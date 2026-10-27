"""Write an EncyclopeDIA ``.dia`` (format 0.8.0) from Constellation peak bundles.

The input is one or more bundles as written by ``massspec convert``
(the peaks, ``scan_metadata.parquet``, ``acquisition_metadata.parquet``
and ``manifest.json``; the peaks are read through ``massspec.bundle``). One bundle gives a
single-run ``.dia``; several give a merged one with each bundle as a
``Fraction``, which is how gas-phase fractions are searched together.

The target is a file the search jar treats exactly as it treats one
written by MSRawJava (the library inside EncyclopeDIA 6.x and MSForest).
Every convention below was read off files that library wrote from the
same ``.raw`` inputs, and is what ``_dia_diff`` compares against:

**Single run**
  - ``SpectrumIndex`` is the instrument scan number; ``SpectrumName`` is
    ``scan=N``; ``Fraction`` is 0 and the ``fractions`` table is empty.
  - ``ScanStartTime`` and ``IonInjectionTime`` are seconds (bundles keep
    injection time in ms).
  - MS1 isolation bounds are the scan range (the ``scan_window_*``
    columns; read from the filter string for bundles that predate them).
  - MS2 ``IsolationWindowTarget`` is the isolation m/z at full
    precision: the centre of the isolation window less any isolation
    offset. The filter string's copy has four decimals, which is not
    enough (it disagrees with the reference for half of all DDA scans).
    A bundle with no filter strings, as one imported from a ``.dia``
    is, uses ``precursor_mz``, where the importer put the target.
  - ``PrecursorCharge`` is 0 when the instrument reports none (always,
    for DIA).
  - ``PrecursorName`` is the parent MS1 scan number as an integer, 0
    when the scan has none (always, for DIA).

**Merged run**
  - ``fractions`` names each source as ``<stem>.dia``.
  - ``SpectrumIndex`` is renumbered from 0 and continuous: each
    fraction's MS1 scans, then its MS2 scans, then the next fraction.
    ``SpectrumName`` keeps the original ``scan=N``, so it repeats across
    fractions.
  - Each fraction's MS1 isolation bounds become that fraction's MS2 span.
  - ``ranges`` holds every fraction's windows; metadata is reduced to
    the file-level keys.

**Everywhere**
  - Scalars are float32-rounded before they are stored (the library
    computes in ``float``), so a value here equals the reference bit for
    bit when the inputs agree. Peak m/z stays float64; intensities are
    float32.
  - ``TIC`` is the *sequential* float32 sum of the stored intensities
    (not a pairwise sum, and not the instrument's own TIC).
  - ``rawFileStructure.isStaggered`` and ``.precursorMarginSize`` are
    written for DIA runs only; a DDA run carries just
    ``rawFileStructure.dataAcquisitionType``.
  - One ``ranges`` row per distinct isolation window:
    ``NumWindows`` = scans in it, ``RtStart`` / ``RtStop`` = its first
    and last scan time, ``DutyCycle`` = ``(last - first) /
    (NumWindows - 1)``. The duty cycle is the one scalar computed from
    the *unrounded* scan times and rounded once at the end; doing it
    from the stored float32 times is off by one unit in the last place
    for about half the windows.
  - Injection time: bundles hold it as float32 milliseconds, the file
    wants float32 seconds. Dividing the float32 by 1000 rounds twice and
    lands one unit off for ~15% of scans, so the millisecond value is
    first read back as the short decimal the instrument reported
    (``31.043``), which is what the reference divides.
  - The reference reads injection time and MS2 isolation width as Thermo
    displays them, rounded to the precision the trailer header declares
    (3 and 2 decimals: an isolation width of 2.0009 is 2.00, so its
    windows are centre ± 1.000). Bundles since scan-metadata v4 hold the
    instrument's full values and record that header
    (``acquisition_metadata.trailer_header``); for them both are rounded
    here as Thermo displays them, ties to even. Bundles without
    the header (older ones, imported ones) already hold what they hold
    and are not rounded.

**DDA ``ranges`` differ from the reference, on purpose.** For a DDA run
the reference lists fewer windows than the run has (134,180 rows for
159,789 distinct windows in the file examined) and their ``NumWindows``
do not add up to the scan count: it describes a sample of the scans.
What it samples is not recoverable from the file, so this writer lists
every distinct window instead. The spectrum-centric search builds its
ranges from the MS2 scans themselves, not from this table. For DIA runs
the table matches the reference exactly.

**Metadata** is not decoration: the jar reads ``rawFileStructure.*`` and
``conversion.*`` to decide whether the run is DIA and whether to
demultiplex it, and copies ``sourcename`` into every result as
``SourceFile``. :func:`_single_run_metadata` lists what is written and
why. Provenance keys MSRawJava also writes (``thermo.*`` method text and
the like) are left out when a bundle does not carry the information,
rather than invented.

Refused, because the format has no honest way to hold them from a
bundle today: profile-mode bundles (the search expects centroids), MSn
beyond MS2, and multi-CV FAIMS runs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.massspec.acquisition_structure import (
    AcquisitionStructure,
    infer_acquisition_structure,
)
from constellation.massspec.bundle import has_peaks, iter_peak_batches
from constellation.massspec.io.encyclopedia._codec import encode_f4, encode_f8
from constellation.massspec.io.encyclopedia._schema import (
    DIA_FORMAT_VERSION,
    DIA_INDEXES_V08,
    DIA_TABLES_V08,
)
from constellation.massspec.readers.thermo._filter import parse_filter_string

_SCAN_COLUMNS = (
    "scan",
    "level",
    "rt",
    "iit",
    "isolation_lower",
    "isolation_upper",
    "precursor_mz",
    "precursor_charge",
    "precursor_scan",
    "ms2_isolation_offset",
    "filter_string",
    "faims_cv",
    "activation_type",
)
# Added to SCAN_METADATA_TABLE in schema v3; older bundles lack them and
# the MS1 scan range is then read out of the filter string.
_SCAN_WINDOW_COLUMNS = ("scan_window_lower", "scan_window_upper")

# Thermo activation tokens (as they appear in a filter string) → the
# ``ActivationType`` names MSRawJava records under ``acq.fragmentations``.
_ACTIVATION_NAMES = {
    "hcd": "HigherEnergyCollisionalDissociation",
    "cid": "CollisionInducedDissociation",
    "etd": "ElectronTransferDissociation",
    "ecd": "ElectronCaptureDissociation",
    "uvpd": "UltraVioletPhotoDissociation",
    "irmpd": "InfraredMultiPhotonDissociation",
}

# Conversion settings MSRawJava stamps on every file it writes. The jar's
# OverlappingDiaPreprocessor reads them; the values say "this file has
# not been demultiplexed", which is true of anything written here.
_CONVERSION_KEYS: tuple[tuple[str, Any], ...] = (
    ("conversion.timsTOF.minimumMS1Intensity", 3),
    ("conversion.demux.includeEdgeSubWindows", "true"),
    ("conversion.demux.massTolerance.type", "PPMMassTolerance"),
    ("conversion.timsTOF.minimumMS2Intensity", 1),
    ("conversion.outputType", "EncyclopeDIA"),
    ("conversion.demux.massTolerance.ppm", 10),
    ("conversion.demultiplex", "false"),
    ("conversion.demux.interpolation", "CUBIC_HERMITE"),
    ("conversion.demux.k", 7),
)

_PRECURSOR_COLS = (
    "Fraction", "SpectrumName", "SpectrumIndex", "ScanStartTime", "IonInjectionTime",
    "IsolationWindowLower", "IsolationWindowUpper", "MassEncodedLength", "MassArray",
    "IntensityEncodedLength", "IntensityArray", "IonMobilityArrayEncodedLength",
    "IonMobilityArray", "TIC",
)  # fmt: skip
_SPECTRA_COLS = (
    "Fraction", "SpectrumName", "PrecursorName", "SpectrumIndex", "ScanStartTime",
    "IonInjectionTime", "IsolationWindowLower", "IsolationWindowTarget",
    "IsolationWindowUpper", "PrecursorCharge", "MassEncodedLength", "MassArray",
    "IntensityEncodedLength", "IntensityArray", "IonMobilityArrayEncodedLength",
    "IonMobilityArray", "TIC",
)  # fmt: skip
_RANGES_COLS = (
    "Start", "Stop", "DutyCycle", "NumWindows", "IonMobilityStart", "IonMobilityStop",
    "RtStart", "RtStop",
)  # fmt: skip

SCAN_MAP_SUFFIX = ".scans.parquet"

SCAN_MAP_SCHEMA = pa.schema(
    [
        pa.field("spectrum_index", pa.int64(), nullable=False),
        pa.field("fraction", pa.int32(), nullable=False),
        pa.field("source_name", pa.string(), nullable=False),
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
    ],
    metadata={b"schema_name": b"DiaScanMap"},
)


class SpectraWriteStopped(RuntimeError):
    """:func:`write_dia` was asked to stop (its ``should_stop`` said so)
    before the file was complete; nothing was left under its name."""


@dataclass(frozen=True)
class DiaWriteSummary:
    """What :func:`write_dia` wrote."""

    path: Path
    scan_map_path: Path
    n_fractions: int
    n_precursor: int
    n_spectra: int
    n_ranges: int
    #: Per input bundle, in fraction order.
    structures: tuple[AcquisitionStructure, ...]


# ──────────────────────────────────────────────────────────────────────
# Bundle loading + preflight
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _Bundle:
    path: Path
    source_path: str  # absolute path of the .raw the bundle came from
    source_name: str  # its basename — becomes ``sourcename`` / ``SourceFile``
    acq: dict[str, Any]
    scans: pa.Table
    structure: AcquisitionStructure

    @property
    def dia_name(self) -> str:
        return f"{Path(self.source_name).stem}.dia"


def _load_bundle(path: Path) -> _Bundle:
    path = Path(path)
    missing = [
        name for name in ("scan_metadata.parquet", "acquisition_metadata.parquet")
        if not (path / name).is_file()
    ]
    if not has_peaks(path):
        missing.insert(0, "its peaks")
    if missing:
        raise FileNotFoundError(
            f"{path} is not a peak bundle: missing {', '.join(missing)} "
            f"(expected the directory `massspec convert` writes)"
        )
    acq = pq.read_table(path / "acquisition_metadata.parquet").to_pylist()[0]
    if acq.get("profile_mode"):
        raise ValueError(
            f"{path} is a profile-mode bundle; a .dia holds centroided spectra. "
            f"Re-run `massspec convert` without --profile."
        )
    source_path = str(acq["source_path"])
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        source_path = str(json.loads(manifest_path.read_text()).get("source_file") or source_path)

    scan_file = pq.ParquetFile(path / "scan_metadata.parquet")
    have = set(scan_file.schema_arrow.names)
    columns = [*_SCAN_COLUMNS, *(c for c in _SCAN_WINDOW_COLUMNS if c in have)]
    scans = scan_file.read(columns=columns).sort_by("scan")
    levels = set(pc.unique(scans.column("level")).to_pylist())
    if not levels <= {1, 2}:
        raise ValueError(
            f"{path} has MS levels {sorted(levels)}; a .dia holds MS1 and MS2 only"
        )
    cvs = [v for v in pc.unique(scans.column("faims_cv")).to_pylist() if v is not None]
    if len(cvs) > 1:
        raise ValueError(
            f"{path} is a multi-CV FAIMS run ({sorted(cvs)} V); a .dia has no "
            f"compensation-voltage dimension. Split it by CV first."
        )
    return _Bundle(
        path=path,
        source_path=source_path,
        source_name=Path(source_path).name,
        acq=acq,
        scans=scans,
        structure=infer_acquisition_structure(scans),
    )


def _f32(x: float) -> float:
    """Round to float32 and return as a Python float (what SQLite stores)."""
    return float(np.float32(x))


_IIT_LABEL = "Ion Injection Time (ms):"
_MS2_WIDTH_LABEL = "MS2 Isolation Width:"


def _display_precision(acq: dict[str, Any]) -> dict[str, int] | None:
    """Label -> the precision Thermo displays it with, or None when the
    bundle records no trailer header (its values are not to be rounded)."""
    header = acq.get("trailer_header")
    if not header:
        return None
    return {h["label"]: int(h["precision"]) for h in header}


def _as_displayed(x: float, places: int) -> float:
    """``x`` rounded to ``places`` decimals as Thermo displays it: the exact
    binary value, ties to even (an injection time of 75.5625 ms shows as
    ``75.562``; 2 of 81,850 scans of a Stellar run are such ties)."""
    exact = Decimal(x).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)
    return float(exact)


def _shortest(x: float) -> float:
    """A float32-held value as the shortest decimal that round-trips to it.

    ``31.04299926757...`` (float32 widened to double) → ``31.043``, i.e.
    the number that was rounded into the float32 in the first place,
    whenever that number had seven significant digits or fewer.
    """
    return float(str(np.float32(x)))


@dataclass
class _ScanRows:
    """Per-scan scalars of one bundle, already in the stored convention."""

    scan: np.ndarray  # int64
    level: np.ndarray  # int8
    rt: np.ndarray  # float64 holding float32-rounded seconds
    rt_exact: np.ndarray  # float64 seconds as the bundle has them
    iit: list[float | None]  # float32-rounded seconds
    lower: np.ndarray  # MS2: isolation; MS1: scan range
    upper: np.ndarray
    target: np.ndarray  # MS2 only (nan for MS1)
    charge: np.ndarray  # int64, 0 when unknown
    parent: np.ndarray  # int64 parent MS1 scan, 0 when none


def _scan_rows(bundle: _Bundle) -> _ScanRows:
    t = bundle.scans
    scan = t.column("scan").to_numpy().astype(np.int64)
    level = t.column("level").to_numpy().astype(np.int8)
    rt_exact = t.column("rt").to_numpy().astype(np.float64)
    rt = rt_exact.astype(np.float32).astype(np.float64)
    displayed = _display_precision(bundle.acq)
    if displayed is None:
        iit_ms = _shortest
    else:
        iit_places = displayed.get(_IIT_LABEL, 3)
        iit_ms = lambda v: _as_displayed(float(np.float32(v)), iit_places)  # noqa: E731
    iit = [None if v is None else _f32(iit_ms(v) / 1000.0) for v in t.column("iit").to_pylist()]
    lower = t.column("isolation_lower").to_numpy(zero_copy_only=False).astype(np.float64)
    upper = t.column("isolation_upper").to_numpy(zero_copy_only=False).astype(np.float64)
    width_places = None if displayed is None else displayed.get(_MS2_WIDTH_LABEL, 2)
    target = np.full(scan.shape, np.nan)
    charge_raw = t.column("precursor_charge").to_pylist()
    charge = np.array([0 if c is None else int(c) for c in charge_raw], dtype=np.int64)
    precursor_mz = t.column("precursor_mz").to_pylist()
    offset = t.column("ms2_isolation_offset").to_pylist()
    parent = np.array(
        [0 if v is None else int(v) for v in t.column("precursor_scan").to_pylist()],
        dtype=np.int64,
    )
    if set(_SCAN_WINDOW_COLUMNS) <= set(t.column_names):
        window_lo = t.column("scan_window_lower").to_pylist()
        window_hi = t.column("scan_window_upper").to_pylist()
    else:
        window_lo = window_hi = [None] * scan.size

    # Filter strings repeat heavily (one per window in DIA), so parse each
    # distinct one once.
    parsed: dict[str | None, dict[str, Any]] = {}
    for i, fs in enumerate(t.column("filter_string").to_pylist()):
        info = parsed.get(fs)
        if info is None:
            info = parse_filter_string(fs) if fs else {}
            parsed[fs] = info
        if level[i] == 1:
            if window_lo[i] is not None and window_hi[i] is not None:
                window = (window_lo[i], window_hi[i])
            else:
                window = info.get("scan_window")
            if window is None:
                raise ValueError(
                    f"{bundle.path}: MS1 scan {scan[i]} has no scan range (no "
                    f"scan_window columns, and none in its filter string {fs!r}); "
                    f"cannot set its isolation bounds"
                )
            lower[i], upper[i] = window
        else:
            if np.isnan(lower[i]) or np.isnan(upper[i]):
                raise ValueError(
                    f"{bundle.path}: MS2 scan {scan[i]} has no isolation bounds"
                )
            if not fs and precursor_mz[i] is not None:
                # A bundle imported from a .dia: the stored target, exactly.
                target[i] = precursor_mz[i]
            else:
                # The bounds were built as target + offset ± width/2 from
                # the instrument's full-precision isolation m/z, so the
                # centre less the offset gives that m/z back.
                target[i] = (lower[i] + upper[i]) / 2.0 - (offset[i] or 0.0)
                if width_places is not None:
                    half = _as_displayed(upper[i] - lower[i], width_places) / 2.0
                    centre = target[i] + (offset[i] or 0.0)
                    lower[i], upper[i] = centre - half, centre + half
    f32 = lambda a: a.astype(np.float32).astype(np.float64)  # noqa: E731
    return _ScanRows(
        scan, level, rt, rt_exact, iit, f32(lower), f32(upper), f32(target), charge, parent
    )


# ──────────────────────────────────────────────────────────────────────
# Peaks: one (scan, mz, intensity) triple at a time, bounded memory
# ──────────────────────────────────────────────────────────────────────


def _iter_scan_peaks(
    bundle_path: Path, batch_rows: int
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(scan, mz, intensity)`` for each scan that has peaks.

    A bundle's peaks are one row per peak, scan-major, across its peak
    files in order. A scan can straddle two record batches (row groups
    are cut on RT bins and on pyarrow's own size limit, neither of which
    respects scan boundaries), so the trailing scan of each batch is
    carried into the next.
    """
    carry_scan: int | None = None
    carry_mz: list[np.ndarray] = []
    carry_int: list[np.ndarray] = []
    for batch in iter_peak_batches(
        bundle_path, columns=["scan", "mz", "intensity"], batch_rows=batch_rows
    ):
        scan = batch.column(0).to_numpy()
        mz = batch.column(1).to_numpy()
        inten = batch.column(2).to_numpy()
        cuts = np.flatnonzero(np.diff(scan)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [scan.size]))
        for s, e in zip(starts.tolist(), ends.tolist()):
            sc = int(scan[s])
            if carry_scan is not None and sc != carry_scan:
                yield carry_scan, np.concatenate(carry_mz), np.concatenate(carry_int)
                carry_mz, carry_int = [], []
            carry_scan = sc
            carry_mz.append(mz[s:e])
            carry_int.append(inten[s:e])
    if carry_scan is not None:
        yield carry_scan, np.concatenate(carry_mz), np.concatenate(carry_int)


def _encode(args: tuple[np.ndarray, np.ndarray]) -> tuple[int, bytes, int, bytes, float]:
    """Compress one scan's arrays and compute its TIC. Thread-pool worker."""
    mz, inten = args
    inten32 = inten.astype(np.float32)
    mz_blob, mz_len = encode_f8(mz)
    int_blob, int_len = encode_f4(inten32)
    # Sequential float32 accumulation, to match the reference writer.
    tic = float(np.cumsum(inten32, dtype=np.float32)[-1]) if inten32.size else 0.0
    return mz_len, mz_blob, int_len, int_blob, tic


# ──────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────


def _run_start_time(creation_date: str | None) -> str | None:
    """Thermo's ``MM/DD/YYYY HH:MM:SS`` creation date as ``YYYY-MM-DDTHH:MM:SSZ``.

    The reference writer relabels the instrument's clock reading as UTC
    without converting it; this does the same so the two agree.
    """
    if not creation_date:
        return None
    try:
        parsed = datetime.strptime(creation_date, "%m/%d/%Y %H:%M:%S")
    except ValueError:
        return None
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _single_run_metadata(
    bundle: _Bundle,
    out_path: Path,
    *,
    total_precursor_tic: float,
) -> list[tuple[str, Any]]:
    """Metadata for a single-run file, in the reference writer's key order.

    Three groups:

    - what the search reads: ``rawFileStructure.*``, ``conversion.*``,
      ``sourcename``, ``totalPrecursorTIC``, ``version``, ``filename``,
      ``filelocation``, ``runStartTime``, ``InstrumentConfigurations``,
      ``SoftwareVersion_*``;
    - provenance a bundle can state exactly: ``file.*``, ``run.*`` scan
      and time extents, ``instrument.*``, ``acq.*``;
    - nothing else. Keys the reference carries that a bundle cannot back
      (instrument-method text, sample vial, integrated-intensity
      summaries) are omitted, not guessed.
    """
    acq, s = bundle.acq, bundle.structure
    scans = bundle.scans
    rt = scans.column("rt").to_numpy()
    scan_no = scans.column("scan").to_numpy()
    analyzers = [f"MassAnalyzer{a}" for a in (acq.get("analyzers") or [])]
    tokens = {
        t.lower() for t in pc.unique(scans.column("activation_type")).to_pylist() if t
    }
    fragmentations = sorted(_ACTIVATION_NAMES.get(t, t) for t in tokens)
    filters = [f for f in pc.unique(scans.column("filter_string")).to_pylist() if f]
    polarities = sorted(
        {
            {"+": "Positive", "-": "Negative"}[p]
            for p in (parse_filter_string(f).get("polarity") for f in filters[:64])
            if p in ("+", "-")
        }
    )
    model = acq.get("instrument_model")
    software = acq.get("software_version")
    start = _run_start_time(acq.get("creation_date"))

    md: list[tuple[str, Any]] = [("file.path", bundle.source_path)]
    if acq.get("creation_date"):
        md.append(("thermo.creation_date", acq["creation_date"]))
    md.append(("file.name", bundle.source_name))
    if os.path.isfile(bundle.source_path):
        md.append(("file.size_bytes", os.path.getsize(bundle.source_path)))
    md += [
        ("run.start_time_min", float(rt.min()) / 60.0),
        ("run.end_time_min", float(rt.max()) / 60.0),
        ("run.start_scan", int(scan_no.min())),
        ("run.end_scan", int(scan_no.max())),
        ("run.total_scans", int(scan_no.size)),
        ("run.gradient_length_seconds", float(rt.max() - rt.min())),
    ]
    if model:
        md += [("instrument.model", model), ("instrument.name", model)]
    if acq.get("instrument_serial"):
        md.append(("instrument.serial_number", acq["instrument_serial"]))
    if software:
        md.append(("instrument.software_version", software))
    if analyzers:
        md.append(("acq.mass_analyzers", ",".join(analyzers)))
    if polarities:
        md.append(("acq.polarities", ",".join(polarities)))
    if fragmentations:
        md.append(("acq.fragmentations", ",".join(fragmentations)))
    md += [
        ("acq.ms1_count", s.n_ms1),
        ("acq.ms2_count", s.n_ms2),
        ("rawFileStructure.dataAcquisitionType", s.acquisition_type),
    ]
    if s.acquisition_type == "DIA":
        md += [
            ("rawFileStructure.isStaggered", "true" if s.is_staggered else "false"),
            # The isolation margin MSRawJava infers for overlapped
            # schemes; 0 for every contiguous-window file examined.
            ("rawFileStructure.precursorMarginSize", 0),
        ]
    md += [
        *_CONVERSION_KEYS,
        ("filelocation", bundle.source_path),
        ("sourcename", bundle.source_name),
    ]
    if start:
        md.append(("runStartTime", start))
    if software:
        md.append(("SoftwareVersion_thermo.instrument.software", software))
    if model and analyzers:
        components = "".join(
            f"[INSTRUMENT-ID-COMPONENT-DELIMITER]order:{i},cvRef:,accessionId:,"
            f"name:{name},type:analyzer"
            for i, name in enumerate(analyzers, start=1)
        )
        md.append(
            ("InstrumentConfigurations", f"configurationId:IC1,accession:,name:{model}{components}")
        )
    md += [
        ("totalPrecursorTIC", total_precursor_tic),
        ("filename", out_path.name),
        ("version", DIA_FORMAT_VERSION),
    ]
    return md


def _merged_metadata(out_path: Path) -> list[tuple[str, Any]]:
    """Metadata for a merged file: file-level keys only, as the reference has it."""
    return [
        ("filelocation", str(out_path)),
        ("sourcename", out_path.name),
        *_CONVERSION_KEYS,
        ("filename", out_path.name),
        ("version", DIA_FORMAT_VERSION),
    ]


# ──────────────────────────────────────────────────────────────────────
# Writer
# ──────────────────────────────────────────────────────────────────────


def write_dia(
    bundles: Sequence[Path | str],
    out_path: Path | str,
    *,
    overwrite: bool = False,
    workers: int | None = None,
    batch_rows: int = 1 << 20,
    chunk_scans: int = 512,
    should_stop: Callable[[], bool] | None = None,
) -> DiaWriteSummary:
    """Write ``out_path`` from one bundle (single run) or several (merged).

    Bundles become fractions in the order given. The file is built beside
    its destination and renamed into place, so a failed or interrupted
    write never leaves a truncated ``.dia`` under the final name. A
    scan map (``<out>.scans.parquet``: spectrum index → fraction, source
    file, scan number) is written with it, because in a merged file
    ``SpectrumIndex`` no longer says which run and scan a spectrum is.

    ``workers`` is the zlib thread count (default: up to 8).

    ``should_stop``, when given, is asked between chunks of scans; once
    it answers true the write ends with :class:`SpectraWriteStopped`.
    For a writer in a thread that a signal cannot interrupt (a search
    fan-out being cancelled), so a stop does not wait for the whole file.
    """
    out_path = Path(out_path).resolve()
    if out_path.suffix.lower() != ".dia":
        raise ValueError(f"output must end in .dia: {out_path}")
    if not bundles:
        raise ValueError("write_dia needs at least one bundle")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists (pass overwrite=True to replace it)")
    loaded = [_load_bundle(Path(b)) for b in bundles]
    names = [b.dia_name for b in loaded]
    if len(set(names)) != len(names):
        raise ValueError(f"bundles share a source name; fractions must be distinct: {names}")
    merged = len(loaded) > 1
    workers = workers or min(8, os.cpu_count() or 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    scan_map_path = out_path.with_name(out_path.name + SCAN_MAP_SUFFIX)
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        # A file being built from scratch has nothing to roll back to.
        con.execute("PRAGMA journal_mode = OFF")
        con.execute("PRAGMA synchronous = OFF")
        for _name, ddl in DIA_TABLES_V08:
            con.execute(ddl)

        n_precursor = n_spectra = n_ranges = 0
        total_precursor_tic = 0.0
        base = 0
        map_parts: list[pa.Table] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fraction, bundle in enumerate(loaded):
                rows = _scan_rows(bundle)
                is_ms1 = rows.level == 1
                n1, n2 = int(is_ms1.sum()), int((~is_ms1).sum())
                if merged:
                    # MS1 block first, then the MS2 block, continuing on
                    # from the previous fraction.
                    index = np.empty(rows.scan.size, dtype=np.int64)
                    index[is_ms1] = base + np.arange(n1)
                    index[~is_ms1] = base + n1 + np.arange(n2)
                    base += n1 + n2
                    con.execute(
                        "INSERT INTO fractions (Fraction, Name) VALUES (?, ?)",
                        (fraction, bundle.dia_name),
                    )
                    ms1_lo = float(rows.lower[~is_ms1].min())
                    ms1_hi = float(rows.upper[~is_ms1].max())
                else:
                    index = rows.scan

                tic_sum, windows = _write_fraction(
                    con,
                    pool,
                    bundle,
                    rows,
                    index,
                    fraction=fraction,
                    ms1_bounds=(ms1_lo, ms1_hi) if merged else None,
                    batch_rows=batch_rows,
                    chunk_scans=chunk_scans,
                    should_stop=should_stop,
                )
                total_precursor_tic += tic_sum
                n_precursor += n1
                n_spectra += n2
                con.executemany(
                    f"INSERT INTO ranges ({', '.join(_RANGES_COLS)}) VALUES (?,?,?,?,?,?,?,?)",
                    _range_rows(windows),
                )
                n_ranges += len(windows)
                map_parts.append(
                    pa.table(
                        {
                            "spectrum_index": pa.array(index, pa.int64()),
                            "fraction": pa.array(np.full(index.size, fraction), pa.int32()),
                            "source_name": pa.array([bundle.source_name] * index.size),
                            "scan": pa.array(rows.scan, pa.int32()),
                            "level": pa.array(rows.level, pa.int8()),
                        },
                        schema=SCAN_MAP_SCHEMA,
                    )
                )

        metadata = (
            _merged_metadata(out_path)
            if merged
            else _single_run_metadata(
                loaded[0], out_path, total_precursor_tic=total_precursor_tic
            )
        )
        con.executemany("INSERT INTO metadata (Key, Value) VALUES (?, ?)", metadata)
        for ddl in DIA_INDEXES_V08:
            con.execute(ddl)
        con.commit()
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    os.replace(tmp, out_path)
    pq.write_table(pa.concat_tables(map_parts), scan_map_path, compression="zstd")
    return DiaWriteSummary(
        path=out_path,
        scan_map_path=scan_map_path,
        n_fractions=len(loaded),
        n_precursor=n_precursor,
        n_spectra=n_spectra,
        n_ranges=n_ranges,
        structures=tuple(b.structure for b in loaded),
    )


def _write_fraction(
    con: sqlite3.Connection,
    pool: ThreadPoolExecutor,
    bundle: _Bundle,
    rows: _ScanRows,
    index: np.ndarray,
    *,
    fraction: int,
    ms1_bounds: tuple[float, float] | None,
    batch_rows: int,
    chunk_scans: int,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[float, dict[tuple[float, float], list[float]]]:
    """Insert one bundle's scans. Returns ``(sum of MS1 TIC, window stats)``.

    Scan metadata drives the loop, not the peak table: a scan with no
    peaks has no peak rows but still exists, and is
    written with empty arrays.
    """
    empty = np.empty(0)
    peaks = _iter_scan_peaks(bundle.path, batch_rows)
    pending = next(peaks, None)
    tic_sum = 0.0
    # (lower, upper) → [n scans, first rt, last rt, first exact rt, last exact rt]
    windows: dict[tuple[float, float], list[float]] = {}
    insert_ms1 = f"INSERT INTO precursor ({', '.join(_PRECURSOR_COLS)}) VALUES ({','.join('?' * len(_PRECURSOR_COLS))})"
    insert_ms2 = f"INSERT INTO spectra ({', '.join(_SPECTRA_COLS)}) VALUES ({','.join('?' * len(_SPECTRA_COLS))})"

    n = rows.scan.size
    for start in range(0, n, chunk_scans):
        if should_stop is not None and should_stop():
            raise SpectraWriteStopped(f"stopped writing {bundle.dia_name} at scan {start} of {n}")
        stop = min(start + chunk_scans, n)
        arrays: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(start, stop):
            sc = int(rows.scan[i])
            if pending is not None and pending[0] < sc:
                raise ValueError(
                    f"{bundle.path}: the peaks have scan {pending[0]}, which "
                    f"scan_metadata.parquet does not list"
                )
            if pending is not None and pending[0] == sc:
                arrays.append((pending[1], pending[2]))
                pending = next(peaks, None)
            else:
                arrays.append((empty, empty))
        ms1_rows: list[tuple[Any, ...]] = []
        ms2_rows: list[tuple[Any, ...]] = []
        for i, (mz_len, mz_blob, int_len, int_blob, tic) in zip(
            range(start, stop), pool.map(_encode, arrays)
        ):
            name = f"scan={int(rows.scan[i])}"
            rt = float(rows.rt[i])
            if rows.level[i] == 1:
                lo, hi = ms1_bounds or (float(rows.lower[i]), float(rows.upper[i]))
                tic_sum += tic
                ms1_rows.append(
                    (fraction, name, int(index[i]), rt, rows.iit[i], lo, hi,
                     mz_len, mz_blob, int_len, int_blob, None, None, tic)
                )  # fmt: skip
            else:
                lo, hi = float(rows.lower[i]), float(rows.upper[i])
                exact = float(rows.rt_exact[i])
                w = windows.get((lo, hi))
                if w is None:
                    windows[(lo, hi)] = [1, rt, rt, exact, exact]
                else:
                    w[0] += 1
                    w[2] = rt
                    w[4] = exact
                ms2_rows.append(
                    (fraction, name, int(rows.parent[i]), int(index[i]), rt, rows.iit[i], lo,
                     float(rows.target[i]), hi, int(rows.charge[i]),
                     mz_len, mz_blob, int_len, int_blob, None, None, tic)
                )  # fmt: skip
        if ms1_rows:
            con.executemany(insert_ms1, ms1_rows)
        if ms2_rows:
            con.executemany(insert_ms2, ms2_rows)
    if pending is not None:
        raise ValueError(
            f"{bundle.path}: the peaks have scan {pending[0]}, which "
            f"scan_metadata.parquet does not list"
        )
    return tic_sum, windows


def _range_rows(
    windows: dict[tuple[float, float], list[float]],
) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for (lo, hi), (n, first, last, first_exact, last_exact) in sorted(windows.items()):
        n = int(n)
        # From the unrounded times, rounded once (see the module docstring).
        duty = _f32((last_exact - first_exact) / (n - 1)) if n > 1 else 0.0
        out.append((lo, hi, duty, n, None, None, first, last))
    return out


__all__ = [
    "SCAN_MAP_SCHEMA",
    "SCAN_MAP_SUFFIX",
    "DiaWriteSummary",
    "SpectraWriteStopped",
    "write_dia",
]
