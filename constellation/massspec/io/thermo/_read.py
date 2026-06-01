"""Thermo ``.raw`` reader + streaming converter.

Two surfaces:

- :class:`ThermoReader` is the in-memory ``RawReader`` subclass — calls
  :meth:`ThermoReader.read` to get a :class:`ReadResult` whose primary
  table is :data:`MS_PEAK_TABLE` and whose companions are the scan- and
  acquisition-level metadata tables. Suitable for small files in
  notebooks; full read fits in RAM.

- :func:`convert` is the streaming-to-disk entry point used by
  ``constellation massspec convert``. It produces a directory bundle
  (``manifest.json`` + ``peaks.parquet`` + ``scan_metadata.parquet`` +
  ``acquisition_metadata.parquet``) with bounded RAM via a per-RT-bin
  row-group flush.

Modes:

- **Centroid** (default): ``GetCentroidStream`` for profile-acquired
  scans, ``GetSegmentedScanFromScanNumber`` for centroid-acquired
  scans. Per-peak ``peak_resolution`` / ``peak_noise`` /
  ``peak_baseline`` are captured from the CentroidStream when present.
- **Profile** (``profile=True``): ``GetSegmentedScanFromScanNumber``
  unconditionally — preserves Thermo's raw FT grid. Peak-aux columns
  remain null; FT conversion coefficients (A/B/C) come from the
  per-scan trailer when present.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from constellation.core.io.bundle import Bundle
from constellation.core.io.readers import RawReader, ReadResult, register_reader
from constellation.core.progress import ProgressCallback, emit_done, emit_progress, emit_start

from constellation.massspec.schemas import (
    ACQUISITION_METADATA_SCHEMA_VERSION,
    ACQUISITION_METADATA_TABLE,
    MS_PEAK_TABLE,
    SCAN_METADATA_TABLE,
)

from ._filter import parse_filter_string
from ._netruntime import load_clr
from ._trailer import (
    TRAILER_KEY_CHARGE,
    TRAILER_KEY_IIT,
    TRAILER_KEY_MASTER_SCAN,
    TRAILER_KEY_MONOISOTOPIC,
    promote_trailer,
    safe_float,
    safe_int,
    trailer_to_dict,
)

logger = logging.getLogger(__name__)


SOURCE_FORMAT = "thermo_raw"
DEFAULT_BATCH_SIZE = 64
DEFAULT_RT_BIN_WIDTH_S = 60.0
_SHA256_CHUNK_SIZE = 1 << 20  # 1 MiB


def _constellation_version() -> str:
    """Resolve the installed constellation package version (fallback to dev sentinel)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("constellation")
    except PackageNotFoundError:
        return "0.0.0+unknown"


# ── ScanBatch shape ───────────────────────────────────────────────────


@dataclass(frozen=True)
class _ScanBatch:
    """A batch of contiguous scans.

    Arrays are concatenated across all scans in the batch; peaks for
    scan ``i`` live at ``mz[scan_offsets[i]:scan_offsets[i + 1]]``.
    ``peak_aux`` is an optional per-peak column dict (resolution /
    noise / baseline from the Thermo CentroidStream).
    """

    mz: np.ndarray
    intensity: np.ndarray
    scan_offsets: np.ndarray
    scan_meta: list[dict]
    peak_aux: dict[str, np.ndarray] | None


# ── .NET array helpers ────────────────────────────────────────────────


def _netarray_to_numpy(net_array) -> np.ndarray:
    """Copy a .NET ``Double[]`` / ``Single[]`` into a contiguous numpy array.

    Prefers a zero-copy ``np.frombuffer`` view on pythonnet ≥ 3, falling
    back to ``System.Runtime.InteropServices.Marshal.Copy`` on older
    builds. Returns an empty array when ``net_array`` is None or empty.
    """
    if net_array is None:
        return np.asarray([], dtype=np.float64)
    try:
        return np.frombuffer(net_array, dtype=np.float64)
    except TypeError:
        pass
    from System import Int64, IntPtr
    from System.Runtime.InteropServices import Marshal

    n = len(net_array)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    dest = np.empty(n, dtype=np.float64)
    try:
        dst_ptr = IntPtr.__overloads__[Int64](dest.__array_interface__["data"][0])
    except TypeError:
        dst_ptr = IntPtr(dest.__array_interface__["data"][0])
    Marshal.Copy(net_array, 0, dst_ptr, n)
    return dest


def _is_sorted_ascending(arr: np.ndarray) -> bool:
    if arr.size < 2:
        return True
    return bool(np.all(arr[1:] >= arr[:-1]))


# ── scan extraction ───────────────────────────────────────────────────


def _extract_centroid_aux(stream, n: int) -> dict[str, np.ndarray] | None:
    """Pull per-peak Resolutions / Noises / Baselines from a CentroidStream.

    Returns ``None`` when none of the three optional arrays are present.
    Handles older CommonCore builds where the accessors are missing or
    return ``None`` outright.
    """
    out: dict[str, np.ndarray] = {}
    for attr, col in (
        ("Resolutions", "peak_resolution"),
        ("Noises", "peak_noise"),
        ("Baselines", "peak_baseline"),
    ):
        net_arr = getattr(stream, attr, None)
        if net_arr is None:
            continue
        vals = _netarray_to_numpy(net_arr)
        if vals.size == 0:
            continue
        if vals.size != n:
            logger.debug(
                "Thermo %s length %d != peak count %d; skipping aux column",
                attr,
                vals.size,
                n,
            )
            continue
        out[col] = vals.astype(np.float32)
    return out or None


def _fetch_centroid_peaks_with_aux(
    source, scan_num: int, stats
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    """Return (mz, intensity, aux) for a centroid-mode scan.

    All returned arrays are sorted ascending by m/z — Thermo typically
    guarantees this but ``torch.searchsorted`` consumers depend on the
    invariant unconditionally.
    """
    try:
        is_centroid = bool(source.IsCentroidScanFromScanNumber(scan_num))
    except Exception:
        is_centroid = False

    aux: dict[str, np.ndarray] | None = None
    if is_centroid:
        seg = source.GetSegmentedScanFromScanNumber(scan_num, stats)
        mz = _netarray_to_numpy(seg.Positions)
        intensity = _netarray_to_numpy(seg.Intensities)
    else:
        stream = source.GetCentroidStream(scan_num, False)
        if stream is None or stream.Length == 0:
            return (
                np.asarray([], dtype=np.float64),
                np.asarray([], dtype=np.float64),
                None,
            )
        mz = _netarray_to_numpy(stream.Masses)
        intensity = _netarray_to_numpy(stream.Intensities)
        aux = _extract_centroid_aux(stream, mz.size)

    if mz.size > 1 and not _is_sorted_ascending(mz):
        order = np.argsort(mz)
        mz = mz[order]
        intensity = intensity[order]
        if aux is not None:
            aux = {k: v[order] for k, v in aux.items()}
    return mz, intensity, aux


def _fetch_profile_peaks(
    source, scan_num: int, stats
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mz, intensity) for the raw FT profile grid.

    Used when the reader is instantiated with ``profile=True``. Returns
    ``GetSegmentedScanFromScanNumber`` data unconditionally — for
    profile-acquired scans this is the native grid; for centroid-acquired
    scans it's the reduced centroid list.
    """
    seg = source.GetSegmentedScanFromScanNumber(scan_num, stats)
    mz = _netarray_to_numpy(seg.Positions)
    intensity = _netarray_to_numpy(seg.Intensities)
    if mz.size > 1 and not _is_sorted_ascending(mz):
        order = np.argsort(mz)
        mz = mz[order]
        intensity = intensity[order]
    return mz, intensity


# ── scan-meta builder ─────────────────────────────────────────────────


def _build_scan_meta(
    scan_num: int,
    rt: float,
    filter_str: str,
    stats,
    trailer: dict,
    filt,
    *,
    profile_mode: bool,
    capture_trailer_extras: bool,
) -> dict[str, Any]:
    """Combine scan filter + stats + trailer into a SCAN_METADATA_TABLE row dict.

    Also populates the peak-broadcast fields the peak writer extends to
    every peak row (``collision_energy``, ``faims_cv``, ``isolation_*``,
    ``precursor_*``, ``ion_mobility``, ``level``, ``rt``).
    """
    parsed = parse_filter_string(filter_str)
    ms_level = int(parsed.get("ms_level", 1))

    iit = safe_float(trailer.get(TRAILER_KEY_IIT))
    tic = safe_float(getattr(stats, "TIC", None))
    bp_mz = safe_float(getattr(stats, "BasePeakMass", None))
    bp_int = safe_float(getattr(stats, "BasePeakIntensity", None))
    peak_count = safe_int(getattr(stats, "PacketCount", None))

    ce: float | None = None
    activation_type: str | None = None
    iso_lower: float | None = None
    iso_upper: float | None = None
    precursor_mz: float | None = None
    precursor_charge: int | None = None

    if ms_level >= 2:
        activations = parsed.get("activations", [])
        if activations:
            _, activation_type, energy = activations[0]
            if energy is not None:
                ce = float(energy)

        iso_target: float | None = None
        try:
            iso_target = float(filt.GetMass(ms_level - 2))
        except Exception:
            iso_target = activations[0][0] if activations else None

        mono = safe_float(trailer.get(TRAILER_KEY_MONOISOTOPIC))
        if mono and mono > 0:
            precursor_mz = mono
        else:
            precursor_mz = iso_target

        charge = safe_int(trailer.get(TRAILER_KEY_CHARGE))
        if charge and charge > 0:
            precursor_charge = charge

        iso_width = safe_float(trailer.get(f"MS{ms_level} Isolation Width:"))
        iso_offset: float = 0.0
        try:
            iso_offset = float(filt.GetIsolationWidthOffset(ms_level - 2))
        except Exception:
            iso_offset = 0.0
        if iso_width is None:
            try:
                iso_width = float(filt.GetIsolationWidth(ms_level - 2))
            except Exception:
                iso_width = None
        if iso_target is not None and iso_width is not None:
            half = iso_width / 2.0
            iso_lower = iso_target + iso_offset - half
            iso_upper = iso_target + iso_offset + half

    faims_cv = parsed.get("faims_cv")
    if faims_cv is not None:
        faims_cv = float(faims_cv)

    master_scan = safe_int(trailer.get(TRAILER_KEY_MASTER_SCAN))
    if master_scan == 0:
        master_scan = None  # Thermo uses 0 to mean "no parent"
    # Promote to float64 to match SCAN_METADATA_TABLE / MS_PEAK_TABLE.
    master_scan_f = float(master_scan) if master_scan is not None else None
    precursor_scan_f = master_scan_f

    monoisotopic_mz_override = safe_float(trailer.get(TRAILER_KEY_MONOISOTOPIC))
    if monoisotopic_mz_override is not None and monoisotopic_mz_override <= 0:
        monoisotopic_mz_override = None

    trailer_map: list[tuple[str, str]] | None = None
    if capture_trailer_extras:
        trailer_map = [(str(k), str(v)) for k, v in trailer.items() if v is not None]

    row: dict[str, Any] = {
        "scan": scan_num,
        "level": ms_level,
        "rt": rt,
        "tic": tic,
        "iit": iit,
        "base_peak_mz": bp_mz,
        "base_peak_intensity": bp_int,
        "peak_count": peak_count,
        "filter_string": filter_str or None,
        "collision_energy": ce,
        "activation_type": activation_type,
        "faims_cv": faims_cv,
        "isolation_lower": iso_lower,
        "isolation_upper": iso_upper,
        "precursor_scan": precursor_scan_f,
        "precursor_mz": precursor_mz,
        "precursor_charge": precursor_charge,
        "ion_mobility": None,
        "master_scan": master_scan_f,
        "monoisotopic_mz_override": monoisotopic_mz_override,
        "trailer_extras": trailer_map,
        # ``analyzer`` is consumed by _AcquisitionStats (for the acquisition
        # ``analyzers`` list) and is also a persisted per-scan column now.
        # ``profile_mode`` stays virtual — consumed then dropped (it is
        # acquisition-level, in ACQUISITION_METADATA_TABLE).
        "profile_mode": profile_mode,
        "analyzer": parsed.get("analyzer"),
    }
    promoted = promote_trailer(trailer) if capture_trailer_extras else {}
    if not capture_trailer_extras:
        # Fill the typed trailer columns with None so the schema matches.
        promoted = {f: None for f in _TRAILER_COLUMNS}
    row.update(promoted)
    return row


# Cache the typed-column name list once at module import.
from ._trailer import _TRAILER_FIELDS as _TRAILER_FIELDS_TUPLE  # noqa: E402

_TRAILER_COLUMNS: tuple[str, ...] = tuple(f.column for f in _TRAILER_FIELDS_TUPLE)


# ── acquisition-metadata helpers ──────────────────────────────────────


def _extract_tune_data(source) -> list[tuple[str, str]] | None:
    """Return ``GetTuneData(0)`` as a list of ``(key, value)`` string pairs.

    Returns ``None`` on any failure — the column stays null rather than
    blocking acqmeta writes.
    """
    try:
        count = int(source.GetTuneDataCount())
    except Exception:
        logger.debug("GetTuneDataCount failed", exc_info=True)
        return None
    if count <= 0:
        return None
    try:
        raw = source.GetTuneData(0)
    except Exception:
        logger.debug("GetTuneData(0) failed", exc_info=True)
        return None
    d = trailer_to_dict(raw)
    if not d:
        return None
    return [(str(k), str(v)) for k, v in d.items() if v is not None]


def _resolve_method_name(source) -> str | None:
    """Return a short label naming the acquisition method.

    Reads ``SampleInformation.InstrumentMethodFile`` and returns only
    the basename. Constellation deliberately does *not* decode the
    method-text blob via ``GetInstrumentMethod(i)`` — it can come back
    as the literal string ``"None"`` on some firmware and isn't
    load-bearing for v1.
    """
    try:
        sample = source.SampleInformation
        meth_file = getattr(sample, "InstrumentMethodFile", None)
        if meth_file:
            path_str = str(meth_file).strip()
            if path_str:
                for sep in ("\\", "/"):
                    if sep in path_str:
                        path_str = path_str.rsplit(sep, 1)[-1]
                return path_str
    except Exception:
        logger.debug("SampleInformation.InstrumentMethodFile failed", exc_info=True)
    return None


# ── stats accumulator ─────────────────────────────────────────────────


class _AcquisitionStats:
    """Accumulates per-acquisition stats across a single iteration pass.

    Captures fields ``ACQUISITION_METADATA_TABLE`` needs that require
    scanning the file: ``n_scans``, ``rt_min_seconds``, ``rt_max_seconds``,
    ``profile_mode`` (true if any scan reports profile-mode; None if
    none does), and ``analyzers`` (de-duplicated, first-seen order;
    None when no scan reported one).
    """

    __slots__ = (
        "n_scans",
        "rt_min",
        "rt_max",
        "any_profile",
        "saw_profile_flag",
        "_analyzers",
        "_analyzer_seen",
    )

    def __init__(self) -> None:
        self.n_scans: int = 0
        self.rt_min: float | None = None
        self.rt_max: float | None = None
        self.any_profile: bool = False
        self.saw_profile_flag: bool = False
        self._analyzers: list[str] = []
        self._analyzer_seen: set[str] = set()

    def observe(self, batch: _ScanBatch) -> None:
        for meta in batch.scan_meta:
            self.n_scans += 1
            rt = meta.get("rt")
            if rt is not None:
                if self.rt_min is None or rt < self.rt_min:
                    self.rt_min = rt
                if self.rt_max is None or rt > self.rt_max:
                    self.rt_max = rt
            profile_flag = meta.get("profile_mode")
            if profile_flag is not None:
                self.saw_profile_flag = True
                if profile_flag:
                    self.any_profile = True
            analyzer = meta.get("analyzer")
            if analyzer and analyzer not in self._analyzer_seen:
                self._analyzer_seen.add(analyzer)
                self._analyzers.append(analyzer)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_scans": self.n_scans,
            "rt_min_seconds": self.rt_min,
            "rt_max_seconds": self.rt_max,
            "profile_mode": self.any_profile if self.saw_profile_flag else None,
            "analyzers": list(self._analyzers) if self._analyzers else None,
        }


# ── table builders ────────────────────────────────────────────────────


def _rows_to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    """Project a list of row-dicts into a pa.Table under ``schema``.

    Missing keys land as null in the appropriate Arrow type.
    """
    columns: dict[str, list] = {name: [] for name in schema.names}
    for row in rows:
        for k in columns:
            columns[k].append(row.get(k))
    return pa.table(
        {k: pa.array(v, type=schema.field(k).type) for k, v in columns.items()},
        schema=schema,
    )


class _PeakBuffer:
    """Accumulate per-peak rows and flush a row group every RT bin.

    Columns are partitioned at construction time:
    - ``mz`` / ``intensity`` are copied from the ScanBatch peak arrays.
    - Any other schema column that appears in the batch's ``peak_aux``
      dict is sourced from that per-peak array.
    - Everything else is broadcast from ``scan_meta`` (one value
      repeated across the scan's peaks). Absent keys become null.
    """

    _PEAK_BUILTIN: frozenset[str] = frozenset({"mz", "intensity"})

    def __init__(self, schema: pa.Schema, writer: pq.ParquetWriter):
        self._schema = schema
        self._writer = writer
        self._columns: dict[str, list] = {name: [] for name in schema.names}
        self._current_bin: int | None = None

    def append_batch(self, batch: _ScanBatch, rt_bin_width: float) -> None:
        offsets = batch.scan_offsets
        mz = batch.mz
        intensity = batch.intensity
        peak_aux = batch.peak_aux or {}
        per_peak = self._PEAK_BUILTIN | (
            frozenset(peak_aux) & frozenset(self._schema.names)
        )
        broadcast_fields = [n for n in self._schema.names if n not in per_peak]

        for i, meta in enumerate(batch.scan_meta):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            n = end - start
            if n == 0:
                continue

            rt = meta.get("rt")
            bin_idx = (
                int(rt // rt_bin_width)
                if rt is not None
                else (self._current_bin if self._current_bin is not None else 0)
            )
            if self._current_bin is None:
                self._current_bin = bin_idx
            elif bin_idx != self._current_bin and self._columns["mz"]:
                self.flush()
                self._current_bin = bin_idx

            for field in broadcast_fields:
                self._columns[field].extend([meta.get(field)] * n)
            self._columns["mz"].extend(mz[start:end].tolist())
            self._columns["intensity"].extend(intensity[start:end].tolist())
            for name in per_peak - self._PEAK_BUILTIN:
                self._columns[name].extend(peak_aux[name][start:end].tolist())

    def flush(self) -> None:
        if not self._columns["mz"]:
            return
        tbl = pa.table(
            {
                k: pa.array(v, type=self._schema.field(k).type)
                for k, v in self._columns.items()
            },
            schema=self._schema,
        )
        self._writer.write_table(tbl)
        for lst in self._columns.values():
            lst.clear()


# ── reader ────────────────────────────────────────────────────────────


@register_reader
class ThermoReader(RawReader):
    """In-memory reader for one Thermo ``.raw`` file.

    Use :meth:`read` for notebook / small-file workflows (returns a
    :class:`ReadResult` with the peak table as ``primary`` and the
    metadata tables as ``companions``); use the module-level
    :func:`convert` for the streaming-to-disk path the CLI invokes.
    """

    suffixes: ClassVar[tuple[str, ...]] = (".raw",)
    modality: ClassVar[str | None] = "ms"

    def __init__(
        self,
        *,
        profile: bool = False,
        capture_trailer_extras: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.profile = profile
        self.capture_trailer_extras = capture_trailer_extras
        self.batch_size = batch_size

    def read(self, source: "Path | Bundle") -> ReadResult:
        """Read the full file into memory and return a ``ReadResult``."""
        path = source.path if isinstance(source, Bundle) else Path(source)
        with _open_raw_file(path) as raw:
            peaks_table, scan_meta_table, acq_table, run_metadata = _read_all_in_memory(
                raw,
                path,
                profile=self.profile,
                capture_trailer_extras=self.capture_trailer_extras,
                batch_size=self.batch_size,
            )
        return ReadResult(
            primary=peaks_table,
            companions={
                "scan_metadata": scan_meta_table,
                "acquisition_metadata": acq_table,
            },
            run_metadata=run_metadata,
        )


# ── source open / close ───────────────────────────────────────────────


class _RawFileHandle:
    """Context-manager wrapper around a CommonCore ``IRawDataPlus`` handle."""

    def __init__(self, source):
        self._source = source

    @property
    def source(self):
        return self._source

    def __enter__(self):
        return self._source

    def __exit__(self, exc_type, exc, tb):
        try:
            self._source.Dispose()
        except TypeError:
            pass
        return None


def _open_raw_file(path: Path) -> _RawFileHandle:
    """Open a ``.raw`` file via CommonCore; raises if the runtime isn't ready."""
    load_clr()
    import ThermoFisher.CommonCore.Data.Business as Business
    from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter

    src = RawFileReaderAdapter.FileFactory(str(path))
    src.SelectInstrument(Business.Device.MS, 1)
    return _RawFileHandle(src)


# ── scan iteration ────────────────────────────────────────────────────


def _iter_scan_batches(
    source,
    *,
    profile: bool,
    capture_trailer_extras: bool,
    batch_size: int,
) -> Iterator[_ScanBatch]:
    """Yield ``_ScanBatch`` objects across all usable spectra in the file."""
    first = int(source.RunHeaderEx.FirstSpectrum)
    last = int(source.RunHeaderEx.LastSpectrum)

    batch_mz: list[np.ndarray] = []
    batch_intensity: list[np.ndarray] = []
    batch_offsets: list[int] = [0]
    batch_meta: list[dict] = []
    batch_aux: dict[str, list[np.ndarray]] = {
        "peak_resolution": [],
        "peak_noise": [],
        "peak_baseline": [],
    }

    for scan_num in range(first, last + 1):
        result = _extract_scan(
            source,
            scan_num,
            profile=profile,
            capture_trailer_extras=capture_trailer_extras,
        )
        if result is None:
            continue
        meta, mz_arr, int_arr, aux = result

        batch_mz.append(mz_arr)
        batch_intensity.append(int_arr)
        batch_offsets.append(batch_offsets[-1] + mz_arr.size)
        batch_meta.append(meta)
        for key, lst in batch_aux.items():
            if aux is not None and key in aux:
                lst.append(aux[key])
            else:
                lst.append(np.full(mz_arr.size, np.nan, dtype=np.float32))

        if len(batch_meta) >= batch_size:
            yield _finalize_batch(
                batch_mz, batch_intensity, batch_offsets, batch_meta, batch_aux
            )
            batch_mz = []
            batch_intensity = []
            batch_offsets = [0]
            batch_meta = []
            batch_aux = {k: [] for k in batch_aux}

    if batch_meta:
        yield _finalize_batch(
            batch_mz, batch_intensity, batch_offsets, batch_meta, batch_aux
        )


def _extract_scan(
    source,
    scan_num: int,
    *,
    profile: bool,
    capture_trailer_extras: bool,
) -> tuple[dict, np.ndarray, np.ndarray, dict[str, np.ndarray] | None] | None:
    """Pull one scan's metadata, peaks, and optional peak-aux arrays.

    Returns ``None`` if the scan cannot be read — malformed scans are
    logged + skipped rather than aborting iteration.
    """
    try:
        stats = source.GetScanStatsForScanNumber(scan_num)
        filt = source.GetFilterForScanNumber(scan_num)
        filter_str = str(filt.ToString()) if filt is not None else ""
        trailer = trailer_to_dict(source.GetTrailerExtraInformation(scan_num))

        rt_minutes = float(source.RetentionTimeFromScanNumber(scan_num))
        rt = rt_minutes * 60.0

        if profile:
            mz_arr, int_arr = _fetch_profile_peaks(source, scan_num, stats)
            aux: dict[str, np.ndarray] | None = None
        else:
            mz_arr, int_arr, aux = _fetch_centroid_peaks_with_aux(source, scan_num, stats)

        meta = _build_scan_meta(
            scan_num,
            rt,
            filter_str,
            stats,
            trailer,
            filt,
            profile_mode=profile,
            capture_trailer_extras=capture_trailer_extras,
        )
        return meta, mz_arr, int_arr, aux
    except Exception:
        logger.warning(
            "Skipping Thermo scan %d: extraction failed", scan_num, exc_info=True
        )
        return None


def _finalize_batch(
    batch_mz: list[np.ndarray],
    batch_intensity: list[np.ndarray],
    batch_offsets: list[int],
    batch_meta: list[dict],
    batch_aux: dict[str, list[np.ndarray]],
) -> _ScanBatch:
    mz = np.concatenate(batch_mz) if batch_mz else np.asarray([], dtype=np.float64)
    intensity = (
        np.concatenate(batch_intensity)
        if batch_intensity
        else np.asarray([], dtype=np.float64)
    )
    offsets = np.asarray(batch_offsets, dtype=np.int64)
    peak_aux: dict[str, np.ndarray] | None
    if batch_aux and any(lst for lst in batch_aux.values()):
        peak_aux = {k: np.concatenate(v) for k, v in batch_aux.items() if v}
    else:
        peak_aux = None
    return _ScanBatch(mz=mz, intensity=intensity, scan_offsets=offsets, scan_meta=batch_meta, peak_aux=peak_aux)


# ── acquisition-metadata builder ──────────────────────────────────────


def _build_run_metadata(source, path: Path) -> dict[str, Any]:
    """Read the run-level metadata exposed by the file header.

    Populates the ``ACQUISITION_METADATA_TABLE`` fields that don't
    require iterating scans (instrument identity, method name, creation
    date, tune data). Iteration-derived fields (n_scans, RT range,
    profile-mode flag, analyzers list) are filled in later by
    :class:`_AcquisitionStats`.
    """
    meta: dict[str, Any] = {
        "source_format": SOURCE_FORMAT,
        "source_path": str(path.resolve()),
        "constellation_version": _constellation_version(),
        "schema_version": ACQUISITION_METADATA_SCHEMA_VERSION,
    }
    try:
        info = source.GetInstrumentData()
        model = getattr(info, "Model", None)
        if model:
            meta["instrument_model"] = str(model)
        serial = getattr(info, "SerialNumber", None)
        if serial:
            meta["instrument_serial"] = str(serial)
        hw_version = getattr(info, "HardwareVersion", None)
        if hw_version:
            meta["hardware_version"] = str(hw_version)
        sw_version = getattr(info, "SoftwareVersion", None)
        if sw_version:
            meta["software_version"] = str(sw_version)
    except Exception:
        logger.warning("GetInstrumentData failed", exc_info=True)

    method_name = _resolve_method_name(source)
    if method_name:
        meta["method_name"] = method_name

    try:
        fh = source.FileHeader
        cdate = getattr(fh, "CreationDate", None)
        if cdate is not None:
            meta["creation_date"] = str(cdate)
    except Exception:
        logger.debug("FileHeader.CreationDate failed", exc_info=True)

    tune_map = _extract_tune_data(source)
    if tune_map is not None:
        meta["tune_data"] = tune_map

    return meta


def _build_acquisition_table(
    run_metadata: dict[str, Any],
    stats: _AcquisitionStats,
) -> pa.Table:
    row = dict(run_metadata)
    for k, v in stats.as_dict().items():
        row.setdefault(k, v)
    return _rows_to_table([row], ACQUISITION_METADATA_TABLE)


def _build_scan_metadata_table(meta_rows: list[dict]) -> pa.Table:
    """Project the accumulated scan-meta rows into ``SCAN_METADATA_TABLE``.

    Drops virtual keys (``profile_mode``, ``analyzer``) that
    ``_AcquisitionStats`` consumed but aren't part of the schema.
    """
    return _rows_to_table(meta_rows, SCAN_METADATA_TABLE)


# ── in-memory read ────────────────────────────────────────────────────


def _read_all_in_memory(
    source,
    path: Path,
    *,
    profile: bool,
    capture_trailer_extras: bool,
    batch_size: int,
) -> tuple[pa.Table, pa.Table, pa.Table, dict[str, Any]]:
    """Accumulate every scan into one in-memory PEAK_TABLE + companions."""
    run_meta = _build_run_metadata(source, path)
    stats = _AcquisitionStats()
    meta_rows: list[dict] = []

    # Build the peak table in a single buffer by writing into a Parquet
    # buffer in memory? Simpler: collect per-scan rows then collate.
    # For consistency with the streaming path, we reuse _PeakBuffer
    # against an in-memory writer.
    sink = pa.BufferOutputStream()
    with pq.ParquetWriter(
        sink, MS_PEAK_TABLE, compression="zstd", use_dictionary=False
    ) as writer:
        buf = _PeakBuffer(MS_PEAK_TABLE, writer)
        for batch in _iter_scan_batches(
            source,
            profile=profile,
            capture_trailer_extras=capture_trailer_extras,
            batch_size=batch_size,
        ):
            meta_rows.extend(batch.scan_meta)
            stats.observe(batch)
            buf.append_batch(batch, DEFAULT_RT_BIN_WIDTH_S)
        buf.flush()
    peaks_table = pq.read_table(pa.BufferReader(sink.getvalue()))
    scan_meta_table = _build_scan_metadata_table(meta_rows)
    acq_table = _build_acquisition_table(run_meta, stats)

    # ReadResult.run_metadata is a flat dict — pull the most useful
    # fields out of the acquisition table for callers who don't want to
    # open the companion. ``tune_data`` is intentionally NOT mirrored
    # here — it can be very large; consumers go through the companion.
    full_run_meta = dict(run_meta)
    full_run_meta.update(stats.as_dict())
    full_run_meta.pop("tune_data", None)
    return peaks_table, scan_meta_table, acq_table, full_run_meta


# ── streaming convert ─────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_SHA256_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def convert(
    source: "str | Path",
    output_dir: "str | Path",
    *,
    rt_bin_width_s: float = DEFAULT_RT_BIN_WIDTH_S,
    profile: bool = False,
    capture_trailer_extras: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_sha256: bool = True,
    force: bool = False,
    progress_cb: ProgressCallback | None = None,
):
    """Stream a Thermo ``.raw`` into a directory bundle.

    Produces ``<output_dir>/manifest.json`` + ``peaks.parquet`` +
    ``scan_metadata.parquet`` + ``acquisition_metadata.parquet``. Peak
    table writes RT-binned row groups — the same selectivity Cartographer
    relies on — but as a single file (per the project's "one bundle per
    raw" output convention).

    ``progress_cb`` receives ``stage_start`` / ``stage_progress`` /
    ``stage_done`` events tagged with stage ``"thermo_convert"``.
    """
    from .manifest import write_manifest, MANIFEST_FILENAME

    src = Path(source)
    out_dir = Path(output_dir)

    if not src.exists():
        raise FileNotFoundError(f"Thermo .raw not found: {src}")
    if not src.is_file():
        raise IsADirectoryError(f"expected a .raw file, got directory: {src}")

    if out_dir.exists():
        if any(out_dir.iterdir()) and not force:
            raise FileExistsError(
                f"bundle dir is non-empty: {out_dir}\n"
                f"  pass --force to overwrite, or choose a different output dir"
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    emit_start(progress_cb, "thermo_convert", message=f"opening {src.name}")

    sha = _sha256_file(src) if compute_sha256 else None

    peaks_path = out_dir / "peaks.parquet"
    scan_meta_path = out_dir / "scan_metadata.parquet"
    acq_path = out_dir / "acquisition_metadata.parquet"

    stats = _AcquisitionStats()
    meta_rows: list[dict] = []

    with _open_raw_file(src) as raw:
        run_meta = _build_run_metadata(raw, src)
        first = int(raw.RunHeaderEx.FirstSpectrum)
        last = int(raw.RunHeaderEx.LastSpectrum)
        n_total = max(last - first + 1, 0)
        emit_progress(
            progress_cb,
            "thermo_convert",
            completed=0,
            total=n_total,
            message=f"{n_total} scans",
        )

        with pq.ParquetWriter(
            str(peaks_path),
            MS_PEAK_TABLE,
            compression="zstd",
            use_dictionary=False,
        ) as writer:
            buf = _PeakBuffer(MS_PEAK_TABLE, writer)
            for batch in _iter_scan_batches(
                raw,
                profile=profile,
                capture_trailer_extras=capture_trailer_extras,
                batch_size=batch_size,
            ):
                meta_rows.extend(batch.scan_meta)
                stats.observe(batch)
                buf.append_batch(batch, rt_bin_width_s)
                emit_progress(
                    progress_cb,
                    "thermo_convert",
                    completed=stats.n_scans,
                    total=n_total,
                )
            buf.flush()

    scan_meta_table = _build_scan_metadata_table(meta_rows)
    pq.write_table(scan_meta_table, str(scan_meta_path), compression="zstd")

    acq_table = _build_acquisition_table(run_meta, stats)
    pq.write_table(acq_table, str(acq_path), compression="zstd")

    # Manifest carries an instrument-summary view of run metadata plus
    # the convert parameters used. tune_data is omitted (full values
    # live in the acquisition table); we record the key list so
    # consumers can tell which tune fields were captured.
    tune_keys: list[str] = []
    if run_meta.get("tune_data"):
        tune_keys = [k for k, _ in run_meta["tune_data"]]

    summary_run_meta: dict[str, Any] = {
        "instrument_model": run_meta.get("instrument_model"),
        "instrument_serial": run_meta.get("instrument_serial"),
        "hardware_version": run_meta.get("hardware_version"),
        "software_version": run_meta.get("software_version"),
        "method_name": run_meta.get("method_name"),
        "creation_date": run_meta.get("creation_date"),
        "n_scans": stats.n_scans,
        "rt_min_seconds": stats.rt_min,
        "rt_max_seconds": stats.rt_max,
        "profile_mode": stats.any_profile if stats.saw_profile_flag else None,
        "analyzers": list(stats._analyzers) if stats._analyzers else None,
        "tune_data_keys": tune_keys,
    }

    from constellation.thirdparty import registry as _registry

    handle = _registry.try_find("thermo")
    dll_version = handle.version if handle is not None else None

    parameters = {
        "rt_bin_width_s": rt_bin_width_s,
        "profile_mode": profile,
        "capture_trailer_extras": capture_trailer_extras,
        "batch_size": batch_size,
        "dll_version": dll_version,
    }

    manifest = write_manifest(
        out_dir / MANIFEST_FILENAME,
        source_file=str(src.resolve()),
        source_sha256=sha,
        parameters=parameters,
        outputs={
            "peaks": peaks_path.name,
            "scan_metadata": scan_meta_path.name,
            "acquisition_metadata": acq_path.name,
        },
        run_metadata=summary_run_meta,
    )

    emit_done(
        progress_cb,
        "thermo_convert",
        completed=stats.n_scans,
        total=stats.n_scans,
        message=f"{stats.n_scans} scans → {out_dir}",
    )
    return manifest


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_RT_BIN_WIDTH_S",
    "SOURCE_FORMAT",
    "ThermoReader",
    "convert",
]
