"""Read a range of scans from an open Thermo ``.raw`` file.

:func:`iter_scan_blocks` walks scans ``first..last`` of a file opened by
:func:`._read._open_raw_file` and yields :class:`ScanBlock` objects: a
block's scans as an Arrow table (:data:`SCAN_METADATA_TABLE`) and their
peaks as numpy arrays, with no Python object per peak.
:meth:`ScanBlock.peak_table` makes the block's peak rows.

How each scan's peaks are made (``ReadOptions.profile`` is ``--profile``):

=========  =====================================  ===================  ============
mode       scan acquired as                       peaks                centroider
=========  =====================================  ===================  ============
centroid   centroid                               stored centroids     instrument
centroid   profile, with a centroid stream        the stream           instrument
           (Orbitrap label data)
centroid   profile, no centroid stream            ``Scan.ToCentroid``  thermo
           (ion trap)
profile    profile                                profile points       (null)
profile    centroid                               stored centroids     instrument
=========  =====================================  ===================  ============

``Scan.ToCentroid`` is the centroider ThermoRawFileParser and msconvert run
on profile scans. ``CENTROIDERS`` maps a centroider's name to the function
that runs it, so another centroider is one more entry.

The DLL calls per scan were chosen by measurement (2026-10-26, Stellar DIA,
CommonCore 5.0.0.93): the filter string is ``ScanStatistics.ScanType`` and
the retention time ``ScanStatistics.StartTime``, both on the statistics
already fetched, where ``GetFilterForScanNumber(...).ToString()`` cost 52 µs
a scan; scan events come in one ``GetScanEvents`` call per range; and the
trailer's labels and values are each joined inside .NET into one string,
because marshalling them element by element cost more than the rest of the
trailer work together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
import pyarrow as pa

from constellation.massspec.bundle import peak_rows
from constellation.massspec.schemas import SCAN_METADATA_TABLE

from ._filter import parse_filter_string
from ._trailer import (
    _TRAILER_FIELDS,
    TRAILER_KEY_CHARGE,
    TRAILER_KEY_IIT,
    TRAILER_KEY_MASTER_SCAN,
    TRAILER_KEY_MONOISOTOPIC,
    safe_float,
    safe_int,
    try_number,
)

logger = logging.getLogger(__name__)

CENTROID = "centroid"
PROFILE = "profile"
INSTRUMENT = "instrument"
THERMO = "thermo"

DEFAULT_BLOCK_SCANS = 512
DEFAULT_BLOCK_PEAKS = 1 << 20

# The peak-table columns a CentroidStream carries per peak.
_STREAM_AUX: tuple[tuple[str, str], ...] = (
    ("Resolutions", "peak_resolution"),
    ("Noises", "peak_noise"),
    ("Baselines", "peak_baseline"),
)

_EMPTY = np.empty(0, dtype=np.float64)


def _as_numpy(net_array) -> np.ndarray:
    """A copy of a .NET ``Double[]`` as a float64 numpy array (empty for None).

    A copy, not a view: a view pins the .NET array for as long as it lives,
    and a block's worth of pinned arrays (512 scans) made every collection
    of the .NET heap slower and more frequent. On ion-trap profile scans,
    whose centroiding allocates large arrays, holding views cost 6.5 ms a
    scan against 1.1 ms copying (2026-10-26).
    """
    if net_array is None:
        return _EMPTY
    try:
        return np.frombuffer(net_array, dtype=np.float64).copy()
    except TypeError:  # pythonnet without the buffer protocol on arrays
        return np.fromiter(net_array, dtype=np.float64, count=len(net_array))


def _thermo_centroids(scan) -> tuple[np.ndarray, np.ndarray]:
    """Thermo's centroider on a profile ``Scan`` (``Scan.ToCentroid``)."""
    from ThermoFisher.CommonCore.Data.Business import Scan

    centroided = Scan.ToCentroid(scan)
    if centroided is None:
        return _EMPTY, _EMPTY
    segmented = centroided.SegmentedScan
    return _as_numpy(segmented.Positions), _as_numpy(segmented.Intensities)


#: Centroiders that can be run on a profile scan at conversion, by name.
CENTROIDERS: dict[str, Callable[[Any], tuple[np.ndarray, np.ndarray]]] = {
    THERMO: _thermo_centroids,
}


@dataclass(frozen=True)
class ReadOptions:
    """What to read. ``profile`` keeps profile points instead of centroiding."""

    profile: bool = False
    centroider: str = THERMO
    capture_trailer_extras: bool = True

    def __post_init__(self) -> None:
        if self.centroider not in CENTROIDERS:
            raise ValueError(
                f"unknown centroider {self.centroider!r}; "
                f"choose from {', '.join(sorted(CENTROIDERS))}"
            )


# ── peaks ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Peaks:
    mz: np.ndarray
    intensity: np.ndarray
    aux: dict[str, np.ndarray] | None
    spectrum_type: str
    centroider: str | None


def _stream_peaks(stream) -> _Peaks:
    mz = _as_numpy(stream.Masses)
    aux: dict[str, np.ndarray] = {}
    for attr, column in _STREAM_AUX:
        values = _as_numpy(getattr(stream, attr, None))
        if values.size == mz.size:
            aux[column] = values
    return _Peaks(mz, _as_numpy(stream.Intensities), aux or None, CENTROID, INSTRUMENT)


def _load_scan(raw, scan_number: int):
    """The whole scan as a CommonCore ``Scan`` (what a centroider takes)."""
    from ThermoFisher.CommonCore.Data.Business import Scan

    return Scan.FromFile(raw, scan_number)


def _read_peaks(raw, scan_number: int, stats, options: ReadOptions) -> _Peaks:
    if stats.IsCentroidScan:
        segmented = raw.GetSegmentedScanFromScanNumber(scan_number, stats)
        return _Peaks(
            _as_numpy(segmented.Positions), _as_numpy(segmented.Intensities),
            None, CENTROID, INSTRUMENT,
        )
    if options.profile:
        segmented = raw.GetSegmentedScanFromScanNumber(scan_number, stats)
        return _Peaks(
            _as_numpy(segmented.Positions), _as_numpy(segmented.Intensities),
            None, PROFILE, None,
        )
    stream = raw.GetCentroidStream(scan_number, False)
    if stream is not None and stream.Length > 0:
        return _stream_peaks(stream)
    scan = _load_scan(raw, scan_number)
    if scan.HasCentroidStream:  # the instrument's stream, empty for this scan
        return _Peaks(_EMPTY, _EMPTY, None, CENTROID, INSTRUMENT)
    mz, intensity = CENTROIDERS[options.centroider](scan)
    return _Peaks(mz, intensity, None, CENTROID, options.centroider)


# ── trailer ───────────────────────────────────────────────────────────


_SEP = "\x1f"  # joins labels and values inside .NET; no trailer text holds it


@dataclass(frozen=True)
class _Layout:
    """What one set of trailer labels maps to, worked out once per set."""

    n: int
    index: dict[str, int]  # label -> position of its (last) value
    # (column, positions of its candidate labels in order, coerce)
    promoted: tuple[tuple[str, tuple[int, ...], Callable[[Any], Any]], ...]
    # unique labels in first-seen order, each with the position of its value
    extras: tuple[tuple[str, int], ...]


_UNSIGNED: frozenset[str] = frozenset(
    f.name for f in SCAN_METADATA_TABLE if pa.types.is_unsigned_integer(f.type)
)
_warned_negative: set[str] = set()


def _non_negative(column: str, coerce: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """``coerce`` for an unsigned column: a negative value becomes null.

    Thermo writes -1 for "not set" in some fields; one such value in an
    unsigned column used to fail the whole conversion.
    """

    def checked(value: Any) -> Any:
        out = coerce(value)
        if out is not None and out < 0:
            if column not in _warned_negative:
                _warned_negative.add(column)
                logger.warning("Thermo trailer gave %s=%r; stored as null", column, out)
            return None
        return out

    return checked


def _layout(labels: list[str]) -> _Layout:
    index: dict[str, int] = {}
    for position, label in enumerate(labels):
        index[label] = position
    promoted = tuple(
        (
            f.column,
            tuple(index[k] for k in f.keys if k in index),
            _non_negative(f.column, f.coerce) if f.column in _UNSIGNED else f.coerce,
        )
        for f in _TRAILER_FIELDS
        if any(k in index for k in f.keys)
    )
    return _Layout(len(labels), index, promoted, tuple(index.items()))


class _Trailer:
    """One scan's trailer: values typed as :func:`try_number` types them."""

    __slots__ = ("layout", "values")

    def __init__(self, layout: _Layout, values: list[Any]):
        self.layout = layout
        self.values = values

    def get(self, label: str) -> Any:
        position = self.layout.index.get(label)
        return None if position is None else self.values[position]

    def promoted(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for column, positions, coerce in self.layout.promoted:
            for position in positions:
                value = coerce(self.values[position])
                if value is not None:
                    out[column] = value
                    break
        return out

    def extras(self) -> list[tuple[str, str]]:
        values = self.values
        return [(label, str(values[p])) for label, p in self.layout.extras if values[p] is not None]


_NO_TRAILER = _Trailer(_layout([]), [])


def _python_join(sep: str, items) -> str:
    return sep.join(str(x) for x in items)


class _TrailerDecoder:
    """Decode trailers, caching the work that repeats from scan to scan."""

    def __init__(self) -> None:
        try:
            from System import String
        except ImportError:  # no CLR loaded: a stand-in source (tests)
            self._join = _python_join
        else:
            self._join = String.Join
        self._layouts: dict[str, _Layout] = {}
        self._typed: dict[str, Any] = {}

    def decode(self, info) -> _Trailer:
        if info is None:
            return _NO_TRAILER
        joined = self._join(_SEP, info.Labels)
        layout = self._layouts.get(joined)
        if layout is None:
            layout = self._layouts[joined] = _layout(joined.split(_SEP) if joined else [])
        if layout.n == 0:
            return _Trailer(layout, [])
        raw = self._join(_SEP, info.Values).split(_SEP)
        if len(raw) != layout.n:  # a value held the separator
            raw = [str(v) for v in info.Values]
        typed = self._typed
        if len(typed) > 200_000:
            typed.clear()
        values = []
        for text in raw:
            value = typed.get(text, typed)
            if value is typed:
                value = typed[text] = try_number(text)
            values.append(value)
        return _Trailer(layout, values)


# ── scan rows ─────────────────────────────────────────────────────────


_TRAILER_COLUMNS: tuple[str, ...] = tuple(f.column for f in _TRAILER_FIELDS)


def _scan_row(
    scan_number: int,
    stats,
    event,
    filter_str: str,
    parsed: dict[str, Any],
    trailer: _Trailer,
    peaks: _Peaks,
    *,
    capture_trailer_extras: bool,
) -> dict[str, Any]:
    """One SCAN_METADATA_TABLE row."""
    ms_level = int(parsed.get("ms_level", 1))

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
            iso_target = float(event.GetMass(ms_level - 2))
        except Exception:
            iso_target = activations[0][0] if activations else None

        mono = safe_float(trailer.get(TRAILER_KEY_MONOISOTOPIC))
        precursor_mz = mono if mono and mono > 0 else iso_target

        charge = safe_int(trailer.get(TRAILER_KEY_CHARGE))
        if charge and charge > 0:
            precursor_charge = charge

        iso_width = safe_float(trailer.get(f"MS{ms_level} Isolation Width:"))
        try:
            iso_offset = float(event.GetIsolationWidthOffset(ms_level - 2))
        except Exception:
            iso_offset = 0.0
        if iso_width is None:
            try:
                iso_width = float(event.GetIsolationWidth(ms_level - 2))
            except Exception:
                iso_width = None
        if iso_target is not None and iso_width is not None:
            half = iso_width / 2.0
            iso_lower = iso_target + iso_offset - half
            iso_upper = iso_target + iso_offset + half

    faims_cv = parsed.get("faims_cv")
    master_scan = safe_int(trailer.get(TRAILER_KEY_MASTER_SCAN))
    master_scan_f = float(master_scan) if master_scan else None  # 0 means no parent
    mono_override = safe_float(trailer.get(TRAILER_KEY_MONOISOTOPIC))
    if mono_override is not None and mono_override <= 0:
        mono_override = None
    scan_window = parsed.get("scan_window")

    row: dict[str, Any] = {
        "scan": scan_number,
        "level": ms_level,
        "rt": float(stats.StartTime) * 60.0,
        "tic": safe_float(stats.TIC),
        "iit": safe_float(trailer.get(TRAILER_KEY_IIT)),
        "base_peak_mz": safe_float(stats.BasePeakMass),
        "base_peak_intensity": safe_float(stats.BasePeakIntensity),
        "peak_count": int(peaks.mz.size),
        "spectrum_type": peaks.spectrum_type,
        "centroider": peaks.centroider,
        "filter_string": filter_str or None,
        "analyzer": parsed.get("analyzer"),
        "collision_energy": ce,
        "activation_type": activation_type,
        "faims_cv": float(faims_cv) if faims_cv is not None else None,
        "isolation_lower": iso_lower,
        "isolation_upper": iso_upper,
        "scan_window_lower": scan_window[0] if scan_window else None,
        "scan_window_upper": scan_window[1] if scan_window else None,
        "precursor_scan": master_scan_f,
        "precursor_mz": precursor_mz,
        "precursor_charge": precursor_charge,
        "master_scan": master_scan_f,
        "monoisotopic_mz_override": mono_override,
        "trailer_extras": trailer.extras() if capture_trailer_extras else None,
    }
    row.update(trailer.promoted())
    return row


# ── blocks ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScanBlock:
    """Consecutive scans: their metadata and their peaks."""

    scans: pa.Table  # SCAN_METADATA_TABLE, one row per scan
    counts: np.ndarray  # int64, peaks per scan
    mz: np.ndarray  # float64, the scans' peaks one after another
    intensity: np.ndarray
    # peak_* column -> (values, valid), as long as mz; None when no scan had any
    aux: dict[str, tuple[np.ndarray, np.ndarray]] | None

    def peak_table(self) -> pa.Table:
        return peak_rows(self.scans, self.counts, self.mz, self.intensity, self.aux)


class _BlockBuilder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.peaks: list[_Peaks] = []
        self.n_peaks = 0

    def add(self, row: dict[str, Any], peaks: _Peaks) -> None:
        self.rows.append(row)
        self.peaks.append(peaks)
        self.n_peaks += int(peaks.mz.size)

    def finish(self) -> ScanBlock:
        counts = np.fromiter((p.mz.size for p in self.peaks), dtype=np.int64, count=len(self.peaks))
        mz = np.concatenate([p.mz for p in self.peaks]) if self.peaks else _EMPTY
        intensity = np.concatenate([p.intensity for p in self.peaks]) if self.peaks else _EMPTY
        aux: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        if any(p.aux for p in self.peaks):
            aux = {}
            for _, column in _STREAM_AUX:
                parts, valid = [], []
                for p in self.peaks:
                    values = p.aux.get(column) if p.aux else None
                    if values is None:
                        parts.append(np.zeros(p.mz.size))
                        valid.append(np.zeros(p.mz.size, dtype=bool))
                    else:
                        parts.append(values)
                        valid.append(np.ones(p.mz.size, dtype=bool))
                aux[column] = (np.concatenate(parts), np.concatenate(valid))
        _sort_within_scans(counts, mz, intensity, aux)
        scans = pa.Table.from_pylist(self.rows, schema=SCAN_METADATA_TABLE)
        return ScanBlock(scans, counts, mz, intensity, aux)


def _sort_within_scans(
    counts: np.ndarray,
    mz: np.ndarray,
    intensity: np.ndarray,
    aux: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> None:
    """Put each scan's peaks in ascending m/z, in place.

    Thermo hands them over sorted; consumers (``searchsorted``, the ``.dia``
    writer) rely on it, so the rare scan that is not gets sorted here.
    """
    if mz.size < 2:
        return
    descending = np.diff(mz) < 0
    ends = np.cumsum(counts)[:-1]
    ends = ends[(ends > 0) & (ends < mz.size)]
    descending[ends - 1] = False  # a drop between two scans is not one
    if not descending.any():
        return
    starts = np.concatenate(([0], np.cumsum(counts)))
    for i in np.unique(np.searchsorted(starts, np.flatnonzero(descending), side="right") - 1):
        lo, hi = int(starts[i]), int(starts[i + 1])
        order = lo + np.argsort(mz[lo:hi], kind="stable")
        mz[lo:hi] = mz[order]
        intensity[lo:hi] = intensity[order]
        for values, valid in (aux or {}).values():
            values[lo:hi] = values[order]
            valid[lo:hi] = valid[order]


def _scan_events(raw, first: int, last: int):
    """Every scan's event in one call, or None to fetch them one by one."""
    try:
        events = raw.GetScanEvents(first, last)
    except Exception:
        logger.debug("GetScanEvents(%d, %d) failed", first, last, exc_info=True)
        return None
    if events is None or len(events) != last - first + 1:
        return None
    return events


def _scan_event(raw, scan_number: int):
    """One scan's event; None if the file cannot give it (MS1 needs none)."""
    try:
        return raw.GetScanEventForScanNumber(scan_number)
    except Exception:
        return None


def iter_scan_blocks(
    raw,
    first: int,
    last: int,
    options: ReadOptions,
    *,
    max_block_scans: int = DEFAULT_BLOCK_SCANS,
    max_block_peaks: int = DEFAULT_BLOCK_PEAKS,
) -> Iterator[ScanBlock]:
    """Blocks of scans ``first..last`` (inclusive), in scan order.

    A block closes at ``max_block_scans`` scans or ``max_block_peaks`` peaks.
    A scan that cannot be read is logged and left out.
    """
    if last < first:
        return
    events = _scan_events(raw, first, last)
    trailers = _TrailerDecoder()
    filters: dict[str, dict[str, Any]] = {}
    block = _BlockBuilder()
    for i, scan_number in enumerate(range(first, last + 1)):
        try:
            stats = raw.GetScanStatsForScanNumber(scan_number)
            event = events[i] if events is not None else _scan_event(raw, scan_number)
            filter_str = str(stats.ScanType or "")
            parsed = filters.get(filter_str)
            if parsed is None:
                parsed = filters[filter_str] = parse_filter_string(filter_str)
            peaks = _read_peaks(raw, scan_number, stats, options)
            trailer = trailers.decode(raw.GetTrailerExtraInformation(scan_number))
            row = _scan_row(
                scan_number, stats, event, filter_str, parsed, trailer, peaks,
                capture_trailer_extras=options.capture_trailer_extras,
            )
        except Exception:
            logger.warning("Skipping Thermo scan %d: extraction failed", scan_number, exc_info=True)
            continue
        block.add(row, peaks)
        if len(block.rows) >= max_block_scans or block.n_peaks >= max_block_peaks:
            yield block.finish()
            block = _BlockBuilder()
    if block.rows:
        yield block.finish()


__all__ = [
    "CENTROID",
    "CENTROIDERS",
    "INSTRUMENT",
    "PROFILE",
    "THERMO",
    "ReadOptions",
    "ScanBlock",
    "iter_scan_blocks",
]
