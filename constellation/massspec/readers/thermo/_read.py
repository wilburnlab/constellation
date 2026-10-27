"""Thermo ``.raw`` reader + streaming converter.

Two surfaces:

- :class:`ThermoReader` is the in-memory ``RawReader`` subclass — calls
  :meth:`ThermoReader.read` to get a :class:`ReadResult` whose primary
  table is :data:`MS_PEAK_TABLE` and whose companions are the scan- and
  acquisition-level metadata tables. Suitable for small files in
  notebooks; full read fits in RAM.

- The run-level metadata, the scan-table summaries and the file opening
  that :func:`._convert.convert` (``constellation massspec convert``,
  streaming to a bundle with bounded RAM) builds on.

Peaks are centroids by default: the instrument's where it stored them, and
Thermo's centroider (``Scan.ToCentroid``) run on profile scans that have no
centroid stream, as ion-trap profile acquisitions do not. ``profile=True``
keeps profile points instead. Each scan records which in
``spectrum_type`` / ``centroider``; the table is in :mod:`._scans`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.bundle import Bundle
from constellation.core.io.readers import RawReader, ReadResult, register_reader
from constellation.massspec.schemas import (
    ACQUISITION_METADATA_SCHEMA_VERSION,
    ACQUISITION_METADATA_TABLE,
    MS_PEAK_TABLE,
    SCAN_METADATA_TABLE,
)

from ._netruntime import load_clr
from ._scans import DEFAULT_BLOCK_SCANS, THERMO, ReadOptions, iter_scan_blocks
from ._trailer import trailer_to_dict

logger = logging.getLogger(__name__)


SOURCE_FORMAT = "thermo_raw"
DEFAULT_BATCH_SIZE = DEFAULT_BLOCK_SCANS

# How a level's peaks were made, as the conversion summary says it.
_MADE_BY = {
    "instrument": "centroided by the instrument",
    THERMO: "centroided by Thermo's centroider",
    "profile": "profile points",
}


class EmptyLevelError(ValueError):
    """A conversion found an MS level whose scans have no peaks at all."""


def _constellation_version() -> str:
    """Resolve the installed constellation package version (fallback to dev sentinel)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("constellation")
    except PackageNotFoundError:
        return "0.0.0+unknown"


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


def _build_run_metadata(source, path: Path) -> dict[str, Any]:
    """Read the run-level metadata exposed by the file header.

    Populates the ``ACQUISITION_METADATA_TABLE`` fields that don't
    require iterating scans (instrument identity, method name, creation
    date, tune data). Fields that do (n_scans, RT range, profile flag,
    analyzers) come from the scan table, :func:`_acquisition_stats`.
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


def _acquisition_stats(scans: pa.Table) -> dict[str, Any]:
    """The acquisition-metadata fields read off the scan table."""
    if scans.num_rows == 0:
        return {
            "n_scans": 0,
            "rt_min_seconds": None,
            "rt_max_seconds": None,
            "profile_mode": None,
            "analyzers": None,
        }
    rt = pc.min_max(scans.column("rt")).as_py()
    # unique() keeps first-seen order.
    analyzers = [a for a in pc.unique(scans.column("analyzer")).to_pylist() if a]
    profile = pc.any(pc.equal(scans.column("spectrum_type"), "profile")).as_py()
    return {
        "n_scans": scans.num_rows,
        "rt_min_seconds": rt["min"],
        "rt_max_seconds": rt["max"],
        "profile_mode": bool(profile),
        "analyzers": analyzers or None,
    }


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


def _build_acquisition_table(run_metadata: dict[str, Any], scans: pa.Table) -> pa.Table:
    row = dict(run_metadata)
    for k, v in _acquisition_stats(scans).items():
        row.setdefault(k, v)
    return _rows_to_table([row], ACQUISITION_METADATA_TABLE)


def _concat_scans(tables: list[pa.Table]) -> pa.Table:
    return pa.concat_tables(tables) if tables else SCAN_METADATA_TABLE.empty_table()


# ── what each level holds ─────────────────────────────────────────────


def level_summary(scans: pa.Table) -> list[dict[str, Any]]:
    """Per MS level: scans, peaks, and how many scans' peaks each way made.

    ``peaks_from`` maps ``instrument`` / ``thermo`` / ``profile`` (a
    ``centroider``, or profile points) to a number of scans.
    """
    if scans.num_rows == 0:
        return []
    grouped = scans.group_by(["level", "centroider"], use_threads=False).aggregate(
        [("scan", "count"), ("peak_count", "sum")]
    )
    levels: dict[int, dict[str, Any]] = {}
    for row in grouped.to_pylist():
        level = int(row["level"])
        entry = levels.setdefault(
            level, {"level": level, "n_scans": 0, "n_peaks": 0, "peaks_from": {}}
        )
        entry["n_scans"] += int(row["scan_count"])
        entry["n_peaks"] += int(row["peak_count_sum"] or 0)
        made_by = row["centroider"] or "profile"
        entry["peaks_from"][made_by] = entry["peaks_from"].get(made_by, 0) + int(row["scan_count"])
    return [levels[k] for k in sorted(levels)]


def describe_levels(summary: list[dict[str, Any]]) -> list[str]:
    """One line per level saying what it holds and how its peaks were made."""
    lines = []
    for entry in summary:
        made = ", ".join(
            f"{n:,} {_MADE_BY.get(name, f'centroided by {name}')}"
            for name, n in sorted(entry["peaks_from"].items())
        )
        lines.append(
            f"MS{entry['level']}: {entry['n_scans']:,} scans, "
            f"{entry['n_peaks']:,} peaks ({made})"
        )
    return lines


def _check_levels(summary: list[dict[str, Any]], scans: pa.Table) -> None:
    """Refuse a level whose scans all came out without peaks."""
    empty = [e for e in summary if e["n_scans"] and not e["n_peaks"]]
    if not empty:
        return
    parts = []
    for entry in empty:
        at_level = scans.filter(pc.equal(scans.column("level"), entry["level"]))
        example = at_level.column("filter_string")[0].as_py()
        parts.append(f"MS{entry['level']} ({entry['n_scans']:,} scans, e.g. {example!r})")
    raise EmptyLevelError(
        "no peaks at " + "; ".join(parts) + ". Nothing was written. If the run "
        "really has no signal there, convert with --allow-empty-levels."
    )


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
        centroider: str = THERMO,
        capture_trailer_extras: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.options = ReadOptions(
            profile=profile,
            centroider=centroider,
            capture_trailer_extras=capture_trailer_extras,
        )
        self.batch_size = batch_size

    def read(self, source: "Path | Bundle") -> ReadResult:
        """Read the full file into memory and return a ``ReadResult``."""
        path = source.path if isinstance(source, Bundle) else Path(source)
        peaks: list[pa.Table] = []
        scans: list[pa.Table] = []
        with _open_raw_file(path) as raw:
            run_meta = _build_run_metadata(raw, path)
            first, last = _scan_range(raw)
            for block in iter_scan_blocks(
                raw, first, last, self.options, max_block_scans=self.batch_size
            ):
                peaks.append(block.peak_table())
                scans.append(block.scans)
        scan_table = _concat_scans(scans)
        peak_table = pa.concat_tables(peaks) if peaks else MS_PEAK_TABLE.empty_table()
        # ReadResult.run_metadata is a flat dict — the most useful fields of
        # the acquisition table for callers who don't want to open the
        # companion. ``tune_data`` is left out; it can be very large.
        run_metadata = dict(run_meta)
        run_metadata.update(_acquisition_stats(scan_table))
        run_metadata.pop("tune_data", None)
        return ReadResult(
            primary=peak_table,
            companions={
                "scan_metadata": scan_table,
                "acquisition_metadata": _build_acquisition_table(run_meta, scan_table),
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
    """Open a ``.raw`` file via CommonCore and select its mass spectrometer.

    The path is resolved first: CommonCore opens a symlink but then finds no
    instrument in it, which surfaced as ``ArgumentOutOfRangeException`` from
    ``SelectInstrument``.
    """
    load_clr()
    import ThermoFisher.CommonCore.Data.Business as Business
    from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter

    resolved = Path(path).resolve()
    src = RawFileReaderAdapter.FileFactory(str(resolved))
    if src is None or src.IsError or not src.IsOpen:
        message = None
        try:
            message = src.FileError.ErrorMessage if src is not None else None
        except Exception:
            pass
        raise OSError(
            f"Thermo RawFileReader could not open {resolved}"
            + (f": {message}" if message else "")
        )
    if src.GetInstrumentCountOfType(Business.Device.MS) < 1:
        src.Dispose()
        raise ValueError(f"{resolved} holds no mass-spectrometer data")
    src.SelectInstrument(Business.Device.MS, 1)
    return _RawFileHandle(src)


def _scan_range(raw) -> tuple[int, int]:
    header = raw.RunHeaderEx
    return int(header.FirstSpectrum), int(header.LastSpectrum)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "SOURCE_FORMAT",
    "EmptyLevelError",
    "ThermoReader",
    "describe_levels",
    "level_summary",
]
