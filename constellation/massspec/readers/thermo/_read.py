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
  ``acquisition_metadata.parquet``) with bounded RAM.

Peaks are centroids by default: the instrument's where it stored them, and
Thermo's centroider (``Scan.ToCentroid``) run on profile scans that have no
centroid stream, as ion-trap profile acquisitions do not. ``profile=True``
keeps profile points instead. Each scan records which in
``spectrum_type`` / ``centroider``; the table is in :mod:`._scans`.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.core.io.bundle import Bundle
from constellation.core.io.readers import RawReader, ReadResult, register_reader
from constellation.core.progress import ProgressCallback, emit_done, emit_progress, emit_start

from constellation.massspec.bundle import DEFAULT_RT_BIN_WIDTH_S, PeakWriter
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
_SHA256_CHUNK_SIZE = 1 << 20  # 1 MiB

PEAKS_FILENAME = "peaks.parquet"
SCAN_METADATA_FILENAME = "scan_metadata.parquet"
ACQUISITION_METADATA_FILENAME = "acquisition_metadata.parquet"
# What a conversion writes into a bundle directory, and what a forced
# re-conversion clears first (a peak index built from the old peaks too).
_BUNDLE_OUTPUTS = (
    "manifest.json",
    PEAKS_FILENAME,
    "peaks",
    SCAN_METADATA_FILENAME,
    ACQUISITION_METADATA_FILENAME,
    "mz_index",
)

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


# ── streaming convert ─────────────────────────────────────────────────


def _sha256_file(path: Path, stop: threading.Event | None = None) -> str | None:
    """The file's SHA-256, or None if ``stop`` was set before it finished."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_SHA256_CHUNK_SIZE):
            if stop is not None and stop.is_set():
                return None
            h.update(chunk)
    return h.hexdigest()


def _clear_outputs(out_dir: Path) -> None:
    for name in _BUNDLE_OUTPUTS:
        target = out_dir / name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()


def _dll_version() -> str | None:
    from constellation.thirdparty import registry as _registry

    handle = _registry.try_find("thermo")
    return handle.version if handle is not None else None


def convert(
    source: "str | Path",
    output_dir: "str | Path",
    *,
    rt_bin_width_s: float = DEFAULT_RT_BIN_WIDTH_S,
    profile: bool = False,
    centroider: str = THERMO,
    capture_trailer_extras: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_sha256: bool = True,
    force: bool = False,
    allow_empty_levels: bool = False,
    progress_cb: ProgressCallback | None = None,
):
    """Stream a Thermo ``.raw`` into a directory bundle.

    Produces ``<output_dir>/manifest.json`` + ``peaks.parquet`` +
    ``scan_metadata.parquet`` + ``acquisition_metadata.parquet``. The
    peak table is written in scan order with a row group per
    ``rt_bin_width_s`` of retention time; the manifest is written last, so
    a bundle without one is unfinished.

    ``profile`` keeps profile points; otherwise profile scans are
    centroided, by ``centroider`` where the instrument stored no centroids.
    A level whose scans all come out without peaks fails the conversion
    unless ``allow_empty_levels``. A failed conversion removes what it
    wrote. ``batch_size`` is the number of scans read per block.

    ``progress_cb`` receives ``stage_start`` / ``stage_progress`` /
    ``stage_done`` events tagged with stage ``"thermo_convert"``.
    """
    from .manifest import MANIFEST_FILENAME, write_manifest

    options = ReadOptions(
        profile=profile, centroider=centroider, capture_trailer_extras=capture_trailer_extras
    )
    src = Path(source)
    out_dir = Path(output_dir)

    if not src.exists():
        raise FileNotFoundError(f"Thermo .raw not found: {src}")
    if not src.is_file():
        raise IsADirectoryError(f"expected a .raw file, got directory: {src}")

    if out_dir.exists():
        if any(out_dir.iterdir()):
            if not force:
                raise FileExistsError(
                    f"bundle dir is non-empty: {out_dir}\n"
                    f"  pass --force to overwrite, or choose a different output dir"
                )
            _clear_outputs(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    emit_start(progress_cb, "thermo_convert", message=f"opening {src.name}")

    peaks_path = out_dir / PEAKS_FILENAME
    scan_meta_path = out_dir / SCAN_METADATA_FILENAME
    acq_path = out_dir / ACQUISITION_METADATA_FILENAME

    # The file is hashed in a thread while it is converted; hashlib releases
    # the GIL on large updates.
    hasher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sha256")
    stop_hashing = threading.Event()
    try:
        sha_future = (
            hasher.submit(_sha256_file, src.resolve(), stop_hashing) if compute_sha256 else None
        )
        scan_tables: list[pa.Table] = []
        with _open_raw_file(src) as raw:
            run_meta = _build_run_metadata(raw, src)
            first, last = _scan_range(raw)
            n_total = max(last - first + 1, 0)
            emit_progress(
                progress_cb, "thermo_convert", completed=0, total=n_total,
                message=f"{n_total} scans",
            )
            done = 0
            with PeakWriter(peaks_path, rt_bin_width_s=rt_bin_width_s) as writer:
                for block in iter_scan_blocks(
                    raw, first, last, options, max_block_scans=batch_size
                ):
                    writer.write(block.peak_table())
                    scan_tables.append(block.scans)
                    done += block.scans.num_rows
                    emit_progress(progress_cb, "thermo_convert", completed=done, total=n_total)

        scan_meta_table = _concat_scans(scan_tables)
        levels = level_summary(scan_meta_table)
        for line in describe_levels(levels):
            logger.info("%s: %s", src.name, line)
        if not allow_empty_levels:
            _check_levels(levels, scan_meta_table)

        pq.write_table(scan_meta_table, str(scan_meta_path), compression="zstd")
        acq_table = _build_acquisition_table(run_meta, scan_meta_table)
        pq.write_table(acq_table, str(acq_path), compression="zstd")
        sha = sha_future.result() if sha_future is not None else None
    except BaseException:
        stop_hashing.set()
        hasher.shutdown(wait=True, cancel_futures=True)
        _clear_outputs(out_dir)
        raise
    hasher.shutdown(wait=True)

    # Manifest carries an instrument-summary view of run metadata plus
    # the convert parameters used. tune_data is omitted (full values
    # live in the acquisition table); we record the key list so
    # consumers can tell which tune fields were captured.
    stats = _acquisition_stats(scan_meta_table)
    summary_run_meta: dict[str, Any] = {
        "instrument_model": run_meta.get("instrument_model"),
        "instrument_serial": run_meta.get("instrument_serial"),
        "hardware_version": run_meta.get("hardware_version"),
        "software_version": run_meta.get("software_version"),
        "method_name": run_meta.get("method_name"),
        "creation_date": run_meta.get("creation_date"),
        **stats,
        "levels": levels,
        "tune_data_keys": [k for k, _ in run_meta.get("tune_data") or ()],
    }

    parameters = {
        "rt_bin_width_s": rt_bin_width_s,
        "profile_mode": profile,
        "peaks": "profile" if profile else "centroid",
        "centroider": centroider,
        "capture_trailer_extras": capture_trailer_extras,
        "batch_size": batch_size,
        "dll_version": _dll_version(),
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
        completed=stats["n_scans"],
        total=stats["n_scans"],
        message=f"{stats['n_scans']} scans → {out_dir}",
    )
    return manifest


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_RT_BIN_WIDTH_S",
    "SOURCE_FORMAT",
    "EmptyLevelError",
    "ThermoReader",
    "convert",
    "describe_levels",
    "level_summary",
]
