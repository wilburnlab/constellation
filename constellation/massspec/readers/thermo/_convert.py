"""Convert Thermo ``.raw`` files to bundles, in worker processes.

A file's conversion is planned as contiguous ranges of scans. Each range is
one task: it reads its scans (:mod:`._scans`), writes their peaks as one
part (``peaks/part-NNNNN.parquet``) and returns their scan metadata. The
parent concatenates the scan metadata in range order and writes it, the
acquisition metadata and, last, the manifest, so a bundle without a
manifest is unfinished.

Tasks run in a pool of spawned processes, one CommonCore handle per file
per process, or in this process when ``workers`` is 1. pythonnet holds the
GIL through every .NET call, so threads would not read in parallel;
processes do (about 3.5x at 8 on a 12-core workstation). With a pool the
parent never loads the CLR: a first task per file opens it and reports its
scan range and run metadata. Ranges go to the pool in scan order, so the
workers read neighbouring parts of the file and disk access stays close to
sequential. :func:`convert_files` gives the ranges of several files to one
pool, so a directory of small files and one large file both keep every
worker busy.

Spawn safety: pythonnet's CoreCLR is documented to deadlock if a process
inherits a partially-initialised CLR via fork, so the pool always spawns.
A worker skips pythonnet's CLR shutdown at exit: it only frees memory the
process is about to give back, and it once took 2.5 h at 21 GB. No .NET
exception leaves a task: its class lives in a CLR namespace the parent
cannot import, so unpickling it there would break the pool and fail every
file in it. A task raises :class:`RawFileReaderError` in its place.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import shutil
import threading
from collections import OrderedDict
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.core.progress import ProgressCallback, emit_done, emit_progress, emit_start
from constellation.massspec.bundle import (
    DEFAULT_RT_BIN_WIDTH_S,
    LEGACY_PEAKS_FILE,
    PEAKS_DIR,
    PeakWriter,
    part_name,
)

from . import _read
from ._scans import DEFAULT_BLOCK_SCANS, THERMO, ReadOptions, iter_scan_blocks

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = DEFAULT_BLOCK_SCANS
#: A range is at least this many scans; a file is cut into about
#: ``RANGES_PER_WORKER`` ranges per worker so that a slow one does not
#: hold up the rest, nor the last round run on few workers.
MIN_RANGE_SCANS = 2000
RANGES_PER_WORKER = 4

_SHA256_CHUNK_SIZE = 1 << 20  # 1 MiB
SCAN_METADATA_FILENAME = "scan_metadata.parquet"
ACQUISITION_METADATA_FILENAME = "acquisition_metadata.parquet"
# What a conversion writes into a bundle directory, and what a forced
# re-conversion clears first (a peak index built from the old peaks too).
_BUNDLE_OUTPUTS = (
    "manifest.json",
    LEGACY_PEAKS_FILE,
    PEAKS_DIR,
    SCAN_METADATA_FILENAME,
    ACQUISITION_METADATA_FILENAME,
    "mz_index",
)


def default_workers() -> int:
    """Worker processes when none are asked for: the cores, up to 8."""
    return max(1, min(8, os.cpu_count() or 1))


@dataclass(frozen=True)
class ConvertOptions:
    """How every file of a conversion is converted."""

    read: ReadOptions = field(default_factory=ReadOptions)
    rt_bin_width_s: float = DEFAULT_RT_BIN_WIDTH_S
    batch_size: int = DEFAULT_BATCH_SIZE
    compute_sha256: bool = True
    allow_empty_levels: bool = False


def plan_ranges(first: int, last: int, workers: int) -> list[tuple[int, int]]:
    """Contiguous, inclusive scan ranges covering ``first..last``, in order.

    One range for one worker; otherwise about ``RANGES_PER_WORKER`` per
    worker, none shorter than ``MIN_RANGE_SCANS`` scans.
    """
    n = last - first + 1
    if n <= 0:
        return []
    k = 1 if workers <= 1 else max(1, min(n // MIN_RANGE_SCANS, workers * RANGES_PER_WORKER))
    edges = [first + (n * i) // k for i in range(k + 1)]
    return [(edges[i], edges[i + 1] - 1) for i in range(k)]


# ── tasks (run in a worker, or inline) ────────────────────────────────


class RawFileReaderError(RuntimeError):
    """An exception Thermo's RawFileReader (.NET) raised, named in the message."""


def _dotnet_name(exc: BaseException) -> str | None:
    """The .NET type of ``exc`` if the CLR raised it, else None."""
    try:
        import System
    except ImportError:  # no CLR loaded here, so nothing raised is .NET
        return None
    if not isinstance(exc, System.Exception):
        return None
    return f"{type(exc).__module__}.{type(exc).__name__}"


def _portable(task):
    """Run ``task`` with a .NET exception it raises turned into a
    :class:`RawFileReaderError`, which the parent can unpickle. The .NET
    exception stays as its cause (in a pool, in the remote traceback)."""

    @functools.wraps(task)
    def run(*args, **kwargs):
        try:
            return task(*args, **kwargs)
        except Exception as exc:
            name = _dotnet_name(exc)
            if name is None:
                raise
            message = getattr(exc, "Message", None) or str(exc)
            raise RawFileReaderError(f"{name}: {message}") from exc

    return run


# Open files of this process: path -> (handle, raw). At most two, so a
# worker moving through a directory of files does not keep them all open.
_OPEN: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
_MAX_OPEN = 2


def _init_worker() -> None:
    from ._netruntime import load_clr

    load_clr()
    import atexit

    import pythonnet

    atexit.unregister(pythonnet.unload)


def _raw_for(path: str):
    entry = _OPEN.get(path)
    if entry is not None:
        _OPEN.move_to_end(path)
        return entry[1]
    handle = _read._open_raw_file(Path(path))
    raw = handle.__enter__()
    _OPEN[path] = (handle, raw)
    while len(_OPEN) > _MAX_OPEN:
        _, (old, _raw) = _OPEN.popitem(last=False)
        old.__exit__(None, None, None)
    return raw


def _close_open_files() -> None:
    while _OPEN:
        _, (handle, _raw) = _OPEN.popitem(last=False)
        handle.__exit__(None, None, None)


def _warm(path: str) -> None:
    """Start a worker on ``path``: open it and pay CommonCore's first-call
    setup (about 1.5 s in ``GetScanEvents``) while the probe runs.

    Only a head start, so it never fails: the range reader fetches events
    one by one where ``GetScanEvents`` does not work, and a file that will
    not open is reported by its probe.
    """
    try:
        raw = _raw_for(path)
        first, _ = _read._scan_range(raw)
        raw.GetScanEvents(first, first)
    except Exception:
        logger.debug("warming a worker on %s failed", path, exc_info=True)


@dataclass(frozen=True)
class _Probe:
    first: int
    last: int
    run_metadata: dict[str, Any]


@_portable
def _probe(path: str) -> _Probe:
    raw = _raw_for(path)
    first, last = _read._scan_range(raw)
    return _Probe(first, last, _read._build_run_metadata(raw, Path(path)))


@dataclass(frozen=True)
class _RangeTask:
    path: str
    first: int
    last: int
    part: str
    options: ConvertOptions


@_portable
def _run_range(task: _RangeTask, on_scans: Callable[[int], None] | None = None) -> pa.Table:
    """Read one range, write its peak part, return its scan metadata."""
    raw = _raw_for(task.path)
    part = Path(task.part)
    tmp = part.with_name(f".{part.name}.tmp")
    scans: list[pa.Table] = []
    with PeakWriter(tmp, rt_bin_width_s=task.options.rt_bin_width_s) as writer:
        for block in iter_scan_blocks(
            raw, task.first, task.last, task.options.read,
            max_block_scans=task.options.batch_size,
        ):
            writer.write(block.peak_table())
            scans.append(block.scans)
            if on_scans is not None:
                on_scans(block.scans.num_rows)
    os.replace(tmp, part)
    return _read._concat_scans(scans)


class _InlineExecutor(Executor):
    """Runs each task as it is submitted, in this process.

    A task's error goes into its future, as a pool's would. An interrupt
    (KeyboardInterrupt, SystemExit) is raised from ``submit`` at once, so
    the driver cleans up instead of running the tasks queued behind it.
    """

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - handed to the caller
            future.set_exception(exc)
        return future


# ── one file, as the parent sees it ───────────────────────────────────


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


class _Job:
    """One file's conversion: its output, its ranges, what came back."""

    def __init__(self, src: Path, out_dir: Path, options: ConvertOptions, *, force: bool):
        self.src = Path(src)
        self.out_dir = Path(out_dir)
        self.options = options
        self.force = force
        self.ranges: list[tuple[int, int]] = []
        self.results: dict[int, pa.Table] = {}
        self.probe: _Probe | None = None
        self.sha: Future | None = None
        self.manifest = None
        self.error: BaseException | None = None
        self.n_scans = 0
        self._wrote = False  # whether anything of ours is in out_dir

    @property
    def done(self) -> bool:
        return self.manifest is not None or self.error is not None

    def prepare(self, hasher: ThreadPoolExecutor, stop: threading.Event) -> None:
        if not self.src.exists():
            raise FileNotFoundError(f"Thermo .raw not found: {self.src}")
        if not self.src.is_file():
            raise IsADirectoryError(f"expected a .raw file, got directory: {self.src}")
        if self.out_dir.exists():
            if any(self.out_dir.iterdir()):
                if not self.force:
                    raise FileExistsError(
                        f"bundle dir is non-empty: {self.out_dir}\n"
                        f"  pass --force to overwrite, or choose a different output dir"
                    )
                _clear_outputs(self.out_dir)
        self._wrote = True
        (self.out_dir / PEAKS_DIR).mkdir(parents=True, exist_ok=True)
        if self.options.compute_sha256:
            self.sha = hasher.submit(_sha256_file, self.src.resolve(), stop)

    def tasks(self, probe: _Probe, workers: int) -> list[_RangeTask]:
        self.probe = probe
        self.n_scans = max(probe.last - probe.first + 1, 0)
        self.ranges = plan_ranges(probe.first, probe.last, workers)
        return [
            _RangeTask(
                str(self.src), lo, hi, str(self.out_dir / PEAKS_DIR / part_name(i)), self.options
            )
            for i, (lo, hi) in enumerate(self.ranges)
        ]

    def complete(self) -> bool:
        return self.probe is not None and len(self.results) == len(self.ranges)

    def finish(self, workers: int):
        """Write the scan and acquisition metadata, then the manifest."""
        from .manifest import MANIFEST_FILENAME, write_manifest

        assert self.probe is not None
        if not self.ranges:  # a file with no scans still gets a (schema-only) part
            with PeakWriter(self.out_dir / PEAKS_DIR / part_name(0)):
                pass
        scans = _read._concat_scans([self.results[i] for i in range(len(self.ranges))])
        _read._check_scans_read(self.n_scans, scans)
        levels = _read.level_summary(scans)
        for line in _read.describe_levels(levels):
            logger.info("%s: %s", self.src.name, line)
        if not self.options.allow_empty_levels:
            _read._check_levels(levels, scans)
        pq.write_table(scans, self.out_dir / SCAN_METADATA_FILENAME, compression="zstd")
        run_meta = self.probe.run_metadata
        pq.write_table(
            _read._build_acquisition_table(run_meta, scans),
            self.out_dir / ACQUISITION_METADATA_FILENAME,
            compression="zstd",
        )
        sha = self.sha.result() if self.sha is not None else None

        stats = _read._acquisition_stats(scans)
        read = self.options.read
        # The manifest carries a summary of the run metadata plus the
        # parameters used. tune_data is omitted (its values are in the
        # acquisition table); its key list says which fields were captured.
        self.manifest = write_manifest(
            self.out_dir / MANIFEST_FILENAME,
            source_file=str(self.src.resolve()),
            source_sha256=sha,
            parameters={
                "rt_bin_width_s": self.options.rt_bin_width_s,
                "profile_mode": read.profile,
                "peaks": "profile" if read.profile else "centroid",
                "centroider": read.centroider,
                "capture_trailer_extras": read.capture_trailer_extras,
                "batch_size": self.options.batch_size,
                "workers": workers,
                "dll_version": _dll_version(),
            },
            outputs={
                "peaks": PEAKS_DIR,
                "scan_metadata": SCAN_METADATA_FILENAME,
                "acquisition_metadata": ACQUISITION_METADATA_FILENAME,
            },
            run_metadata={
                "instrument_model": run_meta.get("instrument_model"),
                "instrument_serial": run_meta.get("instrument_serial"),
                "hardware_version": run_meta.get("hardware_version"),
                "software_version": run_meta.get("software_version"),
                "method_name": run_meta.get("method_name"),
                "creation_date": run_meta.get("creation_date"),
                "instrument_method_names": [
                    m["name"] for m in run_meta.get("instrument_methods") or ()
                ],
                **stats,
                "levels": levels,
                "tune_data_keys": [k for k, _ in run_meta.get("tune_data") or ()],
            },
        )
        self.results.clear()
        return self.manifest

    def fail(self, error: BaseException) -> None:
        """Record ``error`` and remove what this conversion wrote.

        Before :meth:`prepare` succeeds nothing is removed: the directory
        may hold an earlier bundle that was not to be replaced.
        """
        self.error = error
        self.results.clear()
        if self.sha is not None:
            self.sha.cancel()
        if not self._wrote:
            return
        try:
            _clear_outputs(self.out_dir)
        except OSError:
            logger.warning("could not clear %s after a failed conversion", self.out_dir)


# ── the driver ────────────────────────────────────────────────────────


def _run_jobs(
    jobs: list[_Job],
    *,
    workers: int,
    on_scans: Callable[[_Job, int], None] | None = None,
    on_done: Callable[[_Job], None] | None = None,
    executor_factory: Callable[[int], Executor] | None = None,
) -> None:
    """Convert every job; each ends with a manifest or an error.

    ``on_scans(job, n)`` is told as scans are read (per block inline, per
    range with a pool); ``on_done(job)`` when a job has finished or failed.
    An interrupt (Ctrl-C, SIGTERM turned into an exception) clears every
    unfinished job's outputs and is raised.
    """
    inline = workers <= 1 and executor_factory is None
    stop_hashing = threading.Event()
    hasher = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sha256")
    if executor_factory is not None:
        executor = executor_factory(workers)
    elif inline:
        executor = _InlineExecutor()
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker
        )
    pending: dict[Future, tuple[_Job, int | None]] = {}

    def finished(job: _Job) -> None:
        for future, (owner, _) in list(pending.items()):
            if owner is job:
                future.cancel()
                pending.pop(future, None)
        if on_done is not None:
            on_done(job)

    def submit_range(job: _Job, index: int, task: _RangeTask) -> None:
        hook = (lambda n: on_scans(job, n)) if (inline and on_scans) else None
        pending[executor.submit(_run_range, task, hook)] = (job, index)

    try:
        if not inline and jobs:
            # A pool starts a worker per task submitted while none is idle;
            # these start them all now rather than once the probes return.
            for _ in range(workers):
                executor.submit(_warm, str(jobs[0].src))
        for job in jobs:
            try:
                job.prepare(hasher, stop_hashing)
            except Exception as exc:  # noqa: BLE001 - recorded per job
                job.fail(exc)
                finished(job)
                continue
            pending[executor.submit(_probe, str(job.src))] = (job, None)
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for future in done:
                job, index = pending.pop(future, (None, None))
                if job is None or job.done:
                    continue
                try:
                    value = future.result()
                    if index is None:
                        for i, task in enumerate(job.tasks(value, workers)):
                            submit_range(job, i, task)
                    else:
                        job.results[index] = value
                        if on_scans is not None and not inline:
                            on_scans(job, value.num_rows)
                    if job.complete():
                        job.finish(workers)
                        finished(job)
                except Exception as exc:  # noqa: BLE001 - recorded per job
                    job.fail(exc)
                    finished(job)
    except BaseException:
        stop_hashing.set()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        hasher.shutdown(wait=True, cancel_futures=True)
        for job in jobs:
            if not job.done:
                job.fail(RuntimeError("interrupted"))
        if inline:
            _close_open_files()
        raise
    executor.shutdown(wait=True)
    hasher.shutdown(wait=True)
    if inline:
        _close_open_files()


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
    workers: int = 1,
    progress_cb: ProgressCallback | None = None,
):
    """Convert a Thermo ``.raw`` into a directory bundle; return its manifest.

    Produces ``<output_dir>/manifest.json``, ``peaks/part-NNNNN.parquet``,
    ``scan_metadata.parquet`` and ``acquisition_metadata.parquet``. Peaks
    are in scan order across the parts, with a row group per
    ``rt_bin_width_s`` of retention time; the manifest is written last.

    ``profile`` keeps profile points; otherwise profile scans are
    centroided, by ``centroider`` where the instrument stored no centroids.
    A level whose scans all come out without peaks fails the conversion
    unless ``allow_empty_levels``; a file none of whose scans can be read
    fails it always. A failed conversion removes what it wrote. ``workers`` > 1 reads ranges of scans in that many processes.
    ``batch_size`` is the number of scans read per block.

    ``progress_cb`` receives ``stage_start`` / ``stage_progress`` /
    ``stage_done`` events tagged with stage ``"thermo_convert"``.
    """
    options = ConvertOptions(
        read=ReadOptions(
            profile=profile, centroider=centroider, capture_trailer_extras=capture_trailer_extras
        ),
        rt_bin_width_s=rt_bin_width_s,
        batch_size=batch_size,
        compute_sha256=compute_sha256,
        allow_empty_levels=allow_empty_levels,
    )
    job = _Job(Path(source), Path(output_dir), options, force=force)
    emit_start(progress_cb, "thermo_convert", message=f"opening {job.src.name}")
    seen = 0

    def on_scans(j: _Job, n: int) -> None:
        nonlocal seen
        seen += n
        emit_progress(progress_cb, "thermo_convert", completed=seen, total=j.n_scans)

    _run_jobs([job], workers=workers, on_scans=on_scans)
    if job.error is not None:
        raise job.error
    n = job.manifest.run_metadata["n_scans"]
    emit_done(
        progress_cb, "thermo_convert", completed=n, total=n,
        message=f"{n} scans → {job.out_dir}",
    )
    return job.manifest


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "MIN_RANGE_SCANS",
    "RANGES_PER_WORKER",
    "ConvertOptions",
    "RawFileReaderError",
    "convert",
    "default_workers",
    "plan_ranges",
]
