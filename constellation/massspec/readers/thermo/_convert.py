"""Convert Thermo ``.raw`` files to bundles, in worker processes.

A file's conversion is planned as contiguous ranges of scans. Each range is
one task: it reads its scans (:mod:`._scans`), writes their peaks as one
part of the peak bundle (``<run>/peaks/part-NNNNN.parquet``) and their stored
points as one part of the profile bundle
(``<run>.profile/profile/part-NNNNN.parquet``), and returns their scan
metadata. The parent concatenates the scan metadata in range order and writes
it, the acquisition metadata and, last, the manifest of each bundle, so a
bundle without a manifest is unfinished.

Profile data is kept by default, with centroiding as a view of it, because
later work wants the points and a conversion should not throw data away. A
run with no profile scans gets no profile bundle, ``keep_profile=False`` writes
none, and ``centroid_algo="none"`` writes the profile bundle alone.

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
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.core.progress import ProgressCallback, emit_done, emit_progress, emit_start
from constellation.massspec.bundle import (
    DEFAULT_RT_BIN_WIDTH_S,
    PEAKS_DIR,
    PROFILE_BUNDLE_KIND,
    PROFILE_DIR,
    PROFILE_SUFFIX,
    PeakWriter,
    part_name,
)

from . import _read
from ._scans import CENTROIDERS, DEFAULT_BLOCK_SCANS, THERMO, ReadOptions, iter_scan_blocks

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = DEFAULT_BLOCK_SCANS
#: ``centroid_algo`` for what a conversion centroids with by default, and for
#: making no peak list at all.
DEFAULT_CENTROID_ALGO = THERMO
NO_CENTROIDING = "none"
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
    "peaks.parquet",  # what bundles before format v2 held
    PEAKS_DIR,
    PROFILE_DIR,
    SCAN_METADATA_FILENAME,
    ACQUISITION_METADATA_FILENAME,
    "mz_index",
    "mz_axes",
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


def read_options(
    *,
    keep_profile: bool = True,
    centroid_algo: str = DEFAULT_CENTROID_ALGO,
    capture_trailer_extras: bool = True,
) -> ReadOptions:
    """The options for what a conversion was asked to keep and to make."""
    return ReadOptions(
        keep_profile=keep_profile,
        centroider=None if centroid_algo == NO_CENTROIDING else centroid_algo,
        capture_trailer_extras=capture_trailer_extras,
    )


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
    has_profile: bool


@_portable
def _probe(path: str) -> _Probe:
    raw = _raw_for(path)
    first, last = _read._scan_range(raw)
    return _Probe(
        first, last, _read._build_run_metadata(raw, Path(path)),
        _read.has_profile_scans(raw, first, last),
    )


@dataclass(frozen=True)
class _RangeTask:
    path: str
    first: int
    last: int
    part: str | None  # of the peak bundle; None when no peak list is made
    point_part: str | None  # of the profile bundle; None when none is kept
    options: ConvertOptions


@dataclass(frozen=True)
class _RangeResult:
    """A range's scan metadata, as each list it wrote describes its scans."""

    scans: pa.Table | None
    point_scans: pa.Table | None
    n_scans: int


@contextmanager
def _part_writer(part: str | None, rt_bin_width_s: float, **kwargs) -> Iterator[PeakWriter | None]:
    """A writer for one part, moved into place once it is finished."""
    if part is None:
        yield None
        return
    final = Path(part)
    tmp = final.with_name(f".{final.name}.tmp")
    with PeakWriter(tmp, rt_bin_width_s=rt_bin_width_s, **kwargs) as writer:
        yield writer
    os.replace(tmp, final)


@_portable
def _run_range(task: _RangeTask, on_scans: Callable[[int], None] | None = None) -> _RangeResult:
    """Read one range, write its parts, return its scan metadata."""
    raw = _raw_for(task.path)
    width = task.options.rt_bin_width_s
    scans: list[pa.Table] = []
    point_scans: list[pa.Table] = []
    n_scans = 0
    with (
        _part_writer(task.part, width) as peaks,
        _part_writer(task.point_part, width, byte_stream_split=True) as points,
    ):
        for block in iter_scan_blocks(
            raw, task.first, task.last, task.options.read,
            max_block_scans=task.options.batch_size,
        ):
            if peaks is not None:
                peaks.write(block.peak_table())
                scans.append(block.scans)
            if points is not None:
                points.write(block.point_table())
                point_scans.append(block.point_scans())
            n_scans += block.scans.num_rows
            if on_scans is not None:
                on_scans(block.scans.num_rows)
    return _RangeResult(
        _read._concat_scans(scans) if peaks is not None else None,
        _read._concat_scans(point_scans) if points is not None else None,
        n_scans,
    )


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


def _refuse_existing(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"bundle dir is non-empty: {directory}\n"
            f"  pass --force to overwrite, or choose a different output dir"
        )


class _Job:
    """One file's conversion: its outputs, its ranges, what came back."""

    def __init__(self, src: Path, out_dir: Path, options: ConvertOptions, *, force: bool):
        self.src = Path(src)
        self.out_dir = Path(out_dir)
        self.profile_dir = self.out_dir.with_name(self.out_dir.name + PROFILE_SUFFIX)
        self.options = options
        self.read = options.read  # what is asked, until the probe settles it
        self.force = force
        self.ranges: list[tuple[int, int]] = []
        self.results: dict[int, _RangeResult] = {}
        self.probe: _Probe | None = None
        self.sha: Future | None = None
        # The peak bundle's manifest, or the profile bundle's if it is alone.
        self.manifest = None
        self.error: BaseException | None = None
        self.n_scans = 0
        self._claimed: list[Path] = []  # directories this conversion has written to

    @property
    def done(self) -> bool:
        return self.manifest is not None or self.error is not None

    def prepare(self, hasher: ThreadPoolExecutor, stop: threading.Event) -> None:
        if not self.src.exists():
            raise FileNotFoundError(f"Thermo .raw not found: {self.src}")
        if not self.src.is_file():
            raise IsADirectoryError(f"expected a .raw file, got directory: {self.src}")
        if not self.force:
            if self.read.centroider is not None:
                _refuse_existing(self.out_dir)
            if self.read.keep_profile:
                _refuse_existing(self.profile_dir)
        if self.options.compute_sha256:
            self.sha = hasher.submit(_sha256_file, self.src.resolve(), stop)

    def _settle(self, probe: _Probe) -> ReadOptions:
        """What to write, now that the file says whether it has profile scans."""
        read = self.options.read
        if probe.has_profile:
            if read.centroider is None and not read.keep_profile:
                raise ValueError(
                    f"{self.src.name} has profile scans, and keeping no profile with no "
                    f"centroiding would write nothing: drop --no-profile, or name a "
                    f"--centroid-algo."
                )
            return read
        if read.centroider is None or not read.keep_profile:
            logger.info(
                "%s has no profile scans: its stored centroids are its peaks, so "
                "--no-profile and --centroid-algo have no effect",
                self.src.name,
            )
        return replace(read, keep_profile=False, centroider=read.centroider or THERMO)

    def _claim(self, directory: Path, listing: str) -> None:
        if self.force:
            if directory.exists():
                _clear_outputs(directory)
        else:
            _refuse_existing(directory)
        (directory / listing).mkdir(parents=True, exist_ok=True)
        self._claimed.append(directory)

    def tasks(self, probe: _Probe, workers: int) -> list[_RangeTask]:
        self.probe = probe
        self.read = self._settle(probe)
        writes_peaks = self.read.centroider is not None
        if writes_peaks:
            self._claim(self.out_dir, PEAKS_DIR)
        if self.read.keep_profile:
            self._claim(self.profile_dir, PROFILE_DIR)
        self.n_scans = max(probe.last - probe.first + 1, 0)
        self.ranges = plan_ranges(probe.first, probe.last, workers)
        options = replace(self.options, read=self.read)
        return [
            _RangeTask(
                str(self.src), lo, hi,
                str(self.out_dir / PEAKS_DIR / part_name(i)) if writes_peaks else None,
                str(self.profile_dir / PROFILE_DIR / part_name(i))
                if self.read.keep_profile else None,
                options,
            )
            for i, (lo, hi) in enumerate(self.ranges)
        ]

    def complete(self) -> bool:
        return self.probe is not None and len(self.results) == len(self.ranges)

    def _write_bundle(
        self,
        directory: Path,
        scans: pa.Table,
        *,
        noun: str,
        parameters: dict[str, Any],
        outputs: dict[str, str],
        sha: str | None,
        **manifest_kwargs,
    ):
        """Write a bundle's scan and acquisition metadata, then its manifest."""
        from .manifest import MANIFEST_FILENAME, write_manifest

        assert self.probe is not None
        levels = _read.level_summary(scans)
        for line in _read.describe_levels(levels, noun=noun):
            logger.info("%s: %s", directory.name, line)
        pq.write_table(scans, directory / SCAN_METADATA_FILENAME, compression="zstd")
        run_meta = self.probe.run_metadata
        pq.write_table(
            _read._build_acquisition_table(run_meta, scans),
            directory / ACQUISITION_METADATA_FILENAME,
            compression="zstd",
        )
        read = self.read
        # The manifest carries a summary of the run metadata plus the
        # parameters used. tune_data is omitted (its values are in the
        # acquisition table); its key list says which fields were captured.
        return write_manifest(
            directory / MANIFEST_FILENAME,
            source_file=str(self.src.resolve()),
            source_sha256=sha,
            parameters={
                "rt_bin_width_s": self.options.rt_bin_width_s,
                **parameters,
                "capture_trailer_extras": read.capture_trailer_extras,
                "batch_size": self.options.batch_size,
                "dll_version": _dll_version(),
            },
            outputs={
                **outputs,
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
                **_read._acquisition_stats(scans),
                "n_scans_without_injection_time": _read.scans_without_injection_time(scans),
                "levels": levels,
                "tune_data_keys": [k for k, _ in run_meta.get("tune_data") or ()],
            },
            **manifest_kwargs,
        )

    def finish(self, workers: int):
        """Write each bundle's metadata and manifest, the profile bundle first."""
        assert self.probe is not None
        read = self.read
        writes_peaks = read.centroider is not None
        ordered = [self.results[i] for i in range(len(self.ranges))]
        peak_scans = _read._concat_scans([r.scans for r in ordered]) if writes_peaks else None
        point_scans = (
            _read._concat_scans([r.point_scans for r in ordered]) if read.keep_profile else None
        )
        scans = peak_scans if peak_scans is not None else point_scans
        assert scans is not None
        if writes_peaks and not self.ranges:  # no scans: still a (schema-only) part
            with PeakWriter(self.out_dir / PEAKS_DIR / part_name(0)):
                pass
        _read._check_scans_read(self.n_scans, scans)
        if not self.options.allow_empty_levels:
            _read._check_levels(_read.level_summary(scans), scans)
        missing = _read.scans_without_injection_time(scans)
        if missing:
            logger.warning(
                "%s: %d of %d scans have no injection time (a blank trailer, or none "
                "recorded); their injection time, AGC and microscans are null",
                self.src.name, missing, scans.num_rows,
            )
        sha = self.sha.result() if self.sha is not None else None
        manifest = None
        if point_scans is not None:
            manifest = self._write_bundle(
                self.profile_dir, point_scans, noun="points", sha=sha,
                kind=PROFILE_BUNDLE_KIND,
                parameters={
                    "workers": workers,
                    "peak_bundle": self.out_dir.name if writes_peaks else None,
                },
                outputs={"profile": PROFILE_DIR},
            )
        if peak_scans is not None:
            manifest = self._write_bundle(
                self.out_dir, peak_scans, noun="peaks", sha=sha,
                parameters={
                    "peaks": "centroid",
                    "centroider": read.centroider,
                    "workers": workers,
                    "profile_bundle": self.profile_dir.name if read.keep_profile else None,
                },
                outputs={"peaks": PEAKS_DIR},
            )
        self.results.clear()
        self.manifest = manifest
        return manifest

    def fail(self, error: BaseException) -> None:
        """Record ``error`` and remove what this conversion wrote.

        A directory it has not written to is left alone: it may hold an
        earlier bundle that was not to be replaced.
        """
        self.error = error
        self.results.clear()
        if self.sha is not None:
            self.sha.cancel()
        for directory in self._claimed:
            try:
                _clear_outputs(directory)
            except OSError:
                logger.warning("could not clear %s after a failed conversion", directory)
        if self.profile_dir in self._claimed:
            try:
                self.profile_dir.rmdir()  # named for the run: nothing asked for it to stay
            except OSError:
                pass


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
                            on_scans(job, value.n_scans)
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


def _is_own_algorithm(centroid_algo: str) -> bool:
    if centroid_algo in (NO_CENTROIDING, *CENTROIDERS):
        return False
    from constellation.massspec.centroiding import ALGORITHMS

    if centroid_algo not in ALGORITHMS:
        known = sorted({NO_CENTROIDING, *CENTROIDERS, *ALGORITHMS})
        raise ValueError(f"unknown centroid_algo {centroid_algo!r}; choose from {', '.join(known)}")
    return True


def finish_own_algorithm(job: _Job, centroid_algo: str, keep_profile: bool, workers: int):
    """Make ``job``'s peak list with an algorithm of Constellation's own, from
    the profile bundle it just wrote. A run with no profile scans already has
    its stored centroids as its peaks."""
    from constellation.massspec.centroiding import centroid_bundle

    from .manifest import read_manifest_dir

    if job.read.centroider is not None:
        return job.manifest
    try:
        centroid_bundle(
            job.profile_dir, centroid_algo, output_dir=job.out_dir, workers=workers, force=job.force
        )
    except BaseException:
        if not keep_profile:
            shutil.rmtree(job.profile_dir, ignore_errors=True)
        raise
    if not keep_profile:
        shutil.rmtree(job.profile_dir, ignore_errors=True)
    return read_manifest_dir(job.out_dir)


def convert(
    source: "str | Path",
    output_dir: "str | Path",
    *,
    rt_bin_width_s: float = DEFAULT_RT_BIN_WIDTH_S,
    keep_profile: bool = True,
    centroid_algo: str = DEFAULT_CENTROID_ALGO,
    capture_trailer_extras: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_sha256: bool = True,
    force: bool = False,
    allow_empty_levels: bool = False,
    workers: int = 1,
    progress_cb: ProgressCallback | None = None,
):
    """Convert a Thermo ``.raw`` into bundles; return the manifest of the peak
    bundle, or of the profile bundle if it is the only one written.

    Produces the peak bundle ``<output_dir>/``: ``manifest.json``,
    ``peaks/part-NNNNN.parquet``, ``scan_metadata.parquet`` and
    ``acquisition_metadata.parquet``, with peaks in scan order across the
    parts, a row group per ``rt_bin_width_s`` of retention time, and the
    manifest written last. Where the run has profile scans, also the profile
    bundle ``<output_dir>.profile/`` beside it, the same with ``profile/``,
    unless ``keep_profile`` is false.

    ``centroid_algo`` makes the peaks of profile scans with no stored
    centroids; ``"none"`` makes no peak list. An algorithm of
    :mod:`constellation.massspec.centroiding` runs on the profile bundle once
    the worker pass is done, and without ``keep_profile`` the profile is removed
    when it succeeds. A file with no profile scans
    has only stored centroids, and they are its peaks whatever these say. A
    level whose scans all come out empty fails the conversion unless
    ``allow_empty_levels``; a file none of whose scans can be read fails it
    always. A failed conversion removes what it wrote. ``workers`` > 1 reads
    ranges of scans in that many processes. ``batch_size`` is the number of
    scans read per block.

    ``progress_cb`` receives ``stage_start`` / ``stage_progress`` /
    ``stage_done`` events tagged with stage ``"thermo_convert"``.
    """
    own = _is_own_algorithm(centroid_algo)
    options = ConvertOptions(
        read=read_options(
            keep_profile=keep_profile or own,
            centroid_algo=NO_CENTROIDING if own else centroid_algo,
            capture_trailer_extras=capture_trailer_extras,
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
    if own:
        job.manifest = finish_own_algorithm(job, centroid_algo, keep_profile, workers)
    n = job.manifest.run_metadata["n_scans"]
    emit_done(
        progress_cb, "thermo_convert", completed=n, total=n,
        message=f"{n} scans → {job.out_dir}",
    )
    return job.manifest


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CENTROID_ALGO",
    "MIN_RANGE_SCANS",
    "NO_CENTROIDING",
    "RANGES_PER_WORKER",
    "ConvertOptions",
    "RawFileReaderError",
    "convert",
    "default_workers",
    "plan_ranges",
]
