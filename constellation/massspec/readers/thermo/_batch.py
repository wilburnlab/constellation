"""Batch conversion of Thermo ``.raw`` files.

Converts N ``.raw`` files into N independent bundle directories. The work
itself is :mod:`._convert`'s: every file is planned as ranges of scans and
all of them go to one pool of worker processes, so the workers stay busy
whether the batch is a directory of small files or one large file. This
module adds what a batch needs on top: one bundle per input stem, skipping
inputs already converted, and a per-file outcome instead of an exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from constellation.core.progress import (
    ProgressCallback,
    emit_done,
    emit_progress,
    emit_start,
)
from constellation.massspec.bundle import DEFAULT_RT_BIN_WIDTH_S

from ._convert import DEFAULT_BATCH_SIZE, ConvertOptions, _Job, _run_jobs
from ._scans import THERMO, ReadOptions
from .manifest import MANIFEST_FILENAME


BatchStatus = Literal["ok", "skipped", "error"]

_STAGE = "thermo_convert_batch"


@dataclass(frozen=True)
class BatchResult:
    """Per-file outcome from :func:`convert_batch`.

    ``detail`` is the exception ``type+message`` string for
    ``status='error'`` and ``None`` otherwise (``'ok'`` / ``'skipped'``).
    """

    input_path: Path
    bundle_dir: Path
    status: BatchStatus
    detail: str | None = None


def _bundle_dir_for(path: Path, output_parent: Path) -> Path:
    return output_parent / path.stem


def convert_batch(
    paths: list[Path],
    output_parent: Path,
    *,
    n_workers: int = 1,
    force: bool = False,
    rt_bin_width_s: float = DEFAULT_RT_BIN_WIDTH_S,
    profile: bool = False,
    centroider: str = THERMO,
    capture_trailer_extras: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_sha256: bool = True,
    allow_empty_levels: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> list[BatchResult]:
    """Convert ``N`` ``.raw`` files into ``N`` bundles, in ``n_workers`` processes.

    Steps:

    1. Validate input — ``paths`` must be non-empty; bundle dirs (one
       per input stem) must be pairwise distinct.
    2. Pre-filter — when ``force=False``, any path whose
       ``<output_parent>/<stem>/manifest.json`` already exists is
       marked ``'skipped'`` and never converted.
    3. Convert the rest, their scan ranges sharing one pool (in this
       process when ``n_workers <= 1``).
    4. Aggregate. Returns one :class:`BatchResult` per input in the
       original ``paths`` order; a file that fails does not stop the others.

    ``progress_cb`` is the *batch-level* progress callback: one event per
    file finished.
    """
    if not paths:
        raise ValueError("convert_batch: paths is empty")

    output_parent = Path(output_parent)

    bundle_dirs = [_bundle_dir_for(p, output_parent) for p in paths]
    if len(set(bundle_dirs)) != len(bundle_dirs):
        seen: dict[Path, list[Path]] = {}
        for p, b in zip(paths, bundle_dirs):
            seen.setdefault(b, []).append(p)
        conflicts = {b: ps for b, ps in seen.items() if len(ps) > 1}
        lines = [f"  {b}: {', '.join(str(p) for p in ps)}" for b, ps in conflicts.items()]
        raise ValueError(
            "convert_batch: duplicate output bundle dirs (multiple inputs "
            "share the same stem):\n" + "\n".join(lines)
        )

    output_parent.mkdir(parents=True, exist_ok=True)

    # Pre-filter: skip paths whose bundle already has a manifest.json
    # (the last file a conversion writes — its presence is a strong
    # "completed" signal). --force re-converts unconditionally.
    results: dict[Path, BatchResult] = {}
    pending: list[tuple[Path, Path]] = []
    for path, bundle in zip(paths, bundle_dirs):
        if not force and (bundle / MANIFEST_FILENAME).is_file():
            results[path] = BatchResult(path, bundle, "skipped")
        else:
            pending.append((path, bundle))

    total = len(paths)
    emit_start(
        progress_cb,
        _STAGE,
        total=total,
        message=f"{len(pending)} to convert, {total - len(pending)} skipped ({total} total)",
    )
    completed = 0
    for r in results.values():
        completed += 1
        emit_progress(
            progress_cb, _STAGE, completed=completed, total=total,
            message=f"skipped {r.input_path.name}",
        )

    options = ConvertOptions(
        read=ReadOptions(
            profile=profile, centroider=centroider, capture_trailer_extras=capture_trailer_extras
        ),
        rt_bin_width_s=rt_bin_width_s,
        batch_size=batch_size,
        compute_sha256=compute_sha256,
        allow_empty_levels=allow_empty_levels,
    )
    jobs = {_Job(path, bundle, options, force=force): path for path, bundle in pending}

    def on_done(job: _Job) -> None:
        nonlocal completed
        path = jobs[job]
        if job.error is None:
            results[path] = BatchResult(path, job.out_dir, "ok")
        else:
            results[path] = BatchResult(
                path, job.out_dir, "error", f"{type(job.error).__name__}: {job.error}"
            )
        completed += 1
        emit_progress(
            progress_cb, _STAGE, completed=completed, total=total,
            message=f"{path.name} → {results[path].status}",
        )

    if jobs:
        _run_jobs(list(jobs), workers=max(1, n_workers), on_done=on_done)

    n_ok = sum(1 for r in results.values() if r.status == "ok")
    n_skip = sum(1 for r in results.values() if r.status == "skipped")
    n_err = sum(1 for r in results.values() if r.status == "error")
    emit_done(
        progress_cb, _STAGE, completed=completed, total=total,
        message=f"{n_ok} succeeded, {n_skip} skipped, {n_err} failed",
    )

    # Return in original input order — callers and tests depend on this.
    return [results[p] for p in paths]


__all__ = [
    "BatchResult",
    "BatchStatus",
    "convert_batch",
]
