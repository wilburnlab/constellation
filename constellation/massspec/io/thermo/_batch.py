"""Parallel batch conversion for Thermo ``.raw`` files.

Owns parent-process fan-out + aggregation for converting N ``.raw``
files into N independent bundle directories. Stays out of
:mod:`constellation.massspec.io.thermo._read` (the CLR-aware streaming
converter) so the split between "what happens inside one .raw" and
"how many .raws run in parallel" stays clean.

Each ``.raw`` is one self-contained task — no within-file chunking is
needed because each file already produces an independent bundle
directory. We do NOT import from :mod:`constellation.sequencing.parallel`
because the project DAG forbids cross-domain imports; the per-file
granularity here also doesn't need the heavyweight in-flight queue +
shard-per-worker writes + ``_SUCCESS`` markers that ``run_batched``
provides.

Spawn safety: pythonnet's CoreCLR is documented to deadlock if a
process inherits a partially-initialised CLR via fork. The parent
intentionally stays CLR-free (the CLI calls ``require_thermo()`` —
a pure file-existence check — but never ``load_clr()``); workers
lazy-load CLR inside their own ``_open_raw_file`` call. We additionally
force ``mp_context="spawn"`` so even import-time CLR side effects in
the parent can't propagate into workers.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

from constellation.core.progress import (
    ProgressCallback,
    emit_done,
    emit_progress,
    emit_start,
)

from ._read import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RT_BIN_WIDTH_S,
    convert as _convert_one_file,
)
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


def _convert_one_worker(
    src_str: str,
    bundle_dir_str: str,
    *,
    rt_bin_width_s: float,
    profile: bool,
    capture_trailer_extras: bool,
    batch_size: int,
    compute_sha256: bool,
    force: bool,
) -> tuple[str, BatchStatus, str | None]:
    """Top-level (picklable) worker. Runs one ``.raw`` → bundle conversion.

    ``progress_cb`` is intentionally ``None`` here — per-scan
    ``StreamProgress`` events from N workers would interleave on stderr
    and be unreadable. The parent emits one batch-level event per file
    completion instead.

    Exceptions are stringified inside the worker (some pythonnet
    ``.NET`` exception subclasses don't pickle cleanly across
    processes); the worker returns ``(bundle_dir, 'error', detail)``
    rather than re-raising.
    """
    try:
        _convert_one_file(
            src_str,
            bundle_dir_str,
            rt_bin_width_s=rt_bin_width_s,
            profile=profile,
            capture_trailer_extras=capture_trailer_extras,
            batch_size=batch_size,
            compute_sha256=compute_sha256,
            force=force,
            progress_cb=None,
        )
        return bundle_dir_str, "ok", None
    except Exception as exc:  # noqa: BLE001
        return bundle_dir_str, "error", f"{type(exc).__name__}: {exc}"


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
    capture_trailer_extras: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_sha256: bool = True,
    progress_cb: ProgressCallback | None = None,
) -> list[BatchResult]:
    """Convert ``N`` ``.raw`` files into ``N`` bundles, optionally in parallel.

    Steps:

    1. Validate input — ``paths`` must be non-empty; bundle dirs (one
       per input stem) must be pairwise distinct.
    2. Pre-filter — when ``force=False``, any path whose
       ``<output_parent>/<stem>/manifest.json`` already exists is
       marked ``'skipped'`` and never dispatched to a worker.
    3. Clamp ``n_workers`` to ``max(1, min(n_workers, len(remaining)))``.
    4. Fan out remaining paths. When ``n_workers <= 1`` the call runs
       inline (no subprocess) to keep one-file batches cheap and to
       give tests a CLR-free path that doesn't need real ``.raw``
       fixtures. Otherwise: spawn-mode :class:`ProcessPoolExecutor`.
    5. Aggregate. Returns one :class:`BatchResult` per input in the
       original ``paths`` order.

    ``progress_cb`` here is the *batch-level* progress callback; workers
    receive ``progress_cb=None`` unconditionally (see
    :func:`_convert_one_worker`).
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
    # (the last file convert() writes — its presence is a strong
    # "completed" signal). --force re-converts unconditionally.
    skipped: list[BatchResult] = []
    pending: list[tuple[Path, Path]] = []
    for path, bundle in zip(paths, bundle_dirs):
        if not force and (bundle / MANIFEST_FILENAME).is_file():
            skipped.append(
                BatchResult(
                    input_path=path,
                    bundle_dir=bundle,
                    status="skipped",
                    detail=None,
                )
            )
        else:
            pending.append((path, bundle))

    total = len(paths)
    n_pending = len(pending)
    emit_start(
        progress_cb,
        _STAGE,
        total=total,
        message=(
            f"{n_pending} to convert, {len(skipped)} skipped "
            f"({total} total)"
        ),
    )

    # Emit progress events for the skipped paths up front so the
    # running total stays meaningful from the user's perspective.
    completed = 0
    for r in skipped:
        completed += 1
        emit_progress(
            progress_cb,
            _STAGE,
            completed=completed,
            total=total,
            message=f"skipped {r.input_path.name}",
        )

    results_by_path: dict[Path, BatchResult] = {r.input_path: r for r in skipped}

    if pending:
        n_workers_eff = max(1, min(n_workers, n_pending))
        worker_kwargs = dict(
            rt_bin_width_s=rt_bin_width_s,
            profile=profile,
            capture_trailer_extras=capture_trailer_extras,
            batch_size=batch_size,
            compute_sha256=compute_sha256,
            force=force,
        )

        if n_workers_eff <= 1:
            # Inline path — no subprocess. Used for single-file batches
            # AND for tests (the spawn child process re-imports the
            # module fresh, so parent-side monkeypatching of
            # _convert_one_file would not propagate).
            for path, bundle in pending:
                bundle_str, status, detail = _convert_one_worker(
                    str(path), str(bundle), **worker_kwargs
                )
                results_by_path[path] = BatchResult(
                    input_path=path,
                    bundle_dir=Path(bundle_str),
                    status=status,
                    detail=detail,
                )
                completed += 1
                emit_progress(
                    progress_cb,
                    _STAGE,
                    completed=completed,
                    total=total,
                    message=f"{path.name} → {status}",
                )
        else:
            ctx = get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=n_workers_eff, mp_context=ctx
            ) as ex:
                future_to_path: dict = {}
                for path, bundle in pending:
                    fut = ex.submit(
                        _convert_one_worker,
                        str(path),
                        str(bundle),
                        **worker_kwargs,
                    )
                    future_to_path[fut] = path
                for fut in as_completed(future_to_path):
                    path = future_to_path[fut]
                    try:
                        bundle_str, status, detail = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        # Worker process crashed before returning — e.g.
                        # killed by OOM. Surface as a per-file error so
                        # the rest of the batch still completes.
                        bundle_str = str(_bundle_dir_for(path, output_parent))
                        status = "error"
                        detail = f"{type(exc).__name__}: {exc}"
                    results_by_path[path] = BatchResult(
                        input_path=path,
                        bundle_dir=Path(bundle_str),
                        status=status,
                        detail=detail,
                    )
                    completed += 1
                    emit_progress(
                        progress_cb,
                        _STAGE,
                        completed=completed,
                        total=total,
                        message=f"{path.name} → {status}",
                    )

    n_ok = sum(1 for r in results_by_path.values() if r.status == "ok")
    n_skip = sum(1 for r in results_by_path.values() if r.status == "skipped")
    n_err = sum(1 for r in results_by_path.values() if r.status == "error")
    emit_done(
        progress_cb,
        _STAGE,
        completed=completed,
        total=total,
        message=f"{n_ok} succeeded, {n_skip} skipped, {n_err} failed",
    )

    # Return in original input order — callers and tests depend on this.
    return [results_by_path[p] for p in paths]


__all__ = [
    "BatchResult",
    "BatchStatus",
    "convert_batch",
]
