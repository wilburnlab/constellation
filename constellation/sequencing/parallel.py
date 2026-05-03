"""Generic batch-parallel executor for sequencing pipeline stages.

Replaces the Nextflow per-batch fan-out NanoporeAnalysis used.
``run_batched`` consumes an iterator of input :class:`pa.Table`
chunks, dispatches each chunk to a ``ProcessPoolExecutor`` worker, and
writes one ``part-NNNN.parquet`` shard per ``(output_key, batch)``
pair directly into the stage directory. Downstream stages consume the
result via ``pa.dataset.dataset(stage_dir)``.

Two cascading benefits over a "concat in the parent" design:

    1. Workers don't pickle Arrow tables back to the parent — each
       shard is written where it was computed, and only the parent's
       per-shard index/path coordination crosses the IPC boundary.
    2. Stage-level ``--resume`` becomes a directory-existence check
       on the ``_SUCCESS`` marker; partial-run recovery is just
       re-running shard indices whose ``part-NNNN.parquet`` is
       missing or zero-byte.

This module is generic — the demux pipeline lives in
:mod:`sequencing.transcriptome.stages`. Other long-read stages
(assembly, polish, scaffold) plug in by writing their own worker
function and calling ``run_batched``.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.sequencing.progress import (
    ProgressCallback,
    emit_done,
    emit_progress,
    emit_start,
)


# Workers return a dict of ``key → pa.Table``; each entry produces one
# shard file under ``<output_dir>/<key>/part-<idx>.parquet``. Worker
# functions must be importable at module level (multiprocessing
# pickles by qualified name).
WorkerFn = Callable[..., dict[str, pa.Table]]


@dataclass(frozen=True)
class StageOutput:
    """Per-output-key shard directory metadata returned by
    :func:`run_batched`. ``shard_paths`` is the absolute order-stable
    path list (sorted by shard index) so downstream consumers can
    read the dataset deterministically."""

    key: str
    directory: Path
    shard_paths: tuple[Path, ...]


def _shard_filename(idx: int) -> str:
    """Stable zero-padded shard filename. 5 digits gives 100,000 shards
    of head-room — well above any realistic batch count for our scale."""
    return f"part-{idx:05d}.parquet"


def _success_marker(stage_dir: Path) -> Path:
    return stage_dir / "_SUCCESS"


def _is_complete(stage_dir: Path) -> bool:
    return _success_marker(stage_dir).is_file()


def _existing_shard_indices(stage_dir: Path) -> set[int]:
    """Indices of shards already on disk (non-empty)."""
    indices: set[int] = set()
    if not stage_dir.is_dir():
        return indices
    for p in stage_dir.iterdir():
        name = p.name
        if not name.startswith("part-") or not name.endswith(".parquet"):
            continue
        if p.stat().st_size == 0:
            continue
        try:
            idx = int(name[len("part-") : -len(".parquet")])
        except ValueError:
            continue
        indices.add(idx)
    return indices


def _worker_run(
    worker_fn: WorkerFn,
    worker_kwargs: dict[str, Any],
    output_dir: Path,
    output_keys: tuple[str, ...],
    batch_idx: int,
    batch: pa.Table,
) -> tuple[int, dict[str, Path]]:
    """Run a single batch through ``worker_fn`` and write its shards.

    Returns ``(batch_idx, {key: shard_path})``. Lives at module level
    (not a closure) so ``ProcessPoolExecutor`` can pickle it.
    """
    out_tables = worker_fn(batch, **worker_kwargs)
    if set(out_tables.keys()) != set(output_keys):
        raise ValueError(
            f"worker output keys {sorted(out_tables.keys())} differ from "
            f"declared {sorted(output_keys)} for batch {batch_idx}"
        )
    paths: dict[str, Path] = {}
    for key in output_keys:
        stage_dir = output_dir / key
        stage_dir.mkdir(parents=True, exist_ok=True)
        shard_path = stage_dir / _shard_filename(batch_idx)
        pq.write_table(out_tables[key], shard_path)
        paths[key] = shard_path
    return batch_idx, paths


def run_batched(
    worker_fn: WorkerFn,
    batches: Iterable[pa.Table],
    *,
    output_dir: Path,
    output_keys: tuple[str, ...],
    n_workers: int = 1,
    worker_kwargs: dict[str, Any] | None = None,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
    stage_label: str = "stage",
    total: int | None = None,
    in_flight_per_worker: int = 2,
) -> dict[str, StageOutput]:
    """Apply ``worker_fn`` to each batch in parallel — streaming.

    Batches are consumed one at a time from the iterator and submitted
    to the executor with a bounded in-flight window so memory peak
    stays at ``n_workers * in_flight_per_worker`` batches regardless
    of input size. Suitable for 30M+ read pipelines.

    Parameters
    ----------
    worker_fn
        Top-level (importable) function. Receives one
        :class:`pa.Table` plus ``**worker_kwargs`` and returns a
        ``{output_key: pa.Table}`` dict whose key set must match
        ``output_keys``.
    batches
        Iterable of input batches — consumed lazily, never
        materialised in full.
    output_dir
        Top-level directory; per-key shard subdirectories
        (``output_dir / key /``) are created automatically.
    output_keys
        Names of the output tables ``worker_fn`` produces. One shard
        file per ``(key, batch_idx)`` is written.
    n_workers
        ``1`` runs everything in the parent process (no
        ``ProcessPoolExecutor``); higher values fan out via
        multiprocessing.
    worker_kwargs
        Extra kwargs passed to ``worker_fn`` per-batch. Must be
        picklable; commonly the design *name* (string) rather
        than the design instance, since name-loading is cheap.
    progress_cb
        Optional :class:`ProgressCallback`.
    resume
        When True, batches whose ``part-NNNN.parquet`` files already
        exist (non-empty) are skipped. If the stage has a
        ``_SUCCESS`` marker, the entire call short-circuits.
    stage_label
        Free-form name used in progress events (``"demux"``, etc.).
    total
        Optional advance count for progress reporting (e.g. parquet
        row-group count). When ``None``, progress events emit
        ``total=None`` and the stage_done event reports the actual
        consumed count.
    in_flight_per_worker
        How many batches to keep queued per worker before backpressuring
        the iterator. Default 2 = each worker has one running + one
        queued. Bumping higher trades RAM for hiding worker stalls.

    Returns
    -------
    dict[str, StageOutput]
        One entry per output key. ``shard_paths`` is sorted by batch
        index for downstream determinism.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be ≥ 1, got {n_workers}")
    if in_flight_per_worker < 1:
        raise ValueError(
            f"in_flight_per_worker must be ≥ 1, got {in_flight_per_worker}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in output_keys:
        (output_dir / key).mkdir(parents=True, exist_ok=True)

    # Resume short-circuit when every output stage is marked complete.
    if resume and all(_is_complete(output_dir / key) for key in output_keys):
        emit_start(
            progress_cb,
            stage_label,
            message="resume: all output keys already complete",
        )
        result = {}
        for key in output_keys:
            stage_dir = output_dir / key
            shards = sorted(
                p
                for p in stage_dir.iterdir()
                if p.name.startswith("part-")
                and p.name.endswith(".parquet")
                and p.stat().st_size > 0
            )
            result[key] = StageOutput(
                key=key, directory=stage_dir, shard_paths=tuple(shards)
            )
        emit_done(progress_cb, stage_label, message="resumed")
        return result

    # Per-key existing-shard set — used for partial recovery (rerun
    # missing shards in a stage with no _SUCCESS marker).
    existing_per_key: dict[str, set[int]] = {
        key: _existing_shard_indices(output_dir / key) for key in output_keys
    }
    can_skip_idx: set[int] = set()
    if resume and existing_per_key:
        # A batch index is "already done" iff it has a non-empty shard
        # in EVERY output key.
        can_skip_idx = set.intersection(*existing_per_key.values())

    worker_kwargs = worker_kwargs or {}
    emit_start(
        progress_cb,
        stage_label,
        total=total,
        message=(
            f"{total} batches, {n_workers} worker(s)"
            if total is not None
            else f"streaming, {n_workers} worker(s)"
        ),
        payload={"resume": resume, "skipped_idx_count": len(can_skip_idx)},
    )

    completed_paths: dict[str, list[tuple[int, Path]]] = {
        key: [] for key in output_keys
    }
    n_done = 0

    # Pre-populate completed_paths with the resume-skipped shards so
    # the returned StageOutput is order-stable across runs.
    for idx in can_skip_idx:
        for key in output_keys:
            shard = output_dir / key / _shard_filename(idx)
            if shard.is_file() and shard.stat().st_size > 0:
                completed_paths[key].append((idx, shard))

    if can_skip_idx:
        emit_progress(
            progress_cb,
            stage_label,
            completed=len(can_skip_idx),
            total=total,
            message=f"resumed {len(can_skip_idx)} batches",
        )
        n_done = len(can_skip_idx)

    # Index-aware iterator: enumerate the input stream and skip resumed
    # indices on the fly so we never materialise the full batch list.
    enumerated = (
        (idx, batch)
        for idx, batch in enumerate(batches)
        if idx not in can_skip_idx
    )

    def _record(bi: int, shard_map: dict[str, Path]) -> None:
        nonlocal n_done
        for key, path in shard_map.items():
            completed_paths[key].append((bi, path))
        n_done += 1
        emit_progress(
            progress_cb,
            stage_label,
            completed=n_done,
            total=total,
        )

    if n_workers == 1:
        # Single-process path — process each batch immediately and let
        # it go out of scope before pulling the next one.
        for idx, batch in enumerated:
            bi, shard_map = _worker_run(
                worker_fn,
                worker_kwargs,
                output_dir,
                output_keys,
                idx,
                batch,
            )
            _record(bi, shard_map)
    else:
        max_in_flight = n_workers * in_flight_per_worker
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            in_flight: set[Future] = set()
            for idx, batch in enumerated:
                if len(in_flight) >= max_in_flight:
                    done, in_flight = wait(
                        in_flight, return_when=FIRST_COMPLETED
                    )
                    for fut in done:
                        bi, shard_map = fut.result()
                        _record(bi, shard_map)
                in_flight.add(
                    ex.submit(
                        _worker_run,
                        worker_fn,
                        worker_kwargs,
                        output_dir,
                        output_keys,
                        idx,
                        batch,
                    )
                )
                # Drop the local reference so the parent's copy of the
                # batch can be GC'd as soon as the child receives it.
                del batch
            # Drain remaining futures.
            for fut in in_flight:
                bi, shard_map = fut.result()
                _record(bi, shard_map)

    # Sort each key's shard list by batch index so callers reading the
    # dataset see deterministic row order.
    result: dict[str, StageOutput] = {}
    for key in output_keys:
        sorted_shards = sorted(completed_paths[key], key=lambda p: p[0])
        result[key] = StageOutput(
            key=key,
            directory=output_dir / key,
            shard_paths=tuple(path for _idx, path in sorted_shards),
        )
        # Mark the stage complete.
        _success_marker(output_dir / key).touch()

    emit_done(
        progress_cb,
        stage_label,
        completed=n_done,
        total=total if total is not None else n_done,
        message=f"wrote {n_done} shard(s) per output key",
    )
    return result


def read_dataset(stage_output: StageOutput) -> pa.Table:
    """Read a stage's full output as a single concatenated table.

    Use when the downstream consumer needs random access; for streaming
    use, prefer ``pa.dataset.dataset(stage_output.directory)`` directly.
    Reading by explicit shard-path list (rather than directory glob)
    preserves the batch-index ordering.
    """
    if not stage_output.shard_paths:
        raise ValueError(
            f"stage output {stage_output.key!r} has no shards in "
            f"{stage_output.directory}"
        )
    tables = [pq.read_table(p) for p in stage_output.shard_paths]
    return pa.concat_tables(tables)


__all__ = [
    "StageOutput",
    "WorkerFn",
    "read_dataset",
    "run_batched",
]
