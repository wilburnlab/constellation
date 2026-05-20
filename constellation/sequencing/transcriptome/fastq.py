"""Per-sample FASTQ emission from an S1 demux output directory.

A thin sink that consumes the join + window-slice stream from
:func:`constellation.sequencing.align.map._iter_demux_read_batches`
and routes records to one gzipped FASTQ per ``sample_name``.
Aggregation across acquisitions / barcodes that share a ``sample_id``
is automatic because ``sample_id`` is the routing key.

Output layout::

    <demux_dir>/fastq/
        _SUCCESS                       (stage-resume marker)
        <sample_name_1>.fq.gz
        <sample_name_2>.fq.gz
        ...

Filter policy (matches ``_iter_demux_read_batches(only_complete=True)``):
``status == 'Complete'`` + ``sample_id`` not null + ``is_fragment == False``
+ ``is_chimera == False`` + valid transcript window.  Reads with
``sample_id IS NULL`` are skipped silently.
"""

from __future__ import annotations

import gzip
import os
import re
from collections import defaultdict
from pathlib import Path

from constellation.sequencing.align.map import _iter_demux_read_batches
from constellation.sequencing.progress import (
    ProgressCallback,
    emit_done,
    emit_progress,
    emit_start,
)
from constellation.sequencing.samples import Samples


_SANITIZE_RE = re.compile(r"[^\w.\-]")
# Hard-coded write-once compression level — trades ~5% size for ~2×
# throughput vs default 9.  Users who need maximum compression can
# re-gzip out-of-band.
_GZIP_COMPRESSLEVEL = 4


def _sanitize_name(name: str) -> str:
    """Map a sample_name to a filesystem-safe basename.

    Replaces any character outside ``[A-Za-z0-9_.-]`` with ``_``.
    """
    return _SANITIZE_RE.sub("_", name)


def _build_sample_filename_map(samples: Samples) -> dict[int, str]:
    """Resolve ``sample_id -> sanitised filename stem``.

    Raises ``ValueError`` if two distinct ``sample_name`` values
    sanitise to the same string — better to fail loudly than silently
    overwrite one sample's reads with another's.
    """
    ids = samples.samples.column("sample_id").to_pylist()
    names = samples.samples.column("sample_name").to_pylist()
    sanitised = [_sanitize_name(n) for n in names]

    collisions: dict[str, list[str]] = defaultdict(list)
    for original, clean in zip(names, sanitised):
        collisions[clean].append(original)
    duplicates = {clean: origs for clean, origs in collisions.items() if len(origs) > 1}
    if duplicates:
        raise ValueError(
            "sample_name values sanitise to colliding filenames: "
            + "; ".join(f"{clean!r} <- {origs}" for clean, origs in duplicates.items())
        )

    return dict(zip(ids, sanitised))


def emit_per_sample_fastq(
    demux_dir: Path,
    *,
    samples: Samples,
    batch_size: int = 100_000,
    progress_cb: ProgressCallback | None = None,
    resume: bool = False,
) -> Path:
    """Emit one ``.fq.gz`` per sample from an S1 demux output directory.

    ``demux_dir`` must contain partitioned ``reads/`` and ``read_demux/``
    datasets (the standard output of ``run_demux_pipeline``).  Writes
    each sample's reads to ``demux_dir/fastq/<sample_name>.fq.gz`` using
    transcript-window slicing + the strict Complete-only filter; reads
    with ``sample_id IS NULL`` are dropped silently.

    Resume semantics: if ``resume=True`` and
    ``demux_dir/fastq/_SUCCESS`` already exists, returns immediately
    without re-scanning any data.

    Atomicity: each sample writes to ``<name>.fq.gz.tmp`` and is renamed
    via ``os.replace`` after its last write.  ``_SUCCESS`` is touched
    only after every handle closes cleanly — a crashed run leaves
    ``.tmp`` files that a resumed run will overwrite.

    Returns the ``fastq/`` directory path.
    """
    demux_dir = Path(demux_dir)
    fastq_dir = demux_dir / "fastq"
    success_marker = fastq_dir / "_SUCCESS"

    if resume and success_marker.is_file():
        emit_done(
            progress_cb,
            "emit_fastq",
            message=f"resumed (existing fastq/ at {fastq_dir})",
        )
        return fastq_dir

    fastq_dir.mkdir(parents=True, exist_ok=True)
    # Drop any stale _SUCCESS from an earlier aborted/overwriting run so
    # we never advertise completion until this invocation finishes.
    if success_marker.is_file():
        success_marker.unlink()

    name_for_id = _build_sample_filename_map(samples)
    n_samples_known = len(name_for_id)

    emit_start(
        progress_cb,
        "emit_fastq",
        message=f"emitting per-sample FASTQ for up to {n_samples_known} samples",
    )

    handles: dict[int, gzip.GzipFile] = {}
    tmp_paths: dict[int, Path] = {}
    final_paths: dict[int, Path] = {}
    reads_written = 0

    try:
        for batch in _iter_demux_read_batches(
            demux_dir, only_complete=True, batch_size=batch_size
        ):
            per_sample_parts: dict[int, list[str]] = defaultdict(list)
            for rid, sid, seq, qual in zip(
                batch["read_id"],
                batch["sample_id"],
                batch["sequence"],
                batch["quality"],
            ):
                per_sample_parts[sid].append(f"@{rid}\n{seq}\n+\n{qual}\n")

            for sid, parts in per_sample_parts.items():
                handle = handles.get(sid)
                if handle is None:
                    stem = name_for_id.get(sid)
                    if stem is None:
                        raise KeyError(
                            f"read_demux references sample_id={sid} not present "
                            f"in Samples; cannot resolve a filename"
                        )
                    final = fastq_dir / f"{stem}.fq.gz"
                    tmp = fastq_dir / f"{stem}.fq.gz.tmp"
                    handle = gzip.open(
                        tmp, "wb", compresslevel=_GZIP_COMPRESSLEVEL
                    )
                    handles[sid] = handle
                    tmp_paths[sid] = tmp
                    final_paths[sid] = final
                handle.write("".join(parts).encode("ascii"))

            reads_written += sum(len(parts) for parts in per_sample_parts.values())
            emit_progress(
                progress_cb,
                "emit_fastq",
                completed=reads_written,
                message=f"{len(handles)} samples open",
            )
    finally:
        for handle in handles.values():
            try:
                handle.close()
            except OSError:
                pass

    for sid, tmp in tmp_paths.items():
        os.replace(tmp, final_paths[sid])

    success_marker.touch()

    emit_done(
        progress_cb,
        "emit_fastq",
        completed=reads_written,
        message=f"{len(final_paths)} files in {fastq_dir}",
    )
    return fastq_dir


__all__ = ["emit_per_sample_fastq"]
