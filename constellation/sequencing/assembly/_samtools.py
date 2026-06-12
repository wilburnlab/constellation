"""Small samtools helpers shared by the assembly stages (sort / index /
fastq). Resolves the binary via the thirdparty registry.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from constellation.thirdparty.compress import compress_stream_to_path
from constellation.thirdparty.registry import ToolNotFoundError, find


_HINT = (
    "samtools not found; install via bioconda "
    "(`conda install -c bioconda samtools`) or set $CONSTELLATION_SAMTOOLS_HOME"
)


def resolve_samtools() -> Path:
    try:
        return find("samtools").path
    except ToolNotFoundError as exc:
        raise FileNotFoundError(_HINT) from exc


def samtools_sort(in_bam: Path, out_bam: Path, *, threads: int = 8) -> Path:
    """Coordinate-sort ``in_bam`` → ``out_bam``."""
    samtools = str(resolve_samtools())
    out_bam = Path(out_bam)
    out_bam.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [samtools, "sort", "-@", str(int(threads)), "-o", str(out_bam), str(in_bam)],
        check=True,
    )
    return out_bam


def samtools_index(bam: Path, *, threads: int = 8) -> Path:
    """Index ``bam`` → ``bam.bai``."""
    samtools = str(resolve_samtools())
    subprocess.run(
        [samtools, "index", "-@", str(int(threads)), str(bam)],
        check=True,
    )
    return Path(str(bam) + ".bai")


def samtools_fastq(
    in_bam: Path, out_fastq: Path, *, threads: int = 8, codec: str = "bgzf"
) -> Path:
    """Convert ``in_bam`` (unaligned long reads) → FASTQ. Single-end long
    reads all go to one stream.

    ``codec`` selects compression of the output:

      * ``"bgzf"`` (default) — pipe ``samtools fastq`` into ``bgzip`` for
        multithreaded BGZF compression (gzip-compatible; read natively by
        hifiasm). Falls back to single-threaded stdlib gzip if ``bgzip`` is
        unavailable. This replaces the single-threaded Python-gzip sink that
        was ~90% of a multi-hour assembly job.
      * ``"gzip"`` — force the stdlib-gzip path (plain gzip, single-threaded).
      * ``"none"`` — write plain uncompressed FASTQ via ``samtools fastq -o``
        (samtools writes the file itself; zero compression CPU). The caller is
        responsible for giving ``out_fastq`` a non-``.gz`` name.

    Compression-heavy thread split: ``samtools fastq`` decode is I/O-light, so
    it gets at most 2 threads and the remainder goes to ``bgzip``.
    """
    samtools = str(resolve_samtools())
    out = Path(out_fastq)
    out.parent.mkdir(parents=True, exist_ok=True)
    threads = max(1, int(threads))

    if codec == "none":
        subprocess.run(
            [samtools, "fastq", "-@", str(threads), "-o", str(out), str(in_bam)],
            check=True,
        )
        return out

    decode_threads = min(threads, 2)
    compress_threads = max(1, threads - 2)
    producer_cmd = [samtools, "fastq", "-@", str(decode_threads), str(in_bam)]
    compress_stream_to_path(
        producer_cmd,
        out,
        threads=compress_threads,
        allow_bgzip=(codec != "gzip"),
        stage="fastq",
    )
    return out


__all__ = [
    "resolve_samtools",
    "samtools_sort",
    "samtools_index",
    "samtools_fastq",
]
