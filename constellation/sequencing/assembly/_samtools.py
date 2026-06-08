"""Small samtools helpers shared by the assembly stages (sort / index /
fastq). Resolves the binary via the thirdparty registry.
"""

from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

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


def samtools_fastq(in_bam: Path, out_fastq_gz: Path, *, threads: int = 8) -> Path:
    """Convert ``in_bam`` (unaligned long reads) → gzipped FASTQ.

    Streams ``samtools fastq`` through Python gzip (no external ``gzip``
    dependency). Single-end long reads all go to one stream.
    """
    samtools = str(resolve_samtools())
    out = Path(out_fastq_gz)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [samtools, "fastq", "-@", str(int(threads)), str(in_bam)],
        stdout=subprocess.PIPE,
    )
    try:
        with gzip.open(out, "wb") as gz:
            assert proc.stdout is not None
            for chunk in iter(lambda: proc.stdout.read(1 << 20), b""):
                gz.write(chunk)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, ["samtools", "fastq"])
    return out


__all__ = [
    "resolve_samtools",
    "samtools_sort",
    "samtools_index",
    "samtools_fastq",
]
