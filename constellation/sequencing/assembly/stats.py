"""Assembly summary statistics — N50 / L50 / GC / BUSCO completeness.

Pure-functional helpers that take ``Assembly`` (or a contig-length
list) and produce ``ASSEMBLY_STATS``-shaped Arrow rows. BUSCO
completeness is the costly part — defer-loaded so contigs-only stats
don't need a BUSCO install.

Status: STUB. Pending Phase 7 (assembly/{hifiasm, stats}).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from constellation.sequencing.assembly.assembly import Assembly


_PHASE = "Phase 7 (assembly/stats)"


def assembly_stats(assembly: Assembly) -> pa.Table:
    """Return an ``ASSEMBLY_STATS``-shaped Arrow table with one row.

    Computes N50 / L50 / N90 / L90 / total length / largest contig /
    GC content from ``assembly.contigs`` + ``assembly.sequences``.
    BUSCO columns are left null — populate via :func:`busco_stats`.
    """
    raise NotImplementedError(f"assembly_stats pending {_PHASE}")


def n50(lengths: list[int]) -> tuple[int, int]:
    """Classic N50 / L50 from a list of contig lengths.

    Returns ``(n50_length, l50_count)`` — the length of the contig at
    which cumulative-largest-first sum ≥ 50% of total, and the number
    of contigs needed to reach that point.
    """
    raise NotImplementedError(f"n50 pending {_PHASE}")


def gc_content(sequences: pa.Table) -> float:
    """Fraction of G+C bases across all sequences (excludes N)."""
    raise NotImplementedError(f"gc_content pending {_PHASE}")


def busco_stats(
    assembly: Assembly,
    *,
    lineage: str,
    output_dir: Path,
    threads: int = 8,
) -> pa.Table:
    """Run BUSCO and return the busco_* columns of ASSEMBLY_STATS.

    Lineage-data path is resolved via :func:`thirdparty.find('busco')`
    plus the ``BUSCO_DOWNLOADS_PATH`` env var; missing lineage data
    triggers a hard error directing the user to ``busco --download``.
    """
    raise NotImplementedError(f"busco_stats pending {_PHASE}")


__all__ = [
    "assembly_stats",
    "n50",
    "gc_content",
    "busco_stats",
]
