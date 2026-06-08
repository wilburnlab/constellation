"""Assembly summary statistics — N50 / L50 / GC / BUSCO completeness.

The first real Constellation analysis code in the genome-assembly arm
(everything upstream is tool wrapping). Pure-functional helpers that
take an ``Assembly`` (or its component tables) and produce an
``ASSEMBLY_STATS``-shaped one-row Arrow table. Contiguity stats (N50 /
L50 / N90 / L90 / total / largest / GC) are computed Arrow/numpy-native
with no per-base Python loop — GC counting runs at C-level over the
whole sequence column via ``pc.count_substring``, and contig-length
statistics are vectorised numpy over the (small) per-contig length
vector. BUSCO completeness is the costly part — defer-loaded so
contigs-only stats don't need a BUSCO install.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import cast_to_schema
from constellation.sequencing.schemas.assembly import ASSEMBLY_STATS

if TYPE_CHECKING:  # avoid a runtime cycle (assembly.py imports this module)
    from constellation.sequencing.assembly.assembly import Assembly


# ──────────────────────────────────────────────────────────────────────
# Contiguity
# ──────────────────────────────────────────────────────────────────────


def _nx(lengths: Iterable[int], fraction: float) -> tuple[int, int]:
    """Generic Nx / Lx: the largest-first contig that pushes the
    cumulative length past ``fraction`` of the total, and how many
    contigs that took. ``fraction=0.5`` is the classic N50 / L50."""
    arr = np.asarray(list(lengths), dtype=np.int64)
    if arr.size == 0:
        return (0, 0)
    arr = np.sort(arr)[::-1]
    total = int(arr.sum())
    if total <= 0:
        return (0, 0)
    cum = np.cumsum(arr)
    idx = int(np.searchsorted(cum, fraction * total, side="left"))
    if idx >= arr.size:
        idx = arr.size - 1
    return (int(arr[idx]), idx + 1)


def n50(lengths: list[int]) -> tuple[int, int]:
    """Classic N50 / L50 from a list of contig lengths.

    Returns ``(n50_length, l50_count)`` — the length of the contig at
    which the cumulative largest-first sum first reaches 50% of the
    total, and the number of contigs needed to get there.
    """
    return _nx(lengths, 0.5)


def gc_content(sequences: pa.Table) -> float:
    """Fraction of G+C bases across all sequences, excluding N.

    Counts case-insensitively (soft-masked lowercase included) at
    C-level over the whole ``SEQUENCE_TABLE.sequence`` column — no
    per-row Python decode, no whole-genome materialisation. Denominator
    is A+C+G+T (everything non-ACGT, including N and IUPAC ambiguity
    codes, is excluded). Returns 0.0 for an empty / all-N assembly.
    """
    col = sequences.column("sequence")

    def _count(ch: str) -> int:
        s = pc.sum(pc.count_substring(col, ch, ignore_case=True)).as_py()
        return int(s) if s else 0

    g = _count("G")
    c = _count("C")
    a = _count("A")
    t = _count("T")
    gc = g + c
    acgt = a + c + g + t
    return gc / acgt if acgt else 0.0


# ──────────────────────────────────────────────────────────────────────
# ASSEMBLY_STATS assembly
# ──────────────────────────────────────────────────────────────────────


def _stats_from_tables(
    contigs: pa.Table,
    sequences: pa.Table,
    scaffolds: pa.Table | None = None,
    *,
    assembly_id: int = 0,
) -> pa.Table:
    """Compute an ``ASSEMBLY_STATS`` one-row table from raw tables.

    Lets ``Assembly.from_tables`` populate ``stats`` at construction time
    without the chicken-and-egg of needing a fully-built ``Assembly``.
    BUSCO columns are left null — populate via :func:`busco_stats`.
    """
    lengths = [int(x) for x in contigs.column("length").to_pylist()]
    n50_len, l50 = _nx(lengths, 0.5)
    n90_len, l90 = _nx(lengths, 0.9)
    total = sum(lengths)
    largest = max(lengths) if lengths else 0

    n_scaffolds: int | None = None
    if scaffolds is not None and scaffolds.num_rows > 0:
        n_scaffolds = len(set(scaffolds.column("scaffold_id").to_pylist()))

    gc: float | None = gc_content(sequences) if sequences.num_rows > 0 else None

    table = pa.table(
        {
            "assembly_id": pa.array([assembly_id], type=pa.int64()),
            "n_contigs": pa.array([contigs.num_rows], type=pa.int32()),
            "n_scaffolds": pa.array([n_scaffolds], type=pa.int32()),
            "total_length": pa.array([total], type=pa.int64()),
            "n50": pa.array([n50_len], type=pa.int64()),
            "l50": pa.array([l50], type=pa.int32()),
            "n90": pa.array([n90_len], type=pa.int64()),
            "l90": pa.array([l90], type=pa.int32()),
            "largest_contig": pa.array([largest], type=pa.int64()),
            "gc_content": pa.array([gc], type=pa.float32()),
        }
    )
    # cast fills the busco_* columns (absent here) with nulls.
    return cast_to_schema(table, ASSEMBLY_STATS)


def assembly_stats(assembly: "Assembly") -> pa.Table:
    """Return an ``ASSEMBLY_STATS``-shaped Arrow table with one row.

    Computes N50 / L50 / N90 / L90 / total length / largest contig /
    GC content from ``assembly.contigs`` + ``assembly.sequences`` (and
    scaffold count from ``assembly.scaffolds`` when present). BUSCO
    columns are left null — populate via :func:`busco_stats`.
    """
    return _stats_from_tables(assembly.contigs, assembly.sequences, assembly.scaffolds)


def busco_stats(
    assembly: "Assembly",
    *,
    lineage: str,
    output_dir: Path,
    threads: int = 8,
) -> pa.Table:
    """Run BUSCO and return a one-row table of the ``busco_*`` columns.

    Delegates to :class:`sequencing.annotation.busco.BuscoRunner`; merge
    the result into a full ASSEMBLY_STATS row via :func:`apply_busco`.
    Lineage data is resolved via ``busco`` + ``BUSCO_DOWNLOADS_PATH``.
    """
    from constellation.sequencing.annotation.busco import BuscoRunner

    stats_row, _features = BuscoRunner(lineage=lineage, threads=threads).run(
        assembly, Path(output_dir)
    )
    return stats_row


def apply_busco(stats: pa.Table, busco_row: pa.Table) -> pa.Table:
    """Fill the ``busco_*`` + ``busco_lineage`` columns of a one-row
    ASSEMBLY_STATS table from a :func:`busco_stats` row."""
    out = stats
    for name in (
        "busco_complete",
        "busco_single",
        "busco_duplicated",
        "busco_fragmented",
        "busco_missing",
        "busco_lineage",
    ):
        if name in busco_row.column_names:
            idx = out.schema.get_field_index(name)
            out = out.set_column(idx, name, busco_row.column(name))
    return out


__all__ = [
    "assembly_stats",
    "n50",
    "gc_content",
    "busco_stats",
    "apply_busco",
]
