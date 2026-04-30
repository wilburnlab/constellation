"""BAM-canonical alignment schemas.

``ALIGNMENT_TABLE`` carries the fixed columns every BAM alignment has,
plus the 5–6 most-queried optional tags promoted to typed columns
(``nm_tag``, ``as_tag``, ``read_group``). The long tail of optional
BAM tags (``SA``, ``MD``, ``MM``, ``ML``, custom user tags) lives in
``ALIGNMENT_TAG_TABLE`` as long-format ``(alignment_id, tag, type,
value)`` rows. Common-tag queries don't pay a join cost; rare-tag
queries pay a small one.

CIGAR is kept as the string form (``"50M5I20M"``) because that's what
BAM stores, what samtools tooling expects, and what round-trips
losslessly. ``cigar_to_ops`` exposes a parsed-on-demand view as an
Arrow ``list<struct{op, length}>`` for queryability — but we don't
materialize an exploded ops table by default (millions of reads × tens
of ops per read is a 10–100× row blowup we never want eagerly).

PAF (minimap2's tab-separated output) lands in this same schema —
PAF is a strict subset of BAM in column space, so a separate
``PAF_TABLE`` would be redundant. The ``readers/paf.py`` reader fills
the BAM-only columns with nulls.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema


# ──────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────


ALIGNMENT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("alignment_id", pa.int64(), nullable=False),
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("ref_name", pa.string(), nullable=False),
        # 0-based half-open like BAM internal coords
        pa.field("ref_start", pa.int64(), nullable=False),
        pa.field("ref_end", pa.int64(), nullable=False),
        # '+' | '-'
        pa.field("strand", pa.string(), nullable=False),
        pa.field("mapq", pa.int32(), nullable=False),
        # SAM flag bitmask (0x4 unmapped, 0x10 reverse, 0x100 secondary, ...)
        pa.field("flag", pa.int32(), nullable=False),
        pa.field("cigar_string", pa.string(), nullable=False),
        # Promoted-from-tags numeric columns (commonly queried)
        pa.field("nm_tag", pa.int32(), nullable=True),
        pa.field("as_tag", pa.float32(), nullable=True),
        pa.field("read_group", pa.string(), nullable=True),
        pa.field("is_secondary", pa.bool_(), nullable=False),
        pa.field("is_supplementary", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_name": b"AlignmentTable"},
)


ALIGNMENT_TAG_TABLE: pa.Schema = pa.schema(
    [
        pa.field("alignment_id", pa.int64(), nullable=False),
        # Two-character BAM tag — "MD", "SA", "MM", "ML", custom Xx
        pa.field("tag", pa.string(), nullable=False),
        # Single-char BAM type code — 'i', 'f', 'Z', 'B', 'A', 'H'
        pa.field("type", pa.string(), nullable=False),
        # Always serialized as string; cast on read to ``type``-coded value
        pa.field("value", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"AlignmentTagTable"},
)


register_schema("AlignmentTable", ALIGNMENT_TABLE)
register_schema("AlignmentTagTable", ALIGNMENT_TAG_TABLE)


# ──────────────────────────────────────────────────────────────────────
# CIGAR derived view
# ──────────────────────────────────────────────────────────────────────


_PHASE = "Phase 2 (Reader/Writer Protocols + Phred + readers/{fastx, sam_bam, pod5})"


def cigar_to_ops(cigar_string: str) -> pa.Array:
    """Parse a CIGAR string into a ``list<struct{op: string, length:
    int32}>`` Arrow array.

    Used as a derived view — call on demand for queryability without
    storing exploded rows. Single-string variant returns a scalar list;
    Phase 2 will add a vectorized ``cigar_table_to_ops(table) ->
    pa.Array`` for whole-column expansion.
    """
    raise NotImplementedError(f"cigar_to_ops pending {_PHASE}")


__all__ = [
    "ALIGNMENT_TABLE",
    "ALIGNMENT_TAG_TABLE",
    "cigar_to_ops",
]
