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

Alignment-derived per-block / aggregated views (Phase 1 power-user
toggles on ``transcriptome align``):

    ALIGNMENT_BLOCK_TABLE   one row per CIGAR-derived alignment block
                            (M/=/X run between N or large I/D ops).
                            Per-record exon view; primary input to
                            pile-up, junctions, and fingerprints.
    SPLICE_JUNCTION_TABLE   cross-read aggregate keyed on
                            ``(contig_id, donor_pos, acceptor_pos,
                            strand)``. Junctions are aggregated rather
                            than exons because splice sites are *steep*
                            (canonical GT-AG dinucleotide-anchored)
                            while exon termini are fuzzy on first/last
                            blocks (TSS / SSP / polyA scatter).
    READ_FINGERPRINT_TABLE  per-read canonical junction-sequence hash;
                            primary alignments only. The Phase 2
                            cluster key.
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


# ──────────────────────────────────────────────────────────────────────
# Alignment-derived views (Phase 1 power-user toggles)
# ──────────────────────────────────────────────────────────────────────


# One row per CIGAR-derived alignment block. A "block" is a contiguous
# M/=/X run bounded by N (intron) or large D operations (long I never
# breaks a block — insertions add bases to the query within the same
# aligned region). ``n_match``/``n_mismatch`` are populated from
# cs:long when present; nullable when only CIGAR is available.
# Coordinates are 0-based half-open.
#
# Contig is NOT a column here — block rows join back to ``ALIGNMENT_TABLE``
# via ``alignment_id`` for ``ref_name``; aggregations into
# ``SPLICE_JUNCTION_TABLE`` / ``READ_FINGERPRINT_TABLE`` resolve
# ``ref_name → contig_id`` at aggregation time (where the
# GenomeReference is in scope). Saves one column × N_blocks rows
# (~600M rows at PromethION scale).
ALIGNMENT_BLOCK_TABLE: pa.Schema = pa.schema(
    [
        pa.field("alignment_id", pa.int64(), nullable=False),
        pa.field("block_index", pa.int32(), nullable=False),
        pa.field("ref_start", pa.int64(), nullable=False),
        pa.field("ref_end", pa.int64(), nullable=False),
        # Query coords are on the trimmed transcript window that S2
        # streamed into minimap2 (NOT on the raw read).
        pa.field("query_start", pa.int32(), nullable=False),
        pa.field("query_end", pa.int32(), nullable=False),
        pa.field("n_match", pa.int32(), nullable=True),
        pa.field("n_mismatch", pa.int32(), nullable=True),
        pa.field("n_insert", pa.int32(), nullable=False),
        pa.field("n_delete", pa.int32(), nullable=False),
    ],
    metadata={b"schema_name": b"AlignmentBlockTable"},
)


# Cross-read aggregate. One row per distinct (contig, donor, acceptor,
# strand) tuple observed across the alignment_blocks input. ``motif``
# is derived from the genome at donor[0:2] / acceptor[-2:] when a
# GenomeReference is available (e.g. 'GT-AG', 'GC-AG', 'AT-AC',
# 'other'). ``annotated`` is True iff the (donor_pos, acceptor_pos)
# pair matches an intron implied by an Annotation FEATURE_TABLE.
SPLICE_JUNCTION_TABLE: pa.Schema = pa.schema(
    [
        pa.field("junction_id", pa.int64(), nullable=False),
        pa.field("contig_id", pa.int64(), nullable=False),
        pa.field("donor_pos", pa.int64(), nullable=False),
        pa.field("acceptor_pos", pa.int64(), nullable=False),
        # '+' | '-' | '?' (when not splice-motif-disambiguated)
        pa.field("strand", pa.string(), nullable=False),
        pa.field("read_count", pa.int32(), nullable=False),
        pa.field("motif", pa.string(), nullable=True),
        pa.field("annotated", pa.bool_(), nullable=True),
        pa.field("gene_id", pa.int64(), nullable=True),
    ],
    metadata={b"schema_name": b"SpliceJunctionTable"},
)


# Per-read canonical splicing-topology hash. Primary alignments only
# (secondary/supplementary excluded). ``fingerprint_hash`` is computed
# from the canonical tuple ``(contig_id, strand, [(donor//q)*q,
# (acceptor//q)*q for each junction])`` where ``q`` is the
# ``intron_quantum_bp`` knob (default 10) — quantisation absorbs small
# cryptic-splice excursions so they share a fingerprint bucket while
# leaving larger excursions to surface as cluster discordance.
# ``junction_signature`` is a human-readable form for diagnostics, not
# the cluster key.
READ_FINGERPRINT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("contig_id", pa.int64(), nullable=False),
        pa.field("strand", pa.string(), nullable=False),
        pa.field("n_blocks", pa.int32(), nullable=False),
        pa.field("span_start", pa.int64(), nullable=False),
        pa.field("span_end", pa.int64(), nullable=False),
        pa.field("fingerprint_hash", pa.uint64(), nullable=False),
        pa.field("junction_signature", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"ReadFingerprintTable"},
)


# Sidecar to ALIGNMENT_TABLE keyed on alignment_id. minimap2's cs:long
# tag is the per-base substitution / indel record needed for downstream
# weighted-PWM consensus building (Phase 2's `--build-consensus` and
# Phase 3's de novo final stage). Phase 1's worker captures cs from the
# BAM stream when computing ALIGNMENT_BLOCK_TABLE; without this sidecar
# the cs string is consumed and dropped. Off by default behind the
# `transcriptome align --emit-cs-tags` toggle — small additional shard
# write, no change to the existing fast path.
ALIGNMENT_CS_TABLE: pa.Schema = pa.schema(
    [
        pa.field("alignment_id", pa.int64(), nullable=False),
        pa.field("cs_string", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"AlignmentCsTable"},
)


register_schema("AlignmentTable", ALIGNMENT_TABLE)
register_schema("AlignmentTagTable", ALIGNMENT_TAG_TABLE)
register_schema("AlignmentBlockTable", ALIGNMENT_BLOCK_TABLE)
register_schema("SpliceJunctionTable", SPLICE_JUNCTION_TABLE)
register_schema("ReadFingerprintTable", READ_FINGERPRINT_TABLE)
register_schema("AlignmentCsTable", ALIGNMENT_CS_TABLE)


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
    "ALIGNMENT_BLOCK_TABLE",
    "SPLICE_JUNCTION_TABLE",
    "READ_FINGERPRINT_TABLE",
    "ALIGNMENT_CS_TABLE",
    "cigar_to_ops",
]
