"""Tests for ``constellation.sequencing.transcriptome.fingerprints``.

The cluster-key for Phase 2 genome-guided clustering. We're checking:

  1. Reads whose per-read junctions map to the same canonical intron
     IDs collapse to the same fingerprint hash — regardless of whether
     the underlying donor/acceptor positions are bit-identical or only
     within tolerance of an intron seed.
  2. Reads whose junction lists map to different intron IDs produce
     different fingerprint hashes.
  3. Secondary / supplementary alignments are excluded.
  4. Strand differences produce different hashes (strand is part of
     the canonical tuple).
  5. Single-block alignments produce a fingerprint with zero junctions.
  6. Alignments to contigs absent from the genome lookup are dropped
     silently rather than raising.
  7. Reads whose junctions don't appear in the supplied introns table
     are skipped (same silent-drop policy as unknown contigs).
"""

from __future__ import annotations

import pyarrow as pa

from constellation.sequencing.quant.junctions import (
    aggregate_junctions,
    cluster_junctions,
)
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_TABLE,
    INTRON_TABLE,
)
from constellation.sequencing.schemas.reference import CONTIG_TABLE
from constellation.sequencing.transcriptome.fingerprints import (
    compute_read_fingerprints,
)


def _alignment(**overrides) -> dict:
    base = {
        "alignment_id": 0,
        "read_id": "r0",
        "acquisition_id": 1,
        "ref_name": "chr1",
        "ref_start": 0,
        "ref_end": 100,
        "strand": "+",
        "mapq": 60,
        "flag": 0,
        "cigar_string": "100M",
        "nm_tag": None,
        "as_tag": None,
        "read_group": None,
        "is_secondary": False,
        "is_supplementary": False,
    }
    base.update(overrides)
    return base


def _block(**overrides) -> dict:
    base = {
        "alignment_id": 0,
        "block_index": 0,
        "ref_start": 0,
        "ref_end": 100,
        "query_start": 0,
        "query_end": 100,
        "n_match": None,
        "n_mismatch": None,
        "n_insert": 0,
        "n_delete": 0,
    }
    base.update(overrides)
    return base


def _contigs() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"contig_id": 1, "name": "chr1", "length": 1_000_000,
             "topology": None, "circular": None},
            {"contig_id": 2, "name": "chr2", "length": 500_000,
             "topology": None, "circular": None},
        ],
        schema=CONTIG_TABLE,
    )


def _genome() -> GenomeReference:
    """Tiny genome that supplies sequences for motif lookup. The actual
    bases are mostly irrelevant for fingerprint tests; we just need
    contig coverage for ``aggregate_junctions``.
    """
    seq_chr1 = "A" * 1_000_000
    seq_chr2 = "A" * 500_000
    return GenomeReference(
        contigs=_contigs(),
        sequences=pa.Table.from_pylist(
            [
                {"contig_id": 1, "sequence": seq_chr1},
                {"contig_id": 2, "sequence": seq_chr2},
            ]
        ),
    )


def _alignments(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_TABLE)


def _blocks(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_BLOCK_TABLE)


def _introns_for(blocks: pa.Table, alignments: pa.Table,
                  *, tolerance_bp: int = 0) -> pa.Table:
    """Helper: derive the INTRON_TABLE the test's blocks imply.

    Default ``tolerance_bp=0`` produces an identity-clustered table —
    each distinct ``(donor, acceptor)`` pair is its own cluster.
    Tests that want to verify cross-tolerance bucketing pass a larger
    value here.
    """
    raw = aggregate_junctions(blocks, alignments, _genome())
    return cluster_junctions(raw, tolerance_bp=tolerance_bp)


# ──────────────────────────────────────────────────────────────────────


def test_identical_intron_chains_collapse_to_same_hash() -> None:
    """Two reads with the exact same junction sequence get the same hash."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA", ref_start=1000, ref_end=1300),
            _alignment(alignment_id=2, read_id="rB", ref_start=1000, ref_end=1300),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=1, block_index=1, ref_start=1600, ref_end=1800),
            _block(alignment_id=2, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=2, block_index=1, ref_start=1600, ref_end=1800),
        ]
    )
    introns = _introns_for(blocks, al)
    out = compute_read_fingerprints(blocks, al, _contigs(), introns)
    assert out.num_rows == 2
    hashes = out.column("fingerprint_hash").to_pylist()
    assert hashes[0] == hashes[1]


def test_different_intron_chains_produce_different_hashes() -> None:
    """Reads whose junction lists differ get different hashes."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            # rA: junction at (1100, 1600)
            _block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=1, block_index=1, ref_start=1600, ref_end=1800),
            # rB: junction at (1100, 1700) — different acceptor by 100 bp
            _block(alignment_id=2, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=2, block_index=1, ref_start=1700, ref_end=1900),
        ]
    )
    introns = _introns_for(blocks, al, tolerance_bp=10)
    out = compute_read_fingerprints(blocks, al, _contigs(), introns)
    hashes = {r["read_id"]: r["fingerprint_hash"] for r in out.to_pylist()}
    assert hashes["rA"] != hashes["rB"]


def test_clustered_excursion_collapses_to_same_hash() -> None:
    """Two reads with junctions within the cluster tolerance share a hash.

    Replaces the old ``test_cryptic_excursion_within_quantum_collapses``.
    Donor positions 1100 and 1106 are 6 bp apart; with tolerance=10 they
    cluster into a single intron_id, so both reads' fingerprint
    sequences become ``[same_intron_id]``.
    """
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=1, block_index=1, ref_start=1600, ref_end=1800),
            _block(alignment_id=2, block_index=0, ref_start=1000, ref_end=1106),
            _block(alignment_id=2, block_index=1, ref_start=1600, ref_end=1800),
        ]
    )
    introns = _introns_for(blocks, al, tolerance_bp=10)
    out = compute_read_fingerprints(blocks, al, _contigs(), introns)
    hashes = {r["read_id"]: r["fingerprint_hash"] for r in out.to_pylist()}
    assert hashes["rA"] == hashes["rB"]


def test_excursion_beyond_tolerance_does_not_collapse() -> None:
    """Donor delta exceeding tolerance keeps the reads in separate clusters."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            # rA: donor 1100
            _block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=1, block_index=1, ref_start=1600, ref_end=1800),
            # rB: donor 1115 (delta = 15 > 10)
            _block(alignment_id=2, block_index=0, ref_start=1000, ref_end=1115),
            _block(alignment_id=2, block_index=1, ref_start=1600, ref_end=1800),
        ]
    )
    introns = _introns_for(blocks, al, tolerance_bp=10)
    out = compute_read_fingerprints(blocks, al, _contigs(), introns)
    hashes = {r["read_id"]: r["fingerprint_hash"] for r in out.to_pylist()}
    assert hashes["rA"] != hashes["rB"]


def test_secondary_and_supplementary_alignments_are_excluded() -> None:
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB", is_secondary=True),
            _alignment(alignment_id=3, read_id="rC", is_supplementary=True),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=0, ref_end=100),
            _block(alignment_id=2, ref_start=0, ref_end=100),
            _block(alignment_id=3, ref_start=0, ref_end=100),
        ]
    )
    # No junctions in any of these alignments — introns table is empty,
    # but the function still has to run cleanly and emit one zero-junction
    # fingerprint for the primary.
    introns = INTRON_TABLE.empty_table()
    out = compute_read_fingerprints(blocks, al, _contigs(), introns)
    read_ids = out.column("read_id").to_pylist()
    assert read_ids == ["rA"]


def test_strand_difference_changes_hash() -> None:
    """Same junctions on opposite strands produce different hashes."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA", strand="+"),
            _alignment(alignment_id=2, read_id="rB", strand="-"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=1, block_index=1, ref_start=1600, ref_end=1800),
            _block(alignment_id=2, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=2, block_index=1, ref_start=1600, ref_end=1800),
        ]
    )
    introns = _introns_for(blocks, al)
    out = compute_read_fingerprints(blocks, al, _contigs(), introns)
    hashes = {r["read_id"]: r["fingerprint_hash"] for r in out.to_pylist()}
    assert hashes["rA"] != hashes["rB"]


def test_single_block_alignment_produces_zero_junction_fingerprint() -> None:
    al = _alignments(
        [_alignment(alignment_id=1, read_id="rA", ref_start=1000, ref_end=1500)]
    )
    blocks = _blocks(
        [_block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1500)]
    )
    # No junctions at all — empty introns table is fine (no lookups happen).
    introns = INTRON_TABLE.empty_table()
    out = compute_read_fingerprints(blocks, al, _contigs(), introns)
    assert out.num_rows == 1
    row = out.to_pylist()[0]
    assert row["n_blocks"] == 1
    assert row["span_start"] == 1000 and row["span_end"] == 1500
    assert row["junction_signature"] == "chr1:."


def test_unknown_contig_is_dropped_not_raised() -> None:
    """An alignment to a contig the genome doesn't list is silently
    dropped (mirrors gene_set_from_annotation's policy).
    """
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA", ref_name="chrX_unknown"),
            _alignment(alignment_id=2, read_id="rB", ref_name="chr1"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=0, ref_end=100),
            _block(alignment_id=2, ref_start=0, ref_end=100),
        ]
    )
    # Single-block alignments → empty introns table.
    introns = INTRON_TABLE.empty_table()
    out = compute_read_fingerprints(blocks, al, _contigs(), introns)
    assert out.column("read_id").to_pylist() == ["rB"]


def test_blocks_out_of_order_are_sorted_internally() -> None:
    """If alignment_blocks rows come in shuffled order, the fingerprint
    is still the canonical one (sorted by block_index).
    """
    al_in_order = _alignments(
        [_alignment(alignment_id=1, read_id="rA")]
    )
    blocks_in_order = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=1, block_index=1, ref_start=1600, ref_end=1800),
            _block(alignment_id=1, block_index=2, ref_start=2000, ref_end=2100),
        ]
    )
    blocks_shuffled = _blocks(
        [
            _block(alignment_id=1, block_index=2, ref_start=2000, ref_end=2100),
            _block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=1, block_index=1, ref_start=1600, ref_end=1800),
        ]
    )
    introns_a = _introns_for(blocks_in_order, al_in_order)
    introns_b = _introns_for(blocks_shuffled, al_in_order)
    out_a = compute_read_fingerprints(
        blocks_in_order, al_in_order, _contigs(), introns_a
    )
    out_b = compute_read_fingerprints(
        blocks_shuffled, al_in_order, _contigs(), introns_b
    )
    assert (
        out_a.column("fingerprint_hash").to_pylist()
        == out_b.column("fingerprint_hash").to_pylist()
    )


def test_empty_inputs_return_empty_schema_shaped_table() -> None:
    al = ALIGNMENT_TABLE.empty_table()
    blocks = ALIGNMENT_BLOCK_TABLE.empty_table()
    introns = INTRON_TABLE.empty_table()
    out = compute_read_fingerprints(blocks, al, _contigs(), introns)
    assert out.num_rows == 0
    from constellation.sequencing.schemas.alignment import READ_FINGERPRINT_TABLE
    assert out.schema.equals(READ_FINGERPRINT_TABLE)


def test_read_with_unknown_junction_is_dropped() -> None:
    """A read whose per-read junction has no exact match in the supplied
    introns table is skipped — no fingerprint row emitted.

    This is the silent-drop policy for mismatched inputs (e.g. introns
    table from a different alignment run, or filtered to high-support
    only). Mismatched inputs surface as "fewer fingerprints than reads."
    """
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            # rA: junction at (1100, 1600)
            _block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=1, block_index=1, ref_start=1600, ref_end=1800),
            # rB: junction at (5000, 6000) — a different junction
            _block(alignment_id=2, block_index=0, ref_start=4500, ref_end=5000),
            _block(alignment_id=2, block_index=1, ref_start=6000, ref_end=6500),
        ]
    )
    # Hand-crafted introns table that only includes rA's junction.
    introns = pa.Table.from_pylist(
        [
            {
                "intron_id": 0, "contig_id": 1, "strand": "+",
                "donor_pos": 1100, "acceptor_pos": 1600, "read_count": 1,
                "motif": "AA-AA", "is_intron_seed": True, "annotated": None,
            },
        ],
        schema=INTRON_TABLE,
    )
    out = compute_read_fingerprints(blocks, al, _contigs(), introns)
    # Only rA emits a fingerprint; rB is dropped because its (5000, 6000)
    # junction has no INTRON_TABLE entry.
    assert out.column("read_id").to_pylist() == ["rA"]
