"""Tests for ``constellation.sequencing.transcriptome.fingerprints``.

The cluster-key for Phase 2 genome-guided clustering. We're checking:

  1. Reads with identical intron chains (down to the quantum) collapse
     to the same fingerprint hash.
  2. Reads whose intron chain differs by more than the quantum produce
     different fingerprint hashes.
  3. Cryptic-splice excursions within ±``intron_quantum_bp`` collapse;
     larger excursions do not.
  4. Secondary / supplementary alignments are excluded.
  5. Strand differences produce different hashes (even with identical
     junction coordinates).
  6. Single-block alignments produce a fingerprint with zero junctions.
  7. Alignments to contigs absent from the genome lookup are dropped
     silently rather than raising.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_TABLE,
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


def _alignments(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_TABLE)


def _blocks(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_BLOCK_TABLE)


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
    out = compute_read_fingerprints(blocks, al, _contigs())
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
    out = compute_read_fingerprints(blocks, al, _contigs())
    hashes = {r["read_id"]: r["fingerprint_hash"] for r in out.to_pylist()}
    assert hashes["rA"] != hashes["rB"]


def test_cryptic_excursion_within_quantum_collapses() -> None:
    """A 6-bp donor shift is absorbed at q=10."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            # rA: junction donor at 1100
            _block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=1, block_index=1, ref_start=1600, ref_end=1800),
            # rB: junction donor at 1106 (6 bp shift)
            _block(alignment_id=2, block_index=0, ref_start=1000, ref_end=1106),
            _block(alignment_id=2, block_index=1, ref_start=1600, ref_end=1800),
        ]
    )
    out = compute_read_fingerprints(blocks, al, _contigs(), intron_quantum_bp=10)
    hashes = {r["read_id"]: r["fingerprint_hash"] for r in out.to_pylist()}
    assert hashes["rA"] == hashes["rB"]


def test_cryptic_excursion_beyond_quantum_does_not_collapse() -> None:
    """A 15-bp donor shift crosses the q=10 quantum boundary."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            # rA: donor 1100 → quantises to 1100
            _block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1100),
            _block(alignment_id=1, block_index=1, ref_start=1600, ref_end=1800),
            # rB: donor 1115 → quantises to 1110 (different bucket)
            _block(alignment_id=2, block_index=0, ref_start=1000, ref_end=1115),
            _block(alignment_id=2, block_index=1, ref_start=1600, ref_end=1800),
        ]
    )
    out = compute_read_fingerprints(blocks, al, _contigs(), intron_quantum_bp=10)
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
    out = compute_read_fingerprints(blocks, al, _contigs())
    read_ids = out.column("read_id").to_pylist()
    assert read_ids == ["rA"]


def test_strand_difference_changes_hash() -> None:
    """Same junctions on opposite strands produce different hashes
    (the strand is part of the canonical tuple).
    """
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
    out = compute_read_fingerprints(blocks, al, _contigs())
    hashes = {r["read_id"]: r["fingerprint_hash"] for r in out.to_pylist()}
    assert hashes["rA"] != hashes["rB"]


def test_single_block_alignment_produces_zero_junction_fingerprint() -> None:
    al = _alignments(
        [_alignment(alignment_id=1, read_id="rA", ref_start=1000, ref_end=1500)]
    )
    blocks = _blocks(
        [_block(alignment_id=1, block_index=0, ref_start=1000, ref_end=1500)]
    )
    out = compute_read_fingerprints(blocks, al, _contigs())
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
    out = compute_read_fingerprints(blocks, al, _contigs())
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
    out_a = compute_read_fingerprints(blocks_in_order, al_in_order, _contigs())
    out_b = compute_read_fingerprints(blocks_shuffled, al_in_order, _contigs())
    assert (
        out_a.column("fingerprint_hash").to_pylist()
        == out_b.column("fingerprint_hash").to_pylist()
    )


def test_empty_inputs_return_empty_schema_shaped_table() -> None:
    al = ALIGNMENT_TABLE.empty_table()
    blocks = ALIGNMENT_BLOCK_TABLE.empty_table()
    out = compute_read_fingerprints(blocks, al, _contigs())
    assert out.num_rows == 0
    from constellation.sequencing.schemas.alignment import READ_FINGERPRINT_TABLE
    assert out.schema.equals(READ_FINGERPRINT_TABLE)


def test_invalid_quantum_raises() -> None:
    al = ALIGNMENT_TABLE.empty_table()
    blocks = ALIGNMENT_BLOCK_TABLE.empty_table()
    with pytest.raises(ValueError):
        compute_read_fingerprints(
            blocks, al, _contigs(), intron_quantum_bp=0
        )
