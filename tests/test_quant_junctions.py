"""Tests for ``constellation.sequencing.quant.junctions.aggregate_junctions``.

Verify the cross-read junction-aggregation reduction:
    - one row per (contig, donor, acceptor, strand) tuple
    - ``read_count`` counts distinct reads (not distinct alignments)
    - ``motif`` populated from genome at donor[0:2] / acceptor[-2:]
    - ``annotated`` True iff both splice sites match annotated exons
      on the same contig (when an Annotation is provided)
    - ``annotated`` null when no Annotation is provided
    - secondary / supplementary alignments excluded by default
    - alignments to unknown contigs dropped silently
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.quant.junctions import aggregate_junctions
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_TABLE,
)
from constellation.sequencing.schemas.reference import FEATURE_TABLE


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


def _genome(seq: str = None) -> GenomeReference:
    """Tiny genome for motif lookup. By default, builds chr1 with a
    GT...AG canonical splice motif at positions 100→500.
    """
    if seq is None:
        # Position layout used by the canonical-motif tests:
        #   donor at 100 → genome[100:102] = 'GT'
        #   acceptor at 500 → genome[498:500] = 'AG'
        seq = (
            "A" * 100        # 0..99
            + "GT"           # 100..101 — donor
            + "C" * 396      # 102..497
            + "AG"           # 498..499 — acceptor (acceptor_pos=500 → genome[498:500])
            + "T" * 500      # 500..999
        )
    return GenomeReference(
        contigs=pa.Table.from_pylist(
            [{"contig_id": 1, "name": "chr1", "length": len(seq),
              "topology": None, "circular": None}]
        ),
        sequences=pa.Table.from_pylist(
            [{"contig_id": 1, "sequence": seq}]
        ),
    )


def _alignments(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_TABLE)


def _blocks(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_BLOCK_TABLE)


# ──────────────────────────────────────────────────────────────────────


def test_two_reads_same_junction_collapse_to_one_row() -> None:
    """Two reads spliced through the same intron yield one junction
    with read_count=2.
    """
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
            _block(alignment_id=2, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=2, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    out = aggregate_junctions(blocks, al, _genome())
    assert out.num_rows == 1
    row = out.to_pylist()[0]
    assert row["donor_pos"] == 100 and row["acceptor_pos"] == 500
    assert row["read_count"] == 2


def test_motif_lookup_emits_dinucleotide_pair() -> None:
    """Donor + acceptor dinucleotides come straight from the +-strand
    genome.
    """
    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    out = aggregate_junctions(blocks, al, _genome())
    assert out.column("motif").to_pylist() == ["GT-AG"]


def test_motif_out_of_bounds_yields_other() -> None:
    """A junction whose acceptor sits past contig end falls back to
    'other' rather than raising.
    """
    # Tiny 50-bp genome; junction acceptor at 60 is past the end.
    tiny = GenomeReference(
        contigs=pa.Table.from_pylist(
            [{"contig_id": 1, "name": "chr1", "length": 50,
              "topology": None, "circular": None}]
        ),
        sequences=pa.Table.from_pylist(
            [{"contig_id": 1, "sequence": "A" * 50}]
        ),
    )
    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=0, ref_end=10),
            _block(alignment_id=1, block_index=1, ref_start=60, ref_end=70),
        ]
    )
    out = aggregate_junctions(blocks, al, tiny)
    assert out.column("motif").to_pylist() == ["other"]


def test_secondary_and_supplementary_excluded() -> None:
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA", is_secondary=True),
            _alignment(alignment_id=2, read_id="rB", is_supplementary=True),
            _alignment(alignment_id=3, read_id="rC"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
            _block(alignment_id=2, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=2, block_index=1, ref_start=500, ref_end=600),
            _block(alignment_id=3, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=3, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    out = aggregate_junctions(blocks, al, _genome())
    assert out.num_rows == 1
    assert out.column("read_count").to_pylist() == [1]


def test_strand_difference_produces_separate_rows() -> None:
    """Same (donor, acceptor) on opposite strands → two rows."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA", strand="+"),
            _alignment(alignment_id=2, read_id="rB", strand="-"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
            _block(alignment_id=2, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=2, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    out = aggregate_junctions(blocks, al, _genome())
    assert out.num_rows == 2
    strands = out.column("strand").to_pylist()
    assert sorted(strands) == ["+", "-"]


def test_alignment_to_unknown_contig_is_dropped() -> None:
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA", ref_name="chrX_unknown"),
            _alignment(alignment_id=2, read_id="rB", ref_name="chr1"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
            _block(alignment_id=2, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=2, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    out = aggregate_junctions(blocks, al, _genome())
    # chrX dropped silently; only the chr1 junction survives
    assert out.num_rows == 1
    assert out.column("contig_id").to_pylist() == [1]


def test_annotated_flag_null_without_annotation() -> None:
    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    out = aggregate_junctions(blocks, al, _genome())
    assert out.column("annotated").to_pylist() == [None]


def test_annotated_flag_true_when_both_sites_match() -> None:
    """Both donor (= an exon's end) and acceptor (= an exon's start)
    in the annotation set → annotated=True.
    """
    features = pa.Table.from_pylist(
        [
            # Exon ending at 100 (donor site)
            {
                "feature_id": 10, "contig_id": 1, "start": 50, "end": 100,
                "strand": "+", "type": "exon", "name": None,
                "parent_id": None, "source": None, "score": None,
                "phase": None, "attributes_json": None,
            },
            # Exon starting at 500 (acceptor site)
            {
                "feature_id": 20, "contig_id": 1, "start": 500, "end": 600,
                "strand": "+", "type": "exon", "name": None,
                "parent_id": None, "source": None, "score": None,
                "phase": None, "attributes_json": None,
            },
        ],
        schema=FEATURE_TABLE,
    )
    annotation = Annotation(features=features)

    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    out = aggregate_junctions(blocks, al, _genome(), annotation=annotation)
    assert out.column("annotated").to_pylist() == [True]


def test_annotated_flag_false_when_one_site_novel() -> None:
    features = pa.Table.from_pylist(
        [
            {
                "feature_id": 10, "contig_id": 1, "start": 50, "end": 100,
                "strand": "+", "type": "exon", "name": None,
                "parent_id": None, "source": None, "score": None,
                "phase": None, "attributes_json": None,
            },
            # No exon starting at 500 — the observed acceptor is novel
            {
                "feature_id": 20, "contig_id": 1, "start": 700, "end": 800,
                "strand": "+", "type": "exon", "name": None,
                "parent_id": None, "source": None, "score": None,
                "phase": None, "attributes_json": None,
            },
        ],
        schema=FEATURE_TABLE,
    )
    annotation = Annotation(features=features)

    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    out = aggregate_junctions(blocks, al, _genome(), annotation=annotation)
    assert out.column("annotated").to_pylist() == [False]


def test_three_block_alignment_yields_two_junctions() -> None:
    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=0, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
            _block(alignment_id=1, block_index=2, ref_start=800, ref_end=900),
        ]
    )
    out = aggregate_junctions(blocks, al, _genome())
    assert out.num_rows == 2
    rows = sorted(out.to_pylist(), key=lambda r: r["donor_pos"])
    assert rows[0]["donor_pos"] == 100 and rows[0]["acceptor_pos"] == 500
    assert rows[1]["donor_pos"] == 600 and rows[1]["acceptor_pos"] == 800


def test_empty_inputs_return_empty_schema() -> None:
    out = aggregate_junctions(
        ALIGNMENT_BLOCK_TABLE.empty_table(),
        ALIGNMENT_TABLE.empty_table(),
        _genome(),
    )
    from constellation.sequencing.schemas.alignment import INTRON_TABLE
    assert out.schema.equals(INTRON_TABLE)
    assert out.num_rows == 0
