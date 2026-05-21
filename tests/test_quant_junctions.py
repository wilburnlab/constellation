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
from constellation.sequencing.quant.junctions import (
    _aggregate_junctions_legacy,
    aggregate_junctions,
)
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


def test_aggregate_junctions_accepts_large_string_columns() -> None:
    """aggregate_junctions tolerates large_string alignment string columns
    (which can arise upstream at scale) and still emits a plain-string
    INTRON_TABLE."""
    import pyarrow.compute as pc

    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    al = al.set_column(
        al.schema.get_field_index("ref_name"),
        "ref_name",
        pc.cast(al.column("ref_name"), pa.large_string()),
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    out = aggregate_junctions(blocks, al, _genome())
    assert out.num_rows == 1
    assert out.column("strand").type == pa.string()
    assert out.column("read_count").to_pylist() == [1]


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


# ──────────────────────────────────────────────────────────────────────
# Vectorisation parity + scale
# ──────────────────────────────────────────────────────────────────────


def _synth_corpus(
    *,
    n_alignments: int,
    n_blocks_per_alignment: int,
    n_distinct_junctions: int,
    seed: int = 0,
) -> tuple[pa.Table, pa.Table]:
    """Synthetic alignments + blocks fixture for parity / scale tests.

    Each alignment is a single read on chr1 with ``n_blocks_per_alignment``
    blocks → produces ``n_blocks_per_alignment - 1`` junctions per
    alignment. Junctions are drawn from a pool of ``n_distinct_junctions``
    canonical (donor, acceptor) pairs so the cross-read aggregation has
    real overlap to count.
    """
    import random

    rng = random.Random(seed)
    canonical_junctions = [
        (100 + 1000 * i, 500 + 1000 * i) for i in range(n_distinct_junctions)
    ]
    al_rows: list[dict] = []
    block_rows: list[dict] = []
    for aid in range(1, n_alignments + 1):
        # Pick which junctions this alignment uses, in order; first donor
        # establishes the leftmost block; subsequent blocks follow.
        chosen = sorted(
            rng.sample(
                canonical_junctions,
                min(n_blocks_per_alignment - 1, len(canonical_junctions)),
            )
        )
        # Build blocks: first block starts before first donor; each
        # subsequent block goes from the acceptor of the prior junction
        # to the donor of the next.
        first_donor = chosen[0][0]
        last_acceptor = chosen[-1][1]
        al_rows.append(_alignment(
            alignment_id=aid,
            read_id=f"r{aid}",
            ref_start=first_donor - 100,
            ref_end=last_acceptor + 100,
        ))
        # Block 0
        block_rows.append(_block(
            alignment_id=aid, block_index=0,
            ref_start=first_donor - 100, ref_end=first_donor,
        ))
        for bi, ((donor, acceptor), next_pair) in enumerate(
            zip(chosen, chosen[1:] + [(None, None)]), start=1
        ):
            next_donor = next_pair[0] if next_pair[0] is not None else acceptor + 100
            block_rows.append(_block(
                alignment_id=aid, block_index=bi,
                ref_start=acceptor, ref_end=next_donor,
            ))
    al = _alignments(al_rows)
    blocks = _blocks(block_rows)
    return blocks, al


def _normalize_for_compare(t: pa.Table) -> list[dict]:
    """Convert table → list of dicts sorted by the natural key for direct
    equality compare. ``intron_id`` is row-index-derived, so we exclude
    it from the comparison (both implementations must agree on row
    contents, but their intron_id naming can stay row-index-stable as
    long as the deterministic sort is the same).
    """
    rows = t.to_pylist()
    # Sort by the natural key (matches both impls' deterministic sort)
    rows.sort(key=lambda r: (r["contig_id"], r["donor_pos"], r["acceptor_pos"], r["strand"]))
    # Drop intron_id (row index, derived deterministically); compare contents
    for r in rows:
        r.pop("intron_id", None)
    return rows


def test_parity_legacy_vs_vectorized_small() -> None:
    """Small hand-built input → both implementations produce identical
    output. Covers the canonical multi-read multi-junction case."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
            _alignment(alignment_id=3, read_id="rC"),
        ]
    )
    blocks = _blocks(
        [
            # Alignment 1: 100→500 and 600→900
            _block(alignment_id=1, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
            _block(alignment_id=1, block_index=2, ref_start=900, ref_end=950),
            # Alignment 2: same as 1 (different read)
            _block(alignment_id=2, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=2, block_index=1, ref_start=500, ref_end=600),
            # Alignment 3: 100→500 only
            _block(alignment_id=3, block_index=0, ref_start=50, ref_end=100),
            _block(alignment_id=3, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    new = aggregate_junctions(blocks, al, _genome())
    legacy = _aggregate_junctions_legacy(blocks, al, _genome())
    assert _normalize_for_compare(new) == _normalize_for_compare(legacy)


def test_parity_legacy_vs_vectorized_synthetic_corpus() -> None:
    """Larger synthetic corpus: 100 alignments × 4 blocks each, junctions
    drawn from a 20-pool. Confirms parity across the cross-read
    aggregation path (multiple reads supporting the same junction)."""
    blocks, al = _synth_corpus(
        n_alignments=100, n_blocks_per_alignment=4, n_distinct_junctions=20, seed=42,
    )
    # Big enough genome to cover the synthetic junctions
    big_seq = "ACGT" * 6000  # 24kb — covers up to ~position 24000
    genome = GenomeReference(
        contigs=pa.Table.from_pylist(
            [{"contig_id": 1, "name": "chr1", "length": len(big_seq),
              "topology": None, "circular": None}]
        ),
        sequences=pa.Table.from_pylist(
            [{"contig_id": 1, "sequence": big_seq}]
        ),
    )
    new = aggregate_junctions(blocks, al, genome)
    legacy = _aggregate_junctions_legacy(blocks, al, genome)
    assert _normalize_for_compare(new) == _normalize_for_compare(legacy)
    # Sanity-check the test exercises real cross-read aggregation
    rcs = new.column("read_count").to_pylist()
    assert max(rcs) > 1, "test corpus didn't produce any multi-read junctions"


def test_vectorized_scale_smoke() -> None:
    """Smoke test: 10k alignments × 5 blocks each (~40k junction
    observations, ~50 unique junctions) must complete in well under a
    second. The pre-vectorisation impl took multi-hour at this kind of
    scale once the per-row dict.setdefault loop kicked in; the
    vectorized impl is Arrow-native end-to-end."""
    import time

    blocks, al = _synth_corpus(
        n_alignments=10_000, n_blocks_per_alignment=5, n_distinct_junctions=50, seed=7,
    )
    big_seq = "ACGT" * 30_000  # 120kb
    genome = GenomeReference(
        contigs=pa.Table.from_pylist(
            [{"contig_id": 1, "name": "chr1", "length": len(big_seq),
              "topology": None, "circular": None}]
        ),
        sequences=pa.Table.from_pylist(
            [{"contig_id": 1, "sequence": big_seq}]
        ),
    )
    t0 = time.perf_counter()
    out = aggregate_junctions(blocks, al, genome)
    elapsed = time.perf_counter() - t0
    # Generous bound — local laptops finish in <50ms; CI is forgiving.
    assert elapsed < 10.0, f"vectorised path too slow: {elapsed:.2f}s"
    # Real output, not empty
    assert out.num_rows > 0
    # Read counts make sense (some junctions hit by many reads)
    assert max(out.column("read_count").to_pylist()) > 1
