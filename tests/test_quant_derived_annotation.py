"""Tests for ``constellation.sequencing.quant.derived_annotation``.

Cover the 10 scenarios from the plan:

  1. Two-isoform alt-3'SS — 3 exon segments, 1 gene component.
  2. Exon skipping — middle exon's PSI ∈ (0, 1).
  3. Intron retention — single block spans gene; N rows in
     block_assignments per such read.
  4. 5'/3' UTR-only exons — covered intervals with no flanking
     trusted intron snap to coverage edges.
  5. Single-exon transcripts — v1 limitation: not discovered.
  6. Sub-threshold intron — group_by(intron_id).sum(read_count) below
     min_intron_read_count: filtered out at boundary discovery.
  7. Sub-threshold coverage — depth ≤ 4 sliver doesn't seed an exon.
  8. Two genes on same contig, no shared intron — separate components.
  9. Round-trip — write derived_annotation/, load via load_annotation,
     re-derive: same exon set.
 10. Annotation validity — output passes Annotation(features=...)
     construction (PK uniqueness, parent_id FK closure).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa

from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.annotation.io import (
    load_annotation,
    save_annotation,
)
from constellation.sequencing.quant.derived_annotation import (
    assign_blocks_to_exons,
    build_derived_annotation,
    compute_exon_psi,
    derive_exons,
    roll_up_genes,
)
from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_TABLE,
    BLOCK_EXON_ASSIGNMENT_TABLE,
    INTRON_TABLE,
)
from constellation.sequencing.schemas.quant import (
    COVERAGE_TABLE,
    EXON_PSI_TABLE,
)
from constellation.sequencing.schemas.reference import (
    CONTIG_TABLE,
    FEATURE_TABLE,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _contigs(n_chrs: int = 1) -> pa.Table:
    rows = [
        {"contig_id": i + 1, "name": f"chr{i + 1}", "length": 1_000_000,
         "topology": None, "circular": None}
        for i in range(n_chrs)
    ]
    return pa.Table.from_pylist(rows, schema=CONTIG_TABLE)


def _intron(*, intron_id: int, contig_id: int = 1, strand: str = "+",
             donor: int, acceptor: int, read_count: int,
             motif: str = "GT-AG", is_seed: bool = True,
             annotated: bool | None = None) -> dict:
    return {
        "intron_id": int(intron_id),
        "contig_id": int(contig_id),
        "strand": str(strand),
        "donor_pos": int(donor),
        "acceptor_pos": int(acceptor),
        "read_count": int(read_count),
        "motif": str(motif),
        "is_intron_seed": bool(is_seed),
        "annotated": annotated,
    }


def _coverage_rows(*, contig_id: int = 1, sample_id: int = -1,
                    intervals: list[tuple[int, int, int]]) -> list[dict]:
    """Each interval is (start, end, depth)."""
    return [
        {"contig_id": int(contig_id), "sample_id": int(sample_id),
         "start": int(s), "end": int(e), "depth": int(d)}
        for s, e, d in intervals
    ]


def _alignment(**ov) -> dict:
    base = {
        "alignment_id": 0, "read_id": "r0", "acquisition_id": 1,
        "ref_name": "chr1", "ref_start": 0, "ref_end": 100, "strand": "+",
        "mapq": 60, "flag": 0, "cigar_string": "100M",
        "nm_tag": None, "as_tag": None, "read_group": None,
        "is_secondary": False, "is_supplementary": False,
    }
    base.update(ov)
    return base


def _block(**ov) -> dict:
    base = {
        "alignment_id": 0, "block_index": 0,
        "ref_start": 0, "ref_end": 100,
        "query_start": 0, "query_end": 100,
        "n_match": None, "n_mismatch": None,
        "n_insert": 0, "n_delete": 0,
    }
    base.update(ov)
    return base


# ──────────────────────────────────────────────────────────────────────
# Test 1 — Two-isoform alt-3'SS
# ──────────────────────────────────────────────────────────────────────


def test_alt_3ss_yields_three_exon_segments_one_gene() -> None:
    """Trusted introns (100, 200) and (100, 250) over a covered span
    [0, 300). Expected: 3 exon segments — [0,100), [200,250), [250,300).
    [100, 200) is dropped (== intron A); [100, 250) is intron B but
    isn't an exact piece-span. One gene component.
    """
    introns = pa.Table.from_pylist(
        [
            _intron(intron_id=0, donor=100, acceptor=200, read_count=10),
            _intron(intron_id=1, donor=100, acceptor=250, read_count=8),
        ],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 300, 10)]),
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    spans = sorted(
        (int(r["start"]), int(r["end"])) for r in exons.to_pylist()
    )
    assert spans == [(0, 100), (200, 250), (250, 300)]
    # Roll up — single gene component (all exons connected through
    # shared intron donor at 100).
    full = roll_up_genes(exons, introns, min_intron_read_count=3)
    genes = full.filter(pa.compute.equal(full.column("type"), "gene"))
    assert genes.num_rows == 1


# ──────────────────────────────────────────────────────────────────────
# Test 2 — Exon skipping
# ──────────────────────────────────────────────────────────────────────


def test_exon_skipping_middle_exon_psi_strictly_between_zero_and_one() -> None:
    """Three-exon gene: exon1 [0,100), exon2 [200,300), exon3 [400,500).
    Trusted introns: (100, 200), (300, 400) — three-exon isoform —
    AND (100, 400) — exon-2-skipping isoform.
    """
    introns = pa.Table.from_pylist(
        [
            _intron(intron_id=0, donor=100, acceptor=200, read_count=10),
            _intron(intron_id=1, donor=300, acceptor=400, read_count=10),
            _intron(intron_id=2, donor=100, acceptor=400, read_count=5),
        ],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 500, 10)]),
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    spans = sorted(
        (int(r["start"]), int(r["end"])) for r in exons.to_pylist()
    )
    # Cuts: 100, 200, 300, 400 (interior of [0, 500)).
    # Pieces: [0,100) [100,200) [200,300) [300,400) [400,500)
    # Drop pieces == trusted intron span:
    #   [100, 200) == intron 0 → drop
    #   [300, 400) == intron 1 → drop
    # [100, 400) is intron 2 but isn't a single piece (split at 200, 300).
    # Keep: [0,100), [200,300), [400,500) plus the two pieces inside intron 2:
    #   Wait, [200, 300) is inside intron 2 (100, 400). Is it dropped?
    #   The rule says "drop pieces == any trusted intron span". (200, 300)
    #   is not == any intron. KEEP.
    # So expected exons: [0,100), [200,300), [400,500) — 3 exons.
    assert spans == [(0, 100), (200, 300), (400, 500)]

    full = roll_up_genes(exons, introns, min_intron_read_count=3)
    # Now compute PSI: build alignment fixtures.
    # 10 reads of three-exon isoform (include exon2)
    # 5 reads of skipping isoform (exclude exon2)
    al_rows = []
    blk_rows = []
    next_aid = 1
    for i in range(10):  # three-exon reads
        rid = f"three_exon_{i}"
        al_rows.append(_alignment(
            alignment_id=next_aid, read_id=rid, ref_start=0, ref_end=500,
        ))
        # Three blocks: [0,100), [200,300), [400,500)
        blk_rows.append(_block(
            alignment_id=next_aid, block_index=0, ref_start=0, ref_end=100))
        blk_rows.append(_block(
            alignment_id=next_aid, block_index=1, ref_start=200, ref_end=300))
        blk_rows.append(_block(
            alignment_id=next_aid, block_index=2, ref_start=400, ref_end=500))
        next_aid += 1
    for i in range(5):  # skipping reads
        rid = f"skipping_{i}"
        al_rows.append(_alignment(
            alignment_id=next_aid, read_id=rid, ref_start=0, ref_end=500,
        ))
        blk_rows.append(_block(
            alignment_id=next_aid, block_index=0, ref_start=0, ref_end=100))
        blk_rows.append(_block(
            alignment_id=next_aid, block_index=1, ref_start=400, ref_end=500))
        next_aid += 1
    al = pa.Table.from_pylist(al_rows, schema=ALIGNMENT_TABLE)
    blocks = pa.Table.from_pylist(blk_rows, schema=ALIGNMENT_BLOCK_TABLE)
    block_assignments = assign_blocks_to_exons(blocks, al, full, _contigs())
    psi = compute_exon_psi(block_assignments, al, full, _contigs())

    # Locate the middle exon (start=200, end=300) feature_id.
    middle_exon_id = None
    for r in full.to_pylist():
        if r["type"] == "exon" and r["start"] == 200 and r["end"] == 300:
            middle_exon_id = int(r["feature_id"])
            break
    assert middle_exon_id is not None
    middle_psi_rows = [
        r for r in psi.to_pylist()
        if r["data_exon_id"] == middle_exon_id
    ]
    assert len(middle_psi_rows) == 1
    psi_value = middle_psi_rows[0]["psi"]
    assert psi_value is not None
    # 10 inclusion vs 5 exclusion → PSI = 10 / 15 ≈ 0.6667
    assert 0.0 < psi_value < 1.0
    assert middle_psi_rows[0]["n_inclusion_reads"] == 10
    assert middle_psi_rows[0]["n_exclusion_reads"] == 5


# ──────────────────────────────────────────────────────────────────────
# Test 3 — Intron retention
# ──────────────────────────────────────────────────────────────────────


def test_intron_retention_block_emits_multiple_assignments() -> None:
    """A read with a single block spanning the full gene contributes to
    multiple block_exon_assignments rows (one per exon overlap).
    """
    introns = pa.Table.from_pylist(
        [
            _intron(intron_id=0, donor=100, acceptor=200, read_count=10),
            _intron(intron_id=1, donor=300, acceptor=400, read_count=10),
        ],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 500, 10)]),
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    full = roll_up_genes(exons, introns, min_intron_read_count=3)

    # One alignment with one block spanning the whole gene.
    al = pa.Table.from_pylist(
        [_alignment(alignment_id=1, read_id="rIR", ref_start=0, ref_end=500)],
        schema=ALIGNMENT_TABLE,
    )
    blocks = pa.Table.from_pylist(
        [_block(alignment_id=1, block_index=0, ref_start=0, ref_end=500)],
        schema=ALIGNMENT_BLOCK_TABLE,
    )
    block_assignments = assign_blocks_to_exons(blocks, al, full, _contigs())
    # Three exons → three rows with the same (alignment_id, block_index).
    n_exons = full.filter(
        pa.compute.equal(full.column("type"), "exon")
    ).num_rows
    assert block_assignments.num_rows == n_exons


# ──────────────────────────────────────────────────────────────────────
# Test 4 — UTR exons (no flanking trusted intron)
# ──────────────────────────────────────────────────────────────────────


def test_utr_exons_snap_to_coverage_edges() -> None:
    """A two-exon gene with introns at (200, 300). Coverage extends
    from [50, 500). Expected first exon: [50, 200), last exon: [300, 500).
    Boundaries snap to covered-interval edges where no intron exists.
    """
    introns = pa.Table.from_pylist(
        [_intron(intron_id=0, donor=200, acceptor=300, read_count=10)],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(50, 500, 10)]),
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    spans = sorted(
        (int(r["start"]), int(r["end"])) for r in exons.to_pylist()
    )
    assert spans == [(50, 200), (300, 500)]


# ──────────────────────────────────────────────────────────────────────
# Test 5 — Single-exon transcripts
# ──────────────────────────────────────────────────────────────────────


def test_single_exon_transcripts_v1_limitation() -> None:
    """v1: single-exon genes (no introns) are NOT discovered.

    Documented limitation; lifted in v2 with stranded coverage.
    """
    introns = INTRON_TABLE.empty_table()
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(50, 500, 10)]),
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    assert exons.num_rows == 0


# ──────────────────────────────────────────────────────────────────────
# Test 6 — Sub-threshold intron
# ──────────────────────────────────────────────────────────────────────


def test_sub_threshold_intron_filtered() -> None:
    """An intron cluster with total read_count < min_intron_read_count
    doesn't introduce a boundary.
    """
    introns = pa.Table.from_pylist(
        [_intron(intron_id=0, donor=100, acceptor=200, read_count=2)],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 300, 10)]),
        schema=COVERAGE_TABLE,
    )
    # min_intron_read_count=3 filters out the single-intron-with-2-reads.
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    # No trusted introns → no exons emitted (v1 limitation also applies).
    assert exons.num_rows == 0


def test_sub_threshold_intron_aggregated_passes() -> None:
    """Cluster total read_count = sum across member positions; even if
    individual rows are below threshold, the cluster sum can still pass.
    """
    introns = pa.Table.from_pylist(
        [
            _intron(intron_id=0, donor=100, acceptor=200, read_count=2,
                    is_seed=True),
            _intron(intron_id=0, donor=101, acceptor=200, read_count=2,
                    is_seed=False),  # member of same cluster
        ],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 300, 10)]),
        schema=COVERAGE_TABLE,
    )
    # Cluster total = 2 + 2 = 4 >= 3 → trusted.
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    assert exons.num_rows >= 1


# ──────────────────────────────────────────────────────────────────────
# Test 7 — Sub-threshold coverage
# ──────────────────────────────────────────────────────────────────────


def test_sub_threshold_coverage_does_not_seed_exon() -> None:
    """A region with depth < min_exon_depth doesn't seed an exon."""
    introns = pa.Table.from_pylist(
        [_intron(intron_id=0, donor=100, acceptor=200, read_count=10)],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 300, 4)]),  # depth=4 < 5
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    assert exons.num_rows == 0


# ──────────────────────────────────────────────────────────────────────
# Test 8 — Two genes on same contig, no shared intron
# ──────────────────────────────────────────────────────────────────────


def test_two_genes_separate_components() -> None:
    """Two introns on the same contig but in disjoint coverage regions
    → two genes (separate connected components).
    """
    introns = pa.Table.from_pylist(
        [
            _intron(intron_id=0, donor=100, acceptor=200, read_count=10),
            _intron(intron_id=1, donor=600, acceptor=700, read_count=10),
        ],
        schema=INTRON_TABLE,
    )
    # Two non-abutting covered regions (gap between 300 and 500 →
    # not coalesced).
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[
            (0, 300, 10),
            (500, 800, 10),
        ]),
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    full = roll_up_genes(exons, introns, min_intron_read_count=3)
    genes = full.filter(pa.compute.equal(full.column("type"), "gene"))
    assert genes.num_rows == 2


# ──────────────────────────────────────────────────────────────────────
# Test 9 — Round-trip
# ──────────────────────────────────────────────────────────────────────


def test_derived_annotation_roundtrip(tmp_path: Path) -> None:
    """Write derived_annotation/, load via load_annotation, verify
    feature set matches.
    """
    introns = pa.Table.from_pylist(
        [
            _intron(intron_id=0, donor=100, acceptor=200, read_count=10),
            _intron(intron_id=1, donor=300, acceptor=400, read_count=10),
        ],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 500, 10)]),
        schema=COVERAGE_TABLE,
    )
    al = pa.Table.from_pylist(
        [_alignment(alignment_id=1, read_id="rA", ref_start=0, ref_end=500)],
        schema=ALIGNMENT_TABLE,
    )
    blocks = pa.Table.from_pylist(
        [
            _block(alignment_id=1, block_index=0, ref_start=0, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=200, ref_end=300),
            _block(alignment_id=1, block_index=2, ref_start=400, ref_end=500),
        ],
        schema=ALIGNMENT_BLOCK_TABLE,
    )
    annotation, block_assignments, exon_psi = build_derived_annotation(
        coverage=coverage, introns=introns,
        alignment_blocks=blocks, alignments=al, contigs=_contigs(),
    )
    # Sanity: 3 exons + 1 gene.
    assert annotation.features.num_rows == 4
    save_annotation(
        annotation, tmp_path / "derived_annotation",
        format="parquet_dir",
    )
    loaded = load_annotation(tmp_path / "derived_annotation")
    # Schema match.
    assert loaded.features.schema.equals(annotation.features.schema)
    # Same set of (start, end, type, strand).
    def _key(t: pa.Table) -> set:
        return {
            (int(r["start"]), int(r["end"]), r["type"], r["strand"])
            for r in t.to_pylist()
        }
    assert _key(loaded.features) == _key(annotation.features)


# ──────────────────────────────────────────────────────────────────────
# Test 10 — Annotation validity (PK + FK closure)
# ──────────────────────────────────────────────────────────────────────


def test_derived_annotation_passes_annotation_validation() -> None:
    """Output should construct as an Annotation without raising
    (validates PK uniqueness on feature_id + FK closure on parent_id).
    """
    introns = pa.Table.from_pylist(
        [
            _intron(intron_id=0, donor=100, acceptor=200, read_count=10),
            _intron(intron_id=1, donor=300, acceptor=400, read_count=10),
        ],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 500, 10)]),
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    full = roll_up_genes(exons, introns, min_intron_read_count=3)
    # If validation fails this raises.
    annotation = Annotation(features=full)
    # All exons should have parent_id pointing at a valid gene.
    feature_ids = set(annotation.features.column("feature_id").to_pylist())
    for r in annotation.features.to_pylist():
        if r["type"] == "exon":
            assert r["parent_id"] is not None
            assert int(r["parent_id"]) in feature_ids


# ──────────────────────────────────────────────────────────────────────
# Additional invariants
# ──────────────────────────────────────────────────────────────────────


def test_block_assignments_pk_uniqueness() -> None:
    """No duplicate (alignment_id, block_index, data_exon_id) triples."""
    introns = pa.Table.from_pylist(
        [_intron(intron_id=0, donor=100, acceptor=200, read_count=10)],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 300, 10)]),
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    full = roll_up_genes(exons, introns, min_intron_read_count=3)
    al = pa.Table.from_pylist(
        [_alignment(alignment_id=1, read_id="rA", ref_start=0, ref_end=300)],
        schema=ALIGNMENT_TABLE,
    )
    blocks = pa.Table.from_pylist(
        [
            _block(alignment_id=1, block_index=0, ref_start=0, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=200, ref_end=300),
        ],
        schema=ALIGNMENT_BLOCK_TABLE,
    )
    block_assignments = assign_blocks_to_exons(blocks, al, full, _contigs())
    keys = [
        (r["alignment_id"], r["block_index"], r["data_exon_id"])
        for r in block_assignments.to_pylist()
    ]
    assert len(keys) == len(set(keys))


def test_psi_null_when_no_gene_spanning_reads() -> None:
    """An exon with no gene-spanning reads emits no PSI row."""
    introns = pa.Table.from_pylist(
        [_intron(intron_id=0, donor=100, acceptor=200, read_count=10)],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 300, 10)]),
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    full = roll_up_genes(exons, introns, min_intron_read_count=3)
    # No alignments at all.
    al = ALIGNMENT_TABLE.empty_table()
    blocks = ALIGNMENT_BLOCK_TABLE.empty_table()
    block_assignments = assign_blocks_to_exons(blocks, al, full, _contigs())
    psi = compute_exon_psi(block_assignments, al, full, _contigs())
    assert psi.num_rows == 0


def test_empty_inputs_return_empty_schema_shaped_outputs() -> None:
    """Empty introns / coverage → empty derived annotation; build_*
    returns the right schema shapes."""
    annotation, ba, psi = build_derived_annotation(
        coverage=COVERAGE_TABLE.empty_table(),
        introns=INTRON_TABLE.empty_table(),
        alignment_blocks=ALIGNMENT_BLOCK_TABLE.empty_table(),
        alignments=ALIGNMENT_TABLE.empty_table(),
        contigs=_contigs(),
    )
    assert annotation.features.num_rows == 0
    assert annotation.features.schema.equals(FEATURE_TABLE)
    assert ba.num_rows == 0
    assert ba.schema.equals(BLOCK_EXON_ASSIGNMENT_TABLE)
    assert psi.num_rows == 0
    assert psi.schema.equals(EXON_PSI_TABLE)


def test_per_sample_psi_stratification() -> None:
    """When read_to_sample is provided, PSI rows split per sample."""
    introns = pa.Table.from_pylist(
        [
            _intron(intron_id=0, donor=100, acceptor=200, read_count=10),
            _intron(intron_id=1, donor=300, acceptor=400, read_count=10),
            _intron(intron_id=2, donor=100, acceptor=400, read_count=5),
        ],
        schema=INTRON_TABLE,
    )
    coverage = pa.Table.from_pylist(
        _coverage_rows(intervals=[(0, 500, 10)]),
        schema=COVERAGE_TABLE,
    )
    exons = derive_exons(coverage, introns, _contigs(),
                         min_exon_depth=5, min_intron_read_count=3)
    full = roll_up_genes(exons, introns, min_intron_read_count=3)

    # 3 reads in sample 0 (all three-exon), 2 reads in sample 1
    # (skipping middle exon).
    al_rows = []
    blk_rows = []
    read_to_sample = {}
    next_aid = 1
    for i in range(3):
        rid = f"s0_r{i}"
        read_to_sample[rid] = 0
        al_rows.append(_alignment(
            alignment_id=next_aid, read_id=rid,
            ref_start=0, ref_end=500,
        ))
        for bi, (s, e) in enumerate([(0, 100), (200, 300), (400, 500)]):
            blk_rows.append(_block(
                alignment_id=next_aid, block_index=bi,
                ref_start=s, ref_end=e,
            ))
        next_aid += 1
    for i in range(2):
        rid = f"s1_r{i}"
        read_to_sample[rid] = 1
        al_rows.append(_alignment(
            alignment_id=next_aid, read_id=rid,
            ref_start=0, ref_end=500,
        ))
        for bi, (s, e) in enumerate([(0, 100), (400, 500)]):
            blk_rows.append(_block(
                alignment_id=next_aid, block_index=bi,
                ref_start=s, ref_end=e,
            ))
        next_aid += 1

    al = pa.Table.from_pylist(al_rows, schema=ALIGNMENT_TABLE)
    blocks = pa.Table.from_pylist(blk_rows, schema=ALIGNMENT_BLOCK_TABLE)
    ba = assign_blocks_to_exons(blocks, al, full, _contigs())
    psi = compute_exon_psi(ba, al, full, _contigs(),
                           read_to_sample=read_to_sample)

    # Locate middle exon.
    middle_exon_id = None
    for r in full.to_pylist():
        if r["type"] == "exon" and r["start"] == 200 and r["end"] == 300:
            middle_exon_id = int(r["feature_id"])
            break
    assert middle_exon_id is not None

    psi_by_sample = {
        int(r["sample_id"]): r
        for r in psi.to_pylist()
        if r["data_exon_id"] == middle_exon_id
    }
    # Sample 0: PSI = 1.0 (all 3 reads include); sample 1: PSI = 0.0.
    assert psi_by_sample[0]["psi"] == 1.0
    assert psi_by_sample[0]["n_inclusion_reads"] == 3
    assert psi_by_sample[0]["n_exclusion_reads"] == 0
    assert psi_by_sample[1]["psi"] == 0.0
    assert psi_by_sample[1]["n_inclusion_reads"] == 0
    assert psi_by_sample[1]["n_exclusion_reads"] == 2
