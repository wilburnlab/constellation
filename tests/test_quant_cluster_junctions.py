"""Tests for ``constellation.sequencing.quant.junctions.cluster_junctions``.

Greedy support-ranked intron clustering. Verify:

  1. Asymmetric support — GT-AG seed wins.
  2. Bin-edge ambiguity — uniform-quantum splits become single clusters
     when within tolerance.
  3. Genuine alt-splice — donor delta >> tolerance keeps separate clusters.
  4. Cross-strand isolation — same (donor, acceptor) on +/- strands
     stay separate.
  5. Seed determinism — row-order independence.
  6. Membership invariant — exactly one seed per intron_id; sum of
     read_count per intron_id equals the cluster's total support.
  7. tolerance_bp=0 keeps every observed position pair as its own
     singleton cluster.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.sequencing.quant.junctions import cluster_junctions
from constellation.sequencing.schemas.alignment import INTRON_TABLE


def _row(
    *,
    intron_id: int = 0,
    contig_id: int = 1,
    strand: str = "+",
    donor_pos: int,
    acceptor_pos: int,
    read_count: int,
    motif: str = "GT-AG",
    is_intron_seed: bool = True,
    annotated: bool | None = None,
) -> dict:
    return {
        "intron_id": int(intron_id),
        "contig_id": int(contig_id),
        "strand": str(strand),
        "donor_pos": int(donor_pos),
        "acceptor_pos": int(acceptor_pos),
        "read_count": int(read_count),
        "motif": str(motif),
        "is_intron_seed": bool(is_intron_seed),
        "annotated": annotated,
    }


def _table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=INTRON_TABLE)


# ──────────────────────────────────────────────────────────────────────


def test_asymmetric_support_gt_ag_seed_wins() -> None:
    """Two raw junctions: dominant GT-AG vs neighbouring non-canonical.

    Donor delta = 1 bp << tolerance=5; the GT-AG row has 100 reads vs
    the non-canonical row's 5. Expected: one cluster, seed at the
    GT-AG position with is_intron_seed=True.
    """
    rows = [
        _row(donor_pos=1239, acceptor_pos=5670, read_count=100, motif="GT-AG"),
        _row(donor_pos=1240, acceptor_pos=5670, read_count=5, motif="other"),
    ]
    out = cluster_junctions(_table(rows), tolerance_bp=5)
    intron_ids = set(out.column("intron_id").to_pylist())
    assert len(intron_ids) == 1, f"Expected 1 cluster; got {intron_ids}"
    seeds = [r for r in out.to_pylist() if r["is_intron_seed"]]
    assert len(seeds) == 1
    assert seeds[0]["donor_pos"] == 1239
    assert seeds[0]["motif"] == "GT-AG"
    # Members in the cluster.
    members = [r for r in out.to_pylist() if not r["is_intron_seed"]]
    assert len(members) == 1
    assert members[0]["donor_pos"] == 1240


def test_bin_edge_ambiguity_resolved() -> None:
    """Two junctions 4 bp apart with equal read_count should cluster."""
    rows = [
        _row(donor_pos=1234, acceptor_pos=5670, read_count=50),
        _row(donor_pos=1238, acceptor_pos=5670, read_count=50),
    ]
    out = cluster_junctions(_table(rows), tolerance_bp=5)
    assert len(set(out.column("intron_id").to_pylist())) == 1


def test_genuine_alt_splice_preserved() -> None:
    """Donor delta = 66 bp, much larger than the 5 bp tolerance — two
    real alt-5'SS events should stay distinct.
    """
    rows = [
        _row(donor_pos=1234, acceptor_pos=5670, read_count=100),
        _row(donor_pos=1300, acceptor_pos=5670, read_count=80),
    ]
    out = cluster_junctions(_table(rows), tolerance_bp=5)
    assert len(set(out.column("intron_id").to_pylist())) == 2


def test_cross_strand_isolation() -> None:
    """Same (donor, acceptor) on +/- strands → two separate clusters."""
    rows = [
        _row(donor_pos=1000, acceptor_pos=2000, read_count=50, strand="+"),
        _row(donor_pos=1000, acceptor_pos=2000, read_count=50, strand="-"),
    ]
    out = cluster_junctions(_table(rows), tolerance_bp=5)
    assert len(set(out.column("intron_id").to_pylist())) == 2


def test_seed_determinism_across_input_orders() -> None:
    """Reordering the input rows shouldn't change cluster assignments."""
    rows_a = [
        _row(donor_pos=100, acceptor_pos=200, read_count=10),
        _row(donor_pos=105, acceptor_pos=200, read_count=8),
    ]
    rows_b = [
        _row(donor_pos=105, acceptor_pos=200, read_count=8),
        _row(donor_pos=100, acceptor_pos=200, read_count=10),
    ]
    out_a = cluster_junctions(_table(rows_a), tolerance_bp=5)
    out_b = cluster_junctions(_table(rows_b), tolerance_bp=5)
    # Final sort key is (intron_id, is_intron_seed desc, donor, acceptor)
    # → both runs should produce the exact same row sequence.
    keys_a = list(zip(
        out_a.column("intron_id").to_pylist(),
        out_a.column("donor_pos").to_pylist(),
        out_a.column("is_intron_seed").to_pylist(),
        strict=True,
    ))
    keys_b = list(zip(
        out_b.column("intron_id").to_pylist(),
        out_b.column("donor_pos").to_pylist(),
        out_b.column("is_intron_seed").to_pylist(),
        strict=True,
    ))
    assert keys_a == keys_b


def test_membership_invariant_exactly_one_seed_per_cluster() -> None:
    """For every distinct intron_id, exactly one row has is_intron_seed=True."""
    rows = [
        _row(donor_pos=100, acceptor_pos=200, read_count=10),
        _row(donor_pos=102, acceptor_pos=200, read_count=8),  # cluster with above
        _row(donor_pos=500, acceptor_pos=700, read_count=5),
        _row(donor_pos=503, acceptor_pos=700, read_count=4),  # cluster with above
        _row(donor_pos=900, acceptor_pos=1100, read_count=3),  # standalone
    ]
    out = cluster_junctions(_table(rows), tolerance_bp=5)
    intron_ids = out.column("intron_id").to_pylist()
    is_seed = out.column("is_intron_seed").to_pylist()
    seeds_per_intron: dict[int, int] = {}
    for iid, seed in zip(intron_ids, is_seed, strict=True):
        if seed:
            seeds_per_intron[iid] = seeds_per_intron.get(iid, 0) + 1
    distinct_intron_ids = set(intron_ids)
    assert len(distinct_intron_ids) == 3, (
        f"Expected 3 clusters; got {len(distinct_intron_ids)}"
    )
    for iid in distinct_intron_ids:
        assert seeds_per_intron.get(iid, 0) == 1, (
            f"intron_id={iid} should have exactly 1 seed row; "
            f"got {seeds_per_intron.get(iid)}"
        )


def test_membership_read_count_sum_preserved() -> None:
    """Sum of read_count per cluster equals total cluster support."""
    rows = [
        _row(donor_pos=100, acceptor_pos=200, read_count=10),
        _row(donor_pos=102, acceptor_pos=200, read_count=8),
        _row(donor_pos=500, acceptor_pos=700, read_count=5),
    ]
    out = cluster_junctions(_table(rows), tolerance_bp=5)
    sums = out.group_by("intron_id").aggregate([("read_count", "sum")])
    sums_dict = {
        int(iid): int(s)
        for iid, s in zip(
            sums.column("intron_id").to_pylist(),
            sums.column("read_count_sum").to_pylist(),
            strict=True,
        )
    }
    # Two clusters expected: {18, 5}.
    assert sorted(sums_dict.values()) == [5, 18]


def test_tolerance_zero_keeps_singletons() -> None:
    """tolerance_bp=0 absorbs only exact-match positions; observed
    pairs in INTRON_TABLE are unique by construction → each input row
    stays its own cluster.
    """
    rows = [
        _row(donor_pos=1234, acceptor_pos=5670, read_count=50),
        _row(donor_pos=1235, acceptor_pos=5670, read_count=50),
    ]
    out = cluster_junctions(_table(rows), tolerance_bp=0)
    assert len(set(out.column("intron_id").to_pylist())) == 2
    assert all(out.column("is_intron_seed").to_pylist())


def test_motif_priority_breaks_ties_on_equal_count() -> None:
    """Equal read_counts: GT-AG should win over GC-AG."""
    rows = [
        _row(donor_pos=100, acceptor_pos=200, read_count=10, motif="GC-AG"),
        _row(donor_pos=104, acceptor_pos=200, read_count=10, motif="GT-AG"),
    ]
    out = cluster_junctions(_table(rows), tolerance_bp=5)
    seeds = [r for r in out.to_pylist() if r["is_intron_seed"]]
    assert len(seeds) == 1
    assert seeds[0]["motif"] == "GT-AG"


def test_motif_priority_custom_order() -> None:
    """A custom motif_priority list reorders the tiebreak."""
    rows = [
        _row(donor_pos=100, acceptor_pos=200, read_count=10, motif="GT-AG"),
        _row(donor_pos=104, acceptor_pos=200, read_count=10, motif="other"),
    ]
    # Pretend "other" wins over GT-AG.
    out = cluster_junctions(
        _table(rows), tolerance_bp=5,
        motif_priority=("other", "GT-AG", "GC-AG", "AT-AC"),
    )
    seeds = [r for r in out.to_pylist() if r["is_intron_seed"]]
    assert len(seeds) == 1
    assert seeds[0]["motif"] == "other"


def test_empty_input_returns_empty_table() -> None:
    out = cluster_junctions(INTRON_TABLE.empty_table(), tolerance_bp=5)
    assert out.num_rows == 0
    assert out.schema.equals(INTRON_TABLE)


def test_negative_tolerance_raises() -> None:
    rows = [_row(donor_pos=100, acceptor_pos=200, read_count=10)]
    with pytest.raises(ValueError):
        cluster_junctions(_table(rows), tolerance_bp=-1)


def test_three_way_chain_does_not_extend_through_seed() -> None:
    """The greedy algorithm absorbs B into A (within tolerance), but C
    (farther from A but close to B) does NOT get absorbed if it's
    outside ±W of the seed A. This documents the strict L∞-ball-from-
    seed semantics — not single-link chain agglomeration.

    Concrete: A=1100, B=1104, C=1108. With tolerance=5, B absorbs into
    A (|B-A|=4). Should C absorb into A's cluster (|C-A|=8 > 5)? No.
    The greedy algorithm checks against the seed only.
    """
    rows = [
        _row(donor_pos=1100, acceptor_pos=2000, read_count=100),
        _row(donor_pos=1104, acceptor_pos=2000, read_count=50),
        _row(donor_pos=1108, acceptor_pos=2000, read_count=10),
    ]
    out = cluster_junctions(_table(rows), tolerance_bp=5)
    # A and B share one cluster; C is its own.
    by_donor = {r["donor_pos"]: r["intron_id"] for r in out.to_pylist()}
    assert by_donor[1100] == by_donor[1104]
    assert by_donor[1108] != by_donor[1100]
