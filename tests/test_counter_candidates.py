"""Step-6 foundation: the theoretical precursor-candidate index (PR-D). See
docs/plans/counter-real-data-deconstruction.md §9."""

from __future__ import annotations

import torch

from constellation.core.sequence.proforma import Peptidoform
from constellation.massspec.counter import (
    GlobalCalibration,
    Progenitor,
    TheoreticalCandidateIndex,
    channel_overlap_components,
    refine_components_by_rt,
)

_PEP_A = Peptidoform(sequence="PEPTIDEKR")
_PEP_B = Peptidoform(sequence="SAMPLERMKL")


def _index(n_isotopes: int = 3) -> TheoreticalCandidateIndex:
    return TheoreticalCandidateIndex.from_peptides(
        [(0, _PEP_A, [2, 3]), (1, _PEP_B, [2])], n_isotopes=n_isotopes
    )


def test_index_built_sorted_and_sized() -> None:
    idx = _index(n_isotopes=3)
    # target 0: 2 charges × 3 isotopes = 6; target 1: 1 charge × 3 = 3
    assert len(idx) == 9
    assert torch.all(idx.mz[1:] >= idx.mz[:-1])  # mass-sorted
    assert set(idx.target_id.tolist()) == {0, 1}


def test_candidates_query_self_and_tolerance() -> None:
    idx = _index()
    mz0 = float(idx.mz[0])
    c = idx.candidates(mz0, tolerance_ppm=5.0)
    assert len(c) >= 1
    assert int(idx.target_id[0]) in c.target_id.tolist()
    # everything returned is within tolerance of the query
    assert torch.all((c.mz - mz0).abs() / mz0 * 1e6 <= 5.0 + 1e-6)
    # a far-off query resolves to nothing
    assert len(idx.candidates(1.0, tolerance_ppm=5.0)) == 0


def _keys(c) -> set[tuple[int, int, int]]:
    return set(zip(c.target_id.tolist(), c.charge.tolist(), c.isotope.tolist()))


def test_widening_tolerance_only_grows_the_candidate_set() -> None:
    idx = _index()
    mid = float(idx.mz[len(idx) // 2])
    tight = _keys(idx.candidates(mid, tolerance_ppm=1.0))
    wide = _keys(idx.candidates(mid, tolerance_ppm=500.0))
    assert tight <= wide  # candidates are monotone in tolerance (stability)


def test_index_mz_matches_progenitor_grid() -> None:
    # the index must use the SAME envelope as the fit, so an index m/z equals the
    # channel m/z Progenitor.for_peptide builds (else collision edges would diverge
    # from the grids the panel actually scores).
    cal = GlobalCalibration(n_isotopes=3, charges=[2, 3])
    prog = Progenitor.for_peptide(_PEP_A, [2], cal, n_isotopes=3)
    idx = TheoreticalCandidateIndex.from_peptides([(0, _PEP_A, [2])], n_isotopes=3)
    assert torch.allclose(idx.mz.sort().values, prog.channel_mz.sort().values)


# ──────────────────────────────────────────────────────────────────────
# PR-E — channel-overlap connected components
# ──────────────────────────────────────────────────────────────────────


def _chain_index() -> TheoreticalCandidateIndex:
    # three targets at 1-ppm m/z steps: A–B and B–C collide at ≥1 ppm
    long = torch.long
    return TheoreticalCandidateIndex(
        mz=torch.tensor([500.0, 500.0005, 500.0010], dtype=torch.float64),
        target_id=torch.tensor([0, 1, 2], dtype=long),
        charge=torch.tensor([2, 2, 2], dtype=long),
        isotope=torch.tensor([0, 0, 0], dtype=long),
    )


def test_components_partition_all_targets() -> None:
    comps = channel_overlap_components(_index(), collide_ppm=20.0)
    union = set().union(*comps)
    assert union == {0, 1}  # every target present
    assert sum(len(c) for c in comps) == len(union)  # disjoint partition


def test_collide_ppm_sensitivity_singletons_vs_merged() -> None:
    idx = _index()
    singletons = channel_overlap_components(idx, collide_ppm=1e-6)
    assert singletons == [frozenset({0}), frozenset({1})]  # tiny tol → all alone
    merged = channel_overlap_components(idx, collide_ppm=1e7)
    assert merged == [frozenset({0, 1})]  # huge tol → one component


def test_components_transitive_chain() -> None:
    # A–B (1 ppm) and B–C (1 ppm) → {A,B,C} one component even at <2 ppm tol
    assert channel_overlap_components(_chain_index(), collide_ppm=5.0) == [frozenset({0, 1, 2})]
    # tighter than the 1-ppm steps → all singletons
    assert channel_overlap_components(_chain_index(), collide_ppm=0.5) == [
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
    ]


def test_refine_components_by_rt_splits_non_coeluting() -> None:
    comp = [frozenset({0, 1, 2})]  # one m/z-overlap component of three targets
    rts = {0: 100.0, 1: 105.0, 2: 600.0}  # 0,1 co-elute; 2 is 8 min away
    units = refine_components_by_rt(comp, rts, rt_overlap_s=30.0)
    assert units == [frozenset({0, 1}), frozenset({2})]
    # a wide overlap window keeps the whole component together (transitive)
    assert refine_components_by_rt(comp, rts, rt_overlap_s=600.0) == [frozenset({0, 1, 2})]
    # a tight window splits every member off
    assert refine_components_by_rt(comp, rts, rt_overlap_s=1.0) == [
        frozenset({0}), frozenset({1}), frozenset({2})
    ]


def test_refine_components_keeps_rt_less_members_as_singletons() -> None:
    units = refine_components_by_rt([frozenset({0, 1})], {0: 100.0}, rt_overlap_s=30.0)
    assert units == [frozenset({0}), frozenset({1})]  # target 1 has no rt_center


def test_refine_components_span_bounded_not_transitive() -> None:
    # 0–50 and 50–100 are each ≤60 apart, but the 0..100 SPAN exceeds 60: span-bounded
    # grouping must NOT chain them into one unit (it would spread past a co-fit window).
    comp = [frozenset({0, 1, 2})]
    rts = {0: 0.0, 1: 50.0, 2: 100.0}
    units = refine_components_by_rt(comp, rts, rt_overlap_s=60.0)
    assert frozenset({0, 1, 2}) not in units
    for u in units:  # every unit's rt span stays within the window
        spans = [rts[m] for m in u]
        assert max(spans) - min(spans) <= 60.0
