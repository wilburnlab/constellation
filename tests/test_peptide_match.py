"""Tests for ``massspec.peptide.match`` — generic m/z matcher and fragment assignment."""

from __future__ import annotations

import math

import pytest
import torch

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.peptide.match import (
    IonAssignment,
    MzMatch,
    assign_fragments,
    match_mz,
)
from constellation.massspec.peptide.ions import IonType


# ──────────────────────────────────────────────────────────────────────
# match_mz — basics
# ──────────────────────────────────────────────────────────────────────


def test_empty_query_returns_no_matches():
    assert match_mz(torch.tensor([]), torch.tensor([100.0])) == []


def test_empty_reference_returns_no_matches():
    assert match_mz(torch.tensor([100.0]), torch.tensor([])) == []


def test_exact_match_returns_one_record():
    matches = match_mz(
        torch.tensor([500.0]),
        torch.tensor([500.0]),
        tolerance=10.0,  # ppm — very tight
    )
    assert len(matches) == 1
    m = matches[0]
    assert m.query_idx == 0 and m.ref_idx == 0
    assert m.error_da == 0.0 and m.error_ppm == 0.0
    assert math.isnan(m.query_intensity)


def test_query_outside_tolerance_drops():
    """Query at 100, reference at 110, 20 ppm = 0.002 Da — way outside."""
    matches = match_mz(
        torch.tensor([100.0]),
        torch.tensor([110.0]),
        tolerance=20.0,
    )
    assert matches == []


def test_intensity_passes_through():
    matches = match_mz(
        torch.tensor([500.0]),
        torch.tensor([500.0]),
        query_intensity=torch.tensor([1234.5]),
    )
    assert matches[0].query_intensity == pytest.approx(1234.5)


def test_error_signs():
    """error_da = query - ref, signed."""
    # query slightly above reference: positive error
    m = match_mz(torch.tensor([500.001]), torch.tensor([500.0]), tolerance=10.0)
    assert m[0].error_da > 0
    assert m[0].error_ppm > 0
    # query slightly below reference: negative error
    m = match_mz(torch.tensor([499.999]), torch.tensor([500.0]), tolerance=10.0)
    assert m[0].error_da < 0
    assert m[0].error_ppm < 0


# ──────────────────────────────────────────────────────────────────────
# Tolerance — ppm vs Da
# ──────────────────────────────────────────────────────────────────────


def test_ppm_tolerance_scales_with_mz():
    """20 ppm at m/z 100 ≈ 0.002 Da, at m/z 1000 ≈ 0.02 Da. A query
    at 100.005 hits ref at 100.0 with Da=20 mDa tolerance but misses
    at 20 ppm; at m/z 1000 the same 0.005 Da error is well within 20 ppm."""
    # 5 mDa error at low m/z: outside 20 ppm (20 ppm of 100 = 2 mDa)
    assert match_mz(torch.tensor([100.005]), torch.tensor([100.0]), tolerance=20.0) == []
    # 5 mDa error at high m/z: well inside 20 ppm (20 ppm of 1000 = 20 mDa)
    assert len(match_mz(torch.tensor([1000.005]), torch.tensor([1000.0]), tolerance=20.0)) == 1


def test_da_tolerance_is_constant_across_mz():
    matches = match_mz(
        torch.tensor([100.005, 1000.005]),
        torch.tensor([100.0, 1000.0]),
        tolerance=0.01,
        tolerance_unit="Da",
    )
    assert len(matches) == 2


# ──────────────────────────────────────────────────────────────────────
# Sorted-vs-unsorted reference invariance
# ──────────────────────────────────────────────────────────────────────


def test_reference_unsorted_input_preserves_input_indices():
    """ref_idx in the result must index into the *input* ref_mz, not the
    sorted version."""
    # Two refs in descending order; closest to query 500 is ref index 1 (the 500.0)
    matches = match_mz(
        torch.tensor([500.0]),
        torch.tensor([1000.0, 500.0, 100.0]),
        tolerance=10.0,
    )
    assert len(matches) == 1
    assert matches[0].ref_idx == 1
    assert matches[0].ref_mz == pytest.approx(500.0)


# ──────────────────────────────────────────────────────────────────────
# 0/1/2-candidate behaviour and mutual-degeneracy
# ──────────────────────────────────────────────────────────────────────


def test_two_close_refs_within_tolerance_picks_closest_when_not_degenerate():
    """Refs at 500.0 and 500.05; query at 500.002. With 100 ppm tolerance
    (= 0.05 Da at this m/z) both refs are inside the query window AND
    within tolerance of each other ... actually at 100 ppm both are
    degenerate. Pick a sharper tolerance to test the non-degenerate path."""
    # Use Da to control the geometry exactly.
    # Refs 0.020 Da apart, tolerance = 0.015 → only one is in the window
    matches = match_mz(
        torch.tensor([500.0]),
        torch.tensor([500.0, 500.020]),
        tolerance=0.015,
        tolerance_unit="Da",
    )
    # Only the first ref is in the window.
    assert len(matches) == 1
    assert matches[0].ref_idx == 0


def test_mutually_degenerate_refs_double_assign_intensity():
    """Both refs within tolerance of the query AND within tolerance of
    each other → both get the query's intensity (the user's specified
    degenerate-theoretical exception)."""
    # Refs at 500.0 and 500.005, query at 500.002, tolerance = 0.01 Da.
    # The two refs are 0.005 apart, within the 0.01 Da tolerance → degenerate.
    matches = match_mz(
        torch.tensor([500.002]),
        torch.tensor([500.0, 500.005]),
        query_intensity=torch.tensor([42.0]),
        tolerance=0.01,
        tolerance_unit="Da",
    )
    assert len(matches) == 2
    assert {m.ref_idx for m in matches} == {0, 1}
    for m in matches:
        assert m.query_intensity == pytest.approx(42.0)


def test_non_degenerate_two_in_window_picks_closest():
    """Two refs both inside the query window but NOT within tolerance of
    each other → pick the closest one only (no double-assign)."""
    # Refs 0.030 Da apart; query midway, tolerance = 0.02 Da
    # Both are in window from query's perspective but refs span 0.030 > 0.020
    matches = match_mz(
        torch.tensor([500.015]),
        torch.tensor([500.0, 500.030]),
        tolerance=0.02,
        tolerance_unit="Da",
    )
    assert len(matches) == 1
    # 500.015 is equidistant from both; argmin lands on the first encountered.
    assert matches[0].ref_idx in (0, 1)


# ──────────────────────────────────────────────────────────────────────
# Symmetry — match_mz works for obs-vs-obs comparison
# ──────────────────────────────────────────────────────────────────────


def test_obs_vs_obs_symmetry():
    """match_mz has no peptidoform notion. Used to compare two observed
    spectra to each other, the matching should still work."""
    a = torch.tensor([100.001, 200.0, 300.05, 400.0])
    b = torch.tensor([100.0, 199.999, 350.0])  # 350.0 is unmatched
    matches = match_mz(a, b, tolerance=20.0)  # ppm
    pairs = {(m.query_idx, m.ref_idx) for m in matches}
    assert (0, 0) in pairs
    assert (1, 1) in pairs
    # 300.05 vs 350 is way outside tolerance, no match
    assert not any(m.query_idx == 2 for m in matches)


# ──────────────────────────────────────────────────────────────────────
# Validation errors
# ──────────────────────────────────────────────────────────────────────


def test_invalid_tolerance_unit_rejected():
    with pytest.raises(ValueError, match="tolerance_unit"):
        match_mz(torch.tensor([100.0]), torch.tensor([100.0]), tolerance_unit="foo")  # type: ignore


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError, match="tolerance"):
        match_mz(torch.tensor([100.0]), torch.tensor([100.0]), tolerance=-1.0)


# ──────────────────────────────────────────────────────────────────────
# assign_fragments — peptidoform-aware wrapper
# ──────────────────────────────────────────────────────────────────────


def test_assign_fragments_round_trips_through_synthetic_spectrum():
    """Generate the b/y ladder from a synthetic peptide, treat the
    theoretical m/z as 'observed', and check that assign_fragments
    recovers each ion's identity exactly."""
    pep = parse_proforma("PEPTIDE")
    # Get the theoretical fragment ions.
    table, _ = assign_fragments_helper(pep)
    # Use those as fake observed peaks. Pin float64 so the round-trip
    # exercises the matcher's m/z arithmetic, not float32 quantization.
    obs_mz = torch.tensor(
        table.column("mz_theoretical").to_pylist(), dtype=torch.float64
    )
    obs_int = torch.full((obs_mz.numel(),), 100.0, dtype=torch.float64)
    assignments = assign_fragments(
        pep,
        obs_mz,
        obs_int,
        tolerance=1.0,  # ppm — very tight; theoretical-vs-theoretical should match
    )
    # Every fragment ion should be assigned exactly once.
    assert len(assignments) == table.num_rows
    # Verify each assignment is internally consistent.
    for a in assignments:
        assert isinstance(a, IonAssignment)
        assert a.ion_type in (IonType.B, IonType.Y)
        assert a.position >= 0
        assert a.charge >= 1
        assert a.obs_intensity == pytest.approx(100.0)
        assert abs(a.error_da) < 1e-6


def test_assign_fragments_no_match_returns_empty():
    """Observed peaks far from any theoretical fragment → no assignments."""
    pep = parse_proforma("PEPTIDE")
    obs_mz = torch.tensor([1.0, 2.0, 3.0])  # well below any b/y ion
    obs_int = torch.tensor([100.0, 100.0, 100.0])
    assignments = assign_fragments(pep, obs_mz, obs_int, tolerance=20.0)
    assert assignments == []


# helper: just runs fragment_ladder so the test doesn't have to import it
def assign_fragments_helper(pep):
    from constellation.massspec.peptide.ions import fragment_ladder

    return fragment_ladder(pep, return_tensor=False)


# ──────────────────────────────────────────────────────────────────────
# Equivalence: match_mz vs a from-scratch reference of the three-branch
# policy — locks the core.signal re-base against behavioral drift.
# ──────────────────────────────────────────────────────────────────────


def _reference_matches(query_mz, ref_mz, tolerance, unit):
    """Pure-Python reference for match_mz's 0/1/2+ degenerate/closest policy.

    Builds the window via the same ``tolerance_window`` helper so this
    isolates the loop/branch logic from the (separately pinned) window
    arithmetic. Returns the set of ``(query_idx, ref_idx)`` pairs.
    """
    import bisect

    from constellation.core.signal import tolerance_window

    qmz = query_mz.to(torch.float64).tolist()
    rmz = ref_mz.to(torch.float64).tolist()
    if not qmz or not rmz:
        return set()

    order = sorted(range(len(rmz)), key=lambda j: rmz[j])  # stable; refs distinct
    rmz_sorted = [rmz[j] for j in order]
    tol = tolerance_window(query_mz.to(torch.float64), tolerance, unit).tolist()

    pairs = set()
    for i, q in enumerate(qmz):
        lo = bisect.bisect_left(rmz_sorted, q - tol[i])  # searchsorted side="left"
        hi = bisect.bisect_right(rmz_sorted, q + tol[i])  # searchsorted side="right"
        n = hi - lo
        if n == 0:
            continue
        if n == 1:
            pairs.add((i, order[lo]))
            continue
        span = rmz_sorted[hi - 1] - rmz_sorted[lo]
        if span <= tol[i]:  # mutually degenerate → emit all
            for sj in range(lo, hi):
                pairs.add((i, order[sj]))
        else:  # pick closest (strict <, first-encountered wins ties)
            best, best_d = lo, abs(rmz_sorted[lo] - q)
            for sj in range(lo + 1, hi):
                d = abs(rmz_sorted[sj] - q)
                if d < best_d:
                    best, best_d = sj, d
            pairs.add((i, order[best]))
    return pairs


def test_match_mz_matches_reference_policy_over_random_inputs():
    """match_mz agrees pair-for-pair with the reference across random
    spectra spanning ppm/Da, clustered (degenerate) refs, and orphans."""
    import random

    rng = random.Random(20260601)
    for trial in range(200):
        n_ref = rng.randint(1, 25)
        # Distinct refs (avoid exact ties so the sort permutation is
        # unambiguous between torch.sort and Python sorted).
        refs: list[float] = []
        seen: set[float] = set()
        while len(refs) < n_ref:
            base = rng.uniform(100.0, 2000.0)
            # Half the time, cluster near an existing ref to provoke the
            # degenerate / closest-pick branches.
            if refs and rng.random() < 0.5:
                base = refs[rng.randrange(len(refs))] + rng.uniform(-0.02, 0.02)
            val = round(base, 6)
            if val in seen or val <= 0:
                continue
            seen.add(val)
            refs.append(val)

        n_query = rng.randint(0, 20)
        queries: list[float] = []
        for _ in range(n_query):
            if refs and rng.random() < 0.7:  # near a ref
                q = refs[rng.randrange(len(refs))] + rng.uniform(-0.03, 0.03)
            else:  # orphan
                q = rng.uniform(100.0, 2000.0)
            queries.append(q)

        if rng.random() < 0.5:
            tolerance, unit = rng.choice([5.0, 20.0, 100.0]), "ppm"
        else:
            tolerance, unit = rng.choice([0.005, 0.01, 0.05]), "Da"

        query_mz = torch.tensor(queries, dtype=torch.float64)
        ref_mz = torch.tensor(refs, dtype=torch.float64)

        got = {
            (m.query_idx, m.ref_idx)
            for m in match_mz(
                query_mz, ref_mz, tolerance=tolerance, tolerance_unit=unit
            )
        }
        want = _reference_matches(query_mz, ref_mz, tolerance, unit)
        assert got == want, (
            f"trial {trial}: tol={tolerance}{unit} "
            f"refs={refs} queries={queries}\n got={sorted(got)}\nwant={sorted(want)}"
        )
