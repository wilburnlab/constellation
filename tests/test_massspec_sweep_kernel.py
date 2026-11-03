"""The sweep's schedule and matching kernels (`massspec.sweep.{schedule, kernel}`).

Eligibility is checked against a brute force over every (scan, query)
pair; each kernel against a per-scan reference that runs
`core.signal`'s kernels on one unpadded scan at a time; and the two
kernels against each other, which is what lets the sweep pick either.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from constellation.core.signal import bounds_within_tolerance, nearest_within_tolerance
from constellation.massspec.sweep.kernel import match_cells, match_group, match_peaks
from constellation.massspec.sweep.rules import AllWithin, Nearest, SumWithin, rule_named
from constellation.massspec.sweep.schedule import (
    active_queries,
    cell_eligible,
    isolation_groups,
    ragged_arange,
)


def test_ragged_arange():
    assert ragged_arange(np.array([5, 0, 9]), np.array([2, 0, 3])).tolist() == [5, 6, 9, 10, 11]
    assert ragged_arange(np.array([], np.int64), np.array([], np.int64)).size == 0


def test_active_queries_inclusive_overlap():
    lo = np.array([0.0, 10.0, 20.0, -np.inf, np.nan, 5.0])
    hi = np.array([10.0, 20.0, 30.0, np.inf, np.nan, 5.0])
    has = np.array([True, True, True, True, True, False])
    assert active_queries(lo, hi, has, 10.0, 19.0).tolist() == [0, 1, 3]
    assert active_queries(lo, hi, has, 20.0, 20.0).tolist() == [1, 2, 3]


def _scans_and_queries(rng, n_scans=60, n_queries=80):
    windows = [(400.0, 425.0), (425.0, 450.0), (412.5, 437.5), (np.nan, np.nan), (430.1, 431.7)]
    pick = rng.integers(0, len(windows), n_scans)
    lower = np.array([windows[k][0] for k in pick])
    upper = np.array([windows[k][1] for k in pick])
    pmz = rng.uniform(395.0, 455.0, n_queries)
    pmz[:8] = [400.0, 425.0, 450.0, 412.5, 437.5, 430.1, 431.7, np.nan]  # on the edges
    pmz[rng.integers(0, n_queries, 6)] = np.nan
    return lower, upper, pmz


@pytest.mark.parametrize("seed", range(5))
def test_isolation_groups_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    lower, upper, pmz = _scans_and_queries(rng)
    active = np.sort(rng.choice(pmz.size, size=60, replace=False))
    rows = np.arange(lower.size)
    groups = isolation_groups(active, pmz, lower, upper, rows, gate=True)
    seen = np.concatenate([g.rows for g in groups])
    assert sorted(seen.tolist()) == sorted(set(seen.tolist()))
    for g in groups:
        assert np.all(np.diff(g.queries) > 0)
        for r in g.rows:
            lo, hi = lower[r], upper[r]
            want = [q for q in active
                    if np.isnan(lo) or np.isnan(hi) or np.isnan(pmz[q]) or lo <= pmz[q] <= hi]  # fmt: skip
            assert g.queries.tolist() == want
    # Every scan whose window covers some active query is in a group.
    for r in rows:
        lo, hi = lower[r], upper[r]
        if any(np.isnan(lo) or np.isnan(pmz[q]) or lo <= pmz[q] <= hi for q in active):
            assert r in seen


def test_ms1_is_one_group_with_every_active_query():
    active = np.array([2, 5, 7])
    groups = isolation_groups(active, np.full(9, np.nan), np.full(4, np.nan), np.full(4, np.nan),
                              np.arange(4), gate=False)  # fmt: skip
    assert len(groups) == 1 and groups[0].queries.tolist() == [2, 5, 7]


def test_cell_eligible():
    rt_lo = np.array([0.0, 10.0, -np.inf])
    rt_hi = np.array([10.0, 20.0, np.inf])
    assigned = np.array([-1, 7, 9])
    q = np.array([0, 0, 1, 1, 2, 2])
    rt = np.array([10.0, 10.5, 10.0, 15.0, 1.0, 1.0])
    scan = np.array([7, 7, 7, 8, 9, 10])
    assert cell_eligible(q, rt, scan, rt_lo, rt_hi, None).tolist() == [True, False, True, True, True, True]
    assert cell_eligible(q, rt, scan, rt_lo, rt_hi, assigned).tolist() == [
        True, False, True, False, True, False,
    ]  # fmt: skip


# ── kernels ───────────────────────────────────────────────────────────


def _block(rng, n_scans=25, tolerance=20.0, unit="ppm"):
    """Scans with sorted peaks (some duplicated, some empty), and ions that
    land on peaks, between peaks, exactly at the tolerance edge and halfway
    between two peaks."""
    mz_parts, counts = [], []
    for s in range(n_scans):
        n = 0 if s % 9 == 4 else int(rng.integers(1, 40))
        mz = np.sort(rng.uniform(200.0, 1200.0, n))
        if n > 3:
            mz[2] = mz[1]  # a duplicated m/z
        mz_parts.append(mz)
        counts.append(n)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    peaks = np.concatenate(mz_parts)
    intensity = rng.uniform(1.0, 1e6, peaks.size)
    sample = peaks[rng.integers(0, peaks.size, 60)]
    half = sample * tolerance * 1e-6 if unit == "ppm" else np.full(sample.size, tolerance)
    ions = np.concatenate([
        sample,
        sample + half,                      # on the upper edge
        sample - half,                      # on the lower edge
        sample * (1 + 3e-6),
        rng.uniform(200.0, 1200.0, 80),
        peaks[[1, 2, 5]] if peaks.size > 6 else [],  # the duplicate
    ])  # fmt: skip
    mids = (peaks[1:] + peaks[:-1]) / 2.0  # exact ties between neighbours
    ions = np.concatenate([ions, mids[rng.integers(0, mids.size, 30)]])
    return peaks, intensity, offsets, ions.astype(np.float64)


def _reference(peaks, intensity, offsets, rows, ions, rule, tolerance, unit):
    """One scan at a time, unpadded, through core.signal: (row, ion, peak, mz, intensity)."""
    out = []
    q = torch.from_numpy(ions)
    for j, r in enumerate(rows):
        a, b = offsets[r], offsets[r + 1]
        axis = torch.from_numpy(peaks[a:b])
        if rule == "nearest":
            best, within, _ = nearest_within_tolerance(axis, q, tolerance, unit)
            for m in np.flatnonzero(within.numpy()):
                k = int(best[m])
                out.append((j, int(m), k, peaks[a + k], intensity[a + k]))
        else:
            lo, hi = bounds_within_tolerance(axis, q, tolerance, unit)
            for m in range(ions.size):
                ks = list(range(int(lo[m]), int(hi[m])))
                if not ks:
                    continue
                if rule == "all":
                    out.extend((j, m, k, peaks[a + k], intensity[a + k]) for k in ks)
                else:
                    it = intensity[a + np.array(ks)]
                    out.append((j, m, -1, None, float(np.add.reduce(it))))
    return out


def _rows_of(m, rule):
    if rule == "sum":
        return [(int(r), int(i), int(p), None, float(x)) for r, i, p, x in
                zip(m.row, m.ion, m.peak, m.intensity)]  # fmt: skip
    return [(int(r), int(i), int(p), float(z), float(x)) for r, i, p, z, x in
            zip(m.row, m.ion, m.peak, m.mz, m.intensity)]  # fmt: skip


@pytest.mark.parametrize("unit,tolerance", [("ppm", 20.0), ("Da", 0.35), ("ppm", 10.0)])
@pytest.mark.parametrize("rule", ["nearest", "all", "sum"])
@pytest.mark.parametrize("seed", range(3))
def test_kernels_match_the_per_scan_reference(unit, tolerance, rule, seed):
    rng = np.random.default_rng(seed)
    peaks, intensity, offsets, ions = _block(rng, tolerance=tolerance, unit=unit)
    rows = np.array([0, 3, 4, 5, 8, 9, 13, 17, 20, 24])
    want = _reference(peaks, intensity, offsets, rows, ions, rule, tolerance, unit)
    for kernel in (match_cells, match_peaks):
        got = kernel(peaks, intensity, offsets, rows, ions, rule_named(rule), tolerance, unit)
        rows_got = _rows_of(got, rule)
        if rule == "sum":
            assert [r[:3] for r in rows_got] == [r[:3] for r in want], kernel.__name__
            np.testing.assert_allclose([r[4] for r in rows_got], [r[4] for r in want], rtol=1e-12)
        else:
            assert rows_got == want, kernel.__name__


@pytest.mark.parametrize("rule", [Nearest(), AllWithin(), SumWithin()])
def test_the_two_kernels_agree_bit_for_bit(rule):
    rng = np.random.default_rng(11)
    for unit, tolerance in (("ppm", 20.0), ("Da", 0.35)):
        peaks, intensity, offsets, ions = _block(rng, n_scans=40, tolerance=tolerance, unit=unit)
        rows = np.arange(40)
        a = match_cells(peaks, intensity, offsets, rows, ions, rule, tolerance, unit)
        b = match_peaks(peaks, intensity, offsets, rows, ions, rule, tolerance, unit)
        for f in ("row", "ion", "peak", "n_peaks"):
            np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)
        for f in ("mz", "intensity"):
            np.testing.assert_array_equal(getattr(a, f).view(np.int64), getattr(b, f).view(np.int64))


@pytest.mark.parametrize("max_elements", [1, 7, 64, 1 << 21])
def test_cell_chunking_changes_nothing(max_elements):
    rng = np.random.default_rng(3)
    peaks, intensity, offsets, ions = _block(rng)
    rows = np.arange(25)
    for rule in (Nearest(), AllWithin(), SumWithin()):
        whole = match_cells(peaks, intensity, offsets, rows, ions, rule, 20.0, "ppm")
        chunked = match_cells(peaks, intensity, offsets, rows, ions, rule, 20.0, "ppm",
                              max_elements=max_elements)  # fmt: skip
        for f in ("row", "ion", "peak", "n_peaks", "mz", "intensity"):
            np.testing.assert_array_equal(getattr(whole, f), getattr(chunked, f))


def test_auto_picks_peaks_when_ions_outnumber_peaks():
    rng = np.random.default_rng(5)
    peaks, intensity, offsets, ions = _block(rng)
    rows = np.arange(25)
    many = np.concatenate([ions] * 20)
    auto = match_group(peaks, intensity, offsets, rows, many, Nearest(), 20.0, "ppm")
    forced = match_peaks(peaks, intensity, offsets, rows, many, Nearest(), 20.0, "ppm")
    np.testing.assert_array_equal(auto.ion, forced.ion)
    with pytest.raises(ValueError, match="kernel"):
        match_group(peaks, intensity, offsets, rows, ions, Nearest(), 20.0, "ppm", kernel="fast")
    with pytest.raises(ValueError, match="unknown match rule"):
        rule_named("best")
