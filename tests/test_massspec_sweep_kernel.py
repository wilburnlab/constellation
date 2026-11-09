"""The sweep's schedule and matching kernels (`massspec.sweep.{schedule, kernel}`).

Eligibility is checked against a brute force over every (scan, query)
pair; each kernel (channel-driven, point-driven) against a per-scan
reference that runs `core.signal`'s kernels on one unpadded scan at a
time; and the two kernels against each other, which is what lets the
sweep pick either.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from constellation.core.signal import bounds_within_tolerance, nearest_within_tolerance
from constellation.massspec.sweep.kernel import match_by_channel, match_by_point, match_group
from constellation.massspec.sweep.rules import AllWithin, Nearest, SumWithin, rule_named
from constellation.massspec.sweep.schedule import (
    active_queries,
    isolation_groups,
    pair_eligible,
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


def test_pair_eligible():
    rt_lo = np.array([0.0, 10.0, -np.inf])
    rt_hi = np.array([10.0, 20.0, np.inf])
    assigned = np.array([-1, 7, 9])
    q = np.array([0, 0, 1, 1, 2, 2])
    rt = np.array([10.0, 10.5, 10.0, 15.0, 1.0, 1.0])
    scan = np.array([7, 7, 7, 8, 9, 10])
    assert pair_eligible(q, rt, scan, rt_lo, rt_hi, None).tolist() == [True, False, True, True, True, True]
    assert pair_eligible(q, rt, scan, rt_lo, rt_hi, assigned).tolist() == [
        True, False, True, False, True, False,
    ]  # fmt: skip


# ── kernels ───────────────────────────────────────────────────────────


def _block(rng, n_scans=25, tolerance=20.0, unit="ppm"):
    """Scans with sorted points (some duplicated, some empty), and channels
    that land on points, between points, exactly at the tolerance edge and
    halfway between two points."""
    mz_parts, counts = [], []
    for s in range(n_scans):
        n = 0 if s % 9 == 4 else int(rng.integers(1, 40))
        mz = np.sort(rng.uniform(200.0, 1200.0, n))
        if n > 3:
            mz[2] = mz[1]  # a duplicated m/z
        mz_parts.append(mz)
        counts.append(n)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    points = np.concatenate(mz_parts)
    intensity = rng.uniform(1.0, 1e6, points.size)
    sample = points[rng.integers(0, points.size, 60)]
    half = sample * tolerance * 1e-6 if unit == "ppm" else np.full(sample.size, tolerance)
    channels = np.concatenate([
        sample,
        sample + half,                      # on the upper edge
        sample - half,                      # on the lower edge
        sample * (1 + 3e-6),
        rng.uniform(200.0, 1200.0, 80),
        points[[1, 2, 5]] if points.size > 6 else [],  # the duplicate
    ])  # fmt: skip
    mids = (points[1:] + points[:-1]) / 2.0  # exact ties between neighbours
    channels = np.concatenate([channels, mids[rng.integers(0, mids.size, 30)]])
    return points, intensity, offsets, channels.astype(np.float64)


def _reference(points, intensity, offsets, rows, channels, rule, tolerance, unit):
    """One scan at a time, unpadded, through core.signal: (row, channel, point, mz, intensity)."""
    out = []
    q = torch.from_numpy(channels)
    for j, r in enumerate(rows):
        a, b = offsets[r], offsets[r + 1]
        axis = torch.from_numpy(points[a:b])
        if rule == "nearest":
            best, within, _ = nearest_within_tolerance(axis, q, tolerance, unit)
            for m in np.flatnonzero(within.numpy()):
                k = int(best[m])
                out.append((j, int(m), k, points[a + k], intensity[a + k]))
        else:
            lo, hi = bounds_within_tolerance(axis, q, tolerance, unit)
            for m in range(channels.size):
                ks = list(range(int(lo[m]), int(hi[m])))
                if not ks:
                    continue
                if rule == "all":
                    out.extend((j, m, k, points[a + k], intensity[a + k]) for k in ks)
                else:
                    it = intensity[a + np.array(ks)]
                    out.append((j, m, -1, None, float(np.add.reduce(it))))
    return out


def _rows_of(m, rule):
    if rule == "sum":
        return [(int(r), int(i), int(p), None, float(x)) for r, i, p, x in
                zip(m.row, m.channel, m.point, m.intensity)]  # fmt: skip
    return [(int(r), int(i), int(p), float(z), float(x)) for r, i, p, z, x in
            zip(m.row, m.channel, m.point, m.mz, m.intensity)]  # fmt: skip


@pytest.mark.parametrize("unit,tolerance", [("ppm", 20.0), ("Da", 0.35), ("ppm", 10.0)])
@pytest.mark.parametrize("rule", ["nearest", "all", "sum"])
@pytest.mark.parametrize("seed", range(3))
def test_kernels_match_the_per_scan_reference(unit, tolerance, rule, seed):
    rng = np.random.default_rng(seed)
    points, intensity, offsets, channels = _block(rng, tolerance=tolerance, unit=unit)
    rows = np.array([0, 3, 4, 5, 8, 9, 13, 17, 20, 24])
    want = _reference(points, intensity, offsets, rows, channels, rule, tolerance, unit)
    for kernel in (match_by_channel, match_by_point):
        got = kernel(points, intensity, offsets, rows, channels, rule_named(rule), tolerance, unit)
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
        points, intensity, offsets, channels = _block(rng, n_scans=40, tolerance=tolerance, unit=unit)
        rows = np.arange(40)
        a = match_by_channel(points, intensity, offsets, rows, channels, rule, tolerance, unit)
        b = match_by_point(points, intensity, offsets, rows, channels, rule, tolerance, unit)
        for f in ("row", "channel", "point", "n_points"):
            np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)
        for f in ("mz", "intensity"):
            np.testing.assert_array_equal(getattr(a, f).view(np.int64), getattr(b, f).view(np.int64))


@pytest.mark.parametrize("max_elements", [1, 7, 64, 1 << 21])
def test_channel_driven_chunking_changes_nothing(max_elements):
    rng = np.random.default_rng(3)
    points, intensity, offsets, channels = _block(rng)
    rows = np.arange(25)
    for rule in (Nearest(), AllWithin(), SumWithin()):
        whole = match_by_channel(points, intensity, offsets, rows, channels, rule, 20.0, "ppm")
        chunked = match_by_channel(points, intensity, offsets, rows, channels, rule, 20.0, "ppm",
                                   max_elements=max_elements)  # fmt: skip
        for f in ("row", "channel", "point", "n_points", "mz", "intensity"):
            np.testing.assert_array_equal(getattr(whole, f), getattr(chunked, f))


def test_auto_is_point_driven_when_channels_outnumber_points():
    rng = np.random.default_rng(5)
    points, intensity, offsets, channels = _block(rng)
    rows = np.arange(25)
    many = np.concatenate([channels] * 20)
    auto = match_group(points, intensity, offsets, rows, many, Nearest(), 20.0, "ppm")
    forced = match_by_point(points, intensity, offsets, rows, many, Nearest(), 20.0, "ppm")
    np.testing.assert_array_equal(auto.channel, forced.channel)
    with pytest.raises(ValueError, match="kernel"):
        match_group(points, intensity, offsets, rows, channels, Nearest(), 20.0, "ppm", kernel="fast")
    with pytest.raises(ValueError, match="kernel"):
        match_group(points, intensity, offsets, rows, channels, Nearest(), 20.0, "ppm", kernel="cells")
    with pytest.raises(ValueError, match="unknown match rule"):
        rule_named("best")


def test_order_by_is_lexsort_row_for_row():
    """The combined-key sort the match path uses is lexsort's order exactly,
    ties and negative keys included, and falls back when keys do not fit."""
    from constellation.massspec.sweep.schedule import order_by

    rng = np.random.default_rng(11)
    for n in (0, 1, 7, 5000):
        keys = [rng.integers(-3, 4, n), rng.integers(0, 2, n), rng.integers(-1, 900, n), rng.integers(0, 3, n)]
        np.testing.assert_array_equal(order_by(*keys), np.lexsort(keys[::-1]))
    wide = [rng.integers(0, 2**40, 300), rng.integers(0, 2**40, 300)]
    np.testing.assert_array_equal(order_by(*wide), np.lexsort(wide[::-1]))
    floats = [rng.random(50), rng.integers(0, 3, 50)]
    np.testing.assert_array_equal(order_by(*floats), np.lexsort(floats[::-1]))


@pytest.mark.parametrize("unit,tol", [("Da", 0.25), ("ppm", 400.0)])
def test_nearest_without_a_sort_decides_ties_and_duplicates_as_before(unit, tol):
    """The point-driven ``nearest`` reads each candidate's two neighbours
    instead of sorting candidates into cells: duplicated m/z, channels
    exactly between two points and exactly at the tolerance must come out
    as the channel-driven kernel (the definition) has them."""
    from constellation.massspec.sweep.kernel import match_by_channel, match_by_point
    from constellation.massspec.sweep.rules import Nearest

    rng = np.random.default_rng(4)
    grid = np.round(rng.uniform(500.0, 503.0, 40), 2)
    scans = []
    for _ in range(30):
        mz = np.sort(rng.choice(grid, int(rng.integers(0, 12))))  # duplicates on purpose
        scans.append(mz)
    offsets = np.r_[0, np.cumsum([s.size for s in scans])].astype(np.int64)
    points = np.concatenate(scans)
    intensity = rng.uniform(1, 9, points.size)
    mids = (grid[:-1] + grid[1:]) / 2
    edges = grid + (tol if unit == "Da" else grid * tol * 1e-6)
    channels = np.concatenate([grid, mids, edges, rng.uniform(499.5, 503.5, 60)])
    rows = np.arange(len(scans), dtype=np.int64)
    a = match_by_point(points, intensity, offsets, rows, channels, Nearest(), tol, unit)
    b = match_by_channel(points, intensity, offsets, rows, channels, Nearest(), tol, unit)
    for name in ("row", "channel", "point", "n_points", "mz", "intensity"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.n > 50
