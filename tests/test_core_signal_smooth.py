"""Batched, masked smoothing, baselines, noise and consensus (core.signal)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage
from scipy.signal import savgol_filter

from constellation.core.signal._rows import RangeTable, compact, expand
from constellation.core.signal.baseline import opening, rolling_quantile, snip
from constellation.core.signal.consensus import collapse
from constellation.core.signal.noise import diff_mad, residual_mad, rolling_mad
from constellation.core.signal.smooth import gaussian, moving_average, savitzky_golay


@pytest.mark.parametrize(("window", "degree"), [(3, 1), (5, 2), (7, 3), (11, 4)])
def test_savitzky_golay_is_scipys_inside_a_full_row(window, degree) -> None:
    y = np.random.default_rng(window).normal(size=(4, 60))
    h = window // 2
    for d in range(degree + 1):
        got = savitzky_golay(torch.tensor(y), window, degree, derivative=d).numpy()
        want = savgol_filter(y, window, degree, deriv=d, axis=-1)
        np.testing.assert_allclose(got[:, h:-h], want[:, h:-h], atol=1e-10)


def test_savitzky_golay_fits_only_the_samples_a_window_holds() -> None:
    g = np.random.default_rng(1)
    y = g.normal(size=(3, 40))
    valid = g.random((3, 40)) > 0.3
    out = savitzky_golay(torch.tensor(y), 7, 2, valid=torch.tensor(valid)).numpy()
    assert np.isnan(out[~valid]).all() and np.isfinite(out[valid]).all()
    for r, i in ((0, 20), (1, 2), (2, 37)):
        if not valid[r, i]:
            continue
        lo, hi = max(i - 3, 0), min(i + 4, 40)
        xs = np.arange(lo, hi) - i
        ok = valid[r, lo:hi]
        want = np.polyval(np.polyfit(xs[ok], y[r, lo:hi][ok], 2), 0) if ok.sum() >= 3 else y[r, lo:hi][ok].mean()
        assert out[r, i] == pytest.approx(want, abs=1e-10)


def test_smoothers_ignore_invalid_samples_whatever_their_values() -> None:
    y = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64)
    poisoned = y.clone()
    poisoned[0, 2] = 1e9
    valid = torch.tensor([[True, True, False, True, True]])
    for fn in (lambda v, m: moving_average(v, 3, valid=m), lambda v, m: gaussian(v, 1.0, valid=m),
               lambda v, m: savitzky_golay(v, 3, 1, valid=m)):  # fmt: skip
        a, b = fn(y, valid), fn(poisoned, valid)
        assert torch.equal(torch.nan_to_num(a, nan=-1), torch.nan_to_num(b, nan=-1))


def test_rolling_median_and_opening_are_ndimages_inside_the_row() -> None:
    y = np.random.default_rng(2).normal(size=(3, 50))
    med = rolling_quantile(torch.tensor(y), 7).numpy()
    want = np.median(np.lib.stride_tricks.sliding_window_view(y, 7, axis=1), axis=-1)
    np.testing.assert_allclose(med[:, 3:-3], want, atol=0)
    op = opening(torch.tensor(y), 5).numpy()
    np.testing.assert_allclose(op[:, 4:-4], ndimage.grey_opening(y, size=(1, 5))[:, 4:-4], atol=0)


def test_snip_follows_a_sloped_baseline_under_a_peak() -> None:
    t = np.linspace(0, 10, 200)
    base = 3 + 0.2 * t
    trace = base + 50 * np.exp(-0.5 * ((t - 5) / 0.2) ** 2)
    got = snip(torch.tensor(trace[None]), 30).numpy()[0]
    assert np.abs(got[20:180] - base[20:180]).max() < 0.15


def test_noise_estimates_recover_sigma() -> None:
    noise = np.random.default_rng(4).normal(scale=2.0, size=(4, 5000))
    np.testing.assert_allclose(diff_mad(torch.tensor(noise)).numpy(), 2.0, rtol=0.05)
    np.testing.assert_allclose(residual_mad(torch.tensor(noise), torch.zeros(4, 5000, dtype=torch.float64)).numpy(),
                               2.0, rtol=0.05)  # fmt: skip
    local = rolling_mad(torch.tensor(noise[:1, :600]), 101).numpy()
    assert np.nanmedian(local) == pytest.approx(2.0, rel=0.15)
    assert diff_mad(torch.zeros(1, 10, dtype=torch.float64), floor=0.5).item() == 0.5


def test_kth_largest_needs_k_traces_to_rise_together() -> None:
    y = torch.zeros(1, 4, 9, dtype=torch.float64)
    y[0, 0, 4] = 100.0  # one interfering trace
    s, _ = collapse(y, method="kth_largest", k=3)
    assert s.max().item() == 0.0
    y[0, 1:3, 4] = torch.tensor([50.0, 40.0], dtype=torch.float64)
    s, count = collapse(y, method="kth_largest", k=3)
    assert s[0, 4].item() == 40.0 and count[0, 4].item() == 4
    g = torch.Generator().manual_seed(0)
    r = torch.rand(2, 5, 7, generator=g, dtype=torch.float64)
    s, _ = collapse(r, method="kth_largest", k=2, scale=torch.full((2, 5), 2.0, dtype=torch.float64))
    assert torch.allclose(s, 2 * torch.sort(r, dim=-2, descending=True).values[..., 1, :])
    m, _ = collapse(r, method="mean", valid=torch.ones_like(r, dtype=torch.bool))
    assert torch.allclose(m, r.mean(dim=-2))


def test_compaction_round_trips() -> None:
    valid = torch.tensor([[False, True, True, False, True], [True, False, False, False, False]])
    y = torch.arange(10, dtype=torch.float64).reshape(2, 5)
    c = compact(valid)
    assert c.count.tolist() == [3, 1]
    assert c.gather(y, -1.0).tolist() == [[1.0, 2.0, 4.0, -1.0, -1.0], [5.0, -1.0, -1.0, -1.0, -1.0]]
    assert torch.equal(expand(c, c.gather(y), -1.0), torch.where(valid, y, torch.full_like(y, -1.0)))


def test_range_table_searches_are_the_brute_force_ones() -> None:
    g = np.random.default_rng(0)
    for n in (1, 2, 5, 17, 64):
        y = torch.tensor(g.integers(0, 6, (3, n)).astype(float))
        count = torch.tensor([n, max(n - 3, 0), n // 2])
        t = RangeTable.build(y, count)
        yn = y.numpy()
        for r in range(3):
            c = int(count[r])
            for p in range(c):
                full = lambda v: torch.full((3, 1), v)  # noqa: E731
                level = yn[r, p] - 0.5
                want = next((j for j in range(p - 1, -1, -1) if yn[r, j] > level), -1)
                assert int(t.nearest_left(full(p), full(level))[r, 0]) == want
                want = next((j for j in range(p + 1, c) if yn[r, j] < level + 1), -1)
                assert int(t.nearest_right(full(p), full(level + 1), above=False)[r, 0]) == want
                for q in range(p, c):
                    seg = yn[r, p : q + 1]
                    v, i = t.argmin(full(p), full(q), prefer="right")
                    assert float(v[r, 0]) == seg.min() and int(i[r, 0]) == p + int(np.flatnonzero(seg == seg.min())[-1])
