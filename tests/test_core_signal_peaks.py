"""Peak detection along a continuous axis (core.signal.peaks): the stages,
held to scipy where scipy has them, and the picker that composes them."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import find_peaks, peak_prominences, peak_widths

from constellation.core.signal.peaks import (
    DETECTORS,
    DetectorBounds,
    DiffMad,
    Fraction,
    LocalMaxima,
    MatchedFilter,
    Parabolic,
    PeakPicker,
    ResidualMad,
    RobustZ,
    RollingQuantile,
    SavitzkyGolay,
    Trace,
    Valley,
    WalkOut,
    make_stage,
)


def _trace(y: np.ndarray, count: np.ndarray) -> Trace:
    b, n = y.shape
    valid = torch.arange(n) < torch.tensor(count).unsqueeze(-1)
    t = torch.where(valid, torch.tensor(y), torch.tensor(float("nan"), dtype=torch.float64))
    return Trace(y=t, x=torch.arange(n, dtype=torch.float64).expand(b, n).clone(), valid=valid,
                 count=torch.tensor(count))  # fmt: skip


@pytest.mark.parametrize(
    "criteria",
    [dict(), dict(height=0.5), dict(threshold=0.4), dict(distance=4), dict(prominence=1.0), dict(width=2.0),
     dict(prominence=0.5, wlen=11), dict(distance=6, prominence=0.5, width=(1, 6)), dict(height=(None, 2.0))],
)
def test_local_maxima_is_find_peaks_row_by_row(criteria) -> None:
    g = np.random.default_rng(0)
    y = g.normal(size=(40, 80)).cumsum(-1)
    count = np.full(40, 80)
    count[::3] = 73
    c = LocalMaxima(**criteria)(_trace(y, count))
    for r in range(40):
        want, _ = find_peaks(y[r, : count[r]], **criteria)
        assert sorted(c.apex[r][c.valid[r]].tolist()) == list(want)


def test_flat_tops_and_ties_follow_scipy() -> None:
    """A flat top is one peak at its middle; equal heights within a
    distance keep the later one."""
    y = np.array([[0, 1, 1, 1, 0, 2, 2, 0, 0, 3, 0, 3, 0, 0.0]])
    want, _ = find_peaks(y[0])
    c = LocalMaxima()(_trace(y, np.array([14])))
    assert sorted(c.apex[0][c.valid[0]].tolist()) == list(want)
    c = LocalMaxima(distance=3)(_trace(y, np.array([14])))
    assert sorted(c.apex[0][c.valid[0]].tolist()) == list(find_peaks(y[0], distance=3)[0])


def test_prominences_and_widths_are_scipys() -> None:
    g = np.random.default_rng(1)
    y = g.normal(size=(10, 120)).cumsum(-1)
    c = LocalMaxima()(_trace(y, np.full(10, 120)))
    for r in range(10):
        peaks = np.sort(c.apex[r][c.valid[r]].numpy())
        order = np.argsort(c.apex[r][c.valid[r]].numpy())
        prom, lb, rb = peak_prominences(y[r], peaks)
        np.testing.assert_allclose(c.extras["prominence"][r][c.valid[r]].numpy()[order], prom)
        np.testing.assert_array_equal(c.left[r][c.valid[r]].numpy()[order], lb)
        np.testing.assert_array_equal(c.right[r][c.valid[r]].numpy()[order], rb)
        w = peak_widths(y[r], peaks, rel_height=0.5)
        np.testing.assert_allclose(c.extras["width_samples"][r][c.valid[r]].numpy()[order], w[0])
        np.testing.assert_allclose(c.extras["left_ip"][r][c.valid[r]].numpy()[order], w[2])


def _chromatograms(seed: int = 3, b: int = 6, n: int = 120):
    """Irregularly sampled traces: two Gaussian peaks on a sloped baseline
    with noise, a few samples missing."""
    g = np.random.default_rng(seed)
    x = np.cumsum(g.uniform(2.0, 3.5, size=(b, n)), axis=1)
    apex = np.stack([x[:, 30], x[:, 80]], 1)
    sigma = 6.0
    y = 50 * np.exp(-0.5 * ((x - apex[:, :1]) / sigma) ** 2) + 20 * np.exp(-0.5 * ((x - apex[:, 1:]) / sigma) ** 2)
    y += 2.0 + 0.01 * x + g.normal(scale=0.8, size=(b, n))
    valid = g.random((b, n)) > 0.05
    return torch.tensor(y), torch.tensor(x), torch.tensor(valid), apex, sigma


@pytest.mark.parametrize(
    ("detect", "bounds"),
    [(LocalMaxima(prominence=8.0), Fraction(0.05)), (LocalMaxima(prominence=8.0), WalkOut()),
     (MatchedFilter(widths=(1.5, 2.5, 4.0), threshold=4.0), DetectorBounds()),
     (RobustZ(k=5.0, k_low=2.0), DetectorBounds())],
)
def test_every_detector_finds_the_two_peaks_and_measures_them_alike(detect, bounds) -> None:
    y, x, valid, apex, sigma = _chromatograms()
    picker = PeakPicker(detect=detect, smooth=SavitzkyGolay(7, 2), baseline=RollingQuantile(61),
                        noise=ResidualMad(), boundaries=bounds, refine=Parabolic(log=True))  # fmt: skip
    p = picker(y, x=x, valid=valid)
    assert (p.n == 2).all()
    for r in range(y.shape[0]):
        found = np.sort(p.apex_x[r][p.valid[r]].numpy())
        np.testing.assert_allclose(found, apex[r], atol=3.5)  # about one sample spacing
        big = int(torch.argmax(torch.where(p.valid[r], p.height[r], torch.tensor(-1e9, dtype=torch.float64))))
        assert p.width[r, big].item() == pytest.approx(2.3548 * sigma, rel=0.3)
        assert p.snr[r, big].item() > 20 and p.jaggedness[r, big].item() < 0.5
        lo, hi = p.left[r, big].item(), p.right[r, big].item()
        assert valid[r, lo] and valid[r, hi] and lo < p.apex[r, big].item() < hi


def test_peaks_do_not_depend_on_their_neighbours_in_the_batch_or_on_padding() -> None:
    y, x, valid, _, _ = _chromatograms()
    picker = PeakPicker(detect=LocalMaxima(prominence=8.0), smooth=SavitzkyGolay(7, 2), noise=DiffMad())
    whole = picker(y, x=x, valid=valid)
    for r in range(y.shape[0]):
        alone = picker(y[r : r + 1], x=x[r : r + 1], valid=valid[r : r + 1])
        wider = torch.cat([y[r : r + 1], torch.zeros(1, 9, dtype=torch.float64)], dim=-1)
        wider_x = torch.cat([x[r : r + 1], torch.zeros(1, 9, dtype=torch.float64)], dim=-1)
        wider_ok = torch.cat([valid[r : r + 1], torch.zeros(1, 9, dtype=torch.bool)], dim=-1)
        padded = picker(wider, x=wider_x, valid=wider_ok)
        for other in (alone, padded):
            k = int(other.n[0])
            assert k == int(whole.n[r])
            assert torch.equal(other.apex[0, :k], whole.apex[r, :k])
            assert torch.allclose(other.area[0, :k], whole.area[r, :k])


def test_a_boundary_trace_bounds_a_peak_found_in_another() -> None:
    """Detection on a score, extent on the intensity: the apex moves to the
    intensity's maximum nearby, and the width is the intensity's."""
    n = 80
    x = torch.arange(n, dtype=torch.float64)
    intensity = 100 * torch.exp(-0.5 * ((x - 40) / 4) ** 2)
    score = torch.exp(-0.5 * ((x - 38) / 1.5) ** 2)  # a sharper score, off by two samples
    picker = PeakPicker(detect=LocalMaxima(prominence=0.5), boundaries=Fraction(0.05), recenter=3)
    p = picker(score.unsqueeze(0), x=x, boundary_trace=intensity.unsqueeze(0))
    assert p.apex[0, 0].item() == 38 and p.boundary_apex[0, 0].item() == 40
    assert p.width[0, 0].item() == pytest.approx(2.3548 * 4, rel=0.02)
    assert p.area[0, 0].item() == pytest.approx(100 * 4 * np.sqrt(2 * np.pi), rel=0.02)


def test_walk_out_stops_where_a_neighbour_rises() -> None:
    x = torch.arange(100, dtype=torch.float64)
    y = 100 * torch.exp(-0.5 * ((x - 40) / 3) ** 2) + 60 * torch.exp(-0.5 * ((x - 58) / 3) ** 2)
    picker = PeakPicker(detect=LocalMaxima(prominence=10.0), boundaries=WalkOut(rise=1))
    p = picker(y.unsqueeze(0), x=x)
    first = int(torch.argmax(torch.where(p.valid[0], p.height[0], torch.tensor(-1.0, dtype=torch.float64))))
    valley = int(torch.argmin(y[40:58])) + 40
    assert p.right[0, first].item() == valley
    assert p.left[0, first].item() < 32  # the left side runs out below 1 %


def test_robust_z_finds_a_local_outlier_on_a_drifting_baseline() -> None:
    g = np.random.default_rng(7)
    x = np.arange(400.0)
    y = 0.05 * x + g.normal(scale=1.0, size=400)
    y[200:205] += 12.0
    c = PeakPicker(detect=RobustZ(k=6.0, k_low=2.0, window=41), boundaries=DetectorBounds())(torch.tensor(y)[None])
    assert int(c.n[0]) == 1 and 200 <= c.apex[0, 0].item() <= 204
    assert 198 <= c.left[0, 0].item() <= 200 and 204 <= c.right[0, 0].item() <= 206


def test_a_picker_is_named_by_its_stages_and_rebuilt_from_them() -> None:
    picker = PeakPicker(detect=MatchedFilter(widths=(1.0, 2.0)), smooth=SavitzkyGolay(5, 2),
                        baseline=RollingQuantile(31), noise=ResidualMad(), boundaries=WalkOut(rise=3))  # fmt: skip
    again = PeakPicker.from_identity(picker.identity())
    assert again == picker and again.key() == picker.key()
    other = PeakPicker(detect=MatchedFilter(widths=(1.0, 2.5)), smooth=SavitzkyGolay(5, 2),
                       baseline=RollingQuantile(31), noise=ResidualMad(), boundaries=WalkOut(rise=3))  # fmt: skip
    assert other.key() != picker.key()
    assert set(DETECTORS) >= {"local_maxima", "matched_filter", "robust_z"}
    assert make_stage("boundaries", "valley") == Valley()
    with pytest.raises(KeyError, match="available"):
        make_stage("detect", "cwt_ridges")
