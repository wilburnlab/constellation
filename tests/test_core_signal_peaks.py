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


# ── Codex review of 2026-11-11 ─────────────────────────────────────────


def test_padding_does_not_move_a_boundary_on_the_last_sample() -> None:
    """A boundary on a row's last valid sample is that sample's x. Until
    2026-11-11 NaN padding after it made ``right_x`` NaN (the next sample
    read through 0 · NaN)."""
    y = torch.tensor([[0, 1, 4, 1, 0.0]], dtype=torch.float64)
    picker = PeakPicker(detect=LocalMaxima(), boundaries=Valley())
    padded = picker(torch.cat([y, torch.full((1, 3), float("nan"), dtype=torch.float64)], dim=-1))
    assert padded.left_x[0, 0].item() == 0.0 and padded.right_x[0, 0].item() == 4.0
    assert padded.right_x[0, 0].item() == picker(y).right_x[0, 0].item()


def _score_and_intensity(n: int = 60):
    x = torch.arange(n, dtype=torch.float64)
    score = torch.exp(-0.5 * ((x - 30) / 2) ** 2)
    intensity = 100 * torch.exp(-0.5 * ((x - 30) / 4) ** 2)
    return x, score[None], intensity[None]


@pytest.mark.parametrize("bounds", [Valley(), WalkOut(), Fraction(0.01), DetectorBounds()])
def test_a_boundary_trace_hole_is_never_an_apex_or_an_endpoint(bounds) -> None:
    """Holes in the boundary trace (``boundary_valid`` false, or NaN) are
    passed over: a walk stops before one, a search or a minimum ignores it.
    Until 2026-11-11 a walk ended on a hole and a search stopped at one."""
    x, score, intensity = _score_and_intensity()
    holes = [17, 22, 30, 38, 43]  # a fraction's crossings, the flanks, the apex
    bv = torch.ones_like(score, dtype=torch.bool)
    bv[0, holes[:-1]] = False
    intensity = intensity.clone()
    intensity[0, holes[-1]] = float("nan")
    picker = PeakPicker(detect=LocalMaxima(prominence=0.5), boundaries=bounds, recenter=2)
    p = picker(score, x=x, boundary_trace=intensity, boundary_valid=bv)
    assert int(p.n[0]) == 1 and p.apex[0, 0].item() == 30
    for field in (p.boundary_apex, p.left, p.right):
        assert field[0, 0].item() not in holes and field[0, 0].item() >= 0
    assert p.boundary_apex[0, 0].item() in (29, 31)
    if not isinstance(bounds, WalkOut):  # which stops at the apex's hole
        # The area joins the samples on either side of a hole.
        assert p.area[0, 0].item() == pytest.approx(100 * 4 * np.sqrt(2 * np.pi), rel=0.05)


def test_a_walk_stops_before_a_hole_and_a_fraction_passes_it() -> None:
    x, score, intensity = _score_and_intensity()
    bv = torch.ones_like(score, dtype=torch.bool)
    bv[0, [17, 22, 38, 43]] = False
    p = PeakPicker(detect=LocalMaxima(prominence=0.5), boundaries=WalkOut())(score, x=x, boundary_trace=intensity,
                                                                              boundary_valid=bv)  # fmt: skip
    assert (p.left[0, 0].item(), p.right[0, 0].item()) == (23, 37)
    p = PeakPicker(detect=LocalMaxima(prominence=0.5), boundaries=Fraction(0.01))(score, x=x, boundary_trace=intensity,
                                                                                   boundary_valid=bv)  # fmt: skip
    assert (p.left[0, 0].item(), p.right[0, 0].item()) == (16, 44)


def test_a_peak_with_no_boundary_sample_near_keeps_only_what_detection_gave_it() -> None:
    x, score, intensity = _score_and_intensity()
    picker = PeakPicker(detect=LocalMaxima(prominence=0.5), boundaries=WalkOut(), recenter=2)
    for bv in (torch.zeros_like(score, dtype=torch.bool), (x < 20) | (x > 40)):
        p = picker(score, x=x, boundary_trace=intensity, boundary_valid=bv.reshape(1, -1))
        assert int(p.n[0]) == 1 and p.apex[0, 0].item() == 30
        assert p.height[0, 0].item() == 1.0 and np.isfinite(p.prominence[0, 0].item())
        assert (p.boundary_apex[0, 0].item(), p.left[0, 0].item(), p.right[0, 0].item()) == (-1, -1, -1)
        for field in (p.width, p.area, p.asymmetry, p.jaggedness, p.left_x, p.right_x):
            assert torch.isnan(field[0, 0])


def test_a_boundary_trace_hole_at_the_half_height_crossing_is_interpolated_across() -> None:
    x, score, intensity = _score_and_intensity()
    bv = torch.ones_like(score, dtype=torch.bool)
    bv[0, [26, 34]] = False  # the samples just inside each half-height crossing
    picker = PeakPicker(detect=LocalMaxima(prominence=0.5), boundaries=Fraction(0.01))
    whole = picker(score, x=x, boundary_trace=intensity)
    holed = picker(score, x=x, boundary_trace=intensity, boundary_valid=bv)
    assert whole.width[0, 0].item() == pytest.approx(2.3548 * 4, rel=0.01)
    assert holed.width[0, 0].item() == pytest.approx(whole.width[0, 0].item(), rel=0.01)


@pytest.mark.parametrize("template", ["gaussian", "ricker"])
def test_the_matched_filter_finds_nothing_on_a_constant(template) -> None:
    """Until 2026-11-11 a template truncated at the row's edge answered a
    constant, and a response MAD of 0 made it two peaks scored 1e300."""
    ones = torch.ones(2, 100, dtype=torch.float64)
    valid = torch.ones_like(ones, dtype=torch.bool)
    valid[1, [3, 4, 50, 51, 52, 97]] = False
    p = PeakPicker(detect=MatchedFilter(widths=(2.0, 4.0), template=template))(ones, valid=valid)
    assert int(p.n.sum()) == 0


@pytest.mark.parametrize("template", ["gaussian", "ricker"])
def test_the_matched_filter_finds_a_bump_on_an_offset_once(template) -> None:
    g = np.random.default_rng(11)
    x = np.arange(200.0)
    y = 1000.0 + 10 * np.exp(-0.5 * ((x - 120) / 3) ** 2) + g.normal(scale=0.2, size=200)
    picker = PeakPicker(detect=MatchedFilter(widths=(3.0,), template=template, threshold=5.0), max_peaks=8)
    p = picker(torch.tensor(y)[None])
    assert int(p.n[0]) == 1 and abs(p.apex[0, 0].item() - 120) <= 1
    shifted = picker(torch.tensor(y - 1000.0)[None])
    assert torch.equal(shifted.apex, p.apex)


def test_fraction_bounds_are_found_above_a_sloping_baseline() -> None:
    """Until 2026-11-11 the raw trace was compared with a level from the
    apex's baseline alone: under an exact sloping baseline a Gaussian's
    bounds went from [42, 58] to [43, 100]."""
    n = 101
    x = torch.arange(n, dtype=torch.float64)
    peak = 100 * torch.exp(-0.5 * ((x - 50) / 3) ** 2)
    cand = LocalMaxima()(_trace(peak[None].numpy(), np.array([n])))
    flat = Fraction(0.05)(_trace(peak[None].numpy(), np.array([n])), cand, torch.tensor([[50]]))
    for slope in (0.5, -0.5):
        base = 20 + slope * x
        tr = Trace(y=(peak + base)[None], x=x[None], valid=torch.ones(1, n, dtype=torch.bool), count=torch.tensor([n]),
                   baseline=base[None])  # fmt: skip
        got = Fraction(0.05)(tr, cand, torch.tensor([[50]]))
        assert [t.item() for t in got] == [t.item() for t in flat] == [42, 58]


def test_parabolic_refinement_is_exact_for_a_gaussian_on_an_irregular_axis() -> None:
    """Until 2026-11-11 the parabola assumed equal spacing: a Gaussian
    centred at 1.3 sampled at x = 0, 1, 4 refined to 0.68."""
    xi = torch.tensor([[0.0, 1.0, 4.0]], dtype=torch.float64)
    tr = Trace(y=torch.exp(-0.5 * ((xi - 1.3) / 1.0) ** 2), x=xi, valid=torch.ones(1, 3, dtype=torch.bool),
               count=torch.tensor([3]))  # fmt: skip
    assert Parabolic(log=True)(tr, torch.tensor([[1]]), None, None, None)[0].item() == pytest.approx(1.3, abs=1e-12)
    # On an equally spaced axis, the three-point formula as before.
    g = np.random.default_rng(4)
    y = g.uniform(0.1, 1.0, size=(50, 40))
    h, x0 = g.uniform(0.5, 2.0, size=(50, 1)), g.uniform(-5, 5, size=(50, 1))
    x = x0 + h * np.arange(40)
    apex = torch.tensor(g.integers(0, 40, size=(50, 3)))
    for log in (False, True):
        tr = Trace(y=torch.tensor(y), x=torch.tensor(x), valid=torch.ones(50, 40, dtype=torch.bool),
                   count=torch.full((50,), 40))  # fmt: skip
        got = Parabolic(log=log)(tr, apex, None, None, None)[0].numpy()
        v = np.log(y) if log else y
        i = apex.numpy()
        a = np.take_along_axis(v, np.clip(i - 1, 0, 39), 1)
        b = np.take_along_axis(v, i, 1)
        c = np.take_along_axis(v, np.clip(i + 1, 0, 39), 1)
        denom = a - 2 * b + c
        inner = (i > 0) & (i < 39) & (denom < 0)
        shift = np.where(inner, 0.5 * (a - c) / np.where(denom < 0, denom, -1.0), 0.0).clip(-0.5, 0.5)
        np.testing.assert_allclose(got, x0 + h * (i + shift), rtol=1e-12, atol=1e-12)


def _select_by_distance_all_pairs(position, priority, valid, distance):
    """The implementation before 2026-11-11, every pair compared: the
    reference the banded one must equal."""
    m = position.shape[-1]
    p = position.to(torch.float64)
    near = (p.unsqueeze(-1) - p.unsqueeze(-2)).abs() < distance
    near &= valid.unsqueeze(-1) & valid.unsqueeze(-2)
    near &= ~torch.eye(m, dtype=torch.bool).unsqueeze(0)
    j_idx, i_idx = torch.arange(m).view(1, 1, m), torch.arange(m).view(1, m, 1)
    pr = priority
    better = (pr.unsqueeze(-2) > pr.unsqueeze(-1)) | ((pr.unsqueeze(-2) == pr.unsqueeze(-1)) & (j_idx > i_idx))
    rivals = near & better
    undecided, kept = valid.clone(), torch.zeros_like(valid)
    while undecided.any():
        win = undecided & ~(rivals & undecided.unsqueeze(-2)).any(dim=-1)
        kept |= win
        undecided &= ~(win | (near & win.unsqueeze(-2)).any(dim=-1))
    return kept


@pytest.mark.parametrize("distance", [1.0, 1.5, 2.0, 3.0, 4.7, 9.0])
def test_distance_suppression_is_the_all_pairs_choice(distance) -> None:
    from constellation.core.signal.peaks.detect import select_by_distance

    g = torch.Generator().manual_seed(int(distance * 10))
    b, n, m = 20, 300, 60
    pos = torch.stack([torch.randperm(n, generator=g)[:m] for _ in range(b)])  # distinct, unsorted
    prio = torch.randint(0, 6, (b, m), generator=g).to(torch.float64)  # many ties
    valid = torch.rand(b, m, generator=g) > 0.2
    want = _select_by_distance_all_pairs(pos, prio, valid, distance)
    assert torch.equal(select_by_distance(pos, prio, valid, distance), want)


def test_distance_suppression_of_25000_candidates_in_a_row_stays_small() -> None:
    """Until 2026-11-11 one row's candidates were compared pairwise: 25,000
    of them made [M, M] intermediates of 5 GB each."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import resource, torch\n"
        "from constellation.core.signal.peaks.detect import select_by_distance\n"
        "m = 25000\n"
        "pos = (torch.arange(m) * 2 + 1).unsqueeze(0)\n"
        "prio = torch.rand(1, m, generator=torch.Generator().manual_seed(0), dtype=torch.float64)\n"
        "before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "keep = select_by_distance(pos, prio, torch.ones(1, m, dtype=torch.bool), 4)\n"
        "after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "print(int(keep.sum()), (after - before) // 1024)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True,
                         text=True, timeout=300, check=True)  # fmt: skip
    kept, grown_mb = (int(v) for v in out.stdout.split())
    assert 0 < kept < 25000 and grown_mb < 200


def test_robust_z_measures_its_peaks_above_its_own_baseline() -> None:
    """Until 2026-11-11 RobustZ detected above its rolling median but the
    picker measured above 0: an offset of 100 took a peak's width from 7 to
    147 and its area from 91 to 14,777."""
    g = np.random.default_rng(0)
    x = np.arange(200.0)
    y = 10 * np.exp(-0.5 * ((x - 100) / 3) ** 2) + g.normal(scale=0.3, size=200)
    picker = PeakPicker(detect=RobustZ(k=5, k_low=2, window=41), boundaries=DetectorBounds())
    low, high = picker(torch.tensor(y)[None]), picker(torch.tensor(y + 100.0)[None])
    assert int(low.n[0]) == 1 and torch.equal(low.apex, high.apex)
    assert low.width[0, 0].item() == pytest.approx(2.3548 * 3, rel=0.15)
    for field in ("width", "area", "snr", "asymmetry"):
        assert getattr(high, field)[0, 0].item() == pytest.approx(getattr(low, field)[0, 0].item(), rel=1e-9)


@pytest.mark.parametrize("rel_height", [0.0, 0.5, 1.0])
def test_plateau_widths_are_scipys_at_every_relative_height(rel_height) -> None:
    """scipy's walk starts at the peak, so at ``rel_height`` 0 a flat top
    has width 0; until 2026-11-11 it started beside the peak (width 2 for
    ``[0, 1, 1, 1, 0]``)."""
    import warnings

    g = np.random.default_rng(9)
    rows = [np.array([0, 1, 1, 1, 0, 0.0]), np.array([0, 2, 2, 1, 3, 3, 3, 0.0])]
    rows += [g.integers(0, 4, size=40).astype(np.float64) for _ in range(6)]
    for y in rows:
        n = y.size
        c = LocalMaxima(rel_height=rel_height)(_trace(y[None], np.array([n])))
        peaks = np.sort(c.apex[0][c.valid[0]].numpy())
        order = np.argsort(c.apex[0][c.valid[0]].numpy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w, _, left_ip, right_ip = peak_widths(y, peaks, rel_height=rel_height)
        np.testing.assert_allclose(c.extras["width_samples"][0][c.valid[0]].numpy()[order], w)
        np.testing.assert_allclose(c.extras["left_ip"][0][c.valid[0]].numpy()[order], left_ip)
        np.testing.assert_allclose(c.extras["right_ip"][0][c.valid[0]].numpy()[order], right_ip)


def test_a_batch_with_no_samples_or_no_rows_has_no_peaks() -> None:
    """A zero-length axis raised a reshape error (the scorer met it for a
    query with no scans; Codex review, 2026-11-11)."""
    picker = PeakPicker(detect=LocalMaxima(), boundaries=WalkOut())
    for shape in ((3, 0), (0, 12), (2, 4, 0)):
        p = picker(torch.zeros(shape, dtype=torch.float64))
        assert p.valid.shape == (*shape[:-1], 0) and p.apex_x.shape == (*shape[:-1], 0)
