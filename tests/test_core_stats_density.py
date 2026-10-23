"""Gridded 2-D kernel densities: deposition, FFT convolution, renormalization, sampling."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from constellation.core.stats.density import (
    GridAxis,
    bin_2d,
    fft_convolve2d,
    grid_kde_2d,
    kernel_stamp,
    support_mask,
    weighted_quantile,
)


def test_grid_axis_maps_values_and_cells() -> None:
    axis = GridAxis.spanning(-1.0, 4.0, 11)
    assert axis.step == pytest.approx(0.5) and axis.stop == pytest.approx(4.0)
    v = torch.tensor([-1.0, 1.25, 4.0], dtype=torch.float64)
    assert torch.allclose(axis.to_value(axis.to_index(v)), v)
    with pytest.raises(ValueError):
        GridAxis.spanning(1.0, 1.0, 5)


@pytest.mark.parametrize("deposit", ["nearest", "linear"])
def test_deposition_conserves_mass_inside_the_grid(deposit: str) -> None:
    torch.manual_seed(0)
    x, y = torch.rand(500, dtype=torch.float64) * 9, torch.rand(500, dtype=torch.float64) * 4
    w = torch.rand(500, dtype=torch.float64)
    hist = bin_2d(x, y, w, x_axis=GridAxis.spanning(0, 9, 19), y_axis=GridAxis.spanning(0, 4, 9), deposit=deposit)
    assert float(hist.sum()) == pytest.approx(float(w.sum()))


def test_linear_deposition_is_exact_at_cell_centres_and_splits_between_them() -> None:
    ax = GridAxis.spanning(0, 4, 5)
    at_node = bin_2d(torch.tensor([2.0]), torch.tensor([1.0]), x_axis=ax, y_axis=ax)
    assert at_node[2, 1] == 1.0 and float(at_node.sum()) == 1.0
    between = bin_2d(torch.tensor([2.25]), torch.tensor([1.5]), x_axis=ax, y_axis=ax)
    assert between[2, 1] == pytest.approx(0.375) and between[3, 2] == pytest.approx(0.125)


def test_off_grid_weight_is_dropped() -> None:
    ax = GridAxis.spanning(0, 4, 5)
    hist = bin_2d(torch.tensor([-3.0, 2.0]), torch.tensor([1.0, 1.0]), x_axis=ax, y_axis=ax, deposit="nearest")
    assert float(hist.sum()) == 1.0


def test_fft_convolution_equals_direct_convolution() -> None:
    torch.manual_seed(1)
    grid = torch.rand(23, 31, dtype=torch.float64)
    stamp = kernel_stamp((1.5, 2.5))
    direct = F.conv2d(grid[None, None], stamp.flip(0, 1)[None, None], padding=(stamp.shape[0] // 2, stamp.shape[1] // 2))[0, 0]
    assert torch.allclose(fft_convolve2d(grid, stamp), direct, atol=1e-12)


def test_kernel_stamps_are_normalized_and_the_cosine_stops_at_two_sigma() -> None:
    for kernel in ("gaussian", "cosine_approx"):
        assert float(kernel_stamp(3.0, kernel=kernel).sum()) == pytest.approx(1.0)
    cosine = kernel_stamp(3.0, kernel="cosine_approx")
    assert cosine.shape == (13, 13)
    assert cosine[6, 0] == 0.0 and cosine[0, 0] == 0.0 and cosine[6, 6] == cosine.max()


def test_nearest_kde_is_the_sum_of_a_stamp_at_every_point() -> None:
    torch.manual_seed(2)
    ax, ay = GridAxis.spanning(0, 39, 40), GridAxis.spanning(0, 29, 30)
    x = torch.randint(0, 40, (60,)).to(torch.float64)
    y = torch.randint(0, 30, (60,)).to(torch.float64)
    density = grid_kde_2d(x, y, x_axis=ax, y_axis=ay, sigma_cells=2.0, kernel="gaussian", deposit="nearest", zero_below=0.0)
    stamp = kernel_stamp(2.0)
    r = stamp.shape[0] // 2
    expected = torch.zeros(40 + 2 * r, 30 + 2 * r, dtype=torch.float64)
    for i, j in zip(x.long().tolist(), y.long().tolist()):
        expected[i : i + 2 * r + 1, j : j + 2 * r + 1] += stamp
    assert torch.allclose(density.values, expected[r:-r, r:-r], atol=1e-12)


def test_gaussian_kde_matches_a_direct_kernel_sum() -> None:
    torch.manual_seed(3)
    x = torch.randn(400, dtype=torch.float64) * 2 + 10
    y = 0.5 * x + torch.randn(400, dtype=torch.float64)
    ax, ay = GridAxis.spanning(0, 20, 201), GridAxis.spanning(0, 12, 121)
    sigma = 0.6
    density = grid_kde_2d(x, y, x_axis=ax, y_axis=ay, sigma_cells=(sigma / ax.step, sigma / ay.step))
    gx = ax.to_value(torch.arange(201, dtype=torch.float64))[:, None, None]
    gy = ay.to_value(torch.arange(121, dtype=torch.float64))[None, :, None]
    direct = torch.exp(-0.5 * ((gx - x) ** 2 + (gy - y) ** 2) / sigma**2).sum(-1) / (2 * math.pi * sigma**2)
    per_cell = density.values / (ax.step * ay.step)
    assert torch.allclose(per_cell, direct, atol=2e-3 * float(direct.max()))


def test_renormalizing_by_the_support_removes_the_deficit_at_its_edge() -> None:
    torch.manual_seed(4)
    x = torch.rand(40000, dtype=torch.float64) * 10
    y = torch.rand(40000, dtype=torch.float64) * 10
    ax = ay = GridAxis.spanning(-2, 12, 141)
    mask = support_mask(ax, ay, lambda u, v: (u >= 0) & (u <= 10) & (v >= 0) & (v <= 10))
    assert mask[20, 70] == pytest.approx(0.5) and mask[70, 70] == 1.0 and mask[10, 70] == 0.0
    plain = grid_kde_2d(x, y, x_axis=ax, y_axis=ay, sigma_cells=6.0).values
    fixed = grid_kde_2d(x, y, x_axis=ax, y_axis=ay, sigma_cells=6.0, renormalize=mask).values
    centre, edge = (70, 70), (20, 70)  # x = 5 and x = 0
    assert float(plain[edge] / plain[centre]) < 0.6
    assert float(fixed[edge] / fixed[centre]) == pytest.approx(1.0, abs=0.05)


def test_sampling_reads_the_grid_and_is_differentiable() -> None:
    ax, ay = GridAxis.spanning(0, 10, 11), GridAxis.spanning(0, 5, 6)
    values = torch.arange(66, dtype=torch.float64).view(11, 6)
    density = grid_kde_2d(torch.tensor([5.0]), torch.tensor([2.0]), x_axis=ax, y_axis=ay, sigma_cells=1.0)
    density = type(density)(values, ax, ay, (1.0, 1.0), "gaussian", 1.0)
    assert float(density.sample(torch.tensor(3.0), torch.tensor(2.0))) == pytest.approx(float(values[3, 2]))
    assert float(density.sample(torch.tensor(3.5), torch.tensor(2.0))) == pytest.approx(float(values[3:5, 2].mean()))
    assert float(density.sample(torch.tensor(30.0), torch.tensor(2.0))) == 0.0
    q = torch.tensor([3.3], dtype=torch.float64, requires_grad=True)
    density.sample(q, torch.tensor([2.0]), mode="bicubic").sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_weighted_quantile() -> None:
    v = torch.arange(10.0)
    assert float(weighted_quantile(v, 0.5)) == pytest.approx(4.5)
    assert float(weighted_quantile(v, 0.0)) == 0.0 and float(weighted_quantile(v, 1.0)) == 9.0
    w = torch.ones(10)
    w[9] = 1000.0  # its weight's midpoint sits just above the median
    assert 8.9 < float(weighted_quantile(v, 0.5, w)) < 9.0
    assert float(weighted_quantile(v, 0.6, w)) == 9.0
    many = weighted_quantile(v, torch.tensor([0.25, 0.75]))
    assert many.shape == (2,)


def test_zero_weight_values_are_not_observations() -> None:
    v, w = torch.tensor([0.0, 10.0, 20.0]), torch.tensor([0.0, 1.0, 0.0])
    assert weighted_quantile(v, torch.tensor([0.25, 0.75]), w).tolist() == [10.0, 10.0]
    with pytest.raises(ValueError):
        weighted_quantile(v, 0.5, torch.zeros(3))
