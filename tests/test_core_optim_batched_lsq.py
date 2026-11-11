"""Many independent least-squares fits at once (core.optim.batched_lsq)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import least_squares

from constellation.core.optim.batched_lsq import batched_least_squares


def _gaussians(b: int = 512, n: int = 25, seed: int = 0):
    g = np.random.default_rng(seed)
    x = np.linspace(-10, 10, n)[None].repeat(b, 0)
    true = np.stack([g.uniform(5, 50, b), g.uniform(-3, 3, b), g.uniform(1, 3, b)], 1)
    y = true[:, :1] * np.exp(-0.5 * ((x - true[:, 1:2]) / true[:, 2:3]) ** 2) + g.normal(scale=0.5, size=(b, n))
    p0 = np.stack([y.max(1), x[np.arange(b), y.argmax(1)], np.full(b, 2.0)], 1)
    return x, y, p0


def test_each_row_is_scipys_least_squares_fit_with_its_covariance() -> None:
    x, y, p0 = _gaussians()
    xt, yt = torch.tensor(x), torch.tensor(y)

    def residual(q, rows):
        return q[:, :1] * torch.exp(-0.5 * ((xt[rows] - q[:, 1:2]) / q[:, 2:3]) ** 2) - yt[rows]

    lower, upper = np.array([0.0, -10.0, 0.1]), np.array([1e3, 10.0, 10.0])
    out = batched_least_squares(residual, torch.tensor(p0), lower=torch.tensor(lower), upper=torch.tensor(upper))
    assert out.converged.all()
    for i in range(0, 512, 16):
        def f(q, i=i):
            return q[0] * np.exp(-0.5 * ((x[i] - q[1]) / q[2]) ** 2) - y[i]

        s = least_squares(f, p0[i], bounds=(lower, upper), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        np.testing.assert_allclose(out.params[i].numpy(), s.x, atol=1e-5)
        assert out.cost[i].item() == pytest.approx(s.cost, rel=1e-8)
        cov = np.linalg.inv(s.jac.T @ s.jac) * (2 * s.cost / (x.shape[1] - 3))
        np.testing.assert_allclose(out.covariance[i].numpy(), cov, rtol=1e-4, atol=1e-8)


def test_bounds_hold_and_fixed_parameters_stay() -> None:
    x, y, p0 = _gaussians(64)
    xt, yt = torch.tensor(x), torch.tensor(y)

    def residual(q, rows):
        return q[:, :1] * torch.exp(-0.5 * ((xt[rows] - q[:, 1:2]) / q[:, 2:3]) ** 2) - yt[rows]

    upper = torch.tensor([1e3, 10.0, 1.2], dtype=torch.float64)  # a sigma bound the data would pass
    out = batched_least_squares(residual, torch.tensor(p0).clamp(max=upper), upper=upper)
    assert (out.params[:, 2] <= 1.2 + 1e-12).all()
    free = torch.ones(64, 3, dtype=torch.bool)
    free[:, 2] = False
    fixed = batched_least_squares(residual, torch.tensor(p0), free=free)
    assert torch.equal(fixed.params[:, 2], torch.tensor(p0)[:, 2])
    assert (fixed.covariance[:, 2] == 0).all()


def test_masked_residuals_do_not_count_and_rows_do_not_couple() -> None:
    x, y, p0 = _gaussians(32)
    y = y.copy()
    y[:, :5] = 1e6  # outliers, masked away
    mask = torch.ones(32, 25, dtype=torch.bool)
    mask[:, :5] = False
    xt, yt = torch.tensor(x), torch.tensor(y)

    def residual(q, rows):
        return q[:, :1] * torch.exp(-0.5 * ((xt[rows] - q[:, 1:2]) / q[:, 2:3]) ** 2) - yt[rows]

    masked = batched_least_squares(residual, torch.tensor(p0), mask=mask)
    clean = batched_least_squares(lambda q, rows: residual(q, rows)[:, 5:], torch.tensor(p0))
    assert torch.allclose(masked.params, clean.params, atol=1e-8)
    alone = batched_least_squares(lambda q, rows: residual(q, rows + 7)[:, 5:], torch.tensor(p0)[7:8])
    assert torch.allclose(alone.params[0], clean.params[7], atol=1e-8)
