"""Batched interpolation along an irregular, masked axis (core.signal.interpolate)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from constellation.core.signal.interpolate import bracket, interp_rows


def _rows(seed: int = 0, b: int = 6, n: int = 17):
    g = np.random.default_rng(seed)
    x = np.sort(g.uniform(0.0, 100.0, (b, n)), axis=1)
    y = g.normal(size=(b, n))
    valid = g.random((b, n)) > 0.25
    valid[0] = False  # a row with nothing
    valid[1, :] = True
    return x, y, valid


def test_interp_rows_is_np_interp_row_by_row_inside_the_samples() -> None:
    x, y, valid = _rows()
    g = np.random.default_rng(1)
    x_new = g.uniform(-5.0, 105.0, (x.shape[0], 9))
    got = interp_rows(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_new),
                      valid=torch.from_numpy(valid)).numpy()  # fmt: skip
    for r in range(x.shape[0]):
        xs, ys = x[r][valid[r]], y[r][valid[r]]
        for j, q in enumerate(x_new[r]):
            if xs.size == 0 or q < xs[0] or q > xs[-1]:
                assert np.isnan(got[r, j])
            else:
                assert got[r, j] == pytest.approx(np.interp(q, xs, ys), abs=1e-12)


def test_padding_may_sit_anywhere_and_is_never_read() -> None:
    x = torch.tensor([[0.0, 99.0, 10.0, 20.0]], dtype=torch.float64)
    y = torch.tensor([[0.0, 1e9, 10.0, float("nan")]], dtype=torch.float64)
    valid = torch.tensor([[True, False, True, True]])
    out = interp_rows(x, y, torch.tensor([[5.0, 15.0]], dtype=torch.float64), valid=valid)
    # y NaN at 20 is no sample, so 15 is past the last one.
    assert out[0, 0].item() == pytest.approx(5.0) and torch.isnan(out[0, 1])
    clamped = interp_rows(x, y, torch.tensor([[15.0, -1.0]], dtype=torch.float64), valid=valid, extrapolate="clamp")
    assert clamped.tolist() == [[10.0, 0.0]]


def test_bracket_names_the_samples_on_either_side_and_their_weight() -> None:
    x = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float64)
    valid = torch.tensor([[True, False, True, True]])
    b = bracket(x, torch.tensor([[3.0, 4.0, 0.5, 9.0, float("nan")]], dtype=torch.float64), valid=valid)
    assert b.left.tolist() == [[0, 2, -1, 3, -1]]
    assert b.right.tolist() == [[2, 2, 0, -1, -1]]
    assert b.inside.tolist() == [[True, True, False, False, False]]
    assert b.weight[0, 0].item() == pytest.approx(2.0 / 3.0) and b.weight[0, 1].item() == 0.0


def test_max_gap_refuses_to_bridge_a_wide_hole() -> None:
    x = torch.tensor([[0.0, 1.0, 10.0]], dtype=torch.float64)
    y = torch.tensor([[0.0, 1.0, 10.0]], dtype=torch.float64)
    out = interp_rows(x, y, torch.tensor([[0.5, 5.0]], dtype=torch.float64), max_gap=2.0)
    assert out[0, 0].item() == pytest.approx(0.5) and torch.isnan(out[0, 1])
