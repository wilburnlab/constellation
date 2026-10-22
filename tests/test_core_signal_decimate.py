"""M4 trace decimation: each pixel column keeps its first, last, lowest and highest point."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from constellation.core.signal import decimate_trace


def _columns(x: np.ndarray, lo: float, hi: float, n_bins: int) -> np.ndarray:
    return np.minimum(((x - lo) / (hi - lo) * n_bins).astype(np.int64), n_bins - 1)


def test_each_column_keeps_its_ends_and_its_extremes() -> None:
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 6000.0, 180_000))
    y = rng.gamma(2.0, 1e5, x.size)
    n_bins = 800
    keep = decimate_trace(x, y, n_bins=n_bins).numpy()

    assert keep.size <= 4 * n_bins and np.all(np.diff(keep) > 0)
    full_cols = _columns(x, x[0], x[-1], n_bins)
    kept_cols = full_cols[keep]
    for col in np.unique(full_cols):
        members = np.flatnonzero(full_cols == col)
        kept = keep[kept_cols == col]
        assert members[0] in kept and members[-1] in kept
        assert y[kept].max() == y[members].max()
        assert y[kept].min() == y[members].min()


def test_a_short_trace_is_kept_whole() -> None:
    x = np.arange(100.0)
    assert decimate_trace(x, np.ones(100), n_bins=25).tolist() == list(range(100))


def test_a_range_keeps_one_neighbour_beyond_each_edge() -> None:
    x = np.arange(1000.0)
    y = np.sin(x / 7.0)
    keep = decimate_trace(x, y, n_bins=10, x_range=(200.0, 300.0)).numpy()
    assert keep.min() == 199 and keep.max() == 301
    inside = keep[(x[keep] >= 200.0) & (x[keep] <= 300.0)]
    assert inside.size <= 40


def test_tensors_and_arrays_give_the_same_indices() -> None:
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.0, 1.0, 5000))
    y = rng.normal(size=5000)
    from_numpy = decimate_trace(x, y, n_bins=50)
    from_torch = decimate_trace(torch.from_numpy(x), torch.from_numpy(y), n_bins=50)
    assert torch.equal(from_numpy, from_torch)


def test_empty_and_invalid_input() -> None:
    assert decimate_trace(np.array([]), np.array([]), n_bins=10).numel() == 0
    with pytest.raises(ValueError, match="sorted"):
        decimate_trace(np.array([2.0, 1.0]), np.array([0.0, 0.0]), n_bins=10)
    with pytest.raises(ValueError, match="same length"):
        decimate_trace(np.arange(3.0), np.arange(2.0), n_bins=10)
    with pytest.raises(ValueError, match="n_bins"):
        decimate_trace(np.arange(3.0), np.arange(3.0), n_bins=0)
