"""Tolerance-gated matching on a sorted axis.

The shared primitive under extracted-ion-chromatogram extraction, peak
annotation, and spectrum-centric search: pair each *query* value (a
theoretical m/z, a calibration target, ...) to the *observed* value
closest to it within a ppm/Da tolerance window, on an axis that is
already sorted ascending.

Two entry points, both tensor-native, batched, and device-agnostic
(they never materialise Python lists in the hot path — that is the
distinction from :func:`massspec.peptide.match.match_mz`, which returns
``list[MzMatch]`` and is fine only at dlib scale):

  ``nearest_within_tolerance`` — the rectangular fast path. One
  ``torch.searchsorted`` plus a left/right-neighbour check. Returns
  exactly one match per query, so the output is rectangular and lifts
  to a dense tensor with no ragged bookkeeping. The default for XIC.

  ``bounds_within_tolerance`` — the window-bound search. Two
  ``torch.searchsorted`` calls returning the ``[lo, hi)`` candidate
  slice per query. Backs the all-peaks-in-window mode (interference /
  chimeric / DIA work) and the degenerate "assign to both" case.

Both index into the *sorted axis the caller passes*; a caller that
sorts its own input keeps the permutation to map indices back to
original order. ppm tolerance is computed at the query value via
:func:`core.stats.units.ppm_to_da` (``Δm = ppm · m/z · 1e-6``), matching
the convention everywhere else in the package.

1D inputs ``(N,)`` axis + ``(M,)`` queries return ``(M,)`` results.
Batched inputs share a leading dim: ``(B, N)`` axis + ``(B, M)`` queries
return ``(B, M)`` (row-wise searchsorted) — the layout the scan-major
XIC sweep uses, with ``(n_scans, max_peaks)`` observed peaks padded by
``+inf`` against ``(n_scans, n_queries)``.
"""

from __future__ import annotations

from typing import Literal

import torch

from constellation.core.stats.units import ppm_to_da

ToleranceUnit = Literal["ppm", "Da"]


def _window(
    queries: torch.Tensor, tolerance: float, unit: ToleranceUnit
) -> torch.Tensor:
    """Half-width of the tolerance window at each query value (float64)."""
    if unit == "ppm":
        return ppm_to_da(tolerance, queries)
    if unit == "Da":
        return torch.full_like(queries, float(tolerance))
    raise ValueError(f"unit must be 'ppm' or 'Da', got {unit!r}")


def nearest_within_tolerance(
    sorted_axis: torch.Tensor,
    queries: torch.Tensor,
    tolerance: float,
    unit: ToleranceUnit = "ppm",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest observed value to each query, gated by tolerance.

    Parameters
    ----------
    sorted_axis : torch.Tensor
        Ascending values, shape ``(N,)`` or ``(B, N)`` (ascending along
        the last dim). May be ``+inf``-padded (the padding sorts last
        and never wins a nearest-neighbour comparison).
    queries : torch.Tensor
        Query values, shape ``(M,)`` or ``(B, M)`` matching the axis's
        leading dim.
    tolerance : float
        Tolerance half-width. ppm or Da per ``unit``.
    unit : {"ppm", "Da"}
        Interpretation of ``tolerance``.

    Returns
    -------
    best_idx : torch.Tensor
        Long indices into ``sorted_axis`` of the nearest value, shape
        like ``queries``. Meaningful only where ``within_mask`` is True
        (0 elsewhere / for an empty axis).
    within_mask : torch.Tensor
        Bool, True where the nearest value is within tolerance.
    signed_delta : torch.Tensor
        ``queries - sorted_axis[best_idx]`` (float64) — signed, so a
        downstream consumer derives signed Da/ppm error with no
        recompute. ``nan`` where the axis is empty.
    """
    queries = queries.to(torch.float64)
    sorted_axis = sorted_axis.to(torch.float64)
    n = sorted_axis.shape[-1]

    if n == 0:
        best_idx = torch.zeros(queries.shape, dtype=torch.long, device=queries.device)
        within = torch.zeros(queries.shape, dtype=torch.bool, device=queries.device)
        signed = torch.full(
            queries.shape, float("nan"), dtype=torch.float64, device=queries.device
        )
        return best_idx, within, signed

    # Insertion point in [0, n]; nearest is one of {p-1, p}.
    p = torch.searchsorted(sorted_axis, queries.contiguous())
    right = p.clamp(max=n - 1)
    left = (p - 1).clamp(min=0)

    axis_right = torch.gather(sorted_axis, -1, right)
    axis_left = torch.gather(sorted_axis, -1, left)
    d_right = (queries - axis_right).abs()
    d_left = (queries - axis_left).abs()

    use_left = d_left <= d_right
    best_idx = torch.where(use_left, left, right)
    best_axis = torch.where(use_left, axis_left, axis_right)

    signed_delta = queries - best_axis
    within_mask = signed_delta.abs() <= _window(queries, tolerance, unit)
    return best_idx, within_mask, signed_delta


def bounds_within_tolerance(
    sorted_axis: torch.Tensor,
    queries: torch.Tensor,
    tolerance: float,
    unit: ToleranceUnit = "ppm",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Half-open ``[lo, hi)`` slice of axis values within tolerance.

    Returns every candidate in the window rather than just the nearest —
    the substrate for all-peaks-in-window extraction and for the
    degenerate-theoretical "assign to both" case. ``hi == lo`` marks a
    query with no candidate.

    Parameters mirror :func:`nearest_within_tolerance`. Returns ``(lo,
    hi)`` long tensors shaped like ``queries``; both are 0 for every
    query when the axis is empty.
    """
    queries = queries.to(torch.float64)
    sorted_axis = sorted_axis.to(torch.float64)
    half = _window(queries, tolerance, unit)
    lo = torch.searchsorted(sorted_axis, (queries - half).contiguous(), side="left")
    hi = torch.searchsorted(sorted_axis, (queries + half).contiguous(), side="right")
    return lo, hi
