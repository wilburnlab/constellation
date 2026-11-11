"""Detectors: where a batch of traces has its peaks.

A detector takes a compacted :class:`~._types.Trace` and gives
:class:`~._types.Candidates`. Three are implemented; the survey
(``docs/plans/peak-detection.md``) describes the others and how they fit:

- :class:`LocalMaxima`: the samples higher than their neighbours (a flat
  top's middle), kept by height, threshold, distance, prominence and width:
  ``scipy.signal.find_peaks`` in every criterion and order, batched. Its
  score is the prominence (a peak's 0-dimensional persistence, but for a
  peak that reaches the row's edge or a ``wlen`` window).
- :class:`MatchedFilter`: correlation with a peak template (a Gaussian or
  its negative second derivative, the Mexican hat, at one or more widths),
  scaled to the response's noise; peaks are its local maxima above a
  threshold. The detector of CRAWDAD / Skyline and XCMS matchedFilter.
- :class:`RobustZ`: localized outlier runs. Each sample's height above the
  baseline over the noise is a z-score; a peak is a run of samples above
  ``k_low`` that reaches ``k`` (hysteresis), its apex the run's highest.
  It needs a baseline and a noise level, its own or the picker's.

Not implemented, and how they would plug in: CWT ridge lines (Du, Kibbe and
Lin 2006: a ``conv1d`` bank of Mexican hats over scales, local maxima
linked across scales by a loop over the scales; the scale and the ridge
length go in ``extras``, and the best scale gives ``left`` / ``right``);
change-point segmenters, which give their own ``left`` / ``right``;
model-based deconvolution, as a refiner (``refine.ShapeFit``).

The kernels here (:func:`local_maxima`, :func:`prominences`,
:func:`widths`, :func:`select_by_distance`) are public for any detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from constellation.core.signal._rows import RangeTable
from constellation.core.signal.baseline import rolling_quantile
from constellation.core.signal.noise import MAD_TO_SIGMA
from constellation.core.signal.peaks._types import Candidates, Trace
from constellation.core.signal.peaks.stages import Stage, register

__all__ = [
    "LocalMaxima",
    "MatchedFilter",
    "RobustZ",
    "local_maxima",
    "prominences",
    "widths",
    "select_by_distance",
    "padded",
]


def padded(mask: torch.Tensor, *values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The positions where ``mask [B, N]`` is true as ``[B, M]`` (M the most
    a row has; ``-1`` padding), with ``values [B, N]`` gathered there, and
    the validity ``[B, M]`` last."""
    count = mask.sum(dim=-1)
    m = int(count.max()) if count.numel() else 0
    order = torch.argsort((~mask).to(torch.int8), dim=-1, stable=True)[..., :m]
    valid = torch.arange(m, device=mask.device) < count.unsqueeze(-1)
    pos = torch.where(valid, order, torch.full_like(order, -1))
    out = [torch.gather(v, -1, order) for v in values]
    return (pos, *out, valid)


def local_maxima(y: torch.Tensor, count: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``scipy.signal._peak_finding_utils._local_maxima_1d``, batched: a
    sample (not a row's first or last) whose left neighbour is lower and
    whose right neighbour is lower, a flat top counting as one peak at its
    middle (rounded down). Returns ``[B, N]`` masks of the peaks, and each
    peak's plateau's first and last index ``[B, N]`` (valid at peaks)."""
    b, n = y.shape
    idx = torch.arange(n, device=y.device).expand(b, n)
    inside = idx < count.unsqueeze(-1)
    if n < 3:
        z = torch.zeros((b, n), dtype=torch.bool, device=y.device)
        return z, idx.clone(), idx.clone()
    d = y[:, 1:] - y[:, :-1]  # d[k]: y[k+1] - y[k]
    step_ok = inside[:, 1:]
    sign = torch.where(step_ok, torch.sign(d), torch.zeros_like(d))
    sign = torch.nan_to_num(sign, nan=0.0)
    # The last non-flat step at or before each sample's left, and the first
    # at or after its right.
    k = torch.arange(n - 1, device=y.device).expand(b, n - 1)
    moving = sign != 0
    last_k = torch.cummax(torch.where(moving, k, torch.full_like(k, -1)), dim=-1).values  # [B, N-1]
    big = n
    first_k = torch.flip(torch.cummin(torch.flip(torch.where(moving, k, torch.full_like(k, big)), [-1]), dim=-1).values, [-1])
    # For sample i: the step into it from the left is k = i-1 (look at the
    # last moving step <= i-1); the step out of it to the right is k = i.
    left_k = torch.cat([torch.full((b, 1), -1, device=y.device, dtype=torch.long), last_k], dim=-1)  # [B, N]
    right_k = torch.cat([first_k, torch.full((b, 1), big, device=y.device, dtype=torch.long)], dim=-1)
    rises = torch.gather(torch.cat([sign, torch.zeros((b, 1), dtype=sign.dtype)], -1), -1, left_k.clamp(min=0)) > 0
    falls = torch.gather(torch.cat([sign, torch.zeros((b, 1), dtype=sign.dtype)], -1), -1, right_k.clamp(max=n - 2)) < 0
    ok = (left_k >= 0) & (right_k < n - 1) & rises & falls & inside
    ok &= right_k < count.unsqueeze(-1) - 1
    start = left_k + 1
    end = right_k
    middle = (start + end) // 2
    peak = ok & (idx == middle)
    return peak, start, end


def prominences(table: RangeTable, y: torch.Tensor, peak: torch.Tensor, wlen: int | None = None):
    """``scipy.signal.peak_prominences`` for peaks ``[B, M]`` (``-1``
    padding): each peak's prominence and its left and right bases."""
    h = torch.gather(y, -1, peak.clamp(min=0))
    last = (table.count - 1).unsqueeze(-1).expand_as(peak)
    if wlen is not None and wlen > 1:
        lo = (peak - wlen // 2).clamp(min=0)
        hi = torch.minimum(peak + wlen // 2, last)
    else:
        lo, hi = torch.zeros_like(peak), last
    p = peak.clamp(min=0)
    higher_left = table.nearest_left(p, h, above=True, limit=lo)
    a = torch.where(higher_left >= 0, higher_left + 1, lo)
    left_min, left_base = table.argmin(a, p, prefer="right")
    higher_right = table.nearest_right(p, h, above=True, limit=hi)
    b = torch.where(higher_right >= 0, higher_right - 1, hi)
    right_min, right_base = table.argmin(p, b, prefer="left")
    prominence = h - torch.maximum(left_min, right_min)
    return prominence, left_base, right_base


def widths(table: RangeTable, y: torch.Tensor, peak: torch.Tensor, prominence: torch.Tensor,
           left_base: torch.Tensor, right_base: torch.Tensor, rel_height: float = 0.5):  # fmt: skip
    """``scipy.signal.peak_widths``: each peak's width in samples at
    ``rel_height`` of its prominence below it, and the interpolated left and
    right crossing positions (fractional indices)."""
    p = peak.clamp(min=0)
    h = torch.gather(y, -1, p)
    level = h - prominence * rel_height
    # The first sample at or below the level, walking out, not past a base.
    at_most = torch.nextafter(level, torch.full_like(level, float("inf")))
    li = table.nearest_left(p, at_most, above=False, limit=left_base)
    li = torch.where(li >= 0, li, left_base)
    ri = table.nearest_right(p, at_most, above=False, limit=right_base)
    ri = torch.where(ri >= 0, ri, right_base)
    yl, yl1 = torch.gather(y, -1, li), torch.gather(y, -1, (li + 1).clamp(max=y.shape[-1] - 1))
    yr, yr1 = torch.gather(y, -1, ri), torch.gather(y, -1, (ri - 1).clamp(min=0))
    left_ip = li.to(torch.float64) + torch.where(yl < level, (level - yl) / (yl1 - yl), torch.zeros_like(yl))
    right_ip = ri.to(torch.float64) - torch.where(yr < level, (level - yr) / (yr1 - yr), torch.zeros_like(yr))
    return right_ip - left_ip, left_ip, right_ip


def select_by_distance(position: torch.Tensor, priority: torch.Tensor, valid: torch.Tensor, distance: float,
                       *, max_elements: int = 1 << 26) -> torch.Tensor:  # fmt: skip
    """Which candidates ``[B, M]`` survive a minimum ``distance`` (in the
    units of ``position``), the higher ``priority`` first, as scipy's
    greedy rule: a candidate within ``distance`` of a kept, higher one is
    dropped. Done in rounds that keep every candidate no better undecided
    one is near and drop their neighbours: the same choice as the greedy
    walk (ties go to the later candidate, as in scipy)."""
    b, m = position.shape
    keep = torch.zeros_like(valid)
    if m == 0:
        return keep
    rows = max(1, max_elements // max(m * m, 1))
    pos = position.to(torch.float64)
    for a in range(0, b, rows):
        p, pr, v = pos[a : a + rows], priority[a : a + rows], valid[a : a + rows]
        near = (p.unsqueeze(-1) - p.unsqueeze(-2)).abs() < distance
        near &= v.unsqueeze(-1) & v.unsqueeze(-2)
        near &= ~torch.eye(m, dtype=torch.bool).unsqueeze(0)
        j_idx = torch.arange(m).view(1, 1, m)
        i_idx = torch.arange(m).view(1, m, 1)
        # A tie goes to the later candidate, as scipy's walk over a stable
        # ascending sort of the priorities, from its end, takes it first.
        better = (pr.unsqueeze(-2) > pr.unsqueeze(-1)) | ((pr.unsqueeze(-2) == pr.unsqueeze(-1)) & (j_idx > i_idx))
        rivals = near & better  # [r, i, j]: j is near i and better
        undecided = v.clone()
        kept = torch.zeros_like(v)
        while undecided.any():
            blocked = (rivals & undecided.unsqueeze(-2)).any(dim=-1)
            win = undecided & ~blocked
            kept |= win
            removed = win | (near & win.unsqueeze(-2)).any(dim=-1)
            undecided &= ~removed
        keep[a : a + rows] = kept
    return keep


def _pass(value: torch.Tensor, bound) -> torch.Tensor:
    """scipy's ``_select_by_property``: ``bound`` a minimum or a (min, max)."""
    if bound is None:
        return torch.ones_like(value, dtype=torch.bool)
    if isinstance(bound, (tuple, list)):
        lo, hi = bound
        ok = torch.ones_like(value, dtype=torch.bool)
        if lo is not None:
            ok &= value >= lo
        if hi is not None:
            ok &= value <= hi
        return ok
    return value >= bound


@register
@dataclass(frozen=True)
class LocalMaxima(Stage):
    """``scipy.signal.find_peaks``, batched (module docstring). Each
    criterion is a minimum or a ``(min, max)`` pair, applied in scipy's
    order: plateau size, height, threshold, distance (samples; the taller
    first), prominence (within ``wlen`` samples), width (samples, at
    ``rel_height``). ``min_snr`` also keeps a peak only if its height above
    the trace's baseline is that many noise sigmas. The score is the
    prominence."""

    height: float | tuple | None = None
    threshold: float | tuple | None = None
    distance: float | None = None
    prominence: float | tuple | None = None
    width: float | tuple | None = None
    wlen: int | None = None
    rel_height: float = 0.5
    plateau_size: int | tuple | None = None
    min_snr: float | None = None
    kind: ClassVar[str] = "detect"
    name: ClassVar[str] = "local_maxima"

    def __call__(self, trace: Trace) -> Candidates:
        y, count = trace.y, trace.count
        peak_mask, start, end = local_maxima(y, count)
        pos, start, end, valid = padded(peak_mask, start, end)
        p = pos.clamp(min=0)
        h = torch.gather(y, -1, p)
        if self.plateau_size is not None:
            valid &= _pass((end - start + 1).to(torch.float64), self.plateau_size)
        if self.height is not None:
            valid &= _pass(h, self.height)
        if self.threshold is not None:
            # scipy compares the apex sample with its own neighbours (a flat
            # top's middle with its equal neighbours: a threshold of 0).
            left_n = torch.gather(y, -1, (p - 1).clamp(min=0))
            right_n = torch.gather(y, -1, (p + 1).clamp(max=y.shape[-1] - 1))
            valid &= _pass(h - left_n, self.threshold) & _pass(h - right_n, self.threshold)
        if self.min_snr is not None:
            base = torch.gather(trace.baseline_or_zero(), -1, p)
            valid &= (h - base) >= self.min_snr * trace.noise_at(p)
        if self.distance is not None:
            if self.distance < 1:
                raise ValueError("distance must be at least 1 sample")
            valid &= select_by_distance(pos, h, valid, math.ceil(self.distance))
        table = RangeTable.build(y, count)
        prom, lb, rb = prominences(table, y, pos, self.wlen)
        if self.prominence is not None:
            valid &= _pass(prom, self.prominence)
        w, left_ip, right_ip = widths(table, y, pos, prom, lb, rb, self.rel_height)
        if self.width is not None:
            valid &= _pass(w, self.width)
        pos = torch.where(valid, pos, torch.full_like(pos, -1))
        return Candidates(apex=pos, score=torch.where(valid, prom, torch.full_like(prom, float("-inf"))),
                          valid=valid, left=torch.where(valid, lb, -1), right=torch.where(valid, rb, -1),
                          extras={"prominence": prom, "width_samples": w, "left_ip": left_ip,
                                  "right_ip": right_ip})  # fmt: skip


def _template(kind: str, sigma: float, truncate: float = 4.0) -> torch.Tensor:
    h = max(int(math.ceil(truncate * sigma)), 1)
    u = torch.arange(-h, h + 1, dtype=torch.float64) / sigma
    if kind == "gaussian":
        k = torch.exp(-0.5 * u * u)
        k = k - k.mean()  # zero mean: a flat baseline gives no response
    elif kind == "ricker":
        k = (1.0 - u * u) * torch.exp(-0.5 * u * u)
        k = k - k.mean()
    else:
        raise ValueError(f"template must be 'gaussian' or 'ricker', not {kind!r}")
    return k / k.norm()


@register
@dataclass(frozen=True)
class MatchedFilter(Stage):
    """Correlation with a zero-mean template at each width in ``widths``
    (template sigmas, in samples), each response over its row's noise (the
    MAD of the response); a peak is a local maximum of the best response at
    or above ``threshold``, kept apart by ``distance`` samples (default the
    best template's sigma, the taller first). The score is that response;
    ``extras['template_sigma']`` the template that gave it, and the
    detector's own extent is the apex ± 2 of its sigma."""

    widths: tuple[float, ...] = (2.0,)
    template: str = "gaussian"
    threshold: float = 3.0
    distance: float | None = None
    kind: ClassVar[str] = "detect"
    name: ClassVar[str] = "matched_filter"

    def __call__(self, trace: Trace) -> Candidates:
        y, valid = trace.y, trace.valid
        b, n = y.shape
        w = valid.to(torch.float64).reshape(b, 1, n)
        v = torch.where(valid, y - trace.baseline_or_zero(), torch.zeros_like(y)).to(torch.float64).reshape(b, 1, n)
        best = torch.full((b, n), float("-inf"), dtype=torch.float64)
        sigma_at = torch.zeros((b, n), dtype=torch.float64)
        for sigma in self.widths:
            k = _template(self.template, float(sigma))
            h = k.numel() // 2
            r = F.conv1d(v, k.view(1, 1, -1), padding=h)
            energy = F.conv1d(w, (k * k).view(1, 1, -1), padding=h)
            r = torch.where(energy > 0, r / energy.clamp(min=1e-300).sqrt(), torch.zeros_like(r)).reshape(b, n)
            r = torch.where(valid, r, torch.full_like(r, float("nan")))
            med = torch.nanmedian(r, dim=-1, keepdim=True).values
            mad = torch.nanmedian((r - med).abs(), dim=-1, keepdim=True).values * MAD_TO_SIGMA
            z = (r - med) / mad.clamp(min=1e-300)
            z = torch.where(valid, z, torch.full_like(z, float("-inf")))
            better = z > best
            best = torch.where(better, z, best)
            sigma_at = torch.where(better, torch.full_like(sigma_at, float(sigma)), sigma_at)
        filled = torch.where(valid, best, torch.full_like(best, float("-inf")))
        left = F.pad(filled, (1, 0), value=float("-inf"))[:, :-1]
        right = F.pad(filled, (0, 1), value=float("-inf"))[:, 1:]
        is_max = valid & (filled > left) & (filled >= right) & (filled >= self.threshold)
        pos, score, sig, ok = padded(is_max, best, sigma_at)
        d = self.distance
        if ok.any():
            spacing = d if d is not None else float(max(self.widths))
            ok &= select_by_distance(pos, score, ok, max(spacing, 1.0))
        reach = torch.round(2.0 * sig).to(torch.int64)
        last = (trace.count - 1).unsqueeze(-1)
        left_b = torch.where(ok, (pos - reach).clamp(min=0), -1)
        right_b = torch.where(ok, torch.minimum(pos + reach, last), -1)
        pos = torch.where(ok, pos, torch.full_like(pos, -1))
        return Candidates(apex=pos, score=torch.where(ok, score, torch.full_like(score, float("-inf"))),
                          valid=ok, left=left_b, right=right_b, extras={"template_sigma": sig})  # fmt: skip


@register
@dataclass(frozen=True)
class RobustZ(Stage):
    """Localized outlier runs (module docstring): ``z = (y - baseline) /
    noise``, with the picker's baseline and noise, or a rolling median over
    ``window`` samples and the residual's MAD; a peak is a run with z above
    ``k_low`` that reaches ``k`` and lasts ``min_length`` samples. Its apex is
    the run's highest sample, its extent the run, its score the apex's z."""

    k: float = 3.0
    k_low: float = 1.0
    min_length: int = 2
    window: int = 51
    kind: ClassVar[str] = "detect"
    name: ClassVar[str] = "robust_z"

    def __call__(self, trace: Trace) -> Candidates:
        y, valid = trace.y, trace.valid
        b, n = y.shape
        base = trace.baseline
        if base is None:
            base = rolling_quantile(y, self.window if self.window % 2 else self.window + 1, 0.5, valid=valid)
        resid = torch.where(valid, y - base, torch.full_like(y, float("nan")))
        if trace.noise is not None:
            sigma = trace.noise if trace.noise.dim() == 2 else trace.noise.unsqueeze(-1)
        else:
            med = torch.nanmedian(resid, dim=-1, keepdim=True).values
            sigma = torch.nanmedian((resid - med).abs(), dim=-1, keepdim=True).values * MAD_TO_SIGMA
        z = torch.where(valid, resid / sigma.clamp(min=1e-300), torch.full_like(resid, float("-inf")))
        above = valid & (z > self.k_low)
        prev = F.pad(above, (1, 0), value=False)[:, :-1]
        start = above & ~prev
        run = torch.cumsum(start.to(torch.int64), dim=-1) * above  # 1.. per row, 0 outside
        n_runs = int(run.max()) if run.numel() else 0
        if n_runs == 0:
            e = torch.empty((b, 0), dtype=torch.int64)
            return Candidates(apex=e, score=e.to(torch.float64), valid=e.to(torch.bool), left=e, right=e)
        key = (torch.arange(b).unsqueeze(-1) * (n_runs + 1) + run).reshape(-1)
        size = b * (n_runs + 1)
        zf = torch.where(above, z, torch.full_like(z, float("-inf"))).reshape(-1)
        idx = torch.arange(n).expand(b, n).reshape(-1)
        top = torch.full((size,), float("-inf"), dtype=torch.float64).scatter_reduce(0, key, zf, "amax")
        at_top = (zf == top[key]) & above.reshape(-1)
        apex = torch.full((size,), n, dtype=torch.int64).scatter_reduce(
            0, key, torch.where(at_top, idx, torch.full_like(idx, n)), "amin")  # fmt: skip
        first = torch.full((size,), n, dtype=torch.int64).scatter_reduce(
            0, key, torch.where(above.reshape(-1), idx, torch.full_like(idx, n)), "amin")  # fmt: skip
        last = torch.full((size,), -1, dtype=torch.int64).scatter_reduce(
            0, key, torch.where(above.reshape(-1), idx, torch.full_like(idx, -1)), "amax")  # fmt: skip
        shape = (b, n_runs + 1)
        top, apex, first, last = (t.reshape(shape)[:, 1:] for t in (top, apex, first, last))
        ok = (top >= self.k) & ((last - first + 1) >= self.min_length) & (apex < n)
        return Candidates(apex=torch.where(ok, apex, -1), score=torch.where(ok, top, torch.full_like(top, float("-inf"))),
                          valid=ok, left=torch.where(ok, first, -1), right=torch.where(ok, last, -1),
                          extras={"z": top})  # fmt: skip
