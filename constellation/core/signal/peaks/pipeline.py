"""``PeakPicker``: smoothing, baseline, noise, detection, boundaries and
refinement, composed (``stages``), on batches of traces.

A picker takes ``y [..., N]`` with an axis ``x`` (``[..., N]`` or ``[N]``,
irregular allowed; sample indices by default) and a validity mask, and
gives :class:`~._types.Peaks` ``[..., M]``. Each row is compacted first
(its valid samples first), every stage runs on the compacted rows, and the
peaks' indices are mapped back.

With ``boundary_trace`` (on the same axis), detection runs on ``y`` and
the extent and shape on the boundary trace: a peak found in one trace (a
similarity score along time, a derivative) and measured in another (the
intensity it scores). The apex moves to the boundary trace's highest
valid sample within ``recenter`` samples before the boundaries are walked.
The boundary trace keeps the detection trace's samples, holes where it has
no value (``boundary_valid`` false, or not finite), and every rule and
measure passes over them (``boundaries``, ``measures``); a peak with no
valid boundary sample that near keeps what detection gave it, and has no
boundary apex, extent, width, area, asymmetry or jaggedness (``-1`` /
NaN). Until 2026-11-11 the recentring could land on a hole, and a
boundary trace with no valid sample still gave every peak an extent.

With no baseline or noise stage of its own, the picker measures with the
baseline and noise the detector estimated (``Candidates.baseline`` /
``.noise``), on the detection trace and on the boundary trace when it is
the same one: until 2026-11-11 RobustZ found peaks above its rolling
median and the widths and areas were taken above 0.

``identity()`` names every stage and parameter, ``key()`` digests it (the
``core.stages`` key), and ``from_identity`` rebuilds the picker: a result
records how its peaks were found, as a peak list records its centroider.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from constellation.core.signal._rows import compact
from constellation.core.signal.peaks._types import Candidates, Peaks, Trace
from constellation.core.signal.peaks.boundaries import Valley
from constellation.core.signal.peaks.measures import measure
from constellation.core.signal.peaks.refine import NoRefine, x_at
from constellation.core.signal.peaks.stages import Stage, stage_from_identity

__all__ = ["PeakPicker"]

#: Bump when what a picker computes around its stages changes. 2
#: (2026-11-11): a boundary trace's holes, the detector's baseline, and a
#: last sample's x beside padding.
PICKER_VERSION = 2


@dataclass(frozen=True)
class PeakPicker:
    """A peak picker of interchangeable stages (module docstring). Only
    ``detect`` is required; ``max_peaks`` keeps the best by score."""

    detect: Stage
    smooth: Stage | None = None
    baseline: Stage | None = None
    noise: Stage | None = None
    boundaries: Stage = Valley()
    refine: Stage = NoRefine()
    max_peaks: int = 8
    recenter: int = 2

    def _stages(self) -> dict[str, Stage | None]:
        return {"smooth": self.smooth, "baseline": self.baseline, "noise": self.noise, "detect": self.detect,
                "boundaries": self.boundaries, "refine": self.refine}  # fmt: skip

    def identity(self) -> dict[str, Any]:
        return {
            "name": "peak_picker",
            "version": PICKER_VERSION,
            "stages": {kind: None if s is None else s.identity() for kind, s in self._stages().items()},
            "max_peaks": self.max_peaks,
            "recenter": self.recenter,
        }

    def key(self) -> str:
        """The ``core.stages`` digest of :meth:`identity`."""
        from constellation.core.stages import stage_key

        return stage_key(**self.identity())

    @classmethod
    def from_identity(cls, identity: dict[str, Any]) -> PeakPicker:
        if identity.get("version", PICKER_VERSION) != PICKER_VERSION:
            raise ValueError(f"a peak picker of version {identity['version']}; this one is {PICKER_VERSION}")
        stages = {kind: None if s is None else stage_from_identity(s) for kind, s in identity["stages"].items()}
        return cls(**stages, max_peaks=identity["max_peaks"], recenter=identity["recenter"])

    # ── running it ────────────────────────────────────────────────────

    def _prepare(self, y: torch.Tensor, x: torch.Tensor, valid: torch.Tensor, count: torch.Tensor) -> Trace:
        raw = y
        if self.smooth is not None:
            y = self.smooth(y, valid).to(torch.float64)
        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        baseline = None if self.baseline is None else self.baseline(y, valid).to(torch.float64)
        noise = None if self.noise is None else self.noise(y, valid, baseline)
        return Trace(y=y, x=x, valid=valid, count=count, baseline=baseline, noise=noise, raw=raw)

    def __call__(
        self,
        y: torch.Tensor,
        *,
        x: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        boundary_trace: torch.Tensor | None = None,
        boundary_valid: torch.Tensor | None = None,
    ) -> Peaks:
        lead, n = tuple(y.shape[:-1]), y.shape[-1]
        y2 = y.reshape(-1, n).to(torch.float64)
        ok = torch.isfinite(y2) if valid is None else (valid.reshape(-1, n).bool() & torch.isfinite(y2))
        if x is None:
            x2 = torch.arange(n, dtype=torch.float64).expand_as(y2)
        else:
            x2 = torch.broadcast_to(x.to(torch.float64), y.shape).reshape(-1, n)
        c = compact(ok)
        count = c.count
        vc = c.valid
        det = self._prepare(c.gather(y2, float("nan")), c.gather(x2, float("nan")), vc, count)
        if boundary_trace is not None:
            # The detection trace's samples, holes where the boundary trace
            # has no value.
            b2 = boundary_trace.reshape(-1, n).to(torch.float64)
            bok = ok & torch.isfinite(b2)
            if boundary_valid is not None:
                bok &= boundary_valid.reshape(-1, n).bool()
            bvalues = torch.where(bok, b2, torch.full_like(b2, float("nan")))
            bnd = self._prepare(c.gather(bvalues, float("nan")), det.x, vc & c.gather(bok, False), count)
        cand: Candidates = self.detect(det)
        if self.baseline is None and cand.baseline is not None:
            det = replace(det, baseline=cand.baseline.to(torch.float64))
        if self.noise is None and cand.noise is not None:
            det = replace(det, noise=cand.noise.to(torch.float64))
        if boundary_trace is None:
            bnd = det
        # The best max_peaks by score.
        m = min(self.max_peaks, cand.m)
        score = torch.where(cand.valid, cand.score, torch.full_like(cand.score, float("-inf")))
        if cand.m:
            order = torch.sort(score, dim=-1, descending=True, stable=True).indices[..., :m]
            cand = cand.take(order)
        apex = torch.where(cand.valid, cand.apex, -1)
        boundary_apex = apex if boundary_trace is None else _recenter(bnd, apex, self.recenter)
        # The rules bound only the peaks they have an apex for.
        bounded = cand.valid & (boundary_apex >= 0)
        left, right = self.boundaries(bnd, replace(cand, valid=bounded), boundary_apex.clamp(min=0))
        left = torch.where(bounded, left, -1)
        right = torch.where(bounded, right, -1)
        measures = measure(det, bnd, apex, boundary_apex, left, right, cand.valid)
        apex_x, components = self.refine(det, apex.clamp(min=0), left.clamp(min=0), right.clamp(min=0), bounded)

        def back(i):  # compacted → the caller's positions
            return torch.where(cand.valid, c.position(i), torch.full_like(i, -1)).reshape(*lead, m)

        def shaped(t):
            return t.reshape(*lead, m, *t.shape[2:])

        nan = torch.full(apex.shape, float("nan"), dtype=torch.float64)
        return Peaks(
            valid=shaped(cand.valid),
            apex=back(apex),
            apex_x=shaped(torch.where(cand.valid, apex_x, nan)),
            boundary_apex=back(boundary_apex),
            left=back(left),
            right=back(right),
            left_x=shaped(torch.where(bounded, x_at(det.x, left.clamp(min=0).to(torch.float64)), nan)),
            right_x=shaped(torch.where(bounded, x_at(det.x, right.clamp(min=0).to(torch.float64)), nan)),
            left_base=back(measures["left_base"]),
            right_base=back(measures["right_base"]),
            height=shaped(measures["height"]),
            prominence=shaped(measures["prominence"]),
            snr=shaped(measures["snr"]),
            score=shaped(torch.where(cand.valid, cand.score, nan)),
            width=shaped(measures["width"]),
            area=shaped(measures["area"]),
            asymmetry=shaped(measures["asymmetry"]),
            jaggedness=shaped(measures["jaggedness"]),
            extras={k: shaped(v) for k, v in cand.extras.items()},
            components=None if components is None else shaped(components),
            form=None if components is None else getattr(self.refine, "form", None),
        )


def _recenter(trace: Trace, apex: torch.Tensor, reach: int) -> torch.Tensor:
    """Each apex moved to the trace's highest valid sample within ``reach``
    (``-1`` where the trace has none that near)."""
    b, n = trace.y.shape
    p = apex.clamp(min=0)
    if reach <= 0:
        return torch.where((apex >= 0) & torch.gather(trace.valid, -1, p), apex, torch.full_like(apex, -1))
    y = torch.where(trace.valid, trace.y, torch.full_like(trace.y, float("-inf")))
    width = 2 * reach + 1
    windows = F.pad(y, (reach, reach), value=float("-inf")).unfold(-1, width, 1)  # [B, N, 2r+1]
    present = F.pad(trace.valid, (reach, reach), value=False).unfold(-1, width, 1)
    around = torch.gather(windows, 1, p.unsqueeze(-1).expand(*p.shape, width))
    any_valid = torch.gather(present, 1, p.unsqueeze(-1).expand(*p.shape, width)).any(-1)
    moved = (p + torch.argmax(around, dim=-1) - reach).clamp(0, n - 1)
    return torch.where((apex >= 0) & any_valid, moved, torch.full_like(apex, -1))
