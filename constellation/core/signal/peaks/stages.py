"""The stages a peak picker is built from, and the registries that name them.

A peak picker is a pipeline of interchangeable stages (Pipeline in
``docs/plans/peak-detection.md``):

    smooth → baseline → noise → detect → boundaries → refine

and every stage of a kind takes and gives the same things, so any detector
can follow any smoother. A stage is a frozen dataclass: its fields are its
parameters, and ``kind`` / ``name`` / ``version`` (class attributes) say
what it is. ``identity()`` is plain data that names the stage and its
parameters exactly, for a provenance key (``PeakPicker.key``) and to
rebuild it (``make_stage``). A stage's version is raised whenever what it
computes changes.

Each kind has a registry, ``name → class``; :func:`register` adds one, so a
new method (a CWT ridge detector, a change-point segmenter, a learned
picker) is a class with the kind's call signature and nothing else.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from constellation.core.signal import baseline as _baseline
from constellation.core.signal import noise as _noise
from constellation.core.signal import smooth as _smooth

__all__ = [
    "Stage",
    "REGISTRIES",
    "SMOOTHERS",
    "BASELINES",
    "NOISE",
    "DETECTORS",
    "BOUNDARIES",
    "REFINERS",
    "register",
    "make_stage",
    "stage_from_identity",
    "MovingAverage",
    "GaussianSmooth",
    "SavitzkyGolay",
    "RollingQuantile",
    "Opening",
    "Snip",
    "DiffMad",
    "ResidualMad",
    "RollingMad",
]


class Stage:
    """What every stage has (module docstring)."""

    kind: ClassVar[str] = ""
    name: ClassVar[str] = ""
    version: ClassVar[int] = 1

    def parameters(self) -> dict[str, Any]:
        out = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    def identity(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "version": self.version, "parameters": self.parameters()}


SMOOTHERS: dict[str, type] = {}
BASELINES: dict[str, type] = {}
NOISE: dict[str, type] = {}
DETECTORS: dict[str, type] = {}
BOUNDARIES: dict[str, type] = {}
REFINERS: dict[str, type] = {}
REGISTRIES: dict[str, dict[str, type]] = {
    "smooth": SMOOTHERS,
    "baseline": BASELINES,
    "noise": NOISE,
    "detect": DETECTORS,
    "boundaries": BOUNDARIES,
    "refine": REFINERS,
}


def register(cls: type) -> type:
    """Add a stage class to its kind's registry (a class decorator)."""
    registry = REGISTRIES.get(cls.kind)
    if registry is None:
        raise ValueError(f"{cls.__name__} has kind {cls.kind!r}; kinds are {sorted(REGISTRIES)}")
    if cls.name in registry and registry[cls.name] is not cls:
        raise ValueError(f"a {cls.kind} stage named {cls.name!r} is already registered")
    registry[cls.name] = cls
    return cls


def make_stage(kind: str, name: str, **parameters: Any) -> Stage:
    """The stage of ``kind`` named ``name``, with ``parameters``."""
    registry = REGISTRIES.get(kind)
    if registry is None:
        raise KeyError(f"unknown stage kind {kind!r}; kinds are {sorted(REGISTRIES)}")
    cls = registry.get(name)
    if cls is None:
        raise KeyError(f"unknown {kind} stage {name!r}; available: {sorted(registry)}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    args = {k: tuple(v) if isinstance(v, list) and k in fields else v for k, v in parameters.items()}
    return cls(**args)


def stage_from_identity(identity: dict[str, Any]) -> Stage:
    """The stage an ``identity()`` names, refusing another version."""
    stage = make_stage(identity["kind"], identity["name"], **identity.get("parameters", {}))
    if stage.version != identity.get("version", stage.version):
        raise ValueError(
            f"{identity['kind']} stage {identity['name']!r} is version {stage.version}, "
            f"not the {identity['version']} named"
        )
    return stage


# ── preprocessing: smoothing ──────────────────────────────────────────
#
# A smoother: (y [B, N], valid [B, N]) -> y [B, N].


@register
@dataclass(frozen=True)
class MovingAverage(Stage):
    """The mean of each sample's window of valid samples."""

    window: int = 5
    kind: ClassVar[str] = "smooth"
    name: ClassVar[str] = "moving_average"

    def __call__(self, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return _smooth.moving_average(y, self.window, valid=valid)


@register
@dataclass(frozen=True)
class GaussianSmooth(Stage):
    """A Gaussian-weighted average, ``sigma`` in samples."""

    sigma: float = 1.0
    truncate: float = 4.0
    kind: ClassVar[str] = "smooth"
    name: ClassVar[str] = "gaussian"

    def __call__(self, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return _smooth.gaussian(y, self.sigma, valid=valid, truncate=self.truncate)


@register
@dataclass(frozen=True)
class SavitzkyGolay(Stage):
    """The local least-squares polynomial through each window's valid samples."""

    window: int = 5
    degree: int = 2
    kind: ClassVar[str] = "smooth"
    name: ClassVar[str] = "savitzky_golay"

    def __call__(self, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return _smooth.savitzky_golay(y, self.window, self.degree, valid=valid)


# ── preprocessing: baselines ──────────────────────────────────────────
#
# A baseline: (y [B, N], valid [B, N]) -> baseline [B, N].


@register
@dataclass(frozen=True)
class RollingQuantile(Stage):
    """The q-quantile of each sample's window (the moving median at 0.5)."""

    window: int = 51
    q: float = 0.5
    kind: ClassVar[str] = "baseline"
    name: ClassVar[str] = "rolling_quantile"

    def __call__(self, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return _baseline.rolling_quantile(y, self.window, self.q, valid=valid)


@register
@dataclass(frozen=True)
class Opening(Stage):
    """Morphological opening with a flat element of ``window`` samples."""

    window: int = 51
    kind: ClassVar[str] = "baseline"
    name: ClassVar[str] = "opening"

    def __call__(self, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return _baseline.opening(y, self.window, valid=valid)


@register
@dataclass(frozen=True)
class Snip(Stage):
    """SNIP clipping, ``iterations`` half-widths."""

    iterations: int = 20
    transform: bool = True
    kind: ClassVar[str] = "baseline"
    name: ClassVar[str] = "snip"

    def __call__(self, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return _baseline.snip(y, self.iterations, valid=valid, transform=self.transform)


# ── preprocessing: noise ──────────────────────────────────────────────
#
# A noise estimator: (y [B, N], valid [B, N], baseline [B, N] | None)
#   -> sigma [B] (per row) or [B, N] (per sample).


@register
@dataclass(frozen=True)
class DiffMad(Stage):
    """Each row's sigma from the MAD of its first differences."""

    floor: float = 0.0
    kind: ClassVar[str] = "noise"
    name: ClassVar[str] = "diff_mad"

    def __call__(self, y, valid, baseline=None):
        return _noise.diff_mad(y, valid=valid, floor=self.floor)


@register
@dataclass(frozen=True)
class ResidualMad(Stage):
    """Each row's sigma from the MAD of its residual to the baseline (to
    zero without one)."""

    floor: float = 0.0
    kind: ClassVar[str] = "noise"
    name: ClassVar[str] = "residual_mad"

    def __call__(self, y, valid, baseline=None):
        b = torch.zeros_like(y) if baseline is None else baseline
        return _noise.residual_mad(y, b, valid=valid, floor=self.floor)


@register
@dataclass(frozen=True)
class RollingMad(Stage):
    """A sigma at each sample, from the residual's MAD over a window."""

    window: int = 51
    floor: float = 0.0
    kind: ClassVar[str] = "noise"
    name: ClassVar[str] = "rolling_mad"

    def __call__(self, y, valid, baseline=None):
        return _noise.rolling_mad(y, self.window, baseline=baseline, valid=valid, floor=self.floor)
