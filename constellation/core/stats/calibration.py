"""Calibration / dose-response / standard-curve models.

Inspiration: HPLC peak-area calibration, ELISA dose-response curves,
fragment-analyzer ladder calibration, immunoassay 4PL fits. These are
**Parametric** subclasses (not `Distribution` or `PeakShape`) — they
describe deterministic input/output relationships and don't carry a
density.

`log_prob` and `cdf` raise `NotImplementedError` (inherited from
`Parametric` base when used in a context expecting a `Distribution`,
but explicitly here for clarity).

v1 ships:
    Sigmoidal       4-parameter logistic (4PL): bottom, top, x50, slope
    Hill            Vmax · xⁿ / (Kⁿ + xⁿ)
    LogLinear       a · log(x) + b
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .parametric import Parametric

_EPS = 1e-12


class Sigmoidal(Parametric):
    """4-parameter logistic curve.

    `y = bottom + (top - bottom) / (1 + exp(-slope · (x - x50)))`

    Params: `bottom`, `top`, `x50`, `slope` (all in real space — the
    sign of `slope` matters; positive for increasing, negative for
    decreasing). Initial slope of 1 keeps the optimizer well-conditioned.
    """

    def __init__(
        self,
        bottom: float = 0.0,
        top: float = 1.0,
        x50: float = 0.0,
        slope: float = 1.0,
    ) -> None:
        super().__init__()
        self.bottom = nn.Parameter(torch.tensor(bottom, dtype=torch.float64))
        self.top = nn.Parameter(torch.tensor(top, dtype=torch.float64))
        self.x50 = nn.Parameter(torch.tensor(x50, dtype=torch.float64))
        self.slope = nn.Parameter(torch.tensor(slope, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the stable sigmoid formulation: torch.sigmoid handles the
        # large-magnitude regime via internal where(z>0, ...) saturation.
        z = self.slope * (x - self.x50)
        return self.bottom + (self.top - self.bottom) * torch.sigmoid(z)

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {
            "bottom": self.bottom.detach(),
            "top": self.top.detach(),
            "x50": self.x50.detach(),
            "slope": self.slope.detach(),
        }


class Hill(Parametric):
    """Hill equation — sigmoidal binding/dose-response on positive `x`.

    `y = Vmax · xⁿ / (Kⁿ + xⁿ)`

    Params: `log_Vmax`, `log_K`, `log_n`. At `n=1` reduces to
    Michaelis-Menten; `y(K) = Vmax/2` regardless of `n`. Returns `0`
    for `x ≤ 0` (Hill is undefined on the non-positive ray).
    """

    def __init__(
        self, Vmax: float = 1.0, K: float = 1.0, n: float = 1.0
    ) -> None:
        super().__init__()
        self.log_Vmax = nn.Parameter(
            torch.tensor(math.log(Vmax), dtype=torch.float64)
        )
        self.log_K = nn.Parameter(torch.tensor(math.log(K), dtype=torch.float64))
        self.log_n = nn.Parameter(torch.tensor(math.log(n), dtype=torch.float64))

    @property
    def Vmax(self) -> torch.Tensor:
        return self.log_Vmax.exp()

    @property
    def K(self) -> torch.Tensor:
        return self.log_K.exp()

    @property
    def n(self) -> torch.Tensor:
        return self.log_n.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n
        K = self.K
        # x^n via exp(n·log(x.clamp(min=eps))) — torch's `pow` chokes on
        # zero/negative bases for fractional `n`, and we want to short-
        # circuit to 0 for non-positive x anyway.
        x_pos = x.clamp(min=_EPS)
        x_n = torch.exp(n * torch.log(x_pos))
        K_n = torch.exp(n * torch.log(K))
        result = self.Vmax * x_n / (K_n + x_n)
        return torch.where(x > 0, result, torch.zeros_like(result))

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {
            "Vmax": self.Vmax.detach(),
            "K": self.K.detach(),
            "n": self.n.detach(),
        }


class LogLinear(Parametric):
    """Log-linear standard curve: `y = a · log(x) + b` for `x > 0`.

    Params: `a`, `b`. Useful for nucleic-acid ladder calibration
    (mobility vs log-fragment-size) and concentration / signal
    scaling where the linear regime is in log-space.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * torch.log(x.clamp(min=_EPS)) + self.b

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {"a": self.a.detach(), "b": self.b.detach()}
