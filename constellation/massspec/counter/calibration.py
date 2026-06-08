"""`GlobalCalibration` — per-acquisition shared calibrants.

Holds the parameters shared across every progenitor in an acquisition. All
are `nn.Parameter`s so a staged calibrator can freeze/thaw and fit them;
PR-1 keeps them fixed at sensible (IIT-corrected nb42b) defaults while
`estimate_n` fits per-peptide `N` + shape.

Units (the ms convention; see `iit.py`): gain `α(z)` [a.u.·ion⁻¹]; m/z
offset / errors [ppm]; `d_mz` [Da]; `α_mz`, `ν_mz`, `ρ` dimensionless;
`R_ref` resolution, `mz_ref` [Th].

Gain model (`alpha_model`, the switch from the plan):
    "linear"        α(z) = softplus(α₀ + α₁·z)   (default; nb-supported)
    "linear_origin" α(z) = softplus(α₁·z)        (∝ charge; image-current physics)
    "per_z"         α(z) = exp(log_α_z[z])        (one free gain per charge)

m/z-error precision (consumed by `channels.MzVarianceModel`): the power-law
exponent `α_mz` and Student-t dof `ν_mz` are GLOBAL here; the per-peptide
`c_mz` lives on the per-peptide tier. The resolution model `R(m/z) =
R_ref·√(mz_ref/mz)` feeds the *intensity* variance scaling `ρ_R =
(R/R_ref)^ρ` (Eq. 16, grounded in nb27); tying it to the m/z `c_mz` is the
Study-A conjecture and is NOT done here.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch
from torch import nn

from constellation.core.stats.units import da_to_ppm

AlphaModel = Literal["linear", "linear_origin", "per_z"]

_LOG_NU_MIN = math.log(0.5)


class GlobalCalibration(nn.Module):
    """Per-acquisition calibrant parameters. See module docstring for units
    and the `alpha_model` switch."""

    def __init__(
        self,
        *,
        n_isotopes: int = 3,
        charges: Sequence[int] = (2, 3, 4),
        alpha_model: AlphaModel = "linear",
        alpha0: float = 1.0,
        alpha1: float = 15.0,
        alpha_mz: float = 0.6,
        nu_mz: float = 7.8,
        mz_offset_ppm: float = 0.0,
        d_mz_da: Sequence[float] | torch.Tensor | None = None,
        rho: float = 0.5,
        r_ref: float | None = None,
        mz_ref: float | None = None,
        analyzer: str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if alpha_model not in ("linear", "linear_origin", "per_z"):
            raise ValueError(f"unknown alpha_model {alpha_model!r}")
        self.alpha_model: AlphaModel = alpha_model
        self.n_isotopes = int(n_isotopes)
        self.analyzer = analyzer
        self._dtype = dtype
        self.register_buffer(
            "charges", torch.tensor(list(charges), dtype=torch.long)
        )

        # -- gain α(z) --------------------------------------------------
        if alpha_model in ("linear", "linear_origin"):
            self.alpha1 = nn.Parameter(torch.tensor(alpha1, dtype=dtype))
            if alpha_model == "linear":
                self.alpha0 = nn.Parameter(torch.tensor(alpha0, dtype=dtype))
        else:  # per_z
            base = torch.tensor(
                [alpha0 + alpha1 * z for z in charges], dtype=dtype
            ).clamp(min=1e-6)
            self.log_alpha_z = nn.Parameter(base.log())

        # -- m/z calibration --------------------------------------------
        self.mz_offset_ppm = nn.Parameter(torch.tensor(mz_offset_ppm, dtype=dtype))
        if d_mz_da is None:
            d_init = torch.zeros(self.n_isotopes, dtype=dtype)
        else:
            d_init = torch.as_tensor(d_mz_da, dtype=dtype)
            if d_init.shape != (self.n_isotopes,):
                raise ValueError(
                    f"d_mz_da must have shape ({self.n_isotopes},); "
                    f"got {tuple(d_init.shape)}"
                )
        self.d_mz_da = nn.Parameter(d_init)
        self.log_alpha_mz = nn.Parameter(torch.tensor(math.log(alpha_mz), dtype=dtype))
        self.log_nu_mz = nn.Parameter(torch.tensor(math.log(nu_mz), dtype=dtype))

        # -- resolution (intensity-variance scaling) --------------------
        self.rho = nn.Parameter(torch.tensor(rho, dtype=dtype))
        self.r_ref = float(r_ref) if r_ref is not None else None
        self.mz_ref = float(mz_ref) if mz_ref is not None else None

        # -- promoted peak-shape hyperprior -----------------------------
        # Set by StagedCalibration (stage 3) from the calibrant population's
        # fitted HyperEMG shapes; consumed by estimate_n's VB priors. A
        # {full_param_name: (mean, std)} map over log/logit shape params
        # (e.g. "peak.log_sigma" -> (mean, std)) or None until promoted. Not a
        # buffer — it carries no gradient and is persisted via the Arrow table.
        self.peak_shape_prior: dict[str, tuple[float, float]] | None = None

    def set_peak_shape_prior(
        self, prior: dict[str, tuple[float, float]] | None
    ) -> None:
        """Install (or clear) the promoted peak-shape hyperprior. Keys are full
        progenitor parameter names (`peak.log_sigma`, `peak.log_tau_r`,
        `peak.log_tau_l`, `peak.logit_eta`); values are `(mean, std)` Gaussians
        in the parameter's log/logit space."""
        self.peak_shape_prior = prior

    # -- gain -----------------------------------------------------------

    def gain(self, z: torch.Tensor) -> torch.Tensor:
        """α(z) [a.u./ion] for charge value(s) `z` (any broadcastable shape)."""
        z = torch.as_tensor(z, dtype=self._dtype)
        if self.alpha_model == "linear":
            return torch.nn.functional.softplus(self.alpha0 + self.alpha1 * z)
        if self.alpha_model == "linear_origin":
            return torch.nn.functional.softplus(self.alpha1 * z)
        # per_z: map each charge value to its index in `charges`.
        zl = z.round().long()
        idx = torch.searchsorted(self.charges, zl)
        idx = idx.clamp(max=self.charges.numel() - 1)
        return self.log_alpha_z.exp().index_select(0, idx.reshape(-1)).reshape(z.shape)

    # -- m/z ------------------------------------------------------------

    @property
    def alpha_mz(self) -> torch.Tensor:
        """m/z-error power-law exponent (positive)."""
        return self.log_alpha_mz.exp()

    @property
    def nu_mz(self) -> torch.Tensor:
        """m/z-error Student-t degrees of freedom."""
        return self.log_nu_mz.clamp(min=_LOG_NU_MIN).exp()

    def mz_center_ppm(
        self, channel_mz: torch.Tensor, z: torch.Tensor, isotope: torch.Tensor
    ) -> torch.Tensor:
        """Expected m/z-error center [ppm] for channels at observed m/z
        `channel_mz` [Th], charge `z`, isotope index `isotope`.

        `ε_center = mz_offset + (d_mz_k / z) → ppm`. The per-isotope `d_mz_k`
        [Da] is a neutral-mass spacing correction; dividing by `z` gives the
        m/z offset, and `da_to_ppm` converts it relative to the channel m/z
        (the Da→ppm step the manuscript's Eq. 20 left implicit). `d_mz_0` is
        the monoisotopic reference and contributes only `mz_offset`.
        """
        iso = torch.as_tensor(isotope, dtype=torch.long)
        # The monoisotopic (k=0) d_mz is the reference and is pinned to 0: its
        # absolute m/z position is `mz_offset`, so a free d_mz_0 would be
        # degenerate with `mz_offset` (the calibrator cannot separate them). Only
        # the k≥1 isotope-spacing corrections are free; the pin is gradient-safe
        # (index 0 is a constant zero, so no gradient reaches d_mz_da[0]).
        d_pinned = torch.cat([self.d_mz_da.new_zeros(1), self.d_mz_da[1:]])
        d_da = d_pinned.index_select(0, iso.reshape(-1)).reshape(iso.shape)
        mz_offset_da = d_da / torch.as_tensor(z, dtype=self._dtype)
        return self.mz_offset_ppm + da_to_ppm(mz_offset_da, channel_mz)

    # -- resolution -----------------------------------------------------

    def rho_R(self, channel_mz: torch.Tensor) -> torch.Tensor:
        """Intensity-variance resolution factor `ρ_R = (R(mz)/R_ref)^ρ` at
        `channel_mz` [Th]. With `R(mz) = R_ref·√(mz_ref/mz)`, this is
        `(mz_ref/mz)^(ρ/2)`. Returns `1.0` (no scaling) when `R_ref`/`mz_ref`
        are unset."""
        if self.r_ref is None or self.mz_ref is None:
            return torch.ones_like(torch.as_tensor(channel_mz, dtype=self._dtype))
        ratio = self.mz_ref / torch.as_tensor(channel_mz, dtype=self._dtype)
        return ratio ** (0.5 * self.rho)

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "mz_offset_ppm": self.mz_offset_ppm.detach(),
            "d_mz_da": self.d_mz_da.detach(),
            "alpha_mz": self.alpha_mz.detach(),
            "nu_mz": self.nu_mz.detach(),
            "rho": self.rho.detach(),
        }
        if self.alpha_model == "linear":
            out["alpha0"] = self.alpha0.detach()
            out["alpha1"] = self.alpha1.detach()
        elif self.alpha_model == "linear_origin":
            out["alpha1"] = self.alpha1.detach()
        else:
            out["alpha_z"] = self.log_alpha_z.exp().detach()
        return out


__all__ = ["GlobalCalibration", "AlphaModel"]
