"""`Progenitor` — one peptide species, and the packed observation it scores.

A `Progenitor` is the reusable, physics-grounded *score*: given a panel of
observed ions it evaluates the joint m/z + intensity log-likelihood of a
single peptide species producing them, and (run forward) it *simulates* a
panel. It composes a `HyperEMGPeak` (the elution flux `N(t)`) with a
charge partition (`softmax(−E_z)`), the theoretical isotope fractions
`p_k`, and per-peptide noise (`ν_I`, per-isotope `c_mz`), conditioned on a
shared `GlobalCalibration`.

Time base: **everything is in milliseconds** (the ms convention from
`iit.py`), so the flux `N(t)` is [ions/ms], `∫N(t) d(rt_ms) = N_total`
[ions] exactly, and `N_total = peak.integrate()` is the deliverable with no
stray unit factors. Retention times from seconds-based sources are scaled
to ms at the observation boundary (`CounterObservation`).

The calibration is held by reference (not a registered submodule) so that a
`Panel` of several progenitors shares ONE calibration without duplicating
its parameters — and `torch.func.functional_call` still patches it
correctly (same object identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.parametric import Distribution
from constellation.core.stats.peaks import HyperEMGPeak
from constellation.massspec.peptide.envelope import EnvelopeMode, peptide_envelope

from .calibration import GlobalCalibration
from .iit import accumulated_count

__all__ = ["CounterObservation", "Progenitor"]


@dataclass(frozen=True)
class CounterObservation:
    """Packed observation for one panel — a shared `(scan × channel)` grid.

    A *channel* is a `(charge, isotope)` coordinate; the grid is the target's
    theoretical envelope. All tensors are float64 / long and aligned on the
    channel axis. Units: `rt`, `iit` [ms]; `intensity` [a.u./ms]; `mz_error`
    [ppm]; `channel_mz` [Th].
    """

    rt: torch.Tensor  # (S,)
    iit: torch.Tensor  # (S,)
    intensity: torch.Tensor  # (S, C)
    mz_error: torch.Tensor  # (S, C)
    mask: torch.Tensor  # (S, C) bool — True where an ion was observed
    channel_z: torch.Tensor  # (C,) charge (long)
    channel_isotope: torch.Tensor  # (C,) isotope index (long)
    channel_mz: torch.Tensor  # (C,) theoretical m/z (float, Th)
    # Stable physical-peak identity for mapping a fitted panel's soft attribution
    # back to raw peaks ("what's left" after targets are searched, AND cross-panel
    # reconciliation): `scan` is the (S,) scan-number axis; `source_mz` the (S, C)
    # measured m/z [Th] of the peak in each filled cell (NaN elsewhere). The pair
    # `(scan, source_mz)` is IDENTICAL for a peak extracted under different targets
    # — a per-(target,scan,ion) trace row index is not — so it survives as the
    # anti-join / cross-panel key. Optional — None for simulated / legacy
    # observations built without a raw-peak source.
    scan: torch.Tensor | None = None  # (S,) long — scan numbers
    source_mz: torch.Tensor | None = None  # (S, C) float — observed m/z, NaN where unobserved

    @property
    def n_scans(self) -> int:
        return int(self.rt.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.channel_z.shape[0])

    def recovered_count(self, calibration: GlobalCalibration) -> torch.Tensor:
        """Per-channel recovered ion count `N_obs = I·τ/α(z)` `(S, C)` on observed
        cells, 0 on non-detections — the per-ion count substrate for logL-based ion
        selection (L4) and N-weighted seeding (L6). Uses the canonical
        `iit.accumulated_count` (`I·τ/α`); a *derived projection* given a
        calibration, so the observation itself stays raw (the count is not stored).

        The count is floored at `accumulated_count`'s `1e-9` (so it can safely feed
        `log` / `N^{−α_mz}` terms); below that floor it diverges from the likelihood's
        unclamped `n_obs` in `panel_log_prob` — irrelevant at real ion counts (≫1),
        but do not treat the two as identical at the sub-1e-9 boundary."""
        gain_ch = calibration.gain(self.channel_z.to(self.intensity.dtype))  # (C,)
        count = accumulated_count(self.intensity, self.iit[:, None], gain_ch[None, :])
        return torch.where(self.mask, count, torch.zeros_like(count))


def _channel_grid(
    charges: Sequence[int], n_isotopes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a `(charge × isotope)` grid into channel-axis `(z, k)` index
    tensors, charge-major."""
    z = torch.tensor(
        [c for c in charges for _ in range(n_isotopes)], dtype=torch.long
    )
    k = torch.tensor(
        [k for _ in charges for k in range(n_isotopes)], dtype=torch.long
    )
    return z, k


class Progenitor(Distribution):
    """One peptide species over a fixed `(charge × isotope)` channel grid.

    Learnable per-peptide parameters: the `HyperEMGPeak` (`log_N_total`, `mu`,
    `log_sigma`, `log_tau_r`, `log_tau_l`, `logit_eta`), `charge_energy` (Zc,)
    (charge free-energies → `f_z = softmax(−E)`), `log_nu_intensity`,
    `log_c_mz` (K,), and `isotope_energy_offset` (K−1,) — a learnable correction
    on the theoretical isotope log-weights (M+0 reference; see the `p_k`
    property). `calibration` is shared and frozen during per-peptide fits.
    """

    def __init__(
        self,
        *,
        charges: Sequence[int],
        isotope_fractions: torch.Tensor,
        calibration: GlobalCalibration,
        channel_mz: torch.Tensor,
        peak: HyperEMGPeak | None = None,
        nu_intensity: float = 5.0,
        c_mz_init: float = 100.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        p_k = torch.as_tensor(isotope_fractions, dtype=dtype)
        n_isotopes = int(p_k.shape[0])
        z, k = _channel_grid(charges, n_isotopes)
        if channel_mz.shape != z.shape:
            raise ValueError(
                f"channel_mz shape {tuple(channel_mz.shape)} != grid "
                f"({z.numel()},) for {len(charges)} charges × {n_isotopes} isotopes"
            )
        self._dtype = dtype
        self.register_buffer("channel_z", z)
        self.register_buffer("channel_isotope", k)
        self.register_buffer("channel_mz", channel_mz.to(dtype))
        # Theoretical isotope log-weights (fixed), with a learnable per-isotope
        # correction for k≥1 (M+0 is the reference) — see the `p_k` property.
        self.register_buffer("log_p_theo", (p_k / p_k.sum()).log())
        uniq = torch.tensor(list(charges), dtype=torch.long)
        self.register_buffer("charges", uniq)
        # channel → index into `charges`
        charge_idx = torch.searchsorted(uniq, z)
        self.register_buffer("charge_index_of_channel", charge_idx)
        # calibration by reference (not a submodule — see module docstring)
        self._calibration_ref = (calibration,)

        self.peak = peak if peak is not None else HyperEMGPeak()
        self.charge_energy = nn.Parameter(
            torch.zeros(len(charges), dtype=dtype)
        )  # uniform charge prior at init
        self.log_nu_intensity = nn.Parameter(
            torch.tensor(float(nu_intensity), dtype=dtype).log()
        )
        self.log_c_mz = nn.Parameter(
            torch.full((n_isotopes,), float(c_mz_init), dtype=dtype).log()
        )
        # Learnable isotope-fraction correction (free energy on the log-weights)
        # for k≥1; M+0 is the fixed reference. Init 0 → effective p_k = theoretical.
        self.isotope_energy_offset = nn.Parameter(
            torch.zeros(max(0, n_isotopes - 1), dtype=dtype)
        )

    @classmethod
    def for_peptide(
        cls,
        peptidoform: Peptidoform,
        charges: Sequence[int],
        calibration: GlobalCalibration,
        *,
        n_isotopes: int = 3,
        mode: EnvelopeMode = "binned",
        peak: HyperEMGPeak | None = None,
        **kwargs,
    ) -> "Progenitor":
        """Build a progenitor for a peptidoform: compute the theoretical
        isotope envelope (m/z + fractions) at each charge via
        `massspec.peptide.peptide_envelope`, lay the `(charge × isotope)`
        channel grid charge-major, and construct. Used by both the simulator
        and the real-data path (`estimate_n`)."""
        mzs: list[torch.Tensor] = []
        p_k: torch.Tensor | None = None
        for z in charges:
            mz_z, inten = peptide_envelope(
                peptidoform, charge=z, n_peaks=n_isotopes, mode=mode
            )
            mzs.append(mz_z.to(torch.float64))
            if p_k is None:
                p_k = inten.to(torch.float64)
        channel_mz = torch.cat(mzs)
        return cls(
            charges=charges,
            isotope_fractions=p_k,
            calibration=calibration,
            channel_mz=channel_mz,
            peak=peak,
            **kwargs,
        )

    # -- accessors ------------------------------------------------------

    @property
    def calibration(self) -> GlobalCalibration:
        return self._calibration_ref[0]

    @property
    def nu_intensity(self) -> torch.Tensor:
        return self.log_nu_intensity.exp()

    @property
    def p_k(self) -> torch.Tensor:
        """Effective isotope fractions `(K,)` — `softmax` over the theoretical
        log-weights plus the learnable per-isotope correction (M+0 fixed as the
        reference). At init the correction is 0, so `p_k` = the theoretical
        binned envelope. The correction is *required* at high resolution: ¹⁵N/¹³C
        partial separation changes how isotopic species convolve into each binned
        channel — a real, peptide-specific shift of the M+1 (and M+2) relative
        signal that scales with the N/C ratio, which a fixed envelope mis-models
        and which otherwise biases the likelihood."""
        if self.isotope_energy_offset.numel() == 0:
            return torch.softmax(self.log_p_theo, dim=0)
        offset = torch.cat([self.log_p_theo.new_zeros(1), self.isotope_energy_offset])
        return torch.softmax(self.log_p_theo + offset, dim=0)

    def charge_fractions(self) -> torch.Tensor:
        """`f_z = softmax(−E_z)` over the progenitor's charges, `(Zc,)`."""
        return torch.softmax(-self.charge_energy, dim=0)

    def c_mz_per_channel(self) -> torch.Tensor:
        """Per-channel m/z precision constant `c_mz` [ppm²], `(C,)`."""
        return self.log_c_mz.exp().index_select(0, self.channel_isotope)

    # -- forward physics ------------------------------------------------

    def predict(self, obs: CounterObservation) -> tuple[torch.Tensor, torch.Tensor]:
        """`(I_pred, N_count)`, both `(S, C)`.

        `I_pred = α(z)·f_z·p_k·N(t)` [a.u./ms] (Eq. 14 — `τ` cancels in the
        mean); `N_count = N(t)·f_z·p_k·τ` [ions] (Eq. 2 — the latent per-scan
        accumulated count that drives the m/z precision)."""
        n_flux = self.peak.forward(obs.rt)  # (S,) [ions/ms]
        f_z = self.charge_fractions()
        f_ch = f_z.index_select(0, self.charge_index_of_channel)  # (C,)
        p_ch = self.p_k.index_select(0, self.channel_isotope)  # (C,)
        gain_ch = self.calibration.gain(self.channel_z.to(self._dtype))  # (C,)
        weight = (f_ch * p_ch)[None, :]  # (1, C)
        i_pred = n_flux[:, None] * (gain_ch[None, :] * weight)  # (S, C)
        n_count = n_flux[:, None] * weight * obs.iit[:, None]  # (S, C)
        return i_pred, n_count

    # -- score ----------------------------------------------------------

    def log_prob(self, obs: CounterObservation) -> torch.Tensor:
        """Single-species joint log-likelihood (no interference / background) —
        the standalone score. Delegates to the additive panel assembly with a
        one-progenitor panel."""
        from .panel import panel_log_prob

        return panel_log_prob([self], obs, self.calibration)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Progenitor has no closed-form CDF.")

    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return {
            "N_total": self.peak.N_total.detach(),
            "charge_fractions": self.charge_fractions().detach(),
            "isotope_fractions": self.p_k.detach(),
            "nu_intensity": self.nu_intensity.detach(),
            "c_mz": self.log_c_mz.exp().detach(),
        }

    # -- generative simulation -----------------------------------------

    @torch.no_grad()
    def sample(
        self,
        rt: torch.Tensor,
        iit: torch.Tensor,
        *,
        floor_ions: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> CounterObservation:
        """Forward-simulate an observation on this progenitor's grid.

        Count-based generative (the dimensionally-correct model): per channel
        the accumulated count is `Poisson(λ)`, `λ = N(t)·f_z·p_k·τ` (Poisson
        thinning = Poisson-total + multinomial-isotope-split); intensity is
        `α(z)·count/τ`; the m/z error is `center + StudentT(ν_mz)·√(c_mz/N^α_mz)`.
        Channels with `count < floor_ions` are unobserved (mask False)."""
        rt = rt.to(self._dtype)
        iit = iit.to(self._dtype)
        n_flux = self.peak.forward(rt)  # (S,)
        f_z = self.charge_fractions()
        f_ch = f_z.index_select(0, self.charge_index_of_channel)
        p_ch = self.p_k.index_select(0, self.channel_isotope)
        gain_ch = self.calibration.gain(self.channel_z.to(self._dtype))
        lam = n_flux[:, None] * (f_ch * p_ch)[None, :] * iit[:, None]  # (S, C)
        counts = torch.poisson(lam.clamp(min=0.0), generator=generator)
        intensity = gain_ch[None, :] * counts / iit[:, None]
        mask = counts >= floor_ions

        center = self.calibration.mz_center_ppm(
            self.channel_mz, self.channel_z.to(self._dtype), self.channel_isotope
        )  # (C,)
        c_mz_ch = self.c_mz_per_channel()
        scale = (
            c_mz_ch[None, :] / counts.clamp(min=1.0) ** self.calibration.alpha_mz
        ).clamp(min=1e-12).sqrt()
        nu = self.calibration.nu_mz
        student = torch.distributions.StudentT(df=float(nu))
        noise = student.sample(counts.shape).to(self._dtype)
        mz_error = center[None, :] + scale * noise

        zero = torch.zeros_like(intensity)
        return CounterObservation(
            rt=rt,
            iit=iit,
            intensity=torch.where(mask, intensity, zero),
            mz_error=torch.where(mask, mz_error, zero),
            mask=mask,
            channel_z=self.channel_z,
            channel_isotope=self.channel_isotope,
            channel_mz=self.channel_mz,
        )
