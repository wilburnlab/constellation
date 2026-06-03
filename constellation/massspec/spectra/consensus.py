"""Consensus / aggregated MS2 spectrum builder.

Aligns replicate fragment spectra of one precursor to a **fixed
per-precursor channel basis** (the theoretical b/y… ladder) and aggregates
them, so every replicate lives in the same K-dimensional coordinate system.
This is the shared substrate for the Part-I MS2 experiments:

  * exp01 (spectral scoring) compares replicate-vs-replicate and
    replicate-vs-consensus pairs on a common basis;
  * exp02 (multinomial) reads the retained ``per_replicate`` matrix to fit
    the ``Var[p̂_k] = p_k(1−p_k)/N`` mean–variance law.

The builder makes **no blind-inlier assumption**: it keeps the full
``(R, K)`` replicate matrix and reports a per-replicate *deviance-from-bulk*
(``2·N·KL(p̂_r ‖ p̄)``, the multinomial statistic) so a screening / robust
refinement pass can flag replicates that do not look like multinomial draws
from the bulk. First-scope robustness comes from the ``median`` aggregate;
the iterative refinement is a separate (deferred) routine.

All chemistry is reused from ``massspec.peptide`` (``fragment_ladder`` for
the basis, ``match_mz`` for the alignment); the deviance is reused from the
sibling ``similarity`` module. No model state lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import ProFormaResult
from constellation.massspec.peptide.ions import IonType, fragment_ladder
from constellation.massspec.peptide.match import match_mz
from constellation.massspec.spectra.similarity import multinomial_deviance

__all__ = [
    "FragmentBasis",
    "ConsensusSpectrum",
    "fragment_basis",
    "align_to_basis",
    "build_consensus",
]

Aggregate = Literal["sum", "median"]


@dataclass(frozen=True, slots=True)
class FragmentBasis:
    """The fixed K-channel coordinate system for one precursor — one entry
    per theoretical fragment ion, in ``fragment_ladder`` row order.

    Built once per ``(modified_sequence, charge)`` and reused for every
    replicate so the aligned vectors are directly comparable.
    """

    ion_type: torch.Tensor  # (K,) int8 — IonType codes
    position: torch.Tensor  # (K,) int32
    charge: torch.Tensor  # (K,) int32 — fragment charge
    loss_id: tuple[str | None, ...]  # (K,) neutral-loss id (None = baseline)
    mz_theoretical: torch.Tensor  # (K,) float64

    @property
    def K(self) -> int:
        return int(self.mz_theoretical.shape[0])


@dataclass(frozen=True, slots=True)
class ConsensusSpectrum:
    """An aggregated spectrum on a ``FragmentBasis`` plus the substrate the
    downstream multinomial analysis needs."""

    basis: FragmentBasis
    intensity: torch.Tensor  # (K,) aggregated consensus (normalized iff requested)
    dispersion: torch.Tensor  # (K,) across-replicate std of per-replicate proportions
    n_replicates: int
    per_replicate: torch.Tensor  # (R, K) aligned replicate intensities
    deviance_from_bulk: torch.Tensor  # (R,) 2·N·KL(p̂_r ‖ p̄) — the outlier signal

    @property
    def proportions(self) -> torch.Tensor:
        """The consensus as a simplex (L1-normalized intensity)."""
        total = self.intensity.sum().clamp(min=1e-12)
        return self.intensity / total


def fragment_basis(
    peptidoform: ProFormaResult,
    *,
    ion_types: Sequence[IonType] = (IonType.B, IonType.Y),
    max_fragment_charge: int = 2,
    neutral_losses: Sequence[str] | None = None,
    vocab: ModVocab = UNIMOD,
) -> FragmentBasis:
    """Build the fixed channel basis for ``peptidoform`` from its theoretical
    fragment ladder. Channel order is ``fragment_ladder`` row order (stable
    for a given peptidoform + ion-type/charge/loss configuration)."""
    table, _ = fragment_ladder(
        peptidoform,
        ion_types=ion_types,
        max_fragment_charge=max_fragment_charge,
        neutral_losses=neutral_losses,
        return_tensor=False,
        vocab=vocab,
    )
    return FragmentBasis(
        ion_type=torch.tensor(table.column("ion_type").to_pylist(), dtype=torch.int8),
        position=torch.tensor(table.column("position").to_pylist(), dtype=torch.int32),
        charge=torch.tensor(table.column("charge").to_pylist(), dtype=torch.int32),
        loss_id=tuple(table.column("loss_id").to_pylist()),
        mz_theoretical=torch.tensor(
            table.column("mz_theoretical").to_pylist(), dtype=torch.float64
        ),
    )


def align_to_basis(
    obs_mz: torch.Tensor,
    obs_intensity: torch.Tensor,
    basis: FragmentBasis,
    *,
    tolerance: float = 20.0,
    tolerance_unit: Literal["ppm", "Da"] = "ppm",
    return_mz_error: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Project an observed peak list onto ``basis`` → a length-K intensity
    vector (``0.0`` where a basis channel is unmatched).

    Each observed peak is assigned to its closest basis channel within
    tolerance (``match_mz``); observed intensity is scatter-**added** into
    that channel (peaks matching no channel are dropped — the b/y-only
    conditioning). Use analyzer-appropriate tolerance (FTMS ~20 ppm; ITMS
    ~0.5 Da).

    With ``return_mz_error=True`` also returns the per-channel
    **intensity-weighted signed m/z error** in ppm (``match_mz``'s
    ``error_ppm``; ``NaN`` where a channel is unmatched). The matching
    already computes this error — it is simply retained here. It is the
    orthogonal interference discriminant: a genuine fragment sits near
    0 ppm, whereas a contaminant peak that merely falls inside the
    tolerance window sits systematically off-center."""
    obs_mz = torch.as_tensor(obs_mz, dtype=torch.float64).reshape(-1)
    obs_intensity = torch.as_tensor(obs_intensity, dtype=torch.float64).reshape(-1)
    out = torch.zeros(basis.K, dtype=torch.float64)
    werr = torch.zeros(basis.K, dtype=torch.float64) if return_mz_error else None
    if obs_mz.numel() == 0 or basis.K == 0:
        if return_mz_error:
            return out, torch.full((basis.K,), float("nan"), dtype=torch.float64)
        return out
    matches = match_mz(
        obs_mz,
        basis.mz_theoretical,
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
        query_intensity=obs_intensity,
    )
    for m in matches:
        w = obs_intensity[m.query_idx]
        out[m.ref_idx] += w
        if return_mz_error:
            werr[m.ref_idx] += w * m.error_ppm
    if return_mz_error:
        err = torch.where(out > 0, werr / out, torch.full_like(out, float("nan")))
        return out, err
    return out


def build_consensus(
    spectra: Sequence[tuple[torch.Tensor, torch.Tensor]],
    basis: FragmentBasis,
    *,
    aggregate: Aggregate = "sum",
    normalize: bool = False,
    tolerance: float = 20.0,
    tolerance_unit: Literal["ppm", "Da"] = "ppm",
    deviance_pseudocount: float = 0.0,
) -> ConsensusSpectrum:
    """Align every replicate ``(obs_mz, obs_intensity)`` to ``basis`` and
    aggregate per channel.

    ``aggregate``:
      * ``"sum"`` — pooled count-space spectrum (preserves the pooled N; the
        high-N estimate the MSP-provenance forensic wants).
      * ``"median"`` — robust central tendency, resistant to a minority of
        contaminant replicates.
      (``mean`` is omitted: under ``normalize`` it is ``sum/R``.)

    ``normalize`` L1-normalizes the aggregated vector to a simplex. The
    ``proportions`` property gives the simplex regardless of this flag.

    Returns the aggregate plus the retained ``per_replicate`` ``(R, K)``
    matrix and the per-replicate ``deviance_from_bulk`` ``(R,)`` (the
    multinomial outlier signal vs the bulk consensus proportions). No
    replicate is screened here — that is a deliberate downstream choice."""
    if len(spectra) == 0:
        raise ValueError("build_consensus needs at least one spectrum")

    per_replicate = torch.stack(
        [
            align_to_basis(
                mz, inten, basis, tolerance=tolerance, tolerance_unit=tolerance_unit
            )
            for mz, inten in spectra
        ]
    )  # (R, K)
    r = per_replicate.shape[0]

    if aggregate == "sum":
        intensity = per_replicate.sum(dim=0)
    elif aggregate == "median":
        intensity = torch.quantile(per_replicate, 0.5, dim=0)
    else:
        raise ValueError(f"aggregate must be 'sum' or 'median'; got {aggregate!r}")

    # Across-replicate dispersion of the per-replicate *proportions* (the
    # scale-free per-channel spread; population std so R=1 → 0, NaN-free).
    rep_totals = per_replicate.sum(dim=1, keepdim=True).clamp(min=1e-12)
    rep_props = per_replicate / rep_totals  # (R, K)
    dispersion = rep_props.std(dim=0, unbiased=False)

    # Per-replicate deviance from the bulk consensus proportions — the
    # outlier-from-bulk signal a screening pass would threshold. Default
    # ``deviance_pseudocount=0`` so identical replicates score exactly 0
    # (the eps-clamp inside ``multinomial_deviance`` keeps it NaN-free even
    # when a replicate carries signal on a channel the bulk leaves empty).
    bulk_prop = intensity / intensity.sum().clamp(min=1e-12)
    deviance_from_bulk = multinomial_deviance(
        per_replicate,
        bulk_prop.expand_as(per_replicate),
        pseudocount=deviance_pseudocount,
    )

    if normalize:
        intensity = intensity / intensity.sum().clamp(min=1e-12)

    return ConsensusSpectrum(
        basis=basis,
        intensity=intensity,
        dispersion=dispersion,
        n_replicates=r,
        per_replicate=per_replicate,
        deviance_from_bulk=deviance_from_bulk,
    )
