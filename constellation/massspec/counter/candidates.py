"""Theoretical precursor-candidate index — the data-independent half of the
MSFragger/Sage-style lookup, at the PRECURSOR level: a peptide list →
mass-sorted `(target_id, charge, isotope, m/z)`, so an observed m/z resolves to
the candidate progenitors that could have produced it.

The OBSERVED-side mirror is `massspec.quant.peak_index`; composed, the two are a
merge-join in m/z space (the open-search architecture, one level up). This index
is **data-independent given the peptide list** (independent of the raw peaks), so
it — and the channel-overlap components built on it (a follow-up) — are
precomputable once per peptide list. Precursor-scoped today; an MS2-fragment
analogue can join later without re-architecting the lookup.

Linkage stability (the step-7 no-re-extraction invariant): a `candidates(...)`
query takes a `tolerance_ppm` that must exceed the true m/z error + any plausible
global offset, so which candidates an observed peak links to does NOT change when
the global m/z offset is refined — only how they score. Build the index with the
SAME `peptide_envelope(mode, n_isotopes)` the fit uses (`Progenitor.for_peptide`),
so an index m/z matches the channel grid the panel scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from constellation.core.sequence.proforma import Peptidoform
from constellation.massspec.peptide.envelope import EnvelopeMode, peptide_envelope

__all__ = ["CandidateEntry", "TheoreticalCandidateIndex"]


@dataclass(frozen=True)
class CandidateEntry:
    """The candidate `(target_id, charge, isotope, m/z)` rows an observed m/z
    query resolved to — a slice of the index. Empty (length-0 tensors) when the
    query has no candidate within tolerance."""

    target_id: torch.Tensor  # (M,) long
    charge: torch.Tensor  # (M,) long
    isotope: torch.Tensor  # (M,) long
    mz: torch.Tensor  # (M,) theoretical m/z [Th]

    def __len__(self) -> int:
        return int(self.target_id.numel())


@dataclass(frozen=True)
class TheoreticalCandidateIndex:
    """Mass-sorted theoretical precursor-isotope m/z over a peptide list."""

    mz: torch.Tensor  # (N,) sorted theoretical m/z [Th]
    target_id: torch.Tensor  # (N,) long
    charge: torch.Tensor  # (N,) long
    isotope: torch.Tensor  # (N,) long

    @classmethod
    def from_peptides(
        cls,
        entries: Iterable[tuple[int, Peptidoform, Sequence[int]]],
        *,
        n_isotopes: int = 3,
        mode: EnvelopeMode = "binned",
        dtype: torch.dtype = torch.float64,
    ) -> "TheoreticalCandidateIndex":
        """Build from `(target_id, peptidoform, charges)` rows. One index row per
        `(target, charge, isotope)`; uses the SAME `peptide_envelope(mode,
        n_peaks=n_isotopes)` as `Progenitor.for_peptide` so index m/z == channel
        m/z. Sorted by m/z for binary-search queries."""
        tids: list[int] = []
        zs: list[int] = []
        ks: list[int] = []
        mzs: list[float] = []
        for target_id, pep, charges in entries:
            for z in charges:
                mz_z, _inten = peptide_envelope(
                    pep, charge=int(z), n_peaks=n_isotopes, mode=mode
                )
                for k in range(n_isotopes):
                    tids.append(int(target_id))
                    zs.append(int(z))
                    ks.append(k)
                    mzs.append(float(mz_z[k]))
        mz = torch.tensor(mzs, dtype=dtype)
        order = torch.argsort(mz)
        return cls(
            mz=mz[order],
            target_id=torch.tensor(tids, dtype=torch.long)[order],
            charge=torch.tensor(zs, dtype=torch.long)[order],
            isotope=torch.tensor(ks, dtype=torch.long)[order],
        )

    def __len__(self) -> int:
        return int(self.mz.numel())

    def candidates(
        self, mz_query: float, *, tolerance_ppm: float = 20.0
    ) -> CandidateEntry:
        """All index rows within `tolerance_ppm` of `mz_query` — the candidate
        progenitors an observed peak at `mz_query` could be. The window is taken
        relative to `mz_query`; keep `tolerance_ppm` ≫ (true error + plausible
        global offset) so the candidate set is stable under calibration
        refinement (see the module docstring)."""
        if len(self) == 0:
            empty = torch.zeros(0, dtype=torch.long)
            return CandidateEntry(empty, empty, empty, self.mz[:0])
        tol_da = float(mz_query) * float(tolerance_ppm) * 1e-6
        mz_np = self.mz.numpy()
        i0 = int(np.searchsorted(mz_np, float(mz_query) - tol_da, side="left"))
        i1 = int(np.searchsorted(mz_np, float(mz_query) + tol_da, side="right"))
        sl = slice(i0, i1)
        return CandidateEntry(
            target_id=self.target_id[sl],
            charge=self.charge[sl],
            isotope=self.isotope[sl],
            mz=self.mz[sl],
        )
