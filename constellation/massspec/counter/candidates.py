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

__all__ = [
    "CandidateEntry",
    "TheoreticalCandidateIndex",
    "channel_overlap_components",
    "refine_components_by_rt",
    "restrict_to_reference_star",
]


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


def channel_overlap_components(
    index: TheoreticalCandidateIndex, *, collide_ppm: float = 20.0
) -> list[frozenset[int]]:
    """Partition the index's targets into channel-overlap **connected components**:
    two targets share a component when ANY of their theoretical channels fall within
    `collide_ppm` of each other (transitively). Each component is the unit a panel
    co-fits so a shared peak is soft-attributed once across the blend (cross-panel
    reconciliation) instead of double-claimed by independent per-target fits.

    Every target appears in exactly one component; a target that collides with nobody
    is its own size-1 component (so the caller can keep singletons on the
    embarrassingly-parallel path). Returned sorted by each component's smallest
    `target_id` for determinism.

    This is the keep-and-attribute inverse of `search.collision`'s drop semantics
    (which keeps one peptide per cluster); here every co-isobaric candidate is
    retained. `collide_ppm` is static; a calibration-aware widening (loose-then-tight
    as step 7 refines the offset) is a future opt-in, not wired here."""
    # Mass-sorted sweep → collision edges between DISTINCT targets. Sort defensively
    # (from_peptides already sorts, but a hand-built index may not).
    order = torch.argsort(index.mz)
    mz = index.mz[order].numpy()
    tid = index.target_id[order].numpy()
    n = int(mz.shape[0])

    parent: dict[int, int] = {int(t): int(t) for t in tid.tolist()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        hi = mz[i] * (1.0 + collide_ppm * 1e-6)
        j = i + 1
        while j < n and mz[j] <= hi:
            if tid[j] != tid[i]:
                union(int(tid[i]), int(tid[j]))
            j += 1

    comps: dict[int, set[int]] = {}
    for t in parent:
        comps.setdefault(find(t), set()).add(t)
    return sorted((frozenset(c) for c in comps.values()), key=min)


def refine_components_by_rt(
    components: list[frozenset[int]],
    rt_centers: dict[int, float],
    *,
    rt_overlap_s: float,
) -> list[frozenset[int]]:
    """Split each m/z-overlap component into co-**eluting** sub-clusters whose
    `rt_center` **span ≤ rt_overlap_s** (a greedy sweep of RT-sorted members, NOT
    transitive chaining — a chain of pairwise-close members could otherwise spread a
    unit past a single co-fit window, leaving a far member with no signal in the
    reference-grid obs). The units are the panels that actually get co-fit.

    This is an **efficiency** refinement, not a correctness one for the members it
    keeps together: the additive flux + N(t)-weighted soft attribution + per-member
    μ-anchoring separate RT-different members within one window fine — but co-fitting
    m/z-colliding peptides that elute far apart just stretches a panel over dead RT
    span (they do not actually interfere). A member with no `rt_center` (key absent)
    can't be overlap-assessed and is kept as its own singleton. Returned sorted by
    min `target_id`."""
    units: list[frozenset[int]] = []
    for comp in components:
        with_rt = sorted((rt_centers[m], m) for m in comp if m in rt_centers)
        units += [frozenset({m}) for m in comp if m not in rt_centers]  # rt-less → singletons
        if not with_rt:
            continue
        group = [with_rt[0][1]]
        start = with_rt[0][0]
        for rtc, m in with_rt[1:]:
            if rtc - start <= rt_overlap_s:  # span from the group's earliest member
                group.append(m)
            else:
                units.append(frozenset(group))
                group, start = [m], rtc
        units.append(frozenset(group))
    return sorted(units, key=min)


def _channels_overlap(a_mz: list[float], b_mz: list[float], collide_ppm: float) -> bool:
    """True if any channel of `a` is within `collide_ppm` of any channel of `b`
    (ppm relative to the smaller m/z, matching `channel_overlap_components`' edge)."""
    for x in a_mz:
        for y in b_mz:
            if abs(x - y) / min(x, y) * 1e6 <= collide_ppm:
                return True
    return False


def restrict_to_reference_star(
    units: list[frozenset[int]],
    index: TheoreticalCandidateIndex,
    *,
    collide_ppm: float,
) -> list[frozenset[int]]:
    """Re-cluster each (transitive) unit into **reference stars**: a star is a
    reference + the members whose channels DIRECTLY overlap the reference's (within
    `collide_ppm`). Greedy by min `target_id` — emit the first member's star, remove
    it, repeat on the rest.

    This is the correctness fix for the reference-grid co-fit (`estimate_component`):
    the obs is built on the reference member's grid (and its pre-extracted trace), so
    a member is only scorable if its channels lie within the reference's extraction
    tolerance — i.e. it DIRECTLY overlaps the reference, not transitively (A–B, B–C
    with A–C apart would otherwise leave C scored against A's grid where its peaks
    were never extracted → silent ~0 N). A transitively-but-not-directly connected
    member falls into its own star (a singleton if it overlaps no kept reference).
    `collide_ppm` should be ≤ the upstream XIC extraction tolerance, else a member
    within `collide_ppm` of the reference but beyond the extraction tolerance still
    has no extracted signal on the reference grid (the union grid is the full fix)."""
    mz_by_tid: dict[int, list[float]] = {}
    for tid, mz in zip(index.target_id.tolist(), index.mz.tolist()):
        mz_by_tid.setdefault(int(tid), []).append(float(mz))

    out: list[frozenset[int]] = []
    for unit in units:
        remaining = sorted(unit)
        while remaining:
            ref = remaining[0]
            ref_mz = mz_by_tid.get(ref, [])
            star = [ref]
            rest: list[int] = []
            for m in remaining[1:]:
                if _channels_overlap(mz_by_tid.get(m, []), ref_mz, collide_ppm):
                    star.append(m)
                else:
                    rest.append(m)
            out.append(frozenset(star))
            remaining = rest
    return sorted(out, key=min)
