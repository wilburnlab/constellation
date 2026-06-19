"""Step-6 foundation: the theoretical precursor-candidate index (PR-D). See
docs/plans/counter-real-data-deconstruction.md §9."""

from __future__ import annotations

import torch

from constellation.core.sequence.proforma import Peptidoform
from constellation.massspec.counter import (
    GlobalCalibration,
    Progenitor,
    TheoreticalCandidateIndex,
)

_PEP_A = Peptidoform(sequence="PEPTIDEKR")
_PEP_B = Peptidoform(sequence="SAMPLERMKL")


def _index(n_isotopes: int = 3) -> TheoreticalCandidateIndex:
    return TheoreticalCandidateIndex.from_peptides(
        [(0, _PEP_A, [2, 3]), (1, _PEP_B, [2])], n_isotopes=n_isotopes
    )


def test_index_built_sorted_and_sized() -> None:
    idx = _index(n_isotopes=3)
    # target 0: 2 charges × 3 isotopes = 6; target 1: 1 charge × 3 = 3
    assert len(idx) == 9
    assert torch.all(idx.mz[1:] >= idx.mz[:-1])  # mass-sorted
    assert set(idx.target_id.tolist()) == {0, 1}


def test_candidates_query_self_and_tolerance() -> None:
    idx = _index()
    mz0 = float(idx.mz[0])
    c = idx.candidates(mz0, tolerance_ppm=5.0)
    assert len(c) >= 1
    assert int(idx.target_id[0]) in c.target_id.tolist()
    # everything returned is within tolerance of the query
    assert torch.all((c.mz - mz0).abs() / mz0 * 1e6 <= 5.0 + 1e-6)
    # a far-off query resolves to nothing
    assert len(idx.candidates(1.0, tolerance_ppm=5.0)) == 0


def _keys(c) -> set[tuple[int, int, int]]:
    return set(zip(c.target_id.tolist(), c.charge.tolist(), c.isotope.tolist()))


def test_widening_tolerance_only_grows_the_candidate_set() -> None:
    idx = _index()
    mid = float(idx.mz[len(idx) // 2])
    tight = _keys(idx.candidates(mid, tolerance_ppm=1.0))
    wide = _keys(idx.candidates(mid, tolerance_ppm=500.0))
    assert tight <= wide  # candidates are monotone in tolerance (stability)


def test_index_mz_matches_progenitor_grid() -> None:
    # the index must use the SAME envelope as the fit, so an index m/z equals the
    # channel m/z Progenitor.for_peptide builds (else collision edges would diverge
    # from the grids the panel actually scores).
    cal = GlobalCalibration(n_isotopes=3, charges=[2, 3])
    prog = Progenitor.for_peptide(_PEP_A, [2], cal, n_isotopes=3)
    idx = TheoreticalCandidateIndex.from_peptides([(0, _PEP_A, [2])], n_isotopes=3)
    assert torch.allclose(idx.mz.sort().values, prog.channel_mz.sort().values)
