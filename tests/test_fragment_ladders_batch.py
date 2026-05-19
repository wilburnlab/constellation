"""Parity tests for the batched ``fragment_ladders_batch`` API.

``fragment_ladders_batch`` groups peptidoforms by sequence length and
runs the inner tensor math in batched form. These tests prove that for
any peptidoform the batched path produces a ``FragmentIonTable``
row-equal to the per-spectrum ``fragment_ladder`` output (same rows,
same order, same m/z values to bit-equality given identical inputs).
"""

from __future__ import annotations

import pytest

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.peptide.ions import (
    IonType,
    fragment_ladder,
    fragment_ladders_batch,
)


_MODSEQS = [
    "PEPTIDE/2",
    "AAAAAAAAQMHAK/3",
    "PEPTIDE/3",  # same length as #0 — exercises batched grouping
    "AAAAAA/2",
    "AAAAAA/3",  # same length, same isotopes as #3
    "AAS[Phospho]AK/2",
    "AAY[Phospho]AAK/2",
    "AM[Oxidation]ES[Phospho]K/2",
    "AAAATGGG/2",
    "AC[Carbamidomethyl]TS[Phospho]MK/2",
]


@pytest.mark.parametrize(
    "ion_types,max_fragment_charge,neutral_losses",
    [
        ((IonType.B, IonType.Y), 1, None),
        ((IonType.B, IonType.Y), 2, ("H2O", "NH3")),
        (tuple(IonType), 2, ("H2O", "NH3", "HPO3", "H3PO4")),
    ],
)
def test_batch_matches_per_spectrum(
    ion_types: tuple[IonType, ...],
    max_fragment_charge: int,
    neutral_losses: tuple[str, ...] | None,
) -> None:
    peps = [parse_proforma(s) for s in _MODSEQS]
    batched_tables = fragment_ladders_batch(
        peps,
        ion_types=ion_types,
        max_fragment_charge=max_fragment_charge,
        neutral_losses=list(neutral_losses) if neutral_losses else None,
    )
    assert len(batched_tables) == len(peps)
    for i, p in enumerate(peps):
        single_table, _ = fragment_ladder(
            p,
            ion_types=ion_types,
            max_fragment_charge=max_fragment_charge,
            neutral_losses=list(neutral_losses) if neutral_losses else None,
            return_tensor=False,
        )
        assert batched_tables[i].equals(single_table), (
            f"batched/per-spectrum mismatch for {_MODSEQS[i]!r}\n"
            f"batched:\n{batched_tables[i].to_pylist()[:3]}\n"
            f"per-spectrum:\n{single_table.to_pylist()[:3]}"
        )


def test_batch_peptide_idx_threading() -> None:
    peps = [parse_proforma(s) for s in ["PEPTIDE/2", "AAAAAA/2"]]
    tables = fragment_ladders_batch(
        peps,
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=1,
        peptide_idxs=[7, 11],
    )
    assert all(
        v == 7 for v in tables[0].column("peptide_idx").to_pylist()
    )
    assert all(
        v == 11 for v in tables[1].column("peptide_idx").to_pylist()
    )


def test_batch_empty_input() -> None:
    out = fragment_ladders_batch(
        [], ion_types=(IonType.B, IonType.Y), max_fragment_charge=1
    )
    assert out == []
