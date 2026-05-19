"""Parity tests for the vectorized neutral-loss biochem mask.

``_biochem_mask`` in ``massspec.peptide.ions`` replaces the original
per-position Python triple-loop with cumulative-OR tensor operations.
These tests prove the new path is byte-identical to a freshly-inlined
implementation of the legacy semantics across a sweep of peptides,
ion-type subsets, modification configurations, and loss subsets.

The legacy semantics are inlined here (`_legacy_biochem_mask`) rather
than imported from a deleted module so the test suite remains
self-contained as the codebase evolves.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from constellation.core.chem.modifications import UNIMOD
from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.peptide.ions import (
    IonType,
    _biochem_mask,
    _N_SIDE,
    _modref_canonical_id,
)
from constellation.massspec.peptide.neutral_losses import (
    LOSS_REGISTRY,
    loss_applies,
)


def _legacy_biochem_mask(
    *,
    seq: str,
    pos_mod_ids: Sequence[Sequence[str]],
    ion_types: Sequence[IonType],
    loss_ids: Sequence[str],
) -> torch.Tensor:
    """Reference implementation — the original Python triple-loop.

    Mirrors the pre-vectorization body of ``_linear_fragment_ladder``'s
    mask-construction section, kept here so the parity test can compare
    against the exact prior semantics.
    """
    L = len(seq)
    n_pos = L - 1
    n_types = len(ion_types)
    n_losses = 1 + len(loss_ids)
    loss_mask = torch.ones((n_pos, n_types, n_losses), dtype=torch.bool)
    if not loss_ids:
        return loss_mask
    for p in range(n_pos):
        for t_idx, ion_type in enumerate(ion_types):
            if ion_type in _N_SIDE:
                frag_residues = list(seq[: p + 1])
                frag_mods = {
                    i: list(pos_mod_ids[i])
                    for i in range(p + 1)
                    if pos_mod_ids[i]
                }
            else:
                frag_residues = list(seq[p + 1 :])
                offset = p + 1
                frag_mods = {
                    i - offset: list(pos_mod_ids[i])
                    for i in range(p + 1, L)
                    if pos_mod_ids[i]
                }
            for l_idx, lid in enumerate(loss_ids, start=1):
                loss = LOSS_REGISTRY.get(lid)
                loss_mask[p, t_idx, l_idx] = loss_applies(
                    loss,
                    ion_type_name=ion_type.name,
                    fragment_residues=frag_residues,
                    fragment_mods=frag_mods,
                )
    return loss_mask


def _build_pos_mod_ids(modseq: str) -> tuple[str, list[list[str]]]:
    """Parse a ProForma modseq, return ``(sequence, pos_mod_ids)``.

    ``pos_mod_ids[i]`` is the list of canonical mod ids on residue
    ``seq[i]``, matching the layout consumed by ``_biochem_mask`` /
    ``loss_applies``.
    """
    p = parse_proforma(modseq)
    seq = p.sequence
    out: list[list[str]] = [[] for _ in range(len(seq))]
    for pos, taggedmods in p.residue_mods.items():
        for tm in taggedmods:
            if tm.mod is None:
                continue
            cid = _modref_canonical_id(tm.mod, UNIMOD)
            if cid is not None:
                out[pos].append(cid)
    return seq, out


@pytest.mark.parametrize(
    "modseq,loss_ids,ion_types",
    [
        # Plain peptide, all loss types, b/y only
        ("PEPTIDE", ("H2O", "NH3"), (IonType.B, IonType.Y)),
        ("PEPTIDE", ("H2O", "NH3", "HPO3", "H3PO4"), (IonType.B, IonType.Y)),
        # Full six-letter ion-type ladder
        (
            "PEPTIDE",
            ("H2O", "NH3", "HPO3", "H3PO4"),
            tuple(IonType),
        ),
        # No mods → mod-rule losses should never fire
        ("AAAAAAA", ("HPO3", "H3PO4"), (IonType.B, IonType.Y)),
        # Phospho on S → H3PO4 should be biochem-licensed on Y-side of P
        ("AAS[Phospho]AK", ("H2O", "H3PO4", "HPO3"), (IonType.B, IonType.Y)),
        # Phospho on Y instead of S/T → HPO3 (no residue restriction) yes;
        # H3PO4 (requires S/T) no.
        ("AAY[Phospho]AAK", ("H3PO4", "HPO3"), (IonType.B, IonType.Y)),
        # Multi-mod: Oxidation on M + Phospho on S
        (
            "AM[Oxidation]ES[Phospho]K",
            ("H2O", "NH3", "H3PO4"),
            (IonType.B, IonType.Y),
        ),
        # Empty losses → mask is all-True
        ("PEPTIDE", (), (IonType.B, IonType.Y)),
        # Longer peptide, full ion types
        (
            "AAAAAAAAQMHAK",
            ("H2O", "NH3"),
            tuple(IonType),
        ),
        # Min-length peptide
        ("AS", ("H2O",), (IonType.B, IonType.Y)),
        # Peptide where E/T/D/S only at one end — N-side b should reach
        # threshold later than C-side y.
        ("AAAATGGG", ("H2O",), (IonType.B, IonType.Y)),
    ],
)
def test_biochem_mask_matches_legacy(
    modseq: str,
    loss_ids: tuple[str, ...],
    ion_types: tuple[IonType, ...],
) -> None:
    seq, pos_mod_ids = _build_pos_mod_ids(modseq)
    n_pos = len(seq) - 1
    n_types = len(ion_types)
    n_losses = 1 + len(loss_ids)
    type_is_n = torch.tensor(
        [t in _N_SIDE for t in ion_types], dtype=torch.bool
    )

    vectorized = _biochem_mask(
        seq=seq,
        pos_mod_ids=pos_mod_ids,
        ion_types=ion_types,
        loss_ids=loss_ids,
        n_pos=n_pos,
        n_types=n_types,
        n_losses=n_losses,
        type_is_n=type_is_n,
    )
    legacy = _legacy_biochem_mask(
        seq=seq,
        pos_mod_ids=pos_mod_ids,
        ion_types=ion_types,
        loss_ids=loss_ids,
    )
    assert torch.equal(vectorized, legacy), (
        f"vectorized biochem mask diverged from legacy for "
        f"{modseq=} {loss_ids=} {ion_types=}\n"
        f"vectorized=\n{vectorized}\nlegacy=\n{legacy}"
    )


def test_biochem_mask_shape_and_no_loss_slot() -> None:
    """Slot 0 of the loss axis is the no-loss baseline — always True."""
    seq, pos_mod_ids = _build_pos_mod_ids("PEPTIDE")
    ion_types = (IonType.B, IonType.Y)
    loss_ids = ("H2O", "NH3")
    type_is_n = torch.tensor([t in _N_SIDE for t in ion_types], dtype=torch.bool)
    mask = _biochem_mask(
        seq=seq,
        pos_mod_ids=pos_mod_ids,
        ion_types=ion_types,
        loss_ids=loss_ids,
        n_pos=len(seq) - 1,
        n_types=len(ion_types),
        n_losses=1 + len(loss_ids),
        type_is_n=type_is_n,
    )
    assert mask.shape == (len(seq) - 1, len(ion_types), 1 + len(loss_ids))
    assert mask[:, :, 0].all().item()
