"""Backbone fragment-ion enumeration and m/z calculation.

Six canonical ion types from amide-bond / Cα-N bond cleavage:

    a, b, c     N-terminal-side fragments  (a = b - CO; c = b + NH3)
    x, y, z     C-terminal-side fragments  (x = y + CO; z = y - NH3)

Per-ion offsets are computed from `Composition` arithmetic — no
hardcoded numeric constants — so adding ion types or shifting to a
different convention is just a matter of writing the right Composition.

The z-ion offset corresponds to the radical (z•) form. For z+1 (z•+H),
add a hydrogen mass at the call site.

Ladder generation (`fragment_ladder`) returns both an Arrow table
(canonical, storage / round-trip / inspection) and a tensor of shape
`(n_pos, n_ion_types, n_charges, n_losses+1)` (vmap-compatible hot path
for scoring). Loss-validity rules from `peptide.neutral_losses` mask
biochemically invalid ions with NaN in the tensor and omit them from
the Arrow table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum

import pyarrow as pa
import torch

from constellation.core.chem.composition import Composition
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.alphabets import AA, requires_canonical
from constellation.core.sequence.ops import validate
from constellation.massspec.peptide.mz import PROTON_MASS
from constellation.massspec.peptide.neutral_losses import LOSS_REGISTRY, loss_applies
from constellation.massspec.schemas import FRAGMENT_ION_TABLE

# ──────────────────────────────────────────────────────────────────────
# IonType enum + composition offsets
# ──────────────────────────────────────────────────────────────────────


class IonType(IntEnum):
    """Backbone fragment ion types.

    Values are stable integer codes used by the FragmentIonTable schema
    and the ladder tensor. Names match the canonical Roepstorff-Fohlman
    letters and are used as keys when checking `NeutralLoss.applies_to_ion_types`.
    """

    A = 0
    B = 1
    C = 2
    X = 3
    Y = 4
    Z = 5


_N_SIDE = frozenset({IonType.A, IonType.B, IonType.C})
_C_SIDE = frozenset({IonType.X, IonType.Y, IonType.Z})


# Composition deltas added to the residue-sum mass for each ion type.
# Per the standard MS conventions:
#   a = b - CO          (Ø - CO)
#   b = sum_residues    (Ø)
#   c = b + NH3         (+NH3)
#   x = y + CO          (+H2O + CO)
#   y = sum_residues + H2O   (+H2O)
#   z• = y - NH3        (+H2O - NH3)
# Compositions may carry negative counts (Composition is "valid for deltas").
_CO: Composition = Composition.from_dict({"C": 1, "O": 1})
_H2O: Composition = Composition.from_dict({"H": 2, "O": 1})
_NH3: Composition = Composition.from_dict({"N": 1, "H": 3})

ION_OFFSETS: dict[IonType, Composition] = {
    IonType.A: Composition.zeros() - _CO,
    IonType.B: Composition.zeros(),
    IonType.C: _NH3,
    IonType.X: _H2O + _CO,
    IonType.Y: _H2O,
    IonType.Z: _H2O - _NH3,
}

ION_OFFSET_MASSES: dict[IonType, float] = {
    t: comp.mass for t, comp in ION_OFFSETS.items()
}


# ──────────────────────────────────────────────────────────────────────
# FragmentIon record
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FragmentIon:
    """A single fragment-ion record. Mirrors one row of FRAGMENT_ION_TABLE."""

    position: int
    ion_type: IonType
    charge: int
    loss_id: str | None
    mz: float


# ──────────────────────────────────────────────────────────────────────
# Scalar / tensor m/z helper
# ──────────────────────────────────────────────────────────────────────


def fragment_mz(
    residue_sum_mass: float | torch.Tensor,
    *,
    ion_type: IonType,
    charge: int = 1,
) -> float | torch.Tensor:
    """Single-ion m/z from the fragment's *residue-sum mass*.

    `residue_sum_mass` already includes any in-fragment modification
    deltas; this function adds the ion-type offset and the charge term.
    Tensor inputs broadcast on the leading dims.
    """
    if charge <= 0:
        raise ValueError(f"charge must be positive; got {charge}")
    offset = ION_OFFSET_MASSES[ion_type]
    if isinstance(residue_sum_mass, torch.Tensor):
        return (residue_sum_mass + offset + charge * PROTON_MASS) / charge
    return (residue_sum_mass + offset + charge * PROTON_MASS) / charge


# ──────────────────────────────────────────────────────────────────────
# Helpers for modification handling
# ──────────────────────────────────────────────────────────────────────


def _iter_mod_keys(value: object):
    """Yield mod keys from a single value or a list-of-values
    (`parse_modified_sequence` may return either)."""
    if isinstance(value, list):
        yield from value
    else:
        yield value


def _mod_delta_mass(value: object, vocab: ModVocab) -> float:
    """Sum canonical delta masses across one or more mods at a position.

    Strings resolve through the vocabulary (UNIMOD ids or aliases). Floats
    are interpreted as direct mass deltas in Da (mass-notation form from
    `parse_modified_sequence`). Anything else raises.
    """
    total = 0.0
    for k in _iter_mod_keys(value):
        if isinstance(k, str):
            total += vocab.get(k).delta_mass
        elif isinstance(k, (int, float)):
            total += float(k)
        else:
            raise TypeError(
                f"unsupported modification value: {k!r} ({type(k).__name__})"
            )
    return total


# ──────────────────────────────────────────────────────────────────────
# Fragment-ladder driver
# ──────────────────────────────────────────────────────────────────────


@requires_canonical
def fragment_ladder(
    seq: str,
    *,
    modifications: Mapping[int, object] | None = None,
    ion_types: Sequence[IonType] = (IonType.B, IonType.Y),
    max_fragment_charge: int = 1,
    neutral_losses: Sequence[str] | None = None,
    return_tensor: bool = True,
    vocab: ModVocab = UNIMOD,
    peptide_idx: int | None = None,
    include_peptide_seq: bool = True,
) -> tuple[pa.Table, torch.Tensor | None]:
    """Generate the fragment-ion ladder for a (modified) peptide.

    Returns ``(table, tensor)``:

      * ``table`` — Arrow `FragmentIonTable` (one row per biochemically valid ion).
      * ``tensor`` — float64, shape ``(n_pos, n_ion_types, n_charges, n_losses+1)``.
        Slot 0 along the loss axis is the no-loss baseline; subsequent slots
        follow the order of `neutral_losses`. Invalid ions (loss not licensed
        by fragment chemistry) are NaN-masked. ``None`` if ``return_tensor=False``.

    Modifications:
      * `modifications` is keyed by 0-indexed residue position in `seq`.
      * Heavy-isotope mods (those carrying `mass_override`) are handled
        automatically: `Modification.delta_mass` returns the canonical mass.

    Conventions:
      * Cleavage position `p` (0..L-2) splits between residue `p` and `p+1`.
      * N-side fragments (a/b/c) at position `p` cover residues ``[0..p]``.
      * C-side fragments (x/y/z) at position `p` cover residues ``[p+1..L-1]``.
      * Position-0 mods always attach to the N-side fragment of every cleavage;
        position-(L-1) mods always attach to the C-side fragment.
    """
    validate(seq, AA)
    L = len(seq)
    if L < 2:
        raise ValueError(
            f"fragment_ladder requires sequence length ≥ 2; got {L}"
        )
    if max_fragment_charge < 1:
        raise ValueError(
            f"max_fragment_charge must be ≥ 1; got {max_fragment_charge}"
        )
    if not ion_types:
        raise ValueError("ion_types must be non-empty")

    n_pos = L - 1
    n_types = len(ion_types)
    n_charges = max_fragment_charge

    loss_ids: tuple[str, ...] = tuple(neutral_losses) if neutral_losses else ()
    n_losses = 1 + len(loss_ids)  # slot 0 is the no-loss baseline

    mods: dict[int, object] = dict(modifications) if modifications else {}
    for k in mods:
        if not 0 <= k < L:
            raise IndexError(
                f"modification index {k} out of range for length-{L} peptide"
            )

    # ── per-position residue + mod mass ─────────────────────────────
    assert AA.compositions is not None
    residue_masses = [AA.compositions[r].mass for r in seq]
    mod_masses_per_pos = [0.0] * L
    for pos, value in mods.items():
        mod_masses_per_pos[pos] = _mod_delta_mass(value, vocab)
    position_masses_t = torch.tensor(
        [r + m for r, m in zip(residue_masses, mod_masses_per_pos)],
        dtype=torch.float64,
    )

    # ── cumulative sums for N/C side fragment masses ────────────────
    n_cumulative = torch.cumsum(position_masses_t, dim=0)  # (L,) inclusive
    total_residue_sum = n_cumulative[-1]
    # n_side[p] = sum residues [0..p]; valid p in 0..L-2
    n_side = n_cumulative[: L - 1]  # (L-1,)
    # c_side[p] = sum residues [p+1..L-1]; valid p in 0..L-2
    c_side = total_residue_sum - n_side  # (L-1,)

    # ── ion-type offsets and side selectors ─────────────────────────
    type_offsets = torch.tensor(
        [ION_OFFSET_MASSES[t] for t in ion_types], dtype=torch.float64
    )
    type_is_n = torch.tensor(
        [t in _N_SIDE for t in ion_types], dtype=torch.bool
    )
    # side_masses[p, t] = n_side[p] if type_is_n[t] else c_side[p]
    side_masses = torch.where(
        type_is_n[None, :], n_side[:, None], c_side[:, None]
    )  # (n_pos, n_types)
    neutral_no_loss = side_masses + type_offsets[None, :]  # (n_pos, n_types)

    # ── loss-delta table; slot 0 is baseline (0 Da) ─────────────────
    loss_deltas = torch.zeros(n_losses, dtype=torch.float64)
    for i, lid in enumerate(loss_ids, start=1):
        loss_deltas[i] = LOSS_REGISTRY.get(lid).delta_mass

    # neutral mass shape: (n_pos, n_types, n_losses)
    neutral_with_loss = (
        neutral_no_loss[:, :, None] - loss_deltas[None, None, :]
    )
    # m/z shape: (n_pos, n_types, n_charges, n_losses)
    charges_t = torch.arange(1, n_charges + 1, dtype=torch.float64)
    mz_tensor = (
        neutral_with_loss[:, :, None, :]
        + charges_t[None, None, :, None] * PROTON_MASS
    ) / charges_t[None, None, :, None]

    # ── loss-validity mask: per (pos, type, loss_id) ────────────────
    # Baseline (slot 0) is always valid; only the actual-loss slots
    # 1..n_losses-1 need biochemistry checks.
    loss_mask = torch.ones((n_pos, n_types, n_losses), dtype=torch.bool)
    if loss_ids:
        for p in range(n_pos):
            for t_idx, ion_type in enumerate(ion_types):
                if ion_type in _N_SIDE:
                    frag_residues = list(seq[: p + 1])
                    frag_mods = {
                        k: v for k, v in mods.items() if k <= p
                    }
                else:
                    frag_residues = list(seq[p + 1 :])
                    offset = p + 1
                    frag_mods = {
                        k - offset: v for k, v in mods.items() if k > p
                    }
                for l_idx, lid in enumerate(loss_ids, start=1):
                    loss = LOSS_REGISTRY.get(lid)
                    loss_mask[p, t_idx, l_idx] = loss_applies(
                        loss,
                        ion_type_name=ion_type.name,
                        fragment_residues=frag_residues,
                        fragment_mods=frag_mods,
                    )

    # Apply mask: invalid → NaN
    nan = torch.tensor(float("nan"), dtype=torch.float64)
    mz_tensor = torch.where(loss_mask[:, :, None, :], mz_tensor, nan)

    # ── build Arrow table from tensor ───────────────────────────────
    # Iterate finite cells; emit one row per valid ion.
    finite = torch.isfinite(mz_tensor)
    idx_p, idx_t, idx_c, idx_l = torch.where(finite)
    mz_values = mz_tensor[finite].tolist()

    n_rows = idx_p.numel()
    pep_idx_col = (
        [peptide_idx] * n_rows
        if peptide_idx is not None
        else [None] * n_rows
    )
    pep_seq_col = [seq] * n_rows if include_peptide_seq else [None] * n_rows
    positions = idx_p.to(torch.int32).tolist()
    ion_type_codes = [int(ion_types[i]) for i in idx_t.tolist()]
    charges = [c + 1 for c in idx_c.tolist()]  # 0-indexed → 1-indexed
    loss_id_col = [
        None if li == 0 else loss_ids[li - 1] for li in idx_l.tolist()
    ]

    table = pa.table(
        {
            "peptide_idx": pa.array(pep_idx_col, type=pa.int32()),
            "peptide_seq": pa.array(pep_seq_col, type=pa.string()),
            "position": pa.array(positions, type=pa.int32()),
            "ion_type": pa.array(ion_type_codes, type=pa.int8()),
            "charge": pa.array(charges, type=pa.int32()),
            "loss_id": pa.array(loss_id_col, type=pa.string()),
            "mz_theoretical": pa.array(mz_values, type=pa.float64()),
        },
        schema=FRAGMENT_ION_TABLE,
    )

    return table, (mz_tensor if return_tensor else None)


__all__ = [
    "IonType",
    "ION_OFFSETS",
    "ION_OFFSET_MASSES",
    "FragmentIon",
    "fragment_mz",
    "fragment_ladder",
]
