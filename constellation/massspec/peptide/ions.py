"""Backbone fragment-ion enumeration and m/z calculation.

Six canonical ion types from amide-bond / Cα-N bond cleavage:

    a, b, c     N-terminal-side fragments  (a = b - CO; c = b + NH3)
    x, y, z     C-terminal-side fragments  (x = y + CO; z = y - NH3)

Per-ion offsets are computed from `Composition` arithmetic — no
hardcoded numeric constants — so adding ion types or shifting to a
different convention is just a matter of writing the right Composition.

The z-ion offset corresponds to the radical (z•) form. For z+1 (z•+H),
add a hydrogen mass at the call site.

`fragment_ladder` is a dispatcher: linear peptides flow through
`_linear_fragment_ladder`, while cross-linked, branched, and multi-chain
peptidoforms route to dedicated backends (currently NotImplementedError
stubs that document scope). The dispatcher keeps the public API stable
across XL-MS / branch / multichain support landing in future PRs.

Ladder generation returns both an Arrow table (canonical, storage /
round-trip / inspection) and a tensor of shape
``(n_pos, n_ion_types, n_charges, n_losses+1)`` (vmap-compatible hot
path for scoring). Loss-validity rules from `peptide.neutral_losses`
mask biochemically invalid ions with NaN in the tensor and omit them
from the Arrow table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import pyarrow as pa
import torch

from constellation.core.chem.composition import Composition
from constellation.core.chem.elements import ELEMENTS
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.alphabets import AA, requires_canonical
from constellation.core.sequence.ops import validate
from constellation.core.sequence.proforma import (
    ModRef,
    MultiPeptidoform,
    Peptidoform,
    ProFormaResult,
    _has_branches,
    _has_crosslinks,
    parse_isotope_label,
)
from constellation.core.sequence.protein import (
    _CV_ALIASES,
    _count_fixed_mod_locations,
    _modref_contribution,
    _modref_lookup_key,
)
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
# Helpers — global isotope / fixed-mod expansion / canonical mod ids
# ──────────────────────────────────────────────────────────────────────


def _isotope_shift_for_composition(
    comp: Composition, isotopes: tuple[str, ...]
) -> float:
    """Total mass shift when applying global isotope labels to ``comp``.

    Each label substitutes the natural-abundance monoisotopic mass with
    the requested isotope's exact mass for every atom of that element.
    Stacked labels (``<13C><15N>``) are additive.
    """
    if not isotopes:
        return 0.0
    atoms = comp.atoms
    total = 0.0
    for label in isotopes:
        mass_n, sym = parse_isotope_label(label)
        n_atoms = atoms.get(sym, 0)
        if n_atoms == 0:
            continue
        elem = ELEMENTS[sym]
        total += n_atoms * (elem.isotope_mass(mass_n) - elem.monoisotopic_mass)
    return total


def _modref_canonical_id(modref: ModRef, vocab: ModVocab) -> str | None:
    """Canonical mod id used by ``loss_applies`` triggering checks.

    Returns ``None`` for ModRefs that don't resolve through the vocab
    (formula-only, glycan, mass-delta-only, INFO) — those mods simply
    can't trigger UNIMOD-keyed losses, which is the correct behavior.
    """
    if modref.cv == "INFO":
        return None
    if modref.accession is not None:
        cv = modref.cv or "UNIMOD"
        cv = _CV_ALIASES.get(cv, cv)
        return f"{cv}:{modref.accession}"
    if modref.name is not None:
        try:
            return vocab.get(_modref_lookup_key(modref)).id
        except KeyError:
            return None
    return None


def _iter_fixed_mod_residue_indices(
    seq: str, location: str, p: Peptidoform
) -> list[int]:
    """Residue indices a fixed-mod location rule expands to (excluding
    positions with explicit residue mods, which override per spec §4.6.2).

    Returns an empty list for terminal locations (those go through
    ``n_term_extra`` / ``c_term_extra`` in the linear backend, not
    per-residue arrays).
    """
    if location in ("N-term", "C-term"):
        return []
    if location == "*":
        return [i for i in range(len(seq)) if i not in p.residue_mods]
    if len(location) == 1:
        return [
            i
            for i, r in enumerate(seq)
            if r == location and i not in p.residue_mods
        ]
    raise ValueError(f"unsupported fixed-mod location: {location!r}")


# ──────────────────────────────────────────────────────────────────────
# Fragment-ladder dispatcher + backends
# ──────────────────────────────────────────────────────────────────────


def fragment_ladder(
    peptidoform: ProFormaResult,
    *,
    ion_types: Sequence[IonType] = (IonType.B, IonType.Y),
    max_fragment_charge: int = 1,
    neutral_losses: Sequence[str] | None = None,
    return_tensor: bool = True,
    vocab: ModVocab = UNIMOD,
    peptide_idx: int | None = None,
    include_peptide_seq: bool = True,
) -> tuple[pa.Table, torch.Tensor | None]:
    """Generate the fragment-ion ladder for a peptidoform.

    Routes by topology:

      * ``MultiPeptidoform`` (//-separated chains) → ``_multichain_fragment_ladder``
      * Cross-linked peptidoforms (``#XL`` groups) → ``_crosslink_fragment_ladder``
      * Branched peptidoforms (``#BRANCH`` groups) → ``_branched_fragment_ladder``
      * Linear peptides → ``_linear_fragment_ladder``

    The non-linear backends are NotImplementedError stubs in this release;
    they scaffold the dispatch architecture so the public API is stable
    when XL-MS / branch / multichain support lands.

    Returns ``(table, tensor)`` from the chosen backend:

      * ``table`` — Arrow `FragmentIonTable` (one row per biochemically valid ion).
      * ``tensor`` — float64, shape ``(n_pos, n_ion_types, n_charges, n_losses+1)``.
        Slot 0 along the loss axis is the no-loss baseline; subsequent slots
        follow the order of `neutral_losses`. Invalid ions (loss not licensed
        by fragment chemistry) are NaN-masked. ``None`` if ``return_tensor=False``.
    """
    kwargs = dict(
        ion_types=ion_types,
        max_fragment_charge=max_fragment_charge,
        neutral_losses=neutral_losses,
        return_tensor=return_tensor,
        vocab=vocab,
        peptide_idx=peptide_idx,
        include_peptide_seq=include_peptide_seq,
    )
    if isinstance(peptidoform, MultiPeptidoform):
        return _multichain_fragment_ladder(peptidoform, **kwargs)
    if _has_crosslinks(peptidoform):
        return _crosslink_fragment_ladder(peptidoform, **kwargs)
    if _has_branches(peptidoform):
        return _branched_fragment_ladder(peptidoform, **kwargs)
    return _linear_fragment_ladder(peptidoform, **kwargs)


@requires_canonical
def _linear_fragment_ladder(
    peptidoform: Peptidoform,
    *,
    ion_types: Sequence[IonType],
    max_fragment_charge: int,
    neutral_losses: Sequence[str] | None,
    return_tensor: bool,
    vocab: ModVocab,
    peptide_idx: int | None,
    include_peptide_seq: bool,
) -> tuple[pa.Table, torch.Tensor | None]:
    """Linear-peptide ladder. N-term / C-term mods fold into N-side / C-side
    fragment masses uniformly; global isotope labels and global fixed mods
    are expanded into per-residue / per-terminal contributions before the
    cumsum.

    Rejects ambiguity groups, ranges, and labile mods — those need to be
    disambiguated or stripped before ladder generation. A future
    ``Peptidoform.disambiguate()`` helper will do the rewriting.
    """
    seq = peptidoform.sequence
    validate(seq, AA)
    L = len(seq)
    if L < 2:
        raise ValueError(f"fragment_ladder requires sequence length ≥ 2; got {L}")
    if max_fragment_charge < 1:
        raise ValueError(
            f"max_fragment_charge must be ≥ 1; got {max_fragment_charge}"
        )
    if not ion_types:
        raise ValueError("ion_types must be non-empty")

    if peptidoform.ranges:
        raise ValueError(
            "ranges (PRT(ESFRMS)[+19.0523]ISK syntax) are not supported by "
            "the linear fragment-ladder backend; expand the range to a "
            "concrete residue position before calling fragment_ladder"
        )
    if peptidoform.labile_mods:
        raise ValueError(
            "labile mods ({Glycan:Hex} syntax) are not supported by the "
            "linear fragment-ladder backend; precursor-only mods don't "
            "appear in fragments by definition"
        )
    if peptidoform.unknown_pos_mods:
        raise ValueError(
            "unknown-position mods ([Phospho]?PEPTIDE syntax) are not "
            "supported by the linear fragment-ladder backend; "
            "disambiguate to a concrete residue position first"
        )
    for tm in (
        *peptidoform.n_term_mods,
        *peptidoform.c_term_mods,
        *(t for tms in peptidoform.residue_mods.values() for t in tms),
    ):
        if tm.group_id is not None:
            raise ValueError(
                f"ambiguity / cross-link / branch group {tm.group_id!r} "
                "is unsupported by the linear fragment-ladder backend; "
                "disambiguate or route through the crosslink/branch backends"
            )

    for pos in peptidoform.residue_mods:
        if not 0 <= pos < L:
            raise IndexError(
                f"modification index {pos} out of range for length-{L} peptide"
            )

    n_pos = L - 1
    n_types = len(ion_types)
    n_charges = max_fragment_charge
    loss_ids: tuple[str, ...] = tuple(neutral_losses) if neutral_losses else ()
    n_losses = 1 + len(loss_ids)
    isotopes = peptidoform.global_isotopes

    # ── per-residue mass: residue + explicit mods + fixed-mod expansion +
    #    global-isotope shift on the residue's element atoms.
    assert AA.compositions is not None
    residue_comps = [AA.compositions[r] for r in seq]
    pos_mass = [
        c.mass + _isotope_shift_for_composition(c, isotopes) for c in residue_comps
    ]
    pos_mod_ids: list[list[str]] = [[] for _ in range(L)]

    def _add_modref_at(pos: int, modref: ModRef) -> None:
        d_comp, d_mass = _modref_contribution(modref, vocab, monoisotopic=True)
        pos_mass[pos] += d_comp.mass + d_mass + _isotope_shift_for_composition(
            d_comp, isotopes
        )
        cid = _modref_canonical_id(modref, vocab)
        if cid is not None:
            pos_mod_ids[pos].append(cid)

    for pos, taggedmods in peptidoform.residue_mods.items():
        for tm in taggedmods:
            if tm.mod is None:
                continue
            _add_modref_at(pos, tm.mod)

    # ── N-term / C-term scalars: terminal mod masses + their iso shifts.
    n_term_extra = 0.0
    c_term_extra = 0.0
    for tm in peptidoform.n_term_mods:
        if tm.mod is None:
            continue
        d_comp, d_mass = _modref_contribution(tm.mod, vocab, monoisotopic=True)
        n_term_extra += (
            d_comp.mass + d_mass + _isotope_shift_for_composition(d_comp, isotopes)
        )
        cid = _modref_canonical_id(tm.mod, vocab)
        # Fold N-term mod ids onto position 0 for loss-check compatibility
        # with the prior representation.
        if cid is not None:
            pos_mod_ids[0].append(cid)
    for tm in peptidoform.c_term_mods:
        if tm.mod is None:
            continue
        d_comp, d_mass = _modref_contribution(tm.mod, vocab, monoisotopic=True)
        c_term_extra += (
            d_comp.mass + d_mass + _isotope_shift_for_composition(d_comp, isotopes)
        )
        cid = _modref_canonical_id(tm.mod, vocab)
        if cid is not None:
            pos_mod_ids[L - 1].append(cid)

    # ── Global fixed mods: location-rule expansion. Per-residue rules
    #    fold into pos_mass / pos_mod_ids; terminal rules fold into
    #    n_term_extra / c_term_extra. Manual residue mods override at
    #    that position (skipped by _iter_fixed_mod_residue_indices /
    #    _count_fixed_mod_locations).
    for modref, locations in peptidoform.fixed_mods:
        d_comp, d_mass = _modref_contribution(modref, vocab, monoisotopic=True)
        single_extra = (
            d_comp.mass + d_mass + _isotope_shift_for_composition(d_comp, isotopes)
        )
        cid = _modref_canonical_id(modref, vocab)
        for loc in locations:
            if loc == "N-term":
                count = _count_fixed_mod_locations(seq, loc, peptidoform)
                n_term_extra += single_extra * count
                if cid is not None and count and L > 0:
                    pos_mod_ids[0].append(cid)
            elif loc == "C-term":
                count = _count_fixed_mod_locations(seq, loc, peptidoform)
                c_term_extra += single_extra * count
                if cid is not None and count and L > 0:
                    pos_mod_ids[L - 1].append(cid)
            else:
                for i in _iter_fixed_mod_residue_indices(seq, loc, peptidoform):
                    pos_mass[i] += single_extra
                    if cid is not None:
                        pos_mod_ids[i].append(cid)

    # ── Cumulative residue sums ─────────────────────────────────────
    pos_mass_t = torch.tensor(pos_mass, dtype=torch.float64)
    n_cumulative = torch.cumsum(pos_mass_t, dim=0)  # (L,)
    total_residue_sum = n_cumulative[-1]
    n_side_residues = n_cumulative[: L - 1]  # (L-1,) sum residues 0..p
    c_side_residues = total_residue_sum - n_side_residues  # (L-1,) sum p+1..L-1
    n_side = n_side_residues + n_term_extra  # add N-term mods uniformly
    c_side = c_side_residues + c_term_extra  # add C-term mods uniformly

    # ── Ion-type offsets, with isotope shift folded in ──────────────
    type_offsets = torch.tensor(
        [
            ION_OFFSET_MASSES[t]
            + _isotope_shift_for_composition(ION_OFFSETS[t], isotopes)
            for t in ion_types
        ],
        dtype=torch.float64,
    )
    type_is_n = torch.tensor([t in _N_SIDE for t in ion_types], dtype=torch.bool)
    side_masses = torch.where(
        type_is_n[None, :], n_side[:, None], c_side[:, None]
    )  # (n_pos, n_types)
    neutral_no_loss = side_masses + type_offsets[None, :]  # (n_pos, n_types)

    # ── Loss deltas + biochemistry mask ─────────────────────────────
    loss_deltas = torch.zeros(n_losses, dtype=torch.float64)
    for i, lid in enumerate(loss_ids, start=1):
        loss_deltas[i] = LOSS_REGISTRY.get(lid).delta_mass
    neutral_with_loss = (
        neutral_no_loss[:, :, None] - loss_deltas[None, None, :]
    )

    charges_t = torch.arange(1, n_charges + 1, dtype=torch.float64)
    mz_tensor = (
        neutral_with_loss[:, :, None, :]
        + charges_t[None, None, :, None] * PROTON_MASS
    ) / charges_t[None, None, :, None]

    loss_mask = torch.ones((n_pos, n_types, n_losses), dtype=torch.bool)
    if loss_ids:
        for p in range(n_pos):
            for t_idx, ion_type in enumerate(ion_types):
                if ion_type in _N_SIDE:
                    frag_residues = list(seq[: p + 1])
                    frag_mods = {
                        i: pos_mod_ids[i] for i in range(p + 1) if pos_mod_ids[i]
                    }
                else:
                    frag_residues = list(seq[p + 1 :])
                    offset = p + 1
                    frag_mods = {
                        i - offset: pos_mod_ids[i]
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

    nan = torch.tensor(float("nan"), dtype=torch.float64)
    mz_tensor = torch.where(loss_mask[:, :, None, :], mz_tensor, nan)

    # ── Build Arrow table from tensor ───────────────────────────────
    finite = torch.isfinite(mz_tensor)
    idx_p, idx_t, idx_c, idx_l = torch.where(finite)
    mz_values = mz_tensor[finite].tolist()

    n_rows = idx_p.numel()
    pep_idx_col = (
        [peptide_idx] * n_rows if peptide_idx is not None else [None] * n_rows
    )
    pep_seq_col = [seq] * n_rows if include_peptide_seq else [None] * n_rows
    positions = idx_p.to(torch.int32).tolist()
    ion_type_codes = [int(ion_types[i]) for i in idx_t.tolist()]
    charges = [c + 1 for c in idx_c.tolist()]
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


def _crosslink_fragment_ladder(
    peptidoform: Peptidoform, **kwargs: object
) -> tuple[pa.Table, torch.Tensor | None]:
    raise NotImplementedError(
        "Cross-linked peptide fragment generation is not yet implemented. "
        "XL-MS produces multiple coupled fragment series (alpha-chain, "
        "beta-chain, alpha+beta with linker, plus MS-cleavable linker "
        "variants) that require workflow-specific assumptions about "
        "linker chemistry and cleavage modes. Planned alongside the lab's "
        "XL-MS workflow port."
    )


def _branched_fragment_ladder(
    peptidoform: Peptidoform, **kwargs: object
) -> tuple[pa.Table, torch.Tensor | None]:
    raise NotImplementedError(
        "Branched peptide fragment generation (#BRANCH groups) is not yet "
        "implemented. Branch chemistry varies by tag (SUMO, ubiquitin, "
        "ISG15) and merits a dedicated PR with concrete biology in scope."
    )


def _multichain_fragment_ladder(
    peptidoform: MultiPeptidoform, **kwargs: object
) -> tuple[pa.Table, torch.Tensor | None]:
    raise NotImplementedError(
        "Multi-chain peptidoform fragment generation (// separator) is not "
        "yet implemented. Inter-chain cross-link fragmentation is a "
        "specialization of the cross-link case; see _crosslink_fragment_ladder."
    )


__all__ = [
    "IonType",
    "ION_OFFSETS",
    "ION_OFFSET_MASSES",
    "FragmentIon",
    "fragment_mz",
    "fragment_ladder",
]
