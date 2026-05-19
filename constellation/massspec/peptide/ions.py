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
from constellation.massspec.peptide.neutral_losses import LOSS_REGISTRY
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


def _biochem_mask(
    *,
    seq: str,
    pos_mod_ids: Sequence[Sequence[str]],
    ion_types: Sequence[IonType],
    loss_ids: Sequence[str],
    n_pos: int,
    n_types: int,
    n_losses: int,
    type_is_n: torch.Tensor,
) -> torch.Tensor:
    """Vectorized neutral-loss biochemistry mask.

    Returns ``(n_pos, n_types, n_losses)`` bool tensor — slot 0 along
    the loss axis is always True (no-loss baseline); slots 1..len(loss_ids)
    encode whether each loss is biochemically licensed for the
    ``(position, ion_type)`` fragment.

    Replaces the original ``O(n_pos × n_types × n_losses)`` Python
    triple-loop with tensor cumulative-OR operations along the sequence
    axis. The inner-loop equivalent is ``loss_applies`` from
    ``massspec.peptide.neutral_losses``; this routine reproduces its
    semantics in tensor form so the result is byte-identical to the
    legacy path. Tested directly in ``tests/test_biochem_mask.py``.
    """
    if not loss_ids:
        return torch.ones((n_pos, n_types, 1), dtype=torch.bool)

    L = len(seq)
    n_loss_only = len(loss_ids)

    # ── Per-loss attribute tensors ───────────────────────────────────
    losses = [LOSS_REGISTRY.get(lid) for lid in loss_ids]
    has_residue_rule = torch.tensor(
        [bool(l.triggering_residues) for l in losses], dtype=torch.bool
    )
    has_mod_rule = torch.tensor(
        [bool(l.triggering_mods) for l in losses], dtype=torch.bool
    )
    joint_mode = has_residue_rule & has_mod_rule  # (n_loss_only,)

    # ion-type filter — True where loss permits the ion type, or where
    # the loss declares no ion-type restriction (applies to all).
    ion_type_allowed = torch.empty(
        (n_types, n_loss_only), dtype=torch.bool
    )
    for l_idx, loss in enumerate(losses):
        if loss.applies_to_ion_types:
            for t_idx, ion_type in enumerate(ion_types):
                ion_type_allowed[t_idx, l_idx] = (
                    ion_type.name in loss.applies_to_ion_types
                )
        else:
            ion_type_allowed[:, l_idx] = True

    # ── Per-(residue, loss) trigger tensors ──────────────────────────
    # R[i, l] = True if seq[i] is in losses[l].triggering_residues
    # M[i, l] = True if any mod on seq[i] is in losses[l].triggering_mods
    # J[i, l] = R[i, l] AND M[i, l]  (joint-mode predicate, residue carries the mod)
    residue_trigger = torch.zeros((L, n_loss_only), dtype=torch.bool)
    mod_trigger = torch.zeros((L, n_loss_only), dtype=torch.bool)
    for l_idx, loss in enumerate(losses):
        if loss.triggering_residues:
            for i, r in enumerate(seq):
                if r in loss.triggering_residues:
                    residue_trigger[i, l_idx] = True
        if loss.triggering_mods:
            for i in range(L):
                mods_at_i = pos_mod_ids[i]
                if not mods_at_i:
                    continue
                if any(m in loss.triggering_mods for m in mods_at_i):
                    mod_trigger[i, l_idx] = True
    joint_trigger = residue_trigger & mod_trigger

    # ── Cumulative-OR along the residue axis ─────────────────────────
    # For N-side fragments: bond p covers residues 0..p (length p+1).
    # For C-side fragments: bond p covers residues p+1..L-1.
    #
    # torch.cummax(int_tensor, dim=0).values acts as running-OR when
    # the tensor is {0, 1}-valued.
    def _cumor(t: torch.Tensor) -> torch.Tensor:
        return torch.cummax(t.to(torch.int8), dim=0).values.bool()

    # N-side: for each bond p (0 ≤ p < n_pos), want OR over rows 0..p.
    # cumor(t)[p] gives exactly that. Trim to n_pos rows.
    r_n = _cumor(residue_trigger)[:n_pos]  # (n_pos, n_loss_only)
    m_n = _cumor(mod_trigger)[:n_pos]
    j_n = _cumor(joint_trigger)[:n_pos]

    # C-side: for each bond p, want OR over rows p+1..L-1. Equivalent
    # to a reverse cumulative-OR indexed at p+1. Flip → cumor → flip
    # then drop the first row (p=0 case = OR over rows 1..L-1).
    def _rev_cumor(t: torch.Tensor) -> torch.Tensor:
        flipped = torch.flip(t, [0])
        out = _cumor(flipped)
        return torch.flip(out, [0])

    r_c = _rev_cumor(residue_trigger)[1 : 1 + n_pos]  # (n_pos, n_loss_only)
    m_c = _rev_cumor(mod_trigger)[1 : 1 + n_pos]
    j_c = _rev_cumor(joint_trigger)[1 : 1 + n_pos]

    # ── Select N-side vs C-side per ion type ────────────────────────
    # type_is_n: (n_types,) → broadcast to (1, n_types, 1)
    type_is_n_b = type_is_n[None, :, None]
    r_side = torch.where(type_is_n_b, r_n[:, None, :], r_c[:, None, :])
    m_side = torch.where(type_is_n_b, m_n[:, None, :], m_c[:, None, :])
    j_side = torch.where(type_is_n_b, j_n[:, None, :], j_c[:, None, :])
    # shapes: (n_pos, n_types, n_loss_only)

    # ── Combine per-loss rules ───────────────────────────────────────
    #
    # case_no_mod:    applies = (~has_residue_rule) | r_side
    #                 (loss has no mod rule; residue rule satisfied or absent)
    # case_mod_only:  applies = m_side
    #                 (loss has mod rule but no residue rule)
    # case_joint:     applies = j_side
    #                 (loss has both — mod must sit ON a triggering residue)
    has_residue_rule_b = has_residue_rule[None, None, :]
    has_mod_rule_b = has_mod_rule[None, None, :]
    joint_mode_b = joint_mode[None, None, :]

    case_no_mod = (~has_residue_rule_b) | r_side
    case_with_mod = torch.where(joint_mode_b, j_side, m_side)
    applies = torch.where(has_mod_rule_b, case_with_mod, case_no_mod)

    # Final ion-type filter — (1, n_types, n_loss_only) broadcast.
    applies = applies & ion_type_allowed[None, :, :]

    # ── Stitch slot-0 (no-loss baseline) onto the loss axis ─────────
    no_loss = torch.ones((n_pos, n_types, 1), dtype=torch.bool)
    return torch.cat([no_loss, applies], dim=-1)


@dataclass(slots=True)
class _LinearInputs:
    """Per-peptide intermediate computed once per ladder call.

    All the string / ModRef work happens here so the heavy tensor
    operations in ``_run_linear_batch`` can be batched over peptidoforms
    of the same length. ``pos_mass`` includes residue + explicit mods +
    fixed-mod expansion + global-isotope shifts; ``pos_mod_ids`` is
    column-aligned for the biochem mask.
    """

    seq: str
    pos_mass: list[float]  # length L
    pos_mod_ids: list[list[str]]  # length L
    n_term_extra: float
    c_term_extra: float
    isotopes: tuple[str, ...]


def _validate_linear_peptidoform(peptidoform: Peptidoform) -> None:
    """Raise on peptidoform shapes the linear backend doesn't support.

    Hoisted out of ``_linear_fragment_ladder`` so the batched entry
    point shares one validation surface.
    """
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


def _prepare_linear_inputs(
    peptidoform: Peptidoform, *, vocab: ModVocab
) -> _LinearInputs:
    """Pure-Python prep: compute per-residue mass + mod-id lists + terminal extras.

    Mirrors the prep block at the top of ``_linear_fragment_ladder``;
    factored out so ``fragment_ladders_batch`` can call it once per
    peptidoform and then run the heavy tensor math in batched form.
    """
    seq = peptidoform.sequence
    validate(seq, AA)
    L = len(seq)
    if L < 2:
        raise ValueError(
            f"fragment_ladder requires sequence length ≥ 2; got {L}"
        )
    _validate_linear_peptidoform(peptidoform)
    for pos in peptidoform.residue_mods:
        if not 0 <= pos < L:
            raise IndexError(
                f"modification index {pos} out of range for length-{L} peptide"
            )
    isotopes = peptidoform.global_isotopes

    assert AA.compositions is not None
    residue_comps = [AA.compositions[r] for r in seq]
    pos_mass: list[float] = [
        c.mass + _isotope_shift_for_composition(c, isotopes)
        for c in residue_comps
    ]
    pos_mod_ids: list[list[str]] = [[] for _ in range(L)]

    def _add_modref_at(pos: int, modref: ModRef) -> None:
        d_comp, d_mass = _modref_contribution(
            modref, vocab, monoisotopic=True
        )
        pos_mass[pos] += (
            d_comp.mass
            + d_mass
            + _isotope_shift_for_composition(d_comp, isotopes)
        )
        cid = _modref_canonical_id(modref, vocab)
        if cid is not None:
            pos_mod_ids[pos].append(cid)

    for pos, taggedmods in peptidoform.residue_mods.items():
        for tm in taggedmods:
            if tm.mod is None:
                continue
            _add_modref_at(pos, tm.mod)

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

    return _LinearInputs(
        seq=seq,
        pos_mass=pos_mass,
        pos_mod_ids=pos_mod_ids,
        n_term_extra=n_term_extra,
        c_term_extra=c_term_extra,
        isotopes=isotopes,
    )


def _run_linear_batch(
    inputs: Sequence[_LinearInputs],
    *,
    ion_types: Sequence[IonType],
    max_fragment_charge: int,
    loss_ids: Sequence[str],
) -> torch.Tensor:
    """Compute a batched theoretical ladder m/z tensor.

    All peptidoforms must share the same sequence length and global
    isotope labels (callers group by these). Returns a
    ``(B, n_pos, n_types, n_charges, n_losses)`` ``float64`` tensor
    with NaN-masked invalid (position, ion_type, loss) combinations.
    The Arrow-table emission per peptide happens upstream.
    """
    if not inputs:
        raise ValueError("inputs must be non-empty")
    L = len(inputs[0].seq)
    isotopes = inputs[0].isotopes
    for inp in inputs:
        if len(inp.seq) != L:
            raise ValueError(
                "batched linear ladder requires uniform sequence length; "
                f"got mix of {L} and {len(inp.seq)}"
            )
        if inp.isotopes != isotopes:
            raise ValueError(
                "batched linear ladder requires uniform global_isotopes "
                "across the batch"
            )

    n_pos = L - 1
    n_types = len(ion_types)
    n_charges = max_fragment_charge
    n_losses = 1 + len(loss_ids)
    _ = n_charges  # used by broadcasting below; explicit binding for the reader

    # Batched per-residue mass — (B, L)
    pos_mass_t = torch.tensor(
        [inp.pos_mass for inp in inputs], dtype=torch.float64
    )
    n_term_t = torch.tensor(
        [inp.n_term_extra for inp in inputs], dtype=torch.float64
    )
    c_term_t = torch.tensor(
        [inp.c_term_extra for inp in inputs], dtype=torch.float64
    )

    cumulative = torch.cumsum(pos_mass_t, dim=1)  # (B, L)
    total = cumulative[:, -1]  # (B,)
    n_side_residues = cumulative[:, :n_pos]  # (B, n_pos)
    c_side_residues = total[:, None] - n_side_residues  # (B, n_pos)
    n_side = n_side_residues + n_term_t[:, None]  # (B, n_pos)
    c_side = c_side_residues + c_term_t[:, None]

    # Ion-type offsets — global; depend only on isotopes (uniform).
    type_offsets = torch.tensor(
        [
            ION_OFFSET_MASSES[t]
            + _isotope_shift_for_composition(ION_OFFSETS[t], isotopes)
            for t in ion_types
        ],
        dtype=torch.float64,
    )  # (n_types,)
    type_is_n = torch.tensor(
        [t in _N_SIDE for t in ion_types], dtype=torch.bool
    )  # (n_types,)
    # side_masses: (B, n_pos, n_types)
    side_masses = torch.where(
        type_is_n[None, None, :], n_side[:, :, None], c_side[:, :, None]
    )
    neutral_no_loss = side_masses + type_offsets[None, None, :]

    loss_deltas = torch.zeros(n_losses, dtype=torch.float64)
    for i, lid in enumerate(loss_ids, start=1):
        loss_deltas[i] = LOSS_REGISTRY.get(lid).delta_mass
    # (B, n_pos, n_types, n_losses)
    neutral_with_loss = (
        neutral_no_loss[:, :, :, None] - loss_deltas[None, None, None, :]
    )

    charges_t = torch.arange(1, n_charges + 1, dtype=torch.float64)
    # (B, n_pos, n_types, n_charges, n_losses)
    mz_tensor = (
        neutral_with_loss[:, :, :, None, :]
        + charges_t[None, None, None, :, None] * PROTON_MASS
    ) / charges_t[None, None, None, :, None]

    # Batched biochem mask — (B, n_pos, n_types, n_losses)
    loss_mask = _biochem_mask_batched(
        seqs=[inp.seq for inp in inputs],
        pos_mod_ids=[inp.pos_mod_ids for inp in inputs],
        ion_types=ion_types,
        loss_ids=loss_ids,
        n_pos=n_pos,
        n_types=n_types,
        n_losses=n_losses,
        type_is_n=type_is_n,
    )

    nan = torch.tensor(float("nan"), dtype=torch.float64)
    mz_tensor = torch.where(loss_mask[:, :, :, None, :], mz_tensor, nan)
    return mz_tensor


def _biochem_mask_batched(
    *,
    seqs: Sequence[str],
    pos_mod_ids: Sequence[Sequence[Sequence[str]]],
    ion_types: Sequence[IonType],
    loss_ids: Sequence[str],
    n_pos: int,
    n_types: int,
    n_losses: int,
    type_is_n: torch.Tensor,
) -> torch.Tensor:
    """Batched version of ``_biochem_mask``. Returns
    ``(B, n_pos, n_types, n_losses)`` bool tensor."""
    B = len(seqs)
    if not loss_ids:
        return torch.ones((B, n_pos, n_types, 1), dtype=torch.bool)

    L = len(seqs[0])
    n_loss_only = len(loss_ids)
    losses = [LOSS_REGISTRY.get(lid) for lid in loss_ids]

    has_residue_rule = torch.tensor(
        [bool(l.triggering_residues) for l in losses], dtype=torch.bool
    )
    has_mod_rule = torch.tensor(
        [bool(l.triggering_mods) for l in losses], dtype=torch.bool
    )
    joint_mode = has_residue_rule & has_mod_rule  # (n_loss_only,)

    ion_type_allowed = torch.empty(
        (n_types, n_loss_only), dtype=torch.bool
    )
    for l_idx, loss in enumerate(losses):
        if loss.applies_to_ion_types:
            for t_idx, ion_type in enumerate(ion_types):
                ion_type_allowed[t_idx, l_idx] = (
                    ion_type.name in loss.applies_to_ion_types
                )
        else:
            ion_type_allowed[:, l_idx] = True

    # (B, L, n_loss_only) trigger tensors
    residue_trigger = torch.zeros((B, L, n_loss_only), dtype=torch.bool)
    mod_trigger = torch.zeros((B, L, n_loss_only), dtype=torch.bool)
    for b in range(B):
        seq = seqs[b]
        pmods = pos_mod_ids[b]
        for l_idx, loss in enumerate(losses):
            if loss.triggering_residues:
                for i, r in enumerate(seq):
                    if r in loss.triggering_residues:
                        residue_trigger[b, i, l_idx] = True
            if loss.triggering_mods:
                for i in range(L):
                    mods_at_i = pmods[i]
                    if not mods_at_i:
                        continue
                    if any(m in loss.triggering_mods for m in mods_at_i):
                        mod_trigger[b, i, l_idx] = True
    joint_trigger = residue_trigger & mod_trigger

    def _cumor(t: torch.Tensor) -> torch.Tensor:
        return torch.cummax(t.to(torch.int8), dim=1).values.bool()

    r_n = _cumor(residue_trigger)[:, :n_pos, :]  # (B, n_pos, n_loss_only)
    m_n = _cumor(mod_trigger)[:, :n_pos, :]
    j_n = _cumor(joint_trigger)[:, :n_pos, :]

    def _rev_cumor(t: torch.Tensor) -> torch.Tensor:
        flipped = torch.flip(t, [1])
        out = _cumor(flipped)
        return torch.flip(out, [1])

    r_c = _rev_cumor(residue_trigger)[:, 1 : 1 + n_pos, :]
    m_c = _rev_cumor(mod_trigger)[:, 1 : 1 + n_pos, :]
    j_c = _rev_cumor(joint_trigger)[:, 1 : 1 + n_pos, :]

    type_is_n_b = type_is_n[None, None, :, None]  # (1, 1, n_types, 1)
    r_side = torch.where(
        type_is_n_b, r_n[:, :, None, :], r_c[:, :, None, :]
    )  # (B, n_pos, n_types, n_loss_only)
    m_side = torch.where(type_is_n_b, m_n[:, :, None, :], m_c[:, :, None, :])
    j_side = torch.where(type_is_n_b, j_n[:, :, None, :], j_c[:, :, None, :])

    has_residue_rule_b = has_residue_rule[None, None, None, :]
    has_mod_rule_b = has_mod_rule[None, None, None, :]
    joint_mode_b = joint_mode[None, None, None, :]

    case_no_mod = (~has_residue_rule_b) | r_side
    case_with_mod = torch.where(joint_mode_b, j_side, m_side)
    applies = torch.where(has_mod_rule_b, case_with_mod, case_no_mod)
    applies = applies & ion_type_allowed[None, None, :, :]

    no_loss = torch.ones((B, n_pos, n_types, 1), dtype=torch.bool)
    return torch.cat([no_loss, applies], dim=-1)


def _emit_fragment_table(
    *,
    mz_slice: torch.Tensor,  # (n_pos, n_types, n_charges, n_losses)
    seq: str,
    ion_types: Sequence[IonType],
    loss_ids: Sequence[str],
    peptide_idx: int | None,
    include_peptide_seq: bool,
) -> pa.Table:
    """Build a ``FragmentIonTable`` from one peptide's slice."""
    finite = torch.isfinite(mz_slice)
    idx_p, idx_t, idx_c, idx_l = torch.where(finite)
    mz_values = mz_slice[finite].tolist()

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
    return pa.table(
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

    For bulk workloads (many peptidoforms in one pass), prefer
    :func:`fragment_ladders_batch` which amortizes Python / tensor
    dispatch overhead across peptidoforms grouped by sequence length.
    """
    if max_fragment_charge < 1:
        raise ValueError(
            f"max_fragment_charge must be ≥ 1; got {max_fragment_charge}"
        )
    if not ion_types:
        raise ValueError("ion_types must be non-empty")

    loss_ids: tuple[str, ...] = (
        tuple(neutral_losses) if neutral_losses else ()
    )
    inputs = _prepare_linear_inputs(peptidoform, vocab=vocab)
    mz_tensor = _run_linear_batch(
        [inputs],
        ion_types=ion_types,
        max_fragment_charge=max_fragment_charge,
        loss_ids=loss_ids,
    )[0]
    table = _emit_fragment_table(
        mz_slice=mz_tensor,
        seq=inputs.seq,
        ion_types=ion_types,
        loss_ids=loss_ids,
        peptide_idx=peptide_idx,
        include_peptide_seq=include_peptide_seq,
    )
    return table, (mz_tensor if return_tensor else None)


def fragment_ladder_indices_batch(
    peptidoforms: Sequence[Peptidoform],
    *,
    ion_types: Sequence[IonType] = (IonType.B, IonType.Y),
    max_fragment_charge: int = 1,
    neutral_losses: Sequence[str] | None = None,
    vocab: ModVocab = UNIMOD,
) -> list[dict[tuple[int, int, int, str | None], float]]:
    """Batched-ladder fast path for annotated-spectrum readers.

    Returns a list of ``{(ion_type, position, charge, loss_id): mz}``
    dicts in input order — one per peptidoform. Uses the same grouped-
    by-length tensor math as :func:`fragment_ladders_batch` but skips
    the Arrow-table assembly that callers like the MSP reader would
    only deconstruct again. Linear peptidoforms only (no cross-link /
    branch / multichain dispatch). Roughly 2-3× faster than calling
    ``fragment_ladders_batch`` and re-indexing the resulting tables.
    """
    if max_fragment_charge < 1:
        raise ValueError(
            f"max_fragment_charge must be ≥ 1; got {max_fragment_charge}"
        )
    if not ion_types:
        raise ValueError("ion_types must be non-empty")
    loss_ids: tuple[str, ...] = (
        tuple(neutral_losses) if neutral_losses else ()
    )
    n_losses = 1 + len(loss_ids)
    ion_type_codes = tuple(int(t) for t in ion_types)

    # Group linear peptidoforms by (length, isotopes); reject non-linear
    # — callers route those to ``fragment_ladder`` themselves.
    inputs_by_group: dict[
        tuple[int, tuple[str, ...]], list[tuple[int, _LinearInputs]]
    ] = {}
    for idx, p in enumerate(peptidoforms):
        if not isinstance(p, Peptidoform):
            raise TypeError(
                "fragment_ladder_indices_batch supports only linear "
                "Peptidoform inputs; got "
                f"{type(p).__name__}"
            )
        if _has_crosslinks(p) or _has_branches(p):
            raise ValueError(
                "fragment_ladder_indices_batch supports only linear "
                "peptidoforms (no cross-links / branches); route those "
                "through fragment_ladder per-spectrum"
            )
        inp = _prepare_linear_inputs(p, vocab=vocab)
        key = (len(inp.seq), inp.isotopes)
        inputs_by_group.setdefault(key, []).append((idx, inp))

    results: list[dict[tuple[int, int, int, str | None], float] | None] = [
        None
    ] * len(peptidoforms)
    for (_length, _isotopes), items in inputs_by_group.items():
        indices = [i for i, _ in items]
        inputs = [inp for _, inp in items]
        mz_batch = _run_linear_batch(
            inputs,
            ion_types=ion_types,
            max_fragment_charge=max_fragment_charge,
            loss_ids=loss_ids,
        )
        # mz_batch shape: (B, n_pos, n_types, n_charges, n_losses)
        finite_b = torch.isfinite(mz_batch)
        idx_b, idx_p, idx_t, idx_c, idx_l = torch.where(finite_b)
        mz_values = mz_batch[finite_b].tolist()
        # Split per batch slot
        idx_b_l = idx_b.tolist()
        idx_p_l = idx_p.tolist()
        idx_t_l = idx_t.tolist()
        idx_c_l = idx_c.tolist()
        idx_l_l = idx_l.tolist()
        n_batch = len(inputs)
        per_slot: list[dict[tuple[int, int, int, str | None], float]] = [
            {} for _ in range(n_batch)
        ]
        for k in range(len(mz_values)):
            b = idx_b_l[k]
            it = ion_type_codes[idx_t_l[k]]
            pos = idx_p_l[k]
            ch = idx_c_l[k] + 1
            li = idx_l_l[k]
            lid = None if li == 0 else loss_ids[li - 1]
            per_slot[b][(it, pos, ch, lid)] = mz_values[k]
        for slot, idx_into_input in enumerate(indices):
            results[idx_into_input] = per_slot[slot]

    # ``loss_ids``/``n_losses`` used only by the inner branch; the
    # variable below silences an unused-name lint.
    _ = n_losses
    return [r if r is not None else {} for r in results]


def fragment_ladders_batch(
    peptidoforms: Sequence[ProFormaResult],
    *,
    ion_types: Sequence[IonType] = (IonType.B, IonType.Y),
    max_fragment_charge: int = 1,
    neutral_losses: Sequence[str] | None = None,
    vocab: ModVocab = UNIMOD,
    peptide_idxs: Sequence[int | None] | None = None,
    include_peptide_seq: bool = True,
) -> list[pa.Table]:
    """Batched theoretical-ladder construction across many peptidoforms.

    Equivalent to looping ``fragment_ladder(p, ...)`` per peptide, but
    groups inputs by sequence length + global isotope labels so the
    inner tensor math (cumsum, ion-type / charge / loss broadcasts,
    biochem mask, NaN masking) runs once per group rather than once
    per peptide. Returns a list of ``FragmentIonTable``s in input
    order.

    Non-linear peptidoforms (cross-linked, branched, multichain) and
    ranges / labile / unknown-position mods fall back to per-spectrum
    dispatch via :func:`fragment_ladder` so the batched entry point
    works for typical proteomics inputs without restricting the
    public ``fragment_ladder`` surface.

    Big win on MSP / mzSpecLib / search-engine outputs that present
    thousands of peptidoforms per file — amortizes torch dispatch
    overhead across the batch.
    """
    if max_fragment_charge < 1:
        raise ValueError(
            f"max_fragment_charge must be ≥ 1; got {max_fragment_charge}"
        )
    if not ion_types:
        raise ValueError("ion_types must be non-empty")

    loss_ids: tuple[str, ...] = (
        tuple(neutral_losses) if neutral_losses else ()
    )
    peptide_idxs_seq: list[int | None] = (
        list(peptide_idxs)
        if peptide_idxs is not None
        else [None] * len(peptidoforms)
    )
    if len(peptide_idxs_seq) != len(peptidoforms):
        raise ValueError(
            "len(peptide_idxs) must match len(peptidoforms) when supplied"
        )

    results: list[pa.Table | None] = [None] * len(peptidoforms)

    # ── First, route any non-linear / unsupported peptidoforms to the
    #    dispatcher; collect linear ones for batched processing.
    linear_inputs_by_group: dict[
        tuple[int, tuple[str, ...]], list[tuple[int, _LinearInputs]]
    ] = {}

    for idx, p in enumerate(peptidoforms):
        if isinstance(p, MultiPeptidoform) or _has_crosslinks(p) or _has_branches(p):
            tbl, _ = fragment_ladder(
                p,
                ion_types=ion_types,
                max_fragment_charge=max_fragment_charge,
                neutral_losses=list(loss_ids) if loss_ids else None,
                return_tensor=False,
                vocab=vocab,
                peptide_idx=peptide_idxs_seq[idx],
                include_peptide_seq=include_peptide_seq,
            )
            results[idx] = tbl
            continue
        inp = _prepare_linear_inputs(p, vocab=vocab)
        key = (len(inp.seq), inp.isotopes)
        linear_inputs_by_group.setdefault(key, []).append((idx, inp))

    # ── Process each (length, isotopes) group in one batched pass.
    for (_length, _isotopes), items in linear_inputs_by_group.items():
        indices = [i for i, _ in items]
        inputs = [inp for _, inp in items]
        mz_batch = _run_linear_batch(
            inputs,
            ion_types=ion_types,
            max_fragment_charge=max_fragment_charge,
            loss_ids=loss_ids,
        )
        for slot, idx_into_input in enumerate(indices):
            results[idx_into_input] = _emit_fragment_table(
                mz_slice=mz_batch[slot],
                seq=inputs[slot].seq,
                ion_types=ion_types,
                loss_ids=loss_ids,
                peptide_idx=peptide_idxs_seq[idx_into_input],
                include_peptide_seq=include_peptide_seq,
            )

    # All slots are filled now (every input was routed to one of the
    # two branches above).
    return [t for t in results if t is not None]


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
    "fragment_ladders_batch",
    "fragment_ladder_indices_batch",
]
