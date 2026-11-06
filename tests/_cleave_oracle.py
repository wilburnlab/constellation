"""``cleave`` and ``enumerate_modforms`` as they were before 2026-11-05
(``constellation.core.sequence.protein`` at the commit before span
kinds, unspecific digestion and context-aware modification placement),
frozen as the oracle the default paths are held to. Do not edit; it is a
record."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.alphabets import AA
from constellation.core.sequence.ops import validate
from constellation.core.sequence.proforma import ModRef, Peptidoform, TaggedMod
from constellation.core.sequence.protein import Peptide, Protease, cleave_sites

__all__ = ["cleave", "enumerate_modforms"]


def cleave(
    seq: str,
    protease: str | Protease = "Trypsin",
    *,
    missed_cleavages: int = 0,
    min_length: int = 6,
    max_length: int | None = 50,
    semi_specific: bool = False,
    excise_initiator_met: bool = False,
    validate_alphabet: bool = True,
    return_spans: bool = False,
) -> list[str] | list[Peptide]:
    """Enzymatically cleave `seq` into peptides.

    Iterates the protease's cut sites, then enumerates spans across
    0..missed_cleavages consecutive cuts. With `semi_specific=True`,
    additionally emits all suffixes of each fully-specific peptide
    (the C-terminal cut is preserved; the N-terminus floats) — the
    standard "semi-tryptic" search-engine option.

    With `excise_initiator_met=True`, for every N-terminal peptide
    (a span starting at protein position 0) whose first residue is the
    initiator methionine, ALSO emit the Met-clipped form (`peptide[1:]`)
    — keeping both. This is the initiator-Met-excision behaviour search
    engines apply (verified against EncyclopeDIA 6.5.15's predicted
    library: ~87% of protein N-termini carry both the Met-intact and
    Met-clipped precursor). The clipping is **unconditional** (not gated
    on the second residue / biological Met-aminopeptidase rule — the
    library shows clipped forms for every second residue), applies to
    all N-terminal spans (0-missed and missed-cleavage variants), and
    the clipped form is subject to the same length filter independently
    (so a clipped form that drops below `min_length` is omitted while
    its intact partner survives).

    Output ordering: peptides are sorted by `(start ASC, end ASC)`,
    so along the protein you read the 0-missed peptide first, then
    the 1-missed, then 2-missed at each starting cut. Pyteomics
    returns a `set` (random); we explicitly sort.

    Duplicate sequences (same residue content, different positions)
    are de-duplicated by keeping the first occurrence under the
    `(start, end)` ordering.

    With `validate_alphabet=False`, the canonical-AA check is skipped so
    sequences carrying non-canonical residues (selenocysteine ``U``,
    pyrrolysine ``O``, ambiguity codes ``B``/``Z``/``X``/``J``) digest
    normally instead of raising. The cut sites only depend on K/R, so
    peptides not containing the odd residue come out exactly as they
    would for a clean sequence — important when digesting whole
    proteomes, where one selenoprotein must not cost you all of its
    (otherwise canonical) tryptic peptides.

    With `return_spans=True`, returns `list[Peptide]` with the
    parent-protein coordinates intact (no de-duplication on sequence).
    """
    if validate_alphabet:
        validate(seq, AA)
    if missed_cleavages < 0:
        raise ValueError(f"missed_cleavages must be ≥ 0; got {missed_cleavages}")

    sites = cleave_sites(seq, protease, validate_alphabet=validate_alphabet)
    # `sites` includes 0 and len(seq); spans are between consecutive
    # cuts plus extensions across `missed_cleavages` interior cuts.
    spans: list[Peptide] = []
    n_sites = len(sites)
    for i in range(n_sites - 1):
        for k in range(missed_cleavages + 1):
            j = i + 1 + k
            if j >= n_sites:
                break
            start = sites[i]
            end = sites[j]
            peptide = seq[start:end]
            if not _length_ok(peptide, min_length, max_length):
                continue
            spans.append(Peptide(peptide, start, end, n_missed=k))

    if semi_specific:
        # Add all suffixes of each span — N-terminus floats, C-terminus
        # is a real cut. Implemented as: for each span (s, e), add
        # (s+1..e), (s+2..e), ... down to min_length.
        extras: list[Peptide] = []
        for sp in spans:
            for new_start in range(sp.start + 1, sp.end):
                sub = seq[new_start : sp.end]
                if not _length_ok(sub, min_length, max_length):
                    continue
                extras.append(Peptide(sub, new_start, sp.end, n_missed=sp.n_missed))
        spans.extend(extras)

    if excise_initiator_met:
        # For every N-terminal peptide (span at protein position 0)
        # starting with the initiator Met, also emit the Met-clipped
        # form. Only full spans start at 0 (semi suffixes start ≥ 1), so
        # iterating `spans` after the semi block still picks exactly the
        # N-terminal peptides.
        met_clipped: list[Peptide] = []
        for sp in spans:
            if sp.start == 0 and sp.sequence.startswith("M"):
                clipped = sp.sequence[1:]
                if _length_ok(clipped, min_length, max_length):
                    met_clipped.append(
                        Peptide(clipped, 1, sp.end, n_missed=sp.n_missed)
                    )
        spans.extend(met_clipped)

    spans.sort(key=lambda p: (p.start, p.end))

    if return_spans:
        return spans

    # Deduplicate sequences, keep first occurrence under (start, end) order.
    seen: set[str] = set()
    out: list[str] = []
    for sp in spans:
        if sp.sequence in seen:
            continue
        seen.add(sp.sequence)
        out.append(sp.sequence)
    return out


def _length_ok(peptide: str, min_length: int, max_length: int | None) -> bool:
    n = len(peptide)
    if n < min_length:
        return False
    if max_length is not None and n > max_length:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# Modification enumeration
# ──────────────────────────────────────────────────────────────────────

N_TERM = "N-term"
C_TERM = "C-term"


def _normalize_mod_spec(
    spec: Mapping[str, str | Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    """Site → tuple-of-mod-keys, accepting a bare string per site."""
    out: dict[str, tuple[str, ...]] = {}
    for site, keys in (spec or {}).items():
        out[site] = (keys,) if isinstance(keys, str) else tuple(keys)
    return out


def _slots_for_site(site: str, seq: str) -> list[int | str]:
    """Positions in `seq` a site specifier applies to.

    Residue sites yield 0-indexed offsets; ``N-term`` / ``C-term`` yield
    themselves as sentinel slots.
    """
    if site in (N_TERM, C_TERM):
        return [site]
    return [i for i, aa in enumerate(seq) if aa == site]


def _check_specificity(key: str, site: str, vocab: ModVocab, peptide: str) -> None:
    mod = vocab[key]
    if site == N_TERM:
        ok = bool(peptide) and mod.has_terminal_specificity_for(N_TERM, peptide[0])
        detail = f" for a peptide starting with {peptide[:1]!r}" if peptide else ""
    elif site == C_TERM:
        ok = bool(peptide) and mod.has_terminal_specificity_for(C_TERM, peptide[-1])
        detail = f" for a peptide ending with {peptide[-1:]!r}" if peptide else ""
    else:
        ok = mod.has_residue_specificity(site)
        detail = ""
    if not ok:
        raise ValueError(
            f"{key} ({mod.name}) has no UNIMOD specificity for site {site!r}"
            f"{detail}; pass validate_specificity=False to apply it anyway"
        )


def _tagged(key: str) -> TaggedMod:
    cv, _, acc = key.partition(":")
    return TaggedMod(mod=ModRef(cv=cv, accession=acc))


def enumerate_modforms(
    peptide: str,
    *,
    fixed: Mapping[str, str | Sequence[str]] | None = None,
    variable: Mapping[str, str | Sequence[str]] | None = None,
    max_variable: int = 3,
    max_forms: int | None = None,
    vocab: ModVocab = UNIMOD,
    validate_specificity: bool = True,
    validate_alphabet: bool = True,
) -> list[Peptidoform]:
    """Enumerate the modified forms of one peptide.

    ``fixed`` mods are applied to **every** matching site and are not
    enumerated. ``variable`` mods are enumerated over all combinations of
    0..``max_variable`` simultaneously-modified sites. Both map a *site*
    — a residue letter, or the ``N-term`` / ``C-term`` sentinels — to one
    mod key (``"UNIMOD:35"``) or several (``["UNIMOD:21", "UNIMOD:35"]``),
    the latter meaning "any one of these may occupy this site".

    A site already carrying a fixed mod is not offered to the variable
    enumeration, so ``fixed={"C": "UNIMOD:4"}`` and
    ``variable={"C": ...}`` do not stack on the same cysteine.

    Returns forms ordered by increasing variable-mod count, the
    fixed-only form first. Mod keys are validated against ``vocab``, and
    against each mod's UNIMOD specificity unless
    ``validate_specificity=False`` — the latter is the escape hatch for
    deliberately non-canonical placements.

    Raises ``ValueError`` if the enumeration would exceed ``max_forms``
    (when set) rather than silently truncating, since a truncated set
    reads downstream as a complete one.
    """
    if validate_alphabet:
        validate(peptide, AA)
    if max_variable < 0:
        raise ValueError(f"max_variable must be >= 0, got {max_variable}")

    fixed_spec = _normalize_mod_spec(fixed)
    var_spec = _normalize_mod_spec(variable)

    for site, keys in (*fixed_spec.items(), *var_spec.items()):
        for key in keys:
            if key not in vocab:
                raise ValueError(f"unknown modification key {key!r} for site {site!r}")
            if validate_specificity:
                _check_specificity(key, site, vocab, peptide)

    for site, keys in fixed_spec.items():
        if len(keys) != 1:
            raise ValueError(
                f"fixed site {site!r} takes exactly one modification, got {list(keys)}"
            )

    # Fixed application — every matching slot, always.
    fixed_slots: dict[int | str, TaggedMod] = {}
    for site, keys in fixed_spec.items():
        for slot in _slots_for_site(site, peptide):
            fixed_slots[slot] = _tagged(keys[0])

    # Variable candidates — slot → the mod keys that may occupy it.
    candidates: dict[int | str, list[str]] = {}
    for site, keys in var_spec.items():
        for slot in _slots_for_site(site, peptide):
            if slot in fixed_slots:
                continue
            candidates.setdefault(slot, []).extend(keys)

    ordered = sorted(
        candidates, key=lambda s: (isinstance(s, str), s if isinstance(s, int) else str(s))
    )
    k_max = min(max_variable, len(ordered))

    if max_forms is not None:
        total = sum(
            math.prod(len(candidates[s]) for s in combo)
            for k in range(k_max + 1)
            for combo in itertools.combinations(ordered, k)
        )
        if total > max_forms:
            raise ValueError(
                f"{peptide} would yield {total} modforms, exceeding max_forms="
                f"{max_forms}; lower max_variable or narrow the variable mods"
            )

    forms: list[Peptidoform] = []
    for k in range(k_max + 1):
        for combo in itertools.combinations(ordered, k):
            for choice in itertools.product(*(candidates[s] for s in combo)):
                assigned = dict(fixed_slots)
                assigned.update(zip(combo, (_tagged(c) for c in choice), strict=True))
                residue_mods = {
                    slot: (tag,) for slot, tag in assigned.items() if isinstance(slot, int)
                }
                forms.append(
                    Peptidoform(
                        sequence=peptide,
                        residue_mods=residue_mods,
                        n_term_mods=(assigned[N_TERM],) if N_TERM in assigned else (),
                        c_term_mods=(assigned[C_TERM],) if C_TERM in assigned else (),
                    )
                )
    return forms
