"""EncyclopeDIA ``PeptideModSeq`` ↔ ProForma 2.0 translation.

Two directions:

  * ``parse_encyclopedia_modseq(modseq) -> Peptidoform`` — the reader path.
    Translates EncyclopeDIA's ``X[+N.NNN]`` mass-delta notation into a
    canonical ``Peptidoform``, including N-terminal-vs-side-chain
    disambiguation via UNIMOD specificity lookup. EncyclopeDIA collapses
    ``[Mod]-X`` and ``X[Mod]`` into one ``X[Mod]`` form, but they are
    chemically distinct (e.g. K[Acetyl]VERPD vs [Acetyl]-KVERPD —
    lysine ε-amine vs N-terminal α-amine acetylation). The reader is
    the right boundary to fix this; every consumer downstream of
    ``PEPTIDE_TABLE.modified_sequence`` then sees an unambiguous
    ProForma 2.0 string (built via ``format_proforma(peptidoform)``).

  * ``format_encyclopedia_modseq(peptidoform) -> str`` — the writer path.
    Inverse direction. Emits ``X[+N.NNN]`` notation matching cartographer's
    serialization. Lossy on the chemical-fidelity axis: ``[Mod]-X`` collapses
    onto ``X[Mod]`` (residue 0) because EncyclopeDIA cannot represent the
    N-terminal-vs-side-chain distinction. This is why
    ``EncyclopediaWriter.lossy = True``.

Module is pure file-format translation — no JAR invocation, no shell-out,
no third-party tool dependency. Lives next to the rest of the encyclopedia
file-format code in ``massspec.io.encyclopedia``.
"""

from __future__ import annotations

import re

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import (
    ModRef,
    Peptidoform,
    TaggedMod,
)

# ──────────────────────────────────────────────────────────────────────
# parse: EncyclopeDIA modseq → Peptidoform
# ──────────────────────────────────────────────────────────────────────

# EncyclopeDIA modseq grammar (informal):
#   - Bare uppercase residues
#   - Mass-delta blocks ``[+N.NNN]`` or ``[-N.NNN]`` (signed) immediately
#     follow the residue they decorate
#   - No N-term-prefix-style mod blocks; everything is residue-attached
#
# Examples (right column shows format_proforma() of the returned Peptidoform):
#   PEPTIDE                            → "PEPTIDE"
#   PEPC[+57.02146]TIDE                 → "PEPC[UNIMOD:4]TIDE" (Cam on C)
#   K[+42.01057]VERPD                   → "[UNIMOD:1]-KVERPD"  (N-term Acetyl on A — see below)
#   PEPK[+42.01057]TIDE                 → "PEPK[UNIMOD:1]TIDE" (K side-chain Ac)
#   PEPS[+79.96633]TIDE                 → "PEPS[UNIMOD:21]TIDE" (Phospho on S)

_RE_MOD_BLOCK = re.compile(r"\[([+-]?\d+(?:\.\d+)?)\]")


def parse_encyclopedia_modseq(
    modseq: str,
    *,
    vocab: ModVocab = UNIMOD,
    tolerance_da: float = 1e-3,
) -> Peptidoform:
    """Translate EncyclopeDIA-style ``X[+N.NNN]`` → ``Peptidoform``.

    Rules:

      1. Walk the modseq residue-by-residue. When an ``[+N.NNN]`` block
         follows a residue, look up the mass delta in ``vocab``.
      2. If the delta resolves to exactly one UNIMOD modification:
           a. If the residue is at index 0 AND the modification has an
              N-terminal specificity AND the modification CANNOT attach
              to the residue's side chain, render as a terminal
              modification (``[UNIMOD:N]-...``).
           b. Otherwise render as a residue modification (``X[UNIMOD:N]``).
      3. If the delta does not resolve to UNIMOD (no match within
         ``tolerance_da``), pass through as a ProForma mass-delta
         ``ModRef`` (``X[+N.NNN]``) — ProForma 2.0 §4.5.2 supports bare
         deltas.
      4. Multiple UNIMOD candidates: prefer the one whose specificity
         best matches the residue position. Ties leave the modseq as a
         ProForma mass-delta with no claim of identity.

    The function is pure and deterministic; the reader passes its output
    through ``format_proforma(peptidoform)`` to populate
    ``PEPTIDE_TABLE.modified_sequence``.
    """
    if not modseq:
        return Peptidoform(sequence="")

    # Tokenize into (residue, mod_block_or_None) pairs.
    tokens: list[tuple[str, str | None]] = []
    i = 0
    while i < len(modseq):
        c = modseq[i]
        if c.isalpha() and c.isupper():
            residue = c
            i += 1
            mod_str: str | None = None
            if i < len(modseq) and modseq[i] == "[":
                m = _RE_MOD_BLOCK.match(modseq, i)
                if m is None:
                    raise ValueError(
                        f"malformed mod block at index {i} in {modseq!r}"
                    )
                mod_str = m.group(1)
                i = m.end()
            tokens.append((residue, mod_str))
        else:
            raise ValueError(
                f"unexpected character {c!r} at index {i} in {modseq!r}; "
                "EncyclopeDIA modseq must start each residue with an uppercase letter"
            )

    if not tokens:
        return Peptidoform(sequence="")

    sequence = "".join(r for r, _ in tokens)
    n_term_mods: list[TaggedMod] = []
    residue_mods: dict[int, list[TaggedMod]] = {}

    for idx, (residue, mod_str) in enumerate(tokens):
        if mod_str is None:
            continue
        delta = float(mod_str)
        matches = vocab.find_by_mass(delta, tolerance_da=tolerance_da)
        chosen, place_terminal = _choose_modification(matches, residue, idx)
        if chosen is None:
            # Ambiguous or no match → pass through as a ProForma mass-delta.
            modref = ModRef(mass_delta=delta)
        else:
            # ``chosen.id`` is the full canonical id ("UNIMOD:35"); split to
            # populate ``accession`` separately so format_proforma re-emits the
            # canonical form.
            _, _, accession = chosen.id.partition(":")
            modref = ModRef(cv="UNIMOD", accession=accession, name=chosen.name)
        tagged = TaggedMod(mod=modref)
        if place_terminal:
            n_term_mods.append(tagged)
        else:
            residue_mods.setdefault(idx, []).append(tagged)

    return Peptidoform(
        sequence=sequence,
        n_term_mods=tuple(n_term_mods),
        residue_mods={k: tuple(v) for k, v in residue_mods.items()},
    )


def _choose_modification(matches, residue: str, position: int):
    """Pick a UNIMOD modification + decide N-term-vs-side-chain placement.

    Returns ``(modification, place_at_n_terminus)`` or ``(None, False)``
    when no unambiguous choice can be made.
    """
    if not matches:
        return None, False

    compatible = [m for m in matches if _is_compatible(m, residue, position)]
    if not compatible:
        return None, False

    if len(compatible) > 1:
        # Multiple chemically valid candidates of identical mass —
        # defer to ProForma mass-delta to avoid silent misassignment.
        return None, False

    mod = compatible[0]

    if position == 0 and mod.has_n_term_specificity:
        # If the residue cannot host the modification on its side chain,
        # terminal placement is the only chemically valid rendering.
        if not mod.has_residue_specificity(residue):
            return mod, True
        # Both placements are chemically valid; default to side-chain
        # because EncyclopeDIA's literal rendering is X[Mod] and we
        # have no localization evidence to override.
        return mod, False

    return mod, False


def _is_compatible(mod, residue: str, position: int) -> bool:
    """True iff `mod` could chemically attach at (residue, position).

    A modification with empty specificities is treated as **not**
    compatible — many UNIMOD substitution / artefact entries collide on
    mass with canonical biological PTMs but carry no real-world
    placement evidence. Excluding them here is what makes mass-delta
    lookup unambiguous in the common case.
    """
    if not mod.specificities:
        return False
    if mod.has_residue_specificity(residue):
        return True
    if position == 0 and mod.has_n_term_specificity:
        return True
    return False


# ──────────────────────────────────────────────────────────────────────
# format: Peptidoform → EncyclopeDIA modseq
# ──────────────────────────────────────────────────────────────────────


def format_encyclopedia_modseq(
    peptidoform: Peptidoform,
    *,
    vocab: ModVocab = UNIMOD,
) -> str:
    """Inverse of ``parse_encyclopedia_modseq`` — emit ``X[+N.NNN]`` notation.

    Lossy: terminal mods (``[Mod]-X...``) collapse onto residue 0 as
    ``X[+mass]...`` because EncyclopeDIA cannot distinguish terminal
    placement from side-chain placement on the same residue. This is
    documented on ``EncyclopediaWriter.lossy = True``.

    Rejects ProForma features that have no EncyclopeDIA equivalent —
    formula-only mods, glycans, branches, cross-links, multi-chain
    peptidoforms — with a ``NotImplementedError`` carrying a clear message.
    """
    if peptidoform.c_term_mods:
        raise NotImplementedError(
            "C-terminal modifications cannot be expressed in EncyclopeDIA "
            "modseq notation; the format only supports residue-attached "
            f"deltas. Got: {peptidoform.c_term_mods!r}"
        )
    if peptidoform.fixed_mods:
        raise NotImplementedError(
            "Global fixed-mod declarations have no EncyclopeDIA equivalent; "
            "expand them into per-residue modifications before formatting."
        )
    if peptidoform.global_isotopes:
        raise NotImplementedError(
            "Global isotope labels have no EncyclopeDIA equivalent."
        )
    if peptidoform.labile_mods or peptidoform.unknown_pos_mods or peptidoform.ranges:
        raise NotImplementedError(
            "Labile / unknown-position / ranged modifications have no "
            "EncyclopeDIA equivalent."
        )

    seq = peptidoform.sequence
    if not seq:
        return ""

    # Collect deltas per position. N-term mods fold onto residue 0.
    pos_deltas: dict[int, list[float]] = {}
    for tm in peptidoform.n_term_mods:
        pos_deltas.setdefault(0, []).append(_modref_delta(tm.mod, vocab))
    for pos, mods in peptidoform.residue_mods.items():
        for tm in mods:
            pos_deltas.setdefault(pos, []).append(_modref_delta(tm.mod, vocab))

    parts: list[str] = []
    for i, residue in enumerate(seq):
        parts.append(residue)
        if i in pos_deltas:
            for delta in pos_deltas[i]:
                sign = "+" if delta >= 0 else "-"
                parts.append(f"[{sign}{abs(delta):.7f}]")
    return "".join(parts)


def _modref_delta(modref: ModRef | None, vocab: ModVocab) -> float:
    """Resolve a ``ModRef`` to its monoisotopic mass delta.

    UNIMOD references look up the canonical entry; mass-delta-only
    ``ModRef``s pass through. Anything else (formula-only, glycan,
    cross-link / branch group) raises ``NotImplementedError``.
    """
    if modref is None:
        raise NotImplementedError(
            "Empty ModRef has no EncyclopeDIA-renderable mass delta."
        )
    if modref.mass_delta is not None:
        return float(modref.mass_delta)
    if modref.accession is not None:
        cv = (modref.cv or "UNIMOD").upper()
        if cv != "UNIMOD":
            raise NotImplementedError(
                f"Only UNIMOD CV references can be rendered as EncyclopeDIA "
                f"mass deltas; got CV {modref.cv!r}."
            )
        try:
            mod = vocab.get(f"UNIMOD:{modref.accession}")
        except KeyError as exc:
            raise NotImplementedError(
                f"UNIMOD:{modref.accession} not present in the active "
                "vocabulary; cannot render as EncyclopeDIA mass delta."
            ) from exc
        return mod.delta_composition.mass
    if modref.name is not None:
        try:
            mod = vocab.get(modref.name)
        except KeyError as exc:
            raise NotImplementedError(
                f"Modification name {modref.name!r} not in the active "
                "vocabulary; cannot render as EncyclopeDIA mass delta."
            ) from exc
        return mod.delta_composition.mass
    if modref.formula is not None:
        raise NotImplementedError(
            "Formula-only modifications cannot be rendered as a single "
            "EncyclopeDIA mass delta. Resolve to a UNIMOD entry first."
        )
    raise NotImplementedError(
        f"Cannot render {modref!r} as an EncyclopeDIA mass delta — "
        "no accession, name, or mass_delta to draw from."
    )


__all__ = [
    "parse_encyclopedia_modseq",
    "format_encyclopedia_modseq",
]
