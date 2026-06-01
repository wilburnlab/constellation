"""MaxQuant ``Modified sequence`` → ProForma 2.0 ``Peptidoform``.

MaxQuant writes a modseq as ``_<residues+inline mods>_`` where inline
*variable* mods are lowercase codes in parentheses after the modified
residue (``_AAM(ox)R_``); a parenthesised token *before* the first
residue is an N-terminal mod (``_(ac)MAAR_``). *Fixed* modifications
(Carbamidomethyl, etc.) are NOT in the modseq — they are reconstructed
from ``parameters.txt`` and applied to every matching site.

The ``Peptidoform`` is built directly (mirroring
:func:`massspec.io.msp._mods.build_peptidoform`) rather than routing
through ``parse_proforma`` — the latter pays Earley-parse cost per
peptide and can return a ``MultiPeptidoform``. Serialise the result with
``format_proforma`` to populate ``PSM_TABLE.modified_sequence``.
"""

from __future__ import annotations

import re

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import ModRef, Peptidoform, TaggedMod

from constellation.massspec.io.maxquant._abbrev import MAXQUANT_ABBREV_TO_UNIMOD


class MaxQuantModResolutionError(KeyError):
    """A MaxQuant inline code / fixed-mod name didn't resolve in the vocab."""


# Matches a parenthesised inline mod token, e.g. "(ox)" or "(Oxidation (M))".
# Greedy inner class allows nested parens in full-name tokens.
_TOKEN_RE = re.compile(r"\((?P<body>.*?)\)(?=[A-Z_(]|$)")

# Terminal-site spellings MaxQuant uses in ``parameters.txt``.
_N_TERM_SITES = frozenset({"n-term", "any n-term", "protein n-term"})
_C_TERM_SITES = frozenset({"c-term", "any c-term", "protein c-term"})


def _tagged_mod(name: str, vocab: ModVocab, *, context: str) -> TaggedMod:
    """Resolve a canonical UNIMOD ``name`` to ``TaggedMod(ModRef(UNIMOD:n))``."""
    try:
        mod = vocab.get(name)
    except KeyError as exc:
        raise MaxQuantModResolutionError(
            f"cannot resolve modification {name!r} ({context})"
        ) from exc
    _, _, accession = mod.id.partition(":")
    return TaggedMod(mod=ModRef(cv="UNIMOD", accession=accession, name=mod.name))


def _resolve_inline(token: str, vocab: ModVocab, modseq: str) -> TaggedMod:
    """Resolve an inline modseq token (2-letter code or full name)."""
    name = MAXQUANT_ABBREV_TO_UNIMOD.get(token.lower(), token)
    return _tagged_mod(name, vocab, context=f"inline {token!r} in {modseq!r}")


def _has_accession(tagged: list[TaggedMod], accession: str) -> bool:
    return any(
        t.mod is not None and t.mod.accession == accession for t in tagged
    )


def parse_maxquant_modseq(
    modseq: str,
    *,
    fixed_mods: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
    vocab: ModVocab = UNIMOD,
) -> Peptidoform:
    """Translate a MaxQuant modseq into a :class:`Peptidoform`.

    Parameters
    ----------
    modseq
        The MaxQuant ``Modified sequence`` value, with or without the
        flanking underscores (``_AAM(ox)R_`` or ``AAM(ox)R``).
    fixed_mods
        ``(unimod_name, site)`` pairs from
        :func:`._params.parse_fixed_modifications`. Each is applied to
        every matching residue / terminus that doesn't already carry it.
    vocab
        Modification vocabulary (default the bundled UNIMOD).

    Raises
    ------
    MaxQuantModResolutionError
        When an inline code or a fixed-mod name has no vocab entry.
    """
    s = modseq.strip()
    if s.startswith("_"):
        s = s[1:]
    if s.endswith("_"):
        s = s[:-1]

    residues: list[str] = []
    n_term_mods: list[TaggedMod] = []
    residue_mods: dict[int, list[TaggedMod]] = {}

    i = 0
    while i < len(s):
        c = s[i]
        if c == "(":
            m = _TOKEN_RE.match(s, i)
            if m is None:
                raise MaxQuantModResolutionError(
                    f"malformed mod token at index {i} in {modseq!r}"
                )
            tagged = _resolve_inline(m.group("body"), vocab, modseq)
            if not residues:
                # leading token → N-terminal mod
                n_term_mods.append(tagged)
            else:
                residue_mods.setdefault(len(residues) - 1, []).append(tagged)
            i = m.end()
        elif c.isalpha() and c.isupper():
            residues.append(c)
            i += 1
        else:
            raise MaxQuantModResolutionError(
                f"unexpected character {c!r} at index {i} in modseq {modseq!r}"
            )

    sequence = "".join(residues)
    c_term_mods: list[TaggedMod] = []
    _apply_fixed_mods(
        sequence,
        n_term_mods,
        c_term_mods,
        residue_mods,
        fixed_mods,
        vocab,
    )

    return Peptidoform(
        sequence=sequence,
        n_term_mods=tuple(n_term_mods),
        c_term_mods=tuple(c_term_mods),
        residue_mods={idx: tuple(mods) for idx, mods in residue_mods.items()},
    )


def _apply_fixed_mods(
    sequence: str,
    n_term_mods: list[TaggedMod],
    c_term_mods: list[TaggedMod],
    residue_mods: dict[int, list[TaggedMod]],
    fixed_mods: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    vocab: ModVocab,
) -> None:
    """Apply fixed modifications in place (residue + terminal sites).

    Idempotent per site: a site that already carries the modification's
    UNIMOD accession (e.g. an inline variable form of the same mod) is
    left untouched.
    """
    for name, site in fixed_mods:
        tagged = _tagged_mod(name, vocab, context=f"fixed modification {name!r}")
        accession = tagged.mod.accession if tagged.mod is not None else None
        site_norm = site.strip().lower()
        if site_norm in _N_TERM_SITES:
            if accession is None or not _has_accession(n_term_mods, accession):
                n_term_mods.append(tagged)
        elif site_norm in _C_TERM_SITES:
            if accession is None or not _has_accession(c_term_mods, accession):
                c_term_mods.append(tagged)
        elif len(site_norm) == 1 and site_norm.isalpha():
            residue = site_norm.upper()
            for idx, aa in enumerate(sequence):
                if aa != residue:
                    continue
                existing = residue_mods.get(idx, [])
                if accession is None or not _has_accession(existing, accession):
                    residue_mods.setdefault(idx, []).append(tagged)
        # Sites we don't recognise (empty / odd spellings) are skipped
        # rather than mis-placed — the modseq is still usable.


__all__ = [
    "MaxQuantModResolutionError",
    "parse_maxquant_modseq",
]
