"""MaxQuant inline modification abbreviations → canonical UNIMOD names.

MaxQuant 1.5.x writes *variable* modifications inline in the
``Modified sequence`` column as lowercase two-letter codes in parentheses
immediately after the modified residue (e.g. ``M(ox)`` for oxidised
methionine, ``(ac)`` before the first residue for N-terminal acetyl).
*Fixed* modifications are NOT written into the modseq — they are
reconstructed from ``parameters.txt`` (see :mod:`._modseq`).

This table maps the legacy two-letter code to a canonical UNIMOD name
that ``UNIMOD.get`` already aliases. Newer MaxQuant emits the full
modification name inline (e.g. ``(Oxidation (M))``); the modseq parser
falls back to a direct ``UNIMOD.get(token)`` for tokens absent from this
table, so it only has to cover the short codes.
"""

from __future__ import annotations

# code → canonical UNIMOD name (the accession in the trailing comment is
# informational; resolution goes through ``UNIMOD.get(name)``).
MAXQUANT_ABBREV_TO_UNIMOD: dict[str, str] = {
    "ox": "Oxidation",  # UNIMOD:35
    "ac": "Acetyl",  # UNIMOD:1
    "ph": "Phospho",  # UNIMOD:21
    "de": "Deamidated",  # UNIMOD:7  (MaxQuant "Deamidation (NQ)")
    "me": "Methyl",  # UNIMOD:34
    "di": "Dimethyl",  # UNIMOD:36
    "tr": "Trimethyl",  # UNIMOD:37
    "gl": "Gln->pyro-Glu",  # UNIMOD:28
    "ca": "Carbamidomethyl",  # UNIMOD:4 (rarely inline; usually fixed)
}

__all__ = ["MAXQUANT_ABBREV_TO_UNIMOD"]
