"""Default PTM toggle map for EncyclopeDIA predict-library invocations.

Both ``constellation massspec predict-library`` and
``constellation transcriptome-to-proteome`` import from here so the two
CLI surfaces cannot drift on PTM defaults. Prior to this module the two
CLIs maintained their own defaults dictionaries and silently disagreed
on ``ProteinNTermAcetyl`` / ``PyroGluQ`` / ``Oxidation`` — the
orchestrator left all three off while the standalone CLI matched the
EncyclopeDIA jar's own defaults. Users hitting the orchestrator path
got an under-mod-ed library and saw correspondingly lower peptide IDs.

Per lab convention for DIA proteomics the standard variable-mod set is:

- ``Carbamidomethyl`` (C) **fixed**
- ``ProteinNTermAcetyl`` **variable**
- ``PyroGluQ`` **variable**
- ``Oxidation`` (M) **variable**
- everything else off

This DIVERGES from EncyclopeDIA's jar defaults on one point: the jar
leaves ``Oxidation`` off. Constellation promotes M-ox to a default-on
variable mod because virtually every published DIA-proteomics workflow
runs with it enabled, and the prior "off by default" was a footgun.
"""

from __future__ import annotations

from typing import Final

PTM_NAMES: Final[tuple[str, ...]] = (
    "Acetyl",
    "ProteinNTermAcetyl",
    "Carbamidomethyl",
    "Deamidation",
    "Dimethyl",
    "GlyGly",
    "HexNAc",
    "Methyl",
    "Oxidation",
    "Phospho",
    "PyroGluQ",
    "Succinyl",
    "Trimethyl",
    "TMT",
)

PTM_DEFAULTS: Final[dict[str, str]] = {
    "Carbamidomethyl":    "fix",
    "ProteinNTermAcetyl": "var",
    "PyroGluQ":           "var",
    "Oxidation":          "var",
}


def default_for(ptm_name: str) -> str:
    """Return the constellation-wide default toggle (``off|var|fix``) for
    a PTM name; raises ``KeyError`` for unknown names so a typo can't
    silently fall through to ``off``."""
    if ptm_name not in PTM_NAMES:
        raise KeyError(
            f"unknown PTM {ptm_name!r}; expected one of {PTM_NAMES}"
        )
    return PTM_DEFAULTS.get(ptm_name, "off")
