"""Token-level helpers for the mzPAF parser.

mzPAF v1.0 (HUPO-PSI, ratified Aug 2025) annotates a single peak with
one or more interpretations. The grammar is sequential — each
annotation is an ion-prefix followed by a fixed-order chain of
decorations (neutral losses, isotope, adduct, charge, mass error,
confidence). Multiple interpretations of one peak are comma-separated;
multi-analyte (chimeric) interpretations carry an ``N@`` prefix.

We tokenize regex-first and consume left-to-right rather than build a
full Lark grammar, because:

  - mzPAF's decoration order is rigid (no recursion, no precedence).
  - The grammar is small enough (~10 token classes) that a regex pass
    is more readable than a CFG.
  - We only need to round-trip annotations Constellation produces
    (peptide ion ladders) and read annotations external libraries
    supply — both well within the tokenizable subset.

Implemented in :mod:`mzpaf`; this module just exposes the regex bank
and the small enums used to discriminate ion types and decorations.
"""

from __future__ import annotations

import re
from enum import Enum

# ──────────────────────────────────────────────────────────────────────
# Ion-prefix taxonomy
# ──────────────────────────────────────────────────────────────────────


class IonClass(Enum):
    """The mzPAF v1.0 ion-prefix categories."""

    PEPTIDE = "peptide"  # a/b/c/x/y/z + position
    IMMONIUM = "immonium"  # I<residue>[<UNIMOD>]?
    INTERNAL = "internal"  # m<start>:<end>
    PRECURSOR = "precursor"  # p
    REPORTER = "reporter"  # r[<label>]
    NAMED = "named"  # _{<name>}
    FORMULA = "formula"  # f{<formula>}
    SMILES = "smiles"  # s{<smiles>}
    UNKNOWN = "unknown"  # ?


# ──────────────────────────────────────────────────────────────────────
# Mass-error unit
# ──────────────────────────────────────────────────────────────────────


class MassErrorUnit(Enum):
    DA = "Da"
    PPM = "ppm"


# ──────────────────────────────────────────────────────────────────────
# Regex bank
# ──────────────────────────────────────────────────────────────────────

# Peptide ion: letter (a/b/c/x/y/z) + integer position. Capture both.
RE_PEPTIDE_ION = re.compile(r"^([abcxyz])(\d+)")

# Immonium: I + single residue letter, optional [<modification>]
RE_IMMONIUM = re.compile(r"^I([A-Z])(\[[^\]]+\])?")

# Internal: m<start>:<end>
RE_INTERNAL = re.compile(r"^m(\d+):(\d+)")

# Precursor
RE_PRECURSOR = re.compile(r"^p")

# Reporter: r[<label>] — label is opaque (TMT127N, iTRAQ-114, etc.)
RE_REPORTER = re.compile(r"^r\[([^\]]+)\]")

# Named: _{name} (curly braces — note: some tools use square brackets,
# but the v1.0 ratified syntax uses curly).
RE_NAMED = re.compile(r"^_\{([^}]+)\}")

# Formula: f{Hill formula}
RE_FORMULA = re.compile(r"^f\{([^}]+)\}")

# SMILES: s{<smiles>}
RE_SMILES = re.compile(r"^s\{([^}]+)\}")

# Unknown peak
RE_UNKNOWN = re.compile(r"^\?")

# Multi-analyte prefix: <int>@
RE_ANALYTE_PREFIX = re.compile(r"^(\d+)@")

# Charge: ^N (signed N is allowed for negative-mode but rare; v1 ratified
# spec discusses positive charges by default — accept both via the int).
RE_CHARGE = re.compile(r"\^(-?\d+)")

# Isotope: +Ni or -Ni (i is the literal character).
RE_ISOTOPE = re.compile(r"([+-])(\d+)i")

# Adduct: [M+H], [M+H+Na], [M-H2O+H], etc.  Captures the bracketed body.
RE_ADDUCT = re.compile(r"\[(M[+\-][^\]]+)\]")

# Confidence: *<value> where value is a non-negative float.
RE_CONFIDENCE = re.compile(r"\*([0-9]+(?:\.[0-9]+)?)")

# Mass error: /<value>[ppm] — value may be signed; if "ppm" suffix
# absent, the unit is Da. Anchor the slash so we don't over-match URL-
# like substrings (won't appear inside annotations, but be defensive).
RE_MASS_ERROR = re.compile(r"/(-?\d+(?:\.\d+)?)(ppm)?")

# Neutral loss / gain: starts with + or - then either a chemical formula
# (e.g. H2O, CH4O, CO) or a bracket payload [Name] (e.g. [Phospho]).
# Multiple losses can chain: +CO-H2O.
# We match a single loss token at a time:
#   - sign      : +|-
#   - body      : either bracketed [<...>]
#                 or a Hill-style formula [A-Z][a-z]?\d* repeated
RE_LOSS_BRACKET = re.compile(r"^([+\-])\[([^\]]+)\]")
RE_LOSS_FORMULA = re.compile(r"^([+\-])((?:[A-Z][a-z]?\d*)+)")


__all__ = [
    "IonClass",
    "MassErrorUnit",
    "RE_ADDUCT",
    "RE_ANALYTE_PREFIX",
    "RE_CHARGE",
    "RE_CONFIDENCE",
    "RE_FORMULA",
    "RE_IMMONIUM",
    "RE_INTERNAL",
    "RE_ISOTOPE",
    "RE_LOSS_BRACKET",
    "RE_LOSS_FORMULA",
    "RE_MASS_ERROR",
    "RE_NAMED",
    "RE_PEPTIDE_ION",
    "RE_PRECURSOR",
    "RE_REPORTER",
    "RE_SMILES",
    "RE_UNKNOWN",
]
