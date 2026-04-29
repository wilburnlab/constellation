"""Annotation and identifier types for mass-spectrometry data.

This package holds peak-level annotations (mzPAF) and spectrum-level
identifiers (USI) — both are HUPO-PSI standards that reference
peptidoforms and ion types from the broader `massspec` domain. The
package sits at the same conceptual level as `massspec.peptide` because
mzPAF can describe non-peptide ions (reporters, formula tokens, named
ions) and multi-analyte chimeric annotations that are broader than
any single peptidoform.

Per the PSI-bridge architectural invariant (CLAUDE.md #2), these are
the canonical *external* representations; in-memory data continues to
live in flat Arrow columns (`ion_type`, `position`, `charge`, etc. on
`LIBRARY_FRAGMENT_TABLE` / `FRAGMENT_ION_TABLE`). mzPAF strings are
emitted at write time via the projection helpers in `mzpaf`.

Public API:

    USI                    Spectrum identifier (mzspec:...:scan:N:proforma/Z)
    parse_usi              str -> USI
    USIError               Base class for USI parse errors

    Annotation             One mzPAF annotation (single ion, one analyte)
    PeakAnnotation         Full mzPAF parse for one peak (alternates + multi-analyte)
    parse_mzpaf            str -> PeakAnnotation
    format_mzpaf           PeakAnnotation -> str
    MzPAFError             Base class for mzPAF parse errors

    fragment_row_to_mzpaf      dict-like fragment row -> mzPAF string
    fragment_table_to_mzpaf    pa.Table -> pa.Array of mzPAF strings
"""

from __future__ import annotations

from constellation.massspec.annotation.mzpaf import (
    Annotation,
    MzPAFError,
    MzPAFSyntaxError,
    PeakAnnotation,
    format_mzpaf,
    fragment_row_to_mzpaf,
    fragment_table_to_mzpaf,
    parse_mzpaf,
)
from constellation.massspec.annotation.usi import (
    USI,
    USIError,
    USISyntaxError,
    parse_usi,
)

__all__ = [
    "Annotation",
    "MzPAFError",
    "MzPAFSyntaxError",
    "PeakAnnotation",
    "USI",
    "USIError",
    "USISyntaxError",
    "format_mzpaf",
    "fragment_row_to_mzpaf",
    "fragment_table_to_mzpaf",
    "parse_mzpaf",
    "parse_usi",
]
