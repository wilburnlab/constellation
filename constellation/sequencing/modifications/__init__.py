"""RNA / DNA base modifications — calling from BAM tags or signal.

The lab's near-term modifications work falls into two routes:

    1. Tag-based extraction from a Dorado BAM.
       Dorado emits MM (modification calls) and ML (modification
       likelihoods) BAM tags when called with ``--modified-bases``.
       :mod:`modifications.basemod` parses these into a long-format
       per-base call table for downstream analysis (m6A, m5C, ψ at
       per-position resolution).

    2. Direct-current analysis on raw POD5 signal.
       The lab's planned synthetic-library work targets RNA mods at
       the squiggle level — physical-model basecallers (see
       :mod:`signal.models.physical`) recover modifications more
       robustly than CNN basecallers because they parameterize the
       k-mer current shifts directly. Modules for this route land
       under :mod:`signal.models` rather than here.

This package focuses on route 1 today (Phase 11ish — tied to BAM tag
parsing). Status: scaffold.
"""

from __future__ import annotations

from constellation.sequencing.modifications.basemod import (
    BASEMOD_TABLE,
    extract_basemods,
)

__all__ = [
    "BASEMOD_TABLE",
    "extract_basemods",
]
