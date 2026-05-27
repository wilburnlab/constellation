"""Search-engine results — peptide-level / protein-level scores.

Detection-side sibling to :mod:`massspec.library` (theoretical /
sample-agnostic) and :mod:`massspec.quant` (empirical / per-acquisition
abundances). A search engine consumes a ``Library`` plus an
acquisition's spectra and produces a ``Search`` record carrying scores
(percolator scores, q-values, posterior error probabilities) for the
peptides and proteins it identified.

This release ships peptide-level and protein-level score tables — the
shape encyclopedia ``.dlib``/``.elib`` files aggregate to. PSM-level
(one row per spectrum) is held until a real PSM-emitting reader
(mzIdentML, Counter port) drives the schema design; manufacturing PSMs
from peptide-level encyclopedia scores would invent data.

Modules:

    schemas  -- PEPTIDE_SCORE_TABLE, PROTEIN_SCORE_TABLE, NOVEL_PEPTIDE_TABLE
    search   -- Search container (mirrors Library / Quant)
    io       -- Reader/Writer Protocols + ParquetDirReader/Writer
    collision -- apply_collision_filter (opt-in DIA co-elution filter;
                 port of cartographer's filter_elib_by_collision)
"""

from __future__ import annotations

from constellation.massspec.search import schemas as schemas  # noqa: F401  (registers schemas)
from constellation.massspec.search.collision import (
    apply_collision_filter,
    filter_elib_by_losers,
)
from constellation.massspec.search.io import save_search
from constellation.massspec.search.search import Search, assemble_search

__all__ = [
    "Search",
    "apply_collision_filter",
    "assemble_search",
    "filter_elib_by_losers",
    "save_search",
]
