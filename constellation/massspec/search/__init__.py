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
    novel    -- classify_novel_peptides + classify_single_peptide
                (CIGAR-walking classifier; port of cartographer's
                 nanopore.classify_novel_peptides)
"""

from __future__ import annotations

from constellation.massspec.search import schemas as schemas  # noqa: F401  (registers schemas)
from constellation.massspec.search.novel import (
    _CLASSIFICATION_PRIORITY,
    build_gene_map_from_fasta_headers,
    classify_novel_peptides,
    classify_single_peptide,
    read_fasta_proteins,
    save_novel_peptides,
)
from constellation.massspec.search.search import Search, assemble_search

__all__ = [
    "Search",
    "_CLASSIFICATION_PRIORITY",
    "assemble_search",
    "build_gene_map_from_fasta_headers",
    "classify_novel_peptides",
    "classify_single_peptide",
    "read_fasta_proteins",
    "save_novel_peptides",
]
