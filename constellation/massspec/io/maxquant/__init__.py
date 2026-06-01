"""MaxQuant ``combined/txt/`` file-format adapter (search view).

MaxQuant exports its database-search results as a directory of TSV
tables (``msms.txt``, ``evidence.txt``, ``peptides.txt``, ...). v1 reads
``msms.txt`` (one row per peptide-spectrum match) into a :class:`Search`
whose ``psms`` table is populated, plus ``parameters.txt`` as provenance.
This is the first genuinely per-spectrum source driving ``PSM_TABLE``.

Reader only — Constellation does not run MaxQuant. The reader is
format-name-addressed (``load_search(path, format="maxquant")``); see
:mod:`.adapters` for why. Modseqs (``_AAM(ox)R_``) normalise to ProForma
2.0 via :mod:`._modseq`, reconstructing fixed mods from ``parameters.txt``.

Deferred (v2): ``evidence.txt`` → Quant intensities; ``peptides.txt`` /
``proteinGroups.txt`` rollups; ``Matches`` / ``Intensities`` fragment
annotations.
"""

from __future__ import annotations

# Trigger registration in SEARCH_READERS.
from constellation.massspec.io.maxquant import adapters as adapters  # noqa: F401
from constellation.massspec.io.maxquant._modseq import (
    MaxQuantModResolutionError,
    parse_maxquant_modseq,
)
from constellation.massspec.io.maxquant._params import (
    parse_fixed_modifications,
    parse_parameters_txt,
)
from constellation.massspec.io.maxquant._read import read_maxquant_search

__all__ = [
    "MaxQuantModResolutionError",
    "parse_fixed_modifications",
    "parse_maxquant_modseq",
    "parse_parameters_txt",
    "read_maxquant_search",
]
