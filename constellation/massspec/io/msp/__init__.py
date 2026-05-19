"""NIST .msp file-format adapter (library view).

NIST .msp is the classical text-based container for mass spectra and
spectral libraries; in proteomics it is widely used as an interchange
format for annotated peptide libraries (ProteomeTools, MaxQuant /
MSFragger / MSPepSearch exports, ...). The format is a generic
spectra container — entries may be annotated library spectra (peptide
ID + assigned fragments) OR raw spectra (free-text label, no peptide).

This module ships the **library view** only: ``read_msp_library``
produces a :class:`Library` from MSP entries that carry a peptide
identification, and raises ``ValueError`` when an entry can't yield
one. A future ``read_msp_spectra`` reader will plug alongside when a
``RawSpectra`` container ships in ``massspec``.

Mod-name resolution flows through the project's single canonical
sequence path: NIST ``Mods=`` entries → ProForma 2.0 string →
``parse_proforma`` → ``Peptidoform``. Modseq strings populate
``PEPTIDE_TABLE.modified_sequence`` via ``format_proforma``.
"""

from __future__ import annotations

# Trigger registration in LIBRARY_READERS
from constellation.massspec.io.msp import adapters as adapters  # noqa: F401
from constellation.massspec.io.msp._read import read_msp_library

__all__ = ["read_msp_library"]
