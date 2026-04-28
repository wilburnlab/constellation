"""Theoretical / sample-agnostic spectral libraries.

Five Arrow tables modelling the protein → peptide → precursor → fragment
DAG plus the only true M:N edge (protein ↔ peptide). Sample-specific
observations and calibrated efficiencies live in ``massspec.quant``;
search results / PSMs / FDR live in ``massspec.search``. A higher-level
composition layer that unifies all three is intentionally deferred.

    schemas       PROTEIN_TABLE / PEPTIDE_TABLE / PRECURSOR_TABLE /
                  LIBRARY_FRAGMENT_TABLE / PROTEIN_PEPTIDE_EDGE
    library       Library container + assign_ids builder
    io            LibraryReader / LibraryWriter Protocols + registry +
                  native ParquetDirReader/Writer; DlibReader/Writer
                  stubs raise NotImplementedError pending the
                  EncyclopeDIA reader port.
"""

from constellation.massspec.library.io import (
    LIBRARY_READERS,
    LIBRARY_WRITERS,
    DlibReader,
    DlibWriter,
    LibraryReader,
    LibraryWriter,
    ParquetDirReader,
    ParquetDirWriter,
    load_library,
    register_reader,
    register_writer,
    save_library,
)
from constellation.massspec.library.library import Library, assign_ids
from constellation.massspec.library.schemas import (
    LIBRARY_FRAGMENT_TABLE,
    PEPTIDE_TABLE,
    PRECURSOR_TABLE,
    PROTEIN_PEPTIDE_EDGE,
    PROTEIN_TABLE,
)

__all__ = [
    "Library",
    "assign_ids",
    "PROTEIN_TABLE",
    "PEPTIDE_TABLE",
    "PRECURSOR_TABLE",
    "LIBRARY_FRAGMENT_TABLE",
    "PROTEIN_PEPTIDE_EDGE",
    "LibraryReader",
    "LibraryWriter",
    "LIBRARY_READERS",
    "LIBRARY_WRITERS",
    "register_reader",
    "register_writer",
    "save_library",
    "load_library",
    "ParquetDirReader",
    "ParquetDirWriter",
    "DlibReader",
    "DlibWriter",
]
