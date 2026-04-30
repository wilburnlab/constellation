"""Reference container — contigs + sequences + features.

A ``Reference`` is the sequencing analog to ``massspec.library.Library``:
PK uniqueness, FK closure across contigs / sequences / features, and a
ParquetDir lossless native form. Two origins flow into this shape:

    External reference   FASTA + GFF → Reference (e.g. an Ensembl
                         genome download)
    De novo assembly     ``sequencing.assembly.Assembly.to_reference()``
                         lifts a finalized assembly into the same
                         shape so downstream alignment / quant code is
                         uniform across both workflows
"""

from __future__ import annotations

from constellation.sequencing.reference.io import (
    REFERENCE_READERS,
    REFERENCE_WRITERS,
    ReferenceReader,
    ReferenceWriter,
    load_reference,
    register_reader,
    register_writer,
    save_reference,
)
from constellation.sequencing.reference.reference import Reference

__all__ = [
    "Reference",
    "ReferenceReader",
    "ReferenceWriter",
    "REFERENCE_READERS",
    "REFERENCE_WRITERS",
    "register_reader",
    "register_writer",
    "save_reference",
    "load_reference",
]
