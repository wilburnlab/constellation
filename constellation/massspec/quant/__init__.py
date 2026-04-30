"""Empirical / sample-specific MS observations.

Five Arrow tables paired with the theoretical structure in
``massspec.library``: per-tier abundance observations and per-edge
calibrated transmission efficiencies. Run provenance lives in the
shared ``massspec.acquisitions`` module.

Data here is what gets fit/measured from raw acquisitions: predictions
that are sequence-derivable and tied to entity identity (predicted RT,
predicted CCS, predicted fragment intensities) belong in the Library,
not here.

    schemas       PROTEIN_QUANT / PEPTIDE_QUANT / PRECURSOR_QUANT /
                  TRANSMISSION_PROTEIN_PEPTIDE /
                  TRANSMISSION_PEPTIDE_PRECURSOR
    quant         Quant container (strict efficiency ∈ (0, 1] on
                  populated rows; -1.0 = uncalibrated)
    io            QuantReader / QuantWriter Protocols + registry +
                  native ParquetDirReader/Writer. EncyclopeDIA
                  ``.elib`` / ``.dlib`` adapters live in
                  :mod:`massspec.io.encyclopedia` and self-register
                  on package import.
"""

from constellation.massspec.quant.io import (
    QUANT_READERS,
    QUANT_WRITERS,
    ParquetDirReader,
    ParquetDirWriter,
    QuantReader,
    QuantWriter,
    load_quant,
    register_reader,
    register_writer,
    save_quant,
)
from constellation.massspec.quant.quant import Quant, assemble_quant
from constellation.massspec.quant.schemas import (
    PEPTIDE_QUANT,
    PRECURSOR_QUANT,
    PROTEIN_QUANT,
    TRANSMISSION_PEPTIDE_PRECURSOR,
    TRANSMISSION_PROTEIN_PEPTIDE,
)

__all__ = [
    "Quant",
    "assemble_quant",
    "PROTEIN_QUANT",
    "PEPTIDE_QUANT",
    "PRECURSOR_QUANT",
    "TRANSMISSION_PROTEIN_PEPTIDE",
    "TRANSMISSION_PEPTIDE_PRECURSOR",
    "QuantReader",
    "QuantWriter",
    "QUANT_READERS",
    "QUANT_WRITERS",
    "register_reader",
    "register_writer",
    "save_quant",
    "load_quant",
    "ParquetDirReader",
    "ParquetDirWriter",
]
