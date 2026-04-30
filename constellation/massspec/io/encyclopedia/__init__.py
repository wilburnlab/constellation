"""EncyclopeDIA file-format adapter — ``.dlib`` / ``.elib`` / ``.dia``.

Three SQLite-based file formats produced and consumed by EncyclopeDIA:

  * ``.dlib`` — predicted spectral library; sample-agnostic.
  * ``.elib`` — chromatogram library; same SQLite schema as ``.dlib`` with
    additional tables / columns populated for sample-specific data.
    By default EncyclopeDIA writes the full schema and only populates
    the dlib-relevant fields, so format detection happens at the
    populated-column level, not at the file extension.
  * ``.dia`` — raw spectra container (precursor + spectra + ranges + metadata)
    that EncyclopeDIA materialises from an mzML on first load.

The reader produces an ``EncyclopediaReadResult`` carrying ``Library``
(always), ``Quant`` (if sample-specific data is present), and ``Search``
(if percolator-style q-values / scores are present). The writer goes the
other way; the output extension is purely a naming convention. The
``DiaReader`` plugs into the ``core.io.RawReader`` registry for ``.dia``
inputs.

Modseq translation (EncyclopeDIA's ``X[+N.NNN]`` ↔ ProForma 2.0) is in
:mod:`._modseq`. See ``parse_encyclopedia_modseq`` /
``format_encyclopedia_modseq``.
"""

from __future__ import annotations

# Trigger registration in LIBRARY_/QUANT_/SEARCH_ READERS+WRITERS
from constellation.massspec.io.encyclopedia import adapters as adapters  # noqa: F401
from constellation.massspec.io.encyclopedia._dia import DiaReader
from constellation.massspec.io.encyclopedia._modseq import (
    format_encyclopedia_modseq,
    parse_encyclopedia_modseq,
)
from constellation.massspec.io.encyclopedia._read import (
    EncyclopediaReadResult,
    read_encyclopedia,
)
from constellation.massspec.io.encyclopedia._write import (
    WRITER_VERSION,
    write_encyclopedia,
)

__all__ = [
    "parse_encyclopedia_modseq",
    "format_encyclopedia_modseq",
    "read_encyclopedia",
    "write_encyclopedia",
    "EncyclopediaReadResult",
    "DiaReader",
    "WRITER_VERSION",
]
