"""MS raw-data readers — ``RawReader`` subclasses that self-register.

Format-level decoders for a single vendor / instrument file. Importing
this package triggers the ``@register_reader`` side effect on each
submodule so suffix-based lookup via :func:`core.io.readers.find_reader`
resolves the right class for a raw acquisition.

Modules:

    thermo       Thermo ``.raw`` → scan/peak tables (via
                 ``thirdparty.thermo``); ``ThermoReader`` plus the
                 ``convert`` / ``convert_batch`` parquet-bundle writers.

Modules (TODO; not yet shipped): mzml, bruker_d, mzpeak.

Cross-tier container adapters — formats that span Library + Quant +
Search tiers (``.msp`` libraries, MaxQuant search results, EncyclopeDIA
``.dlib`` / ``.elib``) — live in the sibling :mod:`massspec.io` package,
mirroring the ``sequencing.readers`` / ``sequencing.io`` split.
"""

from __future__ import annotations

from constellation.massspec.readers import thermo as thermo  # noqa: F401

__all__ = ["thermo"]
