"""NMR I/O — readers and schemas for instrument-native formats.

Modules:
    schemas    `NMR_FID_TABLE` — ragged 1D, two rows (real/imag) per FID
    bruker     `BrukerReader` — Bruker TopSpin 1D experiment directories

The Bruker reader does NOT register with ``core.io.readers``' suffix-based
dispatch because Bruker experiments are *directories* with conventional
file names (``acqus``, ``fid``), not single files with extensions. Call
``BrukerReader().read(path)`` directly. When a second NMR vendor format
lands (Varian/Agilent ``.fid``, JEOL ``.jdf``, ...), suffix-based dispatch
may earn its keep.
"""

from __future__ import annotations

from constellation.nmr.io.bruker import BrukerReader
from constellation.nmr.io.schemas import NMR_FID_TABLE

__all__ = ["BrukerReader", "NMR_FID_TABLE"]
