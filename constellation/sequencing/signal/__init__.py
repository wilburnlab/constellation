"""POD5 signal processing — squiggle normalization, segmentation, models.

The bridge between raw int16 ADC counts (``RAW_SIGNAL_TABLE``) and
basecaller input. Today this is a thin translation layer over what
Dorado does internally; the lab's research direction is custom
basecallers built from physical models of nanopore translocation +
k-mer-current effects, which will live under :mod:`signal.models`.

Modules:

    normalize     scale/offset application; mad-norm; downstream
                  ML-friendly conversions
    segment       squiggle segmentation (deferred — depends on basecaller
                  research direction)
    models/       custom basecaller-model ABCs (deferred research stub)

Status: STUB. Implementations pending Phase 13 (research area).
"""

from __future__ import annotations

from constellation.sequencing.signal.normalize import (
    mad_normalize,
    raw_to_picoamperes,
)

__all__ = [
    "raw_to_picoamperes",
    "mad_normalize",
]
