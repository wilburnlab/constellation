"""Genome annotation — BUSCO genes, repeats, telomeres, transcript mapping.

Each submodule is a thin wrapper over an external annotation tool that
emits ``FEATURE_TABLE``-shaped rows. The annotation pass is uniform
across external-ref and de-novo-ref workflows because both expose a
``Reference`` (Assembly.to_reference()) — annotators don't need to
know which origin produced the ``Reference``.

Modules:

    busco          BUSCO completeness + ortholog gene calls
    repeats        Repeat-element detection (RepeatMasker / EDTA — TBD)
    telomeres      Telomere / centromere / tandem-repeat detection
    transcripts    Map cDNA reads to genome → transcript features

Each runner returns ``FEATURE_TABLE`` rows that callers append to
``Reference.features`` (via ``Reference.with_features`` — Phase 9).

Status: STUB. All implementations pending Phase 9.
"""

from __future__ import annotations

from constellation.sequencing.annotation.busco import BuscoRunner
from constellation.sequencing.annotation.repeats import RepeatsRunner
from constellation.sequencing.annotation.telomeres import find_telomeres
from constellation.sequencing.annotation.transcripts import map_transcripts

__all__ = [
    "BuscoRunner",
    "RepeatsRunner",
    "find_telomeres",
    "map_transcripts",
]
