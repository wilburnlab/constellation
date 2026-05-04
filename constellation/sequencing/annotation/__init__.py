"""Genome annotation — the ``Annotation`` container plus runner stubs.

Two surfaces:

1. **The ``Annotation`` container** — Arrow ``FEATURE_TABLE`` rows with
   PK uniqueness on ``feature_id`` + parent-id closure + a
   ``validate_against(genome)`` cross-check into a paired
   ``GenomeReference``. Two origins: GFF3 import (paired with an
   external genome FASTA) and de novo annotation passes (BUSCO,
   RepeatMasker, telomere finders).

2. **Runner stubs** for de novo annotation tools that emit FEATURE_TABLE
   rows callers fold into a fresh ``Annotation``. Currently stubbed:
   ``BuscoRunner``, ``RepeatsRunner``, ``find_telomeres``,
   ``map_transcripts``.

The Annotation container is uniform across external-ref and de-novo-ref
workflows because both produce a ``GenomeReference`` (external FASTA
import or ``Assembly.to_genome_reference()``) — annotators don't need
to know which origin produced the genome.
"""

from __future__ import annotations

from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.annotation.busco import BuscoRunner
from constellation.sequencing.annotation.io import (
    ANNOTATION_READERS,
    ANNOTATION_WRITERS,
    AnnotationReader,
    AnnotationWriter,
    load_annotation,
    register_reader,
    register_writer,
    save_annotation,
)
from constellation.sequencing.annotation.repeats import RepeatsRunner
from constellation.sequencing.annotation.telomeres import find_telomeres
from constellation.sequencing.annotation.transcripts import map_transcripts

__all__ = [
    "Annotation",
    "AnnotationReader",
    "AnnotationWriter",
    "ANNOTATION_READERS",
    "ANNOTATION_WRITERS",
    "register_reader",
    "register_writer",
    "save_annotation",
    "load_annotation",
    "BuscoRunner",
    "RepeatsRunner",
    "find_telomeres",
    "map_transcripts",
]
