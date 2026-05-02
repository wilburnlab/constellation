"""Sequencing domain module — Oxford Nanopore long-read pipelines.

Absorbs **NanoporeAnalysis** as a clean rewrite. Imports ``core`` only;
never cross-imports to other domain modules. Cross-modality workflows
(transcriptome → spectral library) live as top-level scripts.

Lab workflows the module serves:

    Genomic DNA       POD5 → Dorado → HiFiASM → polish → RagTag →
                      BUSCO + repeat / telomere annotation
    Transcriptomic    POD5 → Dorado → segmented edlib demux →
                      mmseqs k-mer + abundance-weighted consensus
                      clustering → predicted proteins → bridge to
                      ``massspec.library``
    Direct RNA        future: physics-based basecaller models for
                      RNA modifications (m6A, m5C, ψ); see
                      :mod:`signal.models.physical`

Public surface (most stubs this session — see plan
in-our-development-of-fuzzy-quilt.md):

    Acquisitions, Samples, Reference, Assembly, Alignments, Project
    DoradoModel, DoradoRunner
    LibraryDesign, Adapter, Barcode (transcriptome demux primitives)
    locate_segments, resolve_demux, cluster_reads, build_consensus
    HiFiAsmRunner, PolishRunner, RagTagRunner, BuscoRunner
    pairwise_align, map_reads, locate_substring (alignment primitives)

Schema constants are re-exported under :mod:`sequencing.schemas`.

Status: stub session. Most function bodies raise ``NotImplementedError``
with a phase tag; ``DoradoModel.parse`` and the schema definitions are
fully implemented.
"""

from __future__ import annotations

# Import side-effects: each module's import triggers schema /
# reader / tool-spec registration. Keep these in dependency order
# (schemas first, then containers, then runners).
from constellation.sequencing import schemas  # noqa: F401  registers schemas
from constellation.sequencing import readers  # noqa: F401  registers readers
from constellation.sequencing import io as _io  # noqa: F401  registers cross-tier adapters

# Public re-exports — primary citizens of the sequencing domain
from constellation.sequencing.acquisitions import (
    SEQUENCING_ACQUISITION_TABLE,
    Acquisitions,
    validate_acquisitions,
)
from constellation.sequencing.alignments import Alignments
from constellation.sequencing.assembly import Assembly
from constellation.sequencing.basecall import DoradoModel, DoradoRunner, RunHandle
from constellation.sequencing.projects import Project
from constellation.sequencing.reference import Reference
from constellation.sequencing.samples import (
    SAMPLE_ACQUISITION_EDGE,
    SAMPLE_TABLE,
    Samples,
)

__all__ = [
    "Acquisitions",
    "SEQUENCING_ACQUISITION_TABLE",
    "validate_acquisitions",
    "Samples",
    "SAMPLE_TABLE",
    "SAMPLE_ACQUISITION_EDGE",
    "Reference",
    "Assembly",
    "Alignments",
    "Project",
    "DoradoModel",
    "DoradoRunner",
    "RunHandle",
]
