"""Transcriptome pipeline — cDNA / direct-RNA from reads to consensus.

Subpackages (per pipeline stage):

    demux       S1 demultiplex — segmented edlib-based deconvolution
                (adapters, classify, demux, designs, fastq, orf, quant,
                scoring, simulator).
    align       S2 align — transcriptome-specific align-stage logic
                (currently a placeholder for diagnostic reports; the
                minimap2 orchestrator lives at
                :mod:`constellation.sequencing.align`).
    cluster     Phase 2+ cluster — fingerprint computation, genome-guided
                clustering, weighted-PWM consensus building, de novo
                clustering stubs.

Shared at this level:

    manifest    Demux/Align/Cluster manifest readers/writers.
    stages      Orchestrator wiring demux subpackage modules into the
                ``constellation transcriptome demultiplex`` CLI command.

The public re-exports below preserve the prior flat-namespace API for
callers that ``from constellation.sequencing.transcriptome import X``.
Callers using sub-module paths must reference the new locations under
``demux/`` / ``align/`` / ``cluster/``.
"""

from __future__ import annotations

from constellation.sequencing.transcriptome.demux.adapters import (
    Adapter,
    AdapterSlot,
    Barcode,
    BarcodeSlot,
    LibraryDesign,
    PolyASlot,
    Segment,
    TranscriptSlot,
    UMISlot,
)
from constellation.sequencing.transcriptome.cluster.cluster import cluster_reads
from constellation.sequencing.transcriptome.cluster.consensus import build_consensus
from constellation.sequencing.transcriptome.demux.demux import (
    locate_segments,
    resolve_demux,
)
from constellation.sequencing.transcriptome.cluster.network import build_read_network
from constellation.sequencing.transcriptome.demux.orf import predict_orfs
from constellation.sequencing.transcriptome.demux.simulator import (
    GROUND_TRUTH_TABLE,
    ReadSpec,
    assemble_sequence,
    generate_stress_test_specs,
    mutate,
    simulate_panel,
)

__all__ = [
    "Adapter",
    "Barcode",
    "Segment",
    "AdapterSlot",
    "BarcodeSlot",
    "PolyASlot",
    "UMISlot",
    "TranscriptSlot",
    "LibraryDesign",
    "locate_segments",
    "resolve_demux",
    "predict_orfs",
    "cluster_reads",
    "build_consensus",
    "build_read_network",
    "GROUND_TRUTH_TABLE",
    "ReadSpec",
    "assemble_sequence",
    "generate_stress_test_specs",
    "mutate",
    "simulate_panel",
]
