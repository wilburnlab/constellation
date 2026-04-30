"""Transcriptome pipeline — cDNA / direct-RNA from reads to consensus.

Modules:

    adapters     Adapter / Barcode / LibraryConstruct registry. Encodes
                 the lab's in-house SMARTer-derived chemistry plus the
                 standard ONT kits (PCS111, PCB111, RCB114) as
                 composable Segment layouts.
    demux        Segmented edlib-based deconvolution → READ_SEGMENT_TABLE
                 → READ_DEMUX_TABLE. Beats Dorado's full-primer
                 Smith-Waterman because high homopolymer error in
                 oligo-dT distorts SW scores (empirical mean A≈26
                 against expected A=20).
    orf          Thin wrapper over core.sequence.nucleic.find_orfs;
                 emits ORF_TABLE rows.
    cluster      mmseqs-style kmer clustering with abundance-weighted
                 consensus building. Goal: lift the ~1-3% per-read
                 clustering rate of the current naive ORF-based approach.
    consensus    Read-network → consensus transcript construction.
    network      Read-similarity graph (uses core.graph.Network) →
                 reference-free transcript / gene / allele structure.
"""

from __future__ import annotations

from constellation.sequencing.transcriptome.adapters import (
    Adapter,
    AdapterSlot,
    Barcode,
    BarcodeSlot,
    LibraryConstruct,
    PolyASlot,
    Segment,
    TranscriptSlot,
    UMISlot,
)
from constellation.sequencing.transcriptome.cluster import cluster_reads
from constellation.sequencing.transcriptome.consensus import build_consensus
from constellation.sequencing.transcriptome.demux import (
    locate_segments,
    resolve_demux,
)
from constellation.sequencing.transcriptome.network import build_read_network
from constellation.sequencing.transcriptome.orf import predict_orfs

__all__ = [
    "Adapter",
    "Barcode",
    "Segment",
    "AdapterSlot",
    "BarcodeSlot",
    "PolyASlot",
    "UMISlot",
    "TranscriptSlot",
    "LibraryConstruct",
    "locate_segments",
    "resolve_demux",
    "predict_orfs",
    "cluster_reads",
    "build_consensus",
    "build_read_network",
]
