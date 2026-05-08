"""Genome-aligned quantification — Mode A (reference-guided gene counting).

Sibling to :mod:`sequencing.transcriptome.quant` (which is the demux
pipeline's protein-cluster quantifier). This package hosts genome-mode
counting; future siblings (transcriptome-mode EM, multi-mapper
resolution, ProBAM-derived peptide → gene reduction) live here.

Three pieces, factored so the per-chunk fused worker and the
resolve-stage aggregator share implementations:

    apply_filter_predicates  — kernel: vectorised filter
    compute_gene_overlap     — kernel: sort+searchsorted overlap-join
    count_reads_per_gene     — aggregator: hash-join + group_by-sum to FEATURE_QUANT
    fused_decode_filter_overlap_worker
                             — pipeline worker (decode + filter + overlap, per chunk)

In-memory ``count_reads_per_gene`` and the fused worker share the same
``compute_gene_overlap`` kernel so the streaming and in-memory paths
produce identical assignments on identical inputs.
"""

from __future__ import annotations

from constellation.sequencing.quant._kernels import (
    GENE_ASSIGNMENT_SCHEMA,
    apply_filter_predicates,
    compute_gene_overlap,
    gene_set_from_annotation,
)
from constellation.sequencing.quant.gene_matrix import (
    build_gene_matrix,
    render_gene_matrix_tsv,
)
from constellation.sequencing.quant.genome_count import (
    count_reads_per_gene,
    extract_alignment_blocks,
    fused_decode_filter_overlap_worker,
    serialise_gene_set,
)
from constellation.sequencing.quant.coverage import build_pileup
from constellation.sequencing.quant.derived_annotation import (
    assign_blocks_to_exons,
    build_derived_annotation,
    compute_exon_psi,
    derive_exons,
    roll_up_genes,
)
from constellation.sequencing.quant.junctions import (
    aggregate_junctions,
    cluster_junctions,
)

__all__ = [
    "GENE_ASSIGNMENT_SCHEMA",
    "aggregate_junctions",
    "apply_filter_predicates",
    "assign_blocks_to_exons",
    "build_derived_annotation",
    "build_gene_matrix",
    "build_pileup",
    "cluster_junctions",
    "compute_exon_psi",
    "compute_gene_overlap",
    "count_reads_per_gene",
    "derive_exons",
    "extract_alignment_blocks",
    "fused_decode_filter_overlap_worker",
    "gene_set_from_annotation",
    "render_gene_matrix_tsv",
    "roll_up_genes",
    "serialise_gene_set",
]
