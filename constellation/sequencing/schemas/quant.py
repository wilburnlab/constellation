"""Quantification schemas — per-feature abundance + per-position coverage.

``FEATURE_QUANT`` is the empirical per-sample analog to the
sample-agnostic ``FEATURE_TABLE``. Keyed on
``(feature_id, sample_id, engine, feature_origin)``; holds raw counts +
normalized expression (TPM, CPM) + summary coverage statistics.
Multiple quantification engines (samtools view, featureCounts, HTSeq,
salmon mapping mode) all land here — the engine identity rides as a
column so a single table can carry results from several tools
side-by-side for comparison.

``feature_origin`` partitions the table by *what kind of feature_id*
the row keys against — ``'gene_id'`` rows reference an Annotation,
``'transcript_id'`` rows reference a TranscriptReference,
``'cluster_id'`` rows reference de novo cluster ids,
``'genetic_tool'`` rows reference the bundled GeneticTools DB,
``'protein_uniref'`` rows reference UniRef90 hits, and
``'protein_hash'`` rows reference the legacy naive-ORF clustering
output from S1 transcriptome demux. Within a partition the
``feature_id`` is unique against its corresponding lookup table; **never
sum across partitions** — that conflates fundamentally different
statistical assumptions about what a "feature" measures.

``COVERAGE_TABLE`` is the per-position depth track — kept separate
from FEATURE_QUANT because the row count is dramatically different
(1 row per feature × sample vs. 1 row per non-zero position × sample).
For dense whole-genome coverage, callers use one row per *interval*
of constant depth (run-length-encoded) rather than per-position.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema


FEATURE_QUANT: pa.Schema = pa.schema(
    [
        pa.field("feature_id", pa.int64(), nullable=False),
        pa.field("sample_id", pa.int64(), nullable=False),
        # Engine identity: 'samtools' | 'featurecounts' | 'htseq' |
        # 'mmseqs' | 'custom'; allows side-by-side comparison rows
        pa.field("engine", pa.string(), nullable=False),
        # Which kind of feature_id namespace this row keys into:
        # 'gene_id' | 'transcript_id' | 'cluster_id' | 'genetic_tool' |
        # 'protein_uniref' | 'protein_hash'
        pa.field("feature_origin", pa.string(), nullable=False),
        # Raw read count assigned to this feature
        pa.field("count", pa.float64(), nullable=False),
        # Normalized expression
        pa.field("tpm", pa.float64(), nullable=True),
        pa.field("cpm", pa.float64(), nullable=True),
        # Coverage summaries (mean / median / fraction-covered)
        pa.field("coverage_mean", pa.float32(), nullable=True),
        pa.field("coverage_median", pa.float32(), nullable=True),
        pa.field("coverage_fraction", pa.float32(), nullable=True),
        # Multi-mapping fraction (0..1); useful for paralog-aware QC
        pa.field("multimap_fraction", pa.float32(), nullable=True),
    ],
    metadata={b"schema_name": b"FeatureQuant"},
)


# Allowed values for FEATURE_QUANT.feature_origin — not enforced
# at the schema layer (Arrow doesn't have categorical-with-vocab),
# but consumed by container validators.
FEATURE_ORIGIN_VOCAB: frozenset[str] = frozenset(
    {
        "gene_id",
        "transcript_id",
        "cluster_id",
        "genetic_tool",
        "protein_uniref",
        "protein_hash",
    }
)


COVERAGE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("contig_id", pa.int64(), nullable=False),
        pa.field("sample_id", pa.int64(), nullable=False),
        # Run-length-encoded interval: depth is constant across [start, end)
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("depth", pa.int32(), nullable=False),
    ],
    metadata={b"schema_name": b"CoverageTable"},
)


register_schema("FeatureQuant", FEATURE_QUANT)
register_schema("CoverageTable", COVERAGE_TABLE)


__all__ = [
    "FEATURE_QUANT",
    "FEATURE_ORIGIN_VOCAB",
    "COVERAGE_TABLE",
]
