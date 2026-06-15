"""De novo cluster variant + haplotype schemas (self-registered).

These document *what was collapsed* within each first-round cluster — the
per-position minor alleles (with a context-aware confidence that each is
a real variant vs basecaller error) and the distinct allele-combination
haplotypes (which integrate the covariance / phasing signal).
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema


# One row per consensus position carrying a tested minor allele.
CLUSTER_VARIANT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("cluster_id", pa.int64(), nullable=False),
        # 0-based position on the cluster's consensus_sequence.
        pa.field("consensus_pos", pa.int32(), nullable=False),
        # Major (consensus) allele + tested minor allele: 'A'/'C'/'G'/'T'/'-'.
        pa.field("consensus_allele", pa.string(), nullable=False),
        pa.field("minor_allele", pa.string(), nullable=False),
        # 'substitution' | 'homopolymer_indel' | 'non_hp_indel'
        pa.field("variant_class", pa.string(), nullable=False),
        # Homopolymer run length when variant_class == 'homopolymer_indel'.
        pa.field("homopolymer_run", pa.int32(), nullable=True),
        # Raw read multiplicity (the binomial n) + minor-allele count (a).
        pa.field("depth_total", pa.int32(), nullable=False),
        pa.field("depth_minor", pa.int32(), nullable=False),
        pa.field("minor_fraction", pa.float32(), nullable=False),
        # Upper-tail prob the minor count is explained by error under the
        # context-conditional null (binomial / beta-binomial SF).
        pa.field("p_error", pa.float64(), nullable=False),
        # Context-conditional per-read error rate used for this position.
        pa.field("epsilon_class", pa.float32(), nullable=False),
        # 'real' | 'collapsed_error' | 'ambiguous'
        pa.field("call", pa.string(), nullable=False),
        # Max pairwise r² to any other variant in the cluster (Cut 3).
        pa.field("max_linkage_r2", pa.float32(), nullable=True),
    ],
    metadata={b"schema_name": b"ClusterVariantTable"},
)


# One row per distinct allele-combination (haplotype) within a cluster.
CLUSTER_HAPLOTYPE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("cluster_id", pa.int64(), nullable=False),
        # Rank within the cluster, 0 = most abundant.
        pa.field("haplotype_id", pa.int32(), nullable=False),
        # Alleles at the cluster's variant positions, in consensus order;
        # '.' marks a position the haplotype's reads don't cover.
        pa.field("allele_string", pa.string(), nullable=False),
        pa.field("variant_positions", pa.list_(pa.int32()), nullable=False),
        # Total read multiplicity + distinct unique sequences carrying it.
        pa.field("abundance", pa.int64(), nullable=False),
        pa.field("n_unique_sequences", pa.int32(), nullable=False),
        pa.field("per_sample_abundance", pa.list_(pa.int64()), nullable=True),
        # True iff the haplotype covers all variant positions.
        pa.field("is_complete", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_name": b"ClusterHaplotypeTable"},
)


register_schema("ClusterVariantTable", CLUSTER_VARIANT_TABLE)
register_schema("ClusterHaplotypeTable", CLUSTER_HAPLOTYPE_TABLE)


__all__ = ["CLUSTER_VARIANT_TABLE", "CLUSTER_HAPLOTYPE_TABLE"]
