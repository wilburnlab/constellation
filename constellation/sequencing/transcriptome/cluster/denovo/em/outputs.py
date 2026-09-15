"""Final-round outputs, in the shapes every existing consumer already reads.

The EM loop's internal artifacts (templates / assignments / nodes / lineage)
are addressable per round and stay that way. What a *user* gets is the same
set of files the components path emits — ``clusters.parquet``,
``cluster_membership.parquet``, ``feature_quant.parquet``, ``cluster.fa``,
``proteins.fasta`` — so the viz layer, the quant consumers and the diagnostics
need no branch on which clusterer produced them.

The only schema movement is the ``mode`` vocabulary, which follows the CLI:
``'genome' | 'kmer' | 'em'`` rather than ``'genome-guided' | 'de-novo'``. The
old spellings still load — the column is a plain string and the viz colour
maps keep entries for both — so existing outputs keep working.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.sequencing.schemas.transcriptome import (
    CLUSTER_MEMBERSHIP_TABLE,
    TRANSCRIPT_CLUSTER_TABLE,
)

#: What the ``mode`` column may hold. The first three are canonical; the last
#: two are the pre-rename spellings, still readable so existing
#: clusters.parquet files do not need migrating.
CLUSTER_MODES = ("genome", "kmer", "em")
LEGACY_CLUSTER_MODES = {"genome-guided": "genome", "de-novo": "kmer"}

MODE_EM = "em"


def _sample_ids(membership: pa.Table, samples, m: int) -> pa.Array:
    """Per-membership-row sample_id. Null when the run is unsampled."""
    if samples is not None and "sample_id" in getattr(samples, "column_names", []):
        return samples.column("sample_id").cast(pa.int64())
    return pa.nulls(m, pa.int64())


def build_cluster_tables(
    nodes: pa.Table, assignments: pa.Table, *, identity_threshold: float
) -> tuple[pa.Table, pa.Table]:
    """``(clusters, membership)`` from a final round's nodes + assignments.

    One cluster per node. ``cluster_id`` is the node's own ``template_id``
    would-be — i.e. its row in the final node table — so it is stable within
    the run and independent of how the assignments happen to be ordered.
    """
    n = nodes.num_rows
    if n == 0:
        return (
            TRANSCRIPT_CLUSTER_TABLE.empty_table(),
            CLUSTER_MEMBERSHIP_TABLE.empty_table(),
        )

    parent_id = nodes.column("parent_template_id").to_numpy(zero_copy_only=False)
    cluster_of_parent = {int(p): i for i, p in enumerate(parent_id)}

    assigned = assignments.filter(
        pc.greater_equal(assignments.column("template_id"), 0)
    )
    a_tid = assigned.column("template_id").to_numpy(zero_copy_only=False)
    cluster_id = np.array(
        [cluster_of_parent.get(int(t), -1) for t in a_tid], dtype=np.int64
    )
    keep = cluster_id >= 0
    assigned = assigned.filter(pa.array(keep))
    cluster_id = cluster_id[keep]

    rep_read = _representative_per_cluster(assigned, cluster_id, n)
    counts = np.bincount(cluster_id, minlength=n).astype(np.int32)

    clusters = pa.table(
        {
            "cluster_id": pa.array(np.arange(n, dtype=np.int64)),
            "representative_read_id": pa.array(rep_read, pa.string()),
            "n_reads": pa.array(counts),
            "identity_threshold": pa.array(
                np.full(n, identity_threshold, dtype=np.float32)
            ),
            "consensus_sequence": nodes.column("consensus").cast(pa.string()),
            "predicted_protein": nodes.column("protein").cast(pa.string()),
            "orf_start": nodes.column("orf_start"),
            "orf_end": nodes.column("orf_end"),
            # Seeding is forward-strand only: windows are adapter-oriented,
            # and a '-' hit is dropped in the scan as a correctness matter.
            "orf_strand": pa.array(["+"] * n, pa.string()),
            "codon_table": pa.array(np.ones(n, dtype=np.int32)),
            "mode": pa.array([MODE_EM] * n, pa.string()),
            "contig_id": pa.nulls(n, pa.int64()),
            "strand": pa.nulls(n, pa.string()),
            "span_start": pa.nulls(n, pa.int64()),
            "span_end": pa.nulls(n, pa.int64()),
            "fingerprint_hash": pa.nulls(n, pa.uint64()),
            "n_unique_sequences": pa.array(np.maximum(counts, 0).astype(np.int32)),
            "sample_id": pa.nulls(n, pa.int64()),
        },
        schema=TRANSCRIPT_CLUSTER_TABLE,
    )

    membership = _membership(assigned, cluster_id, rep_read)
    return clusters, membership


def _representative_per_cluster(
    assigned: pa.Table, cluster_id: np.ndarray, n: int
) -> list[str]:
    """The best-scoring read of each cluster.

    From round 2 the frame is a consensus with no read of its own, so there
    is nothing to inherit; the schema requires a non-null representative, and
    the read that fits the consensus best is the honest choice.
    """
    rep = [""] * n
    if assigned.num_rows == 0:
        return rep
    score = assigned.column("as_score").to_numpy(zero_copy_only=False)
    read_id = assigned.column("read_id").to_pylist()
    order = np.lexsort((-score, cluster_id))
    first = np.flatnonzero(
        np.concatenate([[True], cluster_id[order][1:] != cluster_id[order][:-1]])
    )
    for pos in first:
        row = int(order[pos])
        rep[int(cluster_id[row])] = read_id[row]
    return rep


def _membership(
    assigned: pa.Table, cluster_id: np.ndarray, rep_read: list[str]
) -> pa.Table:
    m = assigned.num_rows
    if m == 0:
        return CLUSTER_MEMBERSHIP_TABLE.empty_table()
    read_id = assigned.column("read_id")
    is_rep = pc.equal(read_id, pa.array([rep_read[int(c)] for c in cluster_id]))
    span = pc.subtract(assigned.column("q_end"), assigned.column("q_start"))
    return pa.table(
        {
            "cluster_id": pa.array(cluster_id),
            "read_id": read_id.cast(pa.string()),
            "role": pc.if_else(is_rep, "representative", "member").cast(pa.string()),
            "drift_5p_bp": pa.nulls(m, pa.int32()),
            "drift_3p_bp": pa.nulls(m, pa.int32()),
            "match_rate": pa.nulls(m, pa.float32()),
            "indel_rate": pa.nulls(m, pa.float32()),
            "n_aligned_bp": span.cast(pa.int32()),
        },
        schema=CLUSTER_MEMBERSHIP_TABLE,
    )


def write_em_outputs(
    output_dir: Path,
    clusters: pa.Table,
    membership: pa.Table,
    *,
    samples=None,
    write_fasta: bool = True,
) -> dict[str, Path]:
    """Write the user-facing files and return what was written."""
    from constellation.sequencing.transcriptome.cluster.denovo.quant import (
        cluster_feature_quant,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["clusters"] = output_dir / "clusters.parquet"
    pq.write_table(clusters, paths["clusters"])
    paths["membership"] = output_dir / "cluster_membership.parquet"
    pq.write_table(membership, paths["membership"])

    if membership.num_rows:
        # `cluster_feature_quant` is written against the components path's
        # (read_map, cluster_of) shape: read -> uniq -> cluster. The EM path
        # assigns reads directly, so each membership row IS its own "unique"
        # and the indirection collapses to the identity.
        m = membership.num_rows
        read_map = pa.table(
            {
                "read_id": membership.column("read_id"),
                "uniq_id": pa.array(np.arange(m, dtype=np.int64)),
                "sample_id": _sample_ids(membership, samples, m),
            }
        )
        quant = cluster_feature_quant(
            read_map, membership.column("cluster_id").to_numpy(zero_copy_only=False)
        )
        paths["feature_quant"] = output_dir / "feature_quant.parquet"
        pq.write_table(quant, paths["feature_quant"])

    if write_fasta and clusters.num_rows:
        cids = clusters.column("cluster_id").to_pylist()
        cons = clusters.column("consensus_sequence").to_pylist()
        paths["fasta"] = output_dir / "cluster.fa"
        _write_fasta(paths["fasta"], [f"cluster_{c}" for c in cids], cons)

        prot = clusters.column("predicted_protein").to_pylist()
        named = [(f"cluster_{c}", p) for c, p in zip(cids, prot) if p]
        if named:
            paths["proteins"] = output_dir / "proteins.fasta"
            _write_fasta(
                paths["proteins"], [n for n, _ in named], [p for _, p in named]
            )
    return paths


def _write_fasta(path: Path, names, seqs) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for name, seq in zip(names, seqs):
            if seq:
                fh.write(f">{name}\n{seq}\n")


__all__ = [
    "CLUSTER_MODES",
    "LEGACY_CLUSTER_MODES",
    "MODE_EM",
    "build_cluster_tables",
    "write_em_outputs",
]
