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


def build_cluster_tables(
    nodes: pa.Table,
    assignments: pa.Table,
    node_membership: pa.Table,
    *,
    reads=None,
    identity_threshold: float,
) -> tuple[pa.Table, pa.Table, np.ndarray]:
    """``(clusters, membership)`` from a final round's nodes + membership.

    One cluster per node, and **membership comes from the M-step's own
    per-node read lists**, not from the read -> template mapping in the
    assignment table. That mapping is read -> *parent*, and a parent that the
    M-step split into several nodes has several children behind one id: using
    it collapses every split back into a single cluster and leaves the other
    children with no reads and no representative.
    """
    n = nodes.num_rows
    if n == 0:
        return (
            TRANSCRIPT_CLUSTER_TABLE.empty_table(),
            CLUSTER_MEMBERSHIP_TABLE.empty_table(),
            np.empty(0, dtype=np.int64),
        )

    # (parent_template_id, haplotype_id) is the join key: it is unique within
    # a round and known when a node is emitted, where the child's eventual
    # cluster_id is not.
    node_key = _pack_node_key(
        nodes.column("parent_template_id").to_numpy(zero_copy_only=False),
        nodes.column("haplotype_id").to_numpy(zero_copy_only=False),
    )
    order = np.argsort(node_key, kind="stable")
    sorted_key = node_key[order]

    cluster_id = np.empty(0, dtype=np.int64)
    read_row = np.empty(0, dtype=np.int64)
    if node_membership is not None and node_membership.num_rows:
        mem_key = _pack_node_key(
            node_membership.column("parent_template_id").to_numpy(
                zero_copy_only=False
            ),
            node_membership.column("haplotype_id").to_numpy(zero_copy_only=False),
        )
        pos = np.searchsorted(sorted_key, mem_key)
        ok = (pos < sorted_key.size) & (
            sorted_key[np.clip(pos, 0, max(sorted_key.size - 1, 0))] == mem_key
        )
        cluster_id = order[np.clip(pos, 0, max(sorted_key.size - 1, 0))][ok]
        read_row = (
            node_membership.column("read_row")
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)[ok]
        )

    read_id, sample_id, span = _read_facts(assignments, read_row)
    counts = np.bincount(cluster_id, minlength=n).astype(np.int32) if n else np.zeros(
        0, np.int32
    )
    rep_read = _representative_per_cluster(assignments, cluster_id, read_row, n)
    n_unique = _unique_sequences_per_cluster(reads, cluster_id, read_row, n)

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
            "n_unique_sequences": pa.array(n_unique),
            "sample_id": pa.nulls(n, pa.int64()),
        },
        schema=TRANSCRIPT_CLUSTER_TABLE,
    )
    # sample_id rides alongside rather than inside: CLUSTER_MEMBERSHIP_TABLE
    # has no such column, and quant needs it per read. Taking it from the
    # ASSIGNMENTS is the point — it used to come from an optional argument the
    # round loop never supplied, so every run pooled into one sample_id = -1
    # row and per-sample counts and TPM were silently wrong.
    membership = _membership(cluster_id, read_id, span, rep_read)
    return clusters, membership, sample_id


def _pack_node_key(parent: np.ndarray, hap: np.ndarray) -> np.ndarray:
    """A sortable key over ``(parent_template_id, haplotype_id)``.

    ``haplotype_id`` is bounded by ``max_nodes`` (8 by default, and the M-step
    caps it), so 16 bits is ample headroom and the parent id is not truncated.
    """
    return (np.asarray(parent, dtype=np.int64) << np.int64(16)) | np.asarray(
        hap, dtype=np.int64
    )


def _read_facts(assignments: pa.Table, read_row: np.ndarray):
    """read_id / sample_id / aligned span for each membership row.

    Looked up from the assignment table by ``read_row``, which is a dense
    corpus index — so this is a scatter, not a join.
    """
    n = read_row.size
    if assignments is None or assignments.num_rows == 0 or n == 0:
        return [""] * n, np.full(n, -1, np.int64), np.zeros(n, np.int64)
    rows = assignments.column("read_row").to_numpy(zero_copy_only=False).astype(
        np.int64
    )
    size = int(max(rows.max(initial=-1), read_row.max(initial=-1))) + 1
    slot = np.full(size, -1, dtype=np.int64)
    slot[rows] = np.arange(rows.size)
    take = slot[np.clip(read_row, 0, size - 1)]
    valid = take >= 0
    idx = np.where(valid, take, 0)

    ids = assignments.column("read_id").to_pylist()
    read_id = [ids[int(i)] if v else "" for i, v in zip(idx, valid)]
    sample = assignments.column("sample_id")
    sample_np = pc.fill_null(sample, -1).to_numpy(zero_copy_only=False).astype(np.int64)
    q_end = assignments.column("q_end").to_numpy(zero_copy_only=False).astype(np.int64)
    q_start = assignments.column("q_start").to_numpy(zero_copy_only=False).astype(
        np.int64
    )
    return (
        read_id,
        np.where(valid, sample_np[idx], -1),
        np.where(valid, (q_end - q_start)[idx], 0),
    )


def _representative_per_cluster(
    assignments: pa.Table, cluster_id: np.ndarray, read_row: np.ndarray, n: int
) -> list[str]:
    """The best-scoring read of each cluster.

    From round 2 the frame is a consensus with no read of its own, so there is
    nothing to inherit; the schema requires a non-null representative, and the
    read that fits the consensus best is the honest choice.
    """
    rep = [""] * n
    if cluster_id.size == 0 or assignments is None or assignments.num_rows == 0:
        return rep
    rows = assignments.column("read_row").to_numpy(zero_copy_only=False).astype(
        np.int64
    )
    size = int(max(rows.max(initial=-1), read_row.max(initial=-1))) + 1
    slot = np.full(size, -1, dtype=np.int64)
    slot[rows] = np.arange(rows.size)
    take = slot[np.clip(read_row, 0, size - 1)]
    valid = take >= 0
    if not valid.any():
        return rep
    score = assignments.column("as_score").to_numpy(zero_copy_only=False)
    ids = assignments.column("read_id").to_pylist()
    c, t = cluster_id[valid], take[valid]
    order = np.lexsort((-score[t], c))
    first = np.flatnonzero(np.concatenate([[True], c[order][1:] != c[order][:-1]]))
    for pos in first:
        k = int(order[pos])
        rep[int(c[k])] = ids[int(t[k])]
    return rep


def _unique_sequences_per_cluster(
    reads, cluster_id: np.ndarray, read_row: np.ndarray, n: int
) -> np.ndarray:
    """Distinct trimmed windows per cluster — the heterogeneity diagnostic.

    Counting reads here instead (which is what this used to do) makes the
    unique/read ratio identically 1 and the diagnostic meaningless: identical
    reads are reported as distinct sequences.
    """
    out = np.zeros(n, dtype=np.int32)
    if reads is None or cluster_id.size == 0:
        return np.bincount(cluster_id, minlength=n).astype(np.int32) if n else out
    from constellation.sequencing.transcriptome.cluster.denovo.dereplicate import (
        _hash_sequences,
    )

    digest = _hash_sequences(reads.sequence)
    key = np.frombuffer(
        digest.combine_chunks().buffers()[1]
        if isinstance(digest, pa.ChunkedArray)
        else digest.buffers()[1],
        dtype=np.uint64,
    )[: 2 * reads.n_reads].reshape(-1, 2)[:, 0]
    pair = np.stack([cluster_id, key[read_row].astype(np.int64)], axis=1)
    uniq = np.unique(pair, axis=0)
    counts = np.bincount(uniq[:, 0].astype(np.int64), minlength=n)
    out[: counts.size] = counts.astype(np.int32)
    return out


def _membership(
    cluster_id: np.ndarray,
    read_id: list[str],
    span: np.ndarray,
    rep_read: list[str],
) -> pa.Table:
    m = cluster_id.size
    if m == 0:
        return CLUSTER_MEMBERSHIP_TABLE.empty_table()
    is_rep = np.array(
        [rid == rep_read[int(c)] for rid, c in zip(read_id, cluster_id)], dtype=bool
    )
    return pa.table(
        {
            "cluster_id": pa.array(cluster_id),
            "read_id": pa.array(read_id, pa.string()),
            "role": pa.array(
                np.where(is_rep, "representative", "member"), pa.string()
            ),
            "drift_5p_bp": pa.nulls(m, pa.int32()),
            "drift_3p_bp": pa.nulls(m, pa.int32()),
            "match_rate": pa.nulls(m, pa.float32()),
            "indel_rate": pa.nulls(m, pa.float32()),
            "n_aligned_bp": pa.array(span.astype(np.int32)),
        },
        schema=CLUSTER_MEMBERSHIP_TABLE,
    )


def write_em_outputs(
    output_dir: Path,
    clusters: pa.Table,
    membership: pa.Table,
    sample_id: np.ndarray | None = None,
    *,
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
                "sample_id": pa.array(
                    np.asarray(sample_id, dtype=np.int64)
                    if sample_id is not None and len(sample_id) == m
                    else np.full(m, -1, dtype=np.int64)
                ),
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
