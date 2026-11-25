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

One file is the EM path's alone: ``cluster_edges.parquet``, how the final
clusters relate to each other (:mod:`.graph`). It is additive — nothing that
reads the shared files has to know it exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Aliased: the writers below take a ``write_fasta`` flag.
from constellation.core.io.fasta import write_fasta as _write_fasta
from constellation.core.io.schemas import register_schema
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

CLUSTER_EDGES = "cluster_edges.parquet"

#: Which round node each final cluster is, and what it is made of.
CLUSTER_NODES = "cluster_nodes.parquet"

#: ``clusters.parquet`` numbers clusters by their row among the nodes the
#: final merge kept, which is the order of nothing outside this run; this is
#: the join from a cluster back to the round that emitted it — and through
#: that round's lineage, to its whole history — and the content-derived ids
#: that let two runs, or a transcript and a peptide, name the same sequence.
EM_CLUSTER_NODE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("cluster_id", pa.int64(), nullable=False),
        # The round whose M-step emitted the node (the run's last round).
        pa.field("round", pa.int32(), nullable=False),
        # The node's key in that round's mstep/nodes and node_membership.
        pa.field("parent_template_id", pa.int64(), nullable=False),
        pa.field("haplotype_id", pa.int32(), nullable=False),
        # template_id_for(round + 1, node row): the id the node carries in
        # the round's graph/edges.parquet and merged_final.parquet.
        pa.field("node_template_id", pa.int64(), nullable=False),
        # The final merge absorbed others into it and its consensus was
        # rebuilt from the pooled reads, so it is no node's consensus.
        pa.field("rebuilt", pa.bool_(), nullable=False),
        pa.field("n_absorbed", pa.int32(), nullable=False),
        pa.field("absorbed_node_template_ids", pa.list_(pa.int64()), nullable=False),
        # core.ids.content_id of the consensus and of the protein: attributes,
        # not keys — byte-identical twins share one. Null protein: no ORF.
        pa.field("consensus_content_id", pa.int64(), nullable=False),
        pa.field("protein_content_id", pa.int64(), nullable=True),
        # The final stage's closing fold (2026-11-24; null before): clusters
        # that had become the same as this one after the rebuild, folded in
        # with its consensus kept. Their nodes — each folded cluster's own
        # and what the final merge had absorbed into it — hold reads of
        # this cluster that its consensus was not built from. `n_folded`
        # counts the nodes listed, as `n_absorbed` does.
        pa.field("n_folded", pa.int32(), nullable=True),
        pa.field("folded_node_template_ids", pa.list_(pa.int64()), nullable=True),
    ],
    metadata={b"schema_name": b"EmClusterNodeTable"},
)
register_schema("EmClusterNodeTable", EM_CLUSTER_NODE_TABLE)


def cluster_node_table(
    nodes: pa.Table,
    *,
    round_index: int,
    keep_rows: np.ndarray | None = None,
    rebuilt_rows: np.ndarray | None = None,
    merged: pa.Table | None = None,
    folded: pa.Table | None = None,
) -> pa.Table:
    """:data:`EM_CLUSTER_NODE_TABLE` for the final clusters.

    ``nodes`` are the final clusters' rows (``cluster_id`` = row);
    ``keep_rows`` the rows of the last round's node table they came from
    (identity when nothing was merged or there was no graph);
    ``rebuilt_rows`` the clusters the final merge rebuilt; ``merged`` the
    final merge's ``MERGED_TABLE``; ``folded`` the closing fold's (``None``:
    no fold ran, and the two fold columns are null).
    """
    from constellation.core.ids import content_ids
    from constellation.sequencing.transcriptome.cluster.denovo.em.refine import (
        template_id_for,
    )

    n = nodes.num_rows
    if n == 0:
        return EM_CLUSTER_NODE_TABLE.empty_table()
    keep = (
        np.arange(n, dtype=np.int64)
        if keep_rows is None
        else np.asarray(keep_rows, dtype=np.int64)
    )
    if keep.size != n:
        raise ValueError(
            f"{keep.size} kept node rows for {n} clusters: the final merge's "
            "record does not describe these clusters"
        )
    node_id = np.asarray(template_id_for(int(round_index) + 1, keep), dtype=np.int64)
    absorbed: dict[int, list[int]] = {}
    if merged is not None and merged.num_rows:
        for a, surv in zip(
            merged.column("absorbed_template_id").to_pylist(),
            merged.column("survivor_template_id").to_pylist(),
        ):
            absorbed.setdefault(int(surv), []).append(int(a))
    lists = [sorted(absorbed.get(int(i), [])) for i in node_id.tolist()]
    folded_lists = None
    if folded is not None:
        into: dict[int, list[int]] = {}
        for a, surv in zip(
            folded.column("absorbed_template_id").to_pylist(),
            folded.column("survivor_template_id").to_pylist(),
        ):
            # A folded cluster brings its own node and what the merge had
            # absorbed into it.
            into.setdefault(int(surv), []).extend([int(a), *absorbed.get(int(a), [])])
        folded_lists = [sorted(into.get(int(i), [])) for i in node_id.tolist()]
    rebuilt = np.zeros(n, dtype=bool)
    if rebuilt_rows is not None and len(rebuilt_rows):
        rebuilt[np.asarray(rebuilt_rows, dtype=np.int64)] = True
    proteins = [p or None for p in nodes.column("protein").to_pylist()]
    return pa.table(
        {
            "cluster_id": pa.array(np.arange(n, dtype=np.int64)),
            "round": pa.array(np.full(n, int(round_index), np.int32)),
            "parent_template_id": nodes.column("parent_template_id").cast(pa.int64()),
            "haplotype_id": nodes.column("haplotype_id").cast(pa.int32()),
            "node_template_id": pa.array(node_id),
            "rebuilt": pa.array(rebuilt),
            "n_absorbed": pa.array(np.array([len(x) for x in lists], np.int32)),
            "absorbed_node_template_ids": pa.array(lists, pa.list_(pa.int64())),
            "consensus_content_id": content_ids(
                nodes.column("consensus").to_pylist()
            ),
            "protein_content_id": content_ids(proteins),
            "n_folded": (
                pa.nulls(n, pa.int32())
                if folded_lists is None
                else pa.array(np.array([len(x) for x in folded_lists], np.int32))
            ),
            "folded_node_template_ids": (
                pa.nulls(n, pa.list_(pa.int64()))
                if folded_lists is None
                else pa.array(folded_lists, pa.list_(pa.int64()))
            ),
        },
        schema=EM_CLUSTER_NODE_TABLE,
    )


def write_cluster_nodes(output_dir: Path, table: pa.Table) -> Path:
    path = Path(output_dir) / CLUSTER_NODES
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path


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
            "representative_read_id": rep_read,
            "n_reads": pa.array(counts),
            "identity_threshold": pa.array(
                np.full(n, identity_threshold, dtype=np.float32)
            ),
            "consensus_sequence": _as_string(nodes.column("consensus")),
            "predicted_protein": _as_string(nodes.column("protein")),
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


#: Character bytes one ``pa.string()`` chunk may hold: its offsets are int32,
#: so 2 GiB is the ceiling. A MiB of headroom below it.
_STRING_CHUNK_BYTES = (1 << 31) - (1 << 20)


def _as_string(column) -> pa.ChunkedArray:
    """``large_string`` → ``string``, cut into chunks a 32-bit offset can address.

    The nodes carry their sequences as ``large_string``; the cluster table's
    schema says ``string``. A plain ``cast`` converts chunk by chunk, so one
    chunk of more than 2 GiB of characters cannot be cast at all — and the
    final rebuild combines the consensus column into a single chunk. At 9.4M
    reads and ~1.67M final nodes (~2.7 GB of consensus) that ended a 12-round
    run in its last step ("Failed casting from large_string to string: input
    array too large", 2026-11-18).

    Cutting is not enough on its own: a zero-copy slice keeps its parent's
    ABSOLUTE offsets, and the cast refuses any slice whose offsets pass 2³¹,
    however few characters it holds (measured, pyarrow 22: of a 2.24 GB chunk
    the first slice casts, the second does not). Such a piece is rebased by
    ``concat_arrays`` — one copy of at most a chunk — before it is cast.
    """
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    out: list[pa.Array] = []
    for chunk in chunks:
        if not pa.types.is_large_string(chunk.type):
            out.append(chunk.cast(pa.string()))
            continue
        n = len(chunk)
        # The chunk's own offsets, as a slice of a larger array sees them.
        offsets = np.frombuffer(chunk.buffers()[1], dtype=np.int64)[
            chunk.offset : chunk.offset + n + 1
        ]
        start = 0
        while start < n:
            # The furthest row whose characters still fit after `start`; at
            # least one row, so a single string over the limit fails in the
            # cast with Arrow's own message rather than looping here.
            stop = int(
                np.searchsorted(
                    offsets, offsets[start] + _STRING_CHUNK_BYTES, side="right"
                )
            ) - 1
            stop = min(max(stop, start + 1), n)
            piece = chunk.slice(start, stop - start)
            if int(offsets[stop]) > _STRING_CHUNK_BYTES:
                piece = pa.concat_arrays([piece])  # offsets rebased to 0
            out.append(piece.cast(pa.string()))
            start = stop
    return pa.chunked_array(out, type=pa.string())


def _read_facts(assignments: pa.Table, read_row: np.ndarray):
    """read_id / sample_id / aligned span for each membership row.

    Looked up from the assignment table by ``read_row``, which is a dense
    corpus index — so this is a scatter, not a join.
    """
    n = read_row.size
    if assignments is None or assignments.num_rows == 0 or n == 0:
        return (
            pa.array([""] * n, pa.string()),
            np.full(n, -1, np.int64),
            np.zeros(n, np.int64),
        )
    rows = assignments.column("read_row").to_numpy(zero_copy_only=False).astype(
        np.int64
    )
    size = int(max(rows.max(initial=-1), read_row.max(initial=-1))) + 1
    slot = np.full(size, -1, dtype=np.int64)
    slot[rows] = np.arange(rows.size)
    take = slot[np.clip(read_row, 0, size - 1)]
    valid = take >= 0
    idx = np.where(valid, take, 0)

    # Arrow, not a list of Python strings: one per read of the run (9.4M).
    read_id = pc.if_else(
        pa.array(valid),
        _string_column(assignments.column("read_id")).take(pa.array(idx)),
        pa.scalar("", pa.string()),
    )
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
) -> pa.Array:
    """The best-scoring read of each cluster ("" for a cluster without one).

    From round 2 the frame is a consensus with no read of its own, so there is
    nothing to inherit; the schema requires a non-null representative, and the
    read that fits the consensus best is the honest choice.
    """
    rep = pa.array([""] * n, pa.string())
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
    c, t = cluster_id[valid], take[valid]
    order = np.lexsort((-score[t], c))
    first = np.flatnonzero(np.concatenate([[True], c[order][1:] != c[order][:-1]]))
    k = order[first]
    best = np.full(n, -1, dtype=np.int64)
    best[c[k]] = t[k]
    has = best >= 0
    return pc.if_else(
        pa.array(has),
        _string_column(assignments.column("read_id")).take(pa.array(np.where(has, best, 0))),
        pa.scalar("", pa.string()),
    )


def _string_column(col) -> pa.Array:
    """A read-id column as one ``string`` array."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks() if col.num_chunks != 1 else col.chunk(0)
    return col.cast(pa.string()) if col.type != pa.string() else col


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
    read_id: pa.Array,
    span: np.ndarray,
    rep_read: pa.Array,
) -> pa.Table:
    m = cluster_id.size
    if m == 0:
        return CLUSTER_MEMBERSHIP_TABLE.empty_table()
    is_rep = pc.fill_null(
        pc.equal(read_id, rep_read.take(pa.array(cluster_id))), False
    ).to_numpy(zero_copy_only=False)
    return pa.table(
        {
            "cluster_id": pa.array(cluster_id),
            "read_id": read_id,
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

    # Every artifact below is conditional, and this directory may hold one
    # from an earlier run. A file that the current export does not produce is
    # not "unchanged" — it describes results that no longer exist, and
    # nothing marks it as stale. Clear them first so the directory only ever
    # holds this export.
    for optional in (
        "feature_quant.parquet",
        "cluster.fa",
        "proteins.fasta",
        CLUSTER_EDGES,
    ):
        stale = output_dir / optional
        if stale.exists():
            stale.unlink()

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
        _write_fasta(paths["fasta"], cons, [f"cluster_{c}" for c in cids], skip_empty=True)

        prot = clusters.column("predicted_protein").to_pylist()
        named = [(f"cluster_{c}", p) for c, p in zip(cids, prot) if p]
        if named:
            paths["proteins"] = output_dir / "proteins.fasta"
            _write_fasta(
                paths["proteins"], [p for _, p in named], [n for n, _ in named], skip_empty=True
            )
    return paths


def write_cluster_edges(
    output_dir: Path,
    edges_path: Path,
    keep_rows: np.ndarray,
    clusters: pa.Table,
    *,
    batch_rows: int = 1 << 20,
    cluster_origin: np.ndarray | None = None,
    predicate=None,
) -> tuple[Path, dict[str, int]]:
    """Project the final round's edges onto the clusters that were written.

    ``keep_rows`` are the node rows that survived as clusters, in cluster
    order — cluster ``i`` is node row ``keep_rows[i]``. The edge file is read
    and written a batch at a time: at 1.6M templates it is tens of millions
    of rows, and nothing here needs more than one batch of them.

    Call it AFTER :func:`write_em_outputs`, which clears a stale edge file
    along with the other optional exports.

    Returns the path and three counts the final record carries:
    ``n_cluster_edges``, ``n_still_mergeable`` (edges the run's predicate
    accepts that are still there — refused by a group guard, or inexact where
    the final merge is exact) and ``n_twin_clusters`` (clusters byte-identical
    to another one).

    ``cluster_origin`` and ``predicate`` (both or neither) re-decide each
    edge's ``same_split_origin`` and ``mergeable`` for the clusters as
    written (:func:`~.graph.cluster_edges`).
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        CLUSTER_EDGE_TABLE,
        cluster_edges,
    )

    output_dir = Path(output_dir)
    n = clusters.num_rows
    cluster_len = (
        pc.utf8_length(clusters.column("consensus_sequence"))
        .to_numpy(zero_copy_only=False)
        .astype(np.int64)
    )
    cluster_n_reads = (
        clusters.column("n_reads").to_numpy(zero_copy_only=False).astype(np.int64)
    )
    keep_rows = np.asarray(keep_rows, dtype=np.int64)
    if keep_rows.shape[0] != n:
        raise ValueError(
            f"{keep_rows.shape[0]:,} surviving node rows for {n:,} clusters: "
            f"the edges would be keyed on the wrong clusters"
        )

    path = output_dir / CLUSTER_EDGES
    tmp = path.with_name(path.name + ".tmp")
    counts = {"n_cluster_edges": 0, "n_still_mergeable": 0}
    twin = np.zeros(n, dtype=bool)
    overhangs = (
        "src_overhang_5p",
        "src_overhang_3p",
        "dst_overhang_5p",
        "dst_overhang_3p",
    )
    with pq.ParquetWriter(tmp, CLUSTER_EDGE_TABLE) as writer:
        for batch in pq.ParquetFile(edges_path).iter_batches(batch_size=batch_rows):
            out = cluster_edges(
                pa.Table.from_batches([batch]),
                keep_rows,
                cluster_len=cluster_len,
                cluster_n_reads=cluster_n_reads,
                cluster_origin=cluster_origin,
                predicate=predicate,
            )
            if out.num_rows == 0:
                continue
            writer.write_table(out)
            counts["n_cluster_edges"] += out.num_rows
            counts["n_still_mergeable"] += int(pc.sum(out.column("mergeable")).as_py())

            def col(name):
                return out.column(name).to_numpy(zero_copy_only=False)

            same = (col("n_edits") == 0) & (col("src_len") == col("dst_len"))
            for name in overhangs:
                same &= col(name) == 0
            twin[col("src_cluster_id")[same]] = True
            twin[col("dst_cluster_id")[same]] = True
    tmp.replace(path)
    counts["n_twin_clusters"] = int(twin.sum())
    return path, counts


def write_em_manifest(
    output_dir: Path,
    demux_dir: Path,
    *,
    parameters: dict,
    stages: dict,
    outputs: dict[str, str],
    samples: list[str] | None = None,
) -> Path:
    """Write the ``manifest.json`` every downstream consumer keys on.

    Without it ``read_manifest_dir`` rejects a completed run as an
    unrecognised pipeline output, so the viz layer cannot attach the directory
    as a source however well-formed its parquet is. The manifest is what makes
    an output dir a *stage* rather than a pile of files.
    """
    from constellation.sequencing.transcriptome.manifest import write_cluster_manifest

    write_cluster_manifest(
        Path(output_dir) / "manifest.json",
        reference_handle=None,
        reference_path=None,
        assembly_accession=None,
        # No align dir: this is a reference-free clusterer, the same as the
        # kmer path writes.
        align_dir="",
        demux_dir=str(demux_dir),
        parameters=parameters,
        stages=stages,
        outputs=outputs,
        samples=samples,
    )
    return Path(output_dir) / "manifest.json"


__all__ = [
    "CLUSTER_EDGES",
    "CLUSTER_MODES",
    "LEGACY_CLUSTER_MODES",
    "MODE_EM",
    "build_cluster_tables",
    "write_cluster_edges",
    "write_em_manifest",
    "write_em_outputs",
]
