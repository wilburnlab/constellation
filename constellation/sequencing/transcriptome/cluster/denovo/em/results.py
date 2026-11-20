"""Reading a finished EM run: the notebook API, and the transcriptome browser's.

Pure functions over a run directory (``transcriptome cluster --mode em-*``'s
output), the way ``massspec.{scans, results}`` serve the mass-spec browser:
whatever a kernel shows, a notebook can get the same way. Nothing here runs
the pipeline except :func:`replay_template` / :func:`replay_cluster`, which
re-create one template's M-step to show its evidence (:mod:`.replay`).

Round directories are read through :mod:`.round_io`, which knows which M-step
shards are a round's; a run from before a table existed returns ``None`` for
it rather than an empty table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

from constellation.sequencing.transcriptome.cluster.denovo.em.replay import (
    ReplayError,
    assignment_locator,
    cluster_node_map,
    replay_cluster,
    replay_template,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
    read_round_columns,
    read_round_covariance,
    read_round_disagreements,
    read_round_membership,
    read_round_nodes,
    read_round_reads,
    read_round_templates,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.support import (
    node_differences,
    replay_summary,
    support_tables,
)


def _round_dir(run_dir: Path, r: int) -> Path:
    return Path(run_dir) / "rounds" / f"r{int(r):02d}"


def finished_rounds(run_dir: Path) -> list[int]:
    """Rounds with a ``_SUCCESS``, ascending."""
    rounds = Path(run_dir) / "rounds"
    if not rounds.exists():
        return []
    return sorted(
        int(d.name[1:])
        for d in rounds.glob("r*")
        if d.name[1:].isdigit() and (d / "_SUCCESS").exists()
    )


def run_info(run_dir: Path) -> dict:
    """What the run is: parameters, rounds, sizes, and which evidence it has."""
    run_dir = Path(run_dir)
    manifest = (
        json.loads((run_dir / "manifest.json").read_text())
        if (run_dir / "manifest.json").exists()
        else {}
    )
    rounds = finished_rounds(run_dir)
    last = _round_dir(run_dir, rounds[-1]) if rounds else None
    n_clusters = (
        pq.read_metadata(run_dir / "clusters.parquet").num_rows
        if (run_dir / "clusters.parquet").exists()
        else 0
    )
    return {
        "run_dir": str(run_dir),
        "finished": (run_dir / "clusters.parquet").exists(),
        "rounds": rounds,
        "n_clusters": int(n_clusters),
        "n_reads": _corpus_rows(run_dir),
        "parameters": manifest.get("parameters", {}),
        "demux_dir": manifest.get("demux_dir"),
        "has_cluster_nodes": (run_dir / "cluster_nodes.parquet").exists(),
        "has_mstep_record": bool(last and (last / "mstep" / "templates").exists()),
        "has_compact_evidence": bool(last and (last / "mstep" / "columns").exists()),
        "has_round_params": bool(last and (last / "params.json").exists()),
    }


def _corpus_rows(run_dir: Path) -> int:
    path = Path(run_dir) / "corpus" / "reads.arrow"
    if not path.exists():
        return 0
    with pa.memory_map(str(path), "r") as mm:
        reader = pa.ipc.open_file(mm)
        return int(sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches)))


def round_summaries(run_dir: Path) -> pa.Table:
    """One row per finished round: templates, nodes, churn, E-step and M-step
    counters (``round.json``)."""
    rows = []
    for r in finished_rounds(run_dir):
        path = _round_dir(run_dir, r) / "round.json"
        j = json.loads(path.read_text()) if path.exists() else {}
        est, churn, ms = j.get("estep", {}), j.get("churn", {}), j.get("mstep", {})
        rows.append(
            {
                "round": r,
                "n_templates": j.get("n_templates"),
                "n_nodes": j.get("n_nodes"),
                "n_assigned": est.get("n_assigned"),
                "n_unassigned": est.get("n_unassigned"),
                "frac_unsettled": churn.get("frac_unsettled"),
                "n_moved_genuine": churn.get("n_moved_genuine"),
                "n_split": ms.get("n_split"),
                "n_failed": ms.get("n_failed"),
                "estep_s": (j.get("seconds") or {}).get("estep"),
                "mstep_s": (j.get("seconds") or {}).get("mstep"),
            }
        )
    return pa.Table.from_pylist(rows) if rows else pa.table({"round": pa.array([], pa.int64())})


def clusters(run_dir: Path) -> pa.Table:
    """``clusters.parquet`` with each cluster's round node and content ids."""
    run_dir = Path(run_dir)
    table = pq.read_table(run_dir / "clusters.parquet")
    try:
        nodes = cluster_node_map(run_dir)
    except ReplayError:
        return table
    extra = nodes.select(
        [c for c in nodes.column_names if c != "cluster_id"]
    )
    for name in extra.column_names:
        table = table.append_column(name, extra.column(name))
    return table


def cluster_nodes(run_dir: Path) -> pa.Table:
    """Each final cluster's round node (``cluster_nodes.parquet``, or derived
    for a run from before it)."""
    return cluster_node_map(run_dir)


def nodes(run_dir: Path, round_index: int, columns: list[str] | None = None) -> pa.Table:
    """A round's M-step nodes, in node order; ``columns`` reads only those
    (a query for ids need not hold every consensus)."""
    return read_round_nodes(_round_dir(run_dir, round_index), columns=columns)


def node_membership(run_dir: Path, round_index: int) -> pa.Table:
    return read_round_membership(_round_dir(run_dir, round_index))


def mstep_record(run_dir: Path, round_index: int, what: str = "templates"):
    """One of a round's M-step tables: ``templates``, ``disagreements``,
    ``columns``, ``covariance`` or ``reads``; ``None`` if the round predates it."""
    readers = {
        "templates": read_round_templates,
        "disagreements": read_round_disagreements,
        "columns": read_round_columns,
        "covariance": read_round_covariance,
        "reads": read_round_reads,
    }
    if what not in readers:
        raise ValueError(f"unknown M-step table {what!r}; one of {sorted(readers)}")
    return readers[what](_round_dir(run_dir, round_index))


def lineage(run_dir: Path) -> pa.Table:
    """Every round's lineage rows (``carry`` / ``split`` / ``unrecruited`` /
    ``merge``), concatenated."""
    files = [
        _round_dir(run_dir, r) / "lineage.parquet"
        for r in finished_rounds(run_dir)
        if (_round_dir(run_dir, r) / "lineage.parquet").exists()
    ]
    if not files:
        from constellation.sequencing.transcriptome.cluster.denovo.em.refine import (
            LINEAGE_TABLE,
        )

        return LINEAGE_TABLE.empty_table()
    return pa.concat_tables([pq.read_table(f) for f in files])


def assignments(run_dir: Path, round_index: int, columns: list[str] | None = None):
    """A round's E-step assignment table (one row per corpus read)."""
    files = sorted((_round_dir(run_dir, round_index) / "assignments").glob("part-*.parquet"))
    if not files:
        return None
    return pa_ds.dataset(files).to_table(columns=columns)


def read_rows_for_ids(run_dir: Path, read_ids: list[str]) -> pa.Table:
    """``(read_id, read_row)`` for the ids found in the corpus."""
    path = Path(run_dir) / "corpus" / "reads.arrow"
    want = pa.array(list(read_ids), pa.string())
    out_ids, out_rows = [], []
    with pa.memory_map(str(path), "r") as mm:
        reader = pa.ipc.open_file(mm)
        base = 0
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            ids = batch.column("read_id").cast(pa.string())
            hit = np.flatnonzero(
                pc.is_in(ids, value_set=want).to_numpy(zero_copy_only=False)
            )
            out_ids += ids.take(pa.array(hit)).to_pylist()
            out_rows += (hit + base).tolist()
            base += batch.num_rows
    return pa.table(
        {"read_id": pa.array(out_ids, pa.string()), "read_row": pa.array(out_rows, pa.int64())}
    )


def read_history(run_dir: Path, read_rows) -> pa.Table:
    """Where each read sat, round by round: the template it was assigned to
    (``-1`` unassigned) and the node it ended in."""
    rows = np.asarray(read_rows, dtype=np.int64)
    want = pa.array(rows.astype(np.int32))
    out = []
    for r in finished_rounds(run_dir):
        a = assignments(run_dir, r, columns=["read_row", "template_id"])
        if a is None:
            continue
        a = a.filter(pc.is_in(a.column("read_row"), value_set=want))
        mem = node_membership(run_dir, r)
        mem = mem.filter(pc.is_in(mem.column("read_row"), value_set=want))
        node_of = dict(
            zip(
                mem.column("read_row").to_pylist(),
                zip(
                    mem.column("parent_template_id").to_pylist(),
                    mem.column("haplotype_id").to_pylist(),
                ),
            )
        )
        for row, tid in zip(a.column("read_row").to_pylist(), a.column("template_id").to_pylist()):
            parent, hap = node_of.get(row, (None, None))
            out.append(
                {
                    "read_row": row,
                    "round": r,
                    "template_id": tid,
                    "node_parent_template_id": parent,
                    "node_haplotype_id": hap,
                }
            )
    if not out:
        return pa.table(
            {
                "read_row": pa.array([], pa.int32()),
                "round": pa.array([], pa.int64()),
                "template_id": pa.array([], pa.int64()),
                "node_parent_template_id": pa.array([], pa.int64()),
                "node_haplotype_id": pa.array([], pa.int32()),
            }
        )
    return pa.Table.from_pylist(out).sort_by([("read_row", "ascending"), ("round", "ascending")])


def cluster_index(run_dir: Path) -> pa.Table:
    """One narrow row per final cluster — its size, lengths, round node and
    content ids, and no sequence.

    ``clusters.parquet`` holds every consensus (~2.7 GB at 9.4M reads), so
    a listing that read it to sort a million clusters by size would read the
    sequences to find their lengths. Here the sequence columns are streamed
    a batch at a time, kept only as lengths, and joined to
    ``cluster_nodes.parquet`` (or its derivation for an older run).
    """
    run_dir = Path(run_dir)
    pf = pq.ParquetFile(run_dir / "clusters.parquet")
    base = pf.read(
        columns=["cluster_id", "n_reads", "n_unique_sequences", "orf_start", "orf_end"]
    )
    lengths, protein = [], []
    for batch in pf.iter_batches(
        columns=["consensus_sequence", "predicted_protein"], batch_size=65_536
    ):
        lengths.append(pc.binary_length(batch.column(0)).cast(pa.int32()))
        protein.append(
            pc.fill_null(pc.binary_length(batch.column(1)), 0).cast(pa.int32())
        )
    table = base.append_column(
        "length", pa.chunked_array(lengths, pa.int32())
    ).append_column("protein_length", pa.chunked_array(protein, pa.int32()))
    try:
        nodes = cluster_node_map(run_dir)
    except ReplayError:
        return table
    keep = [
        "cluster_id",
        "round",
        "parent_template_id",
        "haplotype_id",
        "node_template_id",
        "rebuilt",
        "n_absorbed",
        "consensus_content_id",
        "protein_content_id",
    ]
    nodes = nodes.select([c for c in keep if c in nodes.column_names])
    return (
        table.join(nodes, "cluster_id", join_type="left outer")
        .sort_by("cluster_id")
        .combine_chunks()
    )


def find_clusters(
    run_dir: Path, query: str, *, limit: int = 200, index: pa.Table | None = None
) -> pa.Table:
    """Clusters matching ``query``, as :func:`cluster_index` rows: a cluster
    id or a content id, a protein substring (>= 4 upper-case residues), or a
    read id (the cluster that holds it). Each reads only the columns it
    searches; pass ``index`` to reuse one already built."""
    run_dir = Path(run_dir)
    table = cluster_index(run_dir) if index is None else index
    q = query.strip()
    ids = table.column("cluster_id").to_numpy(zero_copy_only=False)
    hits = np.zeros(table.num_rows, dtype=bool)
    if q.isdigit():
        n = int(q)
        hits |= ids == n
        for name in ("consensus_content_id", "protein_content_id"):
            if name in table.column_names:
                hits |= pc.fill_null(pc.equal(table.column(name), n), False).to_numpy(
                    zero_copy_only=False
                )
    found: list[np.ndarray] = []
    if len(q) >= 4 and q.isalpha() and q.upper() == q:
        prot = pq.read_table(
            run_dir / "clusters.parquet", columns=["cluster_id", "predicted_protein"]
        )
        match = pc.fill_null(pc.match_substring(prot.column("predicted_protein"), q), False)
        found.append(prot.column("cluster_id").filter(match).to_numpy(zero_copy_only=False))
    member = pq.read_table(
        run_dir / "cluster_membership.parquet",
        columns=["cluster_id"],
        filters=[("read_id", "==", q)],
    )
    if member.num_rows:
        found.append(member.column("cluster_id").to_numpy(zero_copy_only=False))
    if found:
        hits |= np.isin(ids, np.concatenate(found))
    return table.filter(pa.array(hits)).slice(0, limit)


__all__ = [
    "ReplayError",
    "assignment_locator",
    "assignments",
    "cluster_index",
    "cluster_node_map",
    "cluster_nodes",
    "clusters",
    "find_clusters",
    "finished_rounds",
    "lineage",
    "mstep_record",
    "node_membership",
    "node_differences",
    "nodes",
    "read_history",
    "read_rows_for_ids",
    "replay_cluster",
    "replay_summary",
    "replay_template",
    "round_summaries",
    "run_info",
    "support_tables",
]
