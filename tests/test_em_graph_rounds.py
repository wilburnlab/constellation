"""The template graph inside the round loop — without running a round.

`tests/test_em_rounds.py` drives the whole loop and needs minimap2. Nothing
here does: the graph stage, the merge, what is recorded about them and what a
resume makes of that record are all reachable from a table of nodes and a
directory, so they are tested from exactly that.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em import (
    rounds as rounds_mod,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.diagnostics import (
    section_template_graph,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
    NODE_MEMBERSHIP_TABLE,
    REFINED_NODE_TABLE,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.outputs import (
    write_cluster_edges,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
    EmParams,
    _check_merge_state,
    _final_graph_and_merge,
    _node_graph,
    _refine_and_merge,
    _write_final_record,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
    TEMPLATE_TABLE,
    TemplateStore,
)


def _rnd(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _store(ids):
    n = len(ids)
    return TemplateStore.from_table(
        pa.table(
            {
                "template_id": pa.array(np.asarray(ids, dtype=np.int64)),
                "sequence": pa.array(["ACGT" * 25] * n, pa.large_string()),
                "orf_start": pa.array(np.zeros(n, np.int32)),
                "orf_end": pa.array(np.full(n, 99, np.int32)),
                "orf_aa_length": pa.array(np.full(n, 32, np.int32)),
                "node_weight": pa.array(np.ones(n, np.float64)),
                "orf_replication": pa.array(np.ones(n, np.int64)),
                "seed_read_quality": pa.array(np.full(n, 25.0, np.float32)),
                "seed_read_row": pa.array(np.arange(n, dtype=np.int32)),
                "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
            },
            schema=TEMPLATE_TABLE,
        )
    )


def _nodes(rows, r=1):
    """rows = [(parent_id, parent_row, hap, consensus, n_reads)]"""
    n = len(rows)
    return pa.table(
        {
            "round": pa.array([r] * n, pa.int32()),
            "parent_template_id": pa.array([x[0] for x in rows], pa.int64()),
            "parent_template_row": pa.array([x[1] for x in rows], pa.int32()),
            "haplotype_id": pa.array([x[2] for x in rows], pa.int32()),
            "consensus": pa.array([x[3] for x in rows], pa.large_string()),
            "n_reads": pa.array([x[4] for x in rows], pa.int64()),
            "node_weight": pa.array([float(x[4]) for x in rows], pa.float64()),
            "protein": pa.array([None] * n, pa.large_string()),
            "orf_start": pa.array([0] * n, pa.int32()),
            "orf_end": pa.array([30] * n, pa.int32()),
            "allele_string": pa.array([None] * n, pa.string()),
            "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
            "n_inserted_columns": pa.array([0] * n, pa.int32()),
            "n_extended_5p": pa.array([0] * n, pa.int32()),
            "n_extended_3p": pa.array([0] * n, pa.int32()),
            "n_trimmed_5p": pa.array([0] * n, pa.int32()),
            "n_trimmed_3p": pa.array([0] * n, pa.int32()),
            "n_members_used": pa.array([x[4] for x in rows], pa.int32()),
            "subsample_fraction": pa.array([1.0] * n, pa.float32()),
        },
        schema=REFINED_NODE_TABLE,
    )


def _membership(nodes, r=1):
    """One membership row per read of every node, read rows counted up."""
    pid, hap, row = [], [], []
    k = 0
    for p, h, n in zip(
        nodes.column("parent_template_id").to_pylist(),
        nodes.column("haplotype_id").to_pylist(),
        nodes.column("n_reads").to_pylist(),
    ):
        pid += [p] * n
        hap += [h] * n
        row += list(range(k, k + n))
        k += n
    return pa.table(
        {
            "round": pa.array([r] * len(row), pa.int32()),
            "parent_template_id": pa.array(pid, pa.int64()),
            "haplotype_id": pa.array(hap, pa.int32()),
            "read_row": pa.array(row, pa.int32()),
            "weight": pa.array([1.0] * len(row), pa.float32()),
        },
        schema=NODE_MEMBERSHIP_TABLE,
    )


@pytest.fixture
def panel():
    """Four parents, five nodes.

    node 0, 1   the two children of parent 10, differing by 12 nt of 5' extent
    node 2      parent 11: byte-identical to node 0, from another lineage
    node 3      parent 12: an unrelated transcript
    node 4      parent 13: node 3 truncated by 200 nt at the 3' end
    """
    rng = random.Random(5)
    a, b = _rnd(rng, 900), _rnd(rng, 1100)
    rows = [
        (10, 0, 0, a, 30),
        (10, 0, 1, a[12:], 8),
        (11, 1, 0, a, 5),
        (12, 2, 0, b, 40),
        (13, 3, 0, b[:900], 6),
    ]
    return _nodes(rows), _store([10, 11, 12, 13])


def _rd(tmp_path, r=1):
    rd = tmp_path / "run" / "rounds" / f"r{r:02d}"
    rd.mkdir(parents=True)
    return rd


def _log(_message):
    return None


def _json(path):
    return json.loads(Path(path).read_text())


def _relations(path):
    t = pq.read_table(path)
    return {
        (a, b): (rel, m)
        for a, b, rel, m in zip(
            t.column("src_row").to_pylist(),
            t.column("dst_row").to_pylist(),
            t.column("relation").to_pylist(),
            t.column("mergeable").to_pylist(),
        )
    }


# ── the default: a report ─────────────────────────────────────────────


def test_report_only_relates_the_nodes_and_merges_nothing(panel, tmp_path):
    nodes, store = panel
    rd = _rd(tmp_path)
    refined = _refine_and_merge(
        rd, nodes, store, 1, EmParams(threads=1, merge=False), _log
    )

    assert refined.n_merged == 0 and refined.templates.num_rows == 5
    assert "merge" not in refined.lineage.column("rule").to_pylist()
    rel = _relations(rd / "graph" / "edges.parquet")
    # The twin from another lineage is mergeable; the sibling is equivalent
    # and is not; the truncated transcript is contained and never is.
    assert rel[(0, 2)] == ("equivalent", True)
    assert rel[(1, 0)] == ("equivalent", False)
    assert rel[(4, 3)] == ("contained", False)
    record = _json(rd / "refine.json")
    assert record["graph"] == "ok" and record["merge_applied"] is False
    assert record["n_merged"] == 0 and record["n_templates_after"] == 5
    assert "graph_stamp" not in record, "nothing merged, so nothing is committed"
    assert pq.read_table(rd / "merged.parquet").num_rows == 0
    assert (rd / "graph" / "_SUCCESS").exists()
    assert not list(rd.rglob("*.tmp"))


def test_merge_collapses_the_twin_and_leaves_the_sibling(panel, tmp_path):
    nodes, store = panel
    rd = _rd(tmp_path)
    params = EmParams(threads=1)
    assert params.merge and params.merge_max_edits == 2, "the default merges"
    refined = _refine_and_merge(rd, nodes, store, 1, params, _log)

    assert refined.n_merged == 1 and refined.templates.num_rows == 4
    ids = rounds_mod.rf.template_id_for(2, np.arange(5))
    assert refined.templates.column("template_id").to_pylist() == [
        int(ids[i]) for i in (0, 1, 3, 4)
    ]
    assert refined.templates.column("orf_replication").to_pylist()[0] == 35
    merged = pq.read_table(rd / "merged.parquet").to_pylist()
    assert [(m["absorbed_template_id"], m["survivor_template_id"]) for m in merged] == [
        (int(ids[2]), int(ids[0]))
    ]
    record = _json(rd / "refine.json")
    assert record["merge_applied"] and record["n_merged"] == 1
    assert record["predicate"] == asdict(params.merge_predicate())
    assert record["graph_stamp"]["kernel_version"] >= 1


def test_merge_siblings_lets_the_split_be_undone(panel, tmp_path):
    nodes, store = panel
    refined = _refine_and_merge(
        _rd(tmp_path),
        nodes,
        store,
        1,
        EmParams(threads=1, merge=True, merge_siblings=True),
        _log,
    )
    assert refined.n_merged == 2 and refined.templates.num_rows == 3


def test_a_stricter_tolerance_is_reported_without_merging(panel, tmp_path):
    """The predicate defines `mergeable` whether or not the run merges."""
    nodes, store = panel
    rd = _rd(tmp_path)
    _refine_and_merge(
        rd, nodes, store, 1, EmParams(threads=1, merge_siblings=True), _log
    )
    assert _relations(rd / "graph" / "edges.parquet")[(1, 0)] == ("equivalent", True)
    rd2 = _rd(tmp_path / "strict")
    _refine_and_merge(
        rd2,
        nodes,
        store,
        1,
        EmParams(threads=1, merge_siblings=True, merge_tol_5p=10),
        _log,
    )
    assert _relations(rd2 / "graph" / "edges.parquet")[(1, 0)] == ("equivalent", False)


def test_merge_from_round_waits(panel, tmp_path):
    nodes, store = panel
    rd = _rd(tmp_path)
    params = EmParams(threads=1, merge=True, merge_from_round=2)
    assert _refine_and_merge(rd, nodes, store, 1, params, _log).n_merged == 0
    assert _json(rd / "refine.json")["merge_applied"] is False


def test_a_zero_length_node_does_not_shift_the_merge(tmp_path):
    """The graph is over node rows; templates drop the empty one. Joined by
    row the twin would land on its unrelated neighbour."""
    rng = random.Random(9)
    a, b = _rnd(rng, 800), _rnd(rng, 800)
    nodes = _nodes(
        [(10, 0, 0, "", 2), (11, 1, 0, a, 30), (12, 2, 0, a, 4), (13, 3, 0, b, 7)]
    )
    rd = _rd(tmp_path)
    refined = _refine_and_merge(
        rd, nodes, _store([10, 11, 12, 13]), 1, EmParams(threads=1, merge=True), _log
    )
    assert refined.n_merged == 1
    assert refined.templates.column("sequence").to_pylist() == [a, b]
    assert refined.templates.column("orf_replication").to_pylist() == [34, 7]
    assert _json(rd / "graph" / "stats.json")["n_unpairable_short"] == 1


# ── the escape hatch, and failure ─────────────────────────────────────


def _never(*_a, **_k):
    raise AssertionError("the template graph ran")


def test_template_graph_off_runs_nothing(panel, tmp_path, monkeypatch):
    nodes, store = panel
    monkeypatch.setattr(rounds_mod.gr, "build_graph", _never)
    monkeypatch.setattr(rounds_mod.gr, "split_origins", _never)
    monkeypatch.setattr(rounds_mod.gr, "input_digest", _never)
    rd = _rd(tmp_path)
    params = EmParams(threads=1, template_graph="off", merge=False)
    refined = _refine_and_merge(rd, nodes, store, 1, params, _log)
    assert refined.templates.num_rows == 5 and not (rd / "graph").exists()
    assert _json(rd / "refine.json")["graph"] == "off"

    out_nodes, mem, final = _final_graph_and_merge(
        rd, nodes, _membership(nodes), 1, params, _log
    )
    assert out_nodes.equals(nodes) and final["graph"] == "off"
    assert "edges_path" not in final and not (rd / "graph").exists()


def test_template_graph_final_builds_nothing_between_rounds(
    panel, tmp_path, monkeypatch
):
    nodes, store = panel
    monkeypatch.setattr(rounds_mod.gr, "build_graph", _never)
    rd = _rd(tmp_path)
    _refine_and_merge(
        rd, nodes, store, 1, EmParams(threads=1, template_graph="final"), _log
    )
    assert not (rd / "graph").exists()
    assert _json(rd / "refine.json")["graph"] == "off"


def _boom(*_a, **_k):
    raise MemoryError("no room for the index")


def test_a_failed_report_does_not_sink_the_round(panel, tmp_path, monkeypatch):
    """Nothing reads a graph the run does not merge from, so the round it
    describes is no less finished for its failing."""
    nodes, store = panel
    monkeypatch.setattr(rounds_mod.gr, "build_graph", _boom)
    rd = _rd(tmp_path)
    said = []
    report = EmParams(threads=1, merge=False)
    refined = _refine_and_merge(rd, nodes, store, 1, report, said.append)
    assert refined.templates.num_rows == 5
    record = _json(rd / "refine.json")
    assert record["graph"] == "failed" and "MemoryError" in record["graph_error"]
    assert any("failed" in line for line in said)

    _, _, final = _final_graph_and_merge(rd, nodes, _membership(nodes), 1, report, _log)
    assert final["graph"] == "failed" and "edges_path" not in final


def test_a_failed_graph_is_fatal_when_the_run_merges_from_it(
    panel, tmp_path, monkeypatch
):
    nodes, store = panel
    monkeypatch.setattr(rounds_mod.gr, "build_graph", _boom)
    rd = _rd(tmp_path)
    params = EmParams(threads=1, merge=True)
    with pytest.raises(MemoryError):
        _refine_and_merge(rd, nodes, store, 1, params, _log)
    assert not (rd / "refine.json").exists()
    with pytest.raises(MemoryError):
        _final_graph_and_merge(rd, nodes, _membership(nodes), 1, params, _log)


# ── the cache ─────────────────────────────────────────────────────────


def _spy(monkeypatch):
    built = []
    real = rounds_mod.gr.build_graph

    def _build(*a, **k):
        built.append(k.get("node_round"))
        return real(*a, **k)

    monkeypatch.setattr(rounds_mod.gr, "build_graph", _build)
    return built


def test_the_graph_is_rebuilt_only_when_something_it_depends_on_moved(
    panel, tmp_path, monkeypatch
):
    nodes, _ = panel
    rd = _rd(tmp_path)
    built = _spy(monkeypatch)
    first = _node_graph(rd, nodes, 1, EmParams(threads=1), _log)
    assert built == [1]
    again = _node_graph(rd, nodes, 1, EmParams(threads=3), _log)
    assert built == [1], "the thread count is not something the output depends on"
    assert again.stamp == first.stamp and again.stats == first.stats

    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        GraphParams,
    )

    _node_graph(rd, nodes, 1, EmParams(threads=1, graph=GraphParams(tol_5p=20)), _log)
    assert built == [1, 1]
    _node_graph(rd, nodes, 1, EmParams(threads=1, merge_max_edits=1), _log)
    assert built == [1, 1, 1], "`mergeable` is stored, so the predicate is a dependency"

    i = nodes.schema.get_field_index("n_reads")
    moved = nodes.set_column(
        i, nodes.schema.field(i), pa.array([30, 8, 5, 40, 7], pa.int64())
    )
    _node_graph(rd, moved, 1, EmParams(threads=1, merge_max_edits=1), _log)
    assert built == [1, 1, 1, 1], "the read counts are in the table, so in the digest"


def test_a_graph_without_its_marker_or_its_footer_is_rebuilt(
    panel, tmp_path, monkeypatch
):
    nodes, _ = panel
    rd = _rd(tmp_path)
    built = _spy(monkeypatch)
    _node_graph(rd, nodes, 1, EmParams(threads=1), _log)
    (rd / "graph" / "_SUCCESS").unlink()
    _node_graph(rd, nodes, 1, EmParams(threads=1), _log)
    assert built == [1, 1]
    (rd / "graph" / "edges.parquet").write_bytes(b"PAR1 and then the power went")
    _node_graph(rd, nodes, 1, EmParams(threads=1), _log)
    assert built == [1, 1, 1]
    assert pq.read_metadata(rd / "graph" / "edges.parquet").num_rows >= 3


# ── the final output ──────────────────────────────────────────────────


def _clusters(nodes):
    from constellation.sequencing.schemas.transcriptome import (
        TRANSCRIPT_CLUSTER_TABLE,
    )

    n = nodes.num_rows
    cols = {
        "cluster_id": pa.array(np.arange(n, dtype=np.int64)),
        "n_reads": nodes.column("n_reads").cast(pa.int32()),
        "consensus_sequence": nodes.column("consensus").cast(pa.string()),
    }
    return pa.table(
        {
            f.name: cols.get(f.name, pa.nulls(n, f.type))
            for f in TRANSCRIPT_CLUSTER_TABLE
        }
    )


def test_the_final_merge_follows_the_cap_and_keeps_every_read(panel, tmp_path):
    nodes, _ = panel
    rd = _rd(tmp_path)
    out = tmp_path / "run"
    member = _membership(nodes)
    params = EmParams(threads=1, merge=True, merge_max_edits=3)
    merged_nodes, mem, final = _final_graph_and_merge(
        rd, nodes, member, 1, params, _log
    )

    assert final["merge_applied"] and final["n_merged"] == 1
    assert final["predicate"]["max_edits"] == 3, "the run's predicate, not exact"
    assert final["rebuild"] == "skipped: no corpus"
    assert final["keep_rows"].tolist() == [0, 1, 3, 4]
    assert merged_nodes.column("n_reads").to_pylist() == [35, 8, 40, 6]
    assert mem.num_rows == member.num_rows
    assert pq.read_table(rd / "merged_final.parquet").num_rows == 1

    clusters = _clusters(merged_nodes)
    path, counts = write_cluster_edges(
        out, final["edges_path"], final["keep_rows"], clusters
    )
    edges = pq.read_table(path)
    src = edges.column("src_cluster_id").to_pylist()
    dst = edges.column("dst_cluster_id").to_pylist()
    assert all(0 <= i < 4 for i in src + dst)
    assert all(a != b for a, b in zip(src, dst))
    # Node rows (1, 0) and (4, 3) are clusters (1, 0) and (3, 2) now.
    assert sorted(zip(src, dst)) == [(1, 0), (3, 2)]
    by_pair = dict(zip(zip(src, dst), edges.column("dst_n_reads").to_pylist()))
    assert by_pair[(1, 0)] == 35, "read counts are the clusters', after the merge"
    assert counts["n_cluster_edges"] == 2 and counts["n_twin_clusters"] == 0
    assert not list(out.glob("*.tmp"))

    _write_final_record(rd, final, clusters.num_rows, counts)
    record = _json(rd / "final.json")
    assert record["n_clusters"] == 4 and record["n_cluster_edges"] == 2
    assert "keep_rows" not in record and "edges_path" not in record


def test_cluster_edges_refuse_a_cluster_table_they_do_not_match(panel, tmp_path):
    nodes, _ = panel
    rd = _rd(tmp_path)
    _, _, final = _final_graph_and_merge(
        rd, nodes, _membership(nodes), 1, EmParams(threads=1), _log
    )
    with pytest.raises(ValueError, match="wrong clusters"):
        write_cluster_edges(
            tmp_path,
            final["edges_path"],
            final["keep_rows"],
            _clusters(nodes.slice(0, 3)),
        )


def test_an_unmerged_final_output_says_how_many_clusters_are_twins(panel, tmp_path):
    nodes, _ = panel
    rd = _rd(tmp_path)
    _, _, final = _final_graph_and_merge(
        rd, nodes, _membership(nodes), 1, EmParams(threads=1, merge=False), _log
    )
    assert not final["merge_applied"] and final["keep_rows"].tolist() == [0, 1, 2, 3, 4]
    assert pq.read_table(rd / "merged_final.parquet").num_rows == 0
    _, counts = write_cluster_edges(
        tmp_path / "run", final["edges_path"], final["keep_rows"], _clusters(nodes)
    )
    # (0, 2) are twins. (1, 2) is mergeable too, pairwise: node 1 is node 0's
    # sibling, and node 2 — byte-identical to node 0 — is nobody's. That is
    # the hub the per-group guard exists for.
    assert counts["n_twin_clusters"] == 2 and counts["n_still_mergeable"] == 2


def test_extending_a_run_reuses_the_final_graph_and_retires_the_final_record(
    panel, tmp_path, monkeypatch
):
    """The last round's graph is built by the final stage. When the run is
    extended that round is refined after all — from the same nodes, so from
    the same graph — and is no longer anybody's final round."""
    nodes, store = panel
    rd = _rd(tmp_path)
    built = _spy(monkeypatch)
    params = EmParams(threads=1)
    _, _, final = _final_graph_and_merge(rd, nodes, _membership(nodes), 1, params, _log)
    _write_final_record(rd, final, 5, {})
    assert built == [1] and (rd / "final.json").exists()

    _refine_and_merge(rd, nodes, store, 1, params, _log)
    assert built == [1]
    assert not (rd / "final.json").exists()
    assert not (rd / "merged_final.parquet").exists()
    assert (rd / "refine.json").exists()


# ── what a resume makes of the record ─────────────────────────────────


def _finished(tmp_path, records, *, legacy=None):
    """A rounds/ directory whose rounds are done. records[r] is that round's
    refine.json, or None for a round that was never refined."""
    rounds_dir = tmp_path / "run" / "rounds"
    for r, record in records.items():
        rd = rounds_dir / f"r{r:02d}"
        rd.mkdir(parents=True)
        (rd / "_SUCCESS").write_bytes(b"")
        if record is not None:
            (rd / "refine.json").write_text(json.dumps(record))
    for r, stats in (legacy or {}).items():
        d = rounds_dir / f"r{r:02d}" / "merge"
        d.mkdir(parents=True, exist_ok=True)
        (d / "stats.json").write_text(json.dumps(stats))
    return rounds_dir


def _record(params, r, *, merged):
    record = {"round": r, "merge_applied": merged, "graph": "ok"}
    if merged:
        record["predicate"] = asdict(params.merge_predicate())
        record["graph_stamp"] = json.loads(
            json.dumps(rounds_mod.gr.graph_stamp(params.graph, "digest"))
        )
    return record


def test_nothing_is_checked_unless_the_run_is_resumed(tmp_path):
    report = EmParams(merge=False)
    rounds_dir = _finished(tmp_path, {1: _record(report, 1, merged=False)})
    _check_merge_state(rounds_dir, EmParams(merge=True), resume=False)
    _check_merge_state(tmp_path / "nowhere", EmParams(merge=True), resume=True)


def test_report_only_parameters_may_change_between_invocations(tmp_path):
    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        GraphParams,
    )

    rounds_dir = _finished(
        tmp_path,
        {1: _record(EmParams(merge=False), 1, merged=False), 2: None},
    )
    for params in (
        EmParams(merge=False),
        EmParams(merge=False, graph=GraphParams(tol_5p=10, identity_floor=0.98)),
        EmParams(merge=False, merge_max_edits=4, merge_siblings=True),
        EmParams(merge=False, template_graph="final"),
        EmParams(merge=False, template_graph="off"),
        # Merging from the first round that has not run yet is fine too.
        EmParams(merge_from_round=2),
    ):
        _check_merge_state(rounds_dir, params, resume=True)


def test_merge_cannot_be_switched_on_for_rounds_that_did_not_merge(tmp_path):
    report = EmParams(merge=False)
    rounds_dir = _finished(
        tmp_path,
        {
            1: _record(report, 1, merged=False),
            2: _record(report, 2, merged=False),
            3: None,
        },
    )
    with pytest.raises(
        ValueError, match=r"round 1 did not merge.*--merge-from-round 3"
    ):
        _check_merge_state(rounds_dir, EmParams(merge=True), resume=True)
    # ...but it can start where the record stops. Round 3 was the last round
    # and was never refined, so it accepts either.
    _check_merge_state(
        rounds_dir, EmParams(merge=True, merge_from_round=3), resume=True
    )


def test_merge_cannot_be_switched_off_for_rounds_that_merged(tmp_path):
    merging = EmParams(merge=True)
    rounds_dir = _finished(tmp_path, {1: _record(merging, 1, merged=True), 2: None})
    _check_merge_state(rounds_dir, merging, resume=True)
    with pytest.raises(ValueError, match="round 1 merged"):
        _check_merge_state(rounds_dir, EmParams(merge=False), resume=True)
    with pytest.raises(ValueError, match="round 1 merged"):
        _check_merge_state(
            rounds_dir, EmParams(merge=True, merge_from_round=2), resume=True
        )


def test_the_merge_from_round_it_suggests_is_one_it_accepts(tmp_path):
    """A run that began merging at round 3. "Past everything it refined" is
    round 4 — which would un-merge round 3."""
    began_at_3 = EmParams(merge=True, merge_from_round=3)
    rounds_dir = _finished(
        tmp_path,
        {r: _record(began_at_3, r, merged=r >= 3) for r in (1, 2, 3)} | {4: None},
    )
    for wrong in (
        EmParams(merge=True),
        EmParams(),
        EmParams(merge=True, merge_from_round=4),
    ):
        with pytest.raises(
            ValueError, match=r"--merge --merge-from-round 3, as it was run"
        ):
            _check_merge_state(rounds_dir, wrong, resume=True)
    _check_merge_state(rounds_dir, began_at_3, resume=True)


def test_a_changed_graph_tolerance_is_called_a_graph_parameter(tmp_path):
    """An unset merge tolerance IS the graph's, so the predicate moves with
    it — and naming the predicate would send the user to the wrong flags."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        GraphParams,
    )

    merging = EmParams(merge=True)
    rounds_dir = _finished(tmp_path, {1: _record(merging, 1, merged=True), 2: None})
    with pytest.raises(ValueError, match="graph built with graph") as refusal:
        _check_merge_state(
            rounds_dir, EmParams(merge=True, graph=GraphParams(tol_5p=20)), resume=True
        )
    assert "merge predicate" not in str(refusal.value)


def test_a_round_that_merged_commits_its_predicate_and_its_graph(tmp_path):
    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        GraphParams,
    )

    merging = EmParams(merge=True)
    rounds_dir = _finished(tmp_path, {1: _record(merging, 1, merged=True), 2: None})
    with pytest.raises(ValueError, match="merge predicate"):
        _check_merge_state(
            rounds_dir, EmParams(merge=True, merge_max_edits=1), resume=True
        )
    with pytest.raises(ValueError, match="merge predicate"):
        _check_merge_state(
            rounds_dir, EmParams(merge=True, merge_siblings=True), resume=True
        )
    with pytest.raises(ValueError, match="graph"):
        _check_merge_state(
            rounds_dir,
            EmParams(merge=True, graph=GraphParams(identity_floor=0.98)),
            resume=True,
        )
    # What the output does not depend on is not committed.
    _check_merge_state(
        rounds_dir,
        EmParams(merge=True, threads=48, graph=GraphParams(chunk_rows=1_000)),
        resume=True,
    )


def test_a_run_merged_by_the_removed_detector_cannot_be_resumed(tmp_path):
    rounds_dir = _finished(
        tmp_path,
        {1: None, 2: None},
        legacy={1: {"merge_applied": True, "n_merged": 412, "p_merge": 0.995}},
    )
    for params in (EmParams(), EmParams(merge=True)):
        with pytest.raises(ValueError, match="round 1 merged 412 templates.*minimap2"):
            _check_merge_state(rounds_dir, params, resume=True)


@pytest.mark.parametrize(
    "stats",
    [
        {"merge_applied": False, "n_merged": 0, "removable": 40},
        {"merge_applied": True, "n_merged": 0},
        {},
    ],
)
def test_a_legacy_run_that_merged_nothing_resumes(tmp_path, stats):
    """`--no-merge` under the old detector still scanned and still wrote
    `merge/stats.json`. Its templates are nobody's merge, so it resumes — as
    what it is: a run whose round 1 did not merge."""
    rounds_dir = _finished(tmp_path, {1: None, 2: None}, legacy={1: stats})
    _check_merge_state(rounds_dir, EmParams(merge=False), resume=True)
    _check_merge_state(
        rounds_dir, EmParams(merge=True, merge_from_round=2), resume=True
    )
    with pytest.raises(
        ValueError, match=r"round 1 did not merge.*--merge-from-round 2"
    ):
        _check_merge_state(rounds_dir, EmParams(merge=True), resume=True)


def test_a_legacy_final_merge_fed_no_templates(tmp_path):
    """Only `merge/` is between rounds. `merge_final/` collapsed the output
    of a run's last round, which no later round was built from."""
    rounds_dir = _finished(tmp_path, {1: None})
    d = rounds_dir / "r01" / "merge_final"
    d.mkdir()
    (d / "stats.json").write_text(json.dumps({"merge_applied": True, "n_merged": 9}))
    _check_merge_state(rounds_dir, EmParams(merge=True), resume=True)


# ── the report ────────────────────────────────────────────────────────


def _done(rd):
    (rd / "_SUCCESS").write_bytes(b"")


def test_the_report_counts_what_is_there_after_the_merge(panel, tmp_path):
    nodes, store = panel
    run = tmp_path / "run"
    rd1, rd2 = _rd(tmp_path, 1), _rd(tmp_path, 2)
    params = EmParams(threads=1, merge=True)
    _refine_and_merge(rd1, nodes, store, 1, params, _log)
    _done(rd1)
    merged_nodes, _, final = _final_graph_and_merge(
        rd2, nodes, _membership(nodes), 2, params, _log
    )
    _, counts = write_cluster_edges(
        run, final["edges_path"], final["keep_rows"], _clusters(merged_nodes)
    )
    _write_final_record(rd2, final, merged_nodes.num_rows, counts)
    _done(rd2)

    section = section_template_graph(run)
    assert section.title == "Template relationships"
    assert "| r1 | 5 |" in section.body and "| r2 | 5 |" in section.body
    assert "**4 clusters** (1 merged)" in section.body
    assert "0 clusters byte-identical" in section.body
    assert section.body.count("Final output") == 1
    assert "redundant" not in section.body
    assert not section.flags


def test_the_report_flags_twins_left_in_the_final_output(panel, tmp_path):
    nodes, _ = panel
    run = tmp_path / "run"
    rd = _rd(tmp_path)
    _, _, final = _final_graph_and_merge(
        rd, nodes, _membership(nodes), 1, EmParams(threads=1, merge=False), _log
    )
    _, counts = write_cluster_edges(
        run, final["edges_path"], final["keep_rows"], _clusters(nodes)
    )
    _write_final_record(rd, final, nodes.num_rows, counts)
    _done(rd)
    flags = section_template_graph(run).flags
    assert len(flags) == 1 and "2 clusters (40.0%) are byte-identical" in flags[0]


def test_the_report_says_when_there_was_no_graph(panel, tmp_path, monkeypatch):
    nodes, store = panel
    run = tmp_path / "run"
    rd = _rd(tmp_path)
    params = EmParams(threads=1, template_graph="off", merge=False)
    _refine_and_merge(rd, nodes, store, 1, params, _log)
    _done(rd)
    assert "template graph disabled" in section_template_graph(run).body
    assert section_template_graph(tmp_path / "nowhere").body == "_no completed rounds_"

    failed = tmp_path / "failed"
    rd = _rd(failed)
    monkeypatch.setattr(rounds_mod.gr, "build_graph", _boom)
    _refine_and_merge(rd, nodes, store, 1, EmParams(threads=1, merge=False), _log)
    _done(rd)
    assert "MemoryError" in section_template_graph(failed / "run").body


def test_the_report_does_not_read_the_removed_detector_s_numbers(tmp_path):
    rounds_dir = _finished(
        tmp_path,
        {1: None},
        legacy={1: {"merge_applied": True, "n_merged": 9, "removable_frac": 0.29}},
    )
    section = section_template_graph(rounds_dir.parent)
    assert "minimap2 redundancy scan" in section.body
    assert "29" not in section.body and not section.flags


# ── the bench's gate ──────────────────────────────────────────────────


def _bench():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "bench-template-graph.py"
    spec = importlib.util.spec_from_file_location("bench_template_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gate_stops_whatever_part_of_the_round_is_over_it():
    """The whole round holds every pair a sample of it does. 450M pairs from
    900,000 nodes was once "not judged: fewer than 1,000,000 nodes", exit 0."""
    bench = _bench()
    over, under = bench.GATE_PAIRS + 1, bench.GATE_PAIRS
    for n_all, sampled in ((900_000, False), (1_570_000, True), (1_570_000, False)):
        ok, verdict = bench.gate_verdict(over, n_all, sampled)
        assert not ok and verdict.startswith("STOP"), (n_all, sampled)
    assert bench.gate_verdict(under, 1_570_000, False) == (True, "PASS")
    ok, verdict = bench.gate_verdict(under, 1_570_000, True)
    assert ok and verdict.startswith("not judged")
    ok, verdict = bench.gate_verdict(under, bench.GATE_MIN_NODES - 1, False)
    assert ok and verdict.startswith("not judged")


@pytest.mark.parametrize("mode", ["sketch", "candidates", "full"])
def test_the_bench_reads_a_round_and_writes_nothing_into_it(
    panel, tmp_path, mode, monkeypatch, capsys
):
    nodes, _ = panel
    rd = _rd(tmp_path, 3)
    shards = rd / "mstep" / "nodes"
    shards.mkdir(parents=True)
    pq.write_table(nodes, shards / "part-00000.parquet")
    before = sorted(p.relative_to(rd) for p in rd.rglob("*"))

    bench = _bench()
    monkeypatch.setattr(
        "sys.argv", ["bench", str(rd), "--mode", mode, "--threads", "1"]
    )
    assert bench.main() == 0
    said = capsys.readouterr().out
    assert sorted(p.relative_to(rd) for p in rd.rglob("*")) == before
    assert "JOIN ROWS, exact" in said
    assert ("candidate pairs:" in said) == (mode != "sketch")
    assert ("== edges" in said) == (mode == "full")

    if mode != "sketch":
        monkeypatch.setattr(bench, "GATE_PAIRS", 0)
        assert bench.main() == 3
        assert "STOP" in capsys.readouterr().out


# ── the guards, where a pairwise predicate cannot see them ────────────


@pytest.fixture
def hub():
    """The two children of one split, and a hub from another lineage that
    lies between them.

    node 0   parent 10, 8 reads    the full transcript
    node 1   parent 10, 6 reads    its sibling, 12 nt shorter at the 5' end
    node 2   parent 11, 50 reads   the hub: 6 nt shorter

    Pairwise the hub is mergeable with each child and the children are not
    mergeable with each other — the stored `mergeable` column says exactly
    that. Only the grouping can put all three together.
    """
    a = _rnd(random.Random(13), 900)
    rows = [(10, 0, 0, a, 8), (10, 0, 1, a[12:], 6), (11, 1, 0, a[6:], 50)]
    return _nodes(rows), _store([10, 11])


def test_the_hub_does_not_rejoin_a_split_between_rounds(hub, tmp_path):
    nodes, store = hub
    rd = _rd(tmp_path)
    refined = _refine_and_merge(
        rd, nodes, store, 1, EmParams(threads=1, merge=True), _log
    )
    rel = _relations(rd / "graph" / "edges.parquet")
    assert rel[(2, 0)] == ("equivalent", True) and rel[(1, 2)] == ("equivalent", True)
    assert rel[(1, 0)] == ("equivalent", False)
    assert refined.n_merged == 1 and refined.templates.num_rows == 2
    record = _json(rd / "refine.json")
    assert record["n_merged_kin"] == 0
    assert record["n_still_mergeable"] == 1, "the pair the group guard refused"

    rd2 = _rd(tmp_path / "siblings")
    joined = _refine_and_merge(
        rd2, nodes, store, 1, EmParams(threads=1, merge=True, merge_siblings=True), _log
    )
    assert joined.n_merged == 2
    assert _json(rd2 / "refine.json")["n_merged_kin"] == 0, "neither is kin of the HUB"


def test_the_hub_does_not_rejoin_the_split_a_round_later(hub, tmp_path):
    """The survivor stands for what it absorbed. Round 1: the hub takes one
    child of the split and is refused the other. Round 2: both survivors
    come back unchanged, from two different parents — and if the hub's own
    line were all it remembered, the refused child would be no kin of it."""
    nodes, store = hub
    params = EmParams(threads=1, merge=True)
    rd1, rd2 = _rd(tmp_path, 1), _rd(tmp_path, 2)
    refined = _refine_and_merge(rd1, nodes, store, 1, params, _log)
    assert refined.n_merged == 1
    rounds_mod._write_lineage(rd1, refined.lineage)
    lineage = pq.read_table(rd1 / "lineage.parquet")
    assert sorted(lineage.column("rule").to_pylist()) == ["carry", "merge", "split"]

    ids = refined.templates.column("template_id").to_pylist()
    again = _nodes(
        [
            (tid, row, 0, seq, reads)
            for row, (tid, seq, reads) in enumerate(
                zip(
                    ids,
                    refined.templates.column("sequence").to_pylist(),
                    refined.templates.column("orf_replication").to_pylist(),
                )
            )
        ],
        r=2,
    )
    second = _refine_and_merge(rd2, again, _store(ids), 2, params, _log)
    edges = pq.read_table(rd2 / "graph" / "edges.parquet").to_pylist()
    assert [(e["relation"], e["n_edits"]) for e in edges] == [("equivalent", 0)]
    assert edges[0]["same_split_origin"] and not edges[0]["mergeable"]
    assert second.n_merged == 0 and second.templates.num_rows == 2


def test_the_hub_does_not_rejoin_a_split_in_the_final_output(tmp_path):
    """The final merge claims longest first, so there the hub that could
    rejoin a split is the LONGEST of the three: the full transcript, from
    another lineage than the two children, which are 6 and 12 nt shorter."""
    a = _rnd(random.Random(13), 900)
    nodes = _nodes([(10, 0, 0, a[6:], 8), (10, 0, 1, a[12:], 6), (11, 1, 0, a, 50)])
    member = _membership(nodes)
    rd = _rd(tmp_path)
    out_nodes, mem, final = _final_graph_and_merge(
        rd, nodes, member, 1, EmParams(threads=1, merge=True), _log
    )
    rel = _relations(rd / "graph" / "edges.parquet")
    assert rel[(0, 2)] == ("equivalent", True) and rel[(1, 2)] == ("equivalent", True)
    assert rel[(1, 0)] == ("equivalent", False)
    assert final["n_merged"] == 1 and final["keep_rows"].tolist() == [1, 2]
    assert out_nodes.column("n_reads").to_pylist() == [6, 58]
    assert mem.num_rows == member.num_rows
    _, _, joined = _final_graph_and_merge(
        _rd(tmp_path / "siblings"),
        nodes,
        member,
        1,
        EmParams(threads=1, merge=True, merge_siblings=True),
        _log,
    )
    assert joined["n_merged"] == 2


def _corpus(tmp_path, nodes, *, mutate=None) -> Path:
    """A read corpus laid out as `_membership` numbers the rows: node by
    node, `n_reads` reads each, every read the node's own consensus (or
    `mutate(rng, seq)` of it)."""
    from constellation.sequencing.transcriptome.cluster.denovo._io import _READS_SCHEMA

    rng = random.Random(99)
    seqs = []
    for seq, n in zip(
        nodes.column("consensus").to_pylist(), nodes.column("n_reads").to_pylist()
    ):
        seqs += [mutate(rng, seq) if mutate else seq for _ in range(n)]
    path = tmp_path / "reads.arrow"
    table = pa.table(
        {
            "read_id": pa.array([f"r{i}" for i in range(len(seqs))], pa.string()),
            "sequence": pa.array(seqs, pa.large_string()),
            "sample_id": pa.array(np.zeros(len(seqs), np.int64)),
            "dorado_quality": pa.array(np.full(len(seqs), 30.0, np.float32)),
        },
        schema=_READS_SCHEMA,
    )
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, _READS_SCHEMA) as w:
        w.write_table(table)
    return path


def _one_substitution(rng, seq):
    at = rng.randrange(20, len(seq) - 20)
    other = rng.choice([b for b in "ACGT" if b != seq[at]])
    return seq[:at] + other + seq[at + 1 :]


def test_the_final_survivor_is_the_deepest_and_is_rebuilt_from_the_pool(hub, tmp_path):
    """Between rounds the hub (50 reads) survives and the next M-step
    rebuilds its consensus. In the final output nothing follows, so the
    loop does the rebuild itself: the deepest member survives, every read of
    the group is aligned to it, and the consensus is built from the pool —
    with each read carrying one error of its own, which the pool outvotes.
    The 6 nt the eight full-length reads reach past the hub do NOT come
    back: the fifty hub-length reads are anchored at that end too and vote
    against them, which is the kernel's rule and the deepest form's start.
    Keeping the longest member instead was measured the wrong form more
    often than not (ledger #52)."""
    nodes, store = hub
    full = nodes.column("consensus")[0].as_py()
    between = _refine_and_merge(
        _rd(tmp_path), nodes, store, 1, EmParams(threads=1, merge=True), _log
    )
    assert full[6:] in between.templates.column("sequence").to_pylist()
    assert full not in between.templates.column("sequence").to_pylist()

    rd = _rd(tmp_path / "final")
    member = _membership(nodes)
    out_nodes, mem, final = _final_graph_and_merge(
        rd,
        nodes,
        member,
        1,
        EmParams(threads=1, merge=True),
        _log,
        corpus_path=_corpus(tmp_path, nodes, mutate=_one_substitution),
    )
    assert final["keep_rows"].tolist() == [1, 2], "the hub, not the full-length node"
    assert out_nodes.column("n_reads").to_pylist() == [6, 58]
    assert mem.num_rows == member.num_rows
    merged = pq.read_table(rd / "merged_final.parquet").to_pylist()
    assert [(m["delta_5p"], m["absorbed_n_reads"]) for m in merged] == [(-6, 8)]

    assert final["rebuild"] == "ok" and final["n_rebuilt"] == 1
    assert final["n_rebuild_failed"] == 0 and final["n_rebuild_reads_skipped"] == 0
    rebuilt = out_nodes.slice(1, 1).to_pylist()[0]
    assert rebuilt["consensus"] == full[6:], "the pool outvotes every read's error"
    assert rebuilt["n_members_used"] == 58 and rebuilt["subsample_fraction"] == 1.0
    assert rebuilt["n_extended_5p"] == 0, "8 flanked reads against 50 anchored"
    untouched = out_nodes.slice(0, 1).to_pylist()[0]
    assert untouched["consensus"] == full[12:] and untouched["n_members_used"] == 6


def test_the_final_merge_accepts_an_inexact_pair_within_the_cap_and_rebuilds(tmp_path):
    """One substitution apart. The final merge honours the run's cap, like
    the merges between rounds, and the survivor's consensus comes from the
    pooled reads: 30 of one form and 5 of the other vote the deeper base."""
    a = _rnd(random.Random(17), 900)
    b = a[:450] + ("C" if a[450] != "C" else "G") + a[451:]
    nodes = _nodes([(10, 0, 0, a, 30), (11, 1, 0, b, 5)])
    params = EmParams(threads=1, merge=True, merge_max_edits=1)

    rd = _rd(tmp_path)
    between = _refine_and_merge(rd, nodes, _store([10, 11]), 1, params, _log)
    assert _relations(rd / "graph" / "edges.parquet")[(0, 1)] == ("equivalent", True)
    assert between.n_merged == 1

    corpus = _corpus(tmp_path, nodes)
    out_nodes, _, final = _final_graph_and_merge(
        _rd(tmp_path / "final"),
        nodes,
        _membership(nodes),
        1,
        params,
        _log,
        corpus_path=corpus,
    )
    assert final["merge_applied"] and final["n_merged"] == 1
    assert final["predicate"]["max_edits"] == 1
    assert out_nodes.num_rows == 1
    (node,) = out_nodes.to_pylist()
    assert node["consensus"] == a and node["n_reads"] == 35
    assert node["n_members_used"] == 35 and final["n_rebuilt"] == 1

    # Held exact, the pair stays two clusters — the old final rule, by request.
    out_nodes, _, final = _final_graph_and_merge(
        _rd(tmp_path / "exact"),
        nodes,
        _membership(nodes),
        1,
        EmParams(threads=1, merge=True, merge_max_edits=0),
        _log,
        corpus_path=corpus,
    )
    assert final["n_merged"] == 0 and out_nodes.num_rows == 2
    assert "rebuild" not in final


def test_the_rebuild_runs_in_a_pool_and_agrees_with_the_parent(tmp_path):
    """Three merged groups, two workers: the same table as one worker, and
    the survivors the merge did not touch keep their rows exactly."""
    rng = random.Random(21)
    rows = []
    for p in range(3):
        seq = _rnd(rng, 700 + 100 * p)
        rows += [(10 + p, p, 0, seq, 12), (20 + p, 3 + p, 0, seq[8:], 4)]
    rows.append((30, 6, 0, _rnd(rng, 650), 9))
    nodes = _nodes(rows)
    corpus = _corpus(tmp_path, nodes, mutate=_one_substitution)
    outs = []
    for threads in (1, 2):
        out_nodes, _, final = _final_graph_and_merge(
            _rd(tmp_path / f"t{threads}"),
            nodes,
            _membership(nodes),
            1,
            EmParams(threads=threads, merge=True),
            _log,
            corpus_path=corpus,
        )
        assert final["n_merged"] == 3 and final["n_rebuilt"] == 3
        outs.append(out_nodes)
    assert outs[0].equals(outs[1])
    assert outs[0].num_rows == 4
    by_parent = {r["parent_template_id"]: r for r in outs[0].to_pylist()}
    assert by_parent[30]["n_members_used"] == 9, "untouched"
    for p in range(3):
        assert by_parent[10 + p]["n_reads"] == 16
        assert by_parent[10 + p]["n_members_used"] == 16
        assert by_parent[10 + p]["consensus"] == rows[2 * p][3]


# ── what is left behind ───────────────────────────────────────────────


def test_a_round_without_a_graph_of_its_own_keeps_nobody_else_s(panel, tmp_path):
    """A former final round, refined after all by a run that relates
    nothing there. The graph it was left with is another configuration's."""
    nodes, store = panel
    for mode in ("off", "final"):
        rd = _rd(tmp_path / mode)
        _final_graph_and_merge(
            rd, nodes, _membership(nodes), 1, EmParams(threads=1), _log
        )
        assert (rd / "graph" / "_SUCCESS").exists()
        _refine_and_merge(
            rd,
            nodes,
            store,
            1,
            EmParams(threads=1, template_graph=mode, merge=mode != "off"),
            _log,
        )
        assert not (rd / "graph").exists()
    rd = _rd(tmp_path / "final-off")
    _final_graph_and_merge(rd, nodes, _membership(nodes), 1, EmParams(threads=1), _log)
    _final_graph_and_merge(
        rd,
        nodes,
        _membership(nodes),
        1,
        EmParams(threads=1, template_graph="off", merge=False),
        _log,
    )
    assert not (rd / "graph").exists()


def test_a_round_with_no_nodes_is_not_said_to_have_the_graph_off(tmp_path):
    nodes = REFINED_NODE_TABLE.empty_table()
    member = NODE_MEMBERSHIP_TABLE.empty_table()
    _, _, final = _final_graph_and_merge(
        _rd(tmp_path), nodes, member, 1, EmParams(threads=1), _log
    )
    assert final["graph"] == "empty" and final["template_graph"] == "rounds"


def test_an_interrupt_is_not_a_failed_report(panel, tmp_path, monkeypatch):
    """Ctrl-C during a report-only graph stops the run; it is not recorded
    as a graph that failed and then carried on from."""
    nodes, store = panel

    def _interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(rounds_mod.gr, "build_graph", _interrupt)
    rd = _rd(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        _refine_and_merge(rd, nodes, store, 1, EmParams(threads=1), _log)
    assert not (rd / "refine.json").exists()


def test_a_failed_edge_projection_does_not_cost_the_run_its_manifest(
    panel, tmp_path, monkeypatch
):
    """The clusters are written by then and nothing reads the edge file
    back, merge or no merge. Its failure is recorded; the run goes on to its
    final record."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        _cluster_edges,
    )

    nodes, _ = panel
    out = tmp_path / "run"
    for merge in (False, True):
        rd = _rd(tmp_path / str(merge))
        merged_nodes, _, final = _final_graph_and_merge(
            rd, nodes, _membership(nodes), 1, EmParams(threads=1, merge=merge), _log
        )
        (out / "cluster_edges.parquet.tmp").parent.mkdir(parents=True, exist_ok=True)
        (out / "cluster_edges.parquet.tmp").write_bytes(b"half a file")

        def _full_disk(*_a, **_k):
            raise OSError("No space left on device")

        monkeypatch.setattr(rounds_mod, "write_cluster_edges", _full_disk)
        paths: dict = {}
        said: list[str] = []
        counts = _cluster_edges(out, final, _clusters(merged_nodes), paths, said.append)
        assert counts == {} and "cluster_edges" not in paths
        assert not list(out.glob("cluster_edges.parquet*"))
        assert any("could not be written" in line for line in said)
        _write_final_record(rd, final, merged_nodes.num_rows, counts)
        record = _json(rd / "final.json")
        assert "No space left" in record["cluster_edges_error"]
        assert record["graph"] == "ok" and record["n_clusters"] == merged_nodes.num_rows
        _done(rd)
        body = section_template_graph(rd.parent.parent).body
        assert "could not be written" in body and "No space left" in body
        monkeypatch.undo()


# ── the report's other columns and flags ──────────────────────────────


def test_survivors_the_next_mstep_split_again_are_counted(hub, tmp_path):
    """The M-step clusters span endpoints within 10 nt and the merge
    tolerance is 30, so what is merged on an extent difference in between
    can be separated again a round later. This is the count that shows it."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.diagnostics import (
        _resplit_survivors,
    )

    nodes, store = hub
    rd1, rd2 = _rd(tmp_path, 1), _rd(tmp_path, 2)
    refined = _refine_and_merge(
        rd1, nodes, store, 1, EmParams(threads=1, merge=True), _log
    )
    survivor, other = refined.templates.column("template_id").to_pylist()[::-1]
    assert pq.read_table(rd1 / "merged.parquet").column(
        "survivor_template_id"
    ).to_pylist() == [survivor]
    assert _resplit_survivors(rd1, None) is None
    assert _resplit_survivors(rd1, rd2) is None, "no nodes in the next round yet"

    def next_round(rows):
        d = rd2 / "mstep" / "nodes"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(_nodes(rows, r=2), d / "part-00000.parquet")

    seq = "ACGT" * 25
    next_round(
        [(survivor, 0, 0, seq, 40), (other, 1, 0, seq, 6), (other, 1, 1, seq, 3)]
    )
    assert _resplit_survivors(rd1, rd2) == 0, (
        "the template that split was not the survivor"
    )
    next_round(
        [(survivor, 0, 0, seq, 40), (survivor, 0, 1, seq, 9), (other, 1, 0, seq, 6)]
    )
    assert _resplit_survivors(rd1, rd2) == 1

    _done(rd1)
    _done(rd2)
    body = section_template_graph(tmp_path / "run").body
    row = next(line for line in body.splitlines() if line.startswith("| r1 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[8:11] == ["1", "1", "1"], "merged, still mergeable, re-split"


def test_the_report_flags_a_merge_that_rejoined_a_split(panel, tmp_path):
    """Only when it did. `--merge-siblings` with nothing to rejoin is not a
    finding."""
    nodes, store = panel
    rd = _rd(tmp_path)
    _refine_and_merge(
        rd, nodes, store, 1, EmParams(threads=1, merge=True, merge_siblings=True), _log
    )
    _done(rd)
    assert _json(rd / "refine.json")["n_merged_kin"] == 1
    flags = section_template_graph(tmp_path / "run").flags
    assert len(flags) == 1 and "1 of 2 merges rejoined" in flags[0]

    quiet = _rd(tmp_path / "quiet")
    unrelated = nodes.slice(2, 3)
    _refine_and_merge(
        quiet,
        unrelated,
        store,
        1,
        EmParams(threads=1, merge=True, merge_siblings=True),
        _log,
    )
    _done(quiet)
    assert not section_template_graph(tmp_path / "quiet" / "run").flags


def test_the_report_flags_templates_that_hit_a_candidate_cap(hub, tmp_path):
    """The shortest of the three has two candidates; kept to one, its edge
    list is incomplete and the report has to say so."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        GraphParams,
    )

    nodes, store = hub
    rd = _rd(tmp_path)
    capped = GraphParams(max_candidates=1)
    _refine_and_merge(rd, nodes, store, 1, EmParams(threads=1, graph=capped), _log)
    _done(rd)
    assert _json(rd / "graph" / "stats.json")["n_truncated_templates"] >= 1
    flags = section_template_graph(tmp_path / "run").flags
    assert len(flags) == 1 and "hit a candidate cap" in flags[0]
    assert "edge lists are incomplete" in flags[0]


def test_a_run_that_relates_only_its_final_round_says_so_before_it_gets_there(
    panel, tmp_path
):
    nodes, store = panel
    rd = _rd(tmp_path)
    _refine_and_merge(
        rd, nodes, store, 1, EmParams(threads=1, template_graph="final"), _log
    )
    _done(rd)
    body = section_template_graph(tmp_path / "run").body
    assert "final round only" in body and "disabled" not in body
