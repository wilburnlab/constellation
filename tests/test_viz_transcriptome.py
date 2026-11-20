"""The transcriptome modality over HTTP, on real EM runs of the evidence
fixture (``_em_fixtures.three_gene_panel``: ORF seeding, native E-step, no
minimap2) — one round, whose final merge rebuilds a survivor and whose gene
C template splits."""

from __future__ import annotations

import asyncio
import io
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).parent))
from _em_fixtures import three_gene_panel  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (  # noqa: E402
    EmParams,
    run_em,
)
from constellation.viz.modalities.transcriptome import cache  # noqa: E402
from constellation.viz.server.app import create_app  # noqa: E402
from constellation.viz.server.session import derive_source_id  # noqa: E402


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("tx_viz")
    demux = three_gene_panel(base)
    out = base / "em"
    run_em(
        demux,
        out,
        params=EmParams(
            rounds=1,
            threads=1,
            mstep_workers=1,
            max_window_length=None,
            min_aa_length=30,
            estep_aligner="native",
        ),
        report=False,
    )
    return out


@pytest.fixture
def client() -> TestClient:
    cache.clear_all()
    return TestClient(create_app({}))


def _open(client: TestClient, *runs: Path, **extra) -> str:
    body = {"modality": "transcriptome", "sources": [{"path": str(r), **extra} for r in runs]}
    response = client.post("/api/sessions/open", json=body)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _arrow(response) -> pa.Table:
    assert response.status_code == 200, response.text
    return pa.ipc.RecordBatchStreamReader(io.BytesIO(response.content)).read_all()


def _base(sid: str, run: Path) -> str:
    return f"/api/transcriptome/sessions/{sid}/runs/{derive_source_id(run.resolve(), 'em_run')}"


def _cluster_nodes(run: Path) -> pa.Table:
    return pq.read_table(run / "cluster_nodes.parquet")


# ── the session ───────────────────────────────────────────────────────


def test_a_session_describes_its_runs(client, run_dir) -> None:
    sid = _open(client, run_dir)
    runs = client.get(f"/api/transcriptome/sessions/{sid}/runs").json()
    assert len(runs) == 1
    run = runs[0]
    assert run["finished"] and run["rounds"] == [1] and run["n_reads"] == 110
    assert run["has_compact_evidence"]
    summary = client.get("/api/sessions").json()
    assert any(s["session_id"] == sid and s["modality"] == "transcriptome" for s in summary)


def test_anything_but_an_em_run_is_refused(client, tmp_path) -> None:
    response = client.post(
        "/api/sessions/open",
        json={"modality": "transcriptome", "sources": [{"path": str(tmp_path)}]},
    )
    assert response.status_code == 400
    assert "not an EM run" in response.text


def test_sample_names_come_from_the_demux_directory_given(client, run_dir, tmp_path) -> None:
    # The fixture's demultiplexing output has no samples table: a warning.
    sid = _open(client, run_dir)
    manifest = client.get(f"/api/sessions/{sid}/manifest").json()
    assert any("no samples" in w for w in manifest["warnings"])
    assert manifest["runs"][0]["samples"] == {}
    # A run copied from elsewhere is given its demultiplexing directory.
    (tmp_path / "samples").mkdir()
    pq.write_table(
        pa.table({"sample_id": pa.array([0, 1, 2], pa.int64()), "sample_name": ["a", "b", "c"]}),
        tmp_path / "samples" / "samples.parquet",
    )
    sid = _open(client, run_dir, demux_dir=str(tmp_path))
    manifest = client.get(f"/api/sessions/{sid}/manifest").json()
    assert manifest["runs"][0]["samples"] == {"0": "a", "1": "b", "2": "c"}
    cid = pq.read_table(run_dir / "clusters.parquet").column("cluster_id")[0].as_py()
    detail = client.get(f"{_base(sid, run_dir)}/clusters/{cid}").json()
    assert {s["sample"] for s in detail["samples"]} <= {"a", "b", "c"}


# ── clusters ──────────────────────────────────────────────────────────


def test_clusters_page_sorted_and_counted(client, run_dir) -> None:
    sid = _open(client, run_dir)
    response = client.get(f"{_base(sid, run_dir)}/clusters", params={"limit": 3})
    page = _arrow(response)
    total = pq.read_metadata(run_dir / "clusters.parquet").num_rows
    assert int(response.headers["X-Total-Rows"]) == total
    assert page.num_rows == 3
    n = page.column("n_reads").to_pylist()
    assert n == sorted(n, reverse=True)
    assert "consensus_sequence" not in page.column_names
    assert {"length", "protein_length", "rebuilt", "node_template_id"} <= set(page.column_names)
    # Paging walks the same order.
    rest = _arrow(client.get(f"{_base(sid, run_dir)}/clusters", params={"offset": 3, "limit": 1000}))
    assert page.num_rows + rest.num_rows == total
    assert max(rest.column("n_reads").to_pylist()) <= min(n)


def test_clusters_filtered_searched_and_sorted_otherwise(client, run_dir) -> None:
    sid = _open(client, run_dir)
    big = _arrow(client.get(f"{_base(sid, run_dir)}/clusters", params={"min_reads": 10}))
    assert big.num_rows and min(big.column("n_reads").to_pylist()) >= 10
    cid = big.column("cluster_id")[0].as_py()
    hit = _arrow(client.get(f"{_base(sid, run_dir)}/clusters", params={"q": str(cid)}))
    assert hit.column("cluster_id").to_pylist() == [cid]
    by_len = _arrow(
        client.get(
            f"{_base(sid, run_dir)}/clusters", params={"sort": "length", "descending": "false"}
        )
    )
    lengths = by_len.column("length").to_pylist()
    assert lengths == sorted(lengths)
    assert client.get(f"{_base(sid, run_dir)}/clusters", params={"sort": "consensus"}).status_code == 422


def test_a_cluster_s_detail(client, run_dir) -> None:
    sid = _open(client, run_dir)
    clusters = pq.read_table(run_dir / "clusters.parquet")
    row = clusters.slice(0, 1).to_pylist()[0]
    detail = client.get(f"{_base(sid, run_dir)}/clusters/{row['cluster_id']}").json()
    assert detail["protein"] == row["predicted_protein"]
    assert sum(s["count"] for s in detail["samples"]) == row["n_reads"]
    # 63-bit content ids are strings: a JavaScript number cannot hold them.
    assert isinstance(detail["consensus_content_id"], str)
    assert client.get(f"{_base(sid, run_dir)}/clusters/999999").status_code == 404


# ── the support view ──────────────────────────────────────────────────

TABLES = ("reads", "departures", "insertions", "columns", "arcs", "differences")


def test_a_cluster_s_support_summary_and_tables_agree(client, run_dir) -> None:
    sid = _open(client, run_dir)
    nodes = _cluster_nodes(run_dir)
    plain = nodes.filter(pc.invert(nodes.column("rebuilt"))).column("cluster_id")[0].as_py()
    base = f"{_base(sid, run_dir)}/clusters/{plain}/support"
    meta = client.get(base).json()
    assert meta["kind"] == "node" and meta["verified"], meta["mismatch"]
    assert meta["reference"] == str(meta["haplotype_id"])
    assert len(meta["consensus"]) == meta["n_columns"]
    tables = {name: _arrow(client.get(f"{base}/{name}")) for name in TABLES}
    for name, table in tables.items():
        assert table.num_rows == meta["rows"][name], name
    n_reads = pq.read_table(run_dir / "clusters.parquet", filters=[("cluster_id", "==", plain)])
    assert tables["reads"].num_rows == n_reads.column("n_reads")[0].as_py()
    assert tables["columns"].num_rows == meta["n_columns"]
    assert client.get(f"{base}/nope").status_code == 404


def test_a_split_template_pooled_shows_its_nodes_and_where_they_differ(client, run_dir) -> None:
    sid = _open(client, run_dir)
    templates = pa.concat_tables(
        [pq.read_table(p) for p in sorted((run_dir / "rounds" / "r01" / "mstep" / "templates").glob("*.parquet"))]
    )
    split = templates.filter(pc.greater_equal(templates.column("n_nodes"), 2))
    tid = split.column("template_id")[0].as_py()
    base = f"{_base(sid, run_dir)}/rounds/1/templates/{tid}/support"
    meta = client.get(base, params={"space": "frame"}).json()
    assert meta["kind"] == "template" and meta["verified"]
    assert meta["reference"] == "pooled" and len(meta["nodes"]) >= 2
    reads = _arrow(client.get(f"{base}/reads", params={"space": "frame"}))
    assert set(reads.column("node").to_pylist()) == {n["haplotype_id"] for n in meta["nodes"]}
    diffs = _arrow(client.get(f"{base}/differences", params={"space": "frame"}))
    # Gene C's haplotypes differ at three 3' UTR sites.
    assert diffs.num_rows >= 3 * 2
    arcs = _arrow(client.get(f"{base}/arcs", params={"space": "frame"}))
    assert arcs.num_rows >= 3
    one = client.get(base, params={"node": meta["nodes"][1]["haplotype_id"]}).json()
    assert one["reference"] == str(meta["nodes"][1]["haplotype_id"])


def test_a_rebuilt_cluster_names_what_its_reads_came_from(client, run_dir) -> None:
    sid = _open(client, run_dir)
    nodes = _cluster_nodes(run_dir)
    rebuilt = nodes.filter(nodes.column("rebuilt")).column("cluster_id")[0].as_py()
    base = f"{_base(sid, run_dir)}/clusters/{rebuilt}/support"
    meta = client.get(base).json()
    assert meta["kind"] == "rebuilt" and meta["verified"] and meta["n_absorbed_reads"] > 0
    reads = _arrow(client.get(f"{base}/reads"))
    absorbed = reads.filter(pc.is_valid(reads.column("absorbed_from")))
    assert absorbed.num_rows == meta["n_absorbed_reads"]


def test_six_requests_for_one_view_are_one_replay(client, run_dir) -> None:
    """The browser asks for a summary and five tables at once; the replay
    and the tables are made once (single flight), and kept."""
    sid = _open(client, run_dir)
    nodes = _cluster_nodes(run_dir)
    cid = nodes.column("cluster_id")[0].as_py()
    base = f"{_base(sid, run_dir)}/clusters/{cid}/support"
    replays, supports = cache.REPLAYS.computed, cache.SUPPORT.computed
    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(6) as pool:
        urls = [base] + [f"{base}/{t}" for t in TABLES[:5]]
        codes = list(pool.map(lambda u: client.get(u).status_code, urls))
    assert codes == [200] * 6
    assert cache.REPLAYS.computed - replays == 1
    assert cache.SUPPORT.computed - supports == 1
    client.get(f"{base}/arcs")
    assert cache.SUPPORT.computed - supports == 1
    # Another space of the same cluster reuses the replay.
    client.get(base, params={"space": "frame"})
    assert cache.REPLAYS.computed - replays == 1
    assert cache.SUPPORT.computed - supports == 2


def test_what_is_not_there_is_404(client, run_dir) -> None:
    sid = _open(client, run_dir)
    base = _base(sid, run_dir)
    assert client.get(f"{base}/clusters/999999/support").status_code == 404
    assert client.get(f"{base}/rounds/1/templates/123/support").status_code == 404
    assert client.get(f"/api/transcriptome/sessions/{sid}/runs/nope/clusters").status_code == 404
    assert client.get(f"{base}/clusters/0/support", params={"node": "x"}).status_code == 422


def test_an_unfinished_run_serves_templates_only(client, run_dir, tmp_path) -> None:
    out = tmp_path / "em"
    shutil.copytree(run_dir, out)
    (out / "clusters.parquet").unlink()
    (out / "manifest.json").unlink()
    sid = _open(client, out)
    manifest = client.get(f"/api/sessions/{sid}/manifest").json()
    assert any("templates only" in w for w in manifest["warnings"])
    assert client.get(f"{_base(sid, out)}/clusters").status_code == 404


def test_single_flight_shares_one_computation_and_keeps_no_failure() -> None:
    from concurrent.futures import ThreadPoolExecutor

    flight = cache.SingleFlight(max_entries=2, executor=ThreadPoolExecutor(1))
    calls = []

    def slow():
        import time

        time.sleep(0.05)
        calls.append(1)
        return 42

    async def both():
        return await asyncio.gather(flight.aget("k", slow), flight.aget("k", slow))

    assert asyncio.run(both()) == [42, 42]
    assert calls == [1]

    def boom():
        raise RuntimeError("no")

    with pytest.raises(RuntimeError):
        flight.get("bad", boom)
    assert flight.get("bad", lambda: 7) == 7  # a failure is not kept


def test_the_cli_refuses_to_start_with_nothing_to_open(tmp_path, capsys) -> None:
    from constellation.viz.cli import cmd_viz_transcriptome

    class Args:
        run = None
        saved_session = None
        demux_dir = None
        label = None

    assert cmd_viz_transcriptome(Args()) == 2
    Args.run = [tmp_path]
    assert cmd_viz_transcriptome(Args()) == 1
    assert "not an EM run" in capsys.readouterr().err
