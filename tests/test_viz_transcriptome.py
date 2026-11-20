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


def test_six_requests_for_one_view_are_one_replay(run_dir) -> None:
    """The browser asks for a summary and five tables at once; the replay
    and the tables are made once (single flight), and kept. Sent together on
    one event loop, as a browser's are (a TestClient shared across threads
    is not safe to use so)."""
    import httpx

    cache.clear_all()
    app = create_app({})

    async def go() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            opened = await c.post(
                "/api/sessions/open",
                json={"modality": "transcriptome", "sources": [{"path": str(run_dir)}]},
            )
            sid = opened.json()["session_id"]
            cid = _cluster_nodes(run_dir).column("cluster_id")[0].as_py()
            base = f"{_base(sid, run_dir)}/clusters/{cid}/support"
            replays, supports = cache.REPLAYS.computed, cache.SUPPORT.computed
            urls = [base] + [f"{base}/{t}" for t in TABLES[:5]]
            got = await asyncio.gather(*[c.get(u) for u in urls])
            assert [r.status_code for r in got] == [200] * 6
            assert cache.REPLAYS.computed - replays == 1
            assert cache.SUPPORT.computed - supports == 1
            await c.get(f"{base}/arcs")
            assert cache.SUPPORT.computed - supports == 1
            # Another space of the same cluster reuses the replay.
            await c.get(base, params={"space": "frame"})
            assert cache.REPLAYS.computed - replays == 1
            assert cache.SUPPORT.computed - supports == 2

    asyncio.run(go())


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


# ── review of the support branch (2026-11-20) ─────────────────────────


def test_a_run_finalized_again_is_replayed_again(client, run_dir, tmp_path) -> None:
    """Cached replays are keyed by the run's revision: when the files a
    replay reads change, the old verdict is not served, even after the
    session is reopened."""
    out = tmp_path / "em"
    shutil.copytree(run_dir, out)
    nodes = _cluster_nodes(out)
    row = next(r for r in nodes.to_pylist() if not r["rebuilt"])
    sid = _open(client, out)
    url = f"{_base(sid, out)}/clusters/{row['cluster_id']}/support"
    assert client.get(url).json()["verified"] is True
    ids = nodes.column("consensus_content_id").to_pylist()
    i = nodes.column("cluster_id").to_pylist().index(row["cluster_id"])
    ids[i] ^= 1
    changed = nodes.set_column(
        nodes.schema.get_field_index("consensus_content_id"),
        "consensus_content_id",
        pa.array(ids, pa.int64()),
    )
    pq.write_table(changed, out / "cluster_nodes.parquet")
    sid = _open(client, out)
    got = client.get(f"{_base(sid, out)}/clusters/{row['cluster_id']}/support").json()
    assert got["verified"] is False


def test_a_cancelled_request_neither_poisons_its_key_nor_cancels_others() -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    async def alone(pool) -> None:
        # The only request for a queued key is cancelled: the work is
        # cancelled with it, and the key can be asked for again.
        flight = cache.SingleFlight(2, pool)
        release, running = threading.Event(), threading.Event()

        def block():
            running.set()
            release.wait(5)

        first = asyncio.create_task(flight.aget("blocker", block))
        while not running.is_set():
            await asyncio.sleep(0.001)
        queued = asyncio.create_task(flight.aget("wanted", lambda: 42))
        await asyncio.sleep(0.01)
        queued.cancel()
        await asyncio.gather(queued, return_exceptions=True)
        release.set()
        await first
        assert await asyncio.wait_for(flight.aget("wanted", lambda: 43), 2) == 43

    async def shared(pool) -> None:
        # Two requests share queued work and one is cancelled: the other
        # still gets the value.
        release = threading.Event()
        blocker = pool.submit(lambda: release.wait(5))
        flight = cache.SingleFlight(2, pool)
        a = asyncio.create_task(flight.aget("same", lambda: 42))
        b = asyncio.create_task(flight.aget("same", lambda: 99))
        await asyncio.sleep(0.01)
        a.cancel()
        await asyncio.gather(a, return_exceptions=True)
        release.set()
        blocker.result()
        assert await asyncio.wait_for(b, 2) == 42

    for scenario in (alone, shared):
        with ThreadPoolExecutor(1) as pool:
            asyncio.run(scenario(pool))


def test_adding_or_removing_a_source_keeps_each_demux_dir(client, run_dir, tmp_path) -> None:
    demux = tmp_path / "demux"
    (demux / "samples").mkdir(parents=True)
    pq.write_table(
        pa.table({"sample_id": pa.array([0, 1], pa.int64()), "sample_name": ["a", "b"]}),
        demux / "samples" / "samples.parquet",
    )
    other = tmp_path / "em2"
    shutil.copytree(run_dir, other)
    sid = _open(client, run_dir, demux_dir=str(demux))
    added = client.post(
        f"/api/sessions/{sid}/sources", json={"path": str(other), "demux_dir": str(demux)}
    )
    assert added.status_code == 201, added.text
    runs = {r["path"]: r for r in added.json()["runs"]}
    assert runs[str(run_dir.resolve())]["samples"] == {"0": "a", "1": "b"}
    assert runs[str(other.resolve())]["samples"] == {"0": "a", "1": "b"}
    source_id = derive_source_id(other.resolve(), "em_run")
    removed = client.delete(f"/api/sessions/{sid}/sources/{source_id}")
    assert removed.status_code == 200, removed.text
    assert removed.json()["runs"][0]["samples"] == {"0": "a", "1": "b"}


@pytest.mark.parametrize("value", [[], None, 3, {"kind": "cluster", "parameters": [1]}])
def test_a_manifest_of_another_shape_is_refused(client, tmp_path, value) -> None:
    import json

    (tmp_path / "rounds").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps(value))
    response = client.post(
        "/api/sessions/open",
        json={"modality": "transcriptome", "sources": [{"path": str(tmp_path)}]},
    )
    assert response.status_code == 400
    assert "not an EM run" in response.text


def test_out_of_range_queries_are_refused_or_match_nothing(client, run_dir) -> None:
    sid = _open(client, run_dir)
    url = f"{_base(sid, run_dir)}/clusters"
    for param in ("min_reads", "offset"):
        assert client.get(url, params={param: 2**80}).status_code == 422
    assert _arrow(client.get(url, params={"min_reads": 2**63 - 1})).num_rows == 0
    for q in (str(2**63), str(2**80), "9" * 5000, "²"):
        response = client.get(url, params={"q": q})
        assert _arrow(response).num_rows == 0, q
        assert response.headers["X-Total-Rows"] == "0"
