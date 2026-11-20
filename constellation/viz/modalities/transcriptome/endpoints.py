"""Transcriptome-only endpoints.

- ``GET  /api/transcriptome/sessions/{sid}/runs``                    → each EM run, described
- ``GET  …/runs/{key}/clusters?q=&sort=&descending=&min_reads=&offset=&limit=``
                                                                     → a page of the cluster index (Arrow;
                                                                       ``X-Total-Rows`` says how many matched)
- ``GET  …/runs/{key}/clusters/{cid}``                               → one cluster: index row, protein, counts by sample
- ``GET  …/runs/{key}/clusters/{cid}/support?node=&space=``          → the support view's summary (JSON)
- ``GET  …/runs/{key}/clusters/{cid}/support/{table}?node=&space=``  → one of its tables (Arrow)
- ``GET  …/runs/{key}/rounds/{r}/templates/{tid}/support[/{table}]`` → the same for any template of any round

``node`` is ``cluster`` (a cluster's own node; the default for a cluster),
``pooled`` (every read of the template against its pooled consensus — where
a split shows; the default for a template) or a haplotype id. ``space`` is
``consensus`` or ``frame``. The tables are ``reads``, ``departures``,
``insertions``, ``columns``, ``arcs`` and ``differences`` (the frame columns
where the template's nodes differ).

A summary and its tables come from one replay, made once however many of
them arrive together (``cache.SUPPORT``). Ids above 2**53 (content ids) are
strings in JSON, which a JavaScript number cannot hold; Arrow carries them as
int64.

Imported only through the modality's ``routers()`` factory — it needs fastapi.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query, Request

from constellation.viz.modalities.transcriptome.cache import (
    REPLAYS,
    SUPPORT,
    cluster_index_for,
)
from constellation.viz.modalities.transcriptome.session import TranscriptomeSession, TxRun
from constellation.viz.server.arrow_stream import batches_to_response

router = APIRouter(prefix="/api/transcriptome/sessions/{session_id}", tags=["transcriptome"])

CLUSTER_SORTS = (
    "n_reads",
    "length",
    "protein_length",
    "n_unique_sequences",
    "n_absorbed",
    "cluster_id",
)
SUPPORT_TABLES = ("reads", "departures", "insertions", "columns", "arcs", "differences")
#: JSON integers a JavaScript number holds exactly.
_JS_SAFE = 2**53


def _session(request: Request, session_id: str) -> TranscriptomeSession:
    session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")
    if getattr(session, "modality", None) != TranscriptomeSession.modality:
        raise HTTPException(404, f"session {session_id} is not a transcriptome session")
    return session


def _run(request: Request, session_id: str, key: str) -> TxRun:
    try:
        return _session(request, session_id).run(key)
    except KeyError:
        raise HTTPException(404, f"session {session_id} has no run {key!r}") from None


def _finished(run: TxRun) -> None:
    if not run.info.get("finished"):
        raise HTTPException(404, f"run {run.label} has no final clusters")


def _plain(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= _JS_SAFE:
        return str(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _arrow(table: pa.Table, headers: dict[str, str] | None = None):
    return batches_to_response(table.schema, table.to_batches(), headers=headers)


@router.get("/runs")
def list_runs(request: Request, session_id: str) -> list[dict[str, Any]]:
    return [_plain(r.describe()) for r in _session(request, session_id).runs]


def _page(
    index: pa.Table,
    run: TxRun,
    q: str | None,
    sort: str,
    descending: bool,
    min_reads: int,
    offset: int,
    limit: int,
) -> tuple[pa.Table, int]:
    table = index
    if q and q.strip():
        from constellation.sequencing.transcriptome.cluster.denovo.em.results import (
            find_clusters,
        )

        table = find_clusters(run.path, q, limit=index.num_rows, index=index)
    if min_reads > 0:
        table = table.filter(pc.greater_equal(table.column("n_reads"), min_reads))
    total = table.num_rows
    order = pc.sort_indices(
        table,
        sort_keys=[
            (sort, "descending" if descending else "ascending"),
            ("cluster_id", "ascending"),
        ],
    )
    return table.take(order.slice(offset, limit)), total


@router.get("/runs/{key}/clusters")
async def list_clusters(
    request: Request,
    session_id: str,
    key: str,
    q: str | None = None,
    sort: str = "n_reads",
    descending: bool = True,
    min_reads: int = Query(1, ge=0),
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=5000),
):
    run = _run(request, session_id, key)
    _finished(run)
    if sort not in CLUSTER_SORTS:
        raise HTTPException(422, f"sort must be one of {CLUSTER_SORTS}")
    index = await cluster_index_for(run.path)
    page, total = await asyncio.to_thread(
        _page, index, run, q, sort, descending, min_reads, offset, limit
    )
    return _arrow(page, headers={"X-Total-Rows": str(total)})


def _detail(index: pa.Table, run: TxRun, cluster_id: int) -> dict[str, Any]:
    row = index.filter(pc.equal(index.column("cluster_id"), cluster_id)).to_pylist()
    if not row:
        raise HTTPException(404, f"run {run.label} has no cluster {cluster_id}")
    out = dict(row[0])
    prot = pq.read_table(
        run.path / "clusters.parquet",
        columns=["predicted_protein"],
        filters=[("cluster_id", "==", cluster_id)],
    )
    out["protein"] = prot.column(0)[0].as_py() if prot.num_rows else None
    names = run.info.get("samples") or {}
    counts = []
    fq = run.path / "feature_quant.parquet"
    if fq.exists():
        t = pq.read_table(
            fq,
            columns=["sample_id", "count", "tpm"],
            filters=[("feature_id", "==", cluster_id)],
        )
        for sid, n, tpm in zip(
            t.column("sample_id").to_pylist(),
            t.column("count").to_pylist(),
            t.column("tpm").to_pylist(),
        ):
            counts.append(
                {"sample_id": sid, "sample": names.get(str(sid)), "count": n, "tpm": tpm}
            )
    out["samples"] = sorted(counts, key=lambda c: (c["sample_id"] is None, c["sample_id"]))
    return _plain(out)


@router.get("/runs/{key}/clusters/{cluster_id}")
async def cluster_detail(request: Request, session_id: str, key: str, cluster_id: int):
    run = _run(request, session_id, key)
    _finished(run)
    index = await cluster_index_for(run.path)
    return await asyncio.to_thread(_detail, index, run, cluster_id)


# ── the support view ──────────────────────────────────────────────────


def _replay(run: TxRun, target: tuple):
    from constellation.sequencing.transcriptome.cluster.denovo.em.results import (
        replay_cluster,
        replay_template,
    )

    if target[0] == "cluster":
        return replay_cluster(run.path, target[1])
    return replay_template(run.path, target[1], target[2])


def _node(value: str) -> str | int:
    if value in ("cluster", "pooled"):
        return value
    try:
        return int(value)
    except ValueError:
        raise HTTPException(422, "node must be 'cluster', 'pooled' or a haplotype id") from None


def _make_support(rep, node: str | int, space: str) -> dict[str, Any]:
    from constellation.sequencing.transcriptome.cluster.denovo.em.results import (
        node_differences,
        replay_summary,
        support_tables,
    )

    is_cluster = hasattr(rep, "cluster_id")
    if node == "pooled":
        # A cluster's template, every read against the pooled consensus —
        # under the cluster's verdict. A rebuilt survivor has no template;
        # its pool is its own.
        template = getattr(rep, "template", None)
        st = support_tables(template if template is not None else rep, node=None, space=space)
        if is_cluster:
            st.verified, st.mismatch = bool(rep.verified), list(rep.mismatch)
            st.warnings = list(dict.fromkeys(list(rep.warnings) + list(st.warnings)))
    elif node == "cluster":
        st = support_tables(rep, space=space)
    else:
        st = support_tables(rep, node=int(node), space=space)
    tables = {
        "reads": st.reads,
        "departures": st.departures,
        "insertions": st.insertions,
        "columns": st.columns,
        "arcs": st.arcs,
        "differences": node_differences(rep),
    }
    meta = {
        **replay_summary(rep),
        "reference": st.reference,
        "space": st.space,
        "consensus": st.consensus,
        "n_columns": int(st.n_columns),
        "verified": bool(st.verified),
        "mismatch": list(st.mismatch),
        "warnings": list(st.warnings),
        "rows": {name: int(t.num_rows) for name, t in tables.items()},
    }
    return {"meta": _plain(meta), "tables": tables}


async def _support(run: TxRun, target: tuple, node: str, space: str) -> dict[str, Any]:
    from constellation.sequencing.transcriptome.cluster.denovo.em.results import ReplayError

    rkey = (str(run.path), *target)
    want = _node(node)

    def compute() -> dict[str, Any]:
        rep = REPLAYS.get_here(rkey, lambda: _replay(run, target))
        return _make_support(rep, want, space)

    try:
        return await SUPPORT.aget((*rkey, want, space), compute)
    except ReplayError as exc:
        raise HTTPException(404, str(exc)) from None
    except KeyError as exc:
        raise HTTPException(404, f"not in this run: {exc}") from None


Space = Literal["consensus", "frame"]


@router.get("/runs/{key}/clusters/{cluster_id}/support")
async def cluster_support(
    request: Request,
    session_id: str,
    key: str,
    cluster_id: int,
    node: str = "cluster",
    space: Space = "consensus",
):
    run = _run(request, session_id, key)
    _finished(run)
    return (await _support(run, ("cluster", cluster_id), node, space))["meta"]


@router.get("/runs/{key}/clusters/{cluster_id}/support/{table}")
async def cluster_support_table(
    request: Request,
    session_id: str,
    key: str,
    cluster_id: int,
    table: str,
    node: str = "cluster",
    space: Space = "consensus",
):
    if table not in SUPPORT_TABLES:
        raise HTTPException(404, f"no support table {table!r}; one of {SUPPORT_TABLES}")
    run = _run(request, session_id, key)
    _finished(run)
    return _arrow((await _support(run, ("cluster", cluster_id), node, space))["tables"][table])


@router.get("/runs/{key}/rounds/{round_index}/templates/{template_id}/support")
async def template_support(
    request: Request,
    session_id: str,
    key: str,
    round_index: int,
    template_id: int,
    node: str = "pooled",
    space: Space = "consensus",
):
    run = _run(request, session_id, key)
    return (await _support(run, ("template", round_index, template_id), node, space))["meta"]


@router.get("/runs/{key}/rounds/{round_index}/templates/{template_id}/support/{table}")
async def template_support_table(
    request: Request,
    session_id: str,
    key: str,
    round_index: int,
    template_id: int,
    table: str,
    node: str = "pooled",
    space: Space = "consensus",
):
    if table not in SUPPORT_TABLES:
        raise HTTPException(404, f"no support table {table!r}; one of {SUPPORT_TABLES}")
    run = _run(request, session_id, key)
    got = await _support(run, ("template", round_index, template_id), node, space)
    return _arrow(got["tables"][table])


__all__ = ["CLUSTER_SORTS", "SUPPORT_TABLES", "router"]

