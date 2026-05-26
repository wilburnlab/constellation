"""FastAPI server endpoints — Arrow IPC streaming + session/track routes.

These tests are gated on `fastapi` being importable (it ships in the
``[viz]`` extras only). The smoke ``test_imports.py`` deliberately
skips these modules; this file is the gated home for them.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from constellation.sequencing.schemas.quant import COVERAGE_TABLE  # noqa: E402
from constellation.sequencing.schemas.reference import (  # noqa: E402
    CONTIG_TABLE,
    FEATURE_TABLE,
)
from constellation.viz.server.app import create_app  # noqa: E402
from constellation.viz.server.arrow_stream import (  # noqa: E402
    ARROW_IPC_MEDIA_TYPE,
    collect_to_table,
    encode_ipc_stream,
)
from constellation.viz.server.session import Session  # noqa: E402
from constellation.viz.tracks.coverage_histogram import (  # noqa: E402
    COVERAGE_VECTOR_SCHEMA,
)
from constellation.viz.tracks.cluster_pileup import (  # noqa: E402
    CLUSTER_MEMBER_VECTOR_SCHEMA,
    CLUSTER_PILEUP_VECTOR_SCHEMA,
)
from constellation.viz.tracks.read_pileup import (  # noqa: E402
    READ_PILEUP_VECTOR_SCHEMA,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def fixture_session(tmp_path: Path, monkeypatch) -> Session:
    from _viz_fixtures import build_viz_session

    return build_viz_session(
        tmp_path,
        monkeypatch,
        contigs=[
            {
                "contig_id": 1,
                "name": "chr1",
                "length": 1_000_000,
                "topology": None,
                "circular": None,
            },
            {
                "contig_id": 2,
                "name": "chr2",
                "length": 200_000,
                "topology": None,
                "circular": None,
            },
        ],
        sequences=[
            {"contig_id": 1, "sequence": "N" * 100},
            {"contig_id": 2, "sequence": "N" * 100},
        ],
        align_sources=[
            {
                "coverage": [
                    {"contig_id": 1, "sample_id": -1, "start": 0, "end": 1000, "depth": 5},
                    {"contig_id": 1, "sample_id": -1, "start": 1000, "end": 2000, "depth": 12},
                    {"contig_id": 1, "sample_id": -1, "start": 5000, "end": 6000, "depth": 3},
                ],
            }
        ],
    )


@pytest.fixture
def client(fixture_session: Session) -> TestClient:
    app = create_app(fixture_session)
    return TestClient(app)


# ----------------------------------------------------------------------
# encode_ipc_stream / collect_to_table
# ----------------------------------------------------------------------


def test_encode_ipc_stream_round_trip() -> None:
    schema = pa.schema([pa.field("a", pa.int32()), pa.field("b", pa.float64())])
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], pa.int32()), pa.array([1.5, 2.5, 3.5], pa.float64())],
        schema=schema,
    )
    payload = b"".join(encode_ipc_stream(schema, [batch]))
    reader = pa.ipc.RecordBatchStreamReader(io.BytesIO(payload))
    table = reader.read_all()
    assert table.schema == schema
    assert table.column("a").to_pylist() == [1, 2, 3]


def test_encode_ipc_stream_empty_iter_yields_schema_only() -> None:
    schema = pa.schema([pa.field("x", pa.int64())])
    payload = b"".join(encode_ipc_stream(schema, []))
    reader = pa.ipc.RecordBatchStreamReader(io.BytesIO(payload))
    table = reader.read_all()
    assert table.schema == schema
    assert table.num_rows == 0


def test_collect_to_table_handles_empty() -> None:
    schema = pa.schema([pa.field("x", pa.int64())])
    table = collect_to_table(schema, iter(()))
    assert table.schema == schema
    assert table.num_rows == 0


# ----------------------------------------------------------------------
# Endpoint routes
# ----------------------------------------------------------------------


def test_health_endpoint(client: TestClient, fixture_session: Session) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert fixture_session.session_id in body["sessions"]


def test_list_sessions_returns_summary(
    client: TestClient, fixture_session: Session
) -> None:
    response = client.get("/api/sessions")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    summary = body[0]
    assert summary["session_id"] == fixture_session.session_id
    assert summary["stages_present"]["coverage"] is True


def test_session_manifest_round_trips(
    client: TestClient, fixture_session: Session
) -> None:
    response = client.get(
        f"/api/sessions/{fixture_session.session_id}/manifest"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == fixture_session.session_id
    assert body["reference"]["handle"] == fixture_session.reference_handle
    assert body["sources"][0]["slots"]["coverage"] is not None


def test_session_contigs(
    client: TestClient, fixture_session: Session
) -> None:
    response = client.get(
        f"/api/sessions/{fixture_session.session_id}/contigs"
    )
    assert response.status_code == 200
    body = response.json()
    names = sorted(c["name"] for c in body)
    assert names == ["chr1", "chr2"]


def test_unknown_session_returns_404(client: TestClient) -> None:
    response = client.get("/api/sessions/does-not-exist/manifest")
    assert response.status_code == 404


def _open_payload(fixture_session: Session) -> dict:
    sources = [
        {"path": str(src.path), "kind": src.kind, "label": src.label}
        for src in fixture_session.sources
    ]
    return {
        "reference_handle": fixture_session.reference_handle,
        "sources": sources,
    }


def test_open_session_adds_session(
    fixture_session: Session, tmp_path: Path
) -> None:
    app = create_app({})
    client = TestClient(app)
    # The fixture's monkeypatched CONSTELLATION_REFERENCES_HOME stays
    # active across this nested app since it's a process-wide env var.
    assert client.get("/api/sessions").json() == []

    response = client.post("/api/sessions/open", json=_open_payload(fixture_session))
    assert response.status_code == 201
    body = response.json()
    assert body["reference_handle"] == fixture_session.reference_handle
    assert body["stages_present"]["reference_genome"] is True
    assert body["n_sources"] == len(fixture_session.sources)

    listed = client.get("/api/sessions").json()
    assert len(listed) == 1
    assert listed[0]["session_id"] == body["session_id"]


def test_open_session_is_idempotent_and_evicts_cache(
    fixture_session: Session,
) -> None:
    app = create_app({})
    client = TestClient(app)
    payload = _open_payload(fixture_session)

    first = client.post("/api/sessions/open", json=payload)
    assert first.status_code == 201
    sid = first.json()["session_id"]

    app.state.track_bindings_cache[(sid, "coverage_histogram")] = ["sentinel"]

    second = client.post("/api/sessions/open", json=payload)
    assert second.status_code == 201
    assert second.json()["session_id"] == sid
    assert len(client.get("/api/sessions").json()) == 1
    assert (sid, "coverage_histogram") not in app.state.track_bindings_cache


def test_open_session_unknown_handle_returns_400(tmp_path: Path) -> None:
    app = create_app({})
    client = TestClient(app)
    response = client.post(
        "/api/sessions/open",
        json={"reference_handle": "no_such@local_import-12345678", "sources": []},
    )
    assert response.status_code == 400


def test_inspect_source_reports_kind_and_handle(
    fixture_session: Session,
) -> None:
    app = create_app({})
    client = TestClient(app)
    src = fixture_session.sources[0]
    response = client.post(
        "/api/sessions/inspect-source", json={"path": str(src.path)}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "align"
    assert body["reference_handle"] == fixture_session.reference_handle


def test_inspect_source_missing_dir_returns_400(tmp_path: Path) -> None:
    app = create_app({})
    client = TestClient(app)
    response = client.post(
        "/api/sessions/inspect-source",
        json={"path": str(tmp_path / "does-not-exist")},
    )
    assert response.status_code == 400


def test_list_tracks_includes_coverage_histogram(
    client: TestClient, fixture_session: Session
) -> None:
    response = client.get(
        "/api/tracks", params={"session": fixture_session.session_id}
    )
    assert response.status_code == 200
    body = response.json()
    kinds = {row["kind"] for row in body}
    assert "coverage_histogram" in kinds


def test_track_metadata_round_trips(
    client: TestClient, fixture_session: Session
) -> None:
    response = client.get(
        "/api/tracks/coverage_histogram/metadata",
        params={"session": fixture_session.session_id, "binding": "coverage-0"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "coverage_histogram"
    assert body["depth_max"] == 12.0


def test_track_data_streams_arrow_ipc(
    client: TestClient, fixture_session: Session
) -> None:
    response = client.get(
        "/api/tracks/coverage_histogram/data",
        params={
            "session": fixture_session.session_id,
            "binding": "coverage-0",
            "contig": "chr1",
            "start": 0,
            "end": 3000,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(ARROW_IPC_MEDIA_TYPE)
    assert response.headers["x-track-mode"] == "vector"
    assert response.headers["x-track-kind"] == "coverage_histogram"

    table = pa.ipc.RecordBatchStreamReader(io.BytesIO(response.content)).read_all()
    assert table.schema == COVERAGE_VECTOR_SCHEMA
    # Two coverage rows overlap [0, 3000)
    assert table.num_rows == 2
    depths = sorted(table.column("depth").to_pylist())
    assert depths == [5.0, 12.0]


def test_track_data_unknown_contig_returns_empty_stream(
    client: TestClient, fixture_session: Session
) -> None:
    response = client.get(
        "/api/tracks/coverage_histogram/data",
        params={
            "session": fixture_session.session_id,
            "binding": "coverage-0",
            "contig": "chrX_missing",
            "start": 0,
            "end": 1000,
        },
    )
    assert response.status_code == 200
    table = pa.ipc.RecordBatchStreamReader(io.BytesIO(response.content)).read_all()
    assert table.num_rows == 0


def test_track_data_rejects_inverted_window(
    client: TestClient, fixture_session: Session
) -> None:
    response = client.get(
        "/api/tracks/coverage_histogram/data",
        params={
            "session": fixture_session.session_id,
            "binding": "coverage-0",
            "contig": "chr1",
            "start": 1000,
            "end": 500,
        },
    )
    assert response.status_code == 400


def test_track_data_unknown_kind_returns_404(
    client: TestClient, fixture_session: Session
) -> None:
    response = client.get(
        "/api/tracks/no_such_kind/data",
        params={
            "session": fixture_session.session_id,
            "binding": "coverage-0",
            "contig": "chr1",
            "start": 0,
            "end": 10,
        },
    )
    assert response.status_code == 404


@pytest.fixture
def read_pileup_client(tmp_path: Path, monkeypatch) -> TestClient:
    """A second client with a read_pileup-shaped source — alignments
    plus blocks plus cs plus read_samples. Used by the MAPQ-pushdown
    test below."""
    from _viz_fixtures import build_viz_session

    session = build_viz_session(
        tmp_path,
        monkeypatch,
        contigs=[
            {
                "contig_id": 1,
                "name": "chr1",
                "length": 1_000_000,
                "topology": None,
                "circular": None,
            }
        ],
        sequences=[{"contig_id": 1, "sequence": "N" * 100}],
        align_sources=[
            {
                "alignments": [
                    _alignment_row(1, "r1", 0, 100, mapq=60),
                    _alignment_row(2, "r2", 50, 150, mapq=10),
                    _alignment_row(3, "r3", 200, 300, mapq=30),
                ],
                "alignment_blocks": [
                    _block_row(1, 0, 0, 100),
                    _block_row(2, 0, 50, 150),
                    _block_row(3, 0, 200, 300),
                ],
                "alignment_cs": [
                    {"alignment_id": 1, "cs_string": ""},
                    {"alignment_id": 2, "cs_string": ""},
                    {"alignment_id": 3, "cs_string": ""},
                ],
                "read_samples": [
                    {"read_id": "r1", "sample_id": 1, "sample_name": "a"},
                    {"read_id": "r2", "sample_id": 1, "sample_name": "a"},
                    {"read_id": "r3", "sample_id": 1, "sample_name": "a"},
                ],
            }
        ],
    )
    app = create_app(session)
    client = TestClient(app)
    client.session_id = session.session_id  # type: ignore[attr-defined]
    return client


def _alignment_row(
    alignment_id: int, read_id: str, ref_start: int, ref_end: int, *, mapq: int = 60
) -> dict:
    return {
        "alignment_id": alignment_id,
        "read_id": read_id,
        "acquisition_id": 1,
        "ref_name": "chr1",
        "ref_start": ref_start,
        "ref_end": ref_end,
        "strand": "+",
        "mapq": mapq,
        "flag": 0,
        "cigar_string": f"{ref_end - ref_start}M",
        "nm_tag": None,
        "as_tag": None,
        "read_group": None,
        "is_secondary": False,
        "is_supplementary": False,
    }


def _block_row(
    alignment_id: int, block_index: int, ref_start: int, ref_end: int
) -> dict:
    return {
        "alignment_id": alignment_id,
        "block_index": block_index,
        "ref_start": ref_start,
        "ref_end": ref_end,
        "query_start": 0,
        "query_end": ref_end - ref_start,
        "n_match": None,
        "n_mismatch": None,
        "n_insert": 0,
        "n_delete": 0,
    }


def test_track_data_min_mapq_pushdown_via_endpoint(
    read_pileup_client: TestClient,
) -> None:
    """GET /api/tracks/read_pileup/data?...&min_mapq=20 round-trips
    the new kernel-pushdown filter — the response row count reflects
    the MAPQ predicate applied at scan time."""
    session_id = read_pileup_client.session_id  # type: ignore[attr-defined]

    # Baseline: no MAPQ filter — all three alignments survive.
    baseline = read_pileup_client.get(
        "/api/tracks/read_pileup/data",
        params={
            "session": session_id,
            "binding": "read_pileup-0",
            "contig": "chr1",
            "start": 0,
            "end": 400,
        },
    )
    assert baseline.status_code == 200
    baseline_table = pa.ipc.RecordBatchStreamReader(
        io.BytesIO(baseline.content)
    ).read_all()
    assert baseline_table.schema == READ_PILEUP_VECTOR_SCHEMA
    assert baseline_table.num_rows == 3

    # min_mapq=20 drops the mapq=10 alignment.
    filtered = read_pileup_client.get(
        "/api/tracks/read_pileup/data",
        params={
            "session": session_id,
            "binding": "read_pileup-0",
            "contig": "chr1",
            "start": 0,
            "end": 400,
            "min_mapq": 20,
        },
    )
    assert filtered.status_code == 200
    filtered_table = pa.ipc.RecordBatchStreamReader(
        io.BytesIO(filtered.content)
    ).read_all()
    kept_ids = sorted(filtered_table.column("alignment_id").to_pylist())
    assert kept_ids == [1, 3]


def test_track_data_min_mapq_rejects_negative(
    read_pileup_client: TestClient,
) -> None:
    """FastAPI's ``ge=0`` validation refuses negative MAPQ values
    before reaching the kernel."""
    session_id = read_pileup_client.session_id  # type: ignore[attr-defined]
    response = read_pileup_client.get(
        "/api/tracks/read_pileup/data",
        params={
            "session": session_id,
            "binding": "read_pileup-0",
            "contig": "chr1",
            "start": 0,
            "end": 400,
            "min_mapq": -1,
        },
    )
    assert response.status_code == 422


@pytest.fixture
def cluster_pileup_client(tmp_path: Path, monkeypatch) -> TestClient:
    """A client over a session with both an align source and a cluster
    source — the cluster manifest's align_dir back-pointer makes the
    members view available."""
    from _viz_fixtures import (
        build_viz_session,
        write_align_source,
        DEFAULT_ASSEMBLY,
        DEFAULT_HANDLE,
        install_fake_reference,
    )
    from constellation.sequencing.schemas.transcriptome import (
        CLUSTER_MEMBERSHIP_TABLE,
        TRANSCRIPT_CLUSTER_TABLE,
    )

    # Install reference at a pinned cache root via monkeypatch + a
    # bare write_align_source helper so we can wire align_dir on the
    # cluster manifest at the right path.
    cache_root = tmp_path / "refs"
    cache_root.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache_root))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    install_fake_reference(
        cache_root,
        contigs=[
            {
                "contig_id": 1,
                "name": "chr1",
                "length": 1_000_000,
                "topology": None,
                "circular": None,
            }
        ],
        sequences=[{"contig_id": 1, "sequence": "N" * 100}],
    )

    align_dir = tmp_path / "align_src"
    write_align_source(
        align_dir,
        alignments=[
            _alignment_row(1, "rA", 0, 100, mapq=60),
            _alignment_row(2, "rB", 50, 150, mapq=60),
        ],
        alignment_blocks=[
            _block_row(1, 0, 0, 100),
            _block_row(2, 0, 50, 150),
        ],
        alignment_cs=[
            {"alignment_id": 1, "cs_string": ""},
            {"alignment_id": 2, "cs_string": ""},
        ],
        read_samples=[
            {"read_id": "rA", "sample_id": 1, "sample_name": "a"},
            {"read_id": "rB", "sample_id": 1, "sample_name": "a"},
        ],
    )

    cluster_dir = tmp_path / "cluster_src"
    cluster_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cluster_id": 1,
                    "representative_read_id": "rA",
                    "n_reads": 2,
                    "identity_threshold": None,
                    "consensus_sequence": None,
                    "predicted_protein": None,
                    "orf_start": None,
                    "orf_end": None,
                    "orf_strand": None,
                    "codon_table": None,
                    "mode": "genome-guided",
                    "contig_id": 1,
                    "strand": "+",
                    "span_start": 0,
                    "span_end": 150,
                    "fingerprint_hash": 0,
                    "n_unique_sequences": 1,
                    "sample_id": -1,
                }
            ],
            schema=TRANSCRIPT_CLUSTER_TABLE,
        ),
        cluster_dir / "clusters.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cluster_id": 1,
                    "read_id": "rA",
                    "role": "representative",
                    "drift_5p_bp": None,
                    "drift_3p_bp": None,
                    "match_rate": None,
                    "indel_rate": None,
                    "n_aligned_bp": 100,
                },
                {
                    "cluster_id": 1,
                    "read_id": "rB",
                    "role": "member",
                    "drift_5p_bp": None,
                    "drift_3p_bp": None,
                    "match_rate": None,
                    "indel_rate": None,
                    "n_aligned_bp": 100,
                },
            ],
            schema=CLUSTER_MEMBERSHIP_TABLE,
        ),
        cluster_dir / "cluster_membership.parquet",
    )
    from constellation.sequencing.transcriptome.manifest import (
        write_cluster_manifest,
    )

    write_cluster_manifest(
        cluster_dir / "manifest.json",
        reference_handle=DEFAULT_HANDLE,
        reference_path=str(cache_root / "test_org" / "local_import-20260522"),
        assembly_accession=DEFAULT_ASSEMBLY,
        align_dir=str(align_dir),
        demux_dir="",
        parameters={},
        stages={},
        outputs={
            "clusters": str(cluster_dir / "clusters.parquet"),
            "cluster_membership": str(cluster_dir / "cluster_membership.parquet"),
        },
    )

    session = build_viz_session(
        tmp_path,
        monkeypatch,
        contigs=[
            {
                "contig_id": 1,
                "name": "chr1",
                "length": 1_000_000,
                "topology": None,
                "circular": None,
            }
        ],
        sequences=[{"contig_id": 1, "sequence": "N" * 100}],
    )
    # Re-open with the explicit (align + cluster) source pair.
    from constellation.viz.server.session import Session

    session = Session.open(
        reference_handle=DEFAULT_HANDLE,
        sources=[
            {"path": str(align_dir), "kind": "align"},
            {"path": str(cluster_dir), "kind": "cluster"},
        ],
    )
    app = create_app(session)
    client = TestClient(app)
    client.session_id = session.session_id  # type: ignore[attr-defined]
    return client


def test_track_data_cluster_view_members_via_endpoint(
    cluster_pileup_client: TestClient,
) -> None:
    """GET /api/tracks/cluster_pileup/data?...&cluster_view=members
    streams the expanded member-read schema rather than the cluster
    rectangles. Default (no cluster_view param) keeps the existing
    rectangle behavior."""
    session_id = cluster_pileup_client.session_id  # type: ignore[attr-defined]

    # Default view → CLUSTER_PILEUP_VECTOR_SCHEMA.
    default = cluster_pileup_client.get(
        "/api/tracks/cluster_pileup/data",
        params={
            "session": session_id,
            "binding": "cluster_pileup-1",
            "contig": "chr1",
            "start": 0,
            "end": 200,
        },
    )
    assert default.status_code == 200
    assert default.headers.get("x-track-view") == ""
    default_table = pa.ipc.RecordBatchStreamReader(
        io.BytesIO(default.content)
    ).read_all()
    assert default_table.schema == CLUSTER_PILEUP_VECTOR_SCHEMA

    # Members view → CLUSTER_MEMBER_VECTOR_SCHEMA + per-alignment rows.
    members = cluster_pileup_client.get(
        "/api/tracks/cluster_pileup/data",
        params={
            "session": session_id,
            "binding": "cluster_pileup-1",
            "contig": "chr1",
            "start": 0,
            "end": 200,
            "cluster_view": "members",
        },
    )
    assert members.status_code == 200
    assert members.headers.get("x-track-view") == "members"
    members_table = pa.ipc.RecordBatchStreamReader(
        io.BytesIO(members.content)
    ).read_all()
    assert members_table.schema == CLUSTER_MEMBER_VECTOR_SCHEMA
    # Both reads survive the join.
    assert members_table.num_rows == 2


def test_track_data_cluster_view_rejects_unknown_value(
    cluster_pileup_client: TestClient,
) -> None:
    """The pattern guard refuses cluster_view values outside the
    {clusters, members} set so typos surface as a 422 rather than
    silently rendering an empty stream."""
    session_id = cluster_pileup_client.session_id  # type: ignore[attr-defined]
    response = cluster_pileup_client.get(
        "/api/tracks/cluster_pileup/data",
        params={
            "session": session_id,
            "binding": "cluster_pileup-1",
            "contig": "chr1",
            "start": 0,
            "end": 200,
            "cluster_view": "no_such_view",
        },
    )
    assert response.status_code == 422


def test_root_with_no_static_bundle_returns_pointer(
    fixture_session: Session, tmp_path: Path
) -> None:
    """When the static bundle is absent, `/` returns a JSON pointer to
    the install helper instead of a 404."""
    empty = tmp_path / "no-static"
    empty.mkdir()
    app = create_app(fixture_session, static_root=empty)
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "frontend bundle not installed" in body["message"]
    assert "install-frontend" in body["message"]


def test_root_with_entry_named_index_html_serves_spa(
    fixture_session: Session, tmp_path: Path
) -> None:
    """Bundles whose HTML root is the vanilla `index.html` (or output
    of a Vite rebuild with output rename) → `/` redirects to it."""
    static_root = tmp_path / "vanilla-bundle"
    (static_root / "genome").mkdir(parents=True)
    (static_root / "genome" / "index.html").write_text(
        "<html><body>SPA</body></html>", encoding="utf-8"
    )
    app = create_app(fixture_session, static_root=static_root)
    client = TestClient(app, follow_redirects=False)
    response = client.get("/")
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/static/genome/index.html"

    followed = client.get(response.headers["location"])
    assert followed.status_code == 200
    assert "SPA" in followed.text


def test_root_with_vite_named_index_serves_spa(
    fixture_session: Session, tmp_path: Path
) -> None:
    """REGRESSION: Bundles produced by `vite build` with input
    `index.genome.html` preserve that filename in the output. The
    `/` redirect must point at the entry-specific filename so
    StaticFiles serves it (StaticFiles' `html=True` would only match
    a literal `index.html`)."""
    static_root = tmp_path / "vite-bundle"
    (static_root / "genome").mkdir(parents=True)
    (static_root / "genome" / "index.genome.html").write_text(
        "<!doctype html><html><body><div id='app'></div></body></html>",
        encoding="utf-8",
    )
    (static_root / "genome" / "assets").mkdir()
    (static_root / "genome" / "assets" / "main_genome-abc.js").write_text(
        "console.log('hi')", encoding="utf-8"
    )

    app = create_app(fixture_session, static_root=static_root)
    client = TestClient(app, follow_redirects=False)
    response = client.get("/")
    assert response.status_code in (302, 307)
    # Critical: redirect points at the file Vite actually emitted, not
    # at the directory root (which would 404 with the StaticFiles
    # html=True default).
    assert response.headers["location"] == "/static/genome/index.genome.html"

    followed = client.get(response.headers["location"])
    assert followed.status_code == 200
    assert "<div id='app'>" in followed.text

    # Assets next to the HTML resolve via the StaticFiles mount.
    asset = client.get("/static/genome/assets/main_genome-abc.js")
    assert asset.status_code == 200
    assert asset.text == "console.log('hi')"


# ----------------------------------------------------------------------
# /api/sessions/{id}/search
# ----------------------------------------------------------------------


@pytest.fixture
def search_session(tmp_path: Path, monkeypatch) -> Session:
    """Session with reference + derived annotation parquets for the
    search-endpoint tests. `BRCA1` lives in the curated reference set;
    `derived_gene_42` only in the data-derived bundle; both contain a
    common name fragment ('brc') so substring matches return hits from
    both sources."""
    from _viz_fixtures import build_viz_session

    def _feature(
        *,
        feature_id: int,
        contig_id: int,
        start: int,
        end: int,
        name: str | None,
        type_: str = "gene",
        source: str | None = None,
    ) -> dict:
        return {
            "feature_id": feature_id,
            "contig_id": contig_id,
            "start": start,
            "end": end,
            "strand": "+",
            "type": type_,
            "name": name,
            "parent_id": None,
            "source": source,
            "score": None,
            "phase": None,
            "attributes_json": None,
        }

    return build_viz_session(
        tmp_path,
        monkeypatch,
        contigs=[
            {
                "contig_id": 1,
                "name": "chr1",
                "length": 1_000_000,
                "topology": None,
                "circular": None,
            },
            {
                "contig_id": 2,
                "name": "chrX",
                "length": 250_000,
                "topology": None,
                "circular": None,
            },
        ],
        sequences=[
            {"contig_id": 1, "sequence": "N" * 50},
            {"contig_id": 2, "sequence": "N" * 50},
        ],
        features=[
            _feature(feature_id=1, contig_id=1, start=100, end=400, name="BRCA1"),
            _feature(feature_id=2, contig_id=2, start=500, end=900, name="MYC"),
            _feature(
                feature_id=3,
                contig_id=1,
                start=1100,
                end=1500,
                name=None,
                type_="repeat_region",
            ),
        ],
        align_sources=[
            {
                "derived_annotation_features": [
                    _feature(
                        feature_id=101,
                        contig_id=1,
                        start=2000,
                        end=2400,
                        name="derived_brc_42",
                        source="constellation_derived",
                    ),
                    _feature(
                        feature_id=102,
                        contig_id=2,
                        start=10_000,
                        end=11_000,
                        name="other_derived",
                        source="constellation_derived",
                    ),
                ],
            }
        ],
    )


@pytest.fixture
def search_client(search_session: Session) -> TestClient:
    app = create_app(search_session)
    return TestClient(app)


def test_search_empty_query_returns_empty(
    search_client: TestClient, search_session: Session
) -> None:
    response = search_client.get(
        f"/api/sessions/{search_session.session_id}/search", params={"q": "   "}
    )
    assert response.status_code == 200
    assert response.json() == []


def test_search_substring_matches_across_bindings_reference_first(
    search_client: TestClient, search_session: Session
) -> None:
    response = search_client.get(
        f"/api/sessions/{search_session.session_id}/search", params={"q": "brc"}
    )
    assert response.status_code == 200
    hits = response.json()
    # BRCA1 (reference) + derived_brc_42 (derived); reference must lead.
    names = [h["name"] for h in hits]
    sources = [h["source"] for h in hits]
    assert names == ["BRCA1", "derived_brc_42"]
    assert sources[0] == "reference"
    assert sources[1].startswith("derived")
    assert hits[0]["contig_name"] == "chr1"
    assert hits[0]["start"] == 100 and hits[0]["end"] == 400


def test_search_is_case_insensitive(
    search_client: TestClient, search_session: Session
) -> None:
    response = search_client.get(
        f"/api/sessions/{search_session.session_id}/search", params={"q": "brca1"}
    )
    assert response.status_code == 200
    names = [h["name"] for h in response.json()]
    assert names == ["BRCA1"]


def test_search_numeric_query_matches_feature_id(
    search_client: TestClient, search_session: Session
) -> None:
    response = search_client.get(
        f"/api/sessions/{search_session.session_id}/search", params={"q": "102"}
    )
    assert response.status_code == 200
    hits = response.json()
    # feature_id=102 lives in derived; name does not contain "102".
    assert len(hits) == 1
    assert hits[0]["feature_id"] == 102
    assert hits[0]["source"].startswith("derived")


def test_search_unknown_session_returns_404(search_client: TestClient) -> None:
    response = search_client.get(
        "/api/sessions/does-not-exist/search", params={"q": "brca"}
    )
    assert response.status_code == 404


def test_search_respects_limit(
    search_client: TestClient, search_session: Session
) -> None:
    response = search_client.get(
        f"/api/sessions/{search_session.session_id}/search",
        params={"q": "br", "limit": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    # 'br' matches BRCA1 (reference) and derived_brc_42 (derived); with
    # limit=1 the reference binding must win the ordering tiebreak.
    assert body[0]["source"] == "reference"
    assert body[0]["name"] == "BRCA1"


def test_index_genome_html_preferred_over_index_html_if_both_present(
    fixture_session: Session, tmp_path: Path
) -> None:
    """If a bundle ships BOTH names (unlikely but possible), the
    entry-specific one wins so the redirect URL is predictable."""
    static_root = tmp_path / "both"
    (static_root / "genome").mkdir(parents=True)
    (static_root / "genome" / "index.html").write_text(
        "<html>vanilla</html>", encoding="utf-8"
    )
    (static_root / "genome" / "index.genome.html").write_text(
        "<html>entry-specific</html>", encoding="utf-8"
    )
    app = create_app(fixture_session, static_root=static_root)
    client = TestClient(app, follow_redirects=False)
    response = client.get("/")
    assert response.headers["location"] == "/static/genome/index.genome.html"
