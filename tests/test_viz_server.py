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


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def fixture_session(tmp_path: Path) -> Session:
    root = tmp_path / "run"
    genome = root / "genome"
    genome.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
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
            schema=CONTIG_TABLE,
        ),
        genome / "contigs.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "contig_id": pa.array([1, 2], pa.int64()),
                "sequence": pa.array(["N" * 100, "N" * 100], pa.string()),
            }
        ),
        genome / "sequences.parquet",
    )

    align = root / "S2_align"
    align.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "contig_id": 1,
                    "sample_id": -1,
                    "start": 0,
                    "end": 1000,
                    "depth": 5,
                },
                {
                    "contig_id": 1,
                    "sample_id": -1,
                    "start": 1000,
                    "end": 2000,
                    "depth": 12,
                },
                {
                    "contig_id": 1,
                    "sample_id": -1,
                    "start": 5000,
                    "end": 6000,
                    "depth": 3,
                },
            ],
            schema=COVERAGE_TABLE,
        ),
        align / "coverage.parquet",
    )
    return Session.from_root(root)


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
    assert body["paths"]["coverage"] == "S2_align/coverage.parquet"


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


def test_register_session_adds_session(
    fixture_session: Session, tmp_path: Path
) -> None:
    app = create_app({})
    client = TestClient(app)

    assert client.get("/api/sessions").json() == []

    response = client.post("/api/sessions", json={"path": str(fixture_session.root)})
    assert response.status_code == 201
    body = response.json()
    assert body["session_id"] == fixture_session.session_id
    assert body["root"] == str(fixture_session.root)
    assert body["stages_present"]["reference_genome"] is True

    listed = client.get("/api/sessions").json()
    assert len(listed) == 1
    assert listed[0]["session_id"] == fixture_session.session_id


def test_register_session_is_idempotent_and_evicts_cache(
    fixture_session: Session,
) -> None:
    app = create_app({})
    client = TestClient(app)
    payload = {"path": str(fixture_session.root)}

    first = client.post("/api/sessions", json=payload)
    assert first.status_code == 201
    sid = first.json()["session_id"]

    app.state.track_bindings_cache[(sid, "coverage_histogram")] = ["sentinel"]

    second = client.post("/api/sessions", json=payload)
    assert second.status_code == 201
    assert second.json()["session_id"] == sid
    assert len(client.get("/api/sessions").json()) == 1
    assert (sid, "coverage_histogram") not in app.state.track_bindings_cache


def test_register_session_missing_path_returns_404(tmp_path: Path) -> None:
    app = create_app({})
    client = TestClient(app)
    response = client.post(
        "/api/sessions", json={"path": str(tmp_path / "does-not-exist")}
    )
    assert response.status_code == 404


def test_register_session_empty_path_returns_400() -> None:
    app = create_app({})
    client = TestClient(app)
    response = client.post("/api/sessions", json={"path": "   "})
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
        params={"session": fixture_session.session_id, "binding": "coverage"},
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
            "binding": "coverage",
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
            "binding": "coverage",
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
            "binding": "coverage",
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
            "binding": "coverage",
            "contig": "chr1",
            "start": 0,
            "end": 10,
        },
    )
    assert response.status_code == 404


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
def search_session(tmp_path: Path) -> Session:
    """Session with reference + derived annotation parquets for the
    search-endpoint tests. `BRCA1` lives in the curated reference set;
    `derived_gene_42` only in the data-derived bundle; both contain a
    common name fragment ('brc') so substring matches return hits from
    both sources."""
    root = tmp_path / "search_run"
    genome = root / "genome"
    genome.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
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
            schema=CONTIG_TABLE,
        ),
        genome / "contigs.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "contig_id": pa.array([1, 2], pa.int64()),
                "sequence": pa.array(["N" * 50, "N" * 50], pa.string()),
            }
        ),
        genome / "sequences.parquet",
    )

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

    annotation = root / "annotation"
    annotation.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
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
            schema=FEATURE_TABLE,
        ),
        annotation / "features.parquet",
    )

    align = root / "S2_align"
    derived = align / "derived_annotation"
    derived.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
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
            schema=FEATURE_TABLE,
        ),
        derived / "features.parquet",
    )
    return Session.from_root(root)


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
    assert sources == ["reference", "derived"]
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
    assert hits[0]["source"] == "derived"


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
