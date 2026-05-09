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
from constellation.sequencing.schemas.reference import CONTIG_TABLE  # noqa: E402
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
    the build helper instead of a 404."""
    empty = tmp_path / "no-static"
    empty.mkdir()
    app = create_app(fixture_session, static_root=empty)
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "frontend bundle not built" in body["message"]
