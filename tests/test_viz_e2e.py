"""End-to-end smoke test for the viz server — boots a real uvicorn in
a background thread, hits the JSON + Arrow IPC endpoints, asserts the
wire shapes the frontend depends on.

Marked ``slow`` so it stays out of the default suite. Run via::

    pytest -m slow tests/test_viz_e2e.py
"""

from __future__ import annotations

import io
import socket
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from constellation.sequencing.schemas.quant import COVERAGE_TABLE  # noqa: E402
from constellation.sequencing.schemas.reference import CONTIG_TABLE  # noqa: E402
from constellation.viz.server.app import create_app  # noqa: E402
from constellation.viz.server.arrow_stream import ARROW_IPC_MEDIA_TYPE  # noqa: E402
from constellation.viz.server.session import Session  # noqa: E402


pytestmark = pytest.mark.slow


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _build_fixture_session(tmp_path: Path) -> Session:
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
                }
            ],
            schema=CONTIG_TABLE,
        ),
        genome / "contigs.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "contig_id": pa.array([1], pa.int64()),
                "sequence": pa.array(["N" * 100], pa.string()),
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
            ],
            schema=COVERAGE_TABLE,
        ),
        align / "coverage.parquet",
    )
    return Session.from_root(root)


def test_uvicorn_serves_arrow_ipc_end_to_end(tmp_path: Path) -> None:
    session = _build_fixture_session(tmp_path)
    app = create_app(session)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base}/api/health", timeout=0.5)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail("server did not become ready within 5s")

    try:
        # Health probe carries the version + registered sessions.
        body = httpx.get(f"{base}/api/health").json()
        assert body["ok"] is True
        assert session.session_id in body["sessions"]

        # Manifest round-trip
        manifest = httpx.get(
            f"{base}/api/sessions/{session.session_id}/manifest",
        ).json()
        assert manifest["paths"]["coverage"] == "S2_align/coverage.parquet"

        # Track list — coverage_histogram should be discoverable
        tracks = httpx.get(
            f"{base}/api/tracks", params={"session": session.session_id}
        ).json()
        kinds = {row["kind"] for row in tracks}
        assert "coverage_histogram" in kinds

        # Arrow IPC stream
        r = httpx.get(
            f"{base}/api/tracks/coverage_histogram/data",
            params={
                "session": session.session_id,
                "binding": "coverage",
                "contig": "chr1",
                "start": 0,
                "end": 3000,
            },
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(ARROW_IPC_MEDIA_TYPE)
        assert r.headers["x-track-mode"] == "vector"
        table = pa.ipc.RecordBatchStreamReader(io.BytesIO(r.content)).read_all()
        assert table.num_rows == 2
    finally:
        server.should_exit = True
        thread.join(timeout=5)
