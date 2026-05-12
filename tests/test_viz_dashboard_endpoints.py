"""Dashboard FastAPI endpoints — schema + commands.

Uses ``httpx.AsyncClient`` with ``ASGITransport`` rather than
``TestClient`` so a single persistent event loop spans every request —
``TestClient`` tears down its loop between requests, which kills any
background subprocess pump tasks and prematurely releases the runner
lock. Gated on ``[viz]`` extras (``fastapi`` + ``httpx``).
"""

from __future__ import annotations

import asyncio
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from constellation.viz.runner import registry as registry_mod  # noqa: E402
from constellation.viz.runner import runner as runner_mod  # noqa: E402
from constellation.viz.runner.lock import (  # noqa: E402
    is_locked,
    release_lock,
)
from constellation.viz.server.app import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_runner_state(monkeypatch):
    """Each test starts with a fresh lock + empty job registry +
    a controlled spawn prefix."""
    if is_locked():
        release_lock()
    registry_mod._JOBS.clear()
    # Substitute python -c so tests don't hit the real CLI.
    monkeypatch.setattr(runner_mod, "SPAWN_PREFIX", (sys.executable, "-c"))
    yield
    if is_locked():
        release_lock()
    registry_mod._JOBS.clear()


def _run(coro):
    """Helper: run an async test body to completion on a fresh loop.

    The fresh loop is fine here — we keep it open for the duration of
    the test body so the runner's pump task survives across requests.
    """
    return asyncio.new_event_loop().run_until_complete(coro)


def _client():
    """Build the dashboard app + an ASGI httpx client."""
    app = create_app({})
    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_cli_schema_endpoint_shape():
    async def body():
        async with _client() as c:
            r = await c.get("/api/cli/schema")
        assert r.status_code == 200
        data = r.json()
        assert data["prog"] == "constellation"
        names = {sc["name"] for sc in data["subcommands"]}
        assert {"doctor", "transcriptome", "reference", "viz", "dashboard"} <= names
        assert isinstance(data["curated"], list)
        assert len(data["curated"]) > 0

    _run(body())


def test_post_command_starts_job_and_returns_id():
    async def body():
        async with _client() as c:
            # `import sys; sys.exit(0)` — exits cleanly with no output.
            r = await c.post(
                "/api/commands",
                json={"argv": ["import sys; sys.exit(0)"]},
            )
            assert r.status_code == 200
            data = r.json()
            job_id = data["job_id"]
            assert data["state"] == "running"
            # Wait for completion
            for _ in range(50):
                await asyncio.sleep(0.05)
                snap = await c.get(f"/api/commands/{job_id}")
                if snap.json()["exit_code"] is not None:
                    break
            final = snap.json()
            assert final["state"] == "completed"
            assert final["exit_code"] == 0
        await asyncio.sleep(0.05)
        assert not is_locked()

    _run(body())


def test_concurrent_post_returns_409():
    async def body():
        async with _client() as c:
            r1 = await c.post(
                "/api/commands",
                json={"argv": ["import time; time.sleep(2)"]},
            )
            assert r1.status_code == 200
            r2 = await c.post(
                "/api/commands",
                json={"argv": ["import sys; sys.exit(0)"]},
            )
            assert r2.status_code == 409
            assert "another" in r2.json()["detail"].lower()
            # Cancel the first job so the lock releases quickly
            await c.delete(f"/api/commands/{r1.json()['job_id']}")
            # Wait for cancellation
            for _ in range(60):
                await asyncio.sleep(0.05)
                snap = await c.get(f"/api/commands/{r1.json()['job_id']}")
                if snap.json()["exit_code"] is not None:
                    break
            await asyncio.sleep(0.05)
        # Lock released after cancel completes
        assert not is_locked()

    _run(body())


def test_active_endpoint_reports_running_then_idle():
    async def body():
        async with _client() as c:
            # No active job initially
            r = await c.get("/api/commands/active")
            assert r.status_code == 200
            assert r.json() is None
            # Start a job
            r = await c.post(
                "/api/commands",
                json={"argv": ["import sys; print('hi'); sys.exit(0)"]},
            )
            job_id = r.json()["job_id"]
            # Active right after start (may have already finished — short job)
            r = await c.get("/api/commands/active")
            assert r.status_code == 200
            # Wait for completion
            for _ in range(60):
                await asyncio.sleep(0.05)
                snap = await c.get(f"/api/commands/{job_id}")
                if snap.json()["exit_code"] is not None:
                    break
            # Active now None
            r = await c.get("/api/commands/active")
            assert r.json() is None

    _run(body())


def test_cancel_unknown_job_404():
    async def body():
        async with _client() as c:
            r = await c.delete("/api/commands/00000000-0000-0000-0000-000000000000")
            assert r.status_code == 404

    _run(body())


def test_get_unknown_job_404():
    async def body():
        async with _client() as c:
            r = await c.get("/api/commands/00000000-0000-0000-0000-000000000000")
            assert r.status_code == 404

    _run(body())
