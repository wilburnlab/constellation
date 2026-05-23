"""Saved-session cache — list/read/write/delete + endpoints.

Tests both the standalone cache module (`constellation.viz.sessions`)
and the corresponding FastAPI endpoints (`/api/saved-sessions`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


from constellation.viz.sessions import (
    cache_root as sessions_cache_root,
    delete_saved,
    list_saved,
    read_saved,
    write_saved,
)


# ----------------------------------------------------------------------
# Cache module direct
# ----------------------------------------------------------------------


def test_write_read_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path))
    saved = write_saved(
        label="My Mouse RNA",
        reference_handle="mus_musculus@ensembl-111",
        sources=[
            {"path": "/data/align1", "kind": "align", "label": "align-A"},
            {"path": "/data/cluster1", "kind": "cluster", "label": "cluster-A"},
        ],
        last_viewed_locus={"contig": "chr1", "start": 100, "end": 5000},
    )
    assert saved.path is not None and saved.path.exists()

    reloaded = read_saved(saved.slug)
    assert reloaded.label == "My Mouse RNA"
    assert reloaded.reference_handle == "mus_musculus@ensembl-111"
    assert len(reloaded.sources) == 2
    assert reloaded.sources[0]["kind"] == "align"
    assert reloaded.last_viewed_locus == {
        "contig": "chr1",
        "start": 100,
        "end": 5000,
    }


def test_list_saved_includes_all_entries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path))
    write_saved(label="A", reference_handle="h1@x-1", sources=[
        {"path": "/p", "kind": "align"},
    ])
    write_saved(label="B", reference_handle="h1@x-1", sources=[
        {"path": "/q", "kind": "cluster"},
    ])
    listing = list_saved()
    labels = sorted(s.label for s in listing)
    assert labels == ["A", "B"]


def test_delete_saved_returns_true_on_hit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path))
    saved = write_saved(label="ephemeral", reference_handle="h1@x-1", sources=[
        {"path": "/p", "kind": "align"},
    ])
    assert delete_saved(saved.slug) is True
    assert delete_saved(saved.slug) is False  # idempotent on miss
    with pytest.raises(FileNotFoundError):
        read_saved(saved.slug)


def test_write_validates_required_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path))
    with pytest.raises(ValueError, match="label"):
        write_saved(label="", reference_handle="h", sources=[])
    with pytest.raises(ValueError, match="reference_handle"):
        write_saved(label="x", reference_handle="", sources=[])
    with pytest.raises(ValueError, match="path/kind"):
        write_saved(label="x", reference_handle="h", sources=[{"path": "/x"}])


def test_cache_root_respects_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path / "custom"))
    assert sessions_cache_root() == (tmp_path / "custom").resolve()


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------


def _make_app(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path))
    from constellation.viz.server.app import create_app

    return create_app({})


def test_endpoint_lifecycle(tmp_path: Path, monkeypatch) -> None:
    app = _make_app(tmp_path, monkeypatch)
    client = TestClient(app)

    # Empty list initially.
    r = client.get("/api/saved-sessions")
    assert r.status_code == 200
    assert r.json() == []

    # POST a saved session.
    r = client.post(
        "/api/saved-sessions",
        json={
            "label": "test",
            "reference_handle": "homo_sapiens@ensembl-111",
            "sources": [{"path": "/p", "kind": "align", "label": "A"}],
            "last_viewed_locus": {"contig": "chr1", "start": 0, "end": 1000},
        },
    )
    assert r.status_code == 201
    summary = r.json()
    slug = summary["slug"]

    # GET single.
    r = client.get(f"/api/saved-sessions/{slug}")
    assert r.status_code == 200
    payload = r.json()
    assert payload["label"] == "test"
    assert payload["sources"][0]["path"] == "/p"

    # GET list shows it.
    r = client.get("/api/saved-sessions")
    assert any(item["slug"] == slug for item in r.json())

    # DELETE.
    r = client.delete(f"/api/saved-sessions/{slug}")
    assert r.status_code == 204
    r = client.delete(f"/api/saved-sessions/{slug}")
    assert r.status_code == 404


def test_endpoint_validation_errors(tmp_path: Path, monkeypatch) -> None:
    app = _make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.post(
        "/api/saved-sessions",
        json={"label": "", "reference_handle": "h@x-1", "sources": []},
    )
    assert r.status_code == 400


def test_endpoint_get_missing_slug_404(tmp_path: Path, monkeypatch) -> None:
    app = _make_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get("/api/saved-sessions/missing-slug-0000")
    assert r.status_code == 404
