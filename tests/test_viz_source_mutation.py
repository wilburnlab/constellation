"""Runtime source-mutation endpoints + saved-session v2 layout block.

Covers:
- ``Session.source_id`` stability across add/remove
- ``Session.with_sources()`` preserves ``session_id``
- ``invalidate_binding_cache()`` evicts every ``(session_id, kind)`` entry
- ``POST /api/sessions/{id}/sources`` and ``DELETE /api/sessions/{id}/sources/{source_id}``
- ``[[track_layout]]`` round-trip in saved-session TOML (v2)
- ``v1 → v2`` migration: v1 files read fine, get rewritten as v2 on next save
- ``PATCH /api/saved-sessions/{slug}/layout``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

from _viz_fixtures import (
    DEFAULT_HANDLE,
    build_viz_session,
    install_fake_reference,
    write_align_source,
)


# ----------------------------------------------------------------------
# source_id stability + with_sources()
# ----------------------------------------------------------------------


def test_source_id_stable_across_add(tmp_path: Path, monkeypatch) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[
            {"coverage": [{"sample_id": 0, "contig_id": 1, "start": 0, "end": 10, "depth": 5.0}]},
        ],
    )
    original_id = session.sources[0].source_id

    # Build a second source dir and rebuild via with_sources()
    new_src = tmp_path / "align-extra"
    write_align_source(
        new_src,
        coverage=[{"sample_id": 0, "contig_id": 1, "start": 50, "end": 60, "depth": 9.0}],
    )
    rebuilt = session.with_sources(
        [
            {"path": str(session.sources[0].path), "kind": "align"},
            {"path": str(new_src), "kind": "align"},
        ]
    )
    assert rebuilt.session_id == session.session_id
    # First source's id unchanged
    assert rebuilt.sources[0].source_id == original_id
    # Second source has a different id
    assert rebuilt.sources[1].source_id != original_id


def test_source_id_stable_across_remove(tmp_path: Path, monkeypatch) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[
            {"coverage": [{"sample_id": 0, "contig_id": 1, "start": 0, "end": 10, "depth": 5.0}]},
            {"coverage": [{"sample_id": 0, "contig_id": 1, "start": 0, "end": 10, "depth": 7.0}]},
        ],
    )
    keep_id = session.sources[0].source_id
    rebuilt = session.with_sources(
        [{"path": str(session.sources[0].path), "kind": "align"}]
    )
    assert rebuilt.session_id == session.session_id
    assert len(rebuilt.sources) == 1
    assert rebuilt.sources[0].source_id == keep_id


# ----------------------------------------------------------------------
# Binding-cache invalidator
# ----------------------------------------------------------------------


def test_invalidate_evicts_only_matching_session(tmp_path: Path, monkeypatch) -> None:
    from constellation.viz.server.endpoints.tracks import invalidate_binding_cache

    cache: dict[tuple[str, str], list] = {
        ("sess-A", "coverage_histogram"): [],
        ("sess-A", "read_pileup"): [],
        ("sess-B", "coverage_histogram"): [],
    }
    invalidate_binding_cache(cache, "sess-A")
    assert ("sess-A", "coverage_histogram") not in cache
    assert ("sess-A", "read_pileup") not in cache
    assert ("sess-B", "coverage_histogram") in cache


# ----------------------------------------------------------------------
# POST / DELETE source endpoints
# ----------------------------------------------------------------------


def _build_app(session) -> Any:
    from constellation.viz.server.app import create_app

    return create_app({session.session_id: session})


def test_post_source_appends_and_invalidates(tmp_path: Path, monkeypatch) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[
            {"coverage": [{"sample_id": 0, "contig_id": 1, "start": 0, "end": 10, "depth": 5.0}]},
        ],
    )
    app = _build_app(session)
    client = TestClient(app)

    # Prime cache: list tracks before mutation
    r = client.get(f"/api/tracks?session={session.session_id}")
    assert r.status_code == 200
    initial = r.json()
    assert any(b["kind"] == "coverage_histogram" for b in initial)
    assert len(app.state.track_bindings_cache) > 0

    # Build a new source dir
    new_src = tmp_path / "align-fresh"
    write_align_source(
        new_src,
        coverage=[{"sample_id": 0, "contig_id": 1, "start": 100, "end": 200, "depth": 12.0}],
    )

    r = client.post(
        f"/api/sessions/{session.session_id}/sources",
        json={"path": str(new_src), "kind": "align"},
    )
    assert r.status_code == 201, r.text
    payload = r.json()
    assert len(payload["sources"]) == 2
    # session_id is preserved across rebuild
    assert payload["session_id"] == session.session_id
    # Cache was invalidated (will be re-populated on next /api/tracks)
    cleared = all(k[0] != session.session_id for k in app.state.track_bindings_cache)
    assert cleared

    # Now /api/tracks should reflect both sources' bindings
    r = client.get(f"/api/tracks?session={session.session_id}")
    assert r.status_code == 200
    after = r.json()
    coverage_bindings = [b for b in after if b["kind"] == "coverage_histogram"]
    assert len(coverage_bindings) == 2
    # Each has a non-null source_id
    assert all(b["source_id"] for b in coverage_bindings)


def test_delete_source_removes_and_invalidates(tmp_path: Path, monkeypatch) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[
            {"coverage": [{"sample_id": 0, "contig_id": 1, "start": 0, "end": 10, "depth": 5.0}]},
            {"coverage": [{"sample_id": 0, "contig_id": 1, "start": 0, "end": 10, "depth": 7.0}]},
        ],
    )
    target_id = session.sources[1].source_id
    survivor_id = session.sources[0].source_id

    app = _build_app(session)
    client = TestClient(app)
    # Prime cache
    client.get(f"/api/tracks?session={session.session_id}")
    assert len(app.state.track_bindings_cache) > 0

    r = client.delete(
        f"/api/sessions/{session.session_id}/sources/{target_id}"
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["source_id"] == survivor_id
    cleared = all(k[0] != session.session_id for k in app.state.track_bindings_cache)
    assert cleared


def test_delete_source_unknown_returns_404(tmp_path: Path, monkeypatch) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[
            {"coverage": [{"sample_id": 0, "contig_id": 1, "start": 0, "end": 10, "depth": 5.0}]},
        ],
    )
    app = _build_app(session)
    client = TestClient(app)
    r = client.delete(f"/api/sessions/{session.session_id}/sources/src-deadbeef")
    assert r.status_code == 404


def test_post_source_unknown_session_404(tmp_path: Path, monkeypatch) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[
            {"coverage": [{"sample_id": 0, "contig_id": 1, "start": 0, "end": 10, "depth": 5.0}]},
        ],
    )
    app = _build_app(session)
    client = TestClient(app)
    r = client.post(
        "/api/sessions/no-such-session/sources",
        json={"path": "/tmp/anywhere"},
    )
    assert r.status_code == 404


# ----------------------------------------------------------------------
# Saved-session schema v1 → v2 migration
# ----------------------------------------------------------------------


def test_v1_saved_session_reads_clean(tmp_path: Path, monkeypatch) -> None:
    """A v1 TOML on disk reads as a SavedSession with track_layout=None."""
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path))
    v1_path = tmp_path / "legacy.toml"
    v1_path.write_text(
        "schema_version = 1\n"
        'label = "Legacy"\n'
        'reference_handle = "h@x-1"\n'
        'saved_at = "2026-01-01T00:00:00Z"\n'
        "\n"
        "[[sources]]\n"
        'path = "/x"\n'
        'kind = "align"\n'
        'label = "a"\n',
        encoding="utf-8",
    )
    from constellation.viz.sessions import read_saved

    saved = read_saved("legacy")
    assert saved.label == "Legacy"
    assert saved.track_layout is None


def test_v2_saved_session_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path))
    from constellation.viz.sessions import read_saved, write_saved

    layout = [
        {
            "source_id": "src-aaaa",
            "kind": "coverage_histogram",
            "visible": True,
            "display_order": 0,
            "height_px": 120,
            "collapsed": False,
        },
        {
            "source_id": "",
            "kind": "reference_sequence",
            "visible": False,
            "display_order": 5,
            "height_px": 40,
            "collapsed": True,
        },
    ]
    saved = write_saved(
        label="X",
        reference_handle="h@x-1",
        sources=[{"path": "/p", "kind": "align"}],
        track_layout=layout,
    )
    reloaded = read_saved(saved.slug)
    assert reloaded.track_layout is not None
    assert len(reloaded.track_layout) == 2
    by_kind = {e["kind"]: e for e in reloaded.track_layout}
    assert by_kind["coverage_histogram"]["height_px"] == 120
    assert by_kind["coverage_histogram"]["visible"] is True
    assert by_kind["reference_sequence"]["collapsed"] is True
    assert by_kind["reference_sequence"]["source_id"] == ""


def test_v1_rewrites_as_v2_on_next_write(tmp_path: Path, monkeypatch) -> None:
    """When a v1 file gets re-saved, it comes back as v2."""
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path))
    v1_path = tmp_path / "legacy.toml"
    v1_path.write_text(
        "schema_version = 1\n"
        'label = "Legacy"\n'
        'reference_handle = "h@x-1"\n'
        'saved_at = "2026-01-01T00:00:00Z"\n'
        "\n"
        "[[sources]]\n"
        'path = "/x"\n'
        'kind = "align"\n'
        'label = "a"\n',
        encoding="utf-8",
    )
    from constellation.viz.sessions import read_saved, write_saved

    saved = read_saved("legacy")
    write_saved(
        label=saved.label,
        reference_handle=saved.reference_handle,
        sources=saved.sources,
        slug=saved.slug,
        saved_at=saved.saved_at,
    )
    raw = v1_path.read_text(encoding="utf-8")
    assert "schema_version = 2" in raw


# ----------------------------------------------------------------------
# PATCH /api/saved-sessions/{slug}/layout
# ----------------------------------------------------------------------


def test_patch_layout_writes_track_layout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path))
    from constellation.viz.server.app import create_app

    app = create_app({})
    client = TestClient(app)
    r = client.post(
        "/api/saved-sessions",
        json={
            "label": "L",
            "reference_handle": "h@x-1",
            "sources": [{"path": "/p", "kind": "align", "label": "A"}],
        },
    )
    assert r.status_code == 201
    slug = r.json()["slug"]

    layout = [
        {
            "source_id": "src-aaaa",
            "kind": "coverage_histogram",
            "visible": False,
            "display_order": 1,
            "height_px": 200,
            "collapsed": False,
        }
    ]
    r = client.patch(
        f"/api/saved-sessions/{slug}/layout",
        json={"track_layout": layout},
    )
    assert r.status_code == 200, r.text

    r = client.get(f"/api/saved-sessions/{slug}")
    assert r.status_code == 200
    payload = r.json()
    assert payload["track_layout"][0]["source_id"] == "src-aaaa"
    assert payload["track_layout"][0]["visible"] is False
    assert payload["track_layout"][0]["height_px"] == 200


def test_patch_layout_404_for_missing_slug(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONSTELLATION_SESSIONS_HOME", str(tmp_path))
    from constellation.viz.server.app import create_app

    app = create_app({})
    client = TestClient(app)
    r = client.patch(
        "/api/saved-sessions/no-such/layout",
        json={"track_layout": []},
    )
    assert r.status_code == 404


# ----------------------------------------------------------------------
# Manifest sources include source_id
# ----------------------------------------------------------------------


def test_manifest_sources_carry_source_id(tmp_path: Path, monkeypatch) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[
            {"coverage": [{"sample_id": 0, "contig_id": 1, "start": 0, "end": 10, "depth": 5.0}]},
        ],
    )
    manifest = session.to_manifest()
    for src in manifest["sources"]:
        assert "source_id" in src
        assert src["source_id"].startswith("src-")
