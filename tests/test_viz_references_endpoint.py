"""Reference-cache enumeration endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


from _viz_fixtures import DEFAULT_ASSEMBLY, install_fake_reference


def _make_app():
    from constellation.viz.server.app import create_app

    return create_app({})


def test_references_endpoint_empty_when_cache_empty(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    client = TestClient(_make_app())
    r = client.get("/api/references")
    assert r.status_code == 200
    assert r.json() == []


def test_references_endpoint_enumerates_installed(
    tmp_path: Path, monkeypatch
) -> None:
    cache = tmp_path / "refs"
    cache.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    install_fake_reference(
        cache,
        handle="homo_sapiens@ensembl-111",
        assembly_accession="GRCh38.p14",
    )
    install_fake_reference(
        cache,
        handle="mus_musculus@ensembl-112",
        assembly_accession="GRCm39",
    )
    client = TestClient(_make_app())
    r = client.get("/api/references")
    assert r.status_code == 200
    rows = r.json()
    handles = sorted(row["handle"] for row in rows)
    assert handles == [
        "homo_sapiens@ensembl-111",
        "mus_musculus@ensembl-112",
    ]
    by_handle = {row["handle"]: row for row in rows}
    assert by_handle["homo_sapiens@ensembl-111"]["assembly_accession"] == "GRCh38.p14"
    assert by_handle["mus_musculus@ensembl-112"]["organism"] == "mus_musculus"
    assert by_handle["mus_musculus@ensembl-112"]["is_default"] is False


def test_references_endpoint_surfaces_default_flag(
    tmp_path: Path, monkeypatch
) -> None:
    cache = tmp_path / "refs"
    cache.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    install_fake_reference(cache, handle="homo_sapiens@ensembl-111")
    # Pin it via the handle module's API so the on-disk defaults.toml
    # matches production write semantics.
    from constellation.sequencing.reference.handle import set_default

    set_default("homo_sapiens", "ensembl-111", root=cache)
    client = TestClient(_make_app())
    r = client.get("/api/references")
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["is_default"] is True
