"""Tier A tests for :func:`constellation.catalog.uniprot.fetch_swissprot`.

A local in-process HTTP server serves a synthetic ``reldate.txt`` and a
tiny gzipped ``uniprot_sprot.fasta.gz``; the test sets the
``_TEST_HTTP_BASE_OVERRIDE`` module-level shim to point at it. Covers:

  * Cache miss → writes the expected layout (sprot.fasta + sprot.fasta.gz +
    meta.toml + _SUCCESS) under ``<cache_dir>/swissprot/<release>/``
  * Cache hit → idempotent short-circuit (no HTTP calls on second run)
  * ``force=True`` → re-fetch even if cache is complete
  * Release auto-detect → parses ``Release YYYY_NN`` from reldate.txt
  * Bad reldate.txt → ``ValueError``
"""

from __future__ import annotations

import gzip
import http.server
import socketserver
import threading
import tomllib
from pathlib import Path

import pytest

from constellation.catalog import uniprot as uniprot_mod
from constellation.catalog.uniprot import (
    SwissprotHandle,
    _probe_swissprot_release,
    fetch_swissprot,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


_FASTA_FIXTURE = b""">sp|P12345|TEST_HUMAN Test protein
MKLIGHTPEPTRAAA
>sp|Q67890|FOO_MOUSE Another test
ACDEFGHIKLMNPQRSTVWY
"""


_RELDATE_FIXTURE = b"""UniProt Knowledgebase Release 2026_02 consists of:
Swiss-Prot Release 2026_02 of 14-May-2026
TrEMBL    Release 2026_02 of 14-May-2026
"""


_RELDATE_BAD = b"""no release number here
just garbage
"""


@pytest.fixture
def serve_dir(tmp_path_factory):
    """Spin up an in-process HTTP server rooted at a per-test tmp dir.

    Returns ``(serve_root, base_url)``. Files written to ``serve_root``
    become reachable at ``<base_url>/<filename>``.
    """
    serve_root = tmp_path_factory.mktemp("uniprot_http")

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_root), **kwargs)

        def log_message(self, *args, **kwargs):
            return  # silent

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield serve_root, f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def stub_uniprot_http(serve_dir, monkeypatch):
    """Lay down a complete (reldate + fasta.gz) UniProt-like tree on the
    in-process server and point ``_TEST_HTTP_BASE_OVERRIDE`` at it."""
    serve_root, base_url = serve_dir
    (serve_root / "reldate.txt").write_bytes(_RELDATE_FIXTURE)
    # Gzip the FASTA fixture so the fetcher exercises its gunzip step.
    (serve_root / "uniprot_sprot.fasta.gz").write_bytes(
        gzip.compress(_FASTA_FIXTURE)
    )
    monkeypatch.setattr(uniprot_mod, "_TEST_HTTP_BASE_OVERRIDE", base_url)
    return serve_root, base_url


# ──────────────────────────────────────────────────────────────────────
# Release autodetect
# ──────────────────────────────────────────────────────────────────────


def test_probe_swissprot_release_parses_reldate(stub_uniprot_http) -> None:
    assert _probe_swissprot_release() == "2026_02"


def test_probe_swissprot_release_raises_on_garbage(
    serve_dir, monkeypatch
) -> None:
    serve_root, base_url = serve_dir
    (serve_root / "reldate.txt").write_bytes(_RELDATE_BAD)
    monkeypatch.setattr(uniprot_mod, "_TEST_HTTP_BASE_OVERRIDE", base_url)
    with pytest.raises(ValueError, match="could not parse UniProt release"):
        _probe_swissprot_release()


# ──────────────────────────────────────────────────────────────────────
# fetch_swissprot
# ──────────────────────────────────────────────────────────────────────


def test_fetch_swissprot_first_run_writes_cache(
    stub_uniprot_http, tmp_path
) -> None:
    cache = tmp_path / "swissprot_2026_02"
    handle = fetch_swissprot(release="2026_02", cache_dir=cache)
    assert isinstance(handle, SwissprotHandle)
    assert handle.release == "2026_02"
    assert handle.fasta_path == cache / "sprot.fasta"
    assert handle.fasta_path.read_bytes() == _FASTA_FIXTURE
    assert (cache / "sprot.fasta.gz").is_file()
    assert (cache / "meta.toml").is_file()
    assert (cache / "_SUCCESS").is_file()
    # SHA256 matches the gunzipped FASTA bytes.
    import hashlib
    expected_sha = hashlib.sha256(_FASTA_FIXTURE).hexdigest()
    assert handle.sha256 == expected_sha


def test_fetch_swissprot_meta_toml_shape(stub_uniprot_http, tmp_path) -> None:
    cache = tmp_path / "swissprot_2026_02"
    fetch_swissprot(release="2026_02", cache_dir=cache)
    meta = tomllib.loads((cache / "meta.toml").read_text())
    assert meta["release"] == "2026_02"
    assert meta["source_url"].endswith("/uniprot_sprot.fasta.gz")
    assert len(meta["sha256"]) == 64
    assert "constellation_version" in meta
    assert "fetched_at" in meta


def test_fetch_swissprot_cache_hit_short_circuits(
    stub_uniprot_http, tmp_path, monkeypatch
) -> None:
    """Second call with the same release + cache_dir doesn't re-fetch."""
    cache = tmp_path / "swissprot_2026_02"
    fetch_swissprot(release="2026_02", cache_dir=cache)
    # Sabotage the fasta.gz so a re-fetch would produce different bytes.
    (cache / "sprot.fasta.gz").write_bytes(b"corrupted-bytes")
    # If the second call short-circuits, sprot.fasta keeps the original
    # FASTA bytes.
    handle = fetch_swissprot(release="2026_02", cache_dir=cache)
    assert handle.fasta_path.read_bytes() == _FASTA_FIXTURE


def test_fetch_swissprot_force_refetches(
    stub_uniprot_http, tmp_path
) -> None:
    """``force=True`` ignores the cache and re-downloads."""
    cache = tmp_path / "swissprot_2026_02"
    fetch_swissprot(release="2026_02", cache_dir=cache)
    # Sabotage sprot.fasta — force=True should overwrite it.
    (cache / "sprot.fasta").write_bytes(b"corrupted-bytes")
    handle = fetch_swissprot(release="2026_02", cache_dir=cache, force=True)
    assert handle.fasta_path.read_bytes() == _FASTA_FIXTURE


def test_fetch_swissprot_auto_detect_release(
    stub_uniprot_http, tmp_path
) -> None:
    """release=None autodetects via reldate.txt."""
    cache = tmp_path / "swissprot_auto"
    handle = fetch_swissprot(release=None, cache_dir=cache)
    assert handle.release == "2026_02"


def test_fetch_swissprot_default_cache_location(
    stub_uniprot_http, tmp_path, monkeypatch
) -> None:
    """When cache_dir=None, lands at <cache_root>/swissprot/<release>/."""
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(tmp_path / "refs"))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    handle = fetch_swissprot(release="2026_02")
    expected = tmp_path / "refs" / "swissprot" / "2026_02" / "sprot.fasta"
    assert handle.fasta_path == expected.resolve()
    assert handle.fasta_path.is_file()


def test_fetch_swissprot_partial_dir_cleaned_on_error(
    serve_dir, tmp_path, monkeypatch
) -> None:
    """If the FASTA URL returns 404, the .partial dir is removed so a
    later run isn't blocked by stale state."""
    serve_root, base_url = serve_dir
    (serve_root / "reldate.txt").write_bytes(_RELDATE_FIXTURE)
    # Deliberately do NOT lay down uniprot_sprot.fasta.gz.
    monkeypatch.setattr(uniprot_mod, "_TEST_HTTP_BASE_OVERRIDE", base_url)
    cache = tmp_path / "swissprot_2026_02"
    with pytest.raises(Exception):
        fetch_swissprot(release="2026_02", cache_dir=cache)
    # The release dir should not exist (promote never happened).
    assert not cache.exists()
    # .partial should also be cleaned up.
    assert not Path(str(cache) + ".partial").exists()
