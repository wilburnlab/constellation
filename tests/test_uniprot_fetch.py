"""Tests for the back-compat ``fetch_swissprot`` shim.

After the reference-portal integration, ``fetch_swissprot`` delegates
to the portal's proteome-only fetch path: SwissProt lands at
``<cache_root>/swissprot/uniprot-<release>/protein.faa`` with a v2
``meta.toml`` flagged ``has_genome=false`` / ``has_proteome=true``.

A local in-process HTTP server serves a synthetic ``reldate.txt`` and a
tiny gzipped ``uniprot_sprot.fasta.gz``; the test points
``_TEST_HTTP_BASE_OVERRIDE`` at it. Covers:

  * Cache miss → writes the expected portal layout (``protein.faa`` +
    ``meta.toml`` with the [contents] block) under the new path.
  * Cache hit → idempotent short-circuit (no HTTP calls on second run).
  * ``force=True`` → re-fetch even if cache is complete.
  * Release auto-detect → parses ``Release YYYY_NN`` from reldate.txt.
  * Bad reldate.txt → ``ValueError``.
"""

from __future__ import annotations

import gzip
import hashlib
import http.server
import socketserver
import threading
import tomllib

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
    """Spin up an in-process HTTP server rooted at a per-test tmp dir."""
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
    (serve_root / "uniprot_sprot.fasta.gz").write_bytes(
        gzip.compress(_FASTA_FIXTURE)
    )
    monkeypatch.setattr(uniprot_mod, "_TEST_HTTP_BASE_OVERRIDE", base_url)
    return serve_root, base_url


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Redirect the reference portal cache root for the test."""
    root = tmp_path / "refs"
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(root))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return root


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
# fetch_swissprot (back-compat shim)
# ──────────────────────────────────────────────────────────────────────


def test_fetch_swissprot_first_run_writes_portal_layout(
    stub_uniprot_http, cache_root
) -> None:
    """Cache miss writes the standard portal layout (protein.faa)."""
    handle = fetch_swissprot(release="2026_02")
    assert isinstance(handle, SwissprotHandle)
    assert handle.release == "2026_02"
    # Portal layout: <root>/swissprot/uniprot-<release>/protein.faa
    expected = cache_root / "swissprot" / "uniprot-2026_02" / "protein.faa"
    assert handle.fasta_path == expected.resolve()
    assert handle.fasta_path.read_bytes() == _FASTA_FIXTURE
    # meta.toml is the v2 portal shape (not the legacy one-off).
    assert (handle.fasta_path.parent / "meta.toml").is_file()
    # SHA256 matches the gunzipped FASTA bytes.
    assert handle.sha256 == hashlib.sha256(_FASTA_FIXTURE).hexdigest()


def test_fetch_swissprot_meta_toml_is_portal_v2_shape(
    stub_uniprot_http, cache_root
) -> None:
    """meta.toml carries the v2 [contents] block flagging proteome-only."""
    fetch_swissprot(release="2026_02")
    meta_path = (
        cache_root / "swissprot" / "uniprot-2026_02" / "meta.toml"
    ).resolve()
    meta = tomllib.loads(meta_path.read_text())
    assert meta["schema_version"] == 2
    assert meta["handle"] == "swissprot@uniprot-2026_02"
    assert meta["organism"] == "swissprot"
    assert meta["source"] == "uniprot"
    assert meta["release"] == "2026_02"
    # [contents] block flags this as proteome-only.
    assert meta["contents"]["has_genome"] is False
    assert meta["contents"]["has_annotation"] is False
    assert meta["contents"]["has_proteome"] is True
    assert meta["contents"]["has_cdna"] is False
    # Protein URL provenance lands under [urls.protein].
    assert meta["urls"]["protein"]["url"].endswith("/uniprot_sprot.fasta.gz")


def test_fetch_swissprot_cache_hit_short_circuits(
    stub_uniprot_http, cache_root
) -> None:
    """Second call with the same release doesn't re-fetch."""
    fetch_swissprot(release="2026_02")
    # Sabotage the on-disk FASTA — if the second call short-circuits,
    # the SHA matches the on-disk (sabotaged) bytes.
    protein_path = (
        cache_root / "swissprot" / "uniprot-2026_02" / "protein.faa"
    ).resolve()
    protein_path.write_bytes(b"corrupted")
    handle = fetch_swissprot(release="2026_02")
    assert handle.fasta_path.read_bytes() == b"corrupted"
    # The sha256 reflects the cached (sabotaged) content, confirming
    # the cache hit short-circuited the fetch.
    assert handle.sha256 == hashlib.sha256(b"corrupted").hexdigest()


def test_fetch_swissprot_force_refetches(
    stub_uniprot_http, cache_root
) -> None:
    """force=True ignores the cache and re-downloads."""
    fetch_swissprot(release="2026_02")
    protein_path = (
        cache_root / "swissprot" / "uniprot-2026_02" / "protein.faa"
    ).resolve()
    protein_path.write_bytes(b"corrupted")
    handle = fetch_swissprot(release="2026_02", force=True)
    assert handle.fasta_path.read_bytes() == _FASTA_FIXTURE


def test_fetch_swissprot_auto_detect_release(
    stub_uniprot_http, cache_root
) -> None:
    """release=None autodetects via reldate.txt."""
    handle = fetch_swissprot(release=None)
    assert handle.release == "2026_02"
    # Lands under the autodetected slug.
    assert "uniprot-2026_02" in str(handle.fasta_path)


def test_fetch_swissprot_resolves_via_bare_handle(
    stub_uniprot_http, cache_root
) -> None:
    """After install, Reference.open("swissprot") resolves the bare handle."""
    from constellation.sequencing.reference import Reference

    fetch_swissprot(release="2026_02")
    ref = Reference.open("swissprot")
    assert ref.has_proteome
    assert ref.has_genome is False
    assert ref.protein_fasta_path.read_bytes() == _FASTA_FIXTURE
    assert ref.handle.source == "uniprot"
    assert ref.handle.release == "2026_02"


# ──────────────────────────────────────────────────────────────────────
# `uniprot:swissprot` legacy spec form via _resolve_spec / fetch_reference
# ──────────────────────────────────────────────────────────────────────


def test_resolve_spec_uniprot_swissprot_with_explicit_release(
    stub_uniprot_http,
) -> None:
    """The ``uniprot:swissprot`` spec form returns a proteome-only
    ``_ResolvedSpec`` with the SwissProt URL populated and the standard
    portal handle stamped."""
    from constellation.sequencing.reference.fetch import _resolve_spec

    spec = _resolve_spec("uniprot:swissprot", release="2026_02", source=None)
    assert spec.handle.organism == "swissprot"
    assert spec.handle.source == "uniprot"
    assert spec.handle.release == "2026_02"
    # Proteome-only signal: empty fasta_url + gff_url, populated protein_url.
    assert spec.fasta_url == ""
    assert spec.gff_url == ""
    assert spec.protein_url
    assert spec.protein_url.endswith("/uniprot_sprot.fasta.gz")
    assert spec.cdna_url is None
    assert spec.annotation_release is None


def test_resolve_spec_uniprot_swissprot_autodetects_release(
    stub_uniprot_http,
) -> None:
    """Omitting --release routes through ``_probe_swissprot_release``."""
    from constellation.sequencing.reference.fetch import _resolve_spec

    spec = _resolve_spec("uniprot:swissprot", release=None, source=None)
    assert spec.handle.release == "2026_02"  # from the stubbed reldate fixture


def test_resolve_spec_uniprot_unknown_identifier_raises(stub_uniprot_http) -> None:
    """``uniprot:<not-swissprot>`` raises a KeyError that points the user
    at the bare-species-name path for per-species proteomes."""
    from constellation.sequencing.reference.fetch import _resolve_spec

    with pytest.raises(KeyError, match="unknown uniprot identifier"):
        _resolve_spec("uniprot:UP000005640", release=None, source=None)


def test_fetch_reference_uniprot_swissprot_end_to_end(
    stub_uniprot_http, cache_root
) -> None:
    """``fetch_reference('uniprot:swissprot', release='2026_02')`` writes
    the same portal layout as the back-compat ``fetch_swissprot()`` shim."""
    from constellation.sequencing.reference.fetch import fetch_reference

    result = fetch_reference("uniprot:swissprot", release="2026_02")
    assert result.genome is None
    assert result.annotation is None
    assert result.protein_fasta_path is not None
    assert result.protein_fasta_path.read_bytes() == _FASTA_FIXTURE
    expected = cache_root / "swissprot" / "uniprot-2026_02" / "protein.faa"
    assert result.protein_fasta_path == expected.resolve()
