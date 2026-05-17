"""Integration tests for the reference fetch flow.

Network calls are stubbed: ``_resolve_spec`` returns a fixed
``_ResolvedSpec`` and ``_download_to`` materialises gzipped FASTA/GFF3
fixtures, so the test exercises the full cache-write path
(meta.toml, sha256, atomic rename, idempotency, --output-dir mirror)
without touching the wire.
"""

from __future__ import annotations

import gzip
import json
import tomllib
from pathlib import Path

import pytest

from constellation.sequencing.reference import fetch as ref_fetch
from constellation.sequencing.reference import handle as ref_handle
from constellation.sequencing.reference.fetch import (
    _ResolvedSpec,
    fetch_reference,
)
from constellation.sequencing.reference.handle import (
    Handle,
    parse_handle,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


FASTA_FIXTURE = b""">chr1
ACGTACGTACGT
>chr2
TTTTGGGG
"""


GFF3_FIXTURE = b"""##gff-version 3
chr1\ttest\tgene\t1\t12\t.\t+\t.\tID=gene1;Name=ALPHA
chr1\ttest\texon\t1\t12\t.\t+\t.\tID=exon1;Parent=gene1
chr2\ttest\tgene\t1\t8\t.\t-\t.\tID=gene2;Name=BETA
chr2\ttest\texon\t1\t8\t.\t-\t.\tID=exon2;Parent=gene2
"""


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    cache = tmp_path / "refs"
    cache.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return cache


@pytest.fixture
def stub_network(monkeypatch):
    """Replace ``_resolve_spec`` + ``_download_to`` with deterministic stubs.

    Returned dict lets tests tweak per-call behaviour: set
    ``checksum_kind`` to None to test the "no checksums" path, or
    ``checksum_passes`` to False to test the warn-not-fail branch.
    """
    state = {
        "spec_handle": "homo_sapiens@ensembl-111",
        "assembly_name": "GRCh38.p14",
        "annotation_release": "111",
        "assembly_accession": "GCA_000001405.29",
        "source": "ensembl",
        "checksums_url": "https://test/CHECKSUMS",
        "checksums_kind": "ensembl",
        "checksum_passes": True,
        "download_calls": [],
    }

    def _fake_resolve_spec(spec, *, release=None):
        h = parse_handle(state["spec_handle"])
        return _ResolvedSpec(
            handle=h,
            fasta_url="https://test/foo.dna.toplevel.fa.gz",
            gff_url="https://test/foo.111.gff3.gz",
            checksums_url=state["checksums_url"],
            checksums_kind=state["checksums_kind"],
            assembly_name=state["assembly_name"],
            annotation_release=state["annotation_release"],
            assembly_accession=state["assembly_accession"],
        )

    def _fake_download_to(url, dest, *, timeout=60):
        state["download_calls"].append(url)
        if "gff" in url:
            payload = GFF3_FIXTURE
        else:
            payload = FASTA_FIXTURE
        # Compress and write so the fetch code's gunzip step works.
        dest.write_bytes(gzip.compress(payload))
        return ('"stub-etag"', "Mon, 16 May 2026 12:00:00 GMT")

    def _fake_verify_ensembl(checksums_url, files, *, timeout=60):
        return state["checksum_passes"]

    def _fake_verify_refseq(checksums_url, files, *, timeout=60):
        return state["checksum_passes"]

    monkeypatch.setattr(ref_fetch, "_resolve_spec", _fake_resolve_spec)
    monkeypatch.setattr(ref_fetch, "_download_to", _fake_download_to)
    monkeypatch.setattr(ref_fetch, "_verify_ensembl_checksums", _fake_verify_ensembl)
    monkeypatch.setattr(ref_fetch, "_verify_refseq_checksums", _fake_verify_refseq)
    return state


# ──────────────────────────────────────────────────────────────────────
# Cache-write path
# ──────────────────────────────────────────────────────────────────────


def test_fetch_writes_to_cache_by_default(isolated_cache, stub_network):
    result = fetch_reference("ensembl:human")
    assert result.cache_path == isolated_cache / "homo_sapiens" / "ensembl-111"
    assert result.cache_path.is_dir()
    assert (result.cache_path / "genome" / "manifest.json").exists()
    assert (result.cache_path / "annotation" / "manifest.json").exists()
    assert (result.cache_path / "meta.toml").exists()
    assert result.output_path is None
    assert result.handle.organism == "homo_sapiens"
    assert result.skipped_cache is False


def test_meta_toml_records_full_provenance(isolated_cache, stub_network):
    result = fetch_reference("ensembl:human")
    raw = tomllib.loads((result.cache_path / "meta.toml").read_text())
    assert raw["organism"] == "homo_sapiens"
    assert raw["source"] == "ensembl"
    assert raw["release"] == "111"
    assert raw["assembly_accession"] == "GCA_000001405.29"
    assert raw["assembly_name"] == "GRCh38.p14"
    assert raw["annotation_release"] == "111"
    # Sha256s computed locally — non-empty hex strings.
    assert len(raw["sha256"]["fasta"]) == 64
    assert len(raw["sha256"]["gff3"]) == 64
    assert raw["urls"]["fasta"]["etag"] == '"stub-etag"'
    assert raw["urls"]["fasta"]["last_modified"] == "Mon, 16 May 2026 12:00:00 GMT"
    assert raw["verification"]["source_checksum_verified"] is True


def test_fetch_creates_current_symlink_and_default(isolated_cache, stub_network):
    fetch_reference("ensembl:human")
    organism_dir = isolated_cache / "homo_sapiens"
    sym = organism_dir / "current"
    assert sym.is_symlink()
    assert sym.resolve().name == "ensembl-111"
    # Auto-default on first install.
    defaults = ref_handle.read_defaults()
    assert defaults == {"homo_sapiens": "ensembl-111"}


def test_fetch_idempotent_short_circuit(isolated_cache, stub_network):
    first = fetch_reference("ensembl:human")
    assert first.skipped_cache is False
    download_calls_first = list(stub_network["download_calls"])
    second = fetch_reference("ensembl:human")
    assert second.skipped_cache is True
    # No new downloads issued.
    assert stub_network["download_calls"] == download_calls_first


def test_fetch_force_overrides_idempotency(isolated_cache, stub_network):
    fetch_reference("ensembl:human")
    download_calls_first = list(stub_network["download_calls"])
    second = fetch_reference("ensembl:human", force=True)
    assert second.skipped_cache is False
    assert len(stub_network["download_calls"]) > len(download_calls_first)


def test_fetch_source_checksum_warns_not_fails(
    isolated_cache, stub_network, capsys
):
    stub_network["checksum_passes"] = False
    result = fetch_reference("ensembl:human")
    err = capsys.readouterr().err
    assert "source checksum verification failed" in err
    # Cache write still succeeded.
    raw = tomllib.loads((result.cache_path / "meta.toml").read_text())
    assert raw["verification"]["source_checksum_verified"] is False
    # Local sha256 still recorded.
    assert len(raw["sha256"]["fasta"]) == 64


def test_fetch_no_verify_skips_source_check(
    isolated_cache, stub_network, monkeypatch
):
    # Make the verifier raise if it's called — it must NOT be.
    def _never_called(*args, **kwargs):
        raise AssertionError("verifier should be skipped")

    monkeypatch.setattr(ref_fetch, "_verify_ensembl_checksums", _never_called)
    result = fetch_reference("ensembl:human", verify_source_checksums=False)
    raw = tomllib.loads((result.cache_path / "meta.toml").read_text())
    assert raw["verification"]["source_checksum_verified"] is False


# ──────────────────────────────────────────────────────────────────────
# --output-dir behaviour
# ──────────────────────────────────────────────────────────────────────


def test_output_dir_alone_writes_both_copies(
    isolated_cache, stub_network, tmp_path
):
    out = tmp_path / "local_copy"
    result = fetch_reference("ensembl:human", output_dir=out)
    assert result.cache_path is not None
    assert result.cache_path.is_dir()
    assert result.output_path == out.resolve()
    assert (out / "genome" / "manifest.json").exists()
    assert (out / "annotation" / "manifest.json").exists()


def test_no_cache_with_output_dir(isolated_cache, stub_network, tmp_path):
    out = tmp_path / "scratch"
    result = fetch_reference("ensembl:human", output_dir=out, use_cache=False)
    assert result.cache_path is None
    assert result.output_path == out.resolve()
    assert (out / "genome" / "manifest.json").exists()
    # Cache untouched.
    assert not (isolated_cache / "homo_sapiens").exists()


def test_no_cache_without_output_dir_errors(isolated_cache, stub_network):
    with pytest.raises(ValueError, match="requires a destination"):
        fetch_reference("ensembl:human", use_cache=False)


# ──────────────────────────────────────────────────────────────────────
# Atomic rename — .partial cleanup
# ──────────────────────────────────────────────────────────────────────


def test_crash_during_write_leaves_partial_that_next_fetch_cleans(
    isolated_cache, stub_network, monkeypatch
):
    """Simulate a crash mid-write; next fetch must clean the .partial."""
    # First call: stub the very last "promote_partial" to raise so a
    # .partial/ scratch dir is left behind.
    real_promote = ref_handle.promote_partial
    calls = {"n": 0}

    def _crash_once(release_dir):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash before atomic rename")
        return real_promote(release_dir)

    monkeypatch.setattr(ref_fetch, "promote_partial", _crash_once)
    with pytest.raises(RuntimeError, match="simulated crash"):
        fetch_reference("ensembl:human")

    organism_dir = isolated_cache / "homo_sapiens"
    partial = organism_dir / "ensembl-111.partial"
    assert partial.is_dir()
    assert not (organism_dir / "ensembl-111").exists()

    # Second call (no crash this time) must remove the stale partial
    # and complete the install.
    result = fetch_reference("ensembl:human")
    assert result.cache_path is not None
    assert (organism_dir / "ensembl-111" / "meta.toml").exists()
    assert not partial.exists()


# ──────────────────────────────────────────────────────────────────────
# Sha256 + BSD sum helpers (no network)
# ──────────────────────────────────────────────────────────────────────


def test_sha256_of_matches_hashlib(tmp_path):
    import hashlib

    f = tmp_path / "foo.bin"
    f.write_bytes(b"abc123" * 10000)
    assert ref_fetch._sha256_of(f) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_bsd_sum_of_smoke(tmp_path):
    f = tmp_path / "foo.bin"
    f.write_bytes(b"x" * 2048)
    checksum, blocks = ref_fetch._bsd_sum_of(f)
    assert blocks == 2  # 2048 bytes = 2 × 1024-byte blocks
    assert 0 <= checksum <= 0xFFFF
