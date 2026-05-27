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


PROTEIN_FIXTURE = b""">NP_000001.1 protein gene1
MKLVTLALCAVSLA
>NP_000002.1 protein gene2
ATCDEFGHIK
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
        # When non-None, _ResolvedSpec carries this string as
        # ``protein_url`` and _fake_download_to serves PROTEIN_FIXTURE
        # for it. Set ``protein_returns_404`` to True to simulate a
        # missing-protein-FASTA assembly via a raised HTTPError.
        "protein_url": None,
        "protein_returns_404": False,
    }

    def _fake_resolve_spec(spec, *, release=None, source=None):
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
            protein_url=state["protein_url"],
        )

    def _fake_download_to(url, dest, *, timeout=60):
        state["download_calls"].append(url)
        is_protein = (
            state.get("protein_url") is not None and url == state["protein_url"]
        )
        if is_protein and state.get("protein_returns_404"):
            import urllib.error

            raise urllib.error.HTTPError(
                url, 404, "Not Found", hdrs=None, fp=None
            )
        if is_protein:
            payload = PROTEIN_FIXTURE
        elif "gff" in url:
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
# Protein FASTA fetching (Phase 2.5 — RefSeq protein.faa materialisation)
# ──────────────────────────────────────────────────────────────────────


def test_fetch_omits_protein_when_url_absent(isolated_cache, stub_network):
    """Backward-compat: when the resolver returns no protein_url, the
    fetch result has no protein FASTA and the cache contains no
    `protein.faa`. Mirrors today's Ensembl / unannotated-GenBank
    behaviour."""
    stub_network["protein_url"] = None
    result = fetch_reference("ensembl:human")

    assert result.protein_fasta_path is None
    assert result.cache_path is not None
    assert not (result.cache_path / "protein.faa").exists()
    assert "protein" not in result.sources
    # No protein URL was hit.
    assert not any(
        "protein" in url for url in stub_network["download_calls"]
    )


def test_fetch_writes_protein_fasta_when_url_present(isolated_cache, stub_network):
    """When the resolver returns a protein_url, the protein FASTA is
    downloaded, gunzipped, copied into the cache, recorded in meta.toml
    (urls.protein + sha256.protein), and exposed via
    FetchResult.protein_fasta_path."""
    stub_network["protein_url"] = (
        "https://test/foo_protein.faa.gz"
    )
    result = fetch_reference("refseq:GCF_000001405.40")

    assert result.protein_fasta_path is not None
    assert result.protein_fasta_path.is_file()
    assert result.protein_fasta_path.name == "protein.faa"
    assert result.protein_fasta_path == result.cache_path / "protein.faa"
    # Bytes match the fixture (gunzipped).
    assert result.protein_fasta_path.read_bytes() == PROTEIN_FIXTURE
    assert result.sources["protein"] == "https://test/foo_protein.faa.gz"

    # meta.toml provenance:
    meta = tomllib.loads((result.cache_path / "meta.toml").read_text())
    assert "protein" in meta["urls"]
    assert meta["urls"]["protein"]["url"] == "https://test/foo_protein.faa.gz"
    assert "protein" in meta["sha256"]
    # sha256 is a 64-hex digest.
    assert len(meta["sha256"]["protein"]) == 64
    assert set(meta["sha256"]["protein"]).issubset("0123456789abcdef")


def test_fetch_tolerates_protein_fasta_404(isolated_cache, stub_network, capsys):
    """When the protein URL 404s, the genome + annotation fetch still
    succeeds; protein_fasta_path is None and a warning is printed to
    stderr. Covers the GenBank-only assembly without a published
    protein FASTA path."""
    stub_network["protein_url"] = "https://test/foo_protein.faa.gz"
    stub_network["protein_returns_404"] = True
    result = fetch_reference("refseq:GCF_000999999.1")

    assert result.protein_fasta_path is None
    assert result.cache_path is not None
    assert not (result.cache_path / "protein.faa").exists()
    # Genome + annotation still materialised.
    assert result.genome is not None
    assert result.annotation is not None

    err = capsys.readouterr().err
    assert "protein FASTA fetch failed" in err


def test_fetch_cache_hit_exposes_protein_fasta_path(isolated_cache, stub_network):
    """Second fetch hits the cache; protein_fasta_path is still
    populated from the on-disk protein.faa."""
    stub_network["protein_url"] = "https://test/foo_protein.faa.gz"
    first = fetch_reference("refseq:GCF_000001405.40")
    assert first.protein_fasta_path is not None
    assert not first.skipped_cache

    # Second call — no force, cache should short-circuit.
    second = fetch_reference("refseq:GCF_000001405.40")
    assert second.skipped_cache
    assert second.protein_fasta_path == first.protein_fasta_path
    assert second.protein_fasta_path.is_file()


def test_fetch_cache_hit_without_protein_fasta_returns_none(
    isolated_cache, stub_network
):
    """Backward-compat: when an existing cache slot was written before
    protein FASTA fetching landed (no protein.faa on disk),
    FetchResult.protein_fasta_path is None on cache-hit even if the
    resolver now reports a protein_url. The cache stays valid; users
    can re-fetch with force=True to supplement it."""
    # First fetch with no protein URL → no protein.faa in cache.
    stub_network["protein_url"] = None
    first = fetch_reference("refseq:GCF_000001405.40")
    assert first.protein_fasta_path is None
    assert not first.skipped_cache

    # Second call now claims a protein URL — but the cache is already
    # "complete" by the genome+annotation criterion, so we short-circuit.
    stub_network["protein_url"] = "https://test/foo_protein.faa.gz"
    second = fetch_reference("refseq:GCF_000001405.40")
    assert second.skipped_cache
    assert second.protein_fasta_path is None
    assert second.sources["protein"] == "https://test/foo_protein.faa.gz"


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
