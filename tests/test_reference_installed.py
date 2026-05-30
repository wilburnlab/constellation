"""Tests for ``constellation.sequencing.reference.installed.Reference``.

Focused on the ``require(...)`` / ``_fetch_hint(...)`` machinery: the
error messages have to direct users to the *right* fix when an artifact
is missing, not a hardcoded fallback. RefSeq / Ensembl / Ensembl Genomes
all bundle a protein FASTA alongside the genome, so a missing-proteome
error on those sources should suggest a ``--force`` re-fetch from the
same source, not send the user to UniProt as a first stop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.sequencing.reference import Reference
from constellation.sequencing.reference.handle import (
    Handle,
    ReferenceNotInstalledError,
    write_meta_toml,
)


# ──────────────────────────────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────────────────────────────


def _install_reference(
    root: Path,
    *,
    organism: str,
    source: str,
    release: str,
    has_genome: bool = True,
    has_annotation: bool = True,
    has_proteome: bool = True,
    has_cdna: bool = False,
) -> Reference:
    """Lay down a minimal portal-layout reference directory with the
    requested ``[contents]`` flags and return the opened ``Reference``.
    """
    release_dir = root / organism / f"{source}-{release}"
    release_dir.mkdir(parents=True)
    if has_genome:
        (release_dir / "genome").mkdir()
        (release_dir / "genome" / "manifest.json").write_text("{}")
    if has_annotation:
        (release_dir / "annotation").mkdir()
        (release_dir / "annotation" / "features.parquet").write_text("stub")
        (release_dir / "annotation" / "manifest.json").write_text("{}")
    if has_proteome:
        (release_dir / "protein.faa").write_text(">stub\nMAGCKL\n")
    if has_cdna:
        (release_dir / "cdna.fna").write_text(">stub\nATG\n")
    write_meta_toml(
        release_dir,
        handle=Handle(organism=organism, source=source, release=release),
        assembly_accession=None,
        assembly_name=None,
        annotation_release=None,
        constellation_version="test",
        urls={},
        sha256={},
        source_checksum_verified=False,
        has_genome=has_genome,
        has_annotation=has_annotation,
        has_proteome=has_proteome,
        has_cdna=has_cdna,
    )
    return Reference.open(f"{organism}@{source}-{release}", root=root)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "refs"
    root.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(root))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return root


# ──────────────────────────────────────────────────────────────────────
# require() returns early when artifacts are present
# ──────────────────────────────────────────────────────────────────────


def test_require_passes_when_artifacts_present(cache_root) -> None:
    ref = _install_reference(
        cache_root, organism="homo_sapiens", source="refseq", release="GCF_test",
    )
    # Should not raise.
    ref.require(genome=True, annotation=True, proteome=True)


# ──────────────────────────────────────────────────────────────────────
# Missing proteome hints — the bug-fix surface
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("source", ["refseq", "ensembl", "ensembl_genomes"])
def test_missing_proteome_on_bundled_source_suggests_force_refetch(
    cache_root, source
) -> None:
    """When the reference's source bundles a protein FASTA (RefSeq /
    Ensembl / Ensembl Genomes), the missing-proteome hint should point
    at a ``--force`` re-fetch from the SAME source — not UniProt.
    UniProt should be mentioned only as the fallback."""
    ref = _install_reference(
        cache_root,
        organism="homo_sapiens",
        source=source,
        release="111" if source != "refseq" else "GCF_test",
        has_proteome=False,
    )
    with pytest.raises(ReferenceNotInstalledError) as excinfo:
        ref.require(proteome=True)
    msg = str(excinfo.value)
    # The primary hint pins the same source and adds --force.
    assert f"--source {source} --force" in msg
    # The bundled-source explanation appears.
    assert "bundles a protein FASTA" in msg
    # UniProt is mentioned as a fallback for the "assembly genuinely
    # has no protein FASTA" case.
    assert "--source uniprot" in msg


def test_missing_proteome_on_genbank_falls_back_to_uniprot(
    cache_root,
) -> None:
    """GenBank assemblies often don't ship a protein FASTA at the
    standard path. When the source itself doesn't bundle one, the
    hint goes straight to UniProt."""
    ref = _install_reference(
        cache_root,
        organism="haliotis_rufescens",
        source="genbank",
        release="GCA_023055435.1",
        has_proteome=False,
    )
    with pytest.raises(ReferenceNotInstalledError) as excinfo:
        ref.require(proteome=True)
    msg = str(excinfo.value)
    assert "--source uniprot" in msg
    assert "does not bundle a protein FASTA" in msg
    # Should NOT suggest --force re-fetching from genbank.
    assert "--source genbank --force" not in msg


def test_missing_proteome_on_swissprot_uses_dedicated_spec(
    cache_root,
) -> None:
    """SwissProt has its own bareword spec, not the generic UniProt
    per-species path."""
    ref = _install_reference(
        cache_root,
        organism="swissprot",
        source="uniprot",
        release="2026_02",
        has_genome=False,
        has_annotation=False,
        has_proteome=False,
    )
    with pytest.raises(ReferenceNotInstalledError) as excinfo:
        ref.require(proteome=True)
    msg = str(excinfo.value)
    assert "uniprot:swissprot" in msg
    assert "<species-or-taxid>" not in msg


# ──────────────────────────────────────────────────────────────────────
# Genome / annotation / cDNA hints — pin the reference's source
# ──────────────────────────────────────────────────────────────────────


def test_missing_genome_hints_at_same_source(cache_root) -> None:
    """A missing genome hint pins the same source the reference came from."""
    ref = _install_reference(
        cache_root,
        organism="mus_musculus",
        source="ensembl",
        release="111",
        has_genome=False,
    )
    with pytest.raises(ReferenceNotInstalledError) as excinfo:
        ref.require(genome=True)
    msg = str(excinfo.value)
    assert "--source ensembl" in msg
    # No --force on genome/annotation hints (the user may genuinely
    # have a partially-installed reference; suggest re-fetch but don't
    # imply the file is "already there but broken").
    assert "--force" not in msg


def test_missing_annotation_hints_at_same_source(cache_root) -> None:
    ref = _install_reference(
        cache_root,
        organism="homo_sapiens",
        source="refseq",
        release="GCF_test",
        has_annotation=False,
    )
    with pytest.raises(ReferenceNotInstalledError) as excinfo:
        ref.require(annotation=True)
    msg = str(excinfo.value)
    assert "--source refseq" in msg


def test_missing_cdna_hints_at_same_source(cache_root) -> None:
    ref = _install_reference(
        cache_root,
        organism="homo_sapiens",
        source="refseq",
        release="GCF_test",
        has_cdna=False,
    )
    with pytest.raises(ReferenceNotInstalledError) as excinfo:
        ref.require(cdna=True)
    msg = str(excinfo.value)
    assert "--source refseq" in msg


# ──────────────────────────────────────────────────────────────────────
# Compound: multiple artifacts missing at once
# ──────────────────────────────────────────────────────────────────────


def test_multiple_missing_artifacts_listed_together(cache_root) -> None:
    """All requested missing artifacts surface in one error, each with
    its own hint."""
    ref = _install_reference(
        cache_root,
        organism="homo_sapiens",
        source="refseq",
        release="GCF_test",
        has_annotation=False,
        has_proteome=False,
    )
    with pytest.raises(ReferenceNotInstalledError) as excinfo:
        ref.require(annotation=True, proteome=True)
    msg = str(excinfo.value)
    assert "annotation" in msg
    assert "proteome" in msg
    # Both hints present.
    assert "--source refseq" in msg
    assert "--force" in msg  # proteome hint flags --force
