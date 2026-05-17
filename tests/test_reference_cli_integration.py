"""Integration tests for the reference CLI verbs + doctor cache row.

Covers:
    - ``constellation doctor`` reports the cache row (empty / populated /
      warning states).
    - ``_resolve_reference_argument`` accepts both paths and handles
      (the seam that ``transcriptome align --reference`` rides on).
    - ``reference link`` falls back to a session.toml stub when symlinks
      fail (the WSL/Windows case).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from constellation.cli.__main__ import (
    _cache_integrity_warnings,
    _cmd_reference_link,
    _doctor_reference_cache_row,
    _resolve_reference_argument,
    main,
)
from constellation.sequencing.reference import handle as ref_handle
from constellation.sequencing.reference.handle import (
    ReferenceNotInstalledError,
    parse_handle,
    update_current_pointer,
    write_meta_toml,
)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    cache = tmp_path / "refs"
    cache.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return cache


def _populate(cache: Path, organism: str, slug: str) -> Path:
    rel = cache / organism / slug
    rel.mkdir(parents=True)
    (rel / "genome").mkdir()
    (rel / "annotation").mkdir()
    (rel / "genome" / "manifest.json").write_text("{}")
    (rel / "annotation" / "manifest.json").write_text("{}")
    handle = parse_handle(f"{organism}@{slug}")
    write_meta_toml(
        rel,
        handle=handle,
        assembly_accession=None,
        assembly_name=None,
        annotation_release=None,
        constellation_version="0.0.2",
        urls={"fasta": {"url": "https://test/foo.fa.gz"}},
        sha256={"fasta": "deadbeef" * 8},
        source_checksum_verified=False,
    )
    return rel


# ──────────────────────────────────────────────────────────────────────
# doctor cache row
# ──────────────────────────────────────────────────────────────────────


def test_doctor_cache_row_empty_reports_ok(isolated_cache):
    name, status, refs, location = _doctor_reference_cache_row()
    assert name == "reference cache"
    assert status == "ok"
    assert refs == "0 refs"
    assert "no installs" in location or "not yet populated" in location


def test_doctor_cache_row_populated(isolated_cache):
    _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    _populate(isolated_cache, "mus_musculus", "ensembl-111")
    name, status, refs, location = _doctor_reference_cache_row()
    assert status == "ok"
    assert refs == "2 refs"
    assert str(isolated_cache) in location


def test_doctor_cache_row_warns_on_leftover_partial(isolated_cache):
    _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    # Leftover from a crashed fetch:
    (isolated_cache / "homo_sapiens" / "ensembl-112.partial").mkdir()
    name, status, refs, location = _doctor_reference_cache_row()
    assert status == "warn"
    assert "leftover partial fetch" in location


def test_doctor_cache_row_warns_on_dangling_symlink(isolated_cache):
    _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    sym = isolated_cache / "homo_sapiens" / "current"
    if sym.is_symlink() or sym.exists():
        sym.unlink()
    os.symlink("nonexistent-release", sym)
    warnings = _cache_integrity_warnings(isolated_cache)
    assert any("dangling current symlink" in w for w in warnings)


def test_doctor_command_runs_with_populated_cache(isolated_cache, capsys):
    _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "reference cache" in out
    assert "1 refs" in out
    # Empty cache state is "ok" so doctor's exit code shouldn't be flipped
    # by the cache row alone — the existing third-party rows still gate it.
    assert rc in (0, 1)


# ──────────────────────────────────────────────────────────────────────
# _resolve_reference_argument seam
# ──────────────────────────────────────────────────────────────────────


def test_resolve_reference_argument_accepts_path(isolated_cache, tmp_path):
    plain_dir = tmp_path / "external_ref"
    (plain_dir / "genome").mkdir(parents=True)
    (plain_dir / "annotation").mkdir(parents=True)
    resolved, source = _resolve_reference_argument(str(plain_dir))
    assert Path(resolved) == plain_dir
    assert source == "path"


def test_resolve_reference_argument_accepts_handle(isolated_cache):
    rel = _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    resolved, source = _resolve_reference_argument("homo_sapiens@ensembl-111")
    assert Path(resolved) == rel.resolve()
    assert source == "handle"


def test_resolve_reference_argument_handle_uses_defaults(isolated_cache):
    rel = _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    ref_handle.set_default("homo_sapiens", "ensembl-111")
    resolved, source = _resolve_reference_argument("homo_sapiens")
    assert Path(resolved) == rel.resolve()
    assert source == "handle"


def test_resolve_reference_argument_missing_handle_errors(isolated_cache):
    with pytest.raises(ReferenceNotInstalledError):
        _resolve_reference_argument("nonexistent_species")


# ──────────────────────────────────────────────────────────────────────
# reference link + WSL fallback
# ──────────────────────────────────────────────────────────────────────


def test_reference_link_creates_symlinks(isolated_cache, tmp_path):
    rel = _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    target = tmp_path / "analysis"
    rc = main(
        [
            "reference",
            "link",
            "homo_sapiens@ensembl-111",
            "--into",
            str(target),
        ]
    )
    assert rc == 0
    assert (target / "genome").is_symlink()
    assert (target / "annotation").is_symlink()
    assert (target / "genome").resolve() == (rel / "genome").resolve()


def test_reference_link_force_overwrites(isolated_cache, tmp_path):
    rel = _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    target = tmp_path / "analysis"
    target.mkdir()
    (target / "genome").mkdir()  # pre-existing directory blocks symlink
    rc = main(
        [
            "reference",
            "link",
            "homo_sapiens@ensembl-111",
            "--into",
            str(target),
            "--force",
        ]
    )
    assert rc == 0
    assert (target / "genome").is_symlink()


def test_reference_link_refuses_without_force(isolated_cache, tmp_path, capsys):
    _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    target = tmp_path / "analysis"
    target.mkdir()
    (target / "genome").mkdir()
    rc = main(
        [
            "reference",
            "link",
            "homo_sapiens@ensembl-111",
            "--into",
            str(target),
        ]
    )
    assert rc != 0
    assert "already exists" in capsys.readouterr().err


def test_reference_link_wsl_fallback_writes_session_toml(
    isolated_cache, tmp_path, monkeypatch, capsys
):
    """When os.symlink raises, the link command writes a session.toml
    stub that resolves the handle via the cache."""
    _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    target = tmp_path / "analysis"

    def _no_symlink(*args, **kwargs):
        raise OSError(13, "Operation not permitted")

    monkeypatch.setattr("os.symlink", _no_symlink)
    rc = main(
        [
            "reference",
            "link",
            "homo_sapiens@ensembl-111",
            "--into",
            str(target),
        ]
    )
    assert rc == 0
    session_toml = target / "session.toml"
    assert session_toml.exists()
    content = session_toml.read_text()
    assert 'handle = "homo_sapiens@ensembl-111"' in content
    assert "schema_version = 1" in content
    # The fallback warning is on stderr.
    assert "symlink creation failed" in capsys.readouterr().err


# ──────────────────────────────────────────────────────────────────────
# Bare 'reference default' tour
# ──────────────────────────────────────────────────────────────────────


def test_reference_default_pin_and_inspect(isolated_cache, capsys):
    _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    rc = main(["reference", "default", "homo_sapiens", "ensembl-111"])
    assert rc == 0
    assert "pinned default" in capsys.readouterr().out

    rc = main(["reference", "default"])
    out = capsys.readouterr().out
    assert "homo_sapiens" in out
    assert "ensembl-111" in out


def test_reference_default_unset(isolated_cache, capsys):
    _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    ref_handle.set_default("homo_sapiens", "ensembl-111")
    rc = main(["reference", "default", "homo_sapiens", "--unset"])
    assert rc == 0
    assert ref_handle.read_defaults() == {}


def test_reference_default_use_current(isolated_cache, capsys):
    rel = _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    update_current_pointer(rel.parent, "ensembl-111")
    rc = main(["reference", "default", "homo_sapiens", "--use-current"])
    assert rc == 0
    assert ref_handle.read_defaults() == {"homo_sapiens": "ensembl-111"}


def test_reference_list_and_where(isolated_cache, capsys):
    _populate(isolated_cache, "homo_sapiens", "ensembl-111")
    ref_handle.set_default("homo_sapiens", "ensembl-111")

    rc = main(["reference", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "homo_sapiens@ensembl-111" in out
    # '*' marks the default.
    assert "*" in out

    rc = main(["reference", "where", "homo_sapiens"])
    out = capsys.readouterr().out
    assert "homo_sapiens" in out
    assert "ensembl-111" in out


def test_reference_where_missing_handle_exits_nonzero(isolated_cache, capsys):
    rc = main(["reference", "where", "nonexistent"])
    assert rc == 1
    assert "no cached reference" in capsys.readouterr().err
