"""Unit tests for the reference cache resolver (handle.py).

Exercises handle parsing, cache-root precedence, the four-step resolve
chain, defaults.toml round-trip, atomic-rename, fcntl-based fetch lock,
and the current/symlink + current.txt fallback.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from constellation.sequencing.reference import handle as ref_handle
from constellation.sequencing.reference.handle import (
    Handle,
    InstalledReference,
    ReferenceNotInstalledError,
    acquire_fetch_lock,
    cache_root,
    clean_partial,
    format_size,
    list_installed,
    parse_handle,
    parse_release_slug,
    partial_dir,
    promote_partial,
    read_defaults,
    read_meta_toml,
    resolve,
    set_default,
    unset_default,
    update_current_pointer,
    write_defaults,
    write_meta_toml,
)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Set ``CONSTELLATION_REFERENCES_HOME`` to a fresh tmp dir."""
    cache = tmp_path / "refs"
    cache.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache))
    # Clear the XDG var so it doesn't interfere with the resolution chain.
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return cache


# ──────────────────────────────────────────────────────────────────────
# Handle parsing
# ──────────────────────────────────────────────────────────────────────


def test_parse_unqualified_handle():
    h = parse_handle("homo_sapiens")
    assert h.organism == "homo_sapiens"
    assert h.source is None
    assert h.release is None
    assert not h.is_qualified()
    assert str(h) == "homo_sapiens"


def test_parse_qualified_handle_ensembl():
    h = parse_handle("homo_sapiens@ensembl-111")
    assert h.organism == "homo_sapiens"
    assert h.source == "ensembl"
    assert h.release == "111"
    assert h.is_qualified()
    assert h.release_slug() == "ensembl-111"
    assert str(h) == "homo_sapiens@ensembl-111"


def test_parse_refseq_release_with_dash():
    """RefSeq composite release ``GCF_000001635.27-ar110`` keeps the
    ``-ar110`` suffix in the release field (split-on-first-dash)."""
    h = parse_handle("mus_musculus@refseq-GCF_000001635.27-ar110")
    assert h.source == "refseq"
    assert h.release == "GCF_000001635.27-ar110"
    assert h.release_slug() == "refseq-GCF_000001635.27-ar110"


def test_parse_rejects_invalid_organism():
    for bad in ("Homo_sapiens", "1_thing", "with-dash", ""):
        with pytest.raises(ValueError):
            parse_handle(bad)


def test_parse_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        parse_handle("homo_sapiens@made_up-111")


def test_parse_rejects_release_only():
    with pytest.raises(ValueError):
        parse_handle("ensembl-111")  # no organism, no @


def test_parse_release_slug_with_explicit_organism():
    h = parse_release_slug("ensembl-111", organism="homo_sapiens")
    assert h.is_qualified()
    assert h.organism == "homo_sapiens"


def test_parse_release_slug_full_handle_organism_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        parse_release_slug("mus_musculus@ensembl-110", organism="homo_sapiens")


# ──────────────────────────────────────────────────────────────────────
# Cache root resolution
# ──────────────────────────────────────────────────────────────────────


def test_cache_root_env_var_wins(tmp_path, monkeypatch):
    target = tmp_path / "via_env"
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(target))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert cache_root() == target.resolve()


def test_cache_root_xdg_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("CONSTELLATION_REFERENCES_HOME", raising=False)
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    assert cache_root() == (xdg / "constellation" / "references").resolve()


def test_cache_root_home_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("CONSTELLATION_REFERENCES_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert cache_root() == (tmp_path / "home" / ".constellation" / "references").resolve()


# ──────────────────────────────────────────────────────────────────────
# meta.toml + defaults.toml round-trip
# ──────────────────────────────────────────────────────────────────────


def _make_release_dir(cache: Path, organism: str, slug: str) -> Path:
    rel = cache / organism / slug
    rel.mkdir(parents=True)
    (rel / "genome").mkdir()
    (rel / "annotation").mkdir()
    (rel / "genome" / "manifest.json").write_text("{}")
    (rel / "annotation" / "manifest.json").write_text("{}")
    return rel


def test_meta_toml_round_trip(isolated_cache):
    rel = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    handle = parse_handle("homo_sapiens@ensembl-111")
    write_meta_toml(
        rel,
        handle=handle,
        assembly_accession="GCA_000001405.29",
        assembly_name="GRCh38.p14",
        annotation_release="111",
        constellation_version="0.0.2",
        urls={
            "fasta": {"url": "https://x/y.fa.gz", "etag": '"abc"'},
            "gff3": {"url": "https://x/y.gff3.gz"},
        },
        sha256={"fasta": "ab12", "gff3": "cd34"},
        source_checksum_verified=True,
    )
    parsed = read_meta_toml(rel)
    assert parsed["organism"] == "homo_sapiens"
    assert parsed["source"] == "ensembl"
    assert parsed["release"] == "111"
    assert parsed["assembly_accession"] == "GCA_000001405.29"
    assert parsed["assembly_name"] == "GRCh38.p14"
    assert parsed["urls"]["fasta"]["etag"] == '"abc"'
    assert parsed["sha256"]["fasta"] == "ab12"
    assert parsed["verification"]["source_checksum_verified"] is True


def test_meta_toml_escapes_quotes(isolated_cache, tmp_path):
    rel = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    handle = parse_handle("homo_sapiens@ensembl-111")
    # URL containing both a backslash and an escaped quote.
    weird_url = 'https://x/y"q\\.fa'
    write_meta_toml(
        rel,
        handle=handle,
        assembly_accession=None,
        assembly_name=None,
        annotation_release=None,
        constellation_version="0.0.2",
        urls={"fasta": {"url": weird_url}},
        sha256={},
        source_checksum_verified=False,
    )
    parsed = read_meta_toml(rel)
    assert parsed["urls"]["fasta"]["url"] == weird_url


def test_meta_toml_rejects_unqualified_handle(isolated_cache):
    rel = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    with pytest.raises(ValueError, match="qualified handle"):
        write_meta_toml(
            rel,
            handle=Handle(organism="homo_sapiens"),
            assembly_accession=None,
            assembly_name=None,
            annotation_release=None,
            constellation_version="0.0.2",
            urls={},
            sha256={},
            source_checksum_verified=False,
        )


def test_meta_toml_rejects_unknown_schema_version(isolated_cache):
    rel = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    (rel / "meta.toml").write_text('schema_version = 99\nhandle = "x"\n')
    with pytest.raises(ValueError, match="schema_version=99"):
        read_meta_toml(rel)


def test_defaults_round_trip(isolated_cache):
    write_defaults({"homo_sapiens": "ensembl-111", "mus_musculus": "refseq-GCF_X"})
    assert read_defaults() == {
        "homo_sapiens": "ensembl-111",
        "mus_musculus": "refseq-GCF_X",
    }


def test_defaults_set_unset(isolated_cache):
    set_default("homo_sapiens", "ensembl-111")
    set_default("mus_musculus", "ensembl-110")
    assert read_defaults() == {
        "homo_sapiens": "ensembl-111",
        "mus_musculus": "ensembl-110",
    }
    assert unset_default("mus_musculus") is True
    assert read_defaults() == {"homo_sapiens": "ensembl-111"}
    assert unset_default("mus_musculus") is False  # already gone


def test_defaults_rejects_unknown_schema(isolated_cache):
    (isolated_cache / "defaults.toml").write_text("schema_version = 99\n")
    with pytest.raises(ValueError, match="schema_version=99"):
        read_defaults()


# ──────────────────────────────────────────────────────────────────────
# resolve() precedence chain
# ──────────────────────────────────────────────────────────────────────


def test_resolve_qualified_handle_finds_dir(isolated_cache):
    rel = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    resolved = resolve("homo_sapiens@ensembl-111")
    assert resolved == rel.resolve()


def test_resolve_qualified_missing_release_raises(isolated_cache):
    _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-110")
    with pytest.raises(ReferenceNotInstalledError, match="ensembl-111"):
        resolve("homo_sapiens@ensembl-111")


def test_resolve_bare_organism_uses_defaults(isolated_cache):
    rel_110 = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-110")
    rel_111 = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    # No defaults pinned + no current + ambiguous → error.
    with pytest.raises(ReferenceNotInstalledError, match="ambiguous"):
        resolve("homo_sapiens")
    # Pin a default → bare resolves there.
    set_default("homo_sapiens", "ensembl-110")
    assert resolve("homo_sapiens") == rel_110.resolve()
    set_default("homo_sapiens", "ensembl-111")
    assert resolve("homo_sapiens") == rel_111.resolve()


def test_resolve_bare_organism_uses_current_symlink(isolated_cache):
    _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-110")
    rel_111 = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    update_current_pointer(rel_111.parent, "ensembl-111")
    assert resolve("homo_sapiens") == rel_111.resolve()


def test_resolve_bare_organism_single_install_shortcut(isolated_cache):
    rel = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    # No defaults, no current — but only one install exists.
    assert resolve("homo_sapiens") == rel.resolve()


def test_resolve_qualified_handle_bypasses_defaults(isolated_cache):
    rel_110 = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-110")
    rel_111 = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    set_default("homo_sapiens", "ensembl-110")
    # Default says 110, but the explicit qualified handle wins.
    assert resolve("homo_sapiens@ensembl-111") == rel_111.resolve()
    assert resolve("homo_sapiens@ensembl-110") == rel_110.resolve()


def test_resolve_missing_organism_raises(isolated_cache):
    with pytest.raises(ReferenceNotInstalledError, match="no cached reference"):
        resolve("nonexistent_species")


# ──────────────────────────────────────────────────────────────────────
# current/ pointer + WSL fallback
# ──────────────────────────────────────────────────────────────────────


def test_update_current_writes_symlink(isolated_cache):
    rel = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    kind = update_current_pointer(rel.parent, "ensembl-111")
    assert kind == "symlink"
    sym = rel.parent / "current"
    assert sym.is_symlink()
    assert (rel.parent / "current").resolve() == rel.resolve()


def test_update_current_replaces_existing_atomically(isolated_cache):
    rel_a = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-110")
    rel_b = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    update_current_pointer(rel_a.parent, "ensembl-110")
    update_current_pointer(rel_b.parent, "ensembl-111")
    sym = rel_b.parent / "current"
    assert sym.resolve() == rel_b.resolve()


def test_current_textfile_fallback_when_symlink_fails(isolated_cache, monkeypatch):
    rel = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")

    def _explode(*args, **kwargs):
        raise OSError(13, "Operation not permitted")

    monkeypatch.setattr(os, "symlink", _explode)
    kind = update_current_pointer(rel.parent, "ensembl-111")
    assert kind == "textfile"
    txt = rel.parent / "current.txt"
    assert txt.exists()
    assert txt.read_text().strip() == "ensembl-111"
    # The resolver picks up the textfile form.
    assert resolve("homo_sapiens") == rel.resolve()


def test_unrelated_oserror_during_symlink_propagates(isolated_cache, monkeypatch):
    rel = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")

    def _explode_io(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "symlink", _explode_io)
    with pytest.raises(OSError, match="No space left"):
        update_current_pointer(rel.parent, "ensembl-111")


# ──────────────────────────────────────────────────────────────────────
# Atomic .partial → release rename
# ──────────────────────────────────────────────────────────────────────


def test_partial_promote_renames(isolated_cache):
    organism_dir = isolated_cache / "homo_sapiens"
    organism_dir.mkdir()
    rel = organism_dir / "ensembl-111"
    stage = partial_dir(rel)
    assert stage.name == "ensembl-111.partial"
    stage.mkdir(parents=True)
    (stage / "marker").write_text("ok")
    promote_partial(rel)
    assert rel.is_dir()
    assert not stage.exists()
    assert (rel / "marker").read_text() == "ok"


def test_partial_promote_refuses_overwrite(isolated_cache):
    organism_dir = isolated_cache / "homo_sapiens"
    organism_dir.mkdir()
    rel = organism_dir / "ensembl-111"
    rel.mkdir()
    stage = partial_dir(rel)
    stage.mkdir()
    with pytest.raises(FileExistsError, match="release dir already exists"):
        promote_partial(rel)


def test_clean_partial_removes_scratch(isolated_cache):
    organism_dir = isolated_cache / "homo_sapiens"
    organism_dir.mkdir()
    rel = organism_dir / "ensembl-111"
    stage = partial_dir(rel)
    stage.mkdir(parents=True)
    (stage / "junk").write_text("x")
    clean_partial(rel)
    assert not stage.exists()


# ──────────────────────────────────────────────────────────────────────
# fcntl fetch lock
# ──────────────────────────────────────────────────────────────────────


def test_acquire_fetch_lock_serializes_workers(isolated_cache):
    """Second concurrent attempt to acquire blocks until the first exits."""
    import multiprocessing as mp
    import time

    handle = parse_handle("homo_sapiens@ensembl-111")
    organism_dir = isolated_cache / "homo_sapiens"
    organism_dir.mkdir()

    hold_s = 0.6
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p1 = ctx.Process(
        target=_lock_worker,
        args=(str(handle), str(isolated_cache), hold_s, q),
    )
    p2 = ctx.Process(
        target=_lock_worker,
        args=(str(handle), str(isolated_cache), 0.0, q),
    )
    p1.start()
    time.sleep(0.2)  # let p1 grab the lock first
    p2.start()
    p1.join(timeout=10)
    p2.join(timeout=10)
    events = [q.get_nowait() for _ in range(4)]
    # Each process measures elapsed-since-its-own-start, so the gap is
    # hold_s minus the head-start. Allow 50% scheduling slack — catches
    # a broken lock without flaking.
    acquired_times = sorted(ev[1] for ev in events if ev[0] == "acquired")
    assert len(acquired_times) == 2
    assert acquired_times[1] - acquired_times[0] >= hold_s * 0.5


def _lock_worker(handle_str, root_str, hold_s, q):
    """Module-scope worker (spawn-pickle-able) used by the lock test."""
    import os as _os
    import time

    _os.environ["CONSTELLATION_REFERENCES_HOME"] = root_str
    from constellation.sequencing.reference.handle import (
        acquire_fetch_lock,
        parse_handle,
    )

    t0 = time.time()
    with acquire_fetch_lock(parse_handle(handle_str)):
        q.put(("acquired", time.time() - t0))
        time.sleep(hold_s)
    q.put(("released", time.time() - t0))


def test_acquire_fetch_lock_requires_qualified_handle(isolated_cache):
    with pytest.raises(ValueError, match="qualified handle"):
        with acquire_fetch_lock(Handle(organism="homo_sapiens")):
            pass


# ──────────────────────────────────────────────────────────────────────
# list_installed + InstalledReference
# ──────────────────────────────────────────────────────────────────────


def test_list_installed_walks_cache(isolated_cache):
    rel_a = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-110")
    rel_b = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    rel_c = _make_release_dir(isolated_cache, "mus_musculus", "refseq-GCF_X-ar110")
    h_a = parse_handle("homo_sapiens@ensembl-110")
    h_b = parse_handle("homo_sapiens@ensembl-111")
    h_c = parse_handle("mus_musculus@refseq-GCF_X-ar110")
    for rel, h in [(rel_a, h_a), (rel_b, h_b), (rel_c, h_c)]:
        write_meta_toml(
            rel,
            handle=h,
            assembly_accession=None,
            assembly_name=None,
            annotation_release=None,
            constellation_version="0.0.2",
            urls={},
            sha256={},
            source_checksum_verified=False,
        )

    entries = list_installed()
    handles = sorted(e.handle for e in entries)
    assert handles == [
        "homo_sapiens@ensembl-110",
        "homo_sapiens@ensembl-111",
        "mus_musculus@refseq-GCF_X-ar110",
    ]
    # release_slug always matches the on-disk directory name.
    by_handle = {e.handle: e for e in entries}
    assert by_handle["mus_musculus@refseq-GCF_X-ar110"].release_slug == "refseq-GCF_X-ar110"
    assert by_handle["homo_sapiens@ensembl-111"].path == rel_b.resolve()


def test_list_installed_ignores_scratch_dirs(isolated_cache):
    _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    # Drop a .partial and a .lock — these must not appear.
    (isolated_cache / "homo_sapiens" / "ensembl-112.partial").mkdir()
    (isolated_cache / "homo_sapiens" / "ensembl-110.lock").touch()
    entries = list_installed()
    assert {e.release_slug for e in entries} == {"ensembl-111"}


def test_installed_reference_is_default(isolated_cache):
    rel = _make_release_dir(isolated_cache, "homo_sapiens", "ensembl-111")
    handle = parse_handle("homo_sapiens@ensembl-111")
    write_meta_toml(
        rel,
        handle=handle,
        assembly_accession=None,
        assembly_name=None,
        annotation_release=None,
        constellation_version="0.0.2",
        urls={},
        sha256={},
        source_checksum_verified=False,
    )
    set_default("homo_sapiens", "ensembl-111")
    entry = list_installed()[0]
    assert entry.is_default({"homo_sapiens": "ensembl-111"})
    assert not entry.is_default({"homo_sapiens": "ensembl-110"})
    assert not entry.is_default({})


# ──────────────────────────────────────────────────────────────────────
# format_size
# ──────────────────────────────────────────────────────────────────────


def test_format_size():
    assert format_size(0) == "0 B"
    assert format_size(512) == "512 B"
    assert format_size(2048) == "2.0 KB"
    assert format_size(5 * 1024 * 1024) == "5.0 MB"
    assert format_size(3 * 1024 ** 3) == "3.0 GB"
