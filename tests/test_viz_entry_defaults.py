"""Entry-discovery helpers for the build/install defaults.

After PR 2 the build and install commands default to "every known
entry" rather than just ``genome``. The two helpers under test are
the source of truth for that defaulting:

- ``constellation.viz.frontend.build.known_entries`` — discovers
  entries from ``frontend/index.*.html`` shells (used by the build CLI
  default and by the URL-fetch install path as a fallback).
- ``constellation.viz.install.list_tarball_entries`` — scans a
  release tarball for ``static/<entry>/`` directories (used by the
  ``--from <tarball>`` install path when ``--entry`` is omitted).
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from constellation.viz.frontend.build import known_entries, pack_bundle
from constellation.viz.install import InstallError, list_tarball_entries


def _write_fake_bundle(static_root: Path, entry: str) -> None:
    """Mirror the helper from tests/test_viz_install.py with a minimum
    file set so pack_bundle has something to archive."""
    entry_dir = static_root / entry
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / f"index.{entry}.html").write_text("<!doctype html>\n")
    (entry_dir / "assets").mkdir(exist_ok=True)
    (entry_dir / "assets" / f"main_{entry}-fakehash.js").write_text(
        "console.log('fixture');\n"
    )


def test_known_entries_discovers_shells_from_real_source_tree() -> None:
    """The committed frontend ships index.genome.html + index.dashboard.html;
    both must turn up in the discovery output. New entries (e.g. a future
    spectrum viewer) get picked up automatically when their shell HTML lands."""
    entries = known_entries()
    assert "genome" in entries
    assert "dashboard" in entries
    # Sorted for deterministic build invocations
    assert entries == sorted(entries)


def test_list_tarball_entries_single_entry(tmp_path: Path) -> None:
    """Single-entry tarballs (PR 1.5-era artifacts) return a one-element list."""
    static_root = tmp_path / "static"
    _write_fake_bundle(static_root, entry="genome")
    tarball, _ = pack_bundle(
        entry="genome",
        static_root=static_root,
        dist_dir=tmp_path / "dist",
        version="1.0.0",
    )
    assert list_tarball_entries(tarball) == ["genome"]


def test_list_tarball_entries_multi_entry(tmp_path: Path) -> None:
    """Multi-entry tarballs (PR 2 release artifacts) return every entry,
    sorted. Drives the ``install-frontend --from <tarball>`` default that
    dispatches one install per discovered entry."""
    static_root = tmp_path / "static"
    _write_fake_bundle(static_root, entry="genome")
    _write_fake_bundle(static_root, entry="dashboard")
    tarball, _ = pack_bundle(
        entry=["genome", "dashboard"],
        static_root=static_root,
        dist_dir=tmp_path / "dist",
        version="1.0.0",
    )
    assert list_tarball_entries(tarball) == ["dashboard", "genome"]


def test_list_tarball_entries_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InstallError, match="tarball not found"):
        list_tarball_entries(tmp_path / "does-not-exist.tar.gz")


def test_list_tarball_entries_invalid_layout_raises(tmp_path: Path) -> None:
    """A tarball with no ``static/<entry>/`` directories is rejected with
    an actionable message rather than silently returning an empty list
    (which would cause the install handler to be a no-op)."""
    bad = tmp_path / "bad.tar.gz"
    # Build a tarball whose members don't match the expected layout.
    with tarfile.open(bad, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="not-a-bundle/README")
        payload = b"hello\n"
        info.size = len(payload)
        import io

        tar.addfile(info, fileobj=io.BytesIO(payload))
    with pytest.raises(InstallError, match="no static/<entry>/"):
        list_tarball_entries(bad)
