"""Frontend bundle install + pack — PR 1.5 unit tests.

Exercises ``constellation.viz.install.install_frontend_from_tarball``
end-to-end against tarballs produced by ``constellation.viz.frontend
.build.pack_bundle``. Both modules are stdlib-only at the boundaries
we care about, so these tests run without the ``[viz]`` extras
(no fastapi / uvicorn / datashader / pnpm / npm needed).
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from constellation.viz.frontend.build import pack_bundle
from constellation.viz.install import (
    InstallError,
    install_frontend_from_tarball,
    read_bundle_metadata,
)


# ----------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------


def _write_fake_bundle(static_root: Path, entry: str = "genome") -> Path:
    """Write a minimal "built bundle" tree under ``static_root/<entry>/``.

    Two files are enough to verify directory-shape extraction + asset-
    folder nesting; the real Vite output is many more files but the
    plumbing is the same.
    """
    entry_dir = static_root / entry
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "index.genome.html").write_text(
        "<!doctype html><html><head><title>fixture</title></head>"
        "<body><div id=\"app\"></div></body></html>\n",
        encoding="utf-8",
    )
    (entry_dir / "assets").mkdir(exist_ok=True)
    (entry_dir / "assets" / "main_genome-fakehash.js").write_text(
        "console.log('fixture bundle');\n", encoding="utf-8"
    )
    (entry_dir / "assets" / "main_genome-fakehash.css").write_text(
        "body { color: red; }\n", encoding="utf-8"
    )
    return entry_dir


def _make_packed_bundle(
    tmp_path: Path, entry: str = "genome", version: str = "0.0.0"
) -> tuple[Path, Path, Path]:
    """Build a real packed bundle in tmp_path: returns (tarball, sidecar,
    dest_root) where dest_root is intended for use in install_…
    """
    static_root = tmp_path / "src_static"
    _write_fake_bundle(static_root, entry=entry)
    dist_dir = tmp_path / "dist"
    tarball, sidecar = pack_bundle(
        entry=entry,
        static_root=static_root,
        dist_dir=dist_dir,
        version=version,
    )
    dest_root = tmp_path / "install_root"
    return tarball, sidecar, dest_root


# ----------------------------------------------------------------------
# pack_bundle — producer-side
# ----------------------------------------------------------------------


def test_pack_bundle_produces_tarball_and_sidecar(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    _write_fake_bundle(static_root)
    tarball, sidecar = pack_bundle(
        entry="genome",
        static_root=static_root,
        dist_dir=tmp_path / "dist",
        version="1.2.3",
    )
    assert tarball.is_file()
    assert sidecar.is_file()
    assert tarball.name == "constellation-viz-frontend-1.2.3.tar.gz"
    assert sidecar.name == f"{tarball.name}.sha256"

    # Sidecar format: '<hex>  <filename>\n'
    line = sidecar.read_text(encoding="utf-8").strip()
    digest, filename = line.split(maxsplit=1)
    assert len(digest) == 64
    assert filename == tarball.name

    actual = hashlib.sha256(tarball.read_bytes()).hexdigest()
    assert digest == actual


def test_pack_bundle_layout_matches_spec(tmp_path: Path) -> None:
    static_root = tmp_path / "static"
    _write_fake_bundle(static_root)
    tarball, _ = pack_bundle(
        entry="genome",
        static_root=static_root,
        dist_dir=tmp_path / "dist",
        version="0.9.0",
    )
    with tarfile.open(tarball, "r:gz") as tar:
        names = sorted(tar.getnames())
    prefix = "constellation-viz-frontend-0.9.0"
    assert f"{prefix}/bundle.json" in names
    assert f"{prefix}/static/genome/index.genome.html" in names
    assert any(
        n.startswith(f"{prefix}/static/genome/assets/") for n in names
    )

    # bundle.json carries the expected metadata fields
    with tarfile.open(tarball, "r:gz") as tar:
        f = tar.extractfile(f"{prefix}/bundle.json")
        assert f is not None
        meta = json.loads(f.read().decode("utf-8"))
    assert meta["constellation_version"] == "0.9.0"
    assert meta["entry"] == "genome"
    assert "built_at" in meta


def test_pack_bundle_default_dist_dir_resolves_to_repo_root() -> None:
    """Guard the parents[N] arithmetic in build.py.

    `pack_bundle()` defaults `dist_dir` to ``<repo_root>/dist/`` via
    ``_DIST_DIR = _FRONTEND_DIR.parents[2] / "dist"``. Every other test
    in this file overrides ``dist_dir`` explicitly, so the default is
    untested. PR 1.5 shipped with ``parents[3]`` (one dir above the
    repo root) and CI's first ``--pack`` run failed to find the tarball
    where the upload step expected it. This test pins the default to
    the directory containing the in-tree ``pyproject.toml`` — any
    future off-by-one in the parents[] count fails here.
    """
    from constellation.viz.frontend import build as build_mod

    assert (build_mod._REPO_ROOT / "pyproject.toml").is_file(), (
        f"_REPO_ROOT={build_mod._REPO_ROOT} does not contain pyproject.toml; "
        f"the parents[N] count in constellation/viz/frontend/build.py is wrong"
    )
    assert build_mod._DIST_DIR == build_mod._REPO_ROOT / "dist"


def test_pack_bundle_raises_when_source_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pack_bundle(
            entry="genome",
            static_root=tmp_path / "nope",
            dist_dir=tmp_path / "dist",
            version="0.0.0",
        )


# ----------------------------------------------------------------------
# install_frontend_from_tarball — consumer-side happy paths
# ----------------------------------------------------------------------


def test_install_round_trip_happy_path(tmp_path: Path) -> None:
    tarball, sidecar, dest_root = _make_packed_bundle(tmp_path, version="1.0.0")
    assert sidecar.exists()
    result = install_frontend_from_tarball(
        local_path=tarball, dest_root=dest_root
    )
    assert result.entry == "genome"
    assert (dest_root / "genome" / "index.genome.html").is_file()
    assert (dest_root / "genome" / "assets" / "main_genome-fakehash.js").is_file()
    assert (dest_root / "genome" / "bundle.json").is_file()
    # bundle metadata is mirrored on disk for `doctor` to read later
    meta = read_bundle_metadata(dest_root / "genome")
    assert meta is not None
    assert meta["constellation_version"] == "1.0.0"
    assert result.bundle_metadata["constellation_version"] == "1.0.0"


def test_install_with_no_verify_skips_checksum(tmp_path: Path) -> None:
    tarball, sidecar, dest_root = _make_packed_bundle(tmp_path)
    # Corrupt the sidecar so verification would fail if attempted.
    sidecar.write_text("0" * 64 + "  " + tarball.name + "\n", encoding="utf-8")
    result = install_frontend_from_tarball(
        local_path=tarball, dest_root=dest_root, verify=False
    )
    assert result.bundle_metadata["entry"] == "genome"
    assert (dest_root / "genome" / "index.genome.html").is_file()


def test_install_force_replaces_existing(tmp_path: Path) -> None:
    tarball, _, dest_root = _make_packed_bundle(tmp_path, version="2.0.0")
    install_frontend_from_tarball(local_path=tarball, dest_root=dest_root)

    # Re-pack at a new version, install over the top with --force
    tarball2, _, _ = _make_packed_bundle(
        tmp_path / "second", version="3.0.0"
    )
    result = install_frontend_from_tarball(
        local_path=tarball2, dest_root=dest_root, force=True
    )
    assert result.bundle_metadata["constellation_version"] == "3.0.0"
    on_disk = read_bundle_metadata(dest_root / "genome")
    assert on_disk is not None
    assert on_disk["constellation_version"] == "3.0.0"


# ----------------------------------------------------------------------
# install_frontend_from_tarball — failure modes
# ----------------------------------------------------------------------


def test_install_missing_tarball_raises(tmp_path: Path) -> None:
    with pytest.raises(InstallError, match="tarball not found"):
        install_frontend_from_tarball(
            local_path=tmp_path / "does-not-exist.tar.gz",
            dest_root=tmp_path / "out",
        )


def test_install_missing_sidecar_raises(tmp_path: Path) -> None:
    tarball, sidecar, dest_root = _make_packed_bundle(tmp_path)
    sidecar.unlink()
    with pytest.raises(InstallError, match="sha256 sidecar not found"):
        install_frontend_from_tarball(local_path=tarball, dest_root=dest_root)


def test_install_sha256_mismatch_raises(tmp_path: Path) -> None:
    tarball, sidecar, dest_root = _make_packed_bundle(tmp_path)
    # Replace the digest with a non-matching one
    sidecar.write_text(
        "f" * 64 + "  " + tarball.name + "\n", encoding="utf-8"
    )
    with pytest.raises(InstallError, match="sha256 mismatch"):
        install_frontend_from_tarball(local_path=tarball, dest_root=dest_root)


def test_install_non_empty_target_without_force_raises(tmp_path: Path) -> None:
    tarball, _, dest_root = _make_packed_bundle(tmp_path)
    install_frontend_from_tarball(local_path=tarball, dest_root=dest_root)
    with pytest.raises(InstallError, match="destination not empty"):
        install_frontend_from_tarball(
            local_path=tarball, dest_root=dest_root, force=False
        )


def test_install_rejects_multi_top_level_tarball(tmp_path: Path) -> None:
    # Hand-roll a malformed tarball with two top-level entries
    static_root = tmp_path / "src"
    _write_fake_bundle(static_root)
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        tar.add(static_root, arcname="prefix_a")
        tar.add(static_root, arcname="prefix_b")
    sidecar = tmp_path / f"{bad.name}.sha256"
    sidecar.write_text(
        hashlib.sha256(bad.read_bytes()).hexdigest()
        + "  "
        + bad.name
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(InstallError, match="multiple top-level entries"):
        install_frontend_from_tarball(
            local_path=bad, dest_root=tmp_path / "out"
        )


def test_install_rejects_path_traversal(tmp_path: Path) -> None:
    """A malicious tarball trying to escape `dest_dir` via '..' is refused."""
    bad = tmp_path / "evil.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        # Forge a member whose name escapes dest_dir
        attack = tarfile.TarInfo(
            name="constellation-viz-frontend-0.0.0/static/genome/../../etc/passwd-pwn"
        )
        payload = b"haha\n"
        attack.size = len(payload)
        import io as _io

        tar.addfile(attack, fileobj=_io.BytesIO(payload))
        # Add a benign member so the static-prefix exists
        ok = tarfile.TarInfo(
            name="constellation-viz-frontend-0.0.0/static/genome/index.genome.html"
        )
        body = b"<html></html>"
        ok.size = len(body)
        tar.addfile(ok, fileobj=_io.BytesIO(body))
    sidecar = tmp_path / f"{bad.name}.sha256"
    sidecar.write_text(
        hashlib.sha256(bad.read_bytes()).hexdigest()
        + "  "
        + bad.name
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(InstallError):
        install_frontend_from_tarball(
            local_path=bad, dest_root=tmp_path / "out"
        )


# ----------------------------------------------------------------------
# read_bundle_metadata — doctor helper
# ----------------------------------------------------------------------


def test_read_bundle_metadata_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_bundle_metadata(tmp_path / "nope") is None


def test_read_bundle_metadata_returns_none_on_garbage(tmp_path: Path) -> None:
    target = tmp_path / "genome"
    target.mkdir()
    (target / "bundle.json").write_text("not json", encoding="utf-8")
    assert read_bundle_metadata(target) is None


def test_read_bundle_metadata_round_trip(tmp_path: Path) -> None:
    tarball, _, dest_root = _make_packed_bundle(tmp_path, version="5.5.5")
    install_frontend_from_tarball(local_path=tarball, dest_root=dest_root)
    meta = read_bundle_metadata(dest_root / "genome")
    assert meta is not None
    assert meta["constellation_version"] == "5.5.5"
    assert meta["entry"] == "genome"
