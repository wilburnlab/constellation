"""Unit tests for constellation.core.io.bundle."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from constellation.core.io.bundle import Bundle, DirBundle, OpcBundle


# ----------------------------------------------------------------------
# DirBundle
# ----------------------------------------------------------------------


def test_dir_bundle_single_file_auto_primary(tmp_path):
    f = tmp_path / "only.bin"
    f.write_bytes(b"hello")
    b = DirBundle(tmp_path)
    assert b.primary == "only.bin"
    assert b.members() == ["only.bin"]
    assert b.open("only.bin") == b"hello"


def test_dir_bundle_multi_file_requires_explicit_primary(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "b.bin").write_bytes(b"b")
    with pytest.raises(ValueError, match="specify primary="):
        DirBundle(tmp_path)
    b = DirBundle(tmp_path, primary="a.bin")
    assert b.primary == "a.bin"
    assert sorted(b.members()) == ["a.bin", "b.bin"]
    assert b.open("b.bin") == b"b"


def test_dir_bundle_open_unknown_raises(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"x")
    b = DirBundle(tmp_path)
    with pytest.raises(KeyError):
        b.open("missing.bin")


def test_dir_bundle_rejects_non_dir(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        DirBundle(f)


def test_dir_bundle_primary_must_exist(tmp_path):
    (tmp_path / "real.bin").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        DirBundle(tmp_path, primary="missing.bin")


# ----------------------------------------------------------------------
# OpcBundle
# ----------------------------------------------------------------------


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def test_opc_bundle_eager(tmp_path):
    z = _make_zip(
        tmp_path / "run.dx",
        {"injection.acmd": b"<xml/>", "abc.CH": b"\x03179payload"},
    )
    b = OpcBundle(z)
    assert sorted(b.members()) == ["abc.CH", "injection.acmd"]
    assert b.open("injection.acmd") == b"<xml/>"
    assert b.open("abc.CH") == b"\x03179payload"


def test_opc_bundle_lazy(tmp_path):
    z = _make_zip(tmp_path / "run.dx", {"a.txt": b"hi"})
    b = OpcBundle(z, lazy=True)
    # Lazy: members listed but no payload cached.
    assert b._members == {}
    assert b.open("a.txt") == b"hi"
    # After open, cached.
    assert b._members["a.txt"] == b"hi"


def test_opc_bundle_unknown_member_raises(tmp_path):
    z = _make_zip(tmp_path / "run.dx", {"a.txt": b""})
    b = OpcBundle(z)
    with pytest.raises(KeyError):
        b.open("missing")


# ----------------------------------------------------------------------
# Bundle.from_path
# ----------------------------------------------------------------------


def test_from_path_dispatches_zip(tmp_path):
    z = _make_zip(tmp_path / "run.dx", {"foo": b"x"})
    b = Bundle.from_path(z)
    assert isinstance(b, OpcBundle)


def test_from_path_dispatches_directory(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"x")
    b = Bundle.from_path(tmp_path)
    assert isinstance(b, DirBundle)


def test_from_path_single_regular_file(tmp_path):
    f = tmp_path / "x.raw"
    f.write_bytes(b"x")
    b = Bundle.from_path(f)
    assert isinstance(b, DirBundle)
    assert b.primary == "x.raw"


def test_from_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        Bundle.from_path(tmp_path / "does-not-exist")
