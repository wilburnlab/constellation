"""Sandboxed file-picker endpoint — GET /api/fs/list.

The endpoint widens the server's read surface from "a session's parquet"
to "arbitrary directory listing", so the sandbox (`_resolve_within_roots`)
is the load-bearing control. These tests pin its containment guarantees:
no `..` escape, no symlink escape, nothing outside the configured roots.
Gated on ``[viz]`` extras (``fastapi`` + ``httpx``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from constellation.viz.server.app import create_app  # noqa: E402
from constellation.viz.server.endpoints import fs as fs_ep  # noqa: E402


def _client(roots: list[Path]) -> TestClient:
    return TestClient(create_app({}, fs_roots=roots))


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    """A populated root: two dirs, two files, one escaping symlink."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "align_out").mkdir()
    (root / "sub").mkdir()
    (root / "samples.tsv").write_text("x")
    (root / "notes.md").write_text("y")
    return root.resolve()


# ----------------------------------------------------------------------
# Listing
# ----------------------------------------------------------------------


def test_lists_entries_dirs_first(sandbox: Path) -> None:
    with _client([sandbox]) as c:
        r = c.get("/api/fs/list", params={"path": str(sandbox)})
    assert r.status_code == 200
    data = r.json()
    names = [e["name"] for e in data["entries"]]
    # Directories sort ahead of files.
    assert names == ["align_out", "sub", "notes.md", "samples.tsv"]
    kinds = {e["name"]: e["is_dir"] for e in data["entries"]}
    assert kinds["align_out"] is True
    assert kinds["samples.tsv"] is False


def test_is_root_and_parent(sandbox: Path) -> None:
    with _client([sandbox]) as c:
        r = c.get("/api/fs/list", params={"path": str(sandbox)}).json()
    assert r["is_root"] is True
    # Parent escapes the root → suppressed.
    assert r["parent"] is None
    assert any(root["path"] == str(sandbox) for root in r["roots"])


def test_default_path_is_first_root(sandbox: Path) -> None:
    with _client([sandbox]) as c:
        r = c.get("/api/fs/list").json()
    assert r["path"] == str(sandbox)


def test_dirs_only_hides_files(sandbox: Path) -> None:
    with _client([sandbox]) as c:
        r = c.get(
            "/api/fs/list", params={"path": str(sandbox), "dirs_only": "true"}
        ).json()
    assert [e["name"] for e in r["entries"]] == ["align_out", "sub"]


def test_glob_filters_files(sandbox: Path) -> None:
    with _client([sandbox]) as c:
        r = c.get(
            "/api/fs/list", params={"path": str(sandbox), "globs": "*.tsv"}
        ).json()
    files = [e["name"] for e in r["entries"] if not e["is_dir"]]
    assert files == ["samples.tsv"]  # notes.md filtered out
    # Directories are never filtered by globs.
    assert "align_out" in [e["name"] for e in r["entries"]]


# ----------------------------------------------------------------------
# Sandbox enforcement
# ----------------------------------------------------------------------


def test_parent_dotdot_escape_rejected(sandbox: Path) -> None:
    with _client([sandbox]) as c:
        r = c.get("/api/fs/list", params={"path": f"{sandbox}/../{sandbox.name}/.."})
    assert r.status_code == 403


def test_absolute_outside_root_rejected(sandbox: Path, tmp_path: Path) -> None:
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    with _client([sandbox]) as c:
        r = c.get("/api/fs/list", params={"path": str(outside)})
    assert r.status_code == 403


def test_symlink_escape_rejected_and_hidden(sandbox: Path, tmp_path: Path) -> None:
    outside = (tmp_path / "secret").resolve()
    outside.mkdir()
    (outside / "leak.txt").write_text("s")
    os.symlink(outside, sandbox / "escape_link")

    with _client([sandbox]) as c:
        # Directly navigating the escaping symlink is refused.
        r = c.get("/api/fs/list", params={"path": str(sandbox / "escape_link")})
        assert r.status_code == 403
        # And it never appears in the parent listing.
        listing = c.get("/api/fs/list", params={"path": str(sandbox)}).json()
    assert "escape_link" not in [e["name"] for e in listing["entries"]]


def test_nonexistent_directory_404(sandbox: Path) -> None:
    with _client([sandbox]) as c:
        r = c.get("/api/fs/list", params={"path": str(sandbox / "nope")})
    assert r.status_code == 404


def test_file_path_is_400(sandbox: Path) -> None:
    with _client([sandbox]) as c:
        r = c.get("/api/fs/list", params={"path": str(sandbox / "samples.tsv")})
    assert r.status_code == 400


def test_listing_truncates_at_entry_cap(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepting `_MAX_ENTRIES` stops the scan and flags truncated — the
    response never materializes an unbounded directory."""
    monkeypatch.setattr(fs_ep, "_MAX_ENTRIES", 3)
    for i in range(10):
        (sandbox / f"d{i:02d}").mkdir()
    with _client([sandbox]) as c:
        r = c.get("/api/fs/list", params={"path": str(sandbox)}).json()
    assert r["truncated"] is True
    assert len(r["entries"]) == 3


def test_scan_cap_bounds_work_under_restrictive_filter(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restrictive filter can't walk the whole directory: `_MAX_SCAN`
    caps the raw entries examined even when few/none are accepted."""
    monkeypatch.setattr(fs_ep, "_MAX_SCAN", 2)
    for i in range(10):
        (sandbox / f"f{i:02d}.bam").write_text("x")
    with _client([sandbox]) as c:
        r = c.get(
            "/api/fs/list", params={"path": str(sandbox), "globs": "*.nomatch"}
        ).json()
    assert r["truncated"] is True


# ----------------------------------------------------------------------
# Resolver + WSL units
# ----------------------------------------------------------------------


def test_resolve_new_folder_leaf(sandbox: Path) -> None:
    """A not-yet-existing leaf under a root must resolve (the "new output
    folder" flow) without raising."""
    p = fs_ep._resolve_within_roots(str(sandbox / "brand_new"), [sandbox])
    assert p == sandbox / "brand_new"


def test_resolve_outside_raises_403(sandbox: Path, tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        fs_ep._resolve_within_roots(str(tmp_path / "elsewhere"), [sandbox])
    assert exc.value.status_code == 403


def test_wsl_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fs_ep, "_is_wsl", lambda: True)
    # Force the regex fallback (no wslpath dependency in the test env).
    monkeypatch.setattr(
        fs_ep.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert fs_ep._normalize_wsl(r"C:\Users\me\data") == "/mnt/c/Users/me/data"
    assert fs_ep._normalize_wsl(r"D:\x") == "/mnt/d/x"
    # Non-drive paths pass through untouched.
    assert fs_ep._normalize_wsl("/home/me") == "/home/me"


def test_default_roots_surface_wsl_drives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_c = tmp_path / "mnt_c"
    fake_c.mkdir()
    monkeypatch.setattr(fs_ep, "_is_wsl", lambda: True)
    monkeypatch.setattr(fs_ep, "_wsl_drives", lambda: [fake_c])
    roots = fs_ep.default_fs_roots()
    assert fake_c.resolve() in roots


def test_non_wsl_uses_home_and_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fs_ep, "_is_wsl", lambda: False)
    roots = fs_ep.default_fs_roots()
    # Home + cwd are always present; no WSL drives are injected off-WSL.
    assert Path.home().resolve() in roots
    assert Path.cwd().resolve() in roots
