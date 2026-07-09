"""Sandboxed filesystem-listing endpoint for the dashboard file picker.

``GET /api/fs/list`` powers the frontend ``FilePicker`` widget: it
enumerates the entries of a directory the user is browsing so they can
pick an input/output path without typing it. The dashboard server is
unauthenticated and can be exposed to the LAN via ``--host``, so this is
the one endpoint that materially widens the read surface beyond a
session's parquet. Every path is therefore confined to a set of allowed
roots (``app.state.fs_roots``) — home + cwd + WSL drives by default,
widened at launch with ``constellation dashboard --root DIR``.

The sandbox (``_resolve_within_roots``) mirrors the guarded-``iterdir``
precedent in ``sequencing/reference/handle.py::list_installed``:
``resolve()`` collapses ``..`` and dereferences symlinks, then a
containment check rejects anything outside every root. Symlinks that
point out of the sandbox are dropped from listings so the picker never
surfaces an escape hatch.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/api/fs", tags=["fs"])

# Defensive cap so a pathological directory can't produce a huge payload.
_MAX_ENTRIES = 5000

_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def _is_wsl() -> bool:
    """True when running under WSL (surfaces ``/mnt/<drive>`` roots)."""
    uname = getattr(os, "uname", None)
    if uname is None:  # non-POSIX; WSL is always POSIX
        return False
    try:
        return "microsoft" in uname().release.lower()
    except OSError:
        return False


def _wsl_drives() -> list[Path]:
    """Existing ``/mnt/<letter>`` mount points, if any."""
    mnt = Path("/mnt")
    out: list[Path] = []
    if not mnt.is_dir():
        return out
    try:
        for child in sorted(mnt.iterdir()):
            # WSL mounts single-letter drive dirs (c, d, …).
            if child.is_dir() and len(child.name) == 1 and child.name.isalpha():
                out.append(child)
    except OSError:
        return out
    return out


def default_fs_roots() -> list[Path]:
    """Sandbox roots when the CLI passes no ``--root``.

    Home directory + current working directory + (under WSL) mounted
    Windows drives. De-duplicated and filtered to existing dirs. Defensive
    so app construction never raises on an odd ``$HOME`` / CI environment.
    """
    candidates: list[Path] = []
    for getter in (Path.home, Path.cwd):
        try:
            candidates.append(getter())
        except (OSError, RuntimeError):
            continue
    if _is_wsl():
        candidates.extend(_wsl_drives())
    return _dedup_existing(candidates)


def _dedup_existing(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        try:
            rp = p.expanduser().resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        key = str(rp)
        if key in seen or not rp.is_dir():
            continue
        seen.add(key)
        out.append(rp)
    return out


def _normalize_wsl(raw: str) -> str:
    """Rewrite a pasted Windows path ``C:\\a\\b`` → ``/mnt/c/a/b``.

    No-op when not under WSL or when ``raw`` isn't a drive path. Tries
    ``wslpath -u`` first (handles UNC/edge cases), falls back to a direct
    rewrite.
    """
    if not _is_wsl():
        return raw
    m = _WIN_DRIVE_RE.match(raw)
    if not m:
        return raw
    try:
        res = subprocess.run(
            ["wslpath", "-u", raw],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        out = res.stdout.strip()
        if out:
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    drive, rest = m.group(1).lower(), m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}".rstrip("/") or f"/mnt/{drive}"


def _resolve_roots(request: Request) -> list[Path]:
    roots = getattr(request.app.state, "fs_roots", None)
    if not roots:
        roots = default_fs_roots()
    return [Path(r) for r in roots]


def _resolve_within_roots(raw: str, roots: list[Path]) -> Path:
    """Normalize + resolve ``raw`` and confirm it lies within a root.

    ``strict=False`` keeps a not-yet-existing leaf under a root
    resolvable (the "new output folder" case). Raises 400 on a malformed
    path, 403 when the resolved real path escapes every root.
    """
    normalized = _normalize_wsl(raw)
    try:
        p = Path(normalized).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, f"bad path: {raw}") from exc
    for root in roots:
        if p == root or p.is_relative_to(root):
            return p
    raise HTTPException(403, f"path is outside the allowed roots: {p}")


def _within_any(p: Path, roots: list[Path]) -> bool:
    return any(p == r or p.is_relative_to(r) for r in roots)


def _root_label(root: Path, home: Path | None, cwd: Path | None) -> str:
    if home is not None and root == home:
        return "Home"
    if cwd is not None and root == cwd:
        return "Working dir"
    if root.parent == Path("/mnt") and len(root.name) == 1:
        return f"{root.name.upper()}:"
    return root.name or str(root)


def _roots_payload(roots: list[Path]) -> list[dict]:
    try:
        home = Path.home().resolve(strict=False)
    except (OSError, RuntimeError):
        home = None
    try:
        cwd = Path.cwd().resolve(strict=False)
    except (OSError, RuntimeError):
        cwd = None
    return [{"label": _root_label(r, home, cwd), "path": str(r)} for r in roots]


def _matches_globs(name: str, globs: list[str]) -> bool:
    if not globs:
        return True
    from fnmatch import fnmatch

    return any(fnmatch(name.lower(), g.lower()) for g in globs)


@router.get("/list")
def list_directory(
    request: Request,
    path: str | None = Query(
        None, description="directory to list; omit for first root"
    ),
    dirs_only: bool = Query(False, description="omit files, list subdirectories only"),
    globs: str | None = Query(
        None, description="comma-separated file filters, e.g. *.tsv,*.bed"
    ),
) -> dict:
    """List the entries of a directory, confined to the sandbox roots."""
    roots = _resolve_roots(request)
    if not roots:
        raise HTTPException(500, "no filesystem roots are configured")

    target = _resolve_within_roots(path, roots) if path else roots[0]

    if not target.exists():
        raise HTTPException(404, f"no such directory: {target}")
    if not target.is_dir():
        raise HTTPException(400, f"not a directory: {target}")

    glob_list = [g.strip() for g in globs.split(",")] if globs else []
    glob_list = [g for g in glob_list if g]

    try:
        children = list(target.iterdir())
    except PermissionError as exc:
        raise HTTPException(403, f"permission denied: {target}") from exc
    except OSError as exc:
        raise HTTPException(400, f"cannot read directory: {target}") from exc

    entries: list[dict] = []
    truncated = False
    for entry in sorted(children, key=lambda c: c.name.lower()):
        try:
            real = entry.resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        # Drop symlinks that escape the sandbox — no back door.
        if not _within_any(real, roots):
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if not is_dir:
            if dirs_only:
                continue
            if not _matches_globs(entry.name, glob_list):
                continue
        size: int | None = None
        mtime: float | None = None
        try:
            st = entry.stat()
            mtime = st.st_mtime
            if not is_dir:
                size = st.st_size
        except OSError:
            pass
        entries.append(
            {
                "name": entry.name,
                "path": str(entry),
                "is_dir": is_dir,
                "size": size,
                "mtime": mtime,
            }
        )
        if len(entries) >= _MAX_ENTRIES:
            truncated = True
            break

    # Directories first, then files; each already alphabetical.
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))

    parent = target.parent
    has_parent = target != parent and _within_any(parent, roots)

    return {
        "path": str(target),
        "parent": str(parent) if has_parent else None,
        "is_root": any(target == r for r in roots),
        "roots": _roots_payload(roots),
        "entries": entries,
        "truncated": truncated,
    }
