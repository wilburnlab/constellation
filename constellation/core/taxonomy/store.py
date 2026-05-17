"""On-disk taxonomy cache layout.

Layout under ``$CONSTELLATION_TAXONOMY_HOME`` (or the XDG / dotfile
fallback chain — see ``taxonomy_root()``):

    <root>/
        ncbi-YYYYMMDD/
            nodes.parquet
            names.parquet
            merged.parquet
            meta.toml
        current        — symlink to most recently fetched bundle
                         (text file with the dirname on Windows)

Mirrors the conventions in ``constellation.sequencing.reference.handle``:
``$CONSTELLATION_*_HOME`` → ``$XDG_DATA_HOME/constellation/*`` → dotfile
fallback in the user's home.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


_CACHE_ENV_VAR = "CONSTELLATION_TAXONOMY_HOME"
_HOME_FALLBACK = ".constellation/taxonomy"
_CURRENT_LINK = "current"
_CURRENT_TXT = "current.txt"  # Windows fallback
META_TOML_NAME = "meta.toml"


class CachedTaxonomyMissing(FileNotFoundError):
    """No taxonomy bundle has been fetched into the cache yet."""


# ──────────────────────────────────────────────────────────────────────
# Root resolution
# ──────────────────────────────────────────────────────────────────────


def taxonomy_root() -> Path:
    """Resolve the active taxonomy cache root.

    Precedence:
      1. ``$CONSTELLATION_TAXONOMY_HOME`` (explicit override)
      2. ``$XDG_DATA_HOME/constellation/taxonomy``
      3. ``~/.constellation/taxonomy``
    """
    env = os.environ.get(_CACHE_ENV_VAR)
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "constellation" / "taxonomy"
    return Path.home() / _HOME_FALLBACK


def list_installed(root: Path | None = None) -> list[Path]:
    """Return all bundle directories under the cache root, newest-first.

    Recognises any subdirectory whose ``meta.toml`` validates. The
    ``current`` pointer (symlink or text file) is excluded.
    """
    root = root or taxonomy_root()
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.iterdir():
        if p.name in {_CURRENT_LINK, _CURRENT_TXT}:
            continue
        if not p.is_dir():
            continue
        if not (p / META_TOML_NAME).exists():
            continue
        out.append(p)
    out.sort(key=lambda d: d.name, reverse=True)
    return out


def resolve_current(root: Path | None = None) -> Path | None:
    """Resolve the ``current`` pointer.

    Symlink → target if the target exists; ``current.txt`` (Windows
    fallback) → resolved dir if it contains a valid bundle. Returns
    ``None`` if no current is set.
    """
    root = root or taxonomy_root()
    link = root / _CURRENT_LINK
    if link.is_symlink():
        target = link.resolve()
        if (target / META_TOML_NAME).exists():
            return target
    txt = root / _CURRENT_TXT
    if txt.exists():
        name = txt.read_text(encoding="utf-8").strip()
        target = root / name
        if (target / META_TOML_NAME).exists():
            return target
    return None


def update_current_pointer(target: Path, root: Path | None = None) -> None:
    """Point ``current`` at ``target``. Symlink on POSIX, text file on Windows."""
    root = root or taxonomy_root()
    root.mkdir(parents=True, exist_ok=True)
    link = root / _CURRENT_LINK
    if sys.platform == "win32":
        (root / _CURRENT_TXT).write_text(target.name, encoding="utf-8")
        return
    if link.is_symlink() or link.exists():
        try:
            link.unlink()
        except FileNotFoundError:
            pass
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        # Filesystem doesn't support symlinks — fall back to text marker.
        (root / _CURRENT_TXT).write_text(target.name, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Bundle I/O
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TaxonomyBundle:
    nodes: pa.Table
    names: pa.Table
    merged: pa.Table
    meta: dict[str, Any]


def write_bundle(
    bundle_dir: Path,
    *,
    nodes: pa.Table,
    names: pa.Table,
    merged: pa.Table,
    meta: dict[str, Any],
) -> None:
    """Write a taxonomy bundle (three parquets + meta.toml) atomically.

    Writes go through a sibling ``.partial`` directory, then rename.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = bundle_dir.with_name(bundle_dir.name + ".partial")
    if partial.exists():
        _rmtree(partial)
    partial.mkdir(parents=True)
    pq.write_table(nodes, partial / "nodes.parquet", compression="snappy")
    pq.write_table(names, partial / "names.parquet", compression="snappy")
    pq.write_table(merged, partial / "merged.parquet", compression="snappy")
    (partial / META_TOML_NAME).write_text(_render_meta_toml(meta), encoding="utf-8")
    if bundle_dir.exists():
        _rmtree(bundle_dir)
    partial.rename(bundle_dir)


def read_bundle(bundle_dir: Path) -> TaxonomyBundle:
    bundle_dir = Path(bundle_dir)
    if not (bundle_dir / META_TOML_NAME).exists():
        raise FileNotFoundError(f"no taxonomy bundle at {bundle_dir!s}")
    nodes = pq.read_table(bundle_dir / "nodes.parquet")
    names = pq.read_table(bundle_dir / "names.parquet")
    merged = pq.read_table(bundle_dir / "merged.parquet")
    meta = read_meta_toml(bundle_dir / META_TOML_NAME)
    return TaxonomyBundle(nodes=nodes, names=names, merged=merged, meta=meta)


def load_cached_taxonomy(
    root: Path | None = None,
) -> tuple[pa.Table, pa.Table, pa.Table, dict[str, Any]]:
    """Resolve the active bundle (current pointer, else newest), and read it."""
    root = root or taxonomy_root()
    target = resolve_current(root)
    if target is None:
        installed = list_installed(root)
        if not installed:
            raise CachedTaxonomyMissing(
                f"no taxonomy bundle installed under {root!s}; "
                "run `constellation taxonomy update` to fetch one"
            )
        target = installed[0]
    bundle = read_bundle(target)
    return bundle.nodes, bundle.names, bundle.merged, bundle.meta


# ──────────────────────────────────────────────────────────────────────
# meta.toml
# ──────────────────────────────────────────────────────────────────────


def read_meta_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _render_meta_toml(meta: dict[str, Any]) -> str:
    """Render a flat-ish meta dict to TOML.

    Stays minimal — no nested tables beyond what we need (sha256 + urls).
    """
    lines: list[str] = []
    flat: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            nested[k] = v
        else:
            flat[k] = v
    for k, v in flat.items():
        lines.append(f"{k} = {_toml_value(v)}")
    for table_name, body in nested.items():
        lines.append("")
        lines.append(f"[{table_name}]")
        for k, v in body.items():
            lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines) + "\n"


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value type: {type(v).__name__}")


def _rmtree(path: Path) -> None:
    """Recursive remove for atomic write fallback."""
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    for child in path.iterdir():
        _rmtree(child)
    path.rmdir()


__all__ = [
    "CachedTaxonomyMissing",
    "META_TOML_NAME",
    "TaxonomyBundle",
    "list_installed",
    "load_cached_taxonomy",
    "read_bundle",
    "read_meta_toml",
    "resolve_current",
    "taxonomy_root",
    "update_current_pointer",
    "write_bundle",
]
