"""On-disk catalog cache layout + meta.toml round-trip.

Layout under ``$CONSTELLATION_CATALOGS_HOME`` (or the XDG / dotfile
fallback chain — see ``catalogs_root()``):

    <root>/
        ensembl/<release>/
            catalog.parquet
            meta.toml
        ensembl_genomes/<release>/...
        refseq/<release>/...
        uniprot/<release>/...

Same conventions as ``constellation.core.taxonomy.store`` and
``constellation.sequencing.reference.handle`` — ``$CONSTELLATION_*_HOME``
override, then XDG, then ``~/.constellation/catalogs``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.catalog.schemas import ASSEMBLY_CATALOG_TABLE


_CACHE_ENV_VAR = "CONSTELLATION_CATALOGS_HOME"
_HOME_FALLBACK = ".constellation/catalogs"
META_TOML_NAME = "meta.toml"


class CatalogNotInstalled(FileNotFoundError):
    """No catalog has been fetched for the requested source/release."""


# ──────────────────────────────────────────────────────────────────────
# Root resolution
# ──────────────────────────────────────────────────────────────────────


def catalogs_root() -> Path:
    env = os.environ.get(_CACHE_ENV_VAR)
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "constellation" / "catalogs"
    return Path.home() / _HOME_FALLBACK


def source_root(source: str, *, root: Path | None = None) -> Path:
    return (root or catalogs_root()) / source


def release_dir(source: str, release: str, *, root: Path | None = None) -> Path:
    return source_root(source, root=root) / release


# ──────────────────────────────────────────────────────────────────────
# Bundle I/O
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CatalogBundle:
    table: pa.Table
    meta: dict[str, Any]
    source: str
    release: str


def write_catalog(
    bundle_dir: Path,
    *,
    table: pa.Table,
    meta: dict[str, Any],
) -> None:
    """Write a catalog bundle (parquet + meta.toml). Overwrites in place."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, bundle_dir / "catalog.parquet", compression="snappy")
    (bundle_dir / META_TOML_NAME).write_text(_render_meta_toml(meta), encoding="utf-8")


def read_catalog(bundle_dir: Path) -> CatalogBundle:
    bundle_dir = Path(bundle_dir)
    meta_path = bundle_dir / META_TOML_NAME
    parquet_path = bundle_dir / "catalog.parquet"
    if not meta_path.exists() or not parquet_path.exists():
        raise CatalogNotInstalled(f"no catalog at {bundle_dir!s}")
    table = pq.read_table(parquet_path)
    meta = read_meta_toml(meta_path)
    src = str(meta.get("source", bundle_dir.parent.name))
    rel = str(meta.get("release", bundle_dir.name))
    return CatalogBundle(table=table, meta=meta, source=src, release=rel)


def list_installed(root: Path | None = None) -> list[CatalogBundle]:
    """Walk the catalog root and return every installed (source, release) bundle."""
    root = root or catalogs_root()
    if not root.exists():
        return []
    out: list[CatalogBundle] = []
    for src_dir in sorted(root.iterdir()):
        if not src_dir.is_dir():
            continue
        for rel_dir in sorted(src_dir.iterdir()):
            if not rel_dir.is_dir():
                continue
            try:
                out.append(read_catalog(rel_dir))
            except (CatalogNotInstalled, FileNotFoundError):
                continue
    return out


def latest_release(source: str, *, root: Path | None = None) -> Path | None:
    """Return the newest release directory for ``source`` (lexicographic).

    Returns ``None`` if no releases are installed. Catalog releases are
    either numeric (Ensembl ``111``) or date-stamped (RefSeq
    ``20260517``) so lexicographic sort puts the newest last.
    """
    src = source_root(source, root=root)
    if not src.exists():
        return None
    dirs = sorted(p for p in src.iterdir() if p.is_dir())
    return dirs[-1] if dirs else None


# ──────────────────────────────────────────────────────────────────────
# meta.toml (re-used logic from the taxonomy store — kept inline to
# avoid a circular import between catalog and core.taxonomy)
# ──────────────────────────────────────────────────────────────────────


def read_meta_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _render_meta_toml(meta: dict[str, Any]) -> str:
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


# Re-export for parser convenience
ASSEMBLY_CATALOG_SCHEMA = ASSEMBLY_CATALOG_TABLE


__all__ = [
    "ASSEMBLY_CATALOG_SCHEMA",
    "CatalogBundle",
    "CatalogNotInstalled",
    "META_TOML_NAME",
    "catalogs_root",
    "latest_release",
    "list_installed",
    "read_catalog",
    "read_meta_toml",
    "release_dir",
    "source_root",
    "write_catalog",
]
