"""Saved-session cache — per-user store of named genome-browser sessions.

Layout::

    ~/.constellation/sessions/<slug>.toml

Each TOML records the reference handle plus the data sources the user
attached, so the dashboard can restore a configuration in one click.
Root resolution follows the same precedence as the reference and catalog
caches (env override → XDG → home), so all three siblings under
``~/.constellation/`` move together.

Schema v2 (current)::

    schema_version = 2
    label = "..."
    reference_handle = "<organism>@<source>-<release>"
    saved_at = "2026-05-22T..."

    [[sources]]
    path = "/abs/path"
    kind = "align"
    label = "..."

    [last_viewed_locus]
    contig = "chr1"
    start = 0
    end = 100000

    [options]
    clip_svg = false          # browser-wide preferences; optional

    [[track_layout]]
    source_id = "src-1a2b3c4d"   # "" for reference-only bindings
    kind = "coverage_histogram"
    visible = true
    display_order = 2
    height_px = 80
    collapsed = false

    [track_layout.style]      # optional — per-kernel palette/size overrides
    "palette.gene" = "#5e8cd6"
    feature_opacity = 0.85

    [track_layout.filter]     # optional — per-kernel dataset-slice filters
    visible_samples = [0, 1]

v1 lacked ``[[track_layout]]``; v1 files are accepted transparently
and treated as "all visible, default order, default heights" — they
get rewritten as v2 on the next save.

The ``[options]`` block, plus per-entry ``style`` / ``filter`` tables,
landed after v2 was minted. They are forward-compatibility extensions
to the v2 schema — older Constellations reading a file that includes
them silently drop the unknown keys; newer Constellations reading an
older v2 file see absent options + empty per-entry style/filter and
fall back to defaults. No schema_version bump is required.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAVED_SESSION_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = (1, 2)
_CACHE_ENV_VAR = "CONSTELLATION_SESSIONS_HOME"
_XDG_ENV_VAR = "XDG_DATA_HOME"
_HOME_FALLBACK = ".constellation/sessions"


def cache_root() -> Path:
    """Resolve the active saved-session cache root.

    Precedence chain (first hit wins):
        1. ``$CONSTELLATION_SESSIONS_HOME``
        2. ``$XDG_DATA_HOME/constellation/sessions/``
        3. ``~/.constellation/sessions/``
    """
    env = os.environ.get(_CACHE_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get(_XDG_ENV_VAR)
    if xdg:
        return (Path(xdg).expanduser() / "constellation" / "sessions").resolve()
    return (Path.home() / _HOME_FALLBACK).resolve()


@dataclass(frozen=True, slots=True)
class SavedSessionSource:
    """One data-source entry inside a saved session."""

    path: str
    kind: str  # 'align' | 'cluster'
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "label": self.label}


@dataclass(frozen=True, slots=True)
class SavedSession:
    """A named, persisted genome-browser session configuration."""

    slug: str
    label: str
    reference_handle: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    saved_at: str = ""
    last_viewed_locus: dict[str, Any] | None = None
    track_layout: list[dict[str, Any]] | None = None
    options: dict[str, Any] | None = None
    path: Path | None = None  # populated by readers; absent on fresh constructs

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "label": self.label,
            "reference_handle": self.reference_handle,
            "n_sources": len(self.sources),
            "saved_at": self.saved_at,
            "last_viewed_locus": self.last_viewed_locus,
            "sources": list(self.sources),
            "track_layout": (
                list(self.track_layout) if self.track_layout else None
            ),
            "options": dict(self.options) if self.options else None,
        }


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------


def _slug(s: str) -> str:
    out: list[str] = []
    for ch in s.lower():
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "session"


def derive_slug(label: str, saved_at: str) -> str:
    """Stable slug for a (label, saved_at) pair. Filesystem-safe."""
    payload = f"{label}|{saved_at}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).hexdigest()
    return f"{_slug(label)}-{digest}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _normalize_track_layout(
    entries: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Validate + coerce a track-layout list. Missing source_id maps to ""
    (reference-only binding). Unknown extra keys are dropped silently
    rather than raised — they're forward-compatibility room.

    Optional ``style`` / ``filter`` sub-tables are preserved as opaque
    dicts (renderer-interpreted). Non-dict values are dropped silently
    so a hand-edited TOML can't crash the loader.
    """
    if not entries:
        return []
    out: list[dict[str, Any]] = []
    for entry in entries:
        kind = entry.get("kind")
        if not kind:
            continue
        normalized: dict[str, Any] = {
            "source_id": str(entry.get("source_id") or ""),
            "kind": str(kind),
            "visible": bool(entry.get("visible", True)),
            "display_order": int(entry.get("display_order", 0)),
            "height_px": int(entry.get("height_px", 0)),
            "collapsed": bool(entry.get("collapsed", False)),
        }
        style = entry.get("style")
        if isinstance(style, dict) and style:
            normalized["style"] = dict(style)
        flt = entry.get("filter")
        if isinstance(flt, dict) and flt:
            normalized["filter"] = dict(flt)
        out.append(normalized)
    return out


def _normalize_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Validate + coerce the browser-wide ``[options]`` block. Returns
    an empty dict when no real values are present — callers can then
    skip emitting the section. Defaults are *not* injected; an absent
    key simply means "client falls back to its own default".
    """
    if not isinstance(options, dict):
        return {}
    out: dict[str, Any] = {}
    if "clip_svg" in options:
        out["clip_svg"] = bool(options.get("clip_svg"))
    return out


def write_saved(
    *,
    label: str,
    reference_handle: str,
    sources: list[dict[str, Any]],
    last_viewed_locus: dict[str, Any] | None = None,
    track_layout: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
    saved_at: str | None = None,
    slug: str | None = None,
    root: Path | None = None,
) -> SavedSession:
    """Persist a saved-session TOML; return the resulting dataclass.

    A fresh ``saved_at`` timestamp is generated when not provided. The
    slug defaults to ``derive_slug(label, saved_at)``; pass an explicit
    slug to overwrite an existing entry.
    """
    if not label:
        raise ValueError("saved session requires a non-empty label")
    if not reference_handle:
        raise ValueError("saved session requires a reference_handle")
    if root is None:
        root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    final_saved_at = saved_at or _now_iso()
    final_slug = slug or derive_slug(label, final_saved_at)

    normalized_sources: list[dict[str, Any]] = []
    for src in sources:
        path = src.get("path")
        kind = src.get("kind")
        if not path or not kind:
            raise ValueError(
                f"saved source missing required path/kind: {src!r}"
            )
        normalized_sources.append(
            {
                "path": str(path),
                "kind": str(kind),
                "label": str(src.get("label") or Path(str(path)).name),
            }
        )

    normalized_layout = _normalize_track_layout(track_layout)
    normalized_options = _normalize_options(options)

    lines: list[str] = [
        f"schema_version = {SAVED_SESSION_SCHEMA_VERSION}",
        f'label = "{_toml_escape(label)}"',
        f'reference_handle = "{_toml_escape(reference_handle)}"',
        f'saved_at = "{final_saved_at}"',
    ]
    for src in normalized_sources:
        lines.append("")
        lines.append("[[sources]]")
        lines.append(f'path = "{_toml_escape(src["path"])}"')
        lines.append(f'kind = "{_toml_escape(src["kind"])}"')
        lines.append(f'label = "{_toml_escape(src["label"])}"')
    if last_viewed_locus is not None:
        contig = str(last_viewed_locus.get("contig", ""))
        start = int(last_viewed_locus.get("start", 0))
        end = int(last_viewed_locus.get("end", 0))
        lines.append("")
        lines.append("[last_viewed_locus]")
        lines.append(f'contig = "{_toml_escape(contig)}"')
        lines.append(f"start = {start}")
        lines.append(f"end = {end}")
    if normalized_options:
        lines.append("")
        lines.append("[options]")
        for key, value in normalized_options.items():
            lines.append(_format_toml_kv(key, value))
    for entry in normalized_layout:
        lines.append("")
        lines.append("[[track_layout]]")
        lines.append(f'source_id = "{_toml_escape(entry["source_id"])}"')
        lines.append(f'kind = "{_toml_escape(entry["kind"])}"')
        lines.append(f"visible = {'true' if entry['visible'] else 'false'}")
        lines.append(f"display_order = {entry['display_order']}")
        lines.append(f"height_px = {entry['height_px']}")
        lines.append(f"collapsed = {'true' if entry['collapsed'] else 'false'}")
        style = entry.get("style")
        if isinstance(style, dict) and style:
            lines.append("")
            lines.append("[track_layout.style]")
            for key, value in style.items():
                lines.append(_format_toml_kv(str(key), value))
        flt = entry.get("filter")
        if isinstance(flt, dict) and flt:
            lines.append("")
            lines.append("[track_layout.filter]")
            for key, value in flt.items():
                lines.append(_format_toml_kv(str(key), value))

    path = root / f"{final_slug}.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return SavedSession(
        slug=final_slug,
        label=label,
        reference_handle=reference_handle,
        sources=normalized_sources,
        saved_at=final_saved_at,
        last_viewed_locus=last_viewed_locus,
        track_layout=normalized_layout or None,
        options=normalized_options or None,
        path=path,
    )


def _format_toml_kv(key: str, value: Any) -> str:
    """Emit a single ``key = <value>`` TOML line. Handles strings,
    booleans, numbers, and one-level lists of those — enough to cover
    the style/filter/options shapes we emit. The key is quoted when it
    contains characters TOML's bare-key form forbids (e.g.
    ``palette.gene`` becomes ``"palette.gene"``)."""
    return f"{_format_toml_key(key)} = {_format_toml_value(value)}"


def _format_toml_key(key: str) -> str:
    bare = all(ch.isalnum() or ch in "-_" for ch in key) and key != ""
    return key if bare else f'"{_toml_escape(key)}"'


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return f'"{_toml_escape(value)}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_toml_value(v) for v in value) + "]"
    # Fallback: stringify and quote — keeps the writer from ever raising.
    return f'"{_toml_escape(str(value))}"'


def read_saved(slug: str, *, root: Path | None = None) -> SavedSession:
    """Load one saved session by slug. Raises ``FileNotFoundError`` if
    the TOML does not exist and ``ValueError`` for unsupported schema
    versions."""
    if root is None:
        root = cache_root()
    path = root / f"{slug}.toml"
    if not path.exists():
        raise FileNotFoundError(f"no saved session at {path}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    schema_version = int(raw.get("schema_version", 1))
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported saved-session schema_version={schema_version} "
            f"at {path}; this constellation supports versions "
            f"{list(_SUPPORTED_SCHEMA_VERSIONS)}"
        )
    label = str(raw.get("label") or slug)
    reference_handle = str(raw.get("reference_handle") or "")
    if not reference_handle:
        raise ValueError(
            f"saved session at {path} is missing reference_handle"
        )
    saved_at = str(raw.get("saved_at") or "")
    sources_raw = raw.get("sources") or []
    sources: list[dict[str, Any]] = []
    for entry in sources_raw:
        if not isinstance(entry, dict):
            continue
        path_val = entry.get("path")
        kind_val = entry.get("kind")
        if not path_val or not kind_val:
            continue
        sources.append(
            {
                "path": str(path_val),
                "kind": str(kind_val),
                "label": str(entry.get("label") or Path(str(path_val)).name),
            }
        )
    locus = raw.get("last_viewed_locus")
    last_viewed: dict[str, Any] | None = None
    if isinstance(locus, dict):
        last_viewed = {
            "contig": str(locus.get("contig", "")),
            "start": int(locus.get("start", 0)),
            "end": int(locus.get("end", 0)),
        }
    layout_raw = raw.get("track_layout")
    track_layout = (
        _normalize_track_layout(layout_raw)
        if isinstance(layout_raw, list)
        else []
    )
    options_raw = raw.get("options")
    options = (
        _normalize_options(options_raw)
        if isinstance(options_raw, dict)
        else {}
    )
    return SavedSession(
        slug=slug,
        label=label,
        reference_handle=reference_handle,
        sources=sources,
        saved_at=saved_at,
        last_viewed_locus=last_viewed,
        track_layout=track_layout or None,
        options=options or None,
        path=path,
    )


def list_saved(*, root: Path | None = None) -> list[SavedSession]:
    """Return every saved-session TOML under the cache root.

    Malformed entries are skipped silently — the dashboard would
    otherwise fail to enumerate if any single saved session was
    corrupt. The corrupt files remain on disk for manual inspection.
    """
    if root is None:
        root = cache_root()
    if not root.is_dir():
        return []
    out: list[SavedSession] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_file() or entry.suffix != ".toml":
            continue
        slug = entry.stem
        try:
            out.append(read_saved(slug, root=root))
        except (ValueError, FileNotFoundError, OSError):
            continue
    return out


def delete_saved(slug: str, *, root: Path | None = None) -> bool:
    """Remove a saved session by slug; return True iff it existed."""
    if root is None:
        root = cache_root()
    path = root / f"{slug}.toml"
    if not path.exists():
        return False
    path.unlink()
    return True
