"""Reference cache layout + handle resolution.

Per-user reference cache shared across analysis directories. Replaces
the "every project re-downloads the same FASTA + GFF3" pattern with a
versioned, sha256-checked, atomically-written cache under
``~/.constellation/references/<organism>/<release>/``.

Cache root resolution (first hit wins):

    1. ``$CONSTELLATION_REFERENCES_HOME``
    2. ``$XDG_DATA_HOME/constellation/references/`` (when ``$XDG_DATA_HOME`` is set)
    3. ``~/.constellation/references/``

Not ``~/.cache/`` — reference downloads are slow and remote-availability-
dependent; ``XDG_CACHE_HOME`` semantics ("OS may delete this") are wrong.
The viz-frontend cache at ``~/.cache/constellation/viz-frontend/`` is
genuinely re-fetchable from a GitHub release on demand; reference genomes
aren't.

Handle format::

    <organism_slug>@<source>-<release>   # fully pinned, reproducible
    <organism_slug>                       # resolved via defaults.toml or current/

``organism_slug`` is lowercase underscore-joined (``homo_sapiens``,
``saccharomyces_cerevisiae``). Sources are ``ensembl`` |
``ensembl_genomes`` | ``refseq`` | ``genbank`` | ``local_import``. The
release portion pins both assembly AND annotation — for NCBI this is a
composite ``<accession>-ar<annotation_release>`` because the assembly
accession alone does not pin the GFF (the RefSeq annotation pipeline
versions independently).
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None


# ──────────────────────────────────────────────────────────────────────
# Cache root resolution
# ──────────────────────────────────────────────────────────────────────


_CACHE_ENV_VAR = "CONSTELLATION_REFERENCES_HOME"
_XDG_ENV_VAR = "XDG_DATA_HOME"
_HOME_FALLBACK = ".constellation/references"

VALID_SOURCES: frozenset[str] = frozenset(
    {"ensembl", "ensembl_genomes", "refseq", "genbank", "local_import"}
)


def cache_root() -> Path:
    """Resolve the active reference-cache root."""
    env = os.environ.get(_CACHE_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get(_XDG_ENV_VAR)
    if xdg:
        return (Path(xdg).expanduser() / "constellation" / "references").resolve()
    return (Path.home() / _HOME_FALLBACK).resolve()


# ──────────────────────────────────────────────────────────────────────
# Handle parsing
# ──────────────────────────────────────────────────────────────────────


_ORGANISM_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_RELEASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class Handle:
    """A parsed reference handle.

    Fully-qualified handles carry both ``source`` and ``release``. Bare
    organism handles (``homo_sapiens`` alone) leave both as ``None`` —
    the resolver fills them in from ``defaults.toml`` or the ``current``
    symlink.
    """

    organism: str
    source: str | None = None
    release: str | None = None

    def is_qualified(self) -> bool:
        return self.source is not None and self.release is not None

    def release_slug(self) -> str:
        """The ``<source>-<release>`` directory name under ``<organism>/``."""
        if not self.is_qualified():
            raise ValueError(
                f"release_slug() called on unqualified handle {self!r}; "
                "resolve() the handle first"
            )
        return f"{self.source}-{self.release}"

    def __str__(self) -> str:
        if self.is_qualified():
            return f"{self.organism}@{self.release_slug()}"
        return self.organism


def parse_handle(s: str) -> Handle:
    """Parse a handle string into a ``Handle`` dataclass.

    Accepts ``<organism>``, ``<organism>@<source>-<release>``, and the
    release-only form ``<source>-<release>`` for the
    ``reference default <organism> <release-or-handle>`` CLI shape (the
    organism is supplied separately in that caller).
    """
    s = s.strip()
    if not s:
        raise ValueError("empty handle")
    if "@" not in s:
        if not _ORGANISM_RE.match(s):
            raise ValueError(
                f"invalid organism slug {s!r}; expected lowercase "
                "alphanumeric+underscore, leading letter"
            )
        return Handle(organism=s)
    organism, rest = s.split("@", 1)
    if not _ORGANISM_RE.match(organism):
        raise ValueError(f"invalid organism slug in handle {s!r}")
    source, release = _split_release_slug(rest, full_handle=s)
    return Handle(organism=organism, source=source, release=release)


def parse_release_slug(s: str, *, organism: str) -> Handle:
    """Parse ``<source>-<release>`` (no ``<organism>@``) into a Handle.

    Used by ``reference default <organism> <release-or-handle>`` where
    the organism slug is provided positionally and the user typed only
    the release portion (e.g. ``ensembl-111``).
    """
    if "@" in s:
        h = parse_handle(s)
        if h.organism != organism:
            raise ValueError(
                f"handle organism {h.organism!r} does not match positional "
                f"argument {organism!r}"
            )
        return h
    source, release = _split_release_slug(s, full_handle=f"{organism}@{s}")
    return Handle(organism=organism, source=source, release=release)


def _split_release_slug(rest: str, *, full_handle: str) -> tuple[str, str]:
    """Split ``<source>-<release>`` into its parts."""
    # First '-' splits source from release. Sources are atomic identifiers
    # (no '-' inside), but release identifiers can contain '-' (e.g.
    # NCBI's GCF_000001635.27-ar110), so we split on the first '-' only.
    if "-" not in rest:
        raise ValueError(
            f"invalid handle release portion {rest!r} in {full_handle!r}; "
            f"expected <source>-<release> (e.g. 'ensembl-111')"
        )
    source, release = rest.split("-", 1)
    source = source.strip()
    release = release.strip()
    if source not in VALID_SOURCES:
        raise ValueError(
            f"unknown source {source!r} in handle {full_handle!r}; "
            f"supported: {sorted(VALID_SOURCES)}"
        )
    if not _RELEASE_RE.match(release):
        raise ValueError(
            f"invalid release identifier {release!r} in {full_handle!r}"
        )
    return source, release


# ──────────────────────────────────────────────────────────────────────
# meta.toml + defaults.toml schemas
# ──────────────────────────────────────────────────────────────────────


META_FILENAME = "meta.toml"
DEFAULTS_FILENAME = "defaults.toml"
CURRENT_SYMLINK = "current"
CURRENT_TEXTFILE = "current.txt"  # WSL/Windows fallback when symlinks unavailable
META_SCHEMA_VERSION = 1
DEFAULTS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class InstalledReference:
    """One installed (organism, release) entry in the cache."""

    handle: str
    organism: str
    release_slug: str  # authoritative — matches the on-disk directory name
    source: str
    release: str
    path: Path
    assembly_accession: str | None = None
    assembly_name: str | None = None
    annotation_release: str | None = None
    fetched_at: str | None = None
    size_bytes: int | None = None

    def is_default(self, defaults: dict[str, str]) -> bool:
        target = defaults.get(self.organism)
        if not target:
            return False
        return target == self.release_slug or target == self.handle


def write_meta_toml(
    release_dir: Path,
    *,
    handle: Handle,
    assembly_accession: str | None,
    assembly_name: str | None,
    annotation_release: str | None,
    constellation_version: str,
    urls: dict[str, dict[str, Any]],
    sha256: dict[str, str],
    source_checksum_verified: bool,
    fetched_at: str | None = None,
) -> None:
    """Write a ``meta.toml`` at the cache-release-dir layer.

    The TOML lines below are intentionally hand-rendered: stdlib doesn't
    ship a TOML writer (only a reader in ``tomllib``), and the schema is
    static enough that a serialization library would be overkill. Stays
    valid against ``tomllib``'s parser.
    """
    if not handle.is_qualified():
        raise ValueError(f"meta.toml requires a qualified handle; got {handle!r}")
    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = [
        f"schema_version = {META_SCHEMA_VERSION}",
        f'handle = "{handle}"',
        f'organism = "{handle.organism}"',
        f'source = "{handle.source}"',
        f'release = "{handle.release}"',
    ]
    if assembly_accession is not None:
        lines.append(f'assembly_accession = "{assembly_accession}"')
    if assembly_name is not None:
        lines.append(f'assembly_name = "{assembly_name}"')
    if annotation_release is not None:
        lines.append(f'annotation_release = "{annotation_release}"')
    lines.append(f'constellation_version = "{constellation_version}"')
    lines.append(f'fetched_at = "{fetched_at}"')

    for url_kind, url_info in urls.items():
        lines.append("")
        lines.append(f"[urls.{url_kind}]")
        for key in ("url", "etag", "last_modified"):
            value = url_info.get(key)
            if value is None:
                continue
            lines.append(f'{key} = "{_toml_escape(str(value))}"')

    if sha256:
        lines.append("")
        lines.append("[sha256]")
        for key, value in sorted(sha256.items()):
            lines.append(f'{key} = "{value}"')

    lines.append("")
    lines.append("[verification]")
    lines.append(
        f"source_checksum_verified = {'true' if source_checksum_verified else 'false'}"
    )

    (release_dir / META_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _toml_escape(s: str) -> str:
    """Escape a string for inclusion in a TOML basic string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def read_meta_toml(release_dir: Path) -> dict[str, Any] | None:
    """Read ``<release_dir>/meta.toml``; return ``None`` if missing."""
    path = release_dir / META_FILENAME
    if not path.exists():
        return None
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    schema_version = int(raw.get("schema_version", 1))
    if schema_version != META_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported meta.toml schema_version={schema_version} at {path}; "
            f"this constellation supports v{META_SCHEMA_VERSION}"
        )
    return raw


@dataclass
class _DefaultsFile:
    """In-memory view of ``defaults.toml`` — used by the CLI verbs."""

    schema_version: int = DEFAULTS_SCHEMA_VERSION
    defaults: dict[str, str] = field(default_factory=dict)


def read_defaults(root: Path | None = None) -> dict[str, str]:
    """Read ``defaults.toml`` and return ``{organism: release_slug}``.

    Returns an empty dict if the file is absent. Unknown ``schema_version``
    raises ``ValueError``.
    """
    if root is None:
        root = cache_root()
    path = root / DEFAULTS_FILENAME
    if not path.exists():
        return {}
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    schema_version = int(raw.get("schema_version", 1))
    if schema_version != DEFAULTS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported defaults.toml schema_version={schema_version} at {path}; "
            f"this constellation supports v{DEFAULTS_SCHEMA_VERSION}"
        )
    block = raw.get("defaults", {}) or {}
    out: dict[str, str] = {}
    for organism, release_slug in block.items():
        if not isinstance(organism, str) or not isinstance(release_slug, str):
            continue
        out[organism] = release_slug
    return out


def write_defaults(defaults: dict[str, str], *, root: Path | None = None) -> None:
    """Write ``defaults.toml`` (overwriting any existing file)."""
    if root is None:
        root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    lines = [f"schema_version = {DEFAULTS_SCHEMA_VERSION}", "", "[defaults]"]
    for organism in sorted(defaults):
        lines.append(f'{organism} = "{defaults[organism]}"')
    (root / DEFAULTS_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_default(organism: str, release_slug: str, *, root: Path | None = None) -> None:
    """Pin ``<organism> -> <release_slug>`` in ``defaults.toml``."""
    current = read_defaults(root)
    current[organism] = release_slug
    write_defaults(current, root=root)


def unset_default(organism: str, *, root: Path | None = None) -> bool:
    """Remove a default entry; return True iff one was actually removed."""
    current = read_defaults(root)
    if organism not in current:
        return False
    del current[organism]
    write_defaults(current, root=root)
    return True


# ──────────────────────────────────────────────────────────────────────
# Resolution
# ──────────────────────────────────────────────────────────────────────


class ReferenceNotInstalledError(KeyError):
    """Raised when a handle cannot be resolved to an on-disk cache entry."""


def resolve(handle: Handle | str, *, root: Path | None = None) -> Path:
    """Resolve a handle to the absolute path of its release directory.

    Precedence chain for bare ``<organism>`` handles:

        1. ``defaults.toml`` entry for that organism.
        2. ``<organism>/current`` symlink (or ``current.txt`` on WSL).
        3. Single installed release for that organism.
        4. Raise ``ReferenceNotInstalledError`` with a "did you mean" list.

    Fully-qualified handles (``<organism>@<source>-<release>``) always
    bypass defaults.toml — defaults never override an explicit pin.
    """
    if isinstance(handle, str):
        handle = parse_handle(handle)
    if root is None:
        root = cache_root()

    organism_dir = root / handle.organism
    if not organism_dir.is_dir():
        raise ReferenceNotInstalledError(
            f"no cached reference for organism {handle.organism!r} under {root}; "
            f"run: constellation reference fetch <source>:<id>"
        )

    if handle.is_qualified():
        target = organism_dir / handle.release_slug()
        if not target.is_dir():
            installed = _installed_release_slugs(organism_dir)
            raise ReferenceNotInstalledError(
                f"no cache entry {target}; installed for {handle.organism!r}: "
                f"{sorted(installed) or '<none>'}"
            )
        return target.resolve()

    # Bare organism — walk the precedence chain.
    defaults = read_defaults(root)
    pinned = defaults.get(handle.organism)
    if pinned:
        target = organism_dir / pinned
        if target.is_dir():
            return target.resolve()
        # Defaults references a release that was deleted out from under us.
        # Fall through to current/single-install rules, but mention it.
        # (The 'did you mean' error below carries enough info.)

    current_target = _read_current(organism_dir)
    if current_target is not None:
        return current_target.resolve()

    installed = _installed_release_slugs(organism_dir)
    if len(installed) == 1:
        only = next(iter(installed))
        return (organism_dir / only).resolve()

    msg = (
        f"ambiguous handle {handle.organism!r}: "
        f"{len(installed)} releases installed and no default/current set. "
        f"Installed: {sorted(installed)}. "
        f"Pin one with: constellation reference default {handle.organism} <release-slug>"
    )
    if not installed:
        msg = (
            f"no cache entry for {handle.organism!r}; run: "
            f"constellation reference fetch <source>:<id>"
        )
    raise ReferenceNotInstalledError(msg)


def _read_current(organism_dir: Path) -> Path | None:
    """Return the release dir pointed at by ``current`` symlink or
    ``current.txt`` fallback; ``None`` if neither resolves."""
    symlink = organism_dir / CURRENT_SYMLINK
    if symlink.is_symlink() or symlink.exists():
        target = symlink.resolve() if symlink.is_symlink() else symlink
        if target.is_dir():
            return target
    textfile = organism_dir / CURRENT_TEXTFILE
    if textfile.exists():
        slug = textfile.read_text(encoding="utf-8").strip()
        if slug:
            target = organism_dir / slug
            if target.is_dir():
                return target
    return None


def _installed_release_slugs(organism_dir: Path) -> set[str]:
    """Return the set of release-slug subdirs under ``<organism>/`` —
    excluding the ``current`` symlink, ``current.txt`` fallback, and
    any ``.partial`` / ``.lock`` scratch artifacts."""
    out: set[str] = set()
    for entry in organism_dir.iterdir():
        name = entry.name
        if name in (CURRENT_SYMLINK, CURRENT_TEXTFILE):
            continue
        if name.endswith(".partial") or name.endswith(".lock"):
            continue
        if not entry.is_dir() or entry.is_symlink():
            continue
        out.add(name)
    return out


# ──────────────────────────────────────────────────────────────────────
# current/ symlink (with WSL fallback)
# ──────────────────────────────────────────────────────────────────────


def update_current_pointer(organism_dir: Path, release_slug: str) -> str:
    """Update the ``current`` pointer atomically; return its on-disk
    form (``"symlink"`` or ``"textfile"``).

    Tries ``os.symlink`` first; falls back to a ``current.txt`` file
    when the platform refuses (Windows native paths without dev mode,
    permission errors on some WSL mounts)."""
    organism_dir.mkdir(parents=True, exist_ok=True)
    symlink_path = organism_dir / CURRENT_SYMLINK
    tmp_path = organism_dir / f"{CURRENT_SYMLINK}.tmp"

    # Clean up any leftover tmp pointer from a prior crash.
    if tmp_path.exists() or tmp_path.is_symlink():
        tmp_path.unlink()

    try:
        os.symlink(release_slug, tmp_path)
        os.replace(tmp_path, symlink_path)
        # If we successfully wrote a symlink, remove the textfile
        # fallback so the two pointer forms never disagree.
        textfile_path = organism_dir / CURRENT_TEXTFILE
        if textfile_path.exists():
            textfile_path.unlink()
        return "symlink"
    except (OSError, NotImplementedError) as exc:
        # Treat any symlink-related OS error as a signal to fall back —
        # don't try to enumerate every (Windows | WSL | network-mount)
        # error code, just respond to the failure.
        if isinstance(exc, OSError) and exc.errno not in {
            errno.EPERM,
            errno.EACCES,
            errno.EOPNOTSUPP,
            errno.ENOSYS,
        }:
            raise
        textfile_path = organism_dir / CURRENT_TEXTFILE
        textfile_path.write_text(release_slug + "\n", encoding="utf-8")
        # Best-effort cleanup of a stale symlink the fallback now supersedes.
        if symlink_path.is_symlink() or symlink_path.exists():
            try:
                symlink_path.unlink()
            except OSError:
                pass
        return "textfile"


# ──────────────────────────────────────────────────────────────────────
# Fetch lock + .partial atomic rename
# ──────────────────────────────────────────────────────────────────────


@contextmanager
def acquire_fetch_lock(handle: Handle, *, root: Path | None = None) -> Iterator[Path]:
    """Acquire a per-release fetch lock; blocks concurrent invocations.

    Uses ``fcntl.flock`` on POSIX (Linux + WSL2 + macOS); on platforms
    without ``fcntl`` (Windows native), degrades to a best-effort
    presence check — concurrent fetches on Windows are rare enough that
    a hard mutex isn't justified.
    """
    if not handle.is_qualified():
        raise ValueError(f"fetch lock requires a qualified handle; got {handle!r}")
    if root is None:
        root = cache_root()
    organism_dir = root / handle.organism
    organism_dir.mkdir(parents=True, exist_ok=True)
    lock_path = organism_dir / f"{handle.release_slug()}.lock"

    if _fcntl is None:
        # Windows fallback — write a sentinel file; trust the user not to
        # spawn two concurrent fetches on Windows.
        lock_path.touch()
        try:
            yield lock_path
        finally:
            if lock_path.exists():
                lock_path.unlink()
        return

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
        yield lock_path
    finally:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        finally:
            os.close(fd)
            # Lock file is left behind intentionally — keeps the inode
            # stable for any blocked-but-not-yet-resumed waiters.


def partial_dir(release_dir: Path) -> Path:
    """The sibling scratch directory ``<release_dir>.partial`` used for
    atomic-write staging."""
    return release_dir.with_name(release_dir.name + ".partial")


def clean_partial(release_dir: Path) -> None:
    """Remove a leftover ``.partial`` scratch dir from a prior crashed fetch."""
    p = partial_dir(release_dir)
    if p.exists():
        shutil.rmtree(p)


def promote_partial(release_dir: Path) -> None:
    """Atomically rename ``<release>.partial/`` → ``<release>/``."""
    src = partial_dir(release_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"no .partial dir to promote: {src}")
    if release_dir.exists():
        # Caller is responsible for handling --force / idempotency; we
        # never silently overwrite an existing release directory.
        raise FileExistsError(
            f"release dir already exists: {release_dir}; "
            f"caller must handle conflict (e.g. --force)"
        )
    os.rename(src, release_dir)


# ──────────────────────────────────────────────────────────────────────
# Cache enumeration
# ──────────────────────────────────────────────────────────────────────


def list_installed(*, root: Path | None = None) -> list[InstalledReference]:
    """Walk the cache and return every installed (organism, release)."""
    if root is None:
        root = cache_root()
    if not root.is_dir():
        return []
    out: list[InstalledReference] = []
    for organism_dir in sorted(root.iterdir()):
        if not organism_dir.is_dir():
            continue
        organism = organism_dir.name
        if not _ORGANISM_RE.match(organism):
            continue
        for release_slug in sorted(_installed_release_slugs(organism_dir)):
            release_dir = organism_dir / release_slug
            meta = read_meta_toml(release_dir)
            if meta is not None and meta.get("source") and meta.get("release"):
                source = str(meta["source"])
                release = str(meta["release"])
            else:
                # Stub entry — directory exists but meta.toml missing or
                # incomplete. Best-effort split of the dir name.
                if "-" in release_slug:
                    source, release = release_slug.split("-", 1)
                else:
                    source, release = release_slug, ""
            entry = InstalledReference(
                handle=f"{organism}@{release_slug}",
                organism=organism,
                release_slug=release_slug,
                source=source,
                release=release,
                path=release_dir.resolve(),
                assembly_accession=_str_or_none(meta, "assembly_accession"),
                assembly_name=_str_or_none(meta, "assembly_name"),
                annotation_release=_str_or_none(meta, "annotation_release"),
                fetched_at=_str_or_none(meta, "fetched_at"),
                size_bytes=_dir_size(release_dir),
            )
            out.append(entry)
    return out


def _str_or_none(meta: dict[str, Any] | None, key: str) -> str | None:
    if not meta:
        return None
    value = meta.get(key)
    return str(value) if value is not None else None


def _dir_size(path: Path) -> int:
    """Sum of file sizes under ``path``; silent on transient errors."""
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def format_size(n_bytes: int) -> str:
    """Render a byte count as ``"3.2 GB"`` / ``"812 MB"`` / ``"42 KB"``."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n_bytes /= 1024
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
    return f"{n_bytes:.1f} PB"


__all__ = [
    "Handle",
    "InstalledReference",
    "ReferenceNotInstalledError",
    "VALID_SOURCES",
    "acquire_fetch_lock",
    "cache_root",
    "clean_partial",
    "format_size",
    "list_installed",
    "parse_handle",
    "parse_release_slug",
    "partial_dir",
    "promote_partial",
    "read_defaults",
    "read_meta_toml",
    "resolve",
    "set_default",
    "unset_default",
    "update_current_pointer",
    "write_defaults",
    "write_meta_toml",
]
