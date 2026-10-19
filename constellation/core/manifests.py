"""Typed ``manifest.json`` files that say what a directory is.

Every directory a Constellation command produces carries a
``manifest.json``. Two fields are common to all of them and make a
directory self-describing without knowing which command wrote it:

``kind``
    what this directory is (``"massspec_search"``, ...). A reader asks
    for the kind first and dispatches on it.
``schema_version``
    the version of that kind's fields. A reader that meets a version it
    does not know says so, instead of guessing at the fields.

A kind is a frozen dataclass deriving from :class:`Manifest`::

    @dataclass(frozen=True)
    class SearchManifest(Manifest):
        KIND = "massspec_search"
        SCHEMA_VERSION = 1
        engine: str = ""
        outputs: dict[str, str] = field(default_factory=dict)

Defining the class registers the kind. Fields must be JSON types
(strings, numbers, booleans, ``None``, lists and string-keyed dicts of
those); every field needs a default, so that a manifest written before a
field was added still loads. Adding a field with a default is not a
version bump; renaming, removing or changing the meaning of one is.

:func:`write_manifest` writes atomically, so a directory never holds a
half-written manifest. :func:`read_manifest` returns the right class for
whatever kind it finds; :func:`manifest_kind` is the cheap probe.

``sequencing.transcriptome.manifest`` has its own, older reader for its
three kinds and is not migrated; the two agree on the ``kind`` /
``schema_version`` / ``created_at`` fields.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, TypeVar

MANIFEST_FILENAME = "manifest.json"

_REGISTRY: dict[str, type[Manifest]] = {}

M = TypeVar("M", bound="Manifest")


class ManifestError(ValueError):
    """A manifest is missing, unreadable, of an unknown kind or version,
    or not the kind that was asked for. The message says which, and what
    to do."""


class _NoManifestError(ManifestError):
    """There is no manifest file at all."""


@dataclass(frozen=True)
class Manifest:
    """Base of every typed manifest. Subclasses set ``KIND`` and
    ``SCHEMA_VERSION`` and add fields with defaults."""

    KIND: ClassVar[str] = ""
    SCHEMA_VERSION: ClassVar[int] = 0

    #: UTC, ``YYYY-MM-DDTHH:MM:SSZ``. Filled in by :func:`write_manifest`
    #: when left empty.
    created_at: str = ""
    #: The Constellation version that wrote the directory.
    constellation_version: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = cls.__dict__.get("KIND", "")
        if not kind:
            return  # an intermediate base
        if not isinstance(cls.__dict__.get("SCHEMA_VERSION"), int) or cls.SCHEMA_VERSION < 1:
            raise TypeError(f"{cls.__name__} must set SCHEMA_VERSION to an integer >= 1")
        existing = _REGISTRY.get(kind)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise TypeError(
                f"manifest kind {kind!r} is already registered by {existing.__qualname__}"
            )
        _REGISTRY[kind] = cls

    def to_json(self) -> dict[str, Any]:
        """The JSON document: ``kind`` and ``schema_version`` first."""
        return {
            "kind": self.KIND,
            "schema_version": self.SCHEMA_VERSION,
            **dataclasses.asdict(self),
        }


def registered_kinds() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _manifest_path(location: Path | str) -> Path:
    path = Path(location)
    return path / MANIFEST_FILENAME if path.is_dir() or path.suffix != ".json" else path


def write_manifest(directory: Path | str, manifest: Manifest) -> Path:
    """Write ``manifest`` as ``<directory>/manifest.json`` and return the path.

    Written to a temporary file and renamed, so a reader never sees a
    partial document and an interrupted run leaves the previous one.
    ``created_at`` is filled in when empty.
    """
    if not manifest.KIND:
        raise TypeError(f"{type(manifest).__name__} has no KIND and cannot be written")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not manifest.created_at:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest = dataclasses.replace(manifest, created_at=now)
    path = directory / MANIFEST_FILENAME
    tmp = directory / f".{MANIFEST_FILENAME}.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(manifest.to_json(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _load_json(location: Path | str) -> tuple[Path, dict[str, Any]]:
    path = _manifest_path(location)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _NoManifestError(
            f"no {MANIFEST_FILENAME} at {path}: not a Constellation output directory, "
            f"or the command that writes it did not finish"
        ) from None
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path} is not valid JSON ({exc})") from None
    if not isinstance(raw, dict):
        raise ManifestError(f"{path} does not hold a JSON object")
    return path, raw


def manifest_kind(location: Path | str) -> str | None:
    """The ``kind`` a directory's manifest declares; ``None`` when there
    is no manifest or it declares none (an output from before kinds).
    Never raises for a missing file; does for unreadable JSON."""
    try:
        _path, raw = _load_json(location)
    except _NoManifestError:
        return None
    kind = raw.get("kind")
    return kind if isinstance(kind, str) and kind else None


def read_manifest(location: Path | str, expect: type[M] | None = None) -> M:
    """Read a directory's (or file's) manifest as its typed class.

    ``expect`` names the class the caller needs; any other kind is an
    error that says what was found. Fields the class does not have are
    ignored (a newer writer added them); fields the file lacks take
    their defaults.
    """
    path, raw = _load_json(location)
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ManifestError(
            f"{path} declares no kind; it was written before manifests were typed. "
            f"Re-run the command that produced the directory."
        )
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise ManifestError(
            f"{path} is of kind {kind!r}, which nothing loaded here reads "
            f"(known: {', '.join(registered_kinds()) or 'none'})"
        )
    if expect is not None and cls is not expect:
        raise ManifestError(
            f"{path} is a {kind!r} manifest; a {expect.KIND!r} one was expected"
        )
    version = raw.get("schema_version")
    if version != cls.SCHEMA_VERSION:
        raise ManifestError(
            f"{path} is {kind!r} schema_version {version!r}; this Constellation reads "
            f"version {cls.SCHEMA_VERSION}. Re-run the command that produced the directory."
        )
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})  # type: ignore[return-value]


__all__ = [
    "MANIFEST_FILENAME",
    "Manifest",
    "ManifestError",
    "manifest_kind",
    "read_manifest",
    "registered_kinds",
    "write_manifest",
]
