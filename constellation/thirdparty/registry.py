"""Generic third-party-tool discovery.

Unifies Cartographer's per-tool ``$CARTOGRAPHER_<TOOL>_{JAR,DIR,HOME}``
mix under a single env-var convention (``$CONSTELLATION_<TOOL>_HOME``)
and a single resolution cascade:

    1. env var          ($CONSTELLATION_<TOOL>_HOME)
    2. versioned layout (<repo>/third_party/<tool>/current/)
    3. $PATH lookup     (shutil.which — for conda/system installs)
    4. raise            (ToolNotFoundError with install hint)

A tool registers itself by declaring what a valid install looks like
(primary artifact path under HOME, optional version probe). Adapters
in ``constellation/thirdparty/<tool>.py`` call ``register(...)`` at
import time; ``find(name)`` resolves at call time.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


_REPO_ROOT = Path(__file__).resolve().parents[2]  # constellation/../.. -> repo root
_THIRD_PARTY_ROOT = _REPO_ROOT / "third_party"


class ToolNotFoundError(RuntimeError):
    """Raised when a registered third-party tool cannot be located."""


@dataclass(frozen=True)
class ToolSpec:
    """Contract an adapter declares when registering a tool.

    - ``name``           — short tool identifier (``"encyclopedia"``, ``"thermo"``, ``"mmseqs2"``).
    - ``env_var``        — primary env-var override (``$CONSTELLATION_<NAME>_HOME``).
    - ``artifact``       — path(s) relative to HOME of the file/dir that "proves" a valid install.
                           Accept a single string for the common case; pass a tuple of strings
                           to declare multiple candidate filenames (e.g. across tool versions
                           that renamed the jar — first existing candidate wins).
                           ``None`` means the HOME directory itself is the artifact.
    - ``path_bin``       — optional binary name to look up on ``$PATH`` if HOME resolution fails
                           (e.g. ``"mmseqs"`` for conda-installed mmseqs2).
    - ``install_script`` — path (relative to repo root) of the installer script shown in
                           error messages — e.g. ``"scripts/install-encyclopedia.sh"``.
    - ``version_probe``  — optional callable that takes the resolved path and returns a
                           version string (``"1.12.34"``) for compatibility checks.
    """

    name: str
    env_var: str
    artifact: str | tuple[str, ...] | None = None
    path_bin: str | None = None
    install_script: str | None = None
    version_probe: Callable[[Path], str | None] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ToolHandle:
    """Resolved location for a third-party tool."""

    spec: ToolSpec
    path: Path
    source: str  # "env", "versioned", or "path"
    version: str | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.spec.name} @ {self.path} ({self.source}, {self.version or 'unknown version'})"


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    """Register a tool spec. Called by adapters at import time."""
    _REGISTRY[spec.name] = spec


def registered() -> list[ToolSpec]:
    """Snapshot of currently registered tools (for the doctor CLI)."""
    return list(_REGISTRY.values())


def _env_home(spec: ToolSpec) -> Path | None:
    raw = os.environ.get(spec.env_var)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _versioned_home(spec: ToolSpec) -> Path | None:
    # <repo>/third_party/<tool>/current → symlink-or-dir created by install scripts.
    candidate = _THIRD_PARTY_ROOT / spec.name / "current"
    if not candidate.exists():
        return None
    return candidate.resolve()


def _artifact_paths(home: Path, spec: ToolSpec) -> list[Path]:
    """Candidate artifact paths under ``home``.

    Single-string ``artifact`` → one candidate; tuple → one candidate
    per entry in declaration order (first existing wins). ``None``
    means the HOME directory itself is the artifact.
    """
    if spec.artifact is None:
        return [home]
    if isinstance(spec.artifact, str):
        return [home / spec.artifact]
    return [home / a for a in spec.artifact]


def _probe_version(spec: ToolSpec, path: Path) -> str | None:
    if spec.version_probe is None:
        return None
    try:
        return spec.version_probe(path)
    except Exception:  # noqa: BLE001 - adapters should be tolerant here
        return None


def find(name: str) -> ToolHandle:
    """Resolve a tool to a concrete path, or raise ToolNotFoundError."""
    if name not in _REGISTRY:
        raise KeyError(f"tool {name!r} is not registered in constellation.thirdparty")
    spec = _REGISTRY[name]

    # 1. env var
    home = _env_home(spec)
    if home is not None:
        for artifact in _artifact_paths(home, spec):
            if artifact.exists():
                return ToolHandle(spec, artifact, "env", _probe_version(spec, artifact))

    # 2. versioned on-disk layout
    home = _versioned_home(spec)
    if home is not None:
        for artifact in _artifact_paths(home, spec):
            if artifact.exists():
                return ToolHandle(spec, artifact, "versioned", _probe_version(spec, artifact))

    # 3. $PATH lookup (conda / system binaries)
    if spec.path_bin:
        resolved = shutil.which(spec.path_bin)
        if resolved:
            path = Path(resolved)
            return ToolHandle(spec, path, "path", _probe_version(spec, path))

    # 4. raise with install hint
    hint = (
        f"\n  set {spec.env_var} to an existing install, or"
        f"\n  run `bash {spec.install_script}` from the repo root"
        if spec.install_script
        else f"\n  set {spec.env_var} to point at an existing install"
    )
    raise ToolNotFoundError(f"third-party tool not found: {spec.name}{hint}")


def try_find(name: str) -> ToolHandle | None:
    """Like ``find`` but returns ``None`` instead of raising — for the doctor CLI."""
    try:
        return find(name)
    except (ToolNotFoundError, KeyError):
        return None
