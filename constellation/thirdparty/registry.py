"""Generic third-party-tool discovery.

Unifies Cartographer's per-tool ``$CARTOGRAPHER_<TOOL>_{JAR,DIR,HOME}``
mix under a single env-var convention (``$CONSTELLATION_<TOOL>_HOME``)
and a single resolution cascade:

    1. env var          ($CONSTELLATION_<TOOL>_HOME)
    2. user home cache  (~/.constellation/<tool>/current/ — opt-in per tool)
    3. versioned layout (<repo>/third_party/<tool>/current/)
    4. $PATH lookup     (shutil.which — for conda/system installs)
    5. raise            (ToolNotFoundError with install hint)

The home-cache step (2) is only consulted for tools that declare a
``user_cache_dir`` — heavy app-dir tools (e.g. EncyclopeDIA's install4j
bundle: jar + bundled JRE + native libs) that belong in the per-user
``~/.constellation/`` cache, alongside references/catalogs/taxonomy,
rather than inside the repo's ``third_party/`` tree.

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


def version_tuple(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into an int tuple for ordering.

    ``"6.5.15"`` -> ``(6, 5, 15)``. Tolerant of trailing non-numeric
    junk: each ``.``-separated field contributes its leading run of
    digits, and parsing stops at the first field with no leading digit
    (so ``"6.6-beta"`` -> ``(6, 6)`` and ``"nightly"`` -> ``()``).
    Stdlib only — no ``packaging`` dependency.

    Numeric-tuple comparison is the whole point: lexicographic string
    compare puts ``"6.5.15" < "2.12.30"`` (wrong, because ``"6" < ... no,
    "1" < "6"`` — the field widths differ), but
    ``version_tuple("6.5.15") > version_tuple("2.12.30")`` is ``True``.
    """
    parts: list[int] = []
    for field_str in v.split("."):
        digits = ""
        for ch in field_str:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def version_ge(have: str, minimum: str) -> bool:
    """``True`` when ``have`` >= ``minimum`` by numeric-tuple compare."""
    return version_tuple(have) >= version_tuple(minimum)


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
                           ``None`` means the HOME directory itself is the artifact — UNLESS
                           ``artifact_glob`` is also set, in which case the directory-as-artifact
                           fallback is suppressed (see ``_candidate_paths``).
    - ``artifact_glob``  — optional glob pattern (relative to HOME) for version-suffixed
                           artifacts whose exact filename isn't known ahead of time
                           (e.g. ``"encyclopedia-*.jar"``). Resolved after any exact ``artifact``
                           candidates. Selection among matches is governed by ``pick``.
    - ``pick``           — match-selection policy for ``artifact_glob``: ``"first"`` (lexicographic,
                           the default) or ``"highest"`` (sort by ``version_probe`` and take the
                           newest; un-probeable matches sort last so junk never shadows a real
                           versioned artifact).
    - ``user_cache_dir`` — optional subdirectory name under ``~/.constellation/``. When set, the
                           resolver consults ``~/.constellation/<user_cache_dir>/current/`` (after
                           the env var, before ``third_party/``). For heavy app-dir tools that
                           belong in the per-user home cache rather than the repo tree.
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
    artifact_glob: str | None = None
    pick: str = "first"
    user_cache_dir: str | None = None
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


def _user_cache_home(spec: ToolSpec) -> Path | None:
    # ~/.constellation/<user_cache_dir>/current → symlink-or-dir created by
    # the tool's install script in the per-user home cache. Only consulted
    # when the spec opts in via ``user_cache_dir``. ``Path.home()`` is read
    # at call time (not module load) so tests can override $HOME.
    if spec.user_cache_dir is None:
        return None
    candidate = Path.home() / ".constellation" / spec.user_cache_dir / "current"
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


def _glob_candidates(home: Path, spec: ToolSpec) -> list[Path]:
    """Glob ``home`` for ``artifact_glob`` matches, ordered by ``pick``.

    ``"highest"`` sorts by ``version_probe`` (descending) so the newest
    installed artifact wins; un-probeable matches (``version_probe`` ->
    ``None``) sort last so a junk filename never shadows a real
    versioned one. ``"first"`` returns lexicographic order (cheap,
    stable). ``Path.glob`` only yields extant paths, so every returned
    candidate exists.
    """
    if spec.artifact_glob is None or not home.is_dir():
        return []
    matches = sorted(home.glob(spec.artifact_glob))
    if spec.pick == "highest":
        def _key(p: Path) -> tuple[int, tuple[int, ...]]:
            v = _probe_version(spec, p)
            # (has_version, version_tuple): un-probeable -> (0, ()) sorts
            # first, so reverse=True puts it last.
            return (1, version_tuple(v)) if v else (0, ())
        matches = sorted(matches, key=_key, reverse=True)
    return matches


def _candidate_paths(home: Path, spec: ToolSpec) -> list[Path]:
    """All artifact candidates under ``home``: exact-name first, then glob.

    When ``artifact_glob`` is set and ``artifact`` is ``None``, the
    ``[home]``-is-the-artifact fallback in :func:`_artifact_paths` is
    suppressed — otherwise :func:`find` would return the HOME *directory*
    as the artifact (it exists) before the glob ran, and ``run_jar``
    would try ``java -jar <dir>``.
    """
    if spec.artifact is None and spec.artifact_glob is not None:
        exact: list[Path] = []
    else:
        exact = _artifact_paths(home, spec)
    return exact + _glob_candidates(home, spec)


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
        for artifact in _candidate_paths(home, spec):
            if artifact.exists():
                return ToolHandle(spec, artifact, "env", _probe_version(spec, artifact))

    # 2. per-user home cache (~/.constellation/<tool>/current) — opt-in via
    #    user_cache_dir; sits before third_party so the canonical home-cache
    #    install wins over any legacy repo-local one.
    home = _user_cache_home(spec)
    if home is not None:
        for artifact in _candidate_paths(home, spec):
            if artifact.exists():
                return ToolHandle(spec, artifact, "user_cache", _probe_version(spec, artifact))

    # 3. versioned on-disk layout (<repo>/third_party/<tool>/current)
    home = _versioned_home(spec)
    if home is not None:
        for artifact in _candidate_paths(home, spec):
            if artifact.exists():
                return ToolHandle(spec, artifact, "versioned", _probe_version(spec, artifact))

    # 4. $PATH lookup (conda / system binaries)
    if spec.path_bin:
        resolved = shutil.which(spec.path_bin)
        if resolved:
            path = Path(resolved)
            return ToolHandle(spec, path, "path", _probe_version(spec, path))

    # 5. raise with install hint
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
