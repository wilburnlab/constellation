""".NET runtime + Thermo CommonCore DLL bootstrap.

Thermo ``.raw`` support requires three pieces:

1. ``pythonnet`` (pip, inside the conda env) for .NET CLR interop.
2. .NET 8 runtime (conda-forge ``dotnet-runtime=8.0.*``, pulled in by
   ``environment.yml``).
3. Thermo CommonCore DLLs — fetched separately via
   ``bash scripts/install-thermo-dlls.sh`` because they're proprietary.

DLL location resolves through the standard thirdparty registry
(``constellation/thirdparty/thermo.py`` registers a ``ToolSpec`` with
``$CONSTELLATION_THERMO_HOME`` as the env override and
``third_party/thermo/current/`` as the on-repo default).

``load_clr()`` forces pythonnet to use CoreCLR (not Mono, which is
pythonnet's default on Linux) and points ``$DOTNET_ROOT`` at the
conda-installed runtime when the user hasn't set it explicitly.

Call ``load_clr()`` before importing any ``ThermoFisher.CommonCore.*``
namespace. It is idempotent.

A subtlety: ``OpenMcdf.dll`` is **bundled and AddReference'd** but
Constellation never decodes a method blob through it. CommonCore.RawFileReader
lazy-loads OpenMcdf internally to back its compound-document
method-storage path — without OpenMcdf on disk you get a missing-assembly
error inside a later API call, even from code paths that look unrelated.
We read only ``SampleInformation.InstrumentMethodFile`` (the bare
filename); the in-``.raw`` method-text decode via ``GetInstrumentMethod(i)``
returns the literal string ``"None"`` on some firmware and isn't wired.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# CommonCore DLL names actually loaded by the reader. Matches
# ms_deisotope's minimum surface. Other CommonCore DLLs
# (BackgroundSubtraction, MassPrecisionEstimator) are unused by the
# scan-iteration path and not shipped in the upstream
# ThermoRawFileParser artifact we fetch.
REQUIRED_DLLS: tuple[str, ...] = (
    "ThermoFisher.CommonCore.Data.dll",
    "ThermoFisher.CommonCore.RawFileReader.dll",
    "OpenMcdf.dll",
)

_INSTALL_HINT = (
    "Update the conda env with `conda env update -f environment.yml --prune` "
    "(pulls in dotnet-runtime + pythonnet), then fetch the Thermo CommonCore "
    "DLLs with `bash scripts/install-thermo-dlls.sh`. Override the DLL "
    "location with $CONSTELLATION_THERMO_HOME."
)


def get_dll_dir() -> Path | None:
    """Return the directory expected to contain Thermo CommonCore DLLs.

    Consults the thirdparty registry first (matching every other tool
    Constellation wraps) so users get one uniform discovery path.
    Returns ``None`` when the registry can't find the install at all
    — callers translate that into ``ImportError`` via :func:`require_thermo`.
    """
    # Local imports keep this module a leaf — the thirdparty adapter
    # imports this module from inside its own module body for the CLR
    # bootstrap and we want to avoid an import cycle.
    from constellation.thirdparty import registry

    handle = registry.try_find("thermo")
    if handle is None:
        return None
    # Registered artifact is a file inside the DLL dir; the dir is its parent.
    artifact = handle.path
    return artifact.parent if artifact.is_file() else artifact


def _missing_dlls(dll_dir: Path) -> list[str]:
    return [name for name in REQUIRED_DLLS if not (dll_dir / name).exists()]


def is_thermo_available() -> bool:
    """Return ``True`` if pythonnet is importable AND all DLLs are present.

    Does not load the CLR or reference the DLLs — purely a file-presence
    check. Safe to call from module scope.
    """
    try:
        import pythonnet  # noqa: F401
    except ImportError:
        return False
    dll_dir = get_dll_dir()
    if dll_dir is None:
        return False
    return not _missing_dlls(dll_dir)


def require_thermo() -> None:
    """Raise ``ImportError`` with install guidance if Thermo support is unavailable."""
    try:
        import pythonnet  # noqa: F401
    except ImportError as e:
        raise ImportError("pythonnet is not installed. " + _INSTALL_HINT) from e
    dll_dir = get_dll_dir()
    if dll_dir is None:
        raise ImportError(
            "Thermo CommonCore DLLs are not installed. " + _INSTALL_HINT
        )
    missing = _missing_dlls(dll_dir)
    if missing:
        raise ImportError(
            f"Thermo CommonCore DLLs missing from {dll_dir}: "
            f"{', '.join(missing)}. {_INSTALL_HINT}"
        )


# ── .NET runtime discovery ────────────────────────────────────────────


def _find_dotnet_root() -> Path | None:
    """Locate the .NET runtime in the active conda env (or PATH).

    Resolution order:

    1. ``$DOTNET_ROOT`` env var (user override — unchanged if set).
    2. The ``dotnet`` binary's parent directory on ``$PATH`` (works for
       the conda-forge ``dotnet-runtime`` package, which puts ``dotnet``
       on ``$CONDA_PREFIX/bin``; the actual runtime lives alongside it).
    3. Common conda-env layouts: ``$CONDA_PREFIX/lib/dotnet``,
       ``$CONDA_PREFIX/share/dotnet``.

    Returns the resolved directory or ``None`` if .NET isn't findable.
    """
    existing = os.environ.get("DOTNET_ROOT")
    if existing:
        p = Path(existing)
        if p.is_dir():
            return p

    dotnet_bin = shutil.which("dotnet")
    if dotnet_bin:
        resolved = Path(dotnet_bin).resolve()
        candidate = resolved.parent
        if (candidate / "dotnet").exists():
            return candidate

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for rel in ("lib/dotnet", "share/dotnet"):
            p = Path(conda_prefix) / rel
            if p.is_dir():
                return p

    return None


def _configure_runtime_env() -> None:
    """Point pythonnet at CoreCLR + the conda-installed runtime.

    Sets ``PYTHONNET_RUNTIME=coreclr`` (pythonnet's default on Linux is
    Mono, which isn't installed in the conda env) and ``DOTNET_ROOT``
    to the conda runtime path when unset. User-set values are respected.
    """
    os.environ.setdefault("PYTHONNET_RUNTIME", "coreclr")

    if not os.environ.get("DOTNET_ROOT"):
        dotnet_root = _find_dotnet_root()
        if dotnet_root is not None:
            os.environ["DOTNET_ROOT"] = str(dotnet_root)
            logger.debug("Set DOTNET_ROOT=%s", dotnet_root)
        else:
            logger.warning(
                ".NET runtime not found on PATH or in $CONDA_PREFIX; "
                "pythonnet will likely fail to load. Run "
                "`conda env update -f environment.yml --prune` to install "
                "dotnet-runtime into the active env."
            )


_clr_loaded: bool = False


def load_clr() -> None:
    """Idempotently load the .NET CLR and register Thermo CommonCore DLLs.

    After this returns, the following namespaces are importable:

    * ``ThermoFisher.CommonCore.Data.Business``
    * ``ThermoFisher.CommonCore.RawFileReader``
    * ``System.Runtime.InteropServices`` (for ``Marshal.Copy`` fallback)

    Raises ``ImportError`` via :func:`require_thermo` if prerequisites
    are missing.
    """
    global _clr_loaded
    if _clr_loaded:
        return
    require_thermo()

    # Must be set before ``import clr`` — pythonnet reads these env vars
    # when its runtime loader fires on first import.
    _configure_runtime_env()

    dll_dir = get_dll_dir()
    assert dll_dir is not None  # require_thermo would have raised
    # pythonnet resolves assembly names against sys.path when using bare
    # assembly names (no .dll extension).
    dll_dir_str = str(dll_dir)
    if dll_dir_str not in sys.path:
        sys.path.insert(0, dll_dir_str)

    import clr  # provided by pythonnet

    # Core runtime references needed for the Marshal.Copy zero-copy fallback.
    clr.AddReference("System.Runtime")
    clr.AddReference("System.Runtime.InteropServices")

    # Thermo references — bare assembly names, resolved from sys.path.
    # OpenMcdf is included in REQUIRED_DLLS because CommonCore pulls it in
    # lazily to parse instrument-method storage. Adding the reference
    # eagerly surfaces any DLL problem at load_clr() time rather than deep
    # inside a later GetInstrumentMethod() call.
    for dll in REQUIRED_DLLS:
        clr.AddReference(dll.removesuffix(".dll"))

    _clr_loaded = True


__all__ = [
    "REQUIRED_DLLS",
    "get_dll_dir",
    "is_thermo_available",
    "load_clr",
    "require_thermo",
]
