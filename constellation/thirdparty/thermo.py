"""Thermo CommonCore DLL adapter — ``ToolSpec`` registration only.

Responsibility is *tool discovery*: tell ``constellation doctor`` and
the Thermo reader how to find the CommonCore DLL pack on disk.
File-format work (scan iteration, trailer promotion, manifest writing)
lives with the rest of the Thermo reader at
:mod:`constellation.massspec.readers.thermo` — that's where the
``ThermoReader`` / ``convert`` entry points come from.

Pins: ``scripts/install-thermo-dlls.sh`` installs Thermo's RawFileReader
8.0.42 for .NET 8 from Thermo's own repository (at a fixed commit, since
Thermo keeps no older packages there) and its two dependencies from
nuget.org, every file SHA-256 pinned; ``--legacy-1.4.5`` installs the
CommonCore 5.0.0.93 bundled with ThermoRawFileParser 1.4.5 instead. The
two give identical peaks and scan values; only 8.x reads the instrument
method text on Linux. The script lays the DLLs down under
``third_party/thermo/<version>/`` and repoints
``third_party/thermo/current`` so the registry's standard ``current/``
lookup finds them.

A note on the artifact field: the ToolSpec declares the RawFileReader
DLL as the canonical "this install is present" file. The deeper
"all three DLLs are present and loadable" validation happens inside
:func:`constellation.massspec.readers.thermo._netruntime.require_thermo`.
This split keeps the doctor's discovery cheap while letting the
reader fail fast with a useful message at first ``.raw`` open.
"""

from __future__ import annotations

from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


# The RawFileReader version the install script installs by default (the
# directory under third_party/thermo/; "1.4.5" for the legacy pack).
THERMO_DLL_PACK_VERSION = "8.0.42"


def _probe_version(path: Path) -> str | None:
    """Return the DLL-pack version inferred from the parent directory name.

    The install script lays the DLLs under
    ``third_party/thermo/<version>/`` and points ``current`` at the
    chosen version. So ``path.parent.parent.name`` is the versioned
    folder when the registry resolved via ``current`` (a symlink),
    or ``path.parent.name`` when ``CONSTELLATION_THERMO_HOME`` points
    directly at the versioned folder. We try the closer parent first.
    """
    candidates = (path.parent.name, path.parent.parent.name)
    for candidate in candidates:
        if candidate and candidate[0].isdigit():
            return candidate
    return None


register(
    ToolSpec(
        name="thermo",
        env_var="CONSTELLATION_THERMO_HOME",
        # The CommonCore RawFileReader DLL is the canonical install
        # marker. ``_netruntime.require_thermo`` performs the deeper
        # "all DLLs present" check at first open.
        artifact="ThermoFisher.CommonCore.RawFileReader.dll",
        path_bin=None,  # .NET DLLs — never found via $PATH
        install_script="scripts/install-thermo-dlls.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["THERMO_DLL_PACK_VERSION"]
