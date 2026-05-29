"""Thermo CommonCore DLL adapter — ``ToolSpec`` registration only.

Responsibility is *tool discovery*: tell ``constellation doctor`` and
the Thermo reader how to find the CommonCore DLL pack on disk.
File-format work (scan iteration, trailer promotion, manifest writing)
lives with the rest of the Thermo reader at
:mod:`constellation.massspec.io.thermo` — that's where the
``ThermoReader`` / ``convert`` entry points come from.

Pins: ``scripts/install-thermo-dlls.sh`` hash-pins the **ThermoRawFileParser**
v1.4.5 release archive, which bundles the three DLLs Constellation
loads (``ThermoFisher.CommonCore.Data.dll``,
``ThermoFisher.CommonCore.RawFileReader.dll``, ``OpenMcdf.dll``). The
install script lays them down under
``third_party/thermo/<version>/`` and repoints
``third_party/thermo/current`` so the registry's standard ``current/``
lookup finds them.

A note on the artifact field: the ToolSpec declares the RawFileReader
DLL as the canonical "this install is present" file. The deeper
"all three DLLs are present and loadable" validation happens inside
:func:`constellation.massspec.io.thermo._netruntime.require_thermo`.
This split keeps the doctor's discovery cheap while letting the
reader fail fast with a useful message at first ``.raw`` open.
"""

from __future__ import annotations

from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


# Version of the upstream ThermoRawFileParser release the install
# script downloads. The DLLs themselves come from inside that archive
# (the CommonCore family ships out-of-band of TRFP's own version).
THERMO_DLL_PACK_VERSION = "1.4.5"


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
