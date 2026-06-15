"""BUSCO (Benchmarking Universal Single-Copy Orthologs) — ToolSpec.

Standard genome-completeness metric for eukaryotic assemblies. Installed
into a dedicated conda env by ``scripts/install-busco.sh`` (bioconda
bundles BUSCO's external aligners); the registry artifact ``busco`` is a
wrapper that runs the tool inside that env. Lineage data (eukaryota_odb12,
vertebrata_odb12, ...) lives outside this registry — see the BUSCO docs or
the install script's ``--lineage`` pre-stage.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


BUSCO_VERSION = "6.1.0"


def _probe_version(path: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or "") + (result.stderr or "")
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", output)
    return m.group(1) if m else None


register(
    ToolSpec(
        name="busco",
        env_var="CONSTELLATION_BUSCO_HOME",
        artifact="busco",
        path_bin="busco",
        install_script="scripts/install-busco.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["BUSCO_VERSION"]
