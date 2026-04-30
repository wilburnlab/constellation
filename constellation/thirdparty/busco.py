"""BUSCO (Benchmarking Universal Single-Copy Orthologs) — ToolSpec.

Standard genome-completeness metric for eukaryotic assemblies.
Bioconda-provided. Lineage data (eukaryota_odb10, vertebrata_odb12,
...) lives outside this registry — see the BUSCO docs for downloading,
or the install script which can stage it under
``$CONSTELLATION_BUSCO_HOME/lineages/``.

Status: STUB.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


BUSCO_VERSION = "5.7.1"


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
