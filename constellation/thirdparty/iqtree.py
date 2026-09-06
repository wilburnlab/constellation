"""IQ-TREE 3 (maximum-likelihood phylogenetics) — ToolSpec registration.

Self-contained C++ binary (no Python deps), installed from the upstream
GitHub release by ``scripts/install-iqtree.sh``. Consumed by the
phylogenomics / cross-validation stages of the genome pipeline.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


IQTREE_VERSION = "3.1.2"


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
        name="iqtree",
        env_var="CONSTELLATION_IQTREE_HOME",
        artifact="bin/iqtree3",
        path_bin="iqtree3",
        install_script="scripts/install-iqtree.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["IQTREE_VERSION"]
