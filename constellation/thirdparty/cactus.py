"""Progressive Cactus (whole-genome aligner) — ToolSpec registration.

Installed from the upstream binary release by
``scripts/install-progressive-cactus.sh``, which builds Cactus's
self-contained virtualenv and exposes a wrapper as the artifact
``cactus`` (the wrapper sets the PATH/PYTHONPATH/LD_LIBRARY_PATH its
native binaries require). Consumed by the de novo / comparative stages of
the genome pipeline; also the optional HAL synteny backend for Ragout.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


CACTUS_VERSION = "3.2.1"


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
        name="cactus",
        env_var="CONSTELLATION_CACTUS_HOME",
        artifact="cactus",
        path_bin="cactus",
        install_script="scripts/install-progressive-cactus.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["CACTUS_VERSION"]
