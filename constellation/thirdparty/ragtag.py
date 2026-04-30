"""RagTag (reference-guided scaffolding) — ToolSpec registration.

Status: STUB.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


RAGTAG_VERSION = "2.1.0"


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
        name="ragtag",
        env_var="CONSTELLATION_RAGTAG_HOME",
        artifact="ragtag.py",
        path_bin="ragtag.py",
        install_script="scripts/install-ragtag.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["RAGTAG_VERSION"]
