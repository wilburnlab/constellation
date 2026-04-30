"""HiFiASM (long-read genome assembler) — ToolSpec registration.

The lab's primary genome assembler. Resolution prefers conda-installed
``hifiasm`` on ``$PATH``; ``$CONSTELLATION_HIFIASM_HOME`` overrides
when a from-source build is needed.

Status: STUB. Pinned version is provisional.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


HIFIASM_VERSION = "0.20.0"


def _probe_version(path: Path) -> str | None:
    """``hifiasm --version`` prints e.g. ``"0.20.0-r639"``."""
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
    m = re.search(r"(\d+\.\d+\.\d+)", output)
    return m.group(1) if m else None


register(
    ToolSpec(
        name="hifiasm",
        env_var="CONSTELLATION_HIFIASM_HOME",
        artifact="hifiasm",
        path_bin="hifiasm",
        install_script="scripts/install-hifiasm.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["HIFIASM_VERSION"]
