"""Ragout (multi-reference, rearrangement-aware scaffolding) — ToolSpec.

Bioconda-provided; pins Python 3.8, so ``scripts/install-ragout.sh``
installs it into a dedicated conda env and exposes a wrapper as the
artifact ``ragout``. Sibelia (its default synteny backend) is bundled by
the conda package. Distinct from RagTag (see ``ragtag.py``).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


RAGOUT_VERSION = "2.3"


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
        name="ragout",
        env_var="CONSTELLATION_RAGOUT_HOME",
        artifact="ragout",
        path_bin="ragout",
        install_script="scripts/install-ragout.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["RAGOUT_VERSION"]
