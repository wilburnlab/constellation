"""samtools (BAM/SAM utility suite) — ToolSpec registration.

Bioconda-provided. Used for BAM sort / index / merge during the
polish loop and as a general companion to mappy/minimap2.

Status: STUB.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


SAMTOOLS_VERSION = "1.21"


def _probe_version(path: Path) -> str | None:
    """``samtools --version`` first line is ``"samtools 1.21"``."""
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    first_line = (result.stdout or "").splitlines()[:1]
    if not first_line:
        return None
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", first_line[0])
    return m.group(1) if m else None


register(
    ToolSpec(
        name="samtools",
        env_var="CONSTELLATION_SAMTOOLS_HOME",
        artifact="bin/samtools",
        path_bin="samtools",
        install_script=None,            # bioconda-only
        version_probe=_probe_version,
    )
)


__all__ = ["SAMTOOLS_VERSION"]
