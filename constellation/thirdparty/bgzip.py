"""bgzip (htslib block-gzip compressor) — ToolSpec registration.

Bioconda-provided — ships in the same htslib ``bin/`` as ``samtools``
(no separate install). Used as the multithreaded compressor for the
genome-assembly BAM→FASTQ stage (see ``thirdparty/compress.py``).
Registered so ``constellation doctor`` enumerates it; resolution in
``compress.resolve_bgzip`` prefers the ``samtools`` sibling, with this
registry entry as the ``$PATH`` / env-var fallback.

Status: STUB.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


def _probe_version(path: Path) -> str | None:
    """``bgzip --version`` first line is ``"bgzip (htslib) 1.21"``."""
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
        name="bgzip",
        env_var="CONSTELLATION_BGZIP_HOME",
        artifact="bin/bgzip",
        path_bin="bgzip",
        install_script=None,            # bioconda-only (ships with samtools)
        version_probe=_probe_version,
    )
)


__all__ = ["_probe_version"]
