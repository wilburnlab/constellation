"""mmseqs2 (k-mer clustering & search) — ToolSpec registration.

Already conda-installed in the base environment (`environment.yml`).
Used by :mod:`sequencing.transcriptome.cluster` for k-mer prefiltering
ahead of consensus building. Future: also for protein-database search
on the predicted-protein output that bridges to massspec.

Status: STUB.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


MMSEQS2_VERSION = "15-6f452"


def _probe_version(path: Path) -> str | None:
    """``mmseqs version`` prints e.g. ``"15-6f452"`` (single line)."""
    try:
        result = subprocess.run(
            [str(path), "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or "").strip()
    if not output:
        return None
    # mmseqs version is a single line like "15-6f452" — return as-is
    # (not a strict semver, so don't regex into it)
    return output.splitlines()[0].strip()


register(
    ToolSpec(
        name="mmseqs2",
        env_var="CONSTELLATION_MMSEQS2_HOME",
        artifact="bin/mmseqs",
        path_bin="mmseqs",
        install_script=None,            # bioconda-provided
        version_probe=_probe_version,
    )
)


__all__ = ["MMSEQS2_VERSION"]
