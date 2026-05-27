"""mmseqs2 (k-mer clustering & search) — ToolSpec registration.

Primary install path is conda (bioconda — already in
:file:`environment.yml`). The static-binary fallback at
:file:`scripts/install-mmseqs2.sh` writes to
``third_party/mmseqs2/<version>/bin/mmseqs`` for nodes without conda.

Consumed by:
  * :mod:`sequencing.transcriptome.cluster` — k-mer prefiltering ahead
    of consensus building (existing).
  * :mod:`thirdparty.mmseqs2_run` — generic ``mmseqs easy-search``
    subprocess runner used by the transcriptome-to-proteomics pipeline
    for novel-vs-reference+swissprot alignment.
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
        install_script="scripts/install-mmseqs2.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["MMSEQS2_VERSION"]
