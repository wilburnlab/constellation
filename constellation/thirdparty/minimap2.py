"""minimap2 (long-read aligner) — ToolSpec registration.

Bioconda-provided in production environments. The Python bindings
(``mappy``) avoid subprocess overhead for many use cases; the binary
is needed for full feature coverage and assembly-vs-assembly
workflows.

Status: STUB.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


MINIMAP2_VERSION = "2.28"


def _probe_version(path: Path) -> str | None:
    """``minimap2 --version`` prints ``"2.28-r1209"``."""
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
        name="minimap2",
        env_var="CONSTELLATION_MINIMAP2_HOME",
        artifact="minimap2",
        path_bin="minimap2",
        install_script=None,            # bioconda-only; no installer script
        version_probe=_probe_version,
    )
)


__all__ = ["MINIMAP2_VERSION"]
