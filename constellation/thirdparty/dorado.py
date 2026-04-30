"""Dorado (Oxford Nanopore basecaller) — ToolSpec registration only.

Tool discovery: tells ``constellation doctor`` how to find the
``dorado`` binary. Resolution cascade (per :mod:`thirdparty.registry`)
prefers in this order:

    1. ``$CONSTELLATION_DORADO_HOME`` env var (if set)
    2. ``third_party/dorado/current/`` symlink (set by install script)
    3. ``shutil.which('dorado')`` ($PATH; conda-installed dorado lives here)
    4. raise with install hint

Wrapping logic + subcommand orchestration lives in
:mod:`constellation.sequencing.basecall.dorado`. This module is the
discovery contract only.

License: Dorado is distributed under the Oxford Nanopore Technologies
PLC. Public License Version 1.0 — Research Purposes only. The
Constellation wrapper itself is Apache 2.0; users invoking it
independently accept ONT's terms. See the upstream LICENCE.txt:
https://github.com/nanoporetech/dorado/blob/release-v1.4/LICENCE.txt

Status: STUB. Pinned ``DORADO_VERSION`` is provisional — bump as ONT
releases new model-supporting versions; only requirement is "supports
the latest released models".
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


# Pinned version. Update as needed; the lab's only constraint is that
# the installed Dorado supports the latest released models.
DORADO_VERSION = "0.8.3"


def _probe_version(path: Path) -> str | None:
    """Probe Dorado's version via ``dorado --version``.

    The binary prints something like ``"0.8.3+abc1234"`` on stderr.
    Returns the pre-``+`` segment or None on failure.
    """
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
        name="dorado",
        env_var="CONSTELLATION_DORADO_HOME",
        artifact="bin/dorado",          # under HOME/bin/dorado for tarball installs
        path_bin="dorado",              # primary discovery via $PATH (conda)
        install_script="scripts/install-dorado.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["DORADO_VERSION"]
