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
https://github.com/nanoporetech/dorado/blob/master/LICENCE.txt

Pinned at the latest stable Dorado (``DORADO_VERSION``); bump as ONT
releases new model-supporting versions. Constellation requires Dorado
**>= ``MINIMUM_VERSION``** (2.1.2): the resolver
(``basecall.dorado_run.resolve_dorado``) refuses anything older and
``constellation doctor`` reports it as too old. There is deliberately no
compatibility shim for older releases.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


# Pinned version: what scripts/install-dorado.sh installs. Update as
# needed; the lab's only constraint is that the installed Dorado supports
# the latest released models.
DORADO_VERSION = "2.1.2"

# The oldest Dorado anything here will run. 2.1.1 and 2.1.2 fixed
# non-deterministic calls from the SUP v5+ and HAC v6.0.0 models, so below
# 2.1.2 the same POD5 and model need not give the same reads; 2.0.1 fixed
# duplex with HAC v6.0.0 and a v6 HAC crash on hosts without a GPU. None
# of the flags Constellation passes changed between 2.0.0 and 2.1.2.
MINIMUM_VERSION = "2.1.2"


def _probe_version(path: Path) -> str | None:
    """Probe Dorado's version via ``dorado --version``.

    The binary prints something like ``"2.0.0+abc1234"`` on stderr.
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
        min_version=MINIMUM_VERSION,
    )
)


__all__ = ["DORADO_VERSION", "MINIMUM_VERSION"]
