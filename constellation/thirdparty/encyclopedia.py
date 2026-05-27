"""EncyclopeDIA jar adapter — ``ToolSpec`` registration only.

This module's responsibility is *tool discovery*: tell
``constellation doctor`` how to find the EncyclopeDIA executable jar
on disk. File-format translation (``.dlib``/``.elib`` modseq parsing
and the bidirectional ProForma 2.0 ↔ ``X[+N.NNN]`` translation) lives
with the rest of the encyclopedia file-format code in
:mod:`constellation.massspec.io.encyclopedia` — that's where you want
``parse_encyclopedia_modseq`` / ``format_encyclopedia_modseq`` from.

Pins: ``scripts/install-encyclopedia.sh`` hash-pins 2.12.30 (the
publicly available release that writes ``encyclopedia-2.12.30-executable.jar``
under ``third_party/encyclopedia/<version>/``). The lab's local install
of the unreleased 6.5.15 lays down ``encyclopedia-6.5.15.jar`` (the
install4j installer's naming) under ``$CONSTELLATION_ENCYCLOPEDIA_HOME``.
The adapter accepts both filenames — first existing candidate in HOME
wins, with v6.5.15 checked first so the lab's dev workflow resolves
without needing to flip a constant.

When 6.5.15 ships publicly the install script grows a 6.5.15 path
(install4j unattended) and the v2.12.30 entry can be retired here.
"""

from __future__ import annotations

import re
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register

# v2.12.30 is the SHA256-pinned version in the install script; v6.5.15 is
# the lab's local install (unreleased, env-var path only). Both are
# acceptable; the registry returns whichever exists under HOME.
ENCYCLOPEDIA_VERSION = "2.12.30"

# Candidate artifact filenames, checked in order under the resolved
# HOME (env var first, then `third_party/encyclopedia/current/`).
_JAR_CANDIDATES = (
    "encyclopedia-6.5.15.jar",
    f"encyclopedia-{ENCYCLOPEDIA_VERSION}-executable.jar",
)


_VERSION_RE = re.compile(r"encyclopedia-(\d+\.\d+\.\d+)(?:-executable)?\.jar$")


def _probe_version(path: Path) -> str | None:
    """Extract the version from the jar filename.

    Matches both naming conventions:
    - ``encyclopedia-2.12.30-executable.jar`` → ``"2.12.30"``
    - ``encyclopedia-6.5.15.jar`` → ``"6.5.15"``

    Cheaper than shelling out to ``java -jar``.
    """
    m = _VERSION_RE.match(path.name)
    return m.group(1) if m else None


register(
    ToolSpec(
        name="encyclopedia",
        env_var="CONSTELLATION_ENCYCLOPEDIA_HOME",
        artifact=_JAR_CANDIDATES,
        path_bin=None,  # jar — never found via $PATH
        install_script="scripts/install-encyclopedia.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["ENCYCLOPEDIA_VERSION"]
