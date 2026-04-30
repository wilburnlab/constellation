"""EncyclopeDIA jar adapter — ``ToolSpec`` registration only.

This module's responsibility is *tool discovery*: tell
``constellation doctor`` how to find the EncyclopeDIA executable jar
on disk. File-format translation (``.dlib``/``.elib`` modseq parsing
and the bidirectional ProForma 2.0 ↔ ``X[+N.NNN]`` translation) lives
with the rest of the encyclopedia file-format code in
:mod:`constellation.massspec.io.encyclopedia` — that's where you want
``parse_encyclopedia_modseq`` / ``format_encyclopedia_modseq`` from.

Pin: 2.12.30 (the lab's current stable). ``scripts/install-encyclopedia.sh``
hash-pins that release.
"""

from __future__ import annotations

from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register

ENCYCLOPEDIA_VERSION = "2.12.30"
_JAR_NAME = f"encyclopedia-{ENCYCLOPEDIA_VERSION}-executable.jar"


def _probe_version(path: Path) -> str | None:
    # Jar name encodes the version; cheaper than shelling out to java -jar.
    stem = path.name
    if stem.startswith("encyclopedia-") and stem.endswith("-executable.jar"):
        return stem[len("encyclopedia-") : -len("-executable.jar")]
    return None


register(
    ToolSpec(
        name="encyclopedia",
        env_var="CONSTELLATION_ENCYCLOPEDIA_HOME",
        artifact=_JAR_NAME,
        path_bin=None,  # jar — never found via $PATH
        install_script="scripts/install-encyclopedia.sh",
        version_probe=_probe_version,
    )
)


__all__ = ["ENCYCLOPEDIA_VERSION"]
