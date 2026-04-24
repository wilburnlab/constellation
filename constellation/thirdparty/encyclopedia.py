"""EncyclopeDIA jar adapter — reference integration for the thirdparty pattern.

Cartographer's pinned version lives in ``cartographer/pipeline/__init__.py``
as ``ENCYCLOPEDIA_VERSION``. Constellation uses the same pin until we have
reason to move; ``scripts/install-encyclopedia.sh`` hash-pins that release.
"""

from __future__ import annotations

from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


ENCYCLOPEDIA_VERSION = "3.0.4"
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
