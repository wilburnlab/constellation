"""EncyclopeDIA jar adapter — ``ToolSpec`` registration only.

This module's responsibility is *tool discovery*: tell
``constellation doctor`` (and ``run_jar``) how to find the EncyclopeDIA
executable jar on disk. File-format translation (``.dlib``/``.elib``
modseq parsing and the bidirectional ProForma 2.0 ↔ ``X[+N.NNN]``
translation) lives with the rest of the encyclopedia file-format code in
:mod:`constellation.massspec.io.encyclopedia` — that's where you want
``parse_encyclopedia_modseq`` / ``format_encyclopedia_modseq`` from.

Constellation pins EncyclopeDIA to **>= 6.5.15** (the version available
at public release; older builds lack the ``-convert
-fastaToJChronologerLibrary`` predict-library utility the
transcriptome→proteome pipeline depends on). Discovery therefore:

  * globs ``encyclopedia-*.jar`` under the resolved HOME and picks the
    HIGHEST version (``pick="highest"``), so future releases (6.6.0,
    7.0, ...) resolve with no code change here; and
  * opts into the per-user home cache (``user_cache_dir="encyclopedia"``)
    so :mod:`scripts/install-encyclopedia.sh` can install the heavy
    install4j app dir (jar + bundled JRE + native MSRawJava libs) under
    ``~/.constellation/encyclopedia/<version>/`` instead of the repo's
    ``third_party/`` tree. ``$CONSTELLATION_ENCYCLOPEDIA_HOME`` still
    overrides everything (e.g. an ad-hoc dev install elsewhere).

Minimum-version *enforcement* is NOT done here — the registry just picks
the highest installed jar. The massspec guard layer
(:mod:`constellation.massspec.search.encyclopedia._common`) hard-errors
if that resolved version is still below the minimum, so the user gets a
precise "need >= 6.5.15" message rather than a generic
``ToolNotFoundError``.
"""

from __future__ import annotations

import re
from pathlib import Path

from constellation.thirdparty.registry import ToolSpec, register


_VERSION_RE = re.compile(r"encyclopedia-(\d+\.\d+\.\d+)(?:-executable)?\.jar$")


def _probe_version(path: Path) -> str | None:
    """Extract the version from the jar filename.

    Matches both naming conventions:
    - ``encyclopedia-6.5.15.jar`` → ``"6.5.15"`` (install4j build)
    - ``encyclopedia-2.12.30-executable.jar`` → ``"2.12.30"`` (legacy
      Bitbucket release; resolves, but the guard layer rejects it as
      below the >= 6.5.15 floor)

    Cheaper than shelling out to ``java -jar``.
    """
    m = _VERSION_RE.match(path.name)
    return m.group(1) if m else None


register(
    ToolSpec(
        name="encyclopedia",
        env_var="CONSTELLATION_ENCYCLOPEDIA_HOME",
        artifact_glob="encyclopedia-*.jar",
        pick="highest",
        user_cache_dir="encyclopedia",
        path_bin=None,  # jar — never found via $PATH
        install_script="scripts/install-encyclopedia.sh",
        version_probe=_probe_version,
    )
)
