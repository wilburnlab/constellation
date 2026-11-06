"""EncyclopeDIA jar adapter — ``ToolSpec`` registration only.

This module's responsibility is *tool discovery*: tell
``constellation doctor`` (and ``run_jar``) how to find the EncyclopeDIA
executable jar on disk. File-format translation (``.dlib``/``.elib``
modseq parsing and the bidirectional ProForma 2.0 ↔ ``X[+N.NNN]``
translation) lives with the rest of the encyclopedia file-format code in
:mod:`constellation.massspec.io.encyclopedia` — that's where you want
``parse_encyclopedia_modseq`` / ``format_encyclopedia_modseq`` from.

Constellation pins EncyclopeDIA to **>= 6.9.23** (:data:`MINIMUM_VERSION`,
the one place the floor is written). That build is the one the command
snapshot under ``constellation/data/_raw/encyclopedia/`` was captured
from: ScribeTwo is a single ``-scribetwo -searchType ...`` program, the
``.dia`` cache is format 0.8.0 and result libraries are container
0.1.18. Discovery therefore:

  * globs ``encyclopedia-*.jar`` under the resolved HOME and picks the
    HIGHEST version (``pick="highest"``), so future releases resolve with
    no code change here;
  * opts into the per-user home cache (``user_cache_dir="encyclopedia"``)
    so :mod:`scripts/install-encyclopedia.sh` can install the app dir
    under ``~/.constellation/encyclopedia/<version>/`` instead of the
    repo's ``third_party/`` tree. ``$CONSTELLATION_ENCYCLOPEDIA_HOME``
    still overrides everything (e.g. a shared cluster install); and
  * prefers the JRE bundled beside the jar (``prefer_bundled_jre``). The
    jar warns that execution "may be unstable" outside Java 11-21 and
    ships Temurin 21, while the conda env's ``openjdk`` is unpinned and
    already newer than that range.

The jar is **not standalone**: its manifest ``Class-Path`` points at the
install directory's ``lib/`` (about a hundred jars, MSRawJava among them,
loaded at startup). HOME must be the whole install directory, never a
copy of the jar on its own.

``min_version`` makes ``constellation doctor`` report an older jar as
``too old``. The hard error with the "need >= 6.9.23" message stays in
the massspec guard layer
(:mod:`constellation.massspec.search.encyclopedia._common`), which every
jar-backed command calls before starting the JVM.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from urllib.parse import unquote

from constellation.thirdparty.registry import ToolSpec, register


#: Minimum supported EncyclopeDIA version. Single source of truth;
#: ``massspec.search.encyclopedia._common`` re-exports it.
MINIMUM_VERSION = "6.9.23"

_VERSION_RE = re.compile(r"encyclopedia-(\d+\.\d+\.\d+)(?:-executable)?\.jar$")


def _probe_version(path: Path) -> str | None:
    """Extract the version from the jar filename.

    Matches both naming conventions:
    - ``encyclopedia-6.9.23.jar`` → ``"6.9.23"`` (install4j build)
    - ``encyclopedia-2.12.30-executable.jar`` → ``"2.12.30"`` (legacy
      Bitbucket release; resolves, but the guard layer rejects it as
      below the floor)

    Cheaper than shelling out to ``java -jar``.
    """
    m = _VERSION_RE.match(path.name)
    return m.group(1) if m else None


def manifest_class_path(jar: Path) -> list[Path]:
    """The jars an EncyclopeDIA jar's manifest ``Class-Path`` names,
    resolved against the directory the jar is in.

    These are the jars the JVM actually loads with it. ``lib/`` can hold
    others: the 6.9.23 install has ``jchronologer-1.0.9.jar`` left beside
    the ``jchronologer-1.0.10.jar`` its manifest names.
    """
    with zipfile.ZipFile(jar) as archive:
        text = archive.read("META-INF/MANIFEST.MF").decode("utf-8")
    # A manifest line continues on the next line when that one starts
    # with a single space.
    joined = re.sub(r"\r?\n ", "", text)
    for line in joined.splitlines():
        name, _, value = line.partition(":")
        if name.strip().lower() == "class-path":
            return [jar.parent / unquote(entry) for entry in value.split()]
    return []


def jchronologer_jar(encyclopedia_jar: Path) -> Path:
    """The JChronologer jar (EncyclopeDIA's library predictor and its
    TorchScript models) that ``encyclopedia_jar`` loads."""
    named = [p for p in manifest_class_path(encyclopedia_jar) if p.name.startswith("jchronologer-")]
    if len(named) != 1:
        raise FileNotFoundError(
            f"{encyclopedia_jar} names {len(named)} JChronologer jars in its manifest Class-Path "
            f"({', '.join(p.name for p in named) or 'none'}); expected one"
        )
    if not named[0].is_file():
        raise FileNotFoundError(f"{encyclopedia_jar} names {named[0]}, which is not there")
    return named[0]


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
        min_version=MINIMUM_VERSION,
        prefer_bundled_jre=True,
    )
)
