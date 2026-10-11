"""Capture the installed EncyclopeDIA jar's ``-h`` output into the repo.

Runs ``-h`` for the dispatcher, every program it lists, every converter
``-convert -h`` lists, and the few programs the jar accepts without
listing (see ``EXTRA_PROBES`` in
``constellation/massspec/search/encyclopedia/cli_spec.py``), normalises
the text and writes::

    constellation/data/_raw/encyclopedia/<version>/help/<program>.txt
    constellation/data/_raw/encyclopedia/<version>/programs.json

Needs an installed jar (``constellation doctor`` must show
``encyclopedia`` as ``ok``). Re-run it when the pinned version changes,
then rebuild the packaged flag table and read the diff::

    python3 scripts/survey-encyclopedia.py
    python3 scripts/build-encyclopedia-cli-json.py
    git diff constellation/data

Run from project root.
"""

from __future__ import annotations

import argparse
import sys

from constellation.massspec.search.encyclopedia._common import sha256_file
from constellation.massspec.search.encyclopedia.cli_spec import capture_help, write_capture
from constellation.thirdparty import encyclopedia as _adapter  # noqa: F401  (registers the tool)
from constellation.thirdparty.jvm import _resolve_java
from constellation.thirdparty.registry import find


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="seconds allowed per `-h` call"
    )
    args = parser.parse_args()

    handle = find("encyclopedia")
    if handle.version is None:
        print(f"error: cannot read a version from {handle.path.name}", file=sys.stderr)
        return 1
    java, java_source, java_version = _resolve_java(handle)
    print(f"jar:  {handle.path} ({handle.source})", file=sys.stderr)
    print(f"java: {java} ({java_source})", file=sys.stderr)

    texts, probes = capture_help(jar=handle.path, java=java, timeout=args.timeout)
    out = write_capture(
        texts,
        probes,
        version=handle.version,
        jar_sha256=sha256_file(handle.path),
        java_version=java_version,
    )
    print(f"captured {len(texts)} programs -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
