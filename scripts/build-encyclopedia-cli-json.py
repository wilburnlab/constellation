"""Generate constellation/data/encyclopedia_cli.json from captured help text.

Parses the ``-h`` captures under
``constellation/data/_raw/encyclopedia/<version>/help/`` (written by
``scripts/survey-encyclopedia.py``) into the packaged table of programs
and flags that wrappers and tests validate against. Pure text → JSON:
it needs no jar, so it can be re-run anywhere to check the JSON is in
step with the raw captures.

``<version>`` is the pinned minimum
(``constellation.thirdparty.encyclopedia.MINIMUM_VERSION``).

Run from project root:
    python3 scripts/build-encyclopedia-cli-json.py
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from constellation.massspec.search.encyclopedia.cli_spec import (
    CLI_JSON_NAME,
    build_cli_spec,
    raw_help_dir,
)
from constellation.thirdparty.encyclopedia import MINIMUM_VERSION


def main() -> int:
    spec = build_cli_spec(raw_help_dir(MINIMUM_VERSION), version=MINIMUM_VERSION)
    out = Path(str(resources.files("constellation.data"))) / CLI_JSON_NAME
    out.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    n_flags = sum(len(p["flags"]) for p in spec["programs"].values())
    print(f"wrote {out}: {len(spec['programs'])} programs, {n_flags} flags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
