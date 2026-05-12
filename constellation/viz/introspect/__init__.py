"""Argparse → JSON schema walker for the dashboard's form generator.

Walks the production CLI parser (`constellation.cli.__main__._build_parser()`)
and emits a stable JSON tree the dashboard SPA renders into auto-generated
forms. A curated overlay (`curated.json`) names the small "common tasks"
subset that surfaces in the sidebar's Common mode.

Public surface:

- `walk_parser(parser) -> CommandSchema` — recursive descent
- `load_curated() -> list[CuratedEntry]` — read the packaged overlay
- `build_cli_schema() -> CliSchema` — convenience: parser + curated, one call

The schema shape is documented in `schema.py` as TypedDicts so the
FastAPI endpoint and the TS form generator share a single source of
truth.
"""

from constellation.viz.introspect.schema import (
    ArgumentSchema,
    CliSchema,
    CommandSchema,
    CuratedEntry,
)
from constellation.viz.introspect.walk import (
    build_cli_schema,
    load_curated,
    walk_parser,
)

__all__ = [
    "ArgumentSchema",
    "CliSchema",
    "CommandSchema",
    "CuratedEntry",
    "build_cli_schema",
    "load_curated",
    "walk_parser",
]
