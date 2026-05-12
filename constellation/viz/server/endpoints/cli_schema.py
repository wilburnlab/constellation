"""CLI schema endpoint — exposes the introspected argparse tree.

The dashboard's sidebar + form generator drives off the JSON shape
emitted by :mod:`constellation.viz.introspect`. The result is cached
for the process lifetime — the parser is built once at startup and
never changes for a given binary.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter

from constellation.viz.introspect import CliSchema, build_cli_schema


router = APIRouter(prefix="/api/cli", tags=["cli"])


@lru_cache(maxsize=1)
def _cached_schema() -> CliSchema:
    """Build the CLI schema once per process.

    Imports the CLI parser builder lazily so test environments without
    the full ``constellation.cli`` import surface can still exercise
    the rest of the dashboard endpoints in isolation.
    """
    from constellation.cli.__main__ import _build_parser

    return build_cli_schema(_build_parser())


@router.get("/schema")
def get_schema() -> CliSchema:
    """Return the full introspected CLI tree plus the curated overlay."""
    return _cached_schema()
