"""Saved-session cache for the viz layer.

Per-user cache of named genome-browser sessions — one TOML per saved
session under ``~/.constellation/sessions/`` (or
``$CONSTELLATION_SESSIONS_HOME`` / XDG override). Each saved session
records the reference handle plus the list of data sources the user
attached, so the dashboard can restore a configuration without the user
re-typing every path.

Mirrors the layout of :mod:`constellation.sequencing.reference.handle`
intentionally — sessions, references, and catalogs are siblings under
``~/.constellation/`` with the same per-user-cache shape.
"""

from constellation.viz.sessions.cache import (
    SAVED_SESSION_SCHEMA_VERSION,
    SavedSession,
    SavedSessionSource,
    cache_root,
    delete_saved,
    list_saved,
    read_saved,
    write_saved,
)


__all__ = [
    "SAVED_SESSION_SCHEMA_VERSION",
    "SavedSession",
    "SavedSessionSource",
    "cache_root",
    "delete_saved",
    "list_saved",
    "read_saved",
    "write_saved",
]
