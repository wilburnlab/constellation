"""Has this unit of work already been done, with these inputs and settings?

A long workflow is a set of *stages*, each writing into a directory of
its own. ``--resume`` wants to skip the ones that finished. A bare
"finished" marker is not enough to decide that: rerun with different
reads or a different parameter and the old output is silently reused
while the new settings are recorded beside it. So a finished stage
records a **key**, a digest of everything its output depends on, and is
only considered done when the key it recorded is the key being asked
for.

The pattern, in the order that keeps it safe::

    key = stage_key(inputs=input_identity(paths), threads=..., upstream=previous_key)
    if not is_done(stage_dir, key):
        invalidate(stage_dir)       # the marker never outlives the outputs it describes
        ...do the work...
        mark_done(stage_dir, key)   # only on success

Chaining keys (passing the upstream stage's key as a part) makes a
change early in a workflow invalidate everything after it.

This is the design ``sequencing.assembly.pipeline`` arrived at after
shipping the marker-only version first (its docstrings record the bugs);
it is restated here so other modules can share it. That module keeps its
own copy for now.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

SUCCESS_MARKER = "_SUCCESS"


def stage_key(**parts: object) -> str:
    """Digest of everything a stage's output depends on.

    ``parts`` are named and order-independent. Values go through JSON
    (anything JSON cannot express is taken by ``str()``, so a ``Path``
    is its text), so use plain data: strings, numbers, lists, dicts.
    Pass an upstream stage's key as one of the parts to chain.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def input_identity(paths: Iterable[Path | str]) -> list[str]:
    """``path:size:mtime`` for each input, sorted: a cheap stand-in for
    hashing large files, enough to notice one that was swapped or
    regenerated. A missing input is ``path:missing``, so a stage keyed on
    it cannot match one that ran when the file existed."""
    out: list[str] = []
    for path in sorted(Path(p) for p in paths):
        try:
            st = path.stat()
        except OSError:
            out.append(f"{path}:missing")
        else:
            out.append(f"{path}:{st.st_size}:{st.st_mtime_ns}")
    return out


def recorded_key(stage_dir: Path | str) -> str | None:
    """The key a finished stage recorded: ``None`` when the stage has no
    marker, ``""`` when it has one that carries no key."""
    try:
        return (Path(stage_dir) / SUCCESS_MARKER).read_text().strip()
    except OSError:
        return None


def is_done(stage_dir: Path | str, key: str | None = None) -> bool:
    """True iff the stage finished and, when ``key`` is given, finished
    with that key. A marker with a different key, or with none, is a
    miss: re-running is better than trusting output of unknown origin.
    ``key=None`` asks only whether a marker exists."""
    recorded = recorded_key(stage_dir)
    if recorded is None:
        return False
    return key is None or recorded == key


def invalidate(stage_dir: Path | str) -> None:
    """Remove a stage's marker. Call before rewriting its outputs, so an
    interrupted rerun cannot leave the old marker beside new, partial
    files."""
    (Path(stage_dir) / SUCCESS_MARKER).unlink(missing_ok=True)


def mark_done(stage_dir: Path | str, key: str | None = None) -> None:
    """Record that the stage finished (with ``key``). The last thing a
    stage does, and only on success."""
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SUCCESS_MARKER).write_text("" if key is None else key)


__all__ = [
    "SUCCESS_MARKER",
    "input_identity",
    "invalidate",
    "is_done",
    "mark_done",
    "recorded_key",
    "stage_key",
]
