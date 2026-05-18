"""Helpers shared across the EncyclopeDIA wrapper modules.

Argument-building primitives (sha256, manifest envelope, PTM toggle
encoding) live here so the per-utility wrappers stay focused on flag
translation.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import socket
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

PtmToggle = Literal["off", "var", "fix"]

# EncyclopeDIA versions the wrappers have been authored against. Each
# wrapper warns (not errors) when a different version is resolved, so a
# user with an interim build can still try things.
SUPPORTED_VERSIONS = frozenset({"2.12.30", "6.5.15"})


def sha256_file(path: Path) -> str:
    """SHA256 hex digest of ``path``. Streams 1 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ptm_toggle_args(
    ptms: Mapping[str, PtmToggle] | None,
    *,
    prefix: str = "-ptm",
) -> list[str]:
    """Encode the ``off|var|fix`` per-PTM toggle EncyclopeDIA expects.

    EncyclopeDIA's ``-convert -fastaToJChronologerLibrary`` exposes one
    flag per PTM (``-ptmAcetyl``, ``-ptmCarbamidomethyl``,
    ``-ptmProteinNTermAcetyl`` ...) each taking ``off``, ``var``, or
    ``fix``. Callers pass a Python dict; the order is sorted for
    deterministic argv (helps manifest-hashing reproducibility).
    """
    if not ptms:
        return []
    out: list[str] = []
    for name in sorted(ptms):
        value = ptms[name]
        if value not in ("off", "var", "fix"):
            raise ValueError(
                f"ptm {name!r}: value must be 'off', 'var', or 'fix' (got {value!r})"
            )
        out.append(f"{prefix}{name}")
        out.append(value)
    return out


def encyclopedia_passthrough_args(items: Sequence[str] | None) -> list[str]:
    """Parse ``--encyclopedia-arg FLAG=VALUE`` escape-hatch entries.

    Each item is split on the first ``=`` — ``["-foo=bar"]`` becomes
    ``["-foo", "bar"]``; items without ``=`` pass through as bare flags
    (``["-quiet"]`` stays ``["-quiet"]``). Empty / None input → empty
    list.
    """
    if not items:
        return []
    out: list[str] = []
    for raw in items:
        if "=" in raw:
            flag, _, value = raw.partition("=")
            out.append(flag)
            out.append(value)
        else:
            out.append(raw)
    return out


def default_heap_for_input(size_bytes: int) -> str:
    """Pick a sensible ``-Xmx`` based on the input size.

    Conservative — EncyclopeDIA's library search is memory-hungry, but
    most workstation runs are fine at 12g. The threshold is a soft
    suggestion; callers override via ``--jvm-heap``.
    """
    gib = size_bytes / (1024**3)
    if gib < 1:
        return "8g"
    if gib < 5:
        return "12g"
    if gib < 20:
        return "24g"
    return "48g"


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, second precision."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _input_record(path: Path | None) -> dict[str, Any] | None:
    """Build the ``{path, sha256, size_bytes}`` envelope for one input file.

    ``None`` → ``None`` so callers can pass optional inputs through.
    Missing files raise — manifest fidelity requires real bytes.
    """
    if path is None:
        return None
    p = Path(path)
    return {
        "path": str(p),
        "sha256": sha256_file(p),
        "size_bytes": p.stat().st_size,
    }


def build_manifest_envelope(
    *,
    subcommand: str,
    constellation_version: str,
    constellation_argv: Sequence[str],
    java_argv: Sequence[str] | None,
    tool: Mapping[str, Any],
    inputs: Mapping[str, Path | None],
    outputs: Mapping[str, Path | None],
    runtime: Mapping[str, Any] | None = None,
    ingest: Mapping[str, Any] | None = None,
    encyclopedia_passthrough_args: Sequence[str] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical ``manifest.json`` payload for an encyclopedia
    run.

    Shape (per the PR-approved plan):

    ::

        {
          "constellation_version": ...,
          "subcommand": "massspec search",
          "timestamp_utc": ...,
          "tool": {"name", "version", "jar_path", "jar_sha256",
                   "source", "env_var_set", "java_version",
                   "java_source"},
          "argv": {"constellation": [...], "java": [...]},
          "inputs": {"mzml": {"path", "sha256", "size_bytes"},
                     "library": {...}, "fasta": {...}},
          "outputs": {"elib": ..., "library_pqdir": ..., ...},
          "runtime": {"elapsed_seconds", "returncode", "host",
                      "platform", "python"},
          "ingest": {"skipped", "fragment_tolerance_ppm",
                     "library_counts", ...},
          "encyclopedia_passthrough_args": [...]
        }
    """
    runtime_envelope: dict[str, Any] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    if runtime is not None:
        runtime_envelope.update(runtime)

    envelope: dict[str, Any] = {
        "constellation_version": constellation_version,
        "subcommand": subcommand,
        "timestamp_utc": _utc_now_iso(),
        "tool": dict(tool),
        "argv": {
            "constellation": list(constellation_argv),
            "java": list(java_argv) if java_argv is not None else None,
        },
        "inputs": {name: _input_record(p) for name, p in inputs.items()},
        "outputs": {
            name: (str(p) if p is not None else None) for name, p in outputs.items()
        },
        "runtime": runtime_envelope,
        "ingest": dict(ingest) if ingest is not None else None,
        "encyclopedia_passthrough_args": list(encyclopedia_passthrough_args or []),
    }
    if extras:
        for k, v in extras.items():
            envelope.setdefault(k, v)
    return envelope


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Serialise ``manifest`` to ``path`` as pretty-printed JSON with
    trailing newline.
    """
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n")


__all__ = [
    "PtmToggle",
    "SUPPORTED_VERSIONS",
    "build_manifest_envelope",
    "default_heap_for_input",
    "encyclopedia_passthrough_args",
    "ptm_toggle_args",
    "sha256_file",
    "write_manifest",
]
