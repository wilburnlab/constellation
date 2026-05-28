"""Typed manifest for the Thermo ``.raw`` → directory-bundle conversion.

A single converted ``.raw`` produces a directory containing
``peaks.parquet`` + ``scan_metadata.parquet`` +
``acquisition_metadata.parquet`` plus a ``manifest.json`` that records:

- which source file the bundle came from (path + optional sha256),
- which convert-time parameters produced the parquets (rt-bin-width,
  profile-mode flag, DLL pack version, .NET runtime),
- a small run-summary (instrument identity, n_scans, RT range, list of
  tune-data keys present) so a consumer can decide which bundles to
  open without paying parquet-read cost on every one.

Schema v1 is the initial Constellation port. Manifests bump
``schema_version`` on any additive change. The reader rejects manifests
whose version it doesn't recognise — the bundle producer's command
should be re-run rather than the reader hand-coerced into an older
shape.

Modelled on :mod:`constellation.sequencing.transcriptome.manifest` —
same JSON layout + dataclass dispatch + ISO-8601 ``created_at`` so
multiple bundle types can share future tooling that walks a directory
tree of mixed acquisition bundles.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
BUNDLE_KIND = "thermo_acquisition"


@dataclass(frozen=True, slots=True)
class ThermoAcquisitionManifest:
    """Manifest for one ``.raw`` → bundle conversion."""

    schema_version: int
    kind: Literal["thermo_acquisition"]
    created_at: str
    source_file: str
    source_sha256: str | None
    # Convert-time parameters: ``rt_bin_width_s`` (the parquet row-group
    # chunking knob), ``profile_mode`` (Thermo segmented-scan flag),
    # ``dll_version`` (e.g. ``"1.4.5"`` from ``third_party/thermo/<v>/``),
    # ``dotnet_runtime`` (e.g. ``"net8.0"``), plus anything else worth
    # recording (input flags the user passed at the CLI).
    parameters: dict[str, Any]
    # Path mappings from semantic name → relative filename inside the
    # bundle directory. v1 keys: ``peaks``, ``scan_metadata``,
    # ``acquisition_metadata``. Future readers add keys without breaking
    # existing consumers.
    outputs: dict[str, str]
    # Small instrument/run summary so the dashboard / cross-bundle
    # walker can sort and filter without opening parquets. ``tune_data``
    # is *not* mirrored here — only its key list, so the manifest stays
    # small. Full values live in ``acquisition_metadata.parquet``.
    run_metadata: dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest(
    path: Path,
    *,
    source_file: str,
    source_sha256: str | None,
    parameters: dict[str, Any],
    outputs: dict[str, str],
    run_metadata: dict[str, Any],
    created_at: str | None = None,
) -> ThermoAcquisitionManifest:
    """Write a Thermo-bundle manifest to ``path`` and return the dataclass."""
    manifest = ThermoAcquisitionManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        kind=BUNDLE_KIND,
        created_at=created_at or _now_iso(),
        source_file=source_file,
        source_sha256=source_sha256,
        parameters=parameters,
        outputs=outputs,
        run_metadata=run_metadata,
    )
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> ThermoAcquisitionManifest:
    """Read a Thermo-bundle ``manifest.json``.

    Raises ``ValueError`` with an actionable message for missing-kind
    files or unsupported schema versions — the convert step must be
    re-run rather than silently coerced.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema_version = raw.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest at {path} has schema_version={schema_version!r}; "
            f"this constellation supports v{MANIFEST_SCHEMA_VERSION}. "
            "Re-run `constellation massspec convert` to refresh the bundle."
        )
    kind = raw.get("kind")
    if kind != BUNDLE_KIND:
        raise ValueError(
            f"manifest at {path} has kind={kind!r}; expected {BUNDLE_KIND!r}"
        )
    return ThermoAcquisitionManifest(
        schema_version=schema_version,
        kind=BUNDLE_KIND,
        created_at=str(raw.get("created_at", "")),
        source_file=str(raw.get("source_file", "")),
        source_sha256=raw.get("source_sha256"),
        parameters=dict(raw.get("parameters", {})),
        outputs=dict(raw.get("outputs", {})),
        run_metadata=dict(raw.get("run_metadata", {})),
    )


def read_manifest_dir(bundle_dir: Path) -> ThermoAcquisitionManifest:
    """Read ``<bundle_dir>/manifest.json``; raise if the dir lacks one."""
    manifest_path = bundle_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(
            f"no {MANIFEST_FILENAME} at {manifest_path}; "
            f"{bundle_dir} does not look like a `constellation massspec convert` "
            f"output bundle"
        )
    return read_manifest(manifest_path)


__all__ = [
    "BUNDLE_KIND",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ThermoAcquisitionManifest",
    "read_manifest",
    "read_manifest_dir",
    "write_manifest",
]
