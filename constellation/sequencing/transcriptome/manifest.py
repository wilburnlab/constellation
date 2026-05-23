"""Stage-output manifest readers/writers for transcriptome pipeline stages.

Replaces the ad-hoc ``json.dumps({...})`` blocks that ``constellation
transcriptome align`` and ``constellation transcriptome cluster`` used to
emit. Schema v2 records the reference cache handle directly so downstream
consumers (the genome browser dashboard, future cross-validation passes,
session-cache resume) can match a stage output to its reference without
re-walking the cache by path.

Schema v2 fields (top level)::

    schema_version = 2
    kind            "align" | "cluster"
    reference_handle   "<organism>@<source>-<release>"  or None (escape hatch)
    reference_path     absolute path to the resolved release dir, or to the
                       user-supplied --reference-dir when handle is None
    assembly_accession copied from <release>/meta.toml when available
    created_at         ISO 8601 UTC timestamp
    input_files        list[str]
    parameters         dict[str, Any]  (verb-specific knob snapshot)
    stages             dict[str, Any]  (verb-specific counters)
    outputs            dict[str, str]  (relative or absolute paths)
    samples            list[str] | None
    # align-specific
    demux_dir          str
    # cluster-specific
    align_dir, demux_dir, samples_path

``reference_handle is None`` flags an escape-hatch run made via
``--reference-dir <PATH>`` — the dashboard's source picker treats those
sources as unopenable and steers the user to ``constellation reference
import`` first. Cluster runs propagate ``reference_handle`` from their
upstream align manifest unless the cluster CLI overrides it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


MANIFEST_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class AlignManifest:
    """Schema v2 manifest for ``transcriptome align`` outputs."""

    schema_version: int
    kind: Literal["align"]
    reference_handle: str | None
    reference_path: str
    assembly_accession: str | None
    created_at: str
    demux_dir: str
    input_files: list[str]
    parameters: dict[str, Any]
    stages: dict[str, Any]
    outputs: dict[str, str]
    samples: list[str] | None = None


@dataclass(frozen=True, slots=True)
class ClusterManifest:
    """Schema v2 manifest for ``transcriptome cluster`` outputs."""

    schema_version: int
    kind: Literal["cluster"]
    reference_handle: str | None
    reference_path: str | None
    assembly_accession: str | None
    created_at: str
    align_dir: str
    demux_dir: str
    samples_path: str
    parameters: dict[str, Any]
    stages: dict[str, Any]
    outputs: dict[str, str]
    samples: list[str] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_align_manifest(
    path: Path,
    *,
    reference_handle: str | None,
    reference_path: str,
    assembly_accession: str | None,
    demux_dir: str,
    input_files: list[str],
    parameters: dict[str, Any],
    stages: dict[str, Any],
    outputs: dict[str, str],
    samples: list[str] | None = None,
    created_at: str | None = None,
) -> AlignManifest:
    """Write an align-stage manifest to ``path`` and return the dataclass."""
    manifest = AlignManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        kind="align",
        reference_handle=reference_handle,
        reference_path=reference_path,
        assembly_accession=assembly_accession,
        created_at=created_at or _now_iso(),
        demux_dir=demux_dir,
        input_files=input_files,
        parameters=parameters,
        stages=stages,
        outputs=outputs,
        samples=samples,
    )
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def write_cluster_manifest(
    path: Path,
    *,
    reference_handle: str | None,
    reference_path: str | None,
    assembly_accession: str | None,
    align_dir: str,
    demux_dir: str,
    samples_path: str,
    parameters: dict[str, Any],
    stages: dict[str, Any],
    outputs: dict[str, str],
    samples: list[str] | None = None,
    created_at: str | None = None,
) -> ClusterManifest:
    """Write a cluster-stage manifest to ``path`` and return the dataclass."""
    manifest = ClusterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        kind="cluster",
        reference_handle=reference_handle,
        reference_path=reference_path,
        assembly_accession=assembly_accession,
        created_at=created_at or _now_iso(),
        align_dir=align_dir,
        demux_dir=demux_dir,
        samples_path=samples_path,
        parameters=parameters,
        stages=stages,
        outputs=outputs,
        samples=samples,
    )
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> AlignManifest | ClusterManifest:
    """Read ``<dir>/manifest.json`` and return the typed dataclass.

    Raises ``ValueError`` with an actionable message for missing-kind
    files, unsupported schema versions, or pre-v2 manifests (the clean
    cutover refuses to interpret legacy stage outputs).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema_version = raw.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest at {path} has schema_version={schema_version!r}; "
            f"this constellation supports v{MANIFEST_SCHEMA_VERSION}. "
            "Rerun the producing stage to refresh the manifest."
        )
    kind = raw.get("kind")
    if kind == "align":
        return AlignManifest(
            schema_version=schema_version,
            kind="align",
            reference_handle=raw.get("reference_handle"),
            reference_path=str(raw["reference_path"]),
            assembly_accession=raw.get("assembly_accession"),
            created_at=str(raw.get("created_at", "")),
            demux_dir=str(raw.get("demux_dir", "")),
            input_files=list(raw.get("input_files", [])),
            parameters=dict(raw.get("parameters", {})),
            stages=dict(raw.get("stages", {})),
            outputs=dict(raw.get("outputs", {})),
            samples=raw.get("samples"),
        )
    if kind == "cluster":
        return ClusterManifest(
            schema_version=schema_version,
            kind="cluster",
            reference_handle=raw.get("reference_handle"),
            reference_path=(
                str(raw["reference_path"]) if raw.get("reference_path") else None
            ),
            assembly_accession=raw.get("assembly_accession"),
            created_at=str(raw.get("created_at", "")),
            align_dir=str(raw.get("align_dir", "")),
            demux_dir=str(raw.get("demux_dir", "")),
            samples_path=str(raw.get("samples_path", "")),
            parameters=dict(raw.get("parameters", {})),
            stages=dict(raw.get("stages", {})),
            outputs=dict(raw.get("outputs", {})),
            samples=raw.get("samples"),
        )
    raise ValueError(
        f"manifest at {path} has unknown kind={kind!r}; expected 'align' or 'cluster'"
    )


def read_manifest_dir(source_dir: Path) -> AlignManifest | ClusterManifest:
    """Read ``<source_dir>/manifest.json``; raise if the dir lacks one."""
    manifest_path = source_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(
            f"no manifest.json at {manifest_path}; "
            f"{source_dir} does not look like a `transcriptome align` or "
            f"`transcriptome cluster` output dir"
        )
    return read_manifest(manifest_path)


def detect_source_kind(source_dir: Path) -> Literal["align", "cluster"]:
    """Probe a directory's manifest to identify its producing stage."""
    return read_manifest_dir(source_dir).kind


__all__ = [
    "AlignManifest",
    "ClusterManifest",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "detect_source_kind",
    "read_manifest",
    "read_manifest_dir",
    "write_align_manifest",
    "write_cluster_manifest",
]
