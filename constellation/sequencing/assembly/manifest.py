"""Manifest reader/writer for ``constellation genome assemble`` outputs.

Mirrors :mod:`sequencing.transcriptome.manifest` — a frozen dataclass +
``asdict``/json writer + a typed reader dispatching on ``kind``. One
``AssemblyManifest`` records the whole pipeline run (basecall → harmonize
→ assemble → optional scaffold → optional polish) so the standalone
``genome scaffold`` / ``genome polish`` / ``genome diagnose`` re-entry
verbs can recover the input paths, the harmonized BAM, the basecaller
model, and per-stage stats.

``scaffold_reference_handle is None`` flags either a no-scaffold run or an
escape-hatch ``--scaffold-reference-dir`` run (same convention as the
align manifest's ``reference_handle``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class AssemblyManifest:
    """Schema v1 manifest for a genome-assembly pipeline run."""

    schema_version: int
    kind: Literal["assembly"]
    created_at: str
    input_mode: Literal["pod5", "bam"]
    input_files: list[str]
    # basecall provenance (null in bam mode)
    basecall_model: str | None
    modified_bases: list[str] | None
    device: str | None
    # read-group harmonization
    basecaller_model_ds: str | None
    unified_read_group: str
    # scaffold (null unless it ran)
    scaffold_reference_handle: str | None
    scaffold_reference_path: str | None
    assembly_accession: str | None
    # polish
    polish_rounds: int
    # tool provenance
    tool_versions: dict[str, Any]
    tool_args: dict[str, Any]
    parameters: dict[str, Any]
    stages: dict[str, Any]
    outputs: dict[str, str]
    busco_lineage: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_assembly_manifest(
    path: Path,
    *,
    input_mode: Literal["pod5", "bam"],
    input_files: list[str],
    unified_read_group: str,
    polish_rounds: int,
    parameters: dict[str, Any],
    stages: dict[str, Any],
    outputs: dict[str, str],
    basecall_model: str | None = None,
    modified_bases: list[str] | None = None,
    device: str | None = None,
    basecaller_model_ds: str | None = None,
    scaffold_reference_handle: str | None = None,
    scaffold_reference_path: str | None = None,
    assembly_accession: str | None = None,
    tool_versions: dict[str, Any] | None = None,
    tool_args: dict[str, Any] | None = None,
    busco_lineage: str | None = None,
    created_at: str | None = None,
) -> AssemblyManifest:
    """Write an assembly-pipeline manifest to ``path`` and return it."""
    manifest = AssemblyManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        kind="assembly",
        created_at=created_at or _now_iso(),
        input_mode=input_mode,
        input_files=input_files,
        basecall_model=basecall_model,
        modified_bases=modified_bases,
        device=device,
        basecaller_model_ds=basecaller_model_ds,
        unified_read_group=unified_read_group,
        scaffold_reference_handle=scaffold_reference_handle,
        scaffold_reference_path=scaffold_reference_path,
        assembly_accession=assembly_accession,
        polish_rounds=polish_rounds,
        tool_versions=tool_versions or {},
        tool_args=tool_args or {},
        parameters=parameters,
        stages=stages,
        outputs=outputs,
        busco_lineage=busco_lineage,
    )
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> AssemblyManifest:
    """Read an assembly ``manifest.json`` and return the typed dataclass."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema_version = raw.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest at {path} has schema_version={schema_version!r}; "
            f"this constellation supports v{MANIFEST_SCHEMA_VERSION}. "
            "Rerun `genome assemble` to refresh the manifest."
        )
    kind = raw.get("kind")
    if kind != "assembly":
        raise ValueError(
            f"manifest at {path} has unknown kind={kind!r}; expected 'assembly'"
        )
    return AssemblyManifest(
        schema_version=schema_version,
        kind="assembly",
        created_at=str(raw.get("created_at", "")),
        input_mode=raw.get("input_mode", "bam"),
        input_files=list(raw.get("input_files", [])),
        basecall_model=raw.get("basecall_model"),
        modified_bases=raw.get("modified_bases"),
        device=raw.get("device"),
        basecaller_model_ds=raw.get("basecaller_model_ds"),
        unified_read_group=str(raw.get("unified_read_group", "")),
        scaffold_reference_handle=raw.get("scaffold_reference_handle"),
        scaffold_reference_path=raw.get("scaffold_reference_path"),
        assembly_accession=raw.get("assembly_accession"),
        polish_rounds=int(raw.get("polish_rounds", 0)),
        tool_versions=dict(raw.get("tool_versions", {})),
        tool_args=dict(raw.get("tool_args", {})),
        parameters=dict(raw.get("parameters", {})),
        stages=dict(raw.get("stages", {})),
        outputs=dict(raw.get("outputs", {})),
        busco_lineage=raw.get("busco_lineage"),
    )


def read_manifest_dir(source_dir: Path) -> AssemblyManifest:
    """Read ``<source_dir>/manifest.json``; raise if the dir lacks one."""
    manifest_path = source_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(
            f"no manifest.json at {manifest_path}; "
            f"{source_dir} does not look like a `genome assemble` output dir"
        )
    return read_manifest(manifest_path)


__all__ = [
    "AssemblyManifest",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "read_manifest",
    "read_manifest_dir",
    "write_assembly_manifest",
]
