"""Stage-output manifest readers/writers for transcriptome pipeline stages.

Replaces the ad-hoc ``json.dumps({...})`` blocks that ``constellation
transcriptome demultiplex``, ``... align`` and ``... cluster`` emit.

Schema v4 (current) adds two align-side outputs that the read pile-up
visualization treats as first-class: ``read_samples`` (a small parquet
join read_id → sample_id/sample_name materialized at align resolve time
from the upstream demux dir, so the viz layer never chases a
back-reference) and an always-emitted ``alignment_cs`` (the
``--emit-cs-tags`` flag flipped to default-on, with ``--no-emit-cs-tags``
the opt-out — opting out disables read pile-up viz). v3 manifests must
be regenerated via ``transcriptome align`` (no real-world users yet, so
the reader is a clean cut, not a back-compat shim).

Schema v3 promoted the demux stage to a formal manifest alongside align
+ cluster (it was previously an ad-hoc JSON dict with no
``schema_version`` field) and removed the ``samples_path`` field from
cluster manifests — sample state is now persisted into the demux
output dir as a ParquetDir bundle (``<demux-dir>/samples/``) and read
forward by align + cluster, so the path to the original TSV is no
longer load-bearing.

Schema v4 fields (top level)::

    schema_version  = 4
    kind            "demux" | "align" | "cluster"
    created_at      ISO 8601 UTC timestamp

    # demux-specific
    input_files        list[str]
    acquisition_map    dict[str, int]   file path → acquisition_id
    library_design     str
    parameters         dict[str, Any]
    stages             dict[str, Any]
    outputs            dict[str, str]   includes ``samples`` -> samples/

    # align + cluster:
    reference_handle   "<organism>@<source>-<release>"  or None (escape hatch)
    reference_path     absolute path to the resolved release dir, or to the
                       user-supplied --reference-dir when handle is None
    assembly_accession copied from <release>/meta.toml when available
    input_files        list[str]
    parameters         dict[str, Any]  (verb-specific knob snapshot)
    stages             dict[str, Any]  (verb-specific counters)
    outputs            dict[str, str]  (relative or absolute paths;
                                        align v4 includes ``read_samples``
                                        and (default) ``alignment_cs``)
    samples            list[str] | None
    # align-specific
    demux_dir          str
    # cluster-specific
    align_dir          str
    demux_dir          str

``reference_handle is None`` flags an escape-hatch run made via
``--reference-dir <PATH>`` — the dashboard's source picker treats those
sources as unopenable and steers the user to ``constellation reference
import`` first. Cluster runs propagate ``reference_handle`` from their
upstream align manifest unless the cluster CLI overrides it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


MANIFEST_SCHEMA_VERSION = 4
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class DemuxManifest:
    """Schema v4 manifest for ``transcriptome demultiplex`` outputs.

    The demux stage produces ``read_demux/`` (resolved sample_ids per
    read), ``feature_quant.parquet`` + downstream protein artifacts, and
    a ``samples/`` ParquetDir bundle that downstream stages load
    instead of re-parsing the user's TSV.
    """

    schema_version: int
    kind: Literal["demux"]
    created_at: str
    input_files: list[str]
    acquisition_map: dict[str, int]
    library_design: str
    parameters: dict[str, Any]
    stages: dict[str, Any]
    outputs: dict[str, str]


@dataclass(frozen=True, slots=True)
class AlignManifest:
    """Schema v4 manifest for ``transcriptome align`` outputs.

    v4 adds two ``outputs`` keys consumed by the read pile-up viz:
    ``read_samples`` (per-read sample assignment, materialized from the
    upstream demux dir at align resolve time) and the now-default-on
    ``alignment_cs`` (cs:long strings per primary alignment; suppressed
    only when the user passes ``--no-emit-cs-tags``).
    """

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
    # Final minimap2 argument tuple actually invoked, after preset +
    # explicit-override + extra-args resolution. Captured so a manifest
    # round-trips the exact command line a downstream consumer would
    # need to reproduce the alignment. Backward-compatible default (None)
    # — pre-existing manifests load as None without a schema bump.
    minimap2_resolved_args: list[str] | None = None


@dataclass(frozen=True, slots=True)
class ClusterManifest:
    """Schema v4 manifest for ``transcriptome cluster`` outputs."""

    schema_version: int
    kind: Literal["cluster"]
    reference_handle: str | None
    reference_path: str | None
    assembly_accession: str | None
    created_at: str
    align_dir: str
    demux_dir: str
    parameters: dict[str, Any]
    stages: dict[str, Any]
    outputs: dict[str, str]
    samples: list[str] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_demux_manifest(
    path: Path,
    *,
    input_files: list[str],
    acquisition_map: dict[str, int],
    library_design: str,
    parameters: dict[str, Any],
    stages: dict[str, Any],
    outputs: dict[str, str],
    created_at: str | None = None,
) -> DemuxManifest:
    """Write a demux-stage manifest to ``path`` and return the dataclass."""
    manifest = DemuxManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        kind="demux",
        created_at=created_at or _now_iso(),
        input_files=input_files,
        acquisition_map=acquisition_map,
        library_design=library_design,
        parameters=parameters,
        stages=stages,
        outputs=outputs,
    )
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


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
    minimap2_resolved_args: list[str] | None = None,
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
        minimap2_resolved_args=minimap2_resolved_args,
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
        parameters=parameters,
        stages=stages,
        outputs=outputs,
        samples=samples,
    )
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> DemuxManifest | AlignManifest | ClusterManifest:
    """Read ``<dir>/manifest.json`` and return the typed dataclass.

    Raises ``ValueError`` with an actionable message for missing-kind
    files, unsupported schema versions, or pre-v4 manifests (the clean
    cutover refuses to interpret legacy stage outputs — re-run the
    producing stage to upgrade).
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
    if kind == "demux":
        return DemuxManifest(
            schema_version=schema_version,
            kind="demux",
            created_at=str(raw.get("created_at", "")),
            input_files=list(raw.get("input_files", [])),
            acquisition_map=dict(raw.get("acquisition_map", {})),
            library_design=str(raw.get("library_design", "")),
            parameters=dict(raw.get("parameters", {})),
            stages=dict(raw.get("stages", {})),
            outputs=dict(raw.get("outputs", {})),
        )
    if kind == "align":
        resolved_args_raw = raw.get("minimap2_resolved_args")
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
            minimap2_resolved_args=(
                [str(a) for a in resolved_args_raw]
                if resolved_args_raw is not None
                else None
            ),
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
            parameters=dict(raw.get("parameters", {})),
            stages=dict(raw.get("stages", {})),
            outputs=dict(raw.get("outputs", {})),
            samples=raw.get("samples"),
        )
    raise ValueError(
        f"manifest at {path} has unknown kind={kind!r}; "
        f"expected 'demux', 'align', or 'cluster'"
    )


def read_manifest_dir(
    source_dir: Path,
) -> DemuxManifest | AlignManifest | ClusterManifest:
    """Read ``<source_dir>/manifest.json``; raise if the dir lacks one."""
    manifest_path = source_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(
            f"no manifest.json at {manifest_path}; "
            f"{source_dir} does not look like a `transcriptome "
            f"demultiplex`, `... align`, or `... cluster` output dir"
        )
    return read_manifest(manifest_path)


def detect_source_kind(source_dir: Path) -> Literal["demux", "align", "cluster"]:
    """Probe a directory's manifest to identify its producing stage."""
    return read_manifest_dir(source_dir).kind


__all__ = [
    "AlignManifest",
    "ClusterManifest",
    "DemuxManifest",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "detect_source_kind",
    "read_manifest",
    "read_manifest_dir",
    "write_align_manifest",
    "write_cluster_manifest",
    "write_demux_manifest",
]
