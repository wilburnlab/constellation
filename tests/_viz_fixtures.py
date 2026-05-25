"""Shared test helpers for the reference-cache-first viz layer.

Builds an end-to-end Session backed by a fake reference cache + one or
more `transcriptome align` / `cluster` output dirs with schema-v2
manifests. Used by `tests/test_viz_session.py`, the kernel tests, and
the server endpoint tests.

Intentionally underscore-prefixed so pytest doesn't try to collect it
as a test module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_TABLE,
    INTRON_TABLE,
)
from constellation.sequencing.schemas.quant import COVERAGE_TABLE
from constellation.sequencing.schemas.reference import (
    CONTIG_TABLE,
    FEATURE_TABLE,
    SEQUENCE_TABLE,
)
from constellation.sequencing.schemas.transcriptome import (
    CLUSTER_MEMBERSHIP_TABLE,
    TRANSCRIPT_CLUSTER_TABLE,
)
from constellation.sequencing.transcriptome.manifest import (
    write_align_manifest,
    write_cluster_manifest,
)
from constellation.viz.server.session import Session


DEFAULT_HANDLE = "test_org@local_import-20260522"
DEFAULT_ASSEMBLY = "TestAssembly.1"


_DEFAULT_CONTIGS: list[dict[str, Any]] = [
    {
        "contig_id": 1,
        "name": "chr1",
        "length": 1_000_000,
        "topology": None,
        "circular": None,
    }
]

_DEFAULT_SEQUENCES: list[dict[str, Any]] = [
    {"contig_id": 1, "sequence": "ACGTACGTACGTACGT" * 10}
]


def install_fake_reference(
    cache_root: Path,
    *,
    handle: str = DEFAULT_HANDLE,
    assembly_accession: str | None = DEFAULT_ASSEMBLY,
    contigs: list[dict[str, Any]] | None = None,
    sequences: list[dict[str, Any]] | None = None,
    features: list[dict[str, Any]] | None = None,
) -> Path:
    """Materialize a fake reference at <cache_root>/<organism>/<release>/.

    Returns the release dir.
    """
    if "@" not in handle:
        raise ValueError(f"handle must be qualified: {handle!r}")
    organism, rest = handle.split("@", 1)
    release_slug = rest
    release_dir = cache_root / organism / release_slug
    genome_dir = release_dir / "genome"
    genome_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(contigs or _DEFAULT_CONTIGS, schema=CONTIG_TABLE),
        genome_dir / "contigs.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(sequences or _DEFAULT_SEQUENCES, schema=SEQUENCE_TABLE),
        genome_dir / "sequences.parquet",
    )
    if features is not None:
        annotation_dir = release_dir / "annotation"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(features, schema=FEATURE_TABLE),
            annotation_dir / "features.parquet",
        )
    source_after_at = handle.split("@", 1)[1]
    source, release = source_after_at.split("-", 1)
    meta_lines = [
        "schema_version = 1",
        f'handle = "{handle}"',
        f'organism = "{organism}"',
        f'source = "{source}"',
        f'release = "{release}"',
        f'constellation_version = "0.0.0+test"',
        f'fetched_at = "2026-05-22T00:00:00Z"',
    ]
    if assembly_accession is not None:
        meta_lines.append(f'assembly_accession = "{assembly_accession}"')
    meta_lines.append("")
    meta_lines.append("[verification]")
    meta_lines.append("source_checksum_verified = false")
    (release_dir / "meta.toml").write_text("\n".join(meta_lines) + "\n")
    return release_dir


def write_align_source(
    source_dir: Path,
    *,
    reference_handle: str | None = DEFAULT_HANDLE,
    reference_path: str = "",
    assembly_accession: str | None = DEFAULT_ASSEMBLY,
    alignments: list[dict[str, Any]] | None = None,
    alignment_blocks: list[dict[str, Any]] | None = None,
    coverage: list[dict[str, Any]] | None = None,
    introns: list[dict[str, Any]] | None = None,
    derived_annotation_features: list[dict[str, Any]] | None = None,
    samples: list[str] | None = None,
) -> Path:
    """Write a `transcriptome align` source dir with a v2 manifest.

    Each ``*`` kwarg, when non-None, writes the corresponding artifact
    and records it in the manifest's ``outputs`` map. Missing artifacts
    omit the manifest key (kernels surface only what's present).
    """
    source_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    if alignments is not None:
        path = source_dir / "alignments"
        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(alignments, schema=ALIGNMENT_TABLE),
            path / "part-00000.parquet",
        )
        outputs["alignments"] = str(path)
    if alignment_blocks is not None:
        path = source_dir / "alignment_blocks"
        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(alignment_blocks, schema=ALIGNMENT_BLOCK_TABLE),
            path / "part-00000.parquet",
        )
        outputs["alignment_blocks"] = str(path)
    if coverage is not None:
        path = source_dir / "coverage.parquet"
        pq.write_table(
            pa.Table.from_pylist(coverage, schema=COVERAGE_TABLE),
            path,
        )
        outputs["coverage"] = str(path)
    if introns is not None:
        path = source_dir / "introns.parquet"
        pq.write_table(
            pa.Table.from_pylist(introns, schema=INTRON_TABLE),
            path,
        )
        outputs["introns"] = str(path)
    if derived_annotation_features is not None:
        path = source_dir / "derived_annotation"
        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(derived_annotation_features, schema=FEATURE_TABLE),
            path / "features.parquet",
        )
        outputs["derived_annotation"] = str(path)

    write_align_manifest(
        source_dir / "manifest.json",
        reference_handle=reference_handle,
        reference_path=reference_path or "",
        assembly_accession=assembly_accession,
        demux_dir="",
        input_files=[],
        parameters={},
        stages={},
        outputs=outputs,
        samples=samples,
    )
    return source_dir


def write_cluster_source(
    source_dir: Path,
    *,
    reference_handle: str | None = DEFAULT_HANDLE,
    reference_path: str | None = None,
    assembly_accession: str | None = DEFAULT_ASSEMBLY,
    clusters: list[dict[str, Any]] | None = None,
    cluster_membership: list[dict[str, Any]] | None = None,
    samples: list[str] | None = None,
) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    if clusters is not None:
        path = source_dir / "clusters.parquet"
        pq.write_table(
            pa.Table.from_pylist(clusters, schema=TRANSCRIPT_CLUSTER_TABLE),
            path,
        )
        outputs["clusters"] = str(path)
    if cluster_membership is not None:
        path = source_dir / "cluster_membership.parquet"
        pq.write_table(
            pa.Table.from_pylist(cluster_membership, schema=CLUSTER_MEMBERSHIP_TABLE),
            path,
        )
        outputs["cluster_membership"] = str(path)
    write_cluster_manifest(
        source_dir / "manifest.json",
        reference_handle=reference_handle,
        reference_path=reference_path,
        assembly_accession=assembly_accession,
        align_dir="",
        demux_dir="",
        parameters={},
        stages={},
        outputs=outputs,
        samples=samples,
    )
    return source_dir


def build_viz_session(
    tmp_path: Path,
    monkeypatch,
    *,
    handle: str = DEFAULT_HANDLE,
    assembly_accession: str | None = DEFAULT_ASSEMBLY,
    contigs: list[dict[str, Any]] | None = None,
    sequences: list[dict[str, Any]] | None = None,
    features: list[dict[str, Any]] | None = None,
    align_sources: list[dict[str, Any]] | None = None,
    cluster_sources: list[dict[str, Any]] | None = None,
    label: str | None = None,
) -> Session:
    """Build a Session against a fake reference cache rooted at tmp_path.

    Each entry in ``align_sources`` / ``cluster_sources`` is a kwargs
    dict for :func:`write_align_source` / :func:`write_cluster_source`,
    optionally with a top-level ``label`` for the source name.
    """
    cache_root = tmp_path / "refs"
    cache_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache_root))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    release_dir = install_fake_reference(
        cache_root,
        handle=handle,
        assembly_accession=assembly_accession,
        contigs=contigs,
        sequences=sequences,
        features=features,
    )

    sources_payload: list[dict[str, Any]] = []

    if align_sources:
        for i, src_kwargs in enumerate(align_sources):
            label_arg = src_kwargs.pop("label", None) if "label" in src_kwargs else None
            src_dir = tmp_path / f"align-{i}"
            write_align_source(
                src_dir,
                reference_path=str(release_dir),
                **src_kwargs,
            )
            sources_payload.append(
                {"path": str(src_dir), "kind": "align", "label": label_arg or f"align-{i}"}
            )
    if cluster_sources:
        for i, src_kwargs in enumerate(cluster_sources):
            label_arg = src_kwargs.pop("label", None) if "label" in src_kwargs else None
            src_dir = tmp_path / f"cluster-{i}"
            write_cluster_source(
                src_dir,
                reference_path=str(release_dir),
                **src_kwargs,
            )
            sources_payload.append(
                {"path": str(src_dir), "kind": "cluster", "label": label_arg or f"cluster-{i}"}
            )

    return Session.open(
        reference_handle=handle,
        sources=sources_payload,
        label=label,
    )
