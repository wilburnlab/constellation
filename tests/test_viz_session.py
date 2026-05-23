"""Session shape — Session.open against the reference cache.

Exercises the reference-cache-first construction path:
- builds a fake cache root via monkeypatch + ``CONSTELLATION_REFERENCES_HOME``
- writes one-or-more `transcriptome align` / `cluster` source dirs with
  schema-v2 manifests
- calls ``Session.open(reference_handle, sources)`` and asserts the
  resolved dataclass shape, slot resolution, and mismatch-warning logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.viz.server.session import Session
from _viz_fixtures import (
    DEFAULT_ASSEMBLY,
    DEFAULT_HANDLE,
    build_viz_session,
    install_fake_reference,
    write_align_source,
)


def test_open_with_single_align_source(tmp_path: Path, monkeypatch) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        features=[
            {
                "feature_id": 1,
                "contig_id": 1,
                "type": "gene",
                "name": "gene1",
                "start": 0,
                "end": 1000,
                "strand": "+",
                "parent_id": None,
                "biotype": None,
                "source": "test",
                "attributes": {},
            },
        ],
        align_sources=[
            {
                "coverage": [
                    {
                        "contig_id": 1,
                        "sample_id": -1,
                        "start": 0,
                        "end": 100,
                        "depth": 5,
                    }
                ],
            }
        ],
    )
    assert session.reference_handle == DEFAULT_HANDLE
    assert session.assembly_accession == DEFAULT_ASSEMBLY
    assert session.reference_genome.name == "genome"
    assert session.reference_annotation is not None
    assert len(session.sources) == 1
    src = session.sources[0]
    assert src.kind == "align"
    assert src.coverage is not None and src.coverage.name == "coverage.parquet"
    assert session.warnings == ()


def test_open_multi_source_emits_one_session_per_kind(
    tmp_path: Path, monkeypatch
) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[
            {
                "coverage": [
                    {"contig_id": 1, "sample_id": -1, "start": 0, "end": 50, "depth": 3}
                ],
            },
            {
                "coverage": [
                    {"contig_id": 1, "sample_id": -1, "start": 50, "end": 100, "depth": 7}
                ],
            },
        ],
        cluster_sources=[
            {
                "clusters": [],
                "cluster_membership": [],
            }
        ],
    )
    kinds = [src.kind for src in session.sources]
    assert kinds == ["align", "align", "cluster"]
    assert session.stages_present()["coverage"] is True
    assert session.stages_present()["has_align_sources"] is True
    assert session.stages_present()["has_cluster_sources"] is True


def test_open_mismatched_assembly_emits_warning(
    tmp_path: Path, monkeypatch
) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        assembly_accession="ChosenAssembly.X",
        align_sources=[
            {
                "assembly_accession": "OtherAssembly.Y",
                "coverage": [
                    {"contig_id": 1, "sample_id": -1, "start": 0, "end": 10, "depth": 1}
                ],
            }
        ],
    )
    assert any("OtherAssembly.Y" in w for w in session.warnings)
    # Same-assembly source emits no warning even when handle differs.
    session2 = build_viz_session(
        tmp_path / "second",
        monkeypatch,
        assembly_accession="ChosenAssembly.X",
        align_sources=[
            {
                "assembly_accession": "ChosenAssembly.X",
                "coverage": [
                    {"contig_id": 1, "sample_id": -1, "start": 0, "end": 10, "depth": 1}
                ],
            }
        ],
    )
    assert session2.warnings == ()


def test_open_legacy_manifest_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    cache_root = tmp_path / "refs"
    cache_root.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache_root))
    install_fake_reference(cache_root)
    legacy = tmp_path / "legacy-align"
    legacy.mkdir()
    (legacy / "manifest.json").write_text(
        json.dumps(
            {
                # Pre-v2 manifest shape — no schema_version=2, no kind.
                "demux_dir": "/somewhere",
                "reference": "/old/path",
                "outputs": {},
            }
        )
    )
    with pytest.raises(ValueError, match="schema_version"):
        Session.open(
            reference_handle=DEFAULT_HANDLE,
            sources=[{"path": str(legacy), "kind": "align"}],
        )


def test_open_handle_kind_mismatch_raises(
    tmp_path: Path, monkeypatch
) -> None:
    """Asserting kind='cluster' on an align source must fail loudly."""
    session_dir = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[{"coverage": []}],
    )
    # Build a fresh align source dir we can mis-tag.
    cache_root = tmp_path / "refs"
    align_dir = tmp_path / "rogue-align"
    write_align_source(
        align_dir,
        reference_path=str(cache_root / "test_org" / "local_import-20260522"),
    )
    with pytest.raises(ValueError, match="kind"):
        Session.open(
            reference_handle=DEFAULT_HANDLE,
            sources=[{"path": str(align_dir), "kind": "cluster"}],
        )
    # Sanity: session_dir was built ok and supports to_manifest().
    payload = session_dir.to_manifest()
    assert payload["session_id"] == session_dir.session_id


def test_to_manifest_round_trips(tmp_path: Path, monkeypatch) -> None:
    session = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[
            {
                "coverage": [
                    {"contig_id": 1, "sample_id": -1, "start": 0, "end": 100, "depth": 5}
                ],
            }
        ],
    )
    manifest = session.to_manifest()
    assert manifest["session_id"] == session.session_id
    assert manifest["reference"]["handle"] == session.reference_handle
    assert manifest["reference"]["assembly_accession"] == DEFAULT_ASSEMBLY
    assert manifest["sources"][0]["kind"] == "align"
    assert manifest["sources"][0]["slots"]["coverage"] is not None
    assert manifest["stages_present"]["coverage"] is True


def test_open_unknown_handle_errors(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "refs"
    cache_root.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache_root))
    with pytest.raises(ValueError, match="could not be resolved"):
        Session.open(
            reference_handle="missing_org@local_import-20260522",
            sources=[],
        )
