"""Session discovery — `session.toml` + per-stage fallback walk.

Builds a fixture run-directory layout in tmp_path and exercises both
explicit (``session.toml``) and implicit (directory walk) entry points
into ``constellation.viz.server.session.Session``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.schemas.quant import COVERAGE_TABLE
from constellation.sequencing.schemas.reference import CONTIG_TABLE
from constellation.viz.server.session import Session


def _write_genome(genome_dir: Path) -> None:
    genome_dir.mkdir(parents=True, exist_ok=True)
    contigs = pa.Table.from_pylist(
        [
            {
                "contig_id": 1,
                "name": "chr1",
                "length": 1_000_000,
                "topology": None,
                "circular": None,
            },
        ],
        schema=CONTIG_TABLE,
    )
    pq.write_table(contigs, genome_dir / "contigs.parquet")
    # Tiny stub for the SEQUENCE_TABLE (the reference_sequence kernel
    # consumes this; coverage_histogram doesn't but Session.discover
    # treats `genome/` as a ParquetDir bundle either way).
    pq.write_table(
        pa.table(
            {
                "contig_id": pa.array([1], pa.int64()),
                "sequence": pa.array(["A" * 100], pa.string()),
            }
        ),
        genome_dir / "sequences.parquet",
    )
    (genome_dir / "manifest.json").write_text(
        json.dumps({"format": "parquet_dir", "container": "GenomeReference"})
    )


def _write_coverage(parquet_path: Path, rows: list[dict]) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=COVERAGE_TABLE)
    pq.write_table(table, parquet_path)


# ----------------------------------------------------------------------
# Implicit discovery (no session.toml)
# ----------------------------------------------------------------------


def test_discover_walks_standard_layout(tmp_path: Path) -> None:
    root = tmp_path / "run-2026-05-09"
    _write_genome(root / "genome")
    _write_coverage(
        root / "S2_align" / "coverage.parquet",
        [{"contig_id": 1, "sample_id": -1, "start": 0, "end": 100, "depth": 5}],
    )

    session = Session.from_root(root)
    assert session.root == root.resolve()
    assert session.reference_genome == (root / "genome").resolve()
    assert session.coverage == (root / "S2_align" / "coverage.parquet").resolve()
    # Slot we didn't populate stays None
    assert session.clusters is None
    # Stages-present is the small surface the dashboard reads
    assert session.stages_present()["coverage"] is True
    assert session.stages_present()["clusters"] is False


def test_discover_session_id_is_stable_across_calls(tmp_path: Path) -> None:
    root = tmp_path / "myrun"
    root.mkdir()
    a = Session.from_root(root)
    b = Session.from_root(root)
    assert a.session_id == b.session_id
    # The id encodes the root's basename for human readability
    assert "myrun" in a.session_id


def test_discover_drops_missing_paths(tmp_path: Path) -> None:
    root = tmp_path / "run"
    # Layout exists but the parquet file is missing
    (root / "S2_align").mkdir(parents=True)
    session = Session.from_root(root)
    # _drop_missing should clear the stale slot
    assert session.coverage is None


# ----------------------------------------------------------------------
# session.toml explicit form
# ----------------------------------------------------------------------


def test_session_toml_overrides_directory_walk(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _write_genome(root / "refs" / "Pichia")
    _write_coverage(
        root / "outputs" / "coverage.parquet",
        [{"contig_id": 1, "sample_id": -1, "start": 0, "end": 50, "depth": 3}],
    )
    (root / "session.toml").write_text(
        """
schema_version = 1
label = "explicit-layout"

[reference]
genome = "refs/Pichia"

[stages.s2_align]
path = "outputs"
coverage = "coverage.parquet"
samples = ["sample_a", "sample_b"]
"""
    )
    session = Session.from_root(root)
    assert session.label == "explicit-layout"
    assert session.reference_genome == (root / "refs" / "Pichia").resolve()
    assert session.coverage == (root / "outputs" / "coverage.parquet").resolve()
    assert session.samples == ("sample_a", "sample_b")


def test_session_toml_handle_resolves_via_cache(tmp_path: Path, monkeypatch) -> None:
    """``[reference] handle = "..."`` looks up the cache and resolves to it."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    # Populate the cache with a fake install.
    release_dir = cache / "pichia_pastoris" / "ensembl_genomes-57"
    _write_genome(release_dir / "genome")
    (release_dir / "annotation").mkdir()
    (release_dir / "annotation" / "manifest.json").write_text("{}")
    # Write an empty features.parquet so Session._drop_missing keeps the slot.
    pq.write_table(
        pa.table({"feature_id": pa.array([], pa.int64())}),
        release_dir / "annotation" / "features.parquet",
    )

    root = tmp_path / "analysis"
    root.mkdir()
    (root / "session.toml").write_text(
        """
schema_version = 1
label = "handle-based"

[reference]
handle = "pichia_pastoris@ensembl_genomes-57"
"""
    )
    session = Session.from_root(root)
    assert session.reference_genome == (release_dir / "genome").resolve()
    assert session.reference_annotation == (release_dir / "annotation").resolve()


def test_session_toml_explicit_path_overrides_handle(
    tmp_path: Path, monkeypatch
) -> None:
    """When both ``[reference] genome`` and ``handle`` are set, the
    explicit path wins (per the documented precedence)."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    cache_release = cache / "pichia_pastoris" / "ensembl_genomes-57"
    _write_genome(cache_release / "genome")
    (cache_release / "annotation").mkdir()

    root = tmp_path / "analysis"
    root.mkdir()
    _write_genome(root / "local_refs" / "genome")
    (root / "session.toml").write_text(
        """
schema_version = 1

[reference]
genome = "local_refs/genome"
handle = "pichia_pastoris@ensembl_genomes-57"
"""
    )
    session = Session.from_root(root)
    # Explicit path won.
    assert session.reference_genome == (root / "local_refs" / "genome").resolve()
    # Annotation falls back to the handle since no explicit annotation path.
    assert session.reference_annotation == (cache_release / "annotation").resolve()


def test_session_toml_handle_missing_install_errors(
    tmp_path: Path, monkeypatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    root = tmp_path / "analysis"
    root.mkdir()
    (root / "session.toml").write_text(
        """
schema_version = 1

[reference]
handle = "homo_sapiens@ensembl-111"
"""
    )
    with pytest.raises(ValueError, match="no cached reference"):
        Session.from_root(root)


def test_session_toml_rejects_unknown_schema_version(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    root.mkdir()
    (root / "session.toml").write_text("schema_version = 99\n")
    with pytest.raises(ValueError, match="schema_version=99"):
        Session.from_root(root)


def test_to_manifest_round_trips(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_coverage(
        root / "S2_align" / "coverage.parquet",
        [{"contig_id": 1, "sample_id": -1, "start": 0, "end": 100, "depth": 5}],
    )
    session = Session.from_root(root)
    manifest = session.to_manifest()
    assert manifest["session_id"] == session.session_id
    assert manifest["paths"]["coverage"] == "S2_align/coverage.parquet"
    assert manifest["paths"]["clusters"] is None
    assert manifest["stages_present"]["coverage"] is True
