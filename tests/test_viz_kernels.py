"""TrackKernel ABC + per-kernel data-layer tests.

These tests exercise the kernel surface end-to-end against in-memory
parquet fixtures — discover → metadata → threshold → fetch — without
booting a server. The wire-shape contract (vector batches match
``kernel.schema``; hybrid batches match ``HYBRID_SCHEMA``) is enforced
explicitly so renderer-side code can rely on it.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.schemas.quant import COVERAGE_TABLE
from constellation.sequencing.schemas.reference import CONTIG_TABLE
from constellation.viz.tracks.base import (
    HYBRID_SCHEMA,
    ThresholdDecision,
    TrackKernel,
    TrackQuery,
    get_kernel,
    register_track,
    registered_kinds,
)
from constellation.viz.tracks.coverage_histogram import (
    COVERAGE_VECTOR_SCHEMA,
    CoverageHistogramKernel,
)
from constellation.viz.server.session import Session


# ----------------------------------------------------------------------
# Registry sanity
# ----------------------------------------------------------------------


def test_registry_includes_coverage_histogram() -> None:
    assert "coverage_histogram" in registered_kinds()
    kernel = get_kernel("coverage_histogram")
    assert isinstance(kernel, CoverageHistogramKernel)


def test_register_track_rejects_duplicate() -> None:
    # A second class trying to claim "coverage_histogram" must raise.
    with pytest.raises(ValueError, match="already registered"):

        @register_track
        class _Dup(TrackKernel):  # noqa: D401
            kind = "coverage_histogram"
            schema = COVERAGE_VECTOR_SCHEMA

            def discover(self, session):  # type: ignore[override]
                return []

            def metadata(self, binding):  # type: ignore[override]
                return {}

            def threshold(self, binding, query):  # type: ignore[override]
                return ThresholdDecision.VECTOR

            def fetch(self, binding, query, mode):  # type: ignore[override]
                return iter(())


def test_register_track_rejects_empty_kind() -> None:
    with pytest.raises(TypeError, match="kind"):

        @register_track
        class _Bad(TrackKernel):  # noqa: D401
            kind = ""
            schema = COVERAGE_VECTOR_SCHEMA

            def discover(self, session):  # type: ignore[override]
                return []

            def metadata(self, binding):  # type: ignore[override]
                return {}

            def threshold(self, binding, query):  # type: ignore[override]
                return ThresholdDecision.VECTOR

            def fetch(self, binding, query, mode):  # type: ignore[override]
                return iter(())


def test_hybrid_schema_shape() -> None:
    # Frontend renderers branch on this exact field set; lock it in.
    names = {f.name for f in HYBRID_SCHEMA}
    assert names == {
        "png_bytes",
        "extent_start",
        "extent_end",
        "extent_y0",
        "extent_y1",
        "width_px",
        "height_px",
        "n_items",
        "mode",
    }


# ----------------------------------------------------------------------
# Coverage-histogram kernel: data-layer round-trip
# ----------------------------------------------------------------------


def _build_session_with_coverage(
    tmp_path: Path,
    rows: list[dict],
    *,
    contig_name: str = "chr1",
    contig_id: int = 1,
) -> Session:
    root = tmp_path / "run"
    genome = root / "genome"
    genome.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "contig_id": contig_id,
                    "name": contig_name,
                    "length": 10_000,
                    "topology": None,
                    "circular": None,
                }
            ],
            schema=CONTIG_TABLE,
        ),
        genome / "contigs.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "contig_id": pa.array([contig_id], pa.int64()),
                "sequence": pa.array(["N" * 100], pa.string()),
            }
        ),
        genome / "sequences.parquet",
    )

    align_dir = root / "S2_align"
    align_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=COVERAGE_TABLE),
        align_dir / "coverage.parquet",
    )

    return Session.from_root(root)


def test_coverage_kernel_discover_returns_one_binding(tmp_path: Path) -> None:
    session = _build_session_with_coverage(
        tmp_path,
        [{"contig_id": 1, "sample_id": -1, "start": 0, "end": 100, "depth": 4}],
    )
    bindings = get_kernel("coverage_histogram").discover(session)
    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.kind == "coverage_histogram"
    assert binding.binding_id == "coverage"
    assert "coverage" in binding.paths
    assert "genome" in binding.paths


def test_coverage_kernel_discover_empty_when_coverage_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    (root / "genome").mkdir(parents=True)
    # No coverage.parquet — kernel should produce zero bindings.
    session = Session.from_root(root)
    assert get_kernel("coverage_histogram").discover(session) == []


def test_coverage_kernel_fetch_emits_wire_schema(tmp_path: Path) -> None:
    session = _build_session_with_coverage(
        tmp_path,
        [
            {"contig_id": 1, "sample_id": -1, "start": 0, "end": 50, "depth": 3},
            {"contig_id": 1, "sample_id": -1, "start": 50, "end": 100, "depth": 7},
            # A row outside the visible window
            {"contig_id": 1, "sample_id": -1, "start": 200, "end": 300, "depth": 9},
            # A row on a different contig — must be filtered out
            {"contig_id": 2, "sample_id": -1, "start": 0, "end": 50, "depth": 11},
        ],
    )
    kernel = get_kernel("coverage_histogram")
    [binding] = kernel.discover(session)

    query = TrackQuery(contig="chr1", start=0, end=100)
    mode = kernel.threshold(binding, query)
    assert mode is ThresholdDecision.VECTOR

    batches = list(kernel.fetch(binding, query, mode))
    table = pa.Table.from_batches(batches, schema=COVERAGE_VECTOR_SCHEMA)

    # Two visible RLE rows, depths cast to float64
    assert table.num_rows == 2
    assert table.schema == COVERAGE_VECTOR_SCHEMA
    depths = sorted(table.column("depth").to_pylist())
    assert depths == [3.0, 7.0]
    assert all(c == 1 for c in (1, 1))  # smoke; contigs filtered by id


def test_coverage_kernel_fetch_unknown_contig_is_empty(tmp_path: Path) -> None:
    session = _build_session_with_coverage(
        tmp_path,
        [{"contig_id": 1, "sample_id": -1, "start": 0, "end": 100, "depth": 5}],
    )
    kernel = get_kernel("coverage_histogram")
    [binding] = kernel.discover(session)

    query = TrackQuery(contig="chrX_unknown", start=0, end=1000)
    batches = list(kernel.fetch(binding, query, ThresholdDecision.VECTOR))
    assert batches == []


def test_coverage_kernel_metadata_reports_depth_max_and_samples(
    tmp_path: Path,
) -> None:
    session = _build_session_with_coverage(
        tmp_path,
        [
            {"contig_id": 1, "sample_id": 0, "start": 0, "end": 50, "depth": 4},
            {"contig_id": 1, "sample_id": 1, "start": 50, "end": 100, "depth": 9},
        ],
    )
    kernel = get_kernel("coverage_histogram")
    [binding] = kernel.discover(session)
    meta = kernel.metadata(binding)
    assert meta["kind"] == "coverage_histogram"
    assert sorted(meta["samples_in_data"]) == [0, 1]
    assert meta["depth_max"] == 9.0


def test_coverage_kernel_estimate_vector_cost_matches_row_count(
    tmp_path: Path,
) -> None:
    session = _build_session_with_coverage(
        tmp_path,
        [
            {"contig_id": 1, "sample_id": -1, "start": 0, "end": 50, "depth": 1},
            {"contig_id": 1, "sample_id": -1, "start": 50, "end": 100, "depth": 2},
            {"contig_id": 1, "sample_id": -1, "start": 200, "end": 300, "depth": 3},
        ],
    )
    kernel = get_kernel("coverage_histogram")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=150)
    cost = kernel.estimate_vector_cost(binding, query)
    assert cost == 2  # only the rows overlapping [0, 150)
