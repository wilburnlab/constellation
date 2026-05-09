"""Coverage histogram track — per-sample read depth as RLE step series.

Reads `<root>/S2_align/coverage.parquet` (`COVERAGE_TABLE` schema, RLE
intervals: `contig_id`, `sample_id`, `start`, `end`, `depth`). Emits a
single vector payload per query; the TS renderer draws one `<path>` per
sample.

Always vector — coverage is naturally a 1D summary regardless of the
underlying read count, so we never rasterize. The path point count
scales with the number of RLE intervals in the visible window, which is
bounded by the underlying RLE compression and stays modest at any zoom.

Wire schema diverges intentionally from `COVERAGE_TABLE`:

- We project away `contig_id` (the request was already contig-scoped).
- `depth` widens to `float64` so downstream smoothing / log scales work
  uniformly.
- `sample_id` stays as the integer the upstream pipeline wrote
  (sentinel `-1` = unstratified). The frontend resolves to sample names
  via the `metadata` payload.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

from constellation.viz.server.session import Session
from constellation.viz.tracks.base import (
    ThresholdDecision,
    TrackBinding,
    TrackKernel,
    TrackQuery,
    register_track,
)


COVERAGE_VECTOR_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("depth", pa.float64(), nullable=False),
        pa.field("sample_id", pa.int64(), nullable=False),
    ],
    metadata={b"schema_name": b"VizCoverageHistogram"},
)


@register_track
class CoverageHistogramKernel(TrackKernel):
    """Per-sample read-depth track."""

    kind = "coverage_histogram"
    schema = COVERAGE_VECTOR_SCHEMA

    # Coverage tracks are always vector; thresholds are unused but kept
    # for symmetry with hybrid kernels (a future opt-in raster mode for
    # extreme zooms could be threaded through here without changing the
    # ABC).
    vector_glyph_limit = 200_000  # large RLE expansions stay legal
    vector_bp_per_pixel_limit = float("inf")

    def discover(self, session: Session) -> list[TrackBinding]:
        if session.coverage is None or session.reference_genome is None:
            return []
        return [
            TrackBinding(
                session_id=session.session_id,
                kind=self.kind,
                binding_id="coverage",
                label="Coverage (depth)",
                paths={
                    "coverage": session.coverage,
                    "genome": session.reference_genome,
                },
                config={
                    # Renderer chooses palette per-sample; supply a
                    # sensible default order so colors are stable across
                    # reloads of the same session.
                    "samples": list(session.samples),
                },
            )
        ]

    def metadata(self, binding: TrackBinding) -> dict[str, Any]:
        # Min/max depth is useful for the renderer's y-scale; we read
        # the full coverage table once (it's small — compressed RLE).
        # Samples come from the binding config when known, else from
        # the table.
        path = binding.paths["coverage"]
        table = pq.read_table(path, columns=["sample_id", "depth"])
        if table.num_rows == 0:
            samples: list[int] = []
            depth_max = 0.0
        else:
            samples = sorted(set(table.column("sample_id").to_pylist()))
            depth_max = float(pc.max(table.column("depth")).as_py() or 0.0)
        return {
            "kind": self.kind,
            "binding_id": binding.binding_id,
            "label": binding.label,
            "samples_in_data": samples,
            "depth_max": depth_max,
            "default_height_px": 80,
        }

    def threshold(
        self, binding: TrackBinding, query: TrackQuery
    ) -> ThresholdDecision:
        # Honor an explicit force=hybrid (rasterize anyway) but the
        # default is always vector.
        if query.force is not None:
            return query.force
        return ThresholdDecision.VECTOR

    def fetch(
        self,
        binding: TrackBinding,
        query: TrackQuery,
        mode: ThresholdDecision,
    ) -> Iterator[pa.RecordBatch]:
        if mode is not ThresholdDecision.VECTOR:
            # Forced hybrid is a future enhancement; for now coverage
            # always renders vector. Yielding an empty stream returns a
            # valid IPC frame the client can render as "no data."
            return iter(())

        contig_id = _resolve_contig_id(binding.paths["genome"], query.contig)
        if contig_id is None:
            return iter(())

        dataset = pa_ds.dataset(str(binding.paths["coverage"]), format="parquet")

        # COVERAGE_TABLE rows are half-open intervals [start, end). The
        # visible window is [query.start, query.end). A row is in scope
        # iff its interval overlaps the window.
        contig_field = pc.field("contig_id")
        start_field = pc.field("start")
        end_field = pc.field("end")
        predicate = (
            (contig_field == pa.scalar(contig_id, pa.int64()))
            & (end_field > pa.scalar(int(query.start), pa.int64()))
            & (start_field < pa.scalar(int(query.end), pa.int64()))
        )
        if query.samples:
            sample_ids = _resolve_sample_ids(dataset, query.samples)
            if sample_ids:
                predicate = predicate & pc.field("sample_id").isin(
                    pa.array(sample_ids, pa.int64())
                )

        scanner = dataset.scanner(
            columns=["start", "end", "depth", "sample_id"],
            filter=predicate,
        )
        return _project_batches(scanner.to_batches())

    def estimate_vector_cost(
        self, binding: TrackBinding, query: TrackQuery
    ) -> int | None:
        # One glyph per RLE interval per sample. Quick scan via the
        # filtered dataset's row count; we don't materialize the rows.
        contig_id = _resolve_contig_id(binding.paths["genome"], query.contig)
        if contig_id is None:
            return 0
        dataset = pa_ds.dataset(str(binding.paths["coverage"]), format="parquet")
        contig_field = pc.field("contig_id")
        start_field = pc.field("start")
        end_field = pc.field("end")
        predicate = (
            (contig_field == pa.scalar(contig_id, pa.int64()))
            & (end_field > pa.scalar(int(query.start), pa.int64()))
            & (start_field < pa.scalar(int(query.end), pa.int64()))
        )
        return int(dataset.count_rows(filter=predicate))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_contig_id(genome_dir: "Any", contig_name: str) -> int | None:
    """Look up `contig_id` for the requested contig name in the
    reference's `CONTIG_TABLE` (`<genome_dir>/contigs.parquet`). Returns
    `None` when the contig is unknown to this reference."""
    contigs_path = genome_dir / "contigs.parquet"
    if not contigs_path.exists():
        return None
    table = pq.read_table(contigs_path, columns=["contig_id", "name"])
    matches = table.filter(pc.field("name") == pa.scalar(contig_name, pa.string()))
    if matches.num_rows == 0:
        return None
    return int(matches.column("contig_id")[0].as_py())


def _resolve_sample_ids(
    dataset: pa_ds.Dataset, names_or_ids: tuple[str, ...]
) -> list[int]:
    """Best-effort: callers may pass either string sample names or
    string-encoded integer ids. We resolve integers directly; non-numeric
    strings return an empty list (the kernel falls back to "all
    samples"). Sample-name → id resolution is deferred to PR 2 once the
    Samples container is wired into the session manifest."""
    out: list[int] = []
    for raw in names_or_ids:
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def _project_batches(
    batches: Iterator[pa.RecordBatch],
) -> Iterator[pa.RecordBatch]:
    """Project COVERAGE_TABLE batches into the wire schema.

    `depth` widens int32 → float64; column order is fixed by the wire
    schema. We re-emit each batch as a `RecordBatch` whose schema is
    `COVERAGE_VECTOR_SCHEMA` so the IPC writer accepts it.
    """
    for batch in batches:
        if batch.num_rows == 0:
            continue
        out = pa.RecordBatch.from_arrays(
            [
                batch.column("start"),
                batch.column("end"),
                batch.column("depth").cast(pa.float64()),
                batch.column("sample_id"),
            ],
            schema=COVERAGE_VECTOR_SCHEMA,
        )
        yield out
