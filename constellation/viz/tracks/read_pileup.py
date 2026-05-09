"""Read pile-up track — alignments rendered as stacked horizontal bars.

Reads `<root>/S2_align/alignments/` (`ALIGNMENT_TABLE`, partitioned).
Optionally consumes `<root>/S2_align/alignment_blocks/` for per-CIGAR-
block detail at deep zoom; v1 ignores the blocks dataset and ships
whole-alignment glyphs only.

Two render modes:

- **Vector** (default at high zoom). Per-glyph: each read becomes one
  rectangle in `READ_PILEUP_VECTOR_SCHEMA` with greedy-packed `row`
  and `strand` for arrow direction.
- **Hybrid** (zoom-out / dense). Datashader rasterizes the same
  greedy-packed layout into a PNG embedded via `HYBRID_SCHEMA`.

The `row` column is computed server-side via
`raster.datashader_png.greedy_row_assign` so vector and hybrid agree on
the layout — zooming between modes preserves visual structure.
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
    HYBRID_SCHEMA,
    ThresholdDecision,
    TrackBinding,
    TrackKernel,
    TrackQuery,
    register_track,
)


READ_PILEUP_VECTOR_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("alignment_id", pa.int64(), nullable=False),
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("ref_start", pa.int64(), nullable=False),
        pa.field("ref_end", pa.int64(), nullable=False),
        pa.field("strand", pa.string(), nullable=False),
        pa.field("mapq", pa.int32(), nullable=False),
        pa.field("row", pa.int32(), nullable=False),
    ],
    metadata={b"schema_name": b"VizReadPileup"},
)


@register_track
class ReadPileupKernel(TrackKernel):
    """Per-read pileup track (hybrid)."""

    kind = "read_pileup"
    schema = READ_PILEUP_VECTOR_SCHEMA

    # Hybrid threshold defaults — calibrated against typical workstation
    # widths; users can override via `?force=`. Tuning lives here so it
    # can move per-deployment without touching the threshold method.
    vector_glyph_limit = 4_000
    vector_bp_per_pixel_limit = 50.0

    def discover(self, session: Session) -> list[TrackBinding]:
        if session.alignments is None or session.reference_genome is None:
            return []
        paths: dict[str, Any] = {
            "alignments": session.alignments,
            "genome": session.reference_genome,
        }
        # `alignment_blocks/` is opt-in for v1 — kernels that want
        # per-CIGAR-block visualization can read it via this slot. We
        # surface it in `paths` even when None so future renderers
        # (mismatch coloring, soft-clip triangles) can pick it up.
        if session.alignment_blocks is not None:
            paths["alignment_blocks"] = session.alignment_blocks
        return [
            TrackBinding(
                session_id=session.session_id,
                kind=self.kind,
                binding_id="read_pileup",
                label="Reads",
                paths=paths,
                config={},
            )
        ]

    def metadata(self, binding: TrackBinding) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "binding_id": binding.binding_id,
            "label": binding.label,
            "vector_glyph_limit": self.vector_glyph_limit,
            "vector_bp_per_pixel_limit": self.vector_bp_per_pixel_limit,
            "default_height_px": 240,
        }

    def threshold(
        self, binding: TrackBinding, query: TrackQuery
    ) -> ThresholdDecision:
        if query.force is not None:
            return query.force
        # bp_per_pixel test is fast and zoom-aware; the count test
        # requires hitting the dataset. Try cheap first.
        bp_per_pixel = (query.end - query.start) / max(1, query.viewport_px)
        if bp_per_pixel > self.vector_bp_per_pixel_limit:
            return ThresholdDecision.HYBRID
        # Count alignments in window (cheap predicate-pushdown).
        n = self._count_in_window(binding, query)
        if n > self.vector_glyph_limit:
            return ThresholdDecision.HYBRID
        return ThresholdDecision.VECTOR

    def fetch(
        self,
        binding: TrackBinding,
        query: TrackQuery,
        mode: ThresholdDecision,
    ) -> Iterator[pa.RecordBatch]:
        contig_id_value = _resolve_contig_id_or_name(
            binding.paths["genome"], query.contig
        )
        if contig_id_value is None:
            return iter(())
        # ALIGNMENT_TABLE keys on `ref_name` (string), unlike COVERAGE_
        # TABLE which keys on int contig_id. We pass the literal name.
        contig_name, _contig_id = contig_id_value

        rows = self._scan_window(binding, contig_name, query)
        if rows.num_rows == 0:
            return iter(())

        starts = rows.column("ref_start").to_pylist()
        ends = rows.column("ref_end").to_pylist()
        from constellation.viz.raster.datashader_png import greedy_row_assign

        assigned = greedy_row_assign(starts, ends)

        if mode is ThresholdDecision.VECTOR:
            return self._emit_vector(rows, assigned)
        return self._emit_hybrid(rows, assigned, query)

    def estimate_vector_cost(
        self, binding: TrackBinding, query: TrackQuery
    ) -> int | None:
        return self._count_in_window(binding, query)

    # ------------------------------------------------------------------
    # Internal: dataset scan + emission
    # ------------------------------------------------------------------

    def _count_in_window(
        self, binding: TrackBinding, query: TrackQuery
    ) -> int:
        dataset = pa_ds.dataset(str(binding.paths["alignments"]), format="parquet")
        predicate = self._predicate(query.contig, query.start, query.end)
        return int(dataset.count_rows(filter=predicate))

    def _scan_window(
        self, binding: TrackBinding, contig_name: str, query: TrackQuery
    ) -> pa.Table:
        dataset = pa_ds.dataset(str(binding.paths["alignments"]), format="parquet")
        predicate = self._predicate(contig_name, query.start, query.end)
        scanner = dataset.scanner(
            columns=[
                "alignment_id",
                "read_id",
                "ref_start",
                "ref_end",
                "strand",
                "mapq",
                "is_secondary",
                "is_supplementary",
            ],
            filter=predicate,
        )
        table = scanner.to_table()
        if table.num_rows == 0:
            return table
        # Drop secondary/supplementary alignments — those clutter the
        # pile-up view. A future "show all alignments" toggle would
        # remove this filter.
        primary_mask = pc.and_(
            pc.invert(table.column("is_secondary")),
            pc.invert(table.column("is_supplementary")),
        )
        table = table.filter(primary_mask)
        return table.drop_columns(["is_secondary", "is_supplementary"])

    @staticmethod
    def _predicate(
        contig: str, start: int, end: int
    ) -> Any:
        ref_name = pc.field("ref_name")
        ref_start = pc.field("ref_start")
        ref_end = pc.field("ref_end")
        return (
            (ref_name == pa.scalar(contig, pa.string()))
            & (ref_end > pa.scalar(int(start), pa.int64()))
            & (ref_start < pa.scalar(int(end), pa.int64()))
        )

    def _emit_vector(
        self, rows: pa.Table, assigned: list[int]
    ) -> Iterator[pa.RecordBatch]:
        out = pa.Table.from_arrays(
            [
                rows.column("alignment_id"),
                rows.column("read_id"),
                rows.column("ref_start"),
                rows.column("ref_end"),
                rows.column("strand"),
                rows.column("mapq"),
                pa.array(assigned, pa.int32()),
            ],
            schema=READ_PILEUP_VECTOR_SCHEMA,
        )
        return iter(out.to_batches())

    def _emit_hybrid(
        self, rows: pa.Table, assigned: list[int], query: TrackQuery
    ) -> Iterator[pa.RecordBatch]:
        from constellation.viz.raster.datashader_png import rasterize_segments

        starts = rows.column("ref_start").to_pylist()
        ends = rows.column("ref_end").to_pylist()
        n_rows = max(assigned) + 1 if assigned else 1
        # Cap the rasterized stack height; deep stacks compress
        # vertically but we keep the range bounded so the PNG height
        # stays sensible.
        height_px = min(2_000, max(40, 6 * n_rows))
        png = rasterize_segments(
            starts=starts,
            ends=ends,
            rows=assigned,
            x_range=(int(query.start), int(query.end)),
            n_rows=n_rows,
            width_px=int(query.viewport_px),
            height_px=height_px,
        )
        frame = pa.Table.from_pydict(
            {
                "png_bytes": [png],
                "extent_start": [int(query.start)],
                "extent_end": [int(query.end)],
                "extent_y0": [0.0],
                "extent_y1": [float(n_rows)],
                "width_px": [int(query.viewport_px)],
                "height_px": [int(height_px)],
                "n_items": [int(rows.num_rows)],
                "mode": ["hybrid"],
            },
            schema=HYBRID_SCHEMA,
        )
        return iter(frame.to_batches())


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_contig_id_or_name(
    genome_dir: Any, contig_name: str
) -> tuple[str, int] | None:
    """ALIGNMENT_TABLE keys on `ref_name` (the literal contig name),
    not on the integer `contig_id` used by COVERAGE_TABLE. We still
    look up the id so callers can use it for cross-table joins; the
    name is what the predicate filters on.

    Returns `(name, contig_id)` when the contig exists, `None`
    otherwise — kernels skip the binding when the requested contig
    isn't part of the reference.
    """
    contigs_path = genome_dir / "contigs.parquet"
    if not contigs_path.exists():
        return None
    table = pq.read_table(contigs_path, columns=["contig_id", "name"])
    matches = table.filter(pc.field("name") == pa.scalar(contig_name, pa.string()))
    if matches.num_rows == 0:
        return None
    return contig_name, int(matches.column("contig_id")[0].as_py())
