"""Transcript-cluster pile-up track — isoform-resolution stacked bars.

Reads `<root>/S2_cluster/clusters.parquet` (`TRANSCRIPT_CLUSTER_TABLE`)
and `<root>/S2_cluster/cluster_membership.parquet` (latter referenced
in metadata for a future "expand cluster to member reads" affordance;
v1 ships cluster-level glyphs only).

Each cluster becomes one horizontal bar from `span_start` to
`span_end`, stacked greedily so overlapping isoforms don't collide.
Hybrid threshold mirrors `read_pileup` — bp/pixel + cluster count.
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


CLUSTER_PILEUP_VECTOR_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("cluster_id", pa.int64(), nullable=False),
        pa.field("span_start", pa.int64(), nullable=False),
        pa.field("span_end", pa.int64(), nullable=False),
        pa.field("strand", pa.string(), nullable=False),
        pa.field("n_reads", pa.int32(), nullable=False),
        pa.field("row", pa.int32(), nullable=False),
        pa.field("mode", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"VizClusterPileup"},
)


@register_track
class ClusterPileupKernel(TrackKernel):
    """Per-cluster pile-up track."""

    kind = "cluster_pileup"
    schema = CLUSTER_PILEUP_VECTOR_SCHEMA

    # Cluster counts are an order of magnitude smaller than read counts
    # at the same locus; the threshold is correspondingly higher.
    vector_glyph_limit = 6_000
    vector_bp_per_pixel_limit = 50.0

    def discover(self, session: Session) -> list[TrackBinding]:
        if session.clusters is None or session.reference_genome is None:
            return []
        paths: dict[str, Any] = {
            "clusters": session.clusters,
            "genome": session.reference_genome,
        }
        if session.cluster_membership is not None:
            paths["cluster_membership"] = session.cluster_membership
        return [
            TrackBinding(
                session_id=session.session_id,
                kind=self.kind,
                binding_id="cluster_pileup",
                label="Transcript clusters",
                paths=paths,
                config={},
            )
        ]

    def metadata(self, binding: TrackBinding) -> dict[str, Any]:
        # Surface the unique cluster modes so the renderer can color
        # genome-guided vs de-novo distinctly when both are present.
        path = binding.paths["clusters"]
        modes: list[str] = []
        if path.exists():
            table = pq.read_table(path, columns=["mode"])
            modes = sorted({m for m in table.column("mode").to_pylist() if m})
        return {
            "kind": self.kind,
            "binding_id": binding.binding_id,
            "label": binding.label,
            "modes_in_data": modes,
            "vector_glyph_limit": self.vector_glyph_limit,
            "vector_bp_per_pixel_limit": self.vector_bp_per_pixel_limit,
            "default_height_px": 200,
        }

    def threshold(
        self, binding: TrackBinding, query: TrackQuery
    ) -> ThresholdDecision:
        if query.force is not None:
            return query.force
        bp_per_pixel = (query.end - query.start) / max(1, query.viewport_px)
        if bp_per_pixel > self.vector_bp_per_pixel_limit:
            return ThresholdDecision.HYBRID
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
        contig_id = _resolve_contig_id(binding.paths["genome"], query.contig)
        if contig_id is None:
            return iter(())

        rows = self._scan_window(binding, contig_id, query)
        if rows.num_rows == 0:
            return iter(())

        starts = rows.column("span_start").to_pylist()
        ends = rows.column("span_end").to_pylist()
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_in_window(
        self, binding: TrackBinding, query: TrackQuery
    ) -> int:
        contig_id = _resolve_contig_id(binding.paths["genome"], query.contig)
        if contig_id is None:
            return 0
        # clusters.parquet is a single file (not a partitioned dataset)
        # in the current pipeline output; pa.dataset still handles it.
        dataset = pa_ds.dataset(str(binding.paths["clusters"]), format="parquet")
        return int(dataset.count_rows(filter=self._predicate(contig_id, query)))

    def _scan_window(
        self, binding: TrackBinding, contig_id: int, query: TrackQuery
    ) -> pa.Table:
        dataset = pa_ds.dataset(str(binding.paths["clusters"]), format="parquet")
        scanner = dataset.scanner(
            columns=[
                "cluster_id",
                "span_start",
                "span_end",
                "strand",
                "n_reads",
                "mode",
            ],
            filter=self._predicate(contig_id, query),
        )
        return scanner.to_table()

    @staticmethod
    def _predicate(contig_id: int, query: TrackQuery) -> Any:
        contig_field = pc.field("contig_id")
        span_start = pc.field("span_start")
        span_end = pc.field("span_end")
        return (
            (contig_field == pa.scalar(contig_id, pa.int64()))
            & (span_end > pa.scalar(int(query.start), pa.int64()))
            & (span_start < pa.scalar(int(query.end), pa.int64()))
        )

    def _emit_vector(
        self, rows: pa.Table, assigned: list[int]
    ) -> Iterator[pa.RecordBatch]:
        # n_reads in TRANSCRIPT_CLUSTER_TABLE is int32; strand may
        # be null for unstranded clusters — fill with '.' so the wire
        # schema's non-null promise holds.
        strand = rows.column("strand")
        if strand.null_count:
            strand = pc.fill_null(strand, ".")
        out = pa.Table.from_arrays(
            [
                rows.column("cluster_id"),
                rows.column("span_start"),
                rows.column("span_end"),
                strand,
                rows.column("n_reads"),
                pa.array(assigned, pa.int32()),
                rows.column("mode"),
            ],
            schema=CLUSTER_PILEUP_VECTOR_SCHEMA,
        )
        return iter(out.to_batches())

    def _emit_hybrid(
        self, rows: pa.Table, assigned: list[int], query: TrackQuery
    ) -> Iterator[pa.RecordBatch]:
        from constellation.viz.raster.datashader_png import rasterize_segments

        starts = rows.column("span_start").to_pylist()
        ends = rows.column("span_end").to_pylist()
        n_rows = max(assigned) + 1 if assigned else 1
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


def _resolve_contig_id(genome_dir: Any, contig_name: str) -> int | None:
    contigs_path = genome_dir / "contigs.parquet"
    if not contigs_path.exists():
        return None
    table = pq.read_table(contigs_path, columns=["contig_id", "name"])
    matches = table.filter(pc.field("name") == pa.scalar(contig_name, pa.string()))
    if matches.num_rows == 0:
        return None
    return int(matches.column("contig_id")[0].as_py())
