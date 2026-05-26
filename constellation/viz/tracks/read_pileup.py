"""Read pile-up track — alignments rendered as block-aware stacked bars.

Reads each align source's `alignments/` + `alignment_blocks/` +
`alignment_cs/` + `read_samples.parquet`. Block geometry (per-CIGAR
M/=/X spans) draws solid exon segments; intron gaps between adjacent
blocks draw as dotted connectors; per-base substitutions from cs:long
draw as X glyphs; the alignment's resolved sample id keys a per-sample
palette so users can visually compare coverage across samples and
toggle samples on / off via the gear popover.

**Hard-require.** `alignment_blocks/`, `alignment_cs/`, AND
`read_samples.parquet` must all be present on the source. When any is
missing, `discover()` returns no binding for that source — the upstream
align run was invoked with `--no-emit-cs-tags` (which the v4 CLI
default flipped to "on"), or the source is a legacy v3 output (which
the manifest reader already refuses earlier in the load path). The PR
3 sample-attach assumes the join target is on disk; falling back to
per-strand coloring would split visual conventions.

Two render modes:

- **Vector** (default at high zoom). Per-glyph: each read becomes one
  row in `READ_PILEUP_VECTOR_SCHEMA` with greedy-packed `row`, a nested
  `blocks` list-of-struct that the renderer iterates for solid
  exon-segment rectangles + dotted intron connectors, and a
  `mismatch_positions` int64 list for the per-base X glyphs. Above the
  `mismatch_glyph_bp_per_pixel_limit` zoom threshold the mismatch
  positions are emitted empty (cs:long parse skipped) — at coarser
  zoom the glyphs wouldn't be readable anyway.
- **Hybrid** (zoom-out / dense). Datashader rasterizes the same
  greedy-packed layout into a PNG embedded via `HYBRID_SCHEMA`. Block
  detail is not surfaced in hybrid mode (deferred to PR E in the
  pile-up overhaul plan).

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
from constellation.viz.tracks._alignment_view import (
    BLOCKS_LIST_TYPE,
    MISMATCH_POSITIONS_TYPE,
    attach_blocks,
    attach_mismatch_positions,
)
from constellation.viz.tracks.base import (
    HYBRID_SCHEMA,
    ThresholdDecision,
    TrackBinding,
    TrackKernel,
    TrackQuery,
    iter_sources_with,
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
        pa.field("blocks", BLOCKS_LIST_TYPE, nullable=False),
        pa.field("mismatch_positions", MISMATCH_POSITIONS_TYPE, nullable=False),
        # Nullable so reads in escape-hatch demux setups (no sample
        # bundle, never expected for production runs but defensive
        # against half-populated read_samples.parquet) render under the
        # palette's default color instead of disappearing.
        pa.field("sample_id", pa.int64(), nullable=True),
        pa.field("sample_name", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"VizReadPileup"},
)


@register_track
class ReadPileupKernel(TrackKernel):
    """Per-read pileup track with CIGAR-aware block geometry."""

    kind = "read_pileup"
    schema = READ_PILEUP_VECTOR_SCHEMA

    # Hybrid threshold defaults — calibrated against typical workstation
    # widths; users can override via `?force=`. Tuning lives here so it
    # can move per-deployment without touching the threshold method.
    vector_glyph_limit = 4_000
    vector_bp_per_pixel_limit = 50.0

    # Above this bp/pixel ratio the per-base mismatch X glyphs wouldn't
    # be visually resolvable, so we skip the cs:long parse entirely and
    # emit `mismatch_positions` as empty lists. The per-block exonic
    # rectangles still render at any zoom.
    mismatch_glyph_bp_per_pixel_limit = 5.0

    def discover(self, session: Session) -> list[TrackBinding]:
        if session.reference_genome is None:
            return []
        bindings: list[TrackBinding] = []
        for idx, src in iter_sources_with(
            session,
            "alignments",
            "alignment_blocks",
            "alignment_cs",
            "read_samples",
        ):
            # Hard-require: alignment_blocks/ + alignment_cs/ +
            # read_samples.parquet must all be present. PR 2's schema
            # added `blocks` + `mismatch_positions`; PR 3 adds
            # `sample_id` + `sample_name`. Each derives from one of the
            # four required artifacts. Sources missing any are skipped
            # at discovery time so the Datasets popover surfaces no
            # read_pileup binding for legacy outputs — re-running align
            # (the v4 default emits all four) restores it.
            paths: dict[str, Any] = {
                "alignments": src.alignments,
                "alignment_blocks": src.alignment_blocks,
                "alignment_cs": src.alignment_cs,
                "read_samples": src.read_samples,
                "genome": session.reference_genome,
            }
            label = f"Reads ({src.label})" if len(session.sources) > 1 else "Reads"
            bindings.append(
                TrackBinding(
                    session_id=session.session_id,
                    kind=self.kind,
                    binding_id=f"read_pileup-{idx}",
                    label=label,
                    paths=paths,
                    config={"source_id": src.source_id},
                )
            )
        return bindings

    def metadata(self, binding: TrackBinding) -> dict[str, Any]:
        # Pre-scan the source's read_samples.parquet so the
        # per-sample palette UI knows what to render before the first
        # data fetch. Two parallel lists keep the wire shape numeric
        # (matches coverage_histogram's `samples_in_data` convention
        # for the TrackSettingsPanel.numericList() reader) while still
        # surfacing the human-readable names for the picker labels.
        sample_ids: list[int] = []
        sample_names: list[str | None] = []
        rs_path = binding.paths.get("read_samples")
        if rs_path is not None and rs_path.exists():
            table = pq.read_table(
                rs_path, columns=["sample_id", "sample_name"]
            )
            if table.num_rows > 0:
                # Dedup on sample_id; pick first non-null name per id.
                seen: dict[int, str | None] = {}
                for sid, name in zip(
                    table.column("sample_id").to_pylist(),
                    table.column("sample_name").to_pylist(),
                    strict=True,
                ):
                    if sid is None:
                        continue
                    key = int(sid)
                    if key not in seen or seen[key] is None:
                        seen[key] = name
                sample_ids = sorted(seen.keys())
                sample_names = [seen[sid] for sid in sample_ids]
        return {
            "kind": self.kind,
            "binding_id": binding.binding_id,
            "label": binding.label,
            "vector_glyph_limit": self.vector_glyph_limit,
            "vector_bp_per_pixel_limit": self.vector_bp_per_pixel_limit,
            "mismatch_glyph_bp_per_pixel_limit": (
                self.mismatch_glyph_bp_per_pixel_limit
            ),
            "samples_in_data": sample_ids,
            "sample_names": sample_names,
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

        # Attach sample_id / sample_name from read_samples.parquet
        # before the row-packing step. The sample-filter pushdown then
        # drops disallowed rows before greedy_row_assign so the visible
        # stack height matches what the user asked for; assigning rows
        # first and filtering after would leave empty packing slots.
        rows = _attach_read_samples(rows, binding.paths["read_samples"])
        if query.samples:
            rows = _apply_sample_filter(rows, query.samples)
            if rows.num_rows == 0:
                return iter(())

        starts = rows.column("ref_start").to_pylist()
        ends = rows.column("ref_end").to_pylist()
        from constellation.viz.raster.datashader_png import greedy_row_assign

        assigned = greedy_row_assign(starts, ends)

        if mode is ThresholdDecision.VECTOR:
            # Attach per-CIGAR block geometry + per-base mismatch
            # positions; both columns are part of the vector wire
            # schema. Hybrid mode skips both — the rasterized PNG
            # doesn't surface per-row detail.
            blocks_path = binding.paths["alignment_blocks"]
            blocks_table = self._scan_blocks(blocks_path, rows)
            rows = attach_blocks(rows, blocks_table)
            bp_per_pixel = (query.end - query.start) / max(1, query.viewport_px)
            skip_mismatch = bp_per_pixel > self.mismatch_glyph_bp_per_pixel_limit
            rows = attach_mismatch_positions(
                rows,
                binding.paths["alignment_cs"],
                skip=skip_mismatch,
            )
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
    def _scan_blocks(blocks_path: Any, alignments: pa.Table) -> pa.Table:
        """Read just the block rows whose alignment_id matches the
        in-window alignments. Predicate pushdown bounds the I/O by the
        glyph cap, not the full `alignment_blocks/` cardinality."""
        ids = pc.unique(alignments.column("alignment_id"))
        dataset = pa_ds.dataset(str(blocks_path), format="parquet")
        return dataset.to_table(
            columns=[
                "alignment_id",
                "block_index",
                "ref_start",
                "ref_end",
                "n_match",
                "n_mismatch",
            ],
            filter=pc.field("alignment_id").isin(ids),
        )

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
                rows.column("blocks"),
                rows.column("mismatch_positions"),
                rows.column("sample_id"),
                rows.column("sample_name"),
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


def _attach_read_samples(rows: pa.Table, read_samples_path: Any) -> pa.Table:
    """Left-join the visible-window alignments against
    ``read_samples.parquet`` on ``read_id``. Cardinality of the right
    side is bounded by the unique read count in the file (~10s of M at
    PromethION scale) but the kernel's vector_glyph_limit bounds the
    left side to a few thousand rows at most — the hash-join cost is
    bounded by the left, and we read read_samples once with predicate
    pushdown on ``read_id`` so we don't materialise the whole table.

    Reads with no matching row (which the PR 1 reduction shouldn't
    produce in practice — chimera-disagreement rows are dropped at
    write time) keep ``sample_id=null``, ``sample_name=null`` so the
    renderer falls back to the palette default rather than crashing.
    """
    if rows.num_rows == 0:
        return rows.append_column(
            "sample_id", pa.array([], type=pa.int64())
        ).append_column("sample_name", pa.array([], type=pa.string()))
    read_ids = pc.unique(rows.column("read_id"))
    rs_table = pa_ds.dataset(
        str(read_samples_path), format="parquet"
    ).to_table(
        columns=["read_id", "sample_id", "sample_name"],
        filter=pc.field("read_id").isin(read_ids),
    )
    return rows.join(rs_table, keys="read_id", join_type="left outer")


def _apply_sample_filter(
    rows: pa.Table, samples: tuple[str, ...]
) -> pa.Table:
    """Drop rows whose ``sample_id`` / ``sample_name`` doesn't match
    any entry in ``samples``. Accepts either string sample names or
    stringified numeric ids (matches
    :meth:`coverage_histogram._resolve_sample_ids`'s contract), so the
    same query param feeds both kernels uniformly.
    """
    if not samples:
        return rows
    wanted_names: set[str] = set()
    wanted_ids: set[int] = set()
    for raw in samples:
        s = str(raw)
        wanted_names.add(s)
        try:
            wanted_ids.add(int(s))
        except (TypeError, ValueError):
            continue
    name_col = rows.column("sample_name")
    id_col = rows.column("sample_id")
    name_match = (
        pc.is_in(name_col, pa.array(list(wanted_names), pa.string()))
        if wanted_names
        else pa.array([False] * rows.num_rows)
    )
    id_match = (
        pc.is_in(id_col, pa.array(list(wanted_ids), pa.int64()))
        if wanted_ids
        else pa.array([False] * rows.num_rows)
    )
    # Treat nulls in either match column as False (a null sample_name
    # shouldn't be admitted just because the user passed a name filter).
    name_match = pc.fill_null(name_match, False)
    id_match = pc.fill_null(id_match, False)
    return rows.filter(pc.or_(name_match, id_match))
