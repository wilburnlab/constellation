"""``ms_chromatogram`` — a trace over retention time for one MS level.

Reads only the acquisition's scan table (``scan_metadata.parquet``,
cached per process): the TIC and base peak are recorded per scan, so no
peak is read. See :class:`ChromatogramQuery` for the three traces.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import pyarrow as pa
import pyarrow.compute as pc

from constellation.viz.modalities.massspec.cache import scan_table_for
from constellation.viz.modalities.massspec.kernel import MassSpecTrackKernel, reduce_trace
from constellation.viz.modalities.massspec.query import ChromatogramQuery
from constellation.viz.tracks.base import ThresholdDecision, TrackBinding, register_track

#: One row per point. ``mz`` is the base peak's m/z for ``bpc`` and the
#: scan's precursor m/z for ``scans``; ``decimated`` is true on every row
#: of a reduced trace.
TRACE_SCHEMA = pa.schema(
    [
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("rt", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64()),
        pa.field("mz", pa.float64()),
        pa.field("decimated", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_name": b"MsTrace"},
)


@register_track
class ChromatogramKernel(MassSpecTrackKernel):
    kind: ClassVar[str] = "ms_chromatogram"
    schema: ClassVar[pa.Schema] = TRACE_SCHEMA
    query_model = ChromatogramQuery

    def fetch(  # type: ignore[override]
        self, binding: TrackBinding, query: ChromatogramQuery, mode: ThresholdDecision
    ) -> Iterator[pa.RecordBatch]:
        from constellation.massspec.scans import chromatogram

        scans = scan_table_for(binding.paths["bundle"])
        if query.trace == "scans":
            mask = pc.equal(scans.column("level"), query.level)
            if query.window is not None:
                mask = pc.and_(mask, pc.equal(scans.column("window"), query.window))
            sub = scans.filter(mask)
            table = pa.table(
                {
                    "scan": sub.column("scan"),
                    "rt": sub.column("rt"),
                    "intensity": sub.column("tic").cast(pa.float64()),
                    "mz": sub.column("precursor_mz").cast(pa.float64()),
                }
            )
        else:
            table = chromatogram(scans, level=query.level, kind=query.trace, window=query.window)
            if "mz" not in table.column_names:
                table = table.append_column("mz", pa.nulls(table.num_rows, pa.float64()))

        x_range = None if query.rt_start is None else (query.rt_start, query.rt_end)
        table, reduced = reduce_trace(
            table, "rt", limit=self.decimate_above, width=query.viewport_px, x_range=x_range
        )
        out = pa.table(
            {
                "scan": table.column("scan").cast(pa.int32()),
                "rt": table.column("rt").cast(pa.float64()),
                "intensity": table.column("intensity"),
                "mz": table.column("mz"),
                "decimated": pa.array([reduced] * table.num_rows, pa.bool_()),
            },
            schema=TRACE_SCHEMA,
        )
        # Built eagerly (not a generator), so a refusal is an HTTP 400.
        return iter(out.to_batches())


__all__ = ["TRACE_SCHEMA", "ChromatogramKernel"]
