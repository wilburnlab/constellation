"""``ms_spectrum`` — the peaks of one scan.

Read from ``peaks.parquet`` through row-group pruning on the scan number
(rows are in scan order). A centroided scan is sent whole; a profile scan
long enough to need it is reduced to what the requested width shows.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import pyarrow as pa
import pyarrow.compute as pc

from constellation.viz.modalities.massspec.kernel import MassSpecTrackKernel, reduce_trace
from constellation.viz.modalities.massspec.query import SpectrumQuery
from constellation.viz.tracks.base import ThresholdDecision, TrackBinding, register_track

SPECTRUM_SCHEMA = pa.schema(
    [
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        pa.field("decimated", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_name": b"MsSpectrum"},
)


@register_track
class SpectrumKernel(MassSpecTrackKernel):
    kind: ClassVar[str] = "ms_spectrum"
    schema: ClassVar[pa.Schema] = SPECTRUM_SCHEMA
    query_model = SpectrumQuery

    def fetch(  # type: ignore[override]
        self, binding: TrackBinding, query: SpectrumQuery, mode: ThresholdDecision
    ) -> Iterator[pa.RecordBatch]:
        from constellation.massspec.scans import spectrum

        peaks = spectrum(binding.paths["bundle"], query.scan)
        x_range = None
        if query.mz_start is not None:
            mz = peaks.column("mz")
            peaks = peaks.filter(
                pc.and_(pc.greater_equal(mz, query.mz_start), pc.less_equal(mz, query.mz_end))
            )
            x_range = (query.mz_start, query.mz_end)
        peaks, reduced = reduce_trace(
            peaks, "mz", limit=self.decimate_above, width=query.viewport_px, x_range=x_range
        )
        out = pa.table(
            {
                "mz": peaks.column("mz"),
                "intensity": peaks.column("intensity"),
                "decimated": pa.array([reduced] * peaks.num_rows, pa.bool_()),
            },
            schema=SPECTRUM_SCHEMA,
        )
        # Built eagerly (not a generator), so a refusal is an HTTP 400.
        return iter(out.to_batches())


__all__ = ["SPECTRUM_SCHEMA", "SpectrumKernel"]
