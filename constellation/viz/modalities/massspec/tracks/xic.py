"""``ms_xic`` — extracted-ion chromatograms of given m/z values.

One trace per m/z over one MS level (and DIA window) within an RT window,
zero where the ion is absent (``massspec.scans.xic``): from the bundle's
peak index when it has one, otherwise from ``peaks.parquet`` restricted to
the window, which takes a fraction of a second for a few minutes of a run
and several seconds for all of it. The m/z values say what to trace — a
precursor's isotopes, a peptide's fragments, a peak a user clicked — so
one kernel serves them all.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import pyarrow as pa

from constellation.viz.modalities.massspec.cache import scan_table_for
from constellation.viz.modalities.massspec.kernel import MassSpecTrackKernel
from constellation.viz.modalities.massspec.query import XicQuery
from constellation.viz.tracks.base import ThresholdDecision, TrackBinding, register_track

#: ``massspec.scans.XIC_SCHEMA`` with each trace's ``label`` (restated, as
#: for the spectrum schema).
XIC_WIRE_SCHEMA = pa.schema(
    [
        pa.field("trace", pa.int16(), nullable=False),
        pa.field("label", pa.string()),
        pa.field("mz_target", pa.float64(), nullable=False),
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("rt", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        pa.field("mz_observed", pa.float64()),
    ],
    metadata={b"schema_name": b"MsXic"},
)


@register_track
class XicKernel(MassSpecTrackKernel):
    kind: ClassVar[str] = "ms_xic"
    schema: ClassVar[pa.Schema] = XIC_WIRE_SCHEMA
    query_model = XicQuery

    def fetch(  # type: ignore[override]
        self, binding: TrackBinding, query: XicQuery, mode: ThresholdDecision
    ) -> Iterator[pa.RecordBatch]:
        from constellation.massspec.scans import xic

        bundle = binding.paths["bundle"]
        traces = xic(
            bundle,
            list(query.mz),
            level=query.level,
            window=query.window,
            tolerance=query.tolerance,
            tolerance_unit=query.tolerance_unit,
            rt_range=None if query.rt_start is None else (query.rt_start, query.rt_end),
            scans=scan_table_for(bundle),
        )
        labels = list(query.label) or [f"{m:.4f}" for m in query.mz]
        index = traces.column("trace").to_numpy()
        label = pa.array([labels[i] for i in index], pa.string())
        out = traces.add_column(1, "label", label).cast(XIC_WIRE_SCHEMA)
        return iter(out.to_batches())


__all__ = ["XIC_WIRE_SCHEMA", "XicKernel"]
