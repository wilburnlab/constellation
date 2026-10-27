"""``ms_spectrum`` — the peaks of one scan, annotated when a peptide is given.

Read from the bundle's peaks through row-group pruning on the scan number
(rows are in scan order). A centroided scan is sent whole; a profile scan
long enough to need it is reduced to what the requested width shows.
With a peptide (``modified_sequence`` and ``charge``) each peak carries the
ion it lies on (``massspec.spectra.annotate``); without one those columns
are null, so the wire schema is the same either way.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import pyarrow as pa
import pyarrow.compute as pc

from constellation.viz.modalities.massspec.kernel import MassSpecTrackKernel, reduce_trace
from constellation.viz.modalities.massspec.query import SpectrumQuery
from constellation.viz.tracks.base import ThresholdDecision, TrackBinding, register_track

#: ``massspec.spectra.annotate.ANNOTATED_PEAK_SCHEMA`` plus ``decimated``
#: (restated: the viz layer imports functions from a domain module, not
#: its constants; a test holds the two equal).
SPECTRUM_SCHEMA = pa.schema(
    [
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        pa.field("annotation", pa.string()),
        pa.field("ion_type", pa.string()),
        pa.field("ion_number", pa.int16()),
        pa.field("fragment_charge", pa.int8()),
        pa.field("loss", pa.string()),
        pa.field("mz_theoretical", pa.float64()),
        pa.field("error_da", pa.float64()),
        pa.field("error_ppm", pa.float64()),
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
        from constellation.massspec.spectra.annotate import annotate_spectrum

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
        if query.modified_sequence:
            try:
                annotated = annotate_spectrum(
                    peaks.column("mz").to_numpy(),
                    peaks.column("intensity").to_numpy(),
                    query.modified_sequence,
                    charge=query.charge,
                    tolerance=query.tolerance,
                    tolerance_unit=query.tolerance_unit,
                    ion_types=query.ion_types or ("b", "y"),
                    max_fragment_charge=query.max_fragment_charge,
                    neutral_losses=list(query.neutral_losses) or None,
                )
            except (KeyError, NotImplementedError) as exc:  # an unknown modification or loss
                raise ValueError(f"cannot annotate with {query.modified_sequence!r}: {exc}") from exc
        else:
            annotated = pa.table(
                {
                    f.name: peaks.column(f.name) if f.name in ("mz", "intensity") else pa.nulls(peaks.num_rows, f.type)
                    for f in SPECTRUM_SCHEMA
                    if f.name != "decimated"
                }
            )
        out = annotated.append_column("decimated", pa.array([reduced] * annotated.num_rows, pa.bool_()))
        # Built eagerly (not a generator), so a refusal is an HTTP 400.
        return iter(out.cast(SPECTRUM_SCHEMA).to_batches())


__all__ = ["SPECTRUM_SCHEMA", "SpectrumKernel"]
