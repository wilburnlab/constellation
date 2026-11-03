"""``ms_spectrum`` — one scan's points, annotated when a peptide is given.

Read from the bundle of the peak list asked for, through row-group pruning on
the scan number (rows are in scan order). A centroided scan is sent whole; a
profile scan long enough to need it is reduced to what the requested width
shows, and a window of one keeps a point beyond each edge so its line reaches
the plot's edges. With a peptide (``modified_sequence`` and ``charge``) each peak carries the
ion it lies on (``massspec.spectra.annotate``); without one those columns
are null, so the wire schema is the same either way.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.viz.modalities.massspec.cache import scan_table_for
from constellation.viz.modalities.massspec.kernel import (
    PEAK_LIST_PREFIX,
    MassSpecTrackKernel,
    reduce_trace,
)
from constellation.viz.modalities.massspec.query import SpectrumQuery
from constellation.viz.tracks.base import ThresholdDecision, TrackBinding, register_track

#: ``massspec.spectra.annotate.ANNOTATED_PEAK_SCHEMA`` plus ``decimated`` and
#: ``profile`` (whether this scan's points are profile points, so a line, not
#: sticks; a row column because ``response_headers`` never sees the binding).
#: Restated: the viz layer imports functions from a domain module, not its
#: constants; a test holds the two equal.
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
        pa.field("profile", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_name": b"MsSpectrum"},
)


def _is_profile(bundle: Path, scan: int) -> bool:
    """Whether ``bundle`` holds this scan as profile points (a profile bundle
    keeps a mixed run's centroid-acquired scans as they were stored)."""
    scans = scan_table_for(bundle)
    numbers = scans.column("scan").to_numpy()
    at = int(np.searchsorted(numbers, scan))
    if at >= numbers.size or numbers[at] != scan:
        return False
    return scans.column("spectrum_type")[at].as_py() == "profile"


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

        bundle = binding.paths["bundle"]
        if query.peak_list is not None:
            bundle = binding.paths.get(f"{PEAK_LIST_PREFIX}{query.peak_list}")
            if bundle is None:
                raise ValueError(f"this run has no peak list {query.peak_list!r}")
        peaks = spectrum(bundle, query.scan, mz_axis="latest" if query.corrected else None)
        profile = _is_profile(bundle, query.scan)
        x_range = None
        if query.mz_start is not None:
            mz = peaks.column("mz")
            if profile:
                xs = mz.to_numpy()
                lo = max(int(np.searchsorted(xs, query.mz_start, side="left")) - 1, 0)
                hi = min(int(np.searchsorted(xs, query.mz_end, side="right")) + 1, xs.size)
                peaks = peaks.slice(lo, hi - lo)
            else:
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
                    if f.name not in ("decimated", "profile")
                }
            )
        n = annotated.num_rows
        out = annotated.append_column("decimated", pa.array([reduced] * n, pa.bool_()))
        out = out.append_column("profile", pa.array([profile] * n, pa.bool_()))
        # Built eagerly (not a generator), so a refusal is an HTTP 400.
        return iter(out.cast(SPECTRUM_SCHEMA).to_batches())


__all__ = ["SPECTRUM_SCHEMA", "SpectrumKernel"]
