"""Shared base of the mass-spec kernels.

Every mass-spec kernel has one binding per acquisition of the session.
The binding carries what ``fetch`` needs (``fetch`` never sees the
session): the bundle path and the acquisition's description.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pyarrow as pa

from constellation.viz.tracks.base import (
    ThresholdDecision,
    TrackBinding,
    TrackKernel,
    TrackQuery,
)


class MassSpecTrackKernel(TrackKernel):
    """Base for the kernels of the ``massspec`` modality."""

    modality: ClassVar[str] = "massspec"

    #: A trace or spectrum with more points than this is reduced (M4) to
    #: what the requested width shows. Measured payloads: a DIA chromatogram
    #: is at most a few thousand points and a centroided spectrum under two
    #: thousand peaks, so only a DDA run's all-MS2 trace (~180,000 scans)
    #: and profile spectra cross it.
    decimate_above: ClassVar[int] = 20_000

    def discover(self, session: Any) -> list[TrackBinding]:
        return [
            TrackBinding(
                session_id=session.session_id,
                kind=self.kind,
                binding_id=f"{self.kind}:{acquisition.key}",
                label=acquisition.label,
                paths={"bundle": acquisition.bundle},
                config={
                    "source_id": acquisition.source_id,
                    "acquisition": acquisition.key,
                    "info": acquisition.info,
                },
            )
            for acquisition in session.acquisitions
        ]

    def metadata(self, binding: TrackBinding) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "binding_id": binding.binding_id,
            "label": binding.label,
            "acquisition": binding.config["acquisition"],
            **binding.config["info"],
        }

    def threshold(self, binding: TrackBinding, query: TrackQuery) -> ThresholdDecision:
        return ThresholdDecision.VECTOR


def reduce_trace(
    table: pa.Table,
    x: str,
    *,
    limit: int,
    width: int,
    x_range: tuple[float, float] | None,
) -> tuple[pa.Table, bool]:
    """``table`` as is when it has at most ``limit`` rows, else its M4
    reduction to ``width`` columns of ``x`` (see ``core.signal.decimate``),
    with whether it was reduced."""
    if table.num_rows <= limit:
        return table, False
    from constellation.core.signal.decimate import decimate_trace

    xs = table.column(x).to_numpy()
    ys = table.column("intensity").fill_null(0.0).to_numpy()
    keep = decimate_trace(xs, ys, n_bins=width, x_range=x_range)
    return table.take(pa.array(keep.numpy())), True


__all__ = ["MassSpecTrackKernel", "reduce_trace"]
