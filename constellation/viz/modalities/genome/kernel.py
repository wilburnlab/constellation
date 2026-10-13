"""Base class for the genome browser's track kernels.

Holds what the six genome kernels share and the modality-neutral
``TrackKernel`` ABC should not know: that they belong to the ``genome``
modality, that their queries are loci, and the base-pairs-per-pixel
threshold the two pile-up kernels use to fall back to a raster.
"""

from __future__ import annotations

from typing import ClassVar

from constellation.viz.modalities.genome.query import GenomeQuery
from constellation.viz.tracks.base import ThresholdDecision, TrackKernel, TrackQuery


class GenomeTrackKernel(TrackKernel):
    modality: ClassVar[str] = "genome"

    #: Queries are loci; a kernel that reads more declares a subclass.
    query_model: ClassVar[type[TrackQuery]] = GenomeQuery

    #: Above this many base pairs per pixel a dense kernel rasterizes
    #: instead of emitting one glyph per item. Kernels that are always
    #: vector leave it unused.
    vector_bp_per_pixel_limit: ClassVar[float] = 50.0

    def response_headers(
        self, query: TrackQuery, mode: ThresholdDecision
    ) -> dict[str, str]:
        # Every genome track reports the cluster view that was asked for
        # (empty when none was, or when the kernel has no such view), so
        # the client can branch before inspecting the payload's columns.
        return {"X-Track-View": getattr(query, "cluster_view", None) or ""}
