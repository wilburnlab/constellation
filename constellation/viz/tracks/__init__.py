"""Track-kernel contract and registry.

``tracks.base`` defines what a kernel is — the ``TrackKernel`` ABC, the
query / binding records, the hybrid wire schema — and holds the registry
kernels join via ``@register_track``.

The kernels themselves belong to a modality and live under
``constellation.viz.modalities.<modality>.tracks`` (the genome browser's
six are in ``viz.modalities.genome.tracks``). Importing
``constellation.viz`` imports every shipped modality, which populates
the registry.
"""

from constellation.viz.tracks.base import (
    HYBRID_SCHEMA,
    ThresholdDecision,
    TrackBinding,
    TrackKernel,
    TrackQuery,
    get_kernel,
    register_track,
    registered_kinds,
)

__all__ = [
    "HYBRID_SCHEMA",
    "ThresholdDecision",
    "TrackBinding",
    "TrackKernel",
    "TrackQuery",
    "get_kernel",
    "register_track",
    "registered_kinds",
]
