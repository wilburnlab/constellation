"""Per-modality track kernels for the genome-browser-style viz.

Each kernel module imports `register_track` from `tracks.base` and decorates
its concrete `TrackKernel` subclass at module load. Importing
`constellation.viz` (the parent package) imports every kernel module here
exactly once, populating the registry.

Kernels (reading slot names per `viz.server.session.SessionSource`):

- `reference_sequence` — vector; per-base letters at high zoom, dashes when
  decimated; reads `session.reference_genome / "sequences.parquet"`.
- `gene_annotation`    — vector; rectangles for exons, lines for introns,
  arrows for strand; reads `session.reference_annotation / "features.parquet"`
  (preferred) plus one binding per source's `derived_annotation/features.parquet`.
- `coverage_histogram` — vector; one binding per align source's `coverage.parquet`.
- `read_pileup`        — hybrid (datashader at low zoom, per-glyph high zoom);
  one binding per align source's `alignments/` (+ optional `alignment_blocks/`).
- `cluster_pileup`     — hybrid; one binding per cluster source's
  `clusters.parquet` + `cluster_membership.parquet`.
- `splice_junctions`   — vector; one binding per align source's `introns.parquet`.

The mirror-symmetric TS renderer for each kernel lives under
`constellation/viz/frontend/src/track_renderers/<kind>.ts`.
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
