"""Per-modality track kernels for the genome-browser-style viz.

Each kernel module imports `register_track` from `tracks.base` and decorates
its concrete `TrackKernel` subclass at module load. Importing
`constellation.viz` (the parent package) imports every kernel module here
exactly once, populating the registry.

Kernels shipped in PR 1:

- `reference_sequence` — vector; per-base letters at high zoom, dashes when
  decimated; reads `<root>/genome/SEQUENCE_TABLE`.
- `gene_annotation`    — vector; rectangles for exons, lines for introns,
  arrows for strand; reads `<root>/annotation/FEATURE_TABLE` (preferred) or
  `<root>/S2_align/derived_annotation/`.
- `coverage_histogram` — vector; single `<path>` per sample; reads
  `<root>/S2_align/coverage.parquet`.
- `read_pileup`        — hybrid (datashader at low zoom, per-glyph high zoom);
  reads `<root>/S2_align/alignments/` + `alignment_blocks/`.
- `cluster_pileup`     — hybrid; reads `<root>/S2_cluster/clusters.parquet`
  + `cluster_membership.parquet`.
- `splice_junctions`   — vector; arc glyphs from `<root>/S2_align/introns.parquet`.

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
