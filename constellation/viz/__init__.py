"""Visualization layer — local FastAPI server + per-modality kernels.

`constellation viz <subcommand>` opens a focused tool (genome browser, etc.)
backed by a local FastAPI server that streams Apache Arrow IPC over HTTP from
the existing partitioned-parquet datasets the CLI pipeline produces.

Architectural invariants:

- **Arrow-first wire**. Server streams `pa.RecordBatch` chunks via
  `pa.ipc.new_stream(...)` inside a FastAPI `StreamingResponse`. Browser-side
  decode goes through the `apache-arrow` JS package.
- **SVG-only rendering** on the client. D3 is the algorithmic toolkit (scales,
  axes, brushes, transitions). Three view classes exposed: vector (pure SVG
  primitives), hybrid (SVG envelope around a Datashader PNG `<image>`), raster
  (PNG-only, thumbnails). Hybrid is the interactive default for dense kernels;
  vector is always available for export.
- **Per-modality kernels** live under `constellation.viz.tracks.*`, register
  via `@register_track`, and own their visual vocabulary + threshold logic.
  Each kernel is mirror-symmetric to a TS module under
  `constellation/viz/frontend/src/track_renderers/<kind>.ts`.
- **Read-only over parquet**. The viz server never writes pipeline outputs.
  Long-running compute stays in CLI/notebook; the GUI only consumes.

Public surface (re-exports from submodules):

    from constellation.viz import (
        TrackKernel, TrackBinding, TrackQuery, ThresholdDecision,
        HYBRID_SCHEMA, register_track, get_kernel, registered_kinds,
    )
"""

from __future__ import annotations

# Importing tracks pulls in every kernel module so they can self-register
# via `@register_track` at module load time. Order matches the module layout
# in docs/plans/oh-your-dashboard-design-floofy-scott.md. Kernels are added
# here as their modules land; until then their `kind` strings are simply
# absent from the registry, and the server returns 404 for them.
from constellation.viz.tracks import (  # noqa: F401
    base,
    coverage_histogram,
    gene_annotation,
    reference_sequence,
    splice_junctions,
    read_pileup,
    cluster_pileup,
)
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
