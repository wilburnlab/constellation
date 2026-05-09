"""Server-side rasterization helpers for hybrid-mode tracks.

Hybrid kernels (read_pileup, cluster_pileup) call into this module to
turn high-density Arrow batches into PNG bytes shipped via the
`HYBRID_SCHEMA`. The helpers in `datashader_png` are the project's
*only* sanctioned pandas boundary — datashader's aggregation kernels
require a pandas DataFrame input. We confine the conversion here so
the rest of the package stays Arrow-native (per the project-level
"no pandas inside the package" rule).
"""

from constellation.viz.raster.datashader_png import (
    rasterize_segments,
)

__all__ = ["rasterize_segments"]
