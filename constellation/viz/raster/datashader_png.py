"""Datashader → PNG raster helper for hybrid-mode track payloads.

`rasterize_segments(segments, *, x_range, y_rows, width_px, height_px,
cmap)` returns the PNG bytes the hybrid-mode track kernels embed in
`HYBRID_SCHEMA.png_bytes`. Each segment is a horizontal bar drawn at a
specific row index; datashader handles the aggregation that turns N
overlapping bars into per-pixel density.

This is the **only** module in the package that imports `pandas`.
Datashader's `Canvas.line()` requires a DataFrame input; rather than
spread that requirement across the kernels, we localize it here. The
rest of the package stays Arrow-native.
"""

from __future__ import annotations

import io

import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
from PIL import Image


def rasterize_segments(
    *,
    starts,
    ends,
    rows,
    x_range: tuple[int, int],
    n_rows: int,
    width_px: int,
    height_px: int,
    cmap: tuple[str, ...] = ("#888888", "#1f77b4", "#0d3b66"),
) -> bytes:
    """Rasterize horizontal segments into a PNG.

    Parameters
    ----------
    starts, ends, rows
        Parallel sequences of equal length describing each segment as
        `(start_x, end_x, row_index)`. Coordinates are in genomic-base
        space; rows in slot-index space.
    x_range
        `(x_min, x_max)` for the horizontal axis. `x_min < x_max`.
    n_rows
        Total stack depth (used for `y_range = (-0.5, n_rows - 0.5)`
        so each row lands on its integer center pixel).
    width_px, height_px
        Output image dimensions. The frontend mounts this directly as
        an `<image>` whose `width`/`height` match the SVG envelope.
    cmap
        Density palette — light-to-dark; 1px-thick segments at low
        density show the lightest tone, dense overlaps the darkest.

    Returns
    -------
    bytes
        PNG-encoded image suitable for direct insertion into
        `HYBRID_SCHEMA.png_bytes`. An empty input renders a blank
        canvas of the requested size — the renderer treats this as
        "viewport had no data" without special-casing the schema.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError("width_px and height_px must be positive")
    x_min, x_max = x_range
    if x_min >= x_max:
        raise ValueError(f"x_range invalid: ({x_min}, {x_max})")

    # Empty input → blank canvas.
    if len(starts) == 0:
        img = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # Datashader's line aggregator wants two rows per segment with
    # `pd.NaN` separators between segments to break the line. We build
    # a flat (x, y) frame:
    #
    #   x: [s1, e1, NaN, s2, e2, NaN, ...]
    #   y: [r1, r1, NaN, r2, r2, NaN, ...]
    n = len(starts)
    if not (n == len(ends) == len(rows)):
        raise ValueError("starts/ends/rows must have equal length")
    xs: list[float] = []
    ys: list[float] = []
    for s, e, r in zip(starts, ends, rows):
        xs.extend([float(s), float(e), float("nan")])
        ys.extend([float(r), float(r), float("nan")])
    df = pd.DataFrame({"x": xs, "y": ys})

    canvas = ds.Canvas(
        plot_width=int(width_px),
        plot_height=int(height_px),
        x_range=(float(x_min), float(x_max)),
        y_range=(-0.5, max(0.5, float(n_rows) - 0.5)),
    )
    agg = canvas.line(df, "x", "y", agg=ds.count())
    shaded = tf.shade(agg, cmap=list(cmap), how="log")
    pil = shaded.to_pil()
    if pil.size != (width_px, height_px):
        pil = pil.resize((width_px, height_px))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def greedy_row_assign(starts, ends) -> list[int]:
    """Greedy row assignment for visible reads.

    Sorts reads by `start` and places each in the lowest-numbered row
    whose previous read's `end` is `<=` this read's `start`. Returns
    the row index for each input read in input order.

    Used by both the vector and hybrid renderers so the row assignment
    is identical regardless of which mode the kernel ships — zooming
    in (vector) and zooming out (hybrid) preserve the visual layout.
    """
    if len(starts) == 0:
        return []
    if len(starts) != len(ends):
        raise ValueError("starts and ends must have equal length")
    n = len(starts)
    indexed = sorted(range(n), key=lambda i: (int(starts[i]), int(ends[i])))
    row_ends: list[int] = []
    rows = [0] * n
    for i in indexed:
        s = int(starts[i])
        e = int(ends[i])
        placed = False
        for r, ridx in enumerate(row_ends):
            if ridx <= s:
                row_ends[r] = e
                rows[i] = r
                placed = True
                break
        if not placed:
            rows[i] = len(row_ends)
            row_ends.append(e)
    return rows
