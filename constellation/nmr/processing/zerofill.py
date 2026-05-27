"""Zero-filling — pad the FID before FFT to interpolate the frequency axis.

Zero-filling extends the time-domain FID by appending zeros, increasing
the length of the resulting frequency axis after FFT. Important to
recognize: zero-filling does NOT improve actual spectral resolution
(that's limited by acquisition time, set by the dwell and number of
collected points). What it DOES do:

    - Interpolates the discrete frequency grid (more samples across each
      peak).
    - Smooths peak shapes (no picket-fence sampling artifacts).
    - Improves centroid / fit accuracy because there are more points
      across each peak's width.

Typical NMR practice is to zero-fill to the next power of two (often
2× the acquired length, sometimes 4× for high-resolution work). The
``double()`` shortcut implements the common pattern.

The function operates on the direct dimension only in the 1D-port
phase. For 2D and higher, separate zero-fill calls apply to the
indirect dimensions.
"""

from __future__ import annotations

import pyarrow as pa
import torch

from constellation.core.io.schemas import unpack_metadata
from constellation.nmr.io.schemas import fid_to_complex, log_step, replace_fid_data


def to_size(table: pa.Table, target_size: int) -> pa.Table:
    """Zero-fill the direct dimension to exactly ``target_size`` complex points.

    Pads the FID with complex zeros to reach ``target_size``. If the
    current size is already ``target_size``, returns the original
    table unchanged. Raises if ``target_size`` is smaller than the
    current size (zero-fill is one-way; use a separate truncation
    helper if that's what you want).

    Parameters
    ----------
    table : pa.Table
        ``NMR_FID_TABLE`` carrying the FID.
    target_size : int
        Desired direct-dimension complex-sample count.

    Returns
    -------
    pa.Table
        ``NMR_FID_TABLE`` with the FID extended to ``target_size``;
        ``x.nmr.shape`` updated; other metadata preserved.

    Raises
    ------
    ValueError
        If ``target_size`` is smaller than the current direct-dimension size.
    """
    fid = fid_to_complex(table)
    current = int(fid.shape[-1])
    if target_size < current:
        raise ValueError(
            f"target_size ({target_size}) must be ≥ current direct-dimension "
            f"size ({current}); zero-fill is one-way"
        )
    if target_size == current:
        return table

    # Pad along the last (direct) dimension. torch.nn.functional.pad takes
    # (pad_left, pad_right, ...) starting from the *last* dim, so a 1D pad
    # of (0, n_extra) is correct for any nD tensor.
    n_extra = target_size - current
    padded = torch.nn.functional.pad(fid, (0, n_extra))
    out = replace_fid_data(table, padded)
    return log_step(out, "zerofill.to_size", target_size=target_size)


def double(table: pa.Table) -> pa.Table:
    """Double the direct-dimension length by appending zeros — the classical NMR shortcut.

    Equivalent to ``to_size(table, 2 * current_direct_size)``.

    Parameters
    ----------
    table : pa.Table
        ``NMR_FID_TABLE`` carrying the FID.

    Returns
    -------
    pa.Table
        FID with direct dimension doubled by zero-padding.
    """
    meta = unpack_metadata(table.schema.metadata)
    shape = meta["x.nmr.shape"]
    if isinstance(shape, str):
        import json

        shape = json.loads(shape)
    current = int(shape[-1])
    return to_size(table, 2 * current)


def to_next_power_of_two(table: pa.Table) -> pa.Table:
    """Zero-fill to the next power of two ≥ the current direct-dimension size.

    Common default in NMR processing — FFT is fastest at power-of-two
    lengths, and the next-power-of-two pad is usually within 2× of the
    raw size (so doesn't blow up storage). If the current size is
    already a power of two, returns the original table unchanged.

    Parameters
    ----------
    table : pa.Table
        ``NMR_FID_TABLE`` carrying the FID.

    Returns
    -------
    pa.Table
        FID zero-filled up to the next power of two.
    """
    meta = unpack_metadata(table.schema.metadata)
    shape = meta["x.nmr.shape"]
    if isinstance(shape, str):
        import json

        shape = json.loads(shape)
    current = int(shape[-1])
    if current <= 0:
        raise ValueError("Cannot zero-fill a zero-length FID.")
    # Smallest power of two >= current.
    target = 1
    while target < current:
        target *= 2
    return to_size(table, target)


__all__ = ["to_size", "double", "to_next_power_of_two"]
