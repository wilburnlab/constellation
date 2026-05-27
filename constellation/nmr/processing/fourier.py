"""Fourier transform with Bruker GRPDLY correction.

The classical FFT is one line (``torch.fft.fft``); what earns
``fourier.py`` its NMR-specific seat is **GRPDLY correction**. Modern
Bruker AVANCE instruments apply an oversampling digital filter on
acquisition; the filter delays the FID by a fractional ``GRPDLY``
number of points (typically 67-68 points). The standard fixes:

    - **Time-domain truncation.** Discard the first ``floor(GRPDLY) + 1``
      points and zero-fill back. Loses a small amount of early-FID
      signal; simple but changes FID length.
    - **Frequency-domain correction.** After FFT, multiply each
      spectral point by ``exp(2πi · GRPDLY · k / N)`` — a linear-phase
      ramp that removes the constant time-domain offset without
      truncating. Recommended: TopSpin and nmrglue do it this way for
      standard 1D processing.

This module implements the frequency-domain approach as the default in
:func:`fft`. The GRPDLY value is read from the table's
``x.nmr.grpdly`` metadata key (stamped by the Bruker reader); pass
``apply_grpdly=False`` to skip it (e.g., when working with non-Bruker
data or with explicitly pre-processed datasets where the filter is
already accounted for).

After FFT, the table's ``x.nmr.axis_domain`` flips from
``["time"]`` to ``["freq"]`` for the affected axis. Other metadata
(sweep width, carrier frequency, GRPDLY itself) is preserved — those
are descriptions of the data's *origin* that remain meaningful in the
frequency domain.

For 1D the FFT runs on the only dimension. The 2D extension applies
FFT along the direct dimension first (with the same GRPDLY correction)
and the indirect dimension afterward (typically without GRPDLY since
indirect-dim digital filters work differently — they belong to the t1
acquisition sequence, not the receiver).
"""

from __future__ import annotations

import pyarrow as pa
import torch

from constellation.core.io.schemas import (
    pack_metadata,
    unpack_metadata,
    with_metadata,
)
from constellation.nmr.io.schemas import fid_to_complex, log_step, replace_fid_data


def fft(table: pa.Table, apply_grpdly: bool = True) -> pa.Table:
    """Forward FFT of an NMR FID, with Bruker GRPDLY correction.

    Pipeline:

        1. Extract the complex FID from the table.
        2. ``torch.fft.fft`` along the direct (innermost) dimension.
        3. ``torch.fft.fftshift`` so the carrier sits at the centre of
           the spectrum (standard NMR display convention).
        4. If ``apply_grpdly`` and ``x.nmr.grpdly`` is present and
           non-zero: multiply by ``exp(2πi · GRPDLY · k / N)`` to
           remove the Bruker digital-filter group delay.
        5. Stamp ``x.nmr.axis_domain`` to mark the direct axis as
           ``"freq"``.

    Parameters
    ----------
    table : pa.Table
        ``NMR_FID_TABLE`` with time-domain data and the standard
        ``x.nmr.*`` metadata.
    apply_grpdly : bool
        Apply Bruker GRPDLY correction in the frequency domain.
        Default ``True``. Set ``False`` for non-Bruker data, or for
        Bruker data where GRPDLY has already been handled (e.g.
        time-domain truncation upstream).

    Returns
    -------
    pa.Table
        ``NMR_FID_TABLE`` carrying the complex spectrum (same schema
        re-used; ``x.nmr.axis_domain[-1]`` is now ``"freq"``).
    """
    fid = fid_to_complex(table)
    meta = unpack_metadata(table.schema.metadata)

    # FFT and fftshift on the direct (last) dimension.
    spectrum = torch.fft.fft(fid, dim=-1)
    spectrum = torch.fft.fftshift(spectrum, dim=-1)

    # Bruker GRPDLY correction: multiply by exp(2πi · GRPDLY · k / N)
    # where k runs over the fftshifted index. After fftshift, the
    # carrier (k=0 in the unshifted convention) is at index N//2;
    # constructing the phase ramp pre-shift and shifting along with
    # the data is simpler than reasoning about the post-shift k axis.
    grpdly = float(meta.get("x.nmr.grpdly", 0.0)) if apply_grpdly else 0.0
    if grpdly != 0.0:
        n = spectrum.shape[-1]
        k = torch.arange(n, dtype=torch.float64)
        # Build the ramp at the pre-shift convention (k = 0..N-1), then
        # apply the same fftshift so it lines up with the shifted spectrum.
        phase = 2.0 * torch.pi * grpdly * k / n
        ramp = torch.exp(1j * phase).to(spectrum.dtype)
        ramp = torch.fft.fftshift(ramp, dim=-1)
        spectrum = spectrum * ramp

    out = replace_fid_data(table, spectrum)

    # Mark the direct axis as frequency-domain. axis_domain is a list
    # in outer→inner order, so the direct (last-dim) state is index -1.
    axis_domain = list(meta.get("x.nmr.axis_domain", ["time"]))
    axis_domain[-1] = "freq"
    out = with_metadata(out, {"x.nmr.axis_domain": axis_domain})
    return log_step(out, "fourier.fft", apply_grpdly=apply_grpdly)


def ifft(table: pa.Table, apply_grpdly: bool = True) -> pa.Table:
    """Inverse FFT of an NMR spectrum back to time domain.

    The inverse pipeline: undo GRPDLY (if it was applied on the forward
    pass), inverse-fftshift, ``torch.fft.ifft`` along the direct
    dimension, flip ``axis_domain[-1]`` back to ``"time"``. The round
    trip ``ifft(fft(fid))`` recovers the input to numerical precision.

    Parameters
    ----------
    table : pa.Table
        ``NMR_FID_TABLE`` carrying the frequency-domain spectrum.
    apply_grpdly : bool
        Undo Bruker GRPDLY correction. Match the value used on the
        forward FFT.

    Returns
    -------
    pa.Table
        ``NMR_FID_TABLE`` carrying the time-domain FID.
    """
    spectrum = fid_to_complex(table)
    meta = unpack_metadata(table.schema.metadata)

    grpdly = float(meta.get("x.nmr.grpdly", 0.0)) if apply_grpdly else 0.0
    if grpdly != 0.0:
        n = spectrum.shape[-1]
        k = torch.arange(n, dtype=torch.float64)
        phase = -2.0 * torch.pi * grpdly * k / n  # negate to invert
        ramp = torch.exp(1j * phase).to(spectrum.dtype)
        ramp = torch.fft.fftshift(ramp, dim=-1)
        spectrum = spectrum * ramp

    fid = torch.fft.ifftshift(spectrum, dim=-1)
    fid = torch.fft.ifft(fid, dim=-1)

    out = replace_fid_data(table, fid)
    axis_domain = list(meta.get("x.nmr.axis_domain", ["freq"]))
    axis_domain[-1] = "time"
    out = with_metadata(out, {"x.nmr.axis_domain": axis_domain})
    return log_step(out, "fourier.ifft", apply_grpdly=apply_grpdly)


__all__ = ["fft", "ifft"]
