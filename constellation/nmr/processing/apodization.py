"""NMR-conventional apodization wrappers.

Each function takes an ``NMR_FID_TABLE`` in and returns an
``NMR_FID_TABLE`` out, applying a multiplicative window to the
time-domain FID. The actual window-multiply operation is
:func:`constellation.core.signal.apodize`; the window shapes come from
:func:`constellation.core.signal.windows` (sine bell + DSP standards)
and from :mod:`constellation.core.stats` (Gaussian distribution
evaluated as a window). This module concentrates NMR domain
conventions: Hz line-broadening parameters, dwell-time-derived time
axis, the conventional names ``em`` / ``gaussian`` / ``sine_bell``.

Mathematical content:

    em(lb)              w(t) = exp(-π·lb·t)
                        FT-dual: Lorentzian of HWHM lb/π in frequency
    gaussian(lb, gb)    w(t) = exp(-π·lb·t) · exp(-(π·gb·t)² / (4·ln 2))
                        FT-dual: convolution of a Lorentzian (from lb)
                        and a Gaussian of FWHM gb in frequency
    sine_bell(off,end,power)
                        w[i] = sin(off·π + i/(N-1)·(end-off)·π) ^ power
                        purely geometric window — no distribution kin

Apodization works on the direct dimension only in the 1D-port phase.
For 2D and higher, separate apodization passes apply to the indirect
dimensions; that lands when the 2D reader does.
"""

from __future__ import annotations

import math

import pyarrow as pa
import torch

from constellation.core.io.schemas import unpack_metadata
from constellation.core.signal import apodize as _apodize
from constellation.core.signal.windows import sine_bell as _sine_bell_window
from constellation.nmr.io.schemas import (
    NMR_FID_TABLE,
    fid_to_complex,
    log_step,
    replace_fid_data,
)


def _time_axis_seconds(table: pa.Table) -> torch.Tensor:
    """Reconstruct the direct-dimension time axis from table metadata.

    ``t[i] = i / sw_hz`` for ``i = 0..N-1`` where N is the direct
    dimension size (``x.nmr.shape[-1]``) and ``sw_hz`` is the direct
    sweep width.
    """
    meta = unpack_metadata(table.schema.metadata)
    shape = meta["x.nmr.shape"]
    if isinstance(shape, str):
        import json

        shape = json.loads(shape)
    n_direct = int(shape[-1])
    sw_hz = float(meta["x.nmr.sw_hz"])
    dwell_s = 1.0 / sw_hz
    return torch.arange(n_direct, dtype=torch.float64) * dwell_s


def em(table: pa.Table, lb: float) -> pa.Table:
    """Exponential-multiplication apodization.

    Multiplies the time-domain FID by ``exp(-π·lb·t)``, broadening
    spectral lineshapes via Fourier duality (the FT of an exponential
    decay is a Lorentzian; the FT-dual of multiplying by an exponential
    in time is convolving with a Lorentzian in frequency). ``lb`` is
    the resulting Lorentzian HWHM in Hz × π — i.e., setting ``lb = 1.0``
    Hz convolves the spectrum with a Lorentzian of HWHM ≈ 0.32 Hz.

    Positive ``lb`` broadens (sacrifices resolution for SNR by
    down-weighting the noisy tail of the FID); negative ``lb`` sharpens
    (resolution enhancement, amplifies noise).

    Parameters
    ----------
    table : pa.Table
        ``NMR_FID_TABLE`` carrying the time-domain FID and metadata.
    lb : float
        Line-broadening parameter in Hz. Default for protein 1D ¹H is
        typically 0.3 - 1.0 Hz.

    Returns
    -------
    pa.Table
        Apodized FID, same schema and metadata (shape unchanged).
    """
    t = _time_axis_seconds(table)
    window = torch.exp(-torch.pi * lb * t)
    fid = fid_to_complex(table)
    out = replace_fid_data(table, _apodize(fid, window))
    return log_step(out, "apodization.em", lb=lb)


def gaussian(table: pa.Table, lb: float, gb: float) -> pa.Table:
    """Combined exponential + Gaussian apodization (resolution enhancement).

    Window: ``exp(-π·lb·t) · exp(-(π·gb·t)² / (4·ln 2))``.

    The exponential term is the same as :func:`em` and is typically
    used with negative ``lb`` here (resolution enhancement). The
    Gaussian term provides a smooth taper that prevents the noise
    blow-up that would otherwise come from sharpening with ``em``
    alone; ``gb`` is the FWHM of the resulting Gaussian convolution
    kernel in frequency, expressed as a Hz line-broadening.

    The conversion between Hz line-broadening and time-domain
    Gaussian σ is the standard FT relationship for a Gaussian:
    ``σ_t = √(2·ln 2) / (π·gb)``. The Gaussian factor in time is then
    ``exp(-t² / (2·σ_t²))`` — exactly the unnormalized
    ``NormalDistribution`` shape from :mod:`constellation.core.stats`,
    here evaluated directly because the windowing use case wants the
    unnormalized PDF rather than the probability density.

    Parameters
    ----------
    table : pa.Table
        ``NMR_FID_TABLE`` carrying the time-domain FID.
    lb : float
        Exponential line-broadening in Hz. Often negative here
        (resolution enhancement); positive falls back toward `em()`.
    gb : float
        Gaussian line-broadening (FWHM) in Hz. Must be positive.

    Returns
    -------
    pa.Table
        Apodized FID, same schema and metadata.

    Raises
    ------
    ValueError
        If ``gb <= 0``.
    """
    if gb <= 0.0:
        raise ValueError(f"`gb` must be positive (got {gb}).")
    t = _time_axis_seconds(table)
    exp_term = torch.exp(-torch.pi * lb * t)
    # Gaussian: exp(-(π·gb·t)² / (4·ln 2)) = exp(-t² / (2·σ_t²))
    # where σ_t = √(2·ln 2) / (π·gb).
    sigma_t = math.sqrt(2.0 * math.log(2.0)) / (math.pi * gb)
    gauss_term = torch.exp(-0.5 * (t / sigma_t) ** 2)
    window = exp_term * gauss_term
    fid = fid_to_complex(table)
    out = replace_fid_data(table, _apodize(fid, window))
    return log_step(out, "apodization.gaussian", lb=lb, gb=gb)


def sine_bell(
    table: pa.Table,
    off: float = 0.0,
    end: float = 1.0,
    power: int = 1,
) -> pa.Table:
    """Sine-bell apodization — classical NMR DSP window.

    Forwards to :func:`constellation.core.signal.windows.sine_bell` and
    multiplies into the FID. Common parametrizations:

        ``off=0, end=1, power=1``    pure sine bell — zero at both ends,
                                     peak at midpoint.
        ``off=0.5, end=1, power=1``  shifted sine bell — starts at peak,
                                     decays. Similar to exponential
                                     apodization without an explicit lb.
        ``power=2``                  squared sine bell — sharper taper;
                                     common in 2D processing.

    Parameters
    ----------
    table : pa.Table
        ``NMR_FID_TABLE`` carrying the time-domain FID.
    off : float
        Starting phase as a fraction of π. Default 0.0.
    end : float
        Ending phase as a fraction of π. Default 1.0.
    power : int
        Sine exponent. Default 1 (pure sine bell); 2 = squared sine bell.

    Returns
    -------
    pa.Table
        Apodized FID, same schema and metadata.
    """
    meta = unpack_metadata(table.schema.metadata)
    shape = meta["x.nmr.shape"]
    if isinstance(shape, str):
        import json

        shape = json.loads(shape)
    n_direct = int(shape[-1])
    window = _sine_bell_window(n_direct, off=off, end=end, power=power)
    fid = fid_to_complex(table)
    out = replace_fid_data(table, _apodize(fid, window))
    return log_step(out, "apodization.sine_bell", off=off, end=end, power=power)


__all__ = ["em", "gaussian", "sine_bell"]
