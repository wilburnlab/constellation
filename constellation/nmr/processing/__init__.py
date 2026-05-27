"""NMR-specific processing wrappers.

Thin layer over :mod:`constellation.core.signal` and
:mod:`constellation.core.stats` that converts NMR-conventional
parameters (Hz line-broadening, dwell-time spacing, degrees-of-phase)
into the natural units of the core primitives and stamps NMR metadata
on the resulting Arrow tables.

Each wrapper takes an ``NMR_FID_TABLE``-shaped table in and returns
an ``NMR_FID_TABLE`` out. Per-axis FT state lives in the
``x.nmr.axis_domain`` metadata key: the FFT transitions ``"time"`` →
``"freq"`` on the relevant axis, other operations preserve it.

Modules:
    apodization     em, gaussian, sine_bell — window-based FID weighting
    zerofill        zero-fill before FFT — increase frequency resolution
    fourier         GRPDLY-aware FFT (Bruker digital-filter correction)
    phase           ph0 + ph1 phase correction in degrees (NMR convention)
    reference       SOLVENT_PPM database + set_ppm_scale + hz_axis / ppm_axis
"""

from __future__ import annotations

from constellation.nmr.processing import (  # noqa: F401
    apodization,
    fourier,
    phase,
    reference,
    zerofill,
)

__all__ = ["apodization", "fourier", "phase", "reference", "zerofill"]
