"""ppm-scale referencing and axis-construction helpers.

After Fourier transform, an NMR spectrum lives on a Hz axis: each
spectral bin maps to a frequency offset from the carrier (the
transmitter centre, at index ``N // 2`` after fftshift). To compare
spectra across instruments at different field strengths, the
convention is to normalise by the carrier Larmor frequency and express
positions in **parts per million**:
``ppm = (Hz_signal − Hz_ref) / sfo1_mhz + ppm_ref``. The reference
compound — TMS, DSS, a residual solvent peak — has a known ppm value
(0.00 for TMS / DSS by IUPAC convention; 7.26 for CDCl₃; etc.).

This module composes the generic primitives in
:mod:`constellation.core.signal.calibration` with NMR domain knowledge:

    find_peak_in_window       (core) — argmax of the absorption-mode
                                       spectrum within a Hz search window.
    linear_axis_calibration   (core) — affine transform applied to an
                                       axis given one known calibration
                                       point.
    SOLVENT_PPM               (this module) — NMR standards database.

The public entry point is :func:`set_ppm_scale`, which finds the
reference peak in Hz, looks up its known ppm position, and stamps the
calibration metadata. The ppm axis is then constructed on demand from
the stamped values by :func:`ppm_axis`. Data is stored once; axes are
reconstructed from metadata — the standard "schema-stamped Arrow"
idiom in constellation.
"""

from __future__ import annotations

import json

import pyarrow as pa
import torch

from constellation.core.io.schemas import unpack_metadata, with_metadata
from constellation.core.signal.calibration import (
    find_peak_in_window,
    linear_axis_calibration,
)
from constellation.nmr.io.schemas import fid_to_complex, log_step


# ──────────────────────────────────────────────────────────────────────
# NMR reference-compound standards database.
#
# Values for the residual solvent peaks (CDCl₃, DMSO-d₆, D₂O, CD₃OD)
# are from Gottlieb, Kotlyar & Nudelman (1997) Table 1, the canonical
# reference cited >10,000 times:
#
#     Gottlieb, H. E., Kotlyar, V., & Nudelman, A. (1997).
#     NMR chemical shifts of common laboratory solvents as trace
#     impurities. *J. Org. Chem.* 62(21), 7512-7515.
#     DOI: 10.1021/jo971176v
#
# Fulmer et al. (2010) [DOI: 10.1021/om100106e] extends the Gottlieb
# table to a much larger set of solvents; ship from that source if the
# six entries below ever feel limiting.
#
# TMS = 0.00 and DSS = 0.00 are by IUPAC convention; see
#
#     Harris, R. K., Becker, E. D., Cabral de Menezes, S. M.,
#     Goodfellow, R., & Granger, P. (2001). NMR nomenclature: Nuclear
#     spin properties and conventions for chemical shifts (IUPAC
#     recommendations 2001). *Pure Appl. Chem.* 73(11), 1795-1818.
#     DOI: 10.1351/pac200173111795
#
# DSS specifically as the aqueous primary reference for biomolecular
# NMR is documented in Wishart et al. (1995) *J. Biomol. NMR* 6,
# 135-140 (DOI: 10.1007/BF00211777).
#
# Temperature dependence: the D₂O HDO peak shifts by ~0.01 ppm/°C
# (Gottlieb gives 4.79 at ambient ~22-25 °C). For 37 °C protein NMR
# the actual HDO position is closer to 4.65. Quantitative work at
# non-ambient temperatures should override ``ref_ppm`` in
# :func:`set_ppm_scale` with a temperature-corrected value rather
# than relying on the static dict.
# ──────────────────────────────────────────────────────────────────────


SOLVENT_PPM: dict[str, float] = {
    # Primary references (defined as 0.00 ppm by IUPAC convention)
    "TMS":   0.00,   # tetramethylsilane — primary reference for ¹H in non-aqueous
    "DSS":   0.00,   # 4,4-dimethyl-4-silapentane-1-sulfonate — aqueous biomolecular NMR
    # Residual solvent peaks (Gottlieb 1997 Table 1)
    "CDCl3": 7.26,   # CHCl in CDCl₃
    "DMSO":  2.50,   # DMSO-d₆ residual CHD
    "D2O":   4.79,   # HDO at ambient (~22-25 °C); see temperature-dependence note above
    "MeOD":  3.31,   # CD₃OD residual CHD₂
}


# ──────────────────────────────────────────────────────────────────────
# Axis-construction helpers (read-only; do not modify the table).
# ──────────────────────────────────────────────────────────────────────


def _shape_of(table: pa.Table) -> list[int]:
    """Read ``x.nmr.shape`` from table metadata, JSON-decoding if needed."""
    meta = unpack_metadata(table.schema.metadata)
    shape = meta["x.nmr.shape"]
    if isinstance(shape, str):
        shape = json.loads(shape)
    return [int(n) for n in shape]


def hz_axis(table: pa.Table) -> torch.Tensor:
    """Return the direct-dimension Hz axis (offset from carrier).

    For an fftshifted spectrum, the carrier sits at index ``N // 2``
    where ``Hz = 0``. The axis follows
    ``Hz[k] = (k − N // 2) · sw_hz / N`` for ``k = 0..N-1``, where
    ``N`` is the direct-dimension size and ``sw_hz`` is the direct
    sweep width (read from ``x.nmr.sw_hz``).

    Does not check that the table is in the frequency domain — the
    formula is well-defined regardless, and domain-state validation
    belongs at the calling layer.
    """
    meta = unpack_metadata(table.schema.metadata)
    shape = _shape_of(table)
    n = shape[-1]
    sw_hz = float(meta["x.nmr.sw_hz"])
    k = torch.arange(n, dtype=torch.float64)
    return (k - n // 2) * sw_hz / n


def ppm_axis(table: pa.Table) -> torch.Tensor:
    """Return the direct-dimension ppm axis using stamped calibration metadata.

    Composes :func:`hz_axis` with
    :func:`constellation.core.signal.calibration.linear_axis_calibration`,
    using the reference position recorded by :func:`set_ppm_scale`:
    ``ppm = (Hz_axis − x.nmr.ref_hz) / x.nmr.sfo1_mhz + x.nmr.ref_ppm``.

    Raises
    ------
    ValueError
        If the table has no ppm calibration metadata
        (``x.nmr.ref_hz`` and ``x.nmr.ref_ppm`` not stamped). Call
        :func:`set_ppm_scale` first, or stamp the keys manually.
    """
    meta = unpack_metadata(table.schema.metadata)
    if "x.nmr.ref_hz" not in meta or "x.nmr.ref_ppm" not in meta:
        raise ValueError(
            "Table has no ppm calibration metadata "
            "('x.nmr.ref_hz' and 'x.nmr.ref_ppm' not stamped). "
            "Call `set_ppm_scale` first, or stamp the keys manually."
        )
    sfo1_mhz = float(meta["x.nmr.sfo1_mhz"])
    ref_hz = float(meta["x.nmr.ref_hz"])
    ref_ppm = float(meta["x.nmr.ref_ppm"])
    return linear_axis_calibration(
        hz_axis(table),
        observed_position=ref_hz,
        target_position=ref_ppm,
        scale=sfo1_mhz,
    )


# ──────────────────────────────────────────────────────────────────────
# set_ppm_scale — locate the reference peak and stamp calibration metadata.
# ──────────────────────────────────────────────────────────────────────


def set_ppm_scale(
    table: pa.Table,
    compound: str,
    expected_hz: float,
    *,
    search_half_width_hz: float = 50.0,
    ref_ppm: float | None = None,
) -> pa.Table:
    """Locate a reference compound peak and stamp ppm calibration metadata.

    Procedure:

        1. Look up the compound's known ppm position from
           :data:`SOLVENT_PPM` (or use the explicit ``ref_ppm`` override
           — required for compounds not in the database, useful for
           temperature-corrected water referencing).
        2. Pull the absorption-mode (real-part) spectrum and the
           direct-dimension Hz axis.
        3. Find the actual Hz position of the reference peak via
           :func:`core.signal.calibration.find_peak_in_window` —
           windowed argmax centred on ``expected_hz``.
        4. Stamp ``x.nmr.ref_hz``, ``x.nmr.ref_ppm``, and
           ``x.nmr.reference_compound`` into the table metadata. The
           ppm axis is then reconstructed on demand by
           :func:`ppm_axis`.

    The table itself is not transformed — only the metadata changes.
    Data storage stays in the original form; the ppm convention is a
    metadata-derived view, the standard "schema-stamped Arrow" idiom
    in constellation.

    Parameters
    ----------
    table : pa.Table
        ``NMR_FID_TABLE``-shaped table containing a frequency-domain
        spectrum (typically the output of :func:`fourier.fft`).
    compound : str
        Name of the reference compound (e.g. ``"TMS"``, ``"DSS"``,
        ``"CDCl3"``). Must be a key in :data:`SOLVENT_PPM` unless
        ``ref_ppm`` is supplied.
    expected_hz : float
        Approximate Hz position of the reference peak on the current
        spectrum, used as the centre of the search window. For TMS on
        a 600 MHz spectrometer with the carrier centred around 4 ppm,
        TMS sits near ``-(4 ppm × 600 MHz) ≈ -2400 Hz``.
    search_half_width_hz : float
        Half-width of the windowed argmax search, in Hz. Default
        50.0. Reduce if other peaks are close to the reference;
        increase if the expected position is poorly known.
    ref_ppm : float, optional
        Override the looked-up ppm position. Required for compounds
        not in :data:`SOLVENT_PPM`; useful for temperature-corrected
        water referencing (D₂O at 37 °C ≈ 4.65 ppm vs 4.79 ambient)
        or otherwise non-default references.

    Returns
    -------
    pa.Table
        Same data; updated metadata
        (``x.nmr.ref_hz``, ``x.nmr.ref_ppm``,
        ``x.nmr.reference_compound``); processing history extended
        with one ``"reference.set_ppm_scale"`` entry.

    Raises
    ------
    ValueError
        If ``compound`` is not in :data:`SOLVENT_PPM` and no
        ``ref_ppm`` is supplied; or if the search window is outside
        the available Hz axis.
    """
    if ref_ppm is None:
        if compound not in SOLVENT_PPM:
            raise ValueError(
                f"Unknown reference compound {compound!r}. "
                f"Known compounds: {sorted(SOLVENT_PPM)}. "
                f"Pass `ref_ppm=` to override for non-database references."
            )
        ref_ppm = SOLVENT_PPM[compound]

    spectrum = fid_to_complex(table).real  # absorption mode
    axis = hz_axis(table)
    ref_hz = find_peak_in_window(
        spectrum,
        axis,
        expected_position=expected_hz,
        search_half_width=search_half_width_hz,
    )

    out = with_metadata(table, {
        "x.nmr.ref_hz": ref_hz,
        "x.nmr.ref_ppm": float(ref_ppm),
        "x.nmr.reference_compound": compound,
    })
    return log_step(
        out,
        "reference.set_ppm_scale",
        compound=compound,
        expected_hz=expected_hz,
        search_half_width_hz=search_half_width_hz,
        ref_ppm=float(ref_ppm),
        ref_hz=ref_hz,
    )


__all__ = ["SOLVENT_PPM", "hz_axis", "ppm_axis", "set_ppm_scale"]
