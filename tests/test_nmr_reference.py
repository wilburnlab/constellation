"""Tests for `constellation.nmr.processing.reference` — set_ppm_scale, hz_axis, ppm_axis.

Uses a *damped* complex-exponential FID (envelope ``exp(-t/T2*)``) so
the post-FFT spectrum has a realistic Lorentzian peak shape rather
than the sinc-ringing of an undamped sinusoid — closer to what a real
NMR spectrum looks like at this pipeline stage. ``T2* = 50 ms`` is a
typical small-molecule ¹H value (FWHM ≈ 6.4 Hz on a 10 kHz sweep).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from constellation.core.io.schemas import unpack_metadata
from constellation.nmr.io.bruker import BrukerReader
from constellation.nmr.io.schemas import fid_to_complex
from constellation.nmr.processing import fourier, reference


# ──────────────────────────────────────────────────────────────────────
# Synthetic Bruker dataset generator (duplicated in each NMR test file
# to keep them self-contained; matches the constellation convention of
# fixtures defined where they are used).
# ──────────────────────────────────────────────────────────────────────


_DEFAULT_ACQUS: dict[str, Any] = {
    "TD": 2048,
    "SW_h": 10000.0,
    # SFO1, BF1, O1 satisfy the Bruker relation SFO1 = BF1 + O1/1e6.
    "SFO1": 600.0024,
    "BF1": 600.0,
    "O1": 2400.0,
    "GRPDLY": 67.98,
    "DTYPA": 0,
    "BYTORDA": 0,
    "NS": 16,
    "RG": 256.0,
    "PULPROG": "zg30",
    "NUC1": "1H",
    "TE": 298.0,
    "SOLVENT": "D2O",
    "INSTRUM": "AVANCE600",
}


def _write_acqus(path: Path, params: dict[str, Any]) -> None:
    """Write a JCAMP-DX acqus file from a flat ``{KEY: value}`` dict."""
    lines = [
        "##TITLE= Parameter file, synthetic test fixture",
        "##JCAMP-DX= 5.00 Bruker JCAMP library",
        "##DATA TYPE= Parameter Values",
        "##ORIGIN= constellation tests",
        "##OWNER= constellation",
    ]
    for key, value in params.items():
        if isinstance(value, str):
            lines.append(f"##${key}= <{value}>")
        elif isinstance(value, int):
            lines.append(f"##${key}= {value}")
        else:
            lines.append(f"##${key}= {value:.6f}")
    lines.append("##END=")
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")


def _write_fid_binary(
    path: Path,
    *,
    real: np.ndarray,
    imag: np.ndarray,
    dtypa: int,
    bytorda: int,
) -> None:
    """Write a Bruker FID binary (interleaved real/imag) at the given dtype/order."""
    base = "i4" if dtypa == 0 else "f8"
    order = "<" if bytorda == 0 else ">"
    np_dtype = np.dtype(f"{order}{base}")

    interleaved = np.empty(real.size * 2, dtype=np.float64)
    interleaved[0::2] = real
    interleaved[1::2] = imag
    path.write_bytes(interleaved.astype(np_dtype).tobytes())


def write_synthetic_bruker_1d(
    directory: Path,
    *,
    n_complex: int = 1024,
    real: np.ndarray | None = None,
    imag: np.ndarray | None = None,
    **acqus_overrides: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a minimal synthetic Bruker 1D experiment under ``directory``."""
    params = {**_DEFAULT_ACQUS, **acqus_overrides}
    params["TD"] = 2 * n_complex

    if real is None:
        real = (np.arange(n_complex, dtype=np.float64) - n_complex // 2)
    if imag is None:
        imag = (np.arange(n_complex, dtype=np.float64) - n_complex // 4) * 2

    _write_acqus(directory / "acqus", params)
    _write_fid_binary(
        directory / "fid",
        real=real,
        imag=imag,
        dtypa=int(params["DTYPA"]),
        bytorda=int(params["BYTORDA"]),
    )
    return real, imag


# Match SFO1 in the default fixture so the calibration math lines up.
_DEFAULT_SFO1_MHZ: float = 600.0024


def _spectrum_with_lorentzian_peak(
    tmp_path: Path,
    *,
    peak_hz: float,
    n_complex: int = 2048,
    sw_hz: float = 10000.0,
    sfo1_mhz: float = _DEFAULT_SFO1_MHZ,
    t2_star_s: float = 0.05,
) -> Any:
    """Build a synthetic 1D experiment whose FFT produces a Lorentzian peak at ``peak_hz``.

    A damped complex exponential ``exp(-t/T2*) · exp(2πi · f · t)`` in the
    time domain Fourier-transforms to a Lorentzian peak at frequency f
    (FWHM ``≈ 1 / (π · T2*)``). The default ``T2* = 50 ms`` gives a
    small-molecule-typical peak (FWHM ≈ 6.4 Hz on a 10 kHz spectral
    window). See ``context/explainers/_drafts/figures/test_fixture_fid_spectrum.py``
    for the visualisation.
    """
    n = n_complex
    dwell_s = 1.0 / sw_hz
    t = np.arange(n, dtype=np.float64) * dwell_s
    envelope = np.exp(-t / t2_star_s)
    phase = 2.0 * np.pi * peak_hz * t
    real = envelope * np.cos(phase)
    imag = envelope * np.sin(phase)
    write_synthetic_bruker_1d(
        tmp_path,
        n_complex=n,
        real=real,
        imag=imag,
        DTYPA=2,
        BYTORDA=0,
        GRPDLY=0.0,  # off so the peak Hz isn't shifted by the digital filter
        SW_h=sw_hz,
        SFO1=sfo1_mhz,
    )
    fid_table = BrukerReader().read(tmp_path).primary
    return fourier.fft(fid_table, apply_grpdly=False)


def _spectrum_with_two_lorentzian_peaks(
    tmp_path: Path,
    *,
    peak1_hz: float,
    peak2_hz: float,
    amp1: float = 1.0,
    amp2: float = 0.5,
    n_complex: int = 2048,
    sw_hz: float = 10000.0,
    sfo1_mhz: float = _DEFAULT_SFO1_MHZ,
    t2_star_s: float = 0.05,
) -> Any:
    """Synthetic spectrum with two Lorentzian peaks (sum of two damped sinusoids).

    FFT is linear, so two superposed damped sinusoids produce two
    independent Lorentzians at their respective Hz positions. Used to
    verify that calibrating against one peak correctly positions the
    other on the ppm axis.
    """
    n = n_complex
    dwell_s = 1.0 / sw_hz
    t = np.arange(n, dtype=np.float64) * dwell_s
    envelope = np.exp(-t / t2_star_s)

    sin1 = envelope * np.exp(2j * np.pi * peak1_hz * t)
    sin2 = envelope * np.exp(2j * np.pi * peak2_hz * t)
    composite = amp1 * sin1 + amp2 * sin2

    write_synthetic_bruker_1d(
        tmp_path,
        n_complex=n,
        real=composite.real,
        imag=composite.imag,
        DTYPA=2,
        BYTORDA=0,
        GRPDLY=0.0,
        SW_h=sw_hz,
        SFO1=sfo1_mhz,
    )
    fid_table = BrukerReader().read(tmp_path).primary
    return fourier.fft(fid_table, apply_grpdly=False)


# ──────────────────────────────────────────────────────────────────────
# SOLVENT_PPM database
# ──────────────────────────────────────────────────────────────────────


def test_solvent_ppm_contains_expected_compounds() -> None:
    assert reference.SOLVENT_PPM["TMS"] == 0.00
    assert reference.SOLVENT_PPM["DSS"] == 0.00
    assert reference.SOLVENT_PPM["CDCl3"] == 7.26
    assert reference.SOLVENT_PPM["DMSO"] == 2.50
    assert reference.SOLVENT_PPM["D2O"] == 4.79
    assert reference.SOLVENT_PPM["MeOD"] == 3.31


# ──────────────────────────────────────────────────────────────────────
# hz_axis
# ──────────────────────────────────────────────────────────────────────


def test_hz_axis_endpoints_and_center(tmp_path: Path) -> None:
    """Hz axis: (k − N//2) · sw_hz / N — fftshifted-centred at carrier."""
    n = 1024
    sw_hz = 10000.0
    write_synthetic_bruker_1d(
        tmp_path, n_complex=n, DTYPA=2, BYTORDA=0, SW_h=sw_hz
    )
    spectrum = fourier.fft(BrukerReader().read(tmp_path).primary, apply_grpdly=False)

    axis = reference.hz_axis(spectrum)
    assert axis.shape == (n,)
    # Centre bin (k = N//2) → exactly 0 Hz (the carrier sits here after fftshift).
    assert float(axis[n // 2]) == 0.0
    # Leftmost bin: k = 0 → (0 − N//2) · sw_hz / N = -sw_hz / 2.
    assert float(axis[0]) == pytest.approx(-sw_hz / 2)
    # Uniform spacing of sw_hz / N between bins.
    spacing = float(axis[1] - axis[0])
    assert spacing == pytest.approx(sw_hz / n)


# ──────────────────────────────────────────────────────────────────────
# set_ppm_scale
# ──────────────────────────────────────────────────────────────────────


def test_set_ppm_scale_finds_peak_and_stamps_metadata(tmp_path: Path) -> None:
    """Single Lorentzian peak at known Hz → set_ppm_scale finds it within one bin."""
    peak_hz = -2400.0  # synthetic TMS position (carrier 2400 Hz above TMS)
    spectrum = _spectrum_with_lorentzian_peak(
        tmp_path, peak_hz=peak_hz, n_complex=2048
    )

    out = reference.set_ppm_scale(
        spectrum,
        compound="TMS",
        expected_hz=-2400.0,
        search_half_width_hz=100.0,
    )

    meta = unpack_metadata(out.schema.metadata)
    # ref_hz should be ≈ peak_hz, within a few Hz bins (sw / N ≈ 4.88 Hz).
    assert "x.nmr.ref_hz" in meta
    assert abs(float(meta["x.nmr.ref_hz"]) - peak_hz) < 10.0
    assert meta["x.nmr.ref_ppm"] == 0.00
    assert meta["x.nmr.reference_compound"] == "TMS"


def test_set_ppm_scale_unknown_compound_raises(tmp_path: Path) -> None:
    spectrum = _spectrum_with_lorentzian_peak(tmp_path, peak_hz=0.0)
    with pytest.raises(ValueError, match="Unknown reference compound"):
        reference.set_ppm_scale(
            spectrum, compound="UnobtainiumD", expected_hz=0.0
        )


def test_set_ppm_scale_ref_ppm_override(tmp_path: Path) -> None:
    """The ref_ppm override lets callers calibrate to non-database compounds."""
    peak_hz = -1500.0
    spectrum = _spectrum_with_lorentzian_peak(tmp_path, peak_hz=peak_hz)
    out = reference.set_ppm_scale(
        spectrum,
        compound="UnobtainiumD",
        expected_hz=peak_hz,
        search_half_width_hz=50.0,
        ref_ppm=4.65,  # e.g. D₂O at 37 °C
    )
    meta = unpack_metadata(out.schema.metadata)
    assert meta["x.nmr.ref_ppm"] == 4.65
    assert meta["x.nmr.reference_compound"] == "UnobtainiumD"


def test_set_ppm_scale_logs_step(tmp_path: Path) -> None:
    spectrum = _spectrum_with_lorentzian_peak(tmp_path, peak_hz=-2400.0)
    out = reference.set_ppm_scale(
        spectrum, compound="TMS", expected_hz=-2400.0, search_half_width_hz=50.0
    )
    history = unpack_metadata(out.schema.metadata).get("x.nmr.processing_history", [])
    assert len(history) >= 1
    last = history[-1]
    assert last["step"] == "reference.set_ppm_scale"
    assert last["params"]["compound"] == "TMS"
    assert last["params"]["expected_hz"] == -2400.0


def test_set_ppm_scale_only_finds_intended_peak(tmp_path: Path) -> None:
    """Two well-separated peaks — the search window picks the intended one.

    If the search window correctly bounds the argmax, calibrating to
    peak1 should not be confused by the taller peak2 sitting at a
    distant Hz — `ref_hz` should land near peak1, not peak2.
    """
    peak1_hz = -2400.0   # intended reference, lower amplitude
    peak2_hz = +1800.0   # interferer, taller, far away
    spectrum = _spectrum_with_two_lorentzian_peaks(
        tmp_path,
        peak1_hz=peak1_hz,
        peak2_hz=peak2_hz,
        amp1=0.4,   # smaller — peak2 would dominate without the window
        amp2=1.0,
    )

    out = reference.set_ppm_scale(
        spectrum,
        compound="TMS",
        expected_hz=peak1_hz,
        search_half_width_hz=100.0,
    )
    ref_hz = float(unpack_metadata(out.schema.metadata)["x.nmr.ref_hz"])
    # Must lock onto peak1, not peak2 — search window enforces locality.
    assert abs(ref_hz - peak1_hz) < 10.0
    assert abs(ref_hz - peak2_hz) > 1000.0


# ──────────────────────────────────────────────────────────────────────
# ppm_axis
# ──────────────────────────────────────────────────────────────────────


def test_ppm_axis_without_calibration_raises(tmp_path: Path) -> None:
    spectrum = _spectrum_with_lorentzian_peak(tmp_path, peak_hz=0.0)
    with pytest.raises(ValueError, match="ppm calibration metadata"):
        reference.ppm_axis(spectrum)


def test_ppm_axis_reference_compound_lands_at_zero(tmp_path: Path) -> None:
    """After referencing TMS to 0 ppm, the ppm axis at the reference index ≈ 0."""
    peak_hz = -2400.0
    spectrum = _spectrum_with_lorentzian_peak(
        tmp_path, peak_hz=peak_hz, n_complex=2048,
        sw_hz=10000.0, sfo1_mhz=_DEFAULT_SFO1_MHZ,
    )
    referenced = reference.set_ppm_scale(
        spectrum,
        compound="TMS",
        expected_hz=peak_hz,
        search_half_width_hz=100.0,
    )

    axis_hz = reference.hz_axis(referenced)
    axis_ppm = reference.ppm_axis(referenced)

    # Find the index closest to the recorded ref_hz; ppm there should be ~0.
    meta = unpack_metadata(referenced.schema.metadata)
    ref_hz = float(meta["x.nmr.ref_hz"])
    ref_idx = int(torch.argmin(torch.abs(axis_hz - ref_hz)))
    assert abs(float(axis_ppm[ref_idx]) - 0.0) < 0.001


def test_ppm_axis_slope_matches_one_over_sfo1(tmp_path: Path) -> None:
    """Per-bin ppm spacing should equal Hz spacing divided by sfo1_mhz."""
    n = 1024
    sw_hz = 10000.0
    sfo1_mhz = _DEFAULT_SFO1_MHZ
    spectrum = _spectrum_with_lorentzian_peak(
        tmp_path, peak_hz=-2400.0, n_complex=n, sw_hz=sw_hz, sfo1_mhz=sfo1_mhz,
    )
    referenced = reference.set_ppm_scale(
        spectrum, compound="TMS", expected_hz=-2400.0, search_half_width_hz=100.0
    )

    diffs_hz = torch.diff(reference.hz_axis(referenced))
    diffs_ppm = torch.diff(reference.ppm_axis(referenced))
    # ppm spacing = Hz spacing / sfo1_mhz (linear calibration with scale=sfo1).
    expected = diffs_hz / sfo1_mhz
    assert torch.allclose(diffs_ppm, expected, atol=1e-12)


def test_multi_peak_ppm_offset_matches_expected(tmp_path: Path) -> None:
    """After calibrating via peak1, peak2 should land at the expected ppm offset.

    The *grounded* version of the slope test: instead of verifying
    that Δppm = ΔHz / sfo1_mhz holds locally, this verifies that the
    absolute ppm of a second peak (located by argmax on the
    referenced spectrum) matches what its Hz position implies.

    Calibrate peak1 as TMS (= 0 ppm). Peak2 sits at ``peak2_hz`` in
    the same spectrum; its expected ppm under the calibration is
    ``(peak2_hz − ref_hz) / sfo1_mhz``. Find peak2 by argmax in a
    window around its expected Hz, look up the ppm value at that
    index, and compare.
    """
    n = 2048
    sw_hz = 10000.0
    sfo1_mhz = _DEFAULT_SFO1_MHZ
    peak1_hz = -2400.0   # TMS reference
    peak2_hz = -1200.0   # second peak; offset = 1200 Hz from TMS → ≈ 2.0 ppm at 600 MHz
    spectrum = _spectrum_with_two_lorentzian_peaks(
        tmp_path,
        peak1_hz=peak1_hz,
        peak2_hz=peak2_hz,
        amp1=1.0,
        amp2=0.5,
        n_complex=n,
        sw_hz=sw_hz,
        sfo1_mhz=sfo1_mhz,
    )

    # Calibrate using peak1 as TMS.
    referenced = reference.set_ppm_scale(
        spectrum,
        compound="TMS",
        expected_hz=peak1_hz,
        search_half_width_hz=50.0,
    )

    # Find peak2 by argmax in a window around its known Hz position.
    axis_hz = reference.hz_axis(referenced)
    axis_ppm = reference.ppm_axis(referenced)
    spectrum_real = fid_to_complex(referenced).real
    mask = (axis_hz >= peak2_hz - 50.0) & (axis_hz <= peak2_hz + 50.0)
    local_idx = int(torch.argmax(spectrum_real[mask]))
    peak2_idx = int(torch.where(mask)[0][local_idx])

    actual_peak2_ppm = float(axis_ppm[peak2_idx])

    # Expected ppm using the actual calibrated ref_hz (not the nominal peak1_hz,
    # which can disagree by up to one Hz bin from the argmax-located ref).
    meta = unpack_metadata(referenced.schema.metadata)
    ref_hz = float(meta["x.nmr.ref_hz"])
    expected_peak2_ppm = (float(axis_hz[peak2_idx]) - ref_hz) / sfo1_mhz

    # Tight tolerance: one ppm bin = sw_hz / N / sfo1_mhz ≈ 0.008 ppm at n=2048.
    assert abs(actual_peak2_ppm - expected_peak2_ppm) < 0.01

    # Sanity check on the absolute value: peak1 at -2400, peak2 at -1200,
    # sfo1 ≈ 600 MHz → expected ppm offset ≈ 2.0. Allow ~3 bins of slack
    # for the argmax-located ref_hz.
    nominal_expected = (peak2_hz - peak1_hz) / sfo1_mhz
    bin_ppm = sw_hz / n / sfo1_mhz
    assert abs(actual_peak2_ppm - nominal_expected) < 3 * bin_ppm
