"""Tests for `constellation.nmr.processing.phase` — NMR-conventional wrapper."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from constellation.core.io.schemas import unpack_metadata
from constellation.core.signal.phase import apply_polynomial_phase
from constellation.nmr.io.bruker import BrukerReader
from constellation.nmr.io.schemas import fid_to_complex
from constellation.nmr.processing import fourier, phase

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


def _read_synthetic(tmp_path: Path, **kwargs):
    write_synthetic_bruker_1d(tmp_path, **kwargs)
    return BrukerReader().read(tmp_path)


def test_zero_phase_is_identity(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    spectrum = fourier.fft(result.primary, apply_grpdly=False)
    out = phase.phase_correct(spectrum, ph0=0.0, ph1=0.0, unit="deg")
    assert torch.allclose(
        fid_to_complex(out), fid_to_complex(spectrum), atol=1e-12
    )


def test_ph0_180_deg_negates_spectrum(tmp_path: Path) -> None:
    """A pure 180° zero-order phase = multiply by exp(iπ) = -1."""
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    spectrum = fourier.fft(result.primary, apply_grpdly=False)
    out = phase.phase_correct(spectrum, ph0=180.0, ph1=0.0, unit="deg")
    assert torch.allclose(
        fid_to_complex(out), -fid_to_complex(spectrum), atol=1e-10
    )


def test_matches_apply_polynomial_phase_directly(tmp_path: Path) -> None:
    """Wrapper should be equivalent to calling apply_polynomial_phase with [ph0_rad, ph1_rad]."""
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    spectrum = fourier.fft(result.primary, apply_grpdly=False)
    spec_tensor = fid_to_complex(spectrum)

    ph0_deg, ph1_deg = 30.0, 60.0
    out_wrapper = fid_to_complex(
        phase.phase_correct(spectrum, ph0=ph0_deg, ph1=ph1_deg, unit="deg")
    )
    out_direct = apply_polynomial_phase(
        spec_tensor,
        coefficients=[math.radians(ph0_deg), math.radians(ph1_deg)],
        pivot=None,
    )
    assert torch.allclose(out_wrapper, out_direct, atol=1e-12)


def test_unit_rad_and_deg_equivalence(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    spectrum = fourier.fft(result.primary, apply_grpdly=False)
    ph0_deg, ph1_deg = 45.0, 90.0
    out_deg = fid_to_complex(
        phase.phase_correct(spectrum, ph0=ph0_deg, ph1=ph1_deg, unit="deg")
    )
    out_rad = fid_to_complex(
        phase.phase_correct(
            spectrum,
            ph0=math.radians(ph0_deg),
            ph1=math.radians(ph1_deg),
            unit="rad",
        )
    )
    assert torch.allclose(out_deg, out_rad, atol=1e-12)


def test_unrecognised_unit_raises(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=32, DTYPA=2)
    spectrum = fourier.fft(result.primary, apply_grpdly=False)
    with pytest.raises(ValueError, match="unit"):
        phase.phase_correct(spectrum, ph0=0.0, ph1=0.0, unit="turns")


def test_pivot_only_constant_at_pivot(tmp_path: Path) -> None:
    """At the pivot index, only ph0 contributes."""
    n = 65  # odd so N // 2 is an exact integer index
    result = _read_synthetic(tmp_path, n_complex=n, DTYPA=2)
    spectrum = fourier.fft(result.primary, apply_grpdly=False)
    spec_tensor = fid_to_complex(spectrum)

    pivot = n // 2
    ph0_deg = 90.0
    ph1_deg = 270.0
    out = fid_to_complex(
        phase.phase_correct(
            spectrum, ph0=ph0_deg, ph1=ph1_deg, pivot=pivot, unit="deg"
        )
    )
    expected_at_pivot = spec_tensor[pivot] * torch.exp(
        torch.tensor(1j * math.radians(ph0_deg), dtype=torch.complex128)
    )
    assert abs(complex(out[pivot]) - complex(expected_at_pivot)) < 1e-10


def test_phase_correct_logs_step(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=32, DTYPA=2)
    spectrum = fourier.fft(result.primary, apply_grpdly=False)
    out = phase.phase_correct(spectrum, ph0=45.0, ph1=10.0, unit="deg")
    history = unpack_metadata(out.schema.metadata).get("x.nmr.processing_history", [])
    # fft then phase_correct → two entries.
    assert len(history) == 2
    assert history[-1]["step"] == "phase.phase_correct"
    assert history[-1]["params"]["ph0"] == 45.0
    assert history[-1]["params"]["ph1"] == 10.0
    assert history[-1]["params"]["unit"] == "deg"
