"""Tests for `constellation.nmr.processing.fourier` — fft / ifft with GRPDLY."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from constellation.core.io.schemas import unpack_metadata
from constellation.nmr.io.bruker import BrukerReader
from constellation.nmr.io.schemas import fid_to_complex
from constellation.nmr.processing import fourier

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


def test_fft_changes_axis_domain_to_freq(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    out = fourier.fft(result.primary)
    meta = unpack_metadata(out.schema.metadata)
    assert meta["x.nmr.axis_domain"] == ["freq"]


def test_fft_preserves_shape_and_other_metadata(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    out = fourier.fft(result.primary)
    meta_in = unpack_metadata(result.primary.schema.metadata)
    meta_out = unpack_metadata(out.schema.metadata)
    assert meta_out["x.nmr.shape"] == meta_in["x.nmr.shape"]
    assert meta_out["x.nmr.sw_hz"] == meta_in["x.nmr.sw_hz"]
    assert meta_out["x.nmr.sfo1_mhz"] == meta_in["x.nmr.sfo1_mhz"]


def test_fft_ifft_round_trip_recovers_fid(tmp_path: Path) -> None:
    n = 128
    real = np.sin(np.linspace(0.0, 4 * np.pi, n, dtype=np.float64))
    imag = np.cos(np.linspace(0.0, 4 * np.pi, n, dtype=np.float64))
    result = _read_synthetic(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=2, GRPDLY=0.0
    )
    fid_in = fid_to_complex(result.primary)

    spectrum = fourier.fft(result.primary, apply_grpdly=False)
    fid_back = fid_to_complex(fourier.ifft(spectrum, apply_grpdly=False))

    assert torch.allclose(fid_back, fid_in, atol=1e-12)


def test_fft_no_grpdly_matches_plain_torch_fft(tmp_path: Path) -> None:
    """With GRPDLY=0, fft() should match torch.fft.fftshift(torch.fft.fft(...))."""
    n = 64
    real = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    imag = np.linspace(0.5, -0.5, n, dtype=np.float64)
    result = _read_synthetic(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=2, GRPDLY=0.0
    )
    fid_in = fid_to_complex(result.primary)

    spectrum = fid_to_complex(fourier.fft(result.primary, apply_grpdly=False))
    expected = torch.fft.fftshift(torch.fft.fft(fid_in, dim=-1), dim=-1)
    assert torch.allclose(spectrum, expected, atol=1e-12)


def test_fft_grpdly_round_trip_recovers_fid(tmp_path: Path) -> None:
    """The forward + inverse GRPDLY corrections cancel out."""
    n = 256
    real = np.sin(np.linspace(0.0, 4 * np.pi, n, dtype=np.float64))
    imag = np.cos(np.linspace(0.0, 4 * np.pi, n, dtype=np.float64))
    result = _read_synthetic(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=2, GRPDLY=67.98
    )
    fid_in = fid_to_complex(result.primary)

    spectrum = fourier.fft(result.primary, apply_grpdly=True)
    fid_back = fid_to_complex(fourier.ifft(spectrum, apply_grpdly=True))
    assert torch.allclose(fid_back, fid_in, atol=1e-10)


def test_fft_grpdly_affects_spectrum(tmp_path: Path) -> None:
    """GRPDLY=on vs off should produce a different spectrum (phase ramp applied).

    A constant FID would Fourier-transform to a delta function at DC,
    where the GRPDLY phase ramp contributes 0 — a degenerate fixture.
    Using the default integer-ramp arrays gives a spectrum with
    non-zero amplitude across most frequency bins so the phase ramp
    actually has somewhere to land.
    """
    n = 64
    result = _read_synthetic(tmp_path, n_complex=n, DTYPA=2, GRPDLY=10.0)
    spec_with = fid_to_complex(fourier.fft(result.primary, apply_grpdly=True))
    spec_without = fid_to_complex(fourier.fft(result.primary, apply_grpdly=False))
    # Distinct results.
    assert not torch.allclose(spec_with, spec_without, atol=1e-6)
    # Magnitudes are identical (GRPDLY is a phase-only correction).
    assert torch.allclose(spec_with.abs(), spec_without.abs(), atol=1e-10)


def test_fft_logs_step(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    out = fourier.fft(result.primary, apply_grpdly=True)
    history = unpack_metadata(out.schema.metadata).get("x.nmr.processing_history", [])
    assert len(history) == 1
    assert history[0]["step"] == "fourier.fft"
    assert history[0]["params"] == {"apply_grpdly": True}


def test_ifft_logs_step(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    spectrum = fourier.fft(result.primary, apply_grpdly=False)
    out = fourier.ifft(spectrum, apply_grpdly=False)
    history = unpack_metadata(out.schema.metadata).get("x.nmr.processing_history", [])
    # fft then ifft → two history entries.
    assert len(history) == 2
    assert history[0]["step"] == "fourier.fft"
    assert history[1]["step"] == "fourier.ifft"
