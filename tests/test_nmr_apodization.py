"""Tests for `constellation.nmr.processing.apodization` — em / gaussian / sine_bell."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from constellation.core.io.schemas import unpack_metadata
from constellation.nmr.io.bruker import BrukerReader
from constellation.nmr.io.schemas import fid_to_complex
from constellation.nmr.processing import apodization

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
    """Write + read a synthetic Bruker 1D dataset; return the BrukerReader result."""
    write_synthetic_bruker_1d(tmp_path, **kwargs)
    return BrukerReader().read(tmp_path)


# ──────────────────────────────────────────────────────────────────────
# em — exponential multiplication
# ──────────────────────────────────────────────────────────────────────


def test_em_applies_exponential_decay_window(tmp_path: Path) -> None:
    n = 256
    # Constant FID = 1 + 1j so the windowed result is exactly the window.
    real = np.ones(n, dtype=np.float64)
    imag = np.ones(n, dtype=np.float64)
    result = _read_synthetic(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=2, BYTORDA=0
    )
    sw_hz = 10000.0  # default in the fixture
    lb = 2.0

    out = apodization.em(result.primary, lb=lb)
    fid_out = fid_to_complex(out)

    t = torch.arange(n, dtype=torch.float64) / sw_hz
    expected_window = torch.exp(-torch.pi * lb * t)
    assert torch.allclose(fid_out.real, expected_window, atol=1e-12)
    assert torch.allclose(fid_out.imag, expected_window, atol=1e-12)


def test_em_lb_zero_is_identity(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    fid_in = fid_to_complex(result.primary)
    fid_out = fid_to_complex(apodization.em(result.primary, lb=0.0))
    assert torch.allclose(fid_out, fid_in, atol=1e-12)


def test_em_preserves_metadata_and_shape(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    out = apodization.em(result.primary, lb=1.0)
    meta_in = unpack_metadata(result.primary.schema.metadata)
    meta_out = unpack_metadata(out.schema.metadata)

    assert meta_out["x.nmr.shape"] == meta_in["x.nmr.shape"]
    assert meta_out["x.nmr.sw_hz"] == meta_in["x.nmr.sw_hz"]
    assert meta_out["x.nmr.axis_domain"] == ["time"]
    assert out.num_rows == 2  # still real + imag


def test_em_logs_processing_step(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    out = apodization.em(result.primary, lb=0.5)
    history = unpack_metadata(out.schema.metadata).get("x.nmr.processing_history", [])
    assert len(history) == 1
    assert history[0]["step"] == "apodization.em"
    assert history[0]["params"] == {"lb": 0.5}


# ──────────────────────────────────────────────────────────────────────
# gaussian — combined exponential + Gaussian
# ──────────────────────────────────────────────────────────────────────


def test_gaussian_matches_explicit_formula(tmp_path: Path) -> None:
    n = 128
    real = np.ones(n, dtype=np.float64)
    imag = np.ones(n, dtype=np.float64)
    result = _read_synthetic(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=2
    )
    sw_hz = 10000.0
    lb = -1.0
    gb = 1.5

    out = apodization.gaussian(result.primary, lb=lb, gb=gb)
    fid_out = fid_to_complex(out)

    t = torch.arange(n, dtype=torch.float64) / sw_hz
    sigma_t = math.sqrt(2.0 * math.log(2.0)) / (math.pi * gb)
    expected = (
        torch.exp(-torch.pi * lb * t)
        * torch.exp(-0.5 * (t / sigma_t) ** 2)
    )
    assert torch.allclose(fid_out.real, expected, atol=1e-12)


def test_gaussian_zero_lb_pure_gaussian(tmp_path: Path) -> None:
    n = 64
    real = np.ones(n, dtype=np.float64)
    imag = np.zeros(n, dtype=np.float64)
    result = _read_synthetic(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=2
    )
    gb = 1.0
    out = apodization.gaussian(result.primary, lb=0.0, gb=gb)
    fid_out = fid_to_complex(out)

    sw_hz = 10000.0
    t = torch.arange(n, dtype=torch.float64) / sw_hz
    sigma_t = math.sqrt(2.0 * math.log(2.0)) / (math.pi * gb)
    expected = torch.exp(-0.5 * (t / sigma_t) ** 2)
    assert torch.allclose(fid_out.real, expected, atol=1e-12)


def test_gaussian_negative_gb_raises(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    with pytest.raises(ValueError, match="positive"):
        apodization.gaussian(result.primary, lb=0.0, gb=-1.0)


def test_gaussian_logs_step(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    out = apodization.gaussian(result.primary, lb=-0.5, gb=1.0)
    history = unpack_metadata(out.schema.metadata).get("x.nmr.processing_history", [])
    assert len(history) == 1
    assert history[0]["step"] == "apodization.gaussian"
    assert history[0]["params"] == {"lb": -0.5, "gb": 1.0}


# ──────────────────────────────────────────────────────────────────────
# sine_bell
# ──────────────────────────────────────────────────────────────────────


def test_sine_bell_default_zero_at_endpoints(tmp_path: Path) -> None:
    n = 64
    real = np.ones(n, dtype=np.float64)
    imag = np.ones(n, dtype=np.float64)
    result = _read_synthetic(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=2
    )
    out = apodization.sine_bell(result.primary)
    fid_out = fid_to_complex(out)
    assert abs(float(fid_out.real[0])) < 1e-12
    assert abs(float(fid_out.real[-1])) < 1e-12


def test_sine_bell_shifted_starts_at_peak(tmp_path: Path) -> None:
    n = 64
    real = np.ones(n, dtype=np.float64)
    imag = np.zeros(n, dtype=np.float64)
    result = _read_synthetic(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=2
    )
    out = apodization.sine_bell(result.primary, off=0.5, end=1.0, power=1)
    fid_out = fid_to_complex(out)
    # off=0.5 → starts at sin(π/2) = 1.
    assert abs(float(fid_out.real[0]) - 1.0) < 1e-12
    assert abs(float(fid_out.real[-1])) < 1e-12


def test_sine_bell_logs_step(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    out = apodization.sine_bell(result.primary, off=0.5, end=1.0, power=2)
    history = unpack_metadata(out.schema.metadata).get("x.nmr.processing_history", [])
    assert len(history) == 1
    assert history[0]["step"] == "apodization.sine_bell"
    assert history[0]["params"] == {"off": 0.5, "end": 1.0, "power": 2}


# ──────────────────────────────────────────────────────────────────────
# Chained calls — history grows
# ──────────────────────────────────────────────────────────────────────


def test_chained_apodization_appends_history(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    step1 = apodization.em(result.primary, lb=1.0)
    step2 = apodization.sine_bell(step1, off=0.5)
    history = unpack_metadata(step2.schema.metadata).get("x.nmr.processing_history", [])
    assert len(history) == 2
    assert history[0]["step"] == "apodization.em"
    assert history[1]["step"] == "apodization.sine_bell"
