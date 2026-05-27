"""Tests for `constellation.nmr.processing.zerofill`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from constellation.core.io.schemas import unpack_metadata
from constellation.nmr.io.bruker import BrukerReader
from constellation.nmr.io.schemas import fid_to_complex
from constellation.nmr.processing import zerofill

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


def test_to_size_pads_to_exact_length(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    out = zerofill.to_size(result.primary, target_size=512)
    fid = fid_to_complex(out)
    assert fid.shape == (512,)


def test_to_size_pad_with_zeros(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=64, DTYPA=2)
    fid_in = fid_to_complex(result.primary)
    out = zerofill.to_size(result.primary, target_size=128)
    fid_out = fid_to_complex(out)
    # Original values preserved at the start; zeros at the end.
    assert torch.allclose(fid_out[:64], fid_in, atol=1e-12)
    assert torch.allclose(
        fid_out[64:], torch.zeros(64, dtype=torch.complex128), atol=1e-12
    )


def test_to_size_updates_shape_metadata(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    out = zerofill.to_size(result.primary, target_size=512)
    meta = unpack_metadata(out.schema.metadata)
    assert meta["x.nmr.shape"] == [512]


def test_to_size_preserves_other_metadata(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    out = zerofill.to_size(result.primary, target_size=512)
    meta_in = unpack_metadata(result.primary.schema.metadata)
    meta_out = unpack_metadata(out.schema.metadata)
    assert meta_out["x.nmr.sw_hz"] == meta_in["x.nmr.sw_hz"]
    assert meta_out["x.nmr.sfo1_mhz"] == meta_in["x.nmr.sfo1_mhz"]
    assert meta_out["x.nmr.axis_domain"] == ["time"]


def test_to_size_same_as_current_returns_input(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    out = zerofill.to_size(result.primary, target_size=128)
    assert out is result.primary  # exact identity — no work done


def test_to_size_smaller_raises(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    with pytest.raises(ValueError, match="one-way"):
        zerofill.to_size(result.primary, target_size=64)


def test_double_doubles_direct_dim(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=100, DTYPA=2)
    out = zerofill.double(result.primary)
    assert unpack_metadata(out.schema.metadata)["x.nmr.shape"] == [200]


def test_to_next_power_of_two(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=100, DTYPA=2)
    out = zerofill.to_next_power_of_two(result.primary)
    # Next power of two ≥ 100 is 128.
    assert unpack_metadata(out.schema.metadata)["x.nmr.shape"] == [128]


def test_to_next_power_of_two_already_power(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    out = zerofill.to_next_power_of_two(result.primary)
    # 128 is already a power of two — returned as-is.
    assert out is result.primary


def test_logs_step(tmp_path: Path) -> None:
    result = _read_synthetic(tmp_path, n_complex=128, DTYPA=2)
    out = zerofill.to_size(result.primary, target_size=512)
    history = unpack_metadata(out.schema.metadata).get("x.nmr.processing_history", [])
    assert len(history) == 1
    assert history[0]["step"] == "zerofill.to_size"
    assert history[0]["params"] == {"target_size": 512}
