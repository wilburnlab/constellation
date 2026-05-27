"""Tests for `constellation.nmr.io.bruker.BrukerReader` and helpers.

Uses programmatically generated synthetic Bruker 1D experiment
directories — minimal `acqus` (JCAMP-DX text) plus `fid` binary
(interleaved real / imag). Real Bruker datasets are not vendored yet;
adding an integration test against a real Lysin spectrum is a separate
step once the test-fixture binary handling decision lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import torch

from constellation.core.io.bundle import DirBundle
from constellation.core.io.readers import ReadResult
from constellation.core.io.schemas import unpack_metadata
from constellation.nmr.io.bruker import BrukerReader
from constellation.nmr.io.schemas import NMR_FID_TABLE, fid_to_complex


# ──────────────────────────────────────────────────────────────────────
# Synthetic Bruker dataset generator
# ──────────────────────────────────────────────────────────────────────


# Fields populated by default. Tests override via kwargs.
_DEFAULT_ACQUS: dict[str, Any] = {
    "TD": 2048,         # = 2 × n_complex (default n_complex = 1024)
    "SW_h": 10000.0,
    # SFO1, BF1, O1 satisfy the Bruker relation SFO1 = BF1 + O1/1e6.
    # The carrier sits 2400 Hz above the BF1 frequency, putting TMS
    # (at the BF1 frequency by convention) at -2400 Hz in the
    # carrier-relative spectrum — typical ¹H NMR setup.
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
    """Build a minimal synthetic Bruker 1D experiment under `directory`.

    Returns the ``(real, imag)`` arrays that were written so tests can
    assert numerical recovery. Caller is responsible for ``directory``
    already existing.
    """
    params = {**_DEFAULT_ACQUS, **acqus_overrides}
    params["TD"] = 2 * n_complex

    # Defaults use integer-valued arrays so they round-trip safely through the
    # Bruker int32 default (DTYPA=0). Tests that exercise floating-point
    # storage explicitly override DTYPA=2.
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


# ──────────────────────────────────────────────────────────────────────
# Core read path — happy cases
# ──────────────────────────────────────────────────────────────────────


def test_read_returns_nmr_fid_table_with_two_rows(tmp_path: Path) -> None:
    real, imag = write_synthetic_bruker_1d(tmp_path, n_complex=512)
    result = BrukerReader().read(tmp_path)

    assert isinstance(result, ReadResult)
    assert result.primary.schema.equals(NMR_FID_TABLE.with_metadata(
        result.primary.schema.metadata
    ))
    assert result.primary.num_rows == 2

    components = result.primary.column("component").to_pylist()
    assert set(components) == {"real", "imag"}


def test_read_recovers_fid_values_int32_le(tmp_path: Path) -> None:
    n = 256
    # Use integer values that round-trip cleanly through int32 storage.
    real = np.arange(n, dtype=np.float64) - n // 2  # symmetric around 0
    imag = np.arange(n, dtype=np.float64) * 2
    write_synthetic_bruker_1d(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=0, BYTORDA=0
    )
    result = BrukerReader().read(tmp_path)

    fid = fid_to_complex(result.primary)
    assert fid.dtype == torch.complex128
    assert fid.shape == (n,)
    assert torch.allclose(fid.real, torch.from_numpy(real), atol=1e-10)
    assert torch.allclose(fid.imag, torch.from_numpy(imag), atol=1e-10)


def test_read_recovers_fid_values_float64_le(tmp_path: Path) -> None:
    n = 256
    real = np.sin(np.linspace(0.0, np.pi, n, dtype=np.float64))
    imag = np.cos(np.linspace(0.0, np.pi, n, dtype=np.float64))
    write_synthetic_bruker_1d(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=2, BYTORDA=0
    )
    result = BrukerReader().read(tmp_path)

    fid = fid_to_complex(result.primary)
    assert torch.allclose(fid.real, torch.from_numpy(real), atol=1e-15)
    assert torch.allclose(fid.imag, torch.from_numpy(imag), atol=1e-15)


def test_read_recovers_fid_values_int32_big_endian(tmp_path: Path) -> None:
    n = 128
    real = (np.arange(n, dtype=np.float64) - 64) * 100  # integer values
    imag = (np.arange(n, dtype=np.float64) - 32) * 50
    write_synthetic_bruker_1d(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=0, BYTORDA=1
    )
    result = BrukerReader().read(tmp_path)

    fid = fid_to_complex(result.primary)
    assert torch.allclose(fid.real, torch.from_numpy(real), atol=1e-10)
    assert torch.allclose(fid.imag, torch.from_numpy(imag), atol=1e-10)


def test_read_recovers_fid_values_float64_big_endian(tmp_path: Path) -> None:
    n = 128
    real = np.linspace(0.1, 0.9, n, dtype=np.float64)
    imag = np.linspace(-0.5, 0.5, n, dtype=np.float64)
    write_synthetic_bruker_1d(
        tmp_path, n_complex=n, real=real, imag=imag, DTYPA=2, BYTORDA=1
    )
    result = BrukerReader().read(tmp_path)

    fid = fid_to_complex(result.primary)
    assert torch.allclose(fid.real, torch.from_numpy(real), atol=1e-15)
    assert torch.allclose(fid.imag, torch.from_numpy(imag), atol=1e-15)


# ──────────────────────────────────────────────────────────────────────
# Metadata stamping
# ──────────────────────────────────────────────────────────────────────


def test_metadata_stamps_required_x_nmr_keys(tmp_path: Path) -> None:
    write_synthetic_bruker_1d(tmp_path, n_complex=512)
    result = BrukerReader().read(tmp_path)
    meta = unpack_metadata(result.primary.schema.metadata)

    assert meta["x.nmr.shape"] == [512]
    assert meta["x.nmr.dimensionality"] == 1
    assert meta["x.nmr.axis_domain"] == ["time"]
    assert meta["x.nmr.sw_hz"] == 10000.0
    assert meta["x.nmr.sfo1_mhz"] == 600.0024
    assert meta["x.nmr.bf1_mhz"] == 600.0
    assert meta["x.nmr.o1_hz"] == 2400.0
    assert meta["x.nmr.dtypa"] == 0
    assert meta["x.nmr.bytorda"] == 0


def test_metadata_stamps_optional_keys_when_present(tmp_path: Path) -> None:
    write_synthetic_bruker_1d(tmp_path, n_complex=256)
    result = BrukerReader().read(tmp_path)
    meta = unpack_metadata(result.primary.schema.metadata)

    assert meta["x.nmr.grpdly"] == 67.98
    assert meta["x.nmr.nucleus"] == "1H"
    assert meta["x.nmr.ns"] == 16
    assert meta["x.nmr.rg"] == 256.0
    assert meta["x.nmr.pulprog"] == "zg30"
    assert meta["x.nmr.temp_k"] == 298.0
    assert meta["x.nmr.solvent"] == "D2O"


def test_metadata_skips_optional_keys_when_absent(tmp_path: Path) -> None:
    # Override the optional fields to be absent by passing only required ones.
    minimal_params = {
        "TD": 2 * 256,
        "SW_h": 8000.0,
        # SFO1 = BF1 + O1/1e6 → 500.0 + 0.0025 = 500.0025.
        "SFO1": 500.0025,
        "BF1": 500.0,
        "O1": 2500.0,
        "DTYPA": 0,
        "BYTORDA": 0,
    }
    _write_acqus(tmp_path / "acqus", minimal_params)
    real = np.zeros(256, dtype=np.float64)
    imag = np.zeros(256, dtype=np.float64)
    _write_fid_binary(
        tmp_path / "fid", real=real, imag=imag, dtypa=0, bytorda=0
    )

    result = BrukerReader().read(tmp_path)
    meta = unpack_metadata(result.primary.schema.metadata)

    # Required keys present:
    assert "x.nmr.sw_hz" in meta
    # Optional keys absent (must not be silently sentinel-filled):
    for key in (
        "x.nmr.grpdly",
        "x.nmr.nucleus",
        "x.nmr.ns",
        "x.nmr.rg",
        "x.nmr.pulprog",
        "x.nmr.temp_k",
        "x.nmr.solvent",
    ):
        assert key not in meta, f"unexpected sentinel for absent acqus field: {key}"


def test_run_metadata_populated(tmp_path: Path) -> None:
    expno_dir = tmp_path / "lysin-20260512" / "1"
    expno_dir.mkdir(parents=True)
    write_synthetic_bruker_1d(expno_dir, n_complex=128)

    result = BrukerReader().read(expno_dir)
    assert result.run_metadata["device"] == "Bruker"
    assert result.run_metadata["instrument_id"] == "AVANCE600"
    assert result.run_metadata["run_id"] == "lysin-20260512/1"
    assert "raw_acqus" in result.run_metadata
    assert result.run_metadata["raw_acqus"]["TD"] == 256  # 2 × 128


# ──────────────────────────────────────────────────────────────────────
# DirBundle input
# ──────────────────────────────────────────────────────────────────────


def test_read_accepts_dirbundle(tmp_path: Path) -> None:
    real, imag = write_synthetic_bruker_1d(tmp_path, n_complex=64)
    bundle = DirBundle(tmp_path, primary="fid")

    result = BrukerReader().read(bundle)
    fid = fid_to_complex(result.primary)
    assert torch.allclose(fid.real, torch.from_numpy(real), atol=1e-10)
    assert torch.allclose(fid.imag, torch.from_numpy(imag), atol=1e-10)


# ──────────────────────────────────────────────────────────────────────
# Error paths
# ──────────────────────────────────────────────────────────────────────


def test_missing_acqus_raises(tmp_path: Path) -> None:
    (tmp_path / "fid").write_bytes(b"\x00" * 1024)
    with pytest.raises(FileNotFoundError, match="acqus"):
        BrukerReader().read(tmp_path)


def test_missing_fid_and_ser_raises(tmp_path: Path) -> None:
    _write_acqus(tmp_path / "acqus", _DEFAULT_ACQUS)
    with pytest.raises(FileNotFoundError, match="neither 'fid' nor 'ser'"):
        BrukerReader().read(tmp_path)


def test_ser_present_raises_not_implemented(tmp_path: Path) -> None:
    """2D+ datasets should raise a clear NotImplementedError pointing at the future plan."""
    write_synthetic_bruker_1d(tmp_path, n_complex=64)
    # Add a sham `ser` file alongside the `fid` to trigger the 2D+ dispatch.
    (tmp_path / "ser").write_bytes(b"\x00" * 1024)

    with pytest.raises(NotImplementedError, match="2D"):
        BrukerReader().read(tmp_path)


def test_bad_dtypa_raises(tmp_path: Path) -> None:
    write_synthetic_bruker_1d(tmp_path, n_complex=64, DTYPA=99)
    with pytest.raises(ValueError, match="DTYPA"):
        BrukerReader().read(tmp_path)


def test_bad_bytorda_raises(tmp_path: Path) -> None:
    write_synthetic_bruker_1d(tmp_path, n_complex=64, BYTORDA=99)
    with pytest.raises(ValueError, match="BYTORDA"):
        BrukerReader().read(tmp_path)


def test_non_directory_source_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "not-a-dir.txt"
    bogus.write_text("hello")
    with pytest.raises(NotADirectoryError):
        BrukerReader().read(bogus)


# ──────────────────────────────────────────────────────────────────────
# Schema helper — fid_to_complex
# ──────────────────────────────────────────────────────────────────────


def test_fid_to_complex_rejects_malformed_row_count() -> None:
    # Build a table with three rows instead of two.
    components = pa.array(["real", "imag", "extra"], type=pa.string())
    fid_lists = pa.array(
        [[0.0], [0.0], [0.0]], type=pa.list_(pa.float64())
    )
    table = pa.Table.from_arrays([components, fid_lists], schema=NMR_FID_TABLE)
    with pytest.raises(ValueError, match="exactly 2 rows"):
        fid_to_complex(table)


def test_fid_to_complex_rejects_missing_shape_metadata() -> None:
    components = pa.array(["real", "imag"], type=pa.string())
    fid_lists = pa.array(
        [[0.0, 1.0], [2.0, 3.0]], type=pa.list_(pa.float64())
    )
    table = pa.Table.from_arrays([components, fid_lists], schema=NMR_FID_TABLE)
    # NMR_FID_TABLE already carries schema_name metadata but not x.nmr.shape.
    with pytest.raises(ValueError, match="x.nmr.shape"):
        fid_to_complex(table)
