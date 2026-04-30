"""Tests for ``massspec.io.encyclopedia._codec`` — zlib + big-endian round-trips."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
import torch

from constellation.massspec.io.encyclopedia._codec import (
    compress_chromatogram,
    compress_intensity,
    compress_mz,
    decompress_chromatogram,
    decompress_intensity,
    decompress_mz,
    decompress_quantified_ions,
)


# ──────────────────────────────────────────────────────────────────────
# m/z round-trip (>f8 / float64)
# ──────────────────────────────────────────────────────────────────────


def test_mz_round_trip_preserves_float64_values_exactly():
    """The compress/decompress pair is bit-exact for float64 — m/z values
    must round-trip without quantization."""
    mz = torch.tensor(
        [120.080017, 200.123456, 471.456789, 1012.4529950586647, 1735.8237],
        dtype=torch.float64,
    )
    blob, n = compress_mz(mz)
    back = decompress_mz(blob, n)
    assert back.dtype == torch.float64
    assert torch.allclose(mz, back, atol=0.0, rtol=0.0)


def test_mz_encoded_length_matches_uncompressed_byte_count():
    """encoded_length is the *uncompressed* byte count (8 bytes per
    float64) — that's the contract EncyclopeDIA stores in the sibling
    ``MassEncodedLength`` column."""
    mz = torch.arange(7, dtype=torch.float64)
    _, n = compress_mz(mz)
    assert n == 7 * 8


def test_mz_decode_rejects_wrong_encoded_length():
    """A wrong encoded_length value must surface as a clear error."""
    mz = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
    blob, _n = compress_mz(mz)
    # Buffer is 24 bytes; claim 99 elements (792 bytes) — must fail.
    with pytest.raises(ValueError):
        decompress_mz(blob, 99 * 8)


# ──────────────────────────────────────────────────────────────────────
# Intensity round-trip (>f4 / float32)
# ──────────────────────────────────────────────────────────────────────


def test_intensity_round_trip_preserves_float32_values():
    ints = torch.tensor([1.0, 2.5, 1e6, 1234.5], dtype=torch.float32)
    blob, n = compress_intensity(ints)
    back = decompress_intensity(blob, n)
    assert back.dtype == torch.float32
    assert torch.allclose(ints, back, atol=0.0, rtol=0.0)
    assert n == 4 * 4


def test_chromatogram_round_trip_uses_intensity_codec():
    """MedianChromatogramArray uses the same `>f4` layout as
    IntensityArray — ``decompress_chromatogram`` is just a named alias."""
    chrom = torch.tensor([0.0, 0.5, 1.2, 0.8, 0.3], dtype=torch.float32)
    blob, n = compress_chromatogram(chrom)
    back = decompress_chromatogram(blob, n)
    assert torch.allclose(chrom, back, atol=0.0, rtol=0.0)


# ──────────────────────────────────────────────────────────────────────
# Big-endian on disk — explicitly verify the byte order
# ──────────────────────────────────────────────────────────────────────


def test_disk_payload_is_big_endian_float64():
    """Decoding our blob with little-endian dtype must produce different
    values — proves we really wrote big-endian."""
    mz = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    blob, _n = compress_mz(mz)
    raw = zlib.decompress(blob)
    big = np.frombuffer(raw, dtype=">f8")
    little = np.frombuffer(raw, dtype="<f8")
    assert np.allclose(big, [1.0, 2.0, 3.0])
    assert not np.allclose(little, [1.0, 2.0, 3.0])


def test_disk_payload_is_big_endian_float32():
    ints = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    blob, _n = compress_intensity(ints)
    raw = zlib.decompress(blob)
    big = np.frombuffer(raw, dtype=">f4")
    little = np.frombuffer(raw, dtype="<f4")
    assert np.allclose(big, [1.0, 2.0, 3.0])
    assert not np.allclose(little, [1.0, 2.0, 3.0])


# ──────────────────────────────────────────────────────────────────────
# QuantifiedIonsArray — best-effort
# ──────────────────────────────────────────────────────────────────────


def test_quantified_ions_returns_none_for_null_blob():
    assert decompress_quantified_ions(None) is None


def test_quantified_ions_decodes_clean_float32_payload():
    arr = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    blob, _ = compress_intensity(arr)
    out = decompress_quantified_ions(blob)
    assert out is not None
    assert torch.allclose(out, arr, atol=0.0, rtol=0.0)


def test_quantified_ions_returns_none_for_misaligned_blob():
    """A blob whose decompressed length isn't a multiple of 4 can't be
    a `>f4` array; the helper returns None rather than guessing."""
    blob = zlib.compress(b"\x01\x02\x03")  # 3 bytes
    assert decompress_quantified_ions(blob) is None
