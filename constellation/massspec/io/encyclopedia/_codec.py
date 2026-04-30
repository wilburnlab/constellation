"""zlib + big-endian-float decode/encode for EncyclopeDIA blobs.

EncyclopeDIA stores per-precursor m/z, intensity, chromatogram, and
quantified-ion arrays as zlib-compressed payloads of Java-style
big-endian floats. The uncompressed byte length lives in a sibling
``*EncodedLength`` integer column on the same row — there is no
length-prefix inside the blob itself.

Layout:
    MassArray              zlib(``>f8`` big-endian float64)
    IntensityArray         zlib(``>f4`` big-endian float32)
    MedianChromatogramArray zlib(``>f4`` big-endian float32)
    CorrelationArray       sometimes ``>f4`` (small), often NULL
    QuantifiedIonsArray    variable; usually ``>f4``, occasionally NULL

CLAUDE.md "Torch-first numerics" rule: torch has no native big-endian
dtype as of 2.5, so we do **one** numpy boundary crossing per blob
(``np.frombuffer`` + ``.astype("<f8").copy()``) and then hand a
little-endian torch tensor back. Going the other way, torch → big-endian
bytes uses the same numpy bridge.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# decode
# ──────────────────────────────────────────────────────────────────────


def decompress_mz(blob: bytes, encoded_length: int) -> torch.Tensor:
    """zlib-decode a `>f8` big-endian float64 array → torch.float64 tensor."""
    raw = zlib.decompress(blob)
    arr = np.frombuffer(raw, dtype=">f8")
    n_expected = encoded_length // 8
    if arr.size != n_expected:
        raise ValueError(
            f"mz blob length mismatch: decoded {arr.size} float64 elements "
            f"({len(raw)} bytes); encoded_length={encoded_length} implies {n_expected}"
        )
    return torch.from_numpy(arr.astype("<f8").copy())


def decompress_intensity(blob: bytes, encoded_length: int) -> torch.Tensor:
    """zlib-decode a `>f4` big-endian float32 array → torch.float32 tensor."""
    raw = zlib.decompress(blob)
    arr = np.frombuffer(raw, dtype=">f4")
    n_expected = encoded_length // 4
    if arr.size != n_expected:
        raise ValueError(
            f"intensity blob length mismatch: decoded {arr.size} float32 elements "
            f"({len(raw)} bytes); encoded_length={encoded_length} implies {n_expected}"
        )
    return torch.from_numpy(arr.astype("<f4").copy())


def decompress_chromatogram(blob: bytes, encoded_length: int) -> torch.Tensor:
    """``MedianChromatogramArray`` decode — ``>f4`` like intensity."""
    return decompress_intensity(blob, encoded_length)


def decompress_quantified_ions(blob: bytes | None) -> torch.Tensor | None:
    """Best-effort decode for ``QuantifiedIonsArray``.

    The blob is variable-format and not all dlib/elib files populate it
    consistently. Returns ``None`` when the blob is missing or doesn't
    decode as a clean ``>f4`` array (length not divisible by 4).
    """
    if blob is None:
        return None
    raw = zlib.decompress(blob)
    if len(raw) % 4 != 0:
        return None
    arr = np.frombuffer(raw, dtype=">f4")
    if arr.size == 0:
        return None
    return torch.from_numpy(arr.astype("<f4").copy())


# ──────────────────────────────────────────────────────────────────────
# encode
# ──────────────────────────────────────────────────────────────────────


def compress_mz(arr: torch.Tensor) -> tuple[bytes, int]:
    """Return ``(zlib_blob, uncompressed_byte_length)`` for ``>f8`` storage."""
    np_arr = arr.detach().to(torch.float64).cpu().numpy().astype(">f8")
    raw = np_arr.tobytes()
    return zlib.compress(raw), len(raw)


def compress_intensity(arr: torch.Tensor) -> tuple[bytes, int]:
    """Return ``(zlib_blob, uncompressed_byte_length)`` for ``>f4`` storage."""
    np_arr = arr.detach().to(torch.float32).cpu().numpy().astype(">f4")
    raw = np_arr.tobytes()
    return zlib.compress(raw), len(raw)


def compress_chromatogram(arr: torch.Tensor) -> tuple[bytes, int]:
    return compress_intensity(arr)


__all__ = [
    "decompress_mz",
    "decompress_intensity",
    "decompress_chromatogram",
    "decompress_quantified_ions",
    "compress_mz",
    "compress_intensity",
    "compress_chromatogram",
]
