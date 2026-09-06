"""Sequence encoding + canonical k-mer hashing (torch-vectorized).

The de novo clusterer needs minimizers over millions of unique
sequences. Rather than a Python loop over sequences (catastrophic at
scale), we concatenate a batch of sequences into one ``uint8`` buffer
and compute per-position canonical k-mer hashes in a handful of torch
ops, masking out k-mers that contain a non-ACGT base or straddle a
sequence boundary.

Base codes: ``A=0 C=1 G=2 T=3``; anything else ``=4`` (a k-mer break).
A k-mer is 2-bit packed into an int64 (``k ≤ 31`` fits), then run
through a SplitMix64 finalizer so window-minimizer selection samples a
near-uniform hash rather than a lexicographic one. The *canonical*
hash is ``min(mix(fwd_code), mix(revcomp_code))`` so a read and its
reverse complement sketch identically.

Everything here operates on a single concatenated block; the chunked
driver that bounds peak memory at PromethION scale lives in
:mod:`.minimizers`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# A=0 C=1 G=2 T=3, everything else (incl. N) = 4 → marks a k-mer break.
_BASE_LUT_NP = np.full(256, 4, dtype=np.int8)
for _ch, _code in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _BASE_LUT_NP[ord(_ch)] = _code
    _BASE_LUT_NP[ord(_ch.lower())] = _code

# SplitMix64 constants (wrapped into signed int64 range so torch can hold
# the bit pattern; arithmetic wraps mod 2**64 in two's complement, which
# matches uint64 low-bit semantics).
_I64_WRAP = 1 << 64
_I64_SIGN = 1 << 63


def _i64(v: int) -> int:
    """Wrap an unsigned 64-bit constant into signed int64 range."""
    v &= _I64_WRAP - 1
    return v - _I64_WRAP if v >= _I64_SIGN else v


_SM_GAMMA = _i64(0x9E3779B97F4A7C15)
_SM_C1 = _i64(0xBF58476D1CE4E5B9)
_SM_C2 = _i64(0x94D049BB133111EB)

# 2**63 - 1 — sentinel hash for invalid k-mer positions (never wins a
# window-min).
INVALID_HASH = (1 << 63) - 1


def _logical_rshift(x: torch.Tensor, s: int) -> torch.Tensor:
    """Unsigned right shift on a signed int64 tensor (mask off the
    sign-extended high bits that an arithmetic ``>>`` would carry in)."""
    if s == 0:
        return x
    return (x >> s) & ((1 << (64 - s)) - 1)


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    """SplitMix64 finalizer — scrambles a 2-bit-packed k-mer code into a
    near-uniform 64-bit hash so minimizer selection isn't lexicographic."""
    x = x + _SM_GAMMA
    x = (x ^ _logical_rshift(x, 30)) * _SM_C1
    x = (x ^ _logical_rshift(x, 27)) * _SM_C2
    x = x ^ _logical_rshift(x, 31)
    return x


@dataclass(frozen=True, slots=True)
class KmerHashes:
    """Per-position canonical k-mer hashes for a concatenated block.

    Arrays are length ``n_pos = T - k + 1`` (k-mer start positions over
    the concatenation of ``T`` bases); invalid positions carry
    ``INVALID_HASH`` and ``valid=False``.
    """

    canon_hash: torch.Tensor  # int64 (n_pos,)
    valid: torch.Tensor  # bool (n_pos,)
    seq_id: torch.Tensor  # int64 (n_pos,) — local index within the block
    pos_in_seq: torch.Tensor  # int32 (n_pos,)


def encode_block(
    data: bytes,
    offsets: np.ndarray,
    *,
    k: int,
) -> KmerHashes | None:
    """Canonical k-mer hashes over a concatenated byte block.

    Parameters
    ----------
    data
        Concatenated sequence bytes (ASCII ACGT…), length ``T``.
    offsets
        int64 array of length ``n_seq + 1`` — ``data[offsets[i]:offsets[i+1]]``
        is sequence ``i``. (The Arrow large_string value-offsets buffer.)
    k
        k-mer length (``≤ 31``).

    Returns ``None`` when the block has no valid k-mer start position
    (``T < k``).
    """
    if k < 1 or k > 31:
        raise ValueError(f"k must be in [1, 31], got {k}")
    codes_np = _BASE_LUT_NP[np.frombuffer(data, dtype=np.uint8)]
    total = codes_np.shape[0]
    n_pos = total - k + 1
    if n_pos <= 0:
        return None

    codes = torch.from_numpy(codes_np.astype(np.int64))

    # Per-position 2-bit packed forward + reverse-complement codes.
    # fwd[i] = sum_j base[i+j] << 2*(k-1-j);  rev[i] = sum_j (3-base[i+j]) << 2*j
    fwd = torch.zeros(n_pos, dtype=torch.int64)
    rev = torch.zeros(n_pos, dtype=torch.int64)
    for j in range(k):
        b = codes[j : j + n_pos]
        fwd |= b << (2 * (k - 1 - j))
        rev |= (3 - b) << (2 * j)

    canon = torch.minimum(_splitmix64(fwd), _splitmix64(rev))

    # Validity: every base in the window is ACGT (code < 4) AND the window
    # lies within one sequence.
    is_acgt = (codes < 4).to(torch.int64)
    csum = torch.zeros(total + 1, dtype=torch.int64)
    torch.cumsum(is_acgt, dim=0, out=csum[1:])
    win_acgt = (csum[k : k + n_pos] - csum[0:n_pos]) == k

    seq_lens = np.diff(offsets)
    seq_id_np = np.repeat(np.arange(len(seq_lens), dtype=np.int64), seq_lens)
    seq_id_full = torch.from_numpy(seq_id_np)  # length T
    pos_in_seq_full = torch.from_numpy(
        (np.arange(total, dtype=np.int64) - offsets[seq_id_np]).astype(np.int64)
    )
    same_seq = seq_id_full[0:n_pos] == seq_id_full[k - 1 : k - 1 + n_pos]

    valid = win_acgt & same_seq
    canon = torch.where(valid, canon, torch.tensor(INVALID_HASH, dtype=torch.int64))

    return KmerHashes(
        canon_hash=canon,
        valid=valid,
        seq_id=seq_id_full[0:n_pos].clone(),
        pos_in_seq=pos_in_seq_full[0:n_pos].to(torch.int32),
    )


__all__ = ["KmerHashes", "encode_block", "INVALID_HASH"]
