"""Window minimizers + sorted minimizer index (torch-vectorized).

A *minimizer* is the minimum-hash k-mer in each sliding window of ``w``
consecutive k-mers. Selecting minimizers reduces each sequence to a
sparse, orientation-independent sketch (~``2/(w+1)`` of positions) that
two near-identical reads share even through basecaller error.

The driver chunks the concatenated sequence buffer at sequence
boundaries so the per-position hash intermediate (the dominant memory
cost) stays bounded to ``chunk_bases`` regardless of dataset size.

Output is the sorted minimizer index — three parallel arrays
``(mini_hash, uniq_id, pos)`` sorted by ``mini_hash`` — which
:mod:`.candidates` linear-scans to find shared-minimizer pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from constellation.sequencing.transcriptome.cluster.denovo.encode import (
    encode_block,
)


@dataclass(frozen=True, slots=True)
class MinimizerIndex:
    """Sorted minimizer index — ``mini_hash`` ascending.

    ``uniq_id`` / ``pos`` are int64 / int32 parallel arrays; row ``i`` is
    a minimizer with hash ``mini_hash[i]`` occurring in unique sequence
    ``uniq_id[i]`` at position ``pos[i]``.
    """

    mini_hash: torch.Tensor  # int64 (K,) ascending
    uniq_id: torch.Tensor  # int64 (K,)
    pos: torch.Tensor  # int32 (K,)


def _large_string_buffers(col: pa.Array | pa.ChunkedArray) -> tuple[bytes, np.ndarray]:
    """Raw data bytes + int64 value-offsets from a (large_)string column."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if not pa.types.is_large_string(col.type):
        col = pc.cast(col, pa.large_string())
    if col.offset != 0:
        col = pa.chunked_array([col]).combine_chunks()
    bufs = col.buffers()
    offsets = np.frombuffer(bufs[1], dtype=np.int64, count=len(col) + 1).copy()
    data = bytes(bufs[2]) if bufs[2] is not None else b""
    return data, offsets


def _block_minimizers(
    canon_hash: torch.Tensor,
    valid: torch.Tensor,
    seq_id: torch.Tensor,
    pos_in_seq: torch.Tensor,
    *,
    w: int,
    uniq_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Window-min over one block's per-position hashes → distinct minimizers."""
    n_pos = canon_hash.shape[0]
    if n_pos < 1:
        empty_i = torch.empty(0, dtype=torch.int64)
        return empty_i, empty_i, torch.empty(0, dtype=torch.int32)
    wsize = min(w, n_pos)
    n_win = n_pos - wsize + 1
    last = wsize - 1
    windows = canon_hash.unfold(0, wsize, 1)  # (n_win, wsize) view
    win_min, win_arg = windows.min(dim=1)
    min_pos_idx = torch.arange(n_win, dtype=torch.int64) + win_arg
    # A window is usable iff its first AND last k-mer are valid (non-
    # boundary-crossing) and in the same sequence. Checking both ends is
    # what stops the concatenation from manufacturing windows whose tail
    # k-mers spill across a sequence boundary — those produce spurious
    # minimizers absent from the isolated sequence (breaking revcomp- and
    # chunk-invariance).
    same_seq = seq_id[0:n_win] == seq_id[last : last + n_win]
    usable = same_seq & valid[0:n_win] & valid[last : last + n_win]
    if not bool(usable.any()):
        empty_i = torch.empty(0, dtype=torch.int64)
        return empty_i, empty_i, torch.empty(0, dtype=torch.int32)
    sel_idx = min_pos_idx[usable]
    sel_hash = win_min[usable]
    sel_uniq = seq_id[sel_idx] + uniq_offset
    sel_pos = pos_in_seq[sel_idx]
    # Dedup distinct (uniq, pos): adjacent windows repeatedly select the
    # same minimizer. (uniq < 2**31, pos < 2**31 → pack into one int64.)
    key = (sel_uniq << 32) | sel_pos.to(torch.int64)
    order = torch.argsort(key)
    key_s = key[order]
    keep = torch.ones(key_s.shape[0], dtype=torch.bool)
    keep[1:] = key_s[1:] != key_s[:-1]
    return sel_hash[order][keep], sel_uniq[order][keep], sel_pos[order][keep]


def extract_minimizers(
    seq_col: pa.Array | pa.ChunkedArray,
    *,
    k: int = 15,
    w: int = 10,
    chunk_bases: int = 50_000_000,
) -> MinimizerIndex:
    """Extract canonical minimizers for every sequence, return a sorted index.

    ``seq_col`` is the unique-sequence column; ``uniq_id`` is the row
    index. Sequences are processed in boundary-aligned chunks of
    ~``chunk_bases`` bases so the per-position hash intermediate never
    exceeds one chunk.
    """
    data, offsets = _large_string_buffers(seq_col)
    n_seq = len(offsets) - 1
    if n_seq == 0:
        empty_i = torch.empty(0, dtype=torch.int64)
        return MinimizerIndex(empty_i, empty_i, torch.empty(0, dtype=torch.int32))

    hashes: list[torch.Tensor] = []
    uniqs: list[torch.Tensor] = []
    poss: list[torch.Tensor] = []

    r0 = 0
    while r0 < n_seq:
        # Grow the chunk until it spans ~chunk_bases (always ≥ 1 sequence).
        base0 = int(offsets[r0])
        r1 = r0 + 1
        while r1 < n_seq and int(offsets[r1]) - base0 < chunk_bases:
            r1 += 1
        sub_data = data[base0 : int(offsets[r1])]
        sub_offsets = offsets[r0 : r1 + 1] - base0
        kh = encode_block(sub_data, sub_offsets, k=k)
        if kh is not None:
            h, u, p = _block_minimizers(
                kh.canon_hash, kh.valid, kh.seq_id, kh.pos_in_seq, w=w, uniq_offset=r0
            )
            if h.shape[0] > 0:
                hashes.append(h)
                uniqs.append(u)
                poss.append(p)
        r0 = r1

    if not hashes:
        empty_i = torch.empty(0, dtype=torch.int64)
        return MinimizerIndex(empty_i, empty_i, torch.empty(0, dtype=torch.int32))

    all_hash = torch.cat(hashes)
    all_uniq = torch.cat(uniqs)
    all_pos = torch.cat(poss)
    order = torch.argsort(all_hash)
    return MinimizerIndex(
        mini_hash=all_hash[order],
        uniq_id=all_uniq[order],
        pos=all_pos[order],
    )


__all__ = ["MinimizerIndex", "extract_minimizers"]
