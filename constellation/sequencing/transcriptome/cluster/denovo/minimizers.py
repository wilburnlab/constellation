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
    uniq_id: torch.Tensor  # int32 (K,)
    pos: torch.Tensor  # int32 (K,)


def _iter_seq_blocks(seq_col, chunk_bases: int):
    """Yield ``(data_bytes, rel_offsets, row0)`` boundary-aligned blocks.

    Walks the Arrow chunks directly via zero-copy ``memoryview`` and only
    materialises one ~``chunk_bases`` sub-block at a time — avoids the
    ``combine_chunks()`` + ``bytes()`` copy of the whole (tens-of-GB)
    sequence buffer that a single-buffer extraction would force.
    """
    chunks = seq_col.chunks if isinstance(seq_col, pa.ChunkedArray) else [seq_col]
    row0 = 0
    for ch in chunks:
        nch = len(ch)
        if nch == 0:
            continue
        if not pa.types.is_large_string(ch.type):
            ch = pc.cast(ch, pa.large_string())
        bufs = ch.buffers()
        off = np.frombuffer(bufs[1], dtype=np.int64)
        base = ch.offset
        data = memoryview(bufs[2]) if bufs[2] is not None else memoryview(b"")
        i = 0
        while i < nch:
            byte0 = int(off[base + i])
            j = i + 1
            while j < nch and int(off[base + j]) - byte0 < chunk_bases:
                j += 1
            byte1 = int(off[base + j])
            sub_data = bytes(data[byte0:byte1])  # bounded ~chunk_bases copy
            sub_off = (off[base + i : base + j + 1] - byte0).astype(np.int64)
            yield sub_data, sub_off, row0 + i
            i = j
        row0 += nch


def _cap_per_seq(
    h: torch.Tensor, u: torch.Tensor, p: torch.Tensor, m: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the ``m`` smallest-hash minimizers per sequence (bottom-m sketch).

    Bounds the index to ``n_seq × m`` entries — at ~1 kb reads the uncapped
    density is ~180/kb, so capping is the dominant memory lever. Two
    near-identical reads still share most of their bottom-m set, so
    candidate sensitivity stays high at the identity targets we cluster at.
    Applied per block (sequences are boundary-aligned within a block, so this
    is exactly per-sequence) to avoid a global sort.
    """
    hn = h.numpy()
    un = u.numpy()
    order = np.lexsort((hn, un))  # primary uniq asc, secondary hash asc
    us = un[order]
    n = us.shape[0]
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = us[1:] != us[:-1]
    grp_start = np.maximum.accumulate(np.where(change, np.arange(n), 0))
    keep = (np.arange(n) - grp_start) < m
    sel = torch.from_numpy(np.ascontiguousarray(order[keep]))
    return h[sel], u[sel], p[sel]


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
    max_per_seq: int | None = 50,
    chunk_bases: int = 50_000_000,
) -> MinimizerIndex:
    """Extract canonical minimizers for every sequence, return a sorted index.

    ``seq_col`` is the unique-sequence column; ``uniq_id`` is the row index.
    Sequences are processed in boundary-aligned blocks of ~``chunk_bases``
    bases (zero-copy from the Arrow buffers) so the per-position hash
    intermediate stays bounded. ``max_per_seq`` caps each sequence's sketch
    to its ``m`` smallest-hash minimizers (``None`` = uncapped) — the
    dominant memory lever at scale.
    """
    if len(seq_col) == 0:
        empty_i = torch.empty(0, dtype=torch.int64)
        return MinimizerIndex(empty_i, empty_i, torch.empty(0, dtype=torch.int32))

    hashes: list[torch.Tensor] = []
    uniqs: list[torch.Tensor] = []
    poss: list[torch.Tensor] = []

    for sub_data, sub_offsets, row0 in _iter_seq_blocks(seq_col, chunk_bases):
        kh = encode_block(sub_data, sub_offsets, k=k)
        if kh is None:
            continue
        h, u, p = _block_minimizers(
            kh.canon_hash, kh.valid, kh.seq_id, kh.pos_in_seq, w=w, uniq_offset=row0
        )
        if h.shape[0] == 0:
            continue
        if max_per_seq is not None:
            h, u, p = _cap_per_seq(h, u, p, int(max_per_seq))
        hashes.append(h)
        uniqs.append(u)
        poss.append(p)

    if not hashes:
        empty_i = torch.empty(0, dtype=torch.int64)
        return MinimizerIndex(empty_i, empty_i, torch.empty(0, dtype=torch.int32))

    all_hash = torch.cat(hashes)
    all_uniq = torch.cat(uniqs).to(torch.int32)  # uniq_id < 2**31
    all_pos = torch.cat(poss)
    order = torch.argsort(all_hash)
    return MinimizerIndex(
        mini_hash=all_hash[order],
        uniq_id=all_uniq[order],
        pos=all_pos[order],
    )


__all__ = ["MinimizerIndex", "extract_minimizers"]
