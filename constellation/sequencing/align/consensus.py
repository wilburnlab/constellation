"""Weighted-PWM consensus on a fixed coordinate frame.

The kernel projects each cluster member's aligned bases onto a shared
reference coordinate frame (genome window for Phase 2 genome-guided
clustering, centroid sequence for Phase 3 de novo clustering), then
emits a per-position weighted majority vote as the consensus
sequence.

Substrate-agnostic: the caller supplies a reference sequence + start
offset and a per-member ``(alignment_id, weight, cs_string,
alignment_ref_start)`` tuple. cs:long is parsed in the same grammar
as ``cigar.parse_cs_long_blocks`` — match / substitution / insert /
delete / splice — and projected onto reference positions:

    match (`:N` or `=NNNN`)   vote for reference_sequence[p] at each p
    substitution (`*ab`)      vote for query base b at p
    deletion (`-xxx`)          vote for gap at each spanned p
    insertion (`+xxx`)         skipped (can't be expressed in genome
                               coordinates without expanding the PWM)
    splice (`~aa<N>bb`)        skipped; advance ref_pos by N

After accumulation, ``pwm.argmax(dim=1)`` picks the winning base per
position; gap-winning positions drop out of the emitted consensus, so
the output sequence is the same length as the reference window minus
the count of cluster-wide deletion-dominant positions.

Phase 3 reuses the kernel verbatim with centroid sequence as the
reference frame and a freshly-synthesised cs:long per member from
edlib NW alignment of the member to the centroid.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import torch

from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_CS_TABLE,
)


# A=0, C=1, G=2, T=3, gap=4. Lower-case maps to the same indices.
# 'N' / ambiguous bases are encoded as 5 (sentinel — votes are
# discarded before scatter_add).
_BASE_TO_IDX = {
    "A": 0, "C": 1, "G": 2, "T": 3,
    "a": 0, "c": 1, "g": 2, "t": 3,
}
_GAP = 4
_AMB = 5
_IDX_TO_BASE = "ACGT-"
# Numpy-friendly base-code lookup: index → ASCII byte. Used by the
# vectorized consensus_chars assembly. Position 5 (_AMB) maps to a
# sentinel that we filter out before joining.
_IDX_TO_BASE_NP = np.array(
    [ord("A"), ord("C"), ord("G"), ord("T"), ord("-"), ord("?")],
    dtype=np.uint8,
)


# Reuse the cs grammar from align/cigar.py — kept as a private module-
# level compile here to avoid import cycles when consumers want only
# consensus and not the block parsers.
_CS_TOKEN_RE = re.compile(
    r"""
    (?:
        :(?P<match_short>\d+)
      | =(?P<match_long>[A-Za-z]+)
      | \*(?P<sub_ref>[A-Za-z])(?P<sub_query>[A-Za-z])
      | \+(?P<insert>[A-Za-z]+)
      | -(?P<delete>[A-Za-z]+)
      | ~(?P<splice_donor>[A-Za-z]{2})(?P<splice_len>\d+)(?P<splice_acceptor>[A-Za-z]{2})
    )
    """,
    re.VERBOSE,
)


def _cs_votes(
    cs_string: str,
    *,
    alignment_ref_start: int,
    window_start: int,
    window_end: int,
    reference_sequence: str,
) -> tuple[list[int], list[int]]:
    """Walk a cs:long string and emit (window-relative ref index, base
    code) pairs.

    Positions outside ``[window_start, window_end)`` are clipped — the
    caller's PWM only has columns for positions inside the window.
    Insertions and splice operators are skipped (no projection onto the
    window). Returns parallel lists ``(indices, bases)`` ready for
    scatter_add accumulation.
    """
    indices: list[int] = []
    bases: list[int] = []
    ref_pos = alignment_ref_start
    pos = 0

    def vote_match_block(length: int) -> None:
        # Match block: read agrees with reference at each of length
        # consecutive ref positions starting at ref_pos.
        for offset in range(length):
            p = ref_pos + offset
            if p < window_start or p >= window_end:
                continue
            ref_base = reference_sequence[p - window_start]
            base_idx = _BASE_TO_IDX.get(ref_base, _AMB)
            if base_idx == _AMB:
                continue
            indices.append(p - window_start)
            bases.append(base_idx)

    for m in _CS_TOKEN_RE.finditer(cs_string):
        if m.start() != pos:
            raise ValueError(f"malformed cs string at offset {pos}: {cs_string!r}")
        pos = m.end()
        if (g := m.group("match_short")) is not None:
            length = int(g)
            vote_match_block(length)
            ref_pos += length
        elif (g := m.group("match_long")) is not None:
            length = len(g)
            vote_match_block(length)
            ref_pos += length
        elif m.group("sub_ref") is not None:
            query = m.group("sub_query")
            base_idx = _BASE_TO_IDX.get(query, _AMB)
            if window_start <= ref_pos < window_end and base_idx != _AMB:
                indices.append(ref_pos - window_start)
                bases.append(base_idx)
            ref_pos += 1
        elif (g := m.group("insert")) is not None:
            # Insertions are query-side bases between ref positions —
            # not projectable onto the genome-anchored PWM. Skip.
            pass
        elif (g := m.group("delete")) is not None:
            length = len(g)
            for offset in range(length):
                p = ref_pos + offset
                if window_start <= p < window_end:
                    indices.append(p - window_start)
                    bases.append(_GAP)
            ref_pos += length
        elif (g := m.group("splice_len")) is not None:
            length = int(g)
            ref_pos += length
        else:  # pragma: no cover
            raise ValueError(f"unrecognised cs token at offset {m.start()}")

    if pos != len(cs_string):
        raise ValueError(f"trailing bytes in cs string: {cs_string!r}")

    return indices, bases


def build_consensus(
    *,
    member_alignment_ids: list[int],
    member_weights: list[float],
    member_ref_starts: list[int],
    alignment_cs: pa.Table,
    reference_sequence: str,
    reference_start: int,
    min_depth: float = 1.0,
    threads: int = 1,
    scatter_chunk_size: int = 100_000,
) -> str:
    """Build a per-position weighted-majority-vote consensus.

    Parameters
    ----------
    member_alignment_ids
        Cluster members' ``alignment_id``s — index into ``alignment_cs``.
    member_weights
        Per-member abundance weight (Layer-0 derep count for Phase 2,
        de novo abundance for Phase 3). Same length as
        ``member_alignment_ids``. Weights need not sum to anything in
        particular — only the ratios at each position matter.
    member_ref_starts
        Per-member 0-based reference position where the alignment begins
        (i.e. ``ALIGNMENT_TABLE.ref_start`` for that alignment_id). Same
        length as ``member_alignment_ids``.
    alignment_cs
        ``ALIGNMENT_CS_TABLE``-shaped table containing at least the rows
        for ``member_alignment_ids``. Lookup is by ``alignment_id`` →
        ``cs_string``. Required; ``--build-consensus`` is gated on
        ``--emit-cs-tags`` upstream.
    reference_sequence
        The window the consensus is anchored to. Phase 2: genome slice
        across the cluster's span. Phase 3: centroid read sequence.
    reference_start
        0-based offset of ``reference_sequence`` within whatever
        coordinate frame the caller is using (genome contig pos for
        Phase 2; 0 for Phase 3 where the centroid IS the frame).
    min_depth
        Reject positions whose winning vote weight is below this — the
        consensus emits the reference base at those positions as a
        fallback (preserves window length when the cluster doesn't
        cover certain positions). Default 1.0.
    threads
        Number of worker threads for parallel cs:long parsing. ``1``
        (default) runs the parser inline on the calling thread, which
        is what every existing call site + every test exercises.
        ``threads>1`` dispatches ``_cs_votes`` calls across a
        ``ThreadPoolExecutor`` — Python's regex module releases the
        GIL during match execution, so this scales near-linearly on
        cs:long-bound clusters (mega-clusters with millions of members).
        Output is deterministic regardless of ``threads`` because
        ``index_add_`` is commutative.
    scatter_chunk_size
        Per-chunk member count for the batched torch ``index_add_``
        accumulation. Bounds peak intermediate tensor size to roughly
        ``chunk_size × avg_votes_per_member × 16 bytes``. Default
        100K members ≈ ~200 MB intermediate at typical cs:long density.

    Returns
    -------
    str
        Consensus sequence. Length ≤ ``len(reference_sequence)`` —
        positions where gap wins drop out. ASCII upper-case ACGT.

    Raises
    ------
    ValueError
        If ``alignment_cs`` is empty (caller forgot ``--emit-cs-tags``)
        or if member arrays disagree in length.
    """
    if alignment_cs.num_rows == 0:
        raise ValueError(
            "build_consensus requires populated alignment_cs (run "
            "`transcriptome align --emit-cs-tags` upstream)"
        )
    n_members = len(member_alignment_ids)
    if not (len(member_weights) == n_members == len(member_ref_starts)):
        raise ValueError(
            f"member arrays length mismatch: ids={n_members} "
            f"weights={len(member_weights)} ref_starts={len(member_ref_starts)}"
        )
    if n_members == 0:
        return reference_sequence

    cs_by_aid: dict[int, str] = {
        int(aid): str(cs)
        for aid, cs in zip(
            alignment_cs.column("alignment_id").to_pylist(),
            alignment_cs.column("cs_string").to_pylist(),
            strict=True,
        )
    }

    L = len(reference_sequence)
    window_start = int(reference_start)
    window_end = window_start + L

    # ── Singleton-consensus fast path ────────────────────────
    # With a single member there's no PWM voting — only one vote per
    # position. The per-position loop that argmax-derives the winner
    # from an (L, 5) tensor is pure overhead at this cardinality.
    # We can apply the member's cs:long edits directly to the
    # reference window (mismatches → query base, deletions → drop) in
    # a single numpy pass, which eats the L × .item() torch element-
    # access cost that dominated the slow path for tiny clusters.
    # The contract still respects min_depth: with a single member of
    # weight w, positions with no vote stay at the reference base when
    # w < min_depth (default 1.0 ⇒ no fallback for weight 1.0).
    if n_members == 1:
        aid = int(member_alignment_ids[0])
        weight = float(member_weights[0])
        cs = cs_by_aid.get(aid)
        ref_idx_np = np.frombuffer(
            reference_sequence.encode("ascii"), dtype=np.uint8
        ).copy()
        # Map ascii → base index via a 256-entry lookup table.
        ascii_to_idx = np.full(256, _AMB, dtype=np.int8)
        for ch, code in _BASE_TO_IDX.items():
            ascii_to_idx[ord(ch)] = code
        winner_np = ascii_to_idx[ref_idx_np].astype(np.int64)
        # Drop positions with ambiguous reference bases (matches the
        # old loop which `continue`d on _AMB before appending).
        keep_mask_np = winner_np != _AMB
        if cs is not None and cs:
            idx_list, base_list = _cs_votes(
                cs,
                alignment_ref_start=int(member_ref_starts[0]),
                window_start=window_start,
                window_end=window_end,
                reference_sequence=reference_sequence,
            )
            if idx_list and weight >= float(min_depth):
                idx_np = np.asarray(idx_list, dtype=np.int64)
                base_np = np.asarray(base_list, dtype=np.int64)
                # Overwrite the winner at each voted position.
                winner_np[idx_np] = base_np
                # Re-evaluate keep_mask: drop _GAP and _AMB winners.
                keep_mask_np = (winner_np != _GAP) & (winner_np != _AMB)
        # Vectorized base-char assembly.
        winner_kept = winner_np[keep_mask_np]
        out_bytes = _IDX_TO_BASE_NP[winner_kept]
        return out_bytes.tobytes().decode("ascii")

    pwm = torch.zeros((L, 5), dtype=torch.float64)
    flat_pwm = pwm.view(-1)

    # Per-member worker — closes over the per-call window_start /
    # window_end / reference_sequence so the parser doesn't re-read
    # them from outer scope. Returns the parallel (indices, bases,
    # weight) triple; main thread does the flat-tensor accumulation
    # so the actual torch scatter stays serial (cheap once batched).
    def _parse_one(triple: tuple[int, float, int]) -> tuple[list[int], list[int], float] | None:
        aid, w, aln_ref_start = triple
        cs = cs_by_aid.get(int(aid))
        if cs is None:
            return None
        idx, b = _cs_votes(
            cs,
            alignment_ref_start=int(aln_ref_start),
            window_start=window_start,
            window_end=window_end,
            reference_sequence=reference_sequence,
        )
        if not idx:
            return None
        return idx, b, float(w)

    triples = list(
        zip(
            member_alignment_ids,
            member_weights,
            member_ref_starts,
            strict=True,
        )
    )

    # Chunked batched scatter — replaces the prior 1-scatter-per-member
    # loop. Per-chunk Python list accumulation + 1 index_add_ per chunk
    # cuts torch kernel launch overhead from O(N_members) to
    # O(N_members / chunk_size). At chunk_size=100K that's ~22 scatter
    # calls for a 2.15M-member mega-cluster instead of 2.15M.
    executor: ThreadPoolExecutor | None = None
    if threads > 1 and n_members > 1:
        executor = ThreadPoolExecutor(
            max_workers=min(int(threads), n_members)
        )

    try:
        for chunk_start in range(0, n_members, scatter_chunk_size):
            chunk_end = min(chunk_start + scatter_chunk_size, n_members)
            chunk = triples[chunk_start:chunk_end]
            if executor is not None:
                results = list(executor.map(_parse_one, chunk))
            else:
                results = [_parse_one(t) for t in chunk]
            chunk_flat_idx: list[int] = []
            chunk_weights: list[float] = []
            for r in results:
                if r is None:
                    continue
                idx, b, w = r
                chunk_flat_idx.extend(
                    i * 5 + bb for i, bb in zip(idx, b, strict=True)
                )
                chunk_weights.extend([w] * len(idx))
            if chunk_flat_idx:
                idx_t = torch.tensor(chunk_flat_idx, dtype=torch.long)
                w_t = torch.tensor(chunk_weights, dtype=torch.float64)
                flat_pwm.index_add_(0, idx_t, w_t)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # Per-position winner. Positions with no votes ('row_max == 0') fall
    # back to the reference base (preserves window length when the
    # cluster doesn't cover a region). Vectorized to avoid the O(L)
    # `.item()` + Python list-append loop that dominated wall time for
    # long-gene windows — at L=1 Mbp the prior Python loop was ~1 sec
    # per cluster vs ~ms here.
    row_max, best = pwm.max(dim=1)
    row_max_np = row_max.numpy()
    best_np = best.numpy().astype(np.int64, copy=False)
    # Reference base codes per position via a 256-entry ASCII lookup.
    ref_idx_np = np.frombuffer(
        reference_sequence.encode("ascii"), dtype=np.uint8
    ).copy()
    ascii_to_idx = np.full(256, _AMB, dtype=np.int8)
    for ch, code in _BASE_TO_IDX.items():
        ascii_to_idx[ord(ch)] = code
    ref_base_idx_np = ascii_to_idx[ref_idx_np].astype(np.int64)
    # Winner = ref base when row_max < min_depth, else PWM argmax.
    below_depth = row_max_np < float(min_depth)
    winner_np = np.where(below_depth, ref_base_idx_np, best_np)
    # Drop positions with _GAP (consensus omits gap-winning columns)
    # or _AMB (ambiguous reference base on a fallback position).
    keep_mask_np = (winner_np != _GAP) & (winner_np != _AMB)
    out_bytes = _IDX_TO_BASE_NP[winner_np[keep_mask_np]]
    return out_bytes.tobytes().decode("ascii")


__all__ = [
    "build_consensus",
    "ALIGNMENT_CS_TABLE",
]
