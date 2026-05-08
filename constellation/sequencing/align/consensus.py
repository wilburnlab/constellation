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

    pwm = torch.zeros((L, 5), dtype=torch.float64)
    flat_pwm = pwm.view(-1)

    for aid, weight, aln_ref_start in zip(
        member_alignment_ids, member_weights, member_ref_starts, strict=True
    ):
        cs = cs_by_aid.get(int(aid))
        if cs is None:
            continue
        indices, bases = _cs_votes(
            cs,
            alignment_ref_start=int(aln_ref_start),
            window_start=window_start,
            window_end=window_end,
            reference_sequence=reference_sequence,
        )
        if not indices:
            continue
        flat_idx = torch.tensor(
            [i * 5 + b for i, b in zip(indices, bases, strict=True)],
            dtype=torch.long,
        )
        weights = torch.full(
            (len(flat_idx),), float(weight), dtype=torch.float64
        )
        flat_pwm.index_add_(0, flat_idx, weights)

    # Per-position winner. Positions with no votes ('row_max == 0') fall
    # back to the reference base (preserves window length when the
    # cluster doesn't cover a region).
    row_max, best = pwm.max(dim=1)
    consensus_chars: list[str] = []
    for p in range(L):
        if float(row_max[p].item()) < float(min_depth):
            ref_base = reference_sequence[p]
            ref_base_idx = _BASE_TO_IDX.get(ref_base, _AMB)
            if ref_base_idx == _AMB:
                continue
            consensus_chars.append(_IDX_TO_BASE[ref_base_idx])
            continue
        winner = int(best[p].item())
        if winner == _GAP:
            continue
        consensus_chars.append(_IDX_TO_BASE[winner])
    return "".join(consensus_chars)


__all__ = [
    "build_consensus",
    "ALIGNMENT_CS_TABLE",
]
