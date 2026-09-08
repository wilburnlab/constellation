"""Stage 1 — seeding: one template per distinct sense ORF.

Every read contributes its longest sense ORF (≥30 aa). Those ORF nucleotide
strings are dereplicated exactly, and each distinct ORF elects one read whose
**full cDNA** becomes the template — the frame the M-step's PWM is built on
and the sequence the E-step aligns against.

Two things about this that are easy to get backwards:

* **Replication is a certificate of error-free sequence, not of abundance.**
  The exact-copy count of an ORF scales as exp(−εL), so for a long protein the
  most-replicated exact ORF is usually a *fragment*. Abundance decides which
  ORF anchors a group; it must not decide between proteoforms of different
  length (that is the fold stage's ±3-codon rule).
* **No replication filter.** Akap4's 849-aa proteoform was error-free in 4
  reads and never twice identical, so any "seen ≥2×" gate erases it. Every
  read's ORF is a seed.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import register_schema
from constellation.sequencing.transcriptome.cluster.denovo.dereplicate import (
    _hash_sequences,
    dereplicate,
)
from constellation.sequencing.transcriptome.cluster.denovo.orf import best_sense_orf


# One row per distinct ORF nucleotide sequence.
SEED_ORF_TABLE: pa.Schema = pa.schema(
    [
        pa.field("orf_id", pa.int64(), nullable=False),
        pa.field("orf_nucleotide", pa.large_string(), nullable=False),
        pa.field("orf_aa_length", pa.int32(), nullable=False),
        # Reads whose longest sense ORF is byte-identical to this one.
        pa.field("n_reads", pa.int64(), nullable=False),
        # The elected representative and its full cDNA — the template.
        pa.field("representative_read_id", pa.string(), nullable=False),
        pa.field("template_sequence", pa.large_string(), nullable=False),
        pa.field("orf_start_in_template", pa.int32(), nullable=False),
        pa.field("orf_end_in_template", pa.int32(), nullable=False),
        pa.field("template_length", pa.int32(), nullable=False),
        # Distinct cDNAs carrying this exact ORF — a diagnostic on how much
        # UTR variation the ORF key is collapsing.
        pa.field("n_distinct_templates", pa.int32(), nullable=False),
    ],
    metadata={b"schema_name": b"SeedOrfTable"},
)

READ_ORF_MAP_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("orf_id", pa.int64(), nullable=False),
        pa.field("sample_id", pa.int64(), nullable=True),
    ]
)

register_schema("SeedOrfTable", SEED_ORF_TABLE)


# ── representative-read policies ──────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RepCandidates:
    """The distinct cDNAs carrying one exact ORF, as parallel arrays."""

    template_length: np.ndarray  # int64
    orf_start: np.ndarray  # int64 — 5' flank length
    abundance: np.ndarray  # int64 — reads on each distinct cDNA
    # Abundance-weighted median template_length of the row's own group,
    # broadcast per row so a policy is a pure elementwise function.
    median_length: np.ndarray  # int64


RepresentativePolicy = Callable[[RepCandidates], np.ndarray]


def _rank_median_length(c: RepCandidates) -> np.ndarray:
    return np.abs(c.template_length - c.median_length)


def _rank_longest_template(c: RepCandidates) -> np.ndarray:
    return -c.template_length


def _rank_most_5p_flank(c: RepCandidates) -> np.ndarray:
    return -c.orf_start


def _rank_most_replicated(c: RepCandidates) -> np.ndarray:
    return -c.abundance


REPRESENTATIVE_POLICIES: dict[str, RepresentativePolicy] = {
    # Default. The longest cDNA is the most likely chimera / concatemer /
    # internal-priming artifact, so electing it makes the template a
    # low-support outlier; the central length is the robust estimate of the
    # real extent. Safe ONLY because the consensus kernel can now extend past
    # the frame's ends — before that a too-short template was an
    # unrecoverable ceiling, which is why "longest" would have been the
    # defensive choice. Expect this to be swept.
    "median-length": _rank_median_length,
    "longest-template": _rank_longest_template,
    # Maximises the 5' flank the M-step gets to extend an ORF into.
    "most-5p-flank": _rank_most_5p_flank,
    # Most-replicated exact cDNA; ties fall through to length.
    "most-replicated": _rank_most_replicated,
}


# ── ORF prediction over the unique cDNAs (fork pool) ──────────────────

# Set in the parent before the pool forks; workers read it copy-on-write.
_SEED_SEQS: list[str] | None = None


def _orf_chunk(lo: int, hi: int, min_aa_length: int) -> list[tuple]:
    seqs = _SEED_SEQS
    assert seqs is not None
    out: list[tuple] = []
    for i in range(lo, hi):
        hit = best_sense_orf(seqs[i], min_aa_length=min_aa_length)
        if hit is not None:
            _prot, st, en = hit
            out.append((i, st, en))
    return out


def _predict_orfs(
    seqs: list[str], *, min_aa_length: int, threads: int, chunk: int = 20_000
) -> list[tuple]:
    """``[(uniq_index, orf_start, orf_end), …]`` for the cDNAs that have one."""
    global _SEED_SEQS
    n = len(seqs)
    if n == 0:
        return []
    _SEED_SEQS = seqs
    try:
        if threads <= 1 or n < chunk:
            return _orf_chunk(0, n, min_aa_length)
        bounds = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
        ctx = mp.get_context("fork")
        out: list[tuple] = []
        with ProcessPoolExecutor(max_workers=threads, mp_context=ctx) as ex:
            futs = [ex.submit(_orf_chunk, lo, hi, min_aa_length) for lo, hi in bounds]
            for fut in futs:
                out.extend(fut.result())
        return out
    finally:
        _SEED_SEQS = None


# ── group statistics ──────────────────────────────────────────────────


def _group_bounds(keys: np.ndarray) -> np.ndarray:
    """Start offset of each run in a sorted key array, plus the end sentinel."""
    if keys.shape[0] == 0:
        return np.zeros(1, dtype=np.int64)
    change = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    return np.concatenate(
        [np.zeros(1, dtype=np.int64), change, np.array([keys.shape[0]], np.int64)]
    )


def _weighted_median_per_group(
    starts: np.ndarray, values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Abundance-weighted median of ``values`` per group.

    ``values`` must already be sorted ascending *within* each group. Uses a
    global cumulative-weight scan plus one searchsorted per group boundary —
    no per-group Python loop.
    """
    n_groups = starts.shape[0] - 1
    if n_groups <= 0:
        return np.empty(0, dtype=np.int64)
    cw = np.cumsum(weights)
    lo, hi = starts[:-1], starts[1:]
    base = np.where(lo > 0, cw[np.maximum(lo - 1, 0)], 0.0)
    total = cw[hi - 1] - base
    target = base + total / 2.0
    # side='left' lands on the first row whose cumulative weight reaches half
    # the group's mass — the weighted median row.
    idx = np.searchsorted(cw, target, side="left")
    idx = np.clip(idx, lo, hi - 1)
    return values[idx]


def extract_seed_orfs(
    reads: pa.Table,
    *,
    min_aa_length: int = 30,
    representative: str | RepresentativePolicy = "median-length",
    threads: int = 1,
    progress: Callable[[str], None] | None = None,
) -> tuple[pa.Table, pa.Table]:
    """Seed one template per distinct sense ORF.

    ``reads`` carries ``(read_id, sequence, sample_id)`` — the trimmed
    transcript windows. Returns ``(SEED_ORF_TABLE, READ_ORF_MAP_SCHEMA)``;
    reads with no qualifying ORF are simply absent from the map (they are not
    lost — the E-step still assigns them to whichever template wins).
    """
    log = progress or (lambda _m: None)
    policy: RepresentativePolicy = (
        REPRESENTATIVE_POLICIES[representative]
        if isinstance(representative, str)
        else representative
    )

    # Predict once per distinct cDNA, not once per read: at ~1% error most
    # reads are distinct, but the collapse is free and never wrong.
    log(f"dereplicating {reads.num_rows:,} reads for ORF seeding…")
    uniq, read_map = dereplicate(reads)
    n_uniq = uniq.num_rows
    if n_uniq == 0:
        return SEED_ORF_TABLE.empty_table(), READ_ORF_MAP_SCHEMA.empty_table()

    seqs = uniq.column("sequence").to_pylist()
    log(f"predicting sense ORFs (≥{min_aa_length} aa) over {n_uniq:,} unique cDNAs…")
    hits = _predict_orfs(seqs, min_aa_length=min_aa_length, threads=threads)
    if not hits:
        log("no read carries a qualifying ORF")
        return SEED_ORF_TABLE.empty_table(), READ_ORF_MAP_SCHEMA.empty_table()

    row = np.array([h[0] for h in hits], dtype=np.int64)
    orf_start = np.array([h[1] for h in hits], dtype=np.int64)
    orf_end = np.array([h[2] for h in hits], dtype=np.int64)
    abundance = uniq.column("abundance").to_numpy(zero_copy_only=False)[row]
    tmpl_len = uniq.column("seq_len").to_numpy(zero_copy_only=False)[row].astype(
        np.int64
    )
    orf_nt = pa.array(
        [seqs[int(i)][int(a) : int(b)] for i, a, b in zip(row, orf_start, orf_end)],
        type=pa.large_string(),
    )

    # Group on a fixed-width hash rather than the ORF string itself, for the
    # same reason dereplicate does: Acero's variable-length distinct-key
    # storage overflows its int32 offset once the keys exceed ~2 GB.
    orf_hash = _hash_sequences(orf_nt).to_numpy(zero_copy_only=False)
    log(f"{len(hits):,} unique cDNAs carry an ORF; grouping by exact ORF…")

    # Pass A: sort by (orf, template_length) for the group statistics.
    order = np.lexsort((tmpl_len, orf_hash))
    g_hash = orf_hash[order]
    starts = _group_bounds(g_hash)
    lo, hi = starts[:-1], starts[1:]
    g_abund = abundance[order].astype(np.float64)
    median_len = _weighted_median_per_group(starts, tmpl_len[order], g_abund)
    cw = np.cumsum(g_abund)
    base = np.where(lo > 0, cw[np.maximum(lo - 1, 0)], 0.0)
    n_reads = (cw[hi - 1] - base).astype(np.int64)
    n_templates = (hi - lo).astype(np.int32)

    # Pass B: rank inside each group by the policy, then take the first row.
    group_of_row = np.repeat(np.arange(starts.shape[0] - 1), hi - lo)
    cand = RepCandidates(
        template_length=tmpl_len[order],
        orf_start=orf_start[order],
        abundance=abundance[order].astype(np.int64),
        median_length=median_len[group_of_row],
    )
    rank = np.asarray(policy(cand), dtype=np.int64)
    # Deterministic tie-breaks: more reads, then longer cDNA, then row order.
    pick_order = np.lexsort(
        (
            np.arange(rank.shape[0]),
            -cand.template_length,
            -cand.abundance,
            rank,
            group_of_row,
        )
    )
    # pick_order is sorted by group first, so the first occurrence of each
    # group id in it is that group's elected row.
    n_groups = starts.shape[0] - 1
    first_of_group = pick_order[
        np.searchsorted(group_of_row[pick_order], np.arange(n_groups))
    ]
    rep_row = order[first_of_group]

    rep_uniq = row[rep_row]
    rep_read_id = pc.take(uniq.column("representative_read_id"), pa.array(rep_uniq))
    n_orfs = n_groups
    seed = pa.table(
        {
            "orf_id": pa.array(np.arange(n_orfs, dtype=np.int64)),
            "orf_nucleotide": pc.take(orf_nt, pa.array(rep_row)),
            "orf_aa_length": pa.array(
                ((orf_end[rep_row] - orf_start[rep_row]) // 3 - 1).astype(np.int32)
            ),
            "n_reads": pa.array(n_reads),
            "representative_read_id": rep_read_id,
            "template_sequence": pc.take(
                uniq.column("sequence"), pa.array(rep_uniq)
            ).cast(pa.large_string()),
            "orf_start_in_template": pa.array(orf_start[rep_row].astype(np.int32)),
            "orf_end_in_template": pa.array(orf_end[rep_row].astype(np.int32)),
            "template_length": pa.array(tmpl_len[rep_row].astype(np.int32)),
            "n_distinct_templates": pa.array(n_templates),
        },
        schema=SEED_ORF_TABLE,
    )

    # read → orf_id, via the uniq_id each read dereplicated to.
    uniq_to_orf = pa.table(
        {
            "uniq_id": pa.array(row[order], type=pa.int64()),
            "orf_id": pa.array(group_of_row.astype(np.int64)),
        }
    )
    mapped = read_map.join(uniq_to_orf, keys="uniq_id", join_type="inner")
    read_orf = mapped.select(["read_id", "orf_id", "sample_id"]).cast(
        READ_ORF_MAP_SCHEMA
    )
    log(f"seeded {n_orfs:,} distinct ORFs over {read_orf.num_rows:,} reads")
    return seed, read_orf


__all__ = [
    "SEED_ORF_TABLE",
    "READ_ORF_MAP_SCHEMA",
    "RepCandidates",
    "REPRESENTATIVE_POLICIES",
    "extract_seed_orfs",
]
