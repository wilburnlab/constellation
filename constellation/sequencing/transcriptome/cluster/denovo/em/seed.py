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
    # Best `dorado_quality` among the reads sharing this exact cDNA, or -1
    # where the demux dir predates the qs:f tag. Reads sharing an exact cDNA
    # are byte-identical, so the best of them is the right representative and
    # the choice costs the template sequence nothing.
    quality: np.ndarray  # float64


#: Default floor for the quality-aware policy. The knee of the measured
#: decile curve: median error by `dorado_quality` decile runs 0.02321,
#: 0.01152, 0.00796, 0.00607, then flattens — so everything above the third
#: decile is nearly equivalent and there is nothing to buy above ~Q22.
SEED_QUALITY_FLOOR = 22.0

#: Ranks are int64 lexsort keys. Quality-failing rows are offset past every
#: possible `-template_length` so they sort strictly after every clearing row.
_RANK_FAIL_BASE = 1 << 40


RepresentativePolicy = Callable[[RepCandidates], np.ndarray]


def _rank_median_length(c: RepCandidates) -> np.ndarray:
    return np.abs(c.template_length - c.median_length)


def _rank_longest_template(c: RepCandidates) -> np.ndarray:
    return -c.template_length


def _rank_most_5p_flank(c: RepCandidates) -> np.ndarray:
    return -c.orf_start


def _rank_most_replicated(c: RepCandidates) -> np.ndarray:
    return -c.abundance


def _rank_longest_above_quality(c: RepCandidates) -> np.ndarray:
    """Longest cDNA clearing the quality floor; else the highest-quality one.

    The trade this appears to require does not exist. Over 8.3M reads,
    Spearman(length, error) is **+0.015 to +0.049** in every length stratum —
    length and accuracy are independent — so constraining one and optimising
    the other is nearly free. As selectors of read accuracy they are not
    comparable at all: the top decile by ``dorado_quality`` has median error
    0.00114 (0.25x), by ``mean_quality`` 0.00166, and by ``length`` 0.00501,
    which is **1.10x** — slightly worse than picking at random.

    So quality is the floor and length is the objective, in that order,
    because quality's benefit saturates (the decile curve flattens after the
    third) while length's cost does not: a 500 nt seed for a 3 kb transcript
    can never represent it, and since the covariance M-step made extent
    representable a truncated seed emits a truncated ORF rather than being
    silently extended.

    Honest limit: even the top quality decile has *mean* error 0.0068 against
    *median* 0.00114, so a floor makes bad seeds rarer, not absent. A
    single-read seed is one draw from that tail.
    """
    clears = c.quality >= SEED_QUALITY_FLOOR
    # Failing rows rank by quality descending, after every clearing row — so
    # "fall back to the group's best read" needs no group-level branch.
    fail = _RANK_FAIL_BASE - np.rint(c.quality * 1000.0).astype(np.int64)
    return np.where(clears, -c.template_length, fail)


REPRESENTATIVE_POLICIES: dict[str, RepresentativePolicy] = {
    # Default. The longest cDNA is the most likely chimera / concatemer /
    # internal-priming artifact, so electing it makes the template a
    # low-support outlier; the central length is the robust estimate of the
    # real extent. Safe ONLY because the consensus kernel can now extend past
    # the frame's ends — before that a too-short template was an
    # unrecoverable ceiling, which is why "longest" would have been the
    # defensive choice. Expect this to be swept.
    "median-length": _rank_median_length,
    # The EM path's default. See :func:`_rank_longest_above_quality`.
    "longest-above-quality": _rank_longest_above_quality,
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
    uniq_quality, uniq_best_read = _best_quality_per_uniq(reads, read_map, n_uniq)

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
    tmpl_len = (
        uniq.column("seq_len").to_numpy(zero_copy_only=False)[row].astype(np.int64)
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
        quality=uniq_quality[row[order]],
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
    # Name the best-quality read of the elected cDNA rather than whichever
    # row dereplication happened to keep. The sequence is identical either
    # way — reads share an exact cDNA — so this costs the template nothing
    # and gives the round-1 ranker an accurate seed quality to break ties on.
    rep_read_id = pc.take(uniq.column("representative_read_id"), pa.array(rep_uniq))
    best_ids = uniq_best_read[rep_uniq]
    if best_ids is not None:
        rep_read_id = pc.if_else(
            pa.array(best_ids >= 0),
            pc.take(reads.column("read_id"), pa.array(np.maximum(best_ids, 0))),
            rep_read_id,
        )
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


def _best_quality_per_uniq(
    reads: pa.Table, read_map: pa.Table, n_uniq: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per distinct cDNA: the best ``dorado_quality``, and which read has it.

    Returns ``(quality, read_row)`` with ``-1`` where the column is absent or
    all-null, which makes the quality-aware policy degrade to "highest of
    nothing" — every row fails the floor, so the group falls through to its
    existing abundance / length tie-breaks rather than erroring.
    """
    quality = np.full(n_uniq, -1.0, dtype=np.float64)
    best_row = np.full(n_uniq, -1, dtype=np.int64)
    if "dorado_quality" not in reads.schema.names or read_map.num_rows == 0:
        return quality, best_row

    q_col = reads.column("dorado_quality")
    if q_col.null_count == reads.num_rows:
        return quality, best_row

    # read_map is (read_id, uniq_id, ...); recover each read's ROW in `reads`
    # by position rather than by a string join — dereplicate preserves order,
    # so `pc.index_in` over read_id is the only string touch and it happens
    # once per read rather than once per hit.
    row_of_read = pc.index_in(
        read_map.column("read_id"), value_set=reads.column("read_id").combine_chunks()
    ).to_numpy(zero_copy_only=False)
    uid = read_map.column("uniq_id").to_numpy(zero_copy_only=False)
    q = pc.fill_null(q_col, -1.0).to_numpy(zero_copy_only=False).astype(np.float64)
    valid = ~np.isnan(row_of_read.astype(np.float64))
    if not valid.any():
        return quality, best_row
    rows = row_of_read[valid].astype(np.int64)
    uids = uid[valid].astype(np.int64)
    qv = q[rows]

    # Highest quality first, so the first occurrence of each uniq_id is its best.
    order = np.lexsort((-qv, uids))
    su, sq, sr = uids[order], qv[order], rows[order]
    first = np.flatnonzero(np.concatenate([[True], su[1:] != su[:-1]]))
    quality[su[first]] = sq[first]
    best_row[su[first]] = sr[first]
    return quality, best_row


__all__ = [
    "SEED_ORF_TABLE",
    "SEED_QUALITY_FLOOR",
    "READ_ORF_MAP_SCHEMA",
    "RepCandidates",
    "REPRESENTATIVE_POLICIES",
    "extract_seed_orfs",
]
