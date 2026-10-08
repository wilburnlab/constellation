"""Native E-step candidates: Constellation's own minimizer join.

Why this exists. minimap2 masks high-frequency minimizers as repeats.
In this pipeline a high-frequency minimizer is usually NOT a repeat: it is
a k-mer shared by many templates because the templates are near-duplicates
of one another — similar templates the loop has not consolidated yet, which
is precisely what the E-step must see to consolidate them. Masking those
minimizers blinds the chainer to whole families at once, and the loss is
invisible from the outside: the reads simply come back unassigned, or
assigned to whatever survived the mask. ``--estep-aligner native`` swaps
ONLY the candidate source. The alignment (:func:`~.realign.align_finalist`),
the admission floor, the lazy round-1 walk and the likelihood ranking are
the same code the ``edlib`` path runs — :func:`~.assign.assign_block_edlib`
is called whole, with a synthesized block in place of a scanned one.

The pieces, and where each runs:

``write_read_minimizers``   every corpus read's UNCAPPED sketch, computed
                            once per run in the parent (torch) and stored
                            with the corpus — ~2 billion entries / >30 GB
                            at 9.4M reads is not a resident structure.
                            Workers memory-map it and slice their block.
``TemplateMinimizerIndex``  that round's templates, sketched in the parent
                            (torch) and handed to the fork pool as
                            copy-on-write numpy module state.
``candidates_block``        the bipartite probe join, numpy only: the same
                            windowed-diagonal candidate test as the
                            containment join, ANTISENSE RULE INCLUDED.
``run_native_estep``        the driver: fixed read-row blocks, one shard
                            per block, submission-order collection — the
                            output is independent of the worker count.

*Sense-strand candidates only.* The minimizer hash is canonical, so a read
shares every minimizer with the reverse complement of a template, along an
anti-diagonal. Such pairs fail at alignment anyway, but unrejected they
take shortlist slots from real candidates. The join applies the containment
join's two-clause test: a window is antisense when its hits fit the
anti-diagonal better than the diagonal AND fewer than ``min_shared`` probes
agree on one diagonal (the second clause is what keeps a fragment of a
tandem repeat, whose window spreads along the diagonal axis by one unit per
copy).

*Error k-mers are not a recall problem but ARE a budget problem.* At k=15 a
1.3 Gb template set nearly saturates the k-mer space, so a read's error
k-mers mostly EXIST in the index — in small, unrelated buckets. Probes are
therefore chosen per position stratum by smallest HASH (random with respect
to bucket size), never by smallest bucket: preferring cheap buckets would
spend the probes on exactly those collisions. The per-read row budget then
keeps whatever prefix of the strata fits, never fewer than two probes
(``min_shared`` needs two), and flags the read when it binds.

Workers are torch-free: this module imports torch only inside the two
parent-side builders, and everything below the fork is numpy, pyarrow and
edlib — the same discipline as the graph builder, for the same reason (a
torch op in a forked child deadlocks on OpenMP; the failure is a silent
hang).
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

MINIS_ARROW = "minimizers.arrow"
MINIS_OFFSETS = "minimizers_offsets.arrow"
MINIS_META = "minimizers.json"

#: Entries per values record batch — also the write-side transient bound.
_MINIS_BATCH_ROWS = 1_000_000


def _group_starts(sorted_keys: np.ndarray) -> np.ndarray:
    """First index of each run of equal values in a sorted, non-empty array.

    A copy of ``candidates._group_starts`` rather than an import: that
    module imports the (torch) sketch at its top level, and this one must
    stay importable with no torch anywhere — the AST test below the fork
    holds it to that.
    """
    first = np.empty(sorted_keys.shape[0], dtype=bool)
    first[0] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=first[1:])
    return np.flatnonzero(first)


@dataclass(frozen=True, slots=True)
class NativeParams:
    """The join's knobs. ``kmer`` / ``window`` come from the run's sketch
    parameters (:class:`~.rounds.EmParams`); everything here is stamped into
    ``estep.json`` so a resume under different values is refused."""

    #: One probe per position stratum, smallest hash per stratum. 32 (up
    #: from 16, 2026-10-05): the probe count is also the resolution of the
    #: shortlist key — `n_shared` cannot exceed it — and at 16 a family's
    #: candidates tied so densely that the 16-deep shortlist cut became an
    #: arbitrary subset. CLI: `--estep-probes-per-read`.
    probes_per_read: int = 32
    #: A template bucket larger than this joins through its
    #: ``overflow_anchors`` best-supported entries instead of whole.
    bucket_cap: int = 20_480
    overflow_anchors: int = 32
    #: Probes agreeing within one ``diag_band`` window a candidate needs.
    min_shared: int = 2
    diag_band: int = 64
    #: Candidates kept per read, by (shared probes, template support).
    max_candidates: int = 1_024
    #: Join rows one read may expand; binding it flags the read.
    max_rows_per_read: int = 32_768
    #: Reads per worker block. Execution-only: never in the stamp.
    block_reads: int = 65_536

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            least = 0 if f.name == "diag_band" else 2 if f.name == "bucket_cap" else 1
            if isinstance(v, bool) or not isinstance(v, int) or v < least:
                raise ValueError(f"{f.name} must be an integer >= {least}, got {v!r}")

    def semantic(self) -> dict:
        out = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "block_reads"
        }
        return out


# ──────────────────────────────────────────────────────────────────────
# Read minimizers: once per run, on disk, sorted by read row
# ──────────────────────────────────────────────────────────────────────

_VALUES_SCHEMA = pa.schema(
    [
        pa.field("mini_hash", pa.int64(), nullable=False),
        pa.field("pos", pa.int32(), nullable=False),
    ]
)
_OFFSETS_SCHEMA = pa.schema([pa.field("offset", pa.int64(), nullable=False)])


def write_read_minimizers(
    corpus_arrow: Path | str,
    out_dir: Path | str,
    *,
    kmer: int,
    window: int,
    progress=None,
) -> Path:
    """Sketch every corpus read, uncapped, into ``out_dir`` — once per run.

    Parent only (torch). Reuses a store whose stamp matches; a stamp that
    disagrees on ``kmer`` / ``window`` / the corpus is rebuilt, which is
    safe because the store is derived data and the build is deterministic.
    Values are written per corpus record batch, re-sorted to ``(read row,
    position)``, so the write-side transient is one batch.

    "The corpus" is a digest of its rows (:func:`~.corpus.corpus_digest`),
    not its row count. The store is keyed on the read ROW, and a corpus
    rebuilt at the same path holds the same number of reads in another
    order — a store reused across that hands every read another read's
    minimizers, and nothing fails: the candidates are simply wrong and the
    reads come back unassigned (review of 91e7c69). A stamp from before the
    digest was recorded names no corpus and is rebuilt.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
        corpus_digest,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
        extract_minimizers,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda _m: None)
    values_path = out_dir / MINIS_ARROW
    offsets_path = out_dir / MINIS_OFFSETS
    meta_path = out_dir / MINIS_META

    t_digest = time.time()
    digest = corpus_digest(corpus_arrow)
    t_digest = time.time() - t_digest
    with pa.memory_map(str(corpus_arrow), "r") as mm:
        reader = pa.ipc.open_file(mm)
        n_reads = sum(
            reader.get_batch(b).num_rows for b in range(reader.num_record_batches)
        )
        want = {
            "kmer": int(kmer),
            "window": int(window),
            "n_reads": int(n_reads),
            "corpus_digest": digest,
        }
        if meta_path.exists() and values_path.exists() and offsets_path.exists():
            try:
                have = json.loads(meta_path.read_text())
            except ValueError:
                have = {}
            if {k: have.get(k) for k in want} == want:
                return out_dir
            stale = [k for k in want if have.get(k) != want[k]]
            log(
                f"read minimizers: the store in {out_dir} was built for "
                f"another {' / '.join(stale)}; rebuilding"
            )
        # The stamp goes first: a build that dies half way must not leave
        # new values under a stamp that vouches for the old ones.
        meta_path.unlink(missing_ok=True)

        t0 = time.time()
        offsets = [0]
        n_entries = 0
        tmp_values = values_path.with_suffix(".arrow.tmp")
        with (
            pa.OSFile(str(tmp_values), "wb") as sink,
            pa.ipc.new_file(sink, _VALUES_SCHEMA) as writer,
        ):
            for b in range(reader.num_record_batches):
                batch = reader.get_batch(b)
                seq = batch.column(batch.schema.get_field_index("sequence"))
                index = extract_minimizers(
                    seq, k=int(kmer), w=int(window), max_per_seq=None
                )
                h = index.mini_hash.numpy()
                u = index.uniq_id.numpy().astype(np.int64)
                p = index.pos.numpy()
                order = np.lexsort((p, u))
                counts = np.bincount(u, minlength=batch.num_rows)
                base = offsets[-1]
                offsets.extend((base + np.cumsum(counts)).tolist())
                h, p = h[order], p[order]
                n_entries += int(h.shape[0])
                for lo in range(0, h.shape[0], _MINIS_BATCH_ROWS):
                    writer.write_batch(
                        pa.record_batch(
                            [
                                pa.array(h[lo : lo + _MINIS_BATCH_ROWS]),
                                pa.array(p[lo : lo + _MINIS_BATCH_ROWS]),
                            ],
                            schema=_VALUES_SCHEMA,
                        )
                    )

    tmp_offsets = offsets_path.with_suffix(".arrow.tmp")
    with (
        pa.OSFile(str(tmp_offsets), "wb") as sink,
        pa.ipc.new_file(sink, _OFFSETS_SCHEMA) as writer,
    ):
        writer.write_batch(
            pa.record_batch(
                [pa.array(np.asarray(offsets, dtype=np.int64))],
                schema=_OFFSETS_SCHEMA,
            )
        )
    tmp_values.replace(values_path)
    tmp_offsets.replace(offsets_path)
    meta_path.write_text(json.dumps({**want, "n_entries": n_entries}, indent=2))
    log(
        f"read minimizers: {n_entries:,} entries over {n_reads:,} reads "
        f"(k{kmer}/w{window}, uncapped) in {time.time() - t0:.1f}s "
        f"(corpus digest {t_digest:.1f}s)"
    )
    return out_dir


@dataclass(frozen=True, slots=True)
class ReadMinimizerStore:
    """Memory-mapped per-read minimizers; ``block`` touches only its rows."""

    values: pa.Table
    offsets: np.ndarray  # (n_reads + 1,)
    chunk_starts: np.ndarray  # value-row index at which each chunk starts
    kmer: int
    window: int
    _mm: tuple

    @classmethod
    def open(cls, directory: Path | str) -> ReadMinimizerStore:
        directory = Path(directory)
        meta = json.loads((directory / MINIS_META).read_text())
        mm_v = pa.memory_map(str(directory / MINIS_ARROW), "r")
        values = pa.ipc.open_file(mm_v).read_all()
        mm_o = pa.memory_map(str(directory / MINIS_OFFSETS), "r")
        offsets = (
            pa.ipc.open_file(mm_o).read_all().column("offset").to_numpy()
        ).astype(np.int64)
        lengths = [len(c) for c in values.column(0).chunks]
        starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        return cls(
            values=values,
            offsets=offsets,
            chunk_starts=starts,
            kmer=int(meta["kmer"]),
            window=int(meta["window"]),
            _mm=(mm_v, mm_o),
        )

    @property
    def n_reads(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def block(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(hash, pos, local offsets)`` for reads ``[lo, hi)``.

        One CONTIGUOUS value range, gathered chunk by chunk — each chunk is
        sliced (zero-copy from the map) and only the block's own rows are
        ever copied, so residency is the block, not the store.
        """
        v0, v1 = int(self.offsets[lo]), int(self.offsets[hi])
        local = self.offsets[lo : hi + 1] - v0
        if v1 == v0:
            empty = np.empty(0, dtype=np.int64)
            return empty, np.empty(0, dtype=np.int32), local
        h_parts: list[np.ndarray] = []
        p_parts: list[np.ndarray] = []
        c0 = int(np.searchsorted(self.chunk_starts, v0, side="right")) - 1
        at = v0
        while at < v1:
            c = c0 + len(h_parts)
            lo_c = at - int(self.chunk_starts[c])
            take = min(v1, int(self.chunk_starts[c + 1])) - at
            h_parts.append(
                self.values.column("mini_hash").chunk(c).slice(lo_c, take).to_numpy()
            )
            p_parts.append(
                self.values.column("pos").chunk(c).slice(lo_c, take).to_numpy()
            )
            at += take
        h = h_parts[0] if len(h_parts) == 1 else np.concatenate(h_parts)
        p = p_parts[0] if len(p_parts) == 1 else np.concatenate(p_parts)
        return np.ascontiguousarray(h), np.ascontiguousarray(p), local

    def close(self) -> None:
        for mm in self._mm:
            mm.close()


# ──────────────────────────────────────────────────────────────────────
# Template index: per round, parent-built, fork-shared
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TemplateMinimizerIndex:
    """The round's template minimizers, bucketed by hash. Plain numpy, so a
    forked worker reads it copy-on-write."""

    bucket_hash: np.ndarray  # (B,) int64 ascending — distinct hashes
    bucket_lo: np.ndarray  # (B,) int64 — entry range per bucket
    bucket_hi: np.ndarray
    #: 0 = joined whole, 1 = over the cap (anchor join).
    bucket_class: np.ndarray  # (B,) int8
    entry_row: np.ndarray  # (E,) int64 — template row per entry
    entry_pos: np.ndarray  # (E,) int64
    #: Anchor ranges for class-1 buckets; (0, 0) elsewhere.
    anchor_lo: np.ndarray
    anchor_hi: np.ndarray
    anchor_row: np.ndarray  # entries of the anchor pool, best support first
    anchor_pos: np.ndarray
    t_len: np.ndarray  # (n_templates,) int64
    t_support: np.ndarray  # (n_templates,) float64 — node_weight

    @classmethod
    def build(
        cls, store, *, kmer: int, window: int, params: NativeParams
    ) -> TemplateMinimizerIndex:
        """Parent only (torch): sketch ``store``'s templates, uncapped."""
        from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
            extract_minimizers,
        )

        index = extract_minimizers(
            store.table.column("sequence"),
            k=int(kmer),
            w=int(window),
            max_per_seq=None,
        )
        h = index.mini_hash.numpy()
        row = index.uniq_id.numpy().astype(np.int64)
        pos = index.pos.numpy().astype(np.int64)
        t_len = store.lengths().astype(np.int64)
        support = np.asarray(store.node_weight, dtype=np.float64)
        if h.shape[0] == 0:
            z = np.empty(0, dtype=np.int64)
            return cls(
                bucket_hash=z,
                bucket_lo=z,
                bucket_hi=z,
                bucket_class=np.empty(0, dtype=np.int8),
                entry_row=z,
                entry_pos=z,
                anchor_lo=z,
                anchor_hi=z,
                anchor_row=z,
                anchor_pos=z,
                t_len=t_len,
                t_support=support,
            )
        lo = _group_starts(h)
        hi = np.append(lo[1:], h.shape[0])
        size = hi - lo
        klass = (size > int(params.bucket_cap)).astype(np.int8)

        # The anchor pool of each oversized bucket: its `overflow_anchors`
        # best-supported entries (ties to the lower row, then position).
        anchor_lo = np.zeros(lo.shape[0], dtype=np.int64)
        anchor_hi = np.zeros(lo.shape[0], dtype=np.int64)
        a_rows: list[np.ndarray] = []
        a_poss: list[np.ndarray] = []
        at = 0
        for b in np.flatnonzero(klass == 1).tolist():
            sl = slice(int(lo[b]), int(hi[b]))
            order = np.lexsort((pos[sl], row[sl], -support[row[sl]]))
            take = order[: int(params.overflow_anchors)]
            a_rows.append(row[sl][take])
            a_poss.append(pos[sl][take])
            anchor_lo[b] = at
            at += take.shape[0]
            anchor_hi[b] = at
        return cls(
            bucket_hash=h[lo],
            bucket_lo=lo.astype(np.int64),
            bucket_hi=hi.astype(np.int64),
            bucket_class=klass,
            entry_row=row,
            entry_pos=pos,
            anchor_lo=anchor_lo,
            anchor_hi=anchor_hi,
            anchor_row=(
                np.concatenate(a_rows) if a_rows else np.empty(0, dtype=np.int64)
            ),
            anchor_pos=(
                np.concatenate(a_poss) if a_poss else np.empty(0, dtype=np.int64)
            ),
            t_len=t_len,
            t_support=support,
        )


# ──────────────────────────────────────────────────────────────────────
# The bipartite probe join
# ──────────────────────────────────────────────────────────────────────


#: Projected join rows one expansion sub-chunk may hold. Bounds the block's
#: transient at ~16M rows x ~40 B whatever the family structure.
_SUBCHUNK_ROWS = 16_000_000


def _window_scores(
    key: np.ndarray,
    pair_id: np.ndarray,
    strat: np.ndarray,
    band: int,
    probes_per_read: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, distinct probes)`` in the diagonal window starting at each row.

    Rows are sorted by ``key`` — ``(pair, diagonal)`` packed so that a band
    cannot reach into another pair — and the window starting at row ``i`` is
    the rows from ``i`` on whose key is within ``band`` of it. ``strat`` is
    each row's probe (its position stratum on the read).

    A window is scored by the DISTINCT PROBES in it, not by its rows. A
    k-mer the template repeats puts one probe on several diagonals, and
    counting rows let an unrelated template carrying a 200-nt A-run
    out-score a read's own exact template 130 to 52 at 32 probes, after
    which the shortlist's fraction cut discarded the exact template (review
    of 91e7c69). It also let ONE probe on two diagonals meet ``min_shared``.

    Exact, and no Python loop. Row ``j`` repeats a probe inside the window
    starting at row ``i`` exactly when the previous row of its (pair, probe)
    lies at or after ``i``; the windows that reach ``j`` start at
    ``reach[j]`` or later; so ``j`` is a repeat for the starts
    ``reach[j] .. prev[j]`` — one range update per repeated row, summed
    with a difference array.
    """
    m = int(key.shape[0])
    row_i = np.arange(m, dtype=np.int64)
    hits = np.searchsorted(key, key + band, side="right") - row_i
    # Rows ascend by pair, so grouping them by probe stratum alone — a radix
    # sort on a 16-bit key, linear — leaves each (pair, probe)'s rows
    # adjacent and in row order.
    if probes_per_read <= 1 << 16:
        by_probe = np.argsort(strat.astype(np.uint16), kind="stable")
    else:
        by_probe = np.lexsort((pair_id, strat))
    pair_s, strat_s = pair_id[by_probe], strat[by_probe]
    same = (strat_s[1:] == strat_s[:-1]) & (pair_s[1:] == pair_s[:-1])
    prev = np.full(m, -1, dtype=np.int64)
    prev[by_probe[1:][same]] = by_probe[:-1][same]
    again = np.flatnonzero(prev >= 0)  # few: most probes sit on one diagonal
    if again.shape[0] == 0:
        return hits, hits
    reach = np.searchsorted(key, key[again] - band, side="left")
    inside = prev[again] >= reach
    again, reach = again[inside], reach[inside]
    delta = np.bincount(reach, minlength=m + 1) - np.bincount(
        prev[again] + 1, minlength=m + 1
    )
    return hits, hits - np.cumsum(delta[:m])


def _expand_and_window(
    probe: np.ndarray,
    pb: np.ndarray,
    rep: np.ndarray,
    src_lo: np.ndarray,
    is_anchor: np.ndarray,
    p_owner: np.ndarray,
    p_q: np.ndarray,
    stratum: np.ndarray,
    owned: np.ndarray,
    index: TemplateMinimizerIndex,
    params: NativeParams,
):
    """Expand one probe range into rows; window, antisense-test and reduce.

    Returns ``(local, template_row, n_shared, median diag, only_anchor,
    n_antisense)`` for the pairs that pass, or ``None`` for an empty range.
    """
    n_rows = int(rep.sum())
    if n_rows == 0:
        return None
    r_probe = np.repeat(np.arange(probe.shape[0]), rep)
    within_b = np.arange(n_rows) - np.repeat(np.cumsum(rep) - rep, rep)
    entry = np.repeat(src_lo, rep) + within_b
    anchor_rows = np.repeat(is_anchor, rep)
    # Masked gathers, not np.where: an anchor-free chunk has empty anchor
    # arrays, and np.where evaluates both branches.
    t_row = np.empty(n_rows, dtype=np.int64)
    t_pos = np.empty(n_rows, dtype=np.int64)
    whole = ~anchor_rows
    t_row[whole] = index.entry_row[entry[whole]]
    t_pos[whole] = index.entry_pos[entry[whole]]
    if anchor_rows.any():
        t_row[anchor_rows] = index.anchor_row[entry[anchor_rows]]
        t_pos[anchor_rows] = index.anchor_pos[entry[anchor_rows]]
    local = p_owner[r_probe]
    pos_q = p_q[probe][r_probe].astype(np.int64)
    strat = stratum[probe][r_probe]
    anchor_only_probe = is_anchor[r_probe]
    diag = t_pos - pos_q

    # ── dedupe on (read, template, probe, diagonal) ──
    order = np.lexsort((strat, diag, t_row, local))
    local, t_row, diag, pos_q, strat, anchor_only_probe = (
        a[order] for a in (local, t_row, diag, pos_q, strat, anchor_only_probe)
    )
    fresh = np.ones(local.shape[0], dtype=bool)
    fresh[1:] = (
        (local[1:] != local[:-1])
        | (t_row[1:] != t_row[:-1])
        | (diag[1:] != diag[:-1])
        | (strat[1:] != strat[:-1])
    )
    local, t_row, diag, pos_q, strat, anchor_only_probe = (
        a[fresh] for a in (local, t_row, diag, pos_q, strat, anchor_only_probe)
    )
    m = int(local.shape[0])

    # ── the best diagonal window per (read, template) pair ──
    pair_first = np.flatnonzero(
        np.concatenate([[True], (local[1:] != local[:-1]) | (t_row[1:] != t_row[:-1])])
    )
    n_pairs = int(pair_first.shape[0])
    pair_size = np.diff(np.append(pair_first, m))
    pair_id = np.repeat(np.arange(n_pairs, dtype=np.int64), pair_size)
    # Rows are sorted by (pair, diag); diagonals span (-read, +template), so
    # offset into a field wide enough that the band cannot carry.
    d_min = int(diag.min())
    width = int(diag.max()) - d_min + int(params.diag_band) + 2
    key = pair_id * width + (diag - d_min)
    row_i = np.arange(m, dtype=np.int64)
    hits, distinct = _window_scores(
        key, pair_id, strat, int(params.diag_band), int(params.probes_per_read)
    )
    n_shared = np.maximum.reduceat(distinct, pair_first)
    lead = np.minimum.reduceat(
        np.where(distinct == np.repeat(n_shared, pair_size), row_i, m), pair_first
    )
    #: Rows the chosen window holds — its EXTENT, which the placement hint
    #: and the antisense test read; ``n_shared`` is its score.
    in_window = hits[lead]
    median = diag[lead + (in_window - 1) // 2]

    # ── sense or antisense: the containment join's two-clause rule ──
    spread = diag[lead + in_window - 1] - diag[lead]
    antisense = np.zeros(n_pairs, dtype=bool)
    unsure = np.flatnonzero(spread > 0)
    if unsure.shape[0]:
        anti = np.empty(m + 1, dtype=np.int64)
        anti[:m] = diag + 2 * pos_q
        anti[m] = 0
        bound = np.empty(2 * unsure.shape[0], dtype=np.int64)
        bound[0::2] = lead[unsure]
        bound[1::2] = lead[unsure] + in_window[unsure]
        anti_spread = (
            np.maximum.reduceat(anti, bound)[0::2]
            - np.minimum.reduceat(anti, bound)[0::2]
        )
        # The most rows any ONE diagonal of the pair holds: enough probes
        # agreeing on a diagonal is a candidate whatever else the window
        # holds (the tandem-repeat case).
        run_start = np.ones(m, dtype=bool)
        np.not_equal(diag[1:], diag[:-1], out=run_start[1:])
        run_start[pair_first] = True
        run_first = np.flatnonzero(run_start)
        run_size = np.diff(np.append(run_first, m))
        agreed = np.maximum.reduceat(np.repeat(run_size, run_size), pair_first)
        needed_u = np.minimum(int(params.min_shared), owned[local[pair_first]])
        antisense[unsure] = (anti_spread < spread[unsure]) & (
            agreed[unsure] < needed_u[unsure]
        )
    pair_local = local[pair_first]
    pair_row = t_row[pair_first]
    only_anchor = np.logical_and.reduceat(anchor_only_probe, pair_first)
    needed = np.minimum(int(params.min_shared), owned[pair_local])
    passed = (n_shared >= needed) & ~antisense
    n_antisense = int(((n_shared >= needed) & antisense).sum())
    return (
        pair_local[passed],
        pair_row[passed],
        n_shared[passed],
        median[passed],
        only_anchor[passed],
        n_antisense,
    )


@dataclass(frozen=True, slots=True)
class BlockCandidates:
    """One block's candidates, grouped by ``local`` (the read within the
    block), plus the per-read flags and counters the driver reports."""

    local: np.ndarray  # (C,) int64 ascending
    template_row: np.ndarray  # (C,) int64
    n_shared: np.ndarray  # (C,) int64
    diag: np.ndarray  # (C,) int64 — median diagonal of the best window
    cap_hit: np.ndarray  # (n_block,) bool — budget or candidate cap bound
    overflow: np.ndarray  # (n_block,) bool — some probe joined via anchors
    n_antisense: int
    n_join_rows: int


def candidates_block(
    h_q: np.ndarray,
    p_q: np.ndarray,
    offs: np.ndarray,
    read_len: np.ndarray,
    index: TemplateMinimizerIndex,
    params: NativeParams,
    *,
    kmer: int,
) -> BlockCandidates:
    """Join one block of reads against the template index. numpy only.

    The candidate test is the containment join's: ``min_shared`` probes (or
    as many as the read owns, if fewer) inside one ``diag_band`` diagonal
    window, and the window must not fit the anti-diagonal better unless that
    many probes agree on a single diagonal.
    """
    n_block = int(offs.shape[0] - 1)
    cap_hit = np.zeros(n_block, dtype=bool)
    overflow = np.zeros(n_block, dtype=bool)
    empty = BlockCandidates(
        local=np.empty(0, dtype=np.int64),
        template_row=np.empty(0, dtype=np.int64),
        n_shared=np.empty(0, dtype=np.int64),
        diag=np.empty(0, dtype=np.int64),
        cap_hit=cap_hit,
        overflow=overflow,
        n_antisense=0,
        n_join_rows=0,
    )
    if h_q.shape[0] == 0 or index.bucket_hash.shape[0] == 0:
        return empty

    # ── probes: per (read, stratum), smallest hash among PRESENT buckets ──
    owner = np.repeat(np.arange(n_block, dtype=np.int64), np.diff(offs))
    b = np.searchsorted(index.bucket_hash, h_q)
    np.clip(b, 0, index.bucket_hash.shape[0] - 1, out=b)
    present = index.bucket_hash[b] == h_q
    if not present.any():
        return empty
    span = np.maximum(read_len[owner] - int(kmer) + 1, 1)
    stratum = np.minimum(
        p_q.astype(np.int64) * int(params.probes_per_read) // span,
        int(params.probes_per_read) - 1,
    )
    el = np.flatnonzero(present)
    # Whole-bucket probes beat anchor probes; then smallest hash, position.
    order = np.lexsort(
        (
            p_q[el],
            h_q[el],
            index.bucket_class[b[el]],
            stratum[el],
            owner[el],
        )
    )
    el = el[order]
    first = np.ones(el.shape[0], dtype=bool)
    key_o, key_s = owner[el], stratum[el]
    first[1:] = (key_o[1:] != key_o[:-1]) | (key_s[1:] != key_s[:-1])
    probe = el[first]  # ≤ probes_per_read per read, owner-ascending

    # ── the per-read row budget, in stratum order ──
    pb = b[probe]
    cost = np.where(
        index.bucket_class[pb] == 1,
        index.anchor_hi[pb] - index.anchor_lo[pb],
        index.bucket_hi[pb] - index.bucket_lo[pb],
    )
    p_owner = owner[probe]
    start = np.concatenate([[0], 1 + np.flatnonzero(np.diff(p_owner))])
    within = np.arange(probe.shape[0]) - np.repeat(
        start, np.diff(np.append(start, probe.shape[0]))
    )
    spent = np.cumsum(cost)
    before = spent - cost
    base = np.repeat(before[start], np.diff(np.append(start, probe.shape[0])))
    keep = ((spent - base) <= int(params.max_rows_per_read)) | (within < 2)
    if not keep.all():
        cap_hit[p_owner[~keep]] = True
        probe, pb, cost, p_owner = probe[keep], pb[keep], cost[keep], p_owner[keep]
    owned = np.bincount(p_owner, minlength=n_block)

    # ── expansion, in sub-chunks of bounded projected rows ──
    # The cost is known per probe before any row exists, so a block whose
    # reads sit in large families (every probe a family-sized bucket) is cut
    # into read ranges of at most ``_SUBCHUNK_ROWS`` projected rows rather
    # than expanded whole. Everything below is per read, so the cut cannot
    # change the answer.
    is_anchor = index.bucket_class[pb] == 1
    src_lo = np.where(is_anchor, index.anchor_lo[pb], index.bucket_lo[pb])
    rep_all = cost.astype(np.int64)
    n_rows_total = int(rep_all.sum())
    if n_rows_total == 0:
        return empty

    probe_bound = np.searchsorted(p_owner, np.arange(n_block + 1))
    rows_before = np.concatenate([[0], np.cumsum(rep_all)])
    chunks: list[tuple[int, int]] = []
    r0 = 0
    while r0 < n_block:
        target = rows_before[probe_bound[r0]] + _SUBCHUNK_ROWS
        r1 = int(np.searchsorted(rows_before[probe_bound], target, side="right")) - 1
        r1 = min(max(r1, r0 + 1), n_block)
        if probe_bound[r1] > probe_bound[r0]:
            chunks.append((int(probe_bound[r0]), int(probe_bound[r1])))
        r0 = r1

    out_local: list[np.ndarray] = []
    out_row: list[np.ndarray] = []
    out_shared: list[np.ndarray] = []
    out_diag: list[np.ndarray] = []
    out_anchor: list[np.ndarray] = []
    n_antisense = 0
    for c0, c1 in chunks:
        part = _expand_and_window(
            probe[c0:c1],
            pb[c0:c1],
            rep_all[c0:c1],
            src_lo[c0:c1],
            is_anchor[c0:c1],
            p_owner[c0:c1],
            p_q,
            stratum,
            owned,
            index,
            params,
        )
        if part is None:
            continue
        local_c, row_c, shared_c, diag_c, anchor_c, anti_c = part
        out_local.append(local_c)
        out_row.append(row_c)
        out_shared.append(shared_c)
        out_diag.append(diag_c)
        out_anchor.append(anchor_c)
        n_antisense += anti_c
    if not out_local:
        return BlockCandidates(
            local=np.empty(0, dtype=np.int64),
            template_row=np.empty(0, dtype=np.int64),
            n_shared=np.empty(0, dtype=np.int64),
            diag=np.empty(0, dtype=np.int64),
            cap_hit=cap_hit,
            overflow=overflow,
            n_antisense=n_antisense,
            n_join_rows=n_rows_total,
        )
    pair_local = np.concatenate(out_local)
    pair_row = np.concatenate(out_row)
    n_shared_p = np.concatenate(out_shared)
    median = np.concatenate(out_diag)
    only_anchor = np.concatenate(out_anchor)
    overflow[pair_local[only_anchor]] = True

    # ── the per-read candidate cap ──
    counts = np.bincount(pair_local, minlength=n_block)
    if bool((counts > int(params.max_candidates)).any()):
        rank_order = np.lexsort(
            (pair_row, -index.t_support[pair_row], -n_shared_p, pair_local)
        )
        rank = np.empty(rank_order.shape[0], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        rank[rank_order] = np.arange(rank_order.shape[0]) - np.repeat(starts, counts)
        keep = rank < int(params.max_candidates)
        cap_hit[pair_local[~keep]] = True
        keep_sorted = np.flatnonzero(keep)
        pair_local, pair_row, n_shared_p, median, only_anchor = (
            a[keep_sorted]
            for a in (pair_local, pair_row, n_shared_p, median, only_anchor)
        )
    return BlockCandidates(
        local=pair_local,
        template_row=pair_row,
        n_shared=n_shared_p,
        diag=median,
        cap_hit=cap_hit,
        overflow=overflow,
        n_antisense=n_antisense,
        n_join_rows=n_rows_total,
    )


# ──────────────────────────────────────────────────────────────────────
# The driver: blocks of read rows through assign_block_edlib
# ──────────────────────────────────────────────────────────────────────


class _NativeBlock:
    """Candidates dressed as a hit block, so :func:`~.assign.assign_block_edlib`
    runs UNCHANGED: ``chain_score`` is the shared-probe count (the shortlist
    key), and the "chained" coordinates are the candidate's own PLACEMENT —
    the overlap the join's median diagonal implies.

    Whole-sequence coordinates were measured wrong (review of 91e7c69): the
    finalist aligner windows only the READ from them, so a 500-nt template
    contained in a 2,000-nt read was aligned as the whole read placed infix
    into the 500-nt template and anchor-trimmed to a 130-nt sliver at
    identity 1.0. The diagonal says read position 0 sits at template
    position ``diag``, so the overlap is ``[max(0, -diag), min(q_len,
    t_len - diag))`` on the read and the mirror on the template — which the
    aligner pads, anchors and re-extends exactly as it does a minimap2
    chain, recovering the full containment and the terminal-extension
    evidence the M-step votes on.
    """

    def __init__(
        self,
        read_row: np.ndarray,
        template_row: np.ndarray,
        n_shared: np.ndarray,
        diag: np.ndarray,
        read_len: np.ndarray,
        t_len: np.ndarray,
        n_antisense: int,
    ) -> None:
        self.read_row = read_row
        self.template_row = template_row
        self.chain_score = n_shared.astype(np.int64)
        self.n_dropped_strand = int(n_antisense)
        self.n_dropped_template = 0
        self._q_len = read_len
        self._t_len = t_len
        diag = diag.astype(np.int64)
        q_start = np.maximum(-diag, 0)
        q_end = np.minimum(read_len, t_len - diag)
        t_start = np.maximum(diag, 0)
        t_end = np.minimum(t_len, diag + read_len)
        # A degenerate hint (no overlap under it) falls back to the whole
        # pair rather than an empty window.
        bad = (q_end <= q_start) | (t_end <= t_start)
        self._q_start = np.where(bad, 0, q_start)
        self._q_end = np.where(bad, read_len, q_end)
        self._t_start = np.where(bad, 0, t_start)
        self._t_end = np.where(bad, t_len, t_end)

    def __len__(self) -> int:
        return int(self.read_row.size)

    def int_fields(self, hits: np.ndarray, columns) -> dict[str, np.ndarray]:
        hits = np.asarray(hits, dtype=np.int64)
        full = {
            "q_start": self._q_start[hits],
            "q_end": self._q_end[hits],
            "t_start": self._t_start[hits],
            "t_end": self._t_end[hits],
            "q_len": self._q_len[hits],
            "t_len": self._t_len[hits],
        }
        return {name: full[name] for name in columns}


# (index, params, kmer). Set in the parent before the pool forks, cleared in
# its `finally`; workers read it copy-on-write — plain numpy only, for the
# same page-dirtying reason as the graph builder's kernel state.
_NATIVE_STATE: tuple[TemplateMinimizerIndex, NativeParams, int] | None = None

# Per-process handle caches, like mstep_pool's: a forked worker inherits the
# empty dicts and opens for itself.
_MINIS: dict[str, ReadMinimizerStore] = {}
_READ_LEN: dict[str, np.ndarray] = {}


def _minis(path: str) -> ReadMinimizerStore:
    if path not in _MINIS:
        _MINIS[path] = ReadMinimizerStore.open(path)
    return _MINIS[path]


def _read_lengths(reads, corpus_path: str) -> np.ndarray:
    key = str(corpus_path)
    if key not in _READ_LEN:
        _READ_LEN[key] = reads.lengths().astype(np.int64)
    return _READ_LEN[key]


def _native_block(
    lo: int,
    hi: int,
    shard: int,
    *,
    output_dir: str,
    round_index: int,
    minis_dir: str,
    assign_kwargs: dict,
    corpus_path: str | None = None,
    templates_path: str | None = None,
    store=None,
    reads=None,
) -> tuple[dict, int]:
    """Join, align and rank reads ``[lo, hi)``; write their shard.

    Module-level so a pool can pickle it; everything below here is numpy,
    pyarrow and edlib. Returns ``(counters, shard)``.
    """
    import pyarrow.parquet as pq

    from constellation.sequencing.transcriptome.cluster.denovo.em import mstep_pool
    from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
        EM_ASSIGNMENT_TABLE,
        _reason_counts,
        _unassigned_batch,
        assign_block_edlib,
        reason_tallies,
    )

    state = _NATIVE_STATE
    assert state is not None, "the driver sets the index before the pool forks"
    index, params, kmer = state
    if store is None:
        store = mstep_pool._templates(str(templates_path))
    if reads is None:
        reads = mstep_pool._reads(str(corpus_path))
    read_len = _read_lengths(reads, str(corpus_path))

    h_q, p_q, offs = _minis(minis_dir).block(lo, hi)
    found = candidates_block(h_q, p_q, offs, read_len[lo:hi], index, params, kmer=kmer)
    stats = {
        "n_hits": int(found.local.shape[0]),
        "n_join_rows": found.n_join_rows,
        "n_dropped_strand": found.n_antisense,
        "n_dropped_template": 0,
        "n_reads_seen": hi - lo,
        "n_assigned": 0,
        "n_cap_hit": 0,
        "n_aligned": 0,
        "n_shortlist_truncated": 0,
        "n_overflow_reads": int(found.overflow.sum()),
        **reason_tallies(),
    }
    import pyarrow as pa_

    nb = _NativeBlock(
        read_row=found.local + lo,
        template_row=found.template_row,
        n_shared=found.n_shared,
        diag=found.diag,
        read_len=read_len[found.local + lo],
        t_len=index.t_len[found.template_row],
        n_antisense=found.n_antisense,
    )
    with_cands = np.unique(found.local)
    batch, n_aligned = assign_block_edlib(
        nb,
        store=store,
        reads=reads,
        round_index=round_index,
        minimap2_n=int(params.max_candidates),
        cap_hit=found.cap_hit[with_cands],
        **assign_kwargs,
    )
    stats["n_aligned"] = n_aligned
    parts = []
    if batch.num_rows:
        parts.append(batch)
    bare = np.flatnonzero(~np.isin(np.arange(hi - lo, dtype=np.int64), with_cands))
    if bare.size:
        parts.append(
            _unassigned_batch(bare + lo, reads, round_index, reason="no_candidate")
        )
    if not parts:
        return stats, shard
    table = pa_.Table.from_batches(parts, schema=EM_ASSIGNMENT_TABLE)
    stats["n_assigned"] = int(
        pa_.compute.sum(
            pa_.compute.greater_equal(table.column("template_id"), 0)
        ).as_py()
        or 0
    )
    stats["n_cap_hit"] = int(
        pa_.compute.sum(table.column("candidate_cap_hit")).as_py() or 0
    )
    stats["n_shortlist_truncated"] = int(
        pa_.compute.sum(table.column("shortlist_truncated")).as_py() or 0
    )
    for batch_part in table.to_batches():
        for key, value in _reason_counts(batch_part).items():
            stats[key] += value
    pq.write_table(table, Path(output_dir) / f"part-{shard:05d}.parquet")
    return stats, shard


def run_native_estep(
    *,
    store,
    reads,
    output_dir: Path,
    round_index: int,
    minis_dir: Path,
    kmer: int,
    window: int,
    params: NativeParams | None = None,
    align_workers: int = 1,
    corpus_path=None,
    templates_path=None,
    progress=None,
    **assign_kwargs,
) -> dict:
    """The native E-step: no minimap2 anywhere.

    The template index is built HERE, in the parent, with torch — before the
    pool forks — and handed to the workers copy-on-write. Blocks are fixed
    read-row ranges, each writing its own shard, so the output is
    independent of ``align_workers``. Every read in the corpus gets a row:
    one with no candidate is written with ``unassigned_reason =
    'no_candidate'`` rather than vanishing from the accounting.
    """
    global _NATIVE_STATE

    params = params or NativeParams()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda _m: None)
    for stale in output_dir.glob("part-*.parquet"):
        stale.unlink()

    t0 = time.time()
    index = TemplateMinimizerIndex.build(
        store, kmer=int(kmer), window=int(window), params=params
    )
    n_over = int((index.bucket_class == 1).sum())
    log(
        f"round {round_index}: native index — {index.entry_row.shape[0]:,} "
        f"template minimizers in {index.bucket_hash.shape[0]:,} buckets "
        f"({n_over:,} over the cap) in {time.time() - t0:.1f}s"
    )

    n_reads = int(reads.n_reads)
    bounds = list(range(0, n_reads, int(params.block_reads))) + [n_reads]
    blocks = [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i + 1] > bounds[i]
    ]
    from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
        reason_tallies,
    )

    totals = {
        "n_reads_seen": 0,
        "n_assigned": 0,
        "n_cap_hit": 0,
        "n_hits": 0,
        "n_join_rows": 0,
        "n_dropped_strand": 0,
        "n_dropped_template": 0,
        "n_aligned": 0,
        "n_shortlist_truncated": 0,
        "n_overflow_reads": 0,
        **reason_tallies(),
    }

    def _collect(result: tuple[dict, int]) -> None:
        stats, _shard = result
        for k, v in stats.items():
            totals[k] += v

    common = {
        "output_dir": str(output_dir),
        "round_index": round_index,
        "minis_dir": str(minis_dir),
        "assign_kwargs": assign_kwargs,
    }
    _NATIVE_STATE = (index, params, int(kmer))
    try:
        if int(align_workers) <= 1:
            minis = _minis(str(minis_dir))
            read_len = reads.lengths().astype(np.int64)
            for shard, (lo, hi) in enumerate(blocks):
                _READ_LEN[str(corpus_path)] = read_len
                _collect(
                    _native_block(
                        lo,
                        hi,
                        shard,
                        store=store,
                        reads=reads,
                        corpus_path=str(corpus_path),
                        **common,
                    )
                )
            del minis
        else:
            if corpus_path is None or templates_path is None:
                raise ValueError(
                    "align_workers > 1 needs corpus_path and templates_path: "
                    "pool workers open the stores themselves"
                )
            pending = set()
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=int(align_workers), mp_context=ctx
            ) as ex:
                try:
                    for shard, (lo, hi) in enumerate(blocks):
                        pending.add(
                            ex.submit(
                                _native_block,
                                lo,
                                hi,
                                shard,
                                corpus_path=str(corpus_path),
                                templates_path=str(templates_path),
                                **common,
                            )
                        )
                        if len(pending) >= 2 * int(align_workers):
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            for f in done:
                                _collect(f.result())
                    for f in pending:
                        _collect(f.result())
                finally:
                    for f in pending:
                        f.cancel()
    finally:
        _NATIVE_STATE = None

    totals["n_unmapped"] = totals["n_no_candidate"]
    totals["n_unassigned"] = totals["n_reads_seen"] - totals["n_assigned"]
    totals["n_shards"] = len(blocks)
    denom = totals["n_reads_seen"]
    totals["cap_hit_fraction"] = totals["n_cap_hit"] / denom if denom else 0.0
    totals["shortlist_truncated_fraction"] = (
        totals["n_shortlist_truncated"] / denom if denom else 0.0
    )
    totals["overflow_read_fraction"] = (
        totals["n_overflow_reads"] / denom if denom else 0.0
    )
    totals["aligner"] = "native"
    return totals


__all__ = [
    "MINIS_ARROW",
    "MINIS_META",
    "MINIS_OFFSETS",
    "BlockCandidates",
    "NativeParams",
    "ReadMinimizerStore",
    "TemplateMinimizerIndex",
    "candidates_block",
    "run_native_estep",
    "write_read_minimizers",
]
