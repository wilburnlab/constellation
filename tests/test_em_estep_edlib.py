"""The two-pass E-step (``--estep-aligner edlib``) without minimap2 on PATH.

Hand-written no-``-c`` PAF (chain coordinates + ``s1:i``, no ``AS``/``cg``) is
reduced by :func:`assign_block_edlib` against real template and read
sequences, so what is pinned is the composition: shortlist on chain score,
admission and ranking on the *aligned* identity, and the round-1 lazy walk
that aligns only until the first admitted candidate in replication order.
"""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em import scheduler as sched
from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
    EM_ASSIGNMENT_TABLE,
    _run_edlib_estep,
    assign_block_edlib,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.paf_scan import (
    iter_group_blocks,
    scan_paf_block,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.realign import (
    align_finalist,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
    TEMPLATE_TABLE,
    TemplateStore,
)

_RNG = random.Random(5)
TRUE = "".join(_RNG.choice("ACGT") for _ in range(600))


def _mutant(seq: str, positions) -> str:
    s = list(seq)
    for p in positions:
        s[p] = {"A": "C", "C": "G", "G": "T", "T": "A"}[s[p]]
    return "".join(s)


class _Reads:
    """The slice of ReadStore the reducer touches, over real sequences."""

    def __init__(self, seqs, samples=None):
        self.seqs = list(seqs)
        self.read_id = pa.chunked_array([pa.array([f"r{i}" for i in range(len(seqs))])])
        self.chunk_starts = np.array([0, len(seqs)], dtype=np.int64)
        self.sample_id = np.asarray(
            samples if samples is not None else [0] * len(seqs), dtype=np.int64
        )

    @property
    def n_reads(self):
        return len(self.seqs)

    def take_sequences(self, rows):
        return [self.seqs[int(r)] for r in rows]

    def take_read_ids(self, rows):
        from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
            chunked_take,
        )

        return chunked_take(self.read_id, rows, self.chunk_starts).cast(pa.string())


def _store(sequences, *, replication=None, weight=None):
    n = len(sequences)
    return TemplateStore.from_table(
        pa.table(
            {
                "template_id": pa.array(np.arange(n, dtype=np.int64) + 100),
                "sequence": pa.array(sequences, pa.large_string()),
                "orf_start": pa.array(np.zeros(n, np.int32)),
                "orf_end": pa.array(np.zeros(n, np.int32)),
                "orf_aa_length": pa.array(np.zeros(n, np.int32)),
                "node_weight": pa.array(
                    np.asarray(weight if weight is not None else np.ones(n), float)
                ),
                "orf_replication": pa.array(
                    np.asarray(
                        replication if replication is not None else np.ones(n),
                        dtype=np.int64,
                    )
                ),
                "seed_read_quality": pa.array(np.full(n, 20.0, np.float32)),
                "seed_read_row": pa.array(np.full(n, -1, np.int32)),
                "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
            },
            schema=TEMPLATE_TABLE,
        )
    )


def _row(q, t, *, s1, q_len=600, t_len=600, inset=5):
    """A no-`-c` PAF line: chained coords a few bases inside, `s1:i`, no AS/cg."""
    return (
        f"{q}\t{q_len}\t{inset}\t{q_len - inset}\t+\t{t}\t{t_len}\t{inset}\t"
        f"{t_len - inset}\t{q_len // 2}\t{q_len}\t0\ttp:A:S\tcm:i:50\ts1:i:{s1}\t"
        f"dv:f:0.01\trl:i:0"
    )


def _block(rows):
    buf = np.frombuffer(("\n".join(rows) + "\n").encode(), dtype=np.uint8)
    return scan_paf_block(buf, n_templates=10, chain_score=True)


class _Spy:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, read, template, **kw):
        self.calls.append((read, template))
        return align_finalist(read, template, **kw)


# ── pieces ────────────────────────────────────────────────────────────


def test_s1_is_decoded_and_absent_means_zero():
    hb = _block([_row(0, 0, s1=812), _row(0, 1, s1=97).replace("\ts1:i:97", "")])
    assert hb.chain_score.tolist() == [812, 0]
    # The default scan does not pay for it.
    plain = scan_paf_block(
        np.frombuffer((_row(0, 0, s1=5) + "\n").encode(), np.uint8), n_templates=10
    )
    assert plain.chain_score is None


def test_shortlist_by_chain_frac_then_k():
    s1 = np.array([100, 95, 70, 90, 10, 10, 10])
    ptr = np.array([0, 4, 7])
    keep, rank, n_elig = sched.shortlist_by_chain(s1, ptr, k=2, frac=0.8)
    # read 0: 100, 95, 90 clear 0.8 x 100; k=2 keeps the top two.
    assert keep.tolist() == [True, True, False, False, True, True, False]
    assert rank[:4].tolist() == [0, 1, 3, 2]
    assert n_elig.tolist() == [3, 3]


@pytest.mark.parametrize("chunk", [37, 200, 1 << 16])
def test_group_blocks_never_split_a_read(chunk):
    rows = [_row(q, t, s1=10 * t + 1) for q in range(12) for t in range(3)]
    text = ("\n".join(rows) + "\n").encode()
    blocks = list(
        iter_group_blocks(
            [text[i : i + chunk] for i in range(0, len(text), chunk)],
            block_bytes=chunk,
        )
    )
    assert b"".join(blocks) == text
    for b in blocks:
        names = [ln.split(b"\t")[0] for ln in b.splitlines()]
        assert len(names) % 3 == 0, "a read's hits were split across blocks"


# ── the reducer ───────────────────────────────────────────────────────


def test_round1_aligns_lazily_in_replication_order():
    """The better-replicated candidate is aligned first and, being admitted,
    ends the walk: the other candidate is never aligned at all."""
    store = _store([TRUE, TRUE], replication=[1, 50])
    reads = _Reads([_mutant(TRUE, [100, 300])])
    spy = _Spy()
    batch, n_aligned = assign_block_edlib(
        _block([_row(0, 0, s1=900), _row(0, 1, s1=880)]),
        store=store,
        reads=reads,
        round_index=1,
        align_fn=spy,
    )
    assert n_aligned == 1 and len(spy.calls) == 1
    assert batch.column("template_row").to_pylist() == [1]
    assert batch.column("chain_score").to_pylist() == [880]
    assert batch.column("cigar").to_pylist()[0] is not None


def test_round1_falls_through_a_candidate_that_fails_the_floor():
    """The most-replicated template is a different sequence: aligned, refused
    by p_floor, and the walk moves on to the next."""
    other = "".join(random.Random(9).choice("ACGT") for _ in range(600))
    store = _store([TRUE, other], replication=[1, 50])
    reads = _Reads([_mutant(TRUE, [50])])
    spy = _Spy()
    batch, n_aligned = assign_block_edlib(
        _block([_row(0, 0, s1=900), _row(0, 1, s1=850)]),
        store=store,
        reads=reads,
        round_index=1,
        align_fn=spy,
    )
    assert n_aligned == 2
    assert batch.column("template_row").to_pylist() == [0]
    assert batch.column("n_admitted").to_pylist() == [1]


def test_round2_ranks_on_the_aligned_likelihood():
    """Two candidates, one carrying three substitutions against the read's
    true template: the likelihood over the aligned CIGARs picks the truth,
    even when the chain score slightly prefers the other."""
    store = _store([_mutant(TRUE, [10, 200, 400]), TRUE])
    reads = _Reads([TRUE])
    batch, n_aligned = assign_block_edlib(
        _block([_row(0, 0, s1=905), _row(0, 1, s1=900)]),
        store=store,
        reads=reads,
        round_index=2,
    )
    assert n_aligned == 2
    row = batch.to_pylist()[0]
    assert row["template_row"] == 1
    assert row["cigar"] == "600="
    assert (row["q_start"], row["q_end"], row["t_start"], row["t_end"]) == (
        0,
        600,
        0,
        600,
    )
    assert row["as_score"] == 1200  # map-ont A=2 over 600 matches
    assert row["logl"] is not None and not np.isnan(row["logl"])
    assert row["shortlist_truncated"] is False


def test_unadmitted_read_is_emitted_unassigned():
    other = "".join(random.Random(4).choice("ACGT") for _ in range(600))
    store = _store([other])
    reads = _Reads([TRUE])
    batch, _ = assign_block_edlib(
        _block([_row(0, 0, s1=300)]), store=store, reads=reads, round_index=2
    )
    row = batch.to_pylist()[0]
    assert row["template_id"] == -1 and row["cigar"] is None


def test_round1_is_not_shortlisted_by_chain_score():
    """Round 1 ranks on replication, which the chain score knows nothing
    about: the 50x template wins even with the lowest chain score and k=2.
    A chain-score shortlist here cut 55% of reads on the synthetic panel and
    left 28 templates where minimap2 -c left 13."""
    store = _store([TRUE, TRUE, TRUE], replication=[1, 1, 50])
    reads = _Reads([TRUE])
    batch, n_aligned = assign_block_edlib(
        _block([_row(0, 0, s1=900), _row(0, 1, s1=890), _row(0, 2, s1=100)]),
        store=store,
        reads=reads,
        round_index=1,
        shortlist_k=2,
    )
    assert batch.column("template_row").to_pylist() == [2]
    assert n_aligned == 1
    assert batch.column("shortlist_truncated").to_pylist() == [False]


def test_round1_truncation_means_the_walk_gave_up():
    """k bounds alignment ATTEMPTS in round 1. With k=1 and the best-
    replicated candidate inadmissible, the walk stops unassigned — flagged,
    because an admissible candidate was left untried."""
    other = "".join(random.Random(9).choice("ACGT") for _ in range(600))
    store = _store([TRUE, other], replication=[1, 50])
    reads = _Reads([TRUE])
    batch, n_aligned = assign_block_edlib(
        _block([_row(0, 0, s1=900), _row(0, 1, s1=850)]),
        store=store,
        reads=reads,
        round_index=1,
        shortlist_k=1,
    )
    assert n_aligned == 1
    assert batch.column("template_row").to_pylist() == [-1]
    assert batch.column("shortlist_truncated").to_pylist() == [True]


# ── the driver ────────────────────────────────────────────────────────


def _stream_fixture(n_reads=8):
    rng = random.Random(21)
    t0 = TRUE
    t1 = "".join(rng.choice("ACGT") for _ in range(600))
    seqs, rows = [], []
    for q in range(n_reads):
        src = t0 if q % 2 == 0 else t1
        seqs.append(_mutant(src, [rng.randrange(600) for _ in range(3)]))
        if q == n_reads - 1:
            continue  # minimap2 reported nothing for this read
        rows.append(_row(q, 0, s1=900 if src is t0 else 400))
        rows.append(_row(q, 1, s1=900 if src is t1 else 400))
    text = ("\n".join(rows) + "\n").encode()
    return _store([t0, t1]), _Reads(seqs), text


def _run(tmp_path, workers, store, reads, text, name):
    from constellation.sequencing.transcriptome.cluster.denovo.em import mstep_pool

    # Forked pool workers inherit the parent's handle cache, so these
    # in-memory stores stand in for the mmapped files.
    mstep_pool._OPEN["reads:mem-corpus"] = reads
    mstep_pool._OPEN["templates:mem-templates"] = store
    out = tmp_path / name
    out.mkdir()
    stats = _run_edlib_estep(
        [text[i : i + 97] for i in range(0, len(text), 97)],
        store=store,
        reads=reads,
        output_dir=out,
        round_index=2,
        minimap2_n=50,
        block_bytes=300,
        align_workers=workers,
        corpus_path="mem-corpus",
        templates_path="mem-templates",
        assign_kwargs={},
    )
    import pyarrow.dataset as ds

    table = ds.dataset(sorted(out.glob("part-*.parquet")), schema=EM_ASSIGNMENT_TABLE)
    return stats, table.to_table().sort_by("read_row")


def test_driver_output_is_independent_of_worker_count(tmp_path):
    store, reads, text = _stream_fixture()
    s1, one = _run(tmp_path, 1, store, reads, text, "one")
    s3, three = _run(tmp_path, 3, store, reads, text, "three")
    def rows(t):  # NaN != NaN, so normalise before comparing
        return [
            {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
            for r in t.to_pylist()
        ]

    assert rows(one) == rows(three)
    assert s1["n_shards"] > 2  # the stream really was split into blocks
    for k in ("n_assigned", "n_unmapped", "n_reads_seen", "n_aligned"):
        assert s1[k] == s3[k], k
    assert s1["aligner"] == "edlib"


def test_driver_accounts_for_reads_minimap2_never_reported(tmp_path):
    store, reads, text = _stream_fixture()
    stats, table = _run(tmp_path, 1, store, reads, text, "acct")
    assert stats["n_unmapped"] == 1
    assert table.num_rows == reads.n_reads
    tid = table.column("template_row").to_pylist()
    assert tid[-1] == -1
    # Every reported read found its source template.
    assert tid[:-1] == [0 if q % 2 == 0 else 1 for q in range(reads.n_reads - 1)]
