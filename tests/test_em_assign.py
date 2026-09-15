"""The E-step reducer, end to end from PAF bytes to assignment rows.

The pieces are unit-tested apart (``test_em_paf_scan``, ``test_em_scheduler``,
``test_em_likelihood``); what this file pins is that they compose — in
particular that a read's winner is decoded from the right PAF row, which is
the one thing the deferred-decode design can get silently wrong.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
    EM_ASSIGNMENT_TABLE,
    assign_block,
    assign_blocks,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.paf_scan import (
    iter_hit_blocks,
    scan_paf_block,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
    TEMPLATE_TABLE,
    TemplateStore,
)


class _Reads:
    """The slice of ReadStore the reducer touches."""

    def __init__(self, ids, samples=None):
        self.read_id = pa.chunked_array([pa.array(ids, pa.string())])
        self.sample_id = np.asarray(
            samples if samples is not None else [0] * len(ids), dtype=np.int64
        )


def _store(sequences, *, replication=None, quality=None, weight=None):
    n = len(sequences)
    return TemplateStore.from_table(
        pa.table(
            {
                "template_id": pa.array(np.arange(n, dtype=np.int64) + 100),
                "sequence": pa.array(sequences, pa.large_string()),
                "orf_start": pa.array(np.zeros(n, np.int32)),
                "orf_end": pa.array(np.array([len(s) for s in sequences], np.int32)),
                "orf_aa_length": pa.array(np.zeros(n, np.int32)),
                "node_weight": pa.array(
                    np.asarray(
                        weight if weight is not None else np.ones(n), dtype=np.float64
                    )
                ),
                "orf_replication": pa.array(
                    np.asarray(
                        replication if replication is not None else np.ones(n),
                        dtype=np.int64,
                    )
                ),
                "seed_read_quality": pa.array(
                    np.asarray(
                        quality if quality is not None else np.full(n, 20.0),
                        dtype=np.float32,
                    ),
                    pa.float32(),
                ),
                "seed_read_row": pa.array(np.full(n, -1, np.int32)),
                "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
            },
            schema=TEMPLATE_TABLE,
        )
    )


def _paf(rows) -> np.ndarray:
    return np.frombuffer(("\n".join(rows) + "\n").encode(), dtype=np.uint8)


def _row(
    q,
    t,
    *,
    n_match,
    aln_len,
    as_score,
    cigar,
    q_len=200,
    t_len=200,
    q_start=0,
    q_end=200,
    t_start=0,
    t_end=200,
):
    return (
        f"{q}\t{q_len}\t{q_start}\t{q_end}\t+\t{t}\t{t_len}\t{t_start}\t{t_end}\t"
        f"{n_match}\t{aln_len}\t60\ttp:A:P\tAS:i:{as_score}\tcg:Z:{cigar}"
    )


def test_round1_ranks_on_replication_not_on_score():
    """The self-capture case, composed end to end.

    Hit 0 is the read's own seed (perfect, replication 1); hit 1 is its gene's
    real template at 98% carrying 500 reads. Score says 0, replication says 1.
    """
    seq = "ACGTACGTAC" * 20
    store = _store([seq, seq], replication=[1, 500])
    hb = scan_paf_block(
        _paf(
            [
                _row(0, 0, n_match=200, aln_len=200, as_score=400, cigar="200="),
                _row(0, 1, n_match=196, aln_len=200, as_score=376, cigar="98=1X101="),
            ]
        ),
        n_templates=2,
    )
    batch = assign_block(hb, store=store, reads=_Reads(["r0"]), round_index=1)
    assert batch.num_rows == 1
    assert batch.column("template_row").to_pylist() == [1]
    assert batch.column("template_id").to_pylist() == [101]
    assert batch.column("n_admitted").to_pylist() == [2]


def test_a_read_below_the_floor_is_emitted_as_unassigned():
    seq = "ACGT" * 50
    store = _store([seq])
    hb = scan_paf_block(
        _paf([_row(0, 0, n_match=180, aln_len=200, as_score=300, cigar="200=")]),
        n_templates=1,
    )
    batch = assign_block(hb, store=store, reads=_Reads(["r0"]), round_index=1)
    assert batch.column("template_id").to_pylist() == [-1]
    assert batch.column("template_row").to_pylist() == [-1]
    assert batch.column("n_admitted").to_pylist() == [0]
    assert batch.column("weight").to_pylist() == [0.0]
    assert batch.column("cigar").to_pylist() == [None]


def test_winner_fields_come_from_the_winning_row():
    """The deferred-decode trap: fields must not be read off the wrong row."""
    seq = "ACGTACGTAC" * 20
    store = _store([seq, seq, seq], replication=[1, 1, 900])
    hb = scan_paf_block(
        _paf(
            [
                _row(
                    0,
                    0,
                    n_match=200,
                    aln_len=200,
                    as_score=400,
                    cigar="200=",
                    q_start=1,
                    q_end=191,
                    t_start=11,
                    t_end=201,
                ),
                _row(
                    0,
                    1,
                    n_match=199,
                    aln_len=200,
                    as_score=398,
                    cigar="199=1X",
                    q_start=2,
                    q_end=192,
                    t_start=22,
                    t_end=202,
                ),
                _row(
                    0,
                    2,
                    n_match=198,
                    aln_len=200,
                    as_score=396,
                    cigar="198=2X",
                    q_start=3,
                    q_end=193,
                    t_start=33,
                    t_end=203,
                ),
            ]
        ),
        n_templates=3,
    )
    batch = assign_block(hb, store=store, reads=_Reads(["r0"]), round_index=1)
    assert batch.column("template_row").to_pylist() == [2]
    assert batch.column("q_start").to_pylist() == [3]
    assert batch.column("t_start").to_pylist() == [33]
    assert batch.column("t_end").to_pylist() == [203]
    assert batch.column("offset_5p").to_pylist() == [30]
    assert batch.column("cigar").to_pylist() == ["198=2X"]


def test_round2_prefers_the_homopolymer_explanation():
    """The likelihood decides where AS ties — composed through the reducer."""
    hp = "ACGT" * 10 + "GGGGGG" + "ACGT" * 10
    flat = "ACGT" * 10 + "GATTAC" + "ACGT" * 10
    store = _store([hp, flat], weight=[1.0, 1.0])
    hb = scan_paf_block(
        _paf(
            [
                _row(
                    0,
                    0,
                    n_match=85,
                    aln_len=86,
                    as_score=164,
                    cigar="40=1D45=",
                    q_len=85,
                    q_end=85,
                    t_len=86,
                    t_end=86,
                ),
                _row(
                    0,
                    1,
                    n_match=84,
                    aln_len=85,
                    as_score=164,
                    cigar="40=1X44=",
                    q_len=85,
                    q_end=85,
                    t_len=85,
                    t_end=85,
                ),
            ]
        ),
        n_templates=2,
    )
    batch = assign_block(hb, store=store, reads=_Reads(["r0"]), round_index=2)
    assert batch.column("template_row").to_pylist() == [0]
    assert batch.column("logl").to_pylist()[0] is not None


def test_candidate_cap_hit_flags_a_truncated_pool():
    """A pool at the -N cap was truncated, so the ranking saw a subset."""
    seq = "ACGT" * 50
    store = _store([seq] * 4)
    rows = [
        _row(0, t, n_match=200, aln_len=200, as_score=400 - t, cigar="200=")
        for t in range(4)
    ]
    hb = scan_paf_block(_paf(rows), n_templates=4)
    batch = assign_block(
        hb, store=store, reads=_Reads(["r0"]), round_index=1, minimap2_n=3
    )
    assert batch.column("n_hits").to_pylist() == [4]
    assert batch.column("candidate_cap_hit").to_pylist() == [True]

    batch = assign_block(
        hb, store=store, reads=_Reads(["r0"]), round_index=1, minimap2_n=50
    )
    assert batch.column("candidate_cap_hit").to_pylist() == [False]


def test_many_reads_in_one_block_stay_independent():
    seq = "ACGT" * 50
    store = _store([seq, seq], replication=[1, 50])
    rows = []
    for read in range(5):
        rows.append(_row(read, 0, n_match=200, aln_len=200, as_score=400, cigar="200="))
        if read % 2 == 0:
            rows.append(
                _row(read, 1, n_match=198, aln_len=200, as_score=396, cigar="198=2X")
            )
    hb = scan_paf_block(_paf(rows), n_templates=2)
    batch = assign_block(
        hb, store=store, reads=_Reads([f"r{i}" for i in range(5)]), round_index=1
    )
    assert batch.num_rows == 5
    # Even reads see the better-replicated template and take it; odd ones do not.
    assert batch.column("template_row").to_pylist() == [1, 0, 1, 0, 1]
    assert batch.column("read_row").to_pylist() == [0, 1, 2, 3, 4]
    assert batch.column("read_id").to_pylist() == ["r0", "r1", "r2", "r3", "r4"]


@pytest.mark.parametrize("chunk", [61, 256, 1 << 16])
def test_streaming_is_independent_of_chunk_size(chunk):
    seq = "ACGT" * 50
    store = _store([seq, seq], replication=[1, 50])
    rows = []
    for read in range(20):
        rows.append(_row(read, 0, n_match=200, aln_len=200, as_score=400, cigar="200="))
        rows.append(
            _row(read, 1, n_match=198, aln_len=200, as_score=396, cigar="198=2X")
        )
    text = ("\n".join(rows) + "\n").encode()
    reads = _Reads([f"r{i}" for i in range(20)])

    batches = list(
        assign_blocks(
            iter_hit_blocks(
                [text[i : i + chunk] for i in range(0, len(text), chunk)],
                n_templates=2,
                block_bytes=chunk,
            ),
            store=store,
            reads=reads,
            round_index=1,
        )
    )
    table = pa.Table.from_batches(batches, schema=EM_ASSIGNMENT_TABLE)
    assert table.num_rows == 20
    assert table.column("read_row").to_pylist() == list(range(20))
    assert set(table.column("template_row").to_pylist()) == {1}
