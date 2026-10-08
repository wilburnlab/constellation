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

    def __init__(self, ids, samples=None, quality=None):
        self.read_id = pa.chunked_array([pa.array(ids, pa.string())])
        self.chunk_starts = np.array([0, len(ids)], dtype=np.int64)
        self.sample_id = np.asarray(
            samples if samples is not None else [0] * len(ids), dtype=np.int64
        )
        self.dorado_quality = np.asarray(
            quality if quality is not None else [30.0] * len(ids),
            dtype=np.float64,
        )

    def take_read_ids(self, rows):
        from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
            chunked_take,
        )

        return chunked_take(self.read_id, rows, self.chunk_starts).cast(pa.string())


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


# ── the runner's guards ───────────────────────────────────────────────


def test_a_multipart_index_is_refused():
    """Two things depend on a single index part, not one."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
        _check_single_index_part,
        _parse_size,
    )

    store = _store(["A" * 2000, "C" * 2000])
    _check_single_index_part(store, "16G")  # comfortably above 4 kb

    with pytest.raises(ValueError, match="multi-part index"):
        _check_single_index_part(store, "1K")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("16G", 16_000_000_000), ("4M", 4_000_000), ("500K", 500_000), ("123", 123)],
)
def test_index_size_grammar(text, expected):
    from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
        _parse_size,
    )

    assert _parse_size(text) == expected


def test_minimap2_flags_carry_the_load_bearing_options():
    from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
        TEMPLATE_MINIMAP2_ARGS,
    )

    flags = list(TEMPLATE_MINIMAP2_ARGS)
    # --eqx: without it cg:Z is all M, so mismatches fold into matches and
    # both the identity gate and the likelihood read every hit as perfect.
    assert "--eqx" in flags
    # -p 0.05: the stock 0.8 hides exactly the nested / 5'-truncated
    # proteoforms this pipeline exists to separate.
    assert flags[flags.index("-p") + 1] == "0.05"
    assert "--secondary=yes" in flags
    assert "-c" in flags
    # -N is NOT baked in: it is a per-run correctness parameter, since the
    # pool must contain every template within p_floor.
    assert "-N" not in flags


# ── the per-read floor, and the record of why ─────────────────────────


def test_floor_per_read_eases_with_quality_and_never_rises():
    from constellation.sequencing.transcriptome.cluster.denovo.em.scheduler import (
        floor_per_read,
    )

    q = np.array([30.0, 20.0, 13.0, np.nan, -1.0])
    flat = floor_per_read(q, p_floor=0.97, quality_scale=None)
    assert flat.tolist() == [0.97] * 5
    eased = floor_per_read(q, p_floor=0.97, quality_scale=1.5)
    assert eased[0] == 0.97, "Q30: 1 - 1.5e-3 is above the flat floor"
    assert eased[1] == pytest.approx(0.985) or eased[1] == 0.97
    assert eased[1] == 0.97, "min() with the flat floor: it never rises"
    assert eased[2] == pytest.approx(1 - 1.5 * 10**-1.3)
    assert eased[3] == 0.97 and eased[4] == 0.97, "no quality: flat"


def test_the_minimap2_reducer_records_identity_and_the_reason():
    """One clean read, one whose only hit is under the floor: the winner
    carries its identity and length, the loser the best identity it had and
    `below_floor`."""
    seq = "ACGTACGTAC" * 20
    store = _store([seq])
    hb = scan_paf_block(
        _paf(
            [
                _row(0, 0, n_match=198, aln_len=200, as_score=388, cigar="200="),
                _row(1, 0, n_match=188, aln_len=200, as_score=328, cigar="200="),
            ]
        ),
        n_templates=1,
    )
    batch = assign_block(hb, store=store, reads=_Reads(["a", "b"]), round_index=1)
    got = batch.to_pylist()
    assert got[0]["identity"] == pytest.approx(0.99)
    assert got[0]["aligned_len"] == 200
    assert got[0]["unassigned_reason"] is None
    assert got[1]["template_id"] == -1
    assert got[1]["identity"] == pytest.approx(0.94)
    assert got[1]["aligned_len"] is None
    assert got[1]["unassigned_reason"] == "below_floor"


def test_the_minimap2_reducer_takes_a_per_read_floor():
    """The same 0.94 hit: rejected at the flat floor, admitted once the
    read's Q13 eases its own floor below it."""
    seq = "ACGTACGTAC" * 20
    store = _store([seq])
    rows = [_row(0, 0, n_match=188, aln_len=200, as_score=328, cigar="200=")]
    reads = _Reads(["a"], quality=[13.0])
    flat = assign_block(
        scan_paf_block(_paf(rows), n_templates=1),
        store=store,
        reads=reads,
        round_index=1,
    )
    assert flat.to_pylist()[0]["template_id"] == -1
    eased = assign_block(
        scan_paf_block(_paf(rows), n_templates=1),
        store=store,
        reads=reads,
        round_index=1,
        p_floor_quality_scale=1.5,
    )
    assert eased.to_pylist()[0]["template_id"] == 100


def test_the_minimap2_reducer_runs_round_one_on_the_band_rule_when_asked():
    """Replication-first hands both reads to the 500-replication template;
    the band rule keeps the read whose best match is 1.5 points better."""
    seq = "ACGTACGTAC" * 20
    store = _store([seq, seq], replication=[500, 5])
    rows = [
        _row(0, 0, n_match=195, aln_len=200, as_score=370, cigar="195=5X"),
        _row(0, 1, n_match=198, aln_len=200, as_score=388, cigar="198=2X"),
    ]
    old = assign_block(
        scan_paf_block(_paf(rows), n_templates=2),
        store=store,
        reads=_Reads(["r0"]),
        round_index=1,
    )
    assert old.column("template_row").to_pylist() == [0]
    new = assign_block(
        scan_paf_block(_paf(rows), n_templates=2),
        store=store,
        reads=_Reads(["r0"]),
        round_index=1,
        round1_rule="identity_band",
    )
    assert new.column("template_row").to_pylist() == [1]
    assert new.column("identity").to_pylist()[0] == pytest.approx(0.99)


def test_round_one_measures_length_against_the_read_not_the_aligned_stretch():
    """A local hit covering all of a short template is not a length match
    for a read three times as long. With equal identity, replication and
    seed quality inside the band, the template that explains the whole read
    wins — the aligned length stood in for the read's once, and the two
    tied (review of 91e7c69)."""
    store = _store(["A" * 300, "A" * 1000], replication=[7, 7])
    rows = [
        _row(
            0, 0, n_match=297, aln_len=300, as_score=560, cigar="297=3X",
            q_len=1000, q_end=300, t_len=300, t_end=300,
        ),
        _row(
            0, 1, n_match=990, aln_len=1000, as_score=1900, cigar="990=10X",
            q_len=1000, q_end=1000, t_len=1000, t_end=1000,
        ),
    ]  # fmt: skip
    out = assign_block(
        scan_paf_block(_paf(rows), n_templates=2),
        store=store,
        reads=_Reads(["r0"]),
        round_index=1,
        round1_rule="identity_band",
    )
    assert out.column("template_row").to_pylist() == [1]
    assert out.column("aligned_len").to_pylist() == [1000]


def test_a_sliver_is_not_the_identity_an_unassigned_read_reports():
    """The identity column of an unassigned read is what a floor is
    calibrated from, so it reads only alignments that COUNTED. A sliver at
    1.0 beside a counted 0.95 reports 0.95 / below_floor; slivers alone
    report nothing / short_placement."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
        _identity_columns,
    )

    n_match = np.array([5, 190, 3, 0], dtype=np.int64)
    aln_len = np.array([5, 200, 3, 0], dtype=np.int64)
    placed = np.array([False, True, False, False])
    ptr = np.array([0, 2, 3, 4], dtype=np.int64)
    has = np.zeros(3, dtype=bool)
    ident, length, reason = _identity_columns(
        has, np.zeros(3, dtype=np.int64), n_match, aln_len, ptr, placed=placed
    )
    assert ident.to_pylist() == [pytest.approx(0.95), None, None]
    assert reason.to_pylist() == ["below_floor", "short_placement", "no_alignment"]
    assert length.to_pylist() == [None, None, None]
    # With no guard in play every produced alignment counts, as before.
    ident, _, reason = _identity_columns(
        has, np.zeros(3, dtype=np.int64), n_match, aln_len, ptr
    )
    assert ident.to_pylist() == [pytest.approx(1.0), pytest.approx(1.0), None]
    assert reason.to_pylist() == ["below_floor", "below_floor", "no_alignment"]
