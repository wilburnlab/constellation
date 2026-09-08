"""PAF streaming + the ORF-anchored E-step's band and tie-break.

Everything here runs without minimap2 on ``$PATH``: the parser is fed
hand-written PAF and the ranking is a pure function over hit arrays. That is
deliberate — the band's operating point is the thing most likely to be
re-tuned, and it should be testable without a 500 MB index.
"""

from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
import pytest

from constellation.sequencing.readers.paf import (
    PAF_RECORD_SCHEMA,
    iter_paf_batches,
    paf_to_alignment_table,
    read_paf,
)
from constellation.sequencing.transcriptome.cluster.denovo.orfem.estep import (
    TEMPLATE_MINIMAP2_ARGS,
    TemplateSet,
    assign_reads,
    rank_banded_candidates,
)


def _paf(*lines: str) -> bytes:
    return ("\n".join(lines) + "\n").encode()


def _batches(raw: bytes, chunk_bytes: int = 1 << 20):
    return list(iter_paf_batches(io.BytesIO(raw), chunk_bytes=chunk_bytes))


def _templates(n=3, *, node_weight=None, aa=None, columns=None):
    return TemplateSet(
        name=[f"t{i}" for i in range(n)],
        template_id=np.arange(n, dtype=np.int64),
        sequence=["ACGT" * 100] * n,
        orf_start=np.full(n, 100, dtype=np.int64),
        orf_end=np.full(n, 340, dtype=np.int64),
        orf_aa_length=np.asarray(aa if aa is not None else [80] * n, dtype=np.int64),
        node_weight=np.asarray(
            node_weight if node_weight is not None else [1.0] * n, dtype=np.float64
        ),
        declared_variants=columns or [np.empty(0, dtype=np.int64) for _ in range(n)],
    )


# ── PAF parsing ───────────────────────────────────────────────────────


def test_paf_columns_and_tags():
    raw = _paf(
        "r1\t1500\t10\t1490\t+\tt0\t1600\t20\t1500\t1450\t1480\t60\ttp:A:P\tAS:i:2900\tNM:i:30\tcg:Z:100=1X1379=",
        "r1\t1500\t10\t1490\t+\tt1\t1600\t20\t1500\t1400\t1480\t0\ttp:A:S\tAS:i:2640\tcg:Z:1480=",
        "r2\t900\t0\t900\t-\tt0\t1600\t100\t1000\t880\t900\t60\ttp:A:P\tAS:i:1700",
    )
    tbl = pa.Table.from_batches(_batches(raw), schema=PAF_RECORD_SCHEMA)
    assert tbl.column("q_name").to_pylist() == ["r1", "r1", "r2"]
    assert tbl.column("t_name").to_pylist() == ["t0", "t1", "t0"]
    assert tbl.column("as_score").to_pylist() == [2900, 2640, 1700]
    assert tbl.column("tp").to_pylist() == ["P", "S", "P"]
    assert tbl.column("strand").to_pylist() == ["+", "+", "-"]
    # Absent tags are null, not zero.
    assert tbl.column("nm").to_pylist() == [30, None, None]
    assert tbl.column("cigar").to_pylist() == ["100=1X1379=", "1480=", None]


@pytest.mark.parametrize("chunk", [7, 64, 4096, 1 << 20])
def test_paf_chunk_boundary_invariance(chunk):
    """A record must never be split by a read boundary."""
    raw = _paf(
        *[
            f"r{i}\t900\t0\t900\t+\tt{i % 3}\t1600\t0\t900\t880\t900\t60\tAS:i:{1700 + i}\tcg:Z:900="
            for i in range(40)
        ]
    )
    ref = pa.Table.from_batches(_batches(raw), schema=PAF_RECORD_SCHEMA)
    got = pa.Table.from_batches(_batches(raw, chunk), schema=PAF_RECORD_SCHEMA)
    assert got.equals(ref)
    assert got.num_rows == 40


def test_paf_rejects_short_record():
    with pytest.raises(ValueError, match="malformed PAF record"):
        _batches(b"r1\t1500\t10\n")


def test_paf_handles_missing_trailing_newline(tmp_path):
    p = tmp_path / "x.paf"
    p.write_text("r1\t900\t0\t900\t+\tt0\t1600\t0\t900\t880\t900\t60\tAS:i:1700")
    assert read_paf(p).num_rows == 1


def test_paf_to_alignment_table_round_trips_query_clipping():
    from constellation.sequencing.align.cigar import query_start_from_cigar

    raw = _paf(
        "r1\t1000\t25\t980\t+\tt0\t1600\t20\t975\t950\t955\t60\ttp:A:P\tAS:i:1900\tcg:Z:955="
    )
    aln = paf_to_alignment_table(read_paf(io.BytesIO(raw)))
    assert aln.column("ref_start").to_pylist() == [20]
    assert aln.column("is_secondary").to_pylist() == [False]
    cig = aln.column("cigar_string").to_pylist()[0]
    assert cig == "25S955=20S"
    assert query_start_from_cigar(cig) == 25


# ── the band ──────────────────────────────────────────────────────────


def test_absolute_band_admits_a_near_tie_and_excludes_a_far_hit():
    """AS 3000 / 2960 / 2900. A 40-point absolute band admits the second and
    excludes the third; a 0.99 *ratio* band (30 points here) admits neither,
    and on a 400-nt transcript the same ratio would be ~5 points."""
    as_score = np.array([3000, 2960, 2900], dtype=np.int64)
    tmpl = np.array([0, 1, 2], dtype=np.int64)
    ptr = np.array([0, 3], dtype=np.int64)
    t = _templates()

    _w, n_banded, delta, in_band = rank_banded_candidates(
        as_score, tmpl, ptr, node_weight=t.node_weight,
        orf_aa_length=t.orf_aa_length, band_abs=40,
    )
    assert in_band.tolist() == [True, True, False]
    assert int(n_banded[0]) == 2
    assert delta.tolist() == [0, 40, 100]

    _w, n20, _d20, in20 = rank_banded_candidates(
        as_score, tmpl, ptr, node_weight=t.node_weight,
        orf_aa_length=t.orf_aa_length, band_abs=20,
    )
    assert in20.tolist() == [True, False, False]
    assert int(n20[0]) == 1


def test_band_is_absolute_not_a_ratio():
    """The point of an absolute band: the same slack on a short and a long
    transcript. A ratio band's slack scales with score, which is too tight on
    a short transcript and far too loose on a long one."""
    t = _templates(2)
    for best in (800, 12000):  # a 400-nt and a 6 kb transcript
        as_score = np.array([best, best - 40], dtype=np.int64)
        ptr = np.array([0, 2], dtype=np.int64)
        _w, _n, _d, in_band = rank_banded_candidates(
            np.asarray(as_score), np.array([0, 1]), ptr,
            node_weight=t.node_weight, orf_aa_length=t.orf_aa_length, band_abs=40,
        )
        assert in_band.tolist() == [True, True], "40 points is 40 points"


def test_tie_break_group_size_then_orf_length_then_score():
    ptr = np.array([0, 3], dtype=np.int64)
    scores = np.array([3000, 3000, 3000], dtype=np.int64)
    tmpl = np.array([0, 1, 2], dtype=np.int64)

    # Node weight decides first, even against the higher-indexed template.
    t = _templates(node_weight=[1.0, 68.0, 1.0])
    w, *_ = rank_banded_candidates(
        scores, tmpl, ptr, node_weight=t.node_weight, orf_aa_length=t.orf_aa_length
    )
    assert int(w[0]) == 1

    # Equal weight → longer ORF wins (the least-broken read of a transcript).
    t = _templates(node_weight=[5.0, 5.0, 5.0], aa=[80, 80, 300])
    w, *_ = rank_banded_candidates(
        scores, tmpl, ptr, node_weight=t.node_weight, orf_aa_length=t.orf_aa_length
    )
    assert int(w[0]) == 2

    # Equal weight and length → higher score wins.
    t = _templates(node_weight=[5.0] * 3, aa=[80] * 3)
    w, *_ = rank_banded_candidates(
        np.array([2990, 3000, 2980], dtype=np.int64), tmpl, ptr,
        node_weight=t.node_weight, orf_aa_length=t.orf_aa_length,
    )
    assert int(w[0]) == 1


def test_ranking_handles_many_reads_at_once():
    """The ranker is vectorized over a whole batch, so group offsets have to
    be right — an off-by-one here silently assigns reads to their neighbour's
    best template."""
    rng = np.random.default_rng(5)
    sizes = rng.integers(1, 6, size=200)
    ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    n = int(ptr[-1])
    as_score = rng.integers(1000, 4000, size=n).astype(np.int64)
    tmpl = rng.integers(0, 3, size=n).astype(np.int64)
    t = _templates(node_weight=[1.0, 1.0, 1.0], aa=[80, 80, 80])
    w, n_banded, delta, _ = rank_banded_candidates(
        as_score, tmpl, ptr, node_weight=t.node_weight,
        orf_aa_length=t.orf_aa_length, band_abs=0,
    )
    for g in range(200):
        lo, hi = int(ptr[g]), int(ptr[g + 1])
        assert lo <= int(w[g]) < hi, "winner escaped its own read's hits"
        assert as_score[int(w[g])] == as_score[lo:hi].max()
        assert delta[lo:hi].min() == 0


# ── assignment ────────────────────────────────────────────────────────


def test_assign_reads_drops_antisense_hits():
    """On a '-' hit PAF reports q_start on the forward query while the CIGAR
    is against the reverse complement, so the offsets are unusable."""
    raw = _paf(
        "r1\t900\t0\t900\t-\tt0\t1600\t0\t900\t880\t900\t60\tAS:i:1700\tcg:Z:900=",
        "r2\t900\t0\t900\t+\tt1\t1600\t0\t900\t880\t900\t60\tAS:i:1700\tcg:Z:900=",
    )
    aln, _ = assign_reads(_batches(raw), _templates())
    assert aln.column("read_id").to_pylist() == ["r2"]
    aln, _ = assign_reads(_batches(raw), _templates(), allow_antisense=True)
    assert sorted(aln.column("read_id").to_pylist()) == ["r1", "r2"]


def test_assign_reads_records_offset_and_contest():
    raw = _paf(
        "r1\t900\t12\t900\t+\tt0\t1600\t40\t930\t880\t900\t60\tAS:i:1700\tcg:Z:888=",
        "r1\t900\t12\t900\t+\tt1\t1600\t40\t930\t870\t900\t0\tAS:i:1680\tcg:Z:888=",
    )
    aln, _ = assign_reads(_batches(raw), _templates(node_weight=[9.0, 1.0, 1.0]))
    row = aln.to_pylist()[0]
    assert row["template_id"] == 0
    assert row["n_banded"] == 2  # 20 points apart, inside the 40-point band
    assert row["as_delta"] == 0
    assert row["offset_5p"] == 40 - 12
    assert row["weight"] == pytest.approx(1.0)


def test_fractional_ties_split_by_node_weight_and_sum_to_one():
    raw = _paf(
        "r1\t900\t0\t900\t+\tt0\t1600\t0\t900\t880\t900\t60\tAS:i:1700\tcg:Z:900=",
        "r1\t900\t0\t900\t+\tt1\t1600\t0\t900\t880\t900\t0\tAS:i:1700\tcg:Z:900=",
    )
    t = _templates(node_weight=[30.0, 10.0, 1.0])
    aln, _ = assign_reads(_batches(raw), t, tie_resolution="fractional")
    w = aln.column("weight").to_pylist()
    assert len(w) == 2
    assert sum(w) == pytest.approx(1.0)
    assert sorted(w) == pytest.approx([0.25, 0.75])
    # Rank mode gives the whole read to the heavier node instead.
    aln, _ = assign_reads(_batches(raw), t, tie_resolution="rank")
    assert aln.num_rows == 1
    assert aln.column("template_id").to_pylist() == [0]


def test_read_groups_survive_a_batch_boundary():
    """minimap2 emits a query's hits contiguously, so a group can straddle a
    batch boundary but never appear twice."""
    raw = _paf(
        *[
            f"r{i // 4}\t900\t0\t900\t+\tt{i % 3}\t1600\t0\t900\t880\t900\t60\tAS:i:{2000 - i}\tcg:Z:900="
            for i in range(40)
        ]
    )
    for chunk in (32, 128, 1 << 20):
        aln, _ = assign_reads(_batches(raw, chunk), _templates())
        assert aln.num_rows == 10, f"chunk {chunk} lost or duplicated a read"
        assert aln.column("read_id").to_pylist() == [f"r{i}" for i in range(10)]


def test_allele_string_reads_declared_columns():
    """A read carrying a 1-nt deletion at a declared column reports '-'; a
    read that does not reach the column reports '.'."""
    cols = [np.array([500], dtype=np.int64)] * 3
    t = _templates(columns=cols)
    raw = _paf(
        # spans the column, with a deletion exactly at template position 500
        "del\t900\t0\t899\t+\tt0\t1600\t100\t1000\t880\t900\t60\tAS:i:1700\tcg:Z:400=1D499=",
        # spans the column cleanly
        "ref\t900\t0\t900\t+\tt0\t1600\t100\t1000\t880\t900\t60\tAS:i:1700\tcg:Z:900=",
        # stops well short of it
        "short\t200\t0\t200\t+\tt0\t1600\t100\t300\t195\t200\t60\tAS:i:400\tcg:Z:200=",
    )
    seqs = {"del": "A" * 899, "ref": "C" * 900, "short": "G" * 200}
    aln, _ = assign_reads(_batches(raw), t, read_seq=seqs)
    got = dict(
        zip(aln.column("read_id").to_pylist(), aln.column("allele_string").to_pylist())
    )
    assert got["del"] == "-"
    assert got["ref"] == "C"
    assert got["short"] == "."


def test_non_exclusive_coverage_is_emitted_for_every_banded_hit():
    raw = _paf(
        "r1\t900\t0\t900\t+\tt0\t1600\t100\t900\t880\t900\t60\tAS:i:1700\tcg:Z:800=",
        "r1\t900\t0\t900\t+\tt1\t1600\t100\t900\t870\t900\t0\tAS:i:1690\tcg:Z:800=",
        "r1\t900\t0\t900\t+\tt2\t1600\t100\t900\t500\t900\t0\tAS:i:900\tcg:Z:800=",
    )
    aln, cov = assign_reads(_batches(raw), _templates(), emit_coverage=True)
    assert aln.num_rows == 1
    # Only the two banded hits get support rows; the 800-point-worse one does not.
    assert cov.num_rows == 2
    assert cov.column("is_assigned").to_pylist().count(True) == 1
    assert all(0.0 <= v <= 1.0 for v in cov.column("orf_cov_frac").to_pylist())


def test_minimap2_flags_carry_the_load_bearing_options():
    """`-p 0.05` and a large `-N` are what make a nested template reportable
    at all; `--eqx` is what keeps mismatches out of the match count."""
    args = TEMPLATE_MINIMAP2_ARGS
    assert "--eqx" in args
    assert "--secondary=yes" in args
    assert args[args.index("-p") + 1] == "0.05"
    assert int(args[args.index("-N") + 1]) >= 50
