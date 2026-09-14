"""Candidate column selection for the covariance-driven M-step.

Candidacy is permission to be *tested* for covariance — not retention. So
these tests pin what gets in and, just as importantly, that the things the
old M-step over-called (homopolymers, terminal ramps, deep-but-tiny minor
alleles) are either still admitted here and killed later, or excluded for a
reason that is stated rather than incidental.
"""

from __future__ import annotations

import numpy as np
import pytest

edlib = pytest.importorskip("edlib")

from constellation.sequencing.transcriptome.cluster.denovo.consensus import (  # noqa: E402
    COL_EXT_5P,
    MemberSpec,
    frame_consensus,
)
from constellation.sequencing.transcriptome.cluster.denovo.orfem.columns import (  # noqa: E402
    ROUTE_ALLELIC,
    ROUTE_COVERAGE,
    candidate_columns,
    column_stats,
)
from constellation.sequencing.transcriptome.cluster.denovo.variants import (  # noqa: E402
    ErrorModel,
)


def _rand(rng, n):
    return "".join(rng.choice(list("ACGT"), n))


def _spec(frame, member, weight=1.0, member_id=-1):
    short, long = (frame, member) if len(frame) <= len(member) else (member, frame)
    a = edlib.align(short, long, mode="HW", task="path")
    assert a["editDistance"] >= 0, "fixture members must align"
    return MemberSpec(
        member_seq=member,
        weight=weight,
        cigar=a["cigar"],
        centroid_is_query=(frame == short),
        ref_start=a["locations"][0][0],
        member_id=member_id,
    )


def _build(frame, members):
    """members = [(sequence, weight), ...] → (ConsensusResult, n_assigned)."""
    specs = [_spec(frame, m, w, i) for i, (m, w) in enumerate(members)]
    cres = frame_consensus(frame, specs)
    return cres, float(sum(w for _m, w in members))


def _stats(frame, members, **kw):
    cres, n = _build(frame, members)
    return cres, column_stats(cres, n_assigned=n, **kw)


# ── the allelic route: effect size and significance in opposite regimes ──


def test_a_clean_minor_allele_is_admitted():
    rng = np.random.default_rng(11)
    truth = _rand(rng, 300)
    alt = truth[:150] + ("A" if truth[150] != "A" else "C") + truth[151:]
    cres, stats = _stats(truth, [(truth, 70.0), (alt, 30.0)])
    cand = candidate_columns(stats)
    assert 150 in cand.columns.tolist()
    v = cand.columns.tolist().index(150)
    assert cand.route[v] & ROUTE_ALLELIC
    assert cand.effect[v] == pytest.approx(0.30, abs=0.01)


def test_one_read_at_depth_20_is_not_a_5_percent_variant():
    """Effect size alone fails shallow: at 20 reads a single basecaller error
    *is* a 5% minor allele. Both gates reject it, which is the point — one
    read is p = 0.058 under the substitution null, and ``a_min`` refuses to
    rank it at all."""
    rng = np.random.default_rng(13)
    truth = _rand(rng, 300)
    alt = truth[:150] + ("A" if truth[150] != "A" else "C") + truth[151:]
    _cres, stats = _stats(truth, [(truth, 19.0), (alt, 1.0)])
    assert stats.minor_frac[150] == pytest.approx(0.05), "clears f_min"
    assert 150 not in candidate_columns(stats).columns.tolist()
    assert 150 not in candidate_columns(stats, a_min=1.0).columns.tolist()

    # The same 5% with the depth to judge it is admitted.
    _cres2, deep = _stats(truth, [(truth, 475.0), (alt, 25.0)])
    assert 150 in candidate_columns(deep).columns.tolist()


def test_overdispersion_is_the_depth_penalty():
    """Significance alone fails deep. A 1% minor at depth ~1,500 is
    significant under a point binomial and not under rho = 0.01. ``f_min``
    is lowered here so the effect gate does not mask which gate fired."""
    rng = np.random.default_rng(17)
    truth = _rand(rng, 300)
    alt = truth[:150] + ("A" if truth[150] != "A" else "C") + truth[151:]
    _cres, stats = _stats(truth, [(truth, 1510.0), (alt, 15.0)])
    assert stats.depth[150] == pytest.approx(1525.0)
    kw = {"f_min": 0.005}
    assert 150 in candidate_columns(stats, overdispersion=0.0, **kw).columns.tolist()
    assert (
        150 not in candidate_columns(stats, overdispersion=0.01, **kw).columns.tolist()
    )


def test_overdispersion_does_not_kill_a_real_minority_at_depth():
    """rho caps effective depth; it must not cap effect size. A 20% minor on
    a 1,500-read template is the proteoform this pipeline exists to keep."""
    rng = np.random.default_rng(19)
    truth = _rand(rng, 300)
    alt = truth[:150] + ("A" if truth[150] != "A" else "C") + truth[151:]
    _cres, stats = _stats(truth, [(truth, 1200.0), (alt, 300.0)])
    assert 150 in candidate_columns(stats, overdispersion=0.01).columns.tolist()


def test_a_five_read_minority_is_kept_by_count_not_by_fraction_of_template():
    """Replaces ``min_haplotype_frac``, whose 1% default would have discarded
    a 5-read proteoform on a 20,000-read template.

    The honest limit, measured rather than asserted: at depth 20,000 the
    substitution null already expects ~60 erroneous reads per column, so five
    is **not** recoverable at a column level by any threshold — dropping
    ``f_min`` to 1e-4 does not rescue it, the FDR does not either, and no part
    of this design claims otherwise. What the fraction floor destroyed, and
    this does not, is the same five reads on a template whose depth can
    actually resolve them."""
    rng = np.random.default_rng(23)
    truth = _rand(rng, 300)
    alt = truth[:150] + ("A" if truth[150] != "A" else "C") + truth[151:]

    _cres, deep = _stats(truth, [(truth, 19_995.0), (alt, 5.0)])
    assert deep.minor_frac[150] == pytest.approx(2.5e-4)
    assert 150 not in candidate_columns(deep, f_min=1e-4).columns.tolist()
    assert deep.n_minor[150] * 1.0 < deep.eps[150] * deep.depth[150], (
        "five reads is below the error floor the depth itself creates"
    )

    # 5 of 100 — the fraction floor's actual victim — is admitted.
    _cres2, shallow = _stats(truth, [(truth, 95.0), (alt, 5.0)])
    assert shallow.minor_frac[150] == pytest.approx(0.05)
    assert 150 in candidate_columns(shallow).columns.tolist()

    # And the operating point in between, pinned because it is the one a
    # reader will be surprised by: rho = 0.01 caps effective depth at ~100, so
    # the SAME five reads at depth 200 are p = 0.025 and do not clear q = 0.01.
    # rho is a per-round argument for exactly this reason — it is the
    # regularizer, not a fact about the data.
    _cres3, mid = _stats(truth, [(truth, 195.0), (alt, 5.0)])
    assert 150 not in candidate_columns(mid).columns.tolist()
    assert 150 in candidate_columns(mid, overdispersion=0.001).columns.tolist()


def test_a_minority_insertion_column_is_admitted():
    """Gap-major by construction. Nothing here may gate on 'is the major
    allele a gap' — that asymmetry is what the planned column space removed."""
    rng = np.random.default_rng(29)
    truth = _rand(rng, 400)
    frame = truth[:200] + truth[201:]  # the template LACKS the base
    _cres, stats = _stats(frame, [(truth, 25.0), (frame, 75.0)])
    ins = np.flatnonzero((stats.major == 4) & (stats.minor < 4) & (stats.depth > 0))
    assert ins.size == 1, "one planned insertion column"
    v = int(ins[0])
    assert stats.minor_frac[v] == pytest.approx(0.25)
    assert v in candidate_columns(stats).columns.tolist()


def test_a_homopolymer_is_admitted_here_and_dies_later():
    """Candidacy is not retention. The homopolymer over-calling the old
    M-step suffered from is fixed by covariance, not by a special case in
    selection — so it must still be admitted at this layer."""
    rng = np.random.default_rng(31)
    body, tail = _rand(rng, 150), _rand(rng, 150)
    frame = body + "GGGGG" + tail
    short = body + "GGGG" + tail
    _cres, stats = _stats(frame, [(frame, 78.0), (short, 22.0)])
    hp = np.flatnonzero(stats.is_hp & (stats.minor_frac > 0.1))
    assert hp.size >= 1
    assert set(hp.tolist()) & set(candidate_columns(stats).columns.tolist())


# ── the coverage route ──


def _two_start_modes(seed=37, n_long=10.0, n_short=10.0, start=400, length=600):
    rng = np.random.default_rng(seed)
    truth = _rand(rng, length)
    return _stats(truth, [(truth, n_long), (truth[start:], n_short)])


def test_coverage_boundary_is_admitted_and_collapses_to_one_clique():
    cres, stats = _two_start_modes()
    cand = candidate_columns(stats)
    cov = cand.columns[(cand.route & ROUTE_COVERAGE).astype(bool)]
    assert cov.size == 1, "one representative for the whole boundary block"
    assert cand.run_len[cand.columns.tolist().index(int(cov[0]))] > 300
    assert cand.n_collapsed > 300
    # It really is a coverage split: half the reads are not there at all.
    assert stats.uncov_frac[int(cov[0])] == pytest.approx(0.5, abs=0.05)
    del cres


def test_run_collapse_is_exact_and_stops_at_a_span_boundary():
    """Two blocks separated by a third start mode must NOT fold together —
    their uncovered read-sets differ, so they partition the reads differently.
    """
    rng = np.random.default_rng(41)
    truth = _rand(rng, 900)
    _cres, stats = _stats(
        truth, [(truth, 10.0), (truth[300:], 10.0), (truth[600:], 10.0)]
    )
    cand = candidate_columns(stats)
    cov = cand.columns[(cand.route & ROUTE_COVERAGE).astype(bool)]
    assert cov.size == 2, "two boundaries, two representatives"
    assert np.all(np.diff(np.sort(cov)) > 100)


def test_the_two_coverage_denominators_catch_different_structures():
    """Measured, not assumed: each denominator misses the other's case, so
    the route is their union.

    A fused template whose genes are covered by disjoint read sets is a
    *step*, not a ramp. The local statistic reads 0 inside each gene (every
    local read covers it) and 1.0 in the gap, so it never lands in the band —
    it admitted **one** column of 4,258 on the synthetic fused fixture. The
    global fraction sits at 0.5 across the whole of gene A, which is the
    partition. Fusion is the headline failure the route exists for.
    """
    rng = np.random.default_rng(43)
    truth = _rand(rng, 1200)
    members = [(truth[:500], 10.0), (truth[700:], 10.0)]
    _cres, stats = _stats(truth, members)

    f_min = 0.02
    in_band = lambda x: (x >= f_min) & (x <= 1 - f_min)  # noqa: E731
    glob, loc = in_band(stats.uncov_frac), in_band(stats.uncov_frac_local)
    assert glob[:500].all(), "gene A is uncovered by exactly the gene-B reads"
    assert not loc[:400].any(), "…and the local view sees a step, not a ramp"
    cand = candidate_columns(stats)
    assert (cand.route & ROUTE_COVERAGE).any()


def test_run_collapse_is_what_makes_a_fused_template_affordable():
    """Without collapse the global route admits nearly every column of a
    fused template and the cap binds immediately; with it, a gene block is
    one representative. Measured on the 4,000 nt / 400-read synthetic fused
    fixture: 4,058 candidates capped to 512 with ``Sigma k_i^2 = 5.2e7``,
    versus 3 representatives and 1,017 — and the cap never binds."""
    rng = np.random.default_rng(79)
    truth = _rand(rng, 1200)
    members = [(truth[:500], 10.0), (truth[700:], 10.0)]
    _cres, stats = _stats(truth, members)

    raw = candidate_columns(stats, collapse_runs=False, max_columns=None)
    assert raw.columns.size >= 1000
    assert int(raw.run_len.max()) == 1

    kept = candidate_columns(stats)
    assert kept.columns.size <= 4, "one representative per coverage block"
    assert int(kept.run_len.max()) > 400
    assert kept.n_collapsed == raw.columns.size - kept.columns.size
    assert kept.n_capped == 0


def test_a_degradation_ramp_has_uncorrelated_boundaries():
    """The route admits columns; it is covariance that rejects the ramp. What
    this layer must guarantee is that a ramp is NOT collapsed into one
    perfectly-correlated representative — every read ends somewhere else, so
    every column is its own boundary."""
    rng = np.random.default_rng(47)
    truth = _rand(rng, 600)
    members = [(truth[: 300 + 10 * i], 2.0) for i in range(30)]
    _cres, stats = _stats(truth, members)
    cand = candidate_columns(stats)
    cov_len = cand.run_len[(cand.route & ROUTE_COVERAGE).astype(bool)]
    assert cov_len.size > 5, "many separate representatives, not one block"
    assert int(cov_len.max()) <= 10, "no long perfectly-correlated run"


def test_terminal_columns_are_not_excluded():
    """``in_core`` / the terminal-column exclusion are deleted from this path.
    A 5' extension block is a length difference, which is the point."""
    rng = np.random.default_rng(53)
    truth = _rand(rng, 400)
    frame = truth[60:]
    cres, stats = _stats(frame, [(truth, 10.0), (frame, 10.0)])
    ext = np.flatnonzero(cres.column_kind == COL_EXT_5P)
    assert ext.size == 60
    assert set(ext.tolist()) & set(candidate_columns(stats).columns.tolist())


# ── the error model, the floor, and the homopolymer context ──


def test_eps_floor_is_a_floor_not_a_replacement():
    rng = np.random.default_rng(59)
    body, tail = _rand(rng, 100), _rand(rng, 100)
    frame = body + "GGGGG" + tail
    short = body + "GGGG" + tail
    measured = ErrorModel(eps_hp0=0.22, hp_ref_len=5, hp_slope=0.0, hp_max=0.35)
    cres, _ = _build(frame, [(frame, 78.0), (short, 22.0)])
    s0 = column_stats(cres, model=measured, n_assigned=100.0)
    s1 = column_stats(cres, model=measured, n_assigned=100.0, eps_floor=0.05)
    hp = np.flatnonzero(s0.is_hp)
    sub = np.flatnonzero(~s0.is_gap)
    assert s0.eps[hp].max() == pytest.approx(0.22)
    assert s1.eps[hp].max() == pytest.approx(0.22), "context survives the floor"
    assert s0.eps[sub].max() == pytest.approx(0.003)
    assert s1.eps[sub].max() == pytest.approx(0.05), "…and the floor binds below"


def test_homopolymer_context_spans_a_gap_winning_insertion_column():
    """``cres.frame`` writes '-' at a losing insertion column, which breaks
    the run exactly where the homopolymer null has to judge it."""
    rng = np.random.default_rng(61)
    body, tail = _rand(rng, 100), _rand(rng, 100)
    frame = body + "GGGGG" + tail  # template carries 5
    longer = body + "GGGGGG" + tail  # a minority inserts a 6th
    cres, stats = _stats(frame, [(frame, 80.0), (longer, 20.0)])
    ins = np.flatnonzero((cres.column_kind != 0) | (stats.major == 4))
    ins = np.array([c for c in ins if stats.minor[c] == 2])  # a G was inserted
    assert ins.size >= 1
    assert stats.hp_run[int(ins[0])] >= 6, "the run is not cut by the gap"
    assert stats.is_hp[int(ins[0])]


def test_max_columns_ranks_by_effect_size_not_p_value():
    """Ranking by p-value is what filled the old 64-column budget with
    homopolymers: significance grows with depth and says nothing about how
    much of the template disagrees."""
    rng = np.random.default_rng(67)
    truth = _rand(rng, 400)
    flip = lambda s, p: s[:p] + ("A" if s[p] != "A" else "C") + s[p + 1 :]  # noqa: E731
    # A deep 5' half (1,040 reads) carries a 3.8% minor at column 100; a
    # shallow 3' half (20 reads) carries a 50% minor at column 300.
    members = [
        (truth[:200], 1000.0),
        (flip(truth[:200], 100), 40.0),
        (truth[200:], 10.0),
        (flip(truth, 300)[200:], 10.0),
    ]
    _cres, stats = _stats(truth, members)
    full = candidate_columns(stats, overdispersion=0.0)
    cols = full.columns.tolist()
    assert {100, 300} <= set(cols)
    assert full.effect[cols.index(300)] > full.effect[cols.index(100)]
    assert full.p_value[cols.index(100)] < full.p_value[cols.index(300)], (
        "the low-effect column really is the more significant one"
    )
    capped = candidate_columns(stats, overdispersion=0.0, max_columns=1)
    assert capped.columns.tolist() == [300]
    assert capped.n_capped == len(cols) - 1


def test_column_kind_is_carried_but_never_gates():
    rng = np.random.default_rng(71)
    truth = _rand(rng, 300)
    cres, stats = _stats(truth, [(truth, 10.0)])
    assert stats.column_kind.shape == (cres.pwm.shape[0],)
    assert stats.n_columns == cres.pwm.shape[0]


def test_empty_candidate_set_is_well_formed():
    rng = np.random.default_rng(73)
    truth = _rand(rng, 200)
    _cres, stats = _stats(truth, [(truth, 10.0)])
    cand = candidate_columns(stats)
    assert cand.columns.size == 0
    assert cand.n_allelic == cand.n_coverage == 0
    assert cand.route.dtype == np.int8 and cand.major.dtype == np.int8
