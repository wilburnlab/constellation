"""Read-to-template likelihood over the differing positions.

The load-bearing test is `test_homopolymer_indel_loses_to_a_substitution`:
that single case is the entire justification for computing a likelihood
instead of ranking on AS, which charges both differences the same ~6 points.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em.likelihood import (
    EPS_UNALIGNED,
    parse_cigars,
    read_template_loglik,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
    TEMPLATE_TABLE,
    TemplateStore,
)
from constellation.sequencing.transcriptome.cluster.denovo.variants import ErrorModel


def _store(sequences: list[str]) -> TemplateStore:
    n = len(sequences)
    table = pa.table(
        {
            "template_id": pa.array(np.arange(n, dtype=np.int64)),
            "sequence": pa.array(sequences, pa.large_string()),
            "orf_start": pa.array(np.zeros(n, np.int32)),
            "orf_end": pa.array(np.array([len(s) for s in sequences], np.int32)),
            "orf_aa_length": pa.array(np.zeros(n, np.int32)),
            "node_weight": pa.array(np.ones(n, np.float64)),
            "orf_replication": pa.array(np.ones(n, np.int64)),
            "seed_read_quality": pa.nulls(n, pa.float32()),
            "seed_read_row": pa.array(np.full(n, -1, np.int32)),
            "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
        },
        schema=TEMPLATE_TABLE,
    )
    return TemplateStore.from_table(table)


def _logl(store, cigars, *, t_start, q_start, q_end, q_len, rows, model=None):
    return read_template_loglik(
        pa.array(cigars, pa.large_string()),
        t_start=np.asarray(t_start),
        q_start=np.asarray(q_start),
        q_end=np.asarray(q_end),
        q_len=np.asarray(q_len),
        template_row=np.asarray(rows),
        store=store,
        model=model,
    )


# ── the parser ────────────────────────────────────────────────────────


def test_parse_cigars_matches_a_reference_regex():
    import re

    cigars = ["10=2X5I3=", "1=", "100=50D25=", "7X"]
    ops = parse_cigars(pa.array(cigars, pa.large_string()))
    expect = [
        [(int(n), o) for n, o in re.findall(r"(\d+)([=XIDMSHN])", c)] for c in cigars
    ]
    for i, exp in enumerate(expect):
        sel = ops.candidate == i
        assert ops.length[sel].tolist() == [n for n, _ in exp]
    assert ops.n_candidates == 4


def test_parse_cigars_does_not_bleed_across_string_boundaries():
    """The first op of a CIGAR must not take its digits from the previous one."""
    ops = parse_cigars(pa.array(["12=", "34="], pa.large_string()))
    assert ops.length.tolist() == [12, 34]
    assert ops.candidate.tolist() == [0, 1]


def test_parse_cigars_handles_empty_and_null():
    ops = parse_cigars(pa.array([None, "5="], pa.large_string()))
    assert ops.length.tolist() == [5]
    assert ops.candidate.tolist() == [1]


# ── the physics ───────────────────────────────────────────────────────


def test_homopolymer_indel_loses_to_a_substitution():
    """The case AS cannot see, and the reason this module exists.

    ONE read of 85 bases against two templates. Against A it needs a deletion
    inside a 6-G homopolymer; against B, a substitution. Both cost AS ~6
    points, so AS calls it a tie. The context-aware model must prefer A: a
    homopolymer-length call is something the basecaller does routinely, a
    substitution is not.
    """
    hp = "ACGT" * 10 + "GGGGGG" + "ACGT" * 10
    flat = "ACGT" * 10 + "GATTAC" + "ACGT" * 10
    store = _store([hp, flat])

    logl = _logl(
        store,
        # Both consume all 85 read bases; A's template is one base longer.
        ["40=1D45=", "40=1X44="],
        t_start=[0, 0],
        q_start=[0, 0],
        q_end=[85, 85],
        q_len=[85, 85],
        rows=[0, 1],
    )
    assert logl[0] > logl[1], "a homopolymer slip is weaker evidence than a SNV"


def test_a_measured_homopolymer_rate_widens_the_margin():
    """How much the ranker can discriminate is a property of the error model.

    The shipped default returns 0.0103 at a 5-base run against the 0.22 the
    bench measured at the H3f3b 5-G run (ledger #40), which shrinks the
    homopolymer-vs-substitution margin from ~4.3 nats to ~1.2. Refitting the
    curve — which is what `--error-model empirical` does — restores it, and
    this test pins that the module responds to the model rather than baking
    the defaults in.
    """
    hp = "ACGT" * 10 + "GGGGGG" + "ACGT" * 10
    flat = "ACGT" * 10 + "GATTAC" + "ACGT" * 10
    store = _store([hp, flat])
    args = dict(
        t_start=[0, 0], q_start=[0, 0], q_end=[85, 85], q_len=[85, 85], rows=[0, 1]
    )
    cigars = ["40=1D45=", "40=1X44="]

    default = _logl(store, cigars, **args)
    # eps_hp0 chosen so epsilon_homopolymer(6) lands on the measured 0.22.
    measured = _logl(
        store,
        cigars,
        model=ErrorModel(eps_hp0=0.22, hp_slope=0.0, hp_max=0.5),
        **args,
    )
    assert measured[0] - measured[1] > default[0] - default[1]
    assert measured[0] - measured[1] > 3.0


def test_non_homopolymer_indel_is_near_a_substitution_in_weight():
    seq = "ACGTACGTAC" * 10
    store = _store([seq, seq])
    logl = _logl(
        store,
        ["40=1D50=", "40=1X49="],  # both consume all 90 read bases
        t_start=[0, 0],
        q_start=[0, 0],
        q_end=[90, 90],
        q_len=[90, 90],
        rows=[0, 1],
    )
    # eps_indel 0.004 vs eps_sub 0.003 — close, unlike the homopolymer case.
    assert abs(logl[0] - logl[1]) < 1.0


def test_more_differences_is_always_worse():
    seq = "ACGTACGTAC" * 20
    store = _store([seq, seq, seq])
    logl = _logl(
        store,
        ["100=1X99=", "100=2X98=", "100=3X97="],
        t_start=[0, 0, 0],
        q_start=[0, 0, 0],
        q_end=[200, 200, 200],
        q_len=[200, 200, 200],
        rows=[0, 1, 2],
    )
    assert logl[0] > logl[1] > logl[2]


def test_a_long_near_perfect_alignment_beats_a_short_perfect_one():
    """Identity alone ranks these backwards; the unaligned term is what fixes it."""
    seq = "ACGTACGTAC" * 30
    store = _store([seq, seq])
    logl = _logl(
        store,
        ["299=1X", "100="],  # full-length with one SNV vs a third, perfect
        t_start=[0, 0],
        q_start=[0, 0],
        q_end=[300, 100],
        q_len=[300, 300],
        rows=[0, 1],
    )
    assert logl[0] > logl[1]


def test_unaligned_bases_are_charged():
    seq = "ACGTACGTAC" * 30
    store = _store([seq, seq])
    logl = _logl(
        store,
        ["200=", "200="],
        t_start=[0, 0],
        q_start=[0, 0],
        q_end=[200, 200],
        q_len=[200, 260],  # second read has 60 bases the template never explains
        rows=[0, 1],
    )
    assert logl[0] - logl[1] == pytest.approx(-60 * np.log(EPS_UNALIGNED), rel=1e-9)


def test_missing_cigar_scores_negative_infinity():
    """No alignment is no evidence — it must not score better than a bad one."""
    seq = "ACGT" * 20
    store = _store([seq, seq])
    logl = _logl(
        store,
        [None, "50=10X20="],
        t_start=[0, 0],
        q_start=[0, 0],
        q_end=[80, 80],
        q_len=[80, 80],
        rows=[0, 1],
    )
    assert logl[0] == -np.inf
    assert np.isfinite(logl[1])


def test_indel_run_is_one_event_not_one_per_base():
    """A 3-base homopolymer slip is one mistake, not three independent ones."""
    hp = "ACGT" * 10 + "GGGGGGGG" + "ACGT" * 10
    store = _store([hp, hp])
    logl = _logl(
        store,
        ["40=3D45=", "40=1D47="],
        t_start=[0, 0],
        q_start=[0, 0],
        q_end=[85, 87],
        q_len=[88, 88],
        rows=[0, 1],
    )
    # Both are one run, so the indel terms match; the difference comes only
    # from matched/unaligned bases, never from 3x the indel penalty.
    assert abs(logl[0] - logl[1]) < 3.0


def test_template_boundary_does_not_fabricate_a_homopolymer_run():
    """Concatenating templates must not merge a trailing run into a leading one."""
    store = _store(["AAAA", "AAAA"])
    runs = store.hp_run_at(np.array([0, 1]), np.array([3, 0]))
    assert runs.tolist() == [4, 4], "runs must reset at every template boundary"


def test_error_model_is_respected():
    """A caller-supplied model changes the answer; the defaults are not baked in."""
    seq = "ACGTACGTAC" * 10
    store = _store([seq])
    args = dict(t_start=[0], q_start=[0], q_end=[100], q_len=[100], rows=[0])
    strict = _logl(store, ["99=1X"], model=ErrorModel(eps_sub=1e-6), **args)
    loose = _logl(store, ["99=1X"], model=ErrorModel(eps_sub=1e-1), **args)
    assert loose[0] > strict[0]
