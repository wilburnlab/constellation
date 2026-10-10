"""`--mode {genome, kmer, em}` and the back-compatibility that comes with it.

The old labels described *provenance* ("genome-guided", "de-novo"); what a
user chooses between is the *mechanism*. Renaming them is only safe if the
old spellings keep working on both sides — the flag, and the `mode` column of
every clusters.parquet already on disk.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from constellation.cli.__main__ import _normalise_cluster_mode
from constellation.sequencing.schemas.transcriptome import TRANSCRIPT_CLUSTER_TABLE
from constellation.sequencing.transcriptome.cluster.denovo.em.outputs import (
    CLUSTER_MODES,
    LEGACY_CLUSTER_MODES,
    MODE_EM,
)


@pytest.mark.parametrize("mode", CLUSTER_MODES)
def test_canonical_modes_pass_through(mode):
    assert _normalise_cluster_mode(mode) == mode


@pytest.mark.parametrize(("old", "new"), sorted(LEGACY_CLUSTER_MODES.items()))
def test_deprecated_spellings_normalise_and_warn(old, new, capsys):
    assert _normalise_cluster_mode(old) == new
    err = capsys.readouterr().err
    assert "deprecated" in err
    assert f"--mode {new}" in err


def test_the_parser_still_accepts_the_old_spellings():
    from constellation.cli.__main__ import _build_parser

    parser = _build_parser()
    for spelling in (*CLUSTER_MODES, *LEGACY_CLUSTER_MODES):
        args = parser.parse_args(
            [
                "transcriptome",
                "cluster",
                "--demux-dir",
                "d",
                "--output-dir",
                "o",
                "--mode",
                spelling,
            ]
        )
        assert args.mode == spelling


def test_a_legacy_clusters_parquet_still_loads():
    """The vocabulary widens; it does not replace. No migration needed."""
    for mode in (*CLUSTER_MODES, *LEGACY_CLUSTER_MODES):
        table = pa.table(
            {
                "cluster_id": pa.array([0], pa.int64()),
                "representative_read_id": pa.array(["r0"], pa.string()),
                "n_reads": pa.array([1], pa.int32()),
                "identity_threshold": pa.array([0.97], pa.float32()),
                "consensus_sequence": pa.array(["ACGT"], pa.string()),
                "predicted_protein": pa.nulls(1, pa.string()),
                "orf_start": pa.nulls(1, pa.int32()),
                "orf_end": pa.nulls(1, pa.int32()),
                "orf_strand": pa.nulls(1, pa.string()),
                "codon_table": pa.nulls(1, pa.int32()),
                "mode": pa.array([mode], pa.string()),
                "contig_id": pa.nulls(1, pa.int64()),
                "strand": pa.nulls(1, pa.string()),
                "span_start": pa.nulls(1, pa.int64()),
                "span_end": pa.nulls(1, pa.int64()),
                "fingerprint_hash": pa.nulls(1, pa.uint64()),
                "n_unique_sequences": pa.array([1], pa.int32()),
                "sample_id": pa.nulls(1, pa.int64()),
            },
            schema=TRANSCRIPT_CLUSTER_TABLE,
        )
        assert table.column("mode").to_pylist() == [mode]


def test_the_frontend_colour_maps_keep_the_old_keys():
    """A pre-rename clusters.parquet must not fall through to grey."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "constellation" / "viz" / "frontend"
    # One map now: the renderer's, which its settings popover also reads its
    # default swatches from (there used to be a second copy in the popover).
    for rel in ("src/modalities/genome/renderers/cluster_pileup.ts",):
        text = (root / rel).read_text()
        for key in (*CLUSTER_MODES, *LEGACY_CLUSTER_MODES):
            assert f"'{key}'" in text or f"{key}:" in text, (rel, key)


def test_the_em_path_stamps_the_canonical_mode():
    assert MODE_EM == "em"
    assert MODE_EM in CLUSTER_MODES


# ── per-mode defaults ─────────────────────────────────────────────────


def test_overdispersion_defaults_differ_by_mode():
    """One flag, two right answers — so its default is resolved per handler.

    kmer's `call_variants` wants rho off. The EM path's candidate-column test
    wants it ON at 0.01: at 1,525 reads a point binomial under a 1% null
    admits 1,924 columns where rho=0.01 admits none. A shared default would
    hand the EM path the setting known not to work, silently.
    """
    from constellation.cli.__main__ import _build_parser
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )

    parser = _build_parser()
    base = ["transcriptome", "cluster", "--demux-dir", "d", "--output-dir", "o"]

    args = parser.parse_args(base)
    assert args.overdispersion is None, "the default must be mode-resolved"

    # The EM kernel's own default is the on value.
    assert MStepParams().overdispersion == 0.01

    # An explicit value still wins for either mode.
    args = parser.parse_args([*base, "--overdispersion", "0.05"])
    assert args.overdispersion == 0.05


# ── em seeding: one mechanism, two seeders ────────────────────────────


def _args(*extra):
    from constellation.cli.__main__ import _build_parser

    return _build_parser().parse_args(
        ["transcriptome", "cluster", "--demux-dir", "d", "--output-dir", "o", *extra]
    )


@pytest.mark.parametrize(
    ("spelling", "seeding"),
    [("em", "orf"), ("em-orf", "orf"), ("em-kmer", "kmer")],
)
def test_the_em_spellings_pick_a_seeder_and_keep_one_mode(spelling, seeding):
    """All three are the EM loop; they differ only in how round 1 is seeded.

    So the `mode` COLUMN stays "em" and the vocabulary does not widen —
    nothing downstream (clusters.parquet, the viz colour maps) has to change
    for a seeding choice.
    """
    from constellation.cli.__main__ import _em_seeding, _normalise_cluster_mode

    assert _normalise_cluster_mode(spelling) == MODE_EM
    assert _em_seeding(spelling) == seeding


def test_bare_em_notes_that_it_means_orf_seeding(capsys):
    """Under-specified rather than wrong, so a note and not a deprecation."""
    from constellation.cli.__main__ import _em_seeding

    assert _em_seeding("em") == "orf"
    err = capsys.readouterr().err
    assert "em-orf" in err and "em-kmer" in err


def test_the_default_seeder_has_not_flipped():
    """kmer seeding dominates ORF seeding on every measured round-1 axis.

    Flipping the default before the multi-round comparison runs would destroy
    the baseline that comparison is against, so `em` still means `em-orf`.
    """
    from constellation.cli.__main__ import _em_seeding

    assert _em_seeding("em") == "orf"


def test_the_new_spellings_parse():
    for spelling in ("em-orf", "em-kmer"):
        assert _args("--mode", spelling).mode == spelling


# ── shared flags, per-mode defaults ───────────────────────────────────


def test_the_read_to_read_gate_defaults_differ_by_mode():
    """One flag, two right answers, so the parser holds neither.

    `0.98 / 30:30` is the --mode kmer gate; under --mode em-kmer it costs the
    same 14 h round-1 E-step as ORF seeding, and `0.93 / inf:100` is the 4.8x
    cut. Baking either into the parser hands the other mode a setting known
    to be wrong for it.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.seed_kmer import (
        DEFAULT_SEED_IDENTITY,
        DEFAULT_SEED_MAX_3P,
    )

    args = _args("--mode", "em-kmer")
    assert args.identity is None
    assert args.max_5p_overhang is None and args.max_3p_overhang is None
    assert DEFAULT_SEED_IDENTITY == 0.93
    assert DEFAULT_SEED_MAX_3P == 100


def test_unbounded_overhangs_are_spellable():
    from constellation.sequencing.transcriptome.cluster.denovo.verify import (
        UNBOUNDED_OVERHANG,
    )

    for spelling in ("inf", "none", "-1"):
        args = _args("--mode", "em-kmer", "--max-5p-overhang", spelling)
        assert args.max_5p_overhang == UNBOUNDED_OVERHANG
    assert _args("--mode", "em-kmer", "--max-5p-overhang", "30").max_5p_overhang == 30


# ── inapplicable flags error rather than no-op ────────────────────────


@pytest.mark.parametrize(
    ("mode", "flag", "value"),
    [
        ("em-orf", "--identity", "0.93"),
        ("em-orf", "--max-3p-overhang", "100"),
        ("em-kmer", "--fold-identity", "0.97"),
        ("kmer", "--seed-grouping", "greedy"),
        # The shortlist knobs exist only on the two-pass E-step.
        ("em-kmer", "--estep-shortlist-k", "8"),
        ("em-orf", "--estep-shortlist-frac", "0.7"),
        ("em-kmer", "--estep-align-workers", "4"),
        ("kmer", "--estep-aligner", "edlib"),
        ("kmer", "--p-floor-quality-scale", "1.5"),
        # The template graph and the merge exist only in the em loop.
        ("kmer", "--template-graph", "final"),
        ("kmer", "--merge-max-edits", "1"),
        ("kmer", "--graph-identity-floor", "0.98"),
        # --merge-from-round only schedules merging.
        ("em-kmer --no-merge", "--merge-from-round", "3"),
        ("em-kmer --template-graph off", "--merge-from-round", "3"),
        # With the graph off nothing reads the graph's or the predicate's knobs.
        ("em-kmer --template-graph off", "--graph-5p-tolerance", "20"),
        ("em-kmer --template-graph off", "--merge-max-edits", "1"),
        ("em-orf --template-graph off", "--merge-min-reads", "5"),
        # Under 'final' only the final output merges.
        ("em-kmer --merge --template-graph final", "--merge-from-round", "2"),
        ("em-kmer --template-graph final", "--merge-from-round", "2"),
        # The coverage route is the em M-step's.
        ("kmer", "--coverage-route", ""),
    ],
)
def test_a_flag_this_mode_ignores_is_an_error(mode, flag, value):
    """A swept parameter that silently did nothing makes the run look like
    evidence about it. Same rule as predict-library's backend-only flags."""
    from constellation.cli.__main__ import _em_seeding, _normalise_cluster_mode
    from constellation.cli.__main__ import _reject_inapplicable

    mode, *extra = mode.split()
    args = _args("--mode", mode, *extra, flag, *([value] if value else []))
    canonical = _normalise_cluster_mode(mode)
    seeding = _em_seeding(mode) if canonical == "em" else "orf"
    problem = _reject_inapplicable(args, canonical, seeding)
    assert problem is not None and flag in problem


def test_the_applicable_flags_are_not_rejected():
    from constellation.cli.__main__ import _reject_inapplicable

    args = _args("--mode", "em-kmer", "--identity", "0.90", "--max-3p-overhang", "100")
    assert _reject_inapplicable(args, "em", "kmer") is None
    args = _args("--mode", "em-orf", "--fold-identity", "0.97")
    assert _reject_inapplicable(args, "em", "orf") is None
    args = _args("--mode", "kmer", "--identity", "0.96")
    assert _reject_inapplicable(args, "kmer", "orf") is None


# ── --min-aa-length means one thing, in one place ─────────────────────


def test_min_aa_length_is_the_em_orf_seeding_key_at_30():
    """30, not the parser's old 60.

    At 60 the seeder cannot make a template for Prm1 (51 aa), the most
    abundant transcript in the tissue this pipeline was built for — so the
    effective default had been silently excluding real short-ORF seeds
    (ledger #6).
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        EmParams,
    )

    assert _args("--mode", "em-orf").min_aa_length is None
    assert EmParams().min_aa_length == 30


def test_min_aa_length_still_defaults_to_60_for_plain_kmer_mode():
    """A shared flag with two right answers holds neither in the parser."""
    from constellation.cli.__main__ import _cmd_transcriptome_cluster_denovo  # noqa: F401

    assert _args("--mode", "kmer").min_aa_length is None


def test_min_aa_length_is_the_annotation_floor_in_every_em_mode():
    """It reaches the M-step under both seeders, and under em-orf it is the
    seeding key as well. It was refused under em-kmer while the M-step had
    no floor; that floor is back (a floor of 1 shipped 14,813 sub-30-aa
    proteins in one round), so the flag means something there again."""
    from constellation.cli.__main__ import _reject_inapplicable

    for mode, seeding in (("em-kmer", "kmer"), ("em-orf", "orf")):
        args = _args("--mode", mode, "--min-aa-length", "45")
        assert _reject_inapplicable(args, "em", seeding) is None
        params = _resolved_for(seeding, "--min-aa-length", "45")
        assert params.min_aa_length == 45
        assert params.mstep.min_aa_length == 45
    assert _resolved().mstep.min_aa_length == 30
    assert (
        _reject_inapplicable(
            _args("--mode", "kmer", "--min-aa-length", "60"), "kmer", "orf"
        )
        is None
    )


def test_the_coverage_route_is_off_unless_asked_for():
    from constellation.cli.__main__ import _reject_inapplicable

    assert _resolved().mstep.coverage_route is False
    assert _resolved("--no-coverage-route").mstep.coverage_route is False
    assert _resolved("--coverage-route").mstep.coverage_route is True
    problem = _reject_inapplicable(
        _args("--mode", "kmer", "--coverage-route"), "kmer", "orf"
    )
    assert problem is not None and "--coverage-route" in problem


def test_the_two_pass_estep_is_opt_in_and_its_knobs_apply_under_it():
    from constellation.cli.__main__ import _reject_inapplicable

    assert _args("--mode", "em-kmer").estep_aligner == "minimap2"
    for aligner in ("edlib", "native"):
        args = _args(
            "--mode",
            "em-kmer",
            "--estep-aligner",
            aligner,
            "--estep-shortlist-k",
            "8",
            "--estep-shortlist-frac",
            "0.7",
            "--estep-align-workers",
            "4",
        )
        assert _reject_inapplicable(args, "em", "kmer") is None, aligner


def test_the_native_aligner_and_the_quality_floor_reach_the_params():
    params = _resolved("--estep-aligner", "native", "--p-floor-quality-scale", "1.5")
    assert params.estep_aligner == "native"
    assert params.p_floor_quality_scale == 1.5
    assert _resolved().p_floor_quality_scale is None


def test_the_shortlist_and_probe_counts_default_to_32_and_reach_the_params():
    """Raised from 16 (2026-10-05): under native the shortlist key is
    shared probes, so family candidates tie in droves and a 16-deep cut
    over near-ties was an arbitrary subset that cost deep genes reads."""
    defaults = _resolved()
    assert defaults.estep_shortlist_k == 32
    assert defaults.estep_probes_per_read == 32
    params = _resolved(
        "--estep-aligner",
        "native",
        "--estep-shortlist-k",
        "64",
        "--estep-probes-per-read",
        "48",
    )
    assert params.estep_shortlist_k == 64
    assert params.estep_probes_per_read == 48

    from constellation.cli.__main__ import _reject_inapplicable
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        EmParams,
    )

    for mode_args, seeding in (
        (("--mode", "em-kmer"), "kmer"),  # default aligner is minimap2
        (("--mode", "em-kmer", "--estep-aligner", "edlib"), "kmer"),
    ):
        problem = _reject_inapplicable(
            _args(*mode_args, "--estep-probes-per-read", "48"), "em", seeding
        )
        assert problem is not None and "--estep-probes-per-read" in problem
    with pytest.raises(ValueError, match="estep_probes_per_read"):
        EmParams(estep_probes_per_read=1)


def test_merge_is_off_by_default_and_both_spellings_parse():
    """`--no-merge` stays valid as the explicit default, so a script that
    passed it keeps meaning what it meant."""
    from constellation.cli.__main__ import _reject_inapplicable

    assert _args("--mode", "em-kmer").merge is None
    assert _args("--mode", "em-kmer", "--merge").merge is True
    explicit = _args("--mode", "em-kmer", "--no-merge")
    assert explicit.merge is False
    assert _reject_inapplicable(explicit, "em", "kmer") is None
    for spelling in ("--merge", "--no-merge"):
        problem = _reject_inapplicable(_args("--mode", "kmer", spelling), "kmer", "orf")
        assert problem is not None and spelling in problem


def test_the_predicate_applies_to_the_report_as_well_as_to_a_merge():
    """The predicate defines the graph's `mergeable` column, so its knobs act
    whether or not the run merges — that is how "what would a stricter
    predicate collapse" is asked without collapsing anything."""
    from constellation.cli.__main__ import _reject_inapplicable

    knobs = (
        "--merge-max-edits",
        "1",
        "--merge-5p-tolerance",
        "20",
        "--merge-3p-tolerance",
        "20",
        "--merge-min-reads",
        "5",
        "--merge-siblings",
    )
    assert (
        _reject_inapplicable(_args("--mode", "em-kmer", *knobs), "em", "kmer") is None
    )
    merging = _args("--mode", "em-kmer", "--merge", "--merge-from-round", "2", *knobs)
    assert _reject_inapplicable(merging, "em", "kmer") is None
    final = _args("--mode", "em-orf", "--merge", "--template-graph", "final")
    assert _reject_inapplicable(final, "em", "orf") is None


@pytest.mark.parametrize(
    ("flag", "value", "replacement"),
    [
        ("--p-merge", "0.999", "--merge-max-edits"),
        ("--merge-min-coverage", "0.95", "--merge-5p-tolerance"),
    ],
)
@pytest.mark.parametrize("mode", ["em-kmer", "em-orf", "kmer", "genome"])
def test_the_removed_merge_flags_say_what_replaced_them(mode, flag, value, replacement):
    """What they meant is gone — identity on a local alignment, end tolerance
    as a proportion — so accepting one silently would misreport the run, and
    "this mode does not use it" would be the wrong thing to say."""
    from constellation.cli.__main__ import _em_seeding, _normalise_cluster_mode
    from constellation.cli.__main__ import _reject_inapplicable

    args = _args("--mode", mode, flag, value)
    canonical = _normalise_cluster_mode(mode)
    seeding = _em_seeding(mode) if canonical == "em" else "orf"
    problem = _reject_inapplicable(args, canonical, seeding)
    assert problem is not None
    assert flag in problem and "removed" in problem and replacement in problem


def test_the_removed_flags_are_hidden_from_help():
    from constellation.cli.__main__ import _build_parser

    parser = _build_parser()
    cluster = parser._subparsers._group_actions[0].choices["transcriptome"]
    cluster = cluster._subparsers._group_actions[0].choices["cluster"]
    text = cluster.format_help()
    assert "--p-merge" not in text and "--merge-min-coverage" not in text
    assert "--template-graph" in text and "--merge-max-edits" in text


def test_the_escape_hatch_refuses_merge():
    """`--template-graph off` runs nothing after the M-step, and a merge
    reads the graph."""
    from constellation.cli.__main__ import _reject_inapplicable

    off = _args("--mode", "em-kmer", "--template-graph", "off")
    assert _reject_inapplicable(off, "em", "kmer") is None
    both = _args("--mode", "em-kmer", "--template-graph", "off", "--merge")
    problem = _reject_inapplicable(both, "em", "kmer")
    assert problem is not None and "--template-graph off" in problem


def _resolved_for(seeding, *extra):
    from constellation.cli.__main__ import _em_params
    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        GraphParams,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        EmParams,
    )

    args = _args("--mode", f"em-{seeding}", *extra)
    return _em_params(args, seeding, EmParams, GraphParams, MStepParams)


def _resolved(*extra):
    return _resolved_for("kmer", *extra)


def test_a_run_merges_within_two_edits_by_default():
    """The operating point of the 2026-09-30 sweep: K = 2 nominally best on
    every recall count, node growth between rounds nearly stopped."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        EmParams,
    )

    for params in (EmParams(), _resolved()):
        assert params.template_graph == "rounds"
        assert params.merge is True
        assert all(params.merges_after(r) for r in range(1, 13))
        pred = params.merge_predicate()
        assert (pred.max_edits, pred.tol_5p, pred.tol_3p) == (2, 30, 30)
        assert pred.min_reads == 0 and pred.merge_siblings is False
    assert not hasattr(EmParams(), "p_merge")
    assert not hasattr(EmParams(), "merge_min_coverage")


def test_no_merge_and_the_graph_off_both_switch_it_off():
    off = _resolved("--no-merge")
    assert off.merge is False and not off.merges_after(1)
    assert off.merge_predicate().max_edits == 2, "the report still says mergeable"
    silent = _resolved("--template-graph", "off")
    assert silent.merge is False and silent.template_graph == "off"
    assert _resolved("--template-graph", "final").merge is True


def test_the_edit_budget_follows_the_merge_cap():
    """An edge exists only within `edit_budget = max(min_budget, floor(L *
    (1 - identity_floor)))`, so at 0.99 a pair whose shorter template is
    under 100 K nt could never merge at K edits — 13% of a 9.4M-read run's
    nodes at K = 6 — and the bench passed --graph-identity-floor 0.98 to get
    round it. The budget follows the cap; longer templates are untouched."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        GraphParams,
        edit_budget,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        EmParams,
    )

    assert _resolved().graph.min_budget == 3
    six = _resolved("--merge-max-edits", "6")
    assert six.graph.min_budget == 6
    assert edit_budget(300, six.graph) == 6 and edit_budget(900, six.graph) == 9
    assert EmParams(merge_max_edits=6).graph.min_budget == 6
    assert EmParams(merge_max_edits=1).graph.min_budget == 3
    # An explicit, larger budget is kept; the stamp sees the budget it ran at.
    kept = EmParams(merge_max_edits=6, graph=GraphParams(min_budget=9))
    assert kept.graph.min_budget == 9
    assert EmParams(merge=False, merge_max_edits=6).graph.min_budget == 6
    assert six.graph.semantic()["min_budget"] == 6


def test_the_merge_flags_reach_the_predicate():
    params = _resolved(
        "--merge",
        "--merge-from-round",
        "3",
        "--merge-max-edits",
        "1",
        "--merge-5p-tolerance",
        "10",
        "--merge-min-reads",
        "4",
        "--merge-siblings",
        "--graph-3p-tolerance",
        "50",
        "--graph-identity-floor",
        "0.98",
    )
    assert params.merge and not params.merges_after(2) and params.merges_after(3)
    pred = params.merge_predicate()
    assert (pred.max_edits, pred.tol_5p, pred.min_reads) == (1, 10, 4)
    assert pred.merge_siblings is True
    # An unset merge tolerance is the graph's.
    assert pred.tol_3p == 50 == params.graph.tol_3p
    assert params.graph.identity_floor == 0.98
    assert params.graph.bucket_cap == 20_480


def test_under_final_only_the_final_output_merges():
    params = _resolved("--merge", "--template-graph", "final")
    assert params.merge and not any(params.merges_after(r) for r in range(1, 13))
    # ...and it merges within the cap: the final merge is no longer exact.
    assert params.merge_predicate().max_edits == 2
    assert _resolved("--template-graph", "final", "--merge-max-edits", "1").graph


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"merge": True, "template_graph": "off"}, "needs the template graph"),
        ({"merge_tol_5p": 31}, "merge_tol_5p"),
        ({"merge_tol_3p": -1}, "merge_tol_3p"),
        ({"merge_max_edits": -1}, "merge_max_edits"),
        ({"merge_from_round": 0}, "merge_from_round"),
        ({"template_graph": "sometimes"}, "template_graph"),
        (
            {"merge": True, "template_graph": "final", "merge_from_round": 2},
            "merge_from_round",
        ),
        # A knob that cannot act is an error here too, not only in the parser.
        (
            {"merge": False, "merge_from_round": 3},
            "merge_from_round has no effect without merge",
        ),
        ({"template_graph": "off"}, "merge is on by default; pass merge=False"),
        ({"template_graph": "final", "merge_from_round": 2}, "merge_from_round"),
        (
            {"template_graph": "off", "merge": False, "merge_max_edits": 1},
            "merge_max_edits has no effect",
        ),
        (
            {"template_graph": "off", "merge": False, "merge_tol_5p": 10},
            "merge_tol_5p has no effect",
        ),
        (
            {"template_graph": "off", "merge": False, "merge_min_reads": 3},
            "merge_min_reads has no effect",
        ),
        (
            {"template_graph": "off", "merge": False, "merge_siblings": True},
            "merge_siblings has no effect",
        ),
        # Not counts: a float would be truncated, a string is truthy.
        ({"merge_tol_5p": 10.5}, "merge_tol_5p must be an int"),
        ({"merge_max_edits": 1.5}, "merge_max_edits must be an int"),
        ({"merge_min_reads": True}, "merge_min_reads must be an int"),
        ({"merge": "yes"}, "merge must be a bool"),
    ],
)
def test_emparams_refuses_what_the_cli_refuses(kwargs, match):
    """The library is a second way in, so the rule is not only the parser's."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        EmParams,
    )

    with pytest.raises(ValueError, match=match):
        EmParams(**kwargs)


def test_graph_parameters_have_no_effect_with_the_graph_off():
    from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
        GraphParams,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        EmParams,
    )

    with pytest.raises(ValueError, match="graph parameters have no effect"):
        EmParams(template_graph="off", merge=False, graph=GraphParams(tol_5p=20))
    # How the work is cut up is a parameter of the graph too; with no graph
    # there is no work.
    with pytest.raises(ValueError, match="graph parameters have no effect"):
        EmParams(template_graph="off", merge=False, graph=GraphParams(chunk_rows=1_000))
    assert EmParams(template_graph="off", merge=False).graph == GraphParams()
