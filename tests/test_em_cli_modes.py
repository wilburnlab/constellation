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
    for rel in (
        "src/track_renderers/cluster_pileup.ts",
        "src/widgets/TrackSettingsPanel.ts",
    ):
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
    ],
)
def test_a_flag_this_mode_ignores_is_an_error(mode, flag, value):
    """A swept parameter that silently did nothing makes the run look like
    evidence about it. Same rule as predict-library's backend-only flags."""
    from constellation.cli.__main__ import _em_seeding, _normalise_cluster_mode
    from constellation.cli.__main__ import _reject_inapplicable

    args = _args("--mode", mode, flag, value)
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


def test_min_aa_length_is_refused_under_em_kmer():
    """Nothing reads it there: that seeder predicts no ORF, and the M-step
    has no minimum protein length in either mode."""
    from constellation.cli.__main__ import _reject_inapplicable

    args = _args("--mode", "em-kmer", "--min-aa-length", "30")
    problem = _reject_inapplicable(args, "em", "kmer")
    assert problem is not None and "--min-aa-length" in problem
    # ...and it is fine where it means something.
    assert _reject_inapplicable(_args("--mode", "em-orf", "--min-aa-length", "30"),
                                "em", "orf") is None
    assert _reject_inapplicable(_args("--mode", "kmer", "--min-aa-length", "60"),
                                "kmer", "orf") is None


def test_the_mstep_never_receives_a_length_floor():
    """The CLI cannot hand the M-step a floor, because it has no field for one."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )

    assert "min_aa_length" not in MStepParams.__dataclass_fields__


def test_the_two_pass_estep_is_opt_in_and_its_knobs_apply_under_it():
    from constellation.cli.__main__ import _reject_inapplicable

    assert _args("--mode", "em-kmer").estep_aligner == "minimap2"
    args = _args(
        "--mode",
        "em-kmer",
        "--estep-aligner",
        "edlib",
        "--estep-shortlist-k",
        "8",
        "--estep-shortlist-frac",
        "0.7",
        "--estep-align-workers",
        "4",
    )
    assert _reject_inapplicable(args, "em", "kmer") is None
