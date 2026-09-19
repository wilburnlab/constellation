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
