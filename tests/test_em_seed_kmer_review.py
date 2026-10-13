"""Four defects found in review of feat/em-kmer-seeding. All four are mine."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
    EmParams,
    run_em,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.seed_kmer import (
    READ_CLUSTER_MAP_SCHEMA,
    seed_by_kmer_clustering,
)


# ── 1. a global replace clobbered an unrelated subcommand ─────────────


def test_demultiplex_keeps_its_own_min_aa_length_default():
    """`transcriptome demultiplex --min-aa-length` is a different flag.

    Its handler passes the value straight into ORF prediction, where None
    reaches `min_aa_length * 3` and raises TypeError on the first eligible
    transcript. It was collateral from a `str.replace` aimed at the cluster
    subcommand's identically-formatted argument.
    """
    from constellation.cli.__main__ import _build_parser

    args = _build_parser().parse_args(
        [
            "transcriptome", "demultiplex", "--reads", "r.bam",
            "--samples", "s.tsv", "--output-dir", "o",
        ]
    )
    assert args.min_aa_length == 60
    assert args.min_aa_length * 3 == 180  # the exact expression that raised


# ── 2. "0 = uncapped" has to reach the kernel as None ─────────────────


def test_zero_minimizers_per_seq_means_uncapped_not_zero():
    """The kernel spells uncapped `None`; 0 caps every sketch at nothing.

    Passed through literally the minimizer index is empty, so no candidate
    pair is ever proposed, the graph has no edges, and every read becomes its
    own template — a silent collapse to the worst possible seeding.
    """
    from constellation.cli.__main__ import _build_parser

    args = _build_parser().parse_args(
        [
            "transcriptome", "cluster", "--demux-dir", "d", "--output-dir", "o",
            "--mode", "em-kmer", "--minimizers-per-seq", "0",
        ]
    )
    assert args.minimizers_per_seq == 0
    # EmParams must be able to carry "uncapped" at all.
    assert EmParams(minimizers_per_seq=None).minimizers_per_seq is None


def test_the_em_handler_translates_zero_to_uncapped(monkeypatch):
    """The parse-level half is not enough — the handler does the translating."""
    import constellation.sequencing.transcriptome.cluster.denovo.em.rounds as R
    from constellation.cli.__main__ import _cmd_transcriptome_cluster_em, _build_parser

    seen = {}

    def _fake(demux_dir, output_dir, *, params, **kw):
        seen["params"] = params
        return []

    monkeypatch.setattr(R, "run_em", _fake)
    args = _build_parser().parse_args(
        [
            "transcriptome", "cluster", "--demux-dir", "d", "--output-dir", "o",
            "--mode", "em-kmer", "--minimizers-per-seq", "0",
        ]
    )
    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: "read_demux" in str(self)
                        or "reads" in str(self))
    _cmd_transcriptome_cluster_em(args, seeding="kmer")
    assert seen["params"].minimizers_per_seq is None


def test_uncapped_and_zero_do_not_produce_the_same_partition(tmp_path):
    """The behavioural half: one template per transcript vs one per read."""
    rng = np.random.default_rng(4)
    codons = ["GCT", "TGC", "GAT", "GAA", "TTT", "GGT", "CAT", "ATT"]

    def orf(n):
        return "ATG" + "".join(rng.choice(codons) for _ in range(n)) + "TAA"

    def mutate(s):
        return "".join(
            c if rng.random() > 0.01 else rng.choice(list("ACGT")) for c in s
        )

    truths = [orf(120), orf(140)]
    rows = [
        (f"t{i}_r{j}", mutate(t), 30.0)
        for i, t in enumerate(truths)
        for j in range(30)
    ]
    reads = pa.table(
        {
            "read_id": pa.array([r[0] for r in rows], pa.string()),
            "sequence": pa.array([r[1] for r in rows], pa.large_string()),
            "sample_id": pa.array([0] * len(rows), pa.int64()),
            "dorado_quality": pa.array([r[2] for r in rows], pa.float32()),
        }
    )
    common = dict(identity=0.90, max_cluster_read_frac=0.0, min_chain_cluster_reads=0)
    uncapped = seed_by_kmer_clustering(reads, minimizers_per_seq=None, **common)
    zeroed = seed_by_kmer_clustering(reads, minimizers_per_seq=0, **common)
    assert uncapped.templates.num_rows == 2
    # One template per distinct read: the graph has no edges at all.
    assert zeroed.templates.num_rows == zeroed.stats["n_uniq"]
    assert zeroed.stats["n_minimizers"] == 0
    assert zeroed.stats["n_edges"] == 0


# ── 3. an unstamped seed dir is not self-describing ───────────────────


def _demux(tmp_path, rows):
    from tests.test_em_rounds import _write_demux

    return _write_demux(tmp_path, rows)


@pytest.fixture
def corpus(tmp_path):
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tests"))
    from test_em_rounds import _mutate, _orf, _write_demux

    rng = np.random.default_rng(11)
    truths = [_orf(rng, 120), _orf(rng, 140)]
    rows = [
        (f"t{i}_r{j}", _mutate(rng, t, 0.01), float(rng.uniform(15.0, 35.0)))
        for i, t in enumerate(truths)
        for j in range(30)
    ]
    return _write_demux(tmp_path, rows)


def _p(**kw):
    base = dict(
        rounds=1, threads=1, mstep_workers=1, minimap2_n=50,
        max_window_length=None, min_aa_length=30,
    )
    base.update(kw)
    return EmParams(**base)


def test_a_legacy_seed_dir_is_validated_against_its_manifest(corpus, tmp_path):
    """This release changed ORF seeding's effective floor from 60 to 30.

    So an ordinary legacy resume — same command line — keeps the 60-aa seeds
    and rewrites the manifest to claim 30. The stamp prevents that for new
    runs; pre-stamp dirs need the manifest checked instead.
    """
    out = tmp_path / "em"
    run_em(corpus, out, params=_p(min_aa_length=60))
    (out / "seed" / "params.json").unlink()  # make it look pre-stamp

    with pytest.raises(ValueError, match="min_aa_length"):
        run_em(corpus, out, params=_p(min_aa_length=30), resume=True)
    # ...and the original values still resume cleanly.
    assert run_em(corpus, out, params=_p(min_aa_length=60), resume=True)


def test_an_unverifiable_legacy_seed_dir_is_refused(corpus, tmp_path):
    """No stamp and no manifest means nothing records what produced it."""
    out = tmp_path / "em"
    run_em(corpus, out, params=_p(min_aa_length=60))
    (out / "seed" / "params.json").unlink()
    (out / "manifest.json").unlink()

    with pytest.raises(ValueError, match="never completed"):
        run_em(corpus, out, params=_p(min_aa_length=60), resume=True)


# ── 4. one row per input read, on every exit ──────────────────────────


def test_the_read_map_covers_every_read_even_when_no_cluster_survives():
    """"Which reads were dropped" matters most when ALL of them were."""
    rng = np.random.default_rng(9)
    seqs = ["".join(rng.choice(list("ACGT")) for _ in range(300)) for _ in range(6)]
    reads = pa.table(
        {
            "read_id": pa.array([f"r{i}" for i in range(6)], pa.string()),
            "sequence": pa.array(seqs, pa.large_string()),
            "sample_id": pa.array([0] * 6, pa.int64()),
            "dorado_quality": pa.array([30.0] * 6, pa.float32()),
        }
    )
    # Six unrelated reads, so every cluster is a singleton and none clears 2.
    res = seed_by_kmer_clustering(reads, identity=0.99, min_seed_reads=2)

    assert res.templates.num_rows == 0
    assert res.read_cluster.schema.equals(READ_CLUSTER_MAP_SCHEMA)
    assert res.read_cluster.num_rows == reads.num_rows
    assert set(res.read_cluster.column("cluster_id").to_pylist()) == {-1}
    assert res.read_cluster.column("read_id").to_pylist() == [
        f"r{i}" for i in range(6)
    ]
