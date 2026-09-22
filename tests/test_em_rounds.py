"""The round loop, end to end on a small synthetic corpus.

This is the piece the EM stages were missing: seed / fold / E-step / M-step
all shipped and nothing iterated them. What matters here is that a round
completes, that its outputs are addressable on disk, and that a resumed run
picks up after the last complete round rather than starting over.

Needs minimap2 on ``$PATH``; skipped otherwise, since every other layer is
already tested without it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
    EmParams,
    run_em,
)

pytestmark = pytest.mark.skipif(
    shutil.which("minimap2") is None, reason="minimap2 not on $PATH"
)

_CODONS = ["GCT", "TGC", "GAT", "GAA", "TTT", "GGT", "CAT", "ATT", "AAA", "CTT"]


def _orf(rng, n_codons: int) -> str:
    body = "".join(rng.choice(_CODONS) for _ in range(n_codons - 1))
    return "ATG" + body + "TAA"


def _mutate(rng, seq: str, rate: float) -> str:
    out = []
    for base in seq:
        r = rng.random()
        if r < rate * 0.6:
            out.append(rng.choice([b for b in "ACGT" if b != base]))
        elif r < rate * 0.8:
            continue  # deletion
        elif r < rate:
            out.append(base)
            out.append(rng.choice(list("ACGT")))
        else:
            out.append(base)
    return "".join(out)


def _write_demux(tmp_path: Path, rows) -> Path:
    """rows = [(read_id, window, quality)]"""
    lead, trail = "GGGGGGGGGG", "CCCCCCCCCC"
    reads = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sequence": [lead + r[1] + trail for r in rows],
            "quality": ["I" * (len(r[1]) + 20) for r in rows],
            "dorado_quality": pa.array([r[2] for r in rows], pa.float32()),
        }
    )
    demux = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "transcript_segment_index": [0] * len(rows),
            "sample_id": pa.array([0] * len(rows), pa.int64()),
            "orientation": ["+"] * len(rows),
            "transcript_start": pa.array([len(lead)] * len(rows), pa.int32()),
            "transcript_end": pa.array(
                [len(lead) + len(r[1]) for r in rows], pa.int32()
            ),
            "score": pa.array([1.0] * len(rows), pa.float32()),
            "is_chimera": [False] * len(rows),
            "status": ["Complete"] * len(rows),
            "is_fragment": [False] * len(rows),
            "artifact": ["none"] * len(rows),
        }
    )
    d = tmp_path / "demux"
    (d / "reads").mkdir(parents=True)
    (d / "read_demux").mkdir(parents=True)
    pq.write_table(reads, d / "reads" / "part-00000.parquet")
    pq.write_table(demux, d / "read_demux" / "part-00000.parquet")
    return d


@pytest.fixture
def corpus_dir(tmp_path):
    """Two transcripts, 30 reads each at ~1% error."""
    rng = np.random.default_rng(11)
    truths = [_orf(rng, 120), _orf(rng, 140)]
    rows = []
    for t_idx, truth in enumerate(truths):
        for i in range(30):
            rows.append(
                (
                    f"t{t_idx}_r{i}",
                    _mutate(rng, truth, 0.01),
                    float(rng.uniform(15.0, 35.0)),
                )
            )
    return _write_demux(tmp_path, rows)


def _params(**kw):
    base = dict(
        rounds=2,
        threads=1,
        mstep_workers=1,
        minimap2_n=50,
        max_window_length=None,
        min_aa_length=30,
    )
    base.update(kw)
    return EmParams(**base)


def test_a_round_completes_and_leaves_addressable_outputs(corpus_dir, tmp_path):
    out = tmp_path / "em"
    results = run_em(corpus_dir, out, params=_params(rounds=1))

    assert len(results) == 1
    r = results[0]
    assert r.round_index == 1
    assert r.n_templates > 0

    # Corpus is written once and reused.
    assert (out / "corpus" / "reads.arrow").exists()
    assert (out / "corpus" / "reads.fa").exists()
    assert (out / "corpus" / "_SUCCESS").exists()

    rd = out / "rounds" / "r01"
    assert (rd / "_SUCCESS").exists()
    assert (rd / "templates" / "templates.arrow").exists()
    assert (rd / "templates" / "templates.fa").exists()
    assert sorted((rd / "assignments").glob("part-*.parquet"))
    assert (rd / "round.json").exists()
    assert (out / "churn.tsv").exists()


def test_every_read_is_accounted_for(corpus_dir, tmp_path):
    """Assigned + unassigned must equal the reads minimap2 reported on."""
    results = run_em(corpus_dir, tmp_path / "em", params=_params(rounds=1))
    st = results[0].estep
    assert st["n_assigned"] + st["n_unassigned"] == st["n_reads_seen"]
    assert st["n_assigned"] > 0


def test_templates_that_recruited_nothing_do_not_carry_forward(corpus_dir, tmp_path):
    """The only way a template disappears — and it is a consequence, not a gate."""
    out = tmp_path / "em"
    results = run_em(corpus_dir, out, params=_params(rounds=2))
    if len(results) < 2:
        pytest.skip("converged in one round on this fixture")
    # Round 1 seeds one template per fold group; round 2 sees only the ones
    # that actually recruited, so the set shrinks.
    assert results[1].n_templates <= results[0].n_templates
    assert (out / "rounds" / "r01" / "lineage.parquet").exists()


def test_resume_skips_a_completed_round(corpus_dir, tmp_path):
    out = tmp_path / "em"
    run_em(corpus_dir, out, params=_params(rounds=1))
    rd = out / "rounds" / "r01"
    before = {p.name for p in (rd / "assignments").glob("part-*.parquet")}
    stamp = (rd / "round.json").stat().st_mtime_ns

    run_em(corpus_dir, out, params=_params(rounds=1), resume=True)

    # Round 1 was not re-run.
    assert (rd / "round.json").stat().st_mtime_ns == stamp
    assert {p.name for p in (rd / "assignments").glob("part-*.parquet")} == before


def test_the_corpus_is_not_rewritten_on_resume(corpus_dir, tmp_path):
    out = tmp_path / "em"
    run_em(corpus_dir, out, params=_params(rounds=1))
    stamp = (out / "corpus" / "reads.arrow").stat().st_mtime_ns
    run_em(corpus_dir, out, params=_params(rounds=1), resume=True)
    assert (out / "corpus" / "reads.arrow").stat().st_mtime_ns == stamp


def test_an_empty_corpus_is_not_a_crash(tmp_path):
    demux = _write_demux(tmp_path, [("r0", "ACGT" * 5, 30.0)])
    assert run_em(demux, tmp_path / "em", params=_params(min_aa_length=200)) == []


def test_round_one_dissolves_self_capture_without_a_prune(tmp_path):
    """The claim the whole round-1 ranking rests on.

    Every read seeds a template, so under argmax-AS each would win its own
    seed by ~72 points at 1.2 kb and NOTHING would ever join anything —
    permanently, in every round. Ranking on ORF replication instead, the
    redundant self-templates simply recruit nothing and stop existing, which
    is a consequence of assignment rather than a prune.
    """
    rng = np.random.default_rng(7)
    truths = [_orf(rng, 150) for _ in range(4)]
    rows = [
        (f"g{t}_r{i}", _mutate(rng, truth, 0.012), float(rng.uniform(12, 34)))
        for t, truth in enumerate(truths)
        for i in range(40)
    ]
    out = tmp_path / "em"
    results = run_em(
        _write_demux(tmp_path, rows),
        out,
        params=_params(rounds=3, threads=2, minimap2_n=100, min_aa_length=40),
    )

    assert results[0].n_templates > 50, "one template per fold group to begin with"
    # The great majority recruit nothing and do not carry forward.
    assert results[1].n_templates < results[0].n_templates / 5


def test_the_loop_converges_and_keeps_genes_apart(tmp_path):
    """Churn falls to zero and no cluster mixes two transcripts."""
    rng = np.random.default_rng(7)
    truths = [_orf(rng, 150) for _ in range(4)]
    rows, truth_of = [], {}
    for t, truth in enumerate(truths):
        for i in range(40):
            rid = f"g{t}_r{i}"
            truth_of[rid] = t
            rows.append((rid, _mutate(rng, truth, 0.012), float(rng.uniform(12, 34))))

    out = tmp_path / "em"
    results = run_em(
        _write_demux(tmp_path, rows),
        out,
        params=_params(rounds=3, threads=2, minimap2_n=100, min_aa_length=40),
    )
    assert results[-1].churn.get("frac_changed_lineage", 1.0) < 0.02

    last = pq.read_table(
        out / "rounds" / f"r{results[-1].round_index:02d}" / "assignments"
    )
    per_cluster: dict[int, dict[int, int]] = {}
    for rid, tid in zip(
        last.column("read_id").to_pylist(), last.column("template_id").to_pylist()
    ):
        if tid >= 0:
            per_cluster.setdefault(tid, {}).setdefault(truth_of[rid], 0)
            per_cluster[tid][truth_of[rid]] += 1
    total = sum(sum(c.values()) for c in per_cluster.values())
    dominant = sum(max(c.values()) for c in per_cluster.values())
    assert total > 0
    assert dominant / total >= 0.99, "a cluster must not mix two transcripts"


def test_the_user_facing_outputs_are_written_in_the_shared_shapes(corpus_dir, tmp_path):
    """Every existing consumer reads these; none may need a clusterer branch."""
    import pyarrow as pa_
    from constellation.sequencing.schemas.transcriptome import (
        CLUSTER_MEMBERSHIP_TABLE,
        TRANSCRIPT_CLUSTER_TABLE,
    )

    out = tmp_path / "em"
    run_em(corpus_dir, out, params=_params(rounds=1))

    clusters = pq.read_table(out / "clusters.parquet")
    membership = pq.read_table(out / "cluster_membership.parquet")
    assert clusters.schema.equals(TRANSCRIPT_CLUSTER_TABLE)
    assert membership.schema.equals(CLUSTER_MEMBERSHIP_TABLE)
    assert clusters.num_rows > 0

    # The renamed mode vocabulary, not the pre-rename spelling.
    assert set(clusters.column("mode").to_pylist()) == {"em"}
    # Genome columns stay null: this is a reference-free clusterer.
    assert clusters.column("contig_id").null_count == clusters.num_rows

    # The schema requires a representative even though a round-2 frame is a
    # consensus with no read of its own.
    assert all(clusters.column("representative_read_id").to_pylist())
    assert (out / "cluster.fa").exists()
    assert (out / "feature_quant.parquet").exists()

    # Exactly one representative per cluster that holds reads.
    roles = {}
    for cid, role in zip(
        membership.column("cluster_id").to_pylist(),
        membership.column("role").to_pylist(),
    ):
        roles.setdefault(cid, []).append(role)
    for cid, rs in roles.items():
        assert rs.count("representative") == 1, cid


def test_the_diagnostics_report_is_emitted_and_reads_the_real_artifacts(
    corpus_dir, tmp_path
):
    """Every metric is a pure function over what the loop already wrote."""
    out = tmp_path / "em"
    run_em(corpus_dir, out, params=_params(rounds=2))

    report = (out / "diagnostics" / "report.md").read_text()
    for heading in (
        "Convergence",
        "Candidate pool",
        "Assignment rule",
        "Reference drift",
        "Cluster sizes",
    ):
        assert f"## {heading}" in report

    # Regenerable read-only against a finished run.
    from constellation.sequencing.transcriptome.cluster.denovo.em.diagnostics import (
        build_em_report,
    )

    assert build_em_report(out).exists()


def test_a_saturated_candidate_pool_is_flagged(corpus_dir, tmp_path):
    """-N is a correctness parameter; truncation must be loud, not logged."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.diagnostics import (
        section_candidate_pool,
    )

    out = tmp_path / "em"
    # -N 1 guarantees every read's pool is truncated.
    run_em(corpus_dir, out, params=_params(rounds=1, minimap2_n=1))
    section = section_candidate_pool(out)
    assert section.flags, "a truncated pool must raise a flag"
    assert "TRUNCATED" in section.flags[0]


# ── review regressions ────────────────────────────────────────────────


def test_a_read_minimap2_never_reports_is_still_accounted_for(tmp_path):
    """No PAF line means no row — the read vanished from the rejection rate.

    minimap2 emits nothing for a read with no hit above its chaining
    threshold, so such reads were neither assigned nor unassigned and had no
    row at all. That silently understates the number the admission floor
    exists to make visible.
    """
    rng = np.random.default_rng(5)
    truth = _orf(rng, 120)
    rows = [
        (f"real_{i}", _mutate(rng, truth, 0.01), 30.0) for i in range(8)
    ]
    # Something with no relationship to the rest, and no ORF of its own.
    rows.append(("stranger", "AT" * 300, 30.0))

    out = tmp_path / "em"
    results = run_em(
        _write_demux(tmp_path, rows), out, params=_params(rounds=1, min_aa_length=40)
    )
    st = results[0].estep
    # It really is the unmapped path, not the identity floor: minimap2
    # reported nothing at all for this read.
    assert st["n_unmapped"] == 1
    assert st["n_assigned"] + st["n_unassigned"] == st["n_reads_seen"]
    assert st["n_reads_seen"] == 9, "every corpus read is accounted for"

    table = pq.read_table(out / "rounds" / "r01" / "assignments")
    assert table.num_rows == 9
    ids = table.column("read_id").to_pylist()
    assert "stranger" in ids
    row = table.filter(pa.compute.equal(table.column("read_id"), "stranger"))
    assert row.column("template_id").to_pylist() == [-1]


def test_a_truncated_template_file_falls_back_to_the_finished_round(tmp_path):
    """An interrupted write must not be trusted just because the file exists."""
    corpus = _write_demux(
        tmp_path,
        [
            (f"r{i}", _mutate(np.random.default_rng(i), _orf(np.random.default_rng(1), 120), 0.01), 30.0)
            for i in range(12)
        ],
    )
    out = tmp_path / "em"
    run_em(corpus, out, params=_params(rounds=1))

    # Simulate an interruption during the NEXT round's template write.
    nxt = out / "rounds" / "r02" / "templates"
    nxt.mkdir(parents=True, exist_ok=True)
    (nxt / "templates.arrow").write_bytes(b"ARROW1\x00\x00truncated")

    # Must rebuild from round 1's nodes rather than dying on the stub.
    results = run_em(corpus, out, params=_params(rounds=1), resume=True)
    assert results, "resume must recover rather than raise"


def test_resume_keeps_the_churn_history_and_measures_the_resumed_round(tmp_path):
    """Otherwise churn.tsv holds only this invocation and round 2 reports none."""
    corpus_rows = []
    rng = np.random.default_rng(9)
    for t_idx in range(2):
        truth = _orf(rng, 130)
        for i in range(20):
            corpus_rows.append(
                (f"g{t_idx}_r{i}", _mutate(rng, truth, 0.01), 30.0)
            )
    corpus = _write_demux(tmp_path, corpus_rows)
    out = tmp_path / "em"

    run_em(corpus, out, params=_params(rounds=1, min_aa_length=40))
    run_em(corpus, out, params=_params(rounds=1, min_aa_length=40), resume=True)

    rounds = [
        line.split("\t")[0]
        for line in (out / "churn.tsv").read_text().splitlines()[1:]
    ]
    assert rounds == ["1", "2"], "history must survive the resumed invocation"

    r2 = json.loads((out / "rounds" / "r02" / "round.json").read_text())
    assert r2["churn"], "the resumed round has a previous round to compare to"


def test_a_completed_run_is_a_readable_stage(tmp_path):
    """Without a manifest the viz layer cannot attach the directory at all."""
    from constellation.sequencing.transcriptome.manifest import read_manifest_dir

    out = tmp_path / "em"
    run_em(_write_demux(tmp_path, [
        ("a", "ATG" + "GCT" * 60 + "TAA", 30.0),
        ("b", "ATG" + "GCT" * 60 + "TAA", 30.0),
    ]), out, params=_params(rounds=1, min_aa_length=40))

    manifest = read_manifest_dir(out)
    assert manifest.kind == "cluster"
    assert manifest.stages["n_clusters"] >= 1
    assert manifest.parameters["mode"] == "em"


def test_the_report_shows_the_metric_the_stopping_rule_reads(corpus_dir, tmp_path):
    """A broken section renders as a note, so it must be asserted, not eyeballed."""
    out = tmp_path / "em"
    run_em(corpus_dir, out, params=_params(rounds=2))
    report = (out / "diagnostics" / "report.md").read_text()
    section = report.split("## Convergence", 1)[1].split("## ", 1)[0]
    assert "could not be computed" not in section
    assert "unsettled" in section
    header = (out / "churn.tsv").read_text().splitlines()[0]
    for col in ("frac_unsettled", "reads_gained", "reads_lost"):
        assert col in header


def test_a_retried_estep_does_not_inherit_the_dead_attempt_s_shards(tmp_path):
    """Shards are numbered from zero and the reader globs the directory.

    So an attempt that crashed after writing N shards, followed by one that
    writes fewer, leaves the tail of the DEAD attempt for the reader to pick
    up as this round's assignments — reads counted twice, on templates that
    may no longer exist.
    """
    rng = np.random.default_rng(5)
    truth = _orf(rng, 120)
    corpus = _write_demux(
        tmp_path, [(f"r{i}", _mutate(rng, truth, 0.01), 30.0) for i in range(12)]
    )
    out = tmp_path / "em"
    run_em(corpus, out, params=_params(rounds=1))

    adir = out / "rounds" / "r01" / "assignments"
    real = sorted(adir.glob("part-*.parquet"))
    assert real
    clean = pq.read_table(real[0]).schema
    # A leftover from a previous, longer attempt.
    stale = pq.read_table(real[0])
    pq.write_table(stale.cast(clean), adir / "part-09999.parquet")
    n_stale = stale.num_rows

    shutil.rmtree(out / "rounds" / "r01" / "mstep", ignore_errors=True)
    (out / "rounds" / "r01" / "_SUCCESS").unlink()
    run_em(corpus, out, params=_params(rounds=1))

    assert not (adir / "part-09999.parquet").exists(), "surplus shard survived"
    table = pa_ds_table(adir)
    assert len(set(table.column("read_id").to_pylist())) == table.num_rows, (
        f"{n_stale} reads counted twice from the previous attempt"
    )


def pa_ds_table(directory: Path) -> pa.Table:
    import pyarrow.dataset as pa_ds

    return pa_ds.dataset(sorted(Path(directory).glob("part-*.parquet"))).to_table()


def test_a_round_is_not_marked_done_until_its_lineage_is_on_disk(
    tmp_path, monkeypatch
):
    """`_SUCCESS` means "everything the next round needs is written".

    lineage.parquet is one of those things — without it the next round cannot
    tell a template split from a genuine switch. It used to be written AFTER
    the marker, so a run killed in between left a round that claimed to be
    complete and was not, and the resumed run then died on ArrowInvalid.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em import (
        rounds as rounds_mod,
    )

    rng = np.random.default_rng(7)
    rows = []
    for g in range(2):
        truth = _orf(rng, 130)
        rows += [(f"g{g}_r{i}", _mutate(rng, truth, 0.01), 30.0) for i in range(20)]
    corpus = _write_demux(tmp_path, rows)
    out = tmp_path / "em"

    def _die(*a, **k):
        raise RuntimeError("killed between the marker and the lineage")

    monkeypatch.setattr(rounds_mod.rf, "next_templates", _die)
    with pytest.raises(RuntimeError):
        run_em(corpus, out, params=_params(rounds=2, min_aa_length=40))

    r1 = out / "rounds" / "r01"
    assert not (r1 / "_SUCCESS").exists() or (r1 / "lineage.parquet").exists(), (
        "round 1 is marked complete without the lineage round 2 reads"
    )


def test_a_half_written_lineage_is_replaced_rather_than_trusted(tmp_path):
    """A parquet file is only readable once its footer lands.

    So an interrupted write leaves a path that EXISTS and cannot be opened.
    Recovery keyed on existence declined to rebuild it, and the resumed round
    died reading it.
    """
    rng = np.random.default_rng(13)
    rows = []
    for g in range(2):
        truth = _orf(rng, 130)
        rows += [(f"g{g}_r{i}", _mutate(rng, truth, 0.01), 30.0) for i in range(20)]
    corpus = _write_demux(tmp_path, rows)
    out = tmp_path / "em"
    run_em(corpus, out, params=_params(rounds=1, min_aa_length=40))

    r1 = out / "rounds" / "r01"
    (r1 / "lineage.parquet").write_bytes(b"PAR1\x00\x00truncated")
    shutil.rmtree(out / "rounds" / "r02", ignore_errors=True)

    results = run_em(corpus, out, params=_params(rounds=1, min_aa_length=40),
                     resume=True)
    assert results, "resume must replace the damaged lineage, not die on it"
    pq.read_table(r1 / "lineage.parquet")


def test_stale_optional_exports_do_not_survive_a_later_run(tmp_path):
    """proteins.fasta / cluster.fa / feature_quant are all conditional.

    A file the current export does not produce is not "unchanged" — it
    describes results that no longer exist, and nothing on disk marks it
    stale.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.outputs import (
        CLUSTER_MEMBERSHIP_TABLE,
        TRANSCRIPT_CLUSTER_TABLE,
        write_em_outputs,
    )

    out = tmp_path / "exports"
    out.mkdir()
    for name in ("proteins.fasta", "cluster.fa", "feature_quant.parquet"):
        (out / name).write_bytes(b"from an earlier run\n")

    write_em_outputs(
        out,
        TRANSCRIPT_CLUSTER_TABLE.empty_table(),
        CLUSTER_MEMBERSHIP_TABLE.empty_table(),
        None,
    )
    for name in ("proteins.fasta", "cluster.fa", "feature_quant.parquet"):
        assert not (out / name).exists(), f"{name} outlived the results it described"


# ── seeding mode (`--mode em-orf` vs `--mode em-kmer`) ────────────────


def _kmer_params(**kw):
    """The kmer seeder, with the chaining guard scaled to a fixture.

    The guard's thresholds are fractions of a 9.4M-read corpus; a two-
    transcript fixture is 50% per cluster by construction, so the absolute
    floor is what keeps it quiet here — and that floor is exactly what these
    tests must not depend on.
    """
    base = dict(
        seeding="kmer",
        seed_identity=0.90,
        min_chain_cluster_reads=0,
        max_cluster_read_frac=0.0,
    )
    base.update(kw)
    return _params(**base)


def test_the_kmer_seeder_produces_far_fewer_round_one_templates(corpus_dir, tmp_path):
    """The whole point: one template per read CLUSTER, not per distinct ORF.

    At 9.39M reads that is 3,778,760 ORF templates against 850,450 kmer ones,
    and a round-1 E-step of 13.95 h against 2.89 h. The fixture reproduces the
    direction, not the ratio.
    """
    orf = run_em(corpus_dir, tmp_path / "orf", params=_params(rounds=1))
    kmer = run_em(corpus_dir, tmp_path / "kmer", params=_kmer_params(rounds=1))

    assert kmer[0].n_templates < orf[0].n_templates
    # Two transcripts in, two templates out — the seeder has already done what
    # the ORF path needs a whole E-step to approach.
    assert kmer[0].n_templates == 2


def test_the_kmer_seed_stage_is_addressable_and_stamped(corpus_dir, tmp_path):
    out = tmp_path / "em"
    run_em(corpus_dir, out, params=_kmer_params(rounds=1))

    seed = out / "seed"
    assert (seed / "_SUCCESS").exists()
    assert (seed / "templates.parquet").exists()
    assert (seed / "read_cluster.parquet").exists()
    stamp = json.loads((seed / "params.json").read_text())
    assert stamp["seeding"] == "kmer"
    assert stamp["seed_identity"] == 0.90
    stats = json.loads((seed / "stats.json").read_text())
    assert stats["n_clusters"] == 2
    # Every read accounted for, so the diagnostics can price the size filter.
    assert stats["n_reads"] == stats["n_reads_in_templates"]


def test_the_seeding_mode_reaches_the_manifest(corpus_dir, tmp_path):
    out = tmp_path / "em"
    run_em(corpus_dir, out, params=_kmer_params(rounds=1))
    manifest = json.loads((out / "manifest.json").read_text())
    params = manifest["parameters"]
    assert params["seeding"] == "kmer"
    assert params["seed_max_3p_overhang"] == 100
    # The mode COLUMN stays "em": the mechanism is the EM loop, the seeder is
    # a parameter of it, and widening the column would touch every consumer.
    assert params["mode"] == "em"


def test_a_resume_across_a_changed_seeder_is_refused(corpus_dir, tmp_path):
    """Reusing seed/templates.parquet under a different seeder is silent-wrong.

    The templates would be one seeder's and the manifest would record the
    other's, with nothing on disk marking the disagreement.
    """
    out = tmp_path / "em"
    run_em(corpus_dir, out, params=_params(rounds=1))

    with pytest.raises(ValueError, match="produced by the 'orf' seeder"):
        run_em(corpus_dir, out, params=_kmer_params(rounds=1), resume=True)


def test_a_resume_across_a_changed_seed_gate_is_refused(corpus_dir, tmp_path):
    out = tmp_path / "em"
    run_em(corpus_dir, out, params=_kmer_params(rounds=1))

    with pytest.raises(ValueError, match="seed_identity"):
        run_em(
            corpus_dir,
            out,
            params=_kmer_params(rounds=1, seed_identity=0.96),
            resume=True,
        )


def test_an_unchanged_seed_stage_resumes(corpus_dir, tmp_path):
    out = tmp_path / "em"
    first = run_em(corpus_dir, out, params=_kmer_params(rounds=1))
    again = run_em(corpus_dir, out, params=_kmer_params(rounds=1), resume=True)
    assert again[0].n_templates == first[0].n_templates


def test_a_pre_stamp_seed_dir_cannot_be_resumed_as_kmer(corpus_dir, tmp_path):
    """Output dirs from before this change hold ORF templates, unlabelled."""
    out = tmp_path / "em"
    run_em(corpus_dir, out, params=_params(rounds=1))
    (out / "seed" / "params.json").unlink()

    # The ORF path resumes it, because that is the only thing it can be.
    assert run_em(corpus_dir, out, params=_params(rounds=1), resume=True)
    with pytest.raises(ValueError, match="predates seed-parameter stamping"):
        run_em(corpus_dir, out, params=_kmer_params(rounds=1), resume=True)
