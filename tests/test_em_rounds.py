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


def test_the_two_pass_estep_converges_and_keeps_genes_apart(tmp_path):
    """The same panel under --estep-aligner edlib, through a real worker pool.

    Pins that the two-pass path composes end to end — minimap2 without -c,
    shortlist, edlib finalists, M-step on edlib CIGARs — at the same purity
    bar the single-pass path is held to.
    """
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
        params=_params(
            rounds=3,
            threads=2,
            minimap2_n=100,
            min_aa_length=40,
            estep_aligner="edlib",
            estep_align_workers=2,
        ),
    )
    assert results[-1].estep["aligner"] == "edlib"
    assert results[-1].churn.get("frac_changed_lineage", 1.0) < 0.02
    last = pq.read_table(
        out / "rounds" / f"r{results[-1].round_index:02d}" / "assignments"
    )
    assert pc_count_valid(last.column("chain_score")) > 0
    per_cluster: dict[int, dict[int, int]] = {}
    for rid, tid in zip(
        last.column("read_id").to_pylist(), last.column("template_id").to_pylist()
    ):
        if tid >= 0:
            per_cluster.setdefault(tid, {}).setdefault(truth_of[rid], 0)
            per_cluster[tid][truth_of[rid]] += 1
    total = sum(sum(c.values()) for c in per_cluster.values())
    dominant = sum(max(c.values()) for c in per_cluster.values())
    assert total >= 0.9 * len(rows)
    assert dominant / total >= 0.99, "a cluster must not mix two transcripts"


def pc_count_valid(col) -> int:
    return len(col) - col.null_count


def test_resume_refuses_to_switch_estep_aligners(tmp_path):
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        _check_estep_stamp,
    )

    rounds = tmp_path / "rounds"
    _check_estep_stamp(rounds, _params(), resume=False)
    (rounds / "r01").mkdir()
    (rounds / "r01" / "_SUCCESS").write_bytes(b"")
    _check_estep_stamp(rounds, _params(), resume=True)  # same aligner: fine
    with pytest.raises(ValueError, match="estep-aligner"):
        _check_estep_stamp(rounds, _params(estep_aligner="edlib"), resume=True)
    # A stampless run predates the flag and can only be minimap2's.
    (rounds / "estep.json").unlink()
    with pytest.raises(ValueError, match="minimap2"):
        _check_estep_stamp(rounds, _params(estep_aligner="edlib"), resume=True)


def test_resume_refuses_a_changed_floor_rule_or_native_knob(tmp_path):
    """Admission is per round; a floor that moved mid-run makes the rounds'
    assignments incomparable, exactly like a switched aligner."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        _check_estep_stamp,
    )

    rounds = tmp_path / "rounds"
    _check_estep_stamp(rounds, _params(), resume=False)
    (rounds / "r01").mkdir()
    (rounds / "r01" / "_SUCCESS").write_bytes(b"")
    with pytest.raises(ValueError, match="p_floor_quality_scale"):
        _check_estep_stamp(rounds, _params(p_floor_quality_scale=1.5), resume=True)
    # A stampless directory reads as the defaults, so the default passes...
    (rounds / "estep.json").unlink()
    _check_estep_stamp(rounds, _params(), resume=True)
    # ...and a two-pass run stamped before the shortlist keys existed ran
    # the old 16-deep shortlist, so the new 32 default refuses it unless
    # the old value is restored.
    edlib_dir = tmp_path / "edlib" / "rounds"
    (edlib_dir / "r01").mkdir(parents=True)
    (edlib_dir / "r01" / "_SUCCESS").write_bytes(b"")
    (edlib_dir / "estep.json").write_text(json.dumps({"estep_aligner": "edlib"}))
    with pytest.raises(ValueError, match="estep_shortlist_k"):
        _check_estep_stamp(edlib_dir, _params(estep_aligner="edlib"), resume=True)
    # Restoring the old depth is not enough for THAT run: it also predates
    # the placement guard (review of 91e7c69), a rule with no flag to restore.
    (edlib_dir / "estep.json").write_text(json.dumps({"estep_aligner": "edlib"}))
    with pytest.raises(ValueError, match="earlier two-pass"):
        _check_estep_stamp(
            edlib_dir,
            _params(estep_aligner="edlib", estep_shortlist_k=16),
            resume=True,
        )
    # A run stamped under today's rules resumes at whatever depth it ran.
    (edlib_dir / "estep.json").write_text(
        json.dumps(
            {
                "estep_aligner": "edlib",
                "estep_shortlist_k": 16,
                "estep_shortlist_frac": 0.8,
                "two_pass_rules": 2,
            }
        )
    )
    _check_estep_stamp(
        edlib_dir,
        _params(estep_aligner="edlib", estep_shortlist_k=16),
        resume=True,
    )
    # The single-pass minimap2 path is not what changed, and is not stamped.
    assert "two_pass_rules" not in json.loads((rounds / "estep.json").read_text())
    # ...and the native path stamps its join parameters: hand-edit one and
    # the resume is refused by its name.
    native = tmp_path / "native" / "rounds"
    _check_estep_stamp(native, _params(estep_aligner="native"), resume=False)
    (native / "r01").mkdir()
    (native / "r01" / "_SUCCESS").write_bytes(b"")
    _check_estep_stamp(native, _params(estep_aligner="native"), resume=True)
    stamp = json.loads((native / "estep.json").read_text())
    assert stamp["kmer"] == 15 and stamp["bucket_cap"] == 20_480
    stamp["bucket_cap"] = 1_024
    (native / "estep.json").write_text(json.dumps(stamp))
    with pytest.raises(ValueError, match="bucket_cap"):
        _check_estep_stamp(native, _params(estep_aligner="native"), resume=True)


def test_a_kmer_seeded_native_loop_runs_round_one_on_the_band_rule(
    tmp_path, monkeypatch
):
    """em-kmer + native is the whole loop with no minimap2 anywhere, and
    its round 1 ranks by identity with the noise band — the rule is the
    SEEDER's property and it is stamped, so an old run (stampless, which
    could only have walked by replication) cannot silently continue under
    the new rule."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        _check_estep_stamp,
    )

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    corpus = _panel(tmp_path)
    out = tmp_path / "em"
    params = _params(rounds=2, min_aa_length=40, seeding="kmer", estep_aligner="native")
    assert params.round1_rule() == "identity_band"
    assert _params().round1_rule() == "replication", "em-orf keeps the walk"
    run_em(corpus, out, params=params)
    clusters = pq.read_table(out / "clusters.parquet")
    assert clusters.num_rows == 2
    assert sum(clusters.column("n_reads").to_pylist()) == 40
    assert _json(out / "rounds" / "estep.json")["round1_rule"] == "identity_band"

    # The aligner mismatch fires first on a stampless directory; the rule
    # has to refuse on its own even when the aligner matches.
    stampless = tmp_path / "old" / "rounds"
    (stampless / "r01").mkdir(parents=True)
    (stampless / "r01" / "_SUCCESS").write_bytes(b"")
    with pytest.raises(ValueError, match="round1_rule"):
        _check_estep_stamp(
            stampless,
            _params(rounds=2, min_aa_length=40, seeding="kmer"),
            resume=True,
        )


def test_the_loop_runs_natively_with_no_minimap2_anywhere(tmp_path, monkeypatch):
    """`--estep-aligner native` never launches minimap2: the loop must
    finish with the binary unreachable. The read sketch is built once, in
    the parent, and reused by the second round and by a resume."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    corpus = _panel(tmp_path)
    out = tmp_path / "em"
    run_em(
        corpus,
        out,
        params=_params(rounds=2, min_aa_length=40, estep_aligner="native"),
    )
    clusters = pq.read_table(out / "clusters.parquet")
    assert clusters.num_rows == 2
    assert sum(clusters.column("n_reads").to_pylist()) == 40
    stamp = _json(out / "rounds" / "estep.json")
    assert stamp["estep_aligner"] == "native" and stamp["kmer"] == 15
    minis = out / "corpus" / "minimizers"
    assert (minis / "minimizers.arrow").exists()
    r1 = _json(out / "rounds" / "r01" / "round.json")["estep"]
    assert r1["aligner"] == "native" and r1["n_assigned"] == 40
    assert r1["n_aligned"] <= 45, "round 1 aligns lazily"
    r2 = _json(out / "rounds" / "r02" / "round.json")["estep"]
    assert r2["n_newly_lost"] == 0
    for key in (
        "n_no_candidate",
        "n_below_floor",
        "n_no_alignment",
        "n_short_placement",
    ):
        assert key in r2
    a = pq.read_table(
        next((out / "rounds" / "r02" / "assignments").glob("part-*.parquet"))
    )
    assert all(v is not None for v in a.column("identity").to_pylist())

    before = (minis / "minimizers.arrow").stat().st_mtime_ns
    run_em(
        corpus,
        out,
        params=_params(rounds=1, min_aa_length=40, estep_aligner="native"),
        resume=True,
    )
    assert (minis / "minimizers.arrow").stat().st_mtime_ns == before


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
        "Template relationships",
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
    rows = [(f"real_{i}", _mutate(rng, truth, 0.01), 30.0) for i in range(8)]
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
            (
                f"r{i}",
                _mutate(
                    np.random.default_rng(i), _orf(np.random.default_rng(1), 120), 0.01
                ),
                30.0,
            )
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
            corpus_rows.append((f"g{t_idx}_r{i}", _mutate(rng, truth, 0.01), 30.0))
    corpus = _write_demux(tmp_path, corpus_rows)
    out = tmp_path / "em"

    run_em(corpus, out, params=_params(rounds=1, min_aa_length=40))
    run_em(corpus, out, params=_params(rounds=1, min_aa_length=40), resume=True)

    rounds = [
        line.split("\t")[0] for line in (out / "churn.tsv").read_text().splitlines()[1:]
    ]
    assert rounds == ["1", "2"], "history must survive the resumed invocation"

    r2 = json.loads((out / "rounds" / "r02" / "round.json").read_text())
    assert r2["churn"], "the resumed round has a previous round to compare to"


def test_a_completed_run_is_a_readable_stage(tmp_path):
    """Without a manifest the viz layer cannot attach the directory at all."""
    from constellation.sequencing.transcriptome.manifest import read_manifest_dir

    out = tmp_path / "em"
    run_em(
        _write_demux(
            tmp_path,
            [
                ("a", "ATG" + "GCT" * 60 + "TAA", 30.0),
                ("b", "ATG" + "GCT" * 60 + "TAA", 30.0),
            ],
        ),
        out,
        params=_params(rounds=1, min_aa_length=40),
    )

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


def test_a_round_is_not_marked_done_until_its_lineage_is_on_disk(tmp_path, monkeypatch):
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

    results = run_em(
        corpus, out, params=_params(rounds=1, min_aa_length=40), resume=True
    )
    assert results, "resume must replace the damaged lineage, not die on it"
    pq.read_table(r1 / "lineage.parquet")


def test_stale_optional_exports_do_not_survive_a_later_run(tmp_path):
    """proteins.fasta / cluster.fa / feature_quant / cluster_edges are all
    conditional.

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
    optional = (
        "proteins.fasta",
        "cluster.fa",
        "feature_quant.parquet",
        "cluster_edges.parquet",
    )
    for name in optional:
        (out / name).write_bytes(b"from an earlier run\n")

    write_em_outputs(
        out,
        TRANSCRIPT_CLUSTER_TABLE.empty_table(),
        CLUSTER_MEMBERSHIP_TABLE.empty_table(),
        None,
    )
    for name in optional:
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


# ── the template graph, and the opt-in merge ─────────────────────────
#
# The synthetic panels make nothing mergeable of their own (their templates
# differ in extent or sequence), so these inject it: a second node carrying
# the first one's consensus and half its reads — the shape the real-data
# M-step leaves behind. It is written into the node and membership SHARDS,
# not only returned, because a resumed run rebuilds a round's successor from
# the shards and would otherwise never see it.


def _panel(tmp_path, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(2):
        truth = _orf(rng, 130)
        rows += [(f"g{g}_r{i}", _mutate(rng, truth, 0.01), 30.0) for i in range(20)]
    return _write_demux(tmp_path, rows)


def _twin_first_node(monkeypatch, *, sibling: bool, trim_5p: int = 0):
    """Give node 0 a twin after every M-step.

    ``sibling`` makes the twin a second node of the SAME parent — what a split
    looks like. Otherwise it hangs off another template, so the two are
    unrelated by lineage. ``trim_5p`` shortens the twin's 5' end:
    a byte-identical twin is mergeable whatever its lineage, one that differs
    in extent is not.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em import (
        rounds as rounds_mod,
    )

    real = rounds_mod._run_mstep

    def _with_twin(r, rd, store, corpus, assignments, params, log):
        nodes, membership = real(r, rd, store, corpus, assignments, params, log)
        pid = int(nodes.column("parent_template_id")[0].as_py())
        hap = int(nodes.column("haplotype_id")[0].as_py())
        if sibling:
            twin_pid, twin_row, twin_hap = (
                pid,
                int(nodes.column("parent_template_row")[0].as_py()),
                99,
            )
        else:
            # A template that recruited nothing, if there is one; by round 2
            # there usually is not, and then another parent's node gets a
            # second haplotype. Either way the twin is no kin of node 0.
            rows = nodes.column("parent_template_row").to_pylist()
            free = [i for i in range(store.n_templates) if i not in set(rows)]
            other = [i for i in rows if i != rows[0]]
            twin_row, twin_hap = (free[0], 0) if free else (other[0], 99)
            twin_pid = int(store.template_id[twin_row])

        m_pid = membership.column("parent_template_id").to_numpy()
        m_hap = membership.column("haplotype_id").to_numpy()
        mine = np.flatnonzero((m_pid == pid) & (m_hap == hap))
        moved = mine[: mine.size // 2]
        new_pid, new_hap = m_pid.copy(), m_hap.copy()
        new_pid[moved], new_hap[moved] = twin_pid, twin_hap
        membership = (
            membership.set_column(
                membership.schema.get_field_index("parent_template_id"),
                "parent_template_id",
                pa.array(new_pid, pa.int64()),
            )
            .set_column(
                membership.schema.get_field_index("haplotype_id"),
                "haplotype_id",
                pa.array(new_hap.astype(np.int32)),
            )
            .cast(membership.schema)
        )

        def put(table, name, values, kind):
            i = table.schema.get_field_index(name)
            return table.set_column(i, table.schema.field(i), pa.array(values, kind))

        first, twin = nodes.slice(0, 1), nodes.slice(0, 1)
        first = put(first, "n_reads", [mine.size - moved.size], pa.int64())
        first = put(first, "node_weight", [float(mine.size - moved.size)], pa.float64())
        twin = put(twin, "parent_template_id", [twin_pid], pa.int64())
        twin = put(twin, "parent_template_row", [twin_row], pa.int32())
        twin = put(twin, "haplotype_id", [twin_hap], pa.int32())
        twin = put(twin, "n_reads", [moved.size], pa.int64())
        twin = put(twin, "node_weight", [float(moved.size)], pa.float64())
        if trim_5p:
            seq = twin.column("consensus")[0].as_py()[trim_5p:]
            twin = put(twin, "consensus", [seq], pa.large_string())
        nodes = pa.concat_tables([first, nodes.slice(1), twin]).cast(nodes.schema)

        for sub, table in (("nodes", nodes), ("node_membership", membership)):
            shard_dir = rd / "mstep" / sub
            for old in shard_dir.glob("part-*.parquet"):
                old.unlink()
            pq.write_table(table, shard_dir / "part-00000.parquet")
        return nodes, membership

    monkeypatch.setattr(rounds_mod, "_run_mstep", _with_twin)


def _json(path):
    return json.loads(Path(path).read_text())


def _r2_sequences(out):
    with pa.memory_map(
        str(out / "rounds" / "r02" / "templates" / "templates.arrow")
    ) as mm:
        return sorted(pa.ipc.open_file(mm).read_all().column("sequence").to_pylist())


def _templates_of(out, r):
    path = out / "rounds" / f"r{r:02d}" / "templates" / "templates.arrow"
    with pa.memory_map(str(path)) as mm:
        t = pa.ipc.open_file(mm).read_all()
    return list(
        zip(t.column("template_id").to_pylist(), t.column("sequence").to_pylist())
    )


def test_the_worker_count_does_not_change_the_template_ids(tmp_path):
    """Nodes are written one shard per M-step unit, and the units are a
    bin-packing over the worker count — so concatenated they stood in an
    order that depended on --mstep-workers, and template ids are row
    positions. Measured: the same nodes at two worker counts, one merge
    different in round 1, 94,560 clusters against 94,556 (ledger #60)."""
    # Enough live templates that one worker packs them into its 16 units
    # and three workers do not: 24 transcripts, 3 reads each.
    rng = np.random.default_rng(23)
    rows = []
    for g in range(24):
        truth = _orf(rng, 90 + g)
        rows += [(f"g{g}_r{i}", _mutate(rng, truth, 0.01), 30.0) for i in range(3)]
    corpus = _write_demux(tmp_path, rows)
    runs = {}
    shard_order = {}
    for workers in (1, 3):
        out = tmp_path / f"w{workers}"
        run_em(
            corpus,
            out,
            params=_params(rounds=2, min_aa_length=40, mstep_workers=workers),
        )
        shards = sorted(
            (out / "rounds" / "r01" / "mstep" / "nodes").glob("part-*.parquet")
        )
        shard_order[workers] = [
            pq.read_table(s, columns=["parent_template_id"]).column(0).to_pylist()
            for s in shards
        ]
        runs[workers] = (
            _templates_of(out, 2),
            pq.read_table(out / "clusters.parquet").to_pylist(),
            pq.read_table(out / "cluster_membership.parquet").to_pylist(),
        )
    assert runs[1][0] == runs[3][0], "same ids, same sequences, same rows"
    assert runs[1][1] == runs[3][1]
    assert runs[1][2] == runs[3][2]
    flat = {w: [p for shard in s for p in shard] for w, s in shard_order.items()}
    assert flat[1] != flat[3], "the shards stand in different orders, and that is fine"
    assert len(flat[1]) > 16


def test_report_only_writes_edges_and_merges_nothing(tmp_path, monkeypatch):
    """Report-only: the twin is an edge in the graph, and still a template."""
    _twin_first_node(monkeypatch, sibling=False)
    out = tmp_path / "em"
    run_em(
        _panel(tmp_path), out, params=_params(rounds=2, min_aa_length=40, merge=False)
    )

    r1 = out / "rounds" / "r01"
    record = _json(r1 / "refine.json")
    assert record["graph"] == "ok" and not record["merge_applied"]
    assert record["n_merged"] == 0
    edges = pq.read_table(r1 / "graph" / "edges.parquet")
    twins = edges.filter(pa.array(edges.column("n_edits").to_numpy() == 0))
    assert twins.num_rows >= 1
    assert "equivalent" in twins.column("relation").to_pylist()
    assert any(twins.column("mergeable").to_pylist())
    assert pq.read_table(r1 / "merged.parquet").num_rows == 0
    assert (
        "merge" not in pq.read_table(r1 / "lineage.parquet").column("rule").to_pylist()
    )
    seqs = _r2_sequences(out)
    assert len(seqs) > len(set(seqs)), "the twin should have reached round 2"

    from constellation.sequencing.transcriptome.cluster.denovo.em.diagnostics import (
        section_template_graph,
    )

    body = section_template_graph(out).body
    assert "| r1 |" in body and "Final output (r2)" in body
    assert (out / "cluster_edges.parquet").exists()
    assert _json(out / "manifest.json")["outputs"]["cluster_edges"] == (
        "cluster_edges.parquet"
    )


def test_an_exact_twin_is_merged_by_default(tmp_path, monkeypatch):
    _twin_first_node(monkeypatch, sibling=False)
    out = tmp_path / "em"
    params = _params(rounds=2, min_aa_length=40)
    assert params.merge and params.merge_max_edits == 2
    run_em(_panel(tmp_path), out, params=params)

    r1 = out / "rounds" / "r01"
    record = _json(r1 / "refine.json")
    assert record["merge_applied"] and record["n_merged"] >= 1
    assert record["n_templates_after"] == (
        record["n_templates_before"] - record["n_merged"]
    )
    assert record["predicate"]["max_edits"] == 2 and "graph_stamp" in record
    lin = pq.read_table(r1 / "lineage.parquet")
    assert "merge" in lin.column("rule").to_pylist()
    assert pq.read_table(r1 / "merged.parquet").num_rows == record["n_merged"]
    seqs = _r2_sequences(out)
    assert len(seqs) == len(set(seqs)), "the twin reached round 2"


def test_the_final_output_merges_twins_and_keeps_every_read(tmp_path, monkeypatch):
    """A final node whose reads the M-step split across an identical twin."""
    corpus = _panel(tmp_path)
    base = tmp_path / "base"
    run_em(corpus, base, params=_params(rounds=1, min_aa_length=40))
    base_clusters = pq.read_table(base / "clusters.parquet")

    _twin_first_node(monkeypatch, sibling=False)
    out = tmp_path / "em"
    run_em(corpus, out, params=_params(rounds=1, min_aa_length=40, merge=True))
    clusters = pq.read_table(out / "clusters.parquet")
    assert clusters.num_rows == base_clusters.num_rows
    assert sum(clusters.column("n_reads").to_pylist()) == sum(
        base_clusters.column("n_reads").to_pylist()
    )
    final = _json(out / "rounds" / "r01" / "final.json")
    assert final["merge_applied"] and final["n_merged"] == 1
    assert final["n_clusters"] == clusters.num_rows
    assert final["n_twin_clusters"] == 0
    # The survivor was rebuilt from the pooled reads of both nodes, and the
    # consensus it reports is a sequence those reads support.
    assert final["rebuild"] == "ok" and final["n_rebuilt"] == 1
    assert final["n_rebuild_failed"] == 0
    assert final["predicate"]["max_edits"] == 2
    assert all(clusters.column("consensus_sequence").to_pylist())
    assert pq.read_table(out / "rounds" / "r01" / "merged_final.parquet").num_rows == 1

    edges = pq.read_table(out / "cluster_edges.parquet")
    src = edges.column("src_cluster_id").to_pylist()
    dst = edges.column("dst_cluster_id").to_pylist()
    assert all(0 <= i < clusters.num_rows for i in src + dst)
    assert all(a != b for a, b in zip(src, dst))


def test_split_siblings_are_not_merged_unless_asked(tmp_path, monkeypatch):
    """Two nodes of one parent that differ only in extent are what an M-step
    split on a start mode leaves behind. Merging them back is the limit
    cycle, so it takes --merge-siblings."""
    corpus = _panel(tmp_path)
    _twin_first_node(monkeypatch, sibling=True, trim_5p=12)

    kept = tmp_path / "kept"
    run_em(corpus, kept, params=_params(rounds=1, min_aa_length=40, merge=True))
    assert _json(kept / "rounds" / "r01" / "final.json")["n_merged"] == 0
    edges = pq.read_table(kept / "rounds" / "r01" / "graph" / "edges.parquet")
    kin = edges.filter(edges.column("same_split_origin"))
    assert kin.num_rows >= 1 and not any(kin.column("mergeable").to_pylist())

    joined = tmp_path / "joined"
    run_em(
        corpus,
        joined,
        params=_params(rounds=1, min_aa_length=40, merge=True, merge_siblings=True),
    )
    final = _json(joined / "rounds" / "r01" / "final.json")
    assert final["n_merged"] == 1
    # Nothing rebuilds a consensus after the final merge, so the longer of
    # the two is what is reported: no 12 nt are lost.
    merged = pq.read_table(joined / "rounds" / "r01" / "merged_final.parquet")
    assert merged.column("delta_5p").to_pylist() == [12]


def test_a_round_killed_after_its_graph_finds_it_on_disk(tmp_path, monkeypatch):
    """Killed after the graph and before the marker. The round has no marker,
    so the resume runs it again from its E-step — to the same nodes, and so
    to the graph that is already there."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import (
        rounds as rounds_mod,
    )

    _twin_first_node(monkeypatch, sibling=False)
    corpus = _panel(tmp_path)
    params = _params(rounds=2, min_aa_length=40, merge=True)
    clean = tmp_path / "clean"
    run_em(corpus, clean, params=params)

    out = tmp_path / "em"
    real_apply = rounds_mod.rf.apply_merge

    def _die(*a, **k):
        raise RuntimeError("killed between the graph and the marker")

    monkeypatch.setattr(rounds_mod.rf, "apply_merge", _die)
    with pytest.raises(RuntimeError):
        run_em(corpus, out, params=params)
    r1 = out / "rounds" / "r01"
    assert (r1 / "graph" / "_SUCCESS").exists()
    assert not (r1 / "_SUCCESS").exists() and not (r1 / "refine.json").exists()

    monkeypatch.setattr(rounds_mod.rf, "apply_merge", real_apply)
    built = []
    real_build = rounds_mod.gr.build_graph

    def _spy(*a, **k):
        built.append(k.get("node_round"))
        return real_build(*a, **k)

    monkeypatch.setattr(rounds_mod.gr, "build_graph", _spy)
    run_em(corpus, out, params=params, resume=True)
    assert _r2_sequences(out) == _r2_sequences(clean)
    assert 1 not in built, "round 1's graph was on disk and was built again"


def test_an_extended_run_merges_exactly_as_an_uninterrupted_one(tmp_path, monkeypatch):
    """A finished round has no successor: the resume REBUILDS one from the
    round's node shards, through the same refine-and-merge path. Two rounds
    run as one and then one must give what two rounds run together give."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import (
        rounds as rounds_mod,
    )

    _twin_first_node(monkeypatch, sibling=False)
    corpus = _panel(tmp_path)
    whole = tmp_path / "whole"
    run_em(corpus, whole, params=_params(rounds=2, min_aa_length=40, merge=True))

    out = tmp_path / "em"
    one = _params(rounds=1, min_aa_length=40, merge=True)
    run_em(corpus, out, params=one)
    r1 = out / "rounds" / "r01"
    assert not (r1 / "refine.json").exists(), "the last round is never refined"
    assert _json(r1 / "final.json")["n_merged"] >= 1

    rebuilt = []
    real = rounds_mod._refine_and_merge

    def _spy(rd, *a, **k):
        rebuilt.append(rd.name)
        return real(rd, *a, **k)

    monkeypatch.setattr(rounds_mod, "_refine_and_merge", _spy)
    run_em(corpus, out, params=one, resume=True)
    assert rebuilt == ["r01"], "the resume should have rebuilt round 1's successor"
    assert _r2_sequences(out) == _r2_sequences(whole)
    for name in ("lineage.parquet", "merged.parquet"):
        assert pq.read_table(r1 / name).equals(
            pq.read_table(whole / "rounds" / "r01" / name)
        ), name
    assert (
        _json(r1 / "refine.json")["n_merged"]
        == (_json(whole / "rounds" / "r01" / "refine.json")["n_merged"])
    )
    assert pq.read_table(out / "clusters.parquet").equals(
        pq.read_table(whole / "clusters.parquet")
    )


def test_a_rebuilt_round_replaces_the_lineage_it_found(tmp_path, monkeypatch):
    """A round refined by something that did not merge — an older version,
    here — has a lineage on disk that opens perfectly well. Rebuilt with
    merge on, it needs the `merge` rows: without them the reads of an
    absorbed template count as having chosen differently."""
    _twin_first_node(monkeypatch, sibling=False)
    corpus = _panel(tmp_path)
    out = tmp_path / "em"
    run_em(corpus, out, params=_params(rounds=2, min_aa_length=40, merge=False))
    r1 = out / "rounds" / "r01"
    assert (
        "merge" not in pq.read_table(r1 / "lineage.parquet").column("rule").to_pylist()
    )
    # What such a directory looks like: round 1 done, with its lineage and
    # no record of how it was refined; nothing after it.
    shutil.rmtree(out / "rounds" / "r02")
    (r1 / "refine.json").unlink()
    (out / "_SUCCESS").unlink(missing_ok=True)

    run_em(
        corpus,
        out,
        params=_params(rounds=1, min_aa_length=40, merge=True),
        resume=True,
    )
    record = _json(r1 / "refine.json")
    assert record["merge_applied"] and record["n_merged"] >= 1
    lineage = pq.read_table(r1 / "lineage.parquet")
    assert lineage.column("rule").to_pylist().count("merge") == record["n_merged"]
    seqs = _r2_sequences(out)
    assert len(seqs) == len(set(seqs))


def test_a_graph_left_by_an_earlier_run_does_not_outlive_the_graph_being_off(tmp_path):
    corpus = _panel(tmp_path)
    out = tmp_path / "em"
    run_em(corpus, out, params=_params(rounds=1, min_aa_length=40))
    assert (out / "rounds" / "r01" / "graph").exists()
    assert (out / "cluster_edges.parquet").exists()
    run_em(
        corpus,
        out,
        params=_params(rounds=1, min_aa_length=40, template_graph="off", merge=False),
        resume=True,
    )
    assert not list(out.glob("rounds/r*/graph"))
    assert not (out / "cluster_edges.parquet").exists()
    assert "cluster_edges" not in _json(out / "manifest.json")["outputs"]
    report = (out / "diagnostics" / "report.md").read_text()
    assert "template graph disabled" in report and "| r1 |" not in report


def test_template_graph_off_runs_no_alignment_at_all(tmp_path, monkeypatch):
    """The escape hatch: nothing after the M-step, not a quieter something."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import (
        rounds as rounds_mod,
    )

    def _never(*a, **k):
        raise AssertionError("the template graph ran with template_graph='off'")

    monkeypatch.setattr(rounds_mod.gr, "build_graph", _never)
    monkeypatch.setattr(rounds_mod.gr, "split_origins", _never)
    out = tmp_path / "em"
    run_em(
        _panel(tmp_path),
        out,
        params=_params(rounds=2, min_aa_length=40, template_graph="off", merge=False),
    )
    assert not list(out.glob("rounds/r*/graph"))
    assert not (out / "cluster_edges.parquet").exists()
    assert "cluster_edges" not in _json(out / "manifest.json")["outputs"]
    assert _json(out / "rounds" / "r01" / "refine.json")["graph"] == "off"
    assert _json(out / "rounds" / "r02" / "final.json")["graph"] == "off"
    report = (out / "diagnostics" / "report.md").read_text()
    assert "template graph disabled" in report


def test_template_graph_final_relates_only_the_last_round(tmp_path):
    out = tmp_path / "em"
    run_em(
        _panel(tmp_path),
        out,
        params=_params(rounds=2, min_aa_length=40, template_graph="final"),
    )
    assert not (out / "rounds" / "r01" / "graph").exists()
    assert (out / "rounds" / "r02" / "graph" / "_SUCCESS").exists()
    assert (out / "cluster_edges.parquet").exists()


def test_an_extended_run_has_one_final_round(tmp_path):
    """Every invocation finalises its own last round. When the run is
    extended that round gets a successor, and its final record must go."""
    corpus = _panel(tmp_path)
    out = tmp_path / "em"
    run_em(corpus, out, params=_params(rounds=1, min_aa_length=40))
    assert (out / "rounds" / "r01" / "final.json").exists()
    run_em(corpus, out, params=_params(rounds=1, min_aa_length=40), resume=True)
    assert not (out / "rounds" / "r01" / "final.json").exists()
    assert not (out / "rounds" / "r01" / "merged_final.parquet").exists()
    assert (out / "rounds" / "r02" / "final.json").exists()

    from constellation.sequencing.transcriptome.cluster.denovo.em.diagnostics import (
        section_template_graph,
    )

    assert section_template_graph(out).body.count("Final output") == 1


def test_merge_may_start_after_the_finished_rounds(tmp_path, monkeypatch):
    """The natural experiment: a finished report-only run, extended with
    merge on. Its recorded rounds did not merge, so merging "from round 1"
    is refused, and merging from the first unrecorded round is not."""
    _twin_first_node(monkeypatch, sibling=False)
    corpus = _panel(tmp_path)
    out = tmp_path / "em"
    run_em(corpus, out, params=_params(rounds=2, min_aa_length=40, merge=False))

    with pytest.raises(ValueError, match="--merge-from-round 2"):
        run_em(
            corpus,
            out,
            params=_params(rounds=1, min_aa_length=40, merge=True),
            resume=True,
        )
    run_em(
        corpus,
        out,
        params=_params(rounds=1, min_aa_length=40, merge=True, merge_from_round=2),
        resume=True,
    )
    assert _json(out / "rounds" / "r02" / "refine.json")["merge_applied"]
    assert not _json(out / "rounds" / "r01" / "refine.json")["merge_applied"]
