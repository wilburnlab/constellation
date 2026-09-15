"""The round loop, end to end on a small synthetic corpus.

This is the piece the EM stages were missing: seed / fold / E-step / M-step
all shipped and nothing iterated them. What matters here is that a round
completes, that its outputs are addressable on disk, and that a resumed run
picks up after the last complete round rather than starting over.

Needs minimap2 on ``$PATH``; skipped otherwise, since every other layer is
already tested without it.
"""

from __future__ import annotations

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
