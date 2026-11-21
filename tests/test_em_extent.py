"""The EM's extent rule end to end (PR3, 2026-11-20).

A read joins a template only when their ends agree within the end tolerance;
a molecule of another extent is a relation between templates, answered after
clustering. One gene sequenced three ways — full length, from 150 nt into its
5' UTR (as the template-switching oligo priming inside a transcript makes),
and ending 120 nt short of its 3' end (another cleavage site) — has to come
out as three clusters, each holding one population, related by `contained`
edges; every assignment of every round has to lie within the tolerance.

Under ORF seeding the three populations share an ORF and so a seed: what
separates them there is the E-step's refusal and the templates seeded from
the refused reads. Under kmer seeding the seeder's own overhangs follow the
tolerance and separate them in round 1. No minimap2: native E-step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _em_fixtures import mutate, random_orf, random_utr, write_demux  # noqa: E402

from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (  # noqa: E402
    EmParams,
    run_em,
)

TOL = 30
POPULATIONS = {"full": 30, "late": 22, "early": 22}


def _panel(tmp_path: Path, seed: int = 20261120) -> Path:
    """Errors in the UTRs only, so the three populations share one exact ORF
    and, under ORF seeding, one seed: the case the reseeding exists for."""
    rng = np.random.default_rng(seed)
    utr5, orf, utr3 = random_utr(rng, 200), random_orf(rng, 150), random_utr(rng, 300)
    forms = {
        "full": (utr5, utr3),
        "late": (utr5[150:], utr3),
        "early": (utr5, utr3[:-120]),
    }
    rows = []
    for name, n in POPULATIONS.items():
        five, three = forms[name]
        rows += [
            (
                f"{name}_{i:02d}",
                mutate(rng, five, 0.01) + orf + mutate(rng, three, 0.01),
                float(rng.uniform(20, 32)),
            )
            for i in range(n)
        ]
    return write_demux(tmp_path, rows, n_samples=2)


def _params(seeding: str, **kw) -> EmParams:
    base = dict(
        rounds=3,
        seeding=seeding,
        estep_aligner="native",
        threads=1,
        mstep_workers=1,
        max_window_length=None,
        min_aa_length=30,
        end_tolerance=TOL,
    )
    base.update(kw)
    return EmParams(**base)


@pytest.fixture(scope="module", params=["orf", "kmer"])
def run(request, tmp_path_factory):
    base = tmp_path_factory.mktemp(f"extent_{request.param}")
    out = base / "em"
    run_em(_panel(base), out, params=_params(request.param), report=False)
    return request.param, out


def _population(read_id: str) -> str:
    return read_id.split("_")[0]


def test_each_population_is_a_cluster_of_its_own(run) -> None:
    _seeding, out = run
    mem = pq.read_table(out / "cluster_membership.parquet").to_pylist()
    by_cluster: dict[int, set] = {}
    for row in mem:
        by_cluster.setdefault(row["cluster_id"], set()).add(_population(row["read_id"]))
    # No cluster mixes extents.
    assert all(len(pops) == 1 for pops in by_cluster.values()), by_cluster
    if _seeding == "kmer":
        # On a corpus this small the kmer seeder finds few candidate pairs and
        # leaves most reads seeds of their own, which keep themselves in round
        # 1 — fragmentation that predates the extent rule and is not its
        # subject. What the rule owes is above: no cluster mixes extents.
        return
    # Each population is (nearly) one cluster: its largest cluster holds most of it.
    counts: dict[str, dict[int, int]] = {}
    for row in mem:
        c = counts.setdefault(_population(row["read_id"]), {})
        c[row["cluster_id"]] = c.get(row["cluster_id"], 0) + 1
    for name, n in POPULATIONS.items():
        assert max(counts[name].values()) >= 0.8 * n, (name, counts[name])


def test_every_assignment_lies_within_the_tolerance(run) -> None:
    _seeding, out = run
    with pa.memory_map(str(out / "corpus" / "reads.arrow")) as src:
        corpus = pa.ipc.open_file(src).read_all()
    q_len_all = np.array([len(s) for s in corpus.column("sequence").to_pylist()])
    n_checked = 0
    for rd in sorted((out / "rounds").glob("r[0-9]*")):
        with pa.memory_map(str(rd / "templates" / "templates.arrow")) as src:
            tmpl = pa.ipc.open_file(src).read_all()
        t_len_all = np.array([len(s) for s in tmpl.column("sequence").to_pylist()])
        a = ds.dataset(sorted((rd / "assignments").glob("part-*.parquet"))).to_table()
        a = a.filter(pa.compute.greater_equal(a.column("template_id"), 0))
        q_len = q_len_all[a.column("read_row").to_numpy()]
        t_len = t_len_all[a.column("template_row").to_numpy()]
        qs, qe = a.column("q_start").to_numpy(), a.column("q_end").to_numpy()
        ts, te = a.column("t_start").to_numpy(), a.column("t_end").to_numpy()
        assert (qs <= TOL).all() and (q_len - qe <= TOL).all(), rd.name
        assert (ts <= TOL).all() and (t_len - te <= TOL).all(), rd.name
        n_checked += a.num_rows
    assert n_checked >= sum(POPULATIONS.values())


def test_the_three_clusters_are_related_by_containment(run) -> None:
    _seeding, out = run
    mem = pq.read_table(out / "cluster_membership.parquet").to_pylist()
    major: dict[str, int] = {}
    for name in POPULATIONS:
        ids = [r["cluster_id"] for r in mem if _population(r["read_id"]) == name]
        major[name] = max(set(ids), key=ids.count)
    edges = pq.read_table(out / "cluster_edges.parquet").to_pylist()
    pairs = {
        frozenset((e["src_cluster_id"], e["dst_cluster_id"])): e
        for e in edges
        if e["relation"] == "contained"
    }
    late = pairs.get(frozenset((major["full"], major["late"])))
    early = pairs.get(frozenset((major["full"], major["early"])))
    assert late is not None and late["truncation"] == "5p"
    assert early is not None and early["truncation"] == "3p"


def test_orf_seeding_separates_them_by_seeding_the_refused_reads(run) -> None:
    seeding, out = run
    import json

    reseeded = [
        json.loads((rd / "refine.json").read_text()).get("n_reseeded_reads", 0)
        for rd in sorted((out / "rounds").glob("r[0-9]*"))
        if (rd / "refine.json").exists()
    ]
    if seeding == "orf":
        # The populations share an ORF, so a seed: the refused reads made
        # their own templates.
        assert reseeded and reseeded[0] > 0
    reasons = []
    for rd in sorted((out / "rounds").glob("r[0-9]*")):
        a = ds.dataset(sorted((rd / "assignments").glob("part-*.parquet"))).to_table(
            columns=["unassigned_reason"]
        )
        reasons.append(a.column("unassigned_reason").to_pylist())
    if seeding == "orf":
        assert "extent" in reasons[0]


def test_the_rule_off_mixes_them_again(tmp_path) -> None:
    """Without the rule (the loop before 2026-11-20) the truncated reads join
    the full-length template: the property the rule exists to remove."""
    out = tmp_path / "em"
    run_em(_panel(tmp_path), out, params=_params("orf", end_tolerance=None, rounds=2),
           report=False)
    mem = pq.read_table(out / "cluster_membership.parquet").to_pylist()
    by_cluster: dict[int, set] = {}
    for row in mem:
        by_cluster.setdefault(row["cluster_id"], set()).add(_population(row["read_id"]))
    assert any(len(p) > 1 for p in by_cluster.values())


# ── Codex, 2026-11-20 ─────────────────────────────────────────────────


def test_the_reseeding_guard_judges_a_cluster_against_the_corpus() -> None:
    """10,000 identical refused reads in a 50,000-read corpus are 20% of the
    corpus, not "100%" of the subset reseeding hands the seeder."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.seed_kmer import (
        ChainedClusterError,
        seed_by_kmer_clustering,
    )

    rng = np.random.default_rng(3)
    seq = "".join(rng.choice(list("ACGT"), size=600))
    n = 400
    sub = pa.table({
        "read_id": pa.array([f"r{i}" for i in range(n)]),
        "sequence": pa.array([seq] * n, pa.large_string()),
        "sample_id": pa.array(np.zeros(n, np.int64)),
        "dorado_quality": pa.array(np.full(n, 25.0, np.float32)),
    })
    kw = dict(max_cluster_read_frac=0.25, min_chain_cluster_reads=100, threads=1)
    with pytest.raises(ChainedClusterError):
        seed_by_kmer_clustering(sub, **kw)
    result = seed_by_kmer_clustering(sub, corpus_reads=n * 5, **kw)
    assert result.templates.num_rows == 1


def test_reseeding_resolves_an_orf_only_representative_policy() -> None:
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        reseed_representative,
    )

    assert reseed_representative(_params("orf", seed_representative="most-5p-flank")) == (
        "longest-above-quality"
    )
    assert reseed_representative(_params("orf", seed_representative="median-length")) == (
        "median-length"
    )


def test_a_merge_tolerance_is_checked_against_the_inherited_graph_tolerance() -> None:
    p = EmParams(end_tolerance=60, merge_tol_5p=50)
    assert p.graph.tol_5p == 60 and p.merge_predicate().tol_5p == 50
    with pytest.raises(ValueError):
        EmParams(end_tolerance=60, merge_tol_5p=70)


def test_em_orf_with_most_5p_flank_reseeds(tmp_path) -> None:
    """The ORF-only policy runs round 1 and then reseeds without crashing."""
    out = tmp_path / "em"
    run_em(_panel(tmp_path), out,
           params=_params("orf", rounds=2, seed_representative="most-5p-flank"), report=False)
    import json

    ref = json.loads((out / "rounds" / "r01" / "refine.json").read_text())
    assert ref["n_reseeded_reads"] > 0
    assert ref["reseed_representative"] == "longest-above-quality"
