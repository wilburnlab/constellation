"""Read-level kmer clustering as the EM's round-1 seeder.

The behaviours pinned here are the ones the 9.39M-read sweep settled and
that no small fixture can re-derive: the asymmetric overhang gate (5'
unbounded, 3' capped), the election being a *separate* decision from the
grouping order, and the chaining guard counting reads rather than uniques.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em.elect import (
    assert_read_map_alignment,
    elect_representatives,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.seed_kmer import (
    READ_CLUSTER_MAP_SCHEMA,
    ChainedClusterError,
    seed_by_kmer_clustering,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
    TEMPLATE_TABLE,
)
from constellation.sequencing.transcriptome.cluster.denovo.verify import (
    UNBOUNDED_OVERHANG,
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
            continue
        elif r < rate:
            out.append(base)
            out.append(rng.choice(list("ACGT")))
        else:
            out.append(base)
    return "".join(out)


def _reads(rows) -> pa.Table:
    """rows = [(read_id, sequence, dorado_quality)]"""
    return pa.table(
        {
            "read_id": pa.array([r[0] for r in rows], pa.string()),
            "sequence": pa.array([r[1] for r in rows], pa.large_string()),
            "sample_id": pa.array([0] * len(rows), pa.int64()),
            "dorado_quality": pa.array([r[2] for r in rows], pa.float32()),
        }
    )


@pytest.fixture
def three_transcripts():
    """Three well-separated transcripts, 20 reads each at ~1% error."""
    rng = np.random.default_rng(7)
    truths = [_orf(rng, 150), _orf(rng, 170), _orf(rng, 190)]
    rows = []
    for t_idx, truth in enumerate(truths):
        for i in range(20):
            rows.append(
                (
                    f"t{t_idx}_r{i}",
                    _mutate(rng, truth, 0.01),
                    float(rng.uniform(15.0, 35.0)),
                )
            )
    return _reads(rows), truths


# ── the partition ─────────────────────────────────────────────────────


def test_reads_of_one_transcript_land_in_one_cluster(three_transcripts):
    reads, truths = three_transcripts
    res = seed_by_kmer_clustering(reads, identity=0.90)

    assert res.templates.schema.equals(TEMPLATE_TABLE)
    assert res.read_cluster.schema.equals(READ_CLUSTER_MAP_SCHEMA)
    assert res.templates.num_rows == len(truths)

    cl = res.read_cluster.column("cluster_id").to_pylist()
    rid = res.read_cluster.column("read_id").to_pylist()
    by_truth: dict[str, set[int]] = {}
    for r, c in zip(rid, cl):
        by_truth.setdefault(r.split("_")[0], set()).add(c)
    # One cluster per transcript, and no cluster shared between transcripts.
    assert all(len(v) == 1 for v in by_truth.values())
    assert len({next(iter(v)) for v in by_truth.values()}) == len(truths)


def test_every_read_gets_a_row_and_the_counts_reconcile(three_transcripts):
    reads, _ = three_transcripts
    res = seed_by_kmer_clustering(reads, identity=0.90)

    assert res.read_cluster.num_rows == reads.num_rows
    counts = np.bincount(
        np.array(res.read_cluster.column("cluster_id").to_pylist()),
        minlength=res.templates.num_rows,
    )
    # orf_replication is round 1's ranking key and must BE the cluster's read
    # count — `scheduler.rank_round1` ranks on it directly.
    assert counts.tolist() == res.templates.column("orf_replication").to_pylist()
    assert res.stats["n_reads_in_templates"] == reads.num_rows


# ── the gate: 5' unbounded, 3' capped ─────────────────────────────────


def _truncation_case(five_prime: bool):
    """One transcript plus a read missing 200 nt from one end."""
    rng = np.random.default_rng(3)
    truth = _orf(rng, 200)
    rows = [(f"full_r{i}", _mutate(rng, truth, 0.01), 30.0) for i in range(10)]
    frag = truth[200:] if five_prime else truth[:-200]
    rows.append(("frag_r0", _mutate(rng, frag, 0.01), 30.0))
    return _reads(rows)


def test_a_5p_truncated_read_joins_when_the_5p_end_is_unbounded():
    """30% of Complete windows are 5'-partial, so this recruitment is the point."""
    reads = _truncation_case(five_prime=True)
    joined = seed_by_kmer_clustering(
        reads, identity=0.90, max_5p_overhang=UNBOUNDED_OVERHANG, max_3p_overhang=100
    )
    split = seed_by_kmer_clustering(
        reads, identity=0.90, max_5p_overhang=30, max_3p_overhang=30
    )
    assert joined.templates.num_rows == 1
    assert split.templates.num_rows == 2


def test_a_3p_extended_read_stays_apart_because_the_3p_end_is_capped():
    """The 3' cap is what makes 5'-unbounded recruitment safe at depth."""
    reads = _truncation_case(five_prime=False)
    capped = seed_by_kmer_clustering(
        reads, identity=0.90, max_5p_overhang=UNBOUNDED_OVERHANG, max_3p_overhang=100
    )
    uncapped = seed_by_kmer_clustering(
        reads,
        identity=0.90,
        max_5p_overhang=UNBOUNDED_OVERHANG,
        max_3p_overhang=UNBOUNDED_OVERHANG,
    )
    assert capped.templates.num_rows == 2
    assert uncapped.templates.num_rows == 1


# ── the election, which is NOT the grouping order ─────────────────────


def test_the_template_is_the_longest_read_clearing_the_quality_floor():
    """The elected read is the policy's, not the component centroid's.

    `connected_components` elects its centroid by (abundance, length, id) and
    knows nothing about quality. Here the longest read is Q10 — the centroid
    rule would take it, `longest-above-quality` must not.
    """
    rng = np.random.default_rng(5)
    truth = _orf(rng, 150)
    rows = [(f"r{i}", _mutate(rng, truth, 0.005), 30.0) for i in range(8)]
    # The longest read in the cluster, but below the Q22 floor.
    rows.append(("long_but_bad", truth + "ACGTACGTACGT", 10.0))
    reads = _reads(rows)

    res = seed_by_kmer_clustering(reads, identity=0.90)
    assert res.templates.num_rows == 1
    seed_row = res.templates.column("seed_read_row").to_pylist()[0]
    assert reads.column("read_id").to_pylist()[seed_row] != "long_but_bad"
    assert res.templates.column("seed_read_quality").to_pylist()[0] >= 22.0

    # ...and the policy is honoured, so asking for the longest gets it back.
    longest = seed_by_kmer_clustering(
        reads, identity=0.90, representative="longest-template"
    )
    seed_row = longest.templates.column("seed_read_row").to_pylist()[0]
    assert reads.column("read_id").to_pylist()[seed_row] == "long_but_bad"


def test_an_orf_ranking_policy_is_refused_rather_than_substituted():
    """`most-5p-flank` needs an ORF this seeder has not predicted yet."""
    reads = _reads([("r0", _orf(np.random.default_rng(1), 100), 30.0)])
    with pytest.raises(ValueError, match="most-5p-flank"):
        seed_by_kmer_clustering(reads, representative="most-5p-flank")


# ── the chaining guard ────────────────────────────────────────────────


def test_a_mega_component_fails_the_stage_rather_than_seeding_it():
    """Counts READS: `_MEGA_CLUSTER_UNIQUES` counts uniques and never fired."""
    rng = np.random.default_rng(9)
    truth = _orf(rng, 150)
    rows = [(f"r{i}", _mutate(rng, truth, 0.005), 30.0) for i in range(20)]
    reads = _reads(rows)

    with pytest.raises(ChainedClusterError, match="3' cap"):
        seed_by_kmer_clustering(
            reads,
            identity=0.90,
            max_cluster_read_frac=0.25,
            min_chain_cluster_reads=0,
        )
    # 0 disables it, for a deliberate sweep.
    res = seed_by_kmer_clustering(
        reads, identity=0.90, max_cluster_read_frac=0.0, min_chain_cluster_reads=0
    )
    assert res.templates.num_rows == 1
    assert res.stats["largest_cluster_read_frac"] == 1.0


def test_the_chaining_guard_is_silent_on_a_corpus_too_small_to_judge():
    """A fraction needs a corpus to be a fraction OF.

    Three transcripts at 20x each put 33% of the reads in every cluster,
    correctly. Firing there would make the guard useless on any real small
    run, and its entire evidence base is at 9.4M reads.
    """
    rng = np.random.default_rng(21)
    truths = [_orf(rng, 150), _orf(rng, 170), _orf(rng, 190)]
    rows = [
        (f"t{i}_r{j}", _mutate(rng, truth, 0.01), 30.0)
        for i, truth in enumerate(truths)
        for j in range(20)
    ]
    res = seed_by_kmer_clustering(_reads(rows), identity=0.90)
    assert res.templates.num_rows == 3
    assert res.stats["largest_cluster_read_frac"] > 0.25


# ── the size filter ───────────────────────────────────────────────────


def test_min_seed_reads_drops_low_support_clusters_and_marks_their_reads():
    rng = np.random.default_rng(13)
    truths = [_orf(rng, 150), _orf(rng, 170)]
    rows = [(f"a_r{i}", _mutate(rng, truths[0], 0.01), 30.0) for i in range(10)]
    rows.append(("b_r0", truths[1], 30.0))  # a lone read, its own cluster
    reads = _reads(rows)

    kept = seed_by_kmer_clustering(reads, identity=0.90, min_seed_reads=1)
    dropped = seed_by_kmer_clustering(reads, identity=0.90, min_seed_reads=2)
    assert kept.templates.num_rows == 2
    assert dropped.templates.num_rows == 1
    assert dropped.stats["n_dropped_below_min_seed_reads"] == 1

    # The dropped read keeps a row, flagged, so the cost is countable.
    cl = dict(
        zip(
            dropped.read_cluster.column("read_id").to_pylist(),
            dropped.read_cluster.column("cluster_id").to_pylist(),
        )
    )
    assert cl["b_r0"] == -1
    assert all(v >= 0 for k, v in cl.items() if k != "b_r0")


# ── the ORF is annotation, not part of this algorithm ─────────────────


def test_the_seeder_predicts_no_orf_at_all(three_transcripts):
    """Clustering is by read similarity; the ORF has no say in it.

    Under ORF seeding the ORF is the KEY — a template exists because a
    distinct ORF did. Here nothing in the loop consults one: the partition is
    edit distance, node splitting is column covariance, and round 1 ranks on
    cluster support and seed quality. So no ORF is predicted at seed time.
    """
    reads, _ = three_transcripts
    res = seed_by_kmer_clustering(reads, identity=0.90)

    assert res.templates.num_rows == 3
    assert set(res.templates.column("orf_start").to_pylist()) == {0}
    assert set(res.templates.column("orf_end").to_pylist()) == {0}
    assert set(res.templates.column("orf_aa_length").to_pylist()) == {0}
    # ...even though the fixture's reads all carry a long clean ORF.
    from constellation.sequencing.transcriptome.cluster.denovo.orf import (
        best_sense_orf,
    )

    assert best_sense_orf(res.templates.column("sequence").to_pylist()[0]) is not None


def test_min_aa_length_is_not_a_parameter_of_this_seeder():
    """It cannot be, or a protein-annotation floor would steer the partition."""
    import inspect

    sig = inspect.signature(seed_by_kmer_clustering)
    assert "min_aa_length" not in sig.parameters


def test_an_empty_seed_orf_leaves_the_support_gate_judging_everything():
    """(0, 0) is the honest statement, and gated_orf reads it that way.

    `gated_orf` treats the seed ORF interval as certified BY CONSTRUCTION and
    declines to judge it. Predicting an ORF at seed time would therefore
    relax the M-step's support gate on the strength of a claim this seeder
    never made — which is exactly ORF detection leaking into the algorithm.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep import (
        gated_orf,
    )

    rng = np.random.default_rng(23)
    consensus = _orf(rng, 60)
    # Everything certified: the ORF survives either way.
    certified = np.ones(len(consensus), dtype=bool)
    assert (
        gated_orf(consensus, certified, seed_orf_start=0, seed_orf_end=0) is not None
    )

    # Now strip certification from the ORF's 3' half. With no seed interval
    # the gate judges it and truncates; with the interval asserted it does not.
    certified[len(consensus) // 2 :] = False
    ungated = gated_orf(consensus, certified, seed_orf_start=0, seed_orf_end=0)
    asserted = gated_orf(
        consensus, certified, seed_orf_start=0, seed_orf_end=len(consensus)
    )
    assert ungated is not None and ungated[4] is True, "must be flagged truncated"
    assert asserted is not None and asserted[4] is False
    assert ungated[2] < asserted[2], "the empty seed interval must gate harder"


# ── the shared election helper ────────────────────────────────────────


def test_read_map_is_row_aligned_with_reads():
    """`best_quality_per_uniq` indexes positionally on the strength of this."""
    from constellation.sequencing.transcriptome.cluster.denovo.dereplicate import (
        dereplicate,
    )

    reads = _reads(
        [("r0", "ACGTACGT", 30.0), ("r1", "TTTTTTTT", 20.0), ("r2", "ACGTACGT", 25.0)]
    )
    _uniq, read_map = dereplicate(reads)
    assert_read_map_alignment(reads, read_map)


def test_elect_representatives_reports_empty_groups_rather_than_hiding_them():
    elected, n_reads = elect_representatives(
        np.array([0, 0, 2], dtype=np.int64),
        3,
        template_length=np.array([10, 20, 30], dtype=np.int64),
        abundance=np.array([1, 1, 1], dtype=np.int64),
        quality=np.array([30.0, 30.0, 30.0]),
        policy="longest-template",
    )
    assert elected[0] == 1  # the longer of group 0's two members
    assert elected[1] == -1  # nothing labelled group 1
    assert n_reads.tolist() == [2, 0, 1]
