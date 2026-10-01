"""Between rounds: what carries forward, what does not, and what churned.

The headline negative is `test_a_lone_read_on_its_own_template_carries_forward`:
there is no support prune, so a template holding one read that nothing else
explains survives every round. That is the decision the handoff's §4 argued
against, reversed on the finding that Tuba1a's failure was PWM contamination
(which p_floor fixes) rather than single-read promotion.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
    REFINED_NODE_TABLE,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.refine import (
    LINEAGE_TABLE,
    measure_churn,
    next_templates,
    template_id_for,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
    TEMPLATE_TABLE,
    TemplateStore,
)


def _store(n, *, ids=None):
    ids = np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids)
    return TemplateStore.from_table(
        pa.table(
            {
                "template_id": pa.array(ids),
                "sequence": pa.array(["ACGT" * 25] * n, pa.large_string()),
                "orf_start": pa.array(np.zeros(n, np.int32)),
                "orf_end": pa.array(np.full(n, 99, np.int32)),
                "orf_aa_length": pa.array(np.full(n, 32, np.int32)),
                "node_weight": pa.array(np.ones(n, np.float64)),
                "orf_replication": pa.array(np.ones(n, np.int64)),
                "seed_read_quality": pa.array(np.full(n, 25.0, np.float32)),
                "seed_read_row": pa.array(np.arange(n, dtype=np.int32)),
                "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
            },
            schema=TEMPLATE_TABLE,
        )
    )


def _nodes(rows):
    """rows = [(parent_id, parent_row, hap, consensus, n_reads, weight)]"""
    n = len(rows)
    return pa.table(
        {
            "round": pa.array([1] * n, pa.int32()),
            "parent_template_id": pa.array([r[0] for r in rows], pa.int64()),
            "parent_template_row": pa.array([r[1] for r in rows], pa.int32()),
            "haplotype_id": pa.array([r[2] for r in rows], pa.int32()),
            "consensus": pa.array([r[3] for r in rows], pa.large_string()),
            "n_reads": pa.array([r[4] for r in rows], pa.int64()),
            "node_weight": pa.array([r[5] for r in rows], pa.float64()),
            "protein": pa.array([None] * n, pa.large_string()),
            "orf_start": pa.array([0] * n, pa.int32()),
            "orf_end": pa.array([30] * n, pa.int32()),
            "allele_string": pa.array([None] * n, pa.string()),
            "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
            "n_inserted_columns": pa.array([0] * n, pa.int32()),
            "n_extended_5p": pa.array([0] * n, pa.int32()),
            "n_extended_3p": pa.array([0] * n, pa.int32()),
            "n_trimmed_5p": pa.array([0] * n, pa.int32()),
            "n_trimmed_3p": pa.array([0] * n, pa.int32()),
            "n_members_used": pa.array([r[4] for r in rows], pa.int32()),
            "subsample_fraction": pa.array([1.0] * n, pa.float32()),
        },
        schema=REFINED_NODE_TABLE,
    )


def _assign(pairs):
    """pairs = [(read_row, template_id)]"""
    return pa.table(
        {
            "read_row": pa.array([p[0] for p in pairs], pa.int32()),
            "template_id": pa.array([p[1] for p in pairs], pa.int64()),
        }
    )


# ── what carries forward ──────────────────────────────────────────────


def test_a_lone_read_on_its_own_template_carries_forward():
    """No support prune. A real read that matches nothing else is kept."""
    store = _store(1)
    result = next_templates(
        _nodes([(0, 0, 0, "ACGT" * 25, 1, 1.0)]), store, round_index=1
    )
    assert result.templates.num_rows == 1
    assert result.n_unrecruited == 0
    assert result.lineage.column("rule").to_pylist() == ["carry"]


def test_a_template_that_recruited_nothing_does_not_carry_forward():
    """The only way a template disappears — a consequence, not a rule."""
    store = _store(3)
    result = next_templates(
        _nodes([(1, 1, 0, "ACGT" * 25, 5, 5.0)]), store, round_index=1
    )
    assert result.templates.num_rows == 1
    assert result.n_unrecruited == 2
    rules = result.lineage.column("rule").to_pylist()
    assert sorted(rules) == ["carry", "unrecruited", "unrecruited"]
    # The unrecruited parents are named, so the accounting is explicit rather
    # than inferred from a count difference.
    unrec = result.lineage.filter(
        pa.compute.equal(result.lineage.column("rule"), "unrecruited")
    )
    assert sorted(unrec.column("parent_template_id").to_pylist()) == [0, 2]


def test_a_split_parent_is_recorded_as_such():
    store = _store(1)
    result = next_templates(
        _nodes(
            [
                (0, 0, 0, "ACGT" * 25, 40, 40.0),
                (0, 0, 1, "ACGA" * 25, 12, 12.0),
            ]
        ),
        store,
        round_index=1,
    )
    assert result.templates.num_rows == 2
    assert result.lineage.column("rule").to_pylist() == ["split", "split"]


def test_child_ids_are_deterministic_given_the_round():
    """Resume must rebuild identical ids without a parent-side counter."""
    store = _store(1)
    nodes = _nodes([(0, 0, 0, "ACGT" * 25, 3, 3.0), (0, 0, 1, "ACGC" * 25, 2, 2.0)])
    a = next_templates(nodes, store, round_index=1).templates
    b = next_templates(nodes, store, round_index=1).templates
    assert a.column("template_id").to_pylist() == b.column("template_id").to_pylist()
    assert a.column("template_id").to_pylist() == [
        int(template_id_for(2, 0)),
        int(template_id_for(2, 1)),
    ]


def test_a_child_frame_has_no_seed_quality():
    """From round 2 the frame is a consensus with no read of its own."""
    store = _store(1)
    result = next_templates(
        _nodes([(0, 0, 0, "ACGT" * 25, 9, 9.0)]), store, round_index=1
    )
    assert result.templates.column("seed_read_quality").null_count == 1
    assert result.templates.column("seed_read_row").to_pylist() == [-1]


def test_an_empty_consensus_cannot_become_a_target():
    store = _store(2)
    result = next_templates(
        _nodes([(0, 0, 0, "", 1, 1.0), (1, 1, 0, "ACGT" * 25, 4, 4.0)]),
        store,
        round_index=1,
    )
    assert result.templates.num_rows == 1


def test_no_nodes_at_all_is_not_a_crash():
    store = _store(4)
    result = next_templates(REFINED_NODE_TABLE.empty_table(), store, round_index=1)
    assert result.templates.num_rows == 0
    assert result.n_unrecruited == 4


# ── churn ─────────────────────────────────────────────────────────────


def test_churn_counts_a_genuine_switch():
    prev = _assign([(0, 10), (1, 11)])
    cur = _assign([(0, 11), (1, 11)])
    stats = measure_churn(prev, cur, LINEAGE_TABLE.empty_table(), n_reads=2)
    assert stats["n_compared"] == 2
    assert stats["frac_changed"] == pytest.approx(0.5)
    assert stats["frac_changed_lineage"] == pytest.approx(0.5)


def test_a_read_whose_template_split_is_not_churn():
    """The load-bearing case: otherwise the loop never looks converged."""
    store = _store(1, ids=[10])
    result = next_templates(
        _nodes([(10, 0, 0, "ACGT" * 25, 5, 5.0), (10, 0, 1, "ACGA" * 25, 3, 3.0)]),
        store,
        round_index=1,
    )
    children = result.templates.column("template_id").to_pylist()

    prev = _assign([(0, 10), (1, 10)])
    cur = _assign([(0, children[0]), (1, children[1])])
    stats = measure_churn(prev, cur, result.lineage, n_reads=2)

    assert stats["n_moved_raw"] == 2
    assert stats["n_moved_inherited"] == 2
    assert stats["frac_changed_lineage"] == pytest.approx(0.0)


def test_gained_and_lost_reads_are_counted_separately():
    prev = _assign([(0, 10)])
    cur = _assign([(1, 11)])
    stats = measure_churn(prev, cur, LINEAGE_TABLE.empty_table(), n_reads=2)
    assert stats["n_compared"] == 0
    assert stats["n_gained"] == 1
    assert stats["n_lost"] == 1


# ── final outputs: a split parent keeps both children's reads ─────────


def _node_membership(rows):
    """rows = [(parent_id, hap, read_row, weight)]"""
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        NODE_MEMBERSHIP_TABLE,
    )

    n = len(rows)
    return pa.table(
        {
            "round": pa.array([1] * n, pa.int32()),
            "parent_template_id": pa.array([r[0] for r in rows], pa.int64()),
            "haplotype_id": pa.array([r[1] for r in rows], pa.int32()),
            "read_row": pa.array([r[2] for r in rows], pa.int32()),
            "weight": pa.array([r[3] for r in rows], pa.float32()),
        },
        schema=NODE_MEMBERSHIP_TABLE,
    )


def _assignments_for(read_rows, template_id, *, samples=None):
    from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
        EM_ASSIGNMENT_TABLE,
    )

    n = len(read_rows)
    samples = samples if samples is not None else [0] * n
    return pa.table(
        {
            "read_id": pa.array([f"r{r}" for r in read_rows], pa.string()),
            "read_row": pa.array(read_rows, pa.int32()),
            "template_id": pa.array([template_id] * n, pa.int64()),
            "template_row": pa.array([0] * n, pa.int32()),
            "round": pa.array([1] * n, pa.int32()),
            "weight": pa.array([1.0] * n, pa.float32()),
            "as_score": pa.array(list(range(100, 100 + n)), pa.int32()),
            "as_delta": pa.array([0] * n, pa.int32()),
            "logl": pa.nulls(n, pa.float32()),
            "logl_delta": pa.nulls(n, pa.float32()),
            "n_hits": pa.array([1] * n, pa.int32()),
            "n_admitted": pa.array([1] * n, pa.int32()),
            "candidate_cap_hit": pa.array([False] * n, pa.bool_()),
            "offset_5p": pa.array([0] * n, pa.int32()),
            "q_start": pa.array([0] * n, pa.int32()),
            "q_end": pa.array([100] * n, pa.int32()),
            "t_start": pa.array([0] * n, pa.int32()),
            "t_end": pa.array([100] * n, pa.int32()),
            "cigar": pa.array(["100="] * n, pa.large_string()),
            "sample_id": pa.array(samples, pa.int64()),
            "chain_score": pa.nulls(n, pa.int32()),
            "shortlist_truncated": pa.nulls(n, pa.bool_()),
        },
        schema=EM_ASSIGNMENT_TABLE,
    )


def test_a_split_parent_does_not_collapse_into_one_cluster():
    """The read -> template map is read -> PARENT; using it loses every split.

    One parent emitting two nodes with 3 and 2 reads must export as 3/2. Keyed
    on the parent id alone it exports as 0/5, and the first child has no reads
    and no representative.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.outputs import (
        build_cluster_tables,
    )

    nodes = _nodes(
        [(10, 0, 0, "ACGT" * 25, 3, 3.0), (10, 0, 1, "ACGA" * 25, 2, 2.0)]
    )
    membership = _node_membership(
        [(10, 0, 0, 1.0), (10, 0, 1, 1.0), (10, 0, 2, 1.0),
         (10, 1, 3, 1.0), (10, 1, 4, 1.0)]
    )
    clusters, member_tbl, _ = build_cluster_tables(
        nodes,
        _assignments_for([0, 1, 2, 3, 4], 10),
        membership,
        identity_threshold=0.97,
    )
    assert clusters.column("n_reads").to_pylist() == [3, 2]
    assert all(clusters.column("representative_read_id").to_pylist())
    per = {}
    for cid in member_tbl.column("cluster_id").to_pylist():
        per[cid] = per.get(cid, 0) + 1
    assert per == {0: 3, 1: 2}


def test_sample_ids_come_from_the_assignments():
    """They used to come from an argument the round loop never supplied."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.outputs import (
        build_cluster_tables,
    )

    nodes = _nodes([(10, 0, 0, "ACGT" * 25, 3, 3.0)])
    membership = _node_membership([(10, 0, 0, 1.0), (10, 0, 1, 1.0), (10, 0, 2, 1.0)])
    _, _, sample_id = build_cluster_tables(
        nodes,
        _assignments_for([0, 1, 2], 10, samples=[10, 20, 10]),
        membership,
        identity_threshold=0.97,
    )
    assert sorted(sample_id.tolist()) == [10, 10, 20]
    assert -1 not in sample_id.tolist(), "pooling into sample -1 corrupts TPM"


def test_unique_sequence_count_is_not_the_read_count():
    """Identical reads are one sequence; counting reads makes the ratio 1."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.outputs import (
        build_cluster_tables,
    )

    class _Reads:
        n_reads = 4
        sequence = pa.chunked_array(
            [pa.array(["AAAA", "AAAA", "AAAA", "CCCC"], pa.large_string())]
        )

    nodes = _nodes([(10, 0, 0, "ACGT" * 25, 4, 4.0)])
    membership = _node_membership(
        [(10, 0, 0, 1.0), (10, 0, 1, 1.0), (10, 0, 2, 1.0), (10, 0, 3, 1.0)]
    )
    clusters, _, _ = build_cluster_tables(
        nodes,
        _assignments_for([0, 1, 2, 3], 10),
        membership,
        reads=_Reads(),
        identity_threshold=0.97,
    )
    assert clusters.column("n_reads").to_pylist() == [4]
    assert clusters.column("n_unique_sequences").to_pylist() == [2]


def test_lineage_membership_does_not_collide_at_scale():
    """(child << 21) ^ parent overlaps the two ids past 2**21 template rows.

    Round 1 carries ~4.17M templates, so this is reachable in production: a
    collision reports an unrelated switch as inherited, which reads as zero
    lineage-aware churn and stops the loop early.
    """
    child_a, parent_a = (2 << 40) | 0, (1 << 40) | 0
    child_b, parent_b = (2 << 40) | 1, (1 << 40) | 2_097_152
    # These two edges collide under the old packing.
    assert ((child_a << 21) ^ parent_a) == ((child_b << 21) ^ parent_b)

    lineage = pa.table(
        {
            "round": pa.array([2], pa.int32()),
            "child_template_id": pa.array([child_a], pa.int64()),
            "parent_template_id": pa.array([parent_a], pa.int64()),
            "rule": pa.array(["carry"], pa.string()),
            "n_reads": pa.array([1.0], pa.float64()),
        },
        schema=LINEAGE_TABLE,
    )
    # Read 0 genuinely switched from parent_b to child_b — an unrelated pair.
    stats = measure_churn(
        _assign([(0, parent_b)]), _assign([(0, child_b)]), lineage, n_reads=1
    )
    assert stats["n_moved_inherited"] == 0, "collided onto an unrelated edge"
    assert stats["frac_changed_lineage"] == pytest.approx(1.0)


def test_convergence_counts_reads_that_gain_or_lose_an_assignment():
    """Measured over both-assigned reads alone, losing half scores zero churn."""
    prev = _assign([(0, 10), (1, 10), (2, 10), (3, 10)])
    cur = _assign([(0, 10), (1, 10)])  # two reads lost their assignment
    stats = measure_churn(prev, cur, LINEAGE_TABLE.empty_table(), n_reads=4)

    assert stats["frac_changed_lineage"] == pytest.approx(0.0), "the old measure"
    assert stats["n_lost"] == 2
    # The stopping rule reads this one, so half the corpus vanishing cannot
    # look converged.
    assert stats["frac_unsettled"] == pytest.approx(0.5)


def test_a_capped_template_still_reports_all_of_its_reads():
    """The cap bounds what the CONSENSUS is built from, not what the cluster
    CONTAINS.

    Membership used to carry only the sampled reads, so a template with 20
    reads capped at 5 exported 5 members and n_reads 5. Scaling node_weight
    did not repair it: exported counts and quant are both counted from
    membership rows.
    """
    import pyarrow as pa_
    import pyarrow.parquet as pq_
    import tempfile
    from pathlib import Path as _Path

    from constellation.sequencing.transcriptome.cluster.denovo._io import (
        _READS_SCHEMA,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
        EM_ASSIGNMENT_TABLE,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
        mstep_worker,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
        write_templates,
    )

    d = _Path(tempfile.mkdtemp())
    seq = "ATG" + "GCT" * 40 + "TAA"
    n = 20
    with pa_.OSFile(str(d / "reads.arrow"), "wb") as sink:
        with pa_.ipc.new_file(sink, _READS_SCHEMA) as w:
            w.write_table(
                pa_.table(
                    {
                        "read_id": pa_.array([f"r{i}" for i in range(n)], pa_.string()),
                        "sequence": pa_.array([seq] * n, pa_.large_string()),
                        "sample_id": pa_.array(np.zeros(n, np.int64)),
                        "chain_score": pa_.nulls(n, pa_.int32()),
                        "shortlist_truncated": pa_.nulls(n, pa_.bool_()),
                        "dorado_quality": pa_.array(np.full(n, 30.0, np.float32)),
                    },
                    schema=_READS_SCHEMA,
                )
            )
    write_templates(
        pa_.table(
            {
                "template_id": pa_.array([7], pa_.int64()),
                "sequence": pa_.array([seq], pa_.large_string()),
                "orf_start": pa_.array([0], pa_.int32()),
                "orf_end": pa_.array([len(seq)], pa_.int32()),
                "orf_aa_length": pa_.array([40], pa_.int32()),
                "node_weight": pa_.array([float(n)]),
                "orf_replication": pa_.array([n], pa_.int64()),
                "seed_read_quality": pa_.nulls(1, pa_.float32()),
                "seed_read_row": pa_.array([0], pa_.int32()),
                "declared_variants": pa_.array([[]], pa_.list_(pa_.int64())),
            },
            schema=TEMPLATE_TABLE,
        ),
        d,
    )
    batch = pa_.table(
        {
            "read_id": pa_.array([f"r{i}" for i in range(n)], pa_.string()),
            "read_row": pa_.array(np.arange(n, dtype=np.int32)),
            "template_id": pa_.array(np.full(n, 7, np.int64)),
            "template_row": pa_.array(np.zeros(n, np.int32)),
            "round": pa_.array(np.ones(n, np.int32)),
            "weight": pa_.array(np.ones(n, np.float32)),
            "as_score": pa_.array(np.zeros(n, np.int32)),
            "as_delta": pa_.array(np.zeros(n, np.int32)),
            "logl": pa_.nulls(n, pa_.float32()),
            "logl_delta": pa_.nulls(n, pa_.float32()),
            "n_hits": pa_.array(np.ones(n, np.int32)),
            "n_admitted": pa_.array(np.ones(n, np.int32)),
            "candidate_cap_hit": pa_.array(np.zeros(n, bool)),
            "offset_5p": pa_.array(np.zeros(n, np.int32)),
            "q_start": pa_.array(np.zeros(n, np.int32)),
            "q_end": pa_.array(np.full(n, len(seq), np.int32)),
            "t_start": pa_.array(np.zeros(n, np.int32)),
            "t_end": pa_.array(np.full(n, len(seq), np.int32)),
            "cigar": pa_.array([f"{len(seq)}="] * n, pa_.large_string()),
            "sample_id": pa_.array(np.zeros(n, np.int64)),
            "chain_score": pa_.nulls(n, pa_.int32()),
            "shortlist_truncated": pa_.nulls(n, pa_.bool_()),
        },
        schema=EM_ASSIGNMENT_TABLE,
    )
    for cap in (n, 5):
        out = mstep_worker(
            batch,
            corpus_path=str(d / "reads.arrow"),
            templates_path=str(d / "templates.arrow"),
            round_index=1,
            params=MStepParams(max_members_per_template=cap),
        )
        mem, nodes = out["node_membership"], out["nodes"]
        assert mem.num_rows == n, f"cap={cap} exported {mem.num_rows} of {n} reads"
        assert sorted(mem.column("read_row").to_pylist()) == list(range(n))
        assert sum(nodes.column("n_reads").to_pylist()) == n
        assert sum(nodes.column("node_weight").to_pylist()) == float(n)


def _template_table(seqs):
    n = len(seqs)
    return pa.table(
        {
            "template_id": pa.array(np.arange(n, dtype=np.int64)),
            "sequence": pa.array(seqs, pa.large_string()),
            "orf_start": pa.array(np.zeros(n, np.int32)),
            "orf_end": pa.array(np.array([len(s) for s in seqs], np.int32)),
            "orf_aa_length": pa.array(np.full(n, 1, np.int32)),
            "node_weight": pa.array(np.ones(n)),
            "orf_replication": pa.array(np.ones(n, np.int64)),
            "seed_read_quality": pa.nulls(n, pa.float32()),
            "seed_read_row": pa.array(np.arange(n, dtype=np.int32)),
            "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
        },
        schema=TEMPLATE_TABLE,
    )


def test_a_sliced_template_table_reads_back_its_own_rows():
    """`Array.offset` is not zero after a slice, and `chunk(0)` preserves it.

    Reading the offsets buffer from index 0 then returns a DIFFERENT row —
    silently, with a plausible sequence. Every consumer downstream (the PWM,
    the likelihood, the redundancy detector) would be reading one template's
    bases under another's id. Introduced by the chunk(0) fast path that
    replaced combine_chunks to stop each worker copying the corpus.
    """
    seqs = ["AAAA", "CCCC", "GGGG"]
    table = _template_table(seqs)
    assert table.column("sequence").chunk(0).offset == 0

    for lo in range(len(seqs)):
        for length in range(1, len(seqs) - lo + 1):
            sliced = table.slice(lo, length)
            store = TemplateStore.from_table(sliced)
            got = [store.sequence(i) for i in range(store.n_templates)]
            assert got == seqs[lo : lo + length], f"slice({lo}, {length})"

    # And the unsliced case is unchanged.
    store = TemplateStore.from_table(table)
    assert [store.sequence(i) for i in range(3)] == seqs


def test_an_empty_template_table_opens_without_indexing_a_buffer():
    store = TemplateStore.from_table(_template_table([]))
    assert store.n_templates == 0
    assert store.seq_buffer.size == 0


# ── merge ─────────────────────────────────────────────────────────────
#
# Only the collapse lives in refine.py. WHICH pairs are mergeable is the
# graph's question (tests/test_em_graph*.py); these tests hand the collapse
# pairs directly and check what it does with them.

from constellation.sequencing.transcriptome.cluster.denovo.em.refine import (  # noqa: E402
    MERGED_TABLE,
    apply_merge,
    claim_order,
    merge_nodes,
    select_merges,
)

_TOL = {"tol_5p": 30, "tol_3p": 30}


def _edges(rows):
    """rows = [(a, b)] or [(a, b, off_5p, off_3p)] or [(.., identical)], by ROW."""
    full = [tuple(r) + (0, 0, False)[len(r) - 2 :] for r in rows]
    return {
        "src_row": np.array([r[0] for r in full], dtype=np.int64),
        "dst_row": np.array([r[1] for r in full], dtype=np.int64),
        "off_5p": np.array([r[2] for r in full], dtype=np.int64),
        "off_3p": np.array([r[3] for r in full], dtype=np.int64),
        "identical": np.array([r[4] for r in full], dtype=bool),
    }


def _by_id(rows, ids):
    """The same pairs keyed by template id, as `graph.mergeable_pairs` gives them."""
    e = _edges(rows)
    ids = np.asarray(ids, dtype=np.int64)
    e["src_template_id"] = ids[e["src_row"]]
    e["dst_template_id"] = ids[e["dst_row"]]
    return e


def _survivors(rows, n_reads, seq_len, **kw):
    e = _edges(rows)
    n = len(n_reads)
    return select_merges(
        n,
        e["src_row"],
        e["dst_row"],
        e["off_5p"],
        e["off_3p"],
        np.asarray(n_reads),
        np.asarray(seq_len),
        identical=e["identical"],
        **{**_TOL, **kw},
    ).tolist()


def test_the_best_supported_member_survives():
    # 1 has the most reads and claims its direct neighbour 0; 2 is 0's
    # neighbour, not 1's, so it waits for the next round's pass.
    assert _survivors([(0, 1), (0, 2)], [3, 40, 5], [900, 900, 900]) == [1, 1, 2]
    # Tie on reads: the longer wins, then the lower row.
    assert _survivors([(0, 1)], [5, 5], [900, 950]) == [1, 1]
    assert _survivors([(0, 1)], [5, 5], [900, 900]) == [0, 0]


def test_the_longest_member_survives_when_nothing_will_rebuild_the_consensus():
    """The final output reports the survivor's own sequence, so there the
    order is length first: over the shared span the members agree base for
    base, and the longest discards nobody's extension."""
    rows = [(0, 1, 20, 0)]
    assert _survivors(rows, [40, 3], [900, 920]) == [0, 0]
    assert _survivors(rows, [40, 3], [900, 920], prefer="length") == [1, 1]
    assert claim_order(np.array([40, 3]), np.array([900, 920]), "length").tolist() == [1, 0]
    with pytest.raises(ValueError, match="prefer"):
        claim_order(np.array([1]), np.array([1]), "reads")


def test_merges_do_not_chain():
    """A~B and B~C do not license merging A with C: radius-1 claims only
    DIRECT neighbours of the survivor."""
    assert _survivors([(0, 1), (1, 2)], [50, 10, 5], [900] * 3) == [0, 0, 2]


def test_a_hub_does_not_join_two_split_siblings():
    """Pairwise, the hub is mergeable with each child of the split and the
    children are not mergeable with each other. Grouped as a plain star they
    would all collapse — the limit cycle, reached in one hop."""
    rows = [(2, 0), (2, 1)]
    reads, length = [40, 35, 90], [900] * 3
    origin = np.array([7, 7, -1])
    assert _survivors(rows, reads, length, origin=origin) == [2, 1, 2]
    # --merge-siblings switches the kin guard off by passing no origin.
    assert _survivors(rows, reads, length, origin=None) == [2, 2, 2]
    # Byte-identical siblings cannot be told apart by reads, so they merge.
    twins = [(2, 0, 0, 0, True), (2, 1, 0, 0, True)]
    assert _survivors(twins, reads, length, origin=origin) == [2, 2, 2]


def test_members_on_opposite_sides_of_the_hub_are_too_far_apart():
    """Each is within 30 nt of the hub; from each other they are 50."""
    rows = [(0, 1, 25, 0), (0, 2, -25, 0)]
    assert _survivors(rows, [90, 40, 35], [900] * 3) == [0, 0, 2]
    same_side = [(0, 1, 25, 0), (0, 2, 20, 0)]
    assert _survivors(same_side, [90, 40, 35], [900] * 3) == [0, 0, 0]


def test_a_pair_naming_a_row_that_does_not_exist_is_dropped():
    assert _survivors([(0, 5), (1, 1), (-1, 0), (0, 1)], [5, 1], [900] * 2) == [0, 0]
    assert select_merges(0, *([np.empty(0, np.int64)] * 6), **_TOL).tolist() == []


def _refined(rows, ids):
    return next_templates(_nodes(rows), _store(len(ids), ids=ids), round_index=1)


def test_apply_merge_sums_support_keeps_ids_and_records_lineage():
    seq = "ACGT" * 25
    refined = _refined(
        [
            (10, 0, 0, seq, 30, 30.0),
            (11, 1, 0, seq, 4, 4.0),
            (12, 2, 0, "TTGCA" * 20, 7, 7.0),
        ],
        [10, 11, 12],
    )
    child = refined.templates.column("template_id").to_pylist()
    merged = apply_merge(refined, _by_id([(0, 1)], child), **_TOL)
    assert merged.n_merged == 1 and merged.n_children == 2
    t = merged.templates
    ids = t.column("template_id").to_pylist()
    assert ids == [int(template_id_for(2, 0)), int(template_id_for(2, 2))]
    assert t.column("node_weight").to_pylist()[0] == 34.0
    assert t.column("orf_replication").to_pylist()[0] == 34
    lin = merged.lineage.to_pylist()
    absorbed = [r for r in lin if r["parent_template_id"] == 11][0]
    assert absorbed["rule"] == "merge"
    assert absorbed["child_template_id"] == ids[0]
    # Every other lineage row is exactly what it was.
    assert [r["rule"] for r in lin if r["parent_template_id"] != 11] == ["carry", "carry"]


def test_the_merged_table_records_what_the_lineage_overwrote():
    """The lineage re-points an absorbed child at its survivor, so afterwards
    the absorbed id appears nowhere else."""
    seq = "ACGT" * 25
    refined = _refined(
        [(10, 0, 0, seq + "AC" * 10, 30, 30.0), (11, 1, 0, seq, 4, 4.0)], [10, 11]
    )
    child = refined.templates.column("template_id").to_pylist()
    # By id, 11's child is src and reaches 0 nt beyond; the survivor (10's
    # child) reaches 20 nt further at 3'.
    pairs = _by_id([(1, 0, 0, 20)], child)
    merged = apply_merge(refined, pairs, **_TOL)
    assert merged.merged.schema.equals(MERGED_TABLE, check_metadata=False)
    assert merged.merged.to_pylist() == [
        {
            "round": 2,
            "absorbed_template_id": child[1],
            "survivor_template_id": child[0],
            "delta_5p": 0,
            "delta_3p": 20,
            "absorbed_n_reads": 4,
        }
    ]
    # The same pair stored the other way round reads the same.
    flipped = apply_merge(refined, _by_id([(0, 1, 0, -20)], child), **_TOL)
    assert flipped.merged.to_pylist() == merged.merged.to_pylist()
    # A pass that absorbs nothing still says so, with an empty table.
    nothing = apply_merge(refined, _by_id([], child), **_TOL)
    assert nothing.n_merged == 0 and nothing.merged.num_rows == 0


def test_a_zero_length_node_ahead_of_a_twin_does_not_shift_the_merge():
    """Ids are assigned over node rows and the empty consensus is dropped
    afterwards, so from there on template row != node row. Joined by row, the
    twin at node rows (1, 2) would land on template rows (1, 2): the twin and
    its UNRELATED neighbour."""
    seq = "ACGT" * 25
    refined = _refined(
        [
            (10, 0, 0, "", 2, 2.0),
            (11, 1, 0, seq, 30, 30.0),
            (12, 2, 0, seq, 4, 4.0),
            (13, 3, 0, "TTGCA" * 20, 7, 7.0),
        ],
        [10, 11, 12, 13],
    )
    assert refined.templates.num_rows == 3
    node_ids = template_id_for(2, np.arange(4))
    merged = apply_merge(refined, _by_id([(1, 2)], node_ids), **_TOL)
    t = merged.templates
    assert t.column("template_id").to_pylist() == [int(node_ids[1]), int(node_ids[3])]
    assert t.column("orf_replication").to_pylist() == [34, 7]
    assert t.column("sequence").to_pylist() == [seq, "TTGCA" * 20]
    # A pair with the dropped node is a pair with a template that is not there.
    ghost = apply_merge(refined, _by_id([(0, 1)], node_ids), **_TOL)
    assert ghost.n_merged == 0 and ghost.templates.num_rows == 3


def test_apply_merge_reads_each_template_s_split_origin_by_id():
    """`origin` is indexed by graph NODE. Node 0 has no template, so read by
    template row every origin would be the node's before it."""
    seq = "ACGT" * 25
    nodes = [
        (10, 0, 0, "", 2, 2.0),
        (11, 1, 0, seq, 30, 30.0),
        (11, 1, 1, seq + "A", 4, 4.0),
        (13, 3, 0, seq + "AC", 90, 90.0),
    ]
    refined = _refined(nodes, [10, 11, 12, 13])
    node_ids = template_id_for(2, np.arange(4))
    pairs = _by_id([(3, 1), (3, 2)], node_ids)

    # Nodes 1 and 2 are the children of one split: the hub takes the better
    # supported of them and leaves the other.
    siblings = np.array([-1, 11, 11, -1])
    kept_apart = apply_merge(
        refined, pairs, origin_ids=node_ids, origin=siblings, **_TOL
    )
    assert kept_apart.n_merged == 1
    assert kept_apart.templates.column("template_id").to_pylist() == [
        int(node_ids[2]),
        int(node_ids[3]),
    ]
    # The same origins one node earlier: now it is the EMPTY node and node 1
    # that share one, nodes 1 and 2 do not, and the hub takes both. Read by
    # template row this case and the one above would swap answers.
    shifted = np.array([11, 11, -1, -1])
    both = apply_merge(refined, pairs, origin_ids=node_ids, origin=shifted, **_TOL)
    assert both.n_merged == 2
    assert both.templates.column("template_id").to_pylist() == [int(node_ids[3])]

    assert apply_merge(refined, pairs, **_TOL).n_merged == 2
    with pytest.raises(ValueError, match="origin_ids"):
        apply_merge(refined, pairs, origin=siblings, **_TOL)


def test_churn_reads_a_merge_as_inherited_not_as_movement():
    """A merge is the inverse of a split: reads that moved from an absorbed
    template to its survivor did not choose differently."""
    seq = "ACGT" * 25
    refined = _refined([(10, 0, 0, seq, 3, 3.0), (11, 1, 0, seq, 2, 2.0)], [10, 11])
    child = refined.templates.column("template_id").to_pylist()
    merged = apply_merge(refined, _by_id([(0, 1)], child), **_TOL)
    survivor = merged.templates.column("template_id").to_pylist()[0]
    prev = _assign([(0, 10), (1, 10), (2, 10), (3, 11), (4, 11)])
    cur = _assign([(i, survivor) for i in range(5)])
    churn = measure_churn(prev, cur, merged.lineage, n_reads=5)
    # Every template id changes between rounds, so all five "moved" raw — and
    # through carry (10 -> survivor) and merge (11 -> survivor) edges, none
    # of them genuinely.
    assert churn["n_moved_raw"] == 5
    assert churn["n_moved_genuine"] == 0


def _membership(rows):
    """rows = [(parent_id, hap, read_row)]"""
    return pa.table(
        {
            "round": pa.array([1] * len(rows), pa.int32()),
            "parent_template_id": pa.array([r[0] for r in rows], pa.int64()),
            "haplotype_id": pa.array([r[1] for r in rows], pa.int32()),
            "read_row": pa.array([r[2] for r in rows], pa.int32()),
            "weight": pa.array([1.0] * len(rows), pa.float32()),
        }
    )


def test_final_merge_keeps_every_read():
    seq = "ACGT" * 25
    nodes = _nodes(
        [(10, 0, 0, seq, 3, 3.0), (10, 0, 1, seq, 2, 2.0), (11, 1, 0, "TTGCA" * 20, 1, 1.0)]
    )
    member = _membership(
        [(10, 0, 0), (10, 0, 1), (10, 0, 2), (10, 1, 3), (10, 1, 4), (11, 0, 5)]
    )
    out, mem, n, survivor = merge_nodes(nodes, member, _edges([(0, 1)]), **_TOL)
    assert n == 1 and out.num_rows == 2
    assert survivor.tolist() == [0, 0, 2]
    assert out.column("n_reads").to_pylist() == [5, 1]
    keys = list(zip(mem.column("parent_template_id").to_pylist(),
                    mem.column("haplotype_id").to_pylist()))
    assert keys == [(10, 0)] * 5 + [(11, 0)]
    assert mem.num_rows == member.num_rows
    # Nothing to merge: the tables come back as they went in.
    same, mem2, n2, survivor2 = merge_nodes(nodes, member, _edges([]), **_TOL)
    assert n2 == 0 and same.equals(nodes) and mem2.equals(member)
    assert survivor2.tolist() == [0, 1, 2]


def test_the_final_merge_keeps_the_deepest_node_and_the_caller_rebuilds_it():
    """The deepest member survives, as between rounds. The 5' extension the
    shallower node carries is not lost by this: the caller rebuilds the
    survivor's consensus from the pooled reads (`rebuild.rebuild_survivors`),
    and an extension three reads carry comes back through that. Kept the
    other way round — longest survives, pairs held exact — the longest form
    overshot the annotated TSS in 41% of 914 groups against 24% (ledger #52).
    """
    core = "ACGT" * 25
    nodes = _nodes([(10, 0, 0, core, 30, 30.0), (11, 1, 0, "ATGGCC" + core, 4, 4.0)])
    member = _membership([(10, 0, i) for i in range(30)] + [(11, 0, 30 + i) for i in range(4)])
    out, mem, n, survivor = merge_nodes(nodes, member, _edges([(0, 1, 6, 0)]), **_TOL)
    assert n == 1 and survivor.tolist() == [0, 0]
    assert out.column("consensus").to_pylist() == [core]
    assert out.column("n_reads").to_pylist() == [34]
    assert set(mem.column("parent_template_id").to_pylist()) == {10}
    # The old rule is still there to ask for.
    out, _, _, survivor = merge_nodes(
        nodes, member, _edges([(0, 1, 6, 0)]), prefer="length", **_TOL
    )
    assert survivor.tolist() == [1, 1]
    assert out.column("consensus").to_pylist() == ["ATGGCC" + core]
