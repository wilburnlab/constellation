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
            "orf_certified_end": pa.array([30] * n, pa.int32()),
            "orf_truncated_by_support": pa.array([False] * n, pa.bool_()),
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
