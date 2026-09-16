"""M-step fan-out: member construction and unit scheduling.

The prototype's M-step took 12.7 h on 8 processes at round 1. Two causes, and
both are pinned here: read sequences reached workers as a 9.4M-entry Python
dict (defeating copy-on-write), and templates were handed out as static
contiguous ranges so one mega-template serialised the stage.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
    EM_ASSIGNMENT_TABLE,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
    iter_unit_batches,
    plan_mstep_units,
    sort_assignments_by_template,
    specs_from_assignment_slice,
)


class _Reads:
    def __init__(self, seqs):
        self._seqs = list(seqs)

    def take_sequences(self, rows):
        return [self._seqs[int(r)] for r in rows]


def _assignments(rows):
    """rows = [(read_row, template_row, cigar, t_start, q_start, weight)]"""
    n = len(rows)
    return pa.table(
        {
            "read_id": pa.array([f"r{r[0]}" for r in rows], pa.string()),
            "read_row": pa.array([r[0] for r in rows], pa.int32()),
            "template_id": pa.array([r[1] for r in rows], pa.int64()),
            "template_row": pa.array([r[1] for r in rows], pa.int32()),
            "round": pa.array([1] * n, pa.int32()),
            "weight": pa.array([r[5] for r in rows], pa.float32()),
            "as_score": pa.array([0] * n, pa.int32()),
            "as_delta": pa.array([0] * n, pa.int32()),
            "logl": pa.nulls(n, pa.float32()),
            "logl_delta": pa.nulls(n, pa.float32()),
            "n_hits": pa.array([1] * n, pa.int32()),
            "n_admitted": pa.array([1] * n, pa.int32()),
            "candidate_cap_hit": pa.array([False] * n, pa.bool_()),
            "offset_5p": pa.array([0] * n, pa.int32()),
            "q_start": pa.array([r[4] for r in rows], pa.int32()),
            "q_end": pa.array([100] * n, pa.int32()),
            "t_start": pa.array([r[3] for r in rows], pa.int32()),
            "t_end": pa.array([100] * n, pa.int32()),
            "cigar": pa.array([r[2] for r in rows], pa.large_string()),
            "sample_id": pa.array([0] * n, pa.int64()),
        },
        schema=EM_ASSIGNMENT_TABLE,
    )


# ── members ───────────────────────────────────────────────────────────


def test_members_come_from_the_corpus_by_row_not_by_read_id():
    reads = _Reads(["AAAA", "CCCC", "GGGG"])
    tbl = _assignments([(2, 0, "4=", 0, 0, 1.0), (0, 0, "4=", 3, 1, 2.0)])
    members, read_row, fraction = specs_from_assignment_slice(tbl, reads)
    assert fraction == 1.0
    # The read rows run PARALLEL to members, so a node's member_ids can be
    # turned back into reads — without that, a split parent's children have no
    # recoverable membership.
    assert read_row.tolist() == [2, 0]
    assert [m.member_seq for m in members] == ["GGGG", "AAAA"]
    assert [m.ref_start for m in members] == [0, 3]
    assert [m.member_start for m in members] == [0, 1]
    assert [m.weight for m in members] == [1.0, 2.0]
    # minimap2 emits query -> target, so the template is the reference.
    assert all(not m.centroid_is_query for m in members)


def test_rows_without_a_cigar_are_skipped():
    reads = _Reads(["AAAA", "CCCC"])
    tbl = _assignments([(0, 0, None, 0, 0, 1.0), (1, 0, "4=", 0, 0, 1.0)])
    members, read_row, _ = specs_from_assignment_slice(tbl, reads)
    assert [m.member_seq for m in members] == ["CCCC"]
    assert read_row.tolist() == [1]


def test_member_cap_takes_a_sample_not_a_prefix():
    """The cap must not systematically favour the head of the table.

    Every assigned read has weight 1.0 under the current rule, so a
    weight-ordered stable sort degenerates to "the first max_members rows" —
    a prefix, not a sample. A minority variant that happens to sit late in
    the assignment order then disappears entirely, which is exactly the
    population this pipeline exists to keep.
    """
    reads = _Reads([f"{i:04d}" for i in range(100)])
    rows = [(i, 0, "4=", 0, 0, 1.0) for i in range(100)]
    members, read_row, fraction = specs_from_assignment_slice(
        _assignments(rows), reads, max_members=20, seed=7
    )
    assert len(members) == 20
    assert fraction == pytest.approx(0.2)
    assert sorted(read_row.tolist()) == read_row.tolist(), "order preserved"
    # Not a prefix: the sample must reach the tail of the table.
    assert max(read_row) >= 50
    assert len(set(read_row.tolist())) == 20


def test_the_member_sample_is_reproducible():
    """A re-run and a resumed run must agree on which members were used."""
    reads = _Reads([f"{i:04d}" for i in range(100)])
    tbl = _assignments([(i, 0, "4=", 0, 0, 1.0) for i in range(100)])
    a = specs_from_assignment_slice(tbl, reads, max_members=20, seed=42)[1]
    b = specs_from_assignment_slice(tbl, reads, max_members=20, seed=42)[1]
    c = specs_from_assignment_slice(tbl, reads, max_members=20, seed=43)[1]
    assert a.tolist() == b.tolist()
    assert a.tolist() != c.tolist(), "a different template must sample differently"


def test_empty_slice_is_not_an_error():
    members, read_row, fraction = specs_from_assignment_slice(
        _assignments([]), _Reads([])
    )
    assert members == []
    assert read_row.tolist() == []
    assert fraction == 1.0


# ── scheduling ────────────────────────────────────────────────────────


def test_unassigned_rows_are_excluded_from_the_sort():
    tbl = _assignments([(0, -1, None, 0, 0, 0.0), (1, 3, "4=", 0, 0, 1.0)])
    srt, starts = sort_assignments_by_template(tbl)
    assert srt.num_rows == 1
    assert srt.column("template_row").to_pylist() == [3]
    assert starts.tolist() == [0]


def test_sorting_groups_templates_contiguously():
    rows = [(i, i % 3, "4=", 0, 0, 1.0) for i in range(9)]
    srt, starts = sort_assignments_by_template(_assignments(rows))
    tr = srt.column("template_row").to_pylist()
    assert tr == sorted(tr)
    assert starts.tolist() == [0, 3, 6]


def test_lpt_balances_a_flat_distribution():
    n = 20_000
    rng = np.random.default_rng(3)
    sizes = rng.integers(1, 40, n)
    lo = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    units = plan_mstep_units(
        np.arange(n), lo, lo + sizes, np.full(n, 1500), n_units=128
    )
    loads = np.array([u.cost for u in units])
    assert loads.max() / loads.mean() < 1.01


def test_the_member_cap_is_what_tames_a_mega_template():
    """Without the cap one template is 60x a unit's budget and cannot be split."""
    n = 5_000
    rng = np.random.default_rng(4)
    sizes = rng.integers(1, 40, n)
    sizes[0] = 500_000
    lo = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    args = (np.arange(n), lo, lo + sizes, np.full(n, 1500))

    uncapped = plan_mstep_units(*args, n_units=64, max_members=0)
    capped = plan_mstep_units(*args, n_units=64, max_members=20_000)

    u_loads = np.array([u.cost for u in uncapped])
    c_loads = np.array([u.cost for u in capped])
    assert u_loads.max() / u_loads.mean() > 10.0
    assert c_loads.max() / c_loads.mean() < u_loads.max() / u_loads.mean()


def test_units_are_emitted_largest_first():
    """Combined with a bounded in-flight window this gives LPT + dynamic."""
    n = 500
    sizes = np.arange(1, n + 1)
    lo = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    units = plan_mstep_units(np.arange(n), lo, lo + sizes, np.full(n, 1000), n_units=16)
    loads = [u.cost for u in units]
    assert loads == sorted(loads, reverse=True)


def test_every_live_template_lands_in_exactly_one_unit():
    n = 300
    rng = np.random.default_rng(5)
    sizes = rng.integers(0, 10, n)  # some are zero and must be dropped
    lo = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    units = plan_mstep_units(np.arange(n), lo, lo + sizes, np.full(n, 800), n_units=32)
    seen = np.concatenate([u.rows for u in units]) if units else np.array([])
    assert sorted(seen.tolist()) == sorted(np.flatnonzero(sizes > 0).tolist())


# ── what crosses the process boundary ─────────────────────────────────


def _big_sorted_assignments(n_rows: int, n_templates: int):
    rng = np.random.default_rng(0)
    tr = np.sort(rng.integers(0, n_templates, n_rows)).astype(np.int32)
    rows = [
        (int(i), int(tr[i]), "100=1X50=" * 8, 0, 0, 1.0) for i in range(n_rows)
    ]
    return _assignments(rows), tr


def test_a_unit_pickles_to_its_own_size_not_the_whole_rounds():
    """The 1M-read OOM, in one assertion.

    `pa.concat_tables` of one slice per template gives one CHUNK per template,
    and every Arrow chunk references the whole parent buffer — so pickling for
    ProcessPoolExecutor serialises the parent once per chunk and the payload
    scales with the NUMBER OF TEMPLATES in the unit, not its rows. Measured
    before the fix: 7,360 MB for 3.7 MB of content.

    It also explains why fewer workers made it worse: n_units is
    workers x units_per_worker, so fewer workers means more templates per
    unit means a linearly larger pickle.
    """
    import pickle

    n_rows, n_templates = 40_000, 800
    srt, tr = _big_sorted_assignments(n_rows, n_templates)
    starts = np.flatnonzero(np.concatenate([[True], tr[1:] != tr[:-1]]))
    bounds = np.concatenate([starts, [n_rows]])
    lo = np.zeros(n_templates, np.int64)
    hi = np.zeros(n_templates, np.int64)
    lo[tr[starts]] = bounds[:-1]
    hi[tr[starts]] = bounds[1:]

    # ONE unit holding every template — the worst case, and the one a small
    # --mstep-workers produces.
    units = plan_mstep_units(
        np.arange(n_templates), lo, hi, np.full(n_templates, 1500), n_units=1
    )
    batch = next(iter(iter_unit_batches(srt, units)))
    assert batch.num_rows == n_rows

    pickled = len(pickle.dumps(batch))
    # The payload must track the unit's CONTENT. Anything proportional to the
    # template count is the bug returning.
    assert pickled < 3 * batch.nbytes, (
        f"{pickled / 1e6:.1f} MB pickled for {batch.nbytes / 1e6:.1f} MB of "
        "content — a chunk-per-template table is being serialised"
    )
    assert pickled < srt.nbytes, "a unit must not out-weigh the whole round"


def test_units_carry_only_the_columns_the_worker_reads():
    """read_id and thirteen others crossed the pickle boundary for nothing."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MSTEP_WORKER_COLUMNS,
    )

    srt, tr = _big_sorted_assignments(400, 20)
    starts = np.flatnonzero(np.concatenate([[True], tr[1:] != tr[:-1]]))
    bounds = np.concatenate([starts, [400]])
    lo = np.zeros(20, np.int64)
    hi = np.zeros(20, np.int64)
    lo[tr[starts]] = bounds[:-1]
    hi[tr[starts]] = bounds[1:]
    units = plan_mstep_units(np.arange(20), lo, hi, np.full(20, 1500), n_units=2)
    batch = next(iter(iter_unit_batches(srt, units)))
    assert set(batch.column_names) == set(MSTEP_WORKER_COLUMNS)
    assert "read_id" not in batch.column_names


def test_unit_batches_still_hold_every_row_of_their_templates():
    """The projection and the combine must not change what a unit contains."""
    srt, tr = _big_sorted_assignments(1000, 50)
    starts = np.flatnonzero(np.concatenate([[True], tr[1:] != tr[:-1]]))
    bounds = np.concatenate([starts, [1000]])
    lo = np.zeros(50, np.int64)
    hi = np.zeros(50, np.int64)
    lo[tr[starts]] = bounds[:-1]
    hi[tr[starts]] = bounds[1:]
    units = plan_mstep_units(np.arange(50), lo, hi, np.full(50, 1500), n_units=7)

    seen = []
    for batch in iter_unit_batches(srt, units):
        seen.extend(batch.column("read_row").to_pylist())
    assert sorted(seen) == list(range(1000)), "every row lands in exactly one unit"
