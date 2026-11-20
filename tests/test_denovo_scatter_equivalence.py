"""frame_consensus's PWM scatter, vectorised, against the loop it replaced.

The loop voted gap into each insertion block a member covers and does not fill
one (member, block) at a time — 11 of an 18 s M-step on 600 reads with 530
blocks (scripts/bench-em-replay.py). The vectorised scatter must give the
identical PWM, bit for bit, including at non-integer weights: each column takes
at most one vote per member and members still vote in order, so every column's
sum is accumulated in the same order. The reference below is the replaced
function verbatim (main at a633080).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
    _GAP,
    _classify_columns,
    _lookup_sorted,
    _member_projection,
    member_alleles,
    ColumnPlan,
    MemberSpec,
    ProjectedMember,
    _covers_junction,
    _member_events,
    _scatter_expanded,
    base_codes,
    frame_consensus,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.rebuild import (
    align_to_survivor,
)


def _reference_scatter(
    plan: ColumnPlan,
    frame_codes: np.ndarray,
    frame_weight: float,
    projections: Sequence[tuple[ProjectedMember, float]],
    member_codes: Sequence[np.ndarray],
) -> np.ndarray:
    """Accumulate every vote into the ``(n_columns, 5)`` PWM."""
    n = plan.n_columns
    t_at, b_start, b_width = plan.template_at, plan.block_start, plan.block_width
    active = np.flatnonzero(b_width > 0)
    flat_idx: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    def _add(idx: np.ndarray, codes: np.ndarray, w: float) -> None:
        good = (idx >= 0) & (idx < n) & (codes < 4)
        if good.any():
            flat_idx.append(idx[good] * 5 + codes[good])
            weights.append(np.full(int(good.sum()), w, dtype=np.float64))

    def _add_gaps(idx: np.ndarray, w: float) -> None:
        idx = idx[(idx >= 0) & (idx < n)]
        if idx.shape[0]:
            flat_idx.append(idx * 5 + _GAP)
            weights.append(np.full(idx.shape[0], w, dtype=np.float64))

    if frame_weight:
        _add(t_at, frame_codes, frame_weight)

    for (proj, w), codes in zip(projections, member_codes):
        mf, mb, gf = proj.match_fpos, proj.match_base, proj.gap_fpos
        keep = (mf >= 0) & (mf < plan.n_template)
        _add(t_at[mf[keep]], mb[keep], w)
        gk = gf[(gf >= 0) & (gf < plan.n_template)]
        _add_gaps(t_at[gk], w)

        # Insertion blocks. Everything the member covers but does not fill
        # votes gap, so a block column's depth is the covering depth and its
        # minor-allele fraction is the fraction of covering reads that insert.
        filled = {j: c for j, c in _member_events(proj, codes, plan.n_template)}
        for j in active:
            j = int(j)
            if not _covers_junction(proj, j, plan.n_template):
                continue  # no evidence about this junction — no vote
            wdt = int(b_width[j])
            c = filled.get(j)
            k = 0 if c is None else min(int(c.size), wdt)
            if k:
                # Interior and 3'-terminal blocks abut the column on their
                # left, so they left-justify; junction 0 has no left neighbour
                # and abuts template column 0, so it right-justifies.
                if j == 0:
                    at = b_start[j] + wdt - k
                    used = c[-k:]
                else:
                    at = b_start[j]
                    used = c[:k]
                _add(np.arange(at, at + k, dtype=np.int64), used, w)
            if k < wdt:
                empty = (
                    np.arange(b_start[j], b_start[j] + wdt - k, dtype=np.int64)
                    if j == 0
                    else np.arange(b_start[j] + k, b_start[j] + wdt, dtype=np.int64)
                )
                _add_gaps(empty, w)

    # numpy bincount, not torch — this runs inside fork()ed workers, and the
    # parent has already spawned torch's thread pool during minimizer
    # extraction, so a torch op here deadlocks on OpenMP.
    if flat_idx:
        flat = np.bincount(
            np.concatenate(flat_idx),
            weights=np.concatenate(weights),
            minlength=n * 5,
        )
    else:
        flat = np.zeros(n * 5, dtype=np.float64)
    return flat.reshape(n, 5)



def _members(seed: int, n: int, length: int, ins_rate: float, *, weights=None):
    rng = np.random.default_rng(seed)
    template = "".join(rng.choice(list("ACGT"), size=length))
    members = []
    for i in range(n):
        out = []
        start = int(rng.integers(0, length // 5)) if rng.random() < 0.3 else 0
        stop = length - (int(rng.integers(0, length // 5)) if rng.random() < 0.3 else 0)
        for b in template[start:stop]:
            r = rng.random()
            if r < 0.006:
                out.append(rng.choice([c for c in "ACGT" if c != b]))
            elif r < 0.008:
                continue
            elif r < 0.008 + ins_rate:
                out.append(b)
                out.append("".join(rng.choice(list("ACGT"), size=int(rng.integers(1, 4)))))
            else:
                out.append(b)
        seq = "".join(out)
        placed = align_to_survivor(seq, template, identity_floor=0.85)
        if placed is None:
            continue
        w = 1.0 if weights is None else float(weights(rng))
        members.append(
            MemberSpec(seq, w, placed[0], False, placed[1], placed[2], len(members))
        )
    return template, members


@pytest.mark.parametrize(
    ("seed", "n", "length", "ins_rate", "weights"),
    [
        (1, 120, 400, 0.002, None),
        (2, 200, 600, 0.01, None),  # many blocks
        (3, 150, 500, 0.01, lambda rng: rng.integers(1, 9)),  # abundances
        (4, 150, 500, 0.01, lambda rng: rng.uniform(0.1, 3.0)),  # non-integer
    ],
)
@pytest.mark.parametrize("frame_weight", [0.0, 1.0])
def test_the_vectorised_scatter_is_bit_identical(seed, n, length, ins_rate, weights, frame_weight):
    template, members = _members(seed, n, length, ins_rate, weights=weights)
    cres = frame_consensus(template, members, frame_weight=frame_weight)
    plan = cres.plan
    assert int((plan.block_width > 0).sum()) > 0, "the fixture plans no block"
    frame_codes = base_codes(template).astype(np.int64)
    projections = list(cres.projections)
    codes = list(cres.member_codes)
    new = _scatter_expanded(plan, frame_codes, frame_weight, projections, codes)
    old = _reference_scatter(plan, frame_codes, frame_weight, projections, codes)
    assert new.shape == old.shape
    assert np.array_equal(new, old)
    assert np.array_equal(cres.pwm, old)


def _reference_member_row(row, proj, codes, plan, t_idx, t_want, b_idx, j_of, o_of) -> None:
    """Fill ``row`` (pre-set to -1) with one member's alleles.

    The single definition of "what does this member say at this PWM column",
    shared by the dense :func:`member_alleles` and the sparse
    :func:`member_allele_events` so the two cannot drift.
    """
    if t_idx.shape[0]:
        hit, pos = _lookup_sorted(proj.match_fpos, t_want)
        if hit.any():
            vals = proj.match_base[pos[hit]].astype(np.int8)
            # An ambiguous member base is no evidence, not a gap.
            vals = np.where(vals >= 4, np.int8(-1), vals)
            row[t_idx[hit]] = vals
        hitg, _posg = _lookup_sorted(proj.gap_fpos, t_want)
        if hitg.any():
            row[t_idx[hitg]] = 4

    if b_idx.shape[0]:
        events = dict(_member_events(proj, codes, plan.n_template))
        for k in b_idx:
            j = int(j_of[k])
            if not _covers_junction(proj, j, plan.n_template):
                continue  # no evidence about this junction
            w = int(plan.block_width[j])
            o = int(o_of[k])
            c = events.get(j)
            if c is None or c.size == 0:
                row[k] = 4
                continue
            kk = min(int(c.size), w)
            # Junction 0's block abuts template column 0 on its right, so it
            # right-justifies; every other block abuts the column on its left
            # and left-justifies.
            if j == 0:
                filled = o >= w - kk
                src = int(c.size) - (w - o) if filled else -1
            else:
                filled = o < kk
                src = o if filled else -1
            if not filled:
                row[k] = 4
            else:
                base = int(c[src])
                row[k] = -1 if base >= 4 else base



def _reference_member_alleles(cres, columns):
    plan = cres.plan
    cols = np.asarray(columns, dtype=np.int64)
    out = np.full((len(cres.members), cols.shape[0]), -1, dtype=np.int8)
    t_of, j_of, o_of = _classify_columns(plan, cols)
    t_idx = np.flatnonzero(t_of >= 0)
    b_idx = np.flatnonzero(j_of >= 0)
    for i in range(len(cres.members)):
        proj, codes = _member_projection(cres, i)
        _reference_member_row(
            out[i], proj, codes, plan, t_idx, t_of[t_idx], b_idx, j_of, o_of
        )
    return out


@pytest.mark.parametrize(("seed", "ins_rate"), [(5, 0.002), (6, 0.01), (7, 0.02)])
def test_member_alleles_is_unchanged_by_the_vectorised_blocks(seed, ins_rate):
    template, members = _members(seed, 150, 500, ins_rate)
    cres = frame_consensus(template, members, frame_weight=0.0)
    n = cres.plan.n_columns
    rng = np.random.default_rng(seed)
    for cols in (np.arange(n), np.sort(rng.choice(n, size=n // 3, replace=False))):
        assert np.array_equal(
            member_alleles(cres, cols), _reference_member_alleles(cres, cols)
        )


def test_dense_states_is_its_loop() -> None:
    """The vectorized `dense_states` gives the entry-by-entry loop's matrix:
    columns asked in any order or repeated, repeated entries of one read."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.covariance import (
        ReadStates,
        _dense_states_loop,
        dense_states,
    )

    rng = np.random.default_rng(17)
    for _ in range(300):
        m, v_n = int(rng.integers(0, 60)), int(rng.integers(1, 40))
        counts = rng.integers(0, 8, m)
        ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        nnz = int(ptr[-1])
        lo = rng.integers(0, v_n, m).astype(np.int32)
        hi = (lo + rng.integers(0, v_n, m)).clip(max=v_n).astype(np.int32)
        states = ReadStates(
            ptr=ptr,
            v=rng.integers(0, v_n, nnz).astype(np.int32),
            state=rng.integers(1, 7, nnz).astype(np.int8),
            obs_lo=lo,
            obs_hi=hi,
            weight=np.ones(m),
            is_coverage=rng.random(v_n) < 0.3,
            n_members=m,
            n_candidates=v_n,
        )
        cols = rng.integers(0, v_n, int(rng.integers(0, v_n + 5)))
        assert np.array_equal(dense_states(states, cols), _dense_states_loop(states, cols))


def test_reestimate_is_its_loop() -> None:
    """One weighted bincount gives the per-(pattern, column) loop's
    majorities bit for bit, labels out of range and unseen cells included."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.covariance import (
        _reestimate,
        _reestimate_loop,
    )

    rng = np.random.default_rng(19)
    for _ in range(300):
        m, s, k = int(rng.integers(1, 80)), int(rng.integers(1, 30)), int(rng.integers(1, 6))
        x = rng.integers(0, 7, (m, s)).astype(np.int8)
        observed = rng.random((m, s)) < 0.7
        labels = rng.integers(-1, k + 1, m).astype(np.int32)
        w = rng.choice([1.0, 0.5, 0.1, 0.0, 1 / 3], m)
        patterns = rng.integers(0, 7, (k, s)).astype(np.int8)
        assert np.array_equal(
            _reestimate(x, observed, labels, w, patterns),
            _reestimate_loop(x, observed, labels, w, patterns),
        )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_child_from_reused_projections_is_the_child_from_scratch(seed) -> None:
    """A split child built on the pooled plan from the pool's projections of
    its reads equals the child that projects them itself."""
    template, members = _members(seed, 40, 900, 0.01)
    pooled = frame_consensus(template, members, frame_weight=0.0)
    idx = np.random.default_rng(seed).choice(len(members), size=17, replace=False)
    hap = [members[int(i)] for i in idx]
    scratch = frame_consensus(template, hap, frame_weight=0.0, plan=pooled.plan)
    reused = frame_consensus(
        template,
        hap,
        frame_weight=0.0,
        plan=pooled.plan,
        projections=[pooled.projections[int(i)] for i in idx],
        member_codes=[pooled.member_codes[int(i)] for i in idx],
    )
    assert reused.consensus == scratch.consensus
    assert np.array_equal(reused.pwm, scratch.pwm)
    assert np.array_equal(reused.winner, scratch.winner)
    assert reused.alignments == scratch.alignments
    with pytest.raises(ValueError):
        frame_consensus(template, hap, plan=pooled.plan, projections=pooled.projections[:3],
                        member_codes=pooled.member_codes[:3])
