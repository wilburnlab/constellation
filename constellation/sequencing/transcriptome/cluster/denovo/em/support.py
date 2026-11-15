"""What one replayed template or cluster shows: its reads against a consensus.

The read matrix of a node is reads × PWM columns, but a read agrees with its
consensus almost everywhere, so it is carried as its covered span plus its
*departures* — ``(column, allele)`` where it differs — and its insertions and
clips as markers. That is the M-step's own projection (every member's alleles
on the frame, :func:`~..consensus.member_alleles`), nothing recomputed; it is
lossless against the reference consensus (a test rebuilds every read's
alleles from it), and small enough to render client-side at every zoom.

Coordinates are the frame of the M-step that built the consensus — template
columns plus the insertion blocks its plan reserved — shown either as is
(``space="frame"``) or in the reference consensus's own coordinates
(``space="consensus"``, the default), where a column the consensus does not
carry (a minority insertion, a cluster-wide deletion) is not a column: the
reads' bases there become insertion markers.

The reference is the node a caller asks for (``node=``) — its consensus, on
the shared column plan, so every sibling of a split reads against the same
axis — or, with no node, the template's pooled consensus, against which a
split shows as blocks of reads departing together at the same columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from constellation.sequencing.transcriptome.cluster.denovo.em.replay import (
    ClusterReplay,
    ReplayError,
    TemplateReplay,
)

SPACES = ("consensus", "frame")

READS_SCHEMA = pa.schema(
    [
        pa.field("idx", pa.int32(), nullable=False),
        pa.field("read_row", pa.int32(), nullable=False),
        pa.field("read_id", pa.string(), nullable=True),
        pa.field("sample_id", pa.int64(), nullable=True),
        pa.field("node", pa.int32(), nullable=False),
        # Voted in the PWM. An unsampled read (beyond max_members_per_template)
        # has no alignment here and a null span.
        pa.field("sampled", pa.bool_(), nullable=False),
        pa.field("folded", pa.bool_(), nullable=True),
        pa.field("margin", pa.float32(), nullable=True),
        # Covered columns [lo, hi) in the chosen space.
        pa.field("span_lo", pa.int32(), nullable=True),
        pa.field("span_hi", pa.int32(), nullable=True),
        pa.field("n_departures", pa.int32(), nullable=False),
        pa.field("clip_5p", pa.int32(), nullable=True),
        pa.field("clip_3p", pa.int32(), nullable=True),
        pa.field("dorado_quality", pa.float32(), nullable=True),
        # The E-step's account of the read (null for a rebuilt cluster, whose
        # reads were re-aligned to the survivor, not assigned to it).
        pa.field("identity", pa.float32(), nullable=True),
        pa.field("logl_delta", pa.float32(), nullable=True),
        pa.field("runner_up_template_id", pa.int64(), nullable=True),
        pa.field("runner_up_logl_gap", pa.float32(), nullable=True),
    ]
)

DEPARTURES_SCHEMA = pa.schema(
    [
        pa.field("read_idx", pa.int32(), nullable=False),
        pa.field("col", pa.int32(), nullable=False),
        # 0-3 = ACGT, 4 = gap (the read lacks a base the reference carries).
        pa.field("allele", pa.int8(), nullable=False),
    ]
)

INSERTIONS_SCHEMA = pa.schema(
    [
        pa.field("read_idx", pa.int32(), nullable=False),
        # The read's bases sit immediately before this column (the space's
        # column count: after the last).
        pa.field("before_col", pa.int32(), nullable=False),
        pa.field("length", pa.int32(), nullable=False),
        # True: the bases have PWM columns (voted; the reference just does not
        # carry them). False: the plan reserved no column for them — an
        # unsupported insertion, one longer than its block, or one at the
        # read's own first or last operation — so they never voted.
        pa.field("placed", pa.bool_(), nullable=False),
    ]
)

COLUMNS_SCHEMA = pa.schema(
    [
        pa.field("col", pa.int32(), nullable=False),
        pa.field("frame_col", pa.int32(), nullable=False),
        # -1 on an insertion / extension column of the plan.
        pa.field("template_col", pa.int32(), nullable=False),
        # The reference consensus's base: 0-3 ACGT, 4 gap, 5 unresolved N.
        pa.field("ref_base", pa.int8(), nullable=False),
        pa.field("depth", pa.float64(), nullable=False),
        pa.field("base_cov", pa.float64(), nullable=False),
        pa.field("major", pa.int8(), nullable=False),
        pa.field("minor_frac", pa.float64(), nullable=False),
        # The template's column statistics (pooled), where the kernel ran.
        pa.field("eps", pa.float64(), nullable=True),
        pa.field("hp_run", pa.int32(), nullable=True),
        pa.field("candidate", pa.bool_(), nullable=False),
        pa.field("route", pa.int8(), nullable=True),
        pa.field("signature", pa.int32(), nullable=True),
    ]
)

ARCS_SCHEMA = pa.schema(
    [
        pa.field("u_col", pa.int32(), nullable=False),
        pa.field("v_col", pa.int32(), nullable=False),
        pa.field("self_edge", pa.bool_(), nullable=False),
        pa.field("n11", pa.float64(), nullable=True),
        pa.field("n", pa.float64(), nullable=True),
        pa.field("phi", pa.float64(), nullable=True),
        pa.field("log_odds", pa.float64(), nullable=True),
        pa.field("p_value", pa.float64(), nullable=False),
    ]
)


@dataclass
class SupportTables:
    """The support view's data for one template or cluster."""

    space: str
    #: ``"pooled"`` or the haplotype id whose consensus is the reference.
    reference: str
    #: The reference consensus in the chosen space (frame space: ``-`` where
    #: the reference carries a gap, ``N`` where it is unresolved).
    consensus: str
    n_columns: int
    reads: pa.Table
    departures: pa.Table
    insertions: pa.Table
    columns: pa.Table
    arcs: pa.Table
    verified: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Source:
    pooled: object  # ConsensusResult carrying every member's projection
    children: list  # (haplotype_id, ConsensusResult, lo, hi)
    read_row: np.ndarray  # every read (sampled or not)
    node_of_read: np.ndarray
    member_pos: np.ndarray  # members' positions in read_row
    read_ids: pa.Array
    quality: np.ndarray
    display: pa.Table | None  # E-step columns, aligned to read_row
    internals: object | None
    record_round: int
    template_id: int
    verified: bool
    warnings: list


def _source(replay) -> _Source:
    if isinstance(replay, ClusterReplay) and replay.kind == "node":
        replay = replay.template
    if isinstance(replay, TemplateReplay):
        res = replay.result
        internals = res.internals
        if internals is None:
            raise ReplayError(
                f"template {replay.template_id}: the M-step kernel did not run on "
                f"it (status {res.record.status}); there is no PWM to show"
            )
        return _Source(
            pooled=internals.pooled,
            children=list(internals.children),
            read_row=np.asarray(res.read_row, dtype=np.int64),
            node_of_read=np.asarray(res.haplotype_of_read, dtype=np.int64),
            member_pos=np.asarray(res.member_pos, dtype=np.int64),
            read_ids=replay.read_ids,
            quality=np.asarray(replay.dorado_quality, dtype=np.float64),
            display=replay.rows,
            internals=internals,
            record_round=replay.round,
            template_id=replay.template_id,
            verified=replay.verified,
            warnings=list(replay.warnings),
        )
    if isinstance(replay, ClusterReplay) and replay.kind == "rebuilt":
        from constellation.sequencing.transcriptome.cluster.denovo.em.mstep import (
            _support_span,
        )

        if replay.pooled is None:
            raise ReplayError(f"cluster {replay.cluster_id}: the rebuild placed no read")
        lo, hi = _support_span(replay.pooled)
        pool = np.asarray(replay.pool_read_row, dtype=np.int64)
        order = np.argsort(pool, kind="stable")
        pos = order[np.searchsorted(pool[order], replay.member_read_row)]
        return _Source(
            pooled=replay.pooled,
            children=[(0, replay.pooled, lo, hi)],
            read_row=pool,
            node_of_read=np.zeros(pool.size, dtype=np.int64),
            member_pos=pos.astype(np.int64),
            read_ids=replay.read_ids,
            quality=np.asarray(replay.dorado_quality, dtype=np.float64),
            display=None,
            internals=None,
            record_round=replay.round,
            template_id=-1,
            verified=replay.verified,
            warnings=list(replay.warnings),
        )
    raise TypeError(f"not a replay: {type(replay).__name__}")


def support_tables(replay, *, node: int | None = None, space: str = "consensus"):
    """The support view's tables for a :class:`~.replay.TemplateReplay` or
    :class:`~.replay.ClusterReplay`.

    ``node`` picks the reference consensus (a haplotype id of the template;
    a node replay of a cluster defaults to its own) and restricts the reads
    to that node's; ``None`` reads every read against the pooled consensus.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
        member_alleles,
        member_spans,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep import (
        _support_span,
    )

    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")
    if node is None and isinstance(replay, ClusterReplay):
        node = replay.haplotype_id
    src = _source(replay)
    pooled = src.pooled
    plan = pooled.plan
    n_frame = int(plan.n_columns)

    if node is None:
        ref = pooled
        lo, hi = _support_span(pooled)
        reference = "pooled"
    else:
        found = [c for c in src.children if int(c[0]) == int(node)]
        if not found:
            raise ReplayError(f"no node {node} in this replay")
        _hid, ref, lo, hi = found[0]
        reference = str(int(node))
    winner = np.asarray(ref.winner, dtype=np.int64)
    cons_of_frame = np.asarray(ref.cons_of_frame, dtype=np.int64)
    in_trim = (cons_of_frame >= lo) & (cons_of_frame < hi)
    cons_col = np.where((cons_of_frame >= 0) & in_trim, cons_of_frame - lo, -1)

    # Which reads: all, or the chosen node's.
    reads_sel = (
        np.arange(src.read_row.size)
        if node is None or reference == "pooled"
        else np.flatnonzero(src.node_of_read == int(node))
    )
    member_of_read = np.full(src.read_row.size, -1, dtype=np.int64)
    member_of_read[src.member_pos] = np.arange(src.member_pos.size)
    members = member_of_read[reads_sel]
    sampled = members >= 0
    m_idx = members[sampled]

    alleles = (
        member_alleles(pooled, np.arange(n_frame))[m_idx]
        if m_idx.size
        else np.zeros((0, n_frame), dtype=np.int8)
    )
    spans_lo, spans_hi = member_spans(pooled)

    # Departures in frame space: covered, and not the reference's allele.
    ref_code = np.where(winner == 5, -2, winner)  # an N departs from everything
    differs = (alleles >= 0) & (alleles != ref_code[None, :])
    di, dc = np.nonzero(differs)
    da = alleles[di, dc]
    read_idx_of_member = np.flatnonzero(sampled)  # row in the reads table

    # Insertions the plan never placed, from the members' own events.
    ins_rows: list[tuple[int, int, int, bool]] = []
    block_start = np.asarray(plan.block_start, dtype=np.int64)
    block_width = np.asarray(plan.block_width, dtype=np.int64)
    t_at = np.asarray(plan.template_at, dtype=np.int64)
    n_t = int(plan.n_template)
    projections = pooled.projections
    for k, m in enumerate(m_idx.tolist()):
        proj = projections[m][0]
        for j, codes in proj.ins_events:
            j = int(j)
            length = int(len(codes))
            if length <= 0:
                continue
            b0, bw = int(block_start[j]), int(block_width[j])
            placed = int(((alleles[k, b0 : b0 + bw] >= 0) & (alleles[k, b0 : b0 + bw] < 4)).sum()) if bw else 0
            extra = length - placed
            if extra > 0:
                before = int(t_at[j]) if j < n_t else n_frame
                ins_rows.append((int(read_idx_of_member[k]), before, extra, False))

    # Columns, in frame space first.
    pwm = np.asarray(ref.pwm, dtype=np.float64)
    depth = pwm.sum(axis=1)
    base_cov = pwm[:, :4].sum(axis=1)
    major = np.argmax(pwm, axis=1).astype(np.int8)
    with np.errstate(invalid="ignore", divide="ignore"):
        minor_frac = np.where(depth > 0, (depth - pwm.max(axis=1)) / depth, 0.0)
    eps = hp = route = signature = None
    candidate = np.zeros(n_frame, dtype=bool)
    arcs = ARCS_SCHEMA.empty_table()
    margin = folded = None
    if src.internals is not None:
        from constellation.sequencing.transcriptome.cluster.denovo.em.evidence import (
            compact_evidence,
        )

        st = src.internals.stats
        eps = np.asarray(st.eps, dtype=np.float64)
        hp = np.asarray(st.hp_run).astype(np.int32)
        ev = compact_evidence(
            round_index=src.record_round,
            template_id=src.template_id,
            internals=src.internals,
            read_row=src.read_row,
            haplotype_of_read=src.node_of_read,
            member_pos=src.member_pos,
        )
        cols_t = ev["columns"]
        fc = np.asarray(cols_t.column("frame_col").to_numpy(zero_copy_only=False), np.int64)
        candidate[fc] = True
        route = np.full(n_frame, -1, dtype=np.int8)
        route[fc] = cols_t.column("route").to_numpy(zero_copy_only=False)
        signature = np.full(n_frame, -1, dtype=np.int32)
        signature[fc] = cols_t.column("signature").to_numpy(zero_copy_only=False)
        cov = ev["covariance"]
        if cov.num_rows:
            arcs = pa.table(
                {
                    "u_col": cov.column("u_frame_col"),
                    "v_col": cov.column("v_frame_col"),
                    "self_edge": cov.column("self_edge"),
                    "n11": cov.column("n11"),
                    "n": cov.column("n"),
                    "phi": cov.column("phi"),
                    "log_odds": cov.column("log_odds"),
                    "p_value": cov.column("p_value"),
                },
                schema=ARCS_SCHEMA,
            )
        if ev["reads"].num_rows:
            rr = ev["reads"]
            by_row = dict(
                zip(
                    rr.column("read_row").to_pylist(),
                    zip(rr.column("margin").to_pylist(), rr.column("folded").to_pylist()),
                )
            )
            margin = [by_row.get(int(r), (None, None))[0] for r in src.read_row[reads_sel]]
            folded = [by_row.get(int(r), (None, None))[1] for r in src.read_row[reads_sel]]

    letters = np.array(list("ACGT-N"))
    if space == "frame":
        col_map = np.arange(n_frame, dtype=np.int64)
        keep_col = np.ones(n_frame, dtype=bool)
        consensus = "".join(letters[np.clip(winner, 0, 5)])
        n_cols = n_frame
    else:
        col_map = cons_col
        keep_col = cons_col >= 0
        consensus = ref.consensus[lo:hi]
        n_cols = int(hi - lo)
        # The reads' bases at columns the reference does not carry are
        # insertions before the next column it does carry.
        cons_frames = np.flatnonzero(keep_col)
        gap_cols = np.flatnonzero(~keep_col)
        if gap_cols.size and alleles.shape[0]:
            sub = alleles[:, gap_cols]
            gi, gk = np.nonzero((sub >= 0) & (sub < 4))
            if gi.size:
                before = np.searchsorted(cons_frames, gap_cols[gk])
                before = np.clip(before, 0, n_cols)
                key = gi.astype(np.int64) * (n_cols + 1) + before
                uniq, counts = np.unique(key, return_counts=True)
                for kk, cnt in zip(uniq.tolist(), counts.tolist()):
                    gi_k, b = divmod(kk, n_cols + 1)
                    ins_rows.append((int(read_idx_of_member[gi_k]), int(b), int(cnt), True))
        # Unplaced insertions' positions, into this space.
        ins_rows = [
            (r, int(np.clip(np.searchsorted(cons_frames, b), 0, n_cols)) if not p else b, n, p)
            for r, b, n, p in ins_rows
        ]
        keep_dep = keep_col[dc]
        di, dc, da = di[keep_dep], dc[keep_dep], da[keep_dep]

    dep_col = col_map[dc]
    dep_read = read_idx_of_member[di]
    n_dep = np.bincount(dep_read, minlength=reads_sel.size).astype(np.int32)

    # Spans in the chosen space.
    s_lo = np.full(reads_sel.size, -1, dtype=np.int64)
    s_hi = np.full(reads_sel.size, -1, dtype=np.int64)
    if m_idx.size:
        flo, fhi = spans_lo[m_idx], spans_hi[m_idx]
        if space == "frame":
            s_lo[sampled], s_hi[sampled] = flo, fhi
        else:
            cons_frames = np.flatnonzero(keep_col)
            s_lo[sampled] = np.searchsorted(cons_frames, flo)
            s_hi[sampled] = np.searchsorted(cons_frames, fhi)
    clip5 = np.full(reads_sel.size, -1, dtype=np.int64)
    clip3 = np.full(reads_sel.size, -1, dtype=np.int64)
    for k, m in enumerate(m_idx.tolist()):
        proj = projections[m][0]
        seq_len = len(pooled.members[m].member_seq)
        r = read_idx_of_member[k]
        clip5[r] = int(proj.member_start)
        clip3[r] = int(seq_len - proj.member_end)

    def _disp(name, kind):
        if src.display is None or name not in src.display.column_names:
            return pa.nulls(reads_sel.size, kind)
        return src.display.column(name).take(pa.array(reads_sel)).cast(kind)

    has_span = s_lo >= 0
    reads = pa.table(
        {
            "idx": pa.array(np.arange(reads_sel.size, dtype=np.int32)),
            "read_row": pa.array(src.read_row[reads_sel].astype(np.int32)),
            "read_id": src.read_ids.take(pa.array(reads_sel)).cast(pa.string()),
            "sample_id": _disp("sample_id", pa.int64()),
            "node": pa.array(src.node_of_read[reads_sel].astype(np.int32)),
            "sampled": pa.array(sampled),
            "folded": pa.array(folded, pa.bool_())
            if folded is not None
            else pa.nulls(reads_sel.size, pa.bool_()),
            "margin": pa.array(margin, pa.float32())
            if margin is not None
            else pa.nulls(reads_sel.size, pa.float32()),
            "span_lo": pa.array(s_lo.astype(np.int32), mask=~has_span),
            "span_hi": pa.array(s_hi.astype(np.int32), mask=~has_span),
            "n_departures": pa.array(n_dep),
            "clip_5p": pa.array(clip5.astype(np.int32), mask=clip5 < 0),
            "clip_3p": pa.array(clip3.astype(np.int32), mask=clip3 < 0),
            "dorado_quality": pa.array(
                src.quality[reads_sel].astype(np.float32),
                mask=~np.isfinite(src.quality[reads_sel]),
            ),
            "identity": _disp("identity", pa.float32()),
            "logl_delta": _disp("logl_delta", pa.float32()),
            "runner_up_template_id": _disp("runner_up_template_id", pa.int64()),
            "runner_up_logl_gap": _disp("runner_up_logl_gap", pa.float32()),
        },
        schema=READS_SCHEMA,
    )
    departures = pa.table(
        {
            "read_idx": pa.array(dep_read.astype(np.int32)),
            "col": pa.array(dep_col.astype(np.int32)),
            "allele": pa.array(da.astype(np.int8)),
        },
        schema=DEPARTURES_SCHEMA,
    )
    ins = sorted(ins_rows)
    insertions = pa.table(
        {
            "read_idx": pa.array([r[0] for r in ins], pa.int32()),
            "before_col": pa.array([r[1] for r in ins], pa.int32()),
            "length": pa.array([r[2] for r in ins], pa.int32()),
            "placed": pa.array([r[3] for r in ins], pa.bool_()),
        },
        schema=INSERTIONS_SCHEMA,
    )
    fcols = np.flatnonzero(keep_col)

    def _opt(values, kind):
        if values is None:
            return pa.nulls(fcols.size, kind)
        return pa.array(np.asarray(values)[fcols], kind)

    columns = pa.table(
        {
            "col": pa.array(col_map[fcols].astype(np.int32)),
            "frame_col": pa.array(fcols.astype(np.int32)),
            "template_col": pa.array(
                np.asarray(pooled.template_of_frame, dtype=np.int64)[fcols].astype(np.int32)
            ),
            "ref_base": pa.array(winner[fcols].astype(np.int8)),
            "depth": pa.array(depth[fcols]),
            "base_cov": pa.array(base_cov[fcols]),
            "major": pa.array(major[fcols]),
            "minor_frac": pa.array(minor_frac[fcols]),
            "eps": _opt(eps, pa.float64()),
            "hp_run": _opt(hp, pa.int32()),
            "candidate": pa.array(candidate[fcols]),
            "route": _opt(route, pa.int8()),
            "signature": _opt(signature, pa.int32()),
        },
        schema=COLUMNS_SCHEMA,
    )
    if space == "consensus" and arcs.num_rows:
        u = cons_col[arcs.column("u_col").to_numpy(zero_copy_only=False)]
        v = cons_col[arcs.column("v_col").to_numpy(zero_copy_only=False)]
        ok = (u >= 0) & (v >= 0)
        arcs = arcs.filter(pa.array(ok))
        arcs = arcs.set_column(0, "u_col", pa.array(u[ok].astype(np.int32)))
        arcs = arcs.set_column(1, "v_col", pa.array(v[ok].astype(np.int32)))
    return SupportTables(
        space=space,
        reference=reference,
        consensus=consensus,
        n_columns=n_cols,
        reads=reads,
        departures=departures,
        insertions=insertions,
        columns=columns,
        arcs=arcs,
        verified=src.verified,
        warnings=src.warnings,
    )


__all__ = [
    "ARCS_SCHEMA",
    "COLUMNS_SCHEMA",
    "DEPARTURES_SCHEMA",
    "INSERTIONS_SCHEMA",
    "READS_SCHEMA",
    "SPACES",
    "SupportTables",
    "support_tables",
]
