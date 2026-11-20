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
from pathlib import Path

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
        # The node of the final merge's group this read came from, when the
        # merge absorbed that node into the cluster; null for the survivor's
        # own reads and for any template replay.
        pa.field("absorbed_from", pa.int64(), nullable=True),
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
        # 0-3 = ACGT, 4 = gap (the read lacks a base the reference carries),
        # 5 = N (the read covers the column with an ambiguous base: no
        # evidence, which is not agreement with the reference).
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
    #: The replay's own verdict — for a cluster, the cluster's (its consensus
    #: checked against the cluster's), not only its node's.
    verified: bool
    warnings: list[str] = field(default_factory=list)
    mismatch: list[str] = field(default_factory=list)


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
    #: The template's own reads are read_row[:n_own]; any after are the
    #: final merge's absorbed reads, which the M-step's evidence never saw.
    n_own: int = -1
    #: Per read: the absorbed node it came from, -1 for none.
    absorbed_from: np.ndarray | None = None
    mismatch: list = field(default_factory=list)


def _template_source(replay: TemplateReplay) -> _Source:
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        round_runner_up_measured,
    )

    res = replay.result
    internals = res.internals
    if internals is None:
        raise ReplayError(
            f"template {replay.template_id}: the M-step kernel did not run on "
            f"it (status {res.record.status}); there is no PWM to show"
        )
    read_row = np.asarray(res.read_row, dtype=np.int64)
    warnings = list(replay.warnings)
    rd = Path(replay.run_dir) / "rounds" / f"r{int(replay.round):02d}"
    if not round_runner_up_measured(rd):
        warnings.append(
            "this round's rule never measured a runner-up (the lazy round-1 "
            "walk stops at its first admitted template): a null "
            "runner_up_template_id is unknown, not 'no rival'"
        )
    return _Source(
        pooled=internals.pooled,
        children=list(internals.children),
        read_row=read_row,
        node_of_read=np.asarray(res.haplotype_of_read, dtype=np.int64),
        member_pos=np.asarray(res.member_pos, dtype=np.int64),
        read_ids=replay.read_ids,
        quality=np.asarray(replay.dorado_quality, dtype=np.float64),
        display=replay.rows,
        internals=internals,
        record_round=replay.round,
        template_id=replay.template_id,
        verified=replay.verified,
        warnings=warnings,
        n_own=int(read_row.size),
        absorbed_from=np.full(read_row.size, -1, dtype=np.int64),
        mismatch=list(replay.mismatch),
    )


def _source(replay) -> _Source:
    if isinstance(replay, TemplateReplay):
        return _template_source(replay)
    if isinstance(replay, ClusterReplay) and replay.kind in ("node", "merged"):
        # The node's evidence, under the CLUSTER's verdict: the cluster check
        # (its consensus against the cluster's) is the replay's, and a
        # mismatch there must not come back verified from the template.
        src = _template_source(replay.template)
        src.verified = bool(replay.verified)
        src.mismatch = list(replay.mismatch)
        src.warnings = list(dict.fromkeys(list(replay.warnings) + src.warnings))
        extra = replay.absorbed_read_row
        if replay.kind == "merged" and extra is not None and extra.size:
            n = int(extra.size)
            src.read_row = np.concatenate([src.read_row, extra.astype(np.int64)])
            src.node_of_read = np.concatenate(
                [src.node_of_read, np.full(n, int(replay.haplotype_id), np.int64)]
            )
            src.absorbed_from = np.concatenate(
                [src.absorbed_from, np.asarray(replay.absorbed_node_template_id, np.int64)]
            )
            src.read_ids = pa.concat_arrays(
                [
                    pa.array(src.read_ids).cast(pa.string()),
                    pa.array(replay.absorbed_read_ids).cast(pa.string()),
                ]
            )
            src.quality = np.concatenate(
                [src.quality, np.asarray(replay.absorbed_quality, np.float64)]
            )
            if src.display is not None:
                pad = pa.table(
                    {f.name: pa.nulls(n, f.type) for f in src.display.schema},
                    schema=src.display.schema,
                )
                src.display = pa.concat_tables([src.display, pad])
        return src
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
        # The rebuilt consensus is the SURVIVOR's, and carries its haplotype
        # id: a survivor need not have been its parent's node 0, and asking
        # for node 0 of it raised "no node 1" (Nov2026 run, ledger #78a).
        hid = int(replay.haplotype_id)
        from_node = (
            np.asarray(replay.pool_node_template_id, dtype=np.int64)
            if replay.pool_node_template_id is not None
            else np.full(pool.size, int(replay.node_template_id), np.int64)
        )
        return _Source(
            pooled=replay.pooled,
            children=[(hid, replay.pooled, lo, hi)],
            read_row=pool,
            node_of_read=np.full(pool.size, hid, dtype=np.int64),
            member_pos=pos.astype(np.int64),
            read_ids=replay.read_ids,
            quality=np.asarray(replay.dorado_quality, dtype=np.float64),
            display=None,
            internals=None,
            record_round=replay.round,
            template_id=-1,
            verified=replay.verified,
            warnings=list(replay.warnings),
            n_own=int(pool.size),
            absorbed_from=np.where(
                from_node != int(replay.node_template_id), from_node, -1
            ),
            mismatch=list(replay.mismatch),
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
    if m_idx.size:
        # Inside its span a read says something at every column, and an
        # ambiguous base is "no evidence" — which member_alleles writes as
        # -1, the same code as "uncovered". Left so, span + departures
        # rebuilt an interior N as the reference's base: 5 keeps it an N.
        cols = np.arange(n_frame, dtype=np.int64)
        inside = (cols[None, :] >= spans_lo[m_idx][:, None]) & (
            cols[None, :] < spans_hi[m_idx][:, None]
        )
        alleles = np.where((alleles == -1) & inside, np.int8(5), alleles)

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
            block = alleles[k, b0 : b0 + bw]
            # A base or an N fills a block column; 4 is "inserts nothing here".
            placed = int(((block >= 0) & (block != 4)).sum()) if bw else 0
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
        own = slice(0, src.n_own if src.n_own >= 0 else src.read_row.size)
        ev = compact_evidence(
            round_index=src.record_round,
            template_id=src.template_id,
            internals=src.internals,
            read_row=src.read_row[own],
            haplotype_of_read=src.node_of_read[own],
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
            gi, gk = np.nonzero((sub >= 0) & (sub != 4))
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
    absorbed = (
        src.absorbed_from[reads_sel]
        if src.absorbed_from is not None
        else np.full(reads_sel.size, -1, dtype=np.int64)
    )
    reads = pa.table(
        {
            "idx": pa.array(np.arange(reads_sel.size, dtype=np.int32)),
            "read_row": pa.array(src.read_row[reads_sel].astype(np.int32)),
            "read_id": src.read_ids.take(pa.array(reads_sel)).cast(pa.string()),
            "sample_id": _disp("sample_id", pa.int64()),
            "node": pa.array(src.node_of_read[reads_sel].astype(np.int32)),
            "absorbed_from": pa.array(absorbed, mask=absorbed < 0),
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
        mismatch=src.mismatch,
    )


DIFFERENCES_SCHEMA = pa.schema(
    [
        pa.field("frame_col", pa.int32(), nullable=False),
        pa.field("haplotype_id", pa.int32(), nullable=False),
        # 0-3 ACGT, 4 gap, 5 N: the node's consensus at the column.
        pa.field("allele", pa.int8(), nullable=False),
    ]
)


def _template_of(replay):
    if isinstance(replay, TemplateReplay):
        return replay
    if isinstance(replay, ClusterReplay) and replay.kind in ("node", "merged"):
        return replay.template
    return None


def replay_summary(replay) -> dict:
    """What a replay is, for a header: kind, round, template, the nodes with
    their read counts, how many reads voted in the PWM, and the verdict."""
    tmpl = _template_of(replay)
    out: dict = {
        "kind": "template" if isinstance(replay, TemplateReplay) else replay.kind,
        "round": int(replay.round),
        "verified": bool(replay.verified),
        "mismatch": list(replay.mismatch),
        "warnings": list(replay.warnings),
    }
    if isinstance(replay, ClusterReplay):
        out["cluster_id"] = int(replay.cluster_id)
        out["haplotype_id"] = int(replay.haplotype_id)
        out["node_template_id"] = int(replay.node_template_id)
    if tmpl is not None:
        res = tmpl.result
        hap = np.asarray(res.haplotype_of_read, dtype=np.int64)
        ids, counts = np.unique(hap, return_counts=True) if hap.size else ([], [])
        out.update(
            template_id=int(tmpl.template_id),
            params_source=tmpl.params_source,
            n_reads=int(hap.size),
            n_sampled=int(np.asarray(res.member_pos).size),
            nodes=[
                {"haplotype_id": int(i), "n_reads": int(c)} for i, c in zip(ids, counts)
            ],
            status=getattr(res.record, "status", None),
        )
        absorbed = (
            replay.absorbed_read_row
            if isinstance(replay, ClusterReplay) and replay.kind == "merged"
            else None
        )
        out["n_absorbed_reads"] = int(absorbed.size) if absorbed is not None else 0
    elif isinstance(replay, ClusterReplay):  # rebuilt
        pool = np.asarray(replay.pool_read_row, dtype=np.int64)
        out.update(
            template_id=None,
            n_reads=int(pool.size),
            n_sampled=int(np.asarray(replay.member_read_row).size),
            nodes=[{"haplotype_id": int(replay.haplotype_id), "n_reads": int(pool.size)}],
            n_absorbed_reads=int(
                np.sum(
                    np.asarray(replay.pool_node_template_id, dtype=np.int64)
                    != int(replay.node_template_id)
                )
            )
            if replay.pool_node_template_id is not None
            else 0,
        )
    return out


def node_differences(replay) -> pa.Table:
    """The frame columns where a template's nodes' consensus sequences
    differ, one row per node covering such a column — what a comparison of
    two siblings is restricted to. Empty for a single node or a rebuilt
    survivor (whose replay has one)."""
    tmpl = _template_of(replay)
    internals = tmpl.result.internals if tmpl is not None else None
    if internals is None or len(internals.children) < 2:
        return DIFFERENCES_SCHEMA.empty_table()
    n = int(internals.pooled.plan.n_columns)
    per_node = []
    for hid, child, lo, hi in internals.children:
        winner = np.asarray(child.winner, dtype=np.int64)
        cons = np.asarray(child.cons_of_frame, dtype=np.int64)
        carried = (cons >= lo) & (cons < hi)
        allele = np.full(n, -1, dtype=np.int64)
        if carried.any():
            first, last = np.flatnonzero(carried)[[0, -1]]
            allele[first : last + 1] = 4  # inside its span: a gap unless carried
            allele[carried] = winner[carried]
        per_node.append((int(hid), allele))
    stack = np.stack([a for _h, a in per_node])
    covered = stack >= 0
    lo_val = np.where(covered, stack, 99).min(axis=0)
    hi_val = np.where(covered, stack, -1).max(axis=0)
    cols = np.flatnonzero((covered.sum(axis=0) >= 2) & (lo_val != hi_val))
    rows = [
        (int(c), hid, int(a[c])) for c in cols for hid, a in per_node if a[c] >= 0
    ]
    return pa.table(
        {
            "frame_col": pa.array([r[0] for r in rows], pa.int32()),
            "haplotype_id": pa.array([r[1] for r in rows], pa.int32()),
            "allele": pa.array([r[2] for r in rows], pa.int8()),
        },
        schema=DIFFERENCES_SCHEMA,
    )


__all__ = [
    "ARCS_SCHEMA",
    "COLUMNS_SCHEMA",
    "DEPARTURES_SCHEMA",
    "DIFFERENCES_SCHEMA",
    "INSERTIONS_SCHEMA",
    "READS_SCHEMA",
    "SPACES",
    "SupportTables",
    "node_differences",
    "replay_summary",
    "support_tables",
]
