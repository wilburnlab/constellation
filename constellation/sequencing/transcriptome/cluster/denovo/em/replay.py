"""Re-create one template's M-step — or one final cluster — from a run directory.

A run writes the M-step's compact evidence (:mod:`.evidence`); the dense
evidence behind one template — the PWM, every read's allele at every column,
the coordinate maps from template to frame to consensus — would be tens of GB
a round at full scale and is never written. It is re-created on demand here,
by running the M-step for that one template again, through the SAME code the
round ran (:func:`~.mstep_pool.refine_one_template`), on the inputs the round
left on disk: its templates, its assignment slice, the corpus and its
parameters (``rNN/params.json``; the run's ``manifest.json`` for runs from
before that record).

A replay is only worth showing if it IS the run, so it is checked against
what the round stored — every node row, every read's node, and the
template's record where there is one — and says so (``verified``,
``mismatch``). A run made with code whose M-step has since changed, or under
another numpy (whose ``Generator.choice`` draws the member subsample), fails
the check rather than being shown as if it were the run.

A final cluster is a round node (replayed as above), or, if the final merge
absorbed others into it, a survivor rebuilt from the pooled reads
(:mod:`.rebuild`): that is replayed through the rebuild's own code and checked
against ``clusters.parquet``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

#: Assignment columns a replay reads: the worker's, and the E-step's own
#: account of each read, which the support view shows beside it.
WORKER_COLUMNS = ("template_row", "read_row", "cigar", "t_start", "q_start", "weight")
DISPLAY_COLUMNS = (
    "read_id",
    "sample_id",
    "identity",
    "aligned_len",
    "as_score",
    "logl",
    "logl_delta",
    "runner_up_template_id",
    "runner_up_identity",
    "runner_up_logl_gap",
)


class ReplayError(ValueError):
    """The run directory cannot say what a replay needs."""


# ── where a template's assignment rows are ────────────────────────────


@dataclass(frozen=True)
class AssignmentLocator:
    """Where each template's rows are in a round's assignment shards.

    The shards are written by READ block, so every shard spans every
    template and a filtered scan cannot skip one; this reads the narrow
    ``template_row`` column once (~45 MB a round at 9.4M reads) and keeps,
    for each template, its rows' global positions in the order the M-step
    met them: the sorted shard glob concatenated, then a STABLE sort on
    ``template_row`` (``sort_assignments_by_template``). Cache one per round.
    """

    files: tuple[Path, ...]
    #: (n_files + 1,) cumulative row counts.
    offsets: np.ndarray
    #: Global row positions of assigned rows, stable-sorted by template row.
    order: np.ndarray
    #: (n_template_rows + 1,) CSR starts into ``order``.
    starts: np.ndarray

    def rows_of(self, template_row: int) -> np.ndarray:
        r = int(template_row)
        if r < 0 or r + 1 >= self.starts.size:
            return np.empty(0, dtype=np.int64)
        return self.order[self.starts[r] : self.starts[r + 1]]


def assignment_locator(round_dir: Path) -> AssignmentLocator:
    """Build the :class:`AssignmentLocator` of one round."""
    files = tuple(sorted((Path(round_dir) / "assignments").glob("part-*.parquet")))
    if not files:
        raise ReplayError(f"{Path(round_dir).name}: no assignments to replay from")
    parts = [
        pq.read_table(f, columns=["template_row"]).column(0).to_numpy(
            zero_copy_only=False
        )
        for f in files
    ]
    sizes = np.array([p.size for p in parts], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    tr = np.concatenate(parts).astype(np.int64) if parts else np.empty(0, np.int64)
    assigned = np.flatnonzero(tr >= 0)
    order = assigned[np.argsort(tr[assigned], kind="stable")]
    n_rows = int(tr[assigned].max()) + 1 if assigned.size else 0
    counts = np.bincount(tr[assigned], minlength=n_rows)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return AssignmentLocator(files=files, offsets=offsets, order=order, starts=starts)


def _read_rows(loc: AssignmentLocator, global_rows: np.ndarray) -> pa.Table:
    """The assignment rows at ``global_rows`` (ascending), worker columns and
    whatever display columns the shards carry."""
    global_rows = np.asarray(global_rows, dtype=np.int64)
    if global_rows.size and np.any(np.diff(global_rows) < 0):
        raise ReplayError("a template's rows are not in shard order")
    file_of = np.searchsorted(loc.offsets, global_rows, side="right") - 1
    pieces = []
    for f in np.unique(file_of).tolist():
        path = loc.files[f]
        names = set(pq.read_schema(path).names)
        cols = [c for c in (*WORKER_COLUMNS, *DISPLAY_COLUMNS) if c in names]
        local = global_rows[file_of == f] - loc.offsets[f]
        pieces.append(pq.read_table(path, columns=cols).take(pa.array(local)))
    if not pieces:
        return pa.table({c: pa.array([], pa.int32()) for c in WORKER_COLUMNS})
    return pa.concat_tables(pieces, promote_options="default").combine_chunks()


# ── what a round ran under ────────────────────────────────────────────


def round_params(run_dir: Path, round_index: int):
    """``(MStepParams, EmParams dict, source)`` of a round.

    From ``rNN/params.json`` (written when the round started), else from the
    run's ``manifest.json`` (written when the run finished, with the last
    invocation's values). ``ReplayError`` when neither exists.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )

    run_dir = Path(run_dir)
    rd = run_dir / "rounds" / f"r{int(round_index):02d}"
    em, source = None, None
    if (rd / "params.json").exists():
        em, source = json.loads((rd / "params.json").read_text())["params"], "round"
    elif (run_dir / "manifest.json").exists():
        em = json.loads((run_dir / "manifest.json").read_text()).get("parameters")
        source = "manifest"
    if not em or "mstep" not in em:
        raise ReplayError(
            f"{rd.name}: no params.json and no manifest.json parameters — pass "
            f"params= explicitly"
        )
    return _mstep_params(em["mstep"], MStepParams), em, source


def _mstep_params(d: dict, cls):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ReplayError(
            f"the run's M-step parameters name fields this code does not have "
            f"({unknown}): it was made by a different M-step"
        )
    return cls(**{k: v for k, v in d.items() if k in known})


# ── one template ──────────────────────────────────────────────────────


@dataclass
class TemplateReplay:
    """One template's M-step, re-created and checked against the run."""

    run_dir: Path
    round: int
    template_id: int
    template_row: int
    frame: str
    params_source: str
    #: :class:`~.mstep_pool.TemplateResult` with ``internals`` (the PWM etc.)
    result: object
    #: The template's assignment rows (worker + display columns), slice order.
    rows: pa.Table
    read_ids: pa.Array
    dorado_quality: np.ndarray
    verified: bool = False
    mismatch: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def replay_template(
    run_dir: Path,
    round_index: int,
    template_id: int,
    *,
    params=None,
    locator: AssignmentLocator | None = None,
    strict: bool = False,
) -> TemplateReplay:
    """Re-run the M-step of ``template_id`` in round ``round_index``.

    ``params`` (an ``MStepParams``) overrides what the run recorded;
    ``locator`` is the round's :func:`assignment_locator`, to reuse across
    replays of one round. ``strict`` raises when the replay does not match
    what the round stored instead of returning ``verified=False``.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
        ReadStore,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        refine_one_template,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
        TemplateStore,
    )

    run_dir = Path(run_dir)
    rd = run_dir / "rounds" / f"r{int(round_index):02d}"
    if not (rd / "_SUCCESS").exists():
        raise ReplayError(f"{rd.name} did not finish; nothing it stored to check")
    warnings: list[str] = []
    if params is None:
        params, _em, source = round_params(run_dir, round_index)
    else:
        source = "explicit"
    store = TemplateStore.open(rd / "templates" / "templates.arrow")
    reads = ReadStore.open(run_dir / "corpus" / "reads.arrow")
    try:
        hit = np.flatnonzero(np.asarray(store.template_id) == int(template_id))
        if hit.size == 0:
            raise ReplayError(f"{rd.name} has no template {template_id}")
        row = int(hit[0])
        loc = locator or assignment_locator(rd)
        rows = _read_rows(loc, loc.rows_of(row))
        if rows.num_rows == 0:
            raise ReplayError(
                f"template {template_id} recruited no read in {rd.name}: the "
                f"M-step never ran on it"
            )
        rr = rows.column("read_row").to_numpy(zero_copy_only=False)
        if rr.size > 1 and np.any(np.diff(rr) < 0):
            warnings.append("members are not in ascending read_row order")
        result = refine_one_template(
            rows,
            reads,
            store,
            row,
            params,
            round_index=int(round_index),
            evidence="full",
            on_error="raise",
        )
        frame = store.sequence(row)
        read_rows = result.read_row
        read_ids = reads.take_read_ids(read_rows)
        quality = (
            np.asarray(reads.dorado_quality, dtype=np.float64)[read_rows]
            if np.asarray(reads.dorado_quality).size
            else np.full(read_rows.size, np.nan)
        )
    finally:
        store.close()
        reads.close()
    warnings += _library_warnings(rd)
    out = TemplateReplay(
        run_dir=run_dir,
        round=int(round_index),
        template_id=int(template_id),
        template_row=row,
        frame=frame,
        params_source=source,
        result=result,
        rows=rows,
        read_ids=read_ids,
        dorado_quality=quality,
        warnings=warnings,
    )
    out.mismatch = _verify_template(rd, out)
    out.verified = not out.mismatch
    if strict and out.mismatch:
        raise ReplayError(
            f"replay of template {template_id} in {rd.name} does not match the "
            f"run: {out.mismatch[:3]}"
        )
    return out


def _library_warnings(rd: Path) -> list[str]:
    path = rd / "params.json"
    if not path.exists():
        return ["no params.json: library versions of the run are unknown"]
    from importlib import metadata

    recorded = json.loads(path.read_text()).get("libraries", {})
    out = []
    for name, version in recorded.items():
        try:
            now = metadata.version(name)
        except metadata.PackageNotFoundError:
            now = None
        if version and now and version != now:
            out.append(f"{name} {now} here, {version} in the run")
    return out


def _verify_template(rd: Path, rep: TemplateReplay) -> list[str]:
    """What differs between the replay and what the round stored."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        _rows_to_nodes,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        read_round_membership,
        read_round_nodes,
        read_round_templates,
    )

    out: list[str] = []
    tid = rep.template_id
    stored = read_round_nodes(rd)
    stored = stored.filter(pc.equal(stored.column("parent_template_id"), tid))
    mine = _rows_to_nodes(rep.result.node_rows)
    s_rows = sorted(stored.to_pylist(), key=lambda r: r["haplotype_id"])
    m_rows = sorted(mine.to_pylist(), key=lambda r: r["haplotype_id"])
    if len(s_rows) != len(m_rows):
        out.append(f"{len(m_rows)} nodes replayed, {len(s_rows)} stored")
    for s, m in zip(s_rows, m_rows):
        for name in s:
            if s[name] != m.get(name):
                out.append(f"node {s['haplotype_id']}: {name} differs")
    mem = read_round_membership(rd)
    mem = mem.filter(pc.equal(mem.column("parent_template_id"), tid))
    stored_hap = dict(
        zip(
            mem.column("read_row").to_pylist(),
            mem.column("haplotype_id").to_pylist(),
        )
    )
    mine_hap = dict(
        zip(rep.result.read_row.tolist(), rep.result.haplotype_of_read.tolist())
    )
    if stored_hap != mine_hap:
        moved = sum(stored_hap.get(k) != v for k, v in mine_hap.items())
        out.append(f"{moved} read(s) on a different node than stored")
    tmpl = read_round_templates(rd)
    if tmpl is not None:
        rec = tmpl.filter(pc.equal(tmpl.column("template_id"), tid)).to_pylist()
        if rec:
            rec = rec[0]
            got = rep.result.record
            for name in ("status", "n_assigned", "n_members_used", "n_nodes_emitted"):
                if rec[name] != getattr(got, name):
                    out.append(f"record {name} differs")
            for name, value in got.counters.items():
                if rec.get(name) != value:
                    out.append(f"record {name} differs")
    return out


# ── one final cluster ─────────────────────────────────────────────────


def cluster_node_map(run_dir: Path) -> pa.Table:
    """``cluster_nodes.parquet``, or for a run from before it, the same table
    derived from the last round: its nodes in node order, minus what the
    final merge absorbed (``merged_final.parquet``), in order. A derived
    table's ``rebuilt`` flag is "absorbed something", which a rebuild that
    placed no read would have left un-rebuilt; replay's check tells."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.outputs import (
        cluster_node_table,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        read_round_nodes,
    )

    run_dir = Path(run_dir)
    path = run_dir / "cluster_nodes.parquet"
    if path.exists():
        return pq.read_table(path)
    done = sorted(
        int(d.name[1:])
        for d in (run_dir / "rounds").glob("r*")
        if d.name[1:].isdigit() and (d / "_SUCCESS").exists()
    )
    if not done:
        raise ReplayError(f"{run_dir}: no finished round")
    r = done[-1]
    rd = run_dir / "rounds" / f"r{r:02d}"
    nodes = read_round_nodes(rd)
    merged_path = rd / "merged_final.parquet"
    merged = pq.read_table(merged_path) if merged_path.exists() else None
    absorbed_rows = np.empty(0, dtype=np.int64)
    if merged is not None and merged.num_rows:
        mask = (1 << 40) - 1
        absorbed_rows = (
            merged.column("absorbed_template_id").to_numpy(zero_copy_only=False) & mask
        )
    keep = np.setdiff1d(np.arange(nodes.num_rows), absorbed_rows)
    table = cluster_node_table(
        nodes.take(pa.array(keep)), round_index=r, keep_rows=keep, merged=merged
    )
    rebuilt = pc.greater(table.column("n_absorbed"), 0)
    return table.set_column(
        table.schema.get_field_index("rebuilt"), "rebuilt", rebuilt
    )


@dataclass
class ClusterReplay:
    """One final cluster's evidence, re-created and checked."""

    run_dir: Path
    cluster_id: int
    #: ``"node"`` (a round node; ``template`` holds it) or ``"rebuilt"``.
    kind: str
    round: int
    haplotype_id: int
    template: TemplateReplay | None = None
    #: Rebuilt survivors: the consensus the reads were aligned to (the
    #: survivor node's), the pooled ConsensusResult, the members and their
    #: corpus rows, and every pooled read row.
    frame: str = ""
    pooled: object | None = None
    member_read_row: np.ndarray | None = None
    pool_read_row: np.ndarray | None = None
    read_ids: pa.Array | None = None
    dorado_quality: np.ndarray | None = None
    verified: bool = False
    mismatch: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def replay_cluster(
    run_dir: Path,
    cluster_id: int,
    *,
    params=None,
    locator: AssignmentLocator | None = None,
    strict: bool = False,
) -> ClusterReplay:
    """Re-create the evidence behind final cluster ``cluster_id``."""
    run_dir = Path(run_dir)
    cmap = cluster_node_map(run_dir)
    hit = cmap.filter(pc.equal(cmap.column("cluster_id"), int(cluster_id))).to_pylist()
    if not hit:
        raise ReplayError(f"no cluster {cluster_id}")
    row = hit[0]
    clusters = pq.read_table(run_dir / "clusters.parquet")
    want = clusters.filter(
        pc.equal(clusters.column("cluster_id"), int(cluster_id))
    ).to_pylist()[0]["consensus_sequence"]
    if not row["rebuilt"]:
        rep = replay_template(
            run_dir,
            row["round"],
            row["parent_template_id"],
            params=params,
            locator=locator,
        )
        node = [n for n in rep.result.nodes if n.haplotype_id == row["haplotype_id"]]
        mismatch = list(rep.mismatch)
        if not node:
            mismatch.append(f"no node {row['haplotype_id']} replayed")
        elif node[0].consensus != want:
            mismatch.append("the node's consensus is not the cluster's")
        out = ClusterReplay(
            run_dir=run_dir,
            cluster_id=int(cluster_id),
            kind="node",
            round=int(row["round"]),
            haplotype_id=int(row["haplotype_id"]),
            template=rep,
            mismatch=mismatch,
            warnings=list(rep.warnings),
        )
    else:
        out = _replay_rebuilt(run_dir, row, want, params)
    out.verified = not out.mismatch
    if strict and out.mismatch:
        raise ReplayError(f"replay of cluster {cluster_id}: {out.mismatch[:3]}")
    return out


def _replay_rebuilt(run_dir: Path, row: dict, want: str, params) -> ClusterReplay:
    """A final-merge survivor: its pooled reads, aligned to its node's
    consensus and re-consensused, through the rebuild's own code."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
        ReadStore,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep import (
        pooled_node_with_consensus,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.rebuild import (
        rebuild_members,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        read_round_membership,
        read_round_nodes,
    )

    r = int(row["round"])
    rd = run_dir / "rounds" / f"r{r:02d}"
    mparams, em, source = round_params(run_dir, r)
    if params is not None:
        mparams, source = params, "explicit"
    nodes = read_round_nodes(rd)
    mask = (1 << 40) - 1
    survivor_row = int(row["node_template_id"]) & mask
    consensus = nodes.column("consensus")[survivor_row].as_py()
    group_rows = [survivor_row] + [int(i) & mask for i in row["absorbed_node_template_ids"]]
    keys = {
        (
            nodes.column("parent_template_id")[g].as_py(),
            nodes.column("haplotype_id")[g].as_py(),
        )
        for g in group_rows
    }
    mem = read_round_membership(rd)
    pid = mem.column("parent_template_id").to_pylist()
    hap = mem.column("haplotype_id").to_pylist()
    sel = np.array([(p, h) in keys for p, h in zip(pid, hap)], dtype=bool)
    # Membership order — (parent, haplotype, read_row) — is the order the
    # rebuild pooled them in (rebuild._read_rows_of after the merge re-keyed
    # the absorbed rows onto the survivor, stable-sorted).
    pool = mem.column("read_row").to_numpy(zero_copy_only=False)[sel].astype(np.int64)
    em = em or {}
    members, sampled_rows, _skipped, _fraction = rebuild_members(
        consensus,
        pool,
        template_id=int(nodes.column("parent_template_id")[survivor_row].as_py()),
        corpus_path=str(run_dir / "corpus" / "reads.arrow"),
        params=mparams,
        identity_floor=float(em.get("p_floor", 0.97)),
        quality_scale=em.get("p_floor_quality_scale"),
    )
    node, cres = pooled_node_with_consensus(
        consensus,
        members,
        template_id=int(nodes.column("parent_template_id")[survivor_row].as_py()),
        min_aa_length=mparams.min_aa_length,
        fold_insertions=mparams.fold_insertions,
        min_insertion_support=mparams.min_insertion_support,
        min_extension_support=mparams.min_extension_support,
    )
    mismatch = []
    if node is None:
        mismatch.append("the rebuild placed no read")
    elif node.consensus != want:
        mismatch.append("the rebuilt consensus is not the cluster's")
    member_rows = np.asarray(
        [sampled_rows[m.member_id] for m in members], dtype=np.int64
    )
    reads = ReadStore.open(run_dir / "corpus" / "reads.arrow")
    try:
        read_ids = reads.take_read_ids(pool)
        quality = (
            np.asarray(reads.dorado_quality, dtype=np.float64)[pool]
            if np.asarray(reads.dorado_quality).size
            else np.full(pool.size, np.nan)
        )
    finally:
        reads.close()
    warnings = _library_warnings(rd)
    if source != "round":
        warnings.append(f"parameters from the {source}")
    return ClusterReplay(
        run_dir=run_dir,
        cluster_id=int(row["cluster_id"]),
        kind="rebuilt",
        round=r,
        haplotype_id=int(row["haplotype_id"]),
        frame=consensus,
        pooled=cres,
        member_read_row=member_rows,
        pool_read_row=pool,
        read_ids=read_ids,
        dorado_quality=quality,
        mismatch=mismatch,
        warnings=warnings,
    )


__all__ = [
    "AssignmentLocator",
    "ClusterReplay",
    "ReplayError",
    "TemplateReplay",
    "assignment_locator",
    "cluster_node_map",
    "replay_cluster",
    "replay_template",
    "round_params",
]
