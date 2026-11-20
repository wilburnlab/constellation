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
(:mod:`.rebuild`): that is replayed through the rebuild's own code, under the
final stage's parameters (``final_params.json``), and checked against the
cluster's consensus. A survivor whose rebuild failed keeps its node's
consensus and is replayed as that node, with the absorbed reads listed.
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from collections import OrderedDict
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


# ── where a round's M-step rows are ───────────────────────────────────


@dataclass(frozen=True)
class RoundIndex:
    """Where everything a round stored about one template is, built once.

    A replay used to read the round's whole node table (at 9.4M reads, 1.67M
    rows and ~2.7 GB of consensus), every membership row and every
    assignment shard to check one template: ~15 s a call on OSC even for a
    single-read cluster, 28–44 s from ext4 here. But the M-step writes each
    unit's nodes, membership, record and compact evidence as shards of ONE
    index, so a template's rows all sit in the shards of the unit that
    refined it. This keeps, for every node in node order (``(parent_
    template_id, haplotype_id)``, the order next-round template ids and
    ``merged_final``'s ids are assigned in), its unit and its row there —
    two integer columns read once — and the assignment locator.
    """

    round_dir: Path
    assignments: AssignmentLocator
    #: Each M-step table's shards, by key; None where the round predates it.
    shards: dict
    #: Every node, in node order: its key, its unit (shard index), its row.
    node_parent: np.ndarray
    node_hap: np.ndarray
    node_unit: np.ndarray
    node_local: np.ndarray

    def node_rows_of(self, template_id: int) -> np.ndarray:
        """Node-order rows of ``template_id``'s nodes (contiguous)."""
        lo = int(np.searchsorted(self.node_parent, int(template_id), side="left"))
        hi = int(np.searchsorted(self.node_parent, int(template_id), side="right"))
        return np.arange(lo, hi, dtype=np.int64)

    def unit_table(self, key: str, unit: int) -> pa.Table | None:
        paths = self.shards.get(key)
        if paths is None:
            return None
        return pq.read_table(paths[int(unit)])

    def nodes_at(self, rows) -> pa.Table:
        """The stored node rows at node-order positions ``rows``, in order."""
        rows = np.asarray(rows, dtype=np.int64)
        position = np.arange(rows.size, dtype=np.int64)
        pieces = []
        for unit in np.unique(self.node_unit[rows]).tolist():
            here = self.node_unit[rows] == unit
            pieces.append(
                (
                    position[here],
                    self.unit_table("nodes", unit).take(
                        pa.array(self.node_local[rows[here]])
                    ),
                )
            )
        if not pieces:
            return self.unit_table("nodes", 0).slice(0, 0)
        # Back to the order asked for — by position in ``rows``, never by the
        # row numbers themselves: a rebuilt survivor's group is [survivor,
        # absorbed...], and an absorbed node of a lower row must not take
        # the survivor's place at the head of it.
        order = np.argsort(np.concatenate([p for p, _ in pieces]), kind="stable")
        table = pa.concat_tables([t for _, t in pieces])
        return table.take(pa.array(order))


_INDEX_CACHE: "OrderedDict[tuple, RoundIndex]" = OrderedDict()
_INDEX_LOCK = threading.Lock()
_INDEX_CACHE_SIZE = 2


def index_round(round_dir: Path) -> RoundIndex:
    """The :class:`RoundIndex` of a finished round, cached (two rounds).

    Keyed on the round's ``round.json`` (rewritten whenever the round is), so
    a round re-run in place is indexed afresh.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        MEMBERSHIP_KEY,
        MSTEP_DIR,
        NODES_KEY,
        StaleShardsError,
        round_mstep_shards,
        round_record,
    )

    rd = Path(round_dir)
    stamp = rd / "round.json"
    key = (str(rd.resolve()), stamp.stat().st_mtime_ns if stamp.exists() else 0)
    with _INDEX_LOCK:
        if key in _INDEX_CACHE:
            _INDEX_CACHE.move_to_end(key)
            return _INDEX_CACHE[key]
    shards: dict = {}
    for k in (NODES_KEY, MEMBERSHIP_KEY, "templates", "disagreements", "columns", "covariance", "reads"):
        if (rd / MSTEP_DIR / k).exists():
            shards[k] = round_mstep_shards(rd, k)
        else:
            shards[k] = None
    if not shards.get(NODES_KEY):
        raise ReplayError(f"{rd.name}: no M-step nodes to index")
    parents, haps, units, locals_ = [], [], [], []
    for u, path in enumerate(shards[NODES_KEY]):
        t = pq.read_table(path, columns=["parent_template_id", "haplotype_id"])
        n = t.num_rows
        parents.append(t.column(0).to_numpy(zero_copy_only=False).astype(np.int64))
        haps.append(t.column(1).to_numpy(zero_copy_only=False).astype(np.int64))
        units.append(np.full(n, u, dtype=np.int32))
        locals_.append(np.arange(n, dtype=np.int64))
    parent = np.concatenate(parents)
    hap = np.concatenate(haps)
    order = np.lexsort((hap, parent))
    parent, hap = parent[order], hap[order]
    if parent.size > 1 and np.any((np.diff(parent) == 0) & (np.diff(hap) == 0)):
        raise StaleShardsError(f"{rd.name}: duplicate (parent, haplotype) nodes")
    want = round_record(rd).get("n_nodes")
    if want is not None and parent.size != int(want):
        raise StaleShardsError(
            f"{rd.name}: {parent.size} node rows on disk, round.json records {want}"
        )
    index = RoundIndex(
        round_dir=rd,
        assignments=assignment_locator(rd),
        shards=shards,
        node_parent=parent,
        node_hap=hap,
        node_unit=np.concatenate(units)[order],
        node_local=np.concatenate(locals_)[order],
    )
    with _INDEX_LOCK:
        _INDEX_CACHE[key] = index
        _INDEX_CACHE.move_to_end(key)
        while len(_INDEX_CACHE) > _INDEX_CACHE_SIZE:
            _INDEX_CACHE.popitem(last=False)
    return index


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


def final_params(run_dir: Path, round_index: int):
    """``(MStepParams, EmParams dict, source)`` the final stage ran under.

    From the last round's ``final_params.json`` (written when the final stage
    starts; ``source == "final"``). ``--resume --rounds 0`` finalizes under
    the invoking command's parameters, so the round's own can be the wrong
    ones for a rebuilt survivor; a run from before the record falls back to
    them, and the source says so.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        FINAL_PARAMS,
    )

    path = Path(run_dir) / "rounds" / f"r{int(round_index):02d}" / FINAL_PARAMS
    if path.exists():
        em = json.loads(path.read_text())["params"]
        return _mstep_params(em["mstep"], MStepParams), em, "final"
    return round_params(run_dir, round_index)


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
    index: RoundIndex | None = None,
    strict: bool = False,
) -> TemplateReplay:
    """Re-run the M-step of ``template_id`` in round ``round_index``.

    ``params`` (an ``MStepParams``) overrides what the run recorded. The
    round's :class:`RoundIndex` is built on first use and cached; ``index``
    (or, for the assignment rows alone, ``locator``) may be passed instead.
    ``strict`` raises when the replay does not match what the round stored
    instead of returning ``verified=False``.
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
    index = index or index_round(rd)
    store = TemplateStore.open(rd / "templates" / "templates.arrow")
    reads = ReadStore.open(run_dir / "corpus" / "reads.arrow")
    try:
        hit = np.flatnonzero(np.asarray(store.template_id) == int(template_id))
        if hit.size == 0:
            raise ReplayError(f"{rd.name} has no template {template_id}")
        row = int(hit[0])
        loc = locator or index.assignments
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
    out.mismatch = _verify_template(index, out)
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


#: Relative tolerance for a floating-point value a replay recomputes. Nodes,
#: membership, counts and strings are compared exactly; a p-value or φ
#: computed on another machine or library build can differ in its last bits
#: (on the Nov2026 run, OSC against the workstation: 4.6e-13 relative) with no
#: decision differing — and a decision that differed would show in the nodes
#: or the membership, which are exact.
FLOAT_RTOL = 1e-9


def _same(a, b, drift: list[float]) -> bool:
    """Equality for values read back from Parquet: NaN equals NaN, and floats
    within :data:`FLOAT_RTOL` (the largest relative difference accepted is
    kept in ``drift[0]``)."""
    if isinstance(a, float) and isinstance(b, float):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return True
        if not (math.isfinite(a) and math.isfinite(b)):
            return False
        rel = abs(a - b) / max(abs(a), abs(b))
        if rel <= FLOAT_RTOL or abs(a - b) <= 1e-300:
            drift[0] = max(drift[0], rel)
            return True
        return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, drift) for x, y in zip(a, b))
    return a == b


def _rows_differ(
    stored: pa.Table, mine: pa.Table, sort: list[str], drift: list[float]
) -> str | None:
    """Why two tables of one template's rows differ, or None."""
    if stored.num_rows != mine.num_rows:
        return f"{mine.num_rows} rows replayed, {stored.num_rows} stored"
    keys = [(c, "ascending") for c in sort]
    s_rows = stored.sort_by(keys).to_pylist() if stored.num_rows else []
    m_rows = mine.sort_by(keys).to_pylist() if mine.num_rows else []
    for s_row, m_row in zip(s_rows, m_rows):
        for name, value in s_row.items():
            if name in m_row and not _same(value, m_row[name], drift):
                return f"{name} differs"
    return None


def _verify_template(index: RoundIndex, rep: TemplateReplay) -> list[str]:
    """What differs between the replay and what the round stored: every node
    row, every read's node, the template's record and its compact evidence —
    each read from the one unit's shards that hold this template."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        _rows_to_nodes,
    )

    out: list[str] = []
    drift = [0.0]
    tid = rep.template_id
    rows = index.node_rows_of(tid)
    stored = index.nodes_at(rows)
    mine = _rows_to_nodes(rep.result.node_rows)
    why = _rows_differ(
        stored.select(mine.column_names), mine, ["haplotype_id"], drift
    )
    if why:
        out.append(f"nodes: {why}")
    units = np.unique(index.node_unit[rows]).tolist() if rows.size else []
    if len(units) != 1:
        out.append(f"template {tid}'s nodes are in {len(units)} units, not one")
        return out
    unit = units[0]
    mem = index.unit_table("node_membership", unit)
    mem = mem.filter(pc.equal(mem.column("parent_template_id"), tid))
    stored_hap = dict(
        zip(mem.column("read_row").to_pylist(), mem.column("haplotype_id").to_pylist())
    )
    mine_hap = dict(
        zip(rep.result.read_row.tolist(), rep.result.haplotype_of_read.tolist())
    )
    if stored_hap != mine_hap:
        moved = sum(stored_hap.get(k) != v for k, v in mine_hap.items())
        out.append(f"{moved} read(s) on a different node than stored")
    tmpl = index.unit_table("templates", unit)
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
    # The compact evidence the round wrote for this template — what the
    # browser's arcs and column shading come from — must be what the replay
    # computes, not only the nodes it led to.
    sorts = {
        "columns": ["frame_col"],
        "covariance": ["u_frame_col", "v_frame_col"],
        "reads": ["read_row"],
    }
    evidence = rep.result.evidence or {}
    for key, sort in sorts.items():
        table = index.unit_table(key, unit)
        if table is None or key not in evidence:
            continue
        stored_ev = table.filter(pc.equal(table.column("template_id"), tid))
        why = _rows_differ(stored_ev, evidence[key], sort, drift)
        if why:
            out.append(f"{key}: {why}")
    if drift[0] > 0:
        rep.warnings.append(
            f"floating-point values agree to {drift[0]:.1e} relative, not bit "
            "for bit (another machine or library build)"
        )
    return out


# ── one final cluster ─────────────────────────────────────────────────


_TABLE_CACHE: "OrderedDict[tuple, pa.Table]" = OrderedDict()


def _read_cached(path: Path) -> pa.Table:
    """A small whole-file table, kept while the file is unchanged (two)."""
    st = Path(path).stat()
    key = (str(Path(path).resolve()), st.st_mtime_ns, st.st_size)
    with _INDEX_LOCK:
        if key in _TABLE_CACHE:
            _TABLE_CACHE.move_to_end(key)
            return _TABLE_CACHE[key]
    table = pq.read_table(path)
    with _INDEX_LOCK:
        _TABLE_CACHE[key] = table
        while len(_TABLE_CACHE) > 2:
            _TABLE_CACHE.popitem(last=False)
    return table


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
        return _read_cached(path)
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
    #: ``"node"`` (a round node; ``template`` holds it), ``"rebuilt"`` (a
    #: final-merge survivor re-consensused from its group's reads), or
    #: ``"merged"`` (a survivor whose rebuild failed: its own consensus
    #: stands, ``template`` holds its node, and the reads the merge gave it
    #: are in ``absorbed_*`` — members of the cluster, never aligned to
    #: that node's frame).
    kind: str
    round: int
    haplotype_id: int
    template: TemplateReplay | None = None
    #: The cluster's node id (``cluster_nodes.parquet``'s
    #: ``node_template_id``): what ``absorbed_from`` is measured against.
    node_template_id: int = -1
    #: Rebuilt survivors: the consensus the reads were aligned to (the
    #: survivor node's), the pooled ConsensusResult, the members and their
    #: corpus rows, and every pooled read row.
    frame: str = ""
    pooled: object | None = None
    member_read_row: np.ndarray | None = None
    pool_read_row: np.ndarray | None = None
    #: Per pooled read: the node of the merge group it came from.
    pool_node_template_id: np.ndarray | None = None
    read_ids: pa.Array | None = None
    dorado_quality: np.ndarray | None = None
    #: ``"merged"``: the absorbed nodes' reads, the node each came from,
    #: their ids and qualities.
    absorbed_read_row: np.ndarray | None = None
    absorbed_node_template_id: np.ndarray | None = None
    absorbed_read_ids: pa.Array | None = None
    absorbed_quality: np.ndarray | None = None
    verified: bool = False
    mismatch: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_MASK = (1 << 40) - 1


def _group_reads(index: RoundIndex, node_ids: list[int]) -> pa.Table:
    """Every read of the merge-group nodes ``node_ids`` (graph ids; the low
    40 bits are the node-order row), with the node each came from, in
    membership order — ``(parent, haplotype, read_row)``, the order the
    rebuild pooled them in (``rebuild._read_rows_of`` after the merge re-keyed
    the absorbed rows onto the survivor, stable-sorted)."""
    parts = []
    for nid in node_ids:
        g = int(nid) & _MASK
        mem = index.unit_table("node_membership", int(index.node_unit[g]))
        sel = pc.and_(
            pc.equal(mem.column("parent_template_id"), int(index.node_parent[g])),
            pc.equal(mem.column("haplotype_id"), int(index.node_hap[g])),
        )
        t = mem.filter(sel).select(["parent_template_id", "haplotype_id", "read_row"])
        parts.append(
            t.append_column(
                "node_template_id", pa.array(np.full(t.num_rows, int(nid), np.int64))
            )
        )
    return pa.concat_tables(parts).sort_by(
        [
            ("parent_template_id", "ascending"),
            ("haplotype_id", "ascending"),
            ("read_row", "ascending"),
        ]
    )


def _read_ids_and_quality(run_dir: Path, rows: np.ndarray):
    from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
        ReadStore,
    )

    reads = ReadStore.open(Path(run_dir) / "corpus" / "reads.arrow")
    try:
        ids = reads.take_read_ids(rows)
        quality = (
            np.asarray(reads.dorado_quality, dtype=np.float64)[rows]
            if np.asarray(reads.dorado_quality).size
            else np.full(rows.size, np.nan)
        )
    finally:
        reads.close()
    return ids, quality


def replay_cluster(
    run_dir: Path,
    cluster_id: int,
    *,
    params=None,
    locator: AssignmentLocator | None = None,
    index: RoundIndex | None = None,
    strict: bool = False,
) -> ClusterReplay:
    """Re-create the evidence behind final cluster ``cluster_id``.

    The replayed consensus is checked against the cluster's: by the content
    id ``cluster_nodes.parquet`` records (no read of ``clusters.parquet``,
    whose consensus column is ~2.7 GB at 9.4M reads), or for a run from
    before that file, by the consensus itself.
    """
    run_dir = Path(run_dir)
    cmap = cluster_node_map(run_dir)
    hit = cmap.filter(pc.equal(cmap.column("cluster_id"), int(cluster_id))).to_pylist()
    if not hit:
        raise ReplayError(f"no cluster {cluster_id}")
    row = hit[0]
    is_ours = _cluster_check(run_dir, int(cluster_id), row)
    rd = run_dir / "rounds" / f"r{int(row['round']):02d}"
    index = index or index_round(rd)
    absorbed = [int(i) for i in (row.get("absorbed_node_template_ids") or [])]
    if not row["rebuilt"]:
        rep = replay_template(
            run_dir,
            row["round"],
            row["parent_template_id"],
            params=params,
            locator=locator,
            index=index,
        )
        node = [n for n in rep.result.nodes if n.haplotype_id == row["haplotype_id"]]
        mismatch = list(rep.mismatch)
        if not node:
            mismatch.append(f"no node {row['haplotype_id']} replayed")
        elif not is_ours(node[0].consensus):
            mismatch.append("the node's consensus is not the cluster's")
        out = ClusterReplay(
            run_dir=run_dir,
            cluster_id=int(cluster_id),
            kind="merged" if absorbed else "node",
            round=int(row["round"]),
            haplotype_id=int(row["haplotype_id"]),
            template=rep,
            node_template_id=int(row["node_template_id"]),
            mismatch=mismatch,
            warnings=list(rep.warnings),
        )
        if absorbed:
            # The final merge gave this survivor other nodes' reads and its
            # rebuild failed, so its own consensus stands — but the reads are
            # the cluster's members all the same (cluster_membership.parquet
            # holds them). Shown without them, a 34-read cluster displayed
            # its survivor's 31.
            group = _group_reads(index, absorbed)
            rows = group.column("read_row").to_numpy(zero_copy_only=False).astype(np.int64)
            out.absorbed_read_row = rows
            out.absorbed_node_template_id = (
                group.column("node_template_id").to_numpy(zero_copy_only=False)
            )
            out.absorbed_read_ids, out.absorbed_quality = _read_ids_and_quality(
                run_dir, rows
            )
            out.warnings.append(
                f"the final merge absorbed {len(absorbed)} node(s) ({rows.size} "
                "reads) into this cluster and did not rebuild its consensus; "
                "those reads are members, listed without an alignment to this "
                "node's frame"
            )
    else:
        out = _replay_rebuilt(run_dir, row, is_ours, params, index)
    out.verified = not out.mismatch
    if strict and out.mismatch:
        raise ReplayError(f"replay of cluster {cluster_id}: {out.mismatch[:3]}")
    return out


def _cluster_check(run_dir: Path, cluster_id: int, row: dict):
    """A predicate: is this consensus the cluster's?"""
    if (run_dir / "cluster_nodes.parquet").exists() and row.get(
        "consensus_content_id"
    ) is not None:
        from constellation.core.ids import content_id

        want_id = int(row["consensus_content_id"])
        return lambda consensus: content_id(consensus) == want_id
    table = pq.read_table(
        run_dir / "clusters.parquet",
        columns=["cluster_id", "consensus_sequence"],
        filters=[("cluster_id", "==", cluster_id)],
    )
    want = table.column("consensus_sequence")[0].as_py() if table.num_rows else None
    return lambda consensus: consensus == want


def _replay_rebuilt(run_dir: Path, row: dict, is_ours, params, index: RoundIndex):
    """A final-merge survivor: its pooled reads, aligned to its node's
    consensus and re-consensused, through the rebuild's own code — under the
    final stage's parameters, which ``--resume --rounds 0`` can have changed
    from the round's."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep import (
        pooled_node_with_consensus,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.rebuild import (
        rebuild_members,
    )

    r = int(row["round"])
    rd = run_dir / "rounds" / f"r{r:02d}"
    mparams, em, source = final_params(run_dir, r)
    if params is not None:
        mparams, source = params, "explicit"
    survivor_id = int(row["node_template_id"])
    node_ids = [survivor_id] + [int(i) for i in row["absorbed_node_template_ids"]]
    survivor = index.nodes_at([survivor_id & _MASK])
    consensus = survivor.column("consensus")[0].as_py()
    survivor_parent = int(survivor.column("parent_template_id")[0].as_py())
    pooled_rows = _group_reads(index, node_ids)
    pool = pooled_rows.column("read_row").to_numpy(zero_copy_only=False).astype(np.int64)
    em = em or {}
    members, sampled_rows, _skipped, _fraction = rebuild_members(
        consensus,
        pool,
        template_id=survivor_parent,
        corpus_path=str(run_dir / "corpus" / "reads.arrow"),
        params=mparams,
        identity_floor=float(em.get("p_floor", 0.97)),
        quality_scale=em.get("p_floor_quality_scale"),
    )
    node, cres = pooled_node_with_consensus(
        consensus,
        members,
        template_id=survivor_parent,
        min_aa_length=mparams.min_aa_length,
        fold_insertions=mparams.fold_insertions,
        min_insertion_support=mparams.min_insertion_support,
        min_extension_support=mparams.min_extension_support,
    )
    mismatch = []
    if node is None:
        mismatch.append("the rebuild placed no read")
    elif not is_ours(node.consensus):
        mismatch.append("the rebuilt consensus is not the cluster's")
    member_rows = np.asarray(
        [sampled_rows[m.member_id] for m in members], dtype=np.int64
    )
    read_ids, quality = _read_ids_and_quality(run_dir, pool)
    warnings = _library_warnings(rd)
    if source == "explicit":
        warnings.append("parameters from the caller")
    elif source != "final":
        warnings.append(
            f"the final stage's parameters were not recorded (a run from before "
            f"final_params.json); replayed under the {source}'s, which a "
            f"--resume --rounds 0 finalization may have changed"
        )
    return ClusterReplay(
        run_dir=run_dir,
        cluster_id=int(row["cluster_id"]),
        kind="rebuilt",
        round=r,
        haplotype_id=int(row["haplotype_id"]),
        node_template_id=survivor_id,
        frame=consensus,
        pooled=cres,
        member_read_row=member_rows,
        pool_read_row=pool,
        pool_node_template_id=pooled_rows.column("node_template_id").to_numpy(
            zero_copy_only=False
        ),
        read_ids=read_ids,
        dorado_quality=quality,
        mismatch=mismatch,
        warnings=warnings,
    )


__all__ = [
    "AssignmentLocator",
    "RoundIndex",
    "ClusterReplay",
    "ReplayError",
    "TemplateReplay",
    "assignment_locator",
    "cluster_node_map",
    "replay_cluster",
    "replay_template",
    "final_params",
    "index_round",
    "round_params",
]
