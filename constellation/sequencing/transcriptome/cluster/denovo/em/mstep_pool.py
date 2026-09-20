"""Fanning the M-step out over templates, without a copy-on-write trap.

``refine_template`` handles one template. Making that a round means answering
two questions the prototype got wrong in opposite directions.

**What crosses the fork boundary.** The prototype put a ``dict[read_id, str]``
of every read plus one dict per assignment into a module global and forked,
expecting copy-on-write to share them. Measured: ~16 GB *private* per worker
over a ~30 GB shared base, because CPython writes the refcount word of every
object a worker touches. Here nothing large crosses: a worker receives its
slice of the assignment table and **opens the corpus and the templates
itself** by mmap. That also makes the design correct under ``spawn``, rather
than silently depending on ``fork``.

**How templates are scheduled.** The prototype handed out static contiguous
template ranges, so one mega-template serialised the stage — effective
parallelism was ~2-3 workers of 8. Units are bin-packed largest-cost-first
instead (cost = members x template length, the ``frame_consensus`` term),
which measures at load imbalance 1.000 packing 4M templates into 832 units.

LPT cannot fix a single template that is the critical path on its own, so
``max_members_per_template`` subsamples by abundance above a cap. That is the
likeliest explanation for the prototype's 12.7 h round-1 M-step:
``_CONSENSUS_MAX_MEMBERS`` bounds the components path but nothing bounded
this one.

**No torch, ever, below this line.** ``consensus`` / ``columns`` /
``covariance`` are numpy-only by design because the parent has already spawned
torch's thread pool during minimizer extraction, and a torch op after
``fork()`` deadlocks on OpenMP. The failure is a silent hang, not an
exception.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import register_schema
from constellation.sequencing.transcriptome.cluster.denovo.consensus import MemberSpec


REFINED_NODE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("round", pa.int32(), nullable=False),
        pa.field("parent_template_id", pa.int64(), nullable=False),
        pa.field("parent_template_row", pa.int32(), nullable=False),
        pa.field("haplotype_id", pa.int32(), nullable=False),
        pa.field("consensus", pa.large_string(), nullable=False),
        pa.field("n_reads", pa.int64(), nullable=False),
        pa.field("node_weight", pa.float64(), nullable=False),
        pa.field("protein", pa.large_string(), nullable=True),
        pa.field("orf_start", pa.int32(), nullable=False),
        pa.field("orf_end", pa.int32(), nullable=False),
        pa.field("orf_certified_end", pa.int32(), nullable=False),
        pa.field("orf_truncated_by_support", pa.bool_(), nullable=False),
        pa.field("allele_string", pa.string(), nullable=True),
        pa.field("declared_variants", pa.list_(pa.int64()), nullable=False),
        pa.field("n_inserted_columns", pa.int32(), nullable=False),
        pa.field("n_extended_5p", pa.int32(), nullable=False),
        pa.field("n_extended_3p", pa.int32(), nullable=False),
        pa.field("n_trimmed_5p", pa.int32(), nullable=False),
        pa.field("n_trimmed_3p", pa.int32(), nullable=False),
        pa.field("n_members_used", pa.int32(), nullable=False),
        # <1.0 when max_members_per_template subsampled; node_weight is
        # scaled back up by its reciprocal so quant stays on the real mass.
        pa.field("subsample_fraction", pa.float32(), nullable=False),
    ],
    metadata={b"schema_name": b"EmRefinedNodeTable"},
)

register_schema("EmRefinedNodeTable", REFINED_NODE_TABLE)


#: Which reads a node holds, not merely how many. Keyed on
#: ``(parent_template_id, haplotype_id)`` because that pair is unique within a
#: round and is known at emission, where the child's future ``template_id``
#: is not — it is assigned by row once every shard is concatenated.
#:
#: Without this table the only recoverable mapping is read -> *parent*, and a
#: parent that split into several nodes collapses back into one cluster: every
#: one of its reads lands on whichever child happened to be last.
NODE_MEMBERSHIP_TABLE: pa.Schema = pa.schema(
    [
        pa.field("round", pa.int32(), nullable=False),
        pa.field("parent_template_id", pa.int64(), nullable=False),
        pa.field("haplotype_id", pa.int32(), nullable=False),
        pa.field("read_row", pa.int32(), nullable=False),
        pa.field("weight", pa.float32(), nullable=False),
    ],
    metadata={b"schema_name": b"EmNodeMembershipTable"},
)

register_schema("EmNodeMembershipTable", NODE_MEMBERSHIP_TABLE)


@dataclass(frozen=True, slots=True)
class MStepUnit:
    """A contiguous row range of the template-sorted assignment table."""

    unit_idx: int
    rows: np.ndarray  # template rows in this unit
    row_lo: np.ndarray  # per template, start in the sorted table
    row_hi: np.ndarray
    cost: float


@dataclass(frozen=True, slots=True)
class MStepParams:
    """Every annealing knob, explicitly. No ``**kwargs``.

    Forwarding ``**kwargs`` is how ``--consensus-max-passes`` survived as a
    user-facing flag that did nothing for a release, so the round loop passes
    this object and a typo is a TypeError rather than a silently ignored key.
    """

    min_node_reads: float = 2.0
    f_min: float = 0.02
    a_min: float = 3.0
    q_candidate: float = 0.01
    q_edge: float = 0.01
    gamma: float = 0.6
    max_nodes: int = 8
    max_candidate_columns: int = 512
    support_min_depth: float = 3.0
    support_min_agreement: float = 0.6
    overdispersion: float = 0.01
    eps_floor: float = 0.0
    min_extension_support: float | None = 3.0
    fold_insertions: bool = True
    max_members_per_template: int = 20_000

    def kernel_kwargs(self) -> dict[str, Any]:
        return {
            "min_node_reads": self.min_node_reads,
            "f_min": self.f_min,
            "a_min": self.a_min,
            "q_candidate": self.q_candidate,
            "q_edge": self.q_edge,
            "gamma": self.gamma,
            "max_nodes": self.max_nodes,
            "max_candidate_columns": self.max_candidate_columns,
            "support_min_depth": self.support_min_depth,
            "support_min_agreement": self.support_min_agreement,
            "overdispersion": self.overdispersion,
            "eps_floor": self.eps_floor,
            "min_extension_support": self.min_extension_support,
            "fold_insertions": self.fold_insertions,
        }


def specs_from_assignment_slice(
    rows: pa.Table, reads, *, max_members: int = 20_000, seed: int = 0
) -> tuple[list[MemberSpec], np.ndarray, float]:
    """PWM members for one template.

    Returns ``(members, read_row, subsample_fraction)`` — the read rows run
    parallel to ``members`` so a node's ``member_ids`` can be turned back into
    reads.

    Replaces the dict-per-assignment-row + ``dict[read_id, str]`` form. The
    caller passes this template's contiguous slice of the sorted assignment
    table; sequences come from the mmapped corpus by ``read_row``, so the only
    Python strings that exist are this template's own members — bounded by
    ``max_members``, never by the corpus.

    Orientation is fixed: minimap2 emits query -> target, so the template is
    the reference and ``centroid_is_query`` is False.
    """
    empty = np.empty(0, dtype=np.int64)
    if rows.num_rows == 0:
        return [], empty, 1.0

    keep = pc.is_valid(rows.column("cigar"))
    keep = pc.and_(keep, pc.not_equal(pc.binary_length(rows.column("cigar")), 0))
    rows = rows.filter(keep)
    if rows.num_rows == 0:
        return [], empty, 1.0

    weight = rows.column("weight").to_numpy(zero_copy_only=False).astype(np.float64)
    fraction = 1.0
    if max_members and rows.num_rows > max_members:
        # A UNIFORM RANDOM sample, not the heaviest members. Under the current
        # rule every assigned read has weight 1.0, so "take the heaviest" is a
        # stable sort over ties — i.e. the first `max_members` rows in
        # whatever order the assignment table happened to be in. That is a
        # prefix, not a sample, and it is systematically biased: a 60/40
        # variant mixture can lose the minority entirely, which is precisely
        # the population this pipeline exists to keep.
        #
        # Seeded per template so a re-run and a resumed run agree.
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        order = np.sort(
            rng.choice(rows.num_rows, size=max_members, replace=False)
        )
        fraction = max_members / rows.num_rows
        rows = rows.take(pa.array(order))
        weight = weight[order]

    read_row = rows.column("read_row").to_numpy(zero_copy_only=False).astype(np.int64)
    seqs = reads.take_sequences(read_row)
    cigars = rows.column("cigar").to_pylist()
    t_start = rows.column("t_start").to_numpy(zero_copy_only=False)
    q_start = rows.column("q_start").to_numpy(zero_copy_only=False)

    return [
        MemberSpec(
            member_seq=seqs[i],
            weight=float(weight[i]),
            cigar=cigars[i],
            centroid_is_query=False,
            ref_start=int(t_start[i]),
            member_start=int(q_start[i]),
            member_id=i,
        )
        for i in range(rows.num_rows)
    ], read_row, fraction


def plan_mstep_units(
    template_row: np.ndarray,
    row_lo: np.ndarray,
    row_hi: np.ndarray,
    template_length: np.ndarray,
    *,
    n_units: int,
    max_members: int = 20_000,
) -> list[MStepUnit]:
    """Bin-pack templates into ``n_units``, largest cost first.

    Cost is ``min(members, max_members) x template length`` — the
    ``frame_consensus`` term, capped exactly as the worker caps it. The cap
    has to be in the model or the two disagree: a 500k-member template costs
    the planner 60x a unit's budget and gets its own unit, while the worker
    only ever builds 20k of them.

    It is also why the cap exists at all. LPT balances what it can divide, and
    packing 200k templates with no mega-template measures at load imbalance
    **1.000**; insert one 500k-member template and it goes to **60.6**,
    because no packing can split a single unit of work. Capping members is the
    only thing that bounds it.
    """
    sizes = (row_hi - row_lo).astype(np.int64)
    live = np.flatnonzero(sizes > 0)
    if live.size == 0:
        return []
    effective = (
        np.minimum(sizes[live], int(max_members)) if max_members else sizes[live]
    )
    cost = effective.astype(np.float64) * np.maximum(
        template_length[template_row[live]].astype(np.float64), 1.0
    )
    order = live[np.argsort(-cost, kind="stable")]
    n_units = max(1, min(int(n_units), order.size))

    loads = np.zeros(n_units, dtype=np.float64)
    buckets: list[list[int]] = [[] for _ in range(n_units)]
    cost_of = dict(zip(live.tolist(), cost.tolist()))
    for t in order.tolist():
        b = int(np.argmin(loads))
        buckets[b].append(t)
        loads[b] += cost_of[t]

    units: list[MStepUnit] = []
    for idx in np.argsort(-loads):
        members = buckets[int(idx)]
        if not members:
            continue
        sel = np.array(sorted(members), dtype=np.int64)
        units.append(
            MStepUnit(
                unit_idx=len(units),
                rows=template_row[sel],
                row_lo=row_lo[sel],
                row_hi=row_hi[sel],
                cost=float(loads[int(idx)]),
            )
        )
    return units


def sort_assignments_by_template(assignments: pa.Table) -> tuple[pa.Table, np.ndarray]:
    """Sort by ``template_row`` and return ``(sorted, group_starts)``.

    Sorts on the NARROW numeric projection and then takes once, rather than
    handing Acero the wide table: at this cardinality putting ``cigar`` and
    ``read_id`` through the offsets rewrite pins one core for hours (the
    standing resolve-stage rule).
    """
    assigned = assignments.filter(
        pc.greater_equal(assignments.column("template_row"), 0)
    )
    if assigned.num_rows == 0:
        return assigned, np.zeros(1, dtype=np.int64)
    perm = pc.sort_indices(
        assigned.select(["template_row"]),
        sort_keys=[("template_row", "ascending")],
    )
    srt = assigned.take(perm)
    tr = srt.column("template_row").to_numpy(zero_copy_only=False)
    starts = np.flatnonzero(np.concatenate([[True], tr[1:] != tr[:-1]]))
    return srt, starts


# Per-process handles, opened on first use and reused. A forked worker
# inherits an empty dict and mmaps for itself, so nothing large is pickled and
# nothing depends on the start method.
_OPEN: dict[str, object] = {}


def _reads(path: str):
    key = f"reads:{path}"
    if key not in _OPEN:
        from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
            ReadStore,
        )

        _OPEN[key] = ReadStore.open(path)
    return _OPEN[key]


def _templates(path: str):
    key = f"templates:{path}"
    if key not in _OPEN:
        from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
            TemplateStore,
        )

        _OPEN[key] = TemplateStore.open(path)
    return _OPEN[key]


def mstep_worker(
    batch: pa.Table,
    *,
    corpus_path: str,
    templates_path: str,
    round_index: int,
    params: "MStepParams",
    error_model=None,
) -> dict[str, pa.Table]:
    """Refine every template in one unit. Module-level, so it pickles by name.

    ``batch`` is this unit's slice of the template-sorted assignment table.
    Nothing else crosses the process boundary: the corpus and the templates
    are mmapped here.

    **No torch below this line.** The parent has already spawned torch's
    thread pool during minimizer extraction, and a torch op after ``fork()``
    deadlocks on OpenMP — a silent hang, not an exception. The kernels this
    calls (consensus / columns / covariance) are numpy-only by design.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep import (
        refine_template,
    )

    reads = _reads(corpus_path)
    store = _templates(templates_path)
    if batch.num_rows == 0:
        return {
            "nodes": REFINED_NODE_TABLE.empty_table(),
            "node_membership": NODE_MEMBERSHIP_TABLE.empty_table(),
        }

    tr = batch.column("template_row").to_numpy(zero_copy_only=False)
    starts = np.flatnonzero(np.concatenate([[True], tr[1:] != tr[:-1]]))
    bounds = np.concatenate([starts, [batch.num_rows]])

    rows: list[tuple] = []
    mem_parent: list[np.ndarray] = []
    mem_hap: list[np.ndarray] = []
    mem_read: list[np.ndarray] = []
    mem_weight: list[np.ndarray] = []
    for i in range(starts.size):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        row = int(tr[lo])
        members, member_read_row, fraction = specs_from_assignment_slice(
            batch.slice(lo, hi - lo),
            reads,
            max_members=params.max_members_per_template,
            seed=int(store.template_id[row]),
        )
        if not members:
            continue
        orf = (int(store.orf_start[row]), int(store.orf_end[row]))
        try:
            nodes = refine_template(
                store.sequence(row),
                members,
                template_id=int(store.template_id[row]),
                seed_orf=orf,
                n_assigned=float(sum(m.weight for m in members)),
                error_model=error_model,
                **params.kernel_kwargs(),
            )
        except Exception:  # noqa: BLE001 — one bad template must not kill a unit
            continue
        # Every read assigned to this template, not just the ones sampled into
        # the PWM. The cap bounds what the CONSENSUS is built from; it must not
        # bound what the cluster CONTAINS. Emitting only sampled reads made
        # exported counts and quantification report the cap — 20 reads capped
        # at 5 exported as 5 — and scaling node_weight did not repair it,
        # because both are counted from membership rows.
        all_read_row = (
            batch.slice(lo, hi - lo)
            .column("read_row")
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )
        node_of_member = np.zeros(len(members), dtype=np.int32)
        for h, node in enumerate(nodes):
            node_of_member[np.asarray(node.member_ids, dtype=np.int64)] = h

        # The sample is uniform, so the unsampled tail is exchangeable with it;
        # there is no per-read evidence to place those reads on a minor node.
        # They go to the major one (nodes are emitted mass-descending), which
        # is the approximation the cap buys and is recorded as such.
        sampled = np.zeros(all_read_row.size, dtype=bool)
        pos_of_row = {int(r): k for k, r in enumerate(all_read_row)}
        take_pos = np.array(
            [pos_of_row[int(r)] for r in member_read_row], dtype=np.int64
        )
        sampled[take_pos] = True
        hap_of_row = np.zeros(all_read_row.size, dtype=np.int32)
        hap_of_row[take_pos] = np.array(
            [nodes[int(h)].haplotype_id for h in node_of_member], dtype=np.int32
        )
        hap_of_row[~sampled] = nodes[0].haplotype_id

        counts = np.bincount(
            hap_of_row, minlength=max(n.haplotype_id for n in nodes) + 1
        )
        for node in nodes:
            n_reads = int(counts[node.haplotype_id])
            rows.append(
                _node_row(node, row, round_index, len(members), fraction, n_reads)
            )
        mem_parent.append(
            np.full(all_read_row.size, nodes[0].parent_template_id, np.int64)
        )
        mem_hap.append(hap_of_row)
        mem_read.append(all_read_row)
        mem_weight.append(np.ones(all_read_row.size, dtype=np.float32))

    return {
        "nodes": _rows_to_nodes(rows),
        "node_membership": _rows_to_membership(
            round_index, mem_parent, mem_hap, mem_read, mem_weight
        ),
    }


def _rows_to_membership(round_index, parent, hap, read, weight) -> pa.Table:
    if not parent:
        return NODE_MEMBERSHIP_TABLE.empty_table()
    p_arr = np.concatenate(parent)
    return pa.table(
        {
            "round": pa.array(np.full(p_arr.size, int(round_index), np.int32)),
            "parent_template_id": pa.array(p_arr),
            "haplotype_id": pa.array(np.concatenate(hap)),
            "read_row": pa.array(np.concatenate(read).astype(np.int32)),
            "weight": pa.array(np.concatenate(weight)),
        },
        schema=NODE_MEMBERSHIP_TABLE,
    )


def _node_row(node, parent_row, round_index, n_members, fraction, n_reads) -> tuple:
    return (
        int(round_index),
        int(node.parent_template_id),
        int(parent_row),
        int(node.haplotype_id),
        node.consensus,
        # From MEMBERSHIP, not from the PWM's sampled mass: n_reads and
        # node_weight are what quant and the next round's abundance ranking
        # read, and they have to describe the reads the cluster actually holds.
        int(n_reads),
        float(n_reads),
        node.protein or None,
        int(node.orf_start),
        int(node.orf_end),
        int(node.orf_certified_end),
        bool(node.orf_is_truncated_by_support),
        node.allele_string,
        [int(v) for v in np.asarray(node.declared_variants).tolist()],
        int(node.n_inserted_columns),
        int(node.n_extended_5p),
        int(node.n_extended_3p),
        int(node.n_trimmed_5p),
        int(node.n_trimmed_3p),
        int(n_members),
        float(fraction),
    )


def _rows_to_nodes(rows: list[tuple]) -> pa.Table:
    if not rows:
        return REFINED_NODE_TABLE.empty_table()
    cols = list(zip(*rows))
    return pa.table(
        {
            f.name: pa.array(cols[i], type=f.type)
            for i, f in enumerate(REFINED_NODE_TABLE)
        },
        schema=REFINED_NODE_TABLE,
    )


#: The only columns the worker reads. Projecting before the pickle boundary
#: drops `read_id` and thirteen others that cross it for nothing.
MSTEP_WORKER_COLUMNS: tuple[str, ...] = (
    "template_row",
    "read_row",
    "cigar",
    "t_start",
    "q_start",
    "weight",
)


def iter_unit_batches(
    sorted_assignments: pa.Table, units: list[MStepUnit]
) -> Iterator[pa.Table]:
    """One ``pa.Table`` per unit, largest first.

    Yielded lazily so ``run_batched``'s bounded in-flight window backpressures
    the iterator instead of the whole round's slices existing at once.

    **``combine_chunks()`` here is not tidying — it is the difference between
    running and dying.** ``pa.concat_tables`` of one slice per template gives
    a table with one CHUNK per template, and every Arrow chunk holds a
    reference to the whole parent buffer. Pickling for ``ProcessPoolExecutor``
    then serialises that parent once per chunk, so the payload scales with the
    NUMBER OF TEMPLATES in the unit rather than the rows it contains.

    Measured on a 200k-row stand-in, 400 templates to a unit: **7,360 MB
    pickled, against 3.7 MB after detaching — 1,991x, for byte-identical
    content.** On the real 1M-read data it was ~90 GB per batch against 1.4 MB,
    which is why 100k reads ran and 1M did not, and why *fewer* workers made it
    worse: fewer workers means more templates per unit means a linearly larger
    pickle.

    ``take`` rather than ``concat_tables(...).combine_chunks()``, because
    **combine_chunks is a no-op on a single-chunk column** and a unit holding
    ONE template is exactly one slice. Measured: such a unit pickled at
    **60.4 MB for 6 KB of content** while its parent stayed attached. That is
    not an edge case — a unit holds one template whenever a mega-template is
    packed alone, which LPT does deliberately for the biggest ones, and
    whenever live templates fall below the requested unit count. ``take``
    gathers into fresh buffers unconditionally.
    """
    columns = [c for c in MSTEP_WORKER_COLUMNS if c in sorted_assignments.column_names]
    narrow = sorted_assignments.select(columns)
    for unit in units:
        spans = [
            np.arange(int(lo), int(hi), dtype=np.int64)
            for lo, hi in zip(unit.row_lo, unit.row_hi)
            if hi > lo
        ]
        if spans:
            yield narrow.take(pa.array(np.concatenate(spans)))


__all__ = [
    "MSTEP_WORKER_COLUMNS",
    "NODE_MEMBERSHIP_TABLE",
    "REFINED_NODE_TABLE",
    "MStepParams",
    "MStepUnit",
    "iter_unit_batches",
    "mstep_worker",
    "plan_mstep_units",
    "sort_assignments_by_template",
    "specs_from_assignment_slice",
]
