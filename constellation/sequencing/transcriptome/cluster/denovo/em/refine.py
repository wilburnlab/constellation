"""Between rounds: next templates, lineage, convergence — and no prune.

**There is no support prune, and none is wanted.** The handoff's §4 and
ledger #34 call one a prerequisite, on the reading that template t302020
(Tuba1a — 1,371 reads on a 21-stop sequence) was a single read's own sequence
promoted to a template. The cause established since is different: its
consensus ORF was *contaminated* by reads that should never have entered the
PWM, which is what ``p_floor`` fixes — there was no admission floor at all.
The errored template is a product of unfiltered PWM membership, not of low
support.

So the governing principle: a read was sequenced, it is real, it is just not
like anything else. It is kept as a lowly-supported transcript and the
scientist decides what to make of it. Requiring a read to earn additional
support to justify its own existence is the wrong shape for this tool.

A template therefore disappears exactly one way, and it is a consequence
rather than a rule: **one that recruits no read emits no node**, so it does
not carry forward. Self-capture — which would otherwise keep every round-1
template alive forever — is handled where it arises, in the round-1 ranking
(see :mod:`.scheduler`), not by deleting anything here.

**Merge is opt-in, and its criterion is measured in nucleotides (2026-09-28).**
The merge restored on 2026-09-23 collapsed templates a minimap2 all-vs-all
called redundant — identity >= 0.995 and coverage >= 0.95 both ways — and on
the 9.4M-read bench it did more harm than good: clusters carrying an ORF fell
from 54.4% to 41.3%, RefSeq ``with_hit`` from 0.540 to 0.381 and
``full_length`` from 0.332 to 0.188. Three things were wrong with the
criterion itself. Identity was ``n_match / aln_len`` on a *local* hit, so
unaligned ends were free and a contained sequence scored 100%. Coverage was a
*proportion* standing in for end tolerance, so ~100 nt of differing end was
free at 2 kb and 25 nt at 500 nt, and a pair staggered at both ends passed.
And 51.6% of the merges re-joined two children the M-step had split from one
parent in the same round, a limit cycle moving ~7.5% of reads every round.

What replaces it is a relationship *graph* (:mod:`.graph`): kmer candidates,
two infix alignments per pair, and a typed edge per related pair. A merge is a
predicate over that graph's ``equivalent`` edges — by default at most two
edits over the shared span, ends within 30 nt, and not separated by the same
M-step split. It is on by default: at 9.4M reads a sweep of the edit cap
from 1 to 6 left protein recovery flat while two edits nearly stopped node
growth between rounds (ledger #55).

What lives here is only the collapse. Groups are formed by an
abundance-ordered **radius-1** set cover, so ``A~B~C`` cannot merge ``A`` with
``C``, and the guards hold per *group* rather than per pair
(:func:`~..cluster_graph.guarded_set_cover`): a hub mergeable with both
children of a split would otherwise join them through itself. Between rounds
the best-supported member survives and the next M-step re-derives its
consensus from the union of reads. In the final output no M-step follows, so
the loop does that part itself: the deepest member survives and its consensus
is rebuilt from the group's pooled reads (:mod:`.rebuild`) before the ORF is
called. A merge is recorded in the lineage
as the inverse of a split (``rule='merge'``), so churn does not read it as
mass reassignment, and in :data:`MERGED_TABLE`, because the lineage overwrites
an absorbed template's id with its survivor's.

**Templates are joined to the graph by id, never by row.** The graph is built
over a round's *nodes*; :func:`next_templates` assigns ids over node rows and
then drops zero-length consensus, so from the first empty node on, template
row and node row disagree. Merging by row would collapse a twin into its
unrelated neighbour with no error anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
    TEMPLATE_TABLE,
)


LINEAGE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("round", pa.int32(), nullable=False),
        pa.field("child_template_id", pa.int64(), nullable=False),
        pa.field("parent_template_id", pa.int64(), nullable=False),
        # 'carry'       — the parent emitted exactly one node
        # 'split'       — it emitted several; the read did not really move
        # 'unrecruited' — it emitted none and does not carry forward
        # 'merge'       — its node was absorbed into a redundant survivor;
        #                 child_template_id is the SURVIVOR's id
        pa.field("rule", pa.string(), nullable=False),
        pa.field("n_reads", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"EmLineageTable"},
)

#: Template ids are ``(round << _ROUND_SHIFT) | row`` so they are unique for
#: the life of a run AND deterministic given the round — which is what lets a
#: resumed round rebuild identical ids without a parent-side counter.
_ROUND_SHIFT = 40


def template_id_for(round_index: int, row: int | np.ndarray):
    return (int(round_index) << _ROUND_SHIFT) | np.asarray(row, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class RefineResult:
    templates: pa.Table
    lineage: pa.Table
    n_parents: int
    n_children: int
    n_unrecruited: int
    n_merged: int = 0
    #: ``MERGED_TABLE`` — what the merge absorbed. ``None`` until a merge
    #: pass has run; empty when one ran and absorbed nothing.
    merged: pa.Table | None = None


def next_templates(
    nodes: pa.Table,
    store,
    *,
    round_index: int,
) -> RefineResult:
    """Turn a round's nodes into the next round's templates, plus lineage.

    ``round_index`` is the round the nodes came FROM; children are stamped
    ``round_index + 1``.
    """
    next_round = int(round_index) + 1
    if nodes.num_rows == 0:
        return RefineResult(
            templates=TEMPLATE_TABLE.empty_table(),
            lineage=LINEAGE_TABLE.empty_table(),
            n_parents=int(store.n_templates),
            n_children=0,
            n_unrecruited=int(store.n_templates),
        )

    n = nodes.num_rows
    child_id = template_id_for(next_round, np.arange(n))
    parent_id = nodes.column("parent_template_id").to_numpy(zero_copy_only=False)
    parent_row = nodes.column("parent_template_row").to_numpy(zero_copy_only=False)

    # A parent that emitted several nodes SPLIT; its reads did not move by
    # choosing differently, so the churn measure must not count them.
    _, counts = np.unique(parent_id, return_counts=True)
    per_parent = dict(zip(*np.unique(parent_id, return_counts=True)))
    rule = np.array(
        ["split" if per_parent[int(p)] > 1 else "carry" for p in parent_id],
        dtype=object,
    )

    consensus = nodes.column("consensus")
    seq_len = pc.utf8_length(consensus).to_numpy(zero_copy_only=False)
    orf_start = nodes.column("orf_start").to_numpy(zero_copy_only=False)
    orf_end = nodes.column("orf_end").to_numpy(zero_copy_only=False)

    templates = pa.table(
        {
            "template_id": pa.array(child_id),
            "sequence": consensus.cast(pa.large_string()),
            "orf_start": pa.array(orf_start.astype(np.int32)),
            "orf_end": pa.array(orf_end.astype(np.int32)),
            "orf_aa_length": pa.array(
                np.maximum((orf_end - orf_start) // 3 - 1, 0).astype(np.int32)
            ),
            "node_weight": nodes.column("node_weight").cast(pa.float64()),
            # Carried for provenance only: from round 2 the ranking is the
            # likelihood, and orf_replication is round 1's key alone.
            "orf_replication": pa.array(
                np.maximum(
                    nodes.column("n_reads").to_numpy(zero_copy_only=False), 0
                ).astype(np.int64)
            ),
            # Null from round 2: the frame is a consensus with no read of its
            # own, so there is no seed quality to speak of.
            "seed_read_quality": pa.nulls(n, pa.float32()),
            "seed_read_row": pa.array(np.full(n, -1, dtype=np.int32)),
            "declared_variants": nodes.column("declared_variants"),
        },
        schema=TEMPLATE_TABLE,
    )
    # A zero-length consensus cannot be a minimap2 target.
    templates = templates.filter(pa.array(seq_len > 0))

    recruited = np.unique(parent_row)
    unrecruited = np.setdiff1d(np.arange(store.n_templates), recruited)
    lineage = _lineage_table(
        next_round,
        child_id=child_id,
        parent_id=parent_id,
        rule=rule,
        n_reads=nodes.column("n_reads").to_numpy(zero_copy_only=False).astype(float),
        unrecruited_ids=store.template_id[unrecruited],
    )
    return RefineResult(
        templates=templates,
        lineage=lineage,
        n_parents=int(store.n_templates),
        n_children=int(templates.num_rows),
        n_unrecruited=int(unrecruited.size),
    )


def _lineage_table(
    round_index: int,
    *,
    child_id: np.ndarray,
    parent_id: np.ndarray,
    rule: np.ndarray,
    n_reads: np.ndarray,
    unrecruited_ids: np.ndarray,
) -> pa.Table:
    m = unrecruited_ids.size
    return pa.table(
        {
            "round": pa.array(
                np.full(child_id.size + m, int(round_index), dtype=np.int32)
            ),
            "child_template_id": pa.array(
                np.concatenate([child_id, np.full(m, -1, dtype=np.int64)])
            ),
            "parent_template_id": pa.array(
                np.concatenate([parent_id, unrecruited_ids.astype(np.int64)])
            ),
            "rule": pa.array(list(rule) + ["unrecruited"] * m, pa.string()),
            "n_reads": pa.array(
                np.concatenate([n_reads, np.zeros(m)]).astype(np.float64)
            ),
        },
        schema=LINEAGE_TABLE,
    )


def measure_churn(
    prev_assignments: pa.Table,
    cur_assignments: pa.Table,
    lineage: pa.Table,
    *,
    n_reads: int,
) -> dict[str, float]:
    """Lineage-aware read-switch fraction. O(n_reads), no hash join.

    ``read_row`` is a dense index into the corpus, so both rounds scatter into
    ``(n_reads,)`` arrays and the comparison is elementwise — no Acero join
    and no allocation proportional to the join output.

    Lineage-aware is the load-bearing part: a read whose template merely split
    into several children has not *chosen* differently, and counting it as
    churn would make the loop look unconverged forever.
    """
    prev = np.full(n_reads, -1, dtype=np.int64)
    cur = np.full(n_reads, -1, dtype=np.int64)
    _scatter(prev, prev_assignments)
    _scatter(cur, cur_assignments)

    both = (prev >= 0) & (cur >= 0)
    moved = both & (prev != cur)
    n_both, n_moved = int(both.sum()), int(moved.sum())

    n_inherited = 0
    if n_moved and lineage.num_rows:
        edges = np.unique(
            _pair_key(
                lineage.column("child_template_id").to_numpy(zero_copy_only=False),
                lineage.column("parent_template_id").to_numpy(zero_copy_only=False),
            )
        )
        probe = _pair_key(cur[moved], prev[moved])
        pos = np.searchsorted(edges, probe)
        ok = (pos < edges.size) & (edges[np.clip(pos, 0, edges.size - 1)] == probe)
        n_inherited = int(ok.sum())

    genuine = n_moved - n_inherited
    n_gained = int(((prev < 0) & (cur >= 0)).sum())
    n_lost = int(((prev >= 0) & (cur < 0)).sum())
    # Reads that GAINED or LOST an assignment are movement too. Measured over
    # the both-assigned population alone, losing half the assignments scores
    # zero churn as long as the survivors kept their lineage — so the loop
    # would call that converged. The denominator is therefore every read
    # assigned in EITHER round.
    n_either = n_both + n_gained + n_lost
    return {
        "n_compared": n_both,
        "n_moved_raw": n_moved,
        "n_moved_inherited": n_inherited,
        "n_moved_genuine": genuine,
        "frac_changed": (n_moved / n_both) if n_both else 0.0,
        "frac_changed_lineage": (genuine / n_both) if n_both else 0.0,
        # What the stopping rule reads: unstable reads over reads assigned in
        # either round, so gains and losses cannot hide inside a stable core.
        "frac_unsettled": (
            (genuine + n_gained + n_lost) / n_either if n_either else 0.0
        ),
        "n_gained": n_gained,
        "n_lost": n_lost,
    }


def _scatter(out: np.ndarray, table: pa.Table) -> None:
    if table.num_rows == 0:
        return
    rows = table.column("read_row").to_numpy(zero_copy_only=False).astype(np.int64)
    tid = table.column("template_id").to_numpy(zero_copy_only=False).astype(np.int64)
    keep = (rows >= 0) & (rows < out.size) & (tid >= 0)
    out[rows[keep]] = tid[keep]


#: Exact (id, id) comparison with no packing. The bit-packed form this
#: replaces, ``(child << 21) ^ parent``, overlaps the two ids once a row index
#: reaches 2**21 — and round 1 carries ~4.17M templates, so the collision is
#: reachable in production, not theoretical. A collision reports an unrelated
#: switch as inherited, which reads as zero lineage-aware churn and stops the
#: loop early.
_PAIR_DTYPE = np.dtype([("child", np.int64), ("parent", np.int64)])


def _pair_key(child: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """A lexicographically sortable structured view of (child, parent)."""
    out = np.empty(np.asarray(child).size, dtype=_PAIR_DTYPE)
    out["child"] = child
    out["parent"] = parent
    return out


# ── merge ─────────────────────────────────────────────────────────────

#: What a merge pass absorbed, one row per absorbed template. The lineage
#: cannot give this back: :func:`apply_merge` overwrites an absorbed child's id
#: with its survivor's, so afterwards the absorbed id appears nowhere.
MERGED_TABLE: pa.Schema = pa.schema(
    [
        # The round the ids belong to — the CHILD round, as lineage stamps it.
        pa.field("round", pa.int32(), nullable=False),
        pa.field("absorbed_template_id", pa.int64(), nullable=False),
        pa.field("survivor_template_id", pa.int64(), nullable=False),
        # Signed nt: how far the SURVIVOR reaches beyond the absorbed template
        # at that end. Negative means the absorbed one reached further, and
        # that much extent is what the merge gave up.
        pa.field("delta_5p", pa.int32(), nullable=False),
        pa.field("delta_3p", pa.int32(), nullable=False),
        pa.field("absorbed_n_reads", pa.int64(), nullable=False),
    ],
    metadata={b"schema_name": b"EmMergedTable"},
)

#: Who claims first. ``"support"`` is (reads desc, length desc, row asc) —
#: between rounds, where the next M-step re-derives the consensus from the
#: union of reads. ``"length"`` is (length desc, reads desc, row asc) — the
#: final output, where nothing re-derives it and the survivor's own consensus
#: is what gets reported.
Prefer = Literal["support", "length"]


def claim_order(
    n_reads: np.ndarray, seq_len: np.ndarray, prefer: Prefer = "support"
) -> np.ndarray:
    """Rows in claim priority, best first."""
    reads = np.maximum(np.asarray(n_reads).astype(np.int64), 0)
    length = np.asarray(seq_len).astype(np.int64)
    rows = np.arange(reads.shape[0], dtype=np.int64)
    if prefer == "support":
        return np.lexsort((rows, -length, -reads))
    if prefer == "length":
        return np.lexsort((rows, -reads, -length))
    raise ValueError(f"unknown prefer {prefer!r}; expected 'support' or 'length'")


def select_merges(
    n: int,
    a: np.ndarray,
    b: np.ndarray,
    off_5p: np.ndarray,
    off_3p: np.ndarray,
    n_reads: np.ndarray,
    seq_len: np.ndarray,
    *,
    tol_5p: int,
    tol_3p: int,
    identical: np.ndarray | None = None,
    origin: np.ndarray | None = None,
    prefer: Prefer = "support",
) -> np.ndarray:
    """``survivor_of[i]`` — the row ``i`` merges into (itself if none).

    ``a`` / ``b`` are the mergeable pairs by ROW, ``off_5p`` / ``off_3p`` how
    far ``b`` reaches beyond ``a`` at each end. Radius-1 set cover in
    :func:`claim_order`, with the extent and kin guards enforced per group
    (:func:`~..cluster_graph.guarded_set_cover`). ``origin=None`` switches the
    kin guard off, which is what ``--merge-siblings`` asks for.

    Pairs naming a row outside ``[0, n)`` or pairing a row with itself are
    dropped rather than trusted.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.cluster_graph import (
        guarded_set_cover,
    )

    n = int(n)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    ok = (a >= 0) & (a < n) & (b >= 0) & (b < n) & (a != b)
    res = guarded_set_cover(
        n,
        claim_order(n_reads, seq_len, prefer),
        a[ok],
        b[ok],
        np.asarray(off_5p, dtype=np.int64)[ok],
        np.asarray(off_3p, dtype=np.int64)[ok],
        tol_5p=int(tol_5p),
        tol_3p=int(tol_3p),
        origin=origin,
        edge_identical=None if identical is None else np.asarray(identical, bool)[ok],
    )
    return res.centroid_uniq[res.cluster_of].astype(np.int64)


def _rows_of(ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Row of each ``wanted`` id in ``ids``; ``-1`` where it is absent."""
    wanted = np.asarray(wanted, dtype=np.int64)
    if ids.size == 0:
        return np.full(wanted.shape[0], -1, dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    pos = np.clip(np.searchsorted(sorted_ids, wanted), 0, ids.size - 1)
    return np.where(sorted_ids[pos] == wanted, order[pos], -1).astype(np.int64)


def merged_table(
    ids: np.ndarray,
    survivor: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    off_5p: np.ndarray,
    off_3p: np.ndarray,
    n_reads: np.ndarray,
    *,
    round_index: int,
) -> pa.Table:
    """``MERGED_TABLE`` for one merge pass, from its ``survivor`` array.

    An absorbed row was claimed over a direct edge, so that edge carries the
    offset between the two; it is re-signed to read "survivor beyond
    absorbed" whichever endpoint the survivor was.
    """
    n = survivor.shape[0]
    absorbed = survivor != np.arange(n, dtype=np.int64)
    if not absorbed.any():
        return MERGED_TABLE.empty_table()
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    ok = (a >= 0) & (a < n) & (b >= 0) & (b < n) & (a != b)
    a, b = a[ok], b[ok]
    off_5p = np.asarray(off_5p, dtype=np.int64)[ok]
    off_3p = np.asarray(off_3p, dtype=np.int64)[ok]
    d5 = np.zeros(n, dtype=np.int64)
    d3 = np.zeros(n, dtype=np.int64)
    # Written in reverse so that of a repeated pair the FIRST edge is what
    # remains, which is the one the set cover used.
    into_b = (absorbed[a] & (survivor[a] == b))[::-1]
    into_a = (absorbed[b] & (survivor[b] == a))[::-1]
    ar, br, o5, o3 = a[::-1], b[::-1], off_5p[::-1], off_3p[::-1]
    d5[ar[into_b]], d3[ar[into_b]] = o5[into_b], o3[into_b]
    d5[br[into_a]], d3[br[into_a]] = -o5[into_a], -o3[into_a]
    rows = np.flatnonzero(absorbed)
    return pa.table(
        {
            "round": pa.array(np.full(rows.size, int(round_index), dtype=np.int32)),
            "absorbed_template_id": pa.array(ids[rows].astype(np.int64)),
            "survivor_template_id": pa.array(ids[survivor[rows]].astype(np.int64)),
            "delta_5p": pa.array(d5[rows].astype(np.int32)),
            "delta_3p": pa.array(d3[rows].astype(np.int32)),
            "absorbed_n_reads": pa.array(
                np.maximum(np.asarray(n_reads)[rows], 0).astype(np.int64)
            ),
        },
        schema=MERGED_TABLE,
    )


def apply_merge(
    refined: RefineResult,
    pairs: dict[str, np.ndarray],
    *,
    tol_5p: int,
    tol_3p: int,
    origin_ids: np.ndarray | None = None,
    origin: np.ndarray | None = None,
    prefer: Prefer = "support",
) -> RefineResult:
    """Collapse the mergeable templates of ``refined`` into their survivors.

    ``pairs`` is :func:`.graph.mergeable_pairs` output and is read by
    **template id** (``src_template_id`` / ``dst_template_id``): the graph is
    built over node rows, and a node with an empty consensus has an id and no
    template row, so every row after it is shifted. A pair naming a template
    that is not in ``refined`` is dropped.

    ``origin`` is each graph node's split origin and ``origin_ids`` the
    template id it belongs to; omit both to merge without the kin guard.

    Child ids are stable — the table is filtered, never re-indexed — so
    survivors keep their ids and an absorbed template's lineage row is
    re-pointed at its survivor with ``rule='merge'``: the edge ``(survivor,
    absorbed parent)`` then makes a read moving from the absorbed parent to the
    survivor count as inherited, exactly like a split.
    """
    t = refined.templates
    n = t.num_rows
    no_merge = RefineResult(
        templates=refined.templates,
        lineage=refined.lineage,
        n_parents=refined.n_parents,
        n_children=refined.n_children,
        n_unrecruited=refined.n_unrecruited,
        n_merged=0,
        merged=MERGED_TABLE.empty_table(),
    )
    if n == 0 or pairs["src_template_id"].shape[0] == 0:
        return no_merge

    ids = t.column("template_id").to_numpy(zero_copy_only=False).astype(np.int64)
    a = _rows_of(ids, pairs["src_template_id"])
    b = _rows_of(ids, pairs["dst_template_id"])
    reads = t.column("orf_replication").to_numpy(zero_copy_only=False)
    seq_len = pc.utf8_length(t.column("sequence")).to_numpy(zero_copy_only=False)
    kin = None
    if origin is not None:
        if origin_ids is None:
            raise ValueError("origin needs origin_ids to say whose it is")
        at = _rows_of(np.asarray(origin_ids, dtype=np.int64), ids)
        kin = np.where(at >= 0, np.asarray(origin, dtype=np.int64)[at], -1)
    survivor = select_merges(
        n,
        a,
        b,
        pairs["off_5p"],
        pairs["off_3p"],
        reads,
        seq_len,
        tol_5p=tol_5p,
        tol_3p=tol_3p,
        identical=pairs.get("identical"),
        origin=kin,
        prefer=prefer,
    )
    absorbed = survivor != np.arange(n)
    if not absorbed.any():
        return no_merge

    weight = t.column("node_weight").to_numpy(zero_copy_only=False).astype(float)
    merged_weight = np.bincount(survivor, weights=weight, minlength=n)
    merged_reads = np.bincount(
        survivor, weights=np.maximum(reads, 0).astype(float), minlength=n
    )
    templates = (
        t.set_column(
            t.schema.get_field_index("node_weight"),
            "node_weight",
            pa.array(merged_weight, pa.float64()),
        )
        .set_column(
            t.schema.get_field_index("orf_replication"),
            "orf_replication",
            pa.array(np.rint(merged_reads).astype(np.int64)),
        )
        .filter(pa.array(~absorbed))
    )

    # Re-point the absorbed children. One sorted lookup rather than a dict
    # probed per lineage row: the lineage is one row per node.
    gone = ids[absorbed]
    order = np.argsort(gone, kind="stable")
    gone, kept = gone[order], ids[survivor[absorbed]][order]
    lin = refined.lineage
    child = lin.column("child_template_id").to_numpy(zero_copy_only=False).copy()
    pos = np.clip(np.searchsorted(gone, child), 0, gone.size - 1)
    hit = gone[pos] == child
    child[hit] = kept[pos[hit]]
    lineage = lin.set_column(
        lin.schema.get_field_index("child_template_id"),
        "child_template_id",
        pa.array(child, pa.int64()),
    ).set_column(
        lin.schema.get_field_index("rule"),
        "rule",
        pc.if_else(pa.array(hit), pa.scalar("merge"), lin.column("rule")),
    )
    # set_column drops the declared not-null flags; restore the schemas.
    templates = templates.cast(TEMPLATE_TABLE)
    lineage = lineage.cast(LINEAGE_TABLE)
    return RefineResult(
        templates=templates,
        lineage=lineage,
        n_parents=refined.n_parents,
        n_children=int(templates.num_rows),
        n_unrecruited=refined.n_unrecruited,
        n_merged=int(absorbed.sum()),
        merged=merged_table(
            ids,
            survivor,
            a,
            b,
            pairs["off_5p"],
            pairs["off_3p"],
            reads,
            round_index=int(ids[0] >> _ROUND_SHIFT),
        ),
    )


def merge_nodes(
    nodes: pa.Table,
    node_membership: pa.Table,
    pairs: dict[str, np.ndarray],
    *,
    tol_5p: int,
    tol_3p: int,
    origin: np.ndarray | None = None,
    prefer: Prefer = "support",
) -> tuple[pa.Table, pa.Table, int, np.ndarray]:
    """The final-output merge: collapse mergeable NODES and their membership.

    ``pairs`` is :func:`.graph.mergeable_pairs` output read by **node row**
    (``src_row`` / ``dst_row``) — here the graph's rows and the table's are
    the same rows. Absorbed nodes are dropped and their membership rows
    re-keyed to the survivor's ``(parent_template_id, haplotype_id)``; counts
    and quant are computed from membership, so every read is kept and totals
    are conserved.

    The survivor's consensus is what is written here; the caller rebuilds
    it from the pooled membership afterwards (:mod:`.rebuild`), which is why
    the merge may be edit-tolerant and why the deepest member is kept, as
    between rounds. Keeping the longest instead, with the pairs held exact,
    was measured the wrong form more often than not (ledger #52).

    Returns ``(nodes, membership, n_merged, survivor_of)``; ``survivor_of``
    indexes the INPUT rows.
    """
    n = nodes.num_rows
    unmerged = np.arange(n, dtype=np.int64)
    if n == 0 or pairs["src_row"].shape[0] == 0:
        return nodes, node_membership, 0, unmerged
    n_reads = nodes.column("n_reads").to_numpy(zero_copy_only=False)
    seq_len = pc.utf8_length(nodes.column("consensus")).to_numpy(zero_copy_only=False)
    survivor = select_merges(
        n,
        pairs["src_row"],
        pairs["dst_row"],
        pairs["off_5p"],
        pairs["off_3p"],
        n_reads,
        seq_len,
        tol_5p=tol_5p,
        tol_3p=tol_3p,
        identical=pairs.get("identical"),
        origin=origin,
        prefer=prefer,
    )
    absorbed = survivor != unmerged
    if not absorbed.any():
        return nodes, node_membership, 0, unmerged

    pid = nodes.column("parent_template_id").to_numpy(zero_copy_only=False)
    hap = nodes.column("haplotype_id").to_numpy(zero_copy_only=False)
    sums = {}
    for col in ("n_reads", "node_weight"):
        v = nodes.column(col).to_numpy(zero_copy_only=False).astype(float)
        sums[col] = np.bincount(survivor, weights=np.maximum(v, 0), minlength=n)
    out = nodes.set_column(
        nodes.schema.get_field_index("n_reads"),
        "n_reads",
        pa.array(np.rint(sums["n_reads"]).astype(np.int64)),
    ).set_column(
        nodes.schema.get_field_index("node_weight"),
        "node_weight",
        pa.array(sums["node_weight"], pa.float64()),
    ).filter(pa.array(~absorbed))

    membership = node_membership
    if membership is not None and membership.num_rows:
        # Exact (parent_template_id, haplotype_id) lookup — the same
        # structured-pair comparison the churn measure uses, no bit packing.
        key = _pair_key(pid, hap.astype(np.int64))
        order = np.argsort(key)
        m_pid = membership.column("parent_template_id").to_numpy(zero_copy_only=False)
        m_hap = membership.column("haplotype_id").to_numpy(zero_copy_only=False)
        mkey = _pair_key(m_pid, m_hap.astype(np.int64))
        pos = np.clip(np.searchsorted(key[order], mkey), 0, n - 1)
        found = key[order][pos] == mkey
        target = survivor[order[pos]]
        new_pid = np.where(found, pid[target], m_pid)
        new_hap = np.where(found, hap[target], m_hap)
        membership = membership.set_column(
            membership.schema.get_field_index("parent_template_id"),
            "parent_template_id",
            pa.array(new_pid.astype(np.int64)),
        ).set_column(
            membership.schema.get_field_index("haplotype_id"),
            "haplotype_id",
            pa.array(new_hap.astype(np.int32)),
        )
    out = out.cast(nodes.schema)
    if membership is not None and membership.num_rows:
        membership = membership.cast(node_membership.schema)
    return out, membership, int(absorbed.sum()), survivor


__all__ = [
    "LINEAGE_TABLE",
    "MERGED_TABLE",
    "Prefer",
    "RefineResult",
    "apply_merge",
    "claim_order",
    "measure_churn",
    "merge_nodes",
    "merged_table",
    "next_templates",
    "select_merges",
    "template_id_for",
]
