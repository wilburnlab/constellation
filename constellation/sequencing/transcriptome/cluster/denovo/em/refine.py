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

**Merge is back (2026-09-23), with evidence behind its criterion.** It was
dropped on the argument that redundant templates lose their reads at
assignment time, and :func:`detect_redundant_templates` was left to make that
checkable — but its minimap2 flags combined ``-X`` with ``--secondary=no``,
which minimap2 refuses, so the check never ran. Run with corrected flags on the
finished em-kmer runs, **26.9% (1M reads) and 29.4% (9.4M) of the final
template set was removable redundancy**, the median redundant pair exactly
identical and the largest component 331 templates. The mechanism: round-1
M-steps split at shallow depth, round 2 redistributes a large read population
onto whichever fragment matches best, and nothing revisits the split — so a
split made on evidence that later evaporates is permanent without a merge.

The criterion is ``handoff_scheduler_v3`` §5's: a whole-*template* alignment
(transcripts, never consensus ORFs — the ORF-level merge was struck for
joining parents at 74.5% transcript identity) with identity ≥ ``p_merge``
(0.995, deliberately stricter than the 0.97 read floor, because a merge is a
stronger claim than an assignment) and coverage ≥ 0.95 **in both
directions**. Groups are formed by abundance-ordered **radius-1** greedy set
cover (:func:`select_merges`), so ``A~B~C`` cannot merge ``A`` with ``C``;
whatever a pass leaves is caught next round. The survivor is the
best-supported member; the next M-step re-derives its consensus from the
union of reads the E-step hands it, so a merged group is re-consensused for
free one round later. A merge is recorded in the lineage as the inverse of a
split (``rule='merge'``), so churn does not read it as mass reassignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


# ── the redundancy detector (B2) ──────────────────────────────────────

MERGE_MINIMAP2_ARGS: tuple[str, ...] = (
    "-x",
    "asm5",
    # -c is MANDATORY, not an optimisation: without base-level alignment PAF
    # column 10 is chain-anchor matches — one substitution removes ~19 bases
    # of asm5 anchors — which underestimates identity badly and makes a 0.995
    # threshold mean "exactly identical".
    "-c",
    # -X (= -DP --dual=no) skips self-hits and reports each unordered pair
    # once, halving the work and removing the q_name == t_name filter. It must
    # NOT be combined with --secondary=no: minimap2 refuses the pair outright
    # ("-X/-P and --secondary=no can't be applied at the same time"), which is
    # why this detector never produced a number before 2026-09-23.
    "-X",
    "-p",
    "0.9",
    "-N",
    "5",
)

#: Returned by :func:`detect_redundant_templates`: one row per redundant
#: unordered pair, named by template ROW (the FASTAs name by row index).
REDUNDANT_PAIR_TABLE: pa.Schema = pa.schema(
    [
        pa.field("a_row", pa.int64(), nullable=False),
        pa.field("b_row", pa.int64(), nullable=False),
        pa.field("identity", pa.float64(), nullable=False),
        pa.field("coverage_a", pa.float64(), nullable=False),
        pa.field("coverage_b", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"EmRedundantPairTable"},
)


#: Pairs are REPORTED down to this identity, below the merge threshold, so
#: the near-threshold population — where a genuine single-position variant
#: lives — is visible in the diagnostics rather than silently excluded.
REPORT_IDENTITY_FLOOR = 0.99


def detect_redundant_templates(
    templates_fasta: Path,
    *,
    min_identity: float = REPORT_IDENTITY_FLOOR,
    min_mutual_coverage: float = 0.95,
    threads: int = 8,
    index_batch_size: str = "16G",
) -> pa.Table:
    """Near-identical template pairs (``REDUNDANT_PAIR_TABLE``). Collapses nothing.

    Every pair with identity ≥ ``min_identity`` and both coverages ≥
    ``min_mutual_coverage``; the merge threshold ``p_merge`` is applied by the
    consumers (:func:`redundancy_stats`, :func:`select_merges`), so one scan
    serves both the merge and the near-threshold report.

    Why ``asm5`` rather than ``map-ont`` is not a tuning preference: a preset
    sets the seed length and how densely seeds are sampled. ``map-ont`` uses
    k=15/w=10 because a noisy read has an error every ~100 bp and only a short
    seed has a good chance of being error-free — at the cost of specificity,
    since a 15-mer recurs by chance across a transcriptome. ``asm5`` uses
    k=19/w=19, for sequences that are accurate and <=5% divergent: a 19-mer is
    ~16x more specific and accurate sequences share long exact stretches, so
    it can sample ~2x more sparsely. Template-vs-template at >=99.5% is the
    second situation.

    ``-I`` is passed explicitly: an all-vs-all over a multi-part index never
    compares templates that land in different parts.

    Needs **no CIGAR**: identity and both coverages come from PAF columns.
    The FASTA must name templates by row index, as ``write_templates`` does.
    """
    from constellation.sequencing.align.minimap2 import minimap2_stream
    from constellation.sequencing.readers.paf import iter_paf_batches

    stream = minimap2_stream(
        Path(templates_fasta),
        [Path(templates_fasta)],
        args=(*MERGE_MINIMAP2_ARGS, "-I", str(index_batch_size)),
        threads=threads,
    )
    rows: list[pa.Table] = []
    for batch in iter_paf_batches(stream, want_cigar=False):
        tbl = pa.Table.from_batches([batch])
        f64 = pa.float64()
        identity = pc.divide(
            pc.cast(tbl.column("n_match"), f64),
            pc.max_element_wise(pc.cast(tbl.column("aln_len"), f64), 1.0),
        )
        cov_q = pc.divide(
            pc.cast(pc.subtract(tbl.column("q_end"), tbl.column("q_start")), f64),
            pc.max_element_wise(pc.cast(tbl.column("q_len"), f64), 1.0),
        )
        cov_t = pc.divide(
            pc.cast(pc.subtract(tbl.column("t_end"), tbl.column("t_start")), f64),
            pc.max_element_wise(pc.cast(tbl.column("t_len"), f64), 1.0),
        )
        keep = pc.and_(
            pc.greater_equal(identity, min_identity),
            pc.and_(
                pc.greater_equal(cov_q, min_mutual_coverage),
                pc.greater_equal(cov_t, min_mutual_coverage),
            ),
        )
        sel = pa.table(
            {
                "a_row": pc.cast(tbl.column("q_name"), pa.int64()),
                "b_row": pc.cast(tbl.column("t_name"), pa.int64()),
                "identity": identity,
                "coverage_a": cov_q,
                "coverage_b": cov_t,
            },
            schema=REDUNDANT_PAIR_TABLE,
        ).filter(keep)
        if sel.num_rows:
            rows.append(sel)
    if not rows:
        return REDUNDANT_PAIR_TABLE.empty_table()
    return pa.concat_tables(rows).combine_chunks()


def _pair_edges(
    pairs: pa.Table, n: int, p_merge: float
) -> tuple[np.ndarray, np.ndarray]:
    """The pairs at or above ``p_merge``, as in-range edge arrays."""
    a = pairs.column("a_row").to_numpy(zero_copy_only=False).astype(np.int64)
    b = pairs.column("b_row").to_numpy(zero_copy_only=False).astype(np.int64)
    ident = pairs.column("identity").to_numpy(zero_copy_only=False)
    ok = (a >= 0) & (a < n) & (b >= 0) & (b < n) & (a != b) & (ident >= p_merge)
    return a[ok], b[ok]


#: Identity-histogram bin edges for :func:`redundancy_stats`. The 0.99-0.995
#: band is below the merge threshold and reported so the near-threshold
#: population (where a real single-position variant lives) stays visible.
_IDENTITY_BINS = (0.99, 0.995, 0.998, 0.999, 0.9995, 1.0 - 1e-12, 1.0 + 1e-12)


def redundancy_stats(
    pairs: pa.Table, n_templates: int, *, p_merge: float = 0.995
) -> dict:
    """The handoff's redundancy numbers for one template set.

    ``removable`` is what collapsing every connected component to one would
    remove — the upper bound, and the number that answers "should merge
    exist". The merge itself uses radius-1 greedy (:func:`select_merges`), so
    one pass may remove less; the next round catches the rest.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.cluster_graph import (
        connected_components,
    )

    n = int(n_templates)
    a, b = _pair_edges(pairs, n, p_merge)
    ident = pairs.column("identity").to_numpy(zero_copy_only=False)
    counts, _ = np.histogram(ident, bins=np.asarray(_IDENTITY_BINS))
    labels = ["0.990-0.995", "0.995-0.998", "0.998-0.999", "0.999-0.9995",
              "0.9995-<1", "1.0"]
    out: dict = {
        "n_templates": n,
        "p_merge": float(p_merge),
        "n_pairs": int(a.size),
        "n_near_threshold_pairs": int(((ident < p_merge)).sum()),
        "identity_hist": {k: int(v) for k, v in zip(labels, counts)},
    }
    if a.size == 0 or n == 0:
        out.update(
            n_in_pairs=0,
            n_components=0,
            component_size_median=0.0,
            component_size_p90=0.0,
            component_size_max=0,
            removable=0,
            removable_frac=0.0,
        )
        return out
    comp = connected_components(n, np.ones(n), np.ones(n), a, b).cluster_of
    involved = np.zeros(n, dtype=bool)
    involved[a] = True
    involved[b] = True
    sizes = np.bincount(comp[involved])
    sizes = sizes[sizes > 1]
    removable = int(sizes.sum() - sizes.size)
    out.update(
        n_in_pairs=int(involved.sum()),
        n_components=int(sizes.size),
        component_size_median=float(np.median(sizes)) if sizes.size else 0.0,
        component_size_p90=float(np.percentile(sizes, 90)) if sizes.size else 0.0,
        component_size_max=int(sizes.max()) if sizes.size else 0,
        removable=removable,
        removable_frac=removable / n,
    )
    return out


def select_merges(
    pairs: pa.Table,
    n_reads: np.ndarray,
    seq_len: np.ndarray,
    *,
    p_merge: float = 0.995,
) -> np.ndarray:
    """``survivor_of[i]`` — the row template ``i`` merges into (itself if none).

    Radius-1 greedy set cover in ``(reads desc, length desc, row asc)`` order:
    each unclaimed template claims its unclaimed *direct* redundant
    neighbours, so a chain ``A~B~C`` never merges ``A`` with ``C`` (whose
    identity to each other may be below ``p_merge``). The survivor is the
    group's claimant — its best-supported member.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.cluster_graph import (
        greedy_set_cover,
    )

    n = int(np.asarray(n_reads).size)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    a, b = _pair_edges(pairs, n, p_merge)
    res = greedy_set_cover(
        n,
        np.maximum(np.asarray(n_reads, dtype=np.int64), 0),
        np.asarray(seq_len, dtype=np.int64),
        a,
        b,
    )
    return res.centroid_uniq[res.cluster_of].astype(np.int64)


def apply_merge(
    refined: RefineResult, pairs: pa.Table, *, p_merge: float = 0.995
) -> RefineResult:
    """Collapse the redundant templates of ``refined`` into their survivors.

    ``pairs`` rows index ``refined.templates``. Child ids are stable — the
    table is filtered, never re-indexed — so survivors keep their ids and an
    absorbed template's lineage row is re-pointed at its survivor with
    ``rule='merge'``: the edge ``(survivor, absorbed parent)`` then makes a
    read moving from the absorbed parent to the survivor count as inherited,
    exactly like a split.
    """
    t = refined.templates
    n = t.num_rows
    if n == 0 or pairs.num_rows == 0:
        return refined
    reads = t.column("orf_replication").to_numpy(zero_copy_only=False)
    seq_len = pc.utf8_length(t.column("sequence")).to_numpy(zero_copy_only=False)
    survivor = select_merges(pairs, reads, seq_len, p_merge=p_merge)
    absorbed = survivor != np.arange(n)
    if not absorbed.any():
        return refined

    weight = t.column("node_weight").to_numpy(zero_copy_only=False).astype(float)
    merged_weight = np.bincount(survivor, weights=weight, minlength=n)
    merged_reads = np.bincount(
        survivor, weights=np.maximum(reads, 0).astype(float), minlength=n
    )
    keep = ~absorbed
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
        .filter(pa.array(keep))
    )

    ids = t.column("template_id").to_numpy(zero_copy_only=False)
    remap = dict(zip(ids[absorbed].tolist(), ids[survivor[absorbed]].tolist()))
    lin = refined.lineage
    child = lin.column("child_template_id").to_numpy(zero_copy_only=False).copy()
    rule = np.asarray(lin.column("rule").to_pylist(), dtype=object)
    for i, c in enumerate(child.tolist()):
        s = remap.get(c)
        if s is not None:
            child[i] = s
            rule[i] = "merge"
    lineage = lin.set_column(
        lin.schema.get_field_index("child_template_id"),
        "child_template_id",
        pa.array(child, pa.int64()),
    ).set_column(
        lin.schema.get_field_index("rule"), "rule", pa.array(list(rule), pa.string())
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
    )


def merge_nodes(
    nodes: pa.Table,
    node_membership: pa.Table,
    pairs: pa.Table,
    *,
    p_merge: float = 0.995,
) -> tuple[pa.Table, pa.Table, int]:
    """The final-output merge: collapse redundant NODES and their membership.

    ``pairs`` rows index ``nodes``. Absorbed nodes are dropped and their
    membership rows re-keyed to the survivor's ``(parent_template_id,
    haplotype_id)``; counts and quant are computed from membership, so every
    read is kept and totals are conserved. Returns ``(nodes, membership,
    n_merged)``.
    """
    n = nodes.num_rows
    if n == 0 or pairs.num_rows == 0:
        return nodes, node_membership, 0
    n_reads = nodes.column("n_reads").to_numpy(zero_copy_only=False)
    seq_len = pc.utf8_length(nodes.column("consensus")).to_numpy(zero_copy_only=False)
    survivor = select_merges(pairs, n_reads, seq_len, p_merge=p_merge)
    absorbed = survivor != np.arange(n)
    if not absorbed.any():
        return nodes, node_membership, 0

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
    return out, membership, int(absorbed.sum())


__all__ = [
    "LINEAGE_TABLE",
    "MERGE_MINIMAP2_ARGS",
    "REDUNDANT_PAIR_TABLE",
    "REPORT_IDENTITY_FLOOR",
    "RefineResult",
    "apply_merge",
    "detect_redundant_templates",
    "merge_nodes",
    "redundancy_stats",
    "select_merges",
    "measure_churn",
    "next_templates",
    "template_id_for",
]
