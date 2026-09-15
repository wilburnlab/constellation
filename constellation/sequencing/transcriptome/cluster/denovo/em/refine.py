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

**There is no merge either.** Merge existed to undo self-capture (ledger #25
is explicit: "the between-round merge is the structural fix"), and the
round-1 rule removes that at the source: redundant templates lose their reads
at assignment time. The two worked examples resolve without it — Tcp1 is
separated by the covariance M-step, and H3f3b's elongated frameshift template
loses round 1 outright to the more-replicated correct form.

What remains is the *detector*: :func:`detect_redundant_templates` reports
pairs that a merge would have collapsed, so the decision to drop the
operation is checkable rather than assumed. If the count is negligible, merge
was correctly dropped; if it is not, merge returns with evidence behind its
criterion.
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
        edges = _pack(
            lineage.column("child_template_id").to_numpy(zero_copy_only=False),
            lineage.column("parent_template_id").to_numpy(zero_copy_only=False),
        )
        edges = np.unique(edges)
        probe = _pack(cur[moved], prev[moved])
        pos = np.searchsorted(edges, probe)
        ok = (pos < edges.size) & (edges[np.clip(pos, 0, edges.size - 1)] == probe)
        n_inherited = int(ok.sum())

    genuine = n_moved - n_inherited
    return {
        "n_compared": n_both,
        "n_moved_raw": n_moved,
        "n_moved_inherited": n_inherited,
        "n_moved_genuine": genuine,
        "frac_changed": (n_moved / n_both) if n_both else 0.0,
        "frac_changed_lineage": (genuine / n_both) if n_both else 0.0,
        "n_gained": int((~(prev >= 0) & (cur >= 0)).sum()),
        "n_lost": int(((prev >= 0) & ~(cur >= 0)).sum()),
    }


def _scatter(out: np.ndarray, table: pa.Table) -> None:
    if table.num_rows == 0:
        return
    rows = table.column("read_row").to_numpy(zero_copy_only=False).astype(np.int64)
    tid = table.column("template_id").to_numpy(zero_copy_only=False).astype(np.int64)
    keep = (rows >= 0) & (rows < out.size) & (tid >= 0)
    out[rows[keep]] = tid[keep]


def _pack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A sortable key over an (id, id) pair, without a Python tuple per row."""
    return (np.asarray(a, dtype=np.int64) << np.int64(21)) ^ np.asarray(
        b, dtype=np.int64
    )


# ── the redundancy detector (B2) ──────────────────────────────────────

MERGE_MINIMAP2_ARGS: tuple[str, ...] = (
    "-x",
    "asm5",
    # -c is MANDATORY, not an optimisation: without base-level alignment PAF
    # column 10 is chain-anchor matches, which underestimates identity badly
    # and makes a 0.995 threshold meaningless.
    "-c",
    # -X (= -DP --dual=no) skips self-hits and reports each unordered pair
    # once, halving the work and removing the q_name == t_name filter.
    "-X",
    "--secondary=no",
    "-p",
    "0.9",
    "-N",
    "5",
)


def detect_redundant_templates(
    templates_fasta: Path,
    *,
    p_merge: float = 0.995,
    min_mutual_coverage: float = 0.95,
    threads: int = 8,
) -> pa.Table:
    """Pairs a merge would have collapsed. **Collapses nothing.**

    Runs on the final round only, where the template count is smallest. Why
    ``asm5`` rather than ``map-ont`` is not a tuning preference: a preset sets
    the seed length and how densely seeds are sampled. ``map-ont`` uses
    k=15/w=10 because a noisy read has an error every ~100 bp and only a short
    seed has a good chance of being error-free — at the cost of specificity,
    since a 15-mer recurs by chance across a transcriptome. ``asm5`` uses
    k=19/w=19, for sequences that are accurate and <=5% divergent: a 19-mer is
    ~16x more specific and accurate sequences share long exact stretches, so
    it can sample ~2x more sparsely. Template-vs-template at >=99.5% is the
    second situation.

    Needs **no CIGAR**: identity and both coverages come from PAF columns that
    are already parsed and otherwise unused.
    """
    from constellation.sequencing.align.minimap2 import minimap2_stream
    from constellation.sequencing.readers.paf import iter_paf_batches

    stream = minimap2_stream(
        Path(templates_fasta),
        [Path(templates_fasta)],
        args=MERGE_MINIMAP2_ARGS,
        threads=threads,
    )
    rows: list[pa.Table] = []
    for batch in iter_paf_batches(stream, want_cigar=False):
        tbl = pa.Table.from_batches([batch])
        identity = pc.divide(
            pc.cast(tbl.column("n_match"), pa.float64()),
            pc.max_element_wise(pc.cast(tbl.column("aln_len"), pa.float64()), 1.0),
        )
        cov_q = pc.divide(
            pc.cast(
                pc.subtract(tbl.column("q_end"), tbl.column("q_start")), pa.float64()
            ),
            pc.max_element_wise(pc.cast(tbl.column("q_len"), pa.float64()), 1.0),
        )
        cov_t = pc.divide(
            pc.cast(
                pc.subtract(tbl.column("t_end"), tbl.column("t_start")), pa.float64()
            ),
            pc.max_element_wise(pc.cast(tbl.column("t_len"), pa.float64()), 1.0),
        )
        keep = pc.and_(
            pc.greater_equal(identity, p_merge),
            pc.and_(
                pc.greater_equal(cov_q, min_mutual_coverage),
                pc.greater_equal(cov_t, min_mutual_coverage),
            ),
        )
        sel = pa.table(
            {
                "a_name": tbl.column("q_name"),
                "b_name": tbl.column("t_name"),
                "identity": identity,
                "coverage_a": cov_q,
                "coverage_b": cov_t,
            }
        ).filter(keep)
        if sel.num_rows:
            rows.append(sel)
    if not rows:
        return pa.table(
            {
                "a_name": pa.array([], pa.large_string()),
                "b_name": pa.array([], pa.large_string()),
                "identity": pa.array([], pa.float64()),
                "coverage_a": pa.array([], pa.float64()),
                "coverage_b": pa.array([], pa.float64()),
            }
        )
    return pa.concat_tables(rows)


__all__ = [
    "LINEAGE_TABLE",
    "MERGE_MINIMAP2_ARGS",
    "RefineResult",
    "detect_redundant_templates",
    "measure_churn",
    "next_templates",
    "template_id_for",
]
