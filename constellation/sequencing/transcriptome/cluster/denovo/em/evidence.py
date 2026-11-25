"""What the M-step decided, per template, kept on disk.

Until this module the M-step computed a PWM, column statistics, candidate
columns, a covariance graph, signatures and per-read patterns for every
template — and wrote only the nodes and which reads they hold. Even the
counters ``refine_template`` assembled so that "the method returned one
template" would be *visible rather than inferred from a quiet result* were
built and dropped. The diagnosis of every recent edge case had to rebuild
them by hand.

So each round now writes, beside ``mstep/nodes`` and
``mstep/node_membership``, one row per template it refined
(:data:`MSTEP_TEMPLATE_TABLE`: what happened and every counter), and the
minor-allele mass at high-confidence columns
(:data:`MSTEP_DISAGREEMENT_TABLE`), the substrate a per-round error-model
refit would be fitted and judged on. Both are small — one row per template,
a handful per template — and are always written. The heavier evidence
(candidate columns, covariance edges, per-read margins) is the compact
evidence of ``--mstep-evidence``; the dense evidence (the PWM, every read's
alleles) is never written by a run: it is re-created on demand by replaying
one template through the same code (:mod:`.replay`).

Collecting evidence never changes a result: the nodes are built first, and
the evidence is assembled afterwards, in its own ``try``, from references
the kernel already held (:class:`~.mstep.KernelInternals`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from constellation.core.io.schemas import register_schema

#: Bumped when what an evidence table means changes.
EVIDENCE_VERSION = 1

# ── what happened to a template ───────────────────────────────────────

#: The kernel ran and emitted its nodes.
STATUS_OK = "ok"
#: The kernel raised; the template's reads were pooled into one node by the
#: consensus step alone (``pooled_node``), with no split.
STATUS_FALLBACK_POOLED = "fallback_pooled"
#: The kernel raised and so did the pooled consensus (or it covered
#: nothing): the template carried forward as it was, holding every read.
STATUS_FALLBACK_TEMPLATE = "fallback_template"
#: No assigned read carried an alignment (every CIGAR null or empty), so
#: there was nothing to build a consensus from; the template carried forward
#: holding its reads. Before this record such a template's reads vanished
#: from the round.
STATUS_NO_MEMBERS = "no_members"

STATUSES = (
    STATUS_OK,
    STATUS_FALLBACK_POOLED,
    STATUS_FALLBACK_TEMPLATE,
    STATUS_NO_MEMBERS,
)

#: ``evidence_status``: evidence was not asked for, was collected, or its
#: collection raised (which never changes the nodes — they exist by then).
EVIDENCE_NONE = "none"
EVIDENCE_OK = "ok"
EVIDENCE_FAILED = "failed"

#: The kernel's own counters, in the order the record carries them. A
#: counter the kernel did not reach (no candidates → no covariance pass) is
#: null, not zero: "not tested" and "tested, none significant" differ.
KERNEL_COUNTERS: tuple[str, ...] = (
    "n_columns",
    "n_candidates",
    "n_allelic",
    "n_coverage",
    "n_collapsed",
    "n_candidates_capped",
    # Columns a route would have admitted that the M-step ruled out as
    # terminal extensions (2026-11-24); null in rounds recorded before.
    "n_excluded_extension",
    "n_pairs_seen",
    "n_tested_pairs",
    "n_significant_edges",
    "n_retained",
    "n_columns_dropped",
    "sum_k2",
    "n_signatures",
    "n_tuples_observed",
    "n_nodes",
)

MSTEP_TEMPLATE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("round", pa.int32(), nullable=False),
        pa.field("template_id", pa.int64(), nullable=False),
        pa.field("template_row", pa.int32(), nullable=False),
        # Every read the E-step assigned to the template, sampled or not.
        pa.field("n_assigned", pa.int64(), nullable=False),
        # Reads that voted in the PWM: those with an alignment, capped by
        # max_members_per_template (a uniform sample, seeded by template).
        pa.field("n_members_used", pa.int32(), nullable=False),
        pa.field("subsample_fraction", pa.float32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("error_type", pa.string(), nullable=True),
        pa.field("error_message", pa.string(), nullable=True),
        pa.field("evidence_status", pa.string(), nullable=False),
        # Nodes the template emitted, whatever the status.
        pa.field("n_nodes_emitted", pa.int32(), nullable=False),
        *(pa.field(name, pa.int64(), nullable=True) for name in KERNEL_COUNTERS),
    ],
    metadata={b"schema_name": b"EmMStepTemplateTable"},
)
register_schema("EmMStepTemplateTable", MSTEP_TEMPLATE_TABLE)

#: ``variants.disagreement_stats`` in long form: at columns where the
#: consensus is unambiguous and deep, the minor mass is error, binned by the
#: column's context. ``class_code`` 0 = substitution, 1 = homopolymer indel,
#: 2 = non-homopolymer indel; ``run_len`` is the homopolymer run (0 otherwise).
MSTEP_DISAGREEMENT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("round", pa.int32(), nullable=False),
        pa.field("template_id", pa.int64(), nullable=False),
        pa.field("class_code", pa.int8(), nullable=False),
        pa.field("run_len", pa.int32(), nullable=False),
        pa.field("minor_sum", pa.float64(), nullable=False),
        pa.field("total_sum", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"EmMStepDisagreementTable"},
)
register_schema("EmMStepDisagreementTable", MSTEP_DISAGREEMENT_TABLE)

# ── compact evidence (``--mstep-evidence compact``, the default) ──────

#: One row per candidate column: what the column showed, why it was a
#: candidate, and what the covariance pass made of it. Columns are PWM
#: (frame) columns of the template's pooled plan; ``template_col`` is -1 on
#: an insertion or extension column, which the template does not have.
MSTEP_COLUMN_TABLE: pa.Schema = pa.schema(
    [
        pa.field("round", pa.int32(), nullable=False),
        pa.field("template_id", pa.int64(), nullable=False),
        pa.field("candidate", pa.int32(), nullable=False),
        pa.field("frame_col", pa.int32(), nullable=False),
        pa.field("template_col", pa.int32(), nullable=False),
        # columns.ROUTE_* bits: 1 allelic, 2 coverage.
        pa.field("route", pa.int8(), nullable=False),
        # Admitted columns this representative stands for (exact run collapse).
        pa.field("run_len", pa.int32(), nullable=False),
        # 0-3 = ACGT, 4 = gap.
        pa.field("major", pa.int8(), nullable=False),
        pa.field("minor", pa.int8(), nullable=False),
        pa.field("depth", pa.float64(), nullable=False),
        pa.field("base_cov", pa.float64(), nullable=False),
        pa.field("minor_frac", pa.float64(), nullable=False),
        pa.field("uncov_frac", pa.float64(), nullable=False),
        pa.field("hp_run", pa.int32(), nullable=False),
        # The context-conditional null used for this column, after the floor.
        pa.field("eps", pa.float64(), nullable=False),
        pa.field("effect", pa.float64(), nullable=False),
        # Allelic route only; 1.0 elsewhere.
        pa.field("p_value", pa.float64(), nullable=False),
        pa.field("boundary_mass", pa.float64(), nullable=False),
        pa.field("boundary_expected", pa.float64(), nullable=False),
        # A significant covariance edge touches it.
        pa.field("kept", pa.bool_(), nullable=False),
        # False when the co-occurrence budget dropped it from the pass.
        pa.field("budget_kept", pa.bool_(), nullable=False),
        # Index of the signature (quasi-clique) it belongs to, -1 for none.
        pa.field("signature", pa.int32(), nullable=False),
    ],
    metadata={b"schema_name": b"EmMStepColumnTable"},
)
register_schema("EmMStepColumnTable", MSTEP_COLUMN_TABLE)

#: One row per SIGNIFICANT covariance edge, with the 2x2 it was tested on
#: (weighted counts) and its effect size, so an arc can be drawn by effect
#: rather than by a p-value that grows with depth. A self-edge is a collapsed
#: coverage run judged on its delimiting boundary mass: no table, null counts.
MSTEP_COVARIANCE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("round", pa.int32(), nullable=False),
        pa.field("template_id", pa.int64(), nullable=False),
        pa.field("u", pa.int32(), nullable=False),
        pa.field("v", pa.int32(), nullable=False),
        pa.field("u_frame_col", pa.int32(), nullable=False),
        pa.field("v_frame_col", pa.int32(), nullable=False),
        pa.field("self_edge", pa.bool_(), nullable=False),
        pa.field("n11", pa.float64(), nullable=True),
        pa.field("n1x", pa.float64(), nullable=True),
        pa.field("nx1", pa.float64(), nullable=True),
        pa.field("n", pa.float64(), nullable=True),
        # Phi coefficient of the 2x2, signed: + co-occurring, - exclusive.
        pa.field("phi", pa.float64(), nullable=True),
        # Haldane-corrected log odds ratio.
        pa.field("log_odds", pa.float64(), nullable=True),
        pa.field("p_value", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"EmMStepCovarianceTable"},
)
register_schema("EmMStepCovarianceTable", MSTEP_COVARIANCE_TABLE)

#: How sure the split was about each read — written only for templates whose
#: covariance produced a signature with at least two patterns, i.e. where
#: there was a choice to be sure of. Every other read's margin is null by
#: definition, and the table stays small.
MSTEP_READ_TABLE: pa.Schema = pa.schema(
    [
        pa.field("round", pa.int32(), nullable=False),
        pa.field("template_id", pa.int64(), nullable=False),
        pa.field("read_row", pa.int32(), nullable=False),
        pa.field("haplotype_id", pa.int32(), nullable=False),
        # Voted in the PWM (False: beyond max_members_per_template, given to
        # the major node without evidence of its own).
        pa.field("sampled", pa.bool_(), nullable=False),
        # Its tuple of patterns was too light to be a node and was sent to
        # the nearest surviving one; null for an unsampled read.
        pa.field("folded", pa.bool_(), nullable=True),
        # Smallest best-minus-second-best log-likelihood over the template's
        # multi-pattern signatures (nats); null for an unsampled read.
        pa.field("margin", pa.float32(), nullable=True),
    ],
    metadata={b"schema_name": b"EmMStepReadTable"},
)
register_schema("EmMStepReadTable", MSTEP_READ_TABLE)

#: Output keys of the M-step worker that are always written.
ACCOUNTING_KEYS: tuple[str, ...] = ("templates", "disagreements")
#: ...and the compact evidence's.
COMPACT_KEYS: tuple[str, ...] = ("columns", "covariance", "reads")
EVIDENCE_MODES: tuple[str, ...] = ("compact", "none")


def evidence_keys(evidence: str) -> tuple[str, ...]:
    """The M-step output keys a run under ``--mstep-evidence`` writes."""
    if evidence == "none":
        return ACCOUNTING_KEYS
    if evidence in ("compact", "full"):
        return (*ACCOUNTING_KEYS, *COMPACT_KEYS)
    raise ValueError(f"unknown evidence mode {evidence!r}")


_EMPTY_COMPACT: dict[str, pa.Table] | None = None


def empty_compact() -> dict[str, pa.Table]:
    """The three compact tables, empty. Built once: Arrow tables are
    immutable, and building them per template cost ~0.4 ms each time —
    1.7 s over 4,000 templates, minutes a round at full scale."""
    global _EMPTY_COMPACT
    if _EMPTY_COMPACT is None:
        _EMPTY_COMPACT = {
            "columns": MSTEP_COLUMN_TABLE.empty_table(),
            "covariance": MSTEP_COVARIANCE_TABLE.empty_table(),
            "reads": MSTEP_READ_TABLE.empty_table(),
        }
    return dict(_EMPTY_COMPACT)


def _phi_log_odds(n11, n1x, nx1, n):
    n10 = n1x - n11
    n01 = nx1 - n11
    n00 = n - n1x - nx1 + n11
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sqrt(n1x * (n - n1x) * nx1 * (n - nx1))
        phi = np.where(den > 0, (n11 * n00 - n10 * n01) / den, np.nan)
        lor = np.log(((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5)))
    return phi, lor


def _nullable(values: np.ndarray, kind) -> pa.Array:
    values = np.asarray(values, dtype=np.float64)
    return pa.array(values, type=kind, mask=~np.isfinite(values))


def compact_evidence(
    *,
    round_index: int,
    template_id: int,
    internals,
    read_row: np.ndarray,
    haplotype_of_read: np.ndarray,
    member_pos: np.ndarray,
) -> dict[str, pa.Table]:
    """One template's compact evidence, as small tables, from the kernel's
    working state. Copies, never writes into it: the same arrays built the
    nodes, which exist by now."""
    cand = internals.cand
    stats = internals.stats
    pooled = internals.pooled
    graph = internals.graph
    v = int(np.asarray(cand.columns).size)
    out = empty_compact()
    if v == 0:
        return out
    cols = np.asarray(cand.columns, dtype=np.int64)
    tof = np.asarray(pooled.template_of_frame, dtype=np.int64)
    signature = np.full(v, -1, dtype=np.int32)
    for k, members in enumerate(internals.signatures):
        signature[np.asarray(members, dtype=np.int64)] = k
    keep = (
        np.asarray(graph.keep, dtype=bool) if graph is not None else np.zeros(v, bool)
    )
    budget = (
        np.asarray(graph.budget_kept, dtype=bool)
        if graph is not None and graph.budget_kept is not None
        else np.ones(v, bool)
    )
    out["columns"] = pa.table(
        {
            "round": pa.array(np.full(v, round_index, np.int32)),
            "template_id": pa.array(np.full(v, template_id, np.int64)),
            "candidate": pa.array(np.arange(v, dtype=np.int32)),
            "frame_col": pa.array(cols.astype(np.int32)),
            "template_col": pa.array(tof[cols].astype(np.int32)),
            "route": pa.array(np.asarray(cand.route).astype(np.int8)),
            "run_len": pa.array(np.asarray(cand.run_len).astype(np.int32)),
            "major": pa.array(np.asarray(stats.major)[cols].astype(np.int8)),
            "minor": pa.array(np.asarray(stats.minor)[cols].astype(np.int8)),
            "depth": pa.array(np.asarray(stats.depth, np.float64)[cols]),
            "base_cov": pa.array(np.asarray(stats.base_cov, np.float64)[cols]),
            "minor_frac": pa.array(np.asarray(stats.minor_frac, np.float64)[cols]),
            "uncov_frac": pa.array(np.asarray(stats.uncov_frac, np.float64)[cols]),
            "hp_run": pa.array(np.asarray(stats.hp_run)[cols].astype(np.int32)),
            "eps": pa.array(np.asarray(cand.eps, np.float64)),
            "effect": pa.array(np.asarray(cand.effect, np.float64)),
            "p_value": pa.array(np.asarray(cand.p_value, np.float64)),
            "boundary_mass": pa.array(np.asarray(cand.boundary_mass, np.float64)),
            "boundary_expected": pa.array(
                np.asarray(cand.boundary_expected, np.float64)
            ),
            "kept": pa.array(keep),
            "budget_kept": pa.array(budget),
            "signature": pa.array(signature),
        },
        schema=MSTEP_COLUMN_TABLE,
    )
    if graph is not None and len(graph.edges):
        edges = np.asarray(graph.edges, dtype=np.int64)
        e = edges.shape[0]
        u, w = edges[:, 0], edges[:, 1]
        counts = [
            np.asarray(a, dtype=np.float64) if np.asarray(a).size == e
            else np.full(e, np.nan)
            for a in (graph.edge_n11, graph.edge_n1x, graph.edge_nx1, graph.edge_n)
        ]
        phi, lor = _phi_log_odds(*counts)
        out["covariance"] = pa.table(
            {
                "round": pa.array(np.full(e, round_index, np.int32)),
                "template_id": pa.array(np.full(e, template_id, np.int64)),
                "u": pa.array(u.astype(np.int32)),
                "v": pa.array(w.astype(np.int32)),
                "u_frame_col": pa.array(cols[u].astype(np.int32)),
                "v_frame_col": pa.array(cols[w].astype(np.int32)),
                "self_edge": pa.array(u == w),
                "n11": _nullable(counts[0], pa.float64()),
                "n1x": _nullable(counts[1], pa.float64()),
                "nx1": _nullable(counts[2], pa.float64()),
                "n": _nullable(counts[3], pa.float64()),
                "phi": _nullable(phi, pa.float64()),
                "log_odds": _nullable(lor, pa.float64()),
                "p_value": pa.array(np.asarray(graph.p_value, np.float64)),
            },
            schema=MSTEP_COVARIANCE_TABLE,
        )
    multi = [a for a in internals.assignments if np.asarray(a.patterns).shape[0] >= 2]
    if multi:
        m = int(np.asarray(member_pos).size)
        margin_m = np.min(
            np.stack([np.asarray(a.margin, np.float64) for a in multi], axis=0), axis=0
        )
        if internals.tuple_of_member is not None and internals.surviving_tuples is not None:
            folded_m = ~np.isin(
                np.asarray(internals.tuple_of_member), np.asarray(internals.surviving_tuples)
            )
        else:
            folded_m = np.zeros(m, dtype=bool)
        n = int(np.asarray(read_row).size)
        margin = np.full(n, np.nan)
        folded = np.zeros(n, dtype=bool)
        sampled = np.zeros(n, dtype=bool)
        pos = np.asarray(member_pos, dtype=np.int64)
        margin[pos] = margin_m
        folded[pos] = folded_m
        sampled[pos] = True
        out["reads"] = pa.table(
            {
                "round": pa.array(np.full(n, round_index, np.int32)),
                "template_id": pa.array(np.full(n, template_id, np.int64)),
                "read_row": pa.array(np.asarray(read_row).astype(np.int32)),
                "haplotype_id": pa.array(np.asarray(haplotype_of_read).astype(np.int32)),
                "sampled": pa.array(sampled),
                "folded": pa.array(folded, mask=~sampled),
                "margin": pa.array(margin.astype(np.float32), mask=~sampled),
            },
            schema=MSTEP_READ_TABLE,
        )
    return out

#: Bounded so a pathological message cannot bloat the table.
_MAX_ERROR_MESSAGE = 500


@dataclass(slots=True)
class TemplateRecord:
    """One template's row of :data:`MSTEP_TEMPLATE_TABLE`, plus its
    disagreement statistics."""

    round: int
    template_id: int
    template_row: int
    n_assigned: int
    n_members_used: int = 0
    subsample_fraction: float = 1.0
    status: str = STATUS_OK
    error_type: str | None = None
    error_message: str | None = None
    evidence_status: str = EVIDENCE_NONE
    n_nodes_emitted: int = 0
    counters: dict = field(default_factory=dict)
    #: ``{(class_code, run_len): (minor_sum, total_sum)}``.
    disagreements: dict = field(default_factory=dict)

    def record_error(self, exc: BaseException) -> None:
        self.error_type = type(exc).__name__
        self.error_message = str(exc)[:_MAX_ERROR_MESSAGE]


def template_table(records: list[TemplateRecord]) -> pa.Table:
    if not records:
        return MSTEP_TEMPLATE_TABLE.empty_table()
    cols: dict[str, list] = {
        "round": [r.round for r in records],
        "template_id": [r.template_id for r in records],
        "template_row": [r.template_row for r in records],
        "n_assigned": [r.n_assigned for r in records],
        "n_members_used": [r.n_members_used for r in records],
        "subsample_fraction": [r.subsample_fraction for r in records],
        "status": [r.status for r in records],
        "error_type": [r.error_type for r in records],
        "error_message": [r.error_message for r in records],
        "evidence_status": [r.evidence_status for r in records],
        "n_nodes_emitted": [r.n_nodes_emitted for r in records],
    }
    for name in KERNEL_COUNTERS:
        cols[name] = [
            None if r.counters.get(name) is None else int(r.counters[name])
            for r in records
        ]
    return pa.table(cols, schema=MSTEP_TEMPLATE_TABLE)


def disagreement_table(records: list[TemplateRecord]) -> pa.Table:
    rows = [
        (r.round, r.template_id, int(cls), int(run), float(minor), float(total))
        for r in records
        for (cls, run), (minor, total) in sorted(r.disagreements.items())
    ]
    if not rows:
        return MSTEP_DISAGREEMENT_TABLE.empty_table()
    cols = list(zip(*rows))
    return pa.table(
        {f.name: pa.array(cols[i], type=f.type) for i, f in enumerate(MSTEP_DISAGREEMENT_TABLE)},
        schema=MSTEP_DISAGREEMENT_TABLE,
    )


__all__ = [
    "ACCOUNTING_KEYS",
    "COMPACT_KEYS",
    "EVIDENCE_MODES",
    "MSTEP_COLUMN_TABLE",
    "MSTEP_COVARIANCE_TABLE",
    "MSTEP_READ_TABLE",
    "compact_evidence",
    "empty_compact",
    "evidence_keys",
    "EVIDENCE_FAILED",
    "EVIDENCE_NONE",
    "EVIDENCE_OK",
    "EVIDENCE_VERSION",
    "KERNEL_COUNTERS",
    "MSTEP_DISAGREEMENT_TABLE",
    "MSTEP_TEMPLATE_TABLE",
    "STATUSES",
    "STATUS_FALLBACK_POOLED",
    "STATUS_FALLBACK_TEMPLATE",
    "STATUS_NO_MEMBERS",
    "STATUS_OK",
    "TemplateRecord",
    "disagreement_table",
    "template_table",
]
