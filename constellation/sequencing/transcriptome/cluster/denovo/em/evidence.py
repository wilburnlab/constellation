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

#: Output keys of the M-step worker that are always written.
ACCOUNTING_KEYS: tuple[str, ...] = ("templates", "disagreements")

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
