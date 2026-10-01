"""The round loop — the engine the EM stages were missing.

``seed`` / ``fold`` / E-step / M-step all shipped; nothing iterated them, so
none of it was reachable and the policy could only be measured by a driver
outside the package. This is that driver, in-tree and on the project's own
data rules.

Per round::

    templates -> E-step -> assignments -> M-step -> nodes -> refine -> templates
                                                      |         |
                                                    graph    lineage

and convergence is the lineage-aware read-switch fraction, so a read that
moved only because its template split does not count as churn.

**The graph is a report, and merging is a separate decision.** After the
M-step each round's nodes are related to each other (:mod:`.graph`): which are
the same transcript within an end tolerance, which are contained in which. By
default that is all that happens — ``rounds/rNN/graph/edges.parquet`` is
written and nothing is collapsed. ``merge=True`` collapses the pairs the merge
predicate accepts, between rounds and in the final output;
``template_graph="off"`` runs nothing at all after the M-step. One graph per
round, over that round's *nodes*, serves both: :func:`.refine.next_templates`
turns nodes into next-round templates one for one, so "after every M-step" and
"on the final nodes" are the same sequences.

Because a merge changes every later round's templates it is a commitment of
the run, recorded per round in ``refine.json`` and checked on resume
(:func:`_check_merge_state`). A graph that merged nothing changes nothing, so
its parameters may change freely and its failure never sinks a round.

Everything on disk is addressable and every round is ``_SUCCESS``-gated, so a
resumed run restarts after the last complete round and rebuilds identical
template ids (they are ``(round << 40) | row``, not a counter).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

from constellation.sequencing.parallel import run_batched
from constellation.sequencing.transcriptome.cluster.denovo.em import graph as gr
from constellation.sequencing.transcriptome.cluster.denovo.em import refine as rf
from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
    EM_ASSIGNMENT_TABLE,
    run_em_estep,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
    ReadStore,
    write_corpus,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.fold import fold_orfs
from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
    GraphParams,
    MergePredicate,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.outputs import (
    CLUSTER_EDGES,
    build_cluster_tables,
    write_cluster_edges,
    write_em_manifest,
    write_em_outputs,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
    NODE_MEMBERSHIP_TABLE,
    REFINED_NODE_TABLE,
    MStepParams,
    in_membership_order,
    in_node_order,
    iter_unit_batches,
    mstep_worker,
    plan_mstep_units,
    sort_assignments_by_template,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.rebuild import (
    rebuild_survivors,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.seed import (
    extract_seed_orfs,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.seed_kmer import (
    DEFAULT_MAX_CLUSTER_READ_FRAC,
    DEFAULT_MIN_CHAIN_CLUSTER_READS,
    DEFAULT_SEED_IDENTITY,
    DEFAULT_SEED_MAX_3P,
    DEFAULT_SEED_MAX_5P,
    seed_by_kmer_clustering,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
    TEMPLATE_TABLE,
    TEMPLATES_ARROW,
    TEMPLATES_FASTA,
    TemplateStore,
    templates_from_seeds,
    write_templates,
)

_SUCCESS = "_SUCCESS"


@dataclass(frozen=True, slots=True)
class EmParams:
    """Everything the loop schedules, in one picklable object."""

    rounds: int = 6
    stop_frac_changed: float = 0.005
    # The pool (A1) — a correctness parameter, not a cost one.
    p_floor: float = 0.97
    minimap2_n: int = 500
    index_batch_size: str = "16G"
    # The E-step's base aligner. "minimap2" is the single pass (`-c` on every
    # candidate); "edlib" shortlists on minimap2's chaining score and
    # base-aligns only the shortlist (`em/realign.py`). The shortlist size is
    # an UNMEASURED operating point — `shortlist_truncated` reports its cost
    # per read — which is why edlib is opt-in.
    estep_aligner: Literal["minimap2", "edlib"] = "minimap2"
    estep_shortlist_k: int = 16
    estep_shortlist_frac: float = 0.8
    #: 0 means `threads`.
    estep_align_workers: int = 0
    # The template graph (graph.py). "rounds" relates every round's nodes and
    # the final ones; "final" only the last round's; "off" runs nothing after
    # the M-step — the escape hatch. A report unless `merge` is set.
    template_graph: Literal["rounds", "final", "off"] = "rounds"
    graph: GraphParams = field(default_factory=GraphParams)
    # Merge (refine.py). A pair merges iff it is `equivalent` in the graph,
    # within `merge_max_edits` over the shared span and the end tolerances,
    # and not separated by the same M-step split. ON by default since
    # 2026-09-30: at 9.4M reads a K = 1..6 sweep left full-length proteins
    # and genes flat (±0.3%) while K = 2 nearly stopped node growth (r4→r5
    # +1,790 nodes against +4,543 exact and +5,623 unmerged) and was
    # nominally best on every recall count. The merge this one replaced,
    # which cost 13 points of ORF carriage, judged identity on a local
    # alignment and re-joined what the M-step had split.
    merge: bool = True
    #: Edits over the shared span. An absolute count, like the tolerances: a
    #: fraction lets two differences through above 2 kb and blocks them
    #: below. 2 is the conservative point of the sweep: genotype structure
    #: rises with the count and substitutions carry alleles, so a larger K
    #: wants to be edit-type-aware rather than a bigger number.
    merge_max_edits: int = 2
    #: ``None`` is the graph's tolerance; a merge tolerance may not exceed it,
    #: because `equivalent` is defined by the graph's.
    merge_tol_5p: int | None = None
    merge_tol_3p: int | None = None
    merge_min_reads: int = 0
    #: Let templates separated by the same split merge again. Off: that is the
    #: split/merge limit cycle (51.6% of the old merge's pairs).
    merge_siblings: bool = False
    #: Merge only after rounds >= this (1 = every round).
    merge_from_round: int = 1
    # Round 2+ ranking (A3).
    delta_logl: float = 5.0
    support_ratio: float = 20.0
    near_tie_z: float = 2.0
    read_error_rate: float = 0.01
    # Seeding (Part C).
    #: Which round-1 seeder builds the templates. "orf" is the original —
    #: one template per distinct sense ORF, folded. "kmer" clusters the READS
    #: with the `--mode kmer` kernels and elects one per cluster; measured on
    #: 9.39M reads it is 4.4x fewer templates and a 2.9 h round-1 E-step
    #: against 13.95 h, at the same gene agreement. Fields below are split by
    #: which seeder reads them; the sketch ones are shared because the two
    #: seeders never both run.
    seeding: Literal["orf", "kmer"] = "orf"
    #: The protein-annotation floor on every node (`best_sense_orf` on its
    #: consensus), and under ORF seeding the seeding key as well. One source
    #: of truth: `mstep.min_aa_length` is set from this in `__post_init__`.
    min_aa_length: int = 30
    seed_representative: str = "longest-above-quality"
    min_seed_reads: int = 1
    # orf seeding only — the ORF-level fold.
    fold_identity: float = 0.97
    max_len_delta: int = 9
    # Shared sketch (fold's ORFs / kmer's reads).
    kmer: int = 15
    window: int = 10
    #: ``None`` is uncapped. 0 is NOT — it caps every sketch at zero.
    minimizers_per_seq: int | None = 50
    # kmer seeding only — the read-level gate and grouping.
    seed_identity: float = DEFAULT_SEED_IDENTITY
    seed_max_5p_overhang: int = DEFAULT_SEED_MAX_5P
    seed_max_3p_overhang: int = DEFAULT_SEED_MAX_3P
    seed_grouping: str = "components"
    min_shared: int = 2
    diag_span_max: int = 20
    max_cluster_read_frac: float = DEFAULT_MAX_CLUSTER_READ_FRAC
    min_chain_cluster_reads: int = DEFAULT_MIN_CHAIN_CLUSTER_READS
    # Corpus.
    max_window_length: int | None = 15_000
    threads: int = 8
    mstep_workers: int = 0
    units_per_worker: int = 16
    mstep: MStepParams = field(default_factory=MStepParams)

    def __post_init__(self) -> None:
        if self.estep_aligner not in ("minimap2", "edlib"):
            raise ValueError(f"unknown estep_aligner {self.estep_aligner!r}")
        if self.estep_shortlist_k < 1:
            raise ValueError("estep_shortlist_k must be >= 1")
        if not 0.0 < self.estep_shortlist_frac <= 1.0:
            raise ValueError("estep_shortlist_frac must be in (0, 1]")
        if self.template_graph not in ("rounds", "final", "off"):
            raise ValueError(f"unknown template_graph {self.template_graph!r}")
        for name in ("merge", "merge_siblings"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        for name in (
            "merge_max_edits",
            "merge_min_reads",
            "merge_from_round",
            "merge_tol_5p",
            "merge_tol_3p",
        ):
            value = getattr(self, name)
            if name.startswith("merge_tol") and value is None:
                continue
            # bool is an int and 1.5 truncates to one: neither is a count.
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an int, got {value!r}")
        for name in ("merge_max_edits", "merge_min_reads"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.merge_from_round < 1:
            raise ValueError("merge_from_round must be >= 1")
        if not self.merge and self.merge_from_round != 1:
            raise ValueError(
                "merge_from_round has no effect without merge: it only "
                "schedules merging"
            )
        if self.template_graph == "off":
            # Nothing reads them, and a knob that did nothing makes a run
            # look like evidence about it. Judged against the field defaults,
            # which is all a library caller can be held to: the CLI refuses
            # the FLAGS, whatever their values.
            if self.graph != GraphParams():
                raise ValueError(
                    "graph parameters have no effect under template_graph='off'"
                )
            for f in dataclasses.fields(self):
                if f.name.startswith("merge_") and getattr(self, f.name) != f.default:
                    raise ValueError(
                        f"{f.name} has no effect under template_graph='off': "
                        f"there is no graph for it to define `mergeable` on"
                    )
        for name, limit in (
            ("merge_tol_5p", self.graph.tol_5p),
            ("merge_tol_3p", self.graph.tol_3p),
        ):
            tol = getattr(self, name)
            if tol is not None and not 0 <= int(tol) <= int(limit):
                raise ValueError(
                    f"{name} must be in [0, {limit}]: a pair is `equivalent` "
                    f"under the GRAPH's tolerance, so a wider merge tolerance "
                    f"would ask for pairs the graph filed as `contained`"
                )
        if self.merge and self.template_graph == "off":
            raise ValueError(
                "merge needs the template graph: template_graph='off' runs "
                "nothing after the M-step (merge is on by default; pass "
                "merge=False)"
            )
        if self.merge and self.template_graph == "final":
            # No graph between rounds, so nothing to merge there. A knob
            # that cannot act is an error.
            if self.merge_from_round != 1:
                raise ValueError(
                    "merge_from_round has no effect under template_graph="
                    "'final', which merges the final output only"
                )
        if (
            isinstance(self.min_aa_length, bool)
            or not isinstance(self.min_aa_length, (int, np.integer))
            or self.min_aa_length < 1
        ):
            raise ValueError(
                f"min_aa_length must be an int >= 1, got {self.min_aa_length!r}"
            )
        if self.mstep.min_aa_length != self.min_aa_length:
            object.__setattr__(
                self,
                "mstep",
                dataclasses.replace(self.mstep, min_aa_length=int(self.min_aa_length)),
            )
        # The graph's edit budget may never pre-empt the merge cap: an edge
        # exists only within `edit_budget = max(min_budget, floor(L * (1 -
        # identity_floor)))`, so at the default 0.99 a pair whose shorter
        # template is under 100 * K nt could not merge at K edits however
        # the predicate read — 13% of a 9.4M-read run's nodes at K = 6. The
        # budget follows the cap; long-template edges are unchanged.
        if self.merge_max_edits > self.graph.min_budget:
            object.__setattr__(
                self,
                "graph",
                dataclasses.replace(self.graph, min_budget=int(self.merge_max_edits)),
            )

    def merge_predicate(self) -> MergePredicate:
        """The merge predicate with its tolerances resolved."""
        return MergePredicate(
            max_edits=int(self.merge_max_edits),
            tol_5p=int(
                self.graph.tol_5p if self.merge_tol_5p is None else self.merge_tol_5p
            ),
            tol_3p=int(
                self.graph.tol_3p if self.merge_tol_3p is None else self.merge_tol_3p
            ),
            min_reads=int(self.merge_min_reads),
            merge_siblings=bool(self.merge_siblings),
        )

    def merges_after(self, r: int) -> bool:
        """Does round ``r``'s refine collapse anything?"""
        return bool(
            self.merge
            and self.template_graph == "rounds"
            and r >= self.merge_from_round
        )


@dataclass(frozen=True, slots=True)
class RoundResult:
    round_index: int
    n_templates: int
    estep: dict[str, Any]
    n_nodes: int
    churn: dict[str, float]
    seconds: dict[str, float]


def run_em(
    demux_dir: Path,
    output_dir: Path,
    *,
    params: EmParams | None = None,
    resume: bool = False,
    report: bool = True,
    progress=None,
) -> list[RoundResult]:
    """Run the EM loop to convergence (or ``params.rounds``)."""
    params = params or EmParams()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda _m: None)

    corpus = write_corpus(
        Path(demux_dir),
        output_dir / "corpus",
        max_window_length=params.max_window_length,
        resume=resume,
    )
    log(
        f"corpus: {corpus.n_reads:,} reads "
        f"({corpus.stats.get('n_dropped_long', 0):,} over --max-window-length)"
    )
    if corpus.n_reads == 0:
        return []

    reads = ReadStore.open(corpus.arrow_path)
    try:
        return _loop(
            corpus, reads, output_dir, Path(demux_dir), params, resume, report, log
        )
    finally:
        reads.close()


def _loop(corpus, reads, output_dir, demux_dir, params, resume, report, log) -> list[RoundResult]:
    rounds_dir = output_dir / "rounds"
    # Before anything is reused. The stamp describes what produced ROUND 1's
    # templates, and every later round descends from those — so a resume that
    # restarts at round 4 is just as committed to the original seeder as one
    # that reuses `seed/` directly, and checking only in the latter path left
    # the former silently running one seeder's templates under the other's
    # manifest.
    if resume:
        _check_seed_stamp(output_dir / "seed", params, log)
    _check_estep_stamp(rounds_dir, params, resume)
    _check_merge_state(rounds_dir, params, resume)
    start, templates_table, prev_assignments, history = _resume_point(
        rounds_dir, resume, log, params
    )

    if templates_table is None:
        templates_table = _round_one_templates(
            corpus, reads, output_dir, params, resume, log
        )
        if templates_table.num_rows == 0:
            log(
                "no read carries a qualifying ORF; nothing to cluster"
                if params.seeding == "orf"
                else "no cluster cleared --min-seed-reads; nothing to cluster"
            )
            return []

    results: list[RoundResult] = []
    final_nodes: pa.Table | None = None
    final_assignments: pa.Table | None = None
    final_membership: pa.Table | None = None
    for r in range(start, start + params.rounds):
        rd = rounds_dir / f"r{r:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        tdir = rd / "templates"
        if not (tdir / _SUCCESS).exists():
            write_templates(templates_table, tdir)
            (tdir / _SUCCESS).write_bytes(b"")
        store = TemplateStore.open(tdir / TEMPLATES_ARROW)
        try:
            result, assignments, nodes, node_membership = _one_round(
                r, rd, store, corpus, reads, params, prev_assignments, t0, log
            )
            results.append(result)
            final_nodes, final_assignments = nodes, assignments
            final_membership = node_membership

            last_round = r == start + params.rounds - 1
            # frac_unsettled, not frac_changed_lineage: the latter is measured
            # only over reads assigned in BOTH rounds, so losing half the
            # assignments scores zero churn as long as the survivors kept
            # their lineage, and the loop would call that converged.
            converged = (
                prev_assignments is not None
                and result.churn.get("frac_unsettled", 1.0)
                < params.stop_frac_changed
            )

            # Refine, and write the lineage, BEFORE the round is marked done.
            # `_SUCCESS` means "everything this round owes the next one is on
            # disk", and lineage.parquet is one of those things: without it
            # the next round cannot tell a split from a genuine switch. Dying
            # between the marker and the write left a round that claimed to
            # be complete and was not.
            refined = (
                None if (last_round or converged)
                else _refine_and_merge(rd, nodes, store, r, params, log)
            )
            if refined is not None:
                _write_lineage(rd, refined.lineage)
            (rd / _SUCCESS).write_bytes(b"")

            if last_round:
                break
            if converged:
                log(f"converged at round {r}")
                break
            log(
                f"round {r}: {refined.n_parents:,} templates -> "
                f"{refined.n_children:,} ({refined.n_unrecruited:,} recruited "
                f"nothing, {refined.n_merged:,} merged)"
            )
            if refined.templates.num_rows == 0:
                log("nothing carried forward; stopping")
                break
            templates_table = refined.templates
            prev_assignments = assignments
        finally:
            store.close()

    _write_summary(output_dir, history + results)
    if final_nodes is not None and final_assignments is not None:
        final_rd = rounds_dir / f"r{results[-1].round_index:02d}"
        final_nodes, final_membership, final = _final_graph_and_merge(
            final_rd,
            final_nodes,
            final_membership,
            results[-1].round_index,
            params,
            log,
            corpus_path=corpus.arrow_path,
        )
        clusters, membership, sample_id = build_cluster_tables(
            final_nodes,
            final_assignments,
            final_membership,
            reads=reads,
            identity_threshold=params.p_floor,
        )
        paths = write_em_outputs(output_dir, clusters, membership, sample_id)
        counts = _cluster_edges(output_dir, final, clusters, paths, log)
        _write_final_record(final_rd, final, int(clusters.num_rows), counts)
        # Again, now that the last round's graph has a time to report.
        _write_summary(output_dir, history + results)
        last = results[-1] if results else None
        write_em_manifest(
            output_dir,
            demux_dir,
            parameters={"mode": "em", **asdict(params)},
            stages={
                "n_input_reads": int(corpus.stats.get("n_input", corpus.n_reads)),
                "n_reads": int(corpus.n_reads),
                "n_rounds": len(results),
                "n_clusters": int(clusters.num_rows),
                "n_membership_rows": int(membership.num_rows),
                "converged": bool(
                    last is not None
                    and last.churn.get("frac_unsettled", 1.0)
                    < params.stop_frac_changed
                ),
            },
            outputs={k: v.name for k, v in paths.items()},
        )
        log(
            f"wrote {clusters.num_rows:,} clusters over "
            f"{membership.num_rows:,} assigned reads"
        )
    if report:
        try:
            from constellation.sequencing.transcriptome.cluster.denovo.em import (
                diagnostics as em_diag,
            )

            log(f"diagnostic report: {em_diag.build_em_report(output_dir)}")
        except Exception as exc:  # noqa: BLE001 — a report never sinks a run
            log(
                f"diagnostics failed ({type(exc).__name__}); "
                f"regenerate with `em.diagnostics.build_em_report`: {exc}"
            )
    return results


def _one_round(r, rd, store, corpus, reads, params, prev_assignments, t0, log):
    log(f"round {r}: E-step over {store.n_templates:,} templates")
    estep_stats = run_em_estep(
        rd / "templates" / TEMPLATES_FASTA,
        corpus.fasta_path,
        store=store,
        reads=reads,
        output_dir=rd / "assignments",
        round_index=r,
        threads=params.threads,
        minimap2_n=params.minimap2_n,
        index_batch_size=params.index_batch_size,
        p_floor=params.p_floor,
        delta_logl=params.delta_logl,
        support_ratio=params.support_ratio,
        near_tie_z=params.near_tie_z,
        error_rate=params.read_error_rate,
        progress=log,
        **_estep_aligner_kwargs(params, rd, corpus),
    )
    t_estep = time.time() - t0
    _warn_on_saturation(estep_stats, params, log)

    assignments = _read_assignments(rd / "assignments")
    t1 = time.time()
    nodes, node_membership = _run_mstep(
        r, rd, store, corpus, assignments, params, log
    )
    t_mstep = time.time() - t1

    churn = (
        rf.measure_churn(
            prev_assignments, assignments, _read_lineage(rd), n_reads=corpus.n_reads
        )
        if prev_assignments is not None
        else {}
    )
    result = RoundResult(
        round_index=r,
        n_templates=int(store.n_templates),
        estep=estep_stats,
        n_nodes=int(nodes.num_rows),
        churn=churn,
        seconds={"estep": round(t_estep, 1), "mstep": round(t_mstep, 1)},
    )
    (rd / "round.json").write_text(json.dumps(asdict(result), indent=2, default=str))
    return result, assignments, nodes, node_membership


def _estep_aligner_kwargs(params: EmParams, rd: Path, corpus) -> dict:
    """The extra ``run_em_estep`` arguments for the chosen aligner."""
    if params.estep_aligner == "minimap2":
        return {}
    return {
        "aligner": "edlib",
        "shortlist_k": params.estep_shortlist_k,
        "shortlist_frac": params.estep_shortlist_frac,
        "align_workers": params.estep_align_workers or params.threads,
        "corpus_path": corpus.arrow_path,
        "templates_path": rd / "templates" / TEMPLATES_ARROW,
    }


_ESTEP_STAMP = "estep.json"


def _check_estep_stamp(rounds_dir: Path, params: EmParams, resume: bool) -> None:
    """Refuse a resume that would switch E-step aligners mid-run; stamp it.

    Rounds already on disk were assigned by one aligner, and the M-step built
    the next round's templates from those CIGARs — so a resume that continues
    under the other aligner writes a run whose rounds disagree about how a
    read was aligned while the manifest records one of them. A stampless
    rounds directory predates the flag and can only be minimap2's.
    """
    rounds_dir.mkdir(parents=True, exist_ok=True)
    path = rounds_dir / _ESTEP_STAMP
    want = {"estep_aligner": params.estep_aligner}
    if resume and any(rounds_dir.glob(f"r*/{_SUCCESS}")):
        have = (
            json.loads(path.read_text())
            if path.exists()
            else {"estep_aligner": "minimap2"}
        )
        if have.get("estep_aligner") != want["estep_aligner"]:
            raise ValueError(
                f"--resume would continue {rounds_dir.parent}, whose finished "
                f"rounds were assigned with --estep-aligner "
                f"{have.get('estep_aligner')!r}, under {want['estep_aligner']!r}. "
                f"Later rounds' templates were built from those alignments, so "
                f"the run would mix aligners under one manifest. Choose another "
                f"--output-dir, or restore the original aligner."
            )
    path.write_text(json.dumps(want, indent=2))


def _warn_on_saturation(stats: dict, params: EmParams, log) -> None:
    """The pool must contain every template within p_floor. Say so if it did not.

    minimap2 truncates by SCORE, and among near-clone templates score
    differences are 1-2 units — arbitrary. So a read whose candidate list hit
    the ``-N`` cap had its pool truncated, and both rankers then arbitrated
    over an arbitrary subset of the set they were supposed to rank. That is
    not a degraded answer; it is an answer to a different question, which is
    why this is loud rather than informational.
    """
    frac = stats.get("cap_hit_fraction", 0.0)
    if frac > 0.01:
        log(
            f"  WARNING: {frac:.2%} of reads hit the -N {params.minimap2_n} "
            "candidate cap, so their pool was TRUNCATED and this round's "
            f"rankings are unsound for them. Raise --minimap2-n."
        )
    trunc = stats.get("shortlist_truncated_fraction", 0.0)
    if trunc > 0.01:
        log(
            f"  WARNING: {trunc:.2%} of reads had their edlib shortlist cut "
            f"(--estep-shortlist-k {params.estep_shortlist_k}) where the cut "
            "could have changed their assignment. Raise --estep-shortlist-k "
            "or lower --estep-shortlist-frac."
        )


def _seed_stamp(params: EmParams) -> dict:
    """Only the fields the seeder actually reads, per seeding mode.

    Stamping every field would refuse a resume over a changed `--rounds`,
    which the seed stage neither reads nor is invalidated by.
    """
    shared = ("seeding", "seed_representative", "min_seed_reads")
    per_mode = (
        # `min_aa_length` is the ORF seeder's KEY — change it and the seeds
        # are different ORFs. The kmer seeder never reads it (its templates
        # carry no ORF), so it must not invalidate a kmer resume.
        (
            "min_aa_length",
            "fold_identity",
            "max_len_delta",
            "kmer",
            "window",
            "minimizers_per_seq",
        )
        if params.seeding == "orf"
        else (
            "kmer",
            "window",
            "minimizers_per_seq",
            "seed_identity",
            "seed_max_5p_overhang",
            "seed_max_3p_overhang",
            "seed_grouping",
            "min_shared",
            "diag_span_max",
        )
    )
    return {k: getattr(params, k) for k in (*shared, *per_mode)}


def _check_seed_stamp(seed_dir: Path, params: EmParams, log) -> None:
    """Refuse a resume whose seeding parameters differ from what is on disk."""
    if not (seed_dir / _SUCCESS).exists():
        return  # nothing was seeded here yet; there is nothing to disagree with
    path = seed_dir / "params.json"
    if not path.exists():
        _check_legacy_seed(seed_dir, params)
        return
    want = _seed_stamp(params)
    have = json.loads(path.read_text())
    if have.get("seeding") != want["seeding"]:
        # Report only this. The stamp is per-mode, so every other "difference"
        # is really a field the other seeder does not have, and listing them
        # buries the one fact that matters.
        raise ValueError(
            f"--resume would reuse the seed stage in {seed_dir}, but it was "
            f"produced by the {have.get('seeding')!r} seeder and this run asks "
            f"for {want['seeding']!r}. Round 1's templates would be one "
            f"seeder's while the manifest recorded the other's. Choose another "
            f"--output-dir."
        )
    changed = [k for k, v in want.items() if have.get(k) != v]
    if changed:
        detail = ", ".join(f"{k}: {have.get(k)!r} -> {want[k]!r}" for k in changed)
        raise ValueError(
            f"--resume would reuse the seed stage in {seed_dir}, but its "
            f"parameters differ ({detail}). Re-seeding silently would leave "
            f"round 1's templates disagreeing with the manifest. Choose "
            f"another --output-dir, or restore the original values."
        )
    log(f"--resume: reusing the {have.get('seeding', 'orf')} seed stage")


def _check_legacy_seed(seed_dir: Path, params: EmParams) -> None:
    """An unstamped ``seed/`` predates stamping. Verify it another way, or refuse.

    It can only have come from the ORF seeder, since that is the only one that
    existed — so a kmer run against it would reuse ORF templates under a kmer
    manifest. But "it is ORF-seeded" is not enough on its own: this release
    changed the effective ``--min-aa-length`` for ORF seeding from 60 to 30,
    so an ordinary legacy resume would keep the 60-aa seeds and write a
    manifest claiming 30. That is the same defect the stamp exists to prevent,
    one release earlier.

    A *completed* legacy run recorded its parameters in ``manifest.json``, so
    that is checkable. An interrupted one recorded nothing, and reusing what
    cannot be verified is exactly how an artifact from an earlier attempt gets
    read back as if it belonged to this one — so that case is refused.
    """
    if params.seeding != "orf":
        raise ValueError(
            f"{seed_dir} predates seed-parameter stamping, so it can only "
            f"hold ORF-seeded templates, but this run asks for "
            f"seeding={params.seeding!r}. Choose another --output-dir."
        )

    manifest = seed_dir.parent / "manifest.json"
    if not manifest.exists():
        raise ValueError(
            f"{seed_dir} predates seed-parameter stamping and its run never "
            f"completed, so there is no record of what produced these "
            f"templates — and this release changed the ORF seeder's effective "
            f"--min-aa-length from 60 to 30, so reusing them would put "
            f"60-aa-seeded templates under a manifest claiming 30. Start a "
            f"fresh --output-dir, or delete {seed_dir} to re-seed."
        )
    try:
        recorded = json.loads(manifest.read_text()).get("parameters", {})
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{seed_dir} predates seed-parameter stamping and its "
            f"manifest.json could not be read ({exc}), so what produced these "
            f"templates cannot be verified. Start a fresh --output-dir."
        ) from None

    want = _seed_stamp(params)
    # `seeding` postdates those manifests; everything else is comparable.
    changed = [
        k
        for k, v in want.items()
        if k != "seeding" and k in recorded and recorded[k] != v
    ]
    if changed:
        detail = ", ".join(f"{k}: {recorded[k]!r} -> {want[k]!r}" for k in changed)
        raise ValueError(
            f"--resume would reuse the pre-stamp seed stage in {seed_dir}, but "
            f"the manifest of the run that produced it records different "
            f"parameters ({detail}). Note this release changed the ORF "
            f"seeder's effective --min-aa-length from 60 to 30, so an "
            f"unchanged command line can still land here. Pass the original "
            f"values, or choose another --output-dir."
        )


def _round_one_templates(corpus, reads, output_dir, params, resume, log):
    """Round 1's templates, from whichever seeder ``params.seeding`` names."""
    seed_dir = output_dir / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    # The stamp is already checked in `_loop`, for every resume rather than
    # only the ones that land here.
    if resume and (seed_dir / _SUCCESS).exists():
        return pq.read_table(seed_dir / "templates.parquet")

    if params.seeding == "kmer":
        return _seed_kmer(reads, seed_dir, params, log)

    log(f"seeding ORFs over {reads.n_reads:,} reads…")
    seed, _read_orf = extract_seed_orfs(
        reads.table,
        min_aa_length=params.min_aa_length,
        representative=params.seed_representative,
        threads=params.threads,
        progress=log,
    )
    if seed.num_rows == 0:
        return TEMPLATE_TABLE.empty_table()
    log(f"folding {seed.num_rows:,} distinct ORFs…")
    fold = fold_orfs(
        seed,
        identity=params.fold_identity,
        max_len_delta=params.max_len_delta,
        kmer=params.kmer,
        window=params.window,
        minimizers_per_seq=params.minimizers_per_seq,
        threads=params.threads,
        progress=log,
    )
    row_of_id = {rid: i for i, rid in enumerate(reads.read_id.to_pylist())}
    templates = templates_from_seeds(
        seed,
        group_rep_orf=fold.group_rep_orf,
        group_n_reads=fold.group_n_reads,
        read_quality_of_row=reads.dorado_quality,
        read_row_of_id=row_of_id,
        min_seed_reads=params.min_seed_reads,
    )
    pq.write_table(templates, seed_dir / "templates.parquet")
    pq.write_table(seed, seed_dir / "seed_orfs.parquet")
    # The same shape the kmer seeder reports, so `section_seeding` renders
    # either without branching and the two are comparable at a glance.
    n_reads = np.asarray(
        templates.column("orf_replication").to_numpy(zero_copy_only=False)
    )
    _write_seed_stage(
        seed_dir,
        params,
        {
            "seeding": "orf",
            "n_reads": int(reads.n_reads),
            "n_orfs": int(seed.num_rows),
            "n_clusters": int(templates.num_rows),
            "n_clusters_ge2": int((n_reads >= 2).sum()),
            "template_mb": round(
                float(
                    pc.sum(pc.utf8_length(templates.column("sequence"))).as_py() or 0
                )
                / 1e6,
                3,
            ),
            "reads_per_template": round(
                float(n_reads.sum() / max(templates.num_rows, 1)), 3
            ),
            "singleton_frac": round(
                float((n_reads == 1).mean()) if n_reads.size else 0.0, 6
            ),
            "largest_cluster_reads": int(n_reads.max()) if n_reads.size else 0,
            "largest_cluster_read_frac": round(
                float(n_reads.max() / reads.n_reads) if n_reads.size else 0.0, 6
            ),
        },
    )
    log(f"round 1: {templates.num_rows:,} templates")
    return templates


def _seed_kmer(reads, seed_dir, params, log):
    """Round 1's templates from read-level kmer clustering."""
    log(f"kmer-clustering {reads.n_reads:,} reads for seeding…")
    result = seed_by_kmer_clustering(
        reads.table,
        identity=params.seed_identity,
        max_5p_overhang=params.seed_max_5p_overhang,
        max_3p_overhang=params.seed_max_3p_overhang,
        kmer=params.kmer,
        window=params.window,
        minimizers_per_seq=params.minimizers_per_seq,
        min_shared=params.min_shared,
        diag_span_max=params.diag_span_max,
        grouping=params.seed_grouping,
        min_seed_reads=params.min_seed_reads,
        representative=params.seed_representative,
        max_cluster_read_frac=params.max_cluster_read_frac,
        min_chain_cluster_reads=params.min_chain_cluster_reads,
        threads=params.threads,
        progress=log,
    )
    pq.write_table(result.templates, seed_dir / "templates.parquet")
    pq.write_table(result.read_cluster, seed_dir / "read_cluster.parquet")
    _write_seed_stage(seed_dir, params, result.stats)
    log(f"round 1: {result.templates.num_rows:,} templates")
    return result.templates


def _write_seed_stage(seed_dir: Path, params: EmParams, stats: dict) -> None:
    """Persist the seed stage's stats + parameter stamp, then mark it done.

    `_SUCCESS` last, and only after both writes: it means "everything this
    stage owes the run is on disk", and the stamp is one of those things —
    without it a resume cannot tell which seeder produced these templates.
    """
    (seed_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (seed_dir / "params.json").write_text(
        json.dumps(_seed_stamp(params), indent=2, default=str)
    )
    (seed_dir / _SUCCESS).write_bytes(b"")


def _run_mstep(r, rd, store, corpus, assignments, params, log):
    out = rd / "mstep"
    srt, starts = sort_assignments_by_template(assignments)
    if srt.num_rows == 0:
        return REFINED_NODE_TABLE.empty_table(), NODE_MEMBERSHIP_TABLE.empty_table()

    bounds = np.concatenate([starts, [srt.num_rows]])
    tr = srt.column("template_row").to_numpy(zero_copy_only=False)
    rows = tr[starts].astype(np.int64)
    row_lo = np.zeros(store.n_templates, dtype=np.int64)
    row_hi = np.zeros(store.n_templates, dtype=np.int64)
    row_lo[rows] = bounds[:-1]
    row_hi[rows] = bounds[1:]

    workers = params.mstep_workers or params.threads
    units = plan_mstep_units(
        np.arange(store.n_templates),
        row_lo,
        row_hi,
        store.lengths(),
        n_units=max(workers * params.units_per_worker, 1),
        max_members=params.mstep.max_members_per_template,
    )
    log(f"round {r}: M-step over {rows.size:,} live templates in {len(units)} units")

    outputs = run_batched(
        mstep_worker,
        iter_unit_batches(srt, units),
        output_dir=out,
        output_keys=("nodes", "node_membership"),
        n_workers=max(int(workers), 1),
        worker_kwargs={
            "corpus_path": str(corpus.arrow_path),
            "templates_path": str(rd / "templates" / TEMPLATES_ARROW),
            "round_index": r,
            "params": params.mstep,
        },
        stage_label=f"mstep-r{r}",
        total=len(units),
    )
    shards = outputs["nodes"].shard_paths
    # In node order, not shard order: the shards are one per unit and the
    # units depend on the worker count, while the rows here become the next
    # round's template ids.
    nodes = in_node_order(
        pa_ds.dataset(shards, schema=REFINED_NODE_TABLE).to_table()
        if shards
        else REFINED_NODE_TABLE.empty_table()
    )
    mem_shards = outputs["node_membership"].shard_paths
    membership = in_membership_order(
        pa_ds.dataset(mem_shards, schema=NODE_MEMBERSHIP_TABLE).to_table()
        if mem_shards
        else NODE_MEMBERSHIP_TABLE.empty_table()
    )
    return nodes, membership


def _read_assignments(directory: Path) -> pa.Table:
    files = sorted(Path(directory).glob("part-*.parquet"))
    if not files:
        return EM_ASSIGNMENT_TABLE.empty_table()
    return pa_ds.dataset(files, schema=EM_ASSIGNMENT_TABLE).to_table()


def _write_json(path: Path, payload: dict) -> None:
    """Atomically: a record half-written is a record that lies."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _write_table(path: Path, table: pa.Table) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


@dataclass(frozen=True, slots=True)
class _NodeGraph:
    """One round's graph, as the merge needs it."""

    path: Path
    stats: dict
    stamp: dict
    ids: np.ndarray  # template id of every node row
    origin: np.ndarray  # split origin of every node row, -1 = none
    n_reads: np.ndarray


def _graph_stamp(params: EmParams, digest: str) -> dict:
    """What the cached edges depend on.

    The predicate is in it because the edge table stores ``mergeable``; the
    execution-only knobs (threads, chunk sizes) are not, because the output
    does not depend on them.
    """
    return {
        **gr.graph_stamp(params.graph, digest),
        "predicate": asdict(params.merge_predicate()),
    }


def _node_graph(rd: Path, nodes: pa.Table, r: int, params: EmParams, log) -> _NodeGraph:
    """Relate round ``r``'s nodes, or reuse the graph already on disk.

    Cached at ``rd/graph/`` and reused only when the stamp — kernel version,
    graph parameters, predicate, and a digest of the very sequences, ids, read
    counts and split origins — is equal and the file opens. The cache this
    replaces (``merge/pairs.parquet``) was reused whenever it opened.
    """
    n = nodes.num_rows
    ids = np.asarray(rf.template_id_for(r + 1, np.arange(n)), dtype=np.int64)
    n_reads = (
        nodes.column("n_reads").to_numpy(zero_copy_only=False).astype(np.int64)
    )
    parents = (
        nodes.column("parent_template_id")
        .to_numpy(zero_copy_only=False)
        .astype(np.int64)
    )
    origin = gr.split_origins(rd.parent, r, parents)
    sequences = nodes.column("consensus")
    stamp = _graph_stamp(
        params, gr.input_digest(sequences, ids, n_reads, origin)
    )

    gdir = rd / "graph"
    edges = gdir / "edges.parquet"
    if (gdir / _SUCCESS).exists() and _parquet_opens(edges):
        try:
            have = json.loads((gdir / "params.json").read_text())
            stats = json.loads((gdir / "stats.json").read_text())
        except (OSError, ValueError):
            have = stats = None
        if have == json.loads(json.dumps(stamp)) and stats is not None:
            return _NodeGraph(edges, stats, stamp, ids, origin, n_reads)

    # Anything else in there is from another attempt or other parameters.
    shutil.rmtree(gdir, ignore_errors=True)
    gdir.mkdir(parents=True, exist_ok=True)
    log(f"round {r}: relating {n:,} nodes")
    result = gr.build_graph(
        sequences,
        ids=ids,
        n_reads=n_reads,
        split_origin=origin,
        node_round=r,
        params=params.graph,
        predicate=params.merge_predicate(),
        threads=params.threads,
        output_path=edges,
        progress=log,
    )
    _write_json(gdir / "params.json", stamp)
    _write_json(gdir / "stats.json", result.stats)
    (gdir / _SUCCESS).write_bytes(b"")
    return _NodeGraph(edges, result.stats, stamp, ids, origin, n_reads)


def _graph_or_failure(
    rd: Path, nodes: pa.Table, r: int, params: EmParams, log, *, merging: bool
) -> tuple[_NodeGraph | None, str | None]:
    """The graph, or why there is none.

    A graph nothing merges from is a report: if it fails, the round it
    describes is no less finished, so the failure is recorded and the round
    goes on. A graph a merge reads is part of the round, and its failure is
    the round's.
    """
    try:
        return _node_graph(rd, nodes, r, params, log), None
    except Exception as exc:  # noqa: BLE001 — a report never sinks a round
        if merging:
            raise
        log(
            f"round {r}: template graph failed ({type(exc).__name__}: {exc}); "
            f"nothing depends on it, continuing"
        )
        return None, f"{type(exc).__name__}: {exc}"


def _mergeable(graph: _NodeGraph, params: EmParams, *, exact_only: bool) -> dict:
    """The mergeable pairs of a graph, reading only those rows off disk."""
    edges = pq.read_table(graph.path, filters=[("mergeable", "==", True)])
    return gr.mergeable_pairs(
        edges, params.merge_predicate(), exact_only=exact_only
    )


def _refine_and_merge(rd: Path, nodes, store, r: int, params: EmParams, log):
    """Next templates from this round's nodes; merged, if the run merges.

    One path for the live loop and the resume rebuild, so a resumed round
    does exactly what the live one did. What it did is written to
    ``refine.json`` before the round's marker, because the next resume has to
    be able to check it.
    """
    refined = rf.next_templates(nodes, store, round_index=r)
    if refined.templates.num_rows == 0:
        return refined

    merging = params.merges_after(r)
    record: dict[str, Any] = {
        "round": int(r),
        "template_graph": params.template_graph,
        "graph": "off",
        "merge_applied": False,
        "n_nodes": int(nodes.num_rows),
        "n_templates_before": int(refined.templates.num_rows),
    }
    merged = rf.MERGED_TABLE.empty_table()
    if params.template_graph != "rounds":
        # This round was somebody's final one and is being refined after
        # all, by a run that relates nothing here. The graph it was left
        # with describes a configuration this run is not.
        shutil.rmtree(rd / "graph", ignore_errors=True)
    else:
        graph, failure = _graph_or_failure(
            rd, nodes, r, params, log, merging=merging
        )
        record["graph"] = "ok" if graph is not None else "failed"
        if failure is not None:
            record["graph_error"] = failure
        if graph is not None:
            record["n_still_mergeable"] = int(graph.stats.get("n_mergeable", 0))
        if graph is not None and merging:
            pred = params.merge_predicate()
            pairs = _mergeable(graph, params, exact_only=False)
            refined = rf.apply_merge(
                refined,
                pairs,
                tol_5p=pred.tol_5p,
                tol_3p=pred.tol_3p,
                origin_ids=None if pred.merge_siblings else graph.ids,
                origin=None if pred.merge_siblings else graph.origin,
            )
            merged = refined.merged
            record.update(
                merge_applied=True,
                predicate=asdict(pred),
                graph_stamp=graph.stamp,
                **_merge_counts(merged, pairs, graph),
            )
    record["n_merged"] = int(refined.n_merged)
    record["n_templates_after"] = int(refined.templates.num_rows)

    # Always written, empty or not: a file that is only there when something
    # merged is a file a re-run that merges nothing leaves behind.
    _write_table(rd / "merged.parquet", merged)
    # This round has a successor now, so it is no longer anybody's final one.
    for stale in ("final.json", "merged_final.parquet"):
        (rd / stale).unlink(missing_ok=True)
    _write_json(rd / "refine.json", record)
    if refined.n_merged and log:
        log(f"round {r}: merged {refined.n_merged:,} templates")
    return refined


def _merge_counts(merged: pa.Table, pairs: dict, graph: _NodeGraph) -> dict:
    """What a merge pass left behind, and how much of it undid a split.

    ``n_still_mergeable``: pairs the predicate accepts whose two templates
    both survived — refused by a group guard, or passed over because a
    neighbour was claimed first. ``n_merged_kin``: absorbed templates that
    share a split origin with their survivor, which only
    ``merge_siblings`` (or byte-identity) allows.
    """
    gone = merged.column("absorbed_template_id").to_numpy(zero_copy_only=False)
    kept = merged.column("survivor_template_id").to_numpy(zero_copy_only=False)
    still = ~np.isin(pairs["src_template_id"], gone) & ~np.isin(
        pairs["dst_template_id"], gone
    )
    a = graph.origin[np.searchsorted(graph.ids, gone)]
    b = graph.origin[np.searchsorted(graph.ids, kept)]
    return {
        "n_still_mergeable": int(still.sum()),
        "n_merged_kin": int(((a >= 0) & (a == b)).sum()),
    }


def _final_graph_and_merge(
    rd: Path,
    nodes,
    membership,
    r: int,
    params: EmParams,
    log,
    *,
    corpus_path: Path | str | None = None,
) -> tuple[pa.Table, pa.Table, dict]:
    """The final nodes' graph, and the final merge if the run merges.

    The last round is never refined, so its graph is built here (or found, if
    an earlier invocation stopped at this round). The merge is the run's —
    the same predicate as between rounds, the deepest member surviving — and
    because no M-step follows, the survivor's consensus is rebuilt here from
    the pooled reads of its group (:mod:`.rebuild`) and its ORF called on
    that. Held to exact pairs with the longest member kept, as it once was,
    the final output carried 7-13k of round 6's splits unmerged and the
    wrong form more often than not (ledger #52, #55).

    ``corpus_path`` is where the reads are; without it (tests) the survivors
    keep their own consensus and ``final["rebuild"]`` says so.
    """
    final: dict[str, Any] = {
        "round": int(r),
        "template_graph": params.template_graph,
        "graph": "off",
        "merge_applied": False,
        "n_merged": 0,
        "n_nodes": int(nodes.num_rows),
    }
    if params.template_graph == "off":
        shutil.rmtree(rd / "graph", ignore_errors=True)
        return nodes, membership, final
    if nodes.num_rows == 0:
        final["graph"] = "empty"
        return nodes, membership, final

    graph, failure = _graph_or_failure(
        rd, nodes, r, params, log, merging=params.merge
    )
    if graph is None:
        final.update(graph="failed", graph_error=failure)
        return nodes, membership, final
    final["graph"] = "ok"

    keep = np.arange(nodes.num_rows, dtype=np.int64)
    merged = rf.MERGED_TABLE.empty_table()
    if params.merge:
        pred = params.merge_predicate()
        pairs = _mergeable(graph, params, exact_only=False)
        nodes, membership, n_merged, survivor = rf.merge_nodes(
            nodes,
            membership,
            pairs,
            tol_5p=pred.tol_5p,
            tol_3p=pred.tol_3p,
            origin=None if pred.merge_siblings else graph.origin,
            prefer="support",
        )
        merged = rf.merged_table(
            graph.ids,
            survivor,
            pairs["src_row"],
            pairs["dst_row"],
            pairs["off_5p"],
            pairs["off_3p"],
            graph.n_reads,
            round_index=r + 1,
        )
        keep = np.flatnonzero(survivor == keep)
        final.update(
            merge_applied=True,
            n_merged=int(n_merged),
            predicate=asdict(pred),
            graph_stamp=graph.stamp,
            n_merged_kin=_merge_counts(merged, pairs, graph)["n_merged_kin"],
        )
        if n_merged and log:
            log(f"final output: merged {n_merged:,} nodes")
        if n_merged:
            # Survivors that absorbed something, as rows of the MERGED table.
            absorbed_into = np.bincount(survivor, minlength=survivor.shape[0])[keep]
            grew = np.flatnonzero(absorbed_into > 1)
            if corpus_path is None:
                final["rebuild"] = "skipped: no corpus"
            else:
                nodes, counts = rebuild_survivors(
                    nodes,
                    membership,
                    grew,
                    str(corpus_path),
                    params=params.mstep,
                    identity_floor=params.p_floor,
                    threads=params.threads,
                    log=log,
                )
                final["rebuild"] = "ok"
                final.update(counts)
    _write_table(rd / "merged_final.parquet", merged)
    final["edges_path"] = graph.path
    final["keep_rows"] = keep
    return nodes, membership, final


def _cluster_edges(output_dir: Path, final: dict, clusters, paths: dict, log) -> dict:
    """Write ``cluster_edges.parquet``; return its counts.

    Nothing reads this file back — not even a merge, which has already
    happened — so it is a report whichever way the run was configured, and
    the run it describes is finished: the clusters are on disk. A failure
    here is recorded, and must not cost the run its manifest.
    """
    if final.get("edges_path") is None:
        return {}
    try:
        paths["cluster_edges"], counts = write_cluster_edges(
            output_dir, final["edges_path"], final["keep_rows"], clusters
        )
        return counts
    except Exception as exc:  # noqa: BLE001 — a report never sinks a run
        for name in (CLUSTER_EDGES, CLUSTER_EDGES + ".tmp"):
            (output_dir / name).unlink(missing_ok=True)
        paths.pop("cluster_edges", None)
        final["cluster_edges_error"] = f"{type(exc).__name__}: {exc}"
        log(
            f"cluster_edges.parquet could not be written "
            f"({type(exc).__name__}: {exc}); the clusters are unaffected"
        )
        return {}


def _write_final_record(rd: Path, final: dict, n_clusters: int, counts: dict) -> None:
    """``final.json``: what the final stage did, counted AFTER it did it.

    The record this replaces reported the pre-merge redundancy of a cluster
    table that had already been merged.
    """
    record = {
        k: v for k, v in final.items() if k not in ("edges_path", "keep_rows")
    }
    record["n_clusters"] = int(n_clusters)
    record.update({k: int(v) for k, v in counts.items()})
    _write_json(rd / "final.json", record)


#: Where the merge this version removed left its record. Its presence with a
#: merge in it is how a run directory is known to descend from that criterion.
_LEGACY_MERGE_STATS = Path("merge") / "stats.json"


def _check_merge_state(rounds_dir: Path, params: EmParams, resume: bool) -> None:
    """Refuse a resume that would change what finished rounds merged.

    A merge changes the templates every later round is built from, so it is a
    commitment of the run in the way the seeder and the E-step aligner are —
    but it is made per round (``merge_from_round``), so it is checked per
    round. A completed round with a ``refine.json`` must have merged exactly
    when this invocation would have, under the same predicate and graph.

    A completed round with NO ``refine.json`` was never refined under this
    version: it was the run's last or converged round, it produced no
    templates, or it predates the record. It accepts anything — which is what
    makes turning merge on for the rounds *after* a finished run possible —
    unless it carries the removed detector's record of a merge. Those
    templates cannot be reproduced, and continuing would put two merge
    criteria under one manifest.
    """
    if not resume or not rounds_dir.exists():
        return
    done = sorted(
        int(d.name[1:])
        for d in rounds_dir.glob("r*")
        if d.name[1:].isdigit() and (d / _SUCCESS).exists()
    )
    run = rounds_dir.parent
    # What each refined round did. A round the removed detector refined
    # without merging anything is a round that did not merge, and is held to
    # that like any other.
    record: dict[int, dict] = {}
    for r in done:
        rd = rounds_dir / f"r{r:02d}"
        if (rd / "refine.json").exists():
            record[r] = json.loads((rd / "refine.json").read_text())
            continue
        legacy = rd / _LEGACY_MERGE_STATS
        if not legacy.exists():
            continue
        try:
            old = json.loads(legacy.read_text())
        except (OSError, ValueError):
            old = {}
        if old.get("merge_applied") and int(old.get("n_merged", 0)) > 0:
            raise ValueError(
                f"--resume would continue {run}, whose round {r} merged "
                f"{int(old['n_merged']):,} templates under the minimap2 "
                f"redundancy criterion (local identity, proportional "
                f"coverage) that this version removed. Those templates "
                f"cannot be reproduced, and continuing would put two merge "
                f"criteria under one manifest. Choose another --output-dir."
            )
        record[r] = {"merge_applied": False}

    # The one --merge-from-round this directory accepts: where its merging
    # began, or past everything it refined if it never merged.
    merged = [r for r, have in record.items() if have.get("merge_applied")]
    accepted = min(merged) if merged else (max(record) + 1 if record else 1)
    want = params.merge_predicate()
    for r, have in sorted(record.items()):
        was, would = bool(have.get("merge_applied")), params.merges_after(r)
        if was != would:
            if merged:
                fix = (
                    f"Resume with --merge --merge-from-round {accepted}, as "
                    f"it was run, or choose another --output-dir."
                )
            else:
                fix = (
                    f"To merge only from here on, pass --merge-from-round "
                    f"{accepted}."
                )
            raise ValueError(
                f"--resume would continue {run}, whose round {r} "
                f"{'merged' if was else 'did not merge'} its templates, as a "
                f"run that {'would' if would else 'would not'} have. Later "
                f"rounds' templates descend from that choice, so the run "
                f"would mix both under one manifest. {fix}"
            )
        if not was:
            continue
        # The graph first: an unset merge tolerance IS the graph's, so a
        # changed graph tolerance shows up in the predicate too, and naming
        # the predicate would send the user to the wrong flags.
        old = have.get("graph_stamp") or {}
        new = json.loads(json.dumps(gr.graph_stamp(params.graph, "")))
        for key in ("kernel_version", "graph"):
            if old.get(key) != new.get(key):
                raise ValueError(
                    f"--resume would continue {run}, whose round {r} merged "
                    f"from a graph built with {key} = {old.get(key)}, with "
                    f"{key} = {new.get(key)}. A merge reads the graph, so "
                    f"its parameters are committed once a round has merged. "
                    f"Restore them or choose another --output-dir."
                )
        if have.get("predicate") != asdict(want):
            raise ValueError(
                f"--resume would continue {run}, whose round {r} merged "
                f"under {have.get('predicate')}, with the merge predicate "
                f"{asdict(want)}. Restore the original merge flags or choose "
                f"another --output-dir."
            )


def _write_lineage(rd: Path, lineage: pa.Table) -> Path:
    """Write ``rd/lineage.parquet`` atomically.

    A parquet file is only readable once its footer lands, so a run killed
    mid-write leaves a file that EXISTS and cannot be opened. Recovery then
    sees a present file, declines to rebuild, and the next round dies on
    ArrowInvalid instead. Rename into place so the path is either absent or
    complete.
    """
    path = rd / "lineage.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(lineage, tmp)
    tmp.replace(path)
    return path


def _read_lineage(rd: Path) -> pa.Table:
    path = rd.parent / f"r{int(rd.name[1:]) - 1:02d}" / "lineage.parquet"
    return pq.read_table(path) if path.exists() else rf.LINEAGE_TABLE.empty_table()


def _resume_point(rounds_dir: Path, resume: bool, log, params: EmParams | None = None):
    """Start after the last round with a ``_SUCCESS``; rebuild its successor.

    Returns ``(start_round, templates, prev_assignments, history)``. The last
    two matter: without the previous round's assignments the resumed round has
    nothing to measure churn against and reports none, and without the history
    ``churn.tsv`` is overwritten with only the resumed invocation's rounds.
    """
    if not resume or not rounds_dir.exists():
        return 1, None, None, []
    done = sorted(
        int(d.name[1:])
        for d in rounds_dir.glob("r*")
        if d.name[1:].isdigit() and (d / _SUCCESS).exists()
    )
    if not done:
        return 1, None, None, []
    last = done[-1]
    prev = _read_assignments(rounds_dir / f"r{last:02d}" / "assignments")
    history = _read_history(rounds_dir, done)
    nxt_dir = rounds_dir / f"r{last + 1:02d}" / "templates"
    # The _SUCCESS marker, not the file's existence: a run interrupted DURING
    # that write leaves a truncated Arrow file behind, and trusting it makes
    # resume die with ArrowInvalid instead of rebuilding from the round that
    # did finish.
    if (nxt_dir / _SUCCESS).exists() and (nxt_dir / TEMPLATES_ARROW).exists():
        log(f"resuming at round {last + 1}")
        with pa.memory_map(str(nxt_dir / TEMPLATES_ARROW), "r") as mm:
            with pa.ipc.open_file(mm) as reader:
                return last + 1, reader.read_all(), prev, history
    # The successor's templates were never written: rebuild them from this
    # round's nodes, which with the assignments is all a restart needs.
    rd = rounds_dir / f"r{last:02d}"
    nodes_dir = rd / "mstep" / "nodes"
    store_path = rd / "templates" / TEMPLATES_ARROW
    if not nodes_dir.exists() or not store_path.exists():
        return 1, None, None, []
    log(f"resuming after round {last} (rebuilding its successor's templates)")
    store = TemplateStore.open(store_path)
    try:
        nodes = in_node_order(
            pa_ds.dataset(
                sorted(nodes_dir.glob("part-*.parquet")), schema=REFINED_NODE_TABLE
            ).to_table()
        )
        refined = _refine_and_merge(
            rd, nodes, store, last, params or EmParams(), log
        )
        # Persist the rebuilt lineage too: the resumed round's churn needs it
        # to tell a split from a genuine switch, and rebuilding it without
        # writing it would discard exactly that. Always, not only when the
        # file on disk will not open: the rebuild is deterministic, so for a
        # round this run refined before it writes what is there — and for one
        # it did not (a former final round, or one an older version refined)
        # the file there describes a refine that is not this one, without the
        # `merge` rows that make an absorbed template's reads inherited.
        _write_lineage(rd, refined.lineage)
        return last + 1, refined.templates, prev, history
    finally:
        store.close()


def _parquet_opens(path: Path) -> bool:
    """Does the file have a footer? Without reading what is above it.

    :func:`_lineage_is_readable` reads the table, which for a lineage is one
    row per node; an edge file is tens of millions of rows.
    """
    if not path.exists():
        return False
    try:
        pq.read_metadata(path)
    except Exception:  # noqa: BLE001 - any failure to open means "rebuild it"
        return False
    return True


def _lineage_is_readable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        pq.read_table(path)
    except Exception:  # noqa: BLE001 - any failure to open means "rebuild it"
        return False
    return True


def _read_history(rounds_dir: Path, done: list[int]) -> list[RoundResult]:
    """Rebuild finished rounds' summaries from their own ``round.json``.

    So a resumed run's churn.tsv carries the whole run rather than only the
    rounds this invocation happened to execute.
    """
    out: list[RoundResult] = []
    for r in done:
        j = _round_json(rounds_dir / f"r{r:02d}")
        if not j:
            continue
        out.append(
            RoundResult(
                round_index=int(j.get("round_index", r)),
                n_templates=int(j.get("n_templates", 0)),
                estep=j.get("estep", {}),
                n_nodes=int(j.get("n_nodes", 0)),
                churn=j.get("churn", {}),
                seconds=j.get("seconds", {}),
            )
        )
    return out


def _round_json(d: Path) -> dict:
    path = d / "round.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _graph_seconds(output_dir: Path, r: int) -> float | None:
    """The graph stage's wall time for round ``r``, where it ran.

    Read from the graph's own stats: ``round.json`` is written before the
    graph exists and is not rewritten.
    """
    path = output_dir / "rounds" / f"r{r:02d}" / "graph" / "stats.json"
    try:
        return json.loads(path.read_text()).get("seconds", {}).get("total")
    except (OSError, ValueError):
        return None


def _write_summary(output_dir: Path, results: list[RoundResult]) -> None:
    if not results:
        return
    import pyarrow.csv as pcsv

    rows = [
        {
            "round": r.round_index,
            "templates": r.n_templates,
            "nodes": r.n_nodes,
            "reads_assigned": r.estep.get("n_assigned", 0),
            "reads_unassigned": r.estep.get("n_unassigned", 0),
            "cap_hit_fraction": round(r.estep.get("cap_hit_fraction", 0.0), 5),
            "frac_changed": r.churn.get("frac_changed"),
            "frac_changed_lineage": r.churn.get("frac_changed_lineage"),
            # What the stopping rule reads — shown so the decision is legible
            # rather than inferred from the two columns beside it.
            "frac_unsettled": r.churn.get("frac_unsettled"),
            "reads_gained": r.churn.get("n_gained"),
            "reads_lost": r.churn.get("n_lost"),
            "estep_s": r.seconds.get("estep"),
            "mstep_s": r.seconds.get("mstep"),
            "graph_s": _graph_seconds(output_dir, r.round_index),
        }
        for r in results
    ]
    pcsv.write_csv(
        pa.Table.from_pylist(rows),
        str(output_dir / "churn.tsv"),
        write_options=pcsv.WriteOptions(delimiter="\t"),
    )


__all__ = ["EmParams", "RoundResult", "run_em"]
