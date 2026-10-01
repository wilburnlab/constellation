"""Per-round diagnostics for the EM loop.

Everything here is a pure function over artifacts the loop already writes —
``round.json``, ``assignments/``, ``mstep/nodes/``, ``lineage.parquet`` — so
a report can be regenerated against a finished run without re-running
anything, and adding a metric never changes the cost path.

The sections are chosen to answer the questions the design actually leaves
open, not to tour the outputs:

* **Candidate-pool saturation** is a FLAGS-level metric, not an
  informational one. The pool must hold every template within ``p_floor``,
  both rankers assume that, and minimap2 truncates by *score* — so a read
  whose list hit the ``-N`` cap had its pool truncated and its ranking
  arbitrated over an arbitrary subset.
* **Convergence** is the lineage-aware churn curve, because the raw one
  counts template splits as movement and never settles.
* **Singleton-cluster read fraction** is the number that says whether the
  round-1 ranking is converging singletons — with no prune, it is the only
  thing that reports self-capture.
* **Reference and ORF length per round** is the instrumentation for the
  deferred ledger #42 question (whether the covariance M-step over-trims
  ORFs, or whether that was an artifact the admission floor removes).
* **Template relationships** reports how each round's nodes relate — the same
  transcript within an end tolerance, or one contained in another — and what
  the merge predicate would collapse. It is read from the graph's own
  ``stats.json``, never from the edges, and it counts what is there AFTER a
  merge. The section it replaces flagged the pre-merge redundancy of a
  cluster table that had already been merged, and called templates differing
  at positions the M-step had separated reads on "redundant".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

from constellation.sequencing._render import ReportSection


def _rounds(em_dir: Path) -> list[tuple[int, Path]]:
    rd = Path(em_dir) / "rounds"
    if not rd.exists():
        return []
    out = []
    for d in sorted(rd.glob("r*")):
        if d.name[1:].isdigit() and (d / "_SUCCESS").exists():
            out.append((int(d.name[1:]), d))
    return out


def _round_json(d: Path) -> dict:
    path = d / "round.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _table(directory: Path, pattern: str = "part-*.parquet") -> pa.Table | None:
    files = sorted(Path(directory).glob(pattern))
    if not files:
        return None
    return pa_ds.dataset(files).to_table()


def section_convergence(em_dir: Path) -> ReportSection:
    """Templates, nodes, assignment and churn per round."""
    rows = []
    flags: list[str] = []
    for r, d in _rounds(em_dir):
        j = _round_json(d)
        est, churn = j.get("estep", {}), j.get("churn", {})
        rows.append(
            (
                r,
                j.get("n_templates", 0),
                j.get("n_nodes", 0),
                est.get("n_assigned", 0),
                est.get("n_unassigned", 0),
                churn.get("frac_changed"),
                churn.get("frac_changed_lineage"),
                churn.get("frac_unsettled"),
                churn.get("n_gained"),
                churn.get("n_lost"),
            )
        )
    if not rows:
        return ReportSection(title="Convergence", body="_no completed rounds_")

    lines = [
        "| round | templates | nodes | assigned | unassigned | churn (lineage) "
        "| gained | lost | **unsettled** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r, nt, nn, na, nu, _ch, chl, uns, gain, lost in rows:
        f = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
        i = lambda v: "—" if v is None else f"{v:,}"  # noqa: E731
        lines.append(
            f"| {r} | {nt:,} | {nn:,} | {na:,} | {nu:,} | {f(chl)} "
            f"| {i(gain)} | {i(lost)} | **{f(uns)}** |"
        )

    last = rows[-1]
    if last[7] is not None and last[7] > 0.02:
        flags.append(
            f"the loop had not converged when it stopped — unsettled reads were "
            f"still {last[7]:.3f} at round {last[0]}; consider --rounds"
        )
    body = "\n".join(lines) + (
        "\n\nChurn is **lineage-aware**: a read that moved only because its "
        "template split has not chosen differently, and counting it would "
        "leave the loop looking unconverged forever.\n\n"
        "**unsettled** is what the stopping rule reads — genuine switches plus "
        "gains plus losses, over reads assigned in *either* round. Lineage "
        "churn alone is measured only over reads assigned in **both**, so "
        "losing half the assignments scores zero as long as the survivors keep "
        "their lineage."
    )
    return ReportSection(title="Convergence", body=body, flags=flags)


def section_seeding(em_dir: Path) -> ReportSection:
    """What round 1 was given to work with, from whichever seeder ran.

    Two numbers here are not ordinary instrumentation. **Template count and
    bases** are what round 1's E-step cost is linear in — at 9.39M reads the
    ORF seeder's 3.78M templates / 6.12 Gb cost 13.95 h against the kmer
    seeder's 874k / 1.31 Gb at 2.89 h. And **the largest cluster's read
    fraction** is the chaining guard (ledger #50): connected components with
    unbounded ends put 76% of that corpus into ONE component at purity 0.154,
    a failure invisible below full scale, where the same setting peaks at 3%.
    """
    path = Path(em_dir) / "seed" / "stats.json"
    if not path.exists():
        return ReportSection(
            title="Seeding",
            body="_no seed/stats.json — this run predates seed-stage stats_",
        )
    s = json.loads(path.read_text())
    seeding = s.get("seeding", "orf")
    flags: list[str] = []

    rows = [("seeder", seeding), ("reads", f"{s.get('n_reads', 0):,}")]
    if "n_uniq" in s:
        rows.append(("unique reads", f"{s['n_uniq']:,}"))
    if seeding == "kmer":
        rows += [
            ("identity", f"{s.get('identity', '—')}"),
            (
                "ends (5':3')",
                f"{_ov_label(s.get('max_5p_overhang'))}:"
                f"{_ov_label(s.get('max_3p_overhang'))}",
            ),
            ("grouping", str(s.get("grouping", "—"))),
            ("minimizers", f"{s.get('n_minimizers', 0):,}"),
            ("candidate pairs", f"{s.get('n_candidates', 0):,}"),
            ("verified edges", f"{s.get('n_edges', 0):,}"),
        ]
    for key, label in (
        ("n_clusters", "templates"),
        ("n_orfs", "distinct ORFs"),
        ("n_clusters_ge2", "templates with ≥2 reads"),
        ("n_dropped_below_min_seed_reads", "dropped by --min-seed-reads"),
        ("template_mb", "template bases (Mb)"),
        ("reads_per_template", "reads per template"),
        ("singleton_frac", "singleton fraction"),
        ("seed_quality_median", "elected-read quality (median)"),
        ("seed_quality_ge_floor_frac", "elected reads clearing Q22"),
    ):
        if key in s:
            v = s[key]
            rows.append((label, f"{v:,}" if isinstance(v, int) else f"{v}"))

    largest = s.get("largest_cluster_reads")
    frac = s.get("largest_cluster_read_frac")
    if largest is not None and frac is not None:
        rows.append(("largest cluster", f"{largest:,} reads ({frac:.2%})"))
        # 1% of a real corpus is already two orders of magnitude past any
        # transcript's depth; on real data at the shipped gate it is 0.83%.
        if frac > 0.01 and largest >= 10_000:
            flags.append(
                f"the largest seed cluster holds {largest:,} reads "
                f"({frac:.1%} of the corpus) — far past any transcript's "
                "depth, so this is very likely connected-components chaining; "
                "check --max-3p-overhang / --identity"
            )

    lines = ["| | |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    body = "\n".join(lines) + (
        "\n\nRound 1's E-step cost is close to linear in template bases "
        "(measured: 1,306 Mb → 221 s, 2,455 Mb → 462 s at 9.4M reads), so "
        "this table is also the cost estimate for the round that follows it."
    )
    if seeding == "kmer":
        body += (
            "\n\nNo ORF column here, deliberately: kmer seeding partitions on "
            "read similarity and its templates carry no ORF. Proteins are "
            "predicted per NODE by the M-step, under `--min-aa-length`; see "
            "the reference-drift section."
        )
    return ReportSection(title="Seeding", body=body, flags=flags)


def _ov_label(v) -> str:
    """Render an overhang bound as the sweep writes them (`inf`, `100`)."""
    if v is None:
        return "—"
    return "inf" if int(v) >= (1 << 30) else str(int(v))


def section_candidate_pool(em_dir: Path) -> ReportSection:
    """Saturation of the ``-N`` candidate cap. FLAGS-level, by design."""
    rows, flags = [], []
    for r, d in _rounds(em_dir):
        est = _round_json(d).get("estep", {})
        frac = float(est.get("cap_hit_fraction", 0.0) or 0.0)
        rows.append((r, est.get("n_reads_seen", 0), est.get("n_cap_hit", 0), frac))
        if frac > 0.01:
            flags.append(
                f"round {r}: {frac:.2%} of reads hit the -N candidate cap, so "
                "their pool was TRUNCATED and this round's rankings are "
                "unsound for them — raise --minimap2-n"
            )
    if not rows:
        return ReportSection(title="Candidate pool", body="_no completed rounds_")

    lines = [
        "| round | reads | hit the -N cap | fraction |",
        "|---:|---:|---:|---:|",
    ]
    for r, n, cap, frac in rows:
        lines.append(f"| {r} | {n:,} | {cap:,} | {frac:.4%} |")
    body = "\n".join(lines) + (
        "\n\nThe pool must contain **every** template within `--p-floor`: both "
        "rankers assume it, and minimap2 truncates by *score*, where near-clone "
        "templates differ by 1–2 units. A truncated pool therefore means the "
        "ranking arbitrated over an arbitrary subset of the set it was supposed "
        "to rank — an answer to a different question, not a slightly worse one."
    )
    return ReportSection(title="Candidate pool", body=body, flags=flags)


def section_assignment_rule(em_dir: Path) -> ReportSection:
    """How often the winner is not the argmax, and how contested reads are."""
    rounds = _rounds(em_dir)
    if not rounds:
        return ReportSection(title="Assignment rule", body="_no completed rounds_")
    r, d = rounds[-1]
    tbl = _table(d / "assignments")
    if tbl is None or tbl.num_rows == 0:
        return ReportSection(title="Assignment rule", body="_no assignments_")

    assigned = tbl.filter(pc.greater_equal(tbl.column("template_id"), 0))
    n = assigned.num_rows
    if n == 0:
        return ReportSection(
            title="Assignment rule",
            body="_no read cleared the admission floor_",
            flags=["every read was rejected by --p-floor"],
        )
    n_adm = assigned.column("n_admitted").to_numpy(zero_copy_only=False)
    delta = assigned.column("logl_delta").to_numpy(zero_copy_only=False)
    overridden = np.nansum(delta > 0) if delta.size else 0

    body = (
        f"Round {r}, {n:,} assigned reads.\n\n"
        f"- uncontested (`n_admitted == 1`): **{(n_adm == 1).mean():.2%}**\n"
        f"- median admitted candidates: **{int(np.median(n_adm))}**\n"
        f"- p90 admitted candidates: **{int(np.percentile(n_adm, 90))}**\n"
        f"- winner is not the argmax: **{overridden / n:.2%}** "
        "(the abundance escape; every one is inside `--near-tie-delta-logl` "
        "*and* at `--support-ratio` the support)\n"
    )
    flags = []
    if n > 0 and overridden / n > 0.05:
        flags.append(
            f"the abundance escape fired on {overridden / n:.1%} of reads "
            "(target ≤5%); it is meant to be rare and decisive"
        )
    return ReportSection(title="Assignment rule", body=body, flags=flags)


def section_reference_drift(em_dir: Path) -> ReportSection:
    """Reference and ORF length per round — the ledger #42 instrumentation."""
    rows = []
    for r, d in _rounds(em_dir):
        nodes = _table(d / "mstep" / "nodes")
        if nodes is None or nodes.num_rows == 0:
            continue
        cons_len = pc.utf8_length(nodes.column("consensus")).to_numpy(
            zero_copy_only=False
        )
        orf_len = nodes.column("orf_end").to_numpy(zero_copy_only=False) - nodes.column(
            "orf_start"
        ).to_numpy(zero_copy_only=False)
        orf_len = orf_len[orf_len > 0]
        rows.append(
            (
                r,
                nodes.num_rows,
                float(np.median(cons_len)) if cons_len.size else 0.0,
                float(np.mean(cons_len)) if cons_len.size else 0.0,
                float(np.median(orf_len)) if orf_len.size else 0.0,
                float(orf_len.sum() / max(cons_len.sum(), 1)),
            )
        )
    if not rows:
        return ReportSection(title="Reference drift", body="_no nodes emitted_")

    lines = [
        "| round | nodes | median ref nt | mean ref nt | median ORF nt | coding fraction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r, n, med, mean, orf, frac in rows:
        lines.append(
            f"| {r} | {n:,} | {med:,.0f} | {mean:,.0f} | {orf:,.0f} | {frac:.3f} |"
        )

    flags = []
    if len(rows) > 1:
        first_orf, last_orf = rows[0][4], rows[-1][4]
        if first_orf > 0 and last_orf < 0.85 * first_orf:
            flags.append(
                f"median ORF fell {first_orf:,.0f} → {last_orf:,.0f} nt across "
                "rounds — the ledger #42 over-trim signature, which the "
                "admission floor was expected to remove"
            )
        first_ref, last_ref = rows[0][3], rows[-1][3]
        if first_ref > 0 and last_ref > 1.3 * first_ref:
            flags.append(
                f"mean reference length grew {first_ref:,.0f} → {last_ref:,.0f} nt "
                "— the end-extension ratchet (ledger #36)"
            )
    body = "\n".join(lines) + (
        "\n\nRecorded because ledger #42 measured the covariance M-step trimming "
        "median ORF 543 → 387 nt over six rounds. That was attributed partly to "
        "a faulty 'trimmed' classification and partly to the PWM being too "
        "greedy about which reads it averages — the second of which is exactly "
        "what `--p-floor` fixes, so these columns are how we find out."
    )
    return ReportSection(title="Reference drift", body=body, flags=flags)


def section_cluster_sizes(em_dir: Path) -> ReportSection:
    """Singleton fraction — with no prune, this is how self-capture shows up."""
    path = Path(em_dir) / "clusters.parquet"
    if not path.exists():
        return ReportSection(title="Cluster sizes", body="_no clusters.parquet_")
    clusters = pq.read_table(path, columns=["n_reads"])
    n = clusters.column("n_reads").to_numpy(zero_copy_only=False)
    if n.size == 0:
        return ReportSection(title="Cluster sizes", body="_no clusters_")

    total_reads = int(n.sum())
    singleton_reads = int(n[n == 1].sum())
    body = (
        f"- clusters: **{n.size:,}** over **{total_reads:,}** reads\n"
        f"- singleton clusters: **{(n == 1).mean():.2%}** of clusters, "
        f"**{singleton_reads / max(total_reads, 1):.2%}** of reads\n"
        f"- median / p90 / max cluster size: "
        f"**{int(np.median(n))} / {int(np.percentile(n, 90))} / {int(n.max())}**\n"
        "\nThere is no support prune, so a read that matches nothing else "
        "within `--p-floor` is kept as a lowly-supported transcript by design. "
        "A *high* singleton read fraction is therefore not a bug report — it "
        "is the number that says whether round 1's ranking is converging "
        "singletons or leaving every read on its own seed."
    )
    flags = []
    if singleton_reads / max(total_reads, 1) > 0.5:
        flags.append(
            f"{singleton_reads / total_reads:.0%} of reads sit in singleton "
            "clusters — round 1's ranking may not be converging them"
        )
    return ReportSection(title="Cluster sizes", body=body, flags=flags)


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _resplit_survivors(rd: Path, next_rd: Path | None) -> int | None:
    """Survivors of round ``rd``'s merge that the NEXT M-step split again.

    The M-step clusters span endpoints within 10 nt and the merge tolerance
    is 30, so two templates merged on an extent difference in between can be
    re-separated one round later. This is the count that shows it.
    """
    merged = rd / "merged.parquet"
    if next_rd is None or not merged.exists():
        return None
    survivors = pq.read_table(merged, columns=["survivor_template_id"]).column(0)
    if len(survivors) == 0:
        return 0
    files = sorted((next_rd / "mstep" / "nodes").glob("part-*.parquet"))
    if not files:
        return None
    parents = pa_ds.dataset(files).to_table(columns=["parent_template_id"]).column(0)
    ids, counts = np.unique(
        parents.to_numpy(zero_copy_only=False), return_counts=True
    )
    split = ids[counts > 1]
    return int(np.isin(np.unique(survivors.to_numpy(zero_copy_only=False)), split).sum())


def section_template_graph(em_dir: Path) -> ReportSection:
    """How each round's nodes relate, and what the merge did about it.

    ``equivalent`` is a statement of similarity, not of redundancy: below
    identity 1 the two templates differ at positions the M-step separated
    reads on. "Redundant" is kept for byte-identical twins, which reads
    cannot tell apart.

    A round's graph is shown only if the record of that round says this run
    built it: a directory extended under other settings can still hold a
    graph from before.
    """
    title = "Template relationships"
    rounds = _rounds(em_dir)
    if not rounds:
        return ReportSection(title=title, body="_no completed rounds_")

    final_r, final_d = rounds[-1]
    final = _json(final_d / "final.json")

    rows, flags, notes = [], [], []
    modes: set[str] = set()
    last_graph: dict | None = None
    for i, (r, d) in enumerate(rounds):
        record = _json(d / "refine.json")
        if record is None and d == final_d:
            record = final
        if record is None:
            if (d / "merge" / "stats.json").exists():
                notes.append(
                    f"r{r}: run with the minimap2 redundancy scan this version "
                    f"removed; its `merge/` record is not comparable and is "
                    f"not shown"
                )
            continue
        modes.add(str(record.get("template_graph")))
        if record.get("graph") == "failed":
            notes.append(
                f"r{r}: the template graph failed "
                f"({record.get('graph_error', 'no reason recorded')}); "
                f"nothing depended on it"
            )
        st = _json(d / "graph" / "stats.json")
        if record.get("graph") != "ok" or st is None:
            continue
        last_graph = st
        merged = int(record.get("n_merged", 0))
        # Between rounds what is left is the record's; the final round's is
        # counted over the clusters and reported on the line below.
        still = record.get("n_still_mergeable", st.get("n_mergeable", 0))
        nxt = rounds[i + 1][1] if i + 1 < len(rounds) else None
        resplit = _resplit_survivors(d, nxt) if merged else None
        contained = st.get("contained", {})
        rows.append(
            f"| r{r} | {st.get('n_sequences', 0):,} | {st.get('n_equivalent', 0):,} "
            f"| {st.get('n_exact_twins', 0):,} | {st.get('n_same_split_origin', 0):,} "
            f"| {contained.get('5p', 0):,} / {contained.get('3p', 0):,} / "
            f"{contained.get('both', 0):,} | {st.get('n_exact_nested', 0):,} "
            f"| {st.get('n_mergeable', 0):,} | {merged:,} | {int(still):,} "
            f"| {'—' if resplit is None else f'{resplit:,}'} "
            f"| {st.get('seconds', {}).get('total', 0):,.1f} |"
        )
        hit = st.get("n_overflow_templates", 0) + st.get("n_truncated_templates", 0)
        if hit:
            n = max(int(st.get("n_sequences", 0)), 1)
            flags.append(
                f"r{r}: {hit:,} templates ({hit / n:.1%}) hit a candidate cap "
                f"({st.get('n_overflow_templates', 0):,} in minimizer buckets "
                f"above the cap, {st.get('n_truncated_templates', 0):,} with "
                f"more candidates than are kept) — their edge lists are "
                f"incomplete"
            )
        kin = int(record.get("n_merged_kin", 0))
        if kin:
            flags.append(
                f"r{r}: {kin:,} of {merged:,} merges rejoined templates an "
                f"M-step split had separated (--merge-siblings) — the "
                f"split/merge cycle measured on 2026-09-24"
            )

    parts = []
    if rows:
        parts.append(
            "\n".join(
                [
                    "| nodes of | nodes | equivalent | exact twins | same split "
                    "| contained 5' / 3' / both | exact nested | mergeable "
                    "| merged | still mergeable | re-split next round | graph s |",
                    "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
                    *rows,
                ]
            )
        )
        parts.append(
            "_Counts are edges of that round's nodes, before its merge. "
            "`equivalent`: same extent within the end tolerance. `exact "
            "twins`: byte-identical. `same split`: equivalent pairs an M-step "
            "split separated. `exact nested`: contained at zero edits. "
            "`mergeable`: what the run's merge predicate accepts, whether or "
            "not the run merges; `still mergeable`: those of them whose two "
            "templates were both still there afterwards. `re-split`: "
            "survivors of the merge that the next M-step split again._"
        )
    elif modes and modes <= {"off"}:
        parts.append("_template graph disabled (`--template-graph off`)_")
    elif modes and modes <= {"final", "off"} and final is None:
        parts.append(
            "_the template graph is built for the final round only "
            "(`--template-graph final`), and this run has not reached it_"
        )
    elif not notes:
        parts.append("_no template graph was built_")

    if final is not None and rows:
        if final.get("template_graph") == "off":
            parts.append("Final output: template graph disabled (`--template-graph off`).")
        elif final.get("graph") == "failed":
            parts.append(
                f"Final output: the template graph failed "
                f"({final.get('graph_error', 'no reason recorded')}); the "
                f"clusters are unaffected and `cluster_edges.parquet` was not "
                f"written."
            )
        elif final.get("cluster_edges_error"):
            parts.append(
                f"Final output: `cluster_edges.parquet` could not be written "
                f"({final['cluster_edges_error']}); the clusters are "
                f"unaffected."
            )
        elif final.get("graph") == "ok":
            n_clusters = int(final.get("n_clusters", 0))
            twins = int(final.get("n_twin_clusters", 0))
            rebuilt = ""
            if final.get("rebuild") == "ok":
                rebuilt = (
                    f" The {final.get('n_rebuilt', 0):,} merged survivors were "
                    f"rebuilt from their pooled reads"
                    + (
                        f" ({final['n_rebuild_failed']:,} kept their own "
                        f"consensus, nothing placed)"
                        if final.get("n_rebuild_failed")
                        else ""
                    )
                    + "."
                )
            elif final.get("merge_applied") and final.get("n_merged"):
                rebuilt = f" Survivors were not rebuilt: {final.get('rebuild')}."
            parts.append(
                f"**Final output (r{final_r})**: {final.get('n_nodes', 0):,} "
                f"nodes → **{n_clusters:,} clusters** "
                f"({final.get('n_merged', 0):,} merged), "
                f"{final.get('n_cluster_edges', 0):,} edges between them; "
                f"{final.get('n_still_mergeable', 0):,} still accepted by the "
                f"merge predicate, {twins:,} clusters byte-identical to "
                f"another.{rebuilt}"
            )
            if final.get("rebuild") == "ok" and final.get("n_rebuild_failed"):
                flags.append(
                    f"final output: {final['n_rebuild_failed']:,} merged "
                    f"survivors could not be rebuilt from their reads and kept "
                    f"their own consensus"
                )
            if n_clusters and twins / n_clusters > 0.05:
                flags.append(
                    f"final output: {twins:,} clusters ({twins / n_clusters:.1%}) "
                    f"are byte-identical to another cluster, so their reads "
                    f"are split across identical references; `--merge` "
                    f"collapses them"
                )
    if last_graph is not None:
        hist = last_graph.get("n_edits_hist") or {}
        dropped = last_graph.get("dropped") or {}
        if hist:
            parts.append(
                "Edits over the shared span, `equivalent` edges of the last "
                "graph:\n\n| edits | edges |\n|---|---:|\n"
                + "\n".join(f"| {k} | {v:,} |" for k, v in hist.items())
            )
        if dropped:
            parts.append(
                "Candidate pairs that are not edges, last graph. A pair over "
                "the edit budget is aligned without a trace, so an "
                "alternative end on a short template is counted "
                "`below_floor` and the same end on a long one "
                "`divergent`:\n\n| reason | pairs |\n|---|---:|\n"
                + "\n".join(f"| {k} | {v:,} |" for k, v in dropped.items())
            )
    if notes:
        parts.append("\n".join(f"- {n}" for n in notes))
    return ReportSection(title=title, body="\n\n".join(parts), flags=flags)


def build_em_report(em_dir: Path) -> Path:
    """Assemble the EM diagnostics report under ``diagnostics/report.md``."""
    from constellation.sequencing._render import render_report

    em_dir = Path(em_dir)
    out = em_dir / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    sections = []
    for fn in (
        section_seeding,
        section_convergence,
        section_candidate_pool,
        section_template_graph,
        section_assignment_rule,
        section_reference_drift,
        section_cluster_sizes,
    ):
        try:
            sections.append(fn(em_dir))
        except Exception as exc:  # noqa: BLE001 — one metric never sinks a report
            sections.append(
                ReportSection(
                    title=fn.__name__.replace("section_", "").replace("_", " ").title(),
                    body=f"_could not be computed: {type(exc).__name__}: {exc}_",
                )
            )
    return render_report(
        title="EM clustering diagnostics",
        intro=(
            "`transcriptome cluster --mode em-{orf,kmer}` — seed → "
            "[E-step → M-step → refine] × N. Every metric below is a pure "
            "function over artifacts the loop already wrote, so this can be "
            "regenerated against a finished run without re-running anything."
        ),
        sections=sections,
        output_path=out / "report.md",
    )


__all__ = [
    "build_em_report",
    "section_assignment_rule",
    "section_candidate_pool",
    "section_cluster_sizes",
    "section_convergence",
    "section_reference_drift",
    "section_seeding",
    "section_template_graph",
]
