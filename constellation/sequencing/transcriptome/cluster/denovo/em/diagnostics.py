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


def build_em_report(em_dir: Path) -> Path:
    """Assemble the EM diagnostics report under ``diagnostics/report.md``."""
    from constellation.sequencing._render import render_report

    em_dir = Path(em_dir)
    out = em_dir / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    sections = []
    for fn in (
        section_convergence,
        section_candidate_pool,
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
            "`transcriptome cluster --mode em` — seed → fold → "
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
]
