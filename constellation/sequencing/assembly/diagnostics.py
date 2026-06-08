"""Assembly diagnostics — per-stage + comparative quality reports.

Mirrors ``transcriptome/{align,cluster}/diagnostics.py``: pure
``section_<name>(...) -> ReportSection`` functions over an ``Assembly`` +
its ``ASSEMBLY_STATS`` row, assembled into a ``report.md`` by an
orchestrator that uses the shared :mod:`sequencing._render` helpers. Each
section is wrapped by :func:`_safe` so one broken metric never breaks the
whole report; flags hoist to a top-of-document summary.

The headline deliverable is :func:`section_pipeline_comparison` — how
quality changes across draft → scaffolded → polished (N50 / length /
contig count / GC / BUSCO completeness deltas), which is the analysis the
genome pipeline exists to surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa

from constellation.sequencing._render import (
    ReportSection,
    render_report,
    save_svg_figure,
)
from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.assembly.stats import gc_content


def _plt():
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _safe(fn: Callable[..., ReportSection], *args: Any, **kwargs: Any) -> ReportSection:
    """Run a section function; convert any exception into a stub section."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — a broken metric must not kill the report
        title = getattr(fn, "__name__", "section").replace("section_", "").replace(
            "_", " "
        )
        return ReportSection(
            title=title.capitalize(),
            body=f"_metric failed: {type(exc).__name__}: {exc}_",
        )


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _contig_lengths(assembly: Assembly) -> np.ndarray:
    return np.asarray(
        [int(x) for x in assembly.contigs.column("length").to_pylist()],
        dtype=np.int64,
    )


def _stat(assembly: Assembly, name: str) -> Any:
    rows = assembly.stats.to_pylist()
    return rows[0].get(name) if rows else None


def _per_contig_gc(assembly: Assembly) -> list[float]:
    seqs = assembly.sequences
    out: list[float] = []
    cid_col = seqs.column("contig_id")
    for i in range(seqs.num_rows):
        one = seqs.slice(i, 1)
        out.append(gc_content(one))
    # cid order not needed for the histogram
    _ = cid_col
    return out


# ──────────────────────────────────────────────────────────────────────
# per-stage sections
# ──────────────────────────────────────────────────────────────────────


def section_contiguity(assembly: Assembly, *, figures_dir: Path) -> ReportSection:
    s = assembly.stats.to_pylist()[0]
    flags: list[str] = []
    n_contigs = s["n_contigs"]
    lengths = np.sort(_contig_lengths(assembly))[::-1]
    body_rows = [
        "| metric | value |",
        "| --- | --- |",
        f"| contigs | {n_contigs:,} |",
        f"| scaffolds | {s['n_scaffolds'] if s['n_scaffolds'] is not None else '—'} |",
        f"| total length | {s['total_length']:,} bp |",
        f"| largest | {s['largest_contig']:,} bp |",
        f"| N50 | {s['n50']:,} bp |",
        f"| L50 | {s['l50']:,} |",
        f"| N90 | {s['n90']:,} bp |" if s["n90"] is not None else "| N90 | — |",
        f"| L90 | {s['l90']:,} |" if s["l90"] is not None else "| L90 | — |",
    ]
    figures: list[Path] = []
    if lengths.size:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        cum = np.cumsum(lengths) / lengths.sum() * 100.0
        ax.plot(np.arange(1, lengths.size + 1), cum, marker=".", lw=1)
        ax.axhline(50, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("contig rank (largest first)")
        ax.set_ylabel("cumulative length (%)")
        ax.set_title("Cumulative length curve")
        figures.append(
            save_svg_figure(fig, figures_dir / "contiguity_cumulative.svg")
        )
    if n_contigs and n_contigs > 5000:
        flags.append(f"contiguity: highly fragmented ({n_contigs:,} contigs)")
    return ReportSection(
        title="Contiguity",
        body="\n".join(body_rows),
        figures=figures,
        flags=flags,
    )


def section_size_distribution(
    assembly: Assembly, *, figures_dir: Path
) -> ReportSection:
    lengths = _contig_lengths(assembly)
    figures: list[Path] = []
    if lengths.size:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        logs = np.log10(np.maximum(lengths, 1))
        ax.hist(logs, bins=min(40, max(5, lengths.size)))
        ax.set_xlabel("log10(contig length bp)")
        ax.set_ylabel("count")
        ax.set_title("Contig length distribution")
        figures.append(save_svg_figure(fig, figures_dir / "size_distribution.svg"))
    p50 = int(np.median(lengths)) if lengths.size else 0
    return ReportSection(
        title="Size distribution",
        body=f"{lengths.size:,} contigs; median length {p50:,} bp.",
        figures=figures,
    )


def section_gc(assembly: Assembly, *, figures_dir: Path) -> ReportSection:
    overall = gc_content(assembly.sequences) if assembly.sequences.num_rows else 0.0
    flags: list[str] = []
    figures: list[Path] = []
    gcs = _per_contig_gc(assembly)
    if gcs:
        arr = np.asarray(gcs, dtype=np.float64)
        plt = _plt()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(arr, bins=min(40, max(5, arr.size)), range=(0, 1))
        ax.set_xlabel("per-contig GC fraction")
        ax.set_ylabel("count")
        ax.set_title("GC distribution")
        figures.append(save_svg_figure(fig, figures_dir / "gc_distribution.svg"))
        # crude bimodality check: a big spread can flag contamination
        if arr.size > 10 and float(arr.std()) > 0.12:
            flags.append(
                f"gc: wide per-contig GC spread (sd={arr.std():.3f}) — possible "
                "contamination"
            )
    return ReportSection(
        title="GC content",
        body=f"Overall GC: {overall * 100:.2f}%.",
        figures=figures,
        flags=flags,
    )


def section_coverage(assembly: Assembly, *, figures_dir: Path) -> ReportSection:
    cov = [
        float(x)
        for x in assembly.contigs.column("read_coverage").to_pylist()
        if x is not None
    ]
    if not cov:
        return ReportSection(
            title="Read coverage",
            body="_no per-contig read-coverage recorded (carried only on draft "
            "hifiasm output)._",
        )
    arr = np.asarray(cov, dtype=np.float64)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(arr, bins=min(40, max(5, arr.size)))
    ax.set_xlabel("contig read coverage")
    ax.set_ylabel("count")
    ax.set_title("Read-coverage distribution")
    fig_path = save_svg_figure(fig, figures_dir / "coverage_distribution.svg")
    flags: list[str] = []
    low = int((arr < max(1.0, float(np.median(arr)) * 0.1)).sum())
    if low:
        flags.append(f"coverage: {low:,} very-low-coverage contigs (possible artifacts)")
    return ReportSection(
        title="Read coverage",
        body=f"Median coverage {np.median(arr):.1f}× across {arr.size:,} contigs.",
        figures=[fig_path],
        flags=flags,
    )


def section_busco(assembly: Assembly, *, figures_dir: Path) -> ReportSection:
    s = assembly.stats.to_pylist()[0]
    if s.get("busco_complete") is None:
        return ReportSection(
            title="BUSCO completeness",
            body="_BUSCO not run (pass --busco-lineage to enable)._",
        )
    comp = s["busco_complete"]
    flags: list[str] = []
    fields = [
        ("single", s["busco_single"]),
        ("duplicated", s["busco_duplicated"]),
        ("fragmented", s["busco_fragmented"]),
        ("missing", s["busco_missing"]),
    ]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    labels = [k for k, _ in fields]
    vals = [(v or 0.0) * 100 for _, v in fields]
    ax.bar(labels, vals)
    ax.set_ylabel("% of orthologs")
    ax.set_title(f"BUSCO ({s.get('busco_lineage')})")
    fig_path = save_svg_figure(fig, figures_dir / "busco.svg")
    if comp is not None and comp < 0.9:
        flags.append(f"busco: low completeness (C={comp * 100:.1f}%)")
    if s["busco_duplicated"] is not None and s["busco_duplicated"] > 0.1:
        flags.append(
            f"busco: high duplication (D={s['busco_duplicated'] * 100:.1f}%) — "
            "unresolved haplotypes?"
        )
    return ReportSection(
        title="BUSCO completeness",
        body=(
            f"Complete: {comp * 100:.1f}% "
            f"(single {s['busco_single'] * 100:.1f}%, "
            f"duplicated {s['busco_duplicated'] * 100:.1f}%), "
            f"fragmented {s['busco_fragmented'] * 100:.1f}%, "
            f"missing {s['busco_missing'] * 100:.1f}%."
        ),
        figures=[fig_path],
        flags=flags,
    )


# ──────────────────────────────────────────────────────────────────────
# comparative section
# ──────────────────────────────────────────────────────────────────────


def section_pipeline_comparison(
    stage_stats: dict[str, pa.Table], *, figures_dir: Path
) -> ReportSection:
    """How quality changes across the ordered pipeline stages."""
    order = [s for s in ("draft", "scaffold", "polish") if s in stage_stats]
    rows = {s: stage_stats[s].to_pylist()[0] for s in order}
    flags: list[str] = []

    def fmt(v: Any, *, pct: bool = False) -> str:
        if v is None:
            return "—"
        if pct:
            return f"{v * 100:.1f}%"
        return f"{v:,}"

    header = "| metric | " + " | ".join(order) + " |"
    sep = "| --- |" + " --- |" * len(order)
    lines = [header, sep]
    for label, key, pct in [
        ("contigs", "n_contigs", False),
        ("total length (bp)", "total_length", False),
        ("N50 (bp)", "n50", False),
        ("largest (bp)", "largest_contig", False),
        ("GC", "gc_content", True),
        ("BUSCO complete", "busco_complete", True),
    ]:
        cells = [fmt(rows[s].get(key), pct=pct) for s in order]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # flags on regressions between consecutive stages
    for a, b in zip(order, order[1:]):
        n50_a, n50_b = rows[a].get("n50"), rows[b].get("n50")
        if n50_a and n50_b and n50_b < n50_a:
            flags.append(f"comparison: N50 regressed {a}→{b} ({n50_a:,}→{n50_b:,})")
        bc_a, bc_b = rows[a].get("busco_complete"), rows[b].get("busco_complete")
        if bc_a is not None and bc_b is not None and bc_b < bc_a - 0.005:
            flags.append(
                f"comparison: BUSCO completeness dropped {a}→{b} "
                f"({bc_a * 100:.1f}%→{bc_b * 100:.1f}%)"
            )

    figures: list[Path] = []
    if len(order) >= 2:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        n50s = [rows[s].get("n50") or 0 for s in order]
        ax.bar(order, n50s)
        ax.set_ylabel("N50 (bp)")
        ax.set_title("N50 across pipeline stages")
        figures.append(save_svg_figure(fig, figures_dir / "comparison_n50.svg"))

    return ReportSection(
        title="Pipeline comparison (draft → scaffold → polish)",
        body="\n".join(lines),
        figures=figures,
        flags=flags,
    )


# ──────────────────────────────────────────────────────────────────────
# orchestrators
# ──────────────────────────────────────────────────────────────────────


def generate_assembly_report(
    assembly: Assembly,
    *,
    output_dir: Path,
    stage_label: str = "assembly",
) -> Path:
    """Write a per-stage assembly diagnostic report under ``output_dir``."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    sections = [
        _safe(section_contiguity, assembly, figures_dir=figures_dir),
        _safe(section_size_distribution, assembly, figures_dir=figures_dir),
        _safe(section_gc, assembly, figures_dir=figures_dir),
        _safe(section_coverage, assembly, figures_dir=figures_dir),
        _safe(section_busco, assembly, figures_dir=figures_dir),
    ]
    return render_report(
        title=f"Genome assembly diagnostics — {stage_label}",
        intro=f"Stage: **{stage_label}**.",
        sections=sections,
        output_path=output_dir / "report.md",
    )


def generate_comparative_report(
    stage_stats: dict[str, pa.Table],
    *,
    output_dir: Path,
) -> Path:
    """Write the draft→scaffold→polish comparative report."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    sections = [
        _safe(section_pipeline_comparison, stage_stats, figures_dir=figures_dir),
    ]
    return render_report(
        title="Genome assembly — pipeline comparison",
        intro="How assembly quality changes through the sequential stages.",
        sections=sections,
        output_path=output_dir / "comparison.md",
    )


__all__ = [
    "section_contiguity",
    "section_size_distribution",
    "section_gc",
    "section_coverage",
    "section_busco",
    "section_pipeline_comparison",
    "generate_assembly_report",
    "generate_comparative_report",
]
