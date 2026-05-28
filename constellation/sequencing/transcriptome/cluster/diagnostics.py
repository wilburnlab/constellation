"""Diagnostic-report generator for ``constellation transcriptome cluster`` outputs.

Mirrors :mod:`constellation.sequencing.transcriptome.align.diagnostics`
but operates over the cluster-stage parquet artifacts
(``clusters.parquet``, ``cluster_membership.parquet``) plus optional
joins back to the upstream align dir (for intron-chain complexity) and
annotation (for the multi-gene-cluster-overlap fusion-candidate
diagnostic).

Each metric is a pure function returning a :class:`ReportSection`. The
orchestrator catches per-section exceptions so individual broken
metrics don't break the whole report; failures render as a stub note
on the affected section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq

from constellation.sequencing.transcriptome._render import (
    ReportSection,
    render_report,
    save_svg_figure,
)


# ──────────────────────────────────────────────────────────────────────
# Sections
# ──────────────────────────────────────────────────────────────────────


def section_cluster_size_distribution(
    clusters: pa.Table, *, figures_dir: Path,
) -> ReportSection:
    """Histogram of n_reads per cluster.

    Flags pathological distributions: > 80% singletons (suggests
    over-fragmentation, drift filter too tight) OR a top cluster
    holding > 5% of all reads (suggests an intronless mega-cluster
    swallowing many distinct transcripts).
    """
    flags: list[str] = []
    if clusters.num_rows == 0:
        return ReportSection(
            title="Cluster size distribution",
            body="No clusters.",
        )
    sizes = clusters.column("n_reads").to_numpy(zero_copy_only=False).astype(np.int64)
    n_clusters = int(sizes.size)
    total_reads = int(sizes.sum())
    n_singletons = int((sizes == 1).sum())
    pct_singletons = 100.0 * n_singletons / n_clusters
    top_size = int(sizes.max())
    top_pct = 100.0 * top_size / max(total_reads, 1)
    p50 = int(np.median(sizes))
    p95 = int(np.percentile(sizes, 95))
    p99 = int(np.percentile(sizes, 99))

    body = (
        f"{n_clusters:,} clusters covering {total_reads:,} reads. "
        f"Sizes: median = {p50}, p95 = {p95}, p99 = {p99}, "
        f"max = {top_size:,}. {n_singletons:,} singletons "
        f"({pct_singletons:.2f}%). Largest cluster holds "
        f"{top_size:,} reads ({top_pct:.2f}% of total)."
    )

    if pct_singletons > 80.0:
        flags.append(
            f"cluster_size: {pct_singletons:.1f}% of clusters are "
            "singletons. Likely the drift filter (--max-5p-drift / "
            "--max-3p-drift) is too tight, or many spurious-junction "
            "fingerprints exist (each becomes a unique singleton cluster)."
        )
    if top_pct > 5.0:
        flags.append(
            f"cluster_size: top cluster holds {top_pct:.1f}% of total "
            "reads. Often a mega intronless cluster — every unspliced "
            "read on a (contig, strand) collapses into one bucket. "
            "Inspect cluster_id with the largest n_reads; loosen drift "
            "filter or stratify per-sample to split."
        )

    fig_path = _plot_cluster_size_histogram(
        sizes, figures_dir / "cluster_size_distribution.svg",
    )
    return ReportSection(
        title="Cluster size distribution",
        body=body,
        figures=[fig_path],
        flags=flags,
    )


def section_cluster_span_vs_annotated_genes(
    clusters: pa.Table,
    annotation: Any,
    *,
    figures_dir: Path,
) -> ReportSection:
    """Distribution of cluster span widths vs annotated gene lengths."""
    flags: list[str] = []
    if clusters.num_rows == 0:
        return ReportSection(
            title="Cluster span vs annotated gene length",
            body="No clusters.",
        )
    if annotation is None:
        return ReportSection(
            title="Cluster span vs annotated gene length",
            body="No annotation available — skipping span-vs-gene-length overlay.",
        )

    # Genome-guided clusters have populated span_start/span_end
    spans_t = clusters.filter(
        pc.and_(
            pc.is_valid(clusters.column("span_start")),
            pc.is_valid(clusters.column("span_end")),
        )
    )
    if spans_t.num_rows == 0:
        return ReportSection(
            title="Cluster span vs annotated gene length",
            body="No clusters have span coordinates (de novo mode?).",
        )
    span_widths = (
        pc.subtract(spans_t.column("span_end"), spans_t.column("span_start"))
        .to_numpy(zero_copy_only=False).astype(np.int64)
    )
    span_widths = span_widths[span_widths > 0]
    if span_widths.size == 0:
        return ReportSection(
            title="Cluster span vs annotated gene length",
            body="Cluster spans non-positive (corrupt input?).",
        )

    genes = annotation.features_of_type("gene")
    if genes.num_rows == 0:
        return ReportSection(
            title="Cluster span vs annotated gene length",
            body="Annotation has no gene-type features.",
        )
    gene_lengths = (
        pc.subtract(genes.column("end"), genes.column("start"))
        .to_numpy(zero_copy_only=False).astype(np.int64)
    )
    gene_lengths = gene_lengths[gene_lengths > 0]

    gene_p99 = float(np.percentile(gene_lengths, 99)) if gene_lengths.size else 0.0
    cluster_p50 = int(np.median(span_widths))
    cluster_p99 = int(np.percentile(span_widths, 99))
    n_huge = int((span_widths > 2 * gene_p99).sum()) if gene_p99 > 0 else 0
    pct_huge = 100.0 * n_huge / span_widths.size if span_widths.size else 0.0

    body = (
        f"{span_widths.size:,} genome-guided clusters. Cluster span: "
        f"median = {cluster_p50:,} bp, p99 = {cluster_p99:,} bp. "
        f"Annotated genes p99 length = {gene_p99:,.0f} bp. "
        f"{n_huge:,} clusters ({pct_huge:.2f}%) have span > 2× annotated "
        f"gene p99."
    )

    if pct_huge > 5.0:
        flags.append(
            f"cluster_span: {pct_huge:.1f}% of clusters span > 2× the "
            f"annotated-gene p99 length ({2 * gene_p99:,.0f} bp). "
            "Likely fusion-style clusters created by minimap2 collapsing "
            "intergenic gaps; check the multi-gene-cluster section."
        )

    fig_path = _plot_span_vs_gene_length(
        span_widths, gene_lengths,
        figures_dir / "cluster_span_vs_annotated_gene_length.svg",
    )
    return ReportSection(
        title="Cluster span vs annotated gene length",
        body=body,
        figures=[fig_path],
        flags=flags,
    )


def section_multi_gene_cluster_overlap(
    clusters: pa.Table,
    annotation: Any,
    contig_name_to_id: dict[str, int] | None,
    *,
    figures_dir: Path,
) -> ReportSection:
    """Per-cluster count of overlapping annotated genes.

    Fusion smoking gun for cluster outputs. Same logic as the align-side
    helper but joins the cluster's ``contig_id`` (already an integer FK)
    directly to the annotation's ``contig_id`` — no ref_name → contig_id
    indirection needed.
    """
    flags: list[str] = []
    if clusters.num_rows == 0:
        return ReportSection(
            title="Multi-gene cluster overlap",
            body="No clusters.",
        )
    if annotation is None:
        return ReportSection(
            title="Multi-gene cluster overlap",
            body="No annotation available.",
        )
    genes = annotation.features_of_type("gene")
    if genes.num_rows == 0:
        return ReportSection(
            title="Multi-gene cluster overlap",
            body="Annotation has no gene-type features.",
        )

    spans_t = clusters.filter(
        pc.and_(
            pc.is_valid(clusters.column("contig_id")),
            pc.and_(
                pc.is_valid(clusters.column("span_start")),
                pc.is_valid(clusters.column("span_end")),
            ),
        )
    ).select(["cluster_id", "contig_id", "span_start", "span_end", "n_reads"])
    if spans_t.num_rows == 0:
        return ReportSection(
            title="Multi-gene cluster overlap",
            body=(
                "No clusters have populated contig_id / span (de novo mode?)."
            ),
        )

    overlap_counts = _count_gene_overlaps_per_cluster(spans_t, genes)
    bins = np.bincount(overlap_counts, minlength=5)
    n_total = int(overlap_counts.size)
    n_multi = int((overlap_counts >= 2).sum())
    pct_multi = 100.0 * n_multi / n_total if n_total else 0.0

    # Read-weighted view: how many *reads* are in fusion-candidate clusters?
    n_reads_per = spans_t.column("n_reads").to_numpy(
        zero_copy_only=False
    ).astype(np.int64)
    reads_in_multi = int(n_reads_per[overlap_counts >= 2].sum())
    reads_total = int(n_reads_per.sum())
    pct_reads_multi = (
        100.0 * reads_in_multi / reads_total if reads_total else 0.0
    )

    body = (
        f"Of {n_total:,} genome-guided clusters covering "
        f"{reads_total:,} reads: {n_multi:,} clusters ({pct_multi:.2f}%) "
        f"overlap ≥2 annotated genes, accounting for "
        f"{reads_in_multi:,} reads ({pct_reads_multi:.2f}% of clustered "
        f"reads).\n\n"
        + "| Genes overlapped | Clusters | Reads in clusters |\n"
        + "| --- | ---: | ---: |\n"
        + "\n".join(
            f"| {n} | {int(bins[n]):,} | "
            f"{int(n_reads_per[overlap_counts == n].sum()):,} |"
            for n in range(min(len(bins), 6))
        )
        + (
            f"\n| 6+ | {int(bins[6:].sum()):,} | "
            f"{int(n_reads_per[overlap_counts >= 6].sum()):,} |"
            if len(bins) > 6 else ""
        )
    )

    if pct_multi > 2.0 or pct_reads_multi > 2.0:
        flags.append(
            f"multi_gene_cluster: {pct_multi:.2f}% of clusters "
            f"({pct_reads_multi:.2f}% of clustered reads) overlap ≥2 "
            "annotated genes. The fusion smoking gun for the cluster "
            "stage — every such cluster groups reads that span multiple "
            "biological genes via a shared (likely spurious) splice "
            "topology. Fix the upstream align stage (set "
            "--organism-profile compact_eukaryote on `transcriptome "
            "align`) and re-cluster."
        )

    fig_path = _plot_multi_gene_cluster_bars(
        bins, n_reads_per, overlap_counts,
        figures_dir / "multi_gene_cluster_overlap.svg",
    )
    return ReportSection(
        title="Multi-gene cluster overlap",
        body=body,
        figures=[fig_path],
        flags=flags,
    )


def section_drift_filter_stats(
    membership: pa.Table, *, figures_dir: Path,
) -> ReportSection:
    """Drift filter outcome summary + signed drift distribution."""
    flags: list[str] = []
    if membership.num_rows == 0:
        return ReportSection(
            title="Drift filter statistics",
            body="No membership rows.",
        )
    roles = membership.column("role").to_pylist()
    role_counts: dict[str, int] = {}
    for r in roles:
        role_counts[r] = role_counts.get(r, 0) + 1
    n_total = len(roles)
    n_drifted = role_counts.get("drift_filtered", 0)
    pct_drifted = 100.0 * n_drifted / n_total if n_total else 0.0

    body_rows = []
    for role in sorted(role_counts.keys()):
        body_rows.append(
            f"| {role} | {role_counts[role]:,} | "
            f"{100.0 * role_counts[role] / n_total:.2f}% |"
        )
    body = (
        f"{n_total:,} membership rows. Role breakdown:\n\n"
        "| Role | Count | % |\n"
        "| --- | ---: | ---: |\n"
        + "\n".join(body_rows)
        + f"\n\nDrift-filtered fraction: {pct_drifted:.2f}%."
    )

    if pct_drifted > 20.0:
        flags.append(
            f"drift_filter: {pct_drifted:.1f}% of cluster members were "
            "drift-filtered. Often --max-5p-drift / --max-3p-drift are "
            "set tighter than the biological TSS / polyA scatter for "
            "this organism. Inspect the drift histograms below."
        )

    drift_filtered_rows = membership.filter(
        pc.equal(membership.column("role"), "drift_filtered")
    )
    if drift_filtered_rows.num_rows > 0:
        d5 = drift_filtered_rows.column("drift_5p_bp").to_numpy(
            zero_copy_only=False
        ).astype(np.int64)
        d3 = drift_filtered_rows.column("drift_3p_bp").to_numpy(
            zero_copy_only=False
        ).astype(np.int64)
        fig_path = _plot_drift_histograms(
            d5, d3, figures_dir / "drift_filter_distribution.svg",
        )
        figures = [fig_path]
    else:
        figures = []

    return ReportSection(
        title="Drift filter statistics",
        body=body,
        figures=figures,
        flags=flags,
    )


def section_layer0_derep(
    clusters: pa.Table, *, figures_dir: Path,
) -> ReportSection:
    """Distribution of n_unique_sequences / n_reads per cluster.

    Low ratios mean many reads in the same cluster collapse to the
    same trimmed window — fine for tight clusters of identical
    transcripts. The diagnostic surfaces the distribution shape for
    sanity-checking, but doesn't flag a specific threshold (clusters
    with truly identical sequences are biologically real).
    """
    if clusters.num_rows == 0:
        return ReportSection(
            title="Layer-0 dereplication ratio",
            body="No clusters.",
        )
    n_reads = clusters.column("n_reads").to_numpy(
        zero_copy_only=False
    ).astype(np.float64)
    n_unique = clusters.column("n_unique_sequences").to_numpy(
        zero_copy_only=False
    ).astype(np.float64)
    safe = n_reads > 0
    ratios = n_unique[safe] / n_reads[safe]
    if ratios.size == 0:
        return ReportSection(
            title="Layer-0 dereplication ratio",
            body="No clusters with positive n_reads.",
        )
    p50 = float(np.median(ratios))
    p10 = float(np.percentile(ratios, 10))
    p90 = float(np.percentile(ratios, 90))
    body = (
        f"{ratios.size:,} clusters. n_unique_sequences / n_reads ratio: "
        f"p10 = {p10:.3f}, median = {p50:.3f}, p90 = {p90:.3f}. "
        "Values near 1.0 mean every read in the cluster has a unique "
        "trimmed sequence (substitution noise dominates); values near "
        "0 mean many reads collapse to identical windows (clean cluster, "
        "or basecaller-level homogenization)."
    )
    fig_path = _plot_layer0_ratio(
        ratios, figures_dir / "layer0_derep_ratio.svg"
    )
    return ReportSection(
        title="Layer-0 dereplication ratio",
        body=body,
        figures=[fig_path],
    )


def section_intron_chain_complexity(
    clusters: pa.Table,
    blocks: pa.Table | None,
    alignments: pa.Table | None,
    *,
    figures_dir: Path,
) -> ReportSection:
    """Distribution of intron-chain length per cluster (proxy via the
    representative alignment's block count).

    Joins the cluster representative_read_id back to the align stage's
    alignment_blocks to count n_blocks per representative. Intronless
    clusters (n_blocks == 1) are called out separately — a mega
    intronless cluster on a (contig, strand) often signals an
    intronless-mega-cluster artifact.
    """
    flags: list[str] = []
    if clusters.num_rows == 0:
        return ReportSection(
            title="Intron-chain complexity",
            body="No clusters.",
        )
    if blocks is None or alignments is None:
        return ReportSection(
            title="Intron-chain complexity",
            body=(
                "Upstream `alignment_blocks/` or `alignments/` not "
                "available; skipping intron-chain diagnostic."
            ),
        )

    primary = alignments.filter(
        pc.and_(
            pc.invert(alignments.column("is_secondary")),
            pc.invert(alignments.column("is_supplementary")),
        )
    ).select(["alignment_id", "read_id"])
    rep_ids = clusters.column("representative_read_id")
    # Join cluster reps to alignment_ids via read_id, then count blocks
    rep_to_aln = primary.filter(pc.is_in(primary.column("read_id"), rep_ids))
    if rep_to_aln.num_rows == 0:
        return ReportSection(
            title="Intron-chain complexity",
            body="No representative reads found in primary alignments.",
        )
    aln_to_n_blocks = (
        blocks.select(["alignment_id", "block_index"])
        .join(rep_to_aln, keys="alignment_id", join_type="inner")
        .group_by("alignment_id")
        .aggregate([("block_index", "count")])
    )
    if aln_to_n_blocks.num_rows == 0:
        return ReportSection(
            title="Intron-chain complexity",
            body="No blocks for representative alignments.",
        )

    n_blocks = aln_to_n_blocks.column("block_index_count").to_numpy(
        zero_copy_only=False
    ).astype(np.int64)
    intronless = int((n_blocks == 1).sum())
    pct_intronless = 100.0 * intronless / n_blocks.size

    # Look up the largest intronless cluster (proxy: cluster with
    # max n_reads whose rep alignment has 1 block).
    intronless_aids = aln_to_n_blocks.filter(
        pc.equal(aln_to_n_blocks.column("block_index_count"), 1)
    ).column("alignment_id")
    intronless_read_ids = rep_to_aln.filter(
        pc.is_in(rep_to_aln.column("alignment_id"), intronless_aids)
    ).column("read_id")
    intronless_clusters = clusters.filter(
        pc.is_in(clusters.column("representative_read_id"), intronless_read_ids)
    )
    biggest_intronless = (
        int(intronless_clusters.column("n_reads").to_numpy(
            zero_copy_only=False
        ).max())
        if intronless_clusters.num_rows > 0 else 0
    )

    body = (
        f"{n_blocks.size:,} clusters with resolvable representative "
        f"alignments. n_blocks per representative (proxy for "
        f"intron-chain length): median = {int(np.median(n_blocks))}, "
        f"p99 = {int(np.percentile(n_blocks, 99))}. "
        f"{intronless:,} ({pct_intronless:.2f}%) clusters are intronless "
        f"(rep alignment has 1 block). Largest intronless cluster has "
        f"{biggest_intronless:,} reads."
    )
    if biggest_intronless > 1000:
        flags.append(
            f"intron_chain: largest intronless cluster has "
            f"{biggest_intronless:,} reads. On a dense genome this is "
            "the classic mega-bucket artifact — every unspliced read on "
            "a (contig, strand) collapses into one fingerprint and the "
            "drift filter is the only thing separating distinct "
            "transcripts. Try --per-sample-clusters or loosen drift."
        )

    fig_path = _plot_intron_chain_histogram(
        n_blocks, figures_dir / "intron_chain_complexity.svg",
    )
    return ReportSection(
        title="Intron-chain complexity",
        body=body,
        figures=[fig_path],
        flags=flags,
    )


def section_consensus_quality(
    membership: pa.Table, *, figures_dir: Path,
) -> ReportSection:
    """Distributions of per-member match_rate / indel_rate."""
    flags: list[str] = []
    if membership.num_rows == 0:
        return ReportSection(
            title="Consensus quality (per-member alignment stats)",
            body="No membership rows.",
        )
    mr = membership.column("match_rate").to_numpy(zero_copy_only=False)
    ir = membership.column("indel_rate").to_numpy(zero_copy_only=False)
    mr_valid = mr[~np.isnan(mr)] if mr.dtype.kind == "f" else mr[mr != None]  # noqa: E711
    ir_valid = ir[~np.isnan(ir)] if ir.dtype.kind == "f" else ir[ir != None]  # noqa: E711

    if mr_valid.size == 0:
        return ReportSection(
            title="Consensus quality (per-member alignment stats)",
            body=(
                "match_rate / indel_rate are null for every member — "
                "the upstream align run was invoked with `--no-emit-cs-tags` "
                "(see the align manifest)."
            ),
        )

    mr_p50 = float(np.median(mr_valid))
    mr_p10 = float(np.percentile(mr_valid, 10))
    ir_p50 = float(np.median(ir_valid)) if ir_valid.size else 0.0
    ir_p90 = float(np.percentile(ir_valid, 90)) if ir_valid.size else 0.0

    body = (
        f"{mr_valid.size:,} membership rows with cs:long-derived metrics.\n\n"
        f"match_rate: median = {mr_p50:.3f}, p10 = {mr_p10:.3f}.\n"
        f"indel_rate: median = {ir_p50:.3f}, p90 = {ir_p90:.3f}."
    )

    if mr_p50 < 0.90:
        flags.append(
            f"consensus_quality: median per-member match_rate = {mr_p50:.3f} "
            "(< 0.90). The clustering may be merging reads that diverge "
            "more than basecaller noise can account for — inspect the "
            "fingerprint definition or tighten clustering tolerance."
        )

    fig_path = _plot_consensus_quality(
        mr_valid, ir_valid, figures_dir / "consensus_quality.svg",
    )
    return ReportSection(
        title="Consensus quality (per-member alignment stats)",
        body=body,
        figures=[fig_path],
        flags=flags,
    )


# ──────────────────────────────────────────────────────────────────────
# Per-cluster gene-overlap counter
# ──────────────────────────────────────────────────────────────────────


def _count_gene_overlaps_per_cluster(
    spans: pa.Table, genes: pa.Table,
) -> np.ndarray:
    """Count annotated genes overlapping each cluster's span.

    ``spans`` is the cluster table filtered to populated contig_id /
    span_start / span_end. Same sort+searchsorted-per-contig pattern
    used in the align-side helper, but the join key is the integer
    ``contig_id`` already on both sides.
    """
    g_cid = np.array(genes.column("contig_id").to_pylist(), dtype=np.int64)
    g_start = np.array(genes.column("start").to_pylist(), dtype=np.int64)
    g_end = np.array(genes.column("end").to_pylist(), dtype=np.int64)

    per_contig: dict[int, dict[str, np.ndarray]] = {}
    for cid in np.unique(g_cid):
        mask = g_cid == cid
        s = g_start[mask]
        e = g_end[mask]
        order = np.argsort(s, kind="stable")
        sorted_s = s[order]
        sorted_e = e[order]
        per_contig[int(cid)] = {
            "starts": sorted_s,
            "ends": sorted_e,
            "ends_prefix_max": np.maximum.accumulate(sorted_e),
        }

    c_cid = spans.column("contig_id").to_numpy(zero_copy_only=False).astype(np.int64)
    c_start = spans.column("span_start").to_numpy(zero_copy_only=False).astype(np.int64)
    c_end = spans.column("span_end").to_numpy(zero_copy_only=False).astype(np.int64)

    counts = np.zeros(spans.num_rows, dtype=np.int64)
    for i in range(spans.num_rows):
        bucket = per_contig.get(int(c_cid[i]))
        if bucket is None:
            continue
        hi = int(np.searchsorted(bucket["starts"], int(c_end[i]), side="left"))
        if hi == 0:
            continue
        lo = int(np.searchsorted(
            bucket["ends_prefix_max"], int(c_start[i]), side="right"
        ))
        lo = min(lo, hi)
        if lo >= hi:
            continue
        cand_ends = bucket["ends"][lo:hi]
        counts[i] = int((cand_ends > int(c_start[i])).sum())
    return counts


# ──────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────


def _plot_cluster_size_histogram(sizes: np.ndarray, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(
        0, np.log10(max(int(sizes.max()), 10)), num=40
    )
    ax.hist(sizes, bins=bins, color="#2C7BB6", edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reads per cluster (log)")
    ax.set_ylabel("# clusters (log)")
    ax.set_title("Cluster size distribution")
    return save_svg_figure(fig, out_path)


def _plot_span_vs_gene_length(
    spans: np.ndarray, gene_lengths: np.ndarray, out_path: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    common_min = max(1, min(int(spans.min()), int(gene_lengths.min())))
    common_max = max(int(spans.max()), int(gene_lengths.max()))
    bins = np.logspace(np.log10(common_min), np.log10(common_max + 1), num=40)
    ax.hist(
        gene_lengths, bins=bins, color="#A6CEE3", alpha=0.7,
        label=f"annotated genes (n={gene_lengths.size:,})", edgecolor="none",
    )
    ax.hist(
        spans, bins=bins, color="#E31A1C", alpha=0.5,
        label=f"cluster spans (n={spans.size:,})", edgecolor="none",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Length (bp, log)")
    ax.set_ylabel("count")
    ax.set_title("Cluster span vs annotated gene length")
    ax.legend(loc="best")
    return save_svg_figure(fig, out_path)


def _plot_multi_gene_cluster_bars(
    bins: np.ndarray, n_reads_per: np.ndarray,
    overlap_counts: np.ndarray, out_path: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_cl, ax_r) = plt.subplots(1, 2, figsize=(11, 4))
    labels = [str(i) for i in range(min(len(bins), 6))]
    cl_vals = list(bins[: len(labels)].astype(int))
    if len(bins) > 6:
        labels.append("6+")
        cl_vals.append(int(bins[6:].sum()))
    r_vals = [
        int(n_reads_per[overlap_counts == i].sum())
        for i in range(min(len(bins), 6))
    ]
    if len(bins) > 6:
        r_vals.append(int(n_reads_per[overlap_counts >= 6].sum()))
    colors = ["#2C7BB6"] + ["#D7191C"] * (len(labels) - 1)
    ax_cl.bar(labels, cl_vals, color=colors, edgecolor="none")
    ax_cl.set_xlabel("Annotated genes overlapped")
    ax_cl.set_ylabel("# clusters")
    ax_cl.set_title("Clusters")
    ax_r.bar(labels, r_vals, color=colors, edgecolor="none")
    ax_r.set_xlabel("Annotated genes overlapped")
    ax_r.set_ylabel("# reads in clusters")
    ax_r.set_title("Reads (cluster-weighted)")
    fig.suptitle("Multi-gene cluster overlap (fusion-candidate diagnostic)")
    fig.tight_layout()
    return save_svg_figure(fig, out_path)


def _plot_drift_histograms(
    d5: np.ndarray, d3: np.ndarray, out_path: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax5, ax3) = plt.subplots(1, 2, figsize=(11, 4))
    ax5.hist(d5, bins=40, color="#2C7BB6", edgecolor="none")
    ax5.set_xlabel("5' drift (bp)")
    ax5.set_ylabel("# drift-filtered members")
    ax5.set_title("5' drift")
    ax3.hist(d3, bins=40, color="#FDAE61", edgecolor="none")
    ax3.set_xlabel("3' drift (bp)")
    ax3.set_title("3' drift")
    fig.suptitle("Drift-filtered member distribution")
    fig.tight_layout()
    return save_svg_figure(fig, out_path)


def _plot_layer0_ratio(ratios: np.ndarray, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=np.linspace(0, 1, 30), color="#2C7BB6", edgecolor="none")
    ax.set_xlabel("n_unique_sequences / n_reads per cluster")
    ax.set_ylabel("# clusters")
    ax.set_title("Layer-0 dereplication ratio")
    return save_svg_figure(fig, out_path)


def _plot_intron_chain_histogram(n_blocks: np.ndarray, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    max_x = min(int(n_blocks.max()), 50)
    bins = np.arange(1, max_x + 2)
    ax.hist(np.clip(n_blocks, 1, max_x), bins=bins, color="#2C7BB6", edgecolor="none")
    ax.set_xlabel("Blocks per cluster representative (≈ intron-chain length)")
    ax.set_ylabel("# clusters")
    ax.set_title("Intron-chain complexity")
    return save_svg_figure(fig, out_path)


def _plot_consensus_quality(
    mr: np.ndarray, ir: np.ndarray, out_path: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_mr, ax_ir) = plt.subplots(1, 2, figsize=(11, 4))
    ax_mr.hist(mr, bins=np.linspace(0, 1, 40), color="#2C7BB6", edgecolor="none")
    ax_mr.set_xlabel("match_rate")
    ax_mr.set_ylabel("# members")
    ax_mr.set_title("Per-member match_rate")
    ax_ir.hist(ir, bins=np.linspace(0, 1, 40), color="#FDAE61", edgecolor="none")
    ax_ir.set_xlabel("indel_rate")
    ax_ir.set_title("Per-member indel_rate")
    fig.suptitle("Consensus quality (cs:long-derived)")
    fig.tight_layout()
    return save_svg_figure(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────


def build_cluster_diagnostics_report(
    cluster_dir: Path,
    *,
    reference: Any = None,
    annotation: Any = None,
    output_dir: Path | None = None,
) -> Path:
    """Generate ``<cluster-dir>/diagnostics/report.md`` + figures.

    Reads ``clusters.parquet`` + ``cluster_membership.parquet`` from
    the cluster output dir, plus (optionally) ``alignments/`` +
    ``alignment_blocks/`` from the upstream align dir referenced by the
    cluster manifest (for the intron-chain section).

    If the cluster manifest's ``reference_path`` resolves and the
    caller didn't pass ``reference`` / ``annotation`` explicitly, the
    orchestrator best-effort loads both from disk. Annotation-dependent
    sections degrade to stubs if neither loads.

    Per-section exceptions are caught; broken sections render as stubs
    so the rest of the report still ships.
    """
    from constellation.sequencing.transcriptome.manifest import read_manifest_dir

    cluster_dir = Path(cluster_dir)
    if output_dir is None:
        output_dir = cluster_dir / "diagnostics"
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest_dir(cluster_dir)
    if manifest.kind != "cluster":
        raise ValueError(
            f"{cluster_dir} has manifest kind={manifest.kind!r}; expected 'cluster'"
        )

    if (reference is None or annotation is None) and manifest.reference_path:
        ref_root = Path(manifest.reference_path)
        if ref_root.is_dir():
            try:
                from constellation.sequencing.reference.io import (
                    load_genome_reference,
                )
                from constellation.sequencing.annotation.io import (
                    load_annotation,
                )

                if reference is None and (ref_root / "genome").is_dir():
                    reference = load_genome_reference(ref_root / "genome")
                if annotation is None and (ref_root / "annotation").is_dir():
                    annotation = load_annotation(ref_root / "annotation")
            except Exception:
                pass

    contig_name_to_id: dict[str, int] | None = None
    if reference is not None:
        contig_name_to_id = {
            str(n): int(c)
            for n, c in zip(
                reference.contigs.column("name").to_pylist(),
                reference.contigs.column("contig_id").to_pylist(),
                strict=True,
            )
        }

    clusters_path = cluster_dir / "clusters.parquet"
    membership_path = cluster_dir / "cluster_membership.parquet"
    clusters = pq.read_table(clusters_path) if clusters_path.is_file() else None
    membership = (
        pq.read_table(membership_path) if membership_path.is_file() else None
    )
    if clusters is None or membership is None:
        raise FileNotFoundError(
            f"{cluster_dir} missing clusters.parquet or "
            f"cluster_membership.parquet — was the cluster stage run?"
        )

    # Optional upstream align dir for intron-chain section
    align_dir = Path(manifest.align_dir) if manifest.align_dir else None
    blocks = None
    alignments = None
    if align_dir and align_dir.is_dir():
        blocks_dir = align_dir / "alignment_blocks"
        if blocks_dir.is_dir():
            try:
                blocks = pa_dataset.dataset(blocks_dir).to_table()
            except Exception:
                blocks = None
        alignments_dir = align_dir / "alignments"
        if alignments_dir.is_dir():
            try:
                alignments = pa_dataset.dataset(alignments_dir).to_table(
                    columns=[
                        "alignment_id", "read_id",
                        "is_secondary", "is_supplementary",
                    ]
                )
            except Exception:
                alignments = None

    sections: list[ReportSection] = []

    def _safe(title: str, fn, *args, **kwargs) -> ReportSection:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return ReportSection(
                title=title,
                body=(
                    f"_Section failed to render with `{type(exc).__name__}: "
                    f"{exc}`. Other diagnostics still ran; this is a "
                    "non-fatal report-generation error._"
                ),
            )

    sections.append(_section_run_parameters(manifest))
    sections.append(
        _safe(
            "Cluster size distribution",
            section_cluster_size_distribution, clusters,
            figures_dir=figures_dir,
        )
    )
    sections.append(
        _safe(
            "Cluster span vs annotated gene length",
            section_cluster_span_vs_annotated_genes, clusters, annotation,
            figures_dir=figures_dir,
        )
    )
    sections.append(
        _safe(
            "Multi-gene cluster overlap",
            section_multi_gene_cluster_overlap, clusters, annotation,
            contig_name_to_id, figures_dir=figures_dir,
        )
    )
    sections.append(
        _safe(
            "Drift filter statistics",
            section_drift_filter_stats, membership,
            figures_dir=figures_dir,
        )
    )
    sections.append(
        _safe(
            "Layer-0 dereplication ratio",
            section_layer0_derep, clusters,
            figures_dir=figures_dir,
        )
    )
    sections.append(
        _safe(
            "Intron-chain complexity",
            section_intron_chain_complexity, clusters, blocks, alignments,
            figures_dir=figures_dir,
        )
    )
    sections.append(
        _safe(
            "Consensus quality (per-member alignment stats)",
            section_consensus_quality, membership,
            figures_dir=figures_dir,
        )
    )

    intro = (
        f"Diagnostic report for `transcriptome cluster` output at "
        f"`{cluster_dir}`. Upstream align dir: "
        f"`{manifest.align_dir or 'unknown'}`. Reference: "
        f"`{manifest.reference_handle or manifest.reference_path}`."
    )
    return render_report(
        title="Transcriptome cluster — diagnostics",
        intro=intro,
        sections=sections,
        output_path=output_dir / "report.md",
    )


def _section_run_parameters(manifest: Any) -> ReportSection:
    params = manifest.parameters
    rows = [f"| `{k}` | `{params[k]}` |" for k in sorted(params.keys())]
    body = (
        f"Reference: `{manifest.reference_handle or manifest.reference_path}`.\n"
        f"Align dir: `{manifest.align_dir}`. Demux dir: "
        f"`{manifest.demux_dir}`.\n\n"
        "| Parameter | Value |\n"
        "| --- | --- |\n"
        + "\n".join(rows)
    )
    return ReportSection(title="Run parameters", body=body)


__all__ = [
    "build_cluster_diagnostics_report",
    "section_cluster_size_distribution",
    "section_cluster_span_vs_annotated_genes",
    "section_consensus_quality",
    "section_drift_filter_stats",
    "section_intron_chain_complexity",
    "section_layer0_derep",
    "section_multi_gene_cluster_overlap",
]
