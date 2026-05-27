"""Diagnostic-report generator for ``constellation transcriptome align`` outputs.

Reads the parquet outputs produced by an align run (``introns.parquet``,
``alignments/``, ``alignment_blocks/``, ``gene_assignments/``,
``read_demux/`` via the upstream demux dir, plus the manifest) and emits
a single ``report.md`` + ``figures/*.svg`` bundle under
``<align-dir>/diagnostics/`` (or a caller-supplied path).

The report surfaces the algorithmic signals the pipeline otherwise
throws away — intron-length distribution, splice motif composition,
annotated-junction agreement, alignment complexity, MAPQ /
aligned-fraction distributions, and (the smoking gun for over-fusion
on dense genomes) the per-alignment multi-gene-overlap count.

Each metric function is a pure function from Arrow tables (+ optional
annotation) to a :class:`ReportSection`. The orchestrator wires them
together. Pipeline callers (``_cmd_transcriptome_align``) invoke the
orchestrator after their final ``_SUCCESS`` marker; the standalone
``constellation transcriptome diagnose`` verb re-runs the orchestrator
against an existing align dir without touching the pipeline stages.

Failure mode: any individual metric that raises is caught by the
orchestrator, logged as a warning, and its section replaced with a
stub noting the failure. Reports never break a successful pipeline.
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


# Profile-aware thresholds. Used to flag e.g. "any intron >50 kb on a
# compact_eukaryote run is biologically implausible". Keys match the
# preset ids in constellation/data/sequencing/minimap2_splice_presets.json.
_PROFILE_MAX_BIOLOGICAL_INTRON_BP: dict[str, int] = {
    "compact_eukaryote": 50_000,
    "intermediate_eukaryote": 100_000,
    "animal": 1_000_000,
}


# ──────────────────────────────────────────────────────────────────────
# Per-metric sections
# ──────────────────────────────────────────────────────────────────────


def section_intron_length_distribution(
    introns: pa.Table,
    *,
    figures_dir: Path,
    organism_profile: str | None = None,
) -> ReportSection:
    """Histogram intron lengths (acceptor - donor), colored by motif.

    Flags any intron longer than the profile's biological-plausibility
    cap (50 kb for compact eukaryotes, 100 kb for intermediates, 1 Mb
    for animals).
    """
    flags: list[str] = []
    if introns.num_rows == 0:
        return ReportSection(
            title="Intron length distribution",
            body="No introns observed (empty `introns.parquet`).",
        )

    # Use trusted intron seeds only (one row per cluster). Length =
    # acceptor_pos - donor_pos.
    seeds = introns.filter(pc.equal(introns.column("is_intron_seed"), True))
    if seeds.num_rows == 0:
        return ReportSection(
            title="Intron length distribution",
            body="No intron-cluster seeds in `introns.parquet`.",
        )

    lengths = (
        pc.subtract(seeds.column("acceptor_pos"), seeds.column("donor_pos"))
        .to_numpy(zero_copy_only=False)
        .astype(np.int64)
    )
    lengths = lengths[lengths > 0]
    if lengths.size == 0:
        return ReportSection(
            title="Intron length distribution",
            body="All intron lengths non-positive (corrupt input?).",
        )

    motifs = seeds.column("motif").to_pylist()

    cap = _PROFILE_MAX_BIOLOGICAL_INTRON_BP.get(organism_profile or "", None)
    p50 = int(np.median(lengths))
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    p_max = int(lengths.max())
    n_huge = int((lengths > 2 * p99).sum())

    body = (
        f"{seeds.num_rows:,} intron-cluster seeds. "
        f"Length percentiles: p50 = {p50:,} bp, p95 = {p95:,} bp, "
        f"p99 = {p99:,} bp, max = {p_max:,} bp.\n\n"
        f"{n_huge:,} seeds exceed 2× p99 (long-tail / candidate "
        f"spurious-junction reads)."
    )

    if cap is not None and p_max > cap:
        n_over_cap = int((lengths > cap).sum())
        flags.append(
            f"intron_length: {n_over_cap:,} clusters > {cap:,} bp on a "
            f"{organism_profile!r} run (biologically implausible — "
            f"likely minimap2 collapsing intergenic gaps into spurious "
            f"introns; consider --max-intron-length)"
        )
    if cap is None and p_max > 200_000:
        flags.append(
            f"intron_length: max intron is {p_max:,} bp with no "
            "--organism-profile set; for any compact or intermediate "
            "genome, tune --max-intron-length"
        )

    fig_path = _plot_intron_length_histogram(
        lengths, motifs, figures_dir / "intron_length_distribution.svg",
        organism_profile=organism_profile,
    )
    return ReportSection(
        title="Intron length distribution",
        body=body,
        figures=[fig_path],
        flags=flags,
    )


def section_motif_composition(
    introns: pa.Table, *, figures_dir: Path
) -> ReportSection:
    """Tabulate splice motifs across intron-cluster seeds and read support."""
    flags: list[str] = []
    if introns.num_rows == 0:
        return ReportSection(
            title="Splice motif composition",
            body="No introns observed.",
        )
    seeds = introns.filter(pc.equal(introns.column("is_intron_seed"), True))
    if seeds.num_rows == 0:
        return ReportSection(
            title="Splice motif composition",
            body="No intron-cluster seeds.",
        )

    motifs = seeds.column("motif").to_pylist()
    read_counts = seeds.column("read_count").to_pylist()

    canonical = ("GT-AG", "GC-AG", "AT-AC")
    motif_clusters: dict[str, int] = {m: 0 for m in canonical}
    motif_clusters["other"] = 0
    motif_reads: dict[str, int] = {m: 0 for m in canonical}
    motif_reads["other"] = 0
    for m, n in zip(motifs, read_counts, strict=True):
        bucket = str(m) if m in canonical else "other"
        motif_clusters[bucket] += 1
        motif_reads[bucket] += int(n)

    total_clusters = sum(motif_clusters.values())
    total_reads = sum(motif_reads.values())

    rows = []
    for m in (*canonical, "other"):
        n_cl = motif_clusters[m]
        n_r = motif_reads[m]
        pct_cl = 100.0 * n_cl / total_clusters if total_clusters else 0.0
        pct_r = 100.0 * n_r / total_reads if total_reads else 0.0
        rows.append(
            f"| {m} | {n_cl:,} | {pct_cl:.2f}% | {n_r:,} | {pct_r:.2f}% |"
        )
    table_md = (
        "| Motif | Distinct clusters | % | Total reads | % |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
    )

    other_pct_reads = (
        100.0 * motif_reads["other"] / total_reads if total_reads else 0.0
    )
    if other_pct_reads > 5.0:
        flags.append(
            f"motif_composition: 'other' (non-canonical) motifs carry "
            f"{other_pct_reads:.2f}% of supporting reads — exceeds the 5% "
            "threshold; consider raising --non-canonical-cost or "
            "supplying --junc-bed"
        )

    fig_path = _plot_motif_composition(
        motif_clusters, motif_reads,
        figures_dir / "motif_composition.svg",
    )
    return ReportSection(
        title="Splice motif composition",
        body=(
            f"Among {total_clusters:,} intron-cluster seeds supporting "
            f"{total_reads:,} junction-supporting reads:\n\n"
            + table_md
        ),
        figures=[fig_path],
        flags=flags,
    )


def section_annotated_junction_agreement(
    introns: pa.Table, *, figures_dir: Path,
) -> ReportSection:
    """How well do observed intron-cluster seeds match the input annotation?"""
    flags: list[str] = []
    if introns.num_rows == 0:
        return ReportSection(
            title="Annotated junction agreement",
            body="No introns observed.",
        )
    seeds = introns.filter(pc.equal(introns.column("is_intron_seed"), True))
    if seeds.num_rows == 0:
        return ReportSection(
            title="Annotated junction agreement",
            body="No intron-cluster seeds.",
        )

    ann = seeds.column("annotated").to_pylist()
    counts = seeds.column("read_count").to_pylist()
    has_annotation = any(a is not None for a in ann)
    if not has_annotation:
        return ReportSection(
            title="Annotated junction agreement",
            body=(
                "No annotation was supplied to `aggregate_junctions` "
                "(the `annotated` column is null for every seed). "
                "Re-run align with an annotated reference to enable "
                "this diagnostic."
            ),
        )

    trusted_mask = np.array(
        [bool(a) for a in ann if a is not None], dtype=bool
    )
    n_total = trusted_mask.size
    n_ann = int(trusted_mask.sum())
    n_novel = n_total - n_ann
    pct_novel = 100.0 * n_novel / n_total if n_total else 0.0

    # Per-read-count tier breakdown: how does the annotated fraction
    # change as we restrict to higher-support junctions?
    tiers = [1, 3, 10, 30, 100]
    rows = []
    counts_arr = np.array(counts, dtype=np.int64)
    ann_arr = np.array([1 if a else 0 for a in ann], dtype=np.int64)
    for t in tiers:
        mask = counts_arr >= t
        tot = int(mask.sum())
        if tot == 0:
            rows.append(f"| ≥ {t} reads | 0 | – | – |")
            continue
        ann_tot = int(ann_arr[mask].sum())
        pct = 100.0 * ann_tot / tot
        rows.append(f"| ≥ {t} reads | {tot:,} | {ann_tot:,} | {pct:.2f}% |")
    table_md = (
        "| Min. supporting reads | Total seeds | Annotated seeds | % annotated |\n"
        "| --- | ---: | ---: | ---: |\n"
        + "\n".join(rows)
    )

    # Flag heuristic: at read_count >= 3, expect well-annotated organisms
    # to have <20% novel junctions; >20% suggests the annotation may be
    # outdated OR there are systematic false-positive junctions.
    mask_3 = counts_arr >= 3
    if mask_3.any():
        novel_3 = 1.0 - (ann_arr[mask_3].sum() / mask_3.sum())
        if novel_3 > 0.20:
            flags.append(
                f"junction_agreement: {novel_3 * 100:.1f}% of trusted "
                "(read_count ≥ 3) intron seeds do NOT match the annotation. "
                "Either the annotation is sparse or many trusted-by-count "
                "junctions are systematic alignment artifacts."
            )

    fig_path = _plot_annotated_fraction_by_support(
        counts_arr, ann_arr,
        figures_dir / "annotated_junction_agreement.svg",
    )
    body = (
        f"Of {n_total:,} intron-cluster seeds: {n_ann:,} match the "
        f"annotation, {n_novel:,} ({pct_novel:.2f}%) are novel.\n\n"
        f"Annotated fraction as a function of supporting-read threshold:\n\n"
        + table_md
    )
    return ReportSection(
        title="Annotated junction agreement",
        body=body,
        figures=[fig_path],
        flags=flags,
    )


def section_alignment_complexity(
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    *,
    figures_dir: Path,
) -> ReportSection:
    """Histogram n_blocks per primary alignment (a proxy for splice complexity)."""
    flags: list[str] = []
    if alignments.num_rows == 0 or alignment_blocks.num_rows == 0:
        return ReportSection(
            title="Alignment complexity (blocks per read)",
            body="No alignments / no blocks.",
        )

    primary = alignments.filter(
        pc.and_(
            pc.invert(alignments.column("is_secondary")),
            pc.invert(alignments.column("is_supplementary")),
        )
    ).select(["alignment_id"])
    if primary.num_rows == 0:
        return ReportSection(
            title="Alignment complexity (blocks per read)",
            body="No primary alignments survived filtering.",
        )

    n_blocks_per_aln = (
        alignment_blocks.select(["alignment_id", "block_index"])
        .join(primary, keys="alignment_id", join_type="inner")
        .group_by("alignment_id")
        .aggregate([("block_index", "count")])
    )
    if n_blocks_per_aln.num_rows == 0:
        return ReportSection(
            title="Alignment complexity (blocks per read)",
            body="No primary alignments have any blocks.",
        )

    counts = n_blocks_per_aln.column("block_index_count").to_numpy(
        zero_copy_only=False
    ).astype(np.int64)

    p50 = int(np.median(counts))
    p95 = int(np.percentile(counts, 95))
    p99 = int(np.percentile(counts, 99))
    n_very_complex = int((counts > 30).sum())

    body = (
        f"{counts.size:,} primary alignments. n_blocks per alignment "
        f"(blocks ≈ exons): median = {p50}, p95 = {p95}, "
        f"p99 = {p99}, max = {int(counts.max())}. "
        f"{n_very_complex:,} alignments have > 30 blocks."
    )

    if p50 > 20:
        flags.append(
            f"alignment_complexity: median n_blocks = {p50} (> 20). "
            "For most organisms this is suspicious — even mammalian "
            "transcripts typically have < 15 exons. Likely a "
            "splice-DP-gone-wild signal; check intron-length distribution."
        )

    fig_path = _plot_n_blocks_histogram(
        counts, figures_dir / "alignment_complexity.svg",
    )
    return ReportSection(
        title="Alignment complexity (blocks per read)",
        body=body,
        figures=[fig_path],
        flags=flags,
    )


def section_mapq_distribution(
    alignments: pa.Table, *, figures_dir: Path,
) -> ReportSection:
    """Histogram MAPQ across primary alignments."""
    flags: list[str] = []
    if alignments.num_rows == 0:
        return ReportSection(
            title="MAPQ distribution",
            body="No alignments.",
        )
    primary = alignments.filter(
        pc.and_(
            pc.invert(alignments.column("is_secondary")),
            pc.invert(alignments.column("is_supplementary")),
        )
    )
    if primary.num_rows == 0:
        return ReportSection(title="MAPQ distribution", body="No primary alignments.")

    mapq = primary.column("mapq").to_numpy(zero_copy_only=False).astype(np.int64)
    p50 = int(np.median(mapq))
    pct_zero = 100.0 * int((mapq == 0).sum()) / mapq.size
    pct_low = 100.0 * int((mapq < 30).sum()) / mapq.size

    body = (
        f"{mapq.size:,} primary alignments. Median MAPQ = {p50}; "
        f"{pct_zero:.2f}% at MAPQ 0; {pct_low:.2f}% below MAPQ 30."
    )
    if p50 < 30:
        flags.append(
            f"mapq: median MAPQ = {p50} < 30 — many ambiguous alignments. "
            "Consider --min-mapq 1 (drop MAPQ-0) or higher."
        )

    fig_path = _plot_mapq_histogram(mapq, figures_dir / "mapq_distribution.svg")
    return ReportSection(
        title="MAPQ distribution",
        body=body,
        figures=[fig_path],
        flags=flags,
    )


def section_multi_gene_alignments(
    alignments: pa.Table,
    annotation: Any,
    contig_name_to_id: dict[str, int] | None,
    *,
    figures_dir: Path,
) -> ReportSection:
    """Per-alignment count of overlapping annotated genes.

    The fusion smoking gun: a single primary alignment whose ref_span
    overlaps ≥ 2 annotated genes is either a fusion transcript (rare),
    a transcriptional read-through, or — most often on compact dense
    genomes — minimap2 collapsing intergenic gaps into a spurious
    long intron.

    ``contig_name_to_id`` maps the alignment's ``ref_name`` column
    (BAM contig name) to the annotation's ``contig_id`` integer (FK
    into the genome reference). Without it we can't join the two
    tables and the section degrades to a stub.
    """
    flags: list[str] = []
    if alignments.num_rows == 0:
        return ReportSection(
            title="Multi-gene alignment counts",
            body="No alignments.",
        )
    if annotation is None:
        return ReportSection(
            title="Multi-gene alignment counts",
            body=(
                "No annotation available — pass an annotated reference "
                "or load the manifest's reference to enable this diagnostic."
            ),
        )
    if not contig_name_to_id:
        return ReportSection(
            title="Multi-gene alignment counts",
            body=(
                "Annotation supplied but no GenomeReference contig-name "
                "lookup — the section needs to join the alignments' "
                "`ref_name` column to the annotation's `contig_id` "
                "integers. Pass a `reference` to "
                "`build_align_diagnostics_report` to enable."
            ),
        )
    genes = annotation.features_of_type("gene")
    if genes.num_rows == 0:
        return ReportSection(
            title="Multi-gene alignment counts",
            body="Annotation has no gene-type features.",
        )

    primary = alignments.filter(
        pc.and_(
            pc.invert(alignments.column("is_secondary")),
            pc.invert(alignments.column("is_supplementary")),
        )
    ).select(["alignment_id", "ref_name", "ref_start", "ref_end", "strand"])
    if primary.num_rows == 0:
        return ReportSection(
            title="Multi-gene alignment counts",
            body="No primary alignments.",
        )

    overlap_counts = _count_gene_overlaps_per_alignment(
        primary, genes, contig_name_to_id
    )
    bins = np.bincount(overlap_counts, minlength=5)
    n_aln = int(overlap_counts.size)
    n_multi = int((overlap_counts >= 2).sum())
    pct_multi = 100.0 * n_multi / n_aln if n_aln else 0.0

    body = (
        f"Of {n_aln:,} primary alignments overlapping ≥1 annotated gene: "
        f"{n_multi:,} ({pct_multi:.2f}%) overlap **≥2 annotated genes** — "
        "the over-fusion smoking gun for compact genomes.\n\n"
        + "| Genes overlapped | Alignments |\n"
        + "| --- | ---: |\n"
        + "\n".join(
            f"| {n} | {int(bins[n]):,} |"
            for n in range(min(len(bins), 6))
        )
        + (
            f"\n| 6+ | {int(bins[6:].sum()):,} |"
            if len(bins) > 6 else ""
        )
    )

    if pct_multi > 2.0:
        flags.append(
            f"multi_gene_alignments: {pct_multi:.2f}% of primary alignments "
            "overlap ≥ 2 annotated genes. On a compact genome (Pichia, "
            "yeast, fungi) this is almost certainly minimap2's -G default "
            "letting the splice DP collapse adjacent genes into one read "
            "with a spurious long intron. Set --organism-profile "
            "compact_eukaryote or reduce --max-intron-length."
        )

    fig_path = _plot_multi_gene_overlap_bars(
        bins, figures_dir / "multi_gene_alignment.svg",
    )
    return ReportSection(
        title="Multi-gene alignment counts",
        body=body,
        figures=[fig_path],
        flags=flags,
    )


# ──────────────────────────────────────────────────────────────────────
# Per-alignment gene-overlap counter (vectorised per contig)
# ──────────────────────────────────────────────────────────────────────


def _count_gene_overlaps_per_alignment(
    primary: pa.Table,
    genes: pa.Table,
    contig_name_to_id: dict[str, int],
) -> np.ndarray:
    """Per-alignment count of annotated genes whose interval overlaps
    the alignment's ``[ref_start, ref_end)`` on the same contig.

    Diagnostic-time helper: per-contig sort + two-sided ``searchsorted``
    (upper bound ``gene_start < a_end``; lower bound
    ``ends_prefix_max > a_start``) — same pattern as
    :func:`constellation.sequencing.quant.derived_annotation._expand_block_exon_overlaps_batch`,
    diagnostics-scaled.

    Strand is NOT considered — multi-gene overlaps on opposite strands
    are still counted (a real fusion-or-readthrough event would land on
    one strand; counting opposite-strand neighbours surfaces dense
    bidirectional gene packing as a separate signal).
    """
    g_contig_ids = np.array(genes.column("contig_id").to_pylist(), dtype=np.int64)
    g_starts = np.array(genes.column("start").to_pylist(), dtype=np.int64)
    g_ends = np.array(genes.column("end").to_pylist(), dtype=np.int64)

    per_contig: dict[int, dict[str, np.ndarray]] = {}
    for cid in np.unique(g_contig_ids):
        mask = g_contig_ids == cid
        starts = g_starts[mask]
        ends = g_ends[mask]
        order = np.argsort(starts, kind="stable")
        sorted_starts = starts[order]
        sorted_ends = ends[order]
        per_contig[int(cid)] = {
            "starts": sorted_starts,
            "ends": sorted_ends,
            "ends_prefix_max": np.maximum.accumulate(sorted_ends),
        }

    a_contigs = primary.column("ref_name").to_pylist()
    a_starts = primary.column("ref_start").to_numpy(zero_copy_only=False).astype(np.int64)
    a_ends = primary.column("ref_end").to_numpy(zero_copy_only=False).astype(np.int64)

    counts = np.zeros(primary.num_rows, dtype=np.int64)
    for i, ref_name in enumerate(a_contigs):
        cid = contig_name_to_id.get(str(ref_name))
        if cid is None:
            continue
        bucket = per_contig.get(int(cid))
        if bucket is None:
            continue
        # Upper bound: gene_start < a_end (positions of genes whose
        # start is still strictly before the alignment's end)
        hi = int(np.searchsorted(bucket["starts"], int(a_ends[i]), side="left"))
        if hi == 0:
            continue
        # Lower bound via prefix-max of ends: drop genes that have
        # already ended at-or-before the alignment's start.
        lo = int(
            np.searchsorted(bucket["ends_prefix_max"], int(a_starts[i]), side="right")
        )
        lo = min(lo, hi)
        if lo >= hi:
            continue
        cand_ends = bucket["ends"][lo:hi]
        # Final half-open overlap check on the (possibly small) candidate slice
        overlapping = int((cand_ends > int(a_starts[i])).sum())
        counts[i] = overlapping
    return counts


# ──────────────────────────────────────────────────────────────────────
# Matplotlib plot helpers
# ──────────────────────────────────────────────────────────────────────


_MOTIF_COLORS: dict[str, str] = {
    "GT-AG": "#2C7BB6",
    "GC-AG": "#FDAE61",
    "AT-AC": "#7B3294",
    "other": "#D7191C",
}


def _plot_intron_length_histogram(
    lengths: np.ndarray,
    motifs: list[str | None],
    out_path: Path,
    *,
    organism_profile: str | None = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(
        np.log10(max(int(lengths.min()), 1)),
        np.log10(max(int(lengths.max()), 10)),
        num=50,
    )
    # Stack by motif for a coloured histogram
    canonical = ("GT-AG", "GC-AG", "AT-AC")
    motif_lengths = {m: [] for m in canonical}
    motif_lengths["other"] = []
    for length, m in zip(lengths, motifs, strict=True):
        bucket = str(m) if m in canonical else "other"
        motif_lengths[bucket].append(int(length))

    bottom = np.zeros(len(bins) - 1, dtype=np.float64)
    for m in (*canonical, "other"):
        vals = np.array(motif_lengths[m], dtype=np.int64)
        if vals.size == 0:
            continue
        counts, _ = np.histogram(vals, bins=bins)
        ax.bar(
            bins[:-1], counts, width=np.diff(bins),
            bottom=bottom, color=_MOTIF_COLORS[m], align="edge",
            label=f"{m} (n={vals.size})", edgecolor="none",
        )
        bottom += counts

    cap = _PROFILE_MAX_BIOLOGICAL_INTRON_BP.get(organism_profile or "", None)
    if cap is not None:
        ax.axvline(
            cap, color="k", linestyle="--", linewidth=1.0,
            label=f"profile cap ({cap:,} bp)",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Intron length (bp, log scale)")
    ax.set_ylabel("# intron-cluster seeds")
    ax.set_title("Intron length distribution by splice motif")
    ax.legend(loc="best")
    return save_svg_figure(fig, out_path)


def _plot_motif_composition(
    motif_clusters: dict[str, int],
    motif_reads: dict[str, int],
    out_path: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_cl, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
    motifs = list(motif_clusters.keys())
    cluster_vals = [motif_clusters[m] for m in motifs]
    read_vals = [motif_reads[m] for m in motifs]
    colors = [_MOTIF_COLORS[m] for m in motifs]

    ax_cl.bar(motifs, cluster_vals, color=colors, edgecolor="none")
    ax_cl.set_title("Distinct intron-cluster seeds")
    ax_cl.set_ylabel("count")
    ax_cl.tick_params(axis="x", rotation=20)

    ax_r.bar(motifs, read_vals, color=colors, edgecolor="none")
    ax_r.set_title("Total supporting reads")
    ax_r.set_ylabel("count")
    ax_r.tick_params(axis="x", rotation=20)

    fig.suptitle("Splice motif composition")
    fig.tight_layout()
    return save_svg_figure(fig, out_path)


def _plot_annotated_fraction_by_support(
    counts: np.ndarray, ann: np.ndarray, out_path: Path
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Bin by log-supporting-read-count, plot fraction annotated per bin.
    if counts.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return save_svg_figure(fig, out_path)
    bins = np.unique(np.round(np.geomspace(1, max(int(counts.max()), 2), num=10)))
    bins = np.concatenate([bins, [bins[-1] + 1]]).astype(np.int64)
    centers, fracs, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (counts >= lo) & (counts < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        centers.append((lo + hi - 1) / 2)
        fracs.append(float(ann[mask].sum() / n))
        ns.append(n)

    fig, ax = plt.subplots(figsize=(8, 5))
    if centers:
        ax.plot(centers, fracs, marker="o", linewidth=1.5, color="#2C7BB6")
        for x, y, n in zip(centers, fracs, ns, strict=True):
            ax.annotate(
                f"n={n:,}",
                (x, y), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=7,
            )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Supporting reads per intron cluster (log)")
    ax.set_ylabel("Fraction matching annotation")
    ax.set_title("Annotated junction agreement vs supporting reads")
    return save_svg_figure(fig, out_path)


def _plot_n_blocks_histogram(counts: np.ndarray, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    max_x = min(int(counts.max()), 60)
    bins = np.arange(1, max_x + 2)
    ax.hist(np.clip(counts, 1, max_x), bins=bins, color="#2C7BB6", edgecolor="none")
    ax.set_xlabel("Blocks per primary alignment (≈ exons)")
    ax.set_ylabel("# alignments")
    ax.set_title("Alignment complexity (n_blocks)")
    return save_svg_figure(fig, out_path)


def _plot_mapq_histogram(mapq: np.ndarray, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mapq, bins=np.arange(0, 62), color="#2C7BB6", edgecolor="none")
    ax.set_xlabel("MAPQ")
    ax.set_ylabel("# primary alignments")
    ax.set_title("MAPQ distribution")
    return save_svg_figure(fig, out_path)


def _plot_multi_gene_overlap_bars(bins: np.ndarray, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [str(i) for i in range(min(len(bins), 6))]
    values = list(bins[: len(labels)].astype(int))
    if len(bins) > 6:
        labels.append("6+")
        values.append(int(bins[6:].sum()))
    colors = ["#2C7BB6"] + ["#D7191C"] * (len(labels) - 1)
    ax.bar(labels, values, color=colors, edgecolor="none")
    ax.set_xlabel("# annotated genes overlapped by alignment")
    ax.set_ylabel("# alignments")
    ax.set_title("Multi-gene alignment counts (fusion-candidate diagnostic)")
    return save_svg_figure(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────


def build_align_diagnostics_report(
    align_dir: Path,
    *,
    reference: Any = None,
    annotation: Any = None,
    organism_profile: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Generate ``<align-dir>/diagnostics/report.md`` + figures.

    Reads the manifest at ``align-dir/manifest.json`` for parameter
    snapshot + (when ``organism_profile`` is None) the profile used by
    the producing run. Loads ``introns.parquet``, ``alignments/``, and
    ``alignment_blocks/`` from the standard align-output paths. When
    ``annotation`` is None and a ``GenomeReference`` is also None, the
    annotation-dependent sections (multi-gene overlap, annotated
    junction agreement) are skipped with a stub note rather than
    failing.

    The orchestrator catches per-section exceptions so a single broken
    metric doesn't poison the whole report — broken sections render
    as a stub describing the failure.

    Parameters
    ----------
    align_dir
        Output dir from ``constellation transcriptome align``.
    reference, annotation
        Optional pre-loaded references. When omitted, the orchestrator
        attempts to load them from the manifest's ``reference_path``.
    organism_profile
        Override the profile recorded in the manifest. Affects only
        the flag thresholds (e.g. the biological-plausibility intron
        length cap).
    output_dir
        Defaults to ``<align-dir>/diagnostics``.

    Returns the path to the written ``report.md``.
    """
    from constellation.sequencing.transcriptome.manifest import read_manifest_dir

    align_dir = Path(align_dir)
    if output_dir is None:
        output_dir = align_dir / "diagnostics"
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest_dir(align_dir)
    if manifest.kind != "align":
        raise ValueError(
            f"{align_dir} has manifest kind={manifest.kind!r}; expected 'align'"
        )

    if organism_profile is None:
        organism_profile = manifest.parameters.get("organism_profile")

    # ── Optional reference / annotation auto-load from manifest ─────
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
                # Best-effort — skip annotation-dependent sections
                # rather than break the whole report.
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

    # ── Load parquet inputs ──────────────────────────────────────────
    introns_path = align_dir / "introns.parquet"
    introns = (
        pq.read_table(introns_path) if introns_path.is_file() else None
    )
    alignments_dir = align_dir / "alignments"
    alignments = (
        pa_dataset.dataset(alignments_dir).to_table()
        if alignments_dir.is_dir() else None
    )
    blocks_dir = align_dir / "alignment_blocks"
    blocks = (
        pa_dataset.dataset(blocks_dir).to_table()
        if blocks_dir.is_dir() else None
    )

    # ── Sections ────────────────────────────────────────────────────
    sections: list[ReportSection] = []

    def _safe(title: str, fn, *args, **kwargs) -> ReportSection:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return ReportSection(
                title=title,
                body=(
                    f"_Section failed to render with `{type(exc).__name__}: "
                    f"{exc}`. The other diagnostics still ran; this is a "
                    "non-fatal report-generation error._"
                ),
            )

    sections.append(_section_run_parameters(manifest))

    if introns is not None:
        sections.append(
            _safe(
                "Intron length distribution",
                section_intron_length_distribution,
                introns,
                figures_dir=figures_dir,
                organism_profile=organism_profile,
            )
        )
        sections.append(
            _safe(
                "Splice motif composition",
                section_motif_composition,
                introns,
                figures_dir=figures_dir,
            )
        )
        sections.append(
            _safe(
                "Annotated junction agreement",
                section_annotated_junction_agreement,
                introns,
                figures_dir=figures_dir,
            )
        )
    else:
        sections.append(
            ReportSection(
                title="Intron diagnostics",
                body="`introns.parquet` not found in the align dir.",
            )
        )

    if blocks is not None and alignments is not None:
        sections.append(
            _safe(
                "Alignment complexity (blocks per read)",
                section_alignment_complexity,
                blocks, alignments,
                figures_dir=figures_dir,
            )
        )

    if alignments is not None:
        sections.append(
            _safe(
                "MAPQ distribution",
                section_mapq_distribution,
                alignments,
                figures_dir=figures_dir,
            )
        )
        sections.append(
            _safe(
                "Multi-gene alignment counts",
                section_multi_gene_alignments,
                alignments, annotation, contig_name_to_id,
                figures_dir=figures_dir,
            )
        )

    intro = (
        f"Diagnostic report for `transcriptome align` output at "
        f"`{align_dir}`. Reference: "
        f"`{manifest.reference_handle or manifest.reference_path}`."
    )
    return render_report(
        title="Transcriptome align — diagnostics",
        intro=intro,
        sections=sections,
        output_path=output_dir / "report.md",
    )


def _section_run_parameters(manifest: Any) -> ReportSection:
    """Emit the manifest's parameter snapshot as a markdown table."""
    params = manifest.parameters
    rows = []
    for k in sorted(params.keys()):
        v = params[k]
        rows.append(f"| `{k}` | `{v}` |")
    resolved = getattr(manifest, "minimap2_resolved_args", None)
    body = (
        f"Reference: `{manifest.reference_handle or manifest.reference_path}` "
        f"(`{manifest.assembly_accession or 'no accession'}`).\n\n"
        "| Parameter | Value |\n"
        "| --- | --- |\n"
        + "\n".join(rows)
    )
    if resolved:
        body += (
            "\n\n**Resolved minimap2 command line:**\n\n"
            f"`minimap2 {' '.join(resolved)} <index> -`\n"
        )
    return ReportSection(title="Run parameters", body=body)


__all__ = [
    "build_align_diagnostics_report",
    "section_alignment_complexity",
    "section_annotated_junction_agreement",
    "section_intron_length_distribution",
    "section_mapq_distribution",
    "section_motif_composition",
    "section_multi_gene_alignments",
]
