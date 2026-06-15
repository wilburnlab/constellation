"""Diagnostic report for the de novo cluster stage.

Pure functions over the stage's parquet outputs — cluster size + read
yield, the context-aware variant-class composition (what was collapsed
vs retained), the 5'/3' length variation that was folded in, and
consensus quality. Each metric is a ``section_*`` function returning a
:class:`ReportSection`; the orchestrator wraps each in ``_safe`` so one
broken metric never breaks the whole report, mirroring the genome-guided
cluster diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from constellation.sequencing.transcriptome._render import (
    ReportSection,
    render_report,
    save_svg_figure,
)


def _safe(title: str, fn, *args, **kwargs) -> ReportSection:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return ReportSection(
            title=title,
            body=f"_metric unavailable: {type(exc).__name__}: {exc}_",
        )


def section_parameters(cluster_dir: Path) -> ReportSection:
    manifest = json.loads((cluster_dir / "manifest.json").read_text())
    params = manifest.get("parameters", {})
    stages = manifest.get("stages", {})
    lines = ["| key | value |", "| --- | --- |"]
    for k in sorted(params):
        lines.append(f"| {k} | {params[k]} |")
    for k in sorted(stages):
        lines.append(f"| {k} | {stages[k]} |")
    return ReportSection(title="Run parameters", body="\n".join(lines))


def section_cluster_sizes(cluster_dir: Path, figures_dir: Path) -> ReportSection:
    clusters = pq.read_table(cluster_dir / "clusters.parquet")
    n_reads = clusters.column("n_reads").to_numpy()
    n = clusters.num_rows
    flags: list[str] = []
    if n == 0:
        return ReportSection(title="Cluster sizes", body="_no clusters_")
    singletons = int((n_reads == 1).sum())
    frac_singleton = singletons / n
    top = int(n_reads.max())
    body = (
        f"{n} clusters from {int(n_reads.sum())} clustered reads. "
        f"Singletons: {singletons} ({100 * frac_singleton:.1f}%). "
        f"Largest cluster: {top} reads. "
        f"Median size: {int(np.median(n_reads))}."
    )
    if frac_singleton > 0.8:
        flags.append(
            f"⚠ {100 * frac_singleton:.0f}% of clusters are singletons — "
            "consider lowering --identity (read-to-read gate) or raising "
            "--min-cluster-size."
        )
    figures: list[Path] = []
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(np.clip(n_reads, 1, None), bins=40, log=True, color="#a277ff")
        ax.set_xlabel("reads per cluster")
        ax.set_ylabel("clusters (log)")
        ax.set_title("Cluster size distribution")
        figures.append(save_svg_figure(fig, figures_dir / "cluster_sizes.svg"))
    except Exception:  # noqa: BLE001
        pass
    return ReportSection(title="Cluster sizes", body=body, figures=figures, flags=flags)


def section_variant_composition(cluster_dir: Path, figures_dir: Path) -> ReportSection:
    variants = pq.read_table(cluster_dir / "cluster_variants.parquet")
    if variants.num_rows == 0:
        return ReportSection(
            title="Variant composition",
            body="_no within-cluster variants tested (all clusters single-sequence)_",
        )
    vclass = np.array(variants.column("variant_class").to_pylist())
    call = np.array(variants.column("call").to_pylist())
    classes = ["substitution", "homopolymer_indel", "non_hp_indel"]
    calls = ["real", "ambiguous", "collapsed_error"]
    lines = [
        "| class \\ call | " + " | ".join(calls) + " | total |",
        "| --- | " + " | ".join(["---"] * (len(calls) + 1)) + " |",
    ]
    for c in classes:
        row = [int(((vclass == c) & (call == k)).sum()) for k in calls]
        lines.append(f"| {c} | " + " | ".join(str(x) for x in row) + f" | {sum(row)} |")
    n_real = int((call == "real").sum())
    n_collapsed = int((call == "collapsed_error").sum())
    body = (
        f"{variants.num_rows} positions tested across clusters: "
        f"{n_real} called real variants, {n_collapsed} collapsed as error, "
        f"{int((call == 'ambiguous').sum())} ambiguous (depth-limited).\n\n"
        + "\n".join(lines)
        + "\n\n_Homopolymer indels are held to a higher (run-length-scaled) "
        "error rate than isolated substitutions — the non-uniform Dorado "
        "error model._"
    )
    figures: list[Path] = []
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.arange(len(classes))
        for i, k in enumerate(calls):
            vals = [int(((vclass == c) & (call == k)).sum()) for c in classes]
            ax.bar(x + i * 0.25, vals, width=0.25, label=k)
        ax.set_xticks(x + 0.25)
        ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=8)
        ax.set_ylabel("positions")
        ax.set_title("Variant class × call")
        ax.legend(fontsize=7)
        figures.append(save_svg_figure(fig, figures_dir / "variant_composition.svg"))
    except Exception:  # noqa: BLE001
        pass
    return ReportSection(title="Variant composition", body=body, figures=figures)


def section_length_variation(cluster_dir: Path, figures_dir: Path) -> ReportSection:
    mem = pq.read_table(cluster_dir / "cluster_membership.parquet")
    if mem.num_rows == 0:
        return ReportSection(title="Length variation collapsed", body="_no members_")
    d5 = mem.column("drift_5p_bp").to_numpy(zero_copy_only=False)
    d3 = mem.column("drift_3p_bp").to_numpy(zero_copy_only=False)
    d5 = d5[~np.isnan(d5.astype(float))] if d5.dtype.kind == "f" else d5
    body = (
        "Signed 5'/3' length offset of members vs their cluster consensus "
        "(what was collapsed). "
        f"5': median {int(np.median(d5))} bp, |max| {int(np.abs(d5).max())} bp. "
        f"3': median {int(np.median(d3))} bp, |max| {int(np.abs(d3).max())} bp."
    )
    figures: list[Path] = []
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(d5, bins=40, alpha=0.6, label="5'", color="#a277ff")
        ax.hist(d3, bins=40, alpha=0.6, label="3'", color="#61ffca")
        ax.set_xlabel("offset vs consensus (bp)")
        ax.set_ylabel("reads")
        ax.set_title("Collapsed 5'/3' length variation")
        ax.legend(fontsize=8)
        figures.append(save_svg_figure(fig, figures_dir / "length_variation.svg"))
    except Exception:  # noqa: BLE001
        pass
    return ReportSection(title="Length variation collapsed", body=body, figures=figures)


def section_haplotypes(cluster_dir: Path) -> ReportSection:
    hap_path = cluster_dir / "cluster_haplotypes.parquet"
    if not hap_path.exists():
        return ReportSection(title="Haplotypes", body="_no haplotype table_")
    haps = pq.read_table(hap_path)
    if haps.num_rows == 0:
        return ReportSection(
            title="Haplotypes",
            body="_no multi-variant clusters — nothing to phase_",
        )
    cid = np.array(haps.column("cluster_id").to_pylist())
    per_cluster = np.bincount(cid - cid.min()) if cid.size else np.array([])
    multi = int((per_cluster > 1).sum())
    variants = pq.read_table(cluster_dir / "cluster_variants.parquet")
    r2 = variants.column("max_linkage_r2").to_numpy(zero_copy_only=False)
    r2 = r2[~np.isnan(r2.astype(float))] if r2.dtype.kind == "f" else r2
    phased = int((r2 >= 0.8).sum()) if r2.size else 0
    body = (
        f"{len(per_cluster)} clusters carry called variants; {multi} have ≥2 "
        f"distinct haplotypes (candidate allele/paralog mixtures). "
        f"{phased} variant positions are tightly phased (max r² ≥ 0.8) to "
        "another — co-segregating variants, the signature of real haplotype "
        "structure rather than scattered error.\n\n_Per-cluster haplotype "
        "maps render under `detail/` when `--emit-cluster-detail` is set._"
    )
    return ReportSection(title="Haplotypes", body=body)


def section_consensus_quality(cluster_dir: Path) -> ReportSection:
    mem = pq.read_table(cluster_dir / "cluster_membership.parquet")
    clusters = pq.read_table(cluster_dir / "clusters.parquet")
    if mem.num_rows == 0:
        return ReportSection(title="Consensus quality", body="_no members_")
    mr = mem.column("match_rate").to_numpy(zero_copy_only=False)
    mr = mr[~np.isnan(mr.astype(float))]
    n_protein = int(
        sum(1 for p in clusters.column("predicted_protein").to_pylist() if p)
    )
    body = (
        f"Member→consensus identity: median {100 * np.median(mr):.2f}%, "
        f"10th pct {100 * np.percentile(mr, 10):.2f}%.\n\n"
        f"Clusters with a predicted ORF protein: {n_protein} / "
        f"{clusters.num_rows}."
    )
    return ReportSection(title="Consensus quality", body=body)


# ── per-cluster detail: haplotype map (the MSA-free phasing visual) ───

# Minor-allele colours (consensus rendered muted; uncovered hatched white).
_ALLELE_RGB = {
    "A": (0.30, 0.69, 0.31),  # green
    "C": (0.13, 0.59, 0.95),  # blue
    "G": (1.00, 0.60, 0.00),  # orange
    "T": (0.90, 0.22, 0.21),  # red
    "-": (0.15, 0.15, 0.15),  # deletion = near-black
}
_CONSENSUS_RGB = (0.86, 0.86, 0.88)
_UNCOVERED_RGB = (1.0, 1.0, 1.0)


def render_haplotype_map(
    cluster_id: int,
    hap_rows: list[dict],
    var_rows: list[dict],
    out_path: Path,
    *,
    max_rows: int = 30,
) -> Path | None:
    """Render a cluster's haplotype heatmap: rows = distinct haplotypes
    (abundance-sorted), columns = variant positions, cells coloured by
    allele (consensus muted, minor saturated, uncovered hatched)."""
    if not hap_rows or not var_rows:
        return None
    import numpy as np
    import matplotlib.pyplot as plt

    var_rows = sorted(var_rows, key=lambda r: r["consensus_pos"])
    positions = [r["consensus_pos"] for r in var_rows]
    consensus = [r["consensus_allele"] for r in var_rows]
    calls = [r["call"] for r in var_rows]
    vclasses = [r["variant_class"] for r in var_rows]
    pos_index = {p: i for i, p in enumerate(positions)}
    V = len(positions)

    haps = sorted(hap_rows, key=lambda r: -r["abundance"])
    shown = haps[:max_rows]
    other = haps[max_rows:]
    R = len(shown) + (1 if other else 0)

    img = np.ones((R, V, 3))
    labels = []
    # The haplotype's allele_string is indexed by the cluster's variant
    # positions in ascending consensus order (same order as `positions`).
    for ri, h in enumerate(shown):
        astr = h["allele_string"]
        hap_positions = h["variant_positions"]
        for col_in_hap, p in enumerate(hap_positions):
            if p not in pos_index or col_in_hap >= len(astr):
                continue
            c = pos_index[p]
            ch = astr[col_in_hap]
            if ch == ".":
                img[ri, c] = _UNCOVERED_RGB
            elif ch == consensus[c]:
                img[ri, c] = _CONSENSUS_RGB
            else:
                img[ri, c] = _ALLELE_RGB.get(ch, (0.5, 0.5, 0.5))
        labels.append(f"{h['abundance']} rd / {h['n_unique_sequences']} uq")
    if other:
        img[-1] = (0.95, 0.95, 0.95)
        labels.append(
            f"+{len(other)} more haplotypes ({sum(o['abundance'] for o in other)} rd)"
        )

    fig, ax = plt.subplots(figsize=(max(4, V * 0.35 + 2), max(2.5, R * 0.22 + 1.2)))
    ax.imshow(img, aspect="auto", interpolation="none")
    ax.set_yticks(range(R))
    ax.set_yticklabels(labels, fontsize=7)
    call_mark = {"real": "★", "ambiguous": "?", "collapsed_error": "·"}
    cls_abbr = {"substitution": "sub", "homopolymer_indel": "hp", "non_hp_indel": "ind"}
    ax.set_xticks(range(V))
    ax.set_xticklabels(
        [
            f"{positions[i]}\n{call_mark.get(calls[i], '')}{cls_abbr.get(vclasses[i], '')}"
            for i in range(V)
        ],
        fontsize=6,
    )
    ax.set_title(
        f"cluster {cluster_id} — {len(haps)} haplotypes × {V} variant positions\n"
        "(consensus = grey; minor alleles coloured; ★ real / ? ambiguous / · error)",
        fontsize=8,
    )
    ax.set_xlabel("consensus position", fontsize=8)
    return save_svg_figure(fig, out_path)


def emit_cluster_details(cluster_dir: Path, *, top_n: int = 50) -> int:
    """Write per-cluster haplotype-map SVGs + variant TSVs for the top-N
    clusters by read count. Returns the number of clusters detailed."""
    clusters = pq.read_table(cluster_dir / "clusters.parquet")
    if clusters.num_rows == 0:
        return 0
    variants = pq.read_table(cluster_dir / "cluster_variants.parquet")
    haplotypes = pq.read_table(cluster_dir / "cluster_haplotypes.parquet")
    detail_dir = cluster_dir / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)

    n_reads = np.array(clusters.column("n_reads").to_pylist())
    cids = np.array(clusters.column("cluster_id").to_pylist())
    order = np.argsort(-n_reads)[:top_n]
    top_cids = set(int(cids[i]) for i in order)

    var_by_cluster: dict[int, list[dict]] = {}
    for r in variants.to_pylist():
        if r["cluster_id"] in top_cids:
            var_by_cluster.setdefault(r["cluster_id"], []).append(r)
    hap_by_cluster: dict[int, list[dict]] = {}
    for r in haplotypes.to_pylist():
        if r["cluster_id"] in top_cids:
            hap_by_cluster.setdefault(r["cluster_id"], []).append(r)

    n = 0
    for cid in sorted(top_cids):
        vr = var_by_cluster.get(cid, [])
        hr = hap_by_cluster.get(cid, [])
        if not vr:
            continue
        cdir = detail_dir / f"cluster_{cid}"
        cdir.mkdir(parents=True, exist_ok=True)
        # variant TSV
        cols = list(vr[0].keys())
        with (cdir / "variants.tsv").open("w", encoding="utf-8") as fh:
            fh.write("\t".join(cols) + "\n")
            for row in sorted(vr, key=lambda r: r["consensus_pos"]):
                fh.write("\t".join(str(row[c]) for c in cols) + "\n")
        render_haplotype_map(cid, hr, vr, cdir / "haplotype_map.svg")
        n += 1
    return n


def build_denovo_diagnostics_report(
    cluster_dir: Path, *, output_dir: Path | None = None
) -> Path:
    """Emit ``<cluster_dir>/diagnostics/report.md`` + figures."""
    cluster_dir = Path(cluster_dir)
    diag_dir = (output_dir or cluster_dir) / "diagnostics"
    figures_dir = diag_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        _safe("Run parameters", section_parameters, cluster_dir),
        _safe("Cluster sizes", section_cluster_sizes, cluster_dir, figures_dir),
        _safe(
            "Variant composition",
            section_variant_composition,
            cluster_dir,
            figures_dir,
        ),
        _safe(
            "Length variation collapsed",
            section_length_variation,
            cluster_dir,
            figures_dir,
        ),
        _safe("Haplotypes", section_haplotypes, cluster_dir),
        _safe("Consensus quality", section_consensus_quality, cluster_dir),
    ]
    return render_report(
        title="De novo cluster diagnostics",
        intro=(
            "First-round reference-free transcript assembly — cluster yield, "
            "the context-aware variant calls (what was collapsed vs retained "
            "as real polymorphism), and the length variation folded in."
        ),
        sections=sections,
        output_path=diag_dir / "report.md",
    )


__all__ = ["build_denovo_diagnostics_report"]
