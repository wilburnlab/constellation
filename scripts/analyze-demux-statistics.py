"""Compute calibration distributions from a completed demux output.

Usage:
    python3 scripts/analyze-demux-statistics.py <demux_output_dir> [--out-dir DIR]

Calibration use case
--------------------
The future ProbabilisticScorer needs prior distributions over per-segment
scores: 5'/3' adapter edit distance, polyA length, barcode edit distance,
and barcode best-vs-second-best gap (``score_delta``). This script slices
the demux output by ReadStatus — Complete vs everything else — and emits
the empirical distributions.

The Complete subset gives the cleanest estimate of expected error rates:
these are reads that have a 5' adapter, 3' adapter, polyA tail, and
barcode all confidently identified, so their score distributions are
dominated by sequencing error rather than artifact.

Outputs (under <out-dir>, default = <demux_output_dir>/stats/):
    demux_statistics.parquet  — per-row table with one row per
        (read_id, segment_kind) pair; columns include status,
        is_fragment, segment_kind, edit_distance / length / score_delta.
    demux_statistics_summary.json  — per-status histograms + percentiles.
    demux_statistics.pdf  — 2x2 matplotlib distribution plot
        (5' edit dist / 3' edit dist / polyA length / barcode score_delta);
        emitted only if matplotlib import succeeds.

Run from project root:
    python3 scripts/analyze-demux-statistics.py /path/to/demux_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# Histogram bucket definitions — fixed boundaries so summaries are
# directly comparable across runs / datasets.
_EDIT_DIST_BUCKETS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # +"6+" overflow
_POLYA_LEN_BUCKETS: tuple[tuple[int, int | None], ...] = (
    (0, 14),    # below scorer min (15)
    (15, 19),
    (20, 24),
    (25, 29),
    (30, 34),
    (35, 40),
    (41, None),  # above scorer max (40); should be empty under hard scorer
)
_DELTA_BUCKETS: tuple[int, ...] = (0, 1, 2, 3)  # +"4+" overflow; <0 separate


# ──────────────────────────────────────────────────────────────────────
# Per-(read, segment) table builder
# ──────────────────────────────────────────────────────────────────────


def _build_stats_table(demux: pa.Table, segments: pa.Table) -> pa.Table:
    """Join demux's status / is_fragment onto each segment row.

    Output columns:
        read_id, status, is_fragment, segment_kind,
        edit_distance (= score), length (end - start), score_delta.

    Filters segments to the four kinds we care about for calibration:
    adapter_5p, adapter_3p, polyA, barcode. Transcript / umi rows are
    dropped (no score / length is meaningful as a calibration target).
    """
    keep_kinds = ("adapter_5p", "adapter_3p", "polyA", "barcode")
    seg_filtered = segments.filter(
        pc.is_in(segments.column("segment_kind"), value_set=pa.array(keep_kinds))
    )
    # Compute length column (end - start) — for polyA this is the
    # natural calibration axis; for adapters it's the matched window.
    length_col = pc.subtract(
        seg_filtered.column("end"), seg_filtered.column("start")
    )
    seg_with_len = seg_filtered.append_column("length", length_col)

    # Slim demux to (read_id, status, is_fragment) for the join.
    demux_slim = demux.select(["read_id", "status", "is_fragment"])
    joined = seg_with_len.join(demux_slim, keys="read_id", join_type="inner")

    # Project to the columns the stats consumer needs, in stable order.
    return joined.select(
        [
            "read_id",
            "status",
            "is_fragment",
            "segment_kind",
            "score",
            "score_delta",
            "length",
        ]
    ).rename_columns(
        ["read_id", "status", "is_fragment", "segment_kind", "edit_distance",
         "score_delta", "length"]
    )


# ──────────────────────────────────────────────────────────────────────
# Summary (histograms + percentiles per status)
# ──────────────────────────────────────────────────────────────────────


def _percentiles(values: list[int | float]) -> dict[str, float | None]:
    """Return p25/p50/p75/p90/p99 for a list of numeric values, or
    Nones when the list is empty."""
    if not values:
        return {p: None for p in ("p25", "p50", "p75", "p90", "p99")}
    sorted_vals = sorted(values)

    def _at(frac: float) -> float:
        idx = max(0, min(len(sorted_vals) - 1, int(round(frac * (len(sorted_vals) - 1)))))
        return float(sorted_vals[idx])

    return {
        "p25": _at(0.25),
        "p50": _at(0.50),
        "p75": _at(0.75),
        "p90": _at(0.90),
        "p99": _at(0.99),
    }


def _edit_distance_histogram(values: list[int]) -> dict[str, int]:
    out = {str(b): 0 for b in _EDIT_DIST_BUCKETS}
    out["6+"] = 0
    for v in values:
        if v < 0:
            continue  # skip transcript-style sentinel "-1" rows
        if v in _EDIT_DIST_BUCKETS:
            out[str(v)] += 1
        else:
            out["6+"] += 1
    return out


def _polya_length_histogram(values: list[int]) -> dict[str, int]:
    out = {f"{lo}-{hi if hi is not None else '∞'}": 0 for lo, hi in _POLYA_LEN_BUCKETS}
    for v in values:
        for lo, hi in _POLYA_LEN_BUCKETS:
            if hi is None:
                if v >= lo:
                    out[f"{lo}-∞"] += 1
                    break
            else:
                if lo <= v <= hi:
                    out[f"{lo}-{hi}"] += 1
                    break
    return out


def _delta_histogram(values: list[int]) -> dict[str, int]:
    out: dict[str, int] = {"<0": 0}
    for b in _DELTA_BUCKETS:
        out[str(b)] = 0
    out["4+"] = 0
    for v in values:
        if v < 0:
            out["<0"] += 1
        elif v in _DELTA_BUCKETS:
            out[str(v)] += 1
        else:
            out["4+"] += 1
    return out


def _build_summary(stats: pa.Table) -> dict:
    """Per-status histograms + percentiles for the four target axes.

    Output shape:
        {
          "<status>": {
             "n_reads": int,
             "adapter_5p": {"hist": {...}, "percentiles": {...}, "n": int},
             "adapter_3p": {...},
             "polyA":      {"hist": {...}, "percentiles": {...}, "n": int},
             "barcode":    {"edit_hist": {...}, "edit_percentiles": {...},
                            "delta_hist": {...}, "delta_percentiles": {...},
                            "n": int},
          }
        }
    """
    statuses = sorted({s for s in stats.column("status").to_pylist()})
    out: dict[str, dict] = {}
    rows = stats.to_pylist()
    for status in statuses:
        bucket = [r for r in rows if r["status"] == status]
        n_reads = len({r["read_id"] for r in bucket})
        per_kind: dict[str, dict] = {}
        for kind in ("adapter_5p", "adapter_3p"):
            vals = [
                r["edit_distance"] for r in bucket
                if r["segment_kind"] == kind and r["edit_distance"] is not None
                and r["edit_distance"] >= 0
            ]
            per_kind[kind] = {
                "hist": _edit_distance_histogram(vals),
                "percentiles": _percentiles(vals),
                "n": len(vals),
            }
        polya_vals = [
            r["length"] for r in bucket
            if r["segment_kind"] == "polyA" and r["length"] is not None
        ]
        per_kind["polyA"] = {
            "hist": _polya_length_histogram(polya_vals),
            "percentiles": _percentiles(polya_vals),
            "n": len(polya_vals),
        }
        bc_edit = [
            r["edit_distance"] for r in bucket
            if r["segment_kind"] == "barcode" and r["edit_distance"] is not None
            and r["edit_distance"] >= 0
        ]
        bc_delta = [
            r["score_delta"] for r in bucket
            if r["segment_kind"] == "barcode" and r["score_delta"] is not None
        ]
        per_kind["barcode"] = {
            "edit_hist": _edit_distance_histogram(bc_edit),
            "edit_percentiles": _percentiles(bc_edit),
            "delta_hist": _delta_histogram(bc_delta),
            "delta_percentiles": _percentiles(bc_delta),
            "n": len(bc_edit),
        }
        out[status] = {"n_reads": n_reads, **per_kind}
    return out


# ──────────────────────────────────────────────────────────────────────
# Optional matplotlib PDF
# ──────────────────────────────────────────────────────────────────────


def _emit_pdf(stats: pa.Table, pdf_path: Path) -> None:
    """Render the four-panel distribution plot to PDF.

    Best-effort: import matplotlib lazily so missing-dep environments
    skip the plot without erroring.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    rows = stats.to_pylist()
    statuses = sorted({r["status"] for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()
    panels = [
        ("adapter_5p", "edit_distance", "5' adapter edit distance", axes_flat[0]),
        ("adapter_3p", "edit_distance", "3' adapter edit distance", axes_flat[1]),
        ("polyA", "length", "polyA length", axes_flat[2]),
        ("barcode", "score_delta", "barcode score_delta (best vs 2nd-best)",
         axes_flat[3]),
    ]
    for kind, value_col, title, ax in panels:
        for status in statuses:
            vals = [
                r[value_col] for r in rows
                if r["segment_kind"] == kind
                and r["status"] == status
                and r[value_col] is not None
                and (value_col != "edit_distance" or r[value_col] >= 0)
            ]
            if not vals:
                continue
            ax.hist(vals, bins=30, alpha=0.5, label=status, histtype="stepfilled")
        ax.set_title(title)
        ax.set_xlabel(value_col)
        ax.set_ylabel("count")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Demux statistics — calibration distributions for ProbabilisticScorer")
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def _read_segments_dataset(seg_dir: Path) -> pa.Table:
    """Read the partitioned parquet dataset under ``read_segments/``."""
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"missing segments directory: {seg_dir}")
    dataset = ds.dataset(str(seg_dir), format="parquet")
    return dataset.to_table()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-status calibration distributions from a "
        "completed `constellation transcriptome demultiplex` run."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="path to a demux output directory (contains read_demux/ "
        "+ read_segments/ shards)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory for stats outputs (default: <output_dir>/stats)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (args.output_dir / "stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    demux = ds.dataset(
        args.output_dir / "read_demux", format="parquet"
    ).to_table()
    segments = _read_segments_dataset(args.output_dir / "read_segments")

    stats_table = _build_stats_table(demux, segments)
    pq.write_table(stats_table, out_dir / "demux_statistics.parquet")

    summary = _build_summary(stats_table)
    (out_dir / "demux_statistics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    pdf_path = out_dir / "demux_statistics.pdf"
    try:
        _emit_pdf(stats_table, pdf_path)
        pdf_status = f"wrote PDF → {pdf_path}"
    except ImportError:
        pdf_status = "matplotlib not available; skipping PDF"

    print(f"wrote stats outputs → {out_dir}")
    print(f"  parquet rows: {stats_table.num_rows}")
    print(f"  statuses summarized: {sorted(summary.keys())}")
    print(f"  {pdf_status}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
