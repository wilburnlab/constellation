"""Compute calibration distributions from a completed demux output.

Usage:
    python3 scripts/analyze-demux-statistics.py <demux_output_dir>
        [--out-dir DIR] [--emit-rows]

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

Architecture (per CLAUDE.md invariant #3)
-----------------------------------------
The full Pichia output has ~30M reads × 3-4 segments per read = 100M+
segment rows. We process this in three multithreaded Arrow steps —
NO Python orchestration loop:

  - **One** ``pa.dataset.to_table(filter=..., columns=...)`` call
    against ``read_segments/`` — Arrow does projection-pushdown
    (reads only the 6 columns we need from parquet) and predicate-
    pushdown (filters out non-interesting segment_kinds at scan
    time). Multithreaded across all parquet shards via Arrow's I/O
    thread pool.
  - **One** Arrow hash-join onto the projected
    ``(read_id, status, is_fragment)`` view of ``read_demux/`` —
    multithreaded inside Arrow.
  - **24** vectorised ``filter()`` calls (6 statuses × 4 kinds), each
    one full-table scan with a SIMD boolean mask. The result of each
    is converted directly to a numpy array via
    ``column.to_numpy(zero_copy_only=False)`` — no Python row
    materialisation.

Histograms come from ``np.bincount``; percentiles from
``np.quantile``. The 24 final filters are one-shot, not per-batch:
total Python orchestration is constant regardless of input size.

Outputs (under <out-dir>, default = <demux_output_dir>/stats/):
    demux_statistics_summary.json  — per-status histograms + percentiles.
    demux_statistics.pdf           — 2x2 distribution plot, optional.
    demux_statistics.parquet       — per-row table (opt-in via --emit-rows).

Run from project root:
    python3 scripts/analyze-demux-statistics.py /path/to/demux_output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
_KEEP_KINDS: tuple[str, ...] = ("adapter_5p", "adapter_3p", "polyA", "barcode")


# ──────────────────────────────────────────────────────────────────────
# Per-bucket numpy arrays
# ──────────────────────────────────────────────────────────────────────


def _extract_buckets(
    joined: pa.Table,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Split the joined table into per-(status, segment_kind) numpy
    arrays for the three metric columns (score, score_delta, length).

    24 vectorised ``filter()`` calls (6 statuses × 4 kinds), each
    one full-table scan with a SIMD boolean mask. Each filtered
    sub-table is converted directly to a numpy array via
    ``to_numpy(zero_copy_only=False)`` — no Python row materialization.
    """
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    statuses_arr = joined.column("status")
    kinds_arr = joined.column("segment_kind")
    unique_statuses = pc.unique(statuses_arr).to_pylist()
    for status in unique_statuses:
        sm = pc.equal(statuses_arr, status)
        for kind in _KEEP_KINDS:
            km = pc.equal(kinds_arr, kind)
            both = pc.and_(sm, km)
            sub = joined.filter(both)
            if sub.num_rows == 0:
                continue
            scores = sub.column("score").to_numpy(zero_copy_only=False)
            deltas = sub.column("score_delta").to_numpy(zero_copy_only=False)
            lengths = sub.column("length").to_numpy(zero_copy_only=False)
            # score_delta is nullable; nulls come back as float64 NaN
            # under pyarrow's default conversion. Convert NaN → -1
            # sentinel so the "<0" bucket of the delta histogram
            # captures unscored rows.
            if np.issubdtype(deltas.dtype, np.floating):
                deltas_np = np.where(np.isnan(deltas), -1, deltas).astype(np.int64)
            else:
                deltas_np = deltas.astype(np.int64)
            out[(status, kind)] = {
                "score": scores.astype(np.int64, copy=False),
                "score_delta": deltas_np,
                "length": lengths.astype(np.int64, copy=False),
            }
    return out


# ──────────────────────────────────────────────────────────────────────
# Histogram + percentile builders (numpy-vectorised)
# ──────────────────────────────────────────────────────────────────────


def _percentiles_np(arr: np.ndarray) -> dict[str, float | None]:
    """p25/p50/p75/p90/p99 via ``np.quantile``; None when empty."""
    if arr.size == 0:
        return {p: None for p in ("p25", "p50", "p75", "p90", "p99")}
    qs = np.quantile(arr, [0.25, 0.50, 0.75, 0.90, 0.99])
    return {
        "p25": float(qs[0]),
        "p50": float(qs[1]),
        "p75": float(qs[2]),
        "p90": float(qs[3]),
        "p99": float(qs[4]),
    }


def _edit_dist_hist_np(arr: np.ndarray) -> dict[str, int]:
    """Edit-distance histogram (0..5 buckets + 6+ overflow). Skips
    sentinel ``-1`` values (transcript / unscored rows)."""
    out = {str(b): 0 for b in _EDIT_DIST_BUCKETS}
    out["6+"] = 0
    if arr.size == 0:
        return out
    valid = arr[arr >= 0]
    if valid.size == 0:
        return out
    # Bin via np.bincount up to 5; everything else lands in 6+.
    capped = np.clip(valid, 0, 6)  # 6 → overflow bucket
    counts = np.bincount(capped, minlength=7)
    for b in _EDIT_DIST_BUCKETS:
        out[str(b)] = int(counts[b])
    out["6+"] = int(counts[6])
    return out


def _polya_len_hist_np(arr: np.ndarray) -> dict[str, int]:
    out = {f"{lo}-{hi if hi is not None else '∞'}": 0 for lo, hi in _POLYA_LEN_BUCKETS}
    if arr.size == 0:
        return out
    for lo, hi in _POLYA_LEN_BUCKETS:
        if hi is None:
            mask = arr >= lo
            out[f"{lo}-∞"] = int(mask.sum())
        else:
            mask = (arr >= lo) & (arr <= hi)
            out[f"{lo}-{hi}"] = int(mask.sum())
    return out


def _delta_hist_np(arr: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {"<0": 0}
    for b in _DELTA_BUCKETS:
        out[str(b)] = 0
    out["4+"] = 0
    if arr.size == 0:
        return out
    out["<0"] = int((arr < 0).sum())
    nonneg = arr[arr >= 0]
    if nonneg.size:
        capped = np.clip(nonneg, 0, 4)
        counts = np.bincount(capped, minlength=5)
        for b in _DELTA_BUCKETS:
            out[str(b)] = int(counts[b])
        out["4+"] = int(counts[4])
    return out


def _build_summary(
    buckets: dict[tuple[str, str], dict[str, np.ndarray]],
) -> dict:
    """Convert per-(status, kind) numpy buckets into the summary dict.

    Histograms via numpy (np.bincount); percentiles via np.quantile —
    both vectorised. No Python row iteration anywhere on the hot path.
    """
    statuses = sorted({status for status, _ in buckets})
    empty = np.empty(0, dtype=np.int64)
    out: dict[str, dict] = {}
    for status in statuses:
        n_reads = 0
        for kind in _KEEP_KINDS:
            data = buckets.get((status, kind))
            if data is None:
                continue
            n_reads = max(n_reads, int(data["score"].size))

        per_kind: dict[str, dict] = {}
        for kind in ("adapter_5p", "adapter_3p"):
            data = buckets.get((status, kind))
            scores = data["score"] if data is not None else empty
            valid = scores[scores >= 0]
            per_kind[kind] = {
                "hist": _edit_dist_hist_np(scores),
                "percentiles": _percentiles_np(valid),
                "n": int(valid.size),
            }
        polyA = buckets.get((status, "polyA"))
        lengths = polyA["length"] if polyA is not None else empty
        per_kind["polyA"] = {
            "hist": _polya_len_hist_np(lengths),
            "percentiles": _percentiles_np(lengths),
            "n": int(lengths.size),
        }
        bc = buckets.get((status, "barcode"))
        if bc is None:
            scores, deltas = empty, empty
        else:
            scores, deltas = bc["score"], bc["score_delta"]
        valid_scores = scores[scores >= 0]
        valid_deltas = deltas[deltas >= 0]
        per_kind["barcode"] = {
            "edit_hist": _edit_dist_hist_np(scores),
            "edit_percentiles": _percentiles_np(valid_scores),
            "delta_hist": _delta_hist_np(deltas),
            "delta_percentiles": _percentiles_np(valid_deltas),
            "n": int(valid_scores.size),
        }
        out[status] = {"n_reads": n_reads, **per_kind}
    return out


# ──────────────────────────────────────────────────────────────────────
# Optional matplotlib PDF (numpy-vectorised; no row iteration)
# ──────────────────────────────────────────────────────────────────────


def _emit_pdf(
    buckets: dict[tuple[str, str], dict[str, np.ndarray]],
    pdf_path: Path,
) -> None:
    """Render the four-panel distribution plot from numpy arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    statuses = sorted({status for status, _ in buckets})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()
    panels = [
        ("adapter_5p", "score",       "edit distance", "5' adapter edit distance", axes_flat[0]),
        ("adapter_3p", "score",       "edit distance", "3' adapter edit distance", axes_flat[1]),
        ("polyA",      "length",      "length",        "polyA length",             axes_flat[2]),
        ("barcode",    "score_delta", "score_delta",   "barcode score_delta",       axes_flat[3]),
    ]
    for kind, metric, xlabel, title, ax in panels:
        for status in statuses:
            data = buckets.get((status, kind))
            if data is None:
                continue
            arr = data[metric]
            valid = arr[arr >= 0]
            if valid.size == 0:
                continue
            ax.hist(valid, bins=30, alpha=0.5, label=status, histtype="stepfilled")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Demux statistics — calibration distributions for ProbabilisticScorer")
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


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
    parser.add_argument(
        "--emit-rows",
        action="store_true",
        help="also write the per-(read, segment) joined table as "
        "demux_statistics.parquet (~100M rows on a full-flowcell run; "
        "useful for ad-hoc pandas/duckdb analysis but adds significant "
        "wall time + disk usage; off by default)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="skip the matplotlib distribution PDF",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="print per-stage timings to stderr",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (args.output_dir / "stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_dir = args.output_dir / "read_segments"
    demux_dir = args.output_dir / "read_demux"
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"missing segments directory: {seg_dir}")
    if not demux_dir.is_dir():
        raise FileNotFoundError(f"missing read_demux directory: {demux_dir}")

    import time

    def _tick(label: str, t0: float) -> float:
        if args.progress:
            print(
                f"  [{time.time() - t0:7.2f}s] {label}",
                file=sys.stderr,
                flush=True,
            )
        return time.time()

    t0 = time.time()

    # 1) Project read_demux to join-side columns. ~30M rows × 3 small
    #    fields → ~1 GB pa.Table held throughout.
    if args.progress:
        print("loading read_demux projection ...", file=sys.stderr, flush=True)
    demux_ds = ds.dataset(str(demux_dir), format="parquet")
    demux_proj = demux_ds.to_table(
        columns=["read_id", "status", "is_fragment"]
    )
    t1 = _tick(f"demux projection: {demux_proj.num_rows:,} rows", t0)

    # 2) Filter + project read_segments at dataset level. Arrow does
    #    projection-pushdown into parquet (only reads our 6 columns)
    #    AND predicate-pushdown for the segment_kind filter — entire
    #    operation is multithreaded across all parquet shards via
    #    Arrow's I/O thread pool, no Python in the loop.
    keep_kinds_arr = pa.array(list(_KEEP_KINDS))
    if args.progress:
        seg_ds = ds.dataset(str(seg_dir), format="parquet")
        print(
            f"filtering+projecting read_segments ({len(seg_ds.files)} shards, "
            f"keeping segment_kind in {list(_KEEP_KINDS)}) ...",
            file=sys.stderr,
            flush=True,
        )
    else:
        seg_ds = ds.dataset(str(seg_dir), format="parquet")
    filtered_segs = seg_ds.to_table(
        columns=["read_id", "segment_kind", "score", "score_delta", "start", "end"],
        filter=ds.field("segment_kind").isin(list(_KEEP_KINDS)),
    )
    t2 = _tick(f"filtered segments: {filtered_segs.num_rows:,} rows", t1)

    # 3) Single Arrow hash-join — multithreaded inside Arrow.
    joined = filtered_segs.join(demux_proj, keys="read_id", join_type="inner")
    t3 = _tick(f"joined: {joined.num_rows:,} rows", t2)

    # 4) Add length column (vectorised).
    joined = joined.append_column(
        "length", pc.subtract(joined.column("end"), joined.column("start"))
    )

    # Drop the pre-join projection — peak RAM goes through join
    # output, no longer needs the source table.
    del filtered_segs

    # 5) Optional --emit-rows write (one-shot, no Python loop).
    if args.emit_rows:
        rows_path = out_dir / "demux_statistics.parquet"
        per_row = joined.select(
            ["read_id", "status", "is_fragment", "segment_kind",
             "score", "score_delta", "length"]
        ).rename_columns(
            ["read_id", "status", "is_fragment", "segment_kind",
             "edit_distance", "score_delta", "length"]
        )
        pq.write_table(per_row, rows_path)
        del per_row
        rows_msg = f"wrote per-row table → {rows_path}"
        t3 = _tick(f"emitted rows parquet → {rows_path.name}", t3)
    else:
        rows_msg = "skipped per-row table (use --emit-rows to opt in)"

    # 6) Extract per-(status, kind) buckets (24 vectorised filters,
    #    once each — total Python orchestration constant).
    buckets = _extract_buckets(joined)
    del joined
    t4 = _tick(f"extracted {len(buckets)} (status, kind) buckets", t3)

    # 7) Build summary + write JSON.
    summary = _build_summary(buckets)
    (out_dir / "demux_statistics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    # 8) Optional PDF.
    pdf_path = out_dir / "demux_statistics.pdf"
    if args.no_pdf:
        pdf_status = "skipped PDF (--no-pdf)"
    else:
        try:
            _emit_pdf(buckets, pdf_path)
            pdf_status = f"wrote PDF → {pdf_path}"
        except ImportError:
            pdf_status = "matplotlib not available; skipping PDF"
    _tick("summary + outputs written", t4)

    print(f"wrote stats outputs → {out_dir}")
    print(f"  statuses summarized: {sorted(summary.keys())}")
    print(f"  {rows_msg}")
    print(f"  {pdf_status}")
    print(f"  total: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
