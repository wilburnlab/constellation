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
segment rows. Materialising that as a single ``pa.Table`` and walking
it via ``to_pylist()`` is single-threaded, RAM-unbounded, and slow.

Instead we:

  - Project ``read_demux/`` once to ``(read_id, status, is_fragment)`` —
    ~30M rows × ~30 bytes is bounded RAM, kept as the join-side
    lookup table (never converted to Python).
  - Stream ``read_segments/`` via ``dataset.to_batches(columns=...)``.
    Per batch: Arrow-filter to the four interesting segment_kinds,
    Arrow hash-join onto the demux projection, then **append numpy
    chunks** (not Python rows) to per-``(status, kind)`` accumulators.
  - After all batches, ``np.concatenate`` per (status, kind, metric)
    into dense numpy arrays bounded by the sum of relevant rows.
    Histograms via ``np.bincount``, percentiles via ``np.quantile`` —
    both vectorised.

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
# Streaming accumulator
# ──────────────────────────────────────────────────────────────────────


class _Accumulator:
    """Per-``(status, segment_kind)`` numpy buffer.

    Stores list-of-numpy-chunks per metric (score, score_delta, length).
    ``finalize()`` concatenates each list into one dense array.
    Memory peak per bucket ≈ sum of values seen for that bucket — for
    the full Pichia BAM that's ~30M ints across all buckets, ~120 MB
    total in dense numpy form.
    """

    __slots__ = ("score_chunks", "delta_chunks", "length_chunks")

    def __init__(self) -> None:
        self.score_chunks: list[np.ndarray] = []
        self.delta_chunks: list[np.ndarray] = []
        self.length_chunks: list[np.ndarray] = []

    def extend(
        self,
        scores: np.ndarray,
        deltas: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        if scores.size:
            self.score_chunks.append(scores)
        if deltas.size:
            self.delta_chunks.append(deltas)
        if lengths.size:
            self.length_chunks.append(lengths)

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.concatenate(self.score_chunks) if self.score_chunks else np.empty(0, dtype=np.int64)
        d = np.concatenate(self.delta_chunks) if self.delta_chunks else np.empty(0, dtype=np.int64)
        l_ = np.concatenate(self.length_chunks) if self.length_chunks else np.empty(0, dtype=np.int64)
        return s, d, l_


# ──────────────────────────────────────────────────────────────────────
# Streaming join + accumulate
# ──────────────────────────────────────────────────────────────────────


def _stream_into_accumulators(
    demux_proj: pa.Table,
    seg_dataset: ds.Dataset,
    *,
    batch_size: int,
    progress: bool,
) -> tuple[
    dict[tuple[str, str], _Accumulator],
    pa.Table | None,
]:
    """Walk segments dataset in batches, join + accumulate per
    ``(status, segment_kind)``.

    Returns the accumulators plus, optionally, a per-row table for
    ``--emit-rows``. The per-row table is built incrementally as
    ``pa.Table.from_pylist`` of small per-batch arrow tables, so it
    doesn't itself drive RAM blow-up — but it does add a write-time
    cost the user opts into deliberately.
    """
    keep_kinds_arr = pa.array(list(_KEEP_KINDS))
    accumulators: dict[tuple[str, str], _Accumulator] = {}
    rows_chunks: list[pa.Table] = []

    n_seen = 0
    n_kept = 0
    for batch in seg_dataset.to_batches(
        columns=[
            "read_id", "segment_kind", "score", "score_delta", "start", "end"
        ],
        batch_size=batch_size,
    ):
        n_seen += batch.num_rows
        if batch.num_rows == 0:
            continue
        batch_table = pa.Table.from_batches([batch])

        # 1) Filter to interesting segment kinds — Arrow C-level.
        kind_mask = pc.is_in(batch_table.column("segment_kind"), value_set=keep_kinds_arr)
        filtered = batch_table.filter(kind_mask)
        if filtered.num_rows == 0:
            continue

        # 2) Join (status, is_fragment) onto each segment row via
        #    the small projected demux lookup. Arrow's hash join is
        #    vectorised; right side is ~30M rows held once.
        joined = filtered.join(demux_proj, keys="read_id", join_type="inner")
        if joined.num_rows == 0:
            continue

        # 3) Compute length column (end - start).
        length_col = pc.subtract(joined.column("end"), joined.column("start"))
        joined = joined.append_column("length", length_col)
        n_kept += joined.num_rows

        # 4) For each (status, kind) bucket present in this batch,
        #    extract the three metric columns as numpy and append.
        statuses_arr = joined.column("status")
        kinds_arr = joined.column("segment_kind")

        # Distinct statuses seen in this batch — small set.
        unique_statuses = pc.unique(statuses_arr).to_pylist()
        for status in unique_statuses:
            sm = pc.equal(statuses_arr, status)
            for kind in _KEEP_KINDS:
                km = pc.equal(kinds_arr, kind)
                both = pc.and_(sm, km)
                sub = joined.filter(both)
                if sub.num_rows == 0:
                    continue
                acc = accumulators.setdefault((status, kind), _Accumulator())
                # Numpy extracts: each call is one C-level copy, no
                # Python row materialization.
                scores = sub.column("score").to_numpy(zero_copy_only=False)
                deltas = sub.column("score_delta").to_numpy(zero_copy_only=False)
                lengths = sub.column("length").to_numpy(zero_copy_only=False)
                # score_delta is nullable; null → -1 sentinel for the
                # delta histogram's "<0" bucket. fill_null in Arrow
                # would be more vectorised but the conversion already
                # gave us a numpy array; null-mask via numpy.
                if hasattr(deltas, "mask") or np.issubdtype(deltas.dtype, np.floating):
                    # Pyarrow returns a masked / float64 array when
                    # nulls are present; convert nulls → -1 sentinel.
                    deltas_np = np.where(np.isnan(deltas), -1, deltas).astype(np.int64) \
                        if np.issubdtype(deltas.dtype, np.floating) else \
                        np.where(deltas.mask, -1, deltas.data).astype(np.int64)
                else:
                    deltas_np = deltas.astype(np.int64)
                acc.extend(
                    scores.astype(np.int64, copy=False),
                    deltas_np,
                    lengths.astype(np.int64, copy=False),
                )

        if progress:
            print(
                f"  ... {n_seen:>12,} segments scanned, {n_kept:>12,} kept",
                file=sys.stderr,
                flush=True,
            )
        # Don't accumulate per-row chunks unless caller asked
        # (handled by emit_rows path in main).
        del batch_table, filtered, joined

    return accumulators, None


def _stream_into_accumulators_with_rows(
    demux_proj: pa.Table,
    seg_dataset: ds.Dataset,
    *,
    batch_size: int,
    progress: bool,
    rows_writer: pq.ParquetWriter,
) -> dict[tuple[str, str], _Accumulator]:
    """Same streaming path but additionally writes per-row stats to a
    ParquetWriter as it goes — opt-in via ``--emit-rows``.
    """
    keep_kinds_arr = pa.array(list(_KEEP_KINDS))
    accumulators: dict[tuple[str, str], _Accumulator] = {}

    n_seen = 0
    n_kept = 0
    for batch in seg_dataset.to_batches(
        columns=[
            "read_id", "segment_kind", "score", "score_delta", "start", "end"
        ],
        batch_size=batch_size,
    ):
        n_seen += batch.num_rows
        if batch.num_rows == 0:
            continue
        batch_table = pa.Table.from_batches([batch])

        kind_mask = pc.is_in(batch_table.column("segment_kind"), value_set=keep_kinds_arr)
        filtered = batch_table.filter(kind_mask)
        if filtered.num_rows == 0:
            continue
        joined = filtered.join(demux_proj, keys="read_id", join_type="inner")
        if joined.num_rows == 0:
            continue
        length_col = pc.subtract(joined.column("end"), joined.column("start"))
        joined = joined.append_column("length", length_col)
        n_kept += joined.num_rows

        # Per-row write: rename + project to canonical column order
        # before writing each batch.
        per_row = joined.select(
            ["read_id", "status", "is_fragment", "segment_kind",
             "score", "score_delta", "length"]
        ).rename_columns(
            ["read_id", "status", "is_fragment", "segment_kind",
             "edit_distance", "score_delta", "length"]
        )
        rows_writer.write_table(per_row)

        # Accumulate per-(status, kind) numpy chunks (same as the
        # rows-less variant).
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
                acc = accumulators.setdefault((status, kind), _Accumulator())
                scores = sub.column("score").to_numpy(zero_copy_only=False)
                deltas = sub.column("score_delta").to_numpy(zero_copy_only=False)
                lengths = sub.column("length").to_numpy(zero_copy_only=False)
                if np.issubdtype(deltas.dtype, np.floating):
                    deltas_np = np.where(np.isnan(deltas), -1, deltas).astype(np.int64)
                else:
                    deltas_np = deltas.astype(np.int64)
                acc.extend(
                    scores.astype(np.int64, copy=False),
                    deltas_np,
                    lengths.astype(np.int64, copy=False),
                )

        if progress:
            print(
                f"  ... {n_seen:>12,} segments scanned, {n_kept:>12,} kept",
                file=sys.stderr,
                flush=True,
            )
        del batch_table, filtered, joined, per_row

    return accumulators


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
    accumulators: dict[tuple[str, str], _Accumulator],
) -> dict:
    """Convert per-(status, kind) accumulators into the summary dict.

    Each accumulator's chunked numpy arrays are concatenated once,
    then the histogram + percentiles are computed via numpy on the
    dense arrays. No Python row iteration anywhere on the hot path.
    """
    statuses = sorted({status for status, _ in accumulators})
    out: dict[str, dict] = {}
    for status in statuses:
        # Read-count for this status: derive from any kind's row count
        # (every status-kind pair sees the same set of reads' segments,
        # though not every read has every kind). We use barcode rows
        # (one per read with a barcode found) as the canonical n_reads
        # estimate, falling back to the largest bucket.
        n_reads = 0
        for kind in _KEEP_KINDS:
            acc = accumulators.get((status, kind))
            if acc is None:
                continue
            n = sum(c.size for c in acc.score_chunks)
            n_reads = max(n_reads, n)

        per_kind: dict[str, dict] = {}
        for kind in ("adapter_5p", "adapter_3p"):
            acc = accumulators.get((status, kind))
            if acc is None:
                per_kind[kind] = {
                    "hist": _edit_dist_hist_np(np.empty(0, dtype=np.int64)),
                    "percentiles": _percentiles_np(np.empty(0, dtype=np.int64)),
                    "n": 0,
                }
                continue
            scores, _, _ = acc.finalize()
            valid = scores[scores >= 0]
            per_kind[kind] = {
                "hist": _edit_dist_hist_np(scores),
                "percentiles": _percentiles_np(valid),
                "n": int(valid.size),
            }
        # PolyA: use length, not score.
        polyA_acc = accumulators.get((status, "polyA"))
        if polyA_acc is None:
            per_kind["polyA"] = {
                "hist": _polya_len_hist_np(np.empty(0, dtype=np.int64)),
                "percentiles": _percentiles_np(np.empty(0, dtype=np.int64)),
                "n": 0,
            }
        else:
            _, _, lengths = polyA_acc.finalize()
            per_kind["polyA"] = {
                "hist": _polya_len_hist_np(lengths),
                "percentiles": _percentiles_np(lengths),
                "n": int(lengths.size),
            }
        bc_acc = accumulators.get((status, "barcode"))
        if bc_acc is None:
            per_kind["barcode"] = {
                "edit_hist": _edit_dist_hist_np(np.empty(0, dtype=np.int64)),
                "edit_percentiles": _percentiles_np(np.empty(0, dtype=np.int64)),
                "delta_hist": _delta_hist_np(np.empty(0, dtype=np.int64)),
                "delta_percentiles": _percentiles_np(np.empty(0, dtype=np.int64)),
                "n": 0,
            }
        else:
            scores, deltas, _ = bc_acc.finalize()
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
    accumulators: dict[tuple[str, str], _Accumulator],
    pdf_path: Path,
) -> None:
    """Render the four-panel distribution plot from numpy arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    statuses = sorted({status for status, _ in accumulators})
    finalised: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        key: acc.finalize() for key, acc in accumulators.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()
    panels = [
        ("adapter_5p", 0, "edit distance", "5' adapter edit distance", axes_flat[0]),
        ("adapter_3p", 0, "edit distance", "3' adapter edit distance", axes_flat[1]),
        ("polyA",      2, "length",        "polyA length",             axes_flat[2]),
        ("barcode",    1, "score_delta",   "barcode score_delta",       axes_flat[3]),
    ]
    for kind, metric_idx, xlabel, title, ax in panels:
        for status in statuses:
            arrs = finalised.get((status, kind))
            if arrs is None:
                continue
            arr = arrs[metric_idx]
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
        "--batch-size",
        type=int,
        default=500_000,
        help="rows per segment-dataset batch (default 500000 — tune lower "
        "if RAM-constrained, higher for fewer Arrow op invocations)",
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
        help="print per-batch scan progress to stderr",
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

    # 1) Project demux to the join-side columns. ~30M rows × 3 small
    #    fields → ~1 GB pa.Table held throughout.
    if args.progress:
        print("loading demux projection (read_id, status, is_fragment) ...",
              file=sys.stderr, flush=True)
    demux_ds = ds.dataset(str(demux_dir), format="parquet")
    demux_proj = demux_ds.to_table(
        columns=["read_id", "status", "is_fragment"]
    )
    if args.progress:
        print(f"  loaded {demux_proj.num_rows:,} demux rows",
              file=sys.stderr, flush=True)

    # 2) Stream segments → accumulate.
    seg_ds = ds.dataset(str(seg_dir), format="parquet")
    if args.progress:
        print(f"streaming segments dataset ({len(seg_ds.files)} shards) ...",
              file=sys.stderr, flush=True)

    if args.emit_rows:
        rows_path = out_dir / "demux_statistics.parquet"
        rows_schema = pa.schema([
            pa.field("read_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("is_fragment", pa.bool_()),
            pa.field("segment_kind", pa.string()),
            pa.field("edit_distance", pa.int32()),
            pa.field("score_delta", pa.int32()),
            pa.field("length", pa.int32()),
        ])
        with pq.ParquetWriter(rows_path, rows_schema) as rows_writer:
            accumulators = _stream_into_accumulators_with_rows(
                demux_proj,
                seg_ds,
                batch_size=args.batch_size,
                progress=args.progress,
                rows_writer=rows_writer,
            )
        rows_msg = f"wrote per-row table → {rows_path}"
    else:
        accumulators, _ = _stream_into_accumulators(
            demux_proj,
            seg_ds,
            batch_size=args.batch_size,
            progress=args.progress,
        )
        rows_msg = "skipped per-row table (use --emit-rows to opt in)"

    # 3) Build summary + write JSON.
    summary = _build_summary(accumulators)
    (out_dir / "demux_statistics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    # 4) Optional PDF.
    pdf_path = out_dir / "demux_statistics.pdf"
    if args.no_pdf:
        pdf_status = "skipped PDF (--no-pdf)"
    else:
        try:
            _emit_pdf(accumulators, pdf_path)
            pdf_status = f"wrote PDF → {pdf_path}"
        except ImportError:
            pdf_status = "matplotlib not available; skipping PDF"

    print(f"wrote stats outputs → {out_dir}")
    print(f"  statuses summarized: {sorted(summary.keys())}")
    print(f"  {rows_msg}")
    print(f"  {pdf_status}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
