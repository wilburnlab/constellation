"""Measure the template graph on one finished round, read-only.

Usage:
    python scripts/bench-template-graph.py ROUND_DIR [--mode sketch|candidates|full]
        [--threads 48] [--limit N] [--seed 0] [--output-dir DIR]
        [--kmer 19] [--window 19] [--probes-per-seq 16] [--bucket-cap 20480]
        [--max-candidates 20480] [--identity-floor 0.99] [--tol-5p 30] [--tol-3p 30]

``ROUND_DIR`` is one round of a ``transcriptome cluster --mode em-*`` output,
e.g. ``runs/em/full_kmer_260920/rounds/r06``. Its M-step node shards
(``mstep/nodes/part-*.parquet``) are the sequences the round loop itself
relates, with the parent ids the split-sibling guard needs — which the round's
``templates.arrow`` does not carry.

Why this exists. The graph stage is on by default in every round, and every
cost figure it shipped with rests on an ASSUMED candidate-pair count: pairs are
quadratic in family size (a 2,198-member family yields all 2.4M of its pairs),
and how skewed families are at 1.57M templates has not been measured. The run
directories are not on the development machine, so this is what measures it.

Three modes:

``sketch``      the minimizer index and the probe plan: bucket occupancy and
                the EXACT number of join rows, known before any is expanded.
                Minutes. Projected PAIRS cannot be known without the join.
``candidates``  the same, then the join: the candidate-pair count, and the
                GATE line. The graph stays default-on only if this passes at
                1.57M templates.
``full``        the kernel: relation counts, what was dropped and why, the
                edit histogram, wall and memory — and a grid of how many edges
                each merge predicate would accept, from the ONE measured edge
                table, with no re-alignment. It prints the join's own numbers
                and the GATE line too, from what the builder counted. The
                bucket-size QUANTILES are the one thing only the other two
                modes print: the builder keeps the mean and the largest, and
                sketching a second time to get the rest would double the two
                largest allocations of the run.

The exit status is 3 when the gate says STOP, in every mode that reaches it,
and whatever the size of the round: a sample or a small round that is already
over the gate is over it, since the whole round holds every pair the part
does. Under the gate, only a whole round of at least 1M nodes is a PASS.

Nothing is written inside the run directory. ``--output-dir`` (full mode)
keeps ``edges.parquet`` and ``stats.json`` somewhere else.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from constellation.sequencing.transcriptome.cluster.denovo.em import graph as gr
from constellation.sequencing.transcriptome.cluster.denovo.em.refine import (
    template_id_for,
)

#: Candidate pairs at 1.57M templates above which the stage is too dear to
#: run every round by default: ~45 min a round on 48 threads at the measured
#: 0.3 ms a pair.
GATE_PAIRS = 300_000_000
#: ...and the round size from which the gate is judged at all.
GATE_MIN_NODES = 1_000_000

_NODE_COLUMNS = ["consensus", "n_reads", "parent_template_id", "haplotype_id"]


def _peak_gb() -> tuple[float, float]:
    """Peak RSS of this process and of its largest child, in GB.

    ``ru_maxrss`` is in KiB on Linux. The children's figure is the largest
    single child, not their sum: workers share the sequence buffer, so a sum
    would count it once per worker.
    """
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return own / 1024**2, kids / 1024**2


def _line(label: str, value: object) -> None:
    print(f"  {label:<34s} {value}")


def _head(title: str) -> None:
    print(f"\n== {title}")


def _quantiles(values: np.ndarray, qs=(0, 10, 50, 90, 99, 100)) -> str:
    if values.size == 0:
        return "—"
    return " / ".join(f"{int(np.percentile(values, q)):,}" for q in qs)


def _load(round_dir: Path, limit: int | None, seed: int):
    shards = sorted((round_dir / "mstep" / "nodes").glob("part-*.parquet"))
    if not shards:
        raise SystemExit(
            f"{round_dir} has no mstep/nodes/part-*.parquet — pass one round "
            f"directory of an em run, e.g. RUN/rounds/r06"
        )
    name = round_dir.name
    if not (name.startswith("r") and name[1:].isdigit()):
        raise SystemExit(f"{round_dir.name!r} is not a round directory (rNN)")
    r = int(name[1:])
    nodes = ds.dataset(shards).to_table(columns=_NODE_COLUMNS)
    n_all = nodes.num_rows
    parents = nodes.column("parent_template_id").to_numpy(zero_copy_only=False)
    # Over ALL the nodes, before any sampling: a sample keeps one child of a
    # split more often than both, and would call it a carry.
    origin = gr.split_origins(round_dir.parent, r, parents.astype(np.int64))
    ids = np.asarray(template_id_for(r + 1, np.arange(n_all)), dtype=np.int64)
    n_reads = nodes.column("n_reads").to_numpy(zero_copy_only=False).astype(np.int64)
    sequences = nodes.column("consensus")
    if limit is not None and limit < n_all:
        keep = np.sort(np.random.default_rng(seed).choice(n_all, limit, replace=False))
        sequences = sequences.take(pa.array(keep))
        ids, n_reads, origin = ids[keep], n_reads[keep], origin[keep]
    return r, n_all, sequences, ids, n_reads, origin


def _sketch(sequences, n_reads, params: gr.GraphParams):
    """The index and the probe plan, as `build_graph` makes them."""
    from constellation.sequencing.transcriptome.cluster.denovo import candidates
    from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
        extract_minimizers,
    )

    buffer, offsets = gr._sequence_buffer(sequences)
    n = offsets.shape[0] - 1
    lengths = np.diff(offsets)
    pairable = np.flatnonzero(lengths >= params.kmer + params.window - 1)
    classes = gr._identical_classes(buffer, offsets, pairable)
    rep = classes.representative
    support = (
        np.add.reduceat(n_reads[classes.members], classes.first)
        if rep.size
        else np.zeros(0, dtype=np.int64)
    )
    whole = pa.LargeStringArray.from_buffers(
        n, pa.py_buffer(offsets), pa.py_buffer(buffer)
    )
    if rep.shape[0] != n:
        whole = whole.take(pa.array(rep, pa.int64()))
    started = time.perf_counter()
    index = extract_minimizers(whole, k=params.kmer, w=params.window, max_per_seq=None)
    sketch_s = time.perf_counter() - started
    _, plan = candidates._plan_probes(
        index,
        lengths[rep].astype(np.int64),
        support.astype(np.int64),
        k=params.kmer,
        probes_per_seq=params.probes_per_seq,
        bucket_cap=params.bucket_cap,
        overflow_anchors=params.overflow_anchors,
        max_candidates=params.max_candidates,
        min_shared=params.min_shared,
        diag_band=params.diag_band,
        max_rows=params.max_rows,
    )
    hashes = index.mini_hash.numpy()
    sizes = (
        np.diff(np.flatnonzero(np.r_[True, hashes[1:] != hashes[:-1], True]))
        if hashes.size
        else np.zeros(0, dtype=np.int64)
    )
    return {
        "n": n,
        "lengths": lengths,
        "n_short": n - int(pairable.size),
        "classes": classes,
        "rep_lengths": lengths[rep].astype(np.int64),
        "support": support.astype(np.int64),
        "index": index,
        "sketch_s": sketch_s,
        "plan": plan,
        "bucket_sizes": sizes[sizes >= 2],
    }


def _report_sketch(s: dict, params: gr.GraphParams, n_all: int, sampled: bool) -> None:
    _head("nodes")
    _line("nodes in the round", f"{n_all:,}")
    if sampled:
        _line("nodes sampled (--limit)", f"{s['n']:,}")
        print(
            "  NOTE a sample understates family sizes, and pairs are quadratic "
            "in them:\n       nothing below extrapolates to the whole round."
        )
    _line("bases", f"{int(s['lengths'].sum()):,}")
    _line("length min/p10/p50/p90/p99/max", _quantiles(s["lengths"]))
    _line(
        f"too short to pair (< {params.kmer + params.window - 1} nt)",
        f"{s['n_short']:,}",
    )
    size = s["classes"].size
    _line("distinct sequences", f"{size.shape[0]:,}")
    _line("byte-identical classes (> 1 row)", f"{int((size > 1).sum()):,}")
    _line("largest identical class", f"{int(size.max()) if size.size else 0:,}")

    _head(f"sketch  (k{params.kmer} / w{params.window}, uncapped)")
    n_min = int(s["index"].mini_hash.shape[0])
    bases = max(int(s["rep_lengths"].sum()), 1)
    _line("minimizers", f"{n_min:,}  ({n_min / bases:.3f} per nt)")
    _line(
        "sketch wall",
        f"{s['sketch_s']:.1f} s  ({bases / 1e6 / max(s['sketch_s'], 1e-9):.1f} Mb/s)",
    )
    b = s["bucket_sizes"]
    _line("buckets of size >= 2", f"{b.size:,}")
    if b.size:
        _line("bucket size mean", f"{b.mean():.1f}")
        _line(
            "bucket size, size-biased mean",
            f"{(b.astype(float) ** 2).sum() / b.sum():.1f}",
        )
        _line("bucket size p50/p90/p99/max", _quantiles(b, (50, 90, 99, 100)))
        _line(
            f"buckets above bucket_cap {params.bucket_cap:,}",
            f"{int((b > params.bucket_cap).sum()):,}",
        )

    _head(f"probe plan  ({params.probes_per_seq} probes per sequence)")
    plan = s["plan"]
    _line("sequences with a probe", f"{plan.get('n_with_probes', 0):,}")
    _line("probes", f"{plan.get('n_probes', 0):,}")
    _line("overflow probes (anchor fallback)", f"{plan.get('n_overflow_probes', 0):,}")
    _line("JOIN ROWS, exact", f"{plan.get('n_rows_projected', 0):,}")
    _line(
        "smallest bucket demoted by max_rows",
        f"{plan.get('smallest_demoted_bucket', 0):,}",
    )
    print(
        "  (join rows are exact; candidate PAIRS cannot be known without the "
        "join — run --mode candidates)"
    )


def _candidates(s: dict, params: gr.GraphParams, threads: int):
    from constellation.sequencing.transcriptome.cluster.denovo.candidates import (
        generate_containment_candidates,
    )

    started = time.perf_counter()
    found = generate_containment_candidates(
        s["index"],
        s["rep_lengths"],
        s["support"],
        k=params.kmer,
        probes_per_seq=params.probes_per_seq,
        bucket_cap=params.bucket_cap,
        overflow_anchors=params.overflow_anchors,
        max_candidates=params.max_candidates,
        min_shared=params.min_shared,
        diag_band=params.diag_band,
        chunk_rows=params.chunk_rows,
        max_rows=params.max_rows,
        threads=threads,
    )
    return found, time.perf_counter() - started


def gate_verdict(pairs: int, n_all: int, sampled: bool) -> tuple[bool, str]:
    """``(ok, verdict)`` for a candidate-pair count.

    Over the gate is STOP whatever was measured: the whole round holds every
    pair a sample of it or a smaller round does, so a part that is over is a
    whole that is over. Under it, only a whole round at scale is a PASS —
    the gate is a statement about a 9.4M-read run's round, and a sample
    understates family sizes, in which pairs are quadratic.
    """
    if pairs > GATE_PAIRS:
        of = "a sample of the round" if sampled else f"a round of {n_all:,} nodes"
        return False, f"STOP ({of} is already over the gate)"
    if sampled:
        return True, "not judged: this is a sample"
    if n_all < GATE_MIN_NODES:
        return True, f"not judged: fewer than {GATE_MIN_NODES:,} nodes"
    return True, "PASS"


def _report_candidates(st: dict, seconds: float, n_all: int, sampled: bool) -> bool:
    _head("candidate join")
    pairs = int(st.get("n_candidates", 0))
    _line("join rows", f"{st.get('n_rows', 0):,}")
    _line("templates on the anchor fallback", f"{st.get('n_overflow_templates', 0):,}")
    _line("templates cut at max_candidates", f"{st.get('n_truncated_templates', 0):,}")
    _line("antisense pairs dropped", f"{st.get('n_antisense', 0):,}")
    _line("join wall", f"{seconds:.1f} s")
    ok, verdict = gate_verdict(pairs, n_all, sampled)
    print(
        f"\n  candidate pairs: {pairs:,}  (gate: {GATE_PAIRS:,} at 1.57M "
        f"templates; this round has {n_all:,})  {verdict}"
    )
    if not ok:
        print(
            "  Above the gate the graph is too dear to run every round by "
            "default:\n  the choice is --template-graph final as the default, "
            "or tighter caps."
        )
    return ok


def _report_builder(
    st: dict, params: gr.GraphParams, n_all: int, sampled: bool
) -> None:
    """What `--mode sketch` reports, from the builder's own counts."""
    _head("nodes")
    _line("nodes in the round", f"{n_all:,}")
    if sampled:
        _line("nodes sampled (--limit)", f"{st['n_sequences']:,}")
        print(
            "  NOTE a sample understates family sizes, and pairs are quadratic "
            "in them:\n       nothing below extrapolates to the whole round."
        )
    _line(
        f"too short to pair (< {params.kmer + params.window - 1} nt)",
        f"{st['n_unpairable_short']:,}",
    )
    _line("distinct sequences", f"{st['n_unique']:,}")
    _line("byte-identical classes (> 1 row)", f"{st['n_identical_classes']:,}")
    _line("largest identical class", f"{st.get('largest_identical_class', 0):,}")

    c = st["candidates"]
    _head(f"sketch and probe plan  (k{params.kmer} / w{params.window}, uncapped)")
    _line("buckets", f"{c.get('n_buckets', 0):,}")
    _line("bucket size mean", f"{c.get('bucket_size_mean', 0.0):.1f}")
    _line(
        "bucket size, size-biased mean", f"{c.get('bucket_size_biased_mean', 0.0):.1f}"
    )
    _line("bucket size max", f"{c.get('bucket_size_max', 0):,}")
    _line("sequences with a probe", f"{c.get('n_with_probes', 0):,}")
    _line("probes", f"{c.get('n_probes', 0):,}")
    _line("overflow probes (anchor fallback)", f"{c.get('n_overflow_probes', 0):,}")
    _line("JOIN ROWS, exact", f"{c.get('n_rows_projected', 0):,}")
    _line(
        "smallest bucket demoted by max_rows",
        f"{c.get('smallest_demoted_bucket', 0):,}",
    )
    print("  (bucket-size quantiles: --mode sketch)")


def _grid(edges_path: Path) -> None:
    """How many edges each predicate would accept, from the one edge table."""
    _head("mergeable edges by predicate  (from the measured edges; no re-alignment)")
    tols = ((10, 10), (20, 20), (30, 30))
    print(
        f"  {'max_edits':>9s} {'siblings':>9s} "
        + " ".join(f"{f'{a}:{b}':>12s}" for a, b in tols)
    )
    counts = {
        (e, sib, t): 0 for e in (0, 1, 2, 5) for sib in (False, True) for t in tols
    }
    for batch in pq.ParquetFile(edges_path).iter_batches(batch_size=1 << 20):
        table = pa.Table.from_batches([batch])
        for e, sib, (t5, t3) in counts:
            pred = gr.MergePredicate(
                max_edits=e, tol_5p=t5, tol_3p=t3, merge_siblings=sib
            )
            counts[(e, sib, (t5, t3))] += int(gr.is_mergeable(table, pred).sum())
    for e in (0, 1, 2, 5):
        for sib in (False, True):
            row = " ".join(f"{counts[(e, sib, t)]:>12,}" for t in tols)
            print(f"  {e:>9d} {'merged' if sib else 'excluded':>9s} {row}")
    print(
        "  (columns are 5':3' tolerances in nt; a tolerance above the graph's "
        "own finds nothing more,\n   because `equivalent` is defined by the "
        "graph's. 'excluded' is the default sibling guard.)"
    )


def _full(
    sequences, ids, n_reads, origin, r, params, threads, out: Path | None, n_all: int
) -> bool:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="bench-template-graph-") as scratch:
        path = (out if out is not None else Path(scratch)) / "edges.parquet"
        result = gr.build_graph(
            sequences,
            ids=ids,
            n_reads=n_reads,
            split_origin=origin,
            node_round=r,
            params=params,
            predicate=gr.MergePredicate(tol_5p=params.tol_5p, tol_3p=params.tol_3p),
            threads=threads,
            output_path=path,
            progress=lambda line: print(f"  [graph] {line}", file=sys.stderr),
        )
        st = result.stats
        sampled = st["n_sequences"] != n_all
        _report_builder(st, params, n_all, sampled)
        ok = _report_candidates(
            st["candidates"], st["seconds"]["candidates"], n_all, sampled
        )
        _head("edges")
        _line("candidate pairs aligned", f"{st['n_pairs_aligned']:,}")
        _line("edges written", f"{st['n_edges']:,}")
        _line(
            "equivalent",
            f"{st['n_equivalent']:,}  ({st['n_equivalent_exact']:,} at zero edits)",
        )
        _line("  byte-identical twins", f"{st['n_exact_twins']:,}")
        _line("  separated by the same split", f"{st['n_same_split_origin']:,}")
        c = st["contained"]
        _line(
            "contained 5' / 3' / both",
            f"{c.get('5p', 0):,} / {c.get('3p', 0):,} / {c.get('both', 0):,}",
        )
        _line("  contained at zero edits", f"{st['n_exact_nested']:,}")
        _line("mergeable, default predicate", f"{st['n_mergeable']:,}")
        _head("candidate pairs that are not edges")
        for reason, count in st["dropped"].items():
            _line(reason, f"{count:,}")
        _head("edits over the shared span, equivalent edges")
        for edits, count in st["n_edits_hist"].items():
            _line(edits, f"{count:,}")
        _head("cost")
        sec = st["seconds"]
        for stage in ("sketch", "candidates", "kernel", "total"):
            _line(f"{stage} wall", f"{sec[stage]:,.1f} s")
        if sec["kernel"] > 0:
            _line(
                "pairs per second (kernel)",
                f"{st['n_pairs_aligned'] / sec['kernel']:,.0f}  on {threads} threads",
            )
        _line("edges.parquet", f"{path.stat().st_size / 1e6:,.1f} MB")
        _grid(path)
        if out is not None:
            (out / "stats.json").write_text(json.dumps(st, indent=2))
            print(f"\n  kept: {path}  and  {out / 'stats.json'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("round_dir", type=Path, metavar="ROUND_DIR")
    ap.add_argument("--mode", choices=("sketch", "candidates", "full"), default="full")
    ap.add_argument(
        "--sketch-only", action="store_true", help="alias for --mode sketch"
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--limit", type=int, default=None, help="a seeded random sample of N nodes"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--kmer", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--probes-per-seq", type=int, default=None)
    ap.add_argument("--bucket-cap", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--identity-floor", type=float, default=None)
    ap.add_argument("--tol-5p", type=int, default=None)
    ap.add_argument("--tol-3p", type=int, default=None)
    args = ap.parse_args()
    mode = "sketch" if args.sketch_only else args.mode

    round_dir = args.round_dir.resolve()
    out = None
    if args.output_dir is not None:
        if mode != "full":
            ap.error("--output-dir keeps the edge table, which only --mode full makes")
        out = args.output_dir.resolve()
        run = round_dir.parent.parent
        if out == run or run in out.parents:
            ap.error(f"--output-dir must not be inside the run directory {run}")
        out.mkdir(parents=True, exist_ok=True)

    overrides = {
        "kmer": args.kmer,
        "window": args.window,
        "probes_per_seq": args.probes_per_seq,
        "bucket_cap": args.bucket_cap,
        "max_candidates": args.max_candidates,
        "identity_floor": args.identity_floor,
        "tol_5p": args.tol_5p,
        "tol_3p": args.tol_3p,
    }
    params = replace(
        gr.GraphParams(), **{k: v for k, v in overrides.items() if v is not None}
    )

    started = time.perf_counter()
    r, n_all, sequences, ids, n_reads, origin = _load(round_dir, args.limit, args.seed)
    sampled = len(sequences) != n_all
    print(
        f"template graph bench — {round_dir}  (round {r}, mode {mode}, {args.threads} threads)"
    )
    print(f"graph parameters: {json.dumps(params.semantic())}")

    ok = True
    if mode == "full":
        # build_graph sketches and joins for itself; doing it here as well
        # would double the two largest allocations for nothing.
        ok = _full(sequences, ids, n_reads, origin, r, params, args.threads, out, n_all)
    else:
        s = _sketch(sequences, n_reads, params)
        _report_sketch(s, params, n_all, sampled)
        if mode == "candidates":
            found, seconds = _candidates(s, params, args.threads)
            ok = _report_candidates(found.stats, seconds, n_all, sampled)

    own, kids = _peak_gb()
    _head("process")
    _line("wall, everything", f"{time.perf_counter() - started:,.1f} s")
    _line("peak RSS, this process", f"{own:.2f} GB")
    _line("peak RSS, largest worker", f"{kids:.2f} GB")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
