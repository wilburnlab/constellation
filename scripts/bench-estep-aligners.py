"""Benchmark the EM E-step's two aligners on one fixed input.

Usage:
    python scripts/bench-estep-aligners.py --run-dir RUN --round 1 \
        --n-reads 400000 --threads 96 --out-dir BENCH \
        [--shortlist-k 4 8 16 32] [--shortlist-frac 0.8] [--seed 0]

Runs ``--estep-aligner minimap2`` (minimap2 ``-c`` on every candidate) and
``--estep-aligner edlib`` (minimap2 without ``-c`` to shortlist, edlib on the
shortlist) against the SAME reads and the SAME round's templates of an
existing ``transcriptome cluster --mode em-*`` output dir, and reports the
handoff's acceptance numbers (``handoff_polya_and_estep.md`` §4):

- wall and CPU seconds per aligner (CPU includes minimap2 and pool workers);
- admitted-identity p10 / median, computed the same way for both
  (``n_match / aln_len`` over the winner's CIGAR);
- the fraction of reads assigned the same template, and assigned at all;
- ``t_start`` / ``q_start`` agreement for reads with the same winner;
- the ``shortlist_truncated`` rate, per ``--shortlist-k``.

The default aligner stays minimap2 until this passes: identity p10 ≥ 0.95,
winner agreement high, E-step ≤ 200 s on 400k reads x the em-kmer round-1
template set.

The reads are a random subset of the run's corpus. The subset FASTA keeps the
corpus's row-index names, which is what the E-step reducer keys on.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
    EM_ASSIGNMENT_TABLE,
    run_em_estep,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
    CORPUS_ARROW,
    ReadStore,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
    TEMPLATES_ARROW,
    TEMPLATES_FASTA,
    TemplateStore,
)


class _SubsetReads:
    """A ReadStore with no ``n_reads``, so the runner does not emit every
    read outside the subset as unmapped."""

    def __init__(self, store: ReadStore):
        self._s = store

    def take_sequences(self, rows):
        return self._s.take_sequences(rows)

    def take_read_ids(self, rows):
        return self._s.take_read_ids(rows)

    @property
    def sample_id(self):
        return self._s.sample_id


def _cpu() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


def _write_subset(reads: ReadStore, rows: np.ndarray, path: Path) -> None:
    with path.open("w") as fh:
        for start in range(0, rows.size, 50_000):
            chunk = rows[start : start + 50_000]
            for r, seq in zip(chunk.tolist(), reads.take_sequences(chunk)):
                fh.write(f">{r}\n{seq}\n")


def _identity(cigars: list[str | None]) -> np.ndarray:
    import re

    out = np.full(len(cigars), np.nan)
    for i, c in enumerate(cigars):
        if not c:
            continue
        ops = re.findall(r"(\d+)([=XID])", c)
        eq = sum(int(n) for n, op in ops if op == "=")
        tot = sum(int(n) for n, _ in ops)
        out[i] = eq / tot if tot else np.nan
    return out


def _run(label, aligner, args, templates_dir, store, reads, subset_fa, corpus, **kw):
    out = args.out_dir / label
    t0, c0 = time.time(), _cpu()
    stats = run_em_estep(
        templates_dir / TEMPLATES_FASTA,
        subset_fa,
        store=store,
        reads=reads,
        output_dir=out,
        round_index=args.round,
        threads=args.threads,
        minimap2_n=args.minimap2_n,
        index_batch_size=args.index_batch_size,
        aligner=aligner,
        align_workers=args.threads,
        corpus_path=corpus,
        templates_path=templates_dir / TEMPLATES_ARROW,
        **kw,
    )
    wall, cpu = time.time() - t0, _cpu() - c0
    table = (
        ds.dataset(sorted(out.glob("part-*.parquet")), schema=EM_ASSIGNMENT_TABLE)
        .to_table(
            columns=[
                "read_row",
                "template_id",
                "q_start",
                "t_start",
                "cigar",
                "shortlist_truncated",
            ]
        )
        .sort_by("read_row")
    )
    return {"label": label, "wall_s": wall, "cpu_s": cpu, "stats": stats}, table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--n-reads", type=int, default=400_000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--minimap2-n", type=int, default=2000)
    ap.add_argument("--index-batch-size", default="16G")
    ap.add_argument("--shortlist-k", type=int, nargs="+", default=[16])
    ap.add_argument("--shortlist-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = args.run_dir / "corpus" / CORPUS_ARROW
    templates_dir = args.run_dir / "rounds" / f"r{args.round:02d}" / "templates"
    full = ReadStore.open(corpus)
    store = TemplateStore.open(templates_dir / TEMPLATES_ARROW)
    rng = np.random.default_rng(args.seed)
    n = min(args.n_reads, full.n_reads)
    rows = np.sort(rng.choice(full.n_reads, size=n, replace=False)).astype(np.int64)
    subset_fa = args.out_dir / "reads.subset.fa"
    _write_subset(full, rows, subset_fa)
    reads = _SubsetReads(full)
    print(f"{n:,} reads x {store.n_templates:,} templates (round {args.round})")

    base, mm = _run(
        "minimap2", "minimap2", args, templates_dir, store, reads, subset_fa, corpus
    )
    report = [base]
    mm_ident = _identity(mm.column("cigar").to_pylist())
    mm_tid = mm.column("template_id").to_numpy()
    for k in args.shortlist_k:
        res, ed = _run(
            f"edlib_k{k}",
            "edlib",
            args,
            templates_dir,
            store,
            reads,
            subset_fa,
            corpus,
            shortlist_k=k,
            shortlist_frac=args.shortlist_frac,
        )
        assert ed.column("read_row").equals(mm.column("read_row"))
        ed_tid = ed.column("template_id").to_numpy()
        ed_ident = _identity(ed.column("cigar").to_pylist())
        same = (ed_tid == mm_tid) & (mm_tid >= 0)
        both = (ed_tid >= 0) & (mm_tid >= 0)
        dq = np.abs(ed.column("q_start").to_numpy() - mm.column("q_start").to_numpy())
        dt = np.abs(ed.column("t_start").to_numpy() - mm.column("t_start").to_numpy())
        res.update(
            {
                "assigned_frac": float((ed_tid >= 0).mean()),
                "same_winner_frac_of_both_assigned": float(same.sum() / max(both.sum(), 1)),
                "identity_p10": float(np.nanpercentile(ed_ident, 10)),
                "identity_median": float(np.nanmedian(ed_ident)),
                "t_start_exact_frac_same_winner": float((dt[same] == 0).mean())
                if same.any()
                else None,
                "q_start_exact_frac_same_winner": float((dq[same] == 0).mean())
                if same.any()
                else None,
                "shortlist_truncated_frac": float(
                    pc.mean(pc.cast(ed.column("shortlist_truncated"), "float64")).as_py()
                    or 0.0
                ),
            }
        )
        report.append(res)
    base.update(
        {
            "assigned_frac": float((mm_tid >= 0).mean()),
            "identity_p10": float(np.nanpercentile(mm_ident, 10)),
            "identity_median": float(np.nanmedian(mm_ident)),
        }
    )

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    cols = (
        "label",
        "wall_s",
        "cpu_s",
        "assigned_frac",
        "same_winner_frac_of_both_assigned",
        "identity_p10",
        "identity_median",
        "t_start_exact_frac_same_winner",
        "shortlist_truncated_frac",
    )
    with (args.out_dir / "report.tsv").open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in report:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print((args.out_dir / "report.tsv").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
