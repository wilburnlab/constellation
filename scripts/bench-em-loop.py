#!/usr/bin/env python
"""Profile the de novo EM loop end to end on a synthetic corpus shaped like
real data, and fingerprint everything it writes.

Two jobs, so that a performance change can be shown to be both faster and
exact:

``build``    a demultiplexing directory the EM corpus reader accepts: genes
             with long-tailed depth, alternative first exons, haplotypes,
             5' truncation (random and at a few sharp internal positions),
             3' jitter and homopolymer-biased indels.
``run``      ``run_em`` under cProfile; writes ``bench.json`` (stage seconds
             from the run's own records, wall time, the top functions) and
             ``digests.json`` (a digest of every table the run wrote: each
             round's assignments, nodes and membership, templates, the final
             clusters and membership).
``compare``  two ``run`` outputs: timings side by side, and every digest that
             differs (none, for an exact change).

    python scripts/bench-em-loop.py build /tmp/emb/demux --reads 5000
    python scripts/bench-em-loop.py run /tmp/emb/demux /tmp/emb/before --seeding kmer
    python scripts/bench-em-loop.py compare /tmp/emb/before /tmp/emb/after

The corpus is generated from ``--seed`` alone, so two checkouts build the
same one. Nothing here is imported by the package.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

CODONS = [
    a + b + c
    for a in "ACGT"
    for b in "ACGT"
    for c in "ACGT"
    if a + b + c not in ("TAA", "TAG", "TGA")
]


def _orf(rng, n_codons: int) -> str:
    return "ATG" + "".join(rng.choice(CODONS, size=n_codons - 2)) + "TAA"


def _utr(rng, n: int) -> str:
    """Random sequence with a homopolymer run every ~60 nt."""
    out = []
    while sum(len(x) for x in out) < n:
        out.append("".join(rng.choice(list("ACGT"), size=int(rng.integers(30, 90)))))
        out.append(str(rng.choice(list("ACGT"))) * int(rng.integers(4, 9)))
    return "".join(out)[:n]


def _errors(rng, seq: str, rate: float) -> str:
    """Substitutions, and indels three times likelier inside homopolymers."""
    out = []
    i = 0
    n = len(seq)
    while i < n:
        b = seq[i]
        run = 1
        while i + run < n and seq[i + run] == b:
            run += 1
        hp = run >= 4
        for _ in range(run):
            r = rng.random()
            indel = rate * (0.6 if hp else 0.2)
            if r < indel / 2:
                pass  # deletion
            elif r < indel:
                out.append(b)
                out.append(b if hp else str(rng.choice(list("ACGT"))))
            elif r < indel + rate * 0.5:
                out.append(str(rng.choice([c for c in "ACGT" if c != b])))
            else:
                out.append(b)
        i += run
    return "".join(out)


def build_corpus(out: Path, *, n_reads: int, n_genes: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    genes = []
    for _ in range(n_genes):
        five = _utr(rng, int(rng.integers(40, 200)))
        orf = _orf(rng, int(rng.integers(100, 600)))
        three = _utr(rng, int(rng.integers(100, 800)))
        forms = [five + orf + three]
        if rng.random() < 0.2:  # an alternative first exon
            forms.append(_utr(rng, int(rng.integers(60, 150))) + orf + three)
        if rng.random() < 0.15:  # a haplotype: 2-4 substitutions, 2:1
            alt = list(forms[0])
            for s in rng.choice(len(alt), size=int(rng.integers(2, 5)), replace=False):
                alt[s] = {"A": "C", "C": "G", "G": "T", "T": "A"}[alt[s]]
            forms.append("".join(alt))
        # Sharp internal 5' starts (as the template-switching oligo makes).
        modes = [int(rng.integers(50, max(51, len(forms[0]) // 2))) for _ in range(2)]
        genes.append((forms, modes))
    weights = 1.0 / np.arange(1, n_genes + 1) ** 1.1
    counts = np.maximum(1, np.round(weights / weights.sum() * n_reads)).astype(int)
    rows = []
    for g, ((forms, modes), n) in enumerate(zip(genes, counts)):
        form_p = np.array([2.0] + [1.0] * (len(forms) - 1))
        form_p /= form_p.sum()
        for i in range(int(n)):
            seq = forms[int(rng.choice(len(forms), p=form_p))]
            u = rng.random()
            if u < 0.15:
                seq = seq[int(rng.choice(modes)) :]
            elif u < 0.40:
                seq = seq[int(min(len(seq) - 100, rng.exponential(150))) :]
            jitter = int(rng.integers(-10, 11))
            seq = seq[: len(seq) + jitter] if jitter < 0 else seq
            rows.append(
                (
                    f"g{g:03d}_r{i:05d}",
                    _errors(rng, seq, float(rng.uniform(0.005, 0.02))),
                    float(rng.uniform(12, 35)),
                )
            )
    return _write_demux(out, rows, n_samples=3)


def _write_demux(d: Path, rows, *, n_samples: int) -> Path:
    lead, trail = "GGGGGGGGGG", "CCCCCCCCCC"
    reads = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sequence": [lead + r[1] + trail for r in rows],
            "quality": ["I" * (len(r[1]) + 20) for r in rows],
            "dorado_quality": pa.array([r[2] for r in rows], pa.float32()),
        }
    )
    n = len(rows)
    demux = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "transcript_segment_index": [0] * n,
            "sample_id": pa.array([i % n_samples for i in range(n)], pa.int64()),
            "orientation": ["+"] * n,
            "transcript_start": pa.array([len(lead)] * n, pa.int32()),
            "transcript_end": pa.array([len(lead) + len(r[1]) for r in rows], pa.int32()),
            "score": pa.array([1.0] * n, pa.float32()),
            "is_chimera": [False] * n,
            "status": ["Complete"] * n,
            "is_fragment": [False] * n,
            "artifact": ["none"] * n,
        }
    )
    (d / "reads").mkdir(parents=True, exist_ok=True)
    (d / "read_demux").mkdir(parents=True, exist_ok=True)
    pq.write_table(reads, d / "reads" / "part-00000.parquet")
    pq.write_table(demux, d / "read_demux" / "part-00000.parquet")
    return d


# ── fingerprints ──────────────────────────────────────────────────────


def _digest(table: pa.Table) -> str:
    """Order-independent digest of a table's rows, every column."""
    if table.num_rows:
        table = table.sort_by([(c, "ascending") for c in table.column_names
                               if not pa.types.is_nested(table.schema.field(c).type)])
    blob = json.dumps(table.to_pylist(), sort_keys=True, default=repr).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _read(path: Path) -> pa.Table | None:
    if path.is_file():
        return pq.read_table(path)
    parts = sorted(path.glob("part-*.parquet")) if path.is_dir() else []
    return ds.dataset(parts).to_table() if parts else None


def run_digests(out: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for rd in sorted((out / "rounds").glob("r[0-9]*")):
        for name in ("assignments", "mstep/nodes", "mstep/node_membership",
                     "mstep/templates", "mstep/columns", "mstep/covariance"):
            t = _read(rd / name)
            if t is not None:
                digests[f"{rd.name}/{name}"] = _digest(t)
        tpl = rd / "templates" / "templates.arrow"
        if tpl.exists():
            with pa.memory_map(str(tpl)) as src:
                digests[f"{rd.name}/templates"] = _digest(pa.ipc.open_file(src).read_all())
    for name in ("clusters.parquet", "cluster_membership.parquet", "cluster_nodes.parquet"):
        if (out / name).exists():
            digests[name] = _digest(pq.read_table(out / name))
    return digests


def stage_seconds(out: Path) -> dict[str, float]:
    s: dict[str, float] = {}
    stats = out / "seed" / "stats.json"
    if stats.exists():
        for k, v in json.loads(stats.read_text()).items():
            if k.endswith("_s"):
                s[f"seed/{k[:-2]}"] = float(v)
    for rd in sorted((out / "rounds").glob("r[0-9]*")):
        rj = rd / "round.json"
        if rj.exists():
            for k, v in (json.loads(rj.read_text()).get("seconds") or {}).items():
                s[f"{rd.name}/{k}"] = float(v)
    return s


# ── commands ──────────────────────────────────────────────────────────


def cmd_run(a) -> int:
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        EmParams,
        run_em,
    )

    params = EmParams(
        rounds=a.rounds,
        seeding=a.seeding,
        estep_aligner=a.aligner,
        threads=a.threads,
        mstep_workers=a.workers,
        max_window_length=None,
        min_aa_length=30,
    )
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty", file=sys.stderr)
        return 2
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    run_em(Path(a.demux), out, params=params, report=False)
    prof.disable()
    wall = time.perf_counter() - t0
    buf = io.StringIO()
    st = pstats.Stats(prof, stream=buf)
    top = []
    for sort in ("tottime", "cumulative"):
        st.sort_stats(sort)
        for (file, line, fn), (cc, nc, tt, ct, _) in sorted(
            st.stats.items(), key=lambda kv: -kv[1][2 if sort == "tottime" else 3]
        )[: a.top]:
            top.append({"sort": sort, "where": f"{Path(file).name}:{line}:{fn}",
                        "calls": nc, "tottime": round(tt, 3), "cumtime": round(ct, 3)})
    (out / "bench.json").write_text(json.dumps(
        {"wall_s": round(wall, 2), "stages_s": stage_seconds(out), "top": top,
         "params": {"seeding": a.seeding, "aligner": a.aligner, "rounds": a.rounds,
                    "threads": a.threads, "workers": a.workers}}, indent=1))
    (out / "digests.json").write_text(json.dumps(run_digests(out), indent=1))
    print(f"wall {wall:.1f} s; stages {stage_seconds(out)}")
    return 0


def cmd_compare(a) -> int:
    A, B = Path(a.a), Path(a.b)
    ba, bb = (json.loads((p / "bench.json").read_text()) for p in (A, B))
    print(f"wall: {ba['wall_s']} -> {bb['wall_s']} s")
    for k in sorted(set(ba["stages_s"]) | set(bb["stages_s"])):
        print(f"  {k:28s} {ba['stages_s'].get(k, float('nan')):9.2f} -> "
              f"{bb['stages_s'].get(k, float('nan')):9.2f}")
    da, db = (json.loads((p / "digests.json").read_text()) for p in (A, B))
    diff = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
    print("digests: identical" if not diff else f"digests differ: {diff}")
    return 0 if not diff else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("--reads", type=int, default=5000)
    b.add_argument("--genes", type=int, default=80)
    b.add_argument("--seed", type=int, default=20261115)
    r = sub.add_parser("run")
    r.add_argument("demux")
    r.add_argument("out")
    r.add_argument("--seeding", choices=("kmer", "orf"), default="kmer")
    r.add_argument("--aligner", choices=("native", "edlib", "minimap2"), default="native")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--top", type=int, default=40)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "build":
        d = build_corpus(Path(a.out), n_reads=a.reads, n_genes=a.genes, seed=a.seed)
        print(f"wrote {d}")
        return 0
    return cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
