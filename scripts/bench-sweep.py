"""Benchmark the sweep (`massspec.sweep`) on real acquisitions.

Four subcommands; each reads only its inputs and writes only ``--out``.

``targets``  makes an XIC target table: from an EncyclopeDIA library
             (``--library``, its precursors at their library retention
             times, optionally one m/z range and a sample) or from FASTA
             files (``--fasta``, digested; ``--heavy-fasta`` digested and
             labelled ``<13C><15N>``). ``--decoys`` adds each peptide
             reversed (C-terminal residue kept) at the same charge, with
             ``target_id`` offset by ``DECOY_OFFSET``.
``time``     sweeps one bundle for those targets under each configuration
             asked for (workers, levels, rule, kernel, RT window), each in
             a fresh process so its peak memory is its own, and records
             wall time, peak RSS and match counts as JSON lines.
``gate``     counts the evidence of every target once per ``--top-peaks``
             setting and reports, over a grid of ``min_fragments`` and
             ``min_consecutive``, how many targets and how many decoys pass:
             what the gate's defaults should be measured on.
``compare-old``  runs the frozen target-by-target extractor and the sweep
             on the same scans of a bundle (``--scans LO HI``) and reports
             both times and whether their tables are identical.

Run from the project root, for example::

    python3 scripts/bench-sweep.py targets --library L.elib --mz-range 900 1000 --decoys --out t.parquet
    python3 scripts/bench-sweep.py time --bundle RUN --targets t.parquet --workers 1 4 --out time.jsonl
    python3 scripts/bench-sweep.py gate --bundle RUN --targets t.parquet --top-peaks 0 150 50 --out gate.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DECOY_OFFSET = 1 << 40


# ── targets ───────────────────────────────────────────────────────────


def _reverse(modseq: str) -> str | None:
    """A peptide reversed with its C-terminal residue kept; None for one
    that carries modifications (reversing ProForma text would misplace them)."""
    if "[" in modseq or "<" in modseq or len(modseq) < 3:
        return None
    return modseq[-2::-1] + modseq[-1]


def _cmd_targets(args) -> None:
    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.peptide.mz import precursor_mz
    from constellation.massspec.quant.schemas import XIC_TARGET_TABLE

    rows: list[dict] = []
    if args.library:
        from constellation.massspec.io.encyclopedia import read_encyclopedia
        from constellation.massspec.quant.targets import targets_from_library

        lib = read_encyclopedia(args.library, annotate_fragments=False, psms=False, sidecars=False).library
        t = targets_from_library(lib)
        if args.mz_range:
            mz = t.column("precursor_mz")
            t = t.filter(pc.and_(pc.greater_equal(mz, args.mz_range[0]), pc.less_equal(mz, args.mz_range[1])))
        rows = t.to_pylist()
    for path, heavy in [(f, False) for f in args.fasta or []] + [(f, True) for f in args.heavy_fasta or []]:
        from constellation.massspec.library.digest import precursors_from_fasta

        specs = precursors_from_fasta(
            path, missed_cleavages=args.missed_cleavages, charges=tuple(args.charges),
            variable_mods={"M": "UNIMOD:35", "C": "UNIMOD:4"}, max_variable_mods=2, min_mz=args.min_mz, max_mz=args.max_mz,
        )  # fmt: skip
        for s in specs:
            modseq = f"<13C><15N>{s.modified_sequence}" if heavy else s.modified_sequence
            pmz = precursor_mz(parse_proforma(modseq), charge=s.charge) if heavy else s.precursor_mz
            rows.append(dict(target_id=len(rows) + 1, modified_sequence=modseq, precursor_charge=s.charge,
                             precursor_mz=pmz))  # fmt: skip
    if args.sample and len(rows) > args.sample:
        rng = np.random.default_rng(args.seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.sample, replace=False))]
    if args.decoys:
        decoys = []
        for r in rows:
            rev = _reverse(r["modified_sequence"] or "")
            if rev is None:
                continue
            decoys.append(dict(r, target_id=r["target_id"] + DECOY_OFFSET, modified_sequence=rev,
                               precursor_mz=precursor_mz(parse_proforma(rev), charge=r["precursor_charge"]),
                               precursor_id=None, peptide_id=None))  # fmt: skip
        rows += decoys
    table = pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)
    pq.write_table(table, args.out)
    n_decoy = pc.sum(pc.greater_equal(table["target_id"], DECOY_OFFSET)).as_py() or 0
    print(f"{table.num_rows} targets ({n_decoy} decoys) → {args.out}")


# ── timing ────────────────────────────────────────────────────────────


def _tolerances(args) -> dict[int, tuple[float, str]]:
    return {1: (args.ms1_tolerance, args.ms1_unit), 2: (args.ms2_tolerance, args.ms2_unit)}


def _cmd_run_one(args) -> None:
    """One configuration, in this process; prints one JSON line."""
    import torch

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    from constellation.massspec.quant.chromatogram import XicTraceReducer
    from constellation.massspec.sweep import compile_queries
    from constellation.massspec.sweep.engine import SweepSettings, sweep
    from constellation.massspec.sweep.rules import rule_named

    targets = pq.read_table(args.targets)
    t0 = time.perf_counter()
    plan = compile_queries(targets, levels=args.levels, rt_window=args.rt_window)
    t_compile = time.perf_counter() - t0
    tol = {lv: _tolerances(args)[min(lv, 2)] for lv in args.levels}
    settings = SweepSettings(tolerances=tol, rule=rule_named(args.rule), kernel=args.kernel,
                             block_rows=args.block_rows)  # fmt: skip
    work = Path(args.work_dir) / f"w{args.workers}-{args.kernel}-{args.rule}" if args.workers > 1 else None
    t0 = time.perf_counter()
    result = sweep(args.bundle, plan, settings, reducers=[XicTraceReducer()], workers=args.workers,
                   work_dir=work)  # fmt: skip
    t_sweep = time.perf_counter() - t0
    rows = (result.table("xic_trace").num_rows if not result.parts
            else sum(pq.ParquetFile(p).metadata.num_rows for p in result.parts["xic_trace"]))  # fmt: skip
    print(json.dumps({
        "bundle": Path(args.bundle).name, "targets": Path(args.targets).name, "queries": plan.n_queries,
        "isotope_channels": 0 if plan.isotopes is None else plan.isotopes.n,
        "fragment_channels": 0 if plan.fragments is None else plan.fragments.n,
        "levels": args.levels, "rule": args.rule, "kernel": args.kernel, "workers": args.workers,
        "rt_window": args.rt_window, "block_rows": args.block_rows, "torch_threads": torch.get_num_threads(),
        "compile_s": round(t_compile, 3), "sweep_s": round(t_sweep, 3), "rows": rows,
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
        "children_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 3),
    }), flush=True)  # fmt: skip


def _group_rss_gb(pgid: int) -> float:
    """Resident memory of every process in a process group, from /proc."""
    total = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / "stat").read_text()
            if int(stat.rsplit(")", 1)[1].split()[2]) != pgid:
                continue
            for line in (proc / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except (OSError, ValueError, IndexError):
            continue
    return total / 1e6


def _run_guarded(cmd: list[str], max_rss_gb: float) -> tuple[int, str, str, float]:
    """Run ``cmd`` in a session of its own and kill the whole group if its
    resident memory passes ``max_rss_gb``, so one configuration cannot take
    the machine down. Returns (exit status, stdout, stderr, group peak GB)."""
    import os
    import signal
    import tempfile

    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True, start_new_session=True)
        peak = 0.0
        while proc.poll() is None:
            rss = _group_rss_gb(proc.pid)
            peak = max(peak, rss)
            if rss > max_rss_gb:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                err.write(f"\nkilled: process group passed {max_rss_gb} GB resident ({rss:.1f} GB)")
                break
            time.sleep(0.5)
        out.seek(0)
        err.seek(0)
        return proc.returncode, out.read(), err.read(), round(peak, 2)


def _cmd_time(args) -> None:
    out = open(args.out, "a")
    for workers, kernel, rule, rt_window in itertools.product(args.workers, args.kernels, args.rules,
                                                              args.rt_windows):  # fmt: skip
        cmd = [sys.executable, __file__, "run-one", "--bundle", args.bundle, "--targets", args.targets,
               "--levels", *map(str, args.levels), "--rule", rule, "--kernel", kernel,
               "--workers", str(workers), "--work-dir", args.work_dir, "--block-rows", str(args.block_rows),
               "--ms1-tolerance", str(args.ms1_tolerance), "--ms1-unit", args.ms1_unit,
               "--ms2-tolerance", str(args.ms2_tolerance), "--ms2-unit", args.ms2_unit,
               "--torch-threads", str(args.torch_threads)]  # fmt: skip
        if rt_window > 0:
            cmd += ["--rt-window", str(rt_window)]
        code, stdout, stderr, peak = _run_guarded(cmd, args.max_rss_gb)
        if code == 0:
            row = json.loads(stdout.strip().splitlines()[-1])
            row["group_peak_rss_gb"] = peak
            line = json.dumps(row)
        else:
            line = json.dumps({"error": stderr[-2000:], "returncode": code, "group_peak_rss_gb": peak,
                               "workers": workers, "kernel": kernel, "rule": rule, "rt_window": rt_window})  # fmt: skip
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()
        time.sleep(args.pause_s)


# ── gate grid ─────────────────────────────────────────────────────────


class _GridReducer:
    """Decides every gate of a grid from one sweep's evidence (counted with
    the lowest threshold), and counts passing targets and decoys."""

    name = "gate_grid"
    gated = False
    schema = pa.schema([("min_fragments", pa.int32()), ("min_consecutive", pa.int32()),
                        ("targets_passed", pa.int64()), ("decoys_passed", pa.int64()),
                        ("targets", pa.int64()), ("decoys", pa.int64())])  # fmt: skip

    def __init__(self, ks, runs, top_peaks):
        self.ks, self.runs, self.top_peaks = ks, runs, top_peaks

    def reduce(self, batch):
        from constellation.massspec.sweep.evidence import EvidenceGate, decide

        plan = batch.plan
        decoy = plan.targets.column("target_id").to_numpy()[batch.queries] >= DECOY_OFFSET
        has = plan.fragments.counts()[batch.queries] > 0
        pairs = batch.eligible(2)
        rows = []
        for k, run in itertools.product(self.ks, self.runs):
            gate = EvidenceGate(min_fragments=k, min_consecutive=run, top_peaks=self.top_peaks)
            d = decide(gate, batch.queries, has, lambda: pairs, lambda: (np.empty(0, np.int64),) * 2,
                       batch.evidence, batch.scans)  # fmt: skip
            rows.append(dict(min_fragments=k, min_consecutive=run,
                             targets_passed=int((d.passed & ~decoy).sum()),
                             decoys_passed=int((d.passed & decoy).sum()),
                             targets=int((~decoy).sum()), decoys=int(decoy.sum())))  # fmt: skip
        return pa.Table.from_pylist(rows, schema=self.schema)


def _cmd_gate(args) -> None:
    from constellation.massspec.sweep import compile_queries
    from constellation.massspec.sweep.engine import SweepSettings, sweep
    from constellation.massspec.sweep.evidence import EvidenceGate

    targets = pq.read_table(args.targets)
    plan = compile_queries(targets, levels=[2], rt_window=args.rt_window)
    report = []
    for top in args.top_peaks:
        top_peaks = top or None
        t0 = time.perf_counter()
        settings = SweepSettings(tolerances={2: (args.ms2_tolerance, args.ms2_unit)},
                                 gate=EvidenceGate(min_fragments=1, top_peaks=top_peaks))  # fmt: skip
        result = sweep(args.bundle, plan, settings,
                       reducers=[_GridReducer(args.min_fragments, args.min_consecutive, top_peaks)])  # fmt: skip
        grid = result.table("gate_grid").to_pylist()
        for row in grid:
            row["top_peaks"] = top_peaks
            row["sweep_s"] = round(time.perf_counter() - t0, 2)
            print(json.dumps(row), flush=True)
        report.extend(grid)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


# ── against the old extractor ─────────────────────────────────────────


def _cmd_compare_old(args) -> None:
    import pyarrow.dataset as ds

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    import _xic_scan_major_oracle as oracle

    from constellation.massspec.bundle import peaks_dataset
    from constellation.massspec.quant.chromatogram import extract_xic_scan_major

    lo, hi = args.scans
    peaks = peaks_dataset(args.bundle).to_table(filter=(ds.field("scan") >= lo) & (ds.field("scan") < hi))
    targets = pq.read_table(args.targets)
    report = []
    for level in args.levels:
        tol, unit = _tolerances(args)[level]
        kw = dict(acquisition_id=0, level=level, tolerance=tol, tolerance_unit=unit, rt_window=args.rt_window)
        t0 = time.perf_counter()
        old = oracle.extract_xic_scan_major(peaks, targets, **kw)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        new = extract_xic_scan_major(peaks, targets, **kw)
        t_new = time.perf_counter() - t0
        try:
            oracle.assert_traces_identical(new, old)
            same, why = True, ""
        except AssertionError as exc:
            same, why = False, str(exc)[:500]
        row = dict(level=level, peak_rows=peaks.num_rows, targets=targets.num_rows, rows=old.num_rows,
                   old_s=round(t_old, 2), new_s=round(t_new, 2), identical=same, difference=why)  # fmt: skip
        print(json.dumps(row), flush=True)
        report.append(row)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subs = p.add_subparsers(dest="command", required=True)

    def tolerances(sp):
        sp.add_argument("--ms1-tolerance", type=float, default=0.5)
        sp.add_argument("--ms1-unit", choices=("ppm", "Da"), default="Da")
        sp.add_argument("--ms2-tolerance", type=float, default=0.35)
        sp.add_argument("--ms2-unit", choices=("ppm", "Da"), default="Da")

    t = subs.add_parser("targets")
    t.add_argument("--library")
    t.add_argument("--mz-range", type=float, nargs=2)
    t.add_argument("--fasta", nargs="*")
    t.add_argument("--heavy-fasta", nargs="*")
    t.add_argument("--missed-cleavages", type=int, default=2)
    t.add_argument("--charges", type=int, nargs="*", default=[2, 3, 4])
    t.add_argument("--min-mz", type=float, default=350.0)
    t.add_argument("--max-mz", type=float, default=1250.0)
    t.add_argument("--sample", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--decoys", action="store_true")
    t.add_argument("--out", required=True)
    t.set_defaults(func=_cmd_targets)

    for name, func in (("time", _cmd_time), ("run-one", _cmd_run_one)):
        r = subs.add_parser(name)
        r.add_argument("--bundle", required=True)
        r.add_argument("--targets", required=True)
        r.add_argument("--levels", type=int, nargs="+", default=[1, 2])
        r.add_argument("--block-rows", type=int, default=1 << 20)
        r.add_argument("--work-dir", default="/tmp/sweep-bench-work")
        r.add_argument("--torch-threads", type=int, default=0,
                       help="Threads for torch in a one-process run (default: torch's own).")
        tolerances(r)
        if name == "time":
            r.add_argument("--workers", type=int, nargs="+", default=[1])
            r.add_argument("--kernels", nargs="+", default=["auto"], choices=["auto", "channel", "point"])
            r.add_argument("--rules", nargs="+", default=["nearest"])
            r.add_argument("--rt-windows", type=float, nargs="+", default=[0.0])
            r.add_argument("--max-rss-gb", type=float, default=20.0,
                           help="Kill a configuration whose processes together pass this (default 20).")
            r.add_argument("--pause-s", type=float, default=0.0,
                           help="Rest between configurations, to keep a long run's load moderate.")
            r.add_argument("--out", required=True)
        else:
            r.add_argument("--workers", type=int, default=1)
            r.add_argument("--kernel", default="auto", choices=["auto", "channel", "point"])
            r.add_argument("--rule", default="nearest")
            r.add_argument("--rt-window", type=float, default=None)
        r.set_defaults(func=func)

    g = subs.add_parser("gate")
    g.add_argument("--bundle", required=True)
    g.add_argument("--targets", required=True)
    g.add_argument("--rt-window", type=float, default=None)
    g.add_argument("--top-peaks", type=int, nargs="+", default=[0, 150, 50, 20])
    g.add_argument("--min-fragments", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    g.add_argument("--min-consecutive", type=int, nargs="+", default=[1, 2, 3, 5])
    tolerances(g)
    g.add_argument("--out", required=True)
    g.set_defaults(func=_cmd_gate)

    c = subs.add_parser("compare-old")
    c.add_argument("--bundle", required=True)
    c.add_argument("--targets", required=True)
    c.add_argument("--scans", type=int, nargs=2, required=True)
    c.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    c.add_argument("--rt-window", type=float, default=None)
    tolerances(c)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_compare_old)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
