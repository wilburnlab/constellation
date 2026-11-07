"""Benchmark the sweep (`massspec.sweep`) on real acquisitions.

Four subcommands; each reads only its inputs and writes only ``--out``.

``targets``  makes an XIC target table: from an EncyclopeDIA library
             (``--library``, its precursors at their library retention
             times, optionally one m/z range and a sample) or from FASTA
             files through ``massspec.library.search_space`` (``--fasta``;
             ``--heavy-fasta`` labelled ``<13C><15N>``). ``--decoys`` adds
             EncyclopeDIA-style decoys (first and last residue kept, the
             interior reversed, modifications moving with their residues)
             at the same charges, flagged ``is_decoy``.
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

# ── targets ───────────────────────────────────────────────────────────


def _cmd_targets(args) -> None:
    """Targets from a library (its precursors) or from FASTA files through
    the search space, the package's one enumerator: decoys and labelled
    forms are made there, so a modified or labelled target has its decoy
    as an unmodified light one does."""
    from constellation.massspec.library.decoys import DecoySet
    from constellation.massspec.library.search_space import (
        Digestion,
        ModRule,
        SearchSpaceSettings,
        build_search_space,
    )
    from constellation.massspec.quant.targets import targets_from_library

    if args.library:
        from constellation.massspec.io.encyclopedia import read_encyclopedia

        lib = read_encyclopedia(args.library, annotate_fragments=False, psms=False, sidecars=False).library
        table = targets_from_library(lib)
        if args.mz_range:
            mz = table.column("precursor_mz")
            table = table.filter(pc.and_(pc.greater_equal(mz, args.mz_range[0]), pc.less_equal(mz, args.mz_range[1])))
        if args.decoys:
            table = _with_library_decoys(table)
    else:
        fastas = list(args.fasta or []) + list(args.heavy_fasta or [])
        settings = SearchSpaceSettings(
            digestion=Digestion(missed_cleavages=args.missed_cleavages),
            rules=(ModRule.parse("Oxidation@M", "variable"), ModRule.parse("Carbamidomethyl@C", "variable")),
            max_variable=2,
            labels=tuple((f, ("13C", "15N")) for f in args.heavy_fasta or []),
            charges=tuple(args.charges),
            min_mz=args.min_mz,
            max_mz=args.max_mz,
            decoys=(DecoySet(),) if args.decoys else (),
        )
        table = targets_from_library(build_search_space(fastas, settings))
    if args.sample and table.num_rows > args.sample:
        rng = np.random.default_rng(args.seed)
        table = table.take(np.sort(rng.choice(table.num_rows, args.sample, replace=False)))
    pq.write_table(table, args.out)
    n_decoy = pc.sum(pc.fill_null(table["is_decoy"], False)).as_py() or 0
    print(f"{table.num_rows} targets ({n_decoy} decoys) → {args.out}")


def _with_library_decoys(targets: pa.Table) -> pa.Table:
    """A library's targets with a decoy each, from the search space's decoy
    engine: same composition, so the same precursor m/z and charge."""
    from constellation.core.sequence.proforma import format_proforma, parse_proforma
    from constellation.massspec.library.decoys import DecoySet, il_key, peptide_decoys
    from constellation.massspec.library.ids import canonical_peptide_key, content_id

    rows = targets.to_pylist()
    forms = [parse_proforma(r["modified_sequence"]) for r in rows]
    taken = {il_key(f.sequence) for f in forms}
    out = peptide_decoys(forms, DecoySet(), taken=taken, key=lambda f: canonical_peptide_key(format_proforma(f)))
    decoys = []
    for i, form in zip(out.target, out.decoy, strict=True):
        key = canonical_peptide_key(format_proforma(form))
        r = rows[i]
        decoys.append(dict(r, target_id=content_id(f"{key}/{r['precursor_charge']}"), modified_sequence=key,
                           precursor_id=None, peptide_id=None, is_decoy=True))  # fmt: skip
    return pa.concat_tables([targets, pa.Table.from_pylist(decoys, schema=targets.schema)])


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
        decoy = plan.targets.column("is_decoy").fill_null(False).to_numpy(zero_copy_only=False)[batch.queries]
        has = plan.fragments.counts()[batch.queries] > 0
        pairs = batch.eligible(2, np.unique(batch.evidence.query))  # a query without evidence has no run
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


def _grid_totals(rows: list[dict]) -> list[dict]:
    """One row per gate configuration. A sweep that went to disk reduced its
    queries in buckets, each with its own grid, so a configuration's counts
    are the sum of its buckets' rows."""
    totals: dict[tuple[int, int], dict] = {}
    for row in rows:
        key = (row["min_fragments"], row["min_consecutive"])
        if key not in totals:
            totals[key] = dict(row)
        else:
            for k in ("targets_passed", "decoys_passed", "targets", "decoys"):
                totals[key][k] += row[k]
    return list(totals.values())


def _cmd_gate(args) -> None:
    from constellation.massspec.sweep import compile_queries
    from constellation.massspec.sweep.engine import SweepSettings, sweep
    from constellation.massspec.sweep.evidence import EvidenceGate

    if args.targets.endswith(".parquet"):
        targets = pq.read_table(args.targets)
    else:
        from constellation.massspec.quant.targets import load_targets

        targets = load_targets(args.targets)
    abundance, channels = None, None
    if args.fragment_library or args.top_fragments:
        from constellation.massspec.library.fragment_abundance import LibraryFragments
        from constellation.massspec.sweep import ChannelSettings

        abundance = LibraryFragments.from_directory(args.fragment_library or args.targets)
        coverage = abundance.coverage or {}
        channels = ChannelSettings(top_fragments=args.top_fragments,
                                   max_fragment_charge=int(coverage.get("max_fragment_charge") or 2))  # fmt: skip
    kw = {} if abundance is None else dict(channels=channels, fragment_abundance=abundance)
    plan = compile_queries(targets, levels=[2], rt_window=args.rt_window, **kw)
    report = []
    for top in args.top_peaks:
        top_peaks = top or None
        t0 = time.perf_counter()
        settings = SweepSettings(tolerances={2: (args.ms2_tolerance, args.ms2_unit)},
                                 gate=EvidenceGate(min_fragments=1, top_peaks=top_peaks))  # fmt: skip
        result = sweep(args.bundle, plan, settings,
                       reducers=[_GridReducer(args.min_fragments, args.min_consecutive, top_peaks)])  # fmt: skip
        grid = _grid_totals(result.table("gate_grid").to_pylist())
        for row in grid:
            row["top_peaks"] = top_peaks
            row["sweep_s"] = round(time.perf_counter() - t0, 2)
            print(json.dumps(row), flush=True)
        report.extend(grid)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


# ── compiling queries ─────────────────────────────────────────────────


def _cmd_compile(args) -> None:
    """Time compile_queries on a target file, level set by level set."""
    from constellation.massspec.sweep import compile_queries

    targets = pq.read_table(args.targets)
    for levels in args.levels:
        lv = [int(x) for x in levels.split(",")]
        t0 = time.perf_counter()
        plan = compile_queries(targets, levels=lv)
        dt = time.perf_counter() - t0
        n_channels = sum(c.n for c in (plan.isotopes, plan.fragments) if c is not None)
        print(json.dumps({"levels": lv, "targets": targets.num_rows, "channels": n_channels, "seconds": round(dt, 3),
                          "ms_per_target": round(1e3 * dt / max(targets.num_rows, 1), 4)}), flush=True)  # fmt: skip


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
    g.add_argument("--fragment-library", default=None,
                   help="Take fragment channels' abundances from this library (default: --targets).")
    g.add_argument("--top-fragments", type=int, default=None,
                   help="Keep each target's N most abundant fragment ions (needs a fragment library).")
    tolerances(g)
    g.add_argument("--out", required=True)
    g.set_defaults(func=_cmd_gate)

    k = subs.add_parser("compile")
    k.add_argument("--targets", required=True)
    k.add_argument("--levels", nargs="+", default=["1", "2", "1,2"],
                   help="Level sets to compile, each comma-separated (default: 1 2 1,2).")
    k.set_defaults(func=_cmd_compile)

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
