"""Benchmark the retention-time alignment methods.

Synthetic pairs with a known warp, one factor varied at a time from a
baseline (a curved warp, 0.3 min Gaussian noise, 10 % uniformly wrong
pairs, 10,000 pairs), and, with ``--reference-dir``, the lab's real
EncyclopeDIA fits. Methods: the EncyclopeDIA replay (``encyclopedia``),
its generic form as the jar has it (``greedy``) and with padding,
trimming, a robust bandwidth and renormalization (``greedy+``), the
global path (``global``, the best path; ``global-mean``, the posterior
mean), and the spline
refinements (``global+density``, ``global+modal``, ``greedy+modal``).

Synthetic metrics, against the true warp, over the central 98 % of the
correct pairs' x: mean and max absolute error (min), a failure flag (max
error > 2 min), the smallest slope and the worst inverse round trip on
the fitted domain, how many correct pairs the residual band holds, and
time. The extrapolation family fits on x within the 10th–90th percentile
and scores the held-out ends under ``clamp``, ``linear`` and ``soft``.

Real data (``--reference-dir``, every ``*.rt_fit.txt``): each method
refits the jar's points; reported are the robust spread (MAD) of the
residuals of the pairs the jar called inliers, the largest difference
from the jar's own warp over the inner 90 %, and, fitting on the inner
80 % of library time, the median error and share within 1 min on the
held-out inliers at each end, per extrapolation policy.

Results go to ``--out-dir`` (``synthetic.parquet``, ``extrapolation.parquet``,
``real.parquet``, ``summary.md``, which is also printed). Run from project
root::

    python3 scripts/bench-rt-alignment.py --out-dir /tmp/rt-bench [--reference-dir DIR] [--quick]
"""

from __future__ import annotations

import argparse
import math
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from constellation.core.signal.alignment import (
    GlobalPathOptions,
    GreedyOptions,
    MonotoneAlignment,
    fit_monotone_alignment,
)
from constellation.core.stats.monotone import PiecewiseLinear, TailedCurve
from constellation.massspec.rt.encyclopedia import fit_encyclopedia

# ── warps (minutes) ──

# The lab's predicted-library fit (gas-phase fractions, final pass), at
# its 0, 1, 5, 10, 25, 50, 75, 90, 95, 99 and 100th percentiles.
_GPF_X = [-0.079, 1.479, 2.604, 3.759, 6.913, 11.302, 15.580, 18.965, 20.673, 23.110, 27.413]
_GPF_Y = [10.628, 14.238, 18.980, 24.019, 37.232, 55.662, 73.775, 88.866, 94.451, 100.828, 105.644]
_GPF = PiecewiseLinear(torch.tensor(_GPF_X, dtype=torch.float64), torch.tensor(_GPF_Y, dtype=torch.float64))


def _saturating(x: torch.Tensor) -> torch.Tensor:
    z = (x - 50.0) / 22.0
    return 5.0 + 95.0 * 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


WARPS: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], tuple[float, float]]] = {
    "linear": (lambda x: 5.0 + 1.15 * x, (0.0, 100.0)),
    "curved": (lambda x: 2.0 + 100.0 * ((x + 5.0) / 105.0) ** 0.75, (0.0, 100.0)),
    "kinked": (lambda x: torch.where(x < 60, 0.8 * x + 10.0, 58.0 + 1.8 * (x - 60.0)), (0.0, 100.0)),
    "saturating": (_saturating, (0.0, 100.0)),
    "predicted": (lambda x: _GPF.forward(x), (-0.079, 27.413)),
}


@dataclass(frozen=True)
class Scenario:
    name: str
    warp: str = "curved"
    noise: str = "gaussian"
    sigma: float = 0.3
    outliers: float = 0.10
    structured: str | None = None  # "cluster" | "parallel"
    gap: bool = False
    n: int = 10_000


def _scenarios(quick: bool) -> list[Scenario]:
    base = Scenario("baseline")
    out = [base]
    for w in ("linear", "kinked", "saturating"):
        out.append(Scenario(f"warp={w}", warp=w))
    out.append(Scenario("warp=predicted", warp="predicted", sigma=1.9, outliers=0.02))
    for noise in ("student_t", "heteroscedastic"):
        out.append(Scenario(f"noise={noise}", noise=noise))
    for frac in (0.0, 0.25, 0.5):
        out.append(Scenario(f"outliers={frac}", outliers=frac))
    out.append(Scenario("cluster", structured="cluster"))
    out.append(Scenario("parallel", structured="parallel"))
    out.append(Scenario("gap", gap=True))
    for n in ((200, 1000) if quick else (50, 200, 1000, 100_000)):
        out.append(Scenario(f"n={n}", n=n))
    return out


def _sample(s: Scenario, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairs and a flag for the correct ones."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # torch.distributions samples from the global generator
    warp, (lo, hi) = WARPS[s.warp]
    x = lo + (hi - lo) * torch.distributions.Beta(1.6, 1.4).sample((s.n,)).double()
    if s.gap:
        span = hi - lo
        x = x[(x < lo + 0.40 * span) | (x > lo + 0.55 * span)]
    n = x.numel()
    if s.noise == "gaussian":
        e = s.sigma * torch.randn(n, generator=g, dtype=torch.float64)
    elif s.noise == "student_t":
        e = s.sigma * torch.distributions.StudentT(3.0).sample((n,)).double()
    else:
        e = s.sigma * (0.3 + 1.4 * (x - lo) / (hi - lo)) * torch.randn(n, generator=g, dtype=torch.float64)
    y = warp(x) + e
    good = torch.ones(n, dtype=torch.bool)
    y_lo, y_hi = float(warp(torch.tensor([lo], dtype=torch.float64))), float(warp(torch.tensor([hi], dtype=torch.float64)))
    bad = torch.rand(n, generator=g) < s.outliers
    y[bad] = y_lo + (y_hi - y_lo) * torch.rand(int(bad.sum()), generator=g, dtype=torch.float64)
    good &= ~bad
    if s.structured is not None:
        m = int(0.05 * n) if s.structured == "cluster" else int(0.08 * n)
        if s.structured == "cluster":
            cx = lo + 0.4 * (hi - lo) + 0.03 * (hi - lo) * torch.randn(m, generator=g, dtype=torch.float64)
            cy = warp(cx) + 0.12 * (y_hi - y_lo) + 0.5 * torch.randn(m, generator=g, dtype=torch.float64)
        else:
            cx = lo + (hi - lo) * torch.rand(m, generator=g, dtype=torch.float64)
            cy = warp(cx) + 6.0 + s.sigma * torch.randn(m, generator=g, dtype=torch.float64)
        x, y = torch.cat([x, cx]), torch.cat([y, cy])
        good = torch.cat([good, torch.zeros(m, dtype=torch.bool)])
    order = torch.randperm(x.numel(), generator=g)
    return x[order], y[order], good[order]


# ── methods ──


@dataclass
class Fitted:
    forward: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]
    domain: tuple[float, float]
    band: tuple[float, float]
    alignment: MonotoneAlignment | None = None
    extra: dict = field(default_factory=dict)


def _replay(x, y, policy: str = "clamp") -> Fitted:
    fit = fit_encyclopedia(x, y)
    within = fit.within_band(x, y)
    tailed = TailedCurve.build(fit.curve, policy, x_support=x[within])
    return Fitted(tailed.forward, tailed.inverse, tailed.domain, fit.band)


def _generic(**kw) -> Callable:
    def run(x, y, policy: str = "clamp") -> Fitted:
        a = fit_monotone_alignment(x, y, extrapolation=policy, **kw)
        return Fitted(a.forward, a.inverse, a.domain, a.band, a)
    return run


METHODS: dict[str, Callable] = {
    "encyclopedia": _replay,
    "greedy": _generic(method="greedy"),
    "greedy+": _generic(method="greedy", options=GreedyOptions(bandwidth="robust", trim_fraction=0.001,
                                                                 pad_sigmas=3.0, renormalize=True,
                                                                 deposit="linear")),
    "global": _generic(method="global", refine=None),
    "global-mean": _generic(method="global", refine=None, options=GlobalPathOptions(path="mean")),
    "global+density": _generic(method="global", refine="density"),
    "global+modal": _generic(method="global", refine="modal"),
    "greedy+modal": _generic(method="greedy", refine="modal"),
}


def _timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t0


# ── synthetic ──


def _score(fitted: Fitted, s: Scenario, x, y, good) -> dict:
    warp, _ = WARPS[s.warp]
    xg = x[good]
    a, b = float(torch.quantile(xg, 0.01)), float(torch.quantile(xg, 0.99))
    q = torch.linspace(a, b, 1000, dtype=torch.float64)
    with torch.no_grad():
        err = (fitted.forward(q) - warp(q)).abs()
        lo, hi = fitted.domain
        dq = torch.linspace(lo, hi, 2000, dtype=torch.float64)
        f = fitted.forward(dq)
        slope = torch.diff(f) / torch.diff(dq)
        inv = (fitted.inverse(f) - dq).abs()
        r = y[good] - fitted.forward(xg)
    coverage = float(((r >= fitted.band[0]) & (r <= fitted.band[1])).double().mean())
    return {
        "mae": float(err.mean()),
        "max_error": float(err.max()),
        "failure": bool(float(err.max()) > 2.0),
        "min_slope": float(slope.min()),
        "inverse_error": float(inv.max()),
        "band_coverage": coverage,
    }


def run_synthetic(seeds: int, quick: bool) -> pa.Table:
    rows = []
    for s in _scenarios(quick):
        for seed in range(seeds if s.n < 100_000 else max(1, seeds // 2)):
            x, y, good = _sample(s, 1000 + seed)
            for name, method in METHODS.items():
                if s.n >= 100_000 and name.endswith(("density", "modal")) and quick:
                    continue
                try:
                    fitted, seconds = _timed(method, x, y)
                    metrics = _score(fitted, s, x, y, good)
                except Exception as exc:  # recorded, not fatal: a method may refuse a scenario
                    metrics, seconds = {"error": f"{type(exc).__name__}: {exc}"}, float("nan")
                rows.append({"scenario": s.name, "seed": seed, "method": name, "n": int(x.numel()),
                             "seconds": seconds, **metrics})
                print(f"{s.name:22s} seed={seed} {name:15s} "
                      + (f"mae={metrics['mae']:.4f} max={metrics['max_error']:.3f} t={seconds:.2f}s"
                         if "mae" in metrics else metrics["error"]), file=sys.stderr, flush=True)
    return pa.Table.from_pylist(rows)


def run_extrapolation(seeds: int) -> pa.Table:
    rows = []
    for warp in ("curved", "saturating", "predicted"):
        sigma, outliers = (1.9, 0.02) if warp == "predicted" else (0.3, 0.10)
        s = Scenario(f"tails:{warp}", warp=warp, sigma=sigma, outliers=outliers)
        truth, _ = WARPS[warp]
        for seed in range(seeds):
            x, y, good = _sample(s, 2000 + seed)
            lo, hi = float(torch.quantile(x, 0.10)), float(torch.quantile(x, 0.90))
            inner = (x >= lo) & (x <= hi)
            tails = good & ~inner
            for name in ("encyclopedia", "greedy", "global", "global+modal"):
                for policy in ("clamp", "linear", "soft"):
                    fitted = METHODS[name](x[inner], y[inner], policy)
                    with torch.no_grad():
                        err = (fitted.forward(x[tails]) - truth(x[tails])).abs()
                    rows.append({"scenario": s.name, "seed": seed, "method": name, "policy": policy,
                                 "tail_mae": float(err.mean()), "tail_max": float(err.max())})
    return pa.Table.from_pylist(rows)


# ── real data ──


def run_real(reference_dir: Path) -> pa.Table:
    from constellation.massspec.io.encyclopedia._sidecar import read_tsv

    rows = []
    for path in sorted(reference_dir.glob("*.rt_fit.txt")):
        t = read_tsv(path, ["library", "actual", "warpToactual", "within95p"])
        x = torch.tensor(t.column("library").to_pylist(), dtype=torch.float64)
        y = torch.tensor(t.column("actual").to_pylist(), dtype=torch.float64)
        jar = torch.tensor(t.column("warpToactual").to_pylist(), dtype=torch.float64)
        inlier = torch.tensor(t.column("within95p").to_pylist())
        a, b = float(torch.quantile(x, 0.05)), float(torch.quantile(x, 0.95))
        middle = (x >= a) & (x <= b)
        lo, hi = float(torch.quantile(x, 0.10)), float(torch.quantile(x, 0.90))
        inner = (x >= lo) & (x <= hi)
        held = inlier & ~inner
        for name in ("encyclopedia", "greedy", "greedy+", "global", "global+modal"):
            fitted, seconds = _timed(METHODS[name], x, y)
            with torch.no_grad():
                r = (y - fitted.forward(x))[inlier]
                mad = float(1.4826 * (r - r.median()).abs().median())
                vs_jar = float((fitted.forward(x[middle]) - jar[middle]).abs().max())
            row = {"file": path.name, "method": name, "n": int(x.numel()), "seconds": seconds,
                   "inlier_mad": mad, "max_vs_jar_inner90": vs_jar}
            for policy in ("clamp", "linear", "soft"):
                part = METHODS[name](x[inner], y[inner], policy)
                with torch.no_grad():
                    e = (part.forward(x[held]) - y[held]).abs()
                row[f"holdout_median_{policy}"] = float(e.median())
                row[f"holdout_within1_{policy}"] = float((e <= 1.0).double().mean())
            rows.append(row)
            print(f"{path.name[:40]} {name:13s} mad={mad:.3f} vs_jar={vs_jar:.3f} t={seconds:.2f}s",
                  file=sys.stderr, flush=True)
    return pa.Table.from_pylist(rows)


# ── summary ──


def _median_table(table: pa.Table, keys: list[str], values: list[str]) -> list[dict]:
    rows = table.to_pylist()
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if any(r.get(v) is None for v in values):
            continue
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, members in groups.items():
        rec = dict(zip(keys, key))
        for v in values:
            vals = [float(m[v]) for m in members]
            rec[v] = float(np.median(vals)) if v != "failure" else float(np.mean(vals))
        out.append(rec)
    return out


def _markdown(records: list[dict], columns: list[str], fmt: dict[str, str]) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = ""
    for r in records:
        body += "| " + " | ".join(format(r[c], fmt.get(c, "")) if isinstance(r[c], float) else str(r[c]) for c in columns) + " |\n"
    return head + body


def summarize(synthetic: pa.Table | None, extrapolation: pa.Table | None, real: pa.Table | None) -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    lines = [
        "# RT alignment benchmark",
        "",
        f"git {sha} (+ working tree), torch {torch.__version__}, {platform.processor() or platform.machine()}, "
        f"{torch.get_num_threads()} threads, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB",
        "",
    ]
    if synthetic is not None and synthetic.num_rows:
        lines += ["## Synthetic (median over seeds; failure = share of seeds with max error > 2 min)", ""]
        recs = _median_table(synthetic, ["scenario", "method"],
                             ["mae", "max_error", "failure", "min_slope", "band_coverage", "seconds"])
        lines.append(_markdown(recs, ["scenario", "method", "mae", "max_error", "failure", "min_slope",
                                      "band_coverage", "seconds"],
                               {"mae": ".4f", "max_error": ".3f", "failure": ".2f", "min_slope": ".3g",
                                "band_coverage": ".3f", "seconds": ".2f"}))
    if extrapolation is not None and extrapolation.num_rows:
        lines += ["## Extrapolation (fit on the 10th–90th percentile of x; error on the held-out correct pairs)", ""]
        recs = _median_table(extrapolation, ["scenario", "method", "policy"], ["tail_mae", "tail_max"])
        lines.append(_markdown(recs, ["scenario", "method", "policy", "tail_mae", "tail_max"],
                               {"tail_mae": ".3f", "tail_max": ".3f"}))
    if real is not None and real.num_rows:
        lines += ["## Real EncyclopeDIA fits (minutes)", ""]
        cols = ["file", "method", "inlier_mad", "max_vs_jar_inner90", "holdout_median_clamp",
                "holdout_median_linear", "holdout_median_soft", "holdout_within1_clamp",
                "holdout_within1_soft", "seconds"]
        recs = [{**r, "file": r["file"].split(".dia")[0][-24:] + ("/first" if ".first." in r["file"] else "/final")}
                for r in real.to_pylist()]
        lines.append(_markdown(recs, cols, {c: ".3f" for c in cols if c not in ("file", "method")}))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=None, help="a directory of *.rt_fit.txt (read only)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="fewer sizes, no refinement at 100,000 pairs")
    parser.add_argument("--only", choices=["synthetic", "extrapolation", "real"], action="append")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts = set(args.only or ["synthetic", "extrapolation", "real"])
    synthetic = extrapolation = real = None
    if "synthetic" in parts:
        synthetic = run_synthetic(args.seeds, args.quick)
        pq.write_table(synthetic, args.out_dir / "synthetic.parquet")
    if "extrapolation" in parts:
        extrapolation = run_extrapolation(args.seeds)
        pq.write_table(extrapolation, args.out_dir / "extrapolation.parquet")
    if "real" in parts and args.reference_dir is not None:
        real = run_real(args.reference_dir)
        pq.write_table(real, args.out_dir / "real.parquet")
    text = summarize(synthetic, extrapolation, real)
    (args.out_dir / "summary.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
