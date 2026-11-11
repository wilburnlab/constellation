"""Benchmark scorer v1 (massspec.search.score) on a GPF fraction against
EncyclopeDIA's identifications of the same file.

A subset directory holds ``targets.parquet`` (XIC targets with EncyclopeDIA's
apex and boundaries for the precursors it found, ``ency_rt`` /
``ency_rt_start`` / ``ency_rt_stop``, and library decoys sampled to match
them) and ``library/`` (the predicted library they come from).

    bench-scorer-v1.py run    --subset DIR --bundle BUNDLE --out OUT --config NAME [--workers 4]
    bench-scorer-v1.py report OUT [OUT ...] --subset DIR
    bench-scorer-v1.py grid   --subset DIR [--subset DIR ...] --bundle BUNDLE --out ROOT [--workers 4]

``run`` scores the subset with one named configuration (``CONFIGS``) and
writes ``candidates.parquet`` and ``run.json`` (wall time, peak memory, the
configuration). ``report`` prints, per run: the AUC of EncyclopeDIA's found
targets against the decoys for several of the candidates' columns, how far
the chosen apex is from EncyclopeDIA's and how often it is inside its
boundaries, and a crude target-decoy count at 1 % and 5 % (sampling, not an
FDR: the decoys are a stratified sample and the targets mostly the jar's
own identifications). ``grid`` runs every configuration on every subset,
one factor at a time from ``base``, then reports.
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
import pyarrow.parquet as pq


def configs():
    from constellation.core.signal.peaks import (
        DetectorBounds,
        Fraction,
        GaussianSmooth,
        MatchedFilter,
        RobustZ,
        SavitzkyGolay,
    )
    from constellation.massspec.search.score import ScorerSettings, with_picker

    base = ScorerSettings()
    return {
        "base": base,
        "h1": replace(base, half_window=1),
        "h2": replace(base, half_window=2),
        "neg": replace(base, transform="neg"),
        "minobs2": replace(base, min_observed=2),
        "minobs4": replace(base, min_observed=4),
        "smooth_none": with_picker(base, smooth=None),
        "smooth_wide": with_picker(base, smooth=GaussianSmooth(2.0)),
        "smooth_sg": with_picker(base, smooth=SavitzkyGolay(5, 2)),
        "matched": with_picker(base, detect=MatchedFilter(widths=(1.5, 3.0), threshold=2.0)),
        "robust_z": with_picker(base, detect=RobustZ(k=3.0, k_low=1.0, window=31), boundaries=DetectorBounds()),
        "robust_z_walk": with_picker(base, detect=RobustZ(k=3.0, k_low=1.0, window=31)),
        "bound_fraction": with_picker(base, boundaries=Fraction(0.05)),
    }


def _channels():
    from constellation.massspec.peptide.channels import ChannelSettings

    return ChannelSettings.for_scoring() if hasattr(ChannelSettings, "for_scoring") else ChannelSettings(top_fragments=10)


def run(subset: Path, bundle: Path, out: Path, config: str, workers: int) -> None:
    from constellation.massspec.library.fragment_abundance import LibraryFragments
    from constellation.massspec.search.score import score_acquisition

    settings = configs()[config]
    targets = pq.read_table(subset / "targets.parquet")
    shares = LibraryFragments.from_directory(subset / "library")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = score_acquisition(bundle, targets, fragment_abundance=shares, tolerance=(0.35, "Da"), channels=_channels(),
                               settings=settings, workers=workers, work_dir=out / "work")  # fmt: skip
    result.write("candidates", out / "candidates.parquet")
    wall = time.perf_counter() - t0
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    record = {"config": config, "subset": str(subset), "bundle": str(bundle), "wall_s": round(wall, 1),
              "peak_rss_gb": round(peak / 2**20, 2), "settings": settings.as_dict()}  # fmt: skip
    (out / "run.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps({k: record[k] for k in ("config", "wall_s", "peak_rss_gb")}))


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC, missing values ranked lowest."""
    pos = np.where(np.isnan(pos), -np.inf, pos)
    neg = np.where(np.isnan(neg), -np.inf, neg)
    both = np.concatenate([pos, neg])
    order = np.argsort(both, kind="mergesort")
    ranks = np.empty(both.size)
    ranks[order] = np.arange(1, both.size + 1)
    # Average ties.
    _, inv, counts = np.unique(both, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    a = pos.size
    return float((ranks[:a].sum() - a * (a + 1) / 2) / max(a * neg.size, 1))


def _tdc_counts(score: np.ndarray, decoy: np.ndarray, levels=(0.01, 0.05)) -> list[int]:
    s = np.where(np.isnan(score), -np.inf, score)
    order = np.argsort(-s, kind="mergesort")
    d = np.cumsum(decoy[order])
    t = np.cumsum(~decoy[order])
    fdr = (d + 1) / np.maximum(t, 1)
    out = []
    for level in levels:
        ok = np.flatnonzero(fdr <= level)
        out.append(int(t[ok.max()]) if ok.size else 0)
    return out


def report(outs: list[Path], subset: Path | None) -> None:
    rows = []
    for out in outs:
        meta = json.loads((out / "run.json").read_text())
        sub = Path(subset or meta["subset"])
        targets = pq.read_table(sub / "targets.parquet").select(["target_id", "ency_rt", "ency_rt_start", "ency_rt_stop"])
        cand = pq.read_table(out / "candidates.parquet")
        chosen = cand.filter(cand.column("chosen"))
        joined = chosen.join(targets, "target_id", join_type="left outer")
        get = lambda c: np.asarray(joined.column(c).to_pylist(), dtype=float)  # noqa: E731
        decoy = np.asarray(joined.column("is_decoy").fill_null(False).to_pylist(), bool)
        ency = get("ency_rt")
        found = ~decoy & ~np.isnan(ency)
        rec = {"run": out.name, "config": meta["config"], "subset": Path(meta["subset"]).name, "wall_s": meta["wall_s"],
               "rss_gb": meta["peak_rss_gb"], "found": int(found.sum()), "decoys": int(decoy.sum())}  # fmt: skip
        for name, col, sign in (("score", "score", 1), ("detector", "detector_score", 1),
                                ("kl_smooth", "kl_apex_smoothed", -1), ("kl_ext", "kl_integrated", -1),
                                ("coel", "n_coeluting", 1)):  # fmt: skip
            v = sign * get(col)
            rec[f"auc_{name}"] = round(_auc(v[found], v[decoy]), 3)
        rt = get("rt_apex")
        d = np.abs(rt - ency)[found]
        rec["apex_within_30s"] = round(float(np.mean(d <= 30)), 3)
        rec["apex_median_s"] = round(float(np.nanmedian(d)), 1)
        lo, hi = get("ency_rt_start")[found], get("ency_rt_stop")[found]
        rec["apex_inside_bounds"] = round(float(np.mean((rt[found] >= lo) & (rt[found] <= hi))), 3)
        keep = found | decoy
        rec["tdc_1pct"], rec["tdc_5pct"] = _tdc_counts(get("score")[keep], decoy[keep])
        rows.append(rec)
    keys = list(rows[0])
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r[k]) for k in keys))


def grid(subsets: list[Path], bundle: Path, root: Path, workers: int, names: list[str] | None) -> None:
    outs = []
    for sub in subsets:
        for name in names or list(configs()):
            out = root / f"{sub.name}__{name}"
            if not (out / "run.json").exists():
                run(sub, bundle, out, name, workers)
            outs.append(out)
    report(outs, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--subset", type=Path, required=True)
    r.add_argument("--bundle", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--config", default="base")
    r.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("report")
    p.add_argument("outs", type=Path, nargs="+")
    p.add_argument("--subset", type=Path, default=None)
    g = sub.add_parser("grid")
    g.add_argument("--subset", type=Path, action="append", required=True)
    g.add_argument("--bundle", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--configs", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.subset, a.bundle, a.out, a.config, a.workers)
    elif a.cmd == "report":
        report(a.outs, a.subset)
    else:
        grid(a.subset, a.bundle, a.out, a.workers, a.configs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
