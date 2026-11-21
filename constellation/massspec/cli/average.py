"""``constellation massspec average``: one spectrum per kind of scan from a
direct-infusion profile run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_average_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "average",
        help=(
            "Average a direct-infusion run's profile bundle (<run>.profile, from `massspec "
            "convert`) into one spectrum per kind of scan (MS level, polarity, activation, "
            "energy, precursor, scan range): the mean of every repeat, point by point on the "
            "ion trap's m/z grid, with each point's standard deviation and replicate count. "
            "Written beside its source as <run>.average, a profile bundle any command reads, "
            "so `massspec centroid` makes a peak list from it. Accepts one profile bundle OR a "
            "directory of them (as `massspec convert` writes them); runs already averaged are "
            "skipped unless --force."
        ),
    )
    p.add_argument(
        "bundle",
        type=Path,
        help=(
            "A run's profile bundle (<run>.profile), or a directory holding them "
            "(non-recursive; peak bundles and existing averages are ignored)."
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "The directory each <run>.average is written into (default: beside its source). "
            "Only an earlier average of the same run is ever replaced there, with --force."
        ),
    )
    p.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help=(
            "Leave out the first N scans of each kind, e.g. 1 for the AGC's first fill, which "
            "has no history (default 0: keep every scan). The summary reports their injection time."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Make an average that is already there again (a directory skips averaged runs without "
            "it). The old one is replaced only once the new one is finished."
        ),
    )
    p.set_defaults(func=_cmd_massspec_average)


def _describe(out: Path) -> tuple[str, int]:
    """What an average holds, in a few words, and how many points it left
    out for sitting on no grid point."""
    import json

    run = json.loads((out / "manifest.json").read_text(encoding="utf-8")).get("run_metadata") or {}
    points = sum(int(level.get("n_peaks", 0)) for level in run.get("levels") or [])
    off = int(run.get("n_points_off_grid") or 0)
    text = (
        f"{run.get('n_scans')} kind(s) of scan from {run.get('n_scans_averaged')} scans "
        f"({run.get('n_scans_skipped')} skipped), {points:,} points"
    )
    return text, off


def _warn_off_grid(out: Path, off: int) -> None:
    if off:
        print(
            f"warning: {out.name}: {off:,} points sat on no grid point and were left out "
            f"(per kind of scan in average_summary.parquet, n_points_off_grid)",
            file=sys.stderr,
        )


def _cmd_massspec_average(args: argparse.Namespace) -> int:
    """Exits: 0 done (every run ok or skipped); 2 bad input, or a directory with
    no profile bundles; 4 an average already there, or something else where it
    would go; 5 a run of a directory failed (the rest are still made)."""
    from constellation.massspec.average import average_bundle

    if args.bundle.is_dir() and not (args.bundle / "manifest.json").is_file():
        return _cmd_massspec_average_directory(args)
    try:
        out = average_bundle(
            args.bundle, output_parent=args.output_dir, skip_first=args.skip_first, force=args.force
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text, off = _describe(out)
    print(f"wrote averaged bundle: {out}: {text}")
    _warn_off_grid(out, off)
    return 0


def _cmd_massspec_average_directory(args: argparse.Namespace) -> int:
    from constellation.massspec.average import average_directory

    try:
        results = average_directory(
            args.bundle, output_parent=args.output_dir, skip_first=args.skip_first, force=args.force
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not results:
        print(
            f"error: no profile bundles in {args.bundle} (convert the .raw files first: "
            f"`constellation massspec convert {args.bundle}`)",
            file=sys.stderr,
        )
        return 2
    for r in results:
        where = f" -> {r.output}" if r.output is not None else ""
        note = f" ({r.detail})" if r.detail else ""
        if r.status == "ok":
            text, off = _describe(r.output)
            note = f": {text}"
        print(f"{r.status:>7}  {r.bundle.name}{where}{note}")
        if r.status == "ok":
            _warn_off_grid(r.output, off)
    n = {s: sum(r.status == s for r in results) for s in ("ok", "skipped", "error")}
    print(f"batch average: {n['ok']} averaged, {n['skipped']} skipped, {n['error']} failed", file=sys.stderr)
    return 5 if n["error"] else 0
