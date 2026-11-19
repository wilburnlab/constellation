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
            "One bundle: the averaged bundle's directory. A directory of bundles: the "
            "directory every <run>.average is written into. Default: beside each source."
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
        help="Make it again if it is already there (a directory skips averaged runs without it).",
    )
    p.set_defaults(func=_cmd_massspec_average)


def _cmd_massspec_average(args: argparse.Namespace) -> int:
    """Exits: 0 done (every run ok or skipped); 2 bad input, or a directory with
    no profile bundles; 4 one bundle's average already there; 5 a directory
    run failed (the rest are still made)."""
    from constellation.massspec.averaging import average_bundle

    if args.bundle.is_dir() and not (args.bundle / "manifest.json").is_file():
        return _cmd_massspec_average_directory(args)
    try:
        out = average_bundle(args.bundle, output_dir=args.output_dir, skip_first=args.skip_first, force=args.force)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote averaged bundle: {out}")
    return 0


def _cmd_massspec_average_directory(args: argparse.Namespace) -> int:
    from constellation.massspec.averaging import average_directory, find_profile_bundles

    if not find_profile_bundles(args.bundle):
        print(
            f"error: no profile bundles in {args.bundle} (convert the .raw files first: "
            f"`constellation massspec convert {args.bundle}`)",
            file=sys.stderr,
        )
        return 2
    try:
        results = average_directory(
            args.bundle, output_parent=args.output_dir, skip_first=args.skip_first, force=args.force
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for r in results:
        where = f" -> {r.output}" if r.output is not None else ""
        note = f" ({r.detail})" if r.detail else ""
        print(f"{r.status:>7}  {r.bundle.name}{where}{note}")
    n = {s: sum(r.status == s for r in results) for s in ("ok", "skipped", "error")}
    print(f"batch average: {n['ok']} averaged, {n['skipped']} skipped, {n['error']} failed", file=sys.stderr)
    return 5 if n["error"] else 0
