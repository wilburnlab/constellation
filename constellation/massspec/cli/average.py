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
            "so `massspec centroid` makes a peak list from it."
        ),
    )
    p.add_argument("bundle", type=Path, help="The run's profile bundle.")
    p.add_argument("-o", "--output-dir", type=Path, default=None, help="Where to write it, instead of beside its source.")
    p.add_argument(
        "--skip-first",
        type=int,
        default=1,
        help=(
            "Leave out the first N scans of each kind (default 1: the AGC's first fill, which "
            "has no history). 0 keeps every scan. The summary reports their injection time."
        ),
    )
    p.add_argument("--force", action="store_true", help="Make it again if it is already there.")
    p.set_defaults(func=_cmd_massspec_average)


def _cmd_massspec_average(args: argparse.Namespace) -> int:
    from constellation.massspec.averaging import average_bundle

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
