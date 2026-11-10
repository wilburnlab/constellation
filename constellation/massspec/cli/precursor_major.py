"""``constellation massspec precursor-major``: a bundle's peaks ordered by isolation window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_precursor_major_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "precursor-major",
        help=(
            "Write a converted acquisition's peaks a second time, ordered by the precursor m/z "
            "each can be evidence for (MS2 and above by isolation window, MS1 by m/z), into "
            "<bundle>/precursor_major/, which a search reads one window at a time. `massspec "
            "convert` writes it already; this is for a bundle that has none or an outdated one."
        ),
    )
    p.add_argument("bundle", type=Path, help="A peak bundle (from `massspec convert`, `centroid` or an import).")
    p.add_argument("--workers", type=int, default=1, help="Peak parts written at once, in processes. Default 1.")
    p.add_argument("--force", action="store_true", help="Write it again even when the one there is current.")
    p.set_defaults(func=_cmd_massspec_precursor_major)


def _cmd_massspec_precursor_major(args: argparse.Namespace) -> int:
    from constellation.massspec.precursor_major import build_precursor_major, describe, find_precursor_major

    current = find_precursor_major(args.bundle)
    if current is not None and not args.force:
        print(f"{current.root}: current (levels {', '.join(f'MS{lv}' for lv in current.levels)}); "
              f"--force writes it again")  # fmt: skip
        return 0
    try:
        root = build_precursor_major(args.bundle, workers=max(1, args.workers), force=True)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    info = describe(args.bundle) or {}
    print(f"{root}: {info.get('rows', 0):,} points, levels "
          f"{', '.join(f'MS{lv}' for lv in info.get('levels', []))}")  # fmt: skip
    return 0
