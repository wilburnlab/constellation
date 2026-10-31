"""``constellation massspec mz-axis``: measure a bundle's m/z axis from its data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_mz_axis_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "mz-axis",
        help=(
            "Measure how far a converted acquisition's m/z axis reads high or low, from "
            "where its signal falls on the peptide mass lattice, with no search (on the "
            "principle of Egertson et al. 2012, FineTune). Works on a profile bundle or a "
            "peak list; a peak list's curve includes its centroider's bias. The curve is "
            "stored beside the bundle (mz_axes/<key>.parquet); nothing is rewritten, and "
            "it applies only where it is named: --mz-axis on `massspec search` and "
            "`massspec encyclopedia export-spectra`, or the toggle in the browser."
        ),
    )
    p.add_argument("bundle", type=Path, help="A `massspec convert` bundle.")
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="A reference table (scripts/build-mass-lattice-reference.py); the packaged one by default.",
    )
    p.add_argument(
        "--target-se",
        type=float,
        default=None,
        help="Widen a knot's window until its shift is known this well, in Th. Default 0.01.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure and print, keeping nothing.",
    )
    p.set_defaults(func=_cmd_massspec_mz_axis)


def _cmd_massspec_mz_axis(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from constellation.massspec.mz_axis import (
        MzAxisSettings,
        describe,
        estimate_mz_axis,
        load_reference,
        save_mz_axis,
    )

    settings = MzAxisSettings()
    if args.target_se is not None:
        settings = replace(settings, target_se_th=args.target_se)
    reference = load_reference(args.reference) if args.reference else None
    try:
        axis = estimate_mz_axis(args.bundle, settings=settings, reference=reference)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    scans = axis.parameters["n_scans"]
    print(f"m/z axis {axis.key} of {args.bundle.name}, from {scans['1']:,} MS1 and {scans['2']:,} MS2 scans")
    for line in describe(axis):
        print(f"  {line}")
    if axis.table.num_rows == 0:
        print("error: no window held enough signal to measure", file=sys.stderr)
        return 1
    if not args.dry_run:
        print(f"wrote {save_mz_axis(args.bundle, axis)}")
    return 0
