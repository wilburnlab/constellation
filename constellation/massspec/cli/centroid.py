"""``constellation massspec centroid``: make a peak list from a profile bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THERMO = "thermo"


def _build_centroid_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "centroid",
        help=(
            "Make a peak list from a run's profile bundle (<run>.profile, from `massspec "
            "convert`): a complete bundle of its own beside it, named "
            "<run>.<algorithm>-<digest>, that any command takes like any other, and that "
            "the browser draws over the profile. A changed setting makes another list and "
            "replaces nothing."
        ),
    )
    p.add_argument("bundle", type=Path, help="The run's profile bundle.")
    p.add_argument(
        "--centroid-algo",
        required=True,
        choices=("apex", THERMO),
        help=(
            "'apex': the profile's local maxima, a cheap baseline. 'thermo': Scan.ToCentroid, "
            "which takes a .NET Scan, so it reads the .raw again, found through the manifest "
            "and checked by SHA-256."
        ),
    )
    p.add_argument("-o", "--output-dir", type=Path, default=None, help="Where to write it, instead of beside its source.")
    p.add_argument("--stride", type=int, default=None, help="Pick every Nth scan only: a partial list for looking at, which cannot be searched.")
    p.add_argument("--scans", type=Path, default=None, help="Pick these scans only (one number per line): a partial list, as --stride.")
    p.add_argument("--threads", type=int, default=None, help="Worker processes (default: the cores, up to 8).")
    p.add_argument("--force", action="store_true", help="Make it again if it is already there.")
    p.set_defaults(func=_cmd_massspec_centroid)


def _thermo_from_raw(args: argparse.Namespace) -> int:
    """Thermo's list, by converting the .raw the bundle names, profile aside."""
    from constellation.core import stages
    from constellation.massspec.bundle import read_bundle_manifest
    from constellation.massspec.readers.thermo import convert
    from constellation.massspec.readers.thermo._convert import _sha256_file
    from constellation.massspec.readers.thermo._netruntime import require_thermo

    if args.stride is not None or args.scans is not None:
        print("error: --stride and --scans are for an algorithm of Constellation's own", file=sys.stderr)
        return 2
    manifest = read_bundle_manifest(args.bundle)
    raw = Path(str(manifest.get("source_file") or ""))
    if not raw.is_file():
        print(f"error: {raw} is gone, and thermo reads the .raw: give it back, or use `massspec convert`", file=sys.stderr)
        return 2
    wanted = manifest.get("source_sha256")
    if wanted and _sha256_file(raw.resolve()) != wanted:
        print(f"error: {raw} is not the file this bundle was converted from (SHA-256 differs)", file=sys.stderr)
        return 2
    try:
        require_thermo()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    name = f"{THERMO}-{stages.stage_key(algorithm=THERMO)[:8]}"
    out = args.output_dir or args.bundle.parent / f"{raw.stem}.{name}"
    try:
        convert(raw, out, keep_profile=False, force=args.force, workers=args.threads or 8)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    print(f"wrote peak list: {out}")
    return 0


def _cmd_massspec_centroid(args: argparse.Namespace) -> int:
    if args.centroid_algo == THERMO:
        return _thermo_from_raw(args)
    from constellation.massspec.centroiding import centroid_bundle
    from constellation.massspec.readers.thermo import default_workers

    scans = None
    if args.scans is not None:
        scans = [int(line) for line in args.scans.read_text().split() if line.strip()]
    try:
        out = centroid_bundle(
            args.bundle,
            args.centroid_algo,
            output_dir=args.output_dir,
            scans=scans,
            stride=args.stride,
            workers=args.threads or default_workers(),
            force=args.force,
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote peak list: {out}")
    return 0
