"""``constellation massspec convert``: Thermo ``.raw`` to Parquet bundles.

A profile run comes out as its profile bundle and a peak list made from it by
``DEFAULT_ALGO``; ``--no-profile`` and ``--centroid-algo`` adjust that. A centroid
run comes out as its peaks, and neither flag has anything to do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Restated: importing the reader package for its names would slow every
# command's parser. A test holds these equal to the reader's and the registry's.
DEFAULT_ALGO = "thermo"
NO_ALGO = "none"
OWN_ALGOS = ("apex",)

# ``help`` is what `massspec -h` lists and ``description`` what `massspec convert -h`
# opens with; argparse shows neither in the other's place.
_SUMMARY = (
    "Convert Thermo .raw file(s) into directory bundles. By default a run with "
    "profile scans gives two: <run>.profile/ holds its profile points, and <run>/ "
    f"holds peaks made from them by --centroid-algo (default: {DEFAULT_ALGO}), so "
    "other peak lists can be made later. A run acquired as centroids gives <run>/ "
    "with the instrument's centroids, and --no-profile and --centroid-algo do "
    "nothing to it. Each bundle has manifest.json, scan_metadata.parquet and "
    "acquisition_metadata.parquet beside its points. Accepts a single .raw file OR "
    "a directory of .raw files; --threads worker processes read ranges of scans of "
    "one file or many. Dispatches on the input suffix; .raw routes to "
    "constellation.massspec.readers.thermo.convert. Future mzML / Bruker / Sciex "
    "converters slot in transparently."
)


# ── convert ─────────────────────────────────────────────────────────────


def _build_convert_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser("convert", help=_SUMMARY, description=_SUMMARY)
    p.add_argument(
        "input",
        type=Path,
        help=(
            "Path to a .raw file OR a directory containing .raw files "
            "(non-recursive glob, case-insensitive)."
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory for the output bundle(s): "
            "<output-dir>/<input-stem>/ and, for profile data, "
            "<output-dir>/<input-stem>.profile/. Defaults to the input file's "
            "parent directory (single-file mode) or the input directory "
            "itself (directory mode)."
        ),
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Worker processes reading ranges of scans, of one file or of "
            "every file of a directory (default: the cores, up to 8; 1 "
            "converts in this process). Each worker loads its own .NET "
            "runtime; processes, not threads, because pythonnet holds the "
            "GIL through every .NET call."
        ),
    )
    p.add_argument(
        "--rt-bin-width",
        dest="rt_bin_width_s",
        type=float,
        default=60.0,
        help=(
            "PyArrow row-group RT chunking (seconds). The peak-table "
            "ParquetWriter flushes a new row group every N seconds of "
            "retention time, so downstream RT-window readers get "
            "row-group-level filter pushdown. Purely a write-time "
            "knob — no biological binning happens. Default: 60.0"
        ),
    )
    p.add_argument(
        "--no-profile",
        dest="keep_profile",
        action="store_false",
        help=(
            "Do not keep the profile points of profile-acquired scans. By "
            "default they go into <run>.profile/ beside the peak list, so the "
            "run can be centroided again later (~19,500 points per Stellar MS2 "
            "scan against ~1,200 centroids, and it can be deleted at any time). "
            "No effect on a run with no profile scans."
        ),
    )
    p.add_argument(
        "--centroid-algo",
        choices=(DEFAULT_ALGO, NO_ALGO, *OWN_ALGOS),
        default=DEFAULT_ALGO,
        help=(
            f"How to make peaks from profile scans. Default: {DEFAULT_ALGO}. "
            "'thermo' is Thermo's centroider, Scan.ToCentroid, as "
            "ThermoRawFileParser and msconvert run it. 'none' makes no peak list, "
            "only <run>.profile/, for centroiding later with `massspec centroid`. "
            "'apex' is the profile's local maxima, a cheap baseline. This applies "
            "to profile scans with no stored centroids, as ion-trap profile scans "
            "have none; an Orbitrap's centroid stream and a run's stored centroids "
            "stay the instrument's whatever this says."
        ),
    )
    p.add_argument("--profile", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--no-precursor-major",
        dest="precursor_major",
        action="store_false",
        help=(
            "Do not also write the peaks ordered by isolation window "
            "(<run>/precursor_major/, about the size of peaks/), which a "
            "search reads one window at a time. It can be written later with "
            "`massspec precursor-major BUNDLE`."
        ),
    )
    p.add_argument(
        "--allow-empty-levels",
        action="store_true",
        help=(
            "Finish the bundle even when every scan of an MS level came out "
            "with no peaks. Without it such a conversion fails, so that an "
            "empty bundle is never written silently."
        ),
    )
    p.add_argument(
        "--no-trailer-extras",
        dest="capture_trailer_extras",
        action="store_false",
        help=(
            "Skip capturing the verbatim trailer-extras map. Typed "
            "trailer columns remain populated; only the catch-all "
            "string map is dropped."
        ),
    )
    p.add_argument(
        "--no-sha256",
        dest="compute_sha256",
        action="store_false",
        help=(
            "Skip computing the SHA-256 of the input .raw. Saves I/O on "
            "large files; manifest.source_sha256 is null in that case."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite the run's existing bundles (a peak list made later "
            "with `massspec centroid` is left alone). Without this, a single "
            "file refuses to clobber and exits with an error; a directory "
            "treats existing bundles as skipped instead."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Scans read per block (default 512).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress progress messages on stderr.",
    )
    p.set_defaults(func=_cmd_massspec_convert)


def _cmd_massspec_convert(args: argparse.Namespace) -> int:
    """``constellation massspec convert <input>`` dispatcher.

    Routes a file to the single-file path and a directory to the batch
    path; both run ``--threads`` worker processes (one runs in this one).
    """
    input_path: Path = args.input
    if args.profile:
        print(
            "error: --profile was removed: the profile is now kept by default, in "
            "<run>.profile/ beside the peak list. --centroid-algo none keeps it "
            "without making peaks.",
            file=sys.stderr,
        )
        return 2
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    if args.threads is None:
        from constellation.massspec.readers.thermo import default_workers

        args.threads = default_workers()
    if input_path.is_file():
        return _cmd_massspec_convert_single(args, input_path)
    return _cmd_massspec_convert_batch(args, input_path)


def _cmd_massspec_convert_single(
    args: argparse.Namespace,
    input_path: Path,
) -> int:
    """One file: refuses an existing bundle without --force (exit 4)."""
    suffix = input_path.suffix.lower()
    parent = args.output_dir if args.output_dir is not None else input_path.parent
    bundle_dir = parent / input_path.stem

    progress_cb = None
    if not args.no_progress:
        from constellation.core.progress import StreamProgress

        progress_cb = StreamProgress()

    if suffix == ".raw":
        from constellation.massspec.readers.thermo import convert as thermo_convert
        from constellation.massspec.readers.thermo._netruntime import require_thermo

        try:
            require_thermo()
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3

        try:
            manifest = thermo_convert(
                input_path,
                bundle_dir,
                rt_bin_width_s=args.rt_bin_width_s,
                keep_profile=args.keep_profile,
                centroid_algo=args.centroid_algo,
                capture_trailer_extras=args.capture_trailer_extras,
                batch_size=args.batch_size,
                compute_sha256=args.compute_sha256,
                force=args.force,
                allow_empty_levels=args.allow_empty_levels,
                precursor_major=getattr(args, "precursor_major", True),
                workers=args.threads,
                progress_cb=progress_cb,
            )
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 4
        except Exception as exc:  # noqa: BLE001
            print(
                f"error: thermo conversion failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 5

        from constellation.massspec.bundle import PROFILE_BUNDLE_KIND, PROFILE_SUFFIX
        from constellation.massspec.readers.thermo import describe_levels

        levels = manifest.run_metadata.get("levels") or []
        missing = manifest.run_metadata.get("n_scans_without_injection_time") or 0
        if manifest.kind == PROFILE_BUNDLE_KIND:
            print(f"wrote profile bundle: {parent / (input_path.stem + PROFILE_SUFFIX)}")
            for line in describe_levels(levels, noun="points"):
                print(f"  {line}")
        else:
            print(f"wrote bundle: {bundle_dir}")
            for line in describe_levels(levels):
                print(f"  {line}")
            profile = manifest.parameters.get("profile_bundle")
            print(f"  profile bundle: {parent / profile}" if profile else "  profile bundle: none")
        if missing:
            print(f"  {missing:,} scans have no injection time (a blank trailer, or none recorded)")
        return 0

    print(
        f"error: no MS converter registered for suffix {suffix!r}\n"
        f"  v1 supports .raw (Thermo). mzML / Bruker .d / Sciex .wiff land "
        f"as separate readers when their converters ship.",
        file=sys.stderr,
    )
    return 2


def _collect_raw_files(input_dir: Path) -> list[Path]:
    """Top-level non-recursive glob for ``*.raw`` + ``*.RAW``.

    Deduplicates by resolved path (handles case-insensitive filesystems
    where a single file matches both globs) and sorts by name for
    deterministic order across runs.
    """
    matches: dict[Path, Path] = {}
    for pattern in ("*.raw", "*.RAW"):
        for p in input_dir.glob(pattern):
            if p.is_file():
                matches.setdefault(p.resolve(), p)
    return sorted(matches.values(), key=lambda p: p.name)


def _cmd_massspec_convert_batch(
    args: argparse.Namespace,
    input_path: Path,
) -> int:
    """A directory of .raw files, converted by one pool of workers.

    Exits:
      0 — all results ``ok`` or ``skipped``
      2 — input directory contains no ``.raw`` files
      3 — pythonnet / DLLs missing (parent fails fast)
      5 — at least one per-file result is ``error``
    """
    from constellation.massspec.readers.thermo import convert_batch
    from constellation.massspec.readers.thermo._netruntime import require_thermo

    try:
        require_thermo()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    paths = _collect_raw_files(input_path)
    if not paths:
        print(
            f"error: no .raw files in {input_path}",
            file=sys.stderr,
        )
        return 2
    default_parent = input_path

    output_parent = (
        args.output_dir if args.output_dir is not None else default_parent
    )

    progress_cb = None
    if not args.no_progress:
        from constellation.core.progress import StreamProgress

        progress_cb = StreamProgress()

    results = convert_batch(
        paths,
        output_parent,
        n_workers=args.threads,
        force=args.force,
        rt_bin_width_s=args.rt_bin_width_s,
        keep_profile=args.keep_profile,
        centroid_algo=args.centroid_algo,
        capture_trailer_extras=args.capture_trailer_extras,
        batch_size=args.batch_size,
        compute_sha256=args.compute_sha256,
        allow_empty_levels=args.allow_empty_levels,
        precursor_major=getattr(args, "precursor_major", True),
        progress_cb=progress_cb,
    )

    n_ok = sum(1 for r in results if r.status == "ok")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_err = sum(1 for r in results if r.status == "error")
    print(
        f"batch convert: {n_ok} succeeded, {n_skip} skipped, {n_err} failed",
        file=sys.stderr,
    )
    if n_err:
        print("  failed:", file=sys.stderr)
        for r in results:
            if r.status == "error":
                print(f"    {r.input_path}: {r.detail}", file=sys.stderr)
        return 5
    return 0
