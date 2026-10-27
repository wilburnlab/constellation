"""``constellation massspec convert``: Thermo ``.raw`` to a peak / scan-metadata parquet bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── convert ─────────────────────────────────────────────────────────────


def _build_convert_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "convert",
        help=(
            "Convert Thermo .raw file(s) into directory bundles "
            "(manifest.json + peaks/ + scan_metadata.parquet + "
            "acquisition_metadata.parquet). Accepts a single .raw file "
            "OR a directory of .raw files; --threads worker processes read "
            "ranges of scans of one file or many. Dispatches on the input "
            "suffix; .raw routes to constellation.massspec.readers.thermo.convert. "
            "Future mzML / Bruker / Sciex converters slot in transparently."
        ),
    )
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
            "Parent directory for the output bundle(s). Each bundle dir "
            "is <output-dir>/<input-stem>/. Defaults to the input file's "
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
        "--profile",
        action="store_true",
        help=(
            "Keep the profile points of profile-acquired scans instead of "
            "centroiding them (scans acquired as centroids stay centroids). "
            "By default every scan becomes centroids: the instrument's where it "
            "stored them (with per-peak resolution/noise/baseline from an "
            "Orbitrap's centroid stream), and Thermo's centroider on profile "
            "scans that have none, as ion-trap profile scans do not. A profile "
            "run is far larger: ~19,500 points per Stellar MS2 scan against "
            "~1,200 centroids."
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
            "Overwrite an existing non-empty bundle directory. Without "
            "this, single-file mode refuses to clobber and exits with "
            "an error; batch mode (directory input or --threads > 1) "
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
                profile=args.profile,
                capture_trailer_extras=args.capture_trailer_extras,
                batch_size=args.batch_size,
                compute_sha256=args.compute_sha256,
                force=args.force,
                allow_empty_levels=args.allow_empty_levels,
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

        from constellation.massspec.readers.thermo import describe_levels

        print(f"wrote bundle: {bundle_dir}")
        for line in describe_levels(manifest.run_metadata.get("levels") or []):
            print(f"  {line}")
        print(f"  manifest:                {bundle_dir / 'manifest.json'}")
        print(f"  peaks:                   {bundle_dir / manifest.outputs['peaks']}")
        print(
            f"  scan_metadata:           "
            f"{bundle_dir / manifest.outputs['scan_metadata']}"
        )
        print(
            f"  acquisition_metadata:    "
            f"{bundle_dir / manifest.outputs['acquisition_metadata']}"
        )
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
        profile=args.profile,
        capture_trailer_extras=args.capture_trailer_extras,
        batch_size=args.batch_size,
        compute_sha256=args.compute_sha256,
        allow_empty_levels=args.allow_empty_levels,
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
