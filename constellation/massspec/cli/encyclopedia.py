"""``constellation massspec encyclopedia ...``: EncyclopeDIA file utilities that need no jar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# encyclopedia — EncyclopeDIA-specific utilities
# ──────────────────────────────────────────────────────────────────────
#
# Naming: "DIA" is an acquisition method, and Constellation uses the word
# only for that. EncyclopeDIA happens to call its spectra file `.dia`
# (it holds DDA runs just as well), so the commands that read and write
# that file are named for the tool whose format it is, not for the
# extension.


def _build_encyclopedia_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "encyclopedia",
        help=(
            "EncyclopeDIA-specific utilities. Its spectra file (the `.dia` "
            "the search jar reads; despite the extension it holds DIA or "
            "DDA runs) is written and read by Constellation itself, with "
            "no jar and no MSForest: `export-spectra`, `import-spectra`, "
            "`compare-spectra`. `compare-results` compares two searches."
        ),
    )
    esubs = p.add_subparsers(dest="encyclopedia_subcommand", required=True)

    e = esubs.add_parser(
        "export-spectra",
        help=(
            "Write an EncyclopeDIA spectra file (`.dia`, format 0.8.0) from "
            "one or more `massspec convert` bundles. One bundle gives a "
            "single-run file; several are merged as fractions, in the order "
            "given (gas-phase fractions)."
        ),
    )
    e.add_argument("--peaks-dir", type=Path, nargs="+", required=True,
                   help="Convert bundle(s): directories holding peaks.parquet + "
                        "scan_metadata.parquet + acquisition_metadata.parquet.")
    e.add_argument("--output-file", type=Path, required=True,
                   help="The spectra file to write; must end in .dia, the extension "
                        "EncyclopeDIA expects. A scan map (<name>.dia.scans.parquet) "
                        "is written beside it.")
    e.add_argument("--workers", type=int, default=None,
                   help="zlib compression threads (default: up to 8).")
    e.add_argument("--force", action="store_true",
                   help="Replace an existing output file.")
    e.add_argument("--no-progress", action="store_true",
                   help="Suppress the summary on stderr.")
    e.set_defaults(func=_cmd_encyclopedia_export_spectra)

    i = esubs.add_parser(
        "import-spectra",
        help=(
            "Turn an EncyclopeDIA spectra file (`.dia`, format 0.5.0 or "
            "0.8.0) into Constellation bundles, the same shape `massspec "
            "convert` writes. A merged file gives one bundle per fraction."
        ),
    )
    i.add_argument("--spectra-file", type=Path, required=True,
                   help="The EncyclopeDIA spectra file (.dia) to read.")
    i.add_argument("-o", "--output-dir", type=Path, required=True,
                   help="Parent directory; each bundle is <output-dir>/<run name>/.")
    i.add_argument("--rt-bin-width", type=float, default=60.0, dest="rt_bin_width_s",
                   help="Row-group RT chunking of peaks.parquet, seconds (default 60).")
    i.add_argument("--force", action="store_true",
                   help="Replace existing bundles.")
    i.add_argument("--no-progress", action="store_true",
                   help="Suppress the summary on stderr.")
    i.set_defaults(func=_cmd_encyclopedia_import_spectra)

    d = esubs.add_parser(
        "compare-spectra",
        help=(
            "Compare two EncyclopeDIA spectra files (`.dia`) by content: "
            "every scan's values and peak arrays, the isolation windows, "
            "fractions and metadata. Exits 0 when identical, 1 when not."
        ),
    )
    d.add_argument("a", type=Path, help="First spectra file (.dia).")
    d.add_argument("b", type=Path, help="Second spectra file (.dia).")
    d.add_argument("--rtol", type=float, default=0.0,
                   help="Relative tolerance for scalar values (scan times, window "
                        "bounds, TIC, duty cycle). Default 0: exact.")
    d.add_argument("--mz-atol", type=float, default=0.0,
                   help="Absolute tolerance for peak m/z. Default 0: exact.")
    d.add_argument("--intensity-rtol", type=float, default=0.0,
                   help="Relative tolerance for peak intensities. Default 0: exact.")
    d.add_argument("--all-metadata", action="store_true",
                   help="Also report metadata keys present in only one file "
                        "(by default only keys both files have are compared).")
    d.set_defaults(func=_cmd_encyclopedia_compare_spectra)

    r = esubs.add_parser(
        "compare-results",
        help=(
            "Compare two EncyclopeDIA result libraries (`.elib`, or ScribeTwo's "
            "`.dlib`) from single-run searches: the precursors, peptides and "
            "proteins each identified, and for the shared precursors how far "
            "apart retention time, score and quantity are. Two searches of the "
            "same file can differ by a few precursors at the FDR line, so a "
            "repeat search is the yardstick. Exits 0 when the precursor overlap "
            "reaches --min-overlap, 1 when not."
        ),
    )
    r.add_argument("a", type=Path, help="First result library.")
    r.add_argument("b", type=Path, help="Second result library.")
    r.add_argument("--min-overlap", type=float, default=1.0,
                   help="Smallest acceptable Jaccard overlap of the two precursor "
                        "sets, 0-1. Default 1: identical identifications.")
    r.set_defaults(func=_cmd_encyclopedia_compare_results)


def _cmd_encyclopedia_export_spectra(args: argparse.Namespace) -> int:
    from constellation.massspec.io.encyclopedia import write_dia

    try:
        summary = write_dia(
            args.peaks_dir, args.output_file, overwrite=args.force, workers=args.workers
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.no_progress:
        kinds = ", ".join(
            f"{s.acquisition_type}{' staggered' if s.is_staggered else ''}"
            for s in summary.structures
        )
        print(
            f"wrote {summary.path}: {summary.n_fractions} fraction(s) [{kinds}], "
            f"{summary.n_precursor:,} MS1 + {summary.n_spectra:,} MS2 scans, "
            f"{summary.n_ranges:,} isolation windows\n"
            f"scan map: {summary.scan_map_path}",
            file=sys.stderr,
        )
    return 0


def _cmd_encyclopedia_import_spectra(args: argparse.Namespace) -> int:
    from constellation.massspec.io.encyclopedia import import_dia

    try:
        bundles = import_dia(
            args.spectra_file,
            args.output_dir,
            rt_bin_width_s=args.rt_bin_width_s,
            overwrite=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.no_progress:
        for bundle in bundles:
            print(f"wrote bundle: {bundle}", file=sys.stderr)
    return 0


def _cmd_encyclopedia_compare_spectra(args: argparse.Namespace) -> int:
    from constellation.massspec.io.encyclopedia import diff_dia

    try:
        result = diff_dia(
            args.a,
            args.b,
            rtol=args.rtol,
            mz_atol=args.mz_atol,
            intensity_rtol=args.intensity_rtol,
            metadata_keys="all" if args.all_metadata else "common",
        )
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.report())
    return 0 if result.identical else 1


def _cmd_encyclopedia_compare_results(args: argparse.Namespace) -> int:
    from constellation.massspec.io.encyclopedia import compare_results

    if not 0.0 <= args.min_overlap <= 1.0:
        print("error: --min-overlap must be between 0 and 1", file=sys.stderr)
        return 2
    try:
        result = compare_results(args.a, args.b)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.report())
    return 0 if result.precursors.jaccard >= args.min_overlap else 1
