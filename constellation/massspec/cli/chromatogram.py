"""``constellation massspec chromatogram {build-index, extract}``: XIC extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── chromatogram (XIC extraction) ────────────────────────────────────────


_DEFAULT_CHARGE_RANGE = (1, 4)
_DEFAULT_N_ISOTOPES = 3
_DEFAULT_MAX_FRAG_CHARGE = 2


def _build_chromatogram_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "chromatogram",
        help=(
            "Extracted-ion-chromatogram (XIC) extraction. `build-index` "
            "builds a derived mass-sorted peak index; `extract` pulls XIC "
            "traces for a target list (a library, search results, or a "
            "bare m/z+RT target table)."
        ),
    )
    csubs = p.add_subparsers(dest="chromatogram_subcommand", required=True)
    _build_chromatogram_build_index_parser(csubs)
    _build_chromatogram_extract_parser(csubs)


def _build_chromatogram_build_index_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "build-index",
        help=(
            "Build a derived (ms_level, isolation_window)-partitioned, "
            "mz-then-rt-sorted peak index from a convert bundle. Lossless "
            "f64 by default; --downcast trades precision for size/speed."
        ),
    )
    p.add_argument("--peaks-dir", type=Path, required=True,
                   help="Convert bundle, or a directory of peak parts / one parquet file.")
    p.add_argument("-o", "--output-dir", type=Path, required=True,
                   help="Index output root.")
    p.add_argument("--downcast", default="",
                   help="Comma-separated subset of {mz,rt,intensity} to store as f32 "
                        "(default none → lossless f64 mirror).")
    p.add_argument("--dda-window-width-mz", type=float, default=10.0,
                   help="Synthetic precursor-m/z grid width for binned (DDA) MS2 levels "
                        "(default 10.0).")
    p.add_argument("--max-isolation-windows", type=int, default=512,
                   help="Per-level distinct-window cap above which a level bins (default 512).")
    p.add_argument("--no-progress", action="store_true",
                   help="Suppress progress messages on stderr.")
    p.set_defaults(func=_cmd_chromatogram_build_index)


def _build_chromatogram_extract_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "extract",
        help=(
            "Extract XIC traces. --level selects MS1 (precursor isotopes) "
            "or MS2 (fragments); it is the MS-level selector, NOT a scan "
            "filter — scan selection is derived from each target's RT / "
            "isolation / assigned scan."
        ),
    )
    common = p.add_argument_group("common")
    common.add_argument("--peaks-dir", type=Path, required=True,
                        help="Convert bundle (canonical f64 peaks source).")
    common.add_argument("--targets", type=Path, required=True,
                        help="Target list: a Library / Search / PRECURSOR_QUANT bundle, "
                             "or a bare XIC_TARGET_TABLE .parquet/.tsv (auto-detected).")
    common.add_argument("-o", "--output-dir", type=Path, required=True)
    common.add_argument("--level", type=int, choices=(1, 2), required=True,
                        help="MS level to extract from (1 = precursor isotopes, 2 = fragments).")
    common.add_argument("--acquisition-id", type=int, default=0,
                        help="acquisition_id stamped on output rows (FK; default 0).")
    common.add_argument("--tolerance", type=float, default=20.0)
    common.add_argument("--tolerance-unit", choices=("ppm", "Da"), default="ppm")
    common.add_argument("--rt-window", type=float, default=None,
                        help="Half-width (s) around each target's rt_center; explicit "
                             "rt_start/rt_end in the target override it; absent → all-RT.")
    common.add_argument("--all-in-window", action="store_true",
                        help="Return every peak within tolerance per (scan, ion), not just nearest.")
    common.add_argument("--no-drop-unmatched", dest="drop_unmatched", action="store_false",
                        help="Keep unmatched query rows (intensity 0) instead of dropping.")
    common.add_argument("--no-progress", action="store_true")

    idxg = p.add_argument_group("index (Mode B)")
    idxg.add_argument("--use-index", action="store_true",
                      help="Use the mass index (Mode B); build it in-line if absent.")
    idxg.add_argument("--index-dir", type=Path, default=None,
                      help="Index location override (default <peaks-dir>/mz_index/).")
    idxg.add_argument("--rebuild-index", action="store_true",
                      help="Force-rebuild the index (implies --use-index).")
    idxg.add_argument("--downcast", default="",
                      help="f32 axes when building the index (subset of {mz,rt,intensity}).")
    idxg.add_argument("--exact-error", action="store_true",
                      help="Mode B + f32-m/z index: re-derive f64 mz error for survivors "
                           "from --peaks-dir.")

    ms1g = p.add_argument_group("MS1 precursor-isotope options (with --level 1)")
    ms1g.add_argument("--n-isotopes", type=int, default=_DEFAULT_N_ISOTOPES,
                      help="Isotope peaks M+0..M+(n-1) (default 3).")
    ms1g.add_argument("--charge-range", type=int, nargs=2, metavar=("LO", "HI"),
                      default=list(_DEFAULT_CHARGE_RANGE),
                      help="Precursor charge sweep (default 1 4).")

    ms2g = p.add_argument_group("MS2 fragment options (with --level 2)")
    ms2g.add_argument("--ion-types", default="b,y",
                      help="Comma-separated fragment ion types (default b,y).")
    ms2g.add_argument("--max-fragment-charge", type=int, default=_DEFAULT_MAX_FRAG_CHARGE,
                      help="Max fragment charge, capped by precursor charge (default 2).")
    ms2g.add_argument("--neutral-losses", default=None,
                      help="Comma-separated neutral-loss ids (default none).")
    ms2g.add_argument("--assigned-scans-only", action="store_true",
                      help="Restrict MS2 to each target's assigned scan (requires a scan column).")
    p.set_defaults(func=_cmd_chromatogram_extract)


def _parse_downcast(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _cmd_chromatogram_build_index(args: argparse.Namespace) -> int:
    from constellation.massspec.quant.peak_index import build_peak_index

    if not args.peaks_dir.exists():
        print(f"error: peaks not found: {args.peaks_dir}", file=sys.stderr)
        return 2
    man = build_peak_index(
        args.peaks_dir, args.output_dir,
        downcast=_parse_downcast(args.downcast),
        max_isolation_windows=args.max_isolation_windows,
        dda_window_width_mz=args.dda_window_width_mz,
    )
    if not args.no_progress:
        print(f"built peak index: {man.n_windows} partitions, precision={man.precision}",
              file=sys.stderr)
    return 0


def _resolve_ion_types(spec: str):
    from constellation.massspec.peptide.ions import IonType

    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(IonType[tok.upper()])
        except KeyError:
            raise SystemExit(f"error: unknown ion type {tok!r}; expected a/b/c/x/y/z")
    return tuple(out)


def _cmd_chromatogram_extract(args: argparse.Namespace) -> int:
    from constellation.massspec.quant.chromatogram import (
        extract_xic_indexed,
        extract_xic_scan_major,
        save_xic,
    )
    from constellation.massspec.quant.peak_index import INDEX_DIRNAME, build_peak_index
    from constellation.massspec.quant.targets import load_targets

    if not args.peaks_dir.exists():
        print(f"error: peaks not found: {args.peaks_dir}", file=sys.stderr)
        return 2
    if not args.targets.exists():
        print(f"error: targets not found: {args.targets}", file=sys.stderr)
        return 2

    _warn_level_flag_mismatch(args)
    targets = load_targets(args.targets)
    charge_range = tuple(args.charge_range)
    ion_types = _resolve_ion_types(args.ion_types)
    neutral_losses = _parse_downcast(args.neutral_losses) if args.neutral_losses else None

    use_index = args.use_index or args.rebuild_index
    common = dict(
        acquisition_id=args.acquisition_id, level=args.level,
        n_isotopes=args.n_isotopes, ion_types=ion_types,
        max_fragment_charge=args.max_fragment_charge, neutral_losses=neutral_losses,
        charge_range=charge_range, rt_window=args.rt_window,
        tolerance=args.tolerance, tolerance_unit=args.tolerance_unit,
        all_in_window=args.all_in_window, assigned_scans_only=args.assigned_scans_only,
        drop_unmatched=args.drop_unmatched,
    )

    if use_index:
        index_dir = args.index_dir or (args.peaks_dir / INDEX_DIRNAME)
        manifest_exists = (index_dir / "manifest.json").exists()
        if args.rebuild_index or not manifest_exists:
            if not args.no_progress:
                print(f"building peak index at {index_dir}", file=sys.stderr)
            build_peak_index(args.peaks_dir, index_dir, downcast=_parse_downcast(args.downcast))
        table = extract_xic_indexed(
            index_dir, targets, exact_error=args.exact_error, peaks_dir=args.peaks_dir, **common
        )
    else:
        from constellation.massspec.bundle import open_peaks

        peaks = open_peaks(args.peaks_dir).to_table()
        table = extract_xic_scan_major(peaks, targets, **common)

    save_xic(table, args.output_dir, metadata={
        "level": args.level, "mode": "indexed" if use_index else "scan_major",
        "tolerance": args.tolerance, "tolerance_unit": args.tolerance_unit,
    })
    if not args.no_progress:
        print(f"extracted {table.num_rows} XIC rows → {args.output_dir}", file=sys.stderr)
    return 0


def _warn_level_flag_mismatch(args: argparse.Namespace) -> None:
    """Warn when a flag from the inactive level's group was set."""
    if args.level == 1:
        if args.ion_types != "b,y" or args.max_fragment_charge != _DEFAULT_MAX_FRAG_CHARGE \
                or args.neutral_losses or args.assigned_scans_only:
            print("warning: MS2 fragment options ignored with --level 1", file=sys.stderr)
    elif args.level == 2:
        if args.n_isotopes != _DEFAULT_N_ISOTOPES or tuple(args.charge_range) != _DEFAULT_CHARGE_RANGE:
            print("warning: MS1 isotope options ignored with --level 2", file=sys.stderr)
