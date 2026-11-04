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
            "Extract XIC traces. --level selects MS1 (precursor isotopes), "
            "MS2 (fragments) or both in one pass; it is the MS-level selector, "
            "NOT a scan filter — scan selection is derived from each target's "
            "RT / isolation / assigned scan. A bundle is swept once, every "
            "target matched against each scan together."
        ),
    )
    common = p.add_argument_group("common")
    common.add_argument("--peaks-dir", type=Path, required=True,
                        help="Convert bundle (canonical f64 peaks source).")
    common.add_argument("--targets", type=Path, required=True,
                        help="Target list: a Library / Search / PRECURSOR_QUANT bundle, "
                             "or a bare XIC_TARGET_TABLE .parquet/.tsv (auto-detected).")
    common.add_argument("-o", "--output-dir", type=Path, required=True)
    common.add_argument("--level", choices=("1", "2", "both"), required=True,
                        help="MS level to extract from (1 = precursor isotopes, 2 = fragments, "
                             "both = the two in one sweep).")
    common.add_argument("--acquisition-id", type=int, default=0,
                        help="acquisition_id stamped on output rows (FK; default 0).")
    common.add_argument("--tolerance", type=float, default=20.0,
                        help="Match tolerance at every level without its own (default 20).")
    common.add_argument("--tolerance-unit", choices=("ppm", "Da"), default="ppm")
    for level in (1, 2):
        common.add_argument(f"--ms{level}-tolerance", type=float, default=None,
                            help=f"MS{level} match tolerance (default --tolerance).")
        common.add_argument(f"--ms{level}-tolerance-unit", choices=("ppm", "Da"), default=None,
                            help=f"Unit of --ms{level}-tolerance (default --tolerance-unit).")
    common.add_argument("--rt-window", type=float, default=None,
                        help="Half-width (s) around each target's rt_center; explicit "
                             "rt_start/rt_end in the target override it; absent → all-RT.")
    common.add_argument("--rule", choices=("nearest", "all", "sum"), default=None,
                        help="What a tolerance window gives: the nearest peak (default), every "
                             "peak in it, or their summed intensity as one row.")
    common.add_argument("--all-in-window", action="store_true",
                        help="Same as --rule all.")
    common.add_argument("--no-drop-unmatched", dest="drop_unmatched", action="store_false",
                        help="Keep unmatched query rows (intensity 0) instead of dropping.")
    common.add_argument("--no-progress", action="store_true")

    sweepg = p.add_argument_group("sweep (default mode)")
    sweepg.add_argument("--mz-axis", default=None, metavar="KEY",
                        help="Correct observed m/z with the bundle's measured m/z axis KEY "
                             "(or `latest`) before matching.")
    sweepg.add_argument("--workers", type=int, default=_default_workers(),
                        help="Processes sweeping a bundle's scan ranges (default: the cores, "
                             "up to 4, about 3 GB each; 1 sweeps in this process).")
    sweepg.add_argument("--work-dir", type=Path, default=None,
                        help="Where the workers' matches go (default <output-dir>/.sweep, "
                             "removed when done).")
    sweepg.add_argument("--claims", action="store_true",
                        help="Also write peak_claims.parquet: every peak matched by more than "
                             "one target or ion, with each of them.")
    sweepg.add_argument("--summary", action="store_true",
                        help="Also write query_summary.parquet: per target, the evidence gate's "
                             "verdict, the scans it could be matched in and the ions matched.")

    gateg = p.add_argument_group(
        "evidence gate (sweep; any of these turns it on)",
        "A target counts as seen when a run of its MS2 scans each match enough distinct "
        "fragment bonds among the scan's most intense peaks. At low-resolution tolerances "
        "a bare count passes almost anything; see the summary's counts to set it.",
    )
    gateg.add_argument("--gate", action="store_true",
                       help="Turn the gate on with its defaults (6 bonds among each scan's 50 most "
                            "intense peaks, 3 scans in a row).")
    gateg.add_argument("--min-fragments", type=int, default=None,
                       help="Distinct fragment bonds a scan must show (default 6).")
    gateg.add_argument("--evidence-top-peaks", type=int, default=None,
                       help="Count only each scan's most intense N peaks; 0 for all (default 50).")
    gateg.add_argument("--min-consecutive-scans", type=int, default=None,
                       help="Evidence scans in a row a target must show (default 3).")
    gateg.add_argument("--max-gap-s", type=float, default=None,
                       help="Longest RT step (s) within such a run (default none).")
    gateg.add_argument("--min-isotopes", type=int, default=None,
                       help="Gate targets without fragments on this many isotopes per MS1 scan "
                            "(default: they pass).")
    gateg.add_argument("--only-passing", action="store_true",
                       help="Write only the targets the gate passed into the trace.")

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

    ms1g = p.add_argument_group("MS1 precursor-isotope options (with --level 1 or both)")
    ms1g.add_argument("--n-isotopes", type=int, default=_DEFAULT_N_ISOTOPES,
                      help="Isotope peaks M+0..M+(n-1) (default 3).")
    ms1g.add_argument("--charge-range", type=int, nargs=2, metavar=("LO", "HI"),
                      default=list(_DEFAULT_CHARGE_RANGE),
                      help="Precursor charge sweep (default 1 4).")

    ms2g = p.add_argument_group("MS2 fragment options (with --level 2 or both)")
    ms2g.add_argument("--ion-types", default="b,y",
                      help="Comma-separated fragment ion types (default b,y).")
    ms2g.add_argument("--max-fragment-charge", type=int, default=_DEFAULT_MAX_FRAG_CHARGE,
                      help="Max fragment charge, capped by precursor charge (default 2).")
    ms2g.add_argument("--neutral-losses", default=None,
                      help="Comma-separated neutral-loss ids (default none).")
    ms2g.add_argument("--assigned-scans-only", action="store_true",
                      help="Restrict MS2 to each target's assigned scan (requires a scan column).")
    p.set_defaults(func=_cmd_chromatogram_extract)


def _default_workers() -> int:
    """Four at most: each worker holds about 3 GB at peak, and four sweep
    a whole run 2.4–2.8× faster than one (docs/plans/sweep.md)."""
    import os

    return max(1, min(4, os.cpu_count() or 1))


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
    import shutil

    from constellation.massspec.quant.chromatogram import (
        extract_xic,
        extract_xic_indexed,
        extraction_tolerance_metadata,
        save_xic,
        write_xic_manifest,
    )
    from constellation.massspec.quant.peak_index import INDEX_DIRNAME, build_peak_index
    from constellation.massspec.quant.targets import load_targets
    from constellation.massspec.sweep import IonSettings

    if not args.peaks_dir.exists():
        print(f"error: peaks not found: {args.peaks_dir}", file=sys.stderr)
        return 2
    if not args.targets.exists():
        print(f"error: targets not found: {args.targets}", file=sys.stderr)
        return 2
    levels = [1, 2] if args.level == "both" else [int(args.level)]
    if args.all_in_window and args.rule not in (None, "all"):
        print(f"error: --all-in-window is --rule all; it contradicts --rule {args.rule}", file=sys.stderr)
        return 2
    rule = args.rule or ("all" if args.all_in_window else "nearest")
    tolerances = {
        level: (
            getattr(args, f"ms{level}_tolerance") or args.tolerance,
            getattr(args, f"ms{level}_tolerance_unit") or args.tolerance_unit,
        )
        for level in levels
    }

    _warn_level_flag_mismatch(args)
    targets = load_targets(args.targets)
    charge_range = tuple(args.charge_range)
    ion_types = _resolve_ion_types(args.ion_types)
    neutral_losses = _parse_downcast(args.neutral_losses) if args.neutral_losses else None

    use_index = args.use_index or args.rebuild_index
    if use_index:
        if len(levels) > 1 or rule == "sum":
            print("error: the mass index (--use-index) extracts one level at a time, by the "
                  "nearest peak or all peaks in a window", file=sys.stderr)  # fmt: skip
            return 2
        level = levels[0]
        tolerance, tolerance_unit = tolerances[level]
        index_dir = args.index_dir or (args.peaks_dir / INDEX_DIRNAME)
        manifest_exists = (index_dir / "manifest.json").exists()
        if args.rebuild_index or not manifest_exists:
            if not args.no_progress:
                print(f"building peak index at {index_dir}", file=sys.stderr)
            build_peak_index(args.peaks_dir, index_dir, downcast=_parse_downcast(args.downcast))
        table = extract_xic_indexed(
            index_dir, targets, exact_error=args.exact_error, peaks_dir=args.peaks_dir,
            acquisition_id=args.acquisition_id, level=level, n_isotopes=args.n_isotopes,
            ion_types=ion_types, max_fragment_charge=args.max_fragment_charge,
            neutral_losses=neutral_losses, charge_range=charge_range, rt_window=args.rt_window,
            tolerance=tolerance, tolerance_unit=tolerance_unit,
            all_in_window=rule == "all", assigned_scans_only=args.assigned_scans_only,
            drop_unmatched=args.drop_unmatched,
        )  # fmt: skip
        save_xic(table, args.output_dir, metadata={
            "level": level, "mode": "indexed", "tolerance": tolerance, "tolerance_unit": tolerance_unit,
        })  # fmt: skip
        if not args.no_progress:
            print(f"extracted {table.num_rows} XIC rows → {args.output_dir}", file=sys.stderr)
        return 0

    bundle = (args.peaks_dir / "manifest.json").is_file()
    if bundle:
        _warn_tolerance_against_analyzer(args.peaks_dir, tolerances)
    elif args.mz_axis is not None:
        print("error: --mz-axis needs a converted bundle as --peaks-dir", file=sys.stderr)
        return 2
    gate = None
    if args.gate or any(v is not None for v in (args.min_fragments, args.evidence_top_peaks,
                                                args.min_consecutive_scans, args.max_gap_s,
                                                args.min_isotopes)):  # fmt: skip
        from constellation.massspec.sweep.evidence import EvidenceGate

        default = EvidenceGate()
        top = args.evidence_top_peaks
        gate = EvidenceGate(
            min_fragments=args.min_fragments if args.min_fragments is not None else default.min_fragments,
            top_peaks=default.top_peaks if top is None else (top or None),
            min_consecutive=args.min_consecutive_scans or default.min_consecutive,
            max_gap_s=args.max_gap_s,
            min_isotopes=args.min_isotopes,
        )
    if args.only_passing and gate is None:
        print("error: --only-passing needs a gate (--gate, or --min-fragments and friends)", file=sys.stderr)
        return 2
    on_disk = bundle and (args.workers > 1 or args.work_dir is not None)
    work = (args.work_dir or args.output_dir / ".sweep") if on_disk else None
    progress = None
    if not args.no_progress:
        from constellation.core.progress import StreamProgress

        progress = StreamProgress()
    result = extract_xic(
        args.peaks_dir, targets, acquisition_id=args.acquisition_id, levels=levels,
        tolerances=tolerances,
        ions=IonSettings(n_isotopes=args.n_isotopes, charge_range=charge_range, ion_types=ion_types,
                         max_fragment_charge=args.max_fragment_charge, neutral_losses=neutral_losses),
        rt_window=args.rt_window, rule=rule, assigned_scans_only=args.assigned_scans_only,
        drop_unmatched=args.drop_unmatched, mz_axis=args.mz_axis, gate=gate,
        only_passing=args.only_passing, claims=args.claims, summary=args.summary,
        workers=args.workers if on_disk else 1, work_dir=work, progress=progress,
    )  # fmt: skip
    stamp = extraction_tolerance_metadata(tolerances)
    out = result.write("xic_trace", args.output_dir / "xic_trace.parquet", metadata=stamp)
    tables = ["xic_trace.parquet"]
    for name, flag in (("peak_claims", args.claims), ("query_summary", args.summary)):
        if flag:
            result.write(name, args.output_dir / f"{name}.parquet", metadata=stamp)
            tables.append(f"{name}.parquet")
    metadata = {
        "levels": levels, "mode": "scan_major", "engine": "sweep", "rule": rule,
        "tolerances": {str(lv): {"value": t, "unit": u} for lv, (t, u) in tolerances.items()},
        "mz_axis": result.mz_axis, "tables": tables,
        "gate": None if gate is None else {
            "min_fragments": gate.min_fragments, "top_peaks": gate.top_peaks,
            "min_consecutive": gate.min_consecutive, "max_gap_s": gate.max_gap_s,
            "min_isotopes": gate.min_isotopes, "only_passing": args.only_passing,
        },
    }  # fmt: skip
    if len(levels) == 1:
        metadata.update(level=levels[0], tolerance=tolerances[levels[0]][0],
                        tolerance_unit=tolerances[levels[0]][1])  # fmt: skip
    write_xic_manifest(args.output_dir, metadata=metadata)
    if work is not None and args.work_dir is None:
        shutil.rmtree(work, ignore_errors=True)
    if not args.no_progress:
        import pyarrow.parquet as pq

        rows = pq.ParquetFile(out).metadata.num_rows
        print(f"extracted {rows} XIC rows → {args.output_dir}", file=sys.stderr)
    return 0


def _warn_tolerance_against_analyzer(bundle: Path, tolerances: dict[int, tuple[float, str]]) -> None:
    """Warn when a level's tolerance is in the other unit from what its scans'
    analyzer reads to (ion traps in Da, Orbitraps and TOFs in ppm)."""
    from constellation.massspec.scans import default_tolerance, open_acquisition

    analyzers = open_acquisition(bundle).analyzers
    for level, (tol, unit) in tolerances.items():
        found = analyzers.get(level)
        if not found:
            continue
        usual, usual_unit = default_tolerance(found)
        if usual_unit != unit:
            print(
                f"warning: MS{level} scans were read by {', '.join(found)}, usually matched "
                f"within {usual:g} {usual_unit}; the tolerance given is {tol:g} {unit}",
                file=sys.stderr,
            )


def _warn_level_flag_mismatch(args: argparse.Namespace) -> None:
    """Warn when a flag from the inactive level's group was set."""
    if args.level == "1":
        if args.ion_types != "b,y" or args.max_fragment_charge != _DEFAULT_MAX_FRAG_CHARGE \
                or args.neutral_losses or args.assigned_scans_only:
            print("warning: MS2 fragment options ignored with --level 1", file=sys.stderr)
    elif args.level == "2":
        if args.n_isotopes != _DEFAULT_N_ISOTOPES or tuple(args.charge_range) != _DEFAULT_CHARGE_RANGE:
            print("warning: MS1 isotope options ignored with --level 2", file=sys.stderr)
