"""``constellation massspec chromatogram {build-index, extract}``: XIC extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── chromatogram (XIC extraction) ────────────────────────────────────────


#: The charges a target with no charge of its own is looked for at in MS1.
_FALLBACK_CHARGE_RANGE = (1, 4)
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
    common.add_argument("--acquisition-id", type=int, default=None,
                        help="acquisition_id stamped on output rows (FK; default 0), and the run whose "
                             "targets are taken from a source of several (an experiment, PSMs). Targets "
                             "identified in another run than --peaks-dir's are refused without it.")
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

    sweepg = p.add_argument_group("sweep")
    sweepg.add_argument("--layout", choices=("scan-major", "precursor-major"), default="scan-major",
                        help="How the bundle is read: scan-major (its peaks in scan ranges, every match "
                             "shuffled to the targets; output in target order) or precursor-major (its "
                             "precursor_major/ layout a unit of targets at a time, nothing shuffled; the "
                             "same rows per target, in units of targets by precursor m/z). "
                             "Precursor-major scales with the targets, not their matches: the layout for "
                             "a search at proteome scale.")
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
                        help="Also write point_claims.parquet: every peak matched by more than "
                             "one target, or by two channels of one, with each of them.")
    sweepg.add_argument("--summary", action="store_true",
                        help="Also write query_summary.parquet: per target, the evidence gate's "
                             "verdict, the scans it could be matched in and the cells (a channel "
                             "in a scan) matched.")

    gateg = p.add_argument_group(
        "evidence gate (sweep; any of these turns it on)",
        "A target counts as seen when a run of its MS2 scans each match enough distinct "
        "fragment ions among the scan's most intense peaks. At low-resolution tolerances "
        "a bare count passes almost anything; see the summary's counts to set it.",
    )
    gateg.add_argument("--gate", action="store_true",
                       help="Turn the gate on with its defaults (6 fragment ions among each scan's 50 most "
                            "intense peaks, 3 scans in a row).")
    gateg.add_argument("--min-fragments", type=int, default=None,
                       help="Distinct fragment ions (type and number, any charge or loss) a scan "
                            "must show (default 6).")
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

    # Mode B (the m/z-sorted index, target by target) was retired on
    # 2026-11-09; its flags stay to say what replaces them.
    for flag in _REMOVED_EXTRACT_FLAGS:
        p.add_argument(flag, default=None, nargs="?", const=True, help=argparse.SUPPRESS)

    ms1g = p.add_argument_group("MS1 precursor-isotope options (with --level 1 or both)")
    ms1g.add_argument("--n-isotopes", type=int, default=None,
                      help="Isotope peaks M+0..M+(n-1) (default 3).")
    ms1g.add_argument("--top-isotopes", type=int, default=None, metavar="N",
                      help="Each precursor's N most abundant isotopes, from its envelope (for a "
                           "heavy peptide these need not include M+0); instead of --n-isotopes.")
    ms1g.add_argument("--charge-range", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                      help="Match every target in MS1 at each charge LO..HI: the evidence for an "
                           "analyte across its charge states, as when one precursor of a "
                           "peptide was identified and its others are wanted too (a peptide's "
                           "precursors then share their MS1 channels). Counter models a peptide "
                           "across charges and wants --charge-range 1 4. Default: each target "
                           "at its own charge, and 1-4 for a target with none.")
    ms1g.add_argument("--own-charge", action="store_true",
                      help="Match each target in MS1 at its own charge (the default since "
                           "2026-11-07; accepted for scripts that pass it).")

    ms2g = p.add_argument_group("MS2 fragment options (with --level 2 or both)")
    ms2g.add_argument("--ion-types", default=None,
                      help="Comma-separated fragment ion types (default b,y, or the fragment "
                           "library's).")
    ms2g.add_argument("--max-fragment-charge", type=int, default=None,
                      help="Max fragment charge, capped by precursor charge (default 2, or the "
                           "fragment library's).")
    ms2g.add_argument("--fragment-charge-below-precursor", type=int, default=None, metavar="K",
                      help="Fragment charges below the precursor's by at least K: up to z - K for a precursor "
                           "of charge z, at least 1 (EncyclopeDIA: 1; default 0, up to z).")
    ms2g.add_argument("--min-fragment-mz", type=float, default=None, metavar="MZ",
                      help="Make no fragment channel below this m/z, before --top-fragments chooses "
                           "(default none).")
    ms2g.add_argument("--precursor-exclusion", type=float, default=None, metavar="TH",
                      help="Make no fragment channel within this many Th of the precursor m/z, before "
                           "--top-fragments chooses (EncyclopeDIA's XCorr leaves out 10; default none).")
    ms2g.add_argument("--no-acquired-ranges", dest="acquired_ranges", action="store_false",
                      help="Make fragment channels outside the m/z range the MS2 scans acquired too. By "
                           "default a bundle's ranges (by isolation window) are read and no channel is made "
                           "outside them, so --top-fragments chooses among ions that can be measured.")
    ms2g.add_argument("--fragment-abundance", choices=("none", "uniform", "library"), default="none",
                      help="Where fragment channels' abundances come from: none; uniform (equal "
                           "intensity for every ion asked, the classic pre-Prosit assumption); "
                           "or library (a predicted or empirical library's intensities, by "
                           "precursor id; see --fragment-library). Default none.")
    ms2g.add_argument("--fragment-library", type=Path, default=None, metavar="DIR",
                      help="The library (a library directory, or a predict-library output) to take "
                           "fragment abundances from (default: --targets, when it is one).")
    ms2g.add_argument("--top-fragments", type=int, default=None, metavar="N",
                      help="Keep each target's N most abundant fragment ions (EncyclopeDIA traces "
                           "10); needs --fragment-abundance library.")
    ms2g.add_argument("--ion-weights", default=None, metavar="TYPE=W,...",
                      help="With --fragment-abundance uniform, weights per ion type, e.g. a=0.1.")
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


#: Flags of the retired Mode B, and what to say when one is given.
_REMOVED_EXTRACT_FLAGS = {
    "--use-index": "--use-index was removed (Mode B, retired 2026-11-09): read a bundle by isolation "
                   "window with --layout precursor-major, which gives each target the rows the default "
                   "sweep gives it",
    "--rebuild-index": "--rebuild-index was removed with Mode B: --layout precursor-major reads "
                       "<bundle>/precursor_major/ (`massspec precursor-major BUNDLE` writes it)",
    "--index-dir": "--index-dir was removed with Mode B: --layout precursor-major reads the bundle's own "
                   "precursor_major/",
    "--downcast": "--downcast was removed with Mode B: traces are always the bundle's f64 values",
    "--exact-error": "--exact-error was removed with Mode B: traces are always the bundle's f64 values",
}


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

    from constellation.massspec.quant.chromatogram import extract_xic, write_xic_manifest
    from constellation.massspec.quant.targets import load_targets
    from constellation.massspec.sweep import ChannelSettings

    for flag, message in _REMOVED_EXTRACT_FLAGS.items():
        if getattr(args, flag.lstrip("-").replace("-", "_"), None) is not None:
            print(f"error: {message}", file=sys.stderr)
            return 2
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
    def given(value, default):
        return default if value is None else value  # an explicit 0 is a tolerance

    tolerances = {
        level: (
            given(getattr(args, f"ms{level}_tolerance"), args.tolerance),
            given(getattr(args, f"ms{level}_tolerance_unit"), args.tolerance_unit),
        )
        for level in levels
    }
    bad = [f"MS{lv} {tol:g}" for lv, (tol, _) in tolerances.items() if tol < 0]
    if bad:
        print(f"error: a tolerance cannot be negative ({', '.join(bad)})", file=sys.stderr)
        return 2
    if args.min_consecutive_scans is not None and args.min_consecutive_scans < 1:
        print("error: --min-consecutive-scans is at least 1 (a run of none would pass every target)",
              file=sys.stderr)  # fmt: skip
        return 2
    gate_flags = {
        "--gate": args.gate, "--min-fragments": args.min_fragments is not None,
        "--evidence-top-peaks": args.evidence_top_peaks is not None,
        "--min-consecutive-scans": args.min_consecutive_scans is not None,
        "--max-gap-s": args.max_gap_s is not None, "--min-isotopes": args.min_isotopes is not None,
    }  # fmt: skip

    _warn_level_flag_mismatch(args)
    targets = load_targets(args.targets, acquisition_id=args.acquisition_id)
    mismatch = _other_run(targets, args.peaks_dir)
    if mismatch is not None:
        # Until 2026-11-11 one run's identifications were swept against any
        # bundle, stamped acquisition 0 (review of 2026-11-10, C7).
        if args.acquisition_id is None:
            print(f"error: {mismatch}; pass --acquisition-id to sweep them against this run anyway",
                  file=sys.stderr)  # fmt: skip
            return 2
        print(f"warning: {mismatch}; swept against it as --acquisition-id says", file=sys.stderr)
    if args.rt_window is not None:
        from constellation.massspec.sweep.queries import check_rt_window

        try:
            check_rt_window(targets)
        except ValueError as exc:
            print(f"error: --rt-window: {exc}", file=sys.stderr)
            return 2
    if args.own_charge and args.charge_range is not None:
        print("error: --own-charge and --charge-range contradict each other", file=sys.stderr)
        return 2
    charge_range = tuple(args.charge_range) if args.charge_range is not None else None
    if 1 in levels and not args.no_progress:
        if charge_range is not None:
            print(f"MS1: every target at charges {charge_range[0]}-{charge_range[1]}", file=sys.stderr)
        else:
            none = targets.column("precursor_charge").null_count
            lo, hi = _FALLBACK_CHARGE_RANGE
            print(f"MS1: {targets.num_rows - none} targets at their own charge"
                  + (f", {none} without one at charges {lo}-{hi}" if none else ""), file=sys.stderr)  # fmt: skip
    neutral_losses = _parse_downcast(args.neutral_losses) if args.neutral_losses else None
    try:
        source = _fragment_abundance(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    coverage = getattr(source, "coverage", None) or {}
    if args.fragment_abundance == "library" and not coverage and 2 in levels and (
        args.ion_types is None or args.max_fragment_charge is None
    ):
        # Until 2026-11-09 this fell back to b/y at fragment charge 2, which
        # cost 13-16 % of 3+/4+ precursors an ion of their top 10.
        print(
            "error: the fragment library records no coverage (which ion series and fragment charges its "
            "spectra speak for; an empirical library's do not say): give --ion-types and "
            "--max-fragment-charge",
            file=sys.stderr,
        )
        return 2
    # A library's model says which ions it speaks for: unless told
    # otherwise, its ion series and charge ceiling are the channels'.
    ion_types = _resolve_ion_types(args.ion_types or ",".join(coverage.get("ion_types") or ["b", "y"]))
    max_fragment_charge = (
        args.max_fragment_charge if args.max_fragment_charge is not None
        else int(coverage.get("max_fragment_charge") or _DEFAULT_MAX_FRAG_CHARGE)
    )  # fmt: skip
    try:
        settings = ChannelSettings(
            n_isotopes=args.n_isotopes if args.n_isotopes is not None else _DEFAULT_N_ISOTOPES,
            top_isotopes=args.top_isotopes, charge_range=charge_range, ion_types=ion_types,
            max_fragment_charge=max_fragment_charge, neutral_losses=neutral_losses,
            top_fragments=args.top_fragments,
            fragment_charge_below_precursor=given(args.fragment_charge_below_precursor, 0),
            min_fragment_mz=args.min_fragment_mz, precursor_exclusion_th=args.precursor_exclusion,
        )  # fmt: skip
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.top_isotopes is not None and args.n_isotopes is not None:
        print("error: --top-isotopes and --n-isotopes choose the isotopes two ways; give one", file=sys.stderr)
        return 2

    layout = args.layout.replace("-", "_")
    if layout == "precursor_major":
        refused = [flag for flag, on in {"--claims": args.claims, "--min-isotopes": args.min_isotopes is not None}.items()
                   if on]  # fmt: skip
        if refused:
            print(f"error: {', '.join(refused)} need the scan-major sweep (claims need every target that matched "
                  "a point; MS1 isotope evidence ranks a whole MS1 scan): drop --layout precursor-major",
                  file=sys.stderr)  # fmt: skip
            return 2
    bundle = (args.peaks_dir / "manifest.json").is_file()
    if bundle:
        _warn_tolerance_against_analyzer(args.peaks_dir, tolerances)
    elif args.mz_axis is not None:
        print("error: --mz-axis needs a converted bundle as --peaks-dir", file=sys.stderr)
        return 2
    gate = None
    if any(gate_flags.values()):
        from constellation.massspec.sweep.evidence import EvidenceGate

        default = EvidenceGate()
        top = args.evidence_top_peaks
        gate = EvidenceGate(
            min_fragments=given(args.min_fragments, default.min_fragments),
            top_peaks=default.top_peaks if top is None else (top or None),
            min_consecutive=given(args.min_consecutive_scans, default.min_consecutive),
            max_gap_s=args.max_gap_s,
            min_isotopes=args.min_isotopes,
        )
    if gate is not None and 2 in levels and settings.top_fragments is not None \
            and settings.top_fragments < gate.min_fragments:  # fmt: skip
        print(
            f"error: --top-fragments {settings.top_fragments} keeps fewer ions than the gate's "
            f"--min-fragments {gate.min_fragments}, so no query could pass",
            file=sys.stderr,
        )
        return 2
    if args.only_passing and gate is None:
        print("error: --only-passing needs a gate (--gate, or --min-fragments and friends)", file=sys.stderr)
        return 2
    on_disk = bundle and (args.workers > 1 or args.work_dir is not None)
    work = (args.work_dir or args.output_dir / ".sweep") if on_disk else None
    if layout == "precursor_major" and not bundle:
        print("error: --layout precursor-major reads a converted bundle's precursor_major/; give a bundle as "
              "--peaks-dir", file=sys.stderr)  # fmt: skip
        return 2
    progress = None
    if not args.no_progress:
        from constellation.core.progress import StreamProgress

        progress = StreamProgress()
    try:
        result = extract_xic(
            args.peaks_dir, targets, acquisition_id=0 if args.acquisition_id is None else args.acquisition_id,
            levels=levels,
            tolerances=tolerances,
            channels=settings, fragment_abundance=source,
            rt_window=args.rt_window, rule=rule, assigned_scans_only=args.assigned_scans_only,
            drop_unmatched=args.drop_unmatched, mz_axis=args.mz_axis, gate=gate,
            only_passing=args.only_passing, claims=args.claims, summary=args.summary,
            workers=args.workers if on_disk else 1, work_dir=work, progress=progress, layout=layout,
            # A sweep that outgrows memory carries on beside the outputs, as
            # one on disk does, rather than in the system's temporary directory.
            spill_dir=args.output_dir,
            acquired_ranges="bundle" if args.acquired_ranges else None,
        )  # fmt: skip
    except ValueError as exc:
        if layout != "precursor_major":
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # The result stamps each output with the tolerances (Counter reads the trace's).
    out = result.write("xic_trace", args.output_dir / "xic_trace.parquet")
    tables = ["xic_trace.parquet"]
    for name, flag in (("point_claims", args.claims), ("query_summary", args.summary)):
        if flag:
            result.write(name, args.output_dir / f"{name}.parquet")
            tables.append(f"{name}.parquet")
    metadata = {
        "levels": levels, "mode": layout, "engine": "sweep", "rule": rule,
        "tolerances": {str(lv): {"value": t, "unit": u} for lv, (t, u) in tolerances.items()},
        "mz_axis": result.mz_axis, "tables": tables,
        "channels": settings.to_dict(), "fragment_abundance": source.describe(),
        "acquired_ranges": "bundle" if args.acquired_ranges and bundle and 2 in levels else None,
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


def _other_run(targets, peaks_dir: Path) -> str | None:
    """What is wrong when ``targets`` are one run's identifications and
    ``peaks_dir`` is a bundle of another (by the source file each names),
    else None (a source of no run, or a bundle that names none)."""
    from constellation.massspec.quant.targets import same_run, target_run

    run = target_run(targets)
    if run is None or not (peaks_dir / "manifest.json").is_file():
        return None
    from constellation.massspec.acquisitions import acquisition_record_from_bundle

    try:
        swept = str(acquisition_record_from_bundle(peaks_dir)["source_file"])
    except (FileNotFoundError, ValueError):
        return None
    if same_run(run, swept):
        return None
    return (f"the targets are identifications in run {run!r} and --peaks-dir is a bundle of {swept!r}: their "
            f"retention times and scans are the other run's")  # fmt: skip


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


def _fragment_abundance(args: argparse.Namespace):
    """The fragment abundance source the flags ask for (raises ValueError)."""
    from constellation.massspec.peptide.channels import UNMODELLED, Uniform

    choice = args.fragment_abundance
    if args.ion_weights is not None and choice != "uniform":
        raise ValueError("--ion-weights weigh the uniform series: give --fragment-abundance uniform")
    if args.fragment_library is not None and choice != "library":
        raise ValueError("--fragment-library is read with --fragment-abundance library")
    if args.top_fragments is not None and choice != "library":
        raise ValueError(
            "--top-fragments keeps the most abundant ions, which needs abundances that rank them: "
            "give --fragment-abundance library"
        )
    if choice == "uniform":
        weights = None
        if args.ion_weights:
            from constellation.massspec.peptide.ions import IonType

            weights = {}
            for item in args.ion_weights.split(","):
                name, _, value = item.partition("=")
                try:
                    weights[IonType[name.strip().upper()]] = float(value)
                except (KeyError, ValueError):
                    raise ValueError(f"--ion-weights: {item!r} is not TYPE=WEIGHT (a, b, c, x, y, z)") from None
        return Uniform(ion_type_weights=weights)
    if choice == "library":
        from constellation.massspec.library.fragment_abundance import LibraryFragments

        return LibraryFragments.from_directory(args.fragment_library or args.targets)
    return UNMODELLED


def _warn_level_flag_mismatch(args: argparse.Namespace) -> None:
    """Warn when a flag from the inactive level's group was set."""
    if args.level == "1":
        if args.ion_types is not None or args.max_fragment_charge is not None \
                or args.neutral_losses or args.assigned_scans_only \
                or args.fragment_charge_below_precursor is not None or args.min_fragment_mz is not None \
                or args.precursor_exclusion is not None:
            print("warning: MS2 fragment options ignored with --level 1", file=sys.stderr)
    elif args.level == "2":
        if args.n_isotopes is not None or args.top_isotopes is not None or args.charge_range is not None \
                or args.own_charge:
            print("warning: MS1 isotope options ignored with --level 2", file=sys.stderr)
