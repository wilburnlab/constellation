"""``constellation massspec quantify``: the chromatogram-library workflow, end to end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from constellation.massspec.cli._common import _detach_if_asked, _format_tolerance_summary, _note_stopped
from constellation.massspec.cli.search import _add_run_controls, _add_search_settings


def _build_quantify_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "quantify",
        help=(
            "The chromatogram-library workflow in one command: gas-phase fractions "
            "merged and searched as one against a (predicted) library, giving a "
            "chromatogram library; each run searched against that; a quantitative "
            "report across the runs. Every step is the one `search --merge`, "
            "`search` and `report` run, so steps already done are found with "
            "--resume, and each can be run on its own."
        ),
    )
    p.add_argument(
        "--runs",
        required=True,
        type=Path,
        nargs="+",
        action="extend",
        help="the runs to quantify: bundles, Thermo .raw files or spectra files. Repeatable.",
    )
    p.add_argument(
        "--fractions",
        type=Path,
        nargs="+",
        action="extend",
        default=[],
        help=(
            "gas-phase fractions to build the chromatogram library from: bundles or "
            "Thermo .raw files, merged in the order given. Without them the runs are "
            "searched against --library directly."
        ),
    )
    p.add_argument(
        "--library",
        required=True,
        type=Path,
        help="the library the fractions are searched against (or the runs, without fractions)",
    )
    p.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="background proteome FASTA, the one the library was built from",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="the experiment directory (new, or one to add to)",
    )
    p.add_argument(
        "--merged-name",
        default=None,
        help="name of the merged fractions' spectra file (default: from their names)",
    )
    p.add_argument(
        "--collision-filter",
        action="store_true",
        help=(
            "filter the chromatogram library for co-eluting identifications that "
            "share fragments before searching the runs against it. Off by default "
            "(EncyclopeDIA's own workflow does not filter)."
        ),
    )
    p.add_argument("--collision-rt-threshold", type=float, default=5.0,
                   help="collision filter: largest retention-time difference, seconds (default 5)")
    p.add_argument("--collision-fragment-ppm", type=float, default=20.0,
                   help="collision filter: fragment match tolerance, ppm (default 20; widen for ion traps)")
    p.add_argument("--collision-min-shared-ions", type=int, default=4,
                   help="collision filter: shared fragments that make a collision (default 4)")
    p.add_argument("--no-align", action="store_true",
                   help="report: do not align retention times across runs")
    p.add_argument("--no-report", action="store_true",
                   help="stop after the run searches")
    _add_run_controls(p)
    _add_search_settings(p)
    p.set_defaults(func=_cmd_massspec_quantify)


def _cmd_massspec_quantify(args: argparse.Namespace) -> int:
    """Exit status: 0 done; 1 refused before anything ran, or a search
    failed (the report then waits); otherwise the failing step's."""
    from constellation.massspec.cli.report import _print_report
    from constellation.massspec.cli.search import (
        _describe,
        _exit_status,
        _progress_printer,
        _refusals,
        _report_search,
        _search_parameters,
    )
    from constellation.massspec.workflows.encyclopedia import CollisionSettings, quantify

    detached = _detach_if_asked(args, args.output_dir, "quantify")
    if detached is not None:
        return detached
    collision = (
        CollisionSettings(
            rt_threshold_s=args.collision_rt_threshold,
            fragment_ppm=args.collision_fragment_ppm,
            min_shared_ions=args.collision_min_shared_ions,
        )
        if args.collision_filter
        else None
    )
    try:
        parameters, tolerances = _search_parameters(args)
        print(_format_tolerance_summary(tolerances), file=sys.stderr)
        result = quantify(
            args.output_dir,
            runs=args.runs,
            fractions=args.fractions,
            fasta=args.fasta,
            library=args.library,
            merged_name=args.merged_name,
            collision=collision,
            parameters=parameters,
            jobs=args.jobs,
            align=not args.no_align,
            report=not args.no_report,
            allow_undocumented=tuple(args.allow_undocumented),
            jvm_heap_max=args.jvm_heap_max,
            jvm_heap_min=args.jvm_heap_min,
            jvm_tmpdir=args.jvm_tmpdir,
            annotate_fragments=not args.no_fragments,
            resume=args.resume,
            stream_to_stderr=not args.no_progress and args.jobs <= 1,
            progress=_progress_printer(True),
        )
    except _refusals() as exc:
        print(f"error: {_describe(exc)}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, SystemExit):
        _note_stopped("whatever step had not finished is")
        raise
    except Exception as exc:  # noqa: BLE001 - a step's failure, with its status
        print(f"error: {_describe(exc)}", file=sys.stderr)
        return _exit_status(exc)

    if result.library_search is not None:
        print("chromatogram library:")
        _report_search(result.library_search)
        if result.collision_filtered is not None:
            print(f"  collision-filtered: {result.collision_filtered}")
    print("runs:")
    for search in result.runs.results:
        _report_search(search)
    for plan, exc in result.runs.failures:
        print(f"error: {plan.unit}: {_describe(exc)}", file=sys.stderr)
    if result.runs.failures:
        print("the report waits for every run; --resume runs only what did not finish", file=sys.stderr)
        return 1
    if result.report is not None:
        _print_report(result.report)
    return 0
