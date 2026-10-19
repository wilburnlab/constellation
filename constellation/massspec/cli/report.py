"""``constellation massspec report``: a quantitative report across searches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from constellation.massspec.cli._common import (
    _add_detach_arg,
    _add_encyclopedia_passthrough_arg,
    _add_jvm_args,
    _detach_if_asked,
    _note_stopped,
)


def _build_report_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "report",
        help=(
            "Make EncyclopeDIA's quantitative report (-libexport) across finished "
            "searches of an experiment: retention times aligned across the runs, "
            "every precursor quantified in every run, the whole scored again. "
            "Written to <output-dir>/reports/report-<digest>/ with its tables under "
            "unit=report-<digest>."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="the experiment directory the searches are in",
    )
    p.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help=(
            "the FASTA the searches used. The jar refuses to make a report without "
            "one, whatever its help says."
        ),
    )
    p.add_argument(
        "--search",
        dest="searches",
        action="append",
        default=[],
        metavar="UNIT",
        help=(
            "a search to include, by its directory name under units/ (`massspec "
            "summarize` lists them). Repeatable. Default: every finished "
            "peptide-centric search of a single acquisition, which must all have "
            "used one library."
        ),
    )
    p.add_argument(
        "--library",
        type=Path,
        default=None,
        help="take, by default, only the searches against this library",
    )
    p.add_argument(
        "--no-align",
        action="store_true",
        help="do not align retention times across the runs (the jar's -a false)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="return at once if this very report (same searches and settings) is there",
    )
    p.add_argument(
        "--no-fragments",
        action="store_true",
        help="do not write the fragment table",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress live stderr streaming of the jar's stdout/stderr",
    )
    _add_jvm_args(p)
    _add_encyclopedia_passthrough_arg(p, bare_flags=False)
    p.add_argument(
        "--allow-undocumented",
        action="append",
        default=[],
        metavar="FLAG",
        help="let --encyclopedia-arg carry this flag although the jar's help does not list it",
    )
    _add_detach_arg(p)
    p.set_defaults(func=_cmd_massspec_report)


def _print_report(report) -> None:
    if report.reused:
        print(f"already made with these searches and settings: {report.report_dir}")
        return
    for caveat in report.manifest.caveats:
        print(f"note: {caveat}", file=sys.stderr)
    rows = report.table_rows
    print(
        f"report across {len(report.manifest.units)} searches "
        f"({len(report.manifest.acquisition_ids)} acquisitions): "
        f"{rows.get('precursor', 0):,} precursors, {rows.get('protein_score', 0):,} proteins"
    )
    print(f"  this report: {report.report_dir}")
    print(f"  tables:      {report.experiment_dir / 'tables'} (unit={report.key})")


def _cmd_massspec_report(args: argparse.Namespace) -> int:
    from constellation.massspec.cli.search import _describe, _exit_status, _passthrough_flags, _refusals
    from constellation.massspec.workflows.encyclopedia import report_searches

    detached = _detach_if_asked(args, args.output_dir, "report")
    if detached is not None:
        return detached
    try:
        report = report_searches(
            args.output_dir,
            fasta=args.fasta,
            searches=args.searches or None,
            library=args.library,
            align=not args.no_align,
            extra=_passthrough_flags(args.encyclopedia_arg),
            allow_undocumented=tuple(args.allow_undocumented),
            jvm_heap_max=args.jvm_heap_max,
            jvm_heap_min=args.jvm_heap_min,
            jvm_tmpdir=args.jvm_tmpdir,
            annotate_fragments=not args.no_fragments,
            resume=args.resume,
            stream_to_stderr=not args.no_progress,
        )
    except _refusals() as exc:
        print(f"error: {_describe(exc)}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, SystemExit):
        _note_stopped("the report is")
        raise
    except Exception as exc:  # noqa: BLE001 - the jar's failure, with its status
        print(f"error: {_describe(exc)}", file=sys.stderr)
        return _exit_status(exc)
    _print_report(report)
    return 0
