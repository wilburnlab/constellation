"""``constellation massspec summarize``: what an experiment directory holds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_summarize_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "summarize",
        help=(
            "Say what an experiment directory holds: its acquisitions and, for "
            "each search under units/, what it searched, with which library, "
            "FASTA and settings, whether it finished, what it found, and what "
            "sets it apart from other searches of the same acquisition. Given "
            "one search's directory, shows that search in full (every setting, "
            "the inputs, the command line, the caveats)."
        ),
    )
    p.add_argument(
        "experiment_dir",
        type=Path,
        metavar="DIRECTORY",
        help=(
            "an experiment directory (what `massspec search --output-dir` names), "
            "or one search directory under its units/"
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="print the summary as JSON instead of text",
    )
    p.set_defaults(func=_cmd_massspec_summarize)


def _cmd_massspec_summarize(args: argparse.Namespace) -> int:
    """Exit status: 0, or 2 when the directory is not an experiment or a
    search in one."""
    from constellation.core.manifests import ManifestError
    from constellation.massspec.experiment import format_summary, summarize

    try:
        summary = summarize(args.experiment_dir)
    except (ValueError, ManifestError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        sys.stdout.write(format_summary(summary))
    return 0
