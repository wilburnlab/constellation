"""``constellation massspec search``: EncyclopeDIA search of acquisitions.

Also home to the argument groups the search-running verbs share
(``report`` and ``quantify`` mount them too): the search settings, the
run controls, and the reporting of what a search did.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from constellation.massspec.cli._common import (
    _add_detach_arg,
    _add_encyclopedia_passthrough_arg,
    _add_jvm_args,
    _add_tolerance_args,
    _detach_if_asked,
    _format_tolerance_summary,
    _note_stopped,
    _resolve_tolerance_args,
)

# ── shared argument groups ──────────────────────────────────────────────


def _add_search_settings(p: argparse.ArgumentParser) -> None:
    """The settings a search runs with: tolerances, enzyme, Percolator,
    threads, the JVM, and the pass-through for any other documented flag."""
    _add_tolerance_args(p)
    p.add_argument(
        "--acquisition",
        default=None,
        help="EncyclopeDIA -acquisition. Default: the jar's (DIA).",
    )
    p.add_argument(
        "--enzyme",
        default=None,
        help="EncyclopeDIA -enzyme (Trypsin, LysC, ...). Default: jar default (trypsin).",
    )
    p.add_argument(
        "--fragmentation",
        default=None,
        help="EncyclopeDIA -frag (CID, HCD, ...). Default: jar default (CID).",
    )
    p.add_argument(
        "--percolator-version",
        default=None,
        help=(
            "EncyclopeDIA -percolatorVersion: one of the builds inside the jar "
            "(v2-10, v3-01, v3-05) or the path of a Percolator executable. "
            "Default: the jar's (v3-01). The bundled v3-01 cannot run on a Linux "
            "kernel booted with vsyscall=none (WSL2's default); the command "
            "says so before starting and lists the ways round it."
        ),
    )
    p.add_argument(
        "--percolator-threshold",
        type=float,
        default=None,
        help="EncyclopeDIA -percolatorThreshold (peptide FDR). Default: jar default (0.01).",
    )
    p.add_argument(
        "--percolator-protein-threshold",
        type=float,
        default=None,
        help="EncyclopeDIA -percolatorProteinThreshold. Default: jar default (0.01).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "EncyclopeDIA -numberOfThreadsUsed, per search. Default: the jar's (all "
            "cores). With --jobs N, give each search about cores/N."
        ),
    )
    _add_jvm_args(p)
    _add_encyclopedia_passthrough_arg(p, bare_flags=False)
    p.add_argument(
        "--allow-undocumented",
        action="append",
        default=[],
        metavar="FLAG",
        help=(
            "let --encyclopedia-arg carry this flag although the jar's help does "
            "not list it (the jar accepts some parameters it does not print). "
            "Repeatable. Every other flag is checked against the pinned jar's help "
            "before Java starts."
        ),
    )


def _add_run_controls(p: argparse.ArgumentParser, *, jobs: bool = True) -> None:
    """Resume, fragments, progress, parallel searches, detaching."""
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip what is already in the experiment with the same inputs, settings "
            "and jar. Without this flag it is run again in its own directory. "
            "Anything that differs has a directory of its own either way, and "
            "nothing already there is replaced."
        ),
    )
    p.add_argument(
        "--no-fragments",
        action="store_true",
        help="do not write the fragment table (one row per stored peak)",
    )
    if jobs:
        p.add_argument(
            "--jobs",
            type=int,
            default=1,
            help=(
                "searches to run at once (default 1). Each is a JVM with its own "
                "--threads and --jvm-heap, so the machine needs --jobs times both. "
                "With more than one, the jar's output is not streamed; each "
                "search's log is in its directory."
            ),
        )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress live stderr streaming of the jar's stdout/stderr",
    )
    _add_detach_arg(p)


# ── search ──────────────────────────────────────────────────────────────


def _build_search_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "search",
        help=(
            "Search acquisitions with EncyclopeDIA and write the results into an "
            "experiment directory: the spectra files Constellation wrote, each "
            "search's own directory with the jar's outputs, and the "
            "identifications, quantities and scores as Parquet tables. Takes "
            "converted bundles, Thermo .raw files (converted first) or EncyclopeDIA "
            "spectra files (.dia); several at once (--jobs runs them in parallel), "
            "or several merged and searched as one (--merge, for gas-phase "
            "fractions)."
        ),
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        nargs="+",
        action="extend",
        help=(
            "the acquisitions: `massspec convert` bundle directories, Thermo .raw "
            "files, or EncyclopeDIA spectra files (.dia, single run). Repeatable, "
            "or several after one --input. mzML and Bruker .d are not supported yet."
        ),
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help=(
            "merge the inputs into one spectra file and search it as one (gas-phase "
            "fractions of a pooled sample, which together make a chromatogram "
            "library). The merged file is registered as an acquisition of its own "
            "that names its fractions. Inputs must be bundles or .raw files."
        ),
    )
    p.add_argument(
        "--merged-name",
        default=None,
        help=(
            "name of the merged spectra file, without .dia. Default: the inputs' "
            "common prefix and suffix around 'merged' (gpf-400-500_100min and "
            "gpf-500-600_100min give gpf-merged_100min)."
        ),
    )
    p.add_argument(
        "--library",
        type=Path,
        default=None,
        help=(
            "spectral library (.dlib, or an .elib chromatogram library). Required "
            "for a peptide-centric search; optional for a spectrum-centric one, "
            "which without it predicts every peptide of the FASTA first (see --ptm, "
            "--prediction-cache)."
        ),
    )
    p.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help=(
            "background proteome FASTA; EncyclopeDIA refuses to start without one. "
            "Use the FASTA the library was built from."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "experiment directory. Holds manifest.json, acquisitions/, "
            "spectra/<run>/ (the spectra file, written once per acquisition), "
            "units/<run>-<digest>/ (each search's own directory: a link to the "
            "spectra file with the jar's outputs beside it as <run>.dia.elib "
            "etc., logs, a manifest, _SUCCESS) and "
            "tables/<table>/unit=<run>-<digest>/acquisition_id=<id>/. The digest "
            "is of the library, FASTA, settings and jar version, so an existing "
            "experiment gains a directory for every new search, of a new "
            "acquisition or of one searched before, and nothing in it is "
            "replaced. Anything that is not an experiment must be new or empty."
        ),
    )
    p.add_argument(
        "--search-mode",
        choices=["peptide-centric", "spectrum-centric"],
        default=None,
        help=(
            "which engine. peptide-centric is the original EncyclopeDIA search "
            "(DIA data, needs a library); spectrum-centric is ScribeTwo (DDA or "
            "DIA, library optional). Default: what suits the data, which "
            "Constellation infers from the acquisition (DDA: spectrum-centric)."
        ),
    )
    p.add_argument(
        "--search-type",
        choices=["auto", "dia", "dda", "dda-chimeric"],
        default=None,
        help="spectrum-centric only: ScribeTwo's -searchType. Default: auto.",
    )
    p.add_argument(
        "--ptm",
        action="append",
        default=[],
        metavar="NAME=STATE",
        help=(
            "spectrum-centric, without a library: a modification for the FASTA "
            "prediction, as ScribeTwo's -ptm<NAME> off|var|fix (e.g. Oxidation=var). "
            "Repeatable. Names: Acetyl, Carbamidomethyl, Deamidation, Dimethyl, GlyGly, "
            "HexNAc, Methyl, Oxidation, Phospho, ProteinNTermAcetyl, PyroGluQ, Succinyl, "
            "TMT, Trimethyl. Not with -fixed / -variable."
        ),
    )
    p.add_argument(
        "--prediction-cache",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "spectrum-centric: where ScribeTwo keeps its FASTA predictions, shared by "
            "every search that uses it (about 3 GB per NCE, charge and DDA/DIA for a "
            "human proteome; the first search fills it, which took 86 min on 12 "
            "cores). Default: the jar's, ~/.encyclopedia. Not part of what a search "
            "is: searches in different caches are the same search."
        ),
    )
    _add_run_controls(p)
    _add_search_settings(p)
    p.set_defaults(func=_cmd_massspec_search)


def _passthrough_flags(items: list[str] | None) -> dict[str, str]:
    """``--encyclopedia-arg -flag=value`` entries as a flag mapping."""
    out: dict[str, str] = {}
    for raw in items or []:
        flag, sep, value = raw.partition("=")
        if not sep:
            raise ValueError(
                f"--encyclopedia-arg {raw!r}: write it as -flag=value (an empty value is "
                f"`-flag=`); EncyclopeDIA has no bare flags"
            )
        if flag in out:
            raise ValueError(f"--encyclopedia-arg sets {flag} twice")
        out[flag] = value
    return out


def _search_parameters(args: argparse.Namespace):
    """The command's settings as :class:`SearchParameters`."""
    from constellation.massspec.search.encyclopedia.params import SearchParameters, Tolerance

    resolved = _resolve_tolerance_args(args)

    def tolerance(stem: str):
        value = resolved.get(f"{stem}_tolerance")
        return None if value is None else Tolerance(float(value), str(resolved[f"{stem}_tolerance_unit"]))

    extra = _passthrough_flags(args.encyclopedia_arg)
    ptms: dict[str, str] = {}
    for raw in getattr(args, "ptm", None) or []:
        name, sep, state = raw.partition("=")
        if not sep:
            raise ValueError(f"--ptm {raw!r}: write it as NAME=STATE (e.g. Oxidation=var)")
        if name in ptms:
            raise ValueError(f"--ptm sets {name} twice")
        ptms[name] = state
    if args.acquisition is not None:
        if "-acquisition" in extra:
            raise ValueError("-acquisition is set both by --acquisition and by --encyclopedia-arg")
        extra["-acquisition"] = args.acquisition
    return (
        SearchParameters(
            precursor_tolerance=tolerance("precursor"),
            fragment_tolerance=tolerance("fragment"),
            library_fragment_tolerance=tolerance("library_fragment"),
            enzyme=args.enzyme,
            fragmentation=args.fragmentation,
            percolator_version=args.percolator_version,
            percolator_threshold=args.percolator_threshold,
            percolator_protein_threshold=args.percolator_protein_threshold,
            threads=args.threads,
            ptms=ptms,
            prediction_cache=getattr(args, "prediction_cache", None),
            extra=extra,
        ),
        resolved,
    )


def _refusals():
    """The errors that mean "could not be started as asked" (exit 1)."""
    from constellation.core.manifests import ManifestError
    from constellation.massspec.io.encyclopedia._mirror import MirrorError
    from constellation.massspec.search.encyclopedia.runner import UnsupportedVersionError
    from constellation.thirdparty.registry import ToolNotFoundError

    return (ValueError, KeyError, FileNotFoundError, ToolNotFoundError, UnsupportedVersionError,
            ManifestError, MirrorError)  # fmt: skip


def _exit_status(exc: BaseException) -> int:
    """Exit status for a failure: the jar's own non-zero status, 2 when
    it left no result, 1 for anything else."""
    from constellation.massspec.search.encyclopedia.outputs import MissingOutputError
    from constellation.thirdparty.jvm import JvmRunError

    if isinstance(exc, JvmRunError):
        return exc.returncode or 1
    if isinstance(exc, MissingOutputError):
        return 2
    return 1


def _describe(exc: BaseException) -> str:
    from constellation.thirdparty.jvm import JvmRunError

    if isinstance(exc, JvmRunError):
        return f"encyclopedia jar exited {exc.returncode}; see {exc.stderr_log} for the full log"
    return str(exc.args[0] if isinstance(exc, KeyError) and exc.args else exc)


def _progress_printer(enabled: bool):
    """A ``SearchProgress`` that says when each search starts and ends."""
    import time

    def progress(event: str, plan, outcome) -> None:
        if not enabled:
            return
        stamp = time.strftime("%H:%M:%S")
        if event == "start":
            print(f"[{stamp}] searching {plan.run} -> units/{plan.unit}", file=sys.stderr, flush=True)
        elif event == "failed":
            print(f"[{stamp}] FAILED {plan.unit}: {_describe(outcome)}", file=sys.stderr, flush=True)

    return progress


def _report_search(result) -> None:
    """What a finished search did, on stdout (notes on stderr)."""
    manifest = result.manifest
    if result.reused:
        print(f"already searched with these inputs and settings: {result.unit_dir}")
        return
    for caveat in manifest.caveats:
        if caveat not in result.disagreements:
            print(f"note: {caveat}", file=sys.stderr)
    for line in result.disagreements:
        print(f"warning: {line}", file=sys.stderr)
    if result.disagreements:
        print(
            "warning: the result library does not record the settings this search was "
            "given; check the jar's log before using it",
            file=sys.stderr,
        )
    rows = manifest.runtime.get("table_rows", {})
    print(
        f"{manifest.search_mode} search of {result.run} (acquisition {result.acquisition_id}): "
        f"{rows.get('precursor', 0):,} precursors, {rows.get('peptide', 0):,} peptides, "
        f"{rows.get('protein_score', 0):,} proteins"
    )
    print(f"  this search: {result.unit_dir}")


def _cmd_massspec_search(args: argparse.Namespace) -> int:
    """Search acquisitions (see ``massspec.workflows.encyclopedia``).

    Exit status: 0 every search done (or already done, with
    ``--resume``); 1 the searches could not be started as asked (the
    message says why, and nothing was run), or more than one failed; 2
    the one search that failed left no result library; otherwise the
    jar's own non-zero status for the one that failed.
    """
    from constellation.massspec.workflows.encyclopedia import search_acquisitions

    detached = _detach_if_asked(args, args.output_dir, "search")
    if detached is not None:
        return detached
    try:
        parameters, tolerances = _search_parameters(args)
        print(_format_tolerance_summary(tolerances), file=sys.stderr)
        fan = search_acquisitions(
            args.input,
            experiment_dir=args.output_dir,
            fasta=args.fasta,
            library=args.library,
            parameters=parameters,
            search_mode=args.search_mode,
            search_type=args.search_type,
            allow_undocumented=tuple(args.allow_undocumented),
            merge=args.merge,
            merged_name=args.merged_name,
            jobs=args.jobs,
            jvm_heap_max=args.jvm_heap_max,
            jvm_heap_min=args.jvm_heap_min,
            jvm_tmpdir=args.jvm_tmpdir,
            annotate_fragments=not args.no_fragments,
            resume=args.resume,
            stream_to_stderr=not args.no_progress,
            progress=_progress_printer(len(args.input) > 1 and not args.merge),
        )
    except _refusals() as exc:
        print(f"error: {_describe(exc)}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, SystemExit):
        _note_stopped("every search that had not finished is")
        raise

    for result in fan.results:
        _report_search(result)
    for plan, exc in fan.failures:
        print(f"error: {plan.unit}: {_describe(exc)}", file=sys.stderr)
    if fan.results:
        print(f"experiment: {fan.results[0].experiment_dir}")
        print(f"  tables:    {fan.results[0].experiment_dir / 'tables'}")
    if not fan.failures:
        return 0
    if len(fan.plans) > 1:
        done = len(fan.results)
        print(
            f"{len(fan.failures)} of {len(fan.plans)} searches failed ({done} done); "
            f"--resume runs only what did not finish",
            file=sys.stderr,
        )
        return 1
    return _exit_status(fan.failures[0][1])
