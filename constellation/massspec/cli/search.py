"""``constellation massspec search``: EncyclopeDIA search of one acquisition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from constellation.massspec.cli._common import (
    _add_encyclopedia_passthrough_arg,
    _add_jvm_args,
    _add_tolerance_args,
    _format_tolerance_summary,
    _resolve_tolerance_args,
)


# ── search ──────────────────────────────────────────────────────────────


def _build_search_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "search",
        help=(
            "Search one acquisition with EncyclopeDIA and write the result as an "
            "experiment directory: the spectra file Constellation wrote, the "
            "search's own directory with the jar's outputs, and the "
            "identifications, quantities and scores as Parquet tables. Takes a "
            "converted bundle, a Thermo .raw (converted first) or an EncyclopeDIA "
            "spectra file (.dia)."
        ),
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "the acquisition: a `massspec convert` bundle directory, a Thermo .raw "
            "file, or an EncyclopeDIA spectra file (.dia, single run). mzML and "
            "Bruker .d are not supported yet."
        ),
    )
    p.add_argument(
        "--library",
        type=Path,
        default=None,
        help=(
            "spectral library (.dlib, or an .elib chromatogram library). Required "
            "for a peptide-centric search."
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
            "units/<run>-<digest>/ (this search's own directory: a link to the "
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
            "DIA, library optional; not available here yet). Default: what suits "
            "the data, which Constellation infers from the acquisition."
        ),
    )
    p.add_argument(
        "--search-type",
        choices=["auto", "dia", "dda", "dda-chimeric"],
        default=None,
        help="spectrum-centric only: ScribeTwo's -searchType. Default: auto.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "return at once if this very search (same acquisition, library, FASTA, "
            "settings and jar) is already in the experiment. Without this flag it "
            "is run again in its own directory. A search that differs in anything "
            "has a directory of its own either way."
        ),
    )
    p.add_argument(
        "--no-fragments",
        action="store_true",
        help="do not write the fragment table (one row per stored peak)",
    )
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
        help="EncyclopeDIA -numberOfThreadsUsed. Default: the jar's (all cores).",
    )
    _add_jvm_args(p)
    _add_encyclopedia_passthrough_arg(p)
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
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress live stderr streaming of the jar's stdout/stderr",
    )
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
            extra=extra,
        ),
        resolved,
    )


def _cmd_massspec_search(args: argparse.Namespace) -> int:
    """Search one acquisition (see ``massspec.workflows.encyclopedia``).

    Exit status: 0 done (or already done, with ``--resume``); 1 the
    search could not be started as asked (the message says why, and
    nothing was run); 2 the jar finished without its result library;
    otherwise the jar's own non-zero status.
    """
    from constellation.core.manifests import ManifestError
    from constellation.massspec.io.encyclopedia._mirror import MirrorError
    from constellation.massspec.search.encyclopedia.outputs import MissingOutputError
    from constellation.massspec.search.encyclopedia.runner import UnsupportedVersionError
    from constellation.massspec.workflows.encyclopedia import search_acquisition
    from constellation.thirdparty.jvm import JvmRunError
    from constellation.thirdparty.registry import ToolNotFoundError

    try:
        parameters, tolerances = _search_parameters(args)
        print(_format_tolerance_summary(tolerances), file=sys.stderr)
        result = search_acquisition(
            args.input,
            experiment_dir=args.output_dir,
            fasta=args.fasta,
            library=args.library,
            parameters=parameters,
            search_mode=args.search_mode,
            search_type=args.search_type,
            allow_undocumented=tuple(args.allow_undocumented),
            jvm_heap_max=args.jvm_heap_max,
            jvm_heap_min=args.jvm_heap_min,
            jvm_tmpdir=args.jvm_tmpdir,
            annotate_fragments=not args.no_fragments,
            resume=args.resume,
            stream_to_stderr=not args.no_progress,
        )
    except JvmRunError as exc:
        print(f"error: encyclopedia jar exited {exc.returncode}", file=sys.stderr)
        print(f"  see {exc.stderr_log} for the full log", file=sys.stderr)
        return exc.returncode or 1
    except MissingOutputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, FileNotFoundError, ToolNotFoundError, UnsupportedVersionError,
            ManifestError, MirrorError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        return 1

    manifest = result.manifest
    if result.reused:
        print(f"already searched with these inputs and settings: {result.unit_dir}")
        return 0
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
    print(f"experiment: {result.experiment_dir}")
    print(f"  this search: {result.unit_dir}")
    print(f"  tables:      {result.experiment_dir / 'tables'} (unit={result.unit})")
    return 0
