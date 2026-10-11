"""``constellation massspec search``: EncyclopeDIA library search of one acquisition."""

from __future__ import annotations

import argparse
from pathlib import Path

from constellation.massspec.cli._common import (
    _add_encyclopedia_passthrough_arg,
    _add_jvm_args,
    _add_output_dir_arg,
    _add_tolerance_args,
    _format_tolerance_summary,
    _report_ingest_caveats,
    _resolve_tolerance_args,
)


# ── search ──────────────────────────────────────────────────────────────


def _build_search_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "search",
        help=(
            "DIA library search via EncyclopeDIA. Runs the jar on one "
            "input file against a .dlib/.elib library and writes the "
            "result .elib; auto-ingests into Library/Quant/Search "
            "ParquetDir bundles in the run dir."
        ),
    )
    _add_input_args_for_search(p)
    _add_output_dir_arg(p)
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip the jar invocation if <output-dir>/_SUCCESS exists",
    )
    p.add_argument(
        "--no-ingest",
        action="store_true",
        help=(
            "skip the read_encyclopedia(...) → ParquetDir auto-ingest "
            "after the jar exits"
        ),
    )
    _add_tolerance_args(p)
    p.add_argument(
        "--acquisition",
        default=None,
        help="EncyclopeDIA -acquisition (DIA / DDA). Default: jar default (DIA).",
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
        help="EncyclopeDIA -numberOfThreadsUsed. Default: jar default (20).",
    )
    _add_jvm_args(p)
    _add_encyclopedia_passthrough_arg(p)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress live stderr streaming of the jar's stdout/stderr",
    )
    p.set_defaults(func=_cmd_massspec_search)


def _add_input_args_for_search(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mzml",
        required=True,
        type=Path,
        help=(
            "input mass-spec acquisition file (.mzML, .dia, .raw, or .d "
            "— vendor-native ingest via the bundled MSRawJava in "
            "EncyclopeDIA 6.5.15)"
        ),
    )
    p.add_argument(
        "--library",
        required=True,
        type=Path,
        help=(
            "spectral library (.dlib chromatogram-free or .elib "
            "chromatogram-library)"
        ),
    )
    p.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help=(
            "background proteome FASTA. REQUIRED — EncyclopeDIA 6.5.15's "
            "default search refuses to start without -f ('You are "
            "required to specify an input file (-i), a library file (-l), "
            "and a fasta file (-f)'), even when the library already "
            "carries decoys. Older versions treated it as optional. Use "
            "the same FASTA the library was built from."
        ),
    )


def _cmd_massspec_search(args: argparse.Namespace) -> int:
    """DIA library search via EncyclopeDIA's default mode.

    Runs the jar against an input acquisition (mzML/.dia/.raw/.d) and
    a library (.dlib/.elib), then optionally auto-ingests the produced
    chromatogram .elib into Library/Quant/Search ParquetDir bundles.

    EncyclopeDIA's default search writes its chromatogram .elib
    alongside the input file as ``<input>.elib`` (a side effect we
    can't redirect via -o, which controls only the .encyclopedia.txt
    report). The handler locates the .elib post-run, ingests it, and
    writes the manifest + _SUCCESS to ``--output-dir``.
    """
    import sys as _sys

    from constellation import __version__ as constellation_version
    from constellation.massspec.io.encyclopedia import read_encyclopedia
    from constellation.massspec.library import save_library
    from constellation.massspec.quant.io import save_quant
    from constellation.massspec.search.encyclopedia import (
        build_manifest_envelope,
        encyclopedia_passthrough_args,
        find_search_elib,
        require_min_encyclopedia,
        run_library_search,
        write_manifest,
    )
    from constellation.massspec.search.encyclopedia.percolator import percolator_blocker
    from constellation.massspec.search.io import save_search
    from constellation.thirdparty.jvm import JvmRunError
    from constellation.thirdparty.registry import ToolNotFoundError, find

    input_file = Path(args.mzml).resolve()
    library = Path(args.library).resolve()
    fasta = Path(args.fasta).resolve() if args.fasta is not None else None
    output_dir = Path(args.output_dir).resolve()

    if not input_file.is_file():
        print(f"error: --mzml not found: {input_file}", file=_sys.stderr)
        return 1
    if not library.is_file():
        print(f"error: --library not found: {library}", file=_sys.stderr)
        return 1
    if fasta is not None and not fasta.is_file():
        print(f"error: --fasta not found: {fasta}", file=_sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    success_path = output_dir / "_SUCCESS"
    if success_path.exists() and not args.resume:
        print(
            f"error: --output-dir already complete (touch _SUCCESS exists). "
            f"Pass --resume to re-use it, or delete {output_dir} to start "
            f"fresh.",
            file=_sys.stderr,
        )
        return 1
    if success_path.exists() and args.resume:
        print(f"already complete: {output_dir}")
        return 0

    try:
        handle = find("encyclopedia")
    except ToolNotFoundError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        return 1
    if require_min_encyclopedia(handle):
        return 1

    extra_args = encyclopedia_passthrough_args(args.encyclopedia_arg)

    # The jar runs Percolator only after scoring the whole run; find out
    # now whether the selected build can start on this kernel.
    blocker = percolator_blocker(args.percolator_version, extra_args)
    if blocker is not None:
        print(f"error: {blocker}", file=_sys.stderr)
        return 1

    try:
        tolerances = _resolve_tolerance_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        return 1
    print(_format_tolerance_summary(tolerances), file=_sys.stderr)

    # Report file lands in output_dir; the .elib itself lands beside the input
    # (EncyclopeDIA convention — not redirectable via -o for default search).
    report_path = output_dir / f"{input_file.stem}.encyclopedia.txt"

    try:
        result = run_library_search(
            input_file=input_file,
            library=library,
            fasta=fasta,
            report_output=report_path,
            output_dir=output_dir,
            **tolerances,
            acquisition=args.acquisition,
            enzyme=args.enzyme,
            fragmentation=args.fragmentation,
            percolator_version=args.percolator_version,
            percolator_threshold=args.percolator_threshold,
            percolator_protein_threshold=args.percolator_protein_threshold,
            threads=args.threads,
            jvm_heap_max=args.jvm_heap_max,
            jvm_heap_min=args.jvm_heap_min,
            jvm_tmpdir=args.jvm_tmpdir,
            extra_args=extra_args,
            stream_to_stderr=not args.no_progress,
        )
    except JvmRunError as exc:
        print(f"error: encyclopedia jar exited {exc.returncode}", file=_sys.stderr)
        print(f"  see {exc.stderr_log} for the full log", file=_sys.stderr)
        return exc.returncode

    # Locate the .elib. 6.5.15 writes <input_stem>.elib to the runner's
    # cwd (i.e. output_dir); older versions wrote it next to the input.
    elib_path = find_search_elib(input_file, cwd=output_dir)
    if elib_path is None:
        print(
            f"error: encyclopedia exited 0 but no .elib was found in "
            f"{output_dir} or next to the input at {input_file.parent}; "
            f"check {result.stderr_log}",
            file=_sys.stderr,
        )
        return 2

    # ── auto-ingest ─────────────────────────────────────────────────
    ingest_info: dict[str, object] = {"skipped": bool(args.no_ingest)}
    library_pqdir: Path | None = None
    quant_pqdir: Path | None = None
    search_pqdir: Path | None = None
    if not args.no_ingest:
        try:
            ingest_result = read_encyclopedia(elib_path)
        except Exception as exc:  # noqa: BLE001
            ingest_info["error"] = f"{type(exc).__name__}: {exc}"
            _write_manifest_for_search(
                args=args,
                input_file=input_file,
                library=library,
                fasta=fasta,
                report_path=report_path,
                elib_path=elib_path,
                output_dir=output_dir,
                result=result,
                handle=handle,
                ingest_info=ingest_info,
                library_pqdir=None,
                quant_pqdir=None,
                search_pqdir=None,
                extra_args=extra_args,
                tolerances=tolerances,
                build_manifest_envelope=build_manifest_envelope,
                write_manifest=write_manifest,
                constellation_version=constellation_version,
            )
            print(
                f"error: search produced {elib_path} but auto-ingest "
                f"failed: {exc}. Manifest written; pass --no-ingest to "
                f"retry without ingest.",
                file=_sys.stderr,
            )
            return 3

        library_pqdir = output_dir / "library_pqdir"
        save_library(ingest_result.library, library_pqdir, format="parquet_dir")
        ingest_counts: dict[str, object] = {
            "library": {
                "proteins": ingest_result.library.proteins.num_rows,
                "peptides": ingest_result.library.peptides.num_rows,
                "precursors": ingest_result.library.precursors.num_rows,
                "fragments": ingest_result.library.fragments.num_rows,
            }
        }
        if ingest_result.quant is not None:
            quant_pqdir = output_dir / "quant_pqdir"
            save_quant(ingest_result.quant, quant_pqdir, format="parquet_dir")
            ingest_counts["quant"] = {
                "protein_quant": ingest_result.quant.protein_quant.num_rows,
                "peptide_quant": ingest_result.quant.peptide_quant.num_rows,
                "precursor_quant": ingest_result.quant.precursor_quant.num_rows,
            }
        if ingest_result.search is not None:
            search_pqdir = output_dir / "search_pqdir"
            save_search(ingest_result.search, search_pqdir, format="parquet_dir")
            ingest_counts["search"] = {
                "peptide_scores": ingest_result.search.peptide_scores.num_rows,
                "protein_scores": ingest_result.search.protein_scores.num_rows,
            }
        ingest_info["counts"] = ingest_counts
        _report_ingest_caveats(ingest_result, ingest_info)

    # ── manifest + _SUCCESS ─────────────────────────────────────────
    _write_manifest_for_search(
        args=args,
        input_file=input_file,
        library=library,
        fasta=fasta,
        report_path=report_path,
        elib_path=elib_path,
        output_dir=output_dir,
        result=result,
        handle=handle,
        ingest_info=ingest_info,
        library_pqdir=library_pqdir,
        quant_pqdir=quant_pqdir,
        search_pqdir=search_pqdir,
        extra_args=extra_args,
        tolerances=tolerances,
        build_manifest_envelope=build_manifest_envelope,
        write_manifest=write_manifest,
        constellation_version=constellation_version,
    )
    success_path.write_bytes(b"")

    if not args.no_progress:
        counts = ingest_info.get("counts") if isinstance(ingest_info, dict) else None
        if isinstance(counts, dict):
            lib_c = counts.get("library", {})
            srch_c = counts.get("search", {})
            n_pep = lib_c.get("peptides") if isinstance(lib_c, dict) else None
            n_ppt_scores = srch_c.get("peptide_scores") if isinstance(srch_c, dict) else None
            print(
                f"search done: {elib_path} "
                f"({n_pep} peptides ingested, "
                f"{n_ppt_scores or 0} peptide scores)"
            )
        else:
            print(f"search done: {elib_path}")
    return 0


def _write_manifest_for_search(
    *,
    args: argparse.Namespace,
    input_file: Path,
    library: Path,
    fasta: Path | None,
    report_path: Path,
    elib_path: Path,
    output_dir: Path,
    result,  # JvmResult
    handle,  # ToolHandle
    ingest_info: dict[str, object],
    library_pqdir: Path | None,
    quant_pqdir: Path | None,
    search_pqdir: Path | None,
    extra_args: list[str],
    tolerances: dict[str, object],
    build_manifest_envelope,
    write_manifest,
    constellation_version: str,
) -> None:
    import os as _os
    import sys as _sys

    inputs: dict[str, Path | None] = {
        "input": input_file,
        "library": library,
        "fasta": fasta,
    }
    manifest = build_manifest_envelope(
        subcommand="massspec search",
        constellation_version=constellation_version,
        constellation_argv=_sys.argv,
        java_argv=result.argv,
        tool={
            "name": "encyclopedia",
            "version": handle.version,
            "jar_path": str(handle.path),
            "jar_sha256": result.jar_sha256,
            "source": handle.source,
            "env_var_set": "CONSTELLATION_ENCYCLOPEDIA_HOME" in _os.environ,
            "java_version": result.java_version,
            "java_source": result.java_source,
            "java_path": str(result.java_path),
        },
        inputs=inputs,
        outputs={
            "elib": elib_path,
            "report": report_path,
            "library_pqdir": library_pqdir,
            "quant_pqdir": quant_pqdir,
            "search_pqdir": search_pqdir,
            "stdout_log": result.stdout_log,
            "stderr_log": result.stderr_log,
        },
        runtime={
            "elapsed_seconds": result.elapsed_seconds,
            "returncode": result.returncode,
        },
        ingest=ingest_info,
        encyclopedia_passthrough_args=extra_args,
        # `extras` is a named parameter, NOT **kwargs — passing
        # `search_params=` directly is a TypeError that only surfaces
        # after the jar has already run to completion.
        extras={"search_params": tolerances},
    )
    write_manifest(output_dir / "manifest.json", manifest)
