"""``constellation massspec library-export``: EncyclopeDIA's quantitative report across searches."""

from __future__ import annotations

import argparse
from pathlib import Path

from constellation.massspec.cli._common import (
    _add_encyclopedia_passthrough_arg,
    _add_jvm_args,
    _add_output_dir_arg,
    _report_ingest_caveats,
)


# ── library-export ──────────────────────────────────────────────────────


def _build_library_export_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "library-export",
        help=(
            "Combine per-acquisition EncyclopeDIA searches into a "
            "quant-report .elib (or .blib). The standard quant-report "
            "pathway."
        ),
    )
    p.add_argument(
        "--search-dir",
        required=True,
        type=Path,
        help=(
            "directory containing per-acquisition search outputs, or a "
            "single .elib / .dia / .mzML file"
        ),
    )
    p.add_argument(
        "--library",
        required=True,
        type=Path,
        help="original .dlib / .elib that was searched against",
    )
    p.add_argument(
        "--output-elib",
        required=True,
        type=Path,
        help="output consolidated .elib (or .blib with --blib)",
    )
    _add_output_dir_arg(p)
    p.add_argument(
        "--no-align",
        action="store_true",
        help="skip retention-time alignment across input files",
    )
    p.add_argument(
        "--blib",
        action="store_true",
        help="write the consolidated library in BLIB format instead of ELIB",
    )
    p.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help=(
            "original FASTA (required for Pecan / XCorDIA export modes; "
            "not used by the default ELIB-to-ELIB pathway)"
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip the jar invocation if <output-dir>/_SUCCESS exists",
    )
    p.add_argument(
        "--no-ingest",
        action="store_true",
        help=(
            "skip auto-ingesting the consolidated .elib into ParquetDir "
            "bundles (library_pqdir/quant_pqdir/search_pqdir)"
        ),
    )
    _add_jvm_args(p)
    _add_encyclopedia_passthrough_arg(p)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress live stderr streaming of the jar's stdout/stderr",
    )
    p.set_defaults(func=_cmd_massspec_library_export)


def _cmd_massspec_library_export(args: argparse.Namespace) -> int:
    """Combine per-acquisition search results into a quant-report .elib.

    Runs ``-libexport``, optionally ingests the consolidated .elib into
    ``library_pqdir/`` + ``quant_pqdir/`` + ``search_pqdir/`` ParquetDir
    bundles, writes a ``manifest.json``, and touches ``_SUCCESS`` last.
    """
    import sys as _sys

    from constellation import __version__ as constellation_version
    from constellation.massspec.io.encyclopedia import read_encyclopedia
    from constellation.massspec.library import save_library
    from constellation.massspec.quant import save_quant
    from constellation.massspec.search import save_search
    from constellation.massspec.search.encyclopedia import (
        build_manifest_envelope,
        encyclopedia_passthrough_args,
        require_min_encyclopedia,
        run_library_export,
        write_manifest,
    )
    from constellation.thirdparty.jvm import JvmRunError
    from constellation.thirdparty.registry import ToolNotFoundError, find

    search_dir = Path(args.search_dir).resolve()
    library_path = Path(args.library).resolve()
    output_elib = Path(args.output_elib).resolve()
    output_dir = Path(args.output_dir).resolve()
    fasta_path = Path(args.fasta).resolve() if args.fasta is not None else None

    if not search_dir.exists():
        print(f"error: --search-dir not found: {search_dir}", file=_sys.stderr)
        return 1
    if not library_path.is_file():
        print(f"error: --library not found: {library_path}", file=_sys.stderr)
        return 1
    if fasta_path is not None and not fasta_path.is_file():
        print(f"error: --fasta not found: {fasta_path}", file=_sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_elib.parent.mkdir(parents=True, exist_ok=True)

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

    try:
        result = run_library_export(
            search_dir=search_dir,
            library=library_path,
            output_elib=output_elib,
            output_dir=output_dir,
            align=not args.no_align,
            write_blib=args.blib,
            fasta=fasta_path,
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

    if not output_elib.is_file():
        print(
            f"error: encyclopedia exited 0 but the expected output was not "
            f"produced at {output_elib}; check {result.stderr_log}",
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
            ingest_result = read_encyclopedia(output_elib)
        except Exception as exc:  # noqa: BLE001 — surface in manifest
            ingest_info["error"] = f"{type(exc).__name__}: {exc}"
            _write_manifest_for_library_export(
                args=args,
                search_dir=search_dir,
                library_path=library_path,
                fasta_path=fasta_path,
                output_elib=output_elib,
                output_dir=output_dir,
                result=result,
                handle=handle,
                ingest_info=ingest_info,
                library_pqdir=None,
                quant_pqdir=None,
                search_pqdir=None,
                extra_args=extra_args,
                build_manifest_envelope=build_manifest_envelope,
                write_manifest=write_manifest,
                constellation_version=constellation_version,
            )
            print(
                f"error: library-export produced {output_elib} but "
                f"auto-ingest failed: {exc}. Manifest written; pass "
                f"--no-ingest to retry without ingest.",
                file=_sys.stderr,
            )
            return 3
        library_pqdir = output_dir / "library_pqdir"
        save_library(ingest_result.library, library_pqdir, format="parquet_dir")
        _report_ingest_caveats(ingest_result, ingest_info)
        ingest_info["library_counts"] = {
            "proteins": ingest_result.library.proteins.num_rows,
            "peptides": ingest_result.library.peptides.num_rows,
            "precursors": ingest_result.library.precursors.num_rows,
            "fragments": ingest_result.library.fragments.num_rows,
        }
        if ingest_result.quant is not None:
            quant_pqdir = output_dir / "quant_pqdir"
            save_quant(ingest_result.quant, quant_pqdir, format="parquet_dir")
            ingest_info["quant_counts"] = {
                "precursor_quant": (
                    ingest_result.quant.precursor_quant.num_rows
                    if ingest_result.quant.precursor_quant is not None
                    else 0
                ),
                "peptide_quant": (
                    ingest_result.quant.peptide_quant.num_rows
                    if ingest_result.quant.peptide_quant is not None
                    else 0
                ),
                "protein_quant": (
                    ingest_result.quant.protein_quant.num_rows
                    if ingest_result.quant.protein_quant is not None
                    else 0
                ),
            }
        if ingest_result.search is not None:
            search_pqdir = output_dir / "search_pqdir"
            save_search(ingest_result.search, search_pqdir, format="parquet_dir")
            ingest_info["search_counts"] = {
                "peptide_scores": ingest_result.search.peptide_scores.num_rows,
                "protein_scores": ingest_result.search.protein_scores.num_rows,
            }

    # ── manifest + _SUCCESS ─────────────────────────────────────────
    _write_manifest_for_library_export(
        args=args,
        search_dir=search_dir,
        library_path=library_path,
        fasta_path=fasta_path,
        output_elib=output_elib,
        output_dir=output_dir,
        result=result,
        handle=handle,
        ingest_info=ingest_info,
        library_pqdir=library_pqdir,
        quant_pqdir=quant_pqdir,
        search_pqdir=search_pqdir,
        extra_args=extra_args,
        build_manifest_envelope=build_manifest_envelope,
        write_manifest=write_manifest,
        constellation_version=constellation_version,
    )
    success_path.write_bytes(b"")

    if not args.no_progress:
        counts = ingest_info.get("library_counts")
        if isinstance(counts, dict):
            print(
                f"library-export done: {output_elib} "
                f"({counts.get('peptides')} peptides, "
                f"{counts.get('proteins')} proteins)"
            )
        else:
            print(f"library-export done: {output_elib}")
    return 0


def _write_manifest_for_library_export(
    *,
    args: argparse.Namespace,
    search_dir: Path,
    library_path: Path,
    fasta_path: Path | None,
    output_elib: Path,
    output_dir: Path,
    result,  # JvmResult
    handle,  # ToolHandle
    ingest_info: dict[str, object],
    library_pqdir: Path | None,
    quant_pqdir: Path | None,
    search_pqdir: Path | None,
    extra_args,
    build_manifest_envelope,
    write_manifest,
    constellation_version: str,
) -> None:
    import os as _os
    import sys as _sys

    inputs: dict[str, Path] = {
        "search_dir": search_dir,
        "library": library_path,
    }
    if fasta_path is not None:
        inputs["fasta"] = fasta_path
    manifest = build_manifest_envelope(
        subcommand="massspec library-export",
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
            "elib": output_elib,
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
    )
    write_manifest(output_dir / "manifest.json", manifest)
