"""``constellation massspec process-dia``: jar-side conversion and merging of spectra files."""

from __future__ import annotations

import argparse
from pathlib import Path

from constellation.massspec.cli._common import (
    _add_encyclopedia_passthrough_arg,
    _add_jvm_args,
    _add_output_dir_arg,
)


# ── process-dia ─────────────────────────────────────────────────────────


def _build_process_dia_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "process-dia",
        help=(
            "Preprocess or merge mass-spec acquisitions into a single "
            ".dia cache. Multi-input is the gas-phase-fractions merge "
            "workflow. Accepts .mzML / .raw / .d / .DIA inputs."
        ),
    )
    p.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        type=Path,
        metavar="PATH",
        help=(
            "one or more input acquisitions. Single input = preprocess; "
            "multiple = merge gas-phase fractions. EncyclopeDIA's jar "
            "takes colon-delimited paths; the wrapper joins for you."
        ),
    )
    p.add_argument(
        "--output-dia",
        required=True,
        type=Path,
        help="output merged .dia path (used in merge mode)",
    )
    _add_output_dir_arg(p)
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip the jar invocation if <output-dir>/_SUCCESS exists",
    )
    _add_jvm_args(p)
    _add_encyclopedia_passthrough_arg(p)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress live stderr streaming of the jar's stdout/stderr",
    )
    p.set_defaults(func=_cmd_massspec_process_dia)


def _cmd_massspec_process_dia(args: argparse.Namespace) -> int:
    """Merge / preprocess one or more spectra files into a single .DIA cache.

    Single-input mode preprocesses; multi-input mode merges gas-phase
    fractions. Writes a ``manifest.json`` with input SHA256s + jar
    version + JVM info + runtime, and touches ``_SUCCESS`` last. No
    auto-ingest — the ``.DIA`` is meant for downstream EncyclopeDIA
    consumption (search / library export), not Constellation-native
    analysis.
    """
    import sys as _sys

    from constellation import __version__ as constellation_version
    from constellation.massspec.search.encyclopedia import (
        build_manifest_envelope,
        encyclopedia_passthrough_args,
        require_min_encyclopedia,
        run_process_dia,
        write_manifest,
    )
    from constellation.thirdparty.jvm import JvmRunError
    from constellation.thirdparty.registry import ToolNotFoundError, find

    inputs = [Path(p).resolve() for p in args.inputs]
    output_dia = Path(args.output_dia).resolve()
    output_dir = Path(args.output_dir).resolve()

    for p in inputs:
        if not p.exists():
            print(f"error: input not found: {p}", file=_sys.stderr)
            return 1
    if len(inputs) > 1 and output_dia is None:
        print(
            "error: multi-input merge mode requires --output-dia",
            file=_sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dia.parent.mkdir(parents=True, exist_ok=True)

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
        result = run_process_dia(
            inputs=inputs,
            output_dia=output_dia,
            output_dir=output_dir,
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

    # run_process_dia relocates the jar-named single-input cache onto
    # output_dia, so by this point --output-dia means the same thing
    # regardless of input count.
    if not output_dia.is_file():
        print(
            f"error: encyclopedia exited 0 but the expected .DIA was not "
            f"produced at {output_dia}; check {result.stderr_log}",
            file=_sys.stderr,
        )
        if len(inputs) == 1:
            # Single input ignores -o, so the file lands by convention.
            # Name every place we looked — otherwise diagnosing a new
            # convention means another round-trip to the cluster.
            print(
                "  single-input mode: the jar chooses the output path "
                "itself. Looked for it at:",
                file=_sys.stderr,
            )
            for cand in (
                output_dir / f"{inputs[0].stem}.dia",
                output_dir / f"{inputs[0].name}.dia",
                inputs[0].with_suffix(".dia"),
                inputs[0].parent / f"{inputs[0].name}.dia",
            ):
                print(f"    {cand}", file=_sys.stderr)
        return 2

    # Manifest captures inputs (with SHA256s) + jar + JVM + runtime.
    import os as _os

    manifest = build_manifest_envelope(
        subcommand="massspec process-dia",
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
        inputs={f"input_{i}": p for i, p in enumerate(inputs)},
        outputs={
            "dia": output_dia,
            "stdout_log": result.stdout_log,
            "stderr_log": result.stderr_log,
        },
        runtime={
            "elapsed_seconds": result.elapsed_seconds,
            "returncode": result.returncode,
            "input_count": len(inputs),
            "input_bytes_total": sum(p.stat().st_size for p in inputs),
            "output_bytes": output_dia.stat().st_size,
        },
        encyclopedia_passthrough_args=extra_args,
    )
    write_manifest(output_dir / "manifest.json", manifest)
    success_path.write_bytes(b"")

    if not args.no_progress:
        size_mb = output_dia.stat().st_size / (1024**2)
        print(
            f"process-dia done: {output_dia} "
            f"({len(inputs)} input{'s' if len(inputs) != 1 else ''} → "
            f"{size_mb:.1f} MiB cache)"
        )
    return 0
