"""CLI parser + handlers for ``constellation massspec <subcommand>``.

Mounted on the root ``constellation`` parser by
``constellation/cli/__main__.py`` (next to ``_build_transcriptome_parser``,
``_build_viz_parser``, etc.). Four subcommands shipped in PR 0 — all
four parsers register so the dashboard's argparse introspector picks
them up, but the handlers raise :class:`NotImplementedError` with a
"filled in PR N" pointer:

==================  ===================================================
Subcommand          Wraps (EncyclopeDIA 6.5.15)
==================  ===================================================
``search``          (default) library search of mzML vs .elib
``predict-library`` ``-convert -fastaToJChronologerLibrary``
``process-dia``     ``-convert -processDIA``
``library-export``  ``-libexport``
==================  ===================================================

The PTM-toggle wrappers (``predict-library``) and per-search Percolator
flags are intentionally left to the implementation PRs — PR 0 surfaces
*only* the args needed to drive the runner end-to-end: input paths,
output dir, JVM heap, escape hatch for unwrapped EncyclopeDIA flags.
Per-utility extension comes in PR 1-4 as wrappers fill in.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_PTM_NAMES = (
    "Acetyl",
    "ProteinNTermAcetyl",
    "Carbamidomethyl",
    "Deamidation",
    "Dimethyl",
    "GlyGly",
    "HexNAc",
    "Methyl",
    "Oxidation",
    "Phospho",
    "PyroGluQ",
    "Succinyl",
    "Trimethyl",
    "TMT",
)


def build_parser(subs: argparse._SubParsersAction) -> None:
    """Mount the ``massspec`` subtree on the parent ``constellation`` parser."""
    p_ms = subs.add_parser(
        "massspec",
        help=(
            "Mass-spectrometry workflows (EncyclopeDIA library search, "
            "FASTA → DLIB prediction, gas-phase fraction merging, quant "
            "reports)"
        ),
    )
    ms_subs = p_ms.add_subparsers(dest="ms_subcommand", required=True)

    _build_search_parser(ms_subs)
    _build_predict_library_parser(ms_subs)
    _build_process_dia_parser(ms_subs)
    _build_library_export_parser(ms_subs)


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
        "--elib-name",
        default=None,
        help="basename for the produced .elib (default: <input-stem>.elib)",
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
            "skip the read_encyclopedia(...) → ParquetDir auto-ingest "
            "after the jar exits"
        ),
    )
    p.add_argument(
        "--fragment-tolerance-ppm",
        type=float,
        default=20.0,
        help="ppm tolerance for the ingest-side fragment annotation",
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
            "background proteome FASTA. Used for decoy generation when "
            "the library lacks decoys; surfaced as a separate arg "
            "because the search wrapper always passes -f."
        ),
    )


# ── predict-library ─────────────────────────────────────────────────────


def _build_predict_library_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "predict-library",
        help=(
            "FASTA → predicted .dlib via EncyclopeDIA 6.5.15's bundled "
            "JChronologer (RT) + Sculptor (CCS/IMS) + Electrician "
            "(charge). In-process PyTorch — no Koina round-trip."
        ),
    )
    p.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="input protein FASTA",
    )
    p.add_argument(
        "--output-dlib",
        required=True,
        type=Path,
        help="output predicted .dlib path",
    )
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
            "of the produced .dlib"
        ),
    )
    p.add_argument(
        "--enzyme",
        default="Trypsin",
        help="enzyme name (default Trypsin)",
    )
    p.add_argument(
        "--min-charge",
        type=int,
        default=1,
        help="minimum precursor charge",
    )
    p.add_argument(
        "--max-charge",
        type=int,
        default=6,
        help="maximum precursor charge",
    )
    p.add_argument(
        "--min-mz",
        type=float,
        default=396.4,
        help="minimum precursor m/z filter",
    )
    p.add_argument(
        "--max-mz",
        type=float,
        default=1002.7,
        help="maximum precursor m/z filter",
    )
    p.add_argument(
        "--max-missed-cleavage",
        type=int,
        default=1,
    )
    p.add_argument(
        "--no-decoys",
        action="store_true",
        help="skip reversed-decoy prediction (halves prediction time)",
    )
    p.add_argument(
        "--no-adjust-nce-for-dia",
        action="store_true",
        help="skip the DIA-mode NCE adjustment EncyclopeDIA applies by default",
    )
    p.add_argument(
        "--default-nce",
        type=int,
        default=33,
        help="normalised collision energy for prediction",
    )
    p.add_argument(
        "--default-charge",
        type=int,
        default=3,
        help="default charge used for NCE adjustment",
    )
    p.add_argument(
        "--max-variable-mods",
        type=int,
        default=1,
    )
    p.add_argument(
        "--max-variable-forms",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--generate-protein-entrapments",
        action="store_true",
        help="generate one shuffled protein-level entrapment per target",
    )
    p.add_argument(
        "--entrapment-seed",
        type=int,
        default=1,
        help="RNG seed for reproducible entrapment generation",
    )
    p.add_argument(
        "--prediction-cache",
        type=Path,
        default=None,
        help=(
            "directory for the shared FASTA prediction cache (default: "
            "EncyclopeDIA's OS-specific user data dir)"
        ),
    )
    p.add_argument(
        "--ragged-n-term",
        action="store_true",
        help="enumerate N-terminal ragged variants",
    )
    # Per-PTM toggles — one --ptm-<name> for each EncyclopeDIA-recognised PTM.
    for name in _PTM_NAMES:
        flag = f"--ptm-{_camel_to_kebab(name)}"
        # Distinct defaults per the jar's `-h` output:
        # ProteinNTermAcetyl and PyroGluQ default to "var",
        # Carbamidomethyl defaults to "fix", everything else "off".
        if name == "Carbamidomethyl":
            default = "fix"
        elif name in ("ProteinNTermAcetyl", "PyroGluQ"):
            default = "var"
        else:
            default = "off"
        p.add_argument(
            flag,
            choices=["off", "var", "fix"],
            default=default,
            help=f"{name} PTM mode",
        )
    _add_jvm_args(p)
    _add_encyclopedia_passthrough_arg(p)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress live stderr streaming of the jar's stdout/stderr",
    )
    p.set_defaults(func=_cmd_massspec_predict_library)


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
    _add_jvm_args(p)
    _add_encyclopedia_passthrough_arg(p)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress live stderr streaming of the jar's stdout/stderr",
    )
    p.set_defaults(func=_cmd_massspec_library_export)


# ── shared arg helpers ──────────────────────────────────────────────────


def _add_output_dir_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "run directory: holds the jar's output file, the auto-ingest "
            "parquet bundles (when applicable), logs/{stdout,stderr}.log, "
            "manifest.json, and the _SUCCESS sentinel"
        ),
    )


_JVM_HEAP_RE = re.compile(r"^\d+[kKmMgGtT]$")


def _jvm_heap_value(raw: str) -> str:
    """argparse type validator for ``--jvm-heap`` / ``--jvm-heap-min``.

    The JVM wants ``<integer><unit>`` (e.g. ``24g``, ``2048m``). A bare
    integer like ``24`` parses as 24 BYTES and crashes with the
    cryptic ``Too small maximum heap``. Reject up front with a clear
    message.
    """
    if not _JVM_HEAP_RE.match(raw):
        raise argparse.ArgumentTypeError(
            f"invalid JVM heap value {raw!r}: must be <integer><unit>, "
            f"e.g. '24g' (24 gigabytes), '2048m' (2048 megabytes). "
            f"A bare number is interpreted as bytes by the JVM and will "
            f"crash with 'Too small maximum heap'."
        )
    return raw


def _add_jvm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--jvm-heap",
        dest="jvm_heap_max",
        type=_jvm_heap_value,
        default="12g",
        help="max JVM heap (-Xmx); default 12g. Suffix required (g/m/k).",
    )
    p.add_argument(
        "--jvm-heap-min",
        type=_jvm_heap_value,
        default=None,
        help="initial JVM heap (-Xms); default unset. Suffix required (g/m/k).",
    )
    p.add_argument(
        "--jvm-tmpdir",
        type=Path,
        default=None,
        help="-Djava.io.tmpdir override (default: system tmp)",
    )


def _add_encyclopedia_passthrough_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--encyclopedia-arg",
        action="append",
        default=[],
        metavar="FLAG=VALUE",
        help=(
            "repeatable escape hatch for unwrapped EncyclopeDIA flags "
            "(e.g. --encyclopedia-arg '-percolatorVersion=v3-05'). Items "
            "without '=' pass through as bare flags."
        ),
    )


def _camel_to_kebab(name: str) -> str:
    """``ProteinNTermAcetyl`` → ``protein-n-term-acetyl``; ``TMT`` → ``tmt``.

    Insert a hyphen (a) before an upper-then-lowercase pair that starts
    a CamelCase word, and (b) after a lowercase letter or digit when
    followed by uppercase. All-caps acronyms (``TMT``) stay glued
    because rule (a) never matches and rule (b) requires a preceding
    lowercase.
    """
    s = re.sub(r"(?<!^)(?=[A-Z][a-z])", "-", name)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", s)
    return s.lower()


# ── handlers (PR 0 stubs) ───────────────────────────────────────────────


def _not_yet_wired(subcommand: str) -> int:
    print(
        f"error: `constellation massspec {subcommand}` is registered for "
        f"dashboard introspection but the runner wiring is not yet "
        f"implemented. See docs/plans/encyclopedia-6.5.15-utilities.md for "
        f"the rollout order.",
        file=sys.stderr,
    )
    return 2


def _cmd_massspec_search(_args: argparse.Namespace) -> int:
    return _not_yet_wired("search")


def _cmd_massspec_predict_library(args: argparse.Namespace) -> int:
    """FASTA → predicted .dlib via EncyclopeDIA's JChronologer pipeline.

    Runs the jar, optionally ingests the produced .dlib into a
    ``library_pqdir/`` ParquetDir bundle, writes a ``manifest.json``
    capturing input SHA256s + jar version + JVM + runtime, and touches
    ``_SUCCESS`` last.
    """
    import sys as _sys

    from constellation import __version__ as constellation_version
    from constellation.massspec.io.encyclopedia import read_encyclopedia
    from constellation.massspec.library import save_library
    from constellation.massspec.search.encyclopedia import (
        SUPPORTED_VERSIONS,
        build_manifest_envelope,
        encyclopedia_passthrough_args,
        run_predict_library,
        write_manifest,
    )
    from constellation.thirdparty.jvm import JvmRunError
    from constellation.thirdparty.registry import ToolNotFoundError, find

    fasta = Path(args.fasta).resolve()
    output_dlib = Path(args.output_dlib).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not fasta.is_file():
        print(f"error: --fasta not found: {fasta}", file=_sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dlib.parent.mkdir(parents=True, exist_ok=True)

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

    # Tool version sanity-check (warn, don't error — wrappers stay
    # forward-compatible with interim builds the user may have access
    # to before a release).
    try:
        handle = find("encyclopedia")
    except ToolNotFoundError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        return 1
    if handle.version is not None and handle.version not in SUPPORTED_VERSIONS:
        print(
            f"warning: running EncyclopeDIA {handle.version}; "
            f"constellation has been validated against "
            f"{sorted(SUPPORTED_VERSIONS)}",
            file=_sys.stderr,
        )

    ptms = {
        "Acetyl": args.ptm_acetyl,
        "ProteinNTermAcetyl": args.ptm_protein_n_term_acetyl,
        "Carbamidomethyl": args.ptm_carbamidomethyl,
        "Deamidation": args.ptm_deamidation,
        "Dimethyl": args.ptm_dimethyl,
        "GlyGly": args.ptm_gly_gly,
        "HexNAc": args.ptm_hex_n_ac,
        "Methyl": args.ptm_methyl,
        "Oxidation": args.ptm_oxidation,
        "Phospho": args.ptm_phospho,
        "PyroGluQ": args.ptm_pyro_glu_q,
        "Succinyl": args.ptm_succinyl,
        "Trimethyl": args.ptm_trimethyl,
        "TMT": args.ptm_tmt,
    }
    extra_args = encyclopedia_passthrough_args(args.encyclopedia_arg)

    try:
        result = run_predict_library(
            fasta=fasta,
            output_dlib=output_dlib,
            output_dir=output_dir,
            ptms=ptms,
            min_charge=args.min_charge,
            max_charge=args.max_charge,
            min_mz=args.min_mz,
            max_mz=args.max_mz,
            max_missed_cleavage=args.max_missed_cleavage,
            enzyme=args.enzyme,
            add_decoys=not args.no_decoys,
            adjust_nce_for_dia=not args.no_adjust_nce_for_dia,
            default_nce=args.default_nce,
            default_charge=args.default_charge,
            max_variable_mods=args.max_variable_mods,
            max_variable_forms=args.max_variable_forms,
            generate_protein_entrapments=args.generate_protein_entrapments,
            entrapment_seed=args.entrapment_seed,
            prediction_cache=args.prediction_cache,
            ragged_n_term=args.ragged_n_term,
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

    if not output_dlib.is_file():
        print(
            f"error: encyclopedia exited 0 but the expected .dlib was not "
            f"produced at {output_dlib}; check {result.stderr_log}",
            file=_sys.stderr,
        )
        return 2

    # ── auto-ingest ─────────────────────────────────────────────────
    ingest_info: dict[str, object] = {"skipped": bool(args.no_ingest)}
    library_pqdir: Path | None = None
    if not args.no_ingest:
        try:
            ingest_result = read_encyclopedia(output_dlib)
        except Exception as exc:  # noqa: BLE001 — surface the ingest error in the manifest
            ingest_info["error"] = f"{type(exc).__name__}: {exc}"
            _write_manifest_for_predict_library(
                args=args,
                fasta=fasta,
                output_dlib=output_dlib,
                output_dir=output_dir,
                result=result,
                handle=handle,
                ingest_info=ingest_info,
                library_pqdir=None,
                extra_args=extra_args,
                build_manifest_envelope=build_manifest_envelope,
                write_manifest=write_manifest,
                constellation_version=constellation_version,
            )
            print(
                f"error: predict-library produced {output_dlib} but "
                f"auto-ingest failed: {exc}. Manifest written; pass "
                f"--no-ingest to retry without ingest.",
                file=_sys.stderr,
            )
            return 3
        library_pqdir = output_dir / "library_pqdir"
        save_library(ingest_result.library, library_pqdir, format="parquet_dir")
        ingest_info["library_counts"] = {
            "proteins": ingest_result.library.proteins.num_rows,
            "peptides": ingest_result.library.peptides.num_rows,
            "precursors": ingest_result.library.precursors.num_rows,
            "fragments": ingest_result.library.fragments.num_rows,
        }

    # ── manifest + _SUCCESS ─────────────────────────────────────────
    _write_manifest_for_predict_library(
        args=args,
        fasta=fasta,
        output_dlib=output_dlib,
        output_dir=output_dir,
        result=result,
        handle=handle,
        ingest_info=ingest_info,
        library_pqdir=library_pqdir,
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
                f"predict-library done: {output_dlib} "
                f"({counts.get('peptides')} peptides, "
                f"{counts.get('precursors')} precursors)"
            )
        else:
            print(f"predict-library done: {output_dlib}")
    return 0


def _write_manifest_for_predict_library(
    *,
    args: argparse.Namespace,
    fasta: Path,
    output_dlib: Path,
    output_dir: Path,
    result,  # JvmResult
    handle,  # ToolHandle
    ingest_info: dict[str, object],
    library_pqdir: Path | None,
    extra_args: list[str],
    build_manifest_envelope,
    write_manifest,
    constellation_version: str,
) -> None:
    import os as _os
    import sys as _sys

    manifest = build_manifest_envelope(
        subcommand="massspec predict-library",
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
        inputs={"fasta": fasta},
        outputs={
            "dlib": output_dlib,
            "library_pqdir": library_pqdir,
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
        SUPPORTED_VERSIONS,
        build_manifest_envelope,
        encyclopedia_passthrough_args,
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
    if handle.version is not None and handle.version not in SUPPORTED_VERSIONS:
        print(
            f"warning: running EncyclopeDIA {handle.version}; "
            f"constellation has been validated against "
            f"{sorted(SUPPORTED_VERSIONS)}",
            file=_sys.stderr,
        )

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

    if not output_dia.is_file():
        print(
            f"error: encyclopedia exited 0 but the expected .DIA was not "
            f"produced at {output_dia}; check {result.stderr_log}",
            file=_sys.stderr,
        )
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


def _cmd_massspec_library_export(_args: argparse.Namespace) -> int:
    return _not_yet_wired("library-export")


__all__ = ["build_parser"]
