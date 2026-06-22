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

from constellation.massspec.search.encyclopedia.ptm_defaults import (
    PTM_NAMES as _PTM_NAMES,
    default_for as _ptm_default_for,
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

    _build_convert_parser(ms_subs)
    _build_search_parser(ms_subs)
    _build_predict_library_parser(ms_subs)
    _build_process_dia_parser(ms_subs)
    _build_library_export_parser(ms_subs)
    _build_classify_novel_peptides_parser(ms_subs)
    _build_collision_filter_parser(ms_subs)
    _build_chromatogram_parser(ms_subs)
    _build_counter_parser(ms_subs)


# ── convert ─────────────────────────────────────────────────────────────


def _build_convert_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "convert",
        help=(
            "Convert Thermo .raw file(s) into directory bundles "
            "(manifest.json + peaks.parquet + scan_metadata.parquet + "
            "acquisition_metadata.parquet). Accepts a single .raw file "
            "OR a directory of .raw files; --threads fans the directory "
            "out across worker subprocesses. Dispatches on the input "
            "suffix; .raw routes to constellation.massspec.readers.thermo.convert. "
            "Future mzML / Bruker / Sciex converters slot in transparently."
        ),
    )
    p.add_argument(
        "input",
        type=Path,
        help=(
            "Path to a .raw file OR a directory containing .raw files "
            "(non-recursive glob, case-insensitive)."
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory for the output bundle(s). Each bundle dir "
            "is <output-dir>/<input-stem>/. Defaults to the input file's "
            "parent directory (single-file mode) or the input directory "
            "itself (directory mode)."
        ),
    )
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "Number of parallel converter workers (default 1). Each "
            "worker bootstraps its own .NET CLR; useful when converting "
            "a directory of .raw files. Spawn-mode subprocesses, not "
            "threads — fork-after-CLR-load is a documented pythonnet "
            "deadlock class."
        ),
    )
    p.add_argument(
        "--rt-bin-width",
        dest="rt_bin_width_s",
        type=float,
        default=60.0,
        help=(
            "PyArrow row-group RT chunking (seconds). The peak-table "
            "ParquetWriter flushes a new row group every N seconds of "
            "retention time, so downstream RT-window readers get "
            "row-group-level filter pushdown. Purely a write-time "
            "knob — no biological binning happens. Default: 60.0"
        ),
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Force GetSegmentedScanFromScanNumber unconditionally — "
            "preserves Thermo's raw FT profile grid. Default is centroid "
            "(GetCentroidStream when available, with per-peak "
            "resolution/noise/baseline)."
        ),
    )
    p.add_argument(
        "--no-trailer-extras",
        dest="capture_trailer_extras",
        action="store_false",
        help=(
            "Skip capturing the verbatim trailer-extras map. Typed "
            "trailer columns remain populated; only the catch-all "
            "string map is dropped."
        ),
    )
    p.add_argument(
        "--no-sha256",
        dest="compute_sha256",
        action="store_false",
        help=(
            "Skip computing the SHA-256 of the input .raw. Saves I/O on "
            "large files; manifest.source_sha256 is null in that case."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing non-empty bundle directory. Without "
            "this, single-file mode refuses to clobber and exits with "
            "an error; batch mode (directory input or --threads > 1) "
            "treats existing bundles as skipped instead."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Scans per internal ScanBatch (default 64).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress progress messages on stderr.",
    )
    p.set_defaults(func=_cmd_massspec_convert)


def _cmd_massspec_convert(args: argparse.Namespace) -> int:
    """``constellation massspec convert <input>`` dispatcher.

    Routes to:

    - Legacy single-file path (byte-identical to pre-batch behavior)
      when ``input`` is a file AND ``--threads <= 1``.
    - Batch path (directory glob + spawn-mode ProcessPoolExecutor)
      when ``input`` is a directory OR ``--threads > 1``.
    """
    input_path: Path = args.input
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    if input_path.is_file() and args.threads <= 1:
        return _cmd_massspec_convert_single(args, input_path)
    return _cmd_massspec_convert_batch(args, input_path)


def _cmd_massspec_convert_single(
    args: argparse.Namespace,
    input_path: Path,
) -> int:
    """Legacy single-file convert path. Behavior unchanged from pre-batch."""
    suffix = input_path.suffix.lower()
    parent = args.output_dir if args.output_dir is not None else input_path.parent
    bundle_dir = parent / input_path.stem

    progress_cb = None
    if not args.no_progress:
        from constellation.core.progress import StreamProgress

        progress_cb = StreamProgress()

    if suffix == ".raw":
        from constellation.massspec.readers.thermo import convert as thermo_convert
        from constellation.massspec.readers.thermo._netruntime import require_thermo

        try:
            require_thermo()
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3

        try:
            manifest = thermo_convert(
                input_path,
                bundle_dir,
                rt_bin_width_s=args.rt_bin_width_s,
                profile=args.profile,
                capture_trailer_extras=args.capture_trailer_extras,
                batch_size=args.batch_size,
                compute_sha256=args.compute_sha256,
                force=args.force,
                progress_cb=progress_cb,
            )
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 4
        except Exception as exc:  # noqa: BLE001
            print(
                f"error: thermo conversion failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 5

        print(f"wrote bundle: {bundle_dir}")
        print(f"  manifest:                {bundle_dir / 'manifest.json'}")
        print(f"  peaks:                   {bundle_dir / manifest.outputs['peaks']}")
        print(
            f"  scan_metadata:           "
            f"{bundle_dir / manifest.outputs['scan_metadata']}"
        )
        print(
            f"  acquisition_metadata:    "
            f"{bundle_dir / manifest.outputs['acquisition_metadata']}"
        )
        return 0

    print(
        f"error: no MS converter registered for suffix {suffix!r}\n"
        f"  v1 supports .raw (Thermo). mzML / Bruker .d / Sciex .wiff land "
        f"as separate readers when their converters ship.",
        file=sys.stderr,
    )
    return 2


def _collect_raw_files(input_dir: Path) -> list[Path]:
    """Top-level non-recursive glob for ``*.raw`` + ``*.RAW``.

    Deduplicates by resolved path (handles case-insensitive filesystems
    where a single file matches both globs) and sorts by name for
    deterministic order across runs.
    """
    matches: dict[Path, Path] = {}
    for pattern in ("*.raw", "*.RAW"):
        for p in input_dir.glob(pattern):
            if p.is_file():
                matches.setdefault(p.resolve(), p)
    return sorted(matches.values(), key=lambda p: p.name)


def _cmd_massspec_convert_batch(
    args: argparse.Namespace,
    input_path: Path,
) -> int:
    """Batch convert path: directory glob OR --threads > 1.

    Exits:
      0 — all results ``ok`` or ``skipped``
      2 — input directory contains no ``.raw`` files
      3 — pythonnet / DLLs missing (parent fails fast)
      5 — at least one per-file result is ``error``
    """
    from constellation.massspec.readers.thermo import convert_batch
    from constellation.massspec.readers.thermo._netruntime import require_thermo

    try:
        require_thermo()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if input_path.is_dir():
        paths = _collect_raw_files(input_path)
        if not paths:
            print(
                f"error: no .raw files in {input_path}",
                file=sys.stderr,
            )
            return 2
        default_parent = input_path
    else:
        # Single file routed through batch because --threads > 1.
        if input_path.suffix.lower() != ".raw":
            print(
                f"error: no MS converter registered for suffix "
                f"{input_path.suffix.lower()!r}\n"
                f"  v1 supports .raw (Thermo). mzML / Bruker .d / Sciex "
                f".wiff land as separate readers when their converters "
                f"ship.",
                file=sys.stderr,
            )
            return 2
        paths = [input_path]
        default_parent = input_path.parent

    output_parent = (
        args.output_dir if args.output_dir is not None else default_parent
    )

    progress_cb = None
    if not args.no_progress:
        from constellation.core.progress import StreamProgress

        progress_cb = StreamProgress()

    results = convert_batch(
        paths,
        output_parent,
        n_workers=args.threads,
        force=args.force,
        rt_bin_width_s=args.rt_bin_width_s,
        profile=args.profile,
        capture_trailer_extras=args.capture_trailer_extras,
        batch_size=args.batch_size,
        compute_sha256=args.compute_sha256,
        progress_cb=progress_cb,
    )

    n_ok = sum(1 for r in results if r.status == "ok")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_err = sum(1 for r in results if r.status == "error")
    print(
        f"batch convert: {n_ok} succeeded, {n_skip} skipped, {n_err} failed",
        file=sys.stderr,
    )
    if n_err:
        print("  failed:", file=sys.stderr)
        for r in results:
            if r.status == "error":
                print(f"    {r.input_path}: {r.detail}", file=sys.stderr)
        return 5
    return 0


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
    p.add_argument(
        "--fragment-tolerance-ppm",
        type=float,
        default=None,
        help=(
            "EncyclopeDIA -ftol value (ppm). Default: jar's built-in "
            "default (10 ppm in 6.5.15)."
        ),
    )
    p.add_argument(
        "--precursor-tolerance-ppm",
        type=float,
        default=None,
        help=(
            "EncyclopeDIA -ptol value (ppm). Default: jar's built-in "
            "default (10 ppm in 6.5.15)."
        ),
    )
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
            "EncyclopeDIA -percolatorVersion (v2-10, v3-01, v3-05). "
            "Default: jar default (v3-01 in 6.5.15)."
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
        required=False,
        default=None,
        type=Path,
        help=(
            "background proteome FASTA (optional). Used for decoy "
            "generation when the library lacks decoys. EncyclopeDIA's "
            "default search does not require it when the library "
            "already contains decoys (e.g. predict-library output "
            "with -addDecoys true), but providing it never hurts."
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
    # Per-PTM toggles — one --ptm-<name> for each EncyclopeDIA-recognised
    # PTM. Defaults imported from
    # ``massspec.search.encyclopedia.ptm_defaults`` so the standalone CLI
    # and ``transcriptome-to-proteome`` cannot drift.
    for name in _PTM_NAMES:
        flag = f"--ptm-{_camel_to_kebab(name)}"
        default = _ptm_default_for(name)
        p.add_argument(
            flag,
            choices=["off", "var", "fix"],
            default=default,
            help=f"{name} PTM mode (off|var|fix; default {default})",
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


# ── classify-novel-peptides ─────────────────────────────────────────────


def _build_classify_novel_peptides_parser(
    subs: argparse._SubParsersAction,
) -> None:
    p = subs.add_parser(
        "classify-novel-peptides",
        help=(
            "Classify detected novel peptides against a reference-vs-novel "
            "mmseqs2 alignment. Walks each per-hit CIGAR string and assigns "
            "one of 11 deviation classes (snp / insertion / deletion / "
            "complex / truncations / cutsite mutation / deviations / "
            "unknown / non_reference). Ports the cartographer "
            "nanopore.classify_novel_peptides workflow."
        ),
    )
    p.add_argument(
        "--search-results",
        required=True,
        type=Path,
        help=(
            "Path to a ParquetDir Search bundle, OR a .elib SQLite file. "
            "Only the detected-peptide sequence set is consulted (column "
            "PeptideSeq for .elib; peptide_id+modseq-joined-to-library for "
            "ParquetDir)."
        ),
    )
    p.add_argument(
        "--library",
        required=True,
        type=Path,
        help=(
            "Path to a ParquetDir Library bundle. Used to resolve "
            "peptide_id ↔ peptide_sequence when --search-results is a "
            "ParquetDir Search."
        ),
    )
    p.add_argument(
        "--alignment-hits",
        required=True,
        type=Path,
        help=(
            "Path to an AlignmentHits ParquetDir bundle, OR a headerless "
            "mmseqs2 .tab TSV with the canonical 8-column layout "
            "(query, target, evalue, qstart, qend, tstart, tend, cigar). "
            "May contain mixed reference + swissprot hits per query — "
            "tier inference happens internally based on target membership "
            "in the reference proteome."
        ),
    )
    p.add_argument(
        "--reference-fasta",
        required=True,
        type=Path,
        help=(
            "Reference proteome FASTA. Also serves as the tier-inference "
            "source: any hit whose target accession is not in this FASTA "
            "is treated as non_reference and short-circuits to that "
            "classification."
        ),
    )
    p.add_argument(
        "--novel-fasta",
        required=True,
        type=Path,
        help="Novel proteome FASTA (transcript-derived novel proteins).",
    )
    p.add_argument(
        "--annotation-fasta",
        type=Path,
        default=None,
        action="append",
        help=(
            "Optional additional FASTA(s) scanned for gene-symbol "
            "annotations to populate the `gene` column. Recognises "
            "bracketed `[gene=SYMBOL]` tokens (constellation "
            "combined.fasta convention), bare `gene=SYMBOL` tokens "
            "(cartographer convention), and `GN=SYMBOL` (UniProt / "
            "SwissProt). Pass once per file; can be repeated."
        ),
    )
    p.add_argument(
        "--protein-counts",
        type=Path,
        default=None,
        help=(
            "Optional path to counts_tpm.parquet (PROTEIN_COUNT_TABLE "
            "schema: protein_id, sample_name, tpm, ...). When supplied, "
            "the classifier picks the most-abundant `protein_id` "
            "(highest mean-total TPM across samples) as the "
            "representative when multiple novel proteins produce the "
            "same peptide with the same classification."
        ),
    )
    _add_output_dir_arg(p)
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip the classification if <output-dir>/_SUCCESS exists",
    )
    p.add_argument(
        "--enzyme",
        default="Trypsin",
        help="enzyme for tryptic digestion (default Trypsin)",
    )
    p.add_argument(
        "--max-missed-cleavages",
        type=int,
        default=1,
        help="max missed cleavages in tryptic digest (default 1)",
    )
    p.add_argument(
        "--min-peptide-length",
        type=int,
        default=7,
        help="min peptide length kept in the digest (default 7)",
    )
    p.add_argument(
        "--max-peptide-length",
        type=int,
        default=50,
        help="max peptide length kept in the digest (default 50)",
    )
    p.add_argument(
        "--collision-filter",
        action="store_true",
        help=(
            "Apply cartographer's DIA collision filter: drop detected "
            "modseqs that lose co-elution clusters in the .elib's "
            "entries table. Requires --dia and a .elib --search-results. "
            "Opt-in pending v6.5.15-sweep validation of whether the "
            "newer search internalises this filter."
        ),
    )
    p.add_argument(
        "--dia",
        type=Path,
        default=None,
        help=(
            "Path to combined.dia (the merged-fractions DIA file). "
            "Required when --collision-filter is set; supplies the GPF "
            "isolation windows."
        ),
    )
    p.add_argument(
        "--collision-rt-threshold-s",
        type=float,
        default=5.0,
        help="--collision-filter: max |ΔRT| (s) for a pair (default 5.0)",
    )
    p.add_argument(
        "--collision-frag-ppm-tol",
        type=float,
        default=20.0,
        help="--collision-filter: fragment m/z tolerance ppm (default 20.0)",
    )
    p.add_argument(
        "--collision-min-shared-ions",
        type=int,
        default=4,
        help="--collision-filter: min shared ions to flag (default 4)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress progress messages on stderr",
    )
    p.set_defaults(func=_cmd_massspec_classify_novel_peptides)


# ── collision-filter ────────────────────────────────────────────────────


def _build_collision_filter_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "collision-filter",
        help=(
            "Apply the DIA collision filter to a search .elib and write a "
            "filtered .elib with the collided identifications removed. "
            "Drops co-eluting peptides in the same GPF isolation window "
            "that share fragment ions (keeping the lowest-Score per "
            "cluster), at the SQLite level so the output is consumable by "
            "library-export."
        ),
    )
    p.add_argument(
        "--elib",
        required=True,
        type=Path,
        help="search .elib to filter (the EncyclopeDIA search output)",
    )
    p.add_argument(
        "--dia",
        required=True,
        type=Path,
        help=(
            "combined .dia file — supplies the GPF isolation-window ranges "
            "the filter groups co-eluting peptides by"
        ),
    )
    p.add_argument(
        "--output-elib",
        required=True,
        type=Path,
        help="destination for the collision-filtered .elib",
    )
    _add_output_dir_arg(p)
    p.add_argument(
        "--rt-threshold-s",
        type=float,
        default=5.0,
        help="max |ΔRT| (s) for a candidate co-elution pair (default 5.0)",
    )
    p.add_argument(
        "--frag-ppm-tol",
        type=float,
        default=20.0,
        help="fragment m/z match tolerance in ppm (default 20.0)",
    )
    p.add_argument(
        "--min-shared-ions",
        type=int,
        default=4,
        help="min shared observed fragment ions to flag a pair (default 4)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip if <output-dir>/_SUCCESS exists",
    )
    p.add_argument(
        "--no-ingest",
        action="store_true",
        help=(
            "skip auto-ingesting the filtered .elib into ParquetDir bundles "
            "(library_pqdir / quant_pqdir / search_pqdir)"
        ),
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress progress messages on stderr",
    )
    p.set_defaults(func=_cmd_massspec_collision_filter)


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
            fragment_tolerance_ppm=args.fragment_tolerance_ppm,
            precursor_tolerance_ppm=args.precursor_tolerance_ppm,
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
    )
    write_manifest(output_dir / "manifest.json", manifest)


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
        build_manifest_envelope,
        encyclopedia_passthrough_args,
        require_min_encyclopedia,
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
    if require_min_encyclopedia(handle):
        return 1

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


def _cmd_massspec_classify_novel_peptides(args: argparse.Namespace) -> int:
    """Classify detected novel peptides against a reference-vs-novel
    mmseqs2 alignment and write a NOVEL_PEPTIDE_TABLE-shaped bundle.
    """
    import json as _json
    import sys as _sys

    from constellation import __version__ as constellation_version
    from constellation.core.io.schemas import (
        ALIGNMENT_HIT_TABLE,
        read_mmseqs_tab,
    )
    from constellation.massspec.search import (
        apply_collision_filter,
        build_gene_map_from_fasta_headers,
        classify_novel_peptides,
        read_fasta_proteins,
        save_novel_peptides,
    )

    search_results = Path(args.search_results).resolve()
    library_path = Path(args.library).resolve()
    alignment_hits = Path(args.alignment_hits).resolve()
    reference_fasta = Path(args.reference_fasta).resolve()
    novel_fasta = Path(args.novel_fasta).resolve()
    output_dir = Path(args.output_dir).resolve()
    dia_path: Path | None = (
        Path(args.dia).resolve() if args.dia is not None else None
    )

    if args.collision_filter:
        if dia_path is None:
            print(
                "error: --collision-filter requires --dia (path to combined.dia)",
                file=_sys.stderr,
            )
            return 1
        if search_results.suffix.lower() != ".elib":
            print(
                "error: --collision-filter requires a .elib --search-results "
                "(observed fragment m/z lives in the elib entries table)",
                file=_sys.stderr,
            )
            return 1

    for label, p in [
        ("--search-results", search_results),
        ("--library", library_path),
        ("--alignment-hits", alignment_hits),
        ("--reference-fasta", reference_fasta),
        ("--novel-fasta", novel_fasta),
    ]:
        if not p.exists():
            print(f"error: {label} not found: {p}", file=_sys.stderr)
            return 1
    if dia_path is not None and not dia_path.exists():
        print(f"error: --dia not found: {dia_path}", file=_sys.stderr)
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

    # 1. Load detected peptides
    if not args.no_progress:
        print(
            f"[classify-novel-peptides] loading detected peptides from "
            f"{search_results} ...",
            file=_sys.stderr,
        )
    detected_peptides = _load_detected_peptides(search_results, library_path)

    # 1b. Optional: drop collision-loser modseqs before classification.
    collision_metadata: dict | None = None
    if args.collision_filter:
        if not args.no_progress:
            print(
                f"[classify-novel-peptides] running collision filter "
                f"(rt<{args.collision_rt_threshold_s}s, "
                f"{args.collision_frag_ppm_tol} ppm, "
                f">={args.collision_min_shared_ions} shared ions) "
                f"on {search_results} + {dia_path} ...",
                file=_sys.stderr,
            )
        losers, collision_metadata = apply_collision_filter(
            elib_path=search_results,
            dia_path=dia_path,
            rt_threshold_s=args.collision_rt_threshold_s,
            frag_ppm_tol=args.collision_frag_ppm_tol,
            min_shared_ions=args.collision_min_shared_ions,
            return_metadata=True,
        )
        if losers and detected_peptides.num_rows > 0:
            detected_peptides = _drop_collision_losers(detected_peptides, losers)
        collision_metadata["n_losers"] = len(losers)
        if not args.no_progress:
            print(
                f"[classify-novel-peptides] collision filter dropped "
                f"{len(losers):,} modseqs across "
                f"{len(collision_metadata['clusters']):,} clusters; "
                f"{detected_peptides.num_rows:,} detected peptides remain",
                file=_sys.stderr,
            )

    # 2. Load alignments
    if not args.no_progress:
        print(
            f"[classify-novel-peptides] loading alignments from "
            f"{alignment_hits} ...",
            file=_sys.stderr,
        )
    if alignment_hits.is_dir():
        # ParquetDir bundle
        import pyarrow.parquet as pq
        candidate = alignment_hits / "alignment_hits.parquet"
        if not candidate.is_file():
            candidate = alignment_hits / "alignments.parquet"
        if not candidate.is_file():
            print(
                "error: --alignment-hits dir has no recognised parquet file "
                "(looked for alignment_hits.parquet and alignments.parquet)",
                file=_sys.stderr,
            )
            return 1
        alignments = pq.read_table(candidate)
    else:
        alignments = read_mmseqs_tab(alignment_hits)
    # Ensure schema match before passing to the classifier
    from constellation.core.io.schemas import cast_to_schema
    alignments = cast_to_schema(alignments, ALIGNMENT_HIT_TABLE)

    # 3. Load proteomes
    if not args.no_progress:
        print(
            "[classify-novel-peptides] loading reference + novel proteomes ...",
            file=_sys.stderr,
        )
    reference_proteins = read_fasta_proteins(reference_fasta)
    novel_proteins = read_fasta_proteins(novel_fasta)
    # The bare reference/novel FASTAs typically don't carry gene tags.
    # An optional --annotation-fasta (combined.fasta with [gene=...] tags
    # or SwissProt with GN=...) is the actual source for the gene map.
    annotation_fastas = args.annotation_fasta or []
    gene_map_inputs = [reference_fasta, novel_fasta, *annotation_fastas]
    gene_map = build_gene_map_from_fasta_headers(gene_map_inputs)
    if not args.no_progress:
        print(
            f"[classify-novel-peptides] gene_map: "
            f"{len(gene_map):,} accessions tagged "
            f"from {len(gene_map_inputs)} FASTA(s)",
            file=_sys.stderr,
        )

    # Optional per-protein abundance map (highest-abundance ORF wins
    # the row when multiple novel proteins produce the same peptide).
    protein_abundance: dict[str, float] | None = None
    if args.protein_counts is not None:
        if not args.no_progress:
            print(
                f"[classify-novel-peptides] loading protein abundance from "
                f"{args.protein_counts}",
                file=_sys.stderr,
            )
        import pyarrow.parquet as _pq
        counts_tbl = _pq.read_table(args.protein_counts)
        # PROTEIN_COUNT_TABLE has columns: protein_id, sample_name, count, tpm
        # Aggregate to protein_id -> mean tpm across samples.
        ids = counts_tbl.column("protein_id").to_pylist()
        tpms = counts_tbl.column("tpm").to_pylist()
        from collections import defaultdict as _dd
        sums: dict[str, float] = _dd(float)
        n_obs: dict[str, int] = _dd(int)
        for pid, t in zip(ids, tpms):
            if pid is None or t is None:
                continue
            sums[pid] += float(t)
            n_obs[pid] += 1
        protein_abundance = {
            pid: sums[pid] / n_obs[pid] for pid in sums
        }
        if not args.no_progress:
            print(
                f"[classify-novel-peptides] protein_abundance: "
                f"{len(protein_abundance):,} proteins with mean TPM",
                file=_sys.stderr,
            )

    # 4. Classify
    if not args.no_progress:
        print(
            f"[classify-novel-peptides] classifying "
            f"{detected_peptides.num_rows:,} detected peptides against "
            f"{alignments.num_rows:,} alignment hits "
            f"(novel proteins: {novel_proteins.num_rows:,}; "
            f"reference proteins: {reference_proteins.num_rows:,}) ...",
            file=_sys.stderr,
        )
    result = classify_novel_peptides(
        detected_peptides=detected_peptides,
        alignments=alignments,
        reference_proteins=reference_proteins,
        novel_proteins=novel_proteins,
        gene_map=gene_map,
        protein_abundance=protein_abundance,
        enzyme=args.enzyme,
        max_missed_cleavages=args.max_missed_cleavages,
        min_peptide_length=args.min_peptide_length,
        max_peptide_length=args.max_peptide_length,
    )

    # 5. Write output
    save_novel_peptides(
        result,
        output_dir,
        metadata={
            "constellation_version": constellation_version,
            "search_results": str(search_results),
            "library": str(library_path),
            "alignment_hits": str(alignment_hits),
            "reference_fasta": str(reference_fasta),
            "novel_fasta": str(novel_fasta),
            "enzyme": args.enzyme,
            "max_missed_cleavages": args.max_missed_cleavages,
            "min_peptide_length": args.min_peptide_length,
            "max_peptide_length": args.max_peptide_length,
        },
    )
    if collision_metadata is not None:
        (output_dir / "collision_metadata.json").write_text(
            _json.dumps(collision_metadata, indent=2) + "\n"
        )

    # 6. Run-level manifest
    import os as _os
    import socket
    import platform

    classification_counts: dict[str, int] = {}
    for c in result.column("classification").to_pylist():
        classification_counts[c] = classification_counts.get(c, 0) + 1
    manifest = {
        "constellation_version": constellation_version,
        "subcommand": "massspec classify-novel-peptides",
        "argv": _sys.argv,
        "inputs": {
            "search_results": str(search_results),
            "library": str(library_path),
            "alignment_hits": str(alignment_hits),
            "reference_fasta": str(reference_fasta),
            "novel_fasta": str(novel_fasta),
        },
        "outputs": {
            "novel_peptides_pqdir": str(output_dir),
        },
        "params": {
            "enzyme": args.enzyme,
            "max_missed_cleavages": args.max_missed_cleavages,
            "min_peptide_length": args.min_peptide_length,
            "max_peptide_length": args.max_peptide_length,
            "collision_filter": args.collision_filter,
            "collision_rt_threshold_s": args.collision_rt_threshold_s,
            "collision_frag_ppm_tol": args.collision_frag_ppm_tol,
            "collision_min_shared_ions": args.collision_min_shared_ions,
            "dia": (str(dia_path) if dia_path is not None else None),
        },
        "runtime": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": _sys.version.split()[0],
            "user": _os.environ.get("USER", "unknown"),
        },
        "counts": {
            "novel_peptides": result.num_rows,
            "classifications": classification_counts,
            "collision_losers": (
                collision_metadata["n_losers"]
                if collision_metadata is not None
                else None
            ),
            "collision_clusters": (
                len(collision_metadata["clusters"])
                if collision_metadata is not None
                else None
            ),
        },
    }
    (output_dir / "manifest.json").write_text(_json.dumps(manifest, indent=2) + "\n")
    success_path.write_bytes(b"")

    if not args.no_progress:
        print(
            f"classify-novel-peptides done: {result.num_rows:,} unique novel "
            f"peptides classified → {output_dir}/novel_peptides.parquet",
            file=_sys.stderr,
        )
        for cls in sorted(
            classification_counts, key=lambda c: classification_counts[c],
            reverse=True,
        ):
            print(f"  {cls:30s}  {classification_counts[cls]:>6,}", file=_sys.stderr)
    return 0


def _load_detected_peptides(
    search_results: Path, library_path: Path
) -> "pa.Table":  # noqa: F821 — pa import-on-demand below
    """Load detected peptides as a (peptide_sequence, peptide_id?,
    modified_sequence?) Arrow table.

    Two supported inputs:
      * ``.elib`` SQLite — pull the ``PeptideSeq`` column from the
        ``entries`` table (the cartographer-compatible path).
      * ParquetDir Search bundle — join ``peptide_scores.parquet`` to
        ``<library>/peptides.parquet`` on ``peptide_id`` to recover the
        canonical sequence.
    """
    import pyarrow as pa
    import sqlite3

    if search_results.suffix in (".elib", ".dlib"):
        con = sqlite3.connect(str(search_results))
        try:
            rows = con.execute(
                "SELECT DISTINCT PeptideSeq, PeptideModSeq "
                "FROM entries WHERE PeptideSeq IS NOT NULL"
            ).fetchall()
        finally:
            con.close()
        return pa.table(
            {
                "peptide_sequence": pa.array(
                    [r[0] for r in rows], type=pa.string()
                ),
                "modified_sequence": pa.array(
                    [r[1] for r in rows], type=pa.string()
                ),
            }
        )

    # ParquetDir Search bundle: search-results dir + a library to join on
    import pyarrow.parquet as pq

    from constellation.massspec.library.io import load_library

    library = load_library(library_path, format="parquet_dir")
    peptide_id_to_seq = {
        pid: (seq, modseq)
        for pid, seq, modseq in zip(
            library.peptides.column("peptide_id").to_pylist(),
            library.peptides.column("sequence").to_pylist(),
            library.peptides.column("modified_sequence").to_pylist(),
            strict=True,
        )
    }
    peptide_scores = pq.read_table(search_results / "peptide_scores.parquet")
    seen: set[int] = set()
    peptide_ids: list[int] = []
    seqs: list[str] = []
    modseqs: list[str | None] = []
    for pid in peptide_scores.column("peptide_id").to_pylist():
        if pid in seen:
            continue
        seen.add(pid)
        entry = peptide_id_to_seq.get(pid)
        if entry is None:
            continue
        seqs.append(entry[0])
        modseqs.append(entry[1])
        peptide_ids.append(pid)
    return pa.table(
        {
            "peptide_id": pa.array(peptide_ids, type=pa.int64()),
            "peptide_sequence": pa.array(seqs, type=pa.string()),
            "modified_sequence": pa.array(modseqs, type=pa.string()),
        }
    )


def _drop_collision_losers(
    detected_peptides: "pa.Table",  # noqa: F821 — pa import-on-demand
    losers: set[str],
) -> "pa.Table":  # noqa: F821
    """Filter ``detected_peptides`` to drop rows whose
    ``modified_sequence`` is in the collision-loser set.

    ``losers`` are PeptideModSeq strings in EncyclopeDIA notation
    (``[+N.NNN]`` mass-delta form); ``_load_detected_peptides`` writes
    the same notation into ``modified_sequence`` when the input is an
    ``.elib`` — the only path on which ``--collision-filter`` is
    permitted.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if "modified_sequence" not in detected_peptides.column_names:
        return detected_peptides
    keep_mask = pc.invert(
        pc.is_in(
            detected_peptides.column("modified_sequence"),
            value_set=pa.array(sorted(losers), type=pa.string()),
        )
    )
    return detected_peptides.filter(keep_mask)


def _cmd_massspec_collision_filter(args: argparse.Namespace) -> int:
    """Collision-filter a search .elib and write a filtered .elib.

    Runs ``apply_collision_filter`` to find the collided identifications,
    writes a SQLite-filtered .elib with the loser rows removed (via
    ``filter_elib_by_losers``), drops a ``collision_metadata.json``
    sidecar, optionally auto-ingests the filtered .elib into ParquetDir
    bundles, writes ``manifest.json``, and touches ``_SUCCESS``.
    """
    import json as _json
    import os as _os
    import platform
    import socket
    import sys as _sys

    from constellation import __version__ as constellation_version
    from constellation.massspec.search import (
        apply_collision_filter,
        filter_elib_by_losers,
    )

    elib = Path(args.elib).resolve()
    dia = Path(args.dia).resolve()
    output_elib = Path(args.output_elib).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not elib.is_file():
        print(f"error: --elib not found: {elib}", file=_sys.stderr)
        return 1
    if not dia.is_file():
        print(f"error: --dia not found: {dia}", file=_sys.stderr)
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

    if not args.no_progress:
        print(
            f"[collision-filter] running on {elib} + {dia} "
            f"(rt<{args.rt_threshold_s}s, {args.frag_ppm_tol} ppm, "
            f">={args.min_shared_ions} shared ions) ...",
            file=_sys.stderr,
        )
    losers, collision_metadata = apply_collision_filter(
        elib_path=elib,
        dia_path=dia,
        rt_threshold_s=args.rt_threshold_s,
        frag_ppm_tol=args.frag_ppm_tol,
        min_shared_ions=args.min_shared_ions,
        return_metadata=True,
    )
    rows_deleted = filter_elib_by_losers(elib, losers, output_elib)
    collision_metadata["n_losers"] = len(losers)
    collision_metadata["rows_deleted"] = rows_deleted
    (output_dir / "collision_metadata.json").write_text(
        _json.dumps(collision_metadata, indent=2) + "\n"
    )

    if not args.no_progress:
        print(
            f"[collision-filter] dropped {len(losers):,} modseqs across "
            f"{len(collision_metadata['clusters']):,} clusters "
            f"(of {collision_metadata['n_entries']:,} entries) → "
            f"{output_elib}",
            file=_sys.stderr,
        )

    # ── auto-ingest the filtered .elib ──────────────────────────────
    ingest_info: dict[str, object] = {"skipped": bool(args.no_ingest)}
    library_pqdir: Path | None = None
    quant_pqdir: Path | None = None
    search_pqdir: Path | None = None
    if not args.no_ingest:
        from constellation.massspec.io.encyclopedia import read_encyclopedia
        from constellation.massspec.library import save_library
        from constellation.massspec.quant import save_quant
        from constellation.massspec.search import save_search

        try:
            ingest = read_encyclopedia(output_elib)
        except Exception as exc:  # noqa: BLE001 — surface in manifest
            ingest_info["error"] = f"{type(exc).__name__}: {exc}"
        else:
            library_pqdir = output_dir / "library_pqdir"
            save_library(ingest.library, library_pqdir, format="parquet_dir")
            ingest_info["library_counts"] = {
                "proteins": ingest.library.proteins.num_rows,
                "peptides": ingest.library.peptides.num_rows,
                "precursors": ingest.library.precursors.num_rows,
                "fragments": ingest.library.fragments.num_rows,
            }
            if ingest.quant is not None:
                quant_pqdir = output_dir / "quant_pqdir"
                save_quant(ingest.quant, quant_pqdir, format="parquet_dir")
            if ingest.search is not None:
                search_pqdir = output_dir / "search_pqdir"
                save_search(ingest.search, search_pqdir, format="parquet_dir")

    manifest = {
        "constellation_version": constellation_version,
        "subcommand": "massspec collision-filter",
        "argv": _sys.argv,
        "inputs": {"elib": str(elib), "dia": str(dia)},
        "outputs": {
            "filtered_elib": str(output_elib),
            "library_pqdir": str(library_pqdir) if library_pqdir else None,
            "quant_pqdir": str(quant_pqdir) if quant_pqdir else None,
            "search_pqdir": str(search_pqdir) if search_pqdir else None,
            "collision_metadata": str(output_dir / "collision_metadata.json"),
        },
        "params": {
            "rt_threshold_s": args.rt_threshold_s,
            "frag_ppm_tol": args.frag_ppm_tol,
            "min_shared_ions": args.min_shared_ions,
        },
        "counts": {
            "n_losers": len(losers),
            "n_clusters": len(collision_metadata["clusters"]),
            "n_entries": collision_metadata["n_entries"],
            "rows_deleted": rows_deleted,
        },
        "ingest": ingest_info,
        "runtime": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": _sys.version.split()[0],
            "user": _os.environ.get("USER", "unknown"),
        },
    }
    (output_dir / "manifest.json").write_text(
        _json.dumps(manifest, indent=2) + "\n"
    )
    success_path.write_bytes(b"")
    return 0


# ── chromatogram (XIC extraction) ────────────────────────────────────────


_DEFAULT_CHARGE_RANGE = (1, 4)
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
                   help="Convert bundle (contains peaks.parquet) or a shard dir / parquet file.")
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
            "Extract XIC traces. --level selects MS1 (precursor isotopes) "
            "or MS2 (fragments); it is the MS-level selector, NOT a scan "
            "filter — scan selection is derived from each target's RT / "
            "isolation / assigned scan."
        ),
    )
    common = p.add_argument_group("common")
    common.add_argument("--peaks-dir", type=Path, required=True,
                        help="Convert bundle (canonical f64 peaks source).")
    common.add_argument("--targets", type=Path, required=True,
                        help="Target list: a Library / Search / PRECURSOR_QUANT bundle, "
                             "or a bare XIC_TARGET_TABLE .parquet/.tsv (auto-detected).")
    common.add_argument("-o", "--output-dir", type=Path, required=True)
    common.add_argument("--level", type=int, choices=(1, 2), required=True,
                        help="MS level to extract from (1 = precursor isotopes, 2 = fragments).")
    common.add_argument("--acquisition-id", type=int, default=0,
                        help="acquisition_id stamped on output rows (FK; default 0).")
    common.add_argument("--tolerance", type=float, default=20.0)
    common.add_argument("--tolerance-unit", choices=("ppm", "Da"), default="ppm")
    common.add_argument("--rt-window", type=float, default=None,
                        help="Half-width (s) around each target's rt_center; explicit "
                             "rt_start/rt_end in the target override it; absent → all-RT.")
    common.add_argument("--all-in-window", action="store_true",
                        help="Return every peak within tolerance per (scan, ion), not just nearest.")
    common.add_argument("--no-drop-unmatched", dest="drop_unmatched", action="store_false",
                        help="Keep unmatched query rows (intensity 0) instead of dropping.")
    common.add_argument("--no-progress", action="store_true")

    idxg = p.add_argument_group("index (Mode B)")
    idxg.add_argument("--use-index", action="store_true",
                      help="Use the mass index (Mode B); build it in-line if absent.")
    idxg.add_argument("--index-dir", type=Path, default=None,
                      help="Index location override (default <peaks-dir>/mz_index/).")
    idxg.add_argument("--rebuild-index", action="store_true",
                      help="Force-rebuild the index (implies --use-index).")
    idxg.add_argument("--downcast", default="",
                      help="f32 axes when building the index (subset of {mz,rt,intensity}).")
    idxg.add_argument("--exact-error", action="store_true",
                      help="Mode B + f32-m/z index: re-derive f64 mz error for survivors "
                           "from --peaks-dir.")

    ms1g = p.add_argument_group("MS1 precursor-isotope options (with --level 1)")
    ms1g.add_argument("--n-isotopes", type=int, default=_DEFAULT_N_ISOTOPES,
                      help="Isotope peaks M+0..M+(n-1) (default 3).")
    ms1g.add_argument("--charge-range", type=int, nargs=2, metavar=("LO", "HI"),
                      default=list(_DEFAULT_CHARGE_RANGE),
                      help="Precursor charge sweep (default 1 4).")

    ms2g = p.add_argument_group("MS2 fragment options (with --level 2)")
    ms2g.add_argument("--ion-types", default="b,y",
                      help="Comma-separated fragment ion types (default b,y).")
    ms2g.add_argument("--max-fragment-charge", type=int, default=_DEFAULT_MAX_FRAG_CHARGE,
                      help="Max fragment charge, capped by precursor charge (default 2).")
    ms2g.add_argument("--neutral-losses", default=None,
                      help="Comma-separated neutral-loss ids (default none).")
    ms2g.add_argument("--assigned-scans-only", action="store_true",
                      help="Restrict MS2 to each target's assigned scan (requires a scan column).")
    p.set_defaults(func=_cmd_chromatogram_extract)


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
    import pyarrow.parquet as pq

    from constellation.massspec.quant.chromatogram import (
        extract_xic_indexed,
        extract_xic_scan_major,
        save_xic,
    )
    from constellation.massspec.quant.peak_index import build_peak_index
    from constellation.massspec.quant.targets import load_targets

    if not args.peaks_dir.exists():
        print(f"error: peaks not found: {args.peaks_dir}", file=sys.stderr)
        return 2
    if not args.targets.exists():
        print(f"error: targets not found: {args.targets}", file=sys.stderr)
        return 2

    _warn_level_flag_mismatch(args)
    targets = load_targets(args.targets)
    charge_range = tuple(args.charge_range)
    ion_types = _resolve_ion_types(args.ion_types)
    neutral_losses = _parse_downcast(args.neutral_losses) if args.neutral_losses else None

    use_index = args.use_index or args.rebuild_index
    common = dict(
        acquisition_id=args.acquisition_id, level=args.level,
        n_isotopes=args.n_isotopes, ion_types=ion_types,
        max_fragment_charge=args.max_fragment_charge, neutral_losses=neutral_losses,
        charge_range=charge_range, rt_window=args.rt_window,
        tolerance=args.tolerance, tolerance_unit=args.tolerance_unit,
        all_in_window=args.all_in_window, assigned_scans_only=args.assigned_scans_only,
        drop_unmatched=args.drop_unmatched,
    )

    if use_index:
        index_dir = args.index_dir or (args.peaks_dir / "mz_index")
        manifest_exists = (index_dir / "manifest.json").exists()
        if args.rebuild_index or not manifest_exists:
            if not args.no_progress:
                print(f"building peak index at {index_dir}", file=sys.stderr)
            build_peak_index(args.peaks_dir, index_dir, downcast=_parse_downcast(args.downcast))
        table = extract_xic_indexed(
            index_dir, targets, exact_error=args.exact_error, peaks_dir=args.peaks_dir, **common
        )
    else:
        peaks_file = args.peaks_dir / "peaks.parquet"
        peaks = pq.read_table(peaks_file if peaks_file.exists() else args.peaks_dir)
        table = extract_xic_scan_major(peaks, targets, **common)

    save_xic(table, args.output_dir, metadata={
        "level": args.level, "mode": "indexed" if use_index else "scan_major",
        "tolerance": args.tolerance, "tolerance_unit": args.tolerance_unit,
    })
    if not args.no_progress:
        print(f"extracted {table.num_rows} XIC rows → {args.output_dir}", file=sys.stderr)
    return 0


def _warn_level_flag_mismatch(args: argparse.Namespace) -> None:
    """Warn when a flag from the inactive level's group was set."""
    if args.level == 1:
        if args.ion_types != "b,y" or args.max_fragment_charge != _DEFAULT_MAX_FRAG_CHARGE \
                or args.neutral_losses or args.assigned_scans_only:
            print("warning: MS2 fragment options ignored with --level 1", file=sys.stderr)
    elif args.level == 2:
        if args.n_isotopes != _DEFAULT_N_ISOTOPES or tuple(args.charge_range) != _DEFAULT_CHARGE_RANGE:
            print("warning: MS1 isotope options ignored with --level 2", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────
# counter — panel-shaped ion-count estimation + per-acquisition calibration
# ──────────────────────────────────────────────────────────────────────


def _build_counter_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "counter",
        help=(
            "Counter ion-count estimation. `calibrate` fits a per-acquisition "
            "GlobalCalibration from spiked calibrants; `estimate` runs the "
            "panel-shaped per-seed estimate (one panel per target, parallel "
            "across seeds) producing N_total + credible intervals."
        ),
    )
    csubs = p.add_subparsers(dest="counter_subcommand", required=True)
    _build_counter_calibrate_parser(csubs)
    _build_counter_estimate_parser(csubs)


def _add_counter_common_inputs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trace", type=Path, required=True,
                   help="XIC_TRACE_TABLE .parquet (MS1, ideally an all_in_window extraction).")
    p.add_argument("--scan-metadata", type=Path, required=True,
                   help="SCAN_METADATA_TABLE .parquet (the scan axis; level-1 rows are used).")
    p.add_argument("--rt-window", type=float, default=60.0,
                   help="Half-width (s) of the scan-axis window around each target's rt_center.")
    p.add_argument("--n-isotopes", type=int, default=_DEFAULT_N_ISOTOPES)
    p.add_argument("--acquisition-id", type=int, default=0)
    p.add_argument("--no-progress", action="store_true")


def _build_counter_calibrate_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "calibrate",
        help="Fit a per-acquisition GlobalCalibration (+ peptide params) from calibrants.",
    )
    _add_counter_common_inputs(p)
    p.add_argument("--calibrants", type=Path, required=True,
                   help="XIC_TARGET_TABLE .parquet of calibrant peptides (target_id, "
                        "modified_sequence, precursor_charge, rt_center).")
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("--gain", action="store_true",
                   help="Co-fit the gain alpha(z) (weakly identified; off by default).")
    p.set_defaults(func=_cmd_counter_calibrate)


def _build_counter_estimate_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "estimate",
        help="Panel-shaped per-seed ion-count estimate (parallel across seeds).",
    )
    _add_counter_common_inputs(p)
    p.add_argument("--targets", type=Path, required=True,
                   help="XIC_TARGET_TABLE .parquet of seed peptides to quantify.")
    p.add_argument("--calibration", type=Path, required=True,
                   help="COUNTER_GLOBAL_CALIBRATION_TABLE .parquet (from `counter calibrate`).")
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1,
                   help="Process-pool size over seeds (each seed is an independent panel).")
    p.add_argument("--neighborhood-ppm", type=float, default=100.0,
                   help="Per-channel m/z neighborhood retained as discovery candidate nodes.")
    p.add_argument("--detect-threshold", type=float, default=8.0)
    p.add_argument("--max-candidates", type=int, default=1,
                   help="Max same-grid interferers per panel (default 1; >1 is "
                        "weakly identifiable and overfits until the parsimony pass lands).")
    p.add_argument("--no-background", dest="background", action="store_false",
                   help="Disable the additive background channel.")
    p.add_argument("--collide-ppm", type=float, default=20.0,
                   help="m/z tolerance for grouping targets into channel-overlap "
                        "components that are co-fit jointly (default 20). Should be ≤ the "
                        "upstream XIC extraction tolerance — a member beyond that tolerance "
                        "has no extracted signal on the reference grid.")
    p.add_argument("--rt-overlap-s", type=float, default=None,
                   help="RT-center span (s) within which component members are co-fit as "
                        "one unit (default + max: --rt-window; larger values are clamped, "
                        "since a unit can't span beyond the reference observation window).")
    p.add_argument("--max-component-size", type=int, default=8,
                   help="Co-fit units larger than this fall back to independent "
                        "single-target fits (guards against a dense isobaric region "
                        "chaining into one pool-stalling mega-panel).")
    p.set_defaults(func=_cmd_counter_estimate)


def _counter_targets(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    need = ("target_id", "modified_sequence", "precursor_charge", "rt_center")
    missing = [c for c in need if c not in t.column_names]
    if missing:
        raise SystemExit(f"error: targets missing columns {missing}")
    rows = t.select(list(need)).to_pylist()
    tids = [r["target_id"] for r in rows]
    if len(set(tids)) != len(tids):
        from collections import Counter

        dups = sorted(tid for tid, n in Counter(tids).items() if n > 1)
        raise SystemExit(
            f"error: targets has duplicate target_id(s) (must be unique): {dups[:10]}"
        )
    return rows


# -- estimate: spawn worker pool over seeds (each an independent panel) --

_COUNTER_CTX: dict = {}


def _counter_worker_init(trace_path: str, scan_meta_path: str, calibration_path: str, opts: dict) -> None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from constellation.massspec.counter import calibration_from_table

    trace = pq.read_table(trace_path)
    sm = pq.read_table(scan_meta_path)
    if "level" in sm.column_names:
        sm = sm.filter(pc.equal(sm.column("level"), 1))
    _COUNTER_CTX.update(
        trace=trace,
        ms1=sm.select(["scan", "rt", "iit"]),
        cal=calibration_from_table(pq.read_table(calibration_path)),
        opts=opts,
    )


def _counter_worker_estimate(target: dict) -> dict:

    import pyarrow.compute as pc

    from constellation.core.sequence.proforma import Peptidoform
    from constellation.massspec.counter import (
        DiscoverConfig,
        Panel,
        Progenitor,
        estimate_panel,
        observation_for_region,
    )

    opts, cal, trace, ms1 = (_COUNTER_CTX[k] for k in ("opts", "cal", "trace", "ms1"))
    tid = int(target["target_id"])
    modseq = target["modified_sequence"]
    base = {"acquisition_id": opts["acquisition_id"], "target_id": tid,
            "modified_sequence": modseq, "precursor_charge": target["precursor_charge"]}
    try:
        # Coerce inside the try: a null charge / rt_center / modseq is one bad seed,
        # not a run-aborting crash.
        z = int(target["precursor_charge"])
        rtc = float(target["rt_center"])
        prog = Progenitor.for_peptide(Peptidoform(sequence=modseq), [z], cal,
                                      n_isotopes=opts["n_isotopes"])
        rt = ms1.column("rt")
        win = ms1.filter(pc.less_equal(pc.abs(pc.subtract(rt, rtc)), opts["rt_window"]))
        obs, region = observation_for_region(trace, win, prog, target_id=tid,
                                             neighborhood_ppm=opts["neighborhood_ppm"])
        if int(obs.mask.sum()) == 0:
            return {**base, "status": "no_signal"}
        panel = Panel([prog], cal, background=opts["background"])
        cfg = DiscoverConfig(detect_threshold=opts["detect_threshold"],
                             max_candidates=opts["max_candidates"])
        res = estimate_panel(panel, obs, region, config=cfg, rt_prior_ms=rtc * 1000.0,
                             inference="map")
        return {**base, **res, "status": "ok"}
    except Exception as exc:  # noqa: BLE001 — one bad seed shouldn't kill the run
        return {**base, "status": f"error:{type(exc).__name__}"}


def _counter_cofit_units(targets: list[dict], opts: dict) -> tuple[list[list[dict]], int]:
    """Partition targets into co-fit units: channel-overlap components (shared m/z)
    refined to co-eluting sub-clusters (shared RT), with oversized units split back
    to singletons (the mega-component pool-stall guard). Each unit is a list of
    target dicts sorted by `target_id` ([0] is the reference grid). Targets without a
    `modified_sequence` (no envelope) stay singletons. Returns (units, n_capped)."""
    from constellation.core.sequence.proforma import Peptidoform
    from constellation.massspec.counter import (
        TheoreticalCandidateIndex,
        channel_overlap_components,
        refine_components_by_rt,
        restrict_to_reference_star,
    )

    by_tid = {int(t["target_id"]): t for t in targets}
    entries: list[tuple] = []
    rt_centers: dict[int, float] = {}
    no_envelope: list[int] = []
    for t in targets:
        tid = int(t["target_id"])
        if t.get("rt_center") is not None:
            rt_centers[tid] = float(t["rt_center"])
        # A target needs a modseq AND a charge to form a theoretical envelope; any
        # missing/malformed field → keep it a singleton (the worker reports the error,
        # rather than the parent partitioner crashing the whole run).
        try:
            if t["modified_sequence"] and t.get("precursor_charge") is not None:
                entries.append(
                    (tid, Peptidoform(sequence=t["modified_sequence"]), [int(t["precursor_charge"])])
                )
            else:
                no_envelope.append(tid)
        except Exception:  # noqa: BLE001 — malformed modseq/charge → singleton fallback
            no_envelope.append(tid)

    unit_tids: list = []
    if entries:
        index = TheoreticalCandidateIndex.from_peptides(entries, n_isotopes=opts["n_isotopes"])
        comps = channel_overlap_components(index, collide_ppm=opts["collide_ppm"])
        unit_tids = refine_components_by_rt(comps, rt_centers, rt_overlap_s=opts["rt_overlap_s"])
        # Reference-star restriction: a transitive m/z component co-fit on the
        # reference grid would under-score members that only indirectly overlap (their
        # peaks were never extracted onto the reference grid). Keep only members
        # DIRECTLY overlapping each star's reference; the rest fall into their own
        # stars (singletons if they overlap nobody).
        unit_tids = restrict_to_reference_star(unit_tids, index, collide_ppm=opts["collide_ppm"])
    unit_tids += [frozenset({tid}) for tid in no_envelope]

    cap = int(opts["max_component_size"])
    units: list[list[dict]] = []
    n_capped = 0
    for u in unit_tids:
        members = sorted(u)
        if len(members) > cap:
            n_capped += 1
            units.extend([[by_tid[tid]] for tid in members])  # fall back to singletons
        else:
            units.append([by_tid[tid] for tid in members])
    units.sort(key=lambda u: int(u[0]["target_id"]))  # deterministic work-item order
    return units, n_capped


def _counter_worker_component(unit: list[dict]) -> list[dict]:
    """Joint co-fit of a multi-member unit — one panel over the reference member's
    grid (`unit[0]`), each member scored at its own mass defect. Returns one record
    per member."""
    import pyarrow.compute as pc

    from constellation.core.sequence.proforma import Peptidoform
    from constellation.massspec.counter import (
        DiscoverConfig,
        Progenitor,
        estimate_component,
        observation_for_region,
    )

    opts, cal, trace, ms1 = (_COUNTER_CTX[k] for k in ("opts", "cal", "trace", "ms1"))
    members = sorted(unit, key=lambda t: int(t["target_id"]))  # reference = min target_id
    bases = [
        {
            "acquisition_id": opts["acquisition_id"],
            "target_id": int(t["target_id"]),
            "modified_sequence": t["modified_sequence"],
            "precursor_charge": int(t["precursor_charge"]),
        }
        for t in members
    ]
    try:
        progs = [
            Progenitor.for_peptide(
                Peptidoform(sequence=t["modified_sequence"]),
                [int(t["precursor_charge"])],
                cal,
                n_isotopes=opts["n_isotopes"],
            )
            for t in members
        ]
        rtc_ref = float(members[0]["rt_center"])
        rt = ms1.column("rt")
        win = ms1.filter(pc.less_equal(pc.abs(pc.subtract(rt, rtc_ref)), opts["rt_window"]))
        obs, _region = observation_for_region(
            trace, win, progs[0], target_id=int(members[0]["target_id"]),
            neighborhood_ppm=opts["neighborhood_ppm"],
        )
        if int(obs.mask.sum()) == 0:
            # faint/absent reference grid → don't blanket-no_signal the co-members
            # (the reference is min target_id, arbitrary w.r.t. abundance); fit each
            # member on its OWN grid as an independent singleton.
            return [_counter_worker_estimate(t) for t in members]
        cfg = DiscoverConfig(
            detect_threshold=opts["detect_threshold"], max_candidates=opts["max_candidates"]
        )
        results = estimate_component(
            progs, obs, config=cfg, background=opts["background"],
            rt_priors_ms=[float(t["rt_center"]) * 1000.0 for t in members],
        )
        return [{**b, **r, "status": "ok"} for b, r in zip(bases, results)]
    except Exception as exc:  # noqa: BLE001 — one bad component shouldn't kill the run
        return [{**b, "status": f"error:{type(exc).__name__}"} for b in bases]


def _counter_worker_unit(unit: list[dict]) -> list[dict]:
    """Dispatch a co-fit unit: a singleton → the per-target discovery path; a
    multi-member unit → the joint component co-fit."""
    if len(unit) == 1:
        return [_counter_worker_estimate(unit[0])]
    return _counter_worker_component(unit)


def _cmd_counter_estimate(args: argparse.Namespace) -> int:
    import multiprocessing as mp


    from constellation.massspec.counter import CounterResult, counter_n_table, save_counter

    for path in (args.trace, args.scan_metadata, args.calibration, args.targets):
        if not path.exists():
            print(f"error: not found: {path}", file=sys.stderr)
            return 2
    success = args.output_dir / "_SUCCESS"
    if success.exists():
        print(f"error: --output-dir already complete ({success} exists)", file=sys.stderr)
        return 1

    targets = _counter_targets(args.targets)
    rt_overlap_s = args.rt_overlap_s if args.rt_overlap_s is not None else args.rt_window
    if rt_overlap_s > args.rt_window:
        # a unit's RT span can't exceed the reference's ± rt_window obs window, or a
        # co-member's elution falls outside the observation entirely.
        if not args.no_progress:
            print(
                f"warning: --rt-overlap-s {rt_overlap_s} > --rt-window {args.rt_window}; "
                f"clamping to {args.rt_window}",
                file=sys.stderr,
            )
        rt_overlap_s = args.rt_window
    opts = dict(
        acquisition_id=args.acquisition_id, n_isotopes=args.n_isotopes,
        rt_window=args.rt_window, neighborhood_ppm=args.neighborhood_ppm,
        detect_threshold=args.detect_threshold, max_candidates=args.max_candidates,
        background=args.background, collide_ppm=args.collide_ppm,
        rt_overlap_s=rt_overlap_s, max_component_size=args.max_component_size,
    )
    # Partition into co-fit units in the parent (cheap, data-independent): m/z-overlap
    # components × RT co-elution. Singletons stay on the per-target discovery path; a
    # multi-member unit is one joint panel. The worker maps over UNITS, not targets.
    units, n_capped = _counter_cofit_units(targets, opts)
    initargs = (str(args.trace), str(args.scan_metadata), str(args.calibration), opts)
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_counter_worker_init, initargs=initargs) as pool:
            grouped = pool.map(_counter_worker_unit, units)
    else:
        _counter_worker_init(*initargs)
        try:
            grouped = [_counter_worker_unit(u) for u in units]
        finally:
            _COUNTER_CTX.clear()  # don't leak this run's trace/cal into a later in-process call
    records = [r for group in grouped for r in group]

    ok = [r for r in records if r.get("status") == "ok"]
    skipped = len(records) - len(ok)
    n_multi = sum(1 for u in units if len(u) > 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_counter(CounterResult(counter_n=counter_n_table(ok)), args.output_dir)
    success.touch()
    if not args.no_progress:
        extra = f", {n_multi} co-fit components" if n_multi else ""
        extra += f", {n_capped} oversized split" if n_capped else ""
        print(f"counter estimate: {len(ok)} estimated, {skipped} skipped{extra} → {args.output_dir}",
              file=sys.stderr)
    return 0


def _cmd_counter_calibrate(args: argparse.Namespace) -> int:

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from constellation.core.sequence.proforma import Peptidoform
    from constellation.massspec.counter import (
        GlobalCalibration,
        Progenitor,
        StagedCalibration,
        calibration_to_table,
        observation_for_progenitor,
        peptide_params_to_table,
    )

    for path in (args.trace, args.scan_metadata, args.calibrants):
        if not path.exists():
            print(f"error: not found: {path}", file=sys.stderr)
            return 2
    success = args.output_dir / "_SUCCESS"
    if success.exists():
        print(f"error: --output-dir already complete ({success} exists)", file=sys.stderr)
        return 1

    trace = pq.read_table(args.trace)
    sm = pq.read_table(args.scan_metadata)
    if "level" in sm.column_names:
        sm = sm.filter(pc.equal(sm.column("level"), 1))
    ms1 = sm.select(["scan", "rt", "iit"])
    calibrants = _counter_targets(args.calibrants)

    cal = GlobalCalibration(n_isotopes=args.n_isotopes, charges=(1, 2, 3, 4))
    progs, obss, tids, mods = [], [], [], []
    rt = ms1.column("rt")
    for c in calibrants:
        tid, modseq, z, rtc = (int(c["target_id"]), c["modified_sequence"],
                               int(c["precursor_charge"]), float(c["rt_center"]))
        prog = Progenitor.for_peptide(Peptidoform(sequence=modseq), [z], cal, n_isotopes=args.n_isotopes)
        win = ms1.filter(pc.less_equal(pc.abs(pc.subtract(rt, rtc)), args.rt_window))
        obs = observation_for_progenitor(prog, trace, win, target_id=tid)
        if int(obs.mask.sum()) == 0:
            continue
        progs.append(prog)
        obss.append(obs)
        tids.append(tid)
        mods.append(modseq)
    if not progs:
        print("error: no calibrant yielded signal", file=sys.stderr)
        return 1

    from constellation.massspec.counter.orchestrate import _DEFAULT_GLOBAL_PARAMS, _GAIN_PARAM_NAMES

    gp = tuple(_DEFAULT_GLOBAL_PARAMS) + (_GAIN_PARAM_NAMES if args.gain else ())
    StagedCalibration(progs, obss, cal).run(global_params=gp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(calibration_to_table(cal, acquisition_id=args.acquisition_id),
                   args.output_dir / "global_calibration.parquet")
    pq.write_table(peptide_params_to_table(progs, acquisition_id=args.acquisition_id,
                                           target_ids=tids, modified_sequences=mods),
                   args.output_dir / "peptide_params.parquet")
    success.touch()
    if not args.no_progress:
        print(f"counter calibrate: {len(progs)} calibrants → {args.output_dir}", file=sys.stderr)
    return 0


__all__ = ["build_parser"]
