"""`constellation <subcommand>` dispatcher + per-domain shims.

Usage:
    constellation --help
    constellation doctor

Domain subcommands (mzpeak, koina, pod5, structure) are wired as the
underlying modules are ported. Thin legacy binaries (``mzpeak``,
``koina-library``) forward to the same handlers via the ``main_*``
entry points declared in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

# Import adapter modules for their side-effect of registering ToolSpecs.
# Each adapter calls constellation.thirdparty.registry.register(...) at
# import time so the `doctor` subcommand sees every known tool.
from constellation.thirdparty import (  # noqa: F401
    busco,
    dorado,
    encyclopedia,
    hifiasm,
    minimap2,
    mmseqs2,
    ragtag,
    samtools,
)
from constellation.thirdparty.registry import registered, try_find


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Print a tool-status table."""
    tools = registered()
    if not tools:
        print("no third-party tools registered")
        return 0

    rows: list[tuple[str, str, str, str]] = []
    for spec in tools:
        handle = try_find(spec.name)
        if handle is None:
            status = "not found"
            where = f"(set {spec.env_var} or run {spec.install_script or '—'})"
            version = "—"
        else:
            status = "ok"
            where = f"{handle.source}: {handle.path}"
            version = handle.version or "—"
        rows.append((spec.name, status, version, where))

    name_w = max(len(r[0]) for r in rows + [("tool", "", "", "")])
    status_w = max(len(r[1]) for r in rows + [("", "status", "", "")])
    ver_w = max(len(r[2]) for r in rows + [("", "", "version", "")])

    header = f"{'tool':<{name_w}}  {'status':<{status_w}}  {'version':<{ver_w}}  location"
    print(header)
    print("-" * len(header))
    for name, status, version, where in rows:
        print(f"{name:<{name_w}}  {status:<{status_w}}  {version:<{ver_w}}  {where}")
    # exit nonzero if any tool is missing — lets CI gate on it
    return 0 if all(r[1] == "ok" for r in rows) else 1


def _cmd_not_wired(name: str) -> Callable[[argparse.Namespace], int]:
    def handler(_args: argparse.Namespace) -> int:
        print(
            f"subcommand `{name}` is scaffolded but not implemented yet. "
            "See the 90-day roadmap in CLAUDE.md.",
            file=sys.stderr,
        )
        return 2

    return handler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="constellation")
    subs = parser.add_subparsers(dest="subcommand", required=True)

    p_doctor = subs.add_parser("doctor", help="Report third-party tool status")
    p_doctor.set_defaults(func=_cmd_doctor)

    # Sequencing transcriptomics pipeline (S1 — demultiplex shipped).
    _build_transcriptome_parser(subs)

    # Placeholders so `--help` advertises the intended surface. Wire as
    # the underlying modules are ported.
    for name, summary in [
        ("mzpeak", "Convert raw MS files to Parquet-backed mzpeak (TODO)"),
        ("koina", "Build spectral libraries via Koina (TODO)"),
        ("pod5", "Ingest POD5 signal to Parquet (TODO)"),
        ("structure", "Prepare structures for MD (TODO)"),
        # Other sequencing pipeline verbs — wire as additional sessions land.
        ("basecall", "Run Dorado basecaller against POD5 inputs (TODO)"),
        ("assemble", "Assemble reads with HiFiASM (TODO)"),
        ("polish", "Polish a draft assembly with Dorado + minimap2 (TODO)"),
        ("scaffold", "Reference-guided scaffolding via RagTag (TODO)"),
        ("annotate", "BUSCO + repeat / telomere annotation (TODO)"),
        ("project", "Initialize / inspect a Constellation project (TODO)"),
    ]:
        p = subs.add_parser(name, help=summary)
        p.set_defaults(func=_cmd_not_wired(name))

    return parser


def _build_transcriptome_parser(subs) -> None:
    """Wire the ``constellation transcriptome ...`` sub-tree.

    S1 ships ``demultiplex`` (segments + classifies + barcode-assigns +
    ORF-predicts + builds count matrix). ``cluster`` is a future
    placeholder. Each subcommand defers its real handler until imported,
    so the top-level CLI stays cheap to load.
    """
    p_tx = subs.add_parser(
        "transcriptome",
        help="Long-read cDNA / direct-RNA pipeline",
    )
    tx_subs = p_tx.add_subparsers(dest="tx_subcommand", required=True)

    p_dem = tx_subs.add_parser(
        "demultiplex",
        help=(
            "Segment + classify + barcode-assign + ORF-predict + count "
            "matrix; replaces NanoporeAnalysis transcriptome_nf"
        ),
    )
    p_dem.add_argument(
        "--reads",
        required=True,
        nargs="+",
        help=(
            "path(s) to Dorado-emitted SAM or BAM. Accepts: a single "
            "file, multiple files, or a directory (auto-globs *.bam "
            "and *.sam). Multi-file mode requires the samples TSV to "
            "include a 'file' column tying each (sample_id, barcode_id) "
            "to its source file."
        ),
    )
    p_dem.add_argument(
        "--library-design",
        default="cdna_wilburn_v1",
        help="LibraryDesign name (see designs.available_designs())",
    )
    p_dem.add_argument(
        "--samples",
        required=True,
        help=(
            "TSV file mapping reads → samples. Single-file mode: "
            "columns are sample_id (int) + sample_name (str) + "
            "barcode_id (int, 0-indexed into the design's barcode "
            "panel). Multi-file mode: prepend a 'file' column whose "
            "values match the basenames or paths under --reads — "
            "each row asserts (file, sample_id, barcode_id) and the "
            "same sample_id may appear with multiple (file, barcode) "
            "combos (re-runs / multi-flowcell aggregation). Header "
            "row is optional in single-file mode and required in "
            "multi-file mode."
        ),
    )
    p_dem.add_argument(
        "--output-dir",
        required=True,
        help="output directory (created if missing)",
    )
    p_dem.add_argument(
        "--acquisition-id",
        type=int,
        default=1,
        help="acquisition_id stamped onto every read (default 1)",
    )
    p_dem.add_argument(
        "--min-aa-length",
        type=int,
        default=60,
        help=(
            "minimum protein length (in amino acids) for ORF prediction "
            "(default 60 — matches NanoporeAnalysis _fixed1 baseline)"
        ),
    )
    p_dem.add_argument(
        "--min-protein-count",
        type=int,
        default=2,
        help=(
            "drop proteins with fewer than this many total reads from "
            "the FASTA / counts.tab (default 2 — matches NanoporeAnalysis "
            "_fixed1 baseline)"
        ),
    )
    p_dem.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "number of worker processes for the demux stage (default 1; "
            "≥2 enables ProcessPoolExecutor fan-out — replaces NA's "
            "Nextflow per-batch parallelism)"
        ),
    )
    p_dem.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help=(
            "rows per demux worker batch (default 100000, matches NA's "
            "transcriptome_nf default)"
        ),
    )
    p_dem.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip stages whose output is already present + marked "
            "complete; rerun missing shards in partially-written stages"
        ),
    )
    p_dem.add_argument(
        "--progress",
        action="store_true",
        help="print one-line progress events to stderr",
    )
    p_dem.set_defaults(func=_cmd_transcriptome_demultiplex)

    # Cluster reserved for S4
    p_cluster = tx_subs.add_parser(
        "cluster",
        help="abundance-weighted clustering + consensus + ORF (S4 — TODO)",
    )
    p_cluster.set_defaults(func=_cmd_not_wired("transcriptome cluster"))


def _resolve_input_paths(reads_args: list[str]) -> list[Path]:
    """Expand --reads arguments into a deterministic list of files.

    Accepts (a) one or more file paths, (b) a single directory (auto-
    globs ``*.bam`` and ``*.sam``), or any mix. Sorts the result so
    acquisition_id assignment is deterministic across runs.
    """
    from pathlib import Path

    files: list[Path] = []
    for raw in reads_args:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.bam")))
            files.extend(sorted(p.glob("*.sam")))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"--reads target not found: {p}")
    if not files:
        raise ValueError(
            f"no .sam or .bam files found under --reads {reads_args}"
        )
    # Deduplicate while preserving sort order.
    seen: set[Path] = set()
    deduped: list[Path] = []
    for f in sorted(files):
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


def _parse_samples_tsv(
    tsv_path: str,
    *,
    input_files: list[Path],
    default_acquisition_id: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[Path, int]]:
    """Parse the samples TSV in single- or multi-file mode.

    Single-file mode (TSV header lacks ``file`` column or is absent):
    every row gets ``acquisition_id = default_acquisition_id``. All
    rows must reference barcode_ids from a single panel.

    Multi-file mode (TSV header includes a ``file`` column): each
    row's ``file`` cell is matched against the supplied ``input_files``
    by basename or full path. Each unique referenced file gets a
    deterministic acquisition_id (1..N in sorted-path order).

    Returns ``(sample_rows, edge_rows, file_to_acquisition_id)`` ready
    for ``Samples.from_records`` and the chunk planner.
    """
    from pathlib import Path

    by_basename = {f.name: f for f in input_files}
    by_path = {str(f): f for f in input_files}

    file_to_acq: dict[Path, int] = {
        f: i + 1 for i, f in enumerate(sorted(input_files))
    }

    sample_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    seen_sample_ids: dict[int, str] = {}

    with open(tsv_path, encoding="utf-8") as fh:
        lines = [
            ln.rstrip("\r\n")
            for ln in fh
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    if not lines:
        raise ValueError(f"samples TSV {tsv_path!r} has no rows")

    # Header detection: first row's first cell == "file" or
    # "sample_id" — we treat as header if any of the canonical column
    # names appears as a literal cell value.
    first_cells = [c.strip() for c in lines[0].split("\t")]
    canonical_columns = {"file", "sample_id", "sample_name", "barcode_id"}
    is_header = bool(set(first_cells) & canonical_columns)

    if is_header:
        header = first_cells
        body = lines[1:]
    else:
        # No header — assume single-file 3-column format.
        header = ["sample_id", "sample_name", "barcode_id"]
        body = lines

    multi_file_mode = "file" in header
    if multi_file_mode and len(input_files) <= 1:
        # User supplied a 'file' column but only one input file —
        # require the column anyway, just verify it matches.
        pass
    if not multi_file_mode and len(input_files) > 1:
        raise ValueError(
            f"--reads passed {len(input_files)} files but the samples "
            f"TSV has no 'file' column — multi-file inputs require a "
            f"'file' column tying each (sample_id, barcode_id) row to "
            f"its source file"
        )

    col_idx = {name: i for i, name in enumerate(header)}
    required = {"sample_id", "sample_name", "barcode_id"}
    missing = required - set(col_idx)
    if missing:
        raise ValueError(
            f"samples TSV missing required columns: {sorted(missing)}"
        )

    for raw in body:
        cells = [c.strip() for c in raw.split("\t")]
        if len(cells) < len(header):
            continue  # malformed row — skip
        try:
            sid = int(cells[col_idx["sample_id"]])
            bid = int(cells[col_idx["barcode_id"]])
        except ValueError:
            continue
        sname = cells[col_idx["sample_name"]]
        if multi_file_mode:
            file_cell = cells[col_idx["file"]]
            file_path = by_basename.get(file_cell) or by_path.get(file_cell)
            if file_path is None:
                # Try resolving as a path relative to cwd.
                p = Path(file_cell)
                if p.is_file():
                    file_path = p.resolve()
                    by_path[str(file_path)] = file_path
                    if file_path not in file_to_acq:
                        file_to_acq[file_path] = len(file_to_acq) + 1
                else:
                    raise ValueError(
                        f"samples TSV references file {file_cell!r} "
                        f"which is not in --reads inputs (basenames: "
                        f"{sorted(by_basename)})"
                    )
            acq_id = file_to_acq[file_path]
        else:
            acq_id = default_acquisition_id

        # Sample table row — first time we see each sample_id only.
        if sid not in seen_sample_ids:
            sample_rows.append(
                {"sample_id": sid, "sample_name": sname, "description": None}
            )
            seen_sample_ids[sid] = sname
        edge_rows.append(
            {
                "sample_id": sid,
                "acquisition_id": acq_id,
                "barcode_id": bid,
            }
        )

    return sample_rows, edge_rows, file_to_acq


def _cmd_transcriptome_demultiplex(args: argparse.Namespace) -> int:
    """Run the full S1 demux + ORF + quant pipeline on one or more SAM/BAMs."""
    # Defer heavy imports until the subcommand actually fires so
    # `constellation --help` stays fast.
    import json
    from pathlib import Path

    from constellation.sequencing.progress import (
        NullProgress,
        StreamProgress,
    )
    from constellation.sequencing.samples import Samples
    from constellation.sequencing.transcriptome.stages import (
        run_demux_pipeline,
    )

    input_files = _resolve_input_paths(args.reads)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows, edge_rows, file_to_acq = _parse_samples_tsv(
        args.samples,
        input_files=input_files,
        default_acquisition_id=args.acquisition_id,
    )
    samples = Samples.from_records(samples=sample_rows, edges=edge_rows)

    # Build (file, acquisition_id) input list for the pipeline.
    if len(input_files) == 1 and len(file_to_acq) <= 1:
        # Single-file mode — keep using the user's --acquisition-id as
        # the stamped value.
        pipeline_inputs: list[tuple[Path, int]] = [
            (input_files[0], args.acquisition_id)
        ]
    else:
        pipeline_inputs = [(f, file_to_acq[f]) for f in input_files]

    cb = StreamProgress() if args.progress else NullProgress()

    artefacts = run_demux_pipeline(
        pipeline_inputs,
        library_design=args.library_design,
        samples=samples,
        output_dir=output_dir,
        batch_size=args.batch_size,
        n_workers=args.threads,
        min_aa_length=args.min_aa_length,
        min_protein_count=args.min_protein_count,
        progress_cb=cb,
        resume=args.resume,
    )

    n_reads = artefacts["n_reads"]
    quant_table = artefacts["quant_table"]
    fasta_records = artefacts["fasta_records"]

    # Manifest for reproducibility audit.
    manifest = {
        "input_files": [str(f) for f in input_files],
        "acquisition_map": {str(f): aid for f, aid in pipeline_inputs},
        "library_design": args.library_design,
        "min_aa_length": args.min_aa_length,
        "min_protein_count": args.min_protein_count,
        "n_reads": n_reads,
        "n_proteins_kept": len(fasta_records),
        "n_workers": args.threads,
        "batch_size": args.batch_size,
        "resumed": args.resume,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.progress:
        # Print a one-shot summary when --progress isn't on.
        print(
            f"demultiplex done: {n_reads} reads → "
            f"{len(fasta_records)} proteins, {quant_table.num_rows} quant rows",
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


# Thin legacy shims — `mzpeak`, `koina-library`, etc. Each forces the
# subcommand so the Cartographer muscle memory still works.
def main_mzpeak(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    return main(["mzpeak", *raw])


def main_koina(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    return main(["koina", *raw])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
