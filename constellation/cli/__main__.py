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

    # Reference imports / fetch / summary / validate.
    _build_reference_parser(subs)

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

    # ── transcriptome align — Mode A reference-guided gene counting ──
    p_aln = tx_subs.add_parser(
        "align",
        help=(
            "Align demuxed reads to a reference + filter + count reads "
            "per gene (Mode A). Future: Mode B de novo clustering "
            "shares the same verb (auto-dispatched on --reference presence)."
        ),
    )
    p_aln.add_argument(
        "--demux-dir",
        required=True,
        help=(
            "directory written by `constellation transcriptome demultiplex` "
            "— provides read_demux/, manifest.json (with input BAM paths), "
            "and the implicit acquisition map"
        ),
    )
    p_aln.add_argument(
        "--reference",
        default=None,
        help=(
            "reference root produced by `constellation reference import` / "
            "`fetch` — must contain ``genome/`` + ``annotation/`` "
            "ParquetDir bundles. Presence selects Mode A; absence "
            "errors with the Mode B not-yet-implemented message."
        ),
    )
    p_aln.add_argument(
        "--samples",
        required=True,
        help=(
            "samples TSV (same TSV used at demux time). Future: persist "
            "Samples in the demux output to make this optional."
        ),
    )
    p_aln.add_argument("--output-dir", required=True)
    p_aln.add_argument("--threads", type=int, default=1)
    p_aln.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="records per BAM-decode worker chunk (default 100000)",
    )
    p_aln.add_argument(
        "--min-length",
        type=int,
        default=300,
        help="minimum reference span (bp); default 300",
    )
    p_aln.add_argument(
        "--min-aligned-fraction",
        type=float,
        default=0.85,
        help=(
            "minimum aligned-bp / read-length per SAM v1 §1.4.6; "
            "default 0.85 (tighter than ENCODE's 0.7 published floor — "
            "appropriate for clean full-length cDNA libraries)"
        ),
    )
    p_aln.add_argument(
        "--min-mapq",
        type=int,
        default=0,
        help="minimum minimap2 MAPQ (default 0 — no filter)",
    )
    p_aln.add_argument(
        "--allow-antisense",
        action="store_true",
        help="keep antisense alignments at the gene-overlap step",
    )
    p_aln.add_argument(
        "--min-overlap-fraction",
        type=float,
        default=0.5,
        help=(
            "minimum overlap_bp / ref_span at the gene-overlap step "
            "(default 0.5)"
        ),
    )
    p_aln.add_argument(
        "--matrix-format",
        choices=("count", "tpm", "both", "none"),
        default="both",
        help=(
            "wide gene × sample TSV outputs to write alongside "
            "feature_quant.parquet: 'count' → gene_counts.tsv only, "
            "'tpm' → gene_tpm.tsv only, 'both' (default) → both, "
            "'none' → skip the wide TSV writes (parquet still emitted)"
        ),
    )
    p_aln.add_argument(
        "--matrix-min-count",
        type=int,
        default=0,
        help=(
            "drop gene rows whose summed count across all samples is "
            "below this threshold (default 0 — keep all annotated "
            "genes including zero-observation rows for stable indexing)"
        ),
    )
    p_aln.add_argument("--resume", action="store_true")
    p_aln.add_argument("--progress", action="store_true")
    p_aln.set_defaults(func=_cmd_transcriptome_align)

    # Cluster reserved for S4 (Mode B de novo)
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


def _cmd_transcriptome_align(args: argparse.Namespace) -> int:
    """Mode A — align demuxed reads to a reference + count reads per gene.

    Three stages: align (minimap2 → sorted indexed BAM), decode_filter_overlap
    (N-way fused worker → partitioned shards), count (resolve-stage hash-join
    against read_demux + group_by-sum to FEATURE_QUANT). Memory budget at
    200M-read scale: ~3 GB gene_assignments + ~10 GB read_demux at the
    resolve stage; alignment table never materialised whole.
    """
    import json
    import sys
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.dataset as pa_dataset
    import pyarrow.parquet as pq

    from constellation.sequencing.align.map import map_to_genome
    from constellation.sequencing.annotation.io import load_annotation
    from constellation.sequencing.parallel import run_batched
    from constellation.sequencing.progress import (
        NullProgress,
        StreamProgress,
    )
    from constellation.sequencing.quant import (
        build_gene_matrix,
        count_reads_per_gene,
        fused_decode_filter_overlap_worker,
        gene_set_from_annotation,
        render_gene_matrix_tsv,
        serialise_gene_set,
    )
    from constellation.sequencing.readers.sam_bam import _bam_alignment_chunks
    from constellation.sequencing.reference.io import load_genome_reference
    from constellation.sequencing.samples import Samples

    # ── Mode dispatch ────────────────────────────────────────────────
    if args.reference is None:
        print(
            "Mode B (de novo clustering) not yet implemented in this "
            "release; pass --reference <ref-root> for reference-guided "
            "gene counting.",
            file=sys.stderr,
        )
        return 2

    demux_dir = Path(args.demux_dir)
    if not demux_dir.is_dir():
        raise FileNotFoundError(f"--demux-dir not found: {demux_dir}")
    demux_manifest_path = demux_dir / "manifest.json"
    if not demux_manifest_path.exists():
        raise FileNotFoundError(
            f"--demux-dir missing manifest.json: {demux_dir}"
        )
    if not (demux_dir / "read_demux").is_dir():
        raise FileNotFoundError(
            f"--demux-dir missing read_demux/: {demux_dir}"
        )
    demux_manifest = json.loads(demux_manifest_path.read_text())

    reference = Path(args.reference)
    genome_dir = reference / "genome"
    annotation_dir = reference / "annotation"
    if not genome_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"--reference must contain genome/ + annotation/ subdirs: "
            f"{reference}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "_SUCCESS").exists() and not args.resume:
        print(
            f"output dir already complete: {output_dir} "
            f"(pass --resume to short-circuit; refusing to overwrite)",
            file=sys.stderr,
        )
        return 1

    cb = StreamProgress() if args.progress else NullProgress()

    genome = load_genome_reference(genome_dir)
    annotation = load_annotation(annotation_dir)
    annotation.validate_against(genome)

    # Source BAM paths from the S1 demux manifest are kept solely for
    # samples-TSV multi-file mapping (the `file` column matches against
    # these basenames). We DO NOT open them — alignment streams trimmed
    # transcript windows from the demux parquet partitions instead, so
    # the raw BAMs may be moved/archived after demux without breaking
    # the align stage.
    bam_inputs = [Path(p) for p in demux_manifest.get("input_files", [])]

    sample_rows, edge_rows, _ = _parse_samples_tsv(
        args.samples,
        input_files=bam_inputs,
        default_acquisition_id=int(demux_manifest.get("acquisition_id", 1)),
    )
    samples = Samples.from_records(samples=sample_rows, edges=edge_rows)

    # ── Stage 1: align ───────────────────────────────────────────────
    bam_success = output_dir / "bam" / "_SUCCESS"
    aligned_bam = output_dir / "bam" / "aligned.bam"
    if args.resume and bam_success.exists() and aligned_bam.exists():
        cb_path = aligned_bam
    else:
        cb_path = map_to_genome(
            demux_dir,
            genome,
            output_dir=output_dir,
            threads=args.threads,
            progress_cb=cb,
        )
        bam_success.parent.mkdir(parents=True, exist_ok=True)
        bam_success.write_bytes(b"")

    # ── Stage 2: decode_filter_overlap (N-way fused worker) ──────────
    chunks = _bam_alignment_chunks(cb_path, chunk_size=args.chunk_size)
    chunk_specs = [
        (str(cb_path), int(vo), int(n), idx)
        for idx, (vo, n) in enumerate(chunks)
    ]
    gene_set = gene_set_from_annotation(annotation, genome)
    gene_set_bytes = serialise_gene_set(gene_set)
    filter_kwargs = {
        "min_length": args.min_length,
        "min_aligned_fraction": args.min_aligned_fraction,
        "min_mapq": args.min_mapq,
    }
    overlap_kwargs = {
        "min_overlap_fraction": args.min_overlap_fraction,
        "allow_antisense": args.allow_antisense,
    }
    stage_outputs = run_batched(
        worker_fn=fused_decode_filter_overlap_worker,
        batches=chunk_specs,
        output_dir=output_dir,
        output_keys=("alignments", "alignment_tags", "gene_assignments", "stats"),
        n_workers=args.threads,
        worker_kwargs={
            "gene_set_bytes": gene_set_bytes,
            "filter_kwargs": filter_kwargs,
            "overlap_kwargs": overlap_kwargs,
        },
        progress_cb=cb,
        resume=args.resume,
        stage_label="decode_filter_overlap",
        total=len(chunk_specs),
    )

    # ── Stage 3: count (resolve stage — hash-join + group_by-sum) ────
    gene_assignments = pa_dataset.dataset(
        stage_outputs["gene_assignments"].directory
    ).to_table()
    read_demux = pa_dataset.dataset(demux_dir / "read_demux").to_table()
    feature_quant, count_stats = count_reads_per_gene(
        gene_assignments, read_demux, samples
    )

    fq_dir = output_dir / "feature_quant" / "feature_origin=gene_id"
    fq_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(feature_quant, fq_dir / "part-00000.parquet")
    (output_dir / "feature_quant" / "_SUCCESS").write_bytes(b"")
    pq.write_table(feature_quant, output_dir / "feature_quant.parquet")

    # ── Wide gene × sample TSVs (human-facing) ───────────────────────
    matrix_outputs: dict[str, str] = {}
    if args.matrix_format != "none":
        if args.matrix_format in ("count", "both"):
            counts_matrix = build_gene_matrix(
                feature_quant, annotation, genome, samples,
                value="count", min_count=args.matrix_min_count,
            )
            counts_path = output_dir / "gene_counts.tsv"
            counts_path.write_text(render_gene_matrix_tsv(counts_matrix))
            matrix_outputs["gene_counts_tsv"] = str(counts_path)
        if args.matrix_format in ("tpm", "both"):
            tpm_matrix = build_gene_matrix(
                feature_quant, annotation, genome, samples,
                value="tpm", min_count=args.matrix_min_count,
            )
            tpm_path = output_dir / "gene_tpm.tsv"
            tpm_path.write_text(render_gene_matrix_tsv(tpm_matrix))
            matrix_outputs["gene_tpm_tsv"] = str(tpm_path)

    # ── Aggregate per-worker stats for the manifest ──────────────────
    if stage_outputs["stats"].shard_paths:
        stats_table = pa_dataset.dataset(
            stage_outputs["stats"].directory
        ).to_table()
        per_stage = {
            "decoded": int(pa.compute.sum(stats_table.column("decoded")).as_py()),
            "after_filter": int(
                pa.compute.sum(stats_table.column("after_filter")).as_py()
            ),
            "after_overlap": int(
                pa.compute.sum(stats_table.column("after_overlap")).as_py()
            ),
        }
    else:
        per_stage = {"decoded": 0, "after_filter": 0, "after_overlap": 0}
    per_stage.update(
        {
            "reads_with_sample": count_stats["reads_with_sample"],
            "reads_without_sample": count_stats["reads_without_sample"],
            "unique_(gene,sample)_pairs": count_stats["unique_(gene,sample)_pairs"],
            "total_count": count_stats["total_count"],
            "samples_normalised": count_stats.get("samples_normalised", 0),
        }
    )

    manifest = {
        "demux_dir": str(demux_dir),
        "reference": str(reference),
        "input_files": [str(p) for p in bam_inputs],
        "parameters": {
            "threads": args.threads,
            "chunk_size": args.chunk_size,
            "min_length": args.min_length,
            "min_aligned_fraction": args.min_aligned_fraction,
            "min_mapq": args.min_mapq,
            "allow_antisense": args.allow_antisense,
            "min_overlap_fraction": args.min_overlap_fraction,
            "matrix_format": args.matrix_format,
            "matrix_min_count": args.matrix_min_count,
        },
        "stages": per_stage,
        "outputs": {
            "bam": str(aligned_bam),
            "alignments": str(stage_outputs["alignments"].directory),
            "alignment_tags": str(stage_outputs["alignment_tags"].directory),
            "gene_assignments": str(stage_outputs["gene_assignments"].directory),
            "feature_quant": str(output_dir / "feature_quant.parquet"),
            **matrix_outputs,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "_SUCCESS").write_bytes(b"")

    if not args.progress:
        print(
            f"align done: {per_stage['decoded']} alignments decoded → "
            f"{per_stage['after_filter']} kept → "
            f"{per_stage['after_overlap']} overlapping a gene → "
            f"{per_stage['unique_(gene,sample)_pairs']} (gene, sample) cells, "
            f"total count {per_stage['total_count']}",
            flush=True,
        )
    return 0


def _build_reference_parser(subs) -> None:
    """Wire the ``constellation reference ...`` sub-tree.

    Verbs:
        import        local-files in (FASTA + optional GFF3) →
                      ParquetDir bundles
        fetch         <source>:<id> → download via stdlib HTTP →
                      same ParquetDir bundles as ``import``
        summary       print contig/feature counts on a saved bundle
        validate      load + run ``.validate()`` + cross-check
    """
    p_ref = subs.add_parser(
        "reference",
        help="Import / fetch / summarise / validate genome+annotation references",
    )
    ref_subs = p_ref.add_subparsers(dest="ref_subcommand", required=True)

    p_imp = ref_subs.add_parser(
        "import",
        help="Import local FASTA (+ optional GFF3) into ParquetDir bundles",
    )
    p_imp.add_argument(
        "--fasta",
        required=True,
        help="genome FASTA path (gzip-aware via .gz suffix)",
    )
    p_imp.add_argument(
        "--gff3",
        default=None,
        help="optional GFF3 path; when omitted, only the GenomeReference is written",
    )
    p_imp.add_argument(
        "--output-dir",
        required=True,
        help=(
            "output directory; receives ``genome/`` and (when --gff3 is given) "
            "``annotation/`` ParquetDir subdirectories"
        ),
    )
    p_imp.set_defaults(func=_cmd_reference_import)

    p_fetch = ref_subs.add_parser(
        "fetch",
        help=(
            "Fetch a genome+GFF3 from Ensembl / Ensembl Genomes / RefSeq via "
            "stdlib HTTP"
        ),
    )
    p_fetch.add_argument(
        "spec",
        help=(
            "<source>:<id> — e.g. 'ensembl_genomes:saccharomyces_cerevisiae', "
            "'refseq:GCF_001708105.1', 'ensembl:human'"
        ),
    )
    p_fetch.add_argument(
        "--output-dir",
        required=True,
        help="output directory for the resulting ParquetDir bundles",
    )
    p_fetch.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds (default 600)",
    )
    p_fetch.set_defaults(func=_cmd_reference_fetch)

    p_sum = ref_subs.add_parser(
        "summary",
        help="Print contig + feature stats for a saved reference bundle",
    )
    p_sum.add_argument(
        "ref_dir",
        help=(
            "directory containing ``genome/`` (and optionally ``annotation/``) "
            "ParquetDir bundles, as written by ``import`` or ``fetch``"
        ),
    )
    p_sum.set_defaults(func=_cmd_reference_summary)

    p_val = ref_subs.add_parser(
        "validate",
        help="Load + validate (PK/FK closure + cross-check) a reference bundle",
    )
    p_val.add_argument("ref_dir")
    p_val.set_defaults(func=_cmd_reference_validate)


def _cmd_reference_import(args: argparse.Namespace) -> int:
    """Local FASTA (+ optional GFF3) → ParquetDir bundles."""
    from pathlib import Path

    from constellation.sequencing.annotation.io import save_annotation
    from constellation.sequencing.readers.fastx import read_fasta_genome
    from constellation.sequencing.readers.gff import read_gff3
    from constellation.sequencing.reference.io import save_genome_reference

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    genome = read_fasta_genome(args.fasta)
    save_genome_reference(genome, out / "genome")

    if args.gff3:
        contig_name_to_id = {
            row["name"]: row["contig_id"]
            for row in genome.contigs.to_pylist()
        }
        annotation = read_gff3(args.gff3, contig_name_to_id=contig_name_to_id)
        annotation.validate_against(genome)
        save_annotation(annotation, out / "annotation")
        print(
            f"imported {genome.n_contigs} contigs ({genome.total_length} bp) + "
            f"{annotation.n_features} features → {out}",
            flush=True,
        )
    else:
        print(
            f"imported {genome.n_contigs} contigs ({genome.total_length} bp) → {out}",
            flush=True,
        )
    return 0


def _cmd_reference_fetch(args: argparse.Namespace) -> int:
    """``<source>:<id>`` → ParquetDir bundles via stdlib HTTP."""
    from constellation.sequencing.reference.fetch import fetch_reference

    result = fetch_reference(args.spec, args.output_dir, timeout=args.timeout)
    n_features = result.annotation.n_features if result.annotation else 0
    print(
        f"fetched {result.genome.n_contigs} contigs ({result.genome.total_length} bp) "
        f"+ {n_features} features → {result.output_dir}",
        flush=True,
    )
    print(f"  genome: {result.sources['genome']}")
    print(f"  annotation: {result.sources['annotation']}")
    return 0


def _cmd_reference_summary(args: argparse.Namespace) -> int:
    """Print stats for a saved reference bundle."""
    from collections import Counter
    from pathlib import Path

    from constellation.sequencing.annotation.io import load_annotation
    from constellation.sequencing.reference.io import load_genome_reference

    root = Path(args.ref_dir)
    genome_dir = root / "genome" if (root / "genome").is_dir() else root
    genome = load_genome_reference(genome_dir)
    print(
        f"GenomeReference: {genome.n_contigs} contig(s), "
        f"{genome.total_length} bp total"
    )

    annotation_dir = root / "annotation"
    if annotation_dir.is_dir():
        annotation = load_annotation(annotation_dir)
        print(f"Annotation: {annotation.n_features} feature(s)")
        type_counts = Counter(annotation.features.column("type").to_pylist())
        for ftype, n in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {ftype}: {n}")
        # Are exons present? — informs whether TranscriptReference.from_annotation
        # would yield non-empty output.
        if "exon" in type_counts and ("mRNA" in type_counts or "transcript" in type_counts):
            print("  → TranscriptReference.from_annotation is viable")
    else:
        print("Annotation: (none — only genome was imported/fetched)")
    return 0


def _cmd_reference_validate(args: argparse.Namespace) -> int:
    """Load + run validate() + validate_against on a reference bundle."""
    from pathlib import Path

    from constellation.sequencing.annotation.io import load_annotation
    from constellation.sequencing.reference.io import load_genome_reference

    root = Path(args.ref_dir)
    genome_dir = root / "genome" if (root / "genome").is_dir() else root
    genome = load_genome_reference(genome_dir)
    print(f"  genome: {genome.n_contigs} contigs ok")

    annotation_dir = root / "annotation"
    if annotation_dir.is_dir():
        annotation = load_annotation(annotation_dir)
        annotation.validate_against(genome)
        print(f"  annotation: {annotation.n_features} features, FK closure ok")
    print("validation passed", flush=True)
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
