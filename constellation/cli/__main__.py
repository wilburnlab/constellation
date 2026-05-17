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

    # Frontend bundle row(s) — one per known entry. Doesn't depend on the
    # [viz] extras (we just stat a directory + read JSON), so this runs
    # even when fastapi / uvicorn / datashader aren't installed.
    rows.extend(_doctor_frontend_rows())

    # Reference cache row — single line summarising the per-user cache.
    rows.append(_doctor_reference_cache_row())

    if not rows:
        print("no third-party tools or frontend bundles registered")
        return 0

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


def _doctor_frontend_rows() -> list[tuple[str, str, str, str]]:
    """Probe ``constellation/viz/static/<entry>/`` for installed bundles.

    Iterates over every known entry (``genome``, ``dashboard``) and
    surfaces one row per entry. Reads `bundle.json` via
    `constellation.viz.install.read_bundle_metadata` — a stdlib-only
    helper, so this runs even without the `[viz]` extras installed.
    """
    from pathlib import Path

    from constellation.viz.install import read_bundle_metadata

    static_root = Path(__file__).resolve().parents[1] / "viz" / "static"
    rows: list[tuple[str, str, str, str]] = []
    for entry in ("genome", "dashboard"):
        entry_dir = static_root / entry
        metadata = read_bundle_metadata(entry_dir)
        # Vite's HTML output is named after the source file
        # (`index.<entry>.html` in the current source layout); accept
        # either that or a vanilla `index.html` for forward-compat with
        # future builds that rename the output.
        bundle_present = entry_dir.is_dir() and (
            (entry_dir / f"index.{entry}.html").is_file()
            or (entry_dir / "index.html").is_file()
        )
        name = f"viz frontend ({entry})"
        if bundle_present:
            version = (
                str(metadata.get("constellation_version", "unknown"))
                if metadata
                else "unknown"
            )
            rows.append((name, "ok", version, str(entry_dir)))
        else:
            rows.append(
                (
                    name,
                    "not installed",
                    "—",
                    "(run: constellation viz install-frontend --from <tarball>)",
                )
            )
    return rows


def _doctor_reference_cache_row() -> tuple[str, str, str, str]:
    """Report on the per-user reference cache.

    Status: ``ok`` if at least one installed reference parses; ``empty``
    if the cache root exists but has no entries; ``warn`` if any entry
    has issues (dangling current symlink, missing meta.toml, leftover
    ``.partial/`` scratch dirs).
    """
    from constellation.sequencing.reference.handle import (
        cache_root,
        format_size,
        list_installed,
    )

    root = cache_root()
    name = "reference cache"
    # An empty/missing cache is expected on a fresh install — report "ok"
    # so doctor doesn't fail CI for users who haven't fetched anything yet.
    if not root.is_dir():
        return (name, "ok", "0 refs", f"(not yet populated: {root})")
    entries = list_installed()
    warnings = _cache_integrity_warnings(root)
    total_size = sum((e.size_bytes or 0) for e in entries)
    if not entries and not warnings:
        return (name, "ok", "0 refs", f"(no installs under {root})")
    if warnings:
        # Surface a one-line warning; full detail visible via `reference list`.
        first = warnings[0]
        suffix = f" (+{len(warnings) - 1} more)" if len(warnings) > 1 else ""
        return (
            name,
            "warn",
            f"{len(entries)} refs",
            f"{root} — {first}{suffix}",
        )
    return (
        name,
        "ok",
        f"{len(entries)} refs",
        f"{format_size(total_size)} at {root}",
    )


def _cache_integrity_warnings(root) -> list[str]:
    """Walk the cache and collect human-readable warning lines."""
    from constellation.sequencing.reference.handle import (
        CURRENT_SYMLINK,
        CURRENT_TEXTFILE,
        META_FILENAME,
    )

    warnings: list[str] = []
    for organism_dir in sorted(root.iterdir()):
        if not organism_dir.is_dir():
            continue
        # Dangling current symlink?
        sym = organism_dir / CURRENT_SYMLINK
        if sym.is_symlink() and not sym.exists():
            warnings.append(f"dangling current symlink: {sym}")
        # current.txt without matching dir?
        txt = organism_dir / CURRENT_TEXTFILE
        if txt.exists():
            target = (organism_dir / txt.read_text(encoding="utf-8").strip())
            if not target.is_dir():
                warnings.append(f"stale current.txt: {txt} → {target}")
        for entry in organism_dir.iterdir():
            name = entry.name
            if name in (CURRENT_SYMLINK, CURRENT_TEXTFILE):
                continue
            if name.endswith(".partial"):
                warnings.append(f"leftover partial fetch: {entry}")
                continue
            if name.endswith(".lock"):
                continue
            if entry.is_dir() and not entry.is_symlink():
                if not (entry / META_FILENAME).exists():
                    warnings.append(f"missing meta.toml: {entry}")
    return warnings


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

    # Visualization subtree — `constellation viz genome --session DIR`.
    # The full `[viz]` extras (fastapi / uvicorn / datashader) are
    # only required when the user actually invokes a viz subcommand;
    # the parser registration is cheap import-wise.
    from constellation.viz.cli import (
        build_dashboard_parser as _build_dashboard_parser,
    )
    from constellation.viz.cli import build_parser as _build_viz_parser
    _build_viz_parser(subs)
    # Top-level `constellation dashboard` (also the bare-argv default).
    _build_dashboard_parser(subs)

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
    # Alignment-derived intermediates. Block extraction + intron
    # clustering run by default — they're a structural characteristic
    # of mapping transcripts to a genome, not an opt-in. The --no-*
    # flags exist for diagnostic / footprint-sensitive runs.
    p_aln.add_argument(
        "--no-cluster-junctions",
        action="store_true",
        help=(
            "skip the support-ranked intron-clustering pass. introns.parquet "
            "still emits, but each row gets intron_id=row_index and "
            "is_intron_seed=True (each observed position is its own "
            "singleton cluster — equivalent to --intron-tolerance-bp 0). "
            "Use for diagnostic comparisons against unclustered Dorado "
            "output; not the recommended default."
        ),
    )
    p_aln.add_argument(
        "--intron-tolerance-bp",
        type=int,
        default=5,
        help=(
            "absorption window for the intron clustering pass — a raw "
            "junction at (d, a) is absorbed into an existing seed at "
            "(d_s, a_s) iff |d-d_s| <= W AND |a-a_s| <= W on both axes. "
            "Default 5 absorbs basecaller jitter while preserving genuine "
            "alt-5'SS / alt-3'SS events."
        ),
    )
    p_aln.add_argument(
        "--intron-motif-priority",
        type=str,
        default="GT-AG,GC-AG,AT-AC",
        help=(
            "comma-separated motif tiebreak order for intron clustering. "
            "Earlier entries win on read_count ties. Default GT-AG,GC-AG,"
            "AT-AC follows the canonical splice biology hierarchy."
        ),
    )
    p_aln.add_argument(
        "--no-coverage",
        action="store_true",
        help=(
            "skip writing coverage.parquet — the RLE-encoded per-(contig, "
            "sample) depth track. When --no-derived-annotation is also "
            "set, the in-memory pile-up isn't computed at all (saves "
            "resolve-stage time); otherwise it's computed for the "
            "derived-annotation pass but not persisted."
        ),
    )
    p_aln.add_argument(
        "--no-derived-annotation",
        action="store_true",
        help=(
            "skip the derived-annotation pass. When set, "
            "derived_annotation/ + block_exon_assignments.parquet + "
            "exon_psi.parquet are NOT written."
        ),
    )
    p_aln.add_argument(
        "--min-exon-depth",
        type=int,
        default=5,
        help=(
            "derived-annotation: minimum per-position coverage depth "
            "to seed an exon candidate. Default 5."
        ),
    )
    p_aln.add_argument(
        "--min-intron-read-count",
        type=int,
        default=3,
        help=(
            "derived-annotation: minimum reads supporting an intron "
            "cluster (group_by(intron_id).sum(read_count)) to treat it "
            "as a trusted boundary. Default 3."
        ),
    )
    p_aln.add_argument(
        "--emit-cs-tags",
        action="store_true",
        help=(
            "write alignment_cs/ partitioned dataset preserving "
            "minimap2's cs:long string per alignment. Required by "
            "'transcriptome cluster --build-consensus' for base-resolution "
            "PWM accumulation. Off by default."
        ),
    )
    p_aln.add_argument(
        "--intron-min-bp",
        type=int,
        default=25,
        help=(
            "CIGAR D operations >= this length break a block (used as "
            "intron proxy for assembly-vs-genome where minimap2 emits "
            "D rather than N). Default 25; ignored on the cs:long path "
            "which has its own splice operator."
        ),
    )
    p_aln.add_argument("--resume", action="store_true")
    p_aln.add_argument("--progress", action="store_true")
    p_aln.set_defaults(func=_cmd_transcriptome_align)

    # ── Cluster (Phase 2 genome-guided shipped; --mode de-novo TBD) ──
    p_cluster = tx_subs.add_parser(
        "cluster",
        help=(
            "group reads into transcript / isoform clusters. "
            "Phase 2 --mode genome-guided is wired; --mode de-novo "
            "(Phase 3) and --mode validate (Phase 4) reserved."
        ),
    )
    p_cluster.add_argument(
        "--align-dir",
        required=True,
        help=(
            "directory produced by `transcriptome align` "
            "(must contain alignments/ + alignment_blocks/ + introns.parquet)."
        ),
    )
    p_cluster.add_argument(
        "--demux-dir",
        required=True,
        help="S1 demux output dir; supplies reads/ + read_demux/.",
    )
    p_cluster.add_argument(
        "--reference",
        default=None,
        help=(
            "reference root with genome/ + annotation/ subdirs; "
            "required when --build-consensus is set."
        ),
    )
    p_cluster.add_argument(
        "--samples",
        required=True,
        help="samples TSV (consistent with the upstream align run).",
    )
    p_cluster.add_argument(
        "--output-dir",
        required=True,
        help="cluster output dir (clusters.parquet, cluster.fa, ...).",
    )
    p_cluster.add_argument(
        "--mode",
        choices=("genome-guided",),
        default="genome-guided",
        help=(
            "clustering mode (Phase 2 ships only genome-guided; Phase 3 "
            "adds de-novo; Phase 4 adds validate)."
        ),
    )
    p_cluster.add_argument(
        "--max-5p-drift",
        type=int,
        default=25,
        help=(
            "drop a read from a fingerprint cluster when its 5' end "
            "(strand-aware) deviates from the cluster median by more "
            "than this many bp. Default 25."
        ),
    )
    p_cluster.add_argument(
        "--max-3p-drift",
        type=int,
        default=75,
        help=(
            "as --max-5p-drift but on the 3' end (polyA scatter is wider). "
            "Default 75."
        ),
    )
    p_cluster.add_argument(
        "--intron-tolerance-bp",
        type=int,
        default=None,
        help=(
            "intron-clustering tolerance (the +/- bp absorption window). "
            "When omitted, the value baked into the upstream align run "
            "(read from align_dir/manifest.json) is used as-is. When "
            "set, cluster re-runs cluster_junctions on the raw per-"
            "position rows in introns.parquet before fingerprinting — "
            "no need to re-run align. Replaces the retired "
            "--intron-quantum-bp."
        ),
    )
    p_cluster.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help=(
            "drop clusters whose surviving (post-drift-filter) size is "
            "below this. Default 1 keeps singletons."
        ),
    )
    p_cluster.add_argument(
        "--drop-drift-filtered",
        action="store_true",
        help=(
            "drop drift-filtered reads from cluster_membership "
            "entirely. Default keeps them with role='drift_filtered' "
            "for Phase 4 cross-validation."
        ),
    )
    p_cluster.add_argument(
        "--build-consensus",
        action="store_true",
        help=(
            "populate consensus_sequence per cluster via the shared "
            "weighted-PWM kernel against the genome window. Requires "
            "the upstream align run to have set --emit-cs-tags."
        ),
    )
    p_cluster.add_argument(
        "--per-sample-clusters",
        action="store_true",
        help=(
            "partition by sample_id in addition to (contig, strand). "
            "Default treats clusters as spanning samples."
        ),
    )
    p_cluster.add_argument(
        "--write-fasta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "write cluster.fa with cluster representative sequences "
            "(or consensus when --build-consensus is set). "
            "Default on; pass --no-write-fasta to disable."
        ),
    )
    p_cluster.add_argument(
        "--write-summary",
        action="store_true",
        help=(
            "write cluster_summary.tsv (cluster-size histogram + "
            "n_singletons + fraction_clustered). Default off."
        ),
    )
    p_cluster.add_argument("--threads", type=int, default=1)
    p_cluster.add_argument("--resume", action="store_true")
    p_cluster.add_argument("--progress", action="store_true")
    p_cluster.set_defaults(func=_cmd_transcriptome_cluster)


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
        aggregate_junctions,
        build_gene_matrix,
        build_pileup,
        cluster_junctions,
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

    try:
        resolved_ref, ref_source = _resolve_reference_argument(args.reference)
    except Exception as exc:  # ValueError / ReferenceNotInstalledError
        print(f"error: --reference {args.reference!r}: {exc}", file=sys.stderr)
        return 1
    reference = Path(resolved_ref)
    genome_dir = reference / "genome"
    annotation_dir = reference / "annotation"
    if not genome_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"--reference must contain genome/ + annotation/ subdirs: "
            f"{reference} (source: {ref_source})"
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
    # Block extraction is unconditional now — it's the substrate for
    # the always-on intron-clustering and (when not opted out)
    # derived-annotation passes downstream.
    emit_blocks = True
    emit_cs = bool(args.emit_cs_tags)
    cluster_introns = not bool(args.no_cluster_junctions)
    emit_coverage = not bool(args.no_coverage)
    emit_derived_annotation = not bool(args.no_derived_annotation)
    intron_motif_priority = tuple(
        m.strip() for m in str(args.intron_motif_priority).split(",")
        if m.strip()
    )

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
    output_keys: tuple[str, ...] = (
        "alignments", "alignment_tags", "gene_assignments", "stats"
    )
    if emit_blocks:
        output_keys = output_keys + ("alignment_blocks",)
    if emit_cs:
        output_keys = output_keys + ("alignment_cs",)
    worker_kwargs: dict = {
        "gene_set_bytes": gene_set_bytes,
        "filter_kwargs": filter_kwargs,
        "overlap_kwargs": overlap_kwargs,
    }
    if emit_blocks:
        worker_kwargs["emit_blocks"] = True
        worker_kwargs["intron_min_bp"] = int(args.intron_min_bp)
    if emit_cs:
        worker_kwargs["emit_cs"] = True
    stage_outputs = run_batched(
        worker_fn=fused_decode_filter_overlap_worker,
        batches=chunk_specs,
        output_dir=output_dir,
        output_keys=output_keys,
        n_workers=args.threads,
        worker_kwargs=worker_kwargs,
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

    # ── Resolve-stage aggregations ────────────────────────────────────
    # alignments + alignment_blocks are materialised on demand here.
    # Memory at 200M-read scale: ~25 GB blocks + ~30 GB alignments;
    # cluster-node-friendly but a workstation-RAM stress test.
    # Streaming variant lifts in when we cross 1B alignments.
    emit_outputs: dict[str, str] = {}
    alignments_table = pa_dataset.dataset(
        stage_outputs["alignments"].directory
    ).to_table()
    blocks_table = pa_dataset.dataset(
        stage_outputs["alignment_blocks"].directory
    ).to_table()

    # ── Intron clustering — always-on (introns.parquet) ──────────────
    # The clustered INTRON_TABLE is the canonical splice-junction
    # artifact downstream consumers (transcriptome cluster,
    # derived-annotation pass) consume.
    introns = aggregate_junctions(
        blocks_table, alignments_table, genome, annotation=annotation,
    )
    if cluster_introns:
        introns = cluster_junctions(
            introns,
            tolerance_bp=int(args.intron_tolerance_bp),
            motif_priority=intron_motif_priority,
        )
    introns_path = output_dir / "introns.parquet"
    pq.write_table(introns, introns_path)
    emit_outputs["introns"] = str(introns_path)

    # ── Coverage pile-up + derived annotation ─────────────────────────
    # Both are default-on; the in-memory pile-up is consumed by the
    # derived-annotation pass even when --no-coverage suppresses the
    # on-disk write. Skipping both passes saves the resolve-stage
    # pile-up compute entirely.
    pileup_table: pa.Table | None = None
    need_pileup = emit_coverage or emit_derived_annotation
    if need_pileup:
        rd = read_demux.select(["read_id", "sample_id"])
        valid_mask = pa.compute.is_valid(rd.column("sample_id"))
        rd_valid = rd.filter(valid_mask)
        # If the same read appears in multiple demux rows (chimeras),
        # require all rows to agree on sample_id — match the count
        # stage's policy.
        rd_unique = rd_valid.group_by("read_id").aggregate(
            [("sample_id", "min"), ("sample_id", "max")]
        )
        rd_consistent = rd_unique.filter(
            pa.compute.equal(
                rd_unique.column("sample_id_min"),
                rd_unique.column("sample_id_max"),
            )
        )
        read_to_sample = {
            str(rid): int(sid)
            for rid, sid in zip(
                rd_consistent.column("read_id").to_pylist(),
                rd_consistent.column("sample_id_min").to_pylist(),
                strict=True,
            )
        }
        pileup_table = build_pileup(
            blocks_table, alignments_table, genome.contigs,
            read_to_sample=read_to_sample,
        )

    if emit_coverage and pileup_table is not None:
        pileup_path = output_dir / "coverage.parquet"
        pq.write_table(pileup_table, pileup_path)
        emit_outputs["coverage"] = str(pileup_path)

    if emit_derived_annotation and pileup_table is not None:
        # Local import: derived_annotation depends on Part B schemas
        # registered when the module is imported.
        from constellation.sequencing.quant.derived_annotation import (
            build_derived_annotation,
        )
        from constellation.sequencing.annotation.io import save_annotation

        derived_annotation, block_assignments, exon_psi = (
            build_derived_annotation(
                coverage=pileup_table,
                introns=introns,
                alignment_blocks=blocks_table,
                alignments=alignments_table,
                contigs=genome.contigs,
                read_to_sample=read_to_sample,
                min_exon_depth=int(args.min_exon_depth),
                min_intron_read_count=int(args.min_intron_read_count),
            )
        )
        derived_dir = output_dir / "derived_annotation"
        save_annotation(derived_annotation, derived_dir, format="parquet_dir")
        block_assignments_path = output_dir / "block_exon_assignments.parquet"
        pq.write_table(block_assignments, block_assignments_path)
        exon_psi_path = output_dir / "exon_psi.parquet"
        pq.write_table(exon_psi, exon_psi_path)
        emit_outputs["derived_annotation"] = str(derived_dir)
        emit_outputs["block_exon_assignments"] = str(block_assignments_path)
        emit_outputs["exon_psi"] = str(exon_psi_path)

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

    manifest_outputs: dict[str, str] = {
        "bam": str(aligned_bam),
        "alignments": str(stage_outputs["alignments"].directory),
        "alignment_tags": str(stage_outputs["alignment_tags"].directory),
        "gene_assignments": str(stage_outputs["gene_assignments"].directory),
        "feature_quant": str(output_dir / "feature_quant.parquet"),
    }
    if emit_blocks and "alignment_blocks" in stage_outputs:
        manifest_outputs["alignment_blocks"] = str(
            stage_outputs["alignment_blocks"].directory
        )
    if emit_cs and "alignment_cs" in stage_outputs:
        manifest_outputs["alignment_cs"] = str(
            stage_outputs["alignment_cs"].directory
        )
    manifest_outputs.update(emit_outputs)
    manifest_outputs.update(matrix_outputs)

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
            "no_cluster_junctions": bool(args.no_cluster_junctions),
            "intron_tolerance_bp": int(args.intron_tolerance_bp),
            "intron_motif_priority": str(args.intron_motif_priority),
            "no_coverage": bool(args.no_coverage),
            "no_derived_annotation": bool(args.no_derived_annotation),
            "min_exon_depth": int(args.min_exon_depth),
            "min_intron_read_count": int(args.min_intron_read_count),
            "emit_cs_tags": bool(emit_cs),
            "intron_min_bp": int(args.intron_min_bp),
        },
        "stages": per_stage,
        "outputs": manifest_outputs,
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


def _cmd_transcriptome_cluster(args: argparse.Namespace) -> int:
    """Phase 2 — group reads into transcript / isoform clusters.

    Inputs: a `transcriptome align` output dir (alignment_blocks/ +
    introns.parquet) + the S1 demux dir + samples.tsv. Outputs:
    clusters.parquet, cluster_membership.parquet, optional cluster.fa +
    cluster_summary.tsv.

    Fingerprints are derived fresh at every invocation against the
    INTRON_TABLE in introns.parquet, so the ``--intron-tolerance-bp``
    knob is a clustering parameter rather than an alignment parameter
    — sweeping it never requires re-running align (we re-cluster the
    raw per-position rows on the fly).
    """
    import json
    from collections import Counter
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pa_dataset
    import pyarrow.parquet as pq

    from constellation.sequencing.progress import (
        NullProgress,
        StreamProgress,
    )
    from constellation.sequencing.quant import cluster_junctions
    from constellation.sequencing.reference.io import load_genome_reference
    from constellation.sequencing.samples import Samples
    from constellation.sequencing.transcriptome.cluster_genome import (
        cluster_by_fingerprint,
    )
    from constellation.sequencing.transcriptome.fingerprints import (
        compute_read_fingerprints,
    )

    if args.mode != "genome-guided":  # pragma: no cover — argparse-gated
        print(
            f"--mode {args.mode} not yet implemented; only "
            "genome-guided is wired in this release.",
            file=sys.stderr,
        )
        return 2

    align_dir = Path(args.align_dir)
    blocks_success = align_dir / "alignment_blocks" / "_SUCCESS"
    if not blocks_success.exists():
        print(
            f"--align-dir missing alignment_blocks/_SUCCESS: {align_dir}\n"
            "rerun `transcriptome align` (block extraction is now "
            "always-on; alignment_blocks/ should always be present in "
            "a successful align output).",
            file=sys.stderr,
        )
        return 2
    introns_path = align_dir / "introns.parquet"
    if not introns_path.exists():
        print(
            f"--align-dir missing introns.parquet: {align_dir}\n"
            "rerun `transcriptome align` (intron emission is now "
            "always-on; introns.parquet is the cluster-key substrate).",
            file=sys.stderr,
        )
        return 2
    if args.build_consensus:
        cs_success = align_dir / "alignment_cs" / "_SUCCESS"
        if not cs_success.exists():
            print(
                f"--build-consensus requires alignment_cs/_SUCCESS in "
                f"{align_dir}; rerun `transcriptome align` with "
                "--emit-cs-tags.",
                file=sys.stderr,
            )
            return 2
        if args.reference is None:
            print(
                "--build-consensus requires --reference <ref-root> for "
                "the genome window.",
                file=sys.stderr,
            )
            return 2

    demux_dir = Path(args.demux_dir)
    if not demux_dir.is_dir():
        raise FileNotFoundError(f"--demux-dir not found: {demux_dir}")
    if not (demux_dir / "read_demux").is_dir():
        raise FileNotFoundError(
            f"--demux-dir missing read_demux/: {demux_dir}"
        )
    if not (demux_dir / "reads").is_dir():
        raise FileNotFoundError(
            f"--demux-dir missing reads/: {demux_dir}"
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
    if (output_dir / "_SUCCESS").exists() and args.resume:
        print(f"--resume: output already complete: {output_dir}", flush=True)
        return 0

    # Phase 2 v1 runs single-process; the progress callback is plumbed
    # in when we lift cluster_by_fingerprint into run_batched (per-
    # contig partition parallelism). Until then there are no
    # well-defined inner stages to emit start/progress/done events on,
    # so we leave NullProgress as a no-op placeholder.
    _ = StreamProgress() if args.progress else NullProgress()

    # ── Resolve align-dir manifest for downstream provenance ─────────
    align_manifest_path = align_dir / "manifest.json"
    align_manifest: dict = {}
    if align_manifest_path.exists():
        align_manifest = json.loads(align_manifest_path.read_text())

    # ── Genome (only when --build-consensus) ─────────────────────────
    genome = None
    if args.reference is not None:
        try:
            resolved_ref, ref_source = _resolve_reference_argument(args.reference)
        except Exception as exc:  # ValueError / ReferenceNotInstalledError
            print(
                f"error: --reference {args.reference!r}: {exc}", file=sys.stderr
            )
            return 1
        reference = Path(resolved_ref)
        genome_dir = reference / "genome"
        if not genome_dir.is_dir():
            raise FileNotFoundError(
                f"--reference must contain genome/: {reference} "
                f"(source: {ref_source})"
            )
        genome = load_genome_reference(genome_dir)

    # ── Samples (consistent with the align run) ──────────────────────
    bam_inputs = [
        Path(p) for p in align_manifest.get("input_files", [])
    ]
    sample_rows, edge_rows, _ = _parse_samples_tsv(
        args.samples,
        input_files=bam_inputs,
        default_acquisition_id=int(align_manifest.get("acquisition_id", 1)),
    )
    samples = Samples.from_records(samples=sample_rows, edges=edge_rows)

    # ── Materialise alignments + alignment_blocks ────────────────────
    # Memory budget at 200M reads: ~30 GB alignments + ~25 GB blocks.
    # Workstation-friendly for single-flowcell-shaped runs; streaming
    # variant lifts in when crossing 1B alignments — the cluster_by_
    # fingerprint surface stays the same.
    alignments_table = pa_dataset.dataset(align_dir / "alignments").to_table()
    blocks_table = pa_dataset.dataset(
        align_dir / "alignment_blocks"
    ).to_table()
    alignment_cs_table: pa.Table | None = None
    if args.build_consensus:
        alignment_cs_table = pa_dataset.dataset(
            align_dir / "alignment_cs"
        ).to_table()

    # Genome contig table for fingerprint derivation. When --build-
    # consensus is off we still need contig names → contig_ids, but
    # we don't need full sequences. If no --reference, build the contig
    # table inline from alignments.
    if genome is not None:
        contigs_table = genome.contigs
    else:
        unique_refs = pc.unique(alignments_table.column("ref_name"))
        contigs_table = pa.table(
            {
                "contig_id": pa.array(
                    list(range(len(unique_refs))), type=pa.int64()
                ),
                "name": unique_refs,
                "length": pa.array(
                    [0] * len(unique_refs), type=pa.int64()
                ),
            }
        )

    # ── Load introns.parquet; optionally re-cluster ─────────────────
    introns_table = pq.read_table(align_dir / "introns.parquet")

    # When the user explicitly overrides --intron-tolerance-bp at
    # cluster time, re-run cluster_junctions against the raw per-position
    # rows (each row in introns.parquet is still a distinct observed
    # (donor, acceptor) pair). The motif-priority list rides as a
    # comma-separated align-manifest value.
    align_params = align_manifest.get("parameters", {})
    align_tolerance = align_params.get("intron_tolerance_bp")
    align_motif_priority = str(
        align_params.get("intron_motif_priority", "GT-AG,GC-AG,AT-AC")
    )
    intron_motif_priority = tuple(
        m.strip() for m in align_motif_priority.split(",") if m.strip()
    )
    effective_tolerance: int | None = args.intron_tolerance_bp
    if (
        effective_tolerance is not None
        and align_tolerance is not None
        and int(effective_tolerance) != int(align_tolerance)
    ):
        # Override differs from align — re-cluster.
        introns_table = cluster_junctions(
            introns_table,
            tolerance_bp=int(effective_tolerance),
            motif_priority=intron_motif_priority,
        )
    if effective_tolerance is None:
        effective_tolerance = (
            int(align_tolerance) if align_tolerance is not None else 5
        )

    # ── Derive fingerprints fresh against the (possibly re-clustered)
    # INTRON_TABLE ───────────────────────────────────────────────────
    fingerprints_table = compute_read_fingerprints(
        blocks_table,
        alignments_table,
        contigs_table,
        introns_table,
    )

    # ── Pull trimmed transcript-window sequences for clustered reads ─
    fp_read_ids = set(fingerprints_table.column("read_id").to_pylist())
    fp_read_id_array = pa.array(sorted(fp_read_ids))

    read_demux_columns = [
        "read_id", "transcript_start", "transcript_end", "sample_id",
        "transcript_segment_index",
    ]
    read_demux_filter = pc.is_in(
        pa_dataset.field("read_id"), value_set=fp_read_id_array
    )
    read_demux = pa_dataset.dataset(demux_dir / "read_demux").to_table(
        columns=read_demux_columns,
        filter=read_demux_filter,
    )
    # One transcript window per read for clustering — pick the lowest
    # transcript_segment_index per read_id (chimera resolution downstream
    # is out of scope for genome-guided clustering v1).
    if read_demux.num_rows > 0:
        # Sort, then dedup by read_id keeping first (lowest index).
        read_demux = read_demux.sort_by(
            [("read_id", "ascending"), ("transcript_segment_index", "ascending")]
        )
        # Keep first occurrence per read_id.
        seen: set[str] = set()
        keep_rows: list[int] = []
        rid_col = read_demux.column("read_id").to_pylist()
        for i, rid in enumerate(rid_col):
            if rid in seen:
                continue
            seen.add(rid)
            keep_rows.append(i)
        if len(keep_rows) < read_demux.num_rows:
            read_demux = read_demux.take(pa.array(keep_rows))

    reads_columns = ["read_id", "sequence"]
    reads_dataset = pa_dataset.dataset(demux_dir / "reads")
    reads_schema_names = reads_dataset.schema.names
    if "dorado_quality" in reads_schema_names:
        reads_columns.append("dorado_quality")
    reads_filter = pc.is_in(
        pa_dataset.field("read_id"), value_set=fp_read_id_array
    )
    raw_reads = reads_dataset.to_table(
        columns=reads_columns, filter=reads_filter
    )

    # Join + slice trimmed windows into a flat (read_id, sequence,
    # dorado_quality) table for cluster_by_fingerprint.
    trimmed_table: pa.Table
    if raw_reads.num_rows == 0 or read_demux.num_rows == 0:
        trimmed_table = pa.table(
            {
                "read_id": pa.array([], type=pa.string()),
                "sequence": pa.array([], type=pa.string()),
                "dorado_quality": pa.array([], type=pa.float32()),
            }
        )
    else:
        joined = raw_reads.join(
            read_demux.select(
                ["read_id", "transcript_start", "transcript_end"]
            ),
            keys="read_id",
            join_type="inner",
        )
        rid_list = joined.column("read_id").to_pylist()
        seq_list = joined.column("sequence").to_pylist()
        ts_list = joined.column("transcript_start").to_pylist()
        te_list = joined.column("transcript_end").to_pylist()
        trimmed_seqs: list[str] = []
        for seq, ts, te in zip(seq_list, ts_list, te_list, strict=True):
            if seq is None:
                trimmed_seqs.append("")
                continue
            trimmed_seqs.append(seq[int(ts):int(te)])
        if "dorado_quality" in joined.schema.names:
            quality_list = joined.column("dorado_quality").to_pylist()
        else:
            quality_list = [None] * len(rid_list)
        trimmed_table = pa.table(
            {
                "read_id": pa.array(rid_list, type=pa.string()),
                "sequence": pa.array(trimmed_seqs, type=pa.string()),
                "dorado_quality": pa.array(
                    quality_list, type=pa.float32()
                ),
            }
        )

    # ── read_to_sample for --per-sample-clusters ─────────────────────
    read_to_sample: dict[str, int] | None = None
    if args.per_sample_clusters and read_demux.num_rows > 0:
        rd_valid = read_demux.filter(
            pc.is_valid(read_demux.column("sample_id"))
        )
        read_to_sample = {
            str(rid): int(sid)
            for rid, sid in zip(
                rd_valid.column("read_id").to_pylist(),
                rd_valid.column("sample_id").to_pylist(),
                strict=True,
            )
        }

    # ── Cluster ──────────────────────────────────────────────────────
    clusters, membership = cluster_by_fingerprint(
        fingerprints_table,
        trimmed_table,
        alignments=alignments_table,
        alignment_blocks=blocks_table,
        alignment_cs=alignment_cs_table,
        genome=genome,
        read_to_sample=read_to_sample,
        max_5p_drift=int(args.max_5p_drift),
        max_3p_drift=int(args.max_3p_drift),
        min_cluster_size=int(args.min_cluster_size),
        build_consensus_seq=bool(args.build_consensus),
        drop_drift_filtered=bool(args.drop_drift_filtered),
        per_sample_clusters=bool(args.per_sample_clusters),
        cluster_id_seed=0,
    )

    # ── Write outputs ────────────────────────────────────────────────
    pq.write_table(clusters, output_dir / "clusters.parquet")
    pq.write_table(membership, output_dir / "cluster_membership.parquet")

    cluster_outputs: dict[str, str] = {
        "clusters": str(output_dir / "clusters.parquet"),
        "cluster_membership": str(output_dir / "cluster_membership.parquet"),
    }

    # cluster.fa — representative sequence (or consensus when available).
    if args.write_fasta:
        rep_lookup: dict[str, str] = {
            str(rid): str(seq)
            for rid, seq in zip(
                trimmed_table.column("read_id").to_pylist(),
                trimmed_table.column("sequence").to_pylist(),
                strict=True,
            )
        }
        fasta_path = output_dir / "cluster.fa"
        with fasta_path.open("w") as fh:
            for row in clusters.to_pylist():
                cid = row["cluster_id"]
                rep = row["representative_read_id"]
                consensus = row.get("consensus_sequence")
                seq = consensus if consensus else rep_lookup.get(rep, "")
                if not seq:
                    continue
                fh.write(f">cluster_{cid} representative={rep}\n{seq}\n")
        cluster_outputs["cluster_fa"] = str(fasta_path)

    # cluster_summary.tsv — size histogram + n_singletons + fraction.
    if args.write_summary:
        n_clusters = clusters.num_rows
        sizes = clusters.column("n_reads").to_pylist() if n_clusters else []
        size_hist = Counter(sizes)
        n_singletons = sum(1 for s in sizes if s == 1)
        n_input_reads = fingerprints_table.num_rows
        n_clustered = sum(sizes)
        fraction_clustered = (
            float(n_clustered) / float(n_input_reads)
            if n_input_reads > 0 else 0.0
        )
        summary_path = output_dir / "cluster_summary.tsv"
        with summary_path.open("w") as fh:
            fh.write("metric\tvalue\n")
            fh.write(f"n_clusters\t{n_clusters}\n")
            fh.write(f"n_singletons\t{n_singletons}\n")
            fh.write(f"n_input_reads\t{n_input_reads}\n")
            fh.write(f"n_clustered_reads\t{n_clustered}\n")
            fh.write(f"fraction_clustered\t{fraction_clustered:.6f}\n")
            fh.write("\n")
            fh.write("cluster_size\tn_clusters\n")
            for size in sorted(size_hist.keys()):
                fh.write(f"{size}\t{size_hist[size]}\n")
        cluster_outputs["cluster_summary_tsv"] = str(summary_path)

    # ── Manifest + _SUCCESS ──────────────────────────────────────────
    manifest = {
        "align_dir": str(align_dir),
        "demux_dir": str(demux_dir),
        "reference": (str(args.reference) if args.reference else None),
        "samples_path": str(args.samples),
        "parameters": {
            "mode": str(args.mode),
            "max_5p_drift": int(args.max_5p_drift),
            "max_3p_drift": int(args.max_3p_drift),
            "intron_tolerance_bp": int(effective_tolerance),
            "intron_motif_priority": ",".join(intron_motif_priority),
            "min_cluster_size": int(args.min_cluster_size),
            "drop_drift_filtered": bool(args.drop_drift_filtered),
            "build_consensus": bool(args.build_consensus),
            "per_sample_clusters": bool(args.per_sample_clusters),
            "write_fasta": bool(args.write_fasta),
            "write_summary": bool(args.write_summary),
            "threads": int(args.threads),
        },
        "stages": {
            "n_input_fingerprints": int(fingerprints_table.num_rows),
            "n_clusters": int(clusters.num_rows),
            "n_membership_rows": int(membership.num_rows),
            "n_samples": int(len(list(samples.ids))),
        },
        "outputs": cluster_outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (output_dir / "_SUCCESS").write_bytes(b"")

    if not args.progress:
        print(
            f"cluster done: {fingerprints_table.num_rows} input fingerprints → "
            f"{clusters.num_rows} clusters "
            f"({membership.num_rows} membership rows)",
            flush=True,
        )
    return 0


def _build_reference_parser(subs) -> None:
    """Wire the ``constellation reference ...`` sub-tree.

    Verbs:
        import        local-files in (FASTA + optional GFF3) →
                      ParquetDir bundles
        fetch         <source>:<id> → download via stdlib HTTP →
                      cache (default) and/or --output-dir
        list          print every (organism, release) installed in the cache
        where         print absolute cache path for a handle
        link          symlink a cached reference into an analysis directory
        default       inspect or pin per-organism default handles
        summary       print contig/feature counts on a saved bundle
        validate      load + run ``.validate()`` + cross-check
    """
    p_ref = subs.add_parser(
        "reference",
        help=(
            "Import / fetch / summarise / validate genome+annotation references; "
            "manage the per-user reference cache"
        ),
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
        default=None,
        help=(
            "output directory; receives ``genome/`` and (when --gff3 is given) "
            "``annotation/`` ParquetDir subdirectories. Required unless --to-cache."
        ),
    )
    p_imp.add_argument(
        "--to-cache",
        action="store_true",
        help=(
            "write into the per-user reference cache; requires --handle "
            "<organism_slug>@local-<YYYYMMDD>"
        ),
    )
    p_imp.add_argument(
        "--handle",
        default=None,
        help=(
            "handle for cache writes; required when --to-cache. Example: "
            "'pichia_pastoris@local-20260516'"
        ),
    )
    p_imp.set_defaults(func=_cmd_reference_import)

    p_fetch = ref_subs.add_parser(
        "fetch",
        help=(
            "Fetch a genome+GFF3 from Ensembl / Ensembl Genomes / RefSeq via "
            "stdlib HTTP; writes to the reference cache by default"
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
        default=None,
        help=(
            "optional additional destination; when set alongside the cache, "
            "a second copy is written here. Combine with --no-cache to "
            "preserve the legacy scratch-fetch behaviour."
        ),
    )
    p_fetch.add_argument(
        "--release",
        type=int,
        default=None,
        help=(
            "pin an Ensembl / Ensembl Genomes release number (e.g. 111). "
            "RefSeq/GenBank accessions already pin their assembly version."
        ),
    )
    p_fetch.add_argument(
        "--no-cache",
        action="store_true",
        help="skip the reference cache write; --output-dir then becomes mandatory",
    )
    p_fetch.add_argument(
        "--no-verify-checksums",
        action="store_true",
        help=(
            "skip source-published checksum verification (sha256 of the "
            "downloaded bytes is still computed and recorded locally)"
        ),
    )
    p_fetch.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the cache already has a complete copy",
    )
    p_fetch.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds (default 600)",
    )
    p_fetch.set_defaults(func=_cmd_reference_fetch)

    p_list = ref_subs.add_parser(
        "list",
        help="List every (organism, release) installed in the reference cache",
    )
    p_list.set_defaults(func=_cmd_reference_list)

    p_where = ref_subs.add_parser(
        "where",
        help="Print absolute cache path for a handle (exit 1 if not installed)",
    )
    p_where.add_argument(
        "handle",
        help="handle to resolve, e.g. 'homo_sapiens@ensembl-111' or 'homo_sapiens'",
    )
    p_where.set_defaults(func=_cmd_reference_where)

    p_link = ref_subs.add_parser(
        "link",
        help=(
            "Create <analysis>/genome and <analysis>/annotation symlinks "
            "pointing into the reference cache"
        ),
    )
    p_link.add_argument("handle", help="handle to link (e.g. 'homo_sapiens')")
    p_link.add_argument(
        "--into",
        required=True,
        help="analysis directory to receive the symlinks",
    )
    p_link.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing genome/annotation entries in --into",
    )
    p_link.set_defaults(func=_cmd_reference_link)

    p_def = ref_subs.add_parser(
        "default",
        help="Inspect or pin the per-organism default release used for shorthand handles",
    )
    p_def.add_argument(
        "organism",
        nargs="?",
        default=None,
        help=(
            "organism slug; omit to print all pinned defaults. With an "
            "argument: print the resolution for that organism."
        ),
    )
    p_def.add_argument(
        "release",
        nargs="?",
        default=None,
        help=(
            "release portion (e.g. 'ensembl-111') or full handle "
            "(e.g. 'homo_sapiens@ensembl-111') to pin"
        ),
    )
    p_def.add_argument(
        "--unset",
        action="store_true",
        help="remove the pinned default for <organism>",
    )
    p_def.add_argument(
        "--use-current",
        action="store_true",
        help="snapshot the current-symlink target as the new default",
    )
    p_def.set_defaults(func=_cmd_reference_default)

    p_sum = ref_subs.add_parser(
        "summary",
        help="Print contig + feature stats for a saved reference bundle",
    )
    p_sum.add_argument(
        "ref_dir",
        help=(
            "directory containing ``genome/`` (and optionally ``annotation/``) "
            "ParquetDir bundles, as written by ``import`` or ``fetch``. "
            "Accepts a cache handle too (e.g. 'homo_sapiens@ensembl-111')."
        ),
    )
    p_sum.set_defaults(func=_cmd_reference_summary)

    p_val = ref_subs.add_parser(
        "validate",
        help="Load + validate (PK/FK closure + cross-check) a reference bundle",
    )
    p_val.add_argument(
        "ref_dir",
        help=(
            "directory containing ``genome/`` + ``annotation/`` ParquetDir "
            "bundles. Accepts a cache handle too."
        ),
    )
    p_val.set_defaults(func=_cmd_reference_validate)


def _resolve_reference_argument(arg: str) -> "tuple[object, str]":
    """Resolve a CLI ``ref_dir`` / ``--reference`` argument to a Path.

    Accepts either an on-disk path (current behaviour) or a reference-cache
    handle (e.g. ``'homo_sapiens@ensembl-111'`` or ``'homo_sapiens'``).
    Returns ``(path, source)`` where ``source`` is ``"path"`` or
    ``"handle"`` for error-message attribution.
    """
    from pathlib import Path

    candidate = Path(arg).expanduser()
    if candidate.exists() and candidate.is_dir():
        return candidate, "path"
    # Treat as handle. Lazy-import to keep base CLI startup cheap.
    from constellation.sequencing.reference.handle import resolve as resolve_handle

    return resolve_handle(arg), "handle"


def _cmd_reference_import(args: argparse.Namespace) -> int:
    """Local FASTA (+ optional GFF3) → ParquetDir bundles."""
    import sys
    from pathlib import Path

    from constellation.sequencing.annotation.io import save_annotation
    from constellation.sequencing.readers.fastx import read_fasta_genome
    from constellation.sequencing.readers.gff import read_gff3
    from constellation.sequencing.reference.handle import (
        cache_root,
        parse_handle,
        update_current_pointer,
        write_meta_toml,
    )
    from constellation.sequencing.reference.io import save_genome_reference

    if not args.to_cache and not args.output_dir:
        print(
            "error: reference import requires either --output-dir or --to-cache",
            file=sys.stderr,
        )
        return 2
    if args.to_cache and not args.handle:
        print(
            "error: --to-cache requires --handle "
            "<organism_slug>@local-<YYYYMMDD>",
            file=sys.stderr,
        )
        return 2

    genome = read_fasta_genome(args.fasta)

    if args.gff3:
        contig_name_to_id = {
            row["name"]: row["contig_id"]
            for row in genome.contigs.to_pylist()
        }
        annotation = read_gff3(args.gff3, contig_name_to_id=contig_name_to_id)
        annotation.validate_against(genome)
    else:
        annotation = None

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_genome_reference(genome, out / "genome")
        if annotation is not None:
            save_annotation(annotation, out / "annotation")

    if args.to_cache:
        handle = parse_handle(args.handle)
        if not handle.is_qualified():
            print(
                f"error: --handle must be fully qualified "
                f"(<organism>@local-<YYYYMMDD>); got {args.handle!r}",
                file=sys.stderr,
            )
            return 2
        if handle.source != "local_import":
            # Permit "local-<date>" as syntactic sugar for the user; the
            # handle-parser still validates it.
            pass
        release_dir = cache_root() / handle.organism / handle.release_slug()
        if release_dir.exists():
            print(
                f"error: cache entry already exists: {release_dir}. "
                f"Pass a different --handle date or delete the existing entry.",
                file=sys.stderr,
            )
            return 1
        release_dir.mkdir(parents=True)
        save_genome_reference(genome, release_dir / "genome")
        if annotation is not None:
            save_annotation(annotation, release_dir / "annotation")
        write_meta_toml(
            release_dir,
            handle=handle,
            assembly_accession=None,
            assembly_name=None,
            annotation_release=None,
            constellation_version=__import__("constellation").__version__,
            urls={
                "fasta": {"url": f"file://{Path(args.fasta).resolve()}"},
                **(
                    {"gff3": {"url": f"file://{Path(args.gff3).resolve()}"}}
                    if args.gff3
                    else {}
                ),
            },
            sha256={},
            source_checksum_verified=False,
        )
        update_current_pointer(release_dir.parent, handle.release_slug())

    n_features = annotation.n_features if annotation else 0
    destinations: list[str] = []
    if args.output_dir:
        destinations.append(str(Path(args.output_dir)))
    if args.to_cache:
        destinations.append(f"cache:{args.handle}")
    print(
        f"imported {genome.n_contigs} contigs ({genome.total_length} bp) + "
        f"{n_features} features → {', '.join(destinations)}",
        flush=True,
    )
    return 0


def _cmd_reference_fetch(args: argparse.Namespace) -> int:
    """``<source>:<id>`` → ParquetDir bundles via stdlib HTTP."""
    import sys

    from constellation.sequencing.reference.fetch import fetch_reference

    if args.no_cache and not args.output_dir:
        print(
            "error: --no-cache requires --output-dir (no destination otherwise)",
            file=sys.stderr,
        )
        return 2

    result = fetch_reference(
        args.spec,
        output_dir=args.output_dir,
        release=args.release,
        timeout=args.timeout,
        use_cache=not args.no_cache,
        verify_source_checksums=not args.no_verify_checksums,
        force=args.force,
    )
    n_features = result.annotation.n_features if result.annotation else 0
    destinations: list[str] = []
    if result.cache_path is not None:
        prefix = "cache (cached)" if result.skipped_cache else "cache"
        destinations.append(f"{prefix}: {result.cache_path}")
    if result.output_path is not None:
        destinations.append(str(result.output_path))
    print(
        f"fetched {result.genome.n_contigs} contigs ({result.genome.total_length} bp) "
        f"+ {n_features} features → handle {result.handle} → "
        f"{'; '.join(destinations)}",
        flush=True,
    )
    if not result.skipped_cache:
        print(f"  genome: {result.sources['genome']}")
        print(f"  annotation: {result.sources['annotation']}")
    return 0


def _cmd_reference_list(_args: argparse.Namespace) -> int:
    """Print every (organism, release) entry in the reference cache."""
    from constellation.sequencing.reference.handle import (
        cache_root,
        format_size,
        list_installed,
        read_defaults,
    )

    entries = list_installed()
    if not entries:
        print(f"reference cache empty: {cache_root()}")
        print("  run: constellation reference fetch <source>:<id>")
        return 0
    defaults = read_defaults()
    rows: list[tuple[str, str, str, str, str, str]] = []
    for entry in entries:
        marker = "*" if entry.is_default(defaults) else " "
        fetched = (entry.fetched_at or "")[:10]
        size = format_size(entry.size_bytes or 0)
        rows.append(
            (
                entry.handle,
                marker,
                entry.organism,
                entry.release_slug,
                fetched,
                size,
            )
        )
    headers = ("handle", "*", "organism", "release", "fetched", "size")
    widths = [
        max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)
    ]

    def _fmt(row: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row, widths))

    print(_fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(_fmt(row))
    print(f"\ncache root: {cache_root()}")
    return 0


def _cmd_reference_where(args: argparse.Namespace) -> int:
    """Print absolute cache path for a handle (exit 1 if not installed)."""
    import sys

    from constellation.sequencing.reference.handle import (
        ReferenceNotInstalledError,
        resolve,
    )

    try:
        path = resolve(args.handle)
    except (ValueError, ReferenceNotInstalledError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


def _cmd_reference_link(args: argparse.Namespace) -> int:
    """Symlink a cached reference into an analysis directory."""
    import os
    import sys
    from pathlib import Path

    from constellation.sequencing.reference.handle import (
        ReferenceNotInstalledError,
        resolve,
    )

    try:
        release_dir = resolve(args.handle)
    except (ValueError, ReferenceNotInstalledError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    target_root = Path(args.into).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    sources = [release_dir / "genome", release_dir / "annotation"]
    for src in sources:
        if not src.is_dir():
            continue
        dst = target_root / src.name
        if dst.exists() or dst.is_symlink():
            if not args.force:
                print(
                    f"error: {dst} already exists; pass --force to overwrite",
                    file=sys.stderr,
                )
                return 1
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                import shutil as _shutil

                _shutil.rmtree(dst)
        try:
            os.symlink(src, dst)
        except OSError as exc:
            # WSL/Windows fallback: write a session.toml stub instead of
            # symlinks. Users edit the toml to drop in handles.
            session_toml = target_root / "session.toml"
            session_toml.write_text(
                f"schema_version = 1\n\n[reference]\nhandle = \"{args.handle}\"\n",
                encoding="utf-8",
            )
            print(
                f"warning: symlink creation failed ({exc}); wrote "
                f"{session_toml} as a fallback. The viz Session.from_root "
                f"resolves handles via session.toml the same way.",
                file=sys.stderr,
            )
            return 0
    print(f"linked {args.handle} → {target_root}")
    return 0


def _cmd_reference_default(args: argparse.Namespace) -> int:
    """Inspect or pin per-organism default handles."""
    import sys

    from constellation.sequencing.reference.handle import (
        ReferenceNotInstalledError,
        list_installed,
        parse_release_slug,
        read_defaults,
        resolve,
        set_default,
        unset_default,
        _read_current,
        cache_root,
    )

    if args.organism is None:
        defaults = read_defaults()
        if not defaults:
            print("no defaults pinned. Use: constellation reference default <organism> <release>")
            return 0
        widths = (
            max(len(o) for o in defaults),
            max(len(v) for v in defaults.values()),
        )
        for organism in sorted(defaults):
            print(f"{organism.ljust(widths[0])}  {defaults[organism]}")
        return 0

    organism = args.organism

    if args.unset:
        removed = unset_default(organism)
        if removed:
            print(f"unset default for {organism}")
            return 0
        print(f"no default to unset for {organism}", file=sys.stderr)
        return 1

    if args.use_current:
        organism_dir = cache_root() / organism
        current = _read_current(organism_dir)
        if current is None:
            print(f"error: no current pointer for {organism}", file=sys.stderr)
            return 1
        release_slug = current.name
        set_default(organism, release_slug)
        print(f"pinned default for {organism}: {release_slug}")
        return 0

    if args.release is None:
        # Inspection: print where bare `<organism>` resolves and why.
        defaults = read_defaults()
        pinned = defaults.get(organism)
        try:
            resolved = resolve(organism)
        except (ValueError, ReferenceNotInstalledError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        installed = [
            e.release_slug for e in list_installed() if e.organism == organism
        ]
        if pinned:
            reason = f"pinned in defaults.toml ({pinned})"
        elif (cache_root() / organism / "current").exists() or (
            cache_root() / organism / "current.txt"
        ).exists():
            reason = "current pointer"
        elif len(installed) == 1:
            reason = "single install"
        else:
            reason = "(ambiguous)"
        print(f"{organism} → {resolved.name}  [{reason}]")
        return 0

    # Pin a default.
    try:
        handle = parse_release_slug(args.release, organism=organism)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # Verify the cache slot actually exists before pinning.
    try:
        resolve(handle)
    except ReferenceNotInstalledError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    set_default(organism, handle.release_slug())
    print(f"pinned default for {organism}: {handle.release_slug()}")
    return 0


def _cmd_reference_summary(args: argparse.Namespace) -> int:
    """Print stats for a saved reference bundle."""
    from collections import Counter
    from pathlib import Path

    from constellation.sequencing.annotation.io import load_annotation
    from constellation.sequencing.reference.io import load_genome_reference

    root, _ = _resolve_reference_argument(args.ref_dir)
    root = Path(root)
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

    root, _ = _resolve_reference_argument(args.ref_dir)
    root = Path(root)
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
    raw = list(sys.argv[1:] if argv is None else argv)
    # Bare `constellation` → open the dashboard. PR 2 wires the
    # `dashboard` subcommand; until that ships the rewrite is a no-op
    # (parser will emit "subcommand required" as before).
    if not raw:
        raw = ["dashboard"]
    parser = _build_parser()
    args = parser.parse_args(raw)
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
