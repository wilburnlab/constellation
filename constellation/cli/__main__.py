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
from pathlib import Path
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


def _preload_matplotlib_for_reports() -> None:
    """Pre-import matplotlib so its libstdc++ wins the dynamic linker's
    resolution over later-loaded C extensions.

    pyarrow's bundled compiled extensions dlopen libstdc++ on import; on
    HPC clusters where ``LD_LIBRARY_PATH`` (or ``RUNPATH``) doesn't
    prepend the conda env's lib dir, the loader can pin an older system
    libstdc++ that matplotlib's ``c_internal_utils.so`` can't bind
    against (``CXXABI_1.3.15`` not found). Importing matplotlib FIRST
    gets its libstdc++ requirement satisfied before pyarrow pins
    something incompatible.

    Called at the very top of CLI handlers that auto-emit or regenerate
    diagnostic reports. Silent no-op when matplotlib isn't installed —
    the report generator will fall through to its own graceful-skip
    path. Kept in this module (not under ``constellation.sequencing.*``)
    so calling it doesn't side-effect import pyarrow first, defeating
    the purpose.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        pass


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

    # Taxonomy + catalog rows — surface bundled vs fetched + per-source counts.
    rows.append(_doctor_taxonomy_row())
    rows.append(_doctor_catalogs_row())

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


def _doctor_taxonomy_row() -> tuple[str, str, str, str]:
    """Single-line summary of the active taxonomy source + count."""
    name = "taxonomy"
    try:
        from constellation.core.taxonomy import (
            TaxonomyResolver,
            resolve_current,
            taxonomy_root,
        )
    except Exception as exc:  # pragma: no cover — import error path
        return (name, "warn", "—", f"import error: {exc}")

    root = taxonomy_root()
    try:
        r = TaxonomyResolver.auto()
    except Exception as exc:  # pragma: no cover — corrupt bundle
        return (name, "warn", "—", f"resolver init failed: {exc}")

    source = str(r.source_meta().get("source", "?"))
    n_taxa = r.n_taxa()
    if source.startswith("constellation-v1"):
        suffix = f"bundled starter ({root} empty — run `taxonomy update`)"
    else:
        current = resolve_current(root)
        suffix = f"{source} at {current or root}"
    return (name, "ok", f"{n_taxa} taxa", suffix)


def _doctor_catalogs_row() -> tuple[str, str, str, str]:
    """Single-line summary of installed assembly catalogs."""
    name = "catalogs"
    try:
        from constellation.catalog import catalogs_root, list_installed
    except Exception as exc:  # pragma: no cover
        return (name, "warn", "—", f"import error: {exc}")

    root = catalogs_root()
    if not root.exists():
        return (
            name,
            "ok",
            "0 sources",
            f"(not yet populated: {root})",
        )
    bundles = list_installed(root=root)
    if not bundles:
        return (name, "ok", "0 sources", f"(empty: {root})")
    total = sum(b.table.num_rows for b in bundles)
    by_src = sorted({b.source for b in bundles})
    return (
        name,
        "ok",
        f"{len(bundles)} bundles",
        f"{', '.join(by_src)} — {total} rows at {root}",
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

    # Taxonomy + catalog management.
    _build_taxonomy_parser(subs)
    _build_catalog_parser(subs)

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
            "TSV mapping reads → samples. Required columns: "
            "sample_name (str) + barcode_id (int, 0-indexed into the "
            "design's barcode panel). Optional: 'file' (enables "
            "multi-file mode — each row's file cell matches a basename "
            "or path under --reads), 'description'. Header row is "
            "required. sample_id is auto-assigned per unique "
            "sample_name; rows sharing a sample_name aggregate into "
            "one sample with multiple (file, barcode) edges, which is "
            "how multi-barcode multiplexing and multi-flowcell "
            "aggregation are expressed. The resolved Samples container "
            "is persisted under <output-dir>/samples/ so downstream "
            "stages (align, cluster) don't need --samples re-supplied."
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
    p_dem.add_argument(
        "--emit-fastq",
        action="store_true",
        help=(
            "after demux + resolve, write one gzipped FASTQ per sample to "
            "<output-dir>/fastq/<sample_name>.fq.gz (transcript-trimmed, "
            "Complete-status reads only; aggregates across all "
            "(acquisition, barcode) edges sharing sample_id). Safe to add on "
            "a re-run with --resume — the demux + resolve stages skip and "
            "only the FASTQ emission stage executes."
        ),
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
            "reference cache handle, e.g. `homo_sapiens@ensembl-111` (or a "
            "bare organism slug when defaults.toml resolves it). The handle "
            "is resolved against the per-user reference cache populated by "
            "`constellation reference fetch` / `import`. Presence selects "
            "Mode A; absence errors with the Mode B not-yet-implemented "
            "message. For an off-cache ad-hoc reference dir, use "
            "`--reference-dir <PATH>` instead (escape hatch)."
        ),
    )
    p_aln.add_argument(
        "--reference-dir",
        default=None,
        help=(
            "escape hatch — bare path to a `genome/` + `annotation/` "
            "reference root, used for one-off runs against a reference "
            "that hasn't been imported into the cache. Mutually exclusive "
            "with `--reference`. Manifests written this way omit the "
            "`reference_handle` field and cannot be opened in the genome "
            "browser dashboard until the reference is imported via "
            "`constellation reference import`."
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "write alignment_cs/ partitioned dataset preserving "
            "minimap2's cs:long string per alignment. Required by "
            "'transcriptome cluster --build-consensus' for base-resolution "
            "PWM accumulation AND by the genome browser's read pile-up "
            "track for per-base mismatch glyphs. On by default; "
            "--no-emit-cs-tags opts out and emits a stderr warning that "
            "read pile-up visualization is disabled for this run."
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
    # minimap2 splice-mode tuning. The base splice flags (-ax splice -uf
    # --cs=long --secondary=no) always apply; these add organism-aware
    # caps + escape hatches on top. See
    # constellation/sequencing/align/presets.py for the resolver.
    p_aln.add_argument(
        "--organism-profile",
        choices=("compact_eukaryote", "intermediate_eukaryote", "animal"),
        default=None,
        help=(
            "minimap2 splice-mode preset bundle. compact_eukaryote "
            "(Pichia / yeast / fungi: -G 5000 -C 5), intermediate_eukaryote "
            "(Drosophila / C. elegans / Arabidopsis: -G 50000 -C 5), "
            "animal (mouse / human: -G 200000 -C 5). Default None leaves "
            "minimap2's stock splice defaults in place — usually wrong "
            "for anything but vertebrates."
        ),
    )
    p_aln.add_argument(
        "--max-intron-length",
        type=int,
        default=None,
        help=(
            "minimap2 -G override. Caps the maximum intron length the "
            "splice DP will insert. Overrides the preset value when both "
            "are supplied. Pichia: ~5000; yeast: ~3000; mammals: ~200000."
        ),
    )
    p_aln.add_argument(
        "--non-canonical-cost",
        type=int,
        default=None,
        help=(
            "minimap2 -C override. Score penalty for non-canonical splice "
            "motifs. Default (preset) is 5; raise to disfavor cryptic "
            "junctions more aggressively. 0 = no penalty (minimap2 stock)."
        ),
    )
    p_aln.add_argument(
        "--junc-bed",
        type=str,
        default=None,
        help=(
            "Path to a BED file of annotated junctions to hint minimap2. "
            "Emits --junc-bed PATH; pairs with --junc-bonus to weight "
            "annotated junctions over de novo discoveries."
        ),
    )
    p_aln.add_argument(
        "--junc-bonus",
        type=int,
        default=None,
        help=(
            "minimap2 --junc-bonus N. Score bonus for junctions matching "
            "the --junc-bed file. Only meaningful with --junc-bed."
        ),
    )
    p_aln.add_argument(
        "--minimap2-extra",
        type=str,
        default=None,
        help=(
            "Raw minimap2 extra-args escape hatch (shlex-split). Use for "
            "flags not exposed as kwargs (e.g. --splice-flank=no, -k 14). "
            "Conflicts with the explicit kwarg-managed flags (-G, -C, "
            "--junc-bed, --junc-bonus) raise an error."
        ),
    )
    p_aln.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "auto-emit a diagnostic report (report.md + figures/*.svg) "
            "under <output-dir>/diagnostics/ after the resolve stage. "
            "Pass --no-report to skip. Report generation failures are "
            "logged but never break a successful align run."
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
            "reference cache handle (e.g. `homo_sapiens@ensembl-111`); "
            "required when --build-consensus is set, and otherwise inherits "
            "from the upstream align manifest. For an off-cache ad-hoc "
            "reference dir, use `--reference-dir <PATH>` (escape hatch)."
        ),
    )
    p_cluster.add_argument(
        "--reference-dir",
        default=None,
        help=(
            "escape hatch — bare path to a `genome/` + `annotation/` "
            "reference root. Mutually exclusive with `--reference`. "
            "Manifests written this way omit `reference_handle` and "
            "cannot be opened in the genome browser dashboard."
        ),
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
            "the upstream align run to have alignment_cs/ (default; "
            "absent only when align was invoked with --no-emit-cs-tags)."
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
    p_cluster.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "auto-emit a diagnostic report (report.md + figures/*.svg) "
            "under <output-dir>/diagnostics/ after clustering. Pass "
            "--no-report to skip. Report failures are logged but never "
            "break a successful cluster run."
        ),
    )
    p_cluster.add_argument("--resume", action="store_true")
    p_cluster.add_argument("--progress", action="store_true")
    p_cluster.set_defaults(func=_cmd_transcriptome_cluster)

    # ── Diagnose (standalone report regenerator) ────────────────────
    p_diag = tx_subs.add_parser(
        "diagnose",
        help=(
            "Regenerate diagnostic report(s) for an existing "
            "transcriptome align and/or cluster output dir. "
            "Read-only: never re-runs any pipeline stages, only "
            "rebuilds report.md + figures/*.svg from the parquet "
            "outputs already present."
        ),
    )
    p_diag.add_argument(
        "--align-dir",
        default=None,
        help=(
            "generate <align-dir>/diagnostics/report.md for this "
            "transcriptome-align output dir. Optional; at least one of "
            "--align-dir / --cluster-dir is required."
        ),
    )
    p_diag.add_argument(
        "--cluster-dir",
        default=None,
        help=(
            "generate <cluster-dir>/diagnostics/report.md for this "
            "transcriptome-cluster output dir. Optional; at least one "
            "of --align-dir / --cluster-dir is required."
        ),
    )
    p_diag.add_argument(
        "--reference",
        default=None,
        help=(
            "reference cache handle override. Defaults to the value in "
            "the input dir's manifest.json."
        ),
    )
    p_diag.add_argument(
        "--reference-dir",
        default=None,
        help=(
            "escape-hatch reference dir override. Mutually exclusive "
            "with --reference."
        ),
    )
    p_diag.add_argument(
        "--organism-profile",
        choices=("compact_eukaryote", "intermediate_eukaryote", "animal"),
        default=None,
        help=(
            "override the organism-profile recorded in the manifest. "
            "Affects only the flag thresholds (e.g. biological-"
            "plausibility intron-length cap)."
        ),
    )
    p_diag.set_defaults(func=_cmd_transcriptome_diagnose)


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
    """Parse the samples TSV.

    Required columns: ``sample_name`` (str) + ``barcode_id`` (int).
    Optional columns: ``file`` (enables multi-file mode), ``description``.

    The user-facing identity is ``sample_name``; ``sample_id`` is
    auto-assigned internally as ``1..N`` over the *unique* sample_names
    in sort order. Multiple rows sharing a ``sample_name`` resolve to the
    same auto-assigned ``sample_id`` and contribute additional rows to
    the SAMPLE_ACQUISITION_EDGE table — that's how multi-barcode
    multiplexing and multi-flowcell aggregation are expressed.

    Single-file mode (TSV header lacks ``file``): every row gets
    ``acquisition_id = default_acquisition_id``. All rows must reference
    barcode_ids from a single panel.

    Multi-file mode (TSV header includes a ``file`` column): each row's
    ``file`` cell is matched against the supplied ``input_files`` by
    basename or full path. Each unique referenced file gets a
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

    with open(tsv_path, encoding="utf-8") as fh:
        lines = [
            ln.rstrip("\r\n")
            for ln in fh
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    if not lines:
        raise ValueError(f"samples TSV {tsv_path!r} has no rows")

    # Header is required (no headerless legacy mode). Reject the legacy
    # `sample_id` column with a migration hint — it's now auto-assigned.
    first_cells = [c.strip() for c in lines[0].split("\t")]
    canonical_columns = {"file", "sample_name", "barcode_id", "description"}
    if not (set(first_cells) & canonical_columns):
        raise ValueError(
            f"samples TSV {tsv_path!r} has no header row — header is "
            f"required and must include at least 'sample_name' and "
            f"'barcode_id' (plus optional 'file' / 'description')"
        )
    if "sample_id" in first_cells:
        raise ValueError(
            f"samples TSV {tsv_path!r} has a 'sample_id' column — "
            f"sample_id is now auto-assigned from unique sample_name "
            f"values, so the column is no longer accepted. Drop the "
            f"column and rerun; rows sharing a sample_name aggregate "
            f"into one sample with multiple (acquisition, barcode) edges"
        )

    header = first_cells
    body = lines[1:]

    multi_file_mode = "file" in header
    if not multi_file_mode and len(input_files) > 1:
        raise ValueError(
            f"--reads passed {len(input_files)} files but the samples "
            f"TSV has no 'file' column — multi-file inputs require a "
            f"'file' column tying each (sample_name, barcode_id) row "
            f"to its source file"
        )

    col_idx = {name: i for i, name in enumerate(header)}
    required = {"sample_name", "barcode_id"}
    missing = required - set(col_idx)
    if missing:
        raise ValueError(
            f"samples TSV missing required columns: {sorted(missing)}"
        )

    # Pass 1: collect (sample_name, acquisition_id, barcode_id) edge
    # triples + description-per-name, deferring sample_id assignment
    # until all unique sample_names are known.
    raw_edges: list[tuple[str, int, int]] = []
    description_by_name: dict[str, str | None] = {}

    for raw in body:
        cells = [c.strip() for c in raw.split("\t")]
        if len(cells) < len(header):
            continue  # malformed row — skip
        sname = cells[col_idx["sample_name"]]
        if not sname:
            continue
        try:
            bid = int(cells[col_idx["barcode_id"]])
        except ValueError:
            continue
        if multi_file_mode:
            file_cell = cells[col_idx["file"]]
            file_path = by_basename.get(file_cell) or by_path.get(file_cell)
            if file_path is None:
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

        if "description" in col_idx:
            desc = cells[col_idx["description"]].strip() or None
            if desc is not None:
                # First non-empty description wins on collision.
                description_by_name.setdefault(sname, desc)
        description_by_name.setdefault(sname, None)
        raw_edges.append((sname, acq_id, bid))

    if not raw_edges:
        raise ValueError(
            f"samples TSV {tsv_path!r} parsed zero usable edge rows; "
            f"check that sample_name / barcode_id columns are populated"
        )

    # Auto-assign sample_id: 1..N over unique sample_names in sort order.
    unique_names = sorted(set(name for name, _, _ in raw_edges))
    name_to_sid: dict[str, int] = {n: i + 1 for i, n in enumerate(unique_names)}

    sample_rows: list[dict[str, object]] = [
        {
            "sample_id": name_to_sid[name],
            "sample_name": name,
            "description": description_by_name.get(name),
        }
        for name in unique_names
    ]

    # Dedup truly-identical (sample_name, acquisition_id, barcode_id)
    # tuples; the user's TSV duplicating a row shouldn't blow up FK
    # closure with two indistinguishable edges.
    seen_edges: set[tuple[str, int, int]] = set()
    edge_rows: list[dict[str, object]] = []
    duplicate_edges: list[tuple[str, int, int]] = []
    for name, acq_id, bid in raw_edges:
        key = (name, acq_id, bid)
        if key in seen_edges:
            duplicate_edges.append(key)
            continue
        seen_edges.add(key)
        edge_rows.append(
            {
                "sample_id": name_to_sid[name],
                "acquisition_id": acq_id,
                "barcode_id": bid,
            }
        )
    if duplicate_edges:
        sys.stderr.write(
            f"warning: dropped {len(duplicate_edges)} duplicate "
            f"(sample_name, acquisition_id, barcode_id) row(s) from "
            f"{tsv_path}: {duplicate_edges[:3]}"
            f"{'...' if len(duplicate_edges) > 3 else ''}\n"
        )

    return sample_rows, edge_rows, file_to_acq


def _cmd_transcriptome_demultiplex(args: argparse.Namespace) -> int:
    """Run the full S1 demux + ORF + quant pipeline on one or more SAM/BAMs."""
    # Defer heavy imports until the subcommand actually fires so
    # `constellation --help` stays fast.
    from pathlib import Path

    from constellation.sequencing.progress import (
        NullProgress,
        StreamProgress,
    )
    from constellation.sequencing.samples import Samples, save_samples
    from constellation.sequencing.transcriptome.manifest import (
        write_demux_manifest,
    )
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
        emit_fastq=args.emit_fastq,
    )

    n_reads = artefacts["n_reads"]
    quant_table = artefacts["quant_table"]
    fasta_records = artefacts["fasta_records"]
    fastq_dir = artefacts.get("fastq_dir")

    # Persist the resolved Samples container so align + cluster can
    # consume it without re-parsing the user's TSV.
    samples_dir = output_dir / "samples"
    save_samples(samples, samples_dir)

    outputs: dict[str, str] = {
        "feature_quant": str(output_dir / "feature_quant.parquet"),
        "proteins_fasta": str(output_dir / "proteins.fasta"),
        "protein_counts_tsv": str(output_dir / "protein_counts.tsv"),
        "read_demux": str(output_dir / "read_demux"),
        "samples": str(samples_dir),
    }
    if fastq_dir is not None:
        outputs["fastq_dir"] = str(fastq_dir)

    parameters: dict[str, object] = {
        "min_aa_length": args.min_aa_length,
        "min_protein_count": args.min_protein_count,
        "n_workers": args.threads,
        "batch_size": args.batch_size,
        "resumed": args.resume,
        "emit_fastq": args.emit_fastq,
    }
    stages: dict[str, object] = {
        "n_reads": n_reads,
        "n_proteins_kept": len(fasta_records),
    }
    if fastq_dir is not None:
        fastq_files = {
            p.name.removesuffix(".fq.gz"): str(p)
            for p in sorted(Path(fastq_dir).glob("*.fq.gz"))
        }
        stages["fastq_files"] = fastq_files

    write_demux_manifest(
        output_dir / "manifest.json",
        input_files=[str(f) for f in input_files],
        acquisition_map={str(f): aid for f, aid in pipeline_inputs},
        library_design=args.library_design,
        parameters=parameters,
        stages=stages,
        outputs=outputs,
    )

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
    # Must precede any pyarrow / torch / sequencing import — see
    # _preload_matplotlib_for_reports docstring for the libstdc++
    # ordering rationale.
    _preload_matplotlib_for_reports()

    import sys
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.dataset as pa_dataset
    import pyarrow.parquet as pq

    import shlex

    from constellation.sequencing.align.map import _GENOME_MODE_ARGS, map_to_genome
    from constellation.sequencing.align.presets import resolve_minimap2_args
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
        build_read_samples,
        cluster_junctions,
        count_reads_per_gene,
        fused_decode_filter_overlap_worker,
        gene_set_from_annotation,
        render_gene_matrix_tsv,
        serialise_gene_set,
    )
    from constellation.sequencing.readers.sam_bam import _bam_alignment_chunks
    from constellation.sequencing.reference.io import load_genome_reference
    from constellation.sequencing.samples import load_samples
    from constellation.sequencing.transcriptome.manifest import (
        read_manifest_dir,
    )

    # ── Mode dispatch ────────────────────────────────────────────────
    if args.reference is None and args.reference_dir is None:
        print(
            "Mode B (de novo clustering) not yet implemented in this "
            "release; pass --reference <handle> for reference-guided "
            "gene counting (or --reference-dir <PATH> as an escape hatch).",
            file=sys.stderr,
        )
        return 2
    if args.reference is not None and args.reference_dir is not None:
        print(
            "error: --reference and --reference-dir are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    demux_dir = Path(args.demux_dir).expanduser().resolve()
    if not demux_dir.is_dir():
        raise FileNotFoundError(f"--demux-dir not found: {demux_dir}")
    if not (demux_dir / "read_demux").is_dir():
        raise FileNotFoundError(
            f"--demux-dir missing read_demux/: {demux_dir}"
        )
    demux_manifest = read_manifest_dir(demux_dir)
    if demux_manifest.kind != "demux":
        raise ValueError(
            f"--demux-dir {demux_dir} has manifest kind="
            f"{demux_manifest.kind!r}; expected 'demux'"
        )
    samples_dir = demux_dir / "samples"
    if not samples_dir.is_dir():
        sys.exit(
            f"--demux-dir {demux_dir} does not contain a 'samples/' "
            f"directory — it was produced by an older version of "
            f"constellation; re-run `constellation transcriptome "
            f"demultiplex` to upgrade its schema"
        )
    samples = load_samples(samples_dir)

    reference, reference_handle, assembly_accession = _resolve_reference_args(
        handle_arg=args.reference, dir_arg=args.reference_dir
    )
    if reference is None:
        return 1  # error already printed
    genome_dir = reference / "genome"
    annotation_dir = reference / "annotation"
    if not genome_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"reference must contain genome/ + annotation/ subdirs: {reference}"
        )

    # Resolve --output-dir to an absolute path before any artifact paths
    # are constructed: manifest.json writes record `outputs.*` as `str(
    # output_dir / 'coverage.parquet')` etc., and downstream readers
    # (the viz session loader, future cross-validation passes) require
    # those paths to be unambiguous regardless of which directory the
    # consumer is running from.
    output_dir = Path(args.output_dir).expanduser().resolve()
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
    # the align manifest's provenance record. We DO NOT open them —
    # alignment streams trimmed transcript windows from the demux
    # parquet partitions instead, so the raw BAMs may be moved /
    # archived after demux without breaking the align stage.
    bam_inputs = [Path(p) for p in demux_manifest.input_files]

    # ── Resolve minimap2 arg list (preset + overrides + escape hatch) ──
    # Always computed even on --resume so the manifest reflects the
    # parameters the user requested *this* invocation. The resolved tuple
    # is also written into the manifest for downstream reproducibility.
    try:
        minimap2_resolved_args = resolve_minimap2_args(
            base_args=_GENOME_MODE_ARGS,
            profile=args.organism_profile,
            max_intron_length=args.max_intron_length,
            non_canonical_cost=args.non_canonical_cost,
            junc_bed=Path(args.junc_bed) if args.junc_bed else None,
            junc_bonus=args.junc_bonus,
            extra_args=(
                tuple(shlex.split(args.minimap2_extra))
                if args.minimap2_extra else ()
            ),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.organism_profile is None and args.max_intron_length is None:
        print(
            "WARNING: no --organism-profile or --max-intron-length set; "
            "minimap2 will use its stock -G 200000 intron-length cap. "
            "For compact genomes (e.g. Pichia, yeast, fungi) this lets "
            "the splice DP collapse adjacent genes into one read with a "
            "fictional 50-200 kb 'intron'. Pass `--organism-profile "
            "compact_eukaryote` for sane defaults.",
            file=sys.stderr,
            flush=True,
        )

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
            minimap2_args=minimap2_resolved_args,
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
    if not emit_cs:
        print(
            "WARNING: --no-emit-cs-tags disables read pile-up "
            "visualization in the genome browser (the read_pileup "
            "track requires per-alignment cs:long strings to render "
            "exonic geometry + per-base mismatch glyphs). Re-run without "
            "--no-emit-cs-tags to restore viz support.",
            file=sys.stderr,
            flush=True,
        )
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
    # Project only the columns the resolve stage actually consumes;
    # skips ~10 GB of unused string + small columns (cigar_string,
    # nm_tag, as_tag, read_group, acquisition_id, mapq, flag) at scan
    # time. cigar_string alone is 5-20 GB at PromethION scale.
    alignments_table = pa_dataset.dataset(
        stage_outputs["alignments"].directory
    ).to_table(columns=[
        "alignment_id", "read_id", "ref_name", "strand",
        "ref_start", "ref_end",
        "is_secondary", "is_supplementary",
    ])
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

    # ── Per-read sample resolution (always emitted) ──────────────────
    # read_samples.parquet is the viz-facing per-read sample assignment
    # the genome browser's read pile-up track joins against to color
    # alignments by sample. Also used downstream as `read_to_sample` for
    # the coverage + derived-annotation passes' per-sample
    # stratification, so the reduction runs once and feeds both
    # consumers. Chimera policy lives inside `build_read_samples` — the
    # same agreement rule the count stage uses.
    read_samples_table = build_read_samples(read_demux, samples)
    read_samples_path = output_dir / "read_samples.parquet"
    pq.write_table(read_samples_table, read_samples_path)
    emit_outputs["read_samples"] = str(read_samples_path)
    read_to_sample_table = read_samples_table.select(["read_id", "sample_id"])

    # ── Coverage pile-up + derived annotation ─────────────────────────
    # The two functions are now siblings with path-based handoff:
    # build_pileup streams COVERAGE_TABLE batches to coverage.parquet;
    # build_derived_annotation re-reads that file (small post-RLE), and
    # streams BLOCK_EXON_ASSIGNMENT_TABLE to block_exon_assignments.parquet
    # which compute_exon_psi consumes as a dataset. No giant in-memory
    # handoffs between these passes anymore.
    need_pileup = emit_coverage or emit_derived_annotation
    coverage_path: Path | None = None
    if need_pileup:
        coverage_path = output_dir / "coverage.parquet"
        build_pileup(
            blocks_table, alignments_table, genome.contigs,
            output_path=coverage_path,
            read_to_sample=read_to_sample_table,
        )

    if emit_derived_annotation and coverage_path is not None:
        # Local import: derived_annotation depends on Part B schemas
        # registered when the module is imported.
        from constellation.sequencing.quant.derived_annotation import (
            build_derived_annotation,
        )
        from constellation.sequencing.annotation.io import save_annotation

        block_assignments_path = output_dir / "block_exon_assignments.parquet"
        derived_annotation, _ba_path, exon_psi = build_derived_annotation(
            coverage=coverage_path,
            introns=introns,
            alignment_blocks=blocks_table,
            alignments=alignments_table,
            contigs=genome.contigs,
            read_to_sample=read_to_sample_table,
            block_assignments_output_path=block_assignments_path,
            min_exon_depth=int(args.min_exon_depth),
            min_intron_read_count=int(args.min_intron_read_count),
        )
        derived_dir = output_dir / "derived_annotation"
        save_annotation(derived_annotation, derived_dir, format="parquet_dir")
        exon_psi_path = output_dir / "exon_psi.parquet"
        pq.write_table(exon_psi, exon_psi_path)
        emit_outputs["derived_annotation"] = str(derived_dir)
        emit_outputs["block_exon_assignments"] = str(block_assignments_path)
        emit_outputs["exon_psi"] = str(exon_psi_path)

    if emit_coverage and coverage_path is not None:
        emit_outputs["coverage"] = str(coverage_path)
    elif coverage_path is not None and not emit_coverage:
        # Coverage was needed for derived_annotation but the user opted
        # out of the final output — unlink the transient file.
        try:
            coverage_path.unlink()
        except FileNotFoundError:
            pass

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

    from constellation.sequencing.transcriptome.manifest import write_align_manifest

    sample_names = (
        sorted(set(samples.samples.column("sample_name").to_pylist()))
        if samples.samples.num_rows > 0 else None
    )
    write_align_manifest(
        output_dir / "manifest.json",
        reference_handle=reference_handle,
        reference_path=str(reference),
        assembly_accession=assembly_accession,
        demux_dir=str(demux_dir),
        input_files=[str(p) for p in bam_inputs],
        parameters={
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
            "organism_profile": args.organism_profile,
            "max_intron_length": args.max_intron_length,
            "non_canonical_cost": args.non_canonical_cost,
            "junc_bed": args.junc_bed,
            "junc_bonus": args.junc_bonus,
            "minimap2_extra": args.minimap2_extra,
        },
        stages=per_stage,
        outputs=manifest_outputs,
        samples=sample_names,
        minimap2_resolved_args=list(minimap2_resolved_args),
    )
    (output_dir / "_SUCCESS").write_bytes(b"")

    # ── Diagnostic report ──────────────────────────────────────────────
    # Always attempt unless the user opts out. Failures are logged but
    # never break a successful pipeline run.
    if getattr(args, "report", True):
        try:
            from constellation.sequencing.transcriptome.align.diagnostics import (
                build_align_diagnostics_report,
            )
            report_path = build_align_diagnostics_report(
                output_dir,
                reference=genome,
                annotation=annotation,
                organism_profile=args.organism_profile,
            )
            if not args.progress:
                print(
                    f"diagnostic report: {report_path}",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"WARNING: diagnostic report generation failed: "
                f"{type(exc).__name__}: {exc}. The align outputs are "
                f"complete; re-run with `constellation transcriptome "
                f"diagnose --align-dir {output_dir}` to retry.",
                file=sys.stderr,
                flush=True,
            )

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


def _trim_sequences_to_large_string(
    seq_col,  # pa.Array | pa.ChunkedArray, string or large_string
    starts,  # pa.Array | pa.ChunkedArray, integer
    ends,  # pa.Array | pa.ChunkedArray, integer
    *,
    chunk_size: int = 5_000_000,
):
    """Per-row slice of a string/large_string column → LargeStringArray.

    pyarrow's ``pc.utf8_slice_codeunits`` accepts scalar start/stop only,
    so a per-row trim (transcript-window extraction at PromethION scale)
    has to drop to numpy. The buffer-construction approach copies bytes
    directly between the input values buffer and a new output values
    buffer (no Python-string round-trip), bounding intermediate memory
    to ``chunk_size`` rows' worth of output. Replaces the prior
    ``4 × to_pylist + Python loop`` that materialised tens of GB of
    Python strings on a single thread.

    Null sequences in the input become **empty strings** in the output
    (matches the prior CLI semantics — not nulls, so the output array
    has ``null_count=0``).
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    if isinstance(seq_col, pa.ChunkedArray):
        seq_col = seq_col.combine_chunks()
    if isinstance(starts, pa.ChunkedArray):
        starts = starts.combine_chunks()
    if isinstance(ends, pa.ChunkedArray):
        ends = ends.combine_chunks()
    if pa.types.is_string(seq_col.type):
        seq_col = pc.cast(seq_col, pa.large_string())

    n = len(seq_col)
    if n == 0:
        return pa.chunked_array([], type=pa.large_string())

    seq_offsets = np.frombuffer(seq_col.buffers()[1], dtype=np.int64)
    seq_buf = np.frombuffer(seq_col.buffers()[2], dtype=np.uint8)
    ts_arr = np.asarray(starts).astype(np.int64, copy=False)
    te_arr = np.asarray(ends).astype(np.int64, copy=False)

    # Null sequences → length 0 (empty string in output).
    seq_valid = pc.is_valid(seq_col).to_numpy(zero_copy_only=False)
    ts_eff = np.where(seq_valid, ts_arr, 0).astype(np.int64, copy=False)
    te_eff = np.where(seq_valid, te_arr, 0).astype(np.int64, copy=False)

    chunks: list = []
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_n = chunk_end - chunk_start
        chunk_row_offsets = seq_offsets[chunk_start:chunk_end]
        abs_starts_c = chunk_row_offsets + ts_eff[chunk_start:chunk_end]
        abs_ends_c = chunk_row_offsets + te_eff[chunk_start:chunk_end]
        new_lens = (abs_ends_c - abs_starts_c).astype(np.int64, copy=False)
        new_offsets = np.empty(chunk_n + 1, dtype=np.int64)
        new_offsets[0] = 0
        np.cumsum(new_lens, out=new_offsets[1:])
        total_out = int(new_offsets[-1])
        new_buf = np.empty(total_out, dtype=np.uint8)
        for i in range(chunk_n):
            a = int(abs_starts_c[i])
            b = int(abs_ends_c[i])
            if b > a:
                new_buf[new_offsets[i] : new_offsets[i + 1]] = seq_buf[a:b]
        chunks.append(
            pa.LargeStringArray.from_buffers(
                length=chunk_n,
                value_offsets=pa.py_buffer(new_offsets.tobytes()),
                data=pa.py_buffer(new_buf.tobytes()),
                null_bitmap=None,
                null_count=0,
            )
        )
    return pa.chunked_array(chunks, type=pa.large_string())


def _cmd_transcriptome_cluster(args: argparse.Namespace) -> int:
    """Phase 2 — group reads into transcript / isoform clusters.

    Inputs: a `transcriptome align` output dir (alignment_blocks/ +
    introns.parquet) + the S1 demux dir (whose ``samples/`` subdir
    carries the resolved Samples container — no ``--samples`` TSV
    needed). Outputs: clusters.parquet, cluster_membership.parquet,
    optional cluster.fa + cluster_summary.tsv.

    Fingerprints are derived fresh at every invocation against the
    INTRON_TABLE in introns.parquet, so the ``--intron-tolerance-bp``
    knob is a clustering parameter rather than an alignment parameter
    — sweeping it never requires re-running align (we re-cluster the
    raw per-position rows on the fly).
    """
    # Must precede any pyarrow / torch / sequencing import — see
    # _preload_matplotlib_for_reports docstring for the libstdc++
    # ordering rationale.
    _preload_matplotlib_for_reports()

    from collections import Counter
    from pathlib import Path

    import numpy as np
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
    from constellation.sequencing.samples import load_samples
    from constellation.sequencing.transcriptome.cluster.cluster_genome import (
        cluster_by_fingerprint,
    )
    from constellation.sequencing.transcriptome.cluster.fingerprints import (
        compute_read_fingerprints,
    )
    from constellation.sequencing.transcriptome.manifest import (
        read_manifest_dir,
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
    if args.reference is not None and args.reference_dir is not None:
        print(
            "error: --reference and --reference-dir are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    if args.build_consensus:
        cs_success = align_dir / "alignment_cs" / "_SUCCESS"
        if not cs_success.exists():
            print(
                f"--build-consensus requires alignment_cs/_SUCCESS in "
                f"{align_dir}; the upstream align run was invoked with "
                "--no-emit-cs-tags. Rerun `transcriptome align` without "
                "that flag (cs:long is on by default).",
                file=sys.stderr,
            )
            return 2
        if args.reference is None and args.reference_dir is None:
            print(
                "--build-consensus requires --reference <handle> (or "
                "--reference-dir <PATH>) for the genome window.",
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

    # Resolve --output-dir to absolute so manifest.json's `outputs.*`
    # paths are unambiguous to downstream readers (the viz session
    # loader, future cross-validation passes). See the matching comment
    # in `_cmd_transcriptome_align`.
    output_dir = Path(args.output_dir).expanduser().resolve()
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

    # Stage-boundary timing logger. Under --progress we print
    # `[cluster +Xs] msg` to stderr at every well-known checkpoint
    # below + every K clusters inside cluster_by_fingerprint. Flushes
    # immediately so a cluster job's stdout buffer doesn't hide
    # progress when the wall time runs into hours. No-op when
    # --progress is off.
    import time as _time
    _t0 = _time.perf_counter()
    if args.progress:
        def _log(msg: str) -> None:
            elapsed = _time.perf_counter() - _t0
            print(f"[cluster +{elapsed:8.1f}s] {msg}",
                  file=sys.stderr, flush=True)
    else:
        def _log(msg: str) -> None:
            return None
    _log("start")

    # ── Resolve align-dir manifest for downstream provenance ─────────
    align_manifest = read_manifest_dir(align_dir)
    if align_manifest.kind != "align":
        raise ValueError(
            f"--align-dir {align_dir} has manifest kind="
            f"{align_manifest.kind!r}; expected 'align'"
        )

    # Reference identity: inherit from the upstream align manifest unless
    # the user explicitly overrides via --reference / --reference-dir.
    # The handle (when present) and assembly_accession ride forward into
    # this stage's manifest so downstream consumers can match cluster
    # outputs to the same reference axis as the align outputs.
    inherited_handle: str | None = align_manifest.reference_handle
    inherited_ref_path: str | None = align_manifest.reference_path
    inherited_assembly: str | None = align_manifest.assembly_accession

    reference: Path | None = None
    reference_handle: str | None = inherited_handle
    reference_path_str: str | None = inherited_ref_path
    assembly_accession: str | None = inherited_assembly

    if args.reference is not None or args.reference_dir is not None:
        ref_path, ref_handle, ref_assembly = _resolve_reference_args(
            handle_arg=args.reference, dir_arg=args.reference_dir
        )
        if ref_path is None:
            return 1
        reference = Path(ref_path)
        reference_handle = ref_handle
        reference_path_str = str(ref_path)
        assembly_accession = ref_assembly
        if inherited_handle and ref_handle and inherited_handle != ref_handle:
            print(
                f"warning: --reference {ref_handle!r} overrides align "
                f"manifest's {inherited_handle!r}",
                file=sys.stderr,
            )
    elif inherited_ref_path is not None:
        candidate = Path(inherited_ref_path)
        if candidate.is_dir():
            reference = candidate

    # ── Genome (only when --build-consensus) ─────────────────────────
    genome = None
    if args.build_consensus:
        if reference is None:
            print(
                "--build-consensus requires a reference but none could be "
                "resolved (align manifest carries no reference_path and no "
                "--reference / --reference-dir was passed).",
                file=sys.stderr,
            )
            return 1
        genome_dir = reference / "genome"
        if not genome_dir.is_dir():
            raise FileNotFoundError(
                f"reference must contain genome/: {reference}"
            )
        genome = load_genome_reference(genome_dir)

    # ── Samples (loaded from the demux output's persisted bundle) ────
    samples_dir = demux_dir / "samples"
    if not samples_dir.is_dir():
        sys.exit(
            f"--demux-dir {demux_dir} does not contain a 'samples/' "
            f"directory — it was produced by an older version of "
            f"constellation; re-run `constellation transcriptome "
            f"demultiplex` to upgrade its schema"
        )
    samples = load_samples(samples_dir)
    _log("resolved manifest + samples")

    # ── Materialise alignments + alignment_blocks ────────────────────
    # Project columns at scan time — cluster_by_fingerprint +
    # compute_read_fingerprints don't need cigar_string / read_group /
    # nm_tag / as_tag / mapq / flag / acquisition_id / ref_end /
    # query_start / query_end. At PromethION scale dropping the wide
    # string columns (cigar_string in particular) cuts the alignments
    # load from ~30 GB to ~8 GB.
    _alignments_cols = [
        "alignment_id", "read_id", "ref_name", "ref_start", "strand",
        "is_secondary", "is_supplementary",
    ]
    _blocks_cols = [
        "alignment_id", "block_index", "ref_start", "ref_end",
        "n_match", "n_mismatch", "n_insert", "n_delete",
    ]
    alignments_table = pa_dataset.dataset(
        align_dir / "alignments"
    ).to_table(columns=_alignments_cols)
    _log(f"loaded alignments_table ({alignments_table.num_rows:,} rows)")
    blocks_table = pa_dataset.dataset(
        align_dir / "alignment_blocks"
    ).to_table(columns=_blocks_cols)
    _log(f"loaded blocks_table ({blocks_table.num_rows:,} rows)")
    # alignment_cs is opened as a dataset (not eagerly materialised) when
    # --build-consensus is on — the consensus path filters per-cluster
    # rather than carrying the full cs:long column (~50 GB at full scale)
    # through to every _emit_cluster call. See Fix 5 in the cluster
    # refactor plan.
    alignment_cs_dataset: pa_dataset.Dataset | None = None
    if args.build_consensus:
        alignment_cs_dataset = pa_dataset.dataset(align_dir / "alignment_cs")
        _log("opened alignment_cs dataset (lazy)")

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

    _log("built contigs_table")

    # ── Load introns.parquet; optionally re-cluster ─────────────────
    introns_table = pq.read_table(align_dir / "introns.parquet")
    _log(f"loaded introns_table ({introns_table.num_rows:,} rows)")

    # When the user explicitly overrides --intron-tolerance-bp at
    # cluster time, re-run cluster_junctions against the raw per-position
    # rows (each row in introns.parquet is still a distinct observed
    # (donor, acceptor) pair). The motif-priority list rides as a
    # comma-separated align-manifest value.
    align_params = align_manifest.parameters
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
        _log(
            f"re-clustered introns at tolerance_bp={effective_tolerance} "
            f"({introns_table.num_rows:,} rows)"
        )
    if effective_tolerance is None:
        effective_tolerance = (
            int(align_tolerance) if align_tolerance is not None else 5
        )

    # ── Derive fingerprints fresh against the (possibly re-clustered)
    # INTRON_TABLE ───────────────────────────────────────────────────
    _log("compute_read_fingerprints: start")
    fingerprints_table = compute_read_fingerprints(
        blocks_table,
        alignments_table,
        contigs_table,
        introns_table,
    )
    _log(
        f"compute_read_fingerprints: done "
        f"({fingerprints_table.num_rows:,} fingerprints)"
    )

    # ── Pull trimmed transcript-window sequences for clustered reads ─
    # The fingerprints_table.column("read_id") ChunkedArray rides
    # directly into the dataset filter — no to_pylist + set round-trip
    # (which materialised ~7 GB of Python strings at PromethION scale).
    fp_read_id_chunked = fingerprints_table.column("read_id")

    read_demux_columns = [
        "read_id", "transcript_start", "transcript_end", "sample_id",
        "transcript_segment_index",
    ]
    read_demux_filter = pc.is_in(
        pa_dataset.field("read_id"), value_set=fp_read_id_chunked
    )
    _log("read_demux: scan start")
    read_demux = pa_dataset.dataset(demux_dir / "read_demux").to_table(
        columns=read_demux_columns,
        filter=read_demux_filter,
    )
    _log(f"read_demux: scan done ({read_demux.num_rows:,} rows before dedup)")
    # One transcript window per read for clustering — pick the lowest
    # transcript_segment_index per read_id (chimera resolution downstream
    # is out of scope for genome-guided clustering v1). The dedup replaces
    # the prior to_pylist + Python-set loop with a sort + numpy boundary
    # scan on dict-encoded read_id indices: comparing int32 codes instead
    # of strings cuts the dedup from a 30 GB Python-string Set to a
    # ~1.6 GB int32 numpy array at PromethION scale.
    if read_demux.num_rows > 0:
        read_demux = read_demux.sort_by(
            [
                ("read_id", "ascending"),
                ("transcript_segment_index", "ascending"),
            ]
        )
        rid_col = read_demux.column("read_id")
        rid_da = pc.dictionary_encode(rid_col)
        if isinstance(rid_da, pa.ChunkedArray):
            rid_da = rid_da.combine_chunks()
        rid_idx = rid_da.indices.to_numpy(zero_copy_only=False)
        n_rd = len(rid_idx)
        is_first = np.empty(n_rd, dtype=bool)
        is_first[0] = True
        if n_rd > 1:
            is_first[1:] = rid_idx[1:] != rid_idx[:-1]
        if not bool(np.all(is_first)):
            read_demux = read_demux.filter(pa.array(is_first))
    _log(f"read_demux: dedup done ({read_demux.num_rows:,} rows after dedup)")

    reads_columns = ["read_id", "sequence"]
    reads_dataset = pa_dataset.dataset(demux_dir / "reads")
    reads_schema_names = reads_dataset.schema.names
    if "dorado_quality" in reads_schema_names:
        reads_columns.append("dorado_quality")
    reads_filter = pc.is_in(
        pa_dataset.field("read_id"), value_set=fp_read_id_chunked
    )
    _log("raw_reads: scan start")
    raw_reads = reads_dataset.to_table(
        columns=reads_columns, filter=reads_filter
    )
    _log(f"raw_reads: scan done ({raw_reads.num_rows:,} rows)")
    # Cast sequence to large_string defensively — at PromethION scale
    # (~200M reads × ~500 bytes/window ≈ 100 GB) the column blows past
    # Arrow's 2 GiB int32 string-offset limit when carried through
    # joins / takes / combine_chunks downstream.
    if (
        raw_reads.num_rows > 0
        and pa.types.is_string(raw_reads.schema.field("sequence").type)
    ):
        raw_reads = raw_reads.set_column(
            raw_reads.schema.get_field_index("sequence"),
            "sequence",
            pc.cast(raw_reads.column("sequence"), pa.large_string()),
        )
        _log("raw_reads: sequence cast string → large_string")

    # Join + slice trimmed windows into a flat (read_id, sequence,
    # dorado_quality) table for cluster_by_fingerprint. The trim itself
    # uses _trim_sequences_to_large_string (numpy buffer construction)
    # to avoid the prior 4 × to_pylist + Python loop that materialised
    # tens of GB of Python strings at PromethION scale.
    trimmed_table: pa.Table
    if raw_reads.num_rows == 0 or read_demux.num_rows == 0:
        trimmed_table = pa.table(
            {
                "read_id": pa.array([], type=pa.string()),
                "sequence": pa.array([], type=pa.large_string()),
                "dorado_quality": pa.array([], type=pa.float32()),
            }
        )
    else:
        _log("raw_reads ⨝ read_demux: join start")
        joined = raw_reads.join(
            read_demux.select(
                ["read_id", "transcript_start", "transcript_end"]
            ),
            keys="read_id",
            join_type="inner",
        )
        _log(f"raw_reads ⨝ read_demux: join done ({joined.num_rows:,} rows)")
        _log("trim: vectorized buffer construction start")
        trimmed_seq = _trim_sequences_to_large_string(
            joined.column("sequence"),
            joined.column("transcript_start"),
            joined.column("transcript_end"),
        )
        _log("trim: done")
        cols: dict[str, pa.Array | pa.ChunkedArray] = {
            "read_id": joined.column("read_id"),
            "sequence": trimmed_seq,
        }
        if "dorado_quality" in joined.schema.names:
            cols["dorado_quality"] = pc.cast(
                joined.column("dorado_quality"), pa.float32()
            )
        else:
            cols["dorado_quality"] = pa.nulls(
                joined.num_rows, type=pa.float32()
            )
        trimmed_table = pa.table(cols)
    _log(f"trimmed_table built ({trimmed_table.num_rows:,} rows)")

    # ── read_to_sample for --per-sample-clusters ─────────────────────
    # 2-column Arrow table replaces the prior ~200M-entry Python dict
    # that materialised at this point at mouse scale. The downstream
    # cluster_by_fingerprint consumer joins it via Arrow.
    read_to_sample_table: pa.Table | None = None
    if args.per_sample_clusters and read_demux.num_rows > 0:
        rd_valid = read_demux.filter(
            pc.is_valid(read_demux.column("sample_id"))
        )
        read_to_sample_table = rd_valid.select(["read_id", "sample_id"])
        _log(
            f"read_to_sample built ({read_to_sample_table.num_rows:,} rows)"
        )

    # ── Cluster ──────────────────────────────────────────────────────
    _log("cluster_by_fingerprint: enter")
    clusters, membership = cluster_by_fingerprint(
        fingerprints_table,
        trimmed_table,
        alignments=alignments_table,
        alignment_blocks=blocks_table,
        alignment_cs=alignment_cs_dataset,
        genome=genome,
        read_to_sample=read_to_sample_table,
        max_5p_drift=int(args.max_5p_drift),
        max_3p_drift=int(args.max_3p_drift),
        min_cluster_size=int(args.min_cluster_size),
        build_consensus_seq=bool(args.build_consensus),
        drop_drift_filtered=bool(args.drop_drift_filtered),
        per_sample_clusters=bool(args.per_sample_clusters),
        cluster_id_seed=0,
        consensus_threads=int(args.threads),
        progress=_log if args.progress else None,
    )
    _log(
        f"cluster_by_fingerprint: done "
        f"({clusters.num_rows:,} clusters, {membership.num_rows:,} membership rows)"
    )

    # ── Write outputs ────────────────────────────────────────────────
    pq.write_table(clusters, output_dir / "clusters.parquet")
    pq.write_table(membership, output_dir / "cluster_membership.parquet")
    _log("wrote clusters.parquet + cluster_membership.parquet")

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
        _log("wrote cluster.fa")

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
        _log("wrote cluster_summary.tsv")

    # ── Manifest + _SUCCESS ──────────────────────────────────────────
    from constellation.sequencing.transcriptome.manifest import (
        write_cluster_manifest,
    )

    sample_names = (
        sorted(set(samples.samples.column("sample_name").to_pylist()))
        if samples.samples.num_rows > 0 else None
    )
    write_cluster_manifest(
        output_dir / "manifest.json",
        reference_handle=reference_handle,
        reference_path=reference_path_str,
        assembly_accession=assembly_accession,
        align_dir=str(align_dir),
        demux_dir=str(demux_dir),
        parameters={
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
        stages={
            "n_input_fingerprints": int(fingerprints_table.num_rows),
            "n_clusters": int(clusters.num_rows),
            "n_membership_rows": int(membership.num_rows),
            "n_samples": int(len(list(samples.ids))),
        },
        outputs=cluster_outputs,
        samples=sample_names,
    )
    (output_dir / "_SUCCESS").write_bytes(b"")

    # ── Diagnostic report ──────────────────────────────────────────────
    # Failures don't break the pipeline; they log a re-run hint pointing
    # at the standalone `transcriptome diagnose` verb.
    if getattr(args, "report", True):
        try:
            from constellation.sequencing.transcriptome.cluster.diagnostics import (
                build_cluster_diagnostics_report,
            )
            report_path = build_cluster_diagnostics_report(
                output_dir,
                reference=genome,
            )
            if not args.progress:
                print(
                    f"diagnostic report: {report_path}",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"WARNING: diagnostic report generation failed: "
                f"{type(exc).__name__}: {exc}. The cluster outputs are "
                f"complete; re-run with `constellation transcriptome "
                f"diagnose --cluster-dir {output_dir}` to retry.",
                file=sys.stderr,
                flush=True,
            )

    if not args.progress:
        print(
            f"cluster done: {fingerprints_table.num_rows} input fingerprints → "
            f"{clusters.num_rows} clusters "
            f"({membership.num_rows} membership rows)",
            flush=True,
        )
    return 0


def _cmd_transcriptome_diagnose(args: argparse.Namespace) -> int:
    """Regenerate diagnostic report(s) for an existing align / cluster dir.

    Read-only — never re-runs pipeline stages. Loads each input dir's
    manifest, resolves a reference (from --reference / --reference-dir
    override or the manifest), and calls the corresponding
    build_*_diagnostics_report orchestrator. Output paths default to
    ``<input-dir>/diagnostics/`` per stage.
    """
    # Must precede any pyarrow / sequencing import — see
    # _preload_matplotlib_for_reports docstring for the libstdc++
    # ordering rationale.
    _preload_matplotlib_for_reports()

    import sys
    from pathlib import Path

    if args.reference is not None and args.reference_dir is not None:
        print(
            "error: --reference and --reference-dir are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    if not args.align_dir and not args.cluster_dir:
        print(
            "error: at least one of --align-dir / --cluster-dir is required",
            file=sys.stderr,
        )
        return 2

    # Reference resolution — try the override first; if no override and
    # both stages share the same reference (via manifest), the orchestrators
    # auto-load it themselves. We only resolve here if an override was given.
    reference = None
    annotation = None
    if args.reference is not None or args.reference_dir is not None:
        from constellation.sequencing.annotation.io import load_annotation
        from constellation.sequencing.reference.io import load_genome_reference

        ref_path, _ref_handle, _ref_assembly = _resolve_reference_args(
            handle_arg=args.reference, dir_arg=args.reference_dir
        )
        if ref_path is None:
            return 1
        if (ref_path / "genome").is_dir():
            reference = load_genome_reference(ref_path / "genome")
        if (ref_path / "annotation").is_dir():
            annotation = load_annotation(ref_path / "annotation")

    rc = 0
    if args.align_dir:
        from constellation.sequencing.transcriptome.align.diagnostics import (
            build_align_diagnostics_report,
        )
        align_dir = Path(args.align_dir).expanduser().resolve()
        if not align_dir.is_dir():
            print(f"error: --align-dir not found: {align_dir}", file=sys.stderr)
            return 1
        try:
            report_path = build_align_diagnostics_report(
                align_dir,
                reference=reference,
                annotation=annotation,
                organism_profile=args.organism_profile,
            )
            print(f"align report: {report_path}", flush=True)
        except Exception as exc:
            print(
                f"error: align report generation failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            rc = 1

    if args.cluster_dir:
        from constellation.sequencing.transcriptome.cluster.diagnostics import (
            build_cluster_diagnostics_report,
        )
        cluster_dir = Path(args.cluster_dir).expanduser().resolve()
        if not cluster_dir.is_dir():
            print(f"error: --cluster-dir not found: {cluster_dir}", file=sys.stderr)
            return 1
        try:
            report_path = build_cluster_diagnostics_report(
                cluster_dir,
                reference=reference,
                annotation=annotation,
            )
            print(f"cluster report: {report_path}", flush=True)
        except Exception as exc:
            print(
                f"error: cluster report generation failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            rc = 1

    return rc


def _build_taxonomy_parser(subs) -> None:
    """Wire ``constellation taxonomy ...`` — name lookup + tree queries.

    Verbs:
        resolve <query>       exact lookup; prints taxid + scientific name + rank
        search <substring>    case-insensitive substring search across names
        lineage <query>       root → taxon lineage
        descendants <query>   all descendants (optionally filter by rank)
        update                fetch NCBI taxdump.tar.gz → refold to parquet
        where                 print active taxonomy source + counts
    """
    p_tax = subs.add_parser(
        "taxonomy",
        help=(
            "Name / taxid lookup over NCBI taxonomy; manage the lazy-fetched "
            "full taxonomy cache (bundled starter ships with the package)"
        ),
    )
    tax_subs = p_tax.add_subparsers(dest="tax_subcommand", required=True)

    p_resolve = tax_subs.add_parser(
        "resolve",
        help="Exact lookup of a species name or taxid",
    )
    p_resolve.add_argument(
        "query", help="scientific / common name, or integer taxid"
    )
    p_resolve.add_argument("--json", action="store_true")
    p_resolve.set_defaults(func=_cmd_taxonomy_resolve)

    p_search = tax_subs.add_parser(
        "search",
        help="Substring search across scientific + common + synonym names",
    )
    p_search.add_argument("query", help="substring to match (case-insensitive)")
    p_search.add_argument("--limit", type=int, default=25)
    p_search.add_argument(
        "--rank",
        default=None,
        help="restrict to taxa at a given rank (species / genus / family / ...)",
    )
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=_cmd_taxonomy_search)

    p_lineage = tax_subs.add_parser(
        "lineage",
        help="Print the root → taxon lineage (one rank per line)",
    )
    p_lineage.add_argument("query", help="scientific / common name, or integer taxid")
    p_lineage.add_argument(
        "--ranks-only",
        action="store_true",
        help="emit only rank + name (drop taxid + division)",
    )
    p_lineage.add_argument("--json", action="store_true")
    p_lineage.set_defaults(func=_cmd_taxonomy_lineage)

    p_desc = tax_subs.add_parser(
        "descendants",
        help="Print every descendant taxon below a node",
    )
    p_desc.add_argument("query", help="scientific / common name, or integer taxid")
    p_desc.add_argument(
        "--rank",
        default=None,
        help="restrict to descendants at a given rank",
    )
    p_desc.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="cap depth below the queried node (1 → direct children only)",
    )
    p_desc.add_argument("--json", action="store_true")
    p_desc.set_defaults(func=_cmd_taxonomy_descendants)

    p_update = tax_subs.add_parser(
        "update",
        help="Fetch the latest NCBI taxdump.tar.gz and refold to local parquet",
    )
    p_update.add_argument(
        "--force", action="store_true", help="re-fetch even if already installed today"
    )
    p_update.add_argument(
        "--timeout", type=int, default=900, help="HTTP timeout in seconds"
    )
    p_update.set_defaults(func=_cmd_taxonomy_update)

    p_where = tax_subs.add_parser(
        "where",
        help="Print active taxonomy source + taxon counts",
    )
    p_where.set_defaults(func=_cmd_taxonomy_where)


def _build_catalog_parser(subs) -> None:
    """Wire ``constellation catalog ...`` — assembly / proteome catalog management.

    Verbs:
        update <source>   fetch the source's species/assembly index
        list              one line per installed (source, release)
        show <source> <query>  print every catalog row matching <query>
    """
    p_cat = subs.add_parser(
        "catalog",
        help=(
            "Fetch + browse genome / proteome catalogs from Ensembl / Ensembl "
            "Genomes / RefSeq / UniProt"
        ),
    )
    cat_subs = p_cat.add_subparsers(dest="cat_subcommand", required=True)

    p_update = cat_subs.add_parser(
        "update",
        help="Fetch the source's species/assembly index into the local catalog cache",
    )
    p_update.add_argument(
        "source",
        choices=["ensembl", "ensembl_genomes", "refseq", "uniprot", "all"],
    )
    p_update.add_argument(
        "--release",
        default=None,
        help=(
            "Ensembl / Ensembl Genomes release number, or a UniProt release "
            "tag (e.g. '2024_02'). Defaults to the latest published release "
            "for Ensembl / Ensembl Genomes (probed from the source's FTP "
            "directory listing); defaults to today's date for RefSeq / "
            "UniProt. Pass an older release to download a historical build."
        ),
    )
    p_update.add_argument(
        "--force",
        action="store_true",
        help="re-fetch even if the requested release is already installed",
    )
    p_update.add_argument(
        "--timeout", type=int, default=900, help="HTTP timeout in seconds"
    )
    p_update.set_defaults(func=_cmd_catalog_update)

    p_list = cat_subs.add_parser(
        "list",
        help="List every installed catalog (one line per source/release)",
    )
    p_list.set_defaults(func=_cmd_catalog_list)

    p_show = cat_subs.add_parser(
        "show",
        help="Print every catalog row matching a species name or taxid",
    )
    p_show.add_argument("source", help="source to search (or 'all')")
    p_show.add_argument("query", help="species name, taxid, or substring")
    p_show.add_argument("--limit", type=int, default=25)
    p_show.add_argument("--json", action="store_true")
    p_show.set_defaults(func=_cmd_catalog_show)


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
            "'refseq:GCF_001708105.1', 'ensembl:human'. Also accepts a bare "
            "species name (e.g. 'Haliotis rufescens', 'red abalone') or an "
            "integer taxid (e.g. '9606'), routed through the taxonomy + "
            "catalog layer."
        ),
    )
    p_fetch.add_argument(
        "--source",
        choices=["refseq", "ensembl", "ensembl_genomes", "genbank", "uniprot"],
        default=None,
        help=(
            "pin the catalog source when resolving a bare species name / "
            "taxid (default: RefSeq-first precedence)"
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

    p_search = ref_subs.add_parser(
        "search",
        help=(
            "Search installed catalogs (Ensembl / RefSeq / UniProt / ...) "
            "for assemblies matching a species name or taxid"
        ),
    )
    p_search.add_argument(
        "query",
        help=(
            "species name (scientific / common / vernacular) or substring; "
            "or an integer taxid"
        ),
    )
    p_search.add_argument(
        "--source",
        choices=["refseq", "ensembl", "ensembl_genomes", "genbank", "uniprot"],
        default=None,
        help="restrict to a single catalog source",
    )
    p_search.add_argument(
        "--limit",
        type=int,
        default=25,
        help="maximum rows to print (default 25)",
    )
    p_search.add_argument(
        "--json",
        action="store_true",
        help="emit JSON lines instead of the human-readable table",
    )
    p_search.set_defaults(func=_cmd_reference_search)

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

    p_rm = ref_subs.add_parser(
        "remove",
        help="Delete a cached reference; cleans up defaults + current pointers",
    )
    p_rm.add_argument(
        "handle",
        help=(
            "fully-qualified handle ('homo_sapiens@ensembl-111') or bare "
            "organism slug ('homo_sapiens'). Bare slug removes the single "
            "installed release when unambiguous; with 2+ releases use a "
            "qualified handle or --all-releases."
        ),
    )
    p_rm.add_argument(
        "--all-releases",
        action="store_true",
        help=(
            "remove every installed release for the given organism (handle "
            "must be a bare organism slug, not <organism>@<release>)"
        ),
    )
    p_rm.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the confirmation prompt (only consulted for --all-releases)",
    )
    p_rm.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cleanup plan without deleting anything",
    )
    p_rm.set_defaults(func=_cmd_reference_remove)

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


def _resolve_reference_args(
    *,
    handle_arg: str | None,
    dir_arg: str | None,
) -> "tuple[object | None, str | None, str | None]":
    """Resolve the new dual --reference / --reference-dir CLI shape.

    Returns ``(release_path, reference_handle, assembly_accession)``. When
    the handle path is taken, the release dir is resolved via the
    reference cache and ``assembly_accession`` is read from
    ``meta.toml``. The escape-hatch path returns ``(path, None, None)`` —
    no handle, no assembly_accession, which downstream consumers (the
    dashboard) use to refuse opening the resulting outputs until the user
    runs ``constellation reference import``.

    On error, prints an actionable message to stderr and returns
    ``(None, None, None)``.
    """
    from pathlib import Path
    import sys

    if handle_arg is not None:
        from constellation.sequencing.reference.handle import (
            ReferenceNotInstalledError,
            read_meta_toml,
            resolve as resolve_handle,
        )

        try:
            release_path = resolve_handle(handle_arg)
        except (ValueError, ReferenceNotInstalledError) as exc:
            print(f"error: --reference {handle_arg!r}: {exc}", file=sys.stderr)
            return None, None, None
        meta = read_meta_toml(release_path) or {}
        assembly_accession = meta.get("assembly_accession")
        # Canonical handle string from meta.toml — preserves the qualified
        # form even when the user passed a bare organism slug that the
        # resolver expanded via defaults.toml / current.
        canonical_handle = str(meta.get("handle") or handle_arg)
        return (
            release_path,
            canonical_handle,
            (str(assembly_accession) if assembly_accession is not None else None),
        )

    if dir_arg is not None:
        path = Path(dir_arg).expanduser().resolve()
        if not path.is_dir():
            print(
                f"error: --reference-dir {dir_arg!r}: not a directory",
                file=sys.stderr,
            )
            return None, None, None
        return path, None, None

    return None, None, None


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
        source=args.source,
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


# ──────────────────────────────────────────────────────────────────────
# Taxonomy command handlers
# ──────────────────────────────────────────────────────────────────────


def _cmd_taxonomy_resolve(args: argparse.Namespace) -> int:
    import json
    import sys

    from constellation.core.taxonomy import TaxonomyResolver

    node = TaxonomyResolver.auto().lookup(args.query)
    if node is None:
        print(f"no taxon matches {args.query!r}", file=sys.stderr)
        return 1
    if args.json:
        import dataclasses

        print(json.dumps(dataclasses.asdict(node)))
        return 0
    print(
        f"taxid={node.taxid}  rank={node.rank}  name={node.scientific_name}  "
        f"parent_taxid={node.parent_taxid}  division={node.division}"
    )
    return 0


def _cmd_taxonomy_search(args: argparse.Namespace) -> int:
    import json
    import sys

    from constellation.core.taxonomy import TaxonomyResolver

    hits = TaxonomyResolver.auto().search(
        args.query, limit=args.limit, rank=args.rank
    )
    if not hits:
        print(f"no matches for {args.query!r}", file=sys.stderr)
        return 1
    if args.json:
        import dataclasses

        for node, matched in hits:
            print(
                json.dumps(
                    {"matched_name": matched, **dataclasses.asdict(node)}
                )
            )
        return 0
    print(f"{'taxid':>8}  {'rank':<14}  matched_name")
    for node, matched in hits:
        print(
            f"{node.taxid:>8}  {node.rank:<14}  {matched}  → {node.scientific_name}"
        )
    return 0


def _cmd_taxonomy_lineage(args: argparse.Namespace) -> int:
    import json
    import sys

    from constellation.core.taxonomy import TaxonomyResolver, UnknownTaxonError

    r = TaxonomyResolver.auto()
    try:
        node = r.lookup_strict(args.query)
    except UnknownTaxonError:
        print(f"no taxon matches {args.query!r}", file=sys.stderr)
        return 1
    lineage = r.lineage(node.taxid)
    if args.json:
        import dataclasses

        print(json.dumps([dataclasses.asdict(n) for n in lineage]))
        return 0
    if args.ranks_only:
        for n in lineage:
            print(f"{n.rank:<14}  {n.scientific_name}")
    else:
        for n in lineage:
            print(
                f"{n.rank:<14}  {n.scientific_name:<40}  taxid={n.taxid}"
            )
    return 0


def _cmd_taxonomy_descendants(args: argparse.Namespace) -> int:
    import json
    import sys

    from constellation.core.taxonomy import TaxonomyResolver, UnknownTaxonError

    r = TaxonomyResolver.auto()
    try:
        node = r.lookup_strict(args.query)
    except UnknownTaxonError:
        print(f"no taxon matches {args.query!r}", file=sys.stderr)
        return 1
    desc = r.descendants(node.taxid, rank=args.rank, max_depth=args.max_depth)
    if not desc:
        print(f"no descendants for {args.query!r}", file=sys.stderr)
        return 0
    if args.json:
        import dataclasses

        for d in desc:
            print(json.dumps(dataclasses.asdict(d)))
        return 0
    print(f"# {len(desc)} descendant(s) of {node.scientific_name} (taxid {node.taxid})")
    for d in desc:
        print(f"{d.taxid:>8}  {d.rank:<14}  {d.scientific_name}")
    return 0


def _cmd_taxonomy_update(args: argparse.Namespace) -> int:
    import sys

    from constellation.core.taxonomy import fetch_taxdump

    print("fetching NCBI taxdump.tar.gz (this may take a minute)...", file=sys.stderr)

    def _progress(seen: int, total: int) -> None:
        pct = (seen / total) * 100 if total else 0
        sys.stderr.write(
            f"\r  downloaded {seen / (1 << 20):.1f} / {total / (1 << 20):.1f} "
            f"MiB ({pct:.0f}%)"
        )
        sys.stderr.flush()

    bundle_dir = fetch_taxdump(timeout=args.timeout, progress_cb=_progress)
    sys.stderr.write("\n")
    print(f"installed taxonomy bundle at {bundle_dir}")
    return 0


def _cmd_taxonomy_where(_args: argparse.Namespace) -> int:
    from constellation.core.taxonomy import (
        TaxonomyResolver,
        list_installed,
        resolve_current,
        taxonomy_root,
    )

    root = taxonomy_root()
    print(f"taxonomy_root: {root}")
    current = resolve_current(root)
    if current is not None:
        print(f"current:       {current}")
    installed = list_installed(root)
    if installed:
        print("installed:")
        for p in installed:
            print(f"  {p.name}")
    else:
        print("installed:     (none — using bundled starter)")
    r = TaxonomyResolver.auto()
    src = r.source_meta().get("source", "?")
    print(f"active source: {src}  ({r.n_taxa()} taxa)")
    return 0


# ──────────────────────────────────────────────────────────────────────
# Catalog command handlers
# ──────────────────────────────────────────────────────────────────────


def _cmd_catalog_update(args: argparse.Namespace) -> int:
    import sys
    from datetime import datetime, timezone

    from constellation.catalog import (
        ensembl,
        ensembl_genomes,
        refseq,
        release_dir,
        uniprot,
        write_catalog,
    )

    src = args.source

    def _fetch_one(source_name: str) -> int:
        if source_name == "ensembl":
            if args.release is None:
                try:
                    release_int = ensembl.latest_release(timeout=args.timeout)
                except (RuntimeError, OSError) as exc:
                    print(
                        f"error: could not determine latest Ensembl release ({exc}); "
                        "pass --release N explicitly",
                        file=sys.stderr,
                    )
                    return 2
                print(f"  ensembl: latest release detected as {release_int}", file=sys.stderr)
            else:
                release_int = int(args.release)
            table, meta = ensembl.fetch_catalog(release_int, timeout=args.timeout)
            rel = str(release_int)
        elif source_name == "ensembl_genomes":
            if args.release is None:
                try:
                    release_int = ensembl_genomes.latest_release(timeout=args.timeout)
                except (RuntimeError, OSError) as exc:
                    print(
                        f"error: could not determine latest Ensembl Genomes release ({exc}); "
                        "pass --release N explicitly",
                        file=sys.stderr,
                    )
                    return 2
                print(
                    f"  ensembl_genomes: latest release detected as {release_int}",
                    file=sys.stderr,
                )
            else:
                release_int = int(args.release)
            table, meta = ensembl_genomes.fetch_catalog(
                release_int, timeout=args.timeout
            )
            rel = str(release_int)
        elif source_name == "refseq":
            rel = args.release or datetime.now(timezone.utc).strftime("%Y%m%d")
            table, meta = refseq.fetch_catalog(release_tag=rel, timeout=args.timeout)
        elif source_name == "uniprot":
            rel = args.release or datetime.now(timezone.utc).strftime("%Y%m%d")
            table, meta = uniprot.fetch_catalog(release=rel, timeout=args.timeout)
        else:
            raise ValueError(f"unknown source {source_name!r}")
        bundle = release_dir(source_name, rel)
        if bundle.exists() and not args.force:
            print(
                f"  {source_name}/{rel} already installed (pass --force to refresh)"
            )
            return 0
        write_catalog(bundle, table=table, meta=meta)
        print(f"  {source_name}/{rel}: wrote {table.num_rows} rows to {bundle}")
        return 0

    if src == "all":
        rc = 0
        for s in ("ensembl", "ensembl_genomes", "refseq", "uniprot"):
            print(f"updating {s}...", file=sys.stderr)
            sub_rc = _fetch_one(s)
            if sub_rc != 0:
                rc = sub_rc
        return rc
    return _fetch_one(src)


def _cmd_catalog_list(_args: argparse.Namespace) -> int:
    from constellation.catalog import catalogs_root, list_installed

    root = catalogs_root()
    print(f"catalogs_root: {root}")
    bundles = list_installed(root=root)
    if not bundles:
        print("installed:     (none — run `constellation catalog update <source>`)")
        return 0
    print("installed:")
    for b in bundles:
        n = b.table.num_rows
        print(f"  {b.source:<16}  release={b.release:<16}  n_rows={n}")
    return 0


def _cmd_catalog_show(args: argparse.Namespace) -> int:
    import json
    import sys

    from constellation.catalog import CatalogResolver
    from constellation.core.taxonomy import TaxonomyResolver

    catalogs = CatalogResolver.from_cache()
    if catalogs.is_empty():
        print("no catalogs installed", file=sys.stderr)
        return 2
    src = None if args.source == "all" else args.source

    node = TaxonomyResolver.auto().lookup(args.query)
    if node is not None:
        rows = catalogs.all_for(node.taxid, source=src)
    else:
        rows = catalogs.search(args.query, source=src, limit=args.limit)
    rows = rows[: args.limit]
    if not rows:
        print(f"no catalog hits for {args.query!r}", file=sys.stderr)
        return 1
    if args.json:
        import dataclasses

        for r in rows:
            print(json.dumps(dataclasses.asdict(r)))
        return 0
    for r in rows:
        print(
            f"[{r.source}/{r.release}] taxid={r.taxid} "
            f"{r.species_name} ({r.assembly_accession or r.assembly_name or '-'})"
        )
        print(f"    fasta:   {r.fasta_url}")
        if r.gff_url:
            print(f"    gff:     {r.gff_url}")
        if r.protein_url:
            print(f"    protein: {r.protein_url}")
        if r.cdna_url:
            print(f"    cdna:    {r.cdna_url}")
        print()
    return 0


def _cmd_reference_search(args: argparse.Namespace) -> int:
    """Search installed catalogs by species name / taxid / substring."""
    import json
    import sys

    from constellation.catalog import CatalogResolver
    from constellation.core.taxonomy import TaxonomyResolver

    catalogs = CatalogResolver.from_cache()
    if catalogs.is_empty():
        print(
            "no catalogs installed — run `constellation catalog update refseq` "
            "(and/or ensembl / ensembl_genomes / uniprot)",
            file=sys.stderr,
        )
        return 2

    query = args.query.strip()
    # Try taxonomy first; if it resolves, list every catalog hit for that taxid.
    rows = []
    tax = TaxonomyResolver.auto()
    node = tax.lookup(query)
    if node is not None:
        rows = catalogs.all_for(node.taxid, source=args.source)
        if rows:
            print(
                f"# resolved {query!r} -> taxid {node.taxid} "
                f"({node.scientific_name})",
                file=sys.stderr,
            )
    if not rows:
        rows = catalogs.search(query, source=args.source, limit=args.limit)
    rows = rows[: args.limit]
    if not rows:
        print(f"no catalog hits for {query!r}", file=sys.stderr)
        return 1

    if args.json:
        import dataclasses

        for r in rows:
            print(json.dumps(dataclasses.asdict(r)))
        return 0

    # Human-readable table.
    print(
        f"{'source':<16}  {'taxid':>8}  {'release':<24}  {'assembly':<28}  "
        f"species_name"
    )
    for r in rows:
        print(
            f"{r.source:<16}  {(r.taxid or 0):>8}  {r.release[:24]:<24}  "
            f"{(r.assembly_accession or r.assembly_name or '-')[:28]:<28}  "
            f"{r.species_name}"
        )
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


def _cmd_reference_remove(args: argparse.Namespace) -> int:
    """Delete a cached reference (one release, or all releases for an organism)."""
    import sys

    from constellation.sequencing.reference.handle import (
        ReferenceNotInstalledError,
        _installed_release_slugs,
        _read_current,
        cache_root,
        format_size,
        most_recent_release,
        parse_handle,
        parse_release_slug,
        read_defaults,
        remove_organism_all_releases,
        remove_release,
    )

    try:
        handle = parse_handle(args.handle)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    root = cache_root()
    organism_dir = root / handle.organism

    # ── --all-releases branch ─────────────────────────────────────────
    if args.all_releases:
        if handle.is_qualified():
            print(
                "error: --all-releases requires a bare <organism> handle "
                f"(got qualified handle {args.handle!r})",
                file=sys.stderr,
            )
            return 2
        if not organism_dir.is_dir():
            print(
                f"error: no cached releases for organism {handle.organism!r}",
                file=sys.stderr,
            )
            return 1
        slugs = sorted(_installed_release_slugs(organism_dir))
        if not slugs:
            print(
                f"error: no cached releases for organism {handle.organism!r}",
                file=sys.stderr,
            )
            return 1

        # Show what we're about to do, both for dry-run and for the
        # confirmation prompt.
        sizes_bytes = {
            slug: _dir_size_safe(organism_dir / slug) for slug in slugs
        }
        total_bytes = sum(sizes_bytes.values())

        if args.dry_run:
            print(
                f"would remove {len(slugs)} release(s) "
                f"({format_size(total_bytes)} total) for {handle.organism}:"
            )
            for slug in slugs:
                print(
                    f"  would remove {handle.organism}@{slug} "
                    f"({format_size(sizes_bytes[slug])})"
                )
            if handle.organism in read_defaults(root):
                print(f"  would unset defaults.toml entry for {handle.organism}")
            if _read_current(organism_dir) is not None:
                print(f"  would unlink current pointer for {handle.organism}")
            print(f"  would remove organism dir {handle.organism}")
            return 0

        if not args.yes:
            print(
                f"This will remove {len(slugs)} release(s) "
                f"({format_size(total_bytes)} total) for {handle.organism}:"
            )
            for slug in slugs:
                print(
                    f"  {handle.organism}@{slug}  ({format_size(sizes_bytes[slug])})"
                )
            try:
                reply = input("Proceed? [y/N]: ").strip().lower()
            except EOFError:
                reply = ""
            if reply not in {"y", "yes"}:
                print("aborted", file=sys.stderr)
                return 1

        try:
            results = remove_organism_all_releases(handle.organism)
        except ReferenceNotInstalledError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for r in results:
            print(f"removed {r.handle} ({format_size(r.size_bytes)})")
        last = results[-1]
        if last.default_changed:
            print(f"unset defaults.toml entry for {handle.organism}")
        if last.current_changed:
            print(f"unlinked current pointer for {handle.organism}")
        if last.organism_dir_removed:
            print(
                f"removed organism dir {handle.organism} "
                f"(no releases remaining)"
            )
        return 0

    # ── Single-release branch ─────────────────────────────────────────
    if not handle.is_qualified():
        if not organism_dir.is_dir():
            print(
                f"error: no cached releases for organism {handle.organism!r}",
                file=sys.stderr,
            )
            return 1
        slugs = sorted(_installed_release_slugs(organism_dir))
        if not slugs:
            print(
                f"error: no cached releases for organism {handle.organism!r}",
                file=sys.stderr,
            )
            return 1
        if len(slugs) > 1:
            print(
                f"error: ambiguous handle {handle.organism!r}: "
                f"{len(slugs)} releases installed.",
                file=sys.stderr,
            )
            print(f"Installed: {slugs}", file=sys.stderr)
            print(
                f"Qualify it (e.g. {handle.organism}@{slugs[0]}) "
                f"or pass --all-releases to remove every release for "
                f"{handle.organism}.",
                file=sys.stderr,
            )
            return 1
        try:
            handle = parse_release_slug(slugs[0], organism=handle.organism)
        except ValueError as exc:
            # Defensive — the slug came from the cache so it should parse.
            print(f"error: {exc}", file=sys.stderr)
            return 1

    release_slug = handle.release_slug()
    release_dir = organism_dir / release_slug
    if not release_dir.is_dir():
        print(
            f"error: no cached release at {release_dir}; nothing to remove",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        defaults = read_defaults(root)
        default_targeted = defaults.get(handle.organism) == release_slug
        current_path = _read_current(organism_dir)
        current_targeted = (
            current_path is not None and current_path.name == release_slug
        )
        new_handle = (
            most_recent_release(handle.organism, exclude={release_slug})
            if (default_targeted or current_targeted)
            else None
        )
        new_target = new_handle.release_slug() if new_handle is not None else None
        size_bytes = _dir_size_safe(release_dir)
        print(f"would remove {handle} ({format_size(size_bytes)})")
        if default_targeted:
            if new_target:
                print(
                    f"would retarget defaults.toml: {handle.organism} → {new_target}"
                )
            else:
                print(f"would unset defaults.toml entry for {handle.organism}")
        if current_targeted:
            if new_target:
                print(
                    f"would retarget current pointer: {handle.organism} → {new_target}"
                )
            else:
                print(f"would unlink current pointer for {handle.organism}")
        remaining_after = _installed_release_slugs(organism_dir) - {release_slug}
        if not remaining_after:
            print(f"would remove organism dir {handle.organism} (last release)")
        return 0

    try:
        removed = remove_release(handle)
    except ReferenceNotInstalledError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"removed {removed.handle} ({format_size(removed.size_bytes)})")
    if removed.default_changed:
        if removed.default_retargeted_to:
            print(
                f"retargeted defaults.toml: {handle.organism} → "
                f"{removed.default_retargeted_to}"
            )
        else:
            print(f"unset defaults.toml entry for {handle.organism}")
    if removed.current_changed:
        if removed.current_retargeted_to:
            print(
                f"retargeted current pointer: {handle.organism} → "
                f"{removed.current_retargeted_to}"
            )
        else:
            print(f"unlinked current pointer for {handle.organism}")
    if removed.organism_dir_removed:
        print(f"removed organism dir {handle.organism} (last release)")
    return 0


def _dir_size_safe(path) -> int:
    """Like ``handle._dir_size`` but tolerates a missing path (returns 0)."""
    from constellation.sequencing.reference.handle import _dir_size
    try:
        return _dir_size(path)
    except OSError:
        return 0


def _cmd_reference_summary(args: argparse.Namespace) -> int:
    """Print stats for a saved reference bundle."""
    from collections import Counter
    from pathlib import Path

    from constellation.sequencing.annotation.io import load_annotation
    from constellation.sequencing.reference.handle import read_meta_toml
    from constellation.sequencing.reference.io import load_genome_reference

    root, _ = _resolve_reference_argument(args.ref_dir)
    root = Path(root)
    # Surface the meta.toml header (organism / strain / accession / fetched_at)
    # so users can confirm what's on disk without poking at the file directly.
    meta = read_meta_toml(root)
    if meta:
        for key in (
            "handle",
            "organism",
            "scientific_name",
            "strain",
            "assembly_accession",
            "assembly_name",
            "annotation_release",
            "taxid",
            "fetched_at",
        ):
            value = meta.get(key)
            if value is None or value == "":
                continue
            print(f"{key}: {value}")
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
