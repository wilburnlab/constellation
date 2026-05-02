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
    p_dem.add_argument("--sam", required=True, help="path to a Dorado-emitted SAM")
    p_dem.add_argument(
        "--construct",
        default="cdna_wilburn_v1",
        help="LibraryConstruct panel name (see panels.available_panels())",
    )
    p_dem.add_argument(
        "--samples",
        required=True,
        help=(
            "TSV file: sample_id (int) + sample_name (str) + barcode_id "
            "(int, 0-indexed into the panel) per line; header optional"
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


def _cmd_transcriptome_demultiplex(args: argparse.Namespace) -> int:
    """Run the full S1 demux + ORF + quant pipeline on a SAM."""
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

    sam_path = Path(args.sam)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse the samples TSV. Format: sample_id<TAB>sample_name<TAB>barcode_id.
    # Lines starting with '#' or matching the header are skipped.
    sample_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    with open(args.samples, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cells = line.split("\t")
            if len(cells) < 3:
                continue
            sid_str, name, bid_str = cells[0], cells[1], cells[2]
            if sid_str == "sample_id":  # header line
                continue
            try:
                sid = int(sid_str)
                bid = int(bid_str)
            except ValueError:
                continue
            sample_rows.append(
                {"sample_id": sid, "sample_name": name, "description": None}
            )
            edge_rows.append(
                {
                    "sample_id": sid,
                    "acquisition_id": args.acquisition_id,
                    "barcode_id": bid,
                }
            )
    samples = Samples.from_records(samples=sample_rows, edges=edge_rows)

    cb = StreamProgress() if args.progress else NullProgress()

    artefacts = run_demux_pipeline(
        sam_path,
        construct_name=args.construct,
        samples=samples,
        acquisition_id=args.acquisition_id,
        output_dir=output_dir,
        batch_size=args.batch_size,
        n_workers=args.threads,
        min_protein_count=args.min_protein_count,
        progress_cb=cb,
        resume=args.resume,
    )

    reads = artefacts["reads"]
    demux_table = artefacts["demux_table"]
    quant_table = artefacts["quant_table"]
    fasta_records = artefacts["fasta_records"]

    # Manifest for reproducibility audit.
    manifest = {
        "input_sam": str(sam_path),
        "construct": args.construct,
        "acquisition_id": args.acquisition_id,
        "min_protein_count": args.min_protein_count,
        "n_reads": reads.num_rows,
        "n_demux_rows": demux_table.num_rows,
        "n_proteins_kept": len(fasta_records),
        "n_workers": args.threads,
        "batch_size": args.batch_size,
        "resumed": args.resume,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.progress:
        # Print a one-shot summary when --progress isn't on.
        print(
            f"demultiplex done: {reads.num_rows} reads → "
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
