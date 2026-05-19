"""``constellation pipeline ...`` — cross-modality workflow orchestrators.

The first verb under this group is ``transcriptome-to-proteomics`` —
chains the lab's transcript-counts → novel-protein-search workflow
end-to-end, from per-sample protein counts through alignment-filtered
predicted .dlib, GPF .elib, per-injection searches, library-export
quant report, and novel-peptide classifications. Defaults match the
cartographer pipeline; the collision filter is on by default (the
v6.5.15 sweep validation showed it's still load-bearing) and applied
at every output stage.

Future siblings (genome→spectral, structure→spectral) will land here
when they exist; until then this is the only ``pipeline`` verb.
"""

from __future__ import annotations

import argparse
from pathlib import Path


_VALID_PTM_TOGGLES = ("off", "var", "fix")


def build_parser(subs: argparse._SubParsersAction) -> None:
    """Register the ``pipeline`` subcommand group."""
    p_pipeline = subs.add_parser(
        "pipeline",
        help="Cross-modality orchestrators (e.g. transcriptome → proteomics)",
    )
    pipeline_subs = p_pipeline.add_subparsers(dest="pipeline_subcommand", required=True)
    _build_transcriptome_to_proteomics_parser(pipeline_subs)


def _build_transcriptome_to_proteomics_parser(
    subs: argparse._SubParsersAction,
) -> None:
    p = subs.add_parser(
        "transcriptome-to-proteomics",
        help=(
            "End-to-end: protein counts → alignment-filtered combined "
            "FASTA → predicted .dlib → GPF search → per-injection "
            "searches → library-export quant report. Collision filter "
            "applied at every proteomics output stage (default ON)."
        ),
    )
    # ── Required inputs ────────────────────────────────────────────────
    p.add_argument(
        "--protein-counts",
        required=True,
        type=Path,
        help=(
            "Per-sample protein counts. Accepts the wide TSV from "
            "`constellation transcriptome demultiplex` "
            "(protein_counts.tsv), the long parquet "
            "(feature_quant.parquet), or the demultiplex output "
            "directory."
        ),
    )
    p.add_argument(
        "--proteins-fasta",
        required=True,
        type=Path,
        help="proteins.fasta from `constellation transcriptome demultiplex`",
    )
    p.add_argument(
        "--reference-fasta",
        required=True,
        type=Path,
        help="background reference proteome (e.g. RefSeq for the organism)",
    )
    p.add_argument(
        "--reference-annotation",
        required=True,
        type=Path,
        help=(
            "GFF3 or GBFF reference annotation — provides the "
            "protein_id → gene_symbol mapping for combined-FASTA "
            "header annotation"
        ),
    )
    p.add_argument(
        "--gpf",
        required=True,
        nargs="+",
        type=Path,
        help="one or more GPF .raw or .mzML files (gas-phase fractions)",
    )
    p.add_argument(
        "--injections",
        required=True,
        nargs="+",
        type=Path,
        help="one or more per-sample individual-injection mzML files",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="run directory (per-stage subdirs created here)",
    )

    # ── SwissProt — lazy-fetched by default ────────────────────────────
    p.add_argument(
        "--swissprot-fasta",
        type=Path,
        default=None,
        help=(
            "explicit SwissProt FASTA path; defaults to the lazy-fetched "
            "release at ~/.constellation/references/swissprot/<release>/"
        ),
    )
    p.add_argument(
        "--swissprot-release",
        default=None,
        help="pin a specific UniProt release (e.g. 2026_02); default = current",
    )

    # ── Filter knobs (cartographer defaults) ───────────────────────────
    p.add_argument(
        "--min-avg-tpm",
        type=float,
        default=1.0,
        help="min avg-TPM per protein to enter the alignment + search space (default 1.0)",
    )
    p.add_argument(
        "--min-sequence-length",
        type=int,
        default=100,
        help="min protein length kept in the TPM denominator (default 100)",
    )
    p.add_argument(
        "--evalue-threshold",
        type=float,
        default=1e-20,
        help="mmseqs2 e-value cutoff for the alignment-as-filter step (default 1e-20)",
    )

    # ── Stage toggles ──────────────────────────────────────────────────
    p.add_argument(
        "--no-collision-filter",
        action="store_true",
        help=(
            "skip the DIA collision filter at Stages 7 and 9 "
            "(default: filter ON — v6.5.15 validation showed it's "
            "still load-bearing)"
        ),
    )
    p.add_argument(
        "--collision-rt-threshold-s", type=float, default=5.0,
        help="collision filter: max |ΔRT| (s) for a pair (default 5.0)",
    )
    p.add_argument(
        "--collision-frag-ppm-tol", type=float, default=20.0,
        help="collision filter: fragment m/z tolerance ppm (default 20.0)",
    )
    p.add_argument(
        "--collision-min-shared-ions", type=int, default=4,
        help="collision filter: min shared ions to flag (default 4)",
    )

    # ── Threads + memory ───────────────────────────────────────────────
    p.add_argument(
        "--mmseqs-threads", type=int, default=4,
        help="mmseqs2 worker threads (default 4)",
    )
    p.add_argument(
        "--injection-threads", type=int, default=1,
        help=(
            "parallel injection-search fan-out (default 1; each worker "
            "spawns a full JVM with --jvm-heap, so size carefully)"
        ),
    )
    p.add_argument(
        "--jvm-heap", default="24g", dest="jvm_heap_max",
        help="JVM -Xmx (default 24g)",
    )
    p.add_argument(
        "--jvm-heap-min", default=None, dest="jvm_heap_min",
        help="JVM -Xms (default JVM default)",
    )
    p.add_argument(
        "--jvm-tmpdir", default=None, type=Path,
        help="JVM -Djava.io.tmpdir override",
    )

    # ── Per-stage passthroughs ─────────────────────────────────────────
    p.add_argument(
        "--fragment-tolerance-ppm",
        type=float,
        default=10.0,
        help="fragment m/z tolerance ppm for search (default 10.0)",
    )
    p.add_argument(
        "--precursor-tolerance-ppm",
        type=float,
        default=10.0,
        help="precursor m/z tolerance ppm for search (default 10.0)",
    )
    p.add_argument(
        "--percolator-version",
        default="v3-05",
        help="bundled Percolator version (default v3-05)",
    )
    p.add_argument(
        "--percolator-threshold",
        type=float,
        default=0.01,
        help="Percolator q-value threshold (default 0.01)",
    )
    # PTM passthroughs — mirror the predict-library subcommand's flags.
    for ptm_name, flag in (
        ("Acetyl", "--ptm-acetyl"),
        ("ProteinNTermAcetyl", "--ptm-protein-n-term-acetyl"),
        ("Carbamidomethyl", "--ptm-carbamidomethyl"),
        ("Deamidation", "--ptm-deamidation"),
        ("Dimethyl", "--ptm-dimethyl"),
        ("GlyGly", "--ptm-gly-gly"),
        ("HexNAc", "--ptm-hex-n-ac"),
        ("Methyl", "--ptm-methyl"),
        ("Oxidation", "--ptm-oxidation"),
        ("Phospho", "--ptm-phospho"),
        ("PyroGluQ", "--ptm-pyro-glu-q"),
        ("Succinyl", "--ptm-succinyl"),
        ("Trimethyl", "--ptm-trimethyl"),
        ("TMT", "--ptm-tmt"),
    ):
        default = "fix" if ptm_name == "Carbamidomethyl" else "off"
        p.add_argument(
            flag,
            choices=list(_VALID_PTM_TOGGLES),
            default=default,
            help=f"{ptm_name} mode (off|var|fix; default {default})",
        )

    # ── Standard ───────────────────────────────────────────────────────
    p.add_argument(
        "--resume", action="store_true",
        help="skip stages whose _SUCCESS exists; the orchestrator checks per-stage",
    )
    p.add_argument(
        "--no-progress", action="store_true",
        help="suppress live stderr progress streaming",
    )
    p.add_argument(
        "--no-ingest", action="store_true",
        help="skip auto-ingest of .elib outputs into ParquetDir bundles",
    )
    p.add_argument(
        "--encyclopedia-arg",
        action="append",
        default=[],
        metavar="FLAG=VALUE",
        help=(
            "repeatable escape hatch for unwrapped EncyclopeDIA flags; "
            "applies to every jar-invoking stage"
        ),
    )

    p.set_defaults(func=_cmd_pipeline_transcriptome_to_proteomics)


def _cmd_pipeline_transcriptome_to_proteomics(
    args: argparse.Namespace,
) -> int:
    """Handler — delegates to the orchestrator function in the top-level
    module so notebook / library callers can invoke the same workflow
    without going through argparse."""
    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )
    return run_transcriptome_to_proteomics(args=args)


__all__ = ["build_parser"]
