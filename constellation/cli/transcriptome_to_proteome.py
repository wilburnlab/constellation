"""``constellation transcriptome-to-proteome`` — the lab's transcript-
counts → novel-protein-search workflow.

Chains the end-to-end orchestrator in
:mod:`constellation.transcriptome_to_proteome` (per-sample protein
counts → alignment-filtered predicted .dlib → GPF .elib → per-injection
searches → library-export quant report → novel-peptide
classifications). Defaults match the cartographer pipeline; the
collision filter is on by default (the v6.5.15 sweep validation showed
it's still load-bearing) and applied at the GPF library stage.

Top-level CLI verb per the project rule "single-purpose verbs go at
the top level; umbrellas (`pipeline`, `bridges`) get introduced only
when ≥2 siblings exist." When a second cross-modality workflow lands,
this and that workflow can be collapsed under a shared umbrella —
defer until two workflows exist to compare.
"""

from __future__ import annotations

import argparse
from pathlib import Path


_VALID_PTM_TOGGLES = ("off", "var", "fix")


def build_parser(subs: argparse._SubParsersAction) -> None:
    """Register the ``transcriptome-to-proteome`` top-level verb."""
    p = subs.add_parser(
        "transcriptome-to-proteome",
        help=(
            "End-to-end: protein counts → alignment-filtered combined "
            "FASTA → predicted .dlib → GPF search → per-injection "
            "searches → library-export quant report. Collision filter "
            "applied at the GPF library stage (default ON)."
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
        "--reference-dir",
        type=Path,
        default=None,
        help=(
            "Constellation reference-cache directory, e.g. "
            "``~/.constellation/references/homo_sapiens/refseq-GCF_000001405.40/``. "
            "When provided, auto-resolves ``--reference-fasta`` to "
            "``<dir>/protein.faa`` and ``--reference-annotation`` to "
            "``<dir>/annotation/`` (the Constellation ParquetDir form). "
            "Explicit ``--reference-fasta`` / ``--reference-annotation`` "
            "flags override the auto-resolved defaults. If "
            "``--reference-dir`` is NOT supplied, both other flags are "
            "required."
        ),
    )
    p.add_argument(
        "--reference-fasta",
        type=Path,
        default=None,
        help=(
            "background reference proteome (e.g. RefSeq protein FASTA "
            "for the organism). Required unless ``--reference-dir`` is "
            "supplied."
        ),
    )
    p.add_argument(
        "--reference-annotation",
        type=Path,
        default=None,
        help=(
            "Reference annotation — accepts the Constellation parquet "
            "bundle (directory with features.parquet + manifest.json, "
            "as written by ``constellation reference fetch``), a GFF3 "
            "file, or a GBFF file. Required unless ``--reference-dir`` "
            "is supplied."
        ),
    )
    p.add_argument(
        "--gpf",
        required=True,
        nargs="+",
        type=Path,
        help=(
            "one or more GPF spectra files (gas-phase fractions). "
            "Thermo `.raw` is the canonical lab input; `.mzML` and "
            "preprocessed `.dia` caches are also accepted — Stage 6 "
            "(`process-dia`) handles all three transparently."
        ),
    )
    p.add_argument(
        "--injections",
        required=True,
        nargs="+",
        type=Path,
        help=(
            "one or more per-sample individual-injection spectra "
            "files. Same format flexibility as --gpf: `.raw` is the "
            "canonical lab input; `.mzML` and `.dia` accepted."
        ),
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

    # ── Run identification ─────────────────────────────────────────────
    p.add_argument(
        "--run-name",
        default=None,
        help=(
            "stem used for the merged GPF `.dia` cache "
            "(`<run-name>_combined_GPF.dia` in Stage 6). Defaults "
            "to the basename of --output-dir."
        ),
    )

    # ── Filter knobs (cartographer defaults) ───────────────────────────
    p.add_argument(
        "--min-avg-tpm",
        type=float,
        default=1.0,
        help=(
            "min avg-TPM per protein to enter the alignment + search "
            "space (default 1.0). Embedded into the TPM-cutoff-stamped "
            "output filenames (`combined_<N>TPM.fasta`, "
            "`combined_<N>TPM.dlib`, `combined_<N>TPM.elib`, "
            "`novel_peptides_<N>TPM.parquet`, etc.) so parallel sweeps "
            "with different cutoffs don't collide."
        ),
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
            "skip the DIA collision filter on the GPF library (Stage 7). "
            "Default: filter ON — v6.5.15 validation showed it's still "
            "load-bearing. The filter is applied once to the library; "
            "per-injection searches (Stage 9) inherit it transitively, "
            "so they are not filtered separately."
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

    p.set_defaults(func=_cmd_transcriptome_to_proteome)


def _cmd_transcriptome_to_proteome(args: argparse.Namespace) -> int:
    """Handler — resolves the reference-cache shorthand then delegates
    to the orchestrator function in the top-level module so notebook /
    library callers can invoke the same workflow without going through
    argparse.

    When the user supplies ``--reference-dir`` instead of (or alongside)
    ``--reference-fasta`` and ``--reference-annotation``, we resolve
    the implicit defaults here so the orchestrator's path-resolution
    block can stay simple. Explicit flags override the dir-derived
    defaults.
    """
    import sys

    if args.reference_dir is not None:
        ref_dir = Path(args.reference_dir).expanduser().resolve()
        if not ref_dir.is_dir():
            print(
                f"error: --reference-dir {ref_dir} is not a directory",
                file=sys.stderr,
            )
            return 2
        # Fill in defaults from the cache layout. Explicit flags win.
        if args.reference_fasta is None:
            args.reference_fasta = ref_dir / "protein.faa"
        if args.reference_annotation is None:
            args.reference_annotation = ref_dir / "annotation"

    # After dir-resolution both must be populated.
    missing = []
    if args.reference_fasta is None:
        missing.append("--reference-fasta")
    if args.reference_annotation is None:
        missing.append("--reference-annotation")
    if missing:
        print(
            f"error: {', '.join(missing)} required (or pass --reference-dir "
            f"to auto-resolve from the Constellation reference cache).",
            file=sys.stderr,
        )
        return 2

    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )
    return run_transcriptome_to_proteomics(args=args)


__all__ = ["build_parser"]
