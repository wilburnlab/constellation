"""``constellation transcriptome-to-proteome`` — the lab's transcript-
counts → novel-protein-search workflow.

Chains the end-to-end orchestrator in
:mod:`constellation.transcriptome_to_proteome` (per-sample protein
counts → alignment-filtered predicted .dlib → GPF .elib → per-injection
searches → library-export quant report → novel-peptide
classifications). Defaults match the cartographer pipeline; the
collision filter is on by default (the v6.5.15 sweep validation showed
it's still load-bearing) and applied at the GPF library stage.

Reference handling is portal-native: ``--reference`` accepts a handle
(``<organism>@<source>-<release>`` or a bare organism slug that
resolves via ``defaults.toml`` / ``current/``), ``--reference-from``
inherits the handle from an upstream ``transcriptome align`` /
``transcriptome cluster`` output manifest. SwissProt defaults to
whatever ``Reference.open("swissprot")`` resolves; pin a specific
release with ``--swissprot-reference``. Users with one-off local
FASTA/GFF pairs should register them first via ``constellation
reference import <dir> --organism <slug>`` and then use the handle.

Top-level CLI verb per the project rule "single-purpose verbs go at
the top level; umbrellas get introduced only when ≥2 siblings exist."
When a second cross-modality workflow lands, this and that workflow
can be collapsed under a shared umbrella — defer until two workflows
exist to compare.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from constellation.massspec.search.encyclopedia.ptm_defaults import (
    default_for as _ptm_default_for,
)


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
        "--demux-dir",
        required=True,
        type=Path,
        help=(
            "path to a Constellation demux output directory (preferred); "
            "also accepts a TSV or parquet count table for external "
            "inputs. Demux is reference-free by design — the reference "
            "is supplied separately via --reference or --reference-from."
        ),
    )
    p.add_argument(
        "--protein-fasta",
        type=Path,
        default=None,
        help=(
            "ORF FASTA from a different upstream pipeline; overrides the "
            "novel ORF sequences the orchestrator otherwise derives from "
            "the `--demux-dir` `sequence` column."
        ),
    )
    p.add_argument(
        "--reference",
        type=str,
        default=None,
        help=(
            "reference handle — qualified (``<organism>@<source>-<release>``, "
            "e.g. ``mus_musculus@refseq-GCF_000001635.27-ar109``) or a bare "
            "organism slug that resolves via defaults.toml / current. "
            "Required unless --reference-from is supplied."
        ),
    )
    p.add_argument(
        "--reference-from",
        type=Path,
        default=None,
        help=(
            "inherit the reference handle from an upstream "
            "``transcriptome align`` or ``transcriptome cluster`` output "
            "directory's manifest. Mutually consistent with --reference; "
            "disagreements exit 2."
        ),
    )
    p.add_argument(
        "--gpf",
        required=True,
        nargs="+",
        type=Path,
        help=(
            "one or more GPF spectra inputs (gas-phase fractions). "
            "Each path may be a file (Thermo `.raw`, `.mzML`, `.dia`, "
            "or Bruker `.d`) or a directory; directories are scanned "
            "for spectra files. When the same stem appears in multiple "
            "formats, the orchestrator picks one in this preference "
            "order: `.dia` (preprocessed cache, skip Stage 6) > "
            "`.raw` > `.mzML` > `.d`."
        ),
    )
    p.add_argument(
        "--injections",
        required=True,
        nargs="+",
        type=Path,
        help=(
            "one or more per-sample individual-injection spectra "
            "inputs. Same path semantics as --gpf: files or "
            "directories accepted, same `.dia` > `.raw` > `.mzML` > "
            "`.d` per-stem preference."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="run directory (per-stage subdirs created here)",
    )

    # ── SwissProt — handle-based selection ─────────────────────────────
    p.add_argument(
        "--swissprot-reference",
        type=str,
        default=None,
        help=(
            "pin a specific SwissProt release as a portal handle "
            "(e.g. ``swissprot@uniprot-2026_02``). Default: the bare "
            "``swissprot`` handle, resolved via defaults.toml / current "
            "to whatever is installed. If no SwissProt is installed, "
            "run ``constellation reference fetch uniprot:swissprot`` "
            "(optionally with ``--release 2026_02`` to pin a specific "
            "release)."
        ),
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
        "--threads", type=int, default=8,
        help=(
            "worker threads for both mmseqs2 and EncyclopeDIA (default 8). "
            "In Stage 9 the per-injection EncyclopeDIA searches divide this "
            "across the parallel JVMs (each gets threads // --injection-threads, "
            "min 1), so total core use stays ~--threads."
        ),
    )
    p.add_argument(
        "--mmseqs-threads", type=int, default=None,
        help="override mmseqs2 thread count (default: --threads)",
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
    # PTM passthroughs — flag set + defaults imported from
    # ``massspec.search.encyclopedia.ptm_defaults`` so the orchestrator
    # and ``massspec predict-library`` cannot drift on which mods land
    # in the predicted library.
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
        default = _ptm_default_for(ptm_name)
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
    """Handler — resolves the reference + swissprot_reference handles
    then delegates to the orchestrator function.

    Reference resolution precedence (see plan):
      1. Explicit ``--reference`` always wins.
      2. ``--reference-from DIR`` reads ``reference_handle`` from the
         upstream ``align`` / ``cluster`` manifest. Rejects ``demux``.
      3. Both supplied with disagreeing handles → exit 2.
      4. Neither given → exit 2 with the actionable hint.

    SwissProt: pin via ``--swissprot-reference``, or default to
    ``Reference.open("swissprot")`` which resolves via defaults.toml /
    current. No lazy fetch — missing SwissProt → exit 2 with a hint.
    """
    try:
        reference = _resolve_reference_from_args(args)
    except _ResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        swissprot_reference = _resolve_swissprot_from_args(args)
    except _ResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Expand directories + dedup by stem before handing off to the
    # orchestrator so Stage 6 + Stage 9 see one canonical file per
    # acquisition (avoids re-converting .raw → .dia when a cache is
    # already on disk).
    try:
        args.gpf = _expand_spectra_inputs(args.gpf, flag="--gpf")
        args.injections = _expand_spectra_inputs(
            args.injections, flag="--injections"
        )
    except _ResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from constellation.transcriptome_to_proteome import (
        run_transcriptome_to_proteomics,
    )
    return run_transcriptome_to_proteomics(
        args=args,
        reference=reference,
        swissprot_reference=swissprot_reference,
    )


class _ResolutionError(Exception):
    """CLI-handler-local exception so the resolvers can raise concise
    messages that the handler surfaces as ``error: ...`` + exit 2."""


# ──────────────────────────────────────────────────────────────────────
# Spectra-input expansion (--gpf / --injections)
# ──────────────────────────────────────────────────────────────────────


# Format preference table for per-stem deduplication. Lower rank wins.
# ``.dia`` is preferred over ``.raw`` so Stage 6 picks up an already-
# preprocessed cache and doesn't re-decode the raw bytes. ``.raw`` is
# the canonical Thermo lab input. ``.mzML`` is the legacy intermediate
# (EncyclopeDIA 6.5.15 added native ``.raw`` support so mzML is
# unlikely going forward). ``.d`` is the Bruker timsTOF directory.
_SPECTRA_RANK: dict[str, int] = {
    ".dia": 0,
    ".raw": 1,
    ".mzml": 2,
    ".d": 3,
}


def _is_spectra_path(p: Path) -> bool:
    """True iff ``p`` is a spectra file or a Bruker ``.d`` directory."""
    suffix = p.suffix.lower()
    if suffix not in _SPECTRA_RANK:
        return False
    if suffix == ".d":
        # `.d` is a directory bundle; reject lone files with that suffix.
        return p.is_dir()
    return p.is_file()


def _collect_spectra_files(paths: Sequence[Path]) -> list[Path]:
    """Walk every input path (file or directory) and return spectra files.

    Bruker ``.d`` paths are passed through verbatim (they're directory
    bundles). All other directories are scanned non-recursively for
    spectra files at the top level — nested scans would silently glob
    archived runs that the user didn't intend to process.
    """
    out: list[Path] = []
    for p in paths:
        resolved = Path(p).expanduser().resolve()
        suffix = resolved.suffix.lower()
        if suffix == ".d" and resolved.is_dir():
            # Bruker bundle — pass through.
            out.append(resolved)
            continue
        if resolved.is_dir():
            for entry in sorted(resolved.iterdir()):
                if _is_spectra_path(entry):
                    out.append(entry.resolve())
            continue
        if not resolved.exists():
            raise _ResolutionError(
                f"input path {resolved} does not exist"
            )
        if not _is_spectra_path(resolved):
            raise _ResolutionError(
                f"input path {resolved} is not a recognised spectra file "
                f"(supported: {sorted(_SPECTRA_RANK)})"
            )
        out.append(resolved)
    return out


def _expand_spectra_inputs(
    paths: Sequence[Path], *, flag: str
) -> list[Path]:
    """Expand a list of file/directory paths into a deduplicated
    spectra-file list ready for Stage 6 (process-dia) / Stage 9
    (per-injection search).

    Per-stem dedup: when ``sample01.raw`` and ``sample01.dia`` are both
    present, the ``.dia`` cache wins so Stage 6 doesn't redo the raw
    decode. Dropped entries are logged to stderr so the user can audit
    the choices.

    Raises :class:`_ResolutionError` (handler-surfaced as ``error: ...``)
    if no spectra files are found.
    """
    candidates = _collect_spectra_files(paths)
    if not candidates:
        raise _ResolutionError(
            f"{flag} matched no spectra files; expected one or more "
            f"file or directory paths to "
            f"{sorted(_SPECTRA_RANK)} inputs."
        )

    # Per-stem dedup. Group by stem; keep the highest-preference
    # (lowest rank) entry; log every drop so the user sees the picking.
    by_stem: dict[str, Path] = {}
    drops: list[tuple[Path, Path]] = []
    for cand in candidates:
        stem = cand.stem
        rank_new = _SPECTRA_RANK[cand.suffix.lower()]
        existing = by_stem.get(stem)
        if existing is None:
            by_stem[stem] = cand
            continue
        rank_existing = _SPECTRA_RANK[existing.suffix.lower()]
        if rank_new < rank_existing:
            drops.append((existing, cand))
            by_stem[stem] = cand
        else:
            drops.append((cand, existing))

    for dropped, kept in drops:
        print(
            f"{flag}: stem {kept.stem!r} appears in both "
            f"{dropped.suffix} and {kept.suffix}; using {kept.name}, "
            f"skipping {dropped.name}",
            file=sys.stderr,
        )

    return sorted(by_stem.values())


def _resolve_reference_from_args(args: argparse.Namespace):
    """Return a Reference for the primary (genome/proteome+annotation) input."""
    from constellation.sequencing.reference import Reference
    from constellation.sequencing.reference.handle import (
        ReferenceNotInstalledError,
    )

    handle_explicit: str | None = args.reference
    handle_from_manifest: str | None = None
    if args.reference_from is not None:
        handle_from_manifest = _handle_from_manifest_dir(Path(args.reference_from))

    if handle_explicit is None and handle_from_manifest is None:
        raise _ResolutionError(
            "no reference supplied; pass --reference <handle> or "
            "--reference-from <align-or-cluster-dir>. To use a local "
            "FASTA/GFF pair, register it first with: constellation "
            "reference import <dir> --organism <slug>"
        )

    if (
        handle_explicit is not None
        and handle_from_manifest is not None
        and handle_explicit != handle_from_manifest
    ):
        raise _ResolutionError(
            f"--reference {handle_explicit!r} disagrees with "
            f"--reference-from manifest handle {handle_from_manifest!r}; "
            "re-run with one or the other (or update the upstream "
            "align/cluster run if the organism changed)."
        )

    handle = handle_explicit or handle_from_manifest
    try:
        return Reference.open(handle)
    except ReferenceNotInstalledError as exc:
        raise _ResolutionError(str(exc)) from exc


def _resolve_swissprot_from_args(args: argparse.Namespace):
    """Return a Reference for the SwissProt FASTA used by Stage 3 + Stage 4."""
    from constellation.sequencing.reference import Reference
    from constellation.sequencing.reference.handle import (
        ReferenceNotInstalledError,
    )

    handle = args.swissprot_reference or "swissprot"
    try:
        return Reference.open(handle)
    except ReferenceNotInstalledError as exc:
        raise _ResolutionError(
            f"{exc}\nNo SwissProt reference installed. Run:\n"
            "  constellation reference fetch uniprot:swissprot\n"
            "or pin a specific release with "
            "--swissprot-reference swissprot@uniprot-<release>."
        ) from exc


def _handle_from_manifest_dir(manifest_dir: Path) -> str:
    """Extract a ``reference_handle`` from an upstream align/cluster manifest.

    Demux outputs are rejected — demux is reference-free by design.
    Escape-hatch align/cluster runs (raw-path inputs, no
    ``reference_handle`` stamped) are rejected with a hint pointing at
    explicit ``--reference``.
    """
    from constellation.sequencing.transcriptome.manifest import (
        AlignManifest,
        ClusterManifest,
        read_manifest_dir,
    )

    if not manifest_dir.is_dir():
        raise _ResolutionError(
            f"--reference-from {manifest_dir} is not a directory"
        )
    try:
        manifest = read_manifest_dir(manifest_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise _ResolutionError(
            f"--reference-from {manifest_dir} could not be read as a "
            f"transcriptome manifest dir: {exc}"
        ) from exc
    if not isinstance(manifest, (AlignManifest, ClusterManifest)):
        raise _ResolutionError(
            f"--reference-from points at a {manifest.kind} directory; "
            "demux is reference-free by design — point at an align or "
            "cluster output instead"
        )
    if not manifest.reference_handle:
        raise _ResolutionError(
            f"--reference-from manifest at {manifest_dir} has no "
            "reference_handle (escape-hatch upstream run that used raw "
            "paths); supply --reference explicitly"
        )
    return manifest.reference_handle


__all__ = ["build_parser"]
