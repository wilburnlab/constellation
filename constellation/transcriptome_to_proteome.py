"""Top-level cross-modality workflow: transcriptome → spectral library.

Bridge between :mod:`constellation.sequencing` (de novo transcriptome
assembly producing predicted proteins) and :mod:`constellation.massspec`
(spectral library prediction + DIA / DDA search). The lab's matched
genome + transcriptome + proteome workflow exits here.

This module hosts:

  * **Stage-2/3/4 helpers** for the transcriptome→proteomics pipeline:
    novel-FASTA writer, alignment-as-filter, combined-FASTA builder,
    competitive-target builder. Consumed by the orchestrator (PR 6).
  * **GFF3 / GBFF gene-mapping reader** — extracts protein_id →
    gene_symbol for the combined-FASTA header annotation.
  * **Pipeline orchestrator** (PR 6) — chains every stage of
    ``constellation pipeline transcriptome-to-proteomics``.

Placement: top-level ``constellation/`` per CLAUDE.md's "≥2 workflows
before a folder" rule. When a second cross-modality workflow lands
(structure → spectral, genome → spectral library directly bypassing
transcriptome), this and that workflow co-locate into a dedicated
subpackage with a name TBD (not ``bridges/`` or ``pipelines/``;
candidates: ``asterism/``, ``confluence/``, ``weft/``, ``synthesis/``,
``composer/``, ``lattice/`` — defer until the second workflow appears).
"""

from __future__ import annotations

import gzip
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc


# ──────────────────────────────────────────────────────────────────────
# Sequence-set hashing (matches the convention in
# constellation.sequencing.transcriptome.cluster)
# ──────────────────────────────────────────────────────────────────────


def _seq_hash(sequence: str) -> str:
    """Stable 128-bit blake2b hash of an uppercase AA sequence.

    Used for set-equality dedup against the reference proteome before
    novel proteins enter the search FASTA. Same algorithm
    ``sequencing.transcriptome.cluster`` uses for Layer-0 dereplication
    so the hash space stays consistent across the codebase.
    """
    return hashlib.blake2b(sequence.encode("ascii"), digest_size=16).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# FASTA I/O — local, stdlib-only
# ──────────────────────────────────────────────────────────────────────


def _open_text(path: Path):
    """Open a FASTA / GBFF / GFF3 for text reading, transparently
    handling gzip."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_fasta(path: Path):
    """Yield ``(header, sequence)`` per record. Lifted from
    ``sequencing.readers.fastx._iter_fasta`` to keep this module
    self-contained — that function is private upstream.
    """
    header: str | None = None
    seq_chunks: list[str] = []
    with _open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks).replace(" ", "").replace(
                        "\t", ""
                    )
                header = line[1:].rstrip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
    if header is not None:
        yield header, "".join(seq_chunks).replace(" ", "").replace("\t", "")


def _header_id(header: str) -> str:
    """First whitespace-delimited token of a FASTA header."""
    parts = header.split(None, 1)
    return parts[0] if parts else ""


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — filter novel proteins by avg TPM + dedup vs reference
# ──────────────────────────────────────────────────────────────────────


def filter_and_write_novel_fasta(
    *,
    counts_tpm: pa.Table,
    proteins_fasta: Path | str,
    reference_fasta: Path | str,
    output_path: Path | str,
    min_avg_tpm: float = 1.0,
) -> tuple[pa.Table, int]:
    """Filter the protein FASTA emitted by ``transcriptome demultiplex``
    to the proteins worth searching: those above the TPM threshold AND
    not in the reference proteome.

    Parameters
    ----------
    counts_tpm
        :data:`PROTEIN_COUNTS_LONG_SCHEMA`-shaped (with the ``tpm``
        column added by :func:`tpm_normalize`). Average TPM across
        samples is computed per ``protein_id``; rows below
        ``min_avg_tpm`` drop.
    proteins_fasta
        Source FASTA from demux (``proteins.fasta``). One record per
        unique protein discovered in the transcriptome.
    reference_fasta
        Background proteome (e.g. RefSeq). Used for sequence-set dedup
        — any novel sequence whose blake2b hash matches a reference
        sequence is dropped to avoid duplicating the background in
        the search FASTA.
    output_path
        Destination for the novel-only FASTA. Parent directory is
        created if absent.
    min_avg_tpm
        Average-TPM threshold (default 1.0 — cartographer's default).
        Set per the user's CLI ``--min-avg-tpm`` flag.

    Returns
    -------
    (novel_table, n_written)
        ``novel_table`` is a 3-column Arrow table:
        ``(protein_id, sequence, avg_tpm)``. ``n_written`` is the
        number of FASTA records emitted (matches ``novel_table.num_rows``
        when the demux FASTA covers every passing protein_id).
    """
    proteins_fasta = Path(proteins_fasta)
    reference_fasta = Path(reference_fasta)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Average TPM per protein.
    avg = (
        counts_tpm.group_by(["protein_id"])
        .aggregate([("tpm", "mean")])
        .rename_columns(["protein_id", "avg_tpm"])
    )
    avg = avg.filter(
        pc.greater_equal(avg.column("avg_tpm"), pa.scalar(min_avg_tpm))
    )
    if avg.num_rows == 0:
        output_path.write_text("")
        return pa.table(
            {
                "protein_id": pa.array([], type=pa.string()),
                "sequence": pa.array([], type=pa.string()),
                "avg_tpm": pa.array([], type=pa.float64()),
            }
        ), 0
    passing_ids: set[str] = set(avg.column("protein_id").to_pylist())
    avg_by_id: dict[str, float] = {
        r["protein_id"]: r["avg_tpm"] for r in avg.to_pylist()
    }

    # Load the source FASTA — keep only passing protein_ids.
    source_records: dict[str, str] = {}
    for header, seq in _iter_fasta(proteins_fasta):
        protein_id = _header_id(header)
        if protein_id in passing_ids:
            source_records[protein_id] = seq.upper()

    # Reference dedup via sequence hash.
    ref_hashes: set[str] = set()
    for _, seq in _iter_fasta(reference_fasta):
        ref_hashes.add(_seq_hash(seq.upper()))

    novel_records: list[tuple[str, str, float]] = []
    seen_hashes: set[str] = set(ref_hashes)
    for protein_id, seq in source_records.items():
        h = _seq_hash(seq)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        novel_records.append((protein_id, seq, avg_by_id[protein_id]))

    with output_path.open("w") as out:
        for protein_id, seq, avg_tpm in novel_records:
            out.write(f">{protein_id} avg_tpm={avg_tpm:.4f} source=transcriptome\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i + 60])
                out.write("\n")

    novel_table = pa.table(
        {
            "protein_id": pa.array(
                [r[0] for r in novel_records], type=pa.string()
            ),
            "sequence": pa.array(
                [r[1] for r in novel_records], type=pa.string()
            ),
            "avg_tpm": pa.array(
                [r[2] for r in novel_records], type=pa.float64()
            ),
        }
    )
    return novel_table, len(novel_records)


# ──────────────────────────────────────────────────────────────────────
# Stage 3.5 — alignment as filter
# ──────────────────────────────────────────────────────────────────────


def apply_alignment_filter(
    *,
    novel_table: pa.Table,
    alignment_hits: pa.Table,
    evalue_threshold: float = 1e-20,
) -> pa.Table:
    """Drop novel proteins that have NO mmseqs2 hit below
    ``evalue_threshold``.

    The cartographer rule: include only proteins for which the
    alignment-database search returned at least one match meeting
    the e-value threshold — unmatched proteins are likely sequencing
    errors / truncations / translation artefacts and inflate the
    search-space noise.

    Best-hit-per-query is the cartographer convention: group by
    ``query``, take min(evalue) per group, threshold, then semi-join
    novel_table on ``query == protein_id``.

    Parameters
    ----------
    novel_table
        Output of :func:`filter_and_write_novel_fasta` (or any table
        with a ``protein_id`` string column).
    alignment_hits
        :data:`ALIGNMENT_HIT_TABLE`-shaped — mmseqs2 .tab output
        loaded via :func:`constellation.core.io.schemas.read_mmseqs_tab`.
    evalue_threshold
        Max e-value to keep. Default ``1e-20`` matches cartographer's
        ``run_pipeline.py`` default.

    Returns
    -------
    pa.Table
        Subset of ``novel_table`` whose ``protein_id`` has at least
        one accepted mmseqs2 hit. Column set preserved.
    """
    if alignment_hits.num_rows == 0:
        # No hits at all → no proteins survive the filter.
        return novel_table.slice(0, 0)
    # Best hit per query (min evalue).
    best = (
        alignment_hits.group_by(["query"])
        .aggregate([("evalue", "min")])
        .rename_columns(["query", "best_evalue"])
    )
    best = best.filter(
        pc.less_equal(best.column("best_evalue"), pa.scalar(evalue_threshold))
    )
    passing_queries = best.column("query")
    # Semi-join: keep novel_table rows whose protein_id is in the
    # passing-query set.
    mask = pc.is_in(novel_table.column("protein_id"), value_set=passing_queries)
    return novel_table.filter(mask)


# ──────────────────────────────────────────────────────────────────────
# Reference gene-mapping reader (GFF3 / GBFF)
# ──────────────────────────────────────────────────────────────────────


_GFF_PROTEIN_ID_RE = re.compile(r"protein_id=([^;]+)")
_GFF_GENE_RE = re.compile(r"\bgene=([^;]+)")


def read_reference_gene_map(annotation_path: Path | str) -> dict[str, str]:
    """Parse a GFF3 or GBFF reference annotation → protein_id → gene_symbol.

    Routes by file extension:

      * ``.gff`` / ``.gff3`` (+ ``.gz``) → GFF3 parser. Iterates CDS
        records, extracts ``protein_id=`` and ``gene=`` attributes.
      * ``.gbff`` / ``.gb`` / ``.genbank`` (+ ``.gz``) → GBFF parser.
        Walks LOCUS records, scans CDS feature qualifier blocks for
        ``/protein_id="..."`` + ``/gene="..."`` pairs.

    Used by :func:`build_combined_fasta` to annotate reference protein
    headers with their gene symbol. Unmatched proteins (no ``gene=``
    or ``/gene="..."`` in the annotation) are not included in the
    returned mapping; the caller decides whether to emit them with
    ``[gene=]`` empty or skip the tag entirely.

    Parameters
    ----------
    annotation_path
        Path to a GFF3 or GBFF file (optionally gzipped).

    Returns
    -------
    dict[str, str]
        ``protein_id → gene_symbol`` mapping. Keys are the CDS
        record's ``protein_id`` attribute (RefSeq's NP_/XP_ accession);
        values are the gene symbol (e.g. ``ACTB``).
    """
    p = Path(annotation_path)
    suffixes = [s.lower() for s in p.suffixes]
    if any(s in {".gff", ".gff3"} for s in suffixes):
        return _read_gff3_gene_map(p)
    if any(s in {".gbff", ".gb", ".genbank"} for s in suffixes):
        return _read_gbff_gene_map(p)
    raise ValueError(
        f"unsupported annotation extension {p.suffixes!r}: expected "
        f".gff / .gff3 / .gbff / .gb / .genbank (optionally .gz)"
    )


def _read_gff3_gene_map(path: Path) -> dict[str, str]:
    """Scan GFF3 CDS records for protein_id + gene attributes."""
    out: dict[str, str] = {}
    with _open_text(path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            feature_type = cols[2]
            if feature_type != "CDS":
                continue
            attrs = cols[8]
            m_pid = _GFF_PROTEIN_ID_RE.search(attrs)
            m_gene = _GFF_GENE_RE.search(attrs)
            if m_pid is None or m_gene is None:
                continue
            protein_id = m_pid.group(1).strip()
            gene = m_gene.group(1).strip()
            if protein_id and gene:
                out[protein_id] = gene
    return out


_GBFF_QUALIFIER_RE = re.compile(r'^\s+/(\w+)=(.*)$')


def _read_gbff_gene_map(path: Path) -> dict[str, str]:
    """Scan GBFF CDS qualifier blocks for /protein_id="..." + /gene="..."
    pairs.

    Format reference (GenBank Flat File):

        FEATURES             Location/Qualifiers
             ...
             CDS             1..345
                             /gene="ACTB"
                             /protein_id="NP_001185763.1"
                             ...

    Multi-line qualifier values (continuation lines indented but
    without ``/key=`` prefix) are concatenated. We only need the
    single-line case for protein_id + gene, but the parser tolerates
    continuations gracefully.
    """
    out: dict[str, str] = {}
    in_features = False
    in_cds = False
    current: dict[str, str] = {}

    def _flush():
        nonlocal current
        pid = current.get("protein_id")
        gene = current.get("gene")
        if pid and gene:
            out[pid] = gene
        current = {}

    with _open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("FEATURES"):
                in_features = True
                continue
            if line.startswith("ORIGIN") or line.startswith("//"):
                # End of features block → flush current CDS if open.
                if in_cds:
                    _flush()
                    in_cds = False
                in_features = False
                continue
            if not in_features:
                continue
            # Feature header line: starts at col 5 with the feature
            # type, e.g. ``     CDS             1..345``.
            if line[:5] == "     " and line[5:6].strip():
                # New feature begins — flush any open CDS first.
                if in_cds:
                    _flush()
                head = line[5:21].strip()
                in_cds = head == "CDS"
                continue
            if not in_cds:
                continue
            m = _GBFF_QUALIFIER_RE.match(line)
            if m is None:
                continue
            key = m.group(1)
            val = m.group(2)
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if key in {"protein_id", "gene"}:
                current[key] = val.strip()
    # End-of-file: flush any open CDS.
    if in_cds:
        _flush()
    return out


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — build combined.fasta with annotated headers
# ──────────────────────────────────────────────────────────────────────


def build_combined_fasta(
    *,
    reference_fasta: Path | str,
    filtered_novel: pa.Table,
    alignment_hits: pa.Table,
    reference_gene_map: Mapping[str, str],
    output_path: Path | str,
    dedup_reference_against_self: bool = True,
) -> tuple[int, dict[str, int]]:
    """Concatenate reference + filtered-novel into a single search FASTA
    with annotated headers.

    Header format::

        >protein_id [gene=SYMBOL] [accession=ACC] [aligned_to=refseq|swissprot] source=transcriptome|refseq

    Both reference and novel records always carry ``source=`` (load-
    bearing per the user's "header annotation is not optional" rule)
    and ``[gene=...]`` when the gene-mapping is available (``gene=``
    is omitted for records without a known gene rather than emitting
    an empty value — emit a build-time warning + count instead).

    Parameters
    ----------
    reference_fasta
        Background proteome FASTA.
    filtered_novel
        Output of :func:`apply_alignment_filter` — the novel proteins
        whose mmseqs2 hit survived the e-value threshold.
        ``(protein_id, sequence)`` columns required.
    alignment_hits
        :data:`ALIGNMENT_HIT_TABLE` — used to look up each novel
        protein's ``alignment_tier`` (reference / non_reference)
        and target accession for the ``aligned_to`` + ``accession``
        header tags.
    reference_gene_map
        ``protein_id → gene_symbol`` from
        :func:`read_reference_gene_map`. Used for both reference and
        novel records (novel records inherit the gene tag from the
        reference protein they aligned to, when known).
    output_path
        Destination for the combined FASTA.
    dedup_reference_against_self
        When True (default), drop duplicate sequences within the
        reference FASTA. Cartographer's pipeline does this — RefSeq
        contains many isoforms with identical translated sequences.

    Returns
    -------
    (n_written, stats)
        ``n_written`` is the total record count emitted. ``stats`` is
        a per-source breakdown: ``{"reference": N, "novel": M,
        "no_gene": K, "duplicate_dropped": D}``.
    """
    reference_fasta = Path(reference_fasta)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a lookup from novel protein_id → (best mmseqs target,
    # alignment_tier) by picking the lowest-evalue hit per query.
    novel_target: dict[str, tuple[str | None, str | None]] = {}
    if alignment_hits.num_rows > 0:
        # Sort by evalue ascending so the first row per query is best.
        sorted_hits = alignment_hits.sort_by([("query", "ascending"), ("evalue", "ascending")])
        seen: set[str] = set()
        for row in sorted_hits.to_pylist():
            q = row["query"]
            if q in seen:
                continue
            seen.add(q)
            target = row.get("target")
            tier = row.get("alignment_tier")
            novel_target[q] = (target, tier)

    stats = {
        "reference": 0,
        "novel": 0,
        "no_gene": 0,
        "duplicate_dropped": 0,
    }
    seen_hashes: set[str] = set()

    with output_path.open("w") as out:
        # Reference records first.
        for header, seq in _iter_fasta(reference_fasta):
            seq = seq.upper()
            protein_id = _header_id(header)
            if dedup_reference_against_self:
                h = _seq_hash(seq)
                if h in seen_hashes:
                    stats["duplicate_dropped"] += 1
                    continue
                seen_hashes.add(h)
            gene = reference_gene_map.get(protein_id)
            bracket_tags = [f"[accession={protein_id}]"]
            if gene:
                bracket_tags.insert(0, f"[gene={gene}]")
            else:
                stats["no_gene"] += 1
            header_line = (
                f">{protein_id} {' '.join(bracket_tags)} source=refseq\n"
            )
            out.write(header_line)
            _write_seq(out, seq)
            stats["reference"] += 1

        # Novel records, alignment-annotated.
        for row in filtered_novel.to_pylist():
            protein_id = row["protein_id"]
            seq = row["sequence"].upper()
            if dedup_reference_against_self:
                h = _seq_hash(seq)
                if h in seen_hashes:
                    stats["duplicate_dropped"] += 1
                    continue
                seen_hashes.add(h)
            target, tier = novel_target.get(protein_id, (None, None))
            bracket_tags: list[str] = []
            if target is not None:
                gene = reference_gene_map.get(target)
                if gene:
                    bracket_tags.append(f"[gene={gene}]")
                else:
                    stats["no_gene"] += 1
                bracket_tags.append(f"[accession={target}]")
                if tier:
                    bracket_tags.append(f"[aligned_to={tier}]")
            else:
                stats["no_gene"] += 1
            tag_str = (" " + " ".join(bracket_tags)) if bracket_tags else ""
            out.write(f">{protein_id}{tag_str} source=transcriptome\n")
            _write_seq(out, seq)
            stats["novel"] += 1

    return stats["reference"] + stats["novel"], stats


def _write_seq(fh, sequence: str, *, wrap: int = 60) -> None:
    for i in range(0, len(sequence), wrap):
        fh.write(sequence[i:i + wrap])
        fh.write("\n")


# ──────────────────────────────────────────────────────────────────────
# Stage 3 helper — competitive-target FASTA builder
# ──────────────────────────────────────────────────────────────────────


def write_competitive_target_fasta(
    *,
    reference_fasta: Path | str,
    swissprot_fasta: Path | str,
    output_path: Path | str,
) -> int:
    """Concatenate reference + SwissProt → one FASTA for mmseqs2 to
    search against as the *combined* target database.

    Used by the orchestrator's Stage 3: novel proteins are aligned
    against this combined FASTA via ``mmseqs easy-search``; the
    classifier downstream infers each hit's ``alignment_tier`` from
    the target accession's membership in the reference proteome
    (target in reference → ``"reference"``; otherwise →
    ``"non_reference"``).

    Records are emitted with their original headers preserved — no
    modification, no dedup across the two inputs (DBs may legitimately
    share sequences; the alignment-tier inference handles that).

    Returns the total record count written.
    """
    reference_fasta = Path(reference_fasta)
    swissprot_fasta = Path(swissprot_fasta)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w") as out:
        for src in (reference_fasta, swissprot_fasta):
            for header, seq in _iter_fasta(src):
                out.write(f">{header}\n")
                _write_seq(out, seq.upper())
                n += 1
    return n


# ──────────────────────────────────────────────────────────────────────
# Convenience: collect protein_ids from a FASTA for downstream
# alignment-hits → membership inference.
# ──────────────────────────────────────────────────────────────────────


def collect_protein_ids(fasta_path: Path | str) -> set[str]:
    """Return the set of first-token protein IDs from a FASTA's headers."""
    out: set[str] = set()
    for header, _ in _iter_fasta(Path(fasta_path)):
        pid = _header_id(header)
        if pid:
            out.add(pid)
    return out


def _tag_alignment_tier(
    alignment_hits: "pa.Table", reference_fasta: Path | str
) -> "pa.Table":
    """Set ``alignment_tier`` per hit by target membership in the
    reference proteome: ``target ∈ reference → "refseq"`` else
    ``"swissprot"``. Replaces any existing column.

    The single competitive mmseqs search can't distinguish which DB a
    target came from; accession membership recovers it.
    """
    import pyarrow as pa

    if alignment_hits.num_rows == 0:
        if "alignment_tier" in alignment_hits.column_names:
            return alignment_hits
        return alignment_hits.append_column(
            "alignment_tier", pa.array([], type=pa.string())
        )
    ref_ids = collect_protein_ids(reference_fasta)
    tier = pa.array(
        [
            "refseq" if str(t) in ref_ids else "swissprot"
            for t in alignment_hits.column("target").to_pylist()
        ],
        type=pa.string(),
    )
    if "alignment_tier" in alignment_hits.column_names:
        return alignment_hits.set_column(
            alignment_hits.column_names.index("alignment_tier"),
            "alignment_tier",
            tier,
        )
    return alignment_hits.append_column("alignment_tier", tier)


# ──────────────────────────────────────────────────────────────────────
# Orchestrator entry point
# ──────────────────────────────────────────────────────────────────────


def run_transcriptome_to_proteomics(*, args) -> int:  # args: argparse.Namespace
    """Chain every stage of the transcriptome→proteomics pipeline.

    Stage layout under ``<args.output_dir>/``:

      01_protein_counts        TPM-normalized counts (counts_tpm.parquet)
      02_novel_fasta           TPM + dedup-filtered novel.fasta
      03_alignment             mmseqs2 vs combined reference+swissprot
      04_combined_fasta        annotated combined.fasta (gene/source tags)
      05_predict_library       constellation massspec predict-library
      06_process_dia           constellation massspec process-dia
      07_gpf_search            search + (default) collision-filter
      08_classify_novel_peptides
                               constellation massspec classify-novel-peptides
      09_per_injection         per-mzML search against the filtered library
      10_quant_report          constellation massspec library-export

    Each stage owns its run dir + manifest.json + _SUCCESS. Top-level
    manifest.json at ``<output-dir>/manifest.json`` collates per-stage
    paths + SHA256s of external inputs. ``--resume`` honours per-stage
    _SUCCESS short-circuits.
    """
    import json
    import shutil
    import socket
    import sys
    import time
    from datetime import datetime, timezone

    import pyarrow.parquet as pq

    from constellation import __version__ as constellation_version
    from constellation.catalog.uniprot import fetch_swissprot
    from constellation.core.io.schemas import read_mmseqs_tab
    from constellation.massspec.search import (
        apply_collision_filter,
        filter_elib_by_losers,
    )
    from constellation.massspec.search.encyclopedia import (
        run_library_export,
        run_library_search,
        run_predict_library,
        run_process_dia,
    )
    from constellation.massspec.search.encyclopedia.library_search import (
        find_search_elib,
    )
    from constellation.sequencing.quant.protein_counts import (
        build_tpm_matrix,
        read_protein_counts_tab,
        render_tpm_matrix_tsv,
        tpm_normalize,
    )
    from constellation.thirdparty.mmseqs2_run import run_mmseqs_search

    # ── Path resolution + env validation ───────────────────────────────
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protein_counts = Path(args.protein_counts).resolve()
    proteins_fasta = Path(args.proteins_fasta).resolve()
    reference_fasta = Path(args.reference_fasta).resolve()
    reference_annotation = Path(args.reference_annotation).resolve()
    gpf_files = [Path(p).resolve() for p in args.gpf]
    injection_files = [Path(p).resolve() for p in args.injections]

    progress = not args.no_progress

    def _log(msg: str) -> None:
        if progress:
            print(f"[pipeline] {msg}", file=sys.stderr)

    success_top = output_dir / "_SUCCESS"
    if success_top.exists() and not args.resume:
        print(
            f"error: --output-dir already complete (top-level _SUCCESS exists). "
            f"Pass --resume to short-circuit, or delete {output_dir} to "
            f"start fresh.",
            file=sys.stderr,
        )
        return 1
    if success_top.exists() and args.resume:
        print(f"already complete: {output_dir}")
        return 0

    # ── Resolve SwissProt (lazy fetch if not supplied) ─────────────────
    if args.swissprot_fasta is not None:
        swissprot_fasta = Path(args.swissprot_fasta).resolve()
        swissprot_release: str | None = args.swissprot_release
    else:
        _log("fetching SwissProt FASTA via catalog (~/.constellation/references/swissprot/)")
        sp_handle = fetch_swissprot(release=args.swissprot_release)
        swissprot_fasta = sp_handle.fasta_path
        swissprot_release = sp_handle.release

    t_start = time.monotonic()
    stage_manifests: dict[str, dict] = {}

    # ── Stage 1: read + TPM ────────────────────────────────────────────
    stage_dir = output_dir / "01_protein_counts"
    if not _stage_done(stage_dir, args.resume):
        _log("Stage 1: reading protein counts + TPM-normalizing")
        stage_dir.mkdir(parents=True, exist_ok=True)
        counts = read_protein_counts_tab(protein_counts)
        counts_tpm = tpm_normalize(counts, min_sequence_length=args.min_sequence_length)
        pq.write_table(counts_tpm, stage_dir / "counts_tpm.parquet")
        # Human-facing wide summary alongside the canonical long parquet:
        # one row per (protein_id, sequence), per-sample counts + avg_tpm.
        (stage_dir / "counts_tpm_wide.tsv").write_text(
            render_tpm_matrix_tsv(build_tpm_matrix(counts_tpm))
        )
        _write_stage_manifest(
            stage_dir,
            subcommand="01_protein_counts",
            params={
                "min_sequence_length": args.min_sequence_length,
            },
            counts={"n_rows": counts_tpm.num_rows},
        )
        _touch_success(stage_dir)
    stage_manifests["01_protein_counts"] = _read_stage_manifest(stage_dir)
    counts_tpm = pq.read_table(stage_dir / "counts_tpm.parquet")

    # ── Stage 2: novel FASTA ───────────────────────────────────────────
    stage_dir = output_dir / "02_novel_fasta"
    novel_path = stage_dir / "novel.fasta"
    if not _stage_done(stage_dir, args.resume):
        _log(f"Stage 2: filter avg_tpm >= {args.min_avg_tpm} + dedup vs reference → novel.fasta")
        stage_dir.mkdir(parents=True, exist_ok=True)
        novel_table, n_novel = filter_and_write_novel_fasta(
            counts_tpm=counts_tpm,
            proteins_fasta=proteins_fasta,
            reference_fasta=reference_fasta,
            output_path=novel_path,
            min_avg_tpm=args.min_avg_tpm,
        )
        pq.write_table(novel_table, stage_dir / "novel.parquet")
        _write_stage_manifest(
            stage_dir,
            subcommand="02_novel_fasta",
            params={"min_avg_tpm": args.min_avg_tpm},
            counts={"n_novel": n_novel},
        )
        _touch_success(stage_dir)
    stage_manifests["02_novel_fasta"] = _read_stage_manifest(stage_dir)
    novel_table = pq.read_table(stage_dir / "novel.parquet")

    # ── Stage 3: competitive mmseqs2 alignment ─────────────────────────
    stage_dir = output_dir / "03_alignment"
    alignments_tab = stage_dir / "alignments.tab"
    if not _stage_done(stage_dir, args.resume):
        _log(f"Stage 3: mmseqs2 easy-search ({args.mmseqs_threads} threads)")
        stage_dir.mkdir(parents=True, exist_ok=True)
        scratch = stage_dir / ".scratch"
        target_fasta = stage_dir / "_target.fasta"
        write_competitive_target_fasta(
            reference_fasta=reference_fasta,
            swissprot_fasta=swissprot_fasta,
            output_path=target_fasta,
        )
        mmseqs_result = run_mmseqs_search(
            query_fasta=novel_path,
            target_fasta=target_fasta,
            output_tab=alignments_tab,
            log_dir=stage_dir / "logs",
            evalue=args.evalue_threshold,
            threads=args.mmseqs_threads,
            scratch_dir=scratch,
            stream_to_stderr=progress,
        )
        # Materialise the parsed AlignmentHits as a ParquetDir bundle
        # for downstream classifier consumption.
        alignment_hits = read_mmseqs_tab(alignments_tab)
        # Tag each hit's source tier. The single competitive mmseqs search
        # against (reference ⊕ swissprot) doesn't distinguish which DB a
        # target came from, so derive it from accession membership:
        # target ∈ reference proteome → "refseq", else "swissprot". This
        # restores the source column cartographer's two-tier search emitted
        # (feeds the combined.fasta [aligned_to=...] header tag + the
        # classifier's tier inference).
        alignment_hits = _tag_alignment_tier(alignment_hits, reference_fasta)
        ah_dir = stage_dir / "alignment_hits"
        ah_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(alignment_hits, ah_dir / "alignment_hits.parquet")
        # Re-write the .tab with the 9th source column so the human-facing
        # alignment file carries the tier too (cartographer .tab parity).
        import pyarrow.csv as _pa_csv
        _pa_csv.write_csv(
            alignment_hits,
            alignments_tab,
            write_options=_pa_csv.WriteOptions(delimiter="\t"),
        )
        # Drop the combined target FASTA (recreatable from reference + swissprot).
        target_fasta.unlink(missing_ok=True)
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        _write_stage_manifest(
            stage_dir,
            subcommand="03_alignment",
            params={
                "evalue": args.evalue_threshold,
                "threads": args.mmseqs_threads,
            },
            counts={"n_hits": alignment_hits.num_rows},
            runtime={
                "elapsed_seconds": mmseqs_result.elapsed_seconds,
                "returncode": mmseqs_result.returncode,
                "mmseqs_version": mmseqs_result.mmseqs_version,
            },
        )
        _touch_success(stage_dir)
    stage_manifests["03_alignment"] = _read_stage_manifest(stage_dir)
    alignment_hits = pq.read_table(
        stage_dir / "alignment_hits" / "alignment_hits.parquet"
    )

    # ── Stage 4: combined FASTA ────────────────────────────────────────
    stage_dir = output_dir / "04_combined_fasta"
    combined_path = stage_dir / "combined.fasta"
    if not _stage_done(stage_dir, args.resume):
        _log("Stage 4: alignment filter + combined.fasta with annotated headers")
        stage_dir.mkdir(parents=True, exist_ok=True)
        filtered_novel = apply_alignment_filter(
            novel_table=novel_table,
            alignment_hits=alignment_hits,
            evalue_threshold=args.evalue_threshold,
        )
        gene_map = read_reference_gene_map(reference_annotation)
        n_written, fasta_stats = build_combined_fasta(
            reference_fasta=reference_fasta,
            filtered_novel=filtered_novel,
            alignment_hits=alignment_hits,
            reference_gene_map=gene_map,
            output_path=combined_path,
        )
        (stage_dir / "gene_map.json").write_text(
            json.dumps(gene_map, indent=2, sort_keys=True) + "\n"
        )
        _write_stage_manifest(
            stage_dir,
            subcommand="04_combined_fasta",
            params={
                "evalue_threshold": args.evalue_threshold,
                "gene_map_source": str(reference_annotation),
                "gene_map_size": len(gene_map),
            },
            counts={"n_written": n_written, **fasta_stats},
        )
        _touch_success(stage_dir)
    stage_manifests["04_combined_fasta"] = _read_stage_manifest(stage_dir)

    # ── Stage 5: predict-library ───────────────────────────────────────
    stage_dir = output_dir / "05_predict_library"
    combined_dlib = stage_dir / "library.dlib"
    if not _stage_done(stage_dir, args.resume):
        _log("Stage 5: predict-library (FASTA → .dlib)")
        stage_dir.mkdir(parents=True, exist_ok=True)
        ptms = _build_ptm_map(args)
        result = run_predict_library(
            fasta=combined_path,
            output_dlib=combined_dlib,
            output_dir=stage_dir,
            ptms=ptms,
            jvm_heap_max=args.jvm_heap_max,
            jvm_heap_min=args.jvm_heap_min,
            jvm_tmpdir=args.jvm_tmpdir,
            extra_args=_passthrough_args(args.encyclopedia_arg),
            stream_to_stderr=progress,
        )
        _write_stage_manifest(
            stage_dir,
            subcommand="05_predict_library",
            params={"ptms": ptms},
            runtime={
                "elapsed_seconds": result.elapsed_seconds,
                "returncode": result.returncode,
                "java_version": result.java_version,
            },
        )
        _touch_success(stage_dir)
    stage_manifests["05_predict_library"] = _read_stage_manifest(stage_dir)

    # ── Stage 6: process-dia ───────────────────────────────────────────
    stage_dir = output_dir / "06_process_dia"
    combined_dia = stage_dir / "combined.dia"
    if not _stage_done(stage_dir, args.resume):
        _log(f"Stage 6: process-dia ({len(gpf_files)} GPF input{'s' if len(gpf_files) != 1 else ''})")
        stage_dir.mkdir(parents=True, exist_ok=True)
        result = run_process_dia(
            inputs=gpf_files,
            output_dia=combined_dia,
            output_dir=stage_dir,
            jvm_heap_max=args.jvm_heap_max,
            jvm_heap_min=args.jvm_heap_min,
            jvm_tmpdir=args.jvm_tmpdir,
            extra_args=_passthrough_args(args.encyclopedia_arg),
            stream_to_stderr=progress,
        )
        _write_stage_manifest(
            stage_dir,
            subcommand="06_process_dia",
            params={"n_gpf_inputs": len(gpf_files)},
            runtime={
                "elapsed_seconds": result.elapsed_seconds,
                "returncode": result.returncode,
            },
        )
        _touch_success(stage_dir)
    stage_manifests["06_process_dia"] = _read_stage_manifest(stage_dir)

    # ── Stage 7: GPF search + collision filter ─────────────────────────
    # Transparent elib naming:
    #   _raw/combined.raw.elib       the raw EncyclopeDIA search output
    #   combined.filtered.elib       collision-filtered primary (default)
    #   combined.elib                primary when --no-collision-filter
    stage_dir = output_dir / "07_gpf_search"
    raw_dir = stage_dir / "_raw"
    raw_elib_canonical = raw_dir / "combined.raw.elib"
    gpf_elib_primary = stage_dir / (
        "combined.elib" if args.no_collision_filter else "combined.filtered.elib"
    )
    if not _stage_done(stage_dir, args.resume):
        # Fine-grained resume: a prior run already produced the filtered
        # primary elib, so the (very slow) EncyclopeDIA search + the
        # collision filter are done — only the optional auto-ingest or the
        # manifest/_SUCCESS were left. Pick up from the existing elib
        # instead of re-running the multi-hour search.
        if args.resume and gpf_elib_primary.is_file():
            _log(f"Stage 7: {gpf_elib_primary.name} present — skipping search "
                 "+ collision filter, resuming from existing elib")
            actual_raw_elib = (
                raw_elib_canonical if raw_elib_canonical.is_file()
                else find_search_elib(combined_dia, cwd=raw_dir)
            )
            collision_ran = not args.no_collision_filter
            meta_path = stage_dir / "collision_metadata.json"
            n_losers = 0
            if meta_path.is_file():
                try:
                    n_losers = int(
                        json.loads(meta_path.read_text()).get("n_losers", 0)
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    n_losers = 0
            search_runtime: dict = {"resumed_from_elib": True}
        else:
            _log("Stage 7: GPF library search + collision filter")
            raw_dir.mkdir(parents=True, exist_ok=True)
            result = run_library_search(
                input_file=combined_dia,
                library=combined_dlib,
                fasta=combined_path,
                output_dir=raw_dir,
                jvm_heap_max=args.jvm_heap_max,
                jvm_heap_min=args.jvm_heap_min,
                jvm_tmpdir=args.jvm_tmpdir,
                fragment_tolerance_ppm=args.fragment_tolerance_ppm,
                precursor_tolerance_ppm=args.precursor_tolerance_ppm,
                percolator_version=args.percolator_version,
                percolator_threshold=args.percolator_threshold,
                extra_args=_passthrough_args(args.encyclopedia_arg),
                stream_to_stderr=progress,
            )
            found_raw = find_search_elib(combined_dia, cwd=raw_dir)
            if found_raw is None:
                raise FileNotFoundError(
                    f"GPF search returned 0 but no .elib found under {raw_dir}"
                )
            # EncyclopeDIA names its output <stem>.elib and may write it
            # next to the input .dia (06_process_dia/) rather than into
            # _raw/. Move it to the canonical raw path so the stage dirs
            # stay clean and the raw vs. filtered distinction is explicit.
            if found_raw.resolve() != raw_elib_canonical.resolve():
                shutil.move(str(found_raw), str(raw_elib_canonical))
            actual_raw_elib = raw_elib_canonical
            # Collision filter (default-on).
            if not args.no_collision_filter:
                losers, collision_meta = apply_collision_filter(
                    elib_path=actual_raw_elib,
                    dia_path=combined_dia,
                    rt_threshold_s=args.collision_rt_threshold_s,
                    frag_ppm_tol=args.collision_frag_ppm_tol,
                    min_shared_ions=args.collision_min_shared_ions,
                    return_metadata=True,
                )
                filter_elib_by_losers(actual_raw_elib, losers, gpf_elib_primary)
                (stage_dir / "collision_metadata.json").write_text(
                    json.dumps(
                        {**collision_meta, "n_losers": len(losers)},
                        indent=2,
                    ) + "\n"
                )
                collision_ran = True
            else:
                shutil.copyfile(actual_raw_elib, gpf_elib_primary)
                losers = set()
                collision_ran = False
            n_losers = len(losers)
            search_runtime = {
                "elapsed_seconds": result.elapsed_seconds,
                "returncode": result.returncode,
            }

        # Auto-ingest (no-op under --no-ingest).
        _maybe_auto_ingest_elib(
            gpf_elib_primary, stage_dir, args.no_ingest,
        )
        if collision_ran and actual_raw_elib is not None and actual_raw_elib.is_file():
            _maybe_auto_ingest_elib(
                actual_raw_elib, raw_dir, args.no_ingest,
            )
        _write_stage_manifest(
            stage_dir,
            subcommand="07_gpf_search",
            params={
                "collision_filter": not args.no_collision_filter,
                "fragment_tolerance_ppm": args.fragment_tolerance_ppm,
                "precursor_tolerance_ppm": args.precursor_tolerance_ppm,
            },
            counts={"n_losers": n_losers},
            runtime=search_runtime,
        )
        _touch_success(stage_dir)
    stage_manifests["07_gpf_search"] = _read_stage_manifest(stage_dir)

    # ── Stage 8: classify novel peptides ───────────────────────────────
    stage_dir = output_dir / "08_classify_novel_peptides"
    if not _stage_done(stage_dir, args.resume):
        _log("Stage 8: classify-novel-peptides")
        stage_dir.mkdir(parents=True, exist_ok=True)
        from constellation.massspec.search import (
            build_gene_map_from_fasta_headers,
            classify_novel_peptides,
            read_fasta_proteins,
            save_novel_peptides,
        )
        # Detected peptides from the FILTERED GPF .elib.
        import sqlite3
        con = sqlite3.connect(str(gpf_elib_primary))
        try:
            rows = con.execute(
                "SELECT DISTINCT PeptideSeq, PeptideModSeq "
                "FROM entries WHERE PeptideSeq IS NOT NULL"
            ).fetchall()
        finally:
            con.close()
        import pyarrow as pa
        detected_peptides = pa.table(
            {
                "peptide_sequence": pa.array(
                    [r[0] for r in rows], type=pa.string()
                ),
                "modified_sequence": pa.array(
                    [r[1] for r in rows], type=pa.string()
                ),
            }
        )
        ref_proteins = read_fasta_proteins(reference_fasta)
        # Novel proteins = the combined.fasta minus reference. Simpler:
        # the source proteins.fasta carries every novel protein candidate.
        novel_proteins = read_fasta_proteins(proteins_fasta)
        # Gene tags live on the Stage-4 combined.fasta headers (annotated
        # from the reference GFF3/GBFF). The raw reference + demux FASTAs
        # carry no gene= tags, so the map must come from combined.fasta —
        # building it from the raw inputs yields an empty gene column.
        gene_map_from_fastas = build_gene_map_from_fasta_headers([combined_path])
        result = classify_novel_peptides(
            detected_peptides=detected_peptides,
            alignments=alignment_hits,
            reference_proteins=ref_proteins,
            novel_proteins=novel_proteins,
            gene_map=gene_map_from_fastas,
        )
        save_novel_peptides(
            result,
            stage_dir,
            metadata={
                "constellation_version": constellation_version,
                "reference_fasta": str(reference_fasta),
                "novel_fasta": str(proteins_fasta),
            },
        )
        cls_counts: dict[str, int] = {}
        for cls in result.column("classification").to_pylist():
            cls_counts[cls] = cls_counts.get(cls, 0) + 1
        _write_stage_manifest(
            stage_dir,
            subcommand="08_classify_novel_peptides",
            counts={
                "n_classified": result.num_rows,
                "classifications": cls_counts,
            },
        )
        _touch_success(stage_dir)
    stage_manifests["08_classify_novel_peptides"] = _read_stage_manifest(stage_dir)

    # ── Stage 9: per-injection search (against the filtered library) ───
    stage_dir = output_dir / "09_per_injection"
    if not _stage_done(stage_dir, args.resume):
        _log(f"Stage 9: per-injection searches ({len(injection_files)} mzML, "
             f"{args.injection_threads} thread{'s' if args.injection_threads != 1 else ''})")
        stage_dir.mkdir(parents=True, exist_ok=True)
        _run_per_injection_searches(
            injection_files=injection_files,
            library_elib=gpf_elib_primary,
            fasta=combined_path,
            output_root=stage_dir,
            args=args,
            progress=progress,
        )
        _write_stage_manifest(
            stage_dir,
            subcommand="09_per_injection",
            params={
                # No per-injection collision filter — the searched library
                # (gpf_elib_primary) is already collision-filtered, so the
                # filtering propagates transitively.
                "library_collision_filtered": not args.no_collision_filter,
                "injection_threads": args.injection_threads,
                "n_injections": len(injection_files),
            },
        )
        _touch_success(stage_dir)
    stage_manifests["09_per_injection"] = _read_stage_manifest(stage_dir)

    # ── Stage 10: library-export quant report ──────────────────────────
    stage_dir = output_dir / "10_quant_report"
    quant_report_elib = stage_dir / "quant_report.elib"
    if not _stage_done(stage_dir, args.resume):
        _log("Stage 10: library-export → quant_report.elib")
        stage_dir.mkdir(parents=True, exist_ok=True)
        # Library-export consumes the per-injection .elibs (searched
        # against the already-filtered library). Collect them under a
        # flat dir for the jar's -i flag.
        export_input_dir = stage_dir / "_per_injection_elibs"
        export_input_dir.mkdir(parents=True, exist_ok=True)
        for inj_dir in sorted(
            (output_dir / "09_per_injection").rglob("*.elib")
        ):
            # Use the parent dir name so collisions on injection stems
            # don't clobber.
            target = export_input_dir / f"{inj_dir.parent.name}.elib"
            if not target.exists():
                target.symlink_to(inj_dir) if hasattr(target, "symlink_to") else shutil.copy(
                    inj_dir, target
                )
        result = run_library_export(
            search_dir=export_input_dir,
            library=gpf_elib_primary,
            output_elib=quant_report_elib,
            output_dir=stage_dir,
            # EncyclopeDIA 6.5.15 -libexport requires -f <fasta>.
            fasta=combined_path,
            align=True,
            jvm_heap_max=args.jvm_heap_max,
            jvm_heap_min=args.jvm_heap_min,
            jvm_tmpdir=args.jvm_tmpdir,
            extra_args=_passthrough_args(args.encyclopedia_arg),
            stream_to_stderr=progress,
        )
        _maybe_auto_ingest_elib(
            quant_report_elib, stage_dir, args.no_ingest,
        )
        _write_stage_manifest(
            stage_dir,
            subcommand="10_quant_report",
            runtime={
                "elapsed_seconds": result.elapsed_seconds,
                "returncode": result.returncode,
            },
        )
        _touch_success(stage_dir)
    stage_manifests["10_quant_report"] = _read_stage_manifest(stage_dir)

    # ── Top-level manifest + _SUCCESS ──────────────────────────────────
    elapsed = time.monotonic() - t_start
    top_manifest = {
        "constellation_version": constellation_version,
        "subcommand": "pipeline transcriptome-to-proteomics",
        "argv": sys.argv,
        "inputs": {
            "protein_counts": _input_meta(protein_counts),
            "proteins_fasta": _input_meta(proteins_fasta),
            "reference_fasta": _input_meta(reference_fasta),
            "reference_annotation": _input_meta(reference_annotation),
            "gpf": [_input_meta(p) for p in gpf_files],
            "injections": [_input_meta(p) for p in injection_files],
            "swissprot_fasta": _input_meta(swissprot_fasta),
            "swissprot_release": swissprot_release,
        },
        "stages": {
            name: {
                "path": str((output_dir / name).relative_to(output_dir)),
                "manifest_relpath": f"{name}/manifest.json",
            }
            for name in stage_manifests
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "host": socket.gethostname(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(top_manifest, indent=2, default=str) + "\n"
    )
    success_top.write_bytes(b"")

    _log(f"pipeline done: {elapsed:.1f}s total → {output_dir}")
    return 0


# ──────────────────────────────────────────────────────────────────────
# Orchestrator helpers (file-private; not part of the public API)
# ──────────────────────────────────────────────────────────────────────


def _stage_done(stage_dir: Path, resume: bool) -> bool:
    """True iff the stage's _SUCCESS already exists AND --resume is set."""
    success = stage_dir / "_SUCCESS"
    if success.exists() and not resume:
        raise RuntimeError(
            f"{stage_dir} is already complete (touch _SUCCESS exists). "
            f"Pass --resume to skip it, or delete {stage_dir} to re-run."
        )
    return success.exists() and resume


def _touch_success(stage_dir: Path) -> None:
    (stage_dir / "_SUCCESS").write_bytes(b"")


def _write_stage_manifest(
    stage_dir: Path,
    *,
    subcommand: str,
    params: dict | None = None,
    counts: dict | None = None,
    runtime: dict | None = None,
) -> None:
    import json
    import sys as _sys
    from datetime import datetime, timezone

    from constellation import __version__ as constellation_version

    manifest: dict = {
        "constellation_version": constellation_version,
        "subcommand": subcommand,
        "wrote_at": datetime.now(timezone.utc).isoformat(),
        "argv": _sys.argv,
    }
    if params is not None:
        manifest["params"] = params
    if counts is not None:
        manifest["counts"] = counts
    if runtime is not None:
        manifest["runtime"] = runtime
    (stage_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )


def _read_stage_manifest(stage_dir: Path) -> dict:
    import json
    p = stage_dir / "manifest.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def _input_meta(path: Path) -> dict:
    """Streaming SHA256 + path entry for top-level manifest input records."""
    from constellation.massspec.search.encyclopedia import sha256_file

    if not path.is_file():
        # Directory inputs (e.g. protein_counts pointed at the demux
        # output dir) — record path only.
        return {"path": str(path), "is_dir": path.is_dir()}
    return {"path": str(path), "sha256": sha256_file(path)}


def _build_ptm_map(args) -> dict[str, str]:
    return {
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


def _passthrough_args(arg_list: list[str]) -> list[str]:
    """``--encyclopedia-arg FLAG=VALUE`` → flat argv. Mirrors the
    massspec CLI's helper."""
    from constellation.massspec.search.encyclopedia import (
        encyclopedia_passthrough_args,
    )
    return encyclopedia_passthrough_args(arg_list)


def _maybe_auto_ingest_elib(
    elib_path: Path, run_dir: Path, no_ingest: bool
) -> None:
    """Auto-ingest a .elib into ParquetDir bundles next to it.

    Mirrors the pattern in `_cmd_massspec_search` / `_cmd_massspec_library_export`.
    """
    if no_ingest:
        return
    from constellation.massspec.io.encyclopedia import read_encyclopedia
    from constellation.massspec.library import save_library
    from constellation.massspec.quant import save_quant
    from constellation.massspec.search import save_search

    ingest = read_encyclopedia(elib_path)
    save_library(ingest.library, run_dir / "library_pqdir", format="parquet_dir")
    if ingest.quant is not None:
        save_quant(ingest.quant, run_dir / "quant_pqdir", format="parquet_dir")
    if ingest.search is not None:
        save_search(ingest.search, run_dir / "search_pqdir", format="parquet_dir")


def _run_per_injection_searches(
    *,
    injection_files: list[Path],
    library_elib: Path,
    fasta: Path,
    output_root: Path,
    args,
    progress: bool,
) -> None:
    """Search each injection mzML/raw against the (already collision-
    filtered) GPF library and emit one ``<stem>.elib`` per injection.

    No per-injection collision filter: the search library is the
    Stage-7 collision-filtered ``combined.filtered.elib``, so the
    colliding identifications are already absent from the library and
    cannot be reported in the injection searches. The filtering
    propagates transitively to Stage 10's quant report. (Filtering only
    the library also matches cartographer's validated workflow.)

    Parallel via :class:`concurrent.futures.ThreadPoolExecutor` per the
    plan's footgun #5 (process-level fan-out blows host RAM at
    --jvm-heap per worker).
    """
    from concurrent.futures import ThreadPoolExecutor

    def _run_one(mzml: Path) -> None:
        sample_dir = _sanitise_dirname(mzml.parent.name)
        run_dir = output_root / sample_dir / mzml.stem
        if (run_dir / "_SUCCESS").exists() and args.resume:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        from constellation.massspec.search.encyclopedia import (
            run_library_search,
        )
        from constellation.massspec.search.encyclopedia.library_search import (
            find_search_elib,
        )
        primary_elib = run_dir / f"{mzml.stem}.elib"
        run_library_search(
            input_file=mzml,
            library=library_elib,
            fasta=fasta,
            output_dir=run_dir,
            jvm_heap_max=args.jvm_heap_max,
            jvm_heap_min=args.jvm_heap_min,
            jvm_tmpdir=args.jvm_tmpdir,
            fragment_tolerance_ppm=args.fragment_tolerance_ppm,
            precursor_tolerance_ppm=args.precursor_tolerance_ppm,
            percolator_version=args.percolator_version,
            percolator_threshold=args.percolator_threshold,
            extra_args=_passthrough_args(args.encyclopedia_arg),
            stream_to_stderr=progress,
        )
        found = find_search_elib(mzml, cwd=run_dir)
        if found is None:
            raise FileNotFoundError(
                f"per-injection search returned 0 but no .elib for {mzml}"
            )
        if found.resolve() != primary_elib.resolve():
            import shutil as _shutil
            _shutil.move(str(found), str(primary_elib))
        _maybe_auto_ingest_elib(primary_elib, run_dir, args.no_ingest)
        _touch_success(run_dir)

    if args.injection_threads <= 1:
        for mzml in injection_files:
            _run_one(mzml)
    else:
        with ThreadPoolExecutor(max_workers=args.injection_threads) as ex:
            for _ in ex.map(_run_one, injection_files):
                pass


def _sanitise_dirname(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"_", "-", "."} else "_" for c in name)
    return safe or "sample"


__all__ = [
    "apply_alignment_filter",
    "build_combined_fasta",
    "collect_protein_ids",
    "filter_and_write_novel_fasta",
    "read_reference_gene_map",
    "run_transcriptome_to_proteomics",
    "write_competitive_target_fasta",
]
