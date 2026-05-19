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


__all__ = [
    "apply_alignment_filter",
    "build_combined_fasta",
    "collect_protein_ids",
    "filter_and_write_novel_fasta",
    "read_reference_gene_map",
    "write_competitive_target_fasta",
]
