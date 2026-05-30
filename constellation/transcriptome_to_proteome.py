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
    ``constellation transcriptome-to-proteome``.

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


def deduplicate_fasta(
    input_fasta: Path | str,
    output_fasta: Path | str,
) -> int:
    """Write a deduplicated copy of a FASTA file (first occurrence per
    unique protein SEQUENCE wins).

    Mirrors :func:`cartographer.data.fasta.deduplicate_fasta`. Sequences
    that appear under multiple accessions (RefSeq has many isoform pairs
    where ``NP_*`` and ``XP_*`` translate to identical strings) collapse
    to the first accession encountered. Headers are preserved verbatim
    for the winning accession.

    This is the cartographer-style upstream dedup that ensures mmseqs2
    alignment targets and :func:`build_combined_fasta` consume the same
    accession set — alignment hits cannot reference an accession dropped
    later in combined.fasta, which is what causes gene-map gaps in the
    classifier.

    Parameters
    ----------
    input_fasta
        Source FASTA (may be gzipped).
    output_fasta
        Destination for the deduplicated FASTA.

    Returns
    -------
    int
        Number of records written (= unique-sequence count).
    """
    input_fasta = Path(input_fasta)
    output_fasta = Path(output_fasta)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_written = 0
    with output_fasta.open("w") as out:
        for header, seq in _iter_fasta(input_fasta):
            up = seq.upper()
            if up in seen:
                continue
            seen.add(up)
            out.write(f">{header}\n")
            _write_seq(out, up)
            n_written += 1
    return n_written


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — filter novel proteins by avg TPM + dedup vs reference
# ──────────────────────────────────────────────────────────────────────


def filter_and_write_novel_fasta(
    *,
    counts_tpm: pa.Table,
    reference_fasta: Path | str,
    output_path: Path | str,
    min_avg_tpm: float = 1.0,
    protein_fasta: Path | str | None = None,
) -> tuple[pa.Table, int]:
    """Filter the transcriptome ORFs to the proteins worth searching:
    those above the TPM threshold AND not in the reference proteome.

    Sequences come from ``counts_tpm.column("sequence")`` by default.
    The optional ``protein_fasta`` argument overrides this with an ORF
    FASTA produced by a different upstream pipeline (any path-based
    external caller that doesn't fan its sequences through the demux
    counts table).

    Parameters
    ----------
    counts_tpm
        :data:`PROTEIN_COUNTS_LONG_SCHEMA`-shaped (with the ``tpm``
        column added by :func:`tpm_normalize`). Average TPM across
        samples is computed per ``protein_id``; rows below
        ``min_avg_tpm`` drop. The ``sequence`` column provides the
        ORF sequences when ``protein_fasta`` is None.
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
    protein_fasta
        Optional override produced by a different upstream pipeline.
        When supplied, sequences come from this FASTA instead of
        ``counts_tpm.column("sequence")``. Default ``None`` — derive
        from ``counts_tpm``.

    Returns
    -------
    (novel_table, n_written)
        ``novel_table`` is a 3-column Arrow table:
        ``(protein_id, sequence, avg_tpm)``. ``n_written`` is the
        number of FASTA records emitted.
    """
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

    # Source sequences: from ``counts_tpm`` (default) OR from the
    # ``protein_fasta`` external override. Either way, restrict to
    # ``passing_ids``.
    source_records: dict[str, str] = {}
    if protein_fasta is None:
        # Long-table form: one row per (protein, sample). Sequences
        # repeat across samples for the same protein — take the first
        # non-null occurrence per passing protein_id.
        pid_col = counts_tpm.column("protein_id").to_pylist()
        seq_col = counts_tpm.column("sequence").to_pylist()
        for pid, seq in zip(pid_col, seq_col):
            if pid in passing_ids and pid not in source_records and seq:
                source_records[pid] = seq.upper()
    else:
        protein_fasta = Path(protein_fasta)
        for header, seq in _iter_fasta(protein_fasta):
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


def read_reference_gene_map(annotation_path: Path | str) -> dict[str, str]:
    """Parse a reference annotation → ``protein_id → gene_symbol``.

    Thin path-based dispatcher for callers that have a raw annotation
    path on disk and don't want to construct a
    :class:`~constellation.sequencing.reference.Reference` object. The
    orchestrator itself uses ``Reference.gene_map()`` directly; this
    function is retained for external / notebook callers.

    Accepts three input shapes — pick whichever your upstream provides:

      * **Constellation parquet bundle** (preferred): a directory
        containing ``features.parquet`` + ``manifest.json`` — the
        canonical form produced by ``constellation reference fetch``.
      * ``.gff`` / ``.gff3`` (+ ``.gz``) → legacy GFF3 parser.
      * ``.gbff`` / ``.gb`` / ``.genbank`` (+ ``.gz``) → legacy GBFF
        parser.

    Returns
    -------
    dict[str, str]
        ``protein_id → gene_symbol`` mapping.
    """
    from constellation.sequencing.reference.installed import (
        _gene_map_from_gbff,
        _gene_map_from_gff3,
        _gene_map_from_parquet,
    )

    p = Path(annotation_path)
    if p.is_dir() and (p / "features.parquet").is_file():
        return _gene_map_from_parquet(p)
    suffixes = [s.lower() for s in p.suffixes]
    if any(s in {".gff", ".gff3"} for s in suffixes):
        return _gene_map_from_gff3(p)
    if any(s in {".gbff", ".gb", ".genbank"} for s in suffixes):
        return _gene_map_from_gbff(p)
    raise ValueError(
        f"unsupported annotation path {p!r}: expected a Constellation "
        f"parquet bundle (directory with features.parquet + manifest.json), "
        f"a .gff / .gff3 file, or a .gbff / .gb / .genbank file "
        f"(optionally .gz)"
    )


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
    transcriptome_seq_hashes: set[str] | None = None,
) -> tuple[int, dict[str, int]]:
    """Concatenate reference + filtered-novel into a single search FASTA
    with annotated headers.

    Every record — reference AND novel, with or without complete
    annotation — emits the same 5-bracket-tag shape, so downstream
    consumers (EncyclopeDIA, mass-spec search engines, parsers) can
    rely on a stable positional/token structure regardless of which
    upstream values were populated::

        >protein_id [gene=X|unknown] [accession=X|unknown] [aligned_to=refseq|swissprot|none] [in_transcriptome=true|false|unknown] source=refseq|transcriptome

    Where:

    - ``gene`` — gene symbol when ``gene_map.get(...)`` resolves;
      literal ``unknown`` otherwise. SwissProt entries that ship
      without a ``GN=`` tag land in this branch.
    - ``accession`` — for reference records, the protein's own
      first-token accession; for novel records, the alignment target's
      accession. Literal ``unknown`` for novel ORFs with no mmseqs hit.
    - ``aligned_to`` — for reference records, always ``none``. For
      novel records, the tier of the best alignment hit
      (``refseq`` / ``swissprot``); ``none`` when the alignment row
      carried no tier annotation OR there was no hit.
    - ``in_transcriptome`` — for novel records, always ``true`` (they
      came from the transcriptome by construction). For reference
      records, ``true`` iff the protein's sequence SHA (uppercase,
      via :func:`_seq_hash`) appears in
      ``transcriptome_seq_hashes``; ``false`` otherwise. When the
      caller doesn't supply ``transcriptome_seq_hashes`` (i.e.
      passes ``None``), reference records emit ``unknown`` so
      external callers without the set still see a uniform tag.
    - ``source`` — ``refseq`` for reference records, ``transcriptome``
      for novel records.

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
        protein's ``aligned_to`` (refseq / swissprot) and target
        accession for the ``aligned_to`` + ``accession`` header tags.
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
    transcriptome_seq_hashes
        Optional ``set[str]`` of SHA hashes (via :func:`_seq_hash`,
        uppercase) of every protein sequence the upstream
        transcriptome quant table observed. When supplied, each
        reference record's sequence hash is checked against this set
        to populate the ``[in_transcriptome=true|false]`` header tag.
        When ``None`` (default), reference records emit
        ``[in_transcriptome=unknown]``. Novel records ignore this
        parameter — they always emit ``[in_transcriptome=true]``.

    Returns
    -------
    (n_written, stats)
        ``n_written`` is the total record count emitted. ``stats`` is
        a per-source breakdown, with the no-gene counter split per
        tier so a build can validate which population is missing the
        ``[gene=...]`` lookup::

            {
                "reference": N,
                "novel": M,
                "no_gene_refseq": K1,
                "no_gene_novel_refseq_aligned": K2,
                "no_gene_novel_swissprot_aligned": K3,
                "no_gene_novel_other": K4,         # tier neither refseq nor swissprot
                "no_gene_novel_unaligned": K5,     # novel with no mmseqs hit
                "in_transcriptome_refseq_true":  T1,  # refseq seqs found in transcriptome
                "in_transcriptome_refseq_false": T2,  # refseq seqs NOT found in transcriptome
                "duplicate_dropped": D,
            }

        The ``in_transcriptome_refseq_*`` counters are populated only
        when ``transcriptome_seq_hashes`` was supplied; otherwise refseq
        records emit ``[in_transcriptome=unknown]`` and neither counter
        increments.
    """
    reference_fasta = Path(reference_fasta)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a lookup from novel protein_id → (best mmseqs target,
    # aligned_to) by picking the lowest-evalue hit per query.
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
            tier = row.get("aligned_to")
            novel_target[q] = (target, tier)

    stats = {
        "reference": 0,
        "novel": 0,
        "no_gene_refseq": 0,
        "no_gene_novel_refseq_aligned": 0,
        "no_gene_novel_swissprot_aligned": 0,
        "no_gene_novel_other": 0,
        "no_gene_novel_unaligned": 0,
        "duplicate_dropped": 0,
        "in_transcriptome_refseq_true": 0,
        "in_transcriptome_refseq_false": 0,
    }
    seen_hashes: set[str] = set()

    with output_path.open("w") as out:
        # Reference records first. Every refseq entry emits the same
        # 5-bracket-tag header (gene / accession / aligned_to /
        # in_transcriptome / source); missing values use the literal
        # placeholders ``unknown`` (gene), ``none`` (aligned_to), and
        # ``unknown`` (in_transcriptome when no
        # transcriptome_seq_hashes was supplied).
        for header, seq in _iter_fasta(reference_fasta):
            seq = seq.upper()
            protein_id = _header_id(header)
            if dedup_reference_against_self:
                h = _seq_hash(seq)
                if h in seen_hashes:
                    stats["duplicate_dropped"] += 1
                    continue
                seen_hashes.add(h)
            gene = reference_gene_map.get(protein_id) or "unknown"
            if gene == "unknown":
                stats["no_gene_refseq"] += 1
            if transcriptome_seq_hashes is None:
                in_tx = "unknown"
            else:
                # Match the same sequence-hash key Stage 0 uses to dedup
                # refseq against itself, so a refseq entry whose
                # sequence is byte-identical to a novel ORF the
                # transcriptome called registers as in_transcriptome=true.
                if _seq_hash(seq) in transcriptome_seq_hashes:
                    in_tx = "true"
                    stats["in_transcriptome_refseq_true"] += 1
                else:
                    in_tx = "false"
                    stats["in_transcriptome_refseq_false"] += 1
            bracket_tags = [
                f"[gene={gene}]",
                f"[accession={protein_id}]",
                "[aligned_to=none]",
                f"[in_transcriptome={in_tx}]",
            ]
            header_line = (
                f">{protein_id} {' '.join(bracket_tags)} source=refseq\n"
            )
            out.write(header_line)
            _write_seq(out, seq)
            stats["reference"] += 1

        # Novel records — same 5-bracket-tag shape as refseq.
        # Placeholders flow through identically: gene falls back to
        # "unknown" when gene_map.get(target) returns None (the
        # SwissProt-without-GN= case), accession falls back to
        # "unknown" when the novel ORF has no alignment hit at all,
        # and aligned_to falls back to "none" when the alignment hit
        # carried no tier annotation OR there was no hit.
        # ``in_transcriptome`` is always ``true`` for novel records
        # because they came from the transcriptome by construction.
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
            if target is not None:
                gene = reference_gene_map.get(target) or "unknown"
                accession = str(target)
                aligned_to = str(tier) if tier else "none"
                if gene == "unknown":
                    if aligned_to == "swissprot":
                        stats["no_gene_novel_swissprot_aligned"] += 1
                    elif aligned_to == "refseq":
                        stats["no_gene_novel_refseq_aligned"] += 1
                    else:
                        stats["no_gene_novel_other"] += 1
            else:
                gene = "unknown"
                accession = "unknown"
                aligned_to = "none"
                stats["no_gene_novel_unaligned"] += 1
            bracket_tags = [
                f"[gene={gene}]",
                f"[accession={accession}]",
                f"[aligned_to={aligned_to}]",
                "[in_transcriptome=true]",
            ]
            out.write(
                f">{protein_id} {' '.join(bracket_tags)} source=transcriptome\n"
            )
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
    classifier downstream infers each hit's ``aligned_to`` from
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


def _tag_aligned_to(
    alignment_hits: "pa.Table", reference_fasta: Path | str
) -> "pa.Table":
    """Set ``aligned_to`` per hit by target membership in the
    reference proteome: ``target ∈ reference → "refseq"`` else
    ``"swissprot"``. Replaces any existing column.

    The single competitive mmseqs search can't distinguish which DB a
    target came from; accession membership recovers it.
    """
    import pyarrow as pa

    if alignment_hits.num_rows == 0:
        if "aligned_to" in alignment_hits.column_names:
            return alignment_hits
        return alignment_hits.append_column(
            "aligned_to", pa.array([], type=pa.string())
        )
    ref_ids = collect_protein_ids(reference_fasta)
    tier = pa.array(
        [
            "refseq" if str(t) in ref_ids else "swissprot"
            for t in alignment_hits.column("target").to_pylist()
        ],
        type=pa.string(),
    )
    if "aligned_to" in alignment_hits.column_names:
        return alignment_hits.set_column(
            alignment_hits.column_names.index("aligned_to"),
            "aligned_to",
            tier,
        )
    return alignment_hits.append_column("aligned_to", tier)


# ──────────────────────────────────────────────────────────────────────
# Orchestrator entry point
# ──────────────────────────────────────────────────────────────────────


def run_transcriptome_to_proteomics(
    *,
    args,  # args: argparse.Namespace
    reference,  # constellation.sequencing.reference.Reference
    swissprot_reference,  # constellation.sequencing.reference.Reference
) -> int:
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

    Reference resolution is handled by the CLI handler before this
    function is called — both ``reference`` and ``swissprot_reference``
    are :class:`Reference` objects pointing at installed cache slots
    with their required artifacts (genome+annotation for ``reference``,
    proteome for ``swissprot_reference``).
    """
    import json
    import shutil
    import socket
    import sys
    import time
    from datetime import datetime, timezone

    import pyarrow.parquet as pq

    from constellation import __version__ as constellation_version
    from constellation.core.io.schemas import read_mmseqs_tab
    from constellation.massspec.search import (
        apply_collision_filter,
        filter_elib_by_losers,
        protein_to_gene_from_swissprot,
    )
    from constellation.massspec.search.encyclopedia import (
        require_min_encyclopedia,
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
    from constellation.thirdparty.registry import ToolNotFoundError, find

    # Fail-fast: both references must have their required artifacts installed.
    reference.require(proteome=True, annotation=True)
    swissprot_reference.require(proteome=True)

    # ── Path resolution + env validation ───────────────────────────────
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # TPM-cutoff suffix encoded into every TPM-dependent output filename
    # so parallel sweeps at different cutoffs under the same --output-dir
    # don't collide. Integer cutoffs format compactly ("1TPM" not "1.0TPM");
    # non-integer cutoffs (e.g. 0.5) preserve their decimal but with `.`
    # → `_` so the filename stays POSIX-safe ("0_5TPM"). Stages 0, 1, 2,
    # 3, 6, 9 are TPM-independent (Stage 0 dedupes refseq, Stage 1
    # normalises counts, Stage 2's input IS the TPM filter, Stage 3
    # aligns the same query set, Stage 6 builds a cache from the GPF
    # spectra alone, Stage 9 names per-injection); the suffix applies
    # to Stages 4, 5, 7, 8, 10.
    _tpm_val = args.min_avg_tpm
    if _tpm_val == int(_tpm_val):
        tpm_suffix = f"_{int(_tpm_val)}TPM"
    else:
        tpm_suffix = f"_{str(_tpm_val).replace('.', '_')}TPM"
    demux_dir = Path(args.demux_dir).resolve()
    # Optional override: an ORF FASTA produced by a different upstream
    # pipeline. When None, Stage 2 derives sequences from
    # ``counts_tpm.column("sequence")``.
    protein_fasta = (
        Path(args.protein_fasta).resolve()
        if getattr(args, "protein_fasta", None) is not None
        else None
    )
    # Reference artifacts come from the resolved Reference objects —
    # never raw paths. The CLI resolver guarantees both objects have
    # the required artifacts installed (see require() above).
    reference_fasta = reference.protein_fasta_path
    reference_annotation = reference.annotation_dir
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

    # ── Preflight: require EncyclopeDIA >= 6.5.15 before any work ───────
    # Stages 5/6/7/9/10 invoke the jar, and Stage 5 (predict-library)
    # needs the 6.5.15-only ``-convert -fastaToJChronologerLibrary``
    # utility specifically. Resolve + version-check the jar HERE so a
    # too-old / missing install fails immediately, instead of after
    # Stage 0's refseq dedup, the mmseqs2 alignment, and the SwissProt
    # fetch below have already burned minutes. Stages 5/9 keep their own
    # in-wrapper checks (defense in depth + the standalone CLI paths).
    try:
        _enc_handle = find("encyclopedia")
    except ToolNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if require_min_encyclopedia(_enc_handle):
        return 1

    # ── Resolve SwissProt from the swissprot_reference Reference ──────
    # No lazy fetch — the CLI resolver guarantees an installed SwissProt
    # release (resolved via --swissprot-reference or the default
    # ``swissprot`` bare-handle install).
    swissprot_fasta = swissprot_reference.protein_fasta_path
    swissprot_release: str | None = swissprot_reference.release

    t_start = time.monotonic()
    stage_manifests: dict[str, dict] = {}

    # ── Stage 0: dedupe reference FASTA on protein SEQUENCE ────────────
    # Cartographer-style upstream dedup. Every subsequent stage that
    # consumes the reference proteome (novel-vs-ref filter, alignment
    # target, combined.fasta build) uses this deduped file so all three
    # operate on the same accession set. Without this, the alignment can
    # target XP_* accessions that combined.fasta drops as duplicates,
    # leaving the classifier with no gene-map entry for them.
    deduped_refseq = output_dir / "00_deduped_refseq" / "deduped_refseq.fasta"
    if not deduped_refseq.exists():
        deduped_refseq.parent.mkdir(parents=True, exist_ok=True)
        n_unique = deduplicate_fasta(reference_fasta, deduped_refseq)
        _log(f"Stage 0: deduplicate_fasta wrote {n_unique:,} unique sequences "
             f"→ {deduped_refseq}")
    else:
        _log("Stage 0: deduped_refseq.fasta already exists; reusing")
    # From here on, downstream stages MUST consume `deduped_refseq` rather
    # than the raw `reference_fasta` argument.
    reference_fasta_for_pipeline = deduped_refseq

    # ── Stage 1: read + TPM ────────────────────────────────────────────
    stage_dir = output_dir / "01_protein_counts"
    if not _stage_done(stage_dir, args.resume):
        _log("Stage 1: reading protein counts + TPM-normalizing")
        stage_dir.mkdir(parents=True, exist_ok=True)
        counts = read_protein_counts_tab(demux_dir)
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
            protein_fasta=protein_fasta,
            reference_fasta=reference_fasta_for_pipeline,
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

    # Unified thread budget: --threads drives both mmseqs2 and EncyclopeDIA;
    # --mmseqs-threads (default None) optionally overrides mmseqs2 only. The
    # Stage 9 per-injection JVMs divide --threads across the parallel workers.
    threads = getattr(args, "threads", None) or 8
    args.threads = threads
    if getattr(args, "mmseqs_threads", None) is None:
        args.mmseqs_threads = threads

    # ── Stage 3: competitive mmseqs2 alignment ─────────────────────────
    stage_dir = output_dir / "03_alignment"
    alignments_tab = stage_dir / "alignments.tab"
    if not _stage_done(stage_dir, args.resume):
        _log(f"Stage 3: mmseqs2 easy-search ({args.mmseqs_threads} threads)")
        stage_dir.mkdir(parents=True, exist_ok=True)
        scratch = stage_dir / ".scratch"
        target_fasta = stage_dir / "_target.fasta"
        write_competitive_target_fasta(
            reference_fasta=reference_fasta_for_pipeline,
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
        alignment_hits = _tag_aligned_to(alignment_hits, reference_fasta_for_pipeline)
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
    combined_path = stage_dir / f"combined{tpm_suffix}.fasta"
    if not _stage_done(stage_dir, args.resume):
        _log(f"Stage 4: alignment filter + {combined_path.name} with annotated headers")
        stage_dir.mkdir(parents=True, exist_ok=True)
        filtered_novel = apply_alignment_filter(
            novel_table=novel_table,
            alignment_hits=alignment_hits,
            evalue_threshold=args.evalue_threshold,
        )
        # Merged gene map: RefSeq .gbff (NP_/XP_ → gene) ∪ SwissProt GN= (bare
        # UniProt accession → gene). Cartographer-style — combined.fasta gets
        # [gene=X] tags for both refseq-aligned AND swissprot-aligned novel
        # ORFs, so the classifier needs no other gene source downstream.
        gene_map = reference.gene_map()
        n_refseq_genes = len(gene_map)
        sp_gene_map = protein_to_gene_from_swissprot(swissprot_fasta)
        gene_map.update(sp_gene_map)
        _log(
            f"Stage 4: gene_map = {n_refseq_genes:,} RefSeq + "
            f"{len(sp_gene_map):,} SwissProt = {len(gene_map):,} total entries"
        )
        # Build the in_transcriptome SHA-membership set: every distinct
        # protein sequence the upstream transcriptome quant saw,
        # hashed via the same _seq_hash helper Stage 0 uses for refseq
        # dedup. A refseq entry whose sequence is byte-identical to a
        # transcriptome-called ORF lands as in_transcriptome=true in
        # the combined FASTA header.
        transcriptome_seq_hashes = {
            _seq_hash(s.upper())
            for s in counts_tpm.column("sequence").to_pylist()
            if s is not None
        }
        n_written, fasta_stats = build_combined_fasta(
            reference_fasta=reference_fasta_for_pipeline,
            filtered_novel=filtered_novel,
            alignment_hits=alignment_hits,
            reference_gene_map=gene_map,
            output_path=combined_path,
            transcriptome_seq_hashes=transcriptome_seq_hashes,
        )
        _log(
            f"Stage 4: no_gene breakdown — "
            f"refseq={fasta_stats['no_gene_refseq']:,}, "
            f"novel→refseq={fasta_stats['no_gene_novel_refseq_aligned']:,}, "
            f"novel→swissprot={fasta_stats['no_gene_novel_swissprot_aligned']:,}, "
            f"novel-unaligned={fasta_stats['no_gene_novel_unaligned']:,}"
        )
        _log(
            f"Stage 4: in_transcriptome breakdown — "
            f"refseq=true:{fasta_stats['in_transcriptome_refseq_true']:,}, "
            f"refseq=false:{fasta_stats['in_transcriptome_refseq_false']:,} "
            f"(transcriptome-derived novels are always true)"
        )
        (stage_dir / "gene_map.json").write_text(
            json.dumps(gene_map, indent=2, sort_keys=True) + "\n"
        )
        _write_stage_manifest(
            stage_dir,
            subcommand="04_combined_fasta",
            params={
                "evalue_threshold": args.evalue_threshold,
                "gene_map_refseq_source": str(reference_annotation),
                "gene_map_swissprot_source": str(swissprot_fasta),
                "gene_map_size": len(gene_map),
                "gene_map_refseq_entries": n_refseq_genes,
                "gene_map_swissprot_entries": len(sp_gene_map),
            },
            counts={"n_written": n_written, **fasta_stats},
        )
        _touch_success(stage_dir)
    stage_manifests["04_combined_fasta"] = _read_stage_manifest(stage_dir)

    # ── Stage 5: predict-library ───────────────────────────────────────
    stage_dir = output_dir / "05_predict_library"
    combined_dlib = stage_dir / f"combined{tpm_suffix}.dlib"
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
    # Output cache name: <run-stem>_combined_GPF.dia. The "combined"
    # element reflects that EncyclopeDIA's -processDIA merges every
    # input file into one cache; the <run-stem> (basename of
    # --output-dir or the --run-name override) ties the cache to the
    # run that produced it so a directory listing makes the
    # provenance obvious even when stripped of context.
    stage_dir = output_dir / "06_process_dia"
    run_stem = args.run_name or output_dir.name
    combined_dia = stage_dir / f"{run_stem}_combined_GPF.dia"
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
        f"combined{tpm_suffix}.elib"
        if args.no_collision_filter
        else f"combined{tpm_suffix}.filtered.elib"
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
                threads=args.threads,
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
        # Novel proteins = every transcriptome-derived ORF. Source:
        # ``counts_tpm`` carries the full per-ORF set with sequences,
        # so we derive the 2-column ``(protein_id, sequence)`` table
        # directly. The optional --protein-fasta override (external
        # upstream pipelines that supply their own ORFs) takes precedence.
        if protein_fasta is not None:
            novel_proteins = read_fasta_proteins(protein_fasta)
        else:
            # Long-form counts_tpm has one row per (protein, sample);
            # collapse to unique (protein_id, sequence) pairs. ``set``
            # comprehension would drop the order; use a dict to
            # preserve first-occurrence.
            seen_pid: dict[str, str] = {}
            for pid, seq in zip(
                counts_tpm.column("protein_id").to_pylist(),
                counts_tpm.column("sequence").to_pylist(),
            ):
                if pid is not None and seq is not None and pid not in seen_pid:
                    seen_pid[pid] = seq.upper()
            novel_proteins = pa.table(
                {
                    "protein_id": pa.array(list(seen_pid), type=pa.string()),
                    "sequence": pa.array(list(seen_pid.values()), type=pa.string()),
                }
            )
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
                "novel_fasta": (
                    str(protein_fasta) if protein_fasta is not None
                    else "derived from counts_tpm.parquet"
                ),
            },
            filename=f"novel_peptides{tpm_suffix}.parquet",
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
        _per_jvm_threads = max(1, args.threads // max(1, args.injection_threads))
        _log(f"Stage 9: per-injection searches ({len(injection_files)} mzML, "
             f"{args.injection_threads} JVM{'s' if args.injection_threads != 1 else ''} "
             f"× {_per_jvm_threads} EncyclopeDIA thread"
             f"{'s' if _per_jvm_threads != 1 else ''})")
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
                "encyclopedia_threads_per_jvm": max(
                    1, args.threads // max(1, args.injection_threads)
                ),
                "n_injections": len(injection_files),
            },
        )
        _touch_success(stage_dir)
    stage_manifests["09_per_injection"] = _read_stage_manifest(stage_dir)

    # ── Stage 10: library-export quant report ──────────────────────────
    stage_dir = output_dir / "10_quant_report"
    quant_report_elib = stage_dir / f"quant_report{tpm_suffix}.elib"
    if not _stage_done(stage_dir, args.resume):
        _log(f"Stage 10: library-export → {quant_report_elib.name}")
        stage_dir.mkdir(parents=True, exist_ok=True)
        # EncyclopeDIA -libexport reconstructs each run's analysis from the
        # files sitting next to its spectra file: <X.raw>, <X.raw>.elib
        # (per-run chromatogram library), and the Percolator/feature
        # sidecars <X.raw>.features.txt / <X.raw>.encyclopedia.txt. The
        # Stage 9 search SPLIT these — the .elib landed in the run dir, but
        # EncyclopeDIA wrote the feature/percolator sidecars next to the
        # SOURCE raw — so co-locate the full native-named set in one flat
        # dir. A dir of bare .elib files (or a renamed elib) fails with
        # "Missing feature file" / "Can't find any representative jobs".
        export_input_dir = stage_dir / "_per_injection_elibs"
        export_input_dir.mkdir(parents=True, exist_ok=True)

        def _link(target: Path, src: Path) -> None:
            if src.exists() and not target.exists():
                target.symlink_to(src)

        def _colocate_spectra(target: Path, src: Path) -> None:
            # HARD-link the spectra file, not symlink: EncyclopeDIA's native
            # Thermo reader (MSRawJava / ThermoFisher RawFileReader) fails to
            # open a .raw through a symlink ("instrument index not available"),
            # even though the search read the same file fine at its real path.
            # A hard link is an ordinary directory entry on the same inode —
            # indistinguishable from the original to the reader. Falls back to
            # a symlink only if the link target is on another filesystem.
            #
            # Self-healing: replace a stale symlink left by an earlier run so
            # the hard link actually takes effect without a manual rm -rf.
            import os
            if not src.exists():
                return
            if target.is_symlink() or target.exists():
                if target.is_symlink():
                    target.unlink()  # upgrade stale symlink → hard link
                else:
                    return  # already a real file (hard link) — keep it
            try:
                os.link(src, target)
            except OSError:
                target.symlink_to(src)

        n_export_inputs = 0
        for mzml in injection_files:
            sample_dir = _sanitise_dirname(mzml.parent.name)
            run_dir = output_dir / "09_per_injection" / sample_dir / mzml.stem
            elibs = sorted(run_dir.glob("*.elib"))
            if not elibs:
                continue
            # spectra (hard-linked) + the per-run chromatogram .elib, named
            # <raw>.elib so libexport pairs it with the spectra file.
            _colocate_spectra(export_input_dir / mzml.name, mzml)
            _link(export_input_dir / f"{mzml.name}.elib", elibs[0])
            # Percolator / feature sidecars EncyclopeDIA wrote next to the
            # source raw (<raw>.features.txt, <raw>.encyclopedia.txt, ...).
            # Skip the diagnostic PDFs.
            for sidecar in mzml.parent.glob(f"{mzml.name}.*"):
                if sidecar.suffix.lower() == ".pdf":
                    continue
                _link(export_input_dir / sidecar.name, sidecar)
            n_export_inputs += 1
        if n_export_inputs == 0:
            raise FileNotFoundError(
                "Stage 10: no per-injection .elib results found under "
                f"{output_dir / '09_per_injection'} to export"
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
        # libexport can exit 0 without writing a report (e.g. it found no
        # usable per-run analyses). Fail loudly instead of faking success.
        if not quant_report_elib.is_file():
            raise FileNotFoundError(
                f"Stage 10: library-export exited {result.returncode} but did "
                f"not write {quant_report_elib.name}. Check {stage_dir / 'logs'}"
                " — 'Missing feature file' / 'Can't find any representative "
                "jobs' means the per-injection analysis set (.elib + "
                ".features.txt + .encyclopedia.txt) wasn't co-located."
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
        "subcommand": "transcriptome-to-proteome",
        "argv": sys.argv,
        "inputs": {
            "demux_dir": _input_meta(demux_dir),
            "protein_fasta": _input_meta(protein_fasta),
            "reference_fasta": _input_meta(reference_fasta),
            "reference_annotation": _input_meta(reference_annotation),
            "gpf": [_input_meta(p) for p in gpf_files],
            "injections": [_input_meta(p) for p in injection_files],
            "swissprot_fasta": _input_meta(swissprot_fasta),
            "swissprot_release": swissprot_release,
        },
        "reference": {
            "handle": str(reference.handle),
            "release_dir": str(reference.release_dir),
            "source": reference.source,
            "assembly_accession": reference.assembly_accession,
        },
        "swissprot_reference": {
            "handle": str(swissprot_reference.handle),
            "release_dir": str(swissprot_reference.release_dir),
            "source": swissprot_reference.source,
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
            threads=max(1, args.threads // max(1, args.injection_threads)),
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
