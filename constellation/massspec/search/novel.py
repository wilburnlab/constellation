"""Classify novel peptides against an mmseqs2 alignment of the novel
proteome vs the reference proteome.

Port of cartographer's `classify_novel_peptide` + `classify_novel_peptides`
from ``cartographer/data/nanopore.py`` (lines 661-1129). Same CIGAR-walking
algorithm, same 11-class taxonomy, same per-peptide deduplication
priority. Three meaningful differences from the cartographer original:

  1. **Tryptic digestion goes through ``core.sequence.protein.cleave``**
     instead of cartographer's ``digest_fasta``. Same enzyme registry +
     missed-cleavage enumeration, no functional drift.
  2. **Alignment tier is inferred from target membership in the
     reference proteome** when the input rows have a null
     ``alignment_tier`` column. Cartographer required the user to tag
     each hit explicitly; Constellation accepts a single combined
     ``.tab`` with mixed reference + swissprot hits and infers tier
     internally. Per-row tier annotations on input are still respected
     when present.
  3. **Flexible tabular inputs** — the function takes plain Arrow tables
     for detected peptides, alignments, and proteomes rather than
     paths-on-disk. The disk-based wrapper that mirrors cartographer's
     ``classify_novel_peptides_from_files`` is at the CLI layer.

The 11 classes (priority order — lower = more specific, dedup keeps
the lowest):

    snp                   single mismatch in the peptide span
    insertion             single insertion in the peptide span
    deletion              single deletion in the peptide span
    complex               two or more event types
    n_terminal_truncation peptide flush at the novel N-terminus; the
                          reference protein extends further upstream
    c_terminal_truncation symmetric for the C-terminus
    trypsin_cutsite_mutation
                          peptide is identical to reference but a
                          flanking K/R differs — a new cut site was
                          created by the variant
    n_term_deviation      peptide starts before the alignment range
    c_term_deviation      peptide ends past the alignment range
    unknown               no usable alignment or all-match with no
                          truncation / cutsite explanation
    non_reference         alignment target is not in the reference
                          proteome — handled by short-circuit return
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.core.io.schemas import cast_to_schema
from constellation.core.sequence.protein import cleave
from constellation.massspec.search.schemas import NOVEL_PEPTIDE_TABLE


_CLASSIFICATION_PRIORITY: dict[str, int] = {
    "snp": 0,
    "insertion": 1,
    "deletion": 2,
    "complex": 3,
    "n_terminal_truncation": 4,
    "c_terminal_truncation": 5,
    "trypsin_cutsite_mutation": 6,
    "n_term_deviation": 7,
    "c_term_deviation": 8,
    "unknown": 9,
    "non_reference": 10,
}


_CIGAR_RE = re.compile(r"(\d+)([MIDX=])")


def _parse_cigar(cigar_str: str) -> list[tuple[str, int]]:
    """Parse an mmseqs2 CIGAR string into ``(op, count)`` tuples.

    Supported ops: ``M`` (match/mismatch — needs target comparison to
    discriminate), ``I`` (insertion in query — gap in reference),
    ``D`` (deletion in query — extra residues in reference), ``=``
    (sequence match), ``X`` (mismatch).
    """
    return [(op, int(n)) for n, op in _CIGAR_RE.findall(cigar_str)]


# ──────────────────────────────────────────────────────────────────────
# Single-peptide classifier (verbatim port of cartographer logic)
# ──────────────────────────────────────────────────────────────────────


def classify_single_peptide(
    peptide_seq: str,
    protein_seq: str,
    hit_row: Mapping[str, object],
    reference_seqs: Mapping[str, str],
) -> tuple[str, str]:
    """Classify one ``(peptide, hit)`` pair.

    Direct port of cartographer's ``classify_novel_peptide``
    (``cartographer/data/nanopore.py`` lines 685-850). Returns
    ``(classification, ref_seq)`` where ``classification`` is one of
    the 11 strings in :data:`_CLASSIFICATION_PRIORITY` and ``ref_seq``
    is the aligned reference substring at the peptide locus (empty
    string when not applicable).

    Parameters
    ----------
    peptide_seq
        Bare AA sequence of the detected tryptic peptide.
    protein_seq
        Full AA sequence of the novel protein the peptide comes from.
    hit_row
        One mmseqs2 alignment hit as a mapping. Required keys: ``query``,
        ``target``, ``qstart``, ``qend``, ``tstart``, ``tend``, ``cigar``,
        ``alignment_tier``. Coordinates are 1-indexed inclusive (mmseqs2
        convention). ``alignment_tier == "non_reference"`` (or any string
        other than ``"reference"``) short-circuits to the non_reference
        class — caller is responsible for setting the tier appropriately
        from target-vs-reference-proteome membership.
    reference_seqs
        ``target_accession → AA sequence``. Used to compare per-residue
        identity at aligned positions and to populate
        ``ref_protein_seq`` / look for the peptide in the reference for
        truncation / cutsite analysis.
    """
    if str(hit_row["alignment_tier"]) != "reference":
        return ("non_reference", "")

    pep_start = protein_seq.find(peptide_seq)
    if pep_start == -1:
        return ("unknown", "")
    pep_end = pep_start + len(peptide_seq)

    # Convert mmseqs2 1-indexed inclusive coords → 0-indexed half-open.
    q_start_0 = int(hit_row["qstart"]) - 1
    q_end_0 = int(hit_row["qend"])
    t_start_0 = int(hit_row["tstart"]) - 1

    if pep_start < q_start_0:
        return ("n_term_deviation", "")
    if pep_end > q_end_0:
        return ("c_term_deviation", "")

    ref_seq_full = reference_seqs.get(str(hit_row["target"]), "")
    cigar_ops = _parse_cigar(str(hit_row["cigar"]))

    q_pos = q_start_0
    t_pos = t_start_0
    ref_chars: list[str] = []
    has_insertion = False
    has_deletion = False
    has_mismatch = False

    for op, count in cigar_ops:
        if q_pos >= pep_end:
            break
        if op in ("M", "=", "X"):
            for _ in range(count):
                if pep_start <= q_pos < pep_end:
                    ref_aa = ref_seq_full[t_pos] if t_pos < len(ref_seq_full) else ""
                    ref_chars.append(ref_aa)
                    if protein_seq[q_pos] != ref_aa:
                        has_mismatch = True
                q_pos += 1
                t_pos += 1
        elif op == "I":
            # Insertion in query — gap in reference; t_pos does not advance.
            for _ in range(count):
                if pep_start <= q_pos < pep_end:
                    has_insertion = True
                q_pos += 1
        elif op == "D":
            # Deletion in query — extra residues in reference; q_pos does not advance.
            if pep_start <= q_pos < pep_end:
                ref_chars.extend(ref_seq_full[t_pos : t_pos + count])
                has_deletion = True
            t_pos += count

    ref_seq = "".join(ref_chars)

    n_event_types = sum([has_insertion, has_deletion, has_mismatch])
    if n_event_types > 1:
        return ("complex", ref_seq)
    if has_insertion:
        return ("insertion", ref_seq)
    if has_deletion:
        return ("deletion", ref_seq)
    if has_mismatch:
        return ("snp", ref_seq)

    # All-match block: check whether the peptide appears in the reference
    # protein at all. If yes, it's a truncation or cutsite-mutation case.
    pep_start_ref = ref_seq_full.find(peptide_seq)
    if pep_start_ref != -1:
        if pep_start == 0 and pep_start_ref > 0:
            return ("n_terminal_truncation", ref_seq)
        if (
            pep_end == len(protein_seq)
            and pep_start_ref + len(peptide_seq) < len(ref_seq_full)
        ):
            return ("c_terminal_truncation", ref_seq)

        novel_left = protein_seq[pep_start - 1] if pep_start > 0 else None
        ref_left = ref_seq_full[pep_start_ref - 1] if pep_start_ref > 0 else None
        novel_right = protein_seq[pep_end] if pep_end < len(protein_seq) else None
        ref_right = (
            ref_seq_full[pep_start_ref + len(peptide_seq)]
            if pep_start_ref + len(peptide_seq) < len(ref_seq_full)
            else None
        )
        if novel_left != ref_left or novel_right != ref_right:
            return ("trypsin_cutsite_mutation", ref_seq)

    return ("unknown", "")


# ──────────────────────────────────────────────────────────────────────
# Batch classifier — operates on Arrow tables
# ──────────────────────────────────────────────────────────────────────


def _digest_proteins_into_index(
    proteins: pa.Table,
    *,
    enzyme: str,
    max_missed_cleavages: int,
    min_peptide_length: int,
    max_peptide_length: int,
) -> dict[str, set[str]]:
    """Return ``peptide_sequence → set(protein_id)`` for all tryptic
    peptides of ``proteins``. Driven by
    :func:`constellation.core.sequence.protein.cleave`.

    Proteins whose sequence contains characters outside the canonical
    20-AA alphabet (Sec ``U``, Pyl ``O``, ambiguity codes ``B``/``Z``/``X``/``J``)
    are **skipped** rather than aborting the digest. Real-world UniProt /
    RefSeq FASTAs contain a small number of these residues — the
    cartographer original used regex digestion which silently tolerated
    them; we preserve that behaviour by skipping at the protein granularity
    so a single bad accession doesn't kill the whole batch.
    """
    out: dict[str, set[str]] = {}
    protein_ids = proteins.column("protein_id").to_pylist()
    sequences = proteins.column("sequence").to_pylist()
    for protein_id, seq in zip(protein_ids, sequences, strict=True):
        if not isinstance(seq, str) or not seq:
            continue
        try:
            peptides = cleave(
                seq,
                enzyme,
                missed_cleavages=max_missed_cleavages,
                min_length=min_peptide_length,
                max_length=max_peptide_length,
            )
        except ValueError:
            # Sequence contains non-canonical residues (U / O / B / Z / X / J).
            # Skip this protein — matches cartographer's tolerance.
            continue
        for pep in peptides:
            out.setdefault(pep, set()).add(str(protein_id))
    return out


def _select_best_hit_per_query(alignments: pa.Table) -> pa.Table:
    """For each `query`, keep the single hit with the lowest e-value.

    Matches cartographer.parse_mmseqs_hits's per-query best-hit
    selection. Stable under ties (first occurrence wins).
    """
    if alignments.num_rows == 0:
        return alignments
    sorted_t = alignments.sort_by(
        [("query", "ascending"), ("evalue", "ascending")]
    )
    queries = sorted_t.column("query").to_pylist()
    keep: list[int] = []
    seen: set[str] = set()
    for i, q in enumerate(queries):
        if q in seen:
            continue
        seen.add(q)
        keep.append(i)
    return sorted_t.take(keep)


def _infer_alignment_tier(
    alignments: pa.Table, reference_protein_ids: set[str]
) -> pa.Table:
    """Fill null ``alignment_tier`` values by checking target membership
    in ``reference_protein_ids``. Non-null tier values pass through
    unchanged.

    target ∈ reference → tier = ``"reference"``
    target ∉ reference → tier = ``"non_reference"``
    """
    if "alignment_tier" not in alignments.column_names:
        # Caller passed a table missing the optional column entirely —
        # synthesise it from inference.
        targets = alignments.column("target").to_pylist()
        inferred = [
            "reference" if str(t) in reference_protein_ids else "non_reference"
            for t in targets
        ]
        return alignments.append_column(
            "alignment_tier", pa.array(inferred, type=pa.string())
        )
    existing = alignments.column("alignment_tier").to_pylist()
    targets = alignments.column("target").to_pylist()
    filled = [
        existing[i]
        if existing[i] is not None
        else ("reference" if str(targets[i]) in reference_protein_ids else "non_reference")
        for i in range(len(existing))
    ]
    # Replace the column wholesale to ensure consistent typing
    return alignments.set_column(
        alignments.column_names.index("alignment_tier"),
        "alignment_tier",
        pa.array(filled, type=pa.string()),
    )


def classify_novel_peptides(
    detected_peptides: pa.Table,
    alignments: pa.Table,
    reference_proteins: pa.Table,
    novel_proteins: pa.Table,
    *,
    gene_map: Mapping[str, str] | None = None,
    transcript_map: Mapping[str, str] | None = None,
    enzyme: str = "Trypsin",
    max_missed_cleavages: int = 1,
    min_peptide_length: int = 7,
    max_peptide_length: int = 50,
) -> pa.Table:
    """Classify detected novel peptides against a reference-vs-novel
    alignment.

    Returns a :data:`NOVEL_PEPTIDE_TABLE`-shaped Arrow table with one
    row per unique novel peptide sequence (per-peptide deduplication
    keeps the most-specific classification under
    :data:`_CLASSIFICATION_PRIORITY`).

    Workflow (mirrors cartographer.classify_novel_peptides_from_files):

    1. Tryptic-digest both proteomes via
       :func:`constellation.core.sequence.protein.cleave`.
    2. ``theoretical_novel = novel_digest_keys − reference_digest_keys``
       — peptides reachable by tryptic digest of the novel proteome but
       NOT by digest of the reference proteome.
    3. ``detected_novel = theoretical_novel ∩ detected_peptide_sequences``
       — restrict to peptides actually identified by the search.
    4. For each ``(peptide, protein_id)`` pair where ``protein_id`` is
       a novel protein producing the peptide: look up the mmseqs2 hit
       for that protein, walk the CIGAR, and classify the deviation.
    5. Deduplicate per peptide_sequence by keeping the most-specific
       classification.

    Parameters
    ----------
    detected_peptides
        Arrow table with column ``peptide_sequence`` (string, canonical
        AA). Optional columns ``peptide_id`` and ``modified_sequence``
        are carried through to the output when present.
    alignments
        :data:`ALIGNMENT_HIT_TABLE`-shaped (or compatible). Multiple
        hits per query are allowed; the function keeps the lowest-evalue
        hit per query. ``alignment_tier`` may be null on input — see the
        inference rule above.
    reference_proteins
        Arrow table with columns ``protein_id`` (string) and ``sequence``
        (string). Defines what counts as "reference" — both for the
        tier inference and for the tryptic-digest reference set.
    novel_proteins
        Same shape as ``reference_proteins``. The novel proteome.
    gene_map
        Optional ``protein_id → gene_symbol``. Looked up on the hit's
        ``target`` first, then on the novel ``protein_id``.
    transcript_map
        Optional ``protein_id → transcript_id``. Looked up on the novel
        ``protein_id``.
    enzyme, max_missed_cleavages, min_peptide_length, max_peptide_length
        Tryptic-digest parameters forwarded to
        :func:`core.sequence.protein.cleave`.
    """
    # 1. Digests
    novel_digest = _digest_proteins_into_index(
        novel_proteins,
        enzyme=enzyme,
        max_missed_cleavages=max_missed_cleavages,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
    )
    reference_digest = _digest_proteins_into_index(
        reference_proteins,
        enzyme=enzyme,
        max_missed_cleavages=max_missed_cleavages,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
    )
    reference_peptide_set = set(reference_digest.keys())

    # 2. theoretical novel peptides (in novel digest, not in reference)
    theoretical_novel = set(novel_digest.keys()) - reference_peptide_set

    # 3. Intersection with detected peptides
    detected_sequences = set(
        detected_peptides.column("peptide_sequence").to_pylist()
    )
    detected_novel = theoretical_novel & detected_sequences

    if not detected_novel:
        return NOVEL_PEPTIDE_TABLE.empty_table()

    # 4. Tier inference + best-hit-per-query
    reference_protein_ids = set(
        str(x) for x in reference_proteins.column("protein_id").to_pylist()
    )
    alignments_tiered = _infer_alignment_tier(alignments, reference_protein_ids)
    best_hits = _select_best_hit_per_query(alignments_tiered)
    hits_map: dict[str, dict[str, object]] = {
        str(row["query"]): row for row in best_hits.to_pylist()
    }

    # Protein-id → sequence maps for lookup during classification
    novel_seq_map = {
        str(pid): seq
        for pid, seq in zip(
            novel_proteins.column("protein_id").to_pylist(),
            novel_proteins.column("sequence").to_pylist(),
            strict=True,
        )
        if isinstance(seq, str)
    }
    reference_seq_map = {
        str(pid): seq
        for pid, seq in zip(
            reference_proteins.column("protein_id").to_pylist(),
            reference_proteins.column("sequence").to_pylist(),
            strict=True,
        )
        if isinstance(seq, str)
    }

    # Optional peptide_id / modified_sequence pass-through
    detected_seq_to_extra: dict[str, dict[str, object]] = {}
    has_peptide_id = "peptide_id" in detected_peptides.column_names
    has_modseq = "modified_sequence" in detected_peptides.column_names
    if has_peptide_id or has_modseq:
        cols = ["peptide_sequence"]
        if has_peptide_id:
            cols.append("peptide_id")
        if has_modseq:
            cols.append("modified_sequence")
        for row in detected_peptides.select(cols).to_pylist():
            seq = row["peptide_sequence"]
            extra: dict[str, object] = {}
            if has_peptide_id and row.get("peptide_id") is not None:
                extra["peptide_id"] = row["peptide_id"]
            if has_modseq and row.get("modified_sequence") is not None:
                extra["modified_sequence"] = row["modified_sequence"]
            # Keep the first encountered extras per peptide_sequence
            detected_seq_to_extra.setdefault(seq, extra)

    _gene_map: Mapping[str, str] = gene_map or {}
    _transcript_map: Mapping[str, str] = transcript_map or {}

    # 5. Classify each (peptide_seq, novel_protein_id) pair
    rows: list[dict[str, object]] = []
    for pep_seq in detected_novel:
        novel_protein_ids = novel_digest[pep_seq]
        for protein_id in novel_protein_ids:
            protein_seq = novel_seq_map.get(protein_id)
            hit = hits_map.get(protein_id)
            extras = detected_seq_to_extra.get(pep_seq, {})

            if hit is None or protein_seq is None:
                rows.append(
                    {
                        "peptide_id": extras.get("peptide_id"),
                        "peptide_sequence": pep_seq,
                        "modified_sequence": extras.get("modified_sequence"),
                        "classification": "unknown",
                        "ref_seq": None,
                        "protein_id": protein_id,
                        "ref_protein_id": None,
                        "gene": _gene_map.get(protein_id),
                        "transcript": _transcript_map.get(protein_id),
                        "cigar": None,
                        "alignment_tier": None,
                        "novel_protein_seq": protein_seq,
                        "ref_protein_seq": None,
                    }
                )
                continue

            cls, ref_seq = classify_single_peptide(
                pep_seq, protein_seq, hit, reference_seq_map
            )
            target = str(hit["target"])
            rows.append(
                {
                    "peptide_id": extras.get("peptide_id"),
                    "peptide_sequence": pep_seq,
                    "modified_sequence": extras.get("modified_sequence"),
                    "classification": cls,
                    "ref_seq": ref_seq,
                    "protein_id": protein_id,
                    "ref_protein_id": target,
                    "gene": _gene_map.get(target) or _gene_map.get(protein_id),
                    "transcript": _transcript_map.get(protein_id),
                    "cigar": str(hit["cigar"]),
                    "alignment_tier": hit.get("alignment_tier"),
                    "novel_protein_seq": protein_seq,
                    "ref_protein_seq": reference_seq_map.get(target),
                }
            )

    # 6. Deduplicate per peptide_sequence, keeping the most-specific class
    # (lowest priority value wins).
    rows.sort(
        key=lambda r: (
            r["peptide_sequence"],
            _CLASSIFICATION_PRIORITY.get(r["classification"], 99),
        )
    )
    deduped: list[dict[str, object]] = []
    seen_peptides: set[str] = set()
    for row in rows:
        if row["peptide_sequence"] in seen_peptides:
            continue
        seen_peptides.add(row["peptide_sequence"])
        # `novel_seq` column is the same value as `peptide_sequence` —
        # add explicitly so the output is readable without
        # cross-referencing.
        row_with_novel = {"novel_seq": row["peptide_sequence"], **row}
        deduped.append(row_with_novel)

    if not deduped:
        return NOVEL_PEPTIDE_TABLE.empty_table()

    table = pa.Table.from_pylist(deduped)
    return cast_to_schema(table, NOVEL_PEPTIDE_TABLE)


# ──────────────────────────────────────────────────────────────────────
# Output writer
# ──────────────────────────────────────────────────────────────────────


def save_novel_peptides(
    table: pa.Table,
    path: "Path | str",
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write a ``NOVEL_PEPTIDE_TABLE`` to a ParquetDir-style bundle.

    Mirrors the existing ``save_library`` / ``save_search`` pattern:
    one parquet file (``novel_peptides.parquet``) + a ``manifest.json``
    describing the format and carrying optional metadata (e.g. the
    alignment source, the reference proteome id, the search engine
    that produced the upstream peptide table).
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / "novel_peptides.parquet")
    manifest = {
        "format": "parquet_dir",
        "tables": ["novel_peptides"],
        "metadata": dict(metadata) if metadata else {},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def build_gene_map_from_fasta_headers(
    fasta_paths: Iterable["Path | str"],
) -> dict[str, str]:
    """Extract ``protein_accession → gene_symbol`` from FASTA headers.

    Looks for ``gene=<symbol>`` tokens — the convention cartographer's
    pipeline uses for the combined FASTA. Pure helper; defined here so
    the CLI handler can pre-build the map from one or more FASTAs and
    pass the dict in to :func:`classify_novel_peptides`.
    """
    out: dict[str, str] = {}
    for fasta_path in fasta_paths:
        p = Path(fasta_path)
        if not p.is_file():
            continue
        with p.open() as fh:
            for line in fh:
                if not line.startswith(">"):
                    continue
                header = line[1:].rstrip("\n")
                tokens = header.split()
                if not tokens:
                    continue
                accession = tokens[0]
                for token in tokens[1:]:
                    if token.startswith("gene="):
                        out[accession] = token[len("gene="):]
                        break
    return out


def read_fasta_proteins(path: "Path | str") -> pa.Table:
    """Read a FASTA file into a 2-column Arrow table.

    Convenience reader for the CLI handler: yields a
    ``(protein_id: string, sequence: string)`` table where ``protein_id``
    is the first whitespace-delimited token of each header line and
    ``sequence`` is the concatenated uppercase AA sequence.
    """
    p = Path(path)
    protein_ids: list[str] = []
    sequences: list[str] = []
    current_id: str | None = None
    current_seq: list[str] = []
    with p.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    protein_ids.append(current_id)
                    sequences.append("".join(current_seq).upper())
                current_id = line[1:].split()[0] if len(line) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            protein_ids.append(current_id)
            sequences.append("".join(current_seq).upper())
    return pa.table(
        {
            "protein_id": pa.array(protein_ids, type=pa.string()),
            "sequence": pa.array(sequences, type=pa.string()),
        }
    )


__all__ = [
    "_CLASSIFICATION_PRIORITY",
    "build_gene_map_from_fasta_headers",
    "classify_novel_peptides",
    "classify_single_peptide",
    "read_fasta_proteins",
    "save_novel_peptides",
]
