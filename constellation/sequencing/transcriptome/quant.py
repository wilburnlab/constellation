"""Protein × Sample count-matrix builder — final stage of the
transcriptomics demux pipeline.

NanoporeAnalysis's ``extract_proteins.py`` filter:

    cDNA status == 'Complete'  AND  Protein.is_valid()

Reads passing both contribute one count per (Protein, Sample_ID) cell.
We mirror that here, taking the demux + ORF tables as inputs and
emitting:

    - feature_quant Arrow table (FEATURE_QUANT-shaped — one row per
      (protein, sample) pair with a non-zero count)
    - protein FASTA records (unique protein sequences with P0/P1/...
      labels in count-descending order — matches NA's labelling)
    - count-matrix TSV bytes (Protein × Sample matrix, columns
      ordered by sample-count descending)

The output FASTA + TSV are designed to compare cleanly to NA's
``qJS00x_proteins.fasta`` / ``qJS00x_protein-counts.tab`` after
canonicalising column / row order (sort by protein sequence for the
TSV; FASTA set-equality on (label-free, sequence) pairs).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pyarrow as pa

from constellation.sequencing.samples import Samples


# Per-row count record. Shape mirrors FEATURE_QUANT but specialised to
# the cDNA workflow: feature_id is the protein label (P0, P1, ...);
# sample_id is the M:N-resolved sample.
PROTEIN_COUNT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("protein_label", pa.string(), nullable=False),
        pa.field("protein_sequence", pa.string(), nullable=False),
        pa.field("sample_id", pa.int64(), nullable=False),
        pa.field("sample_name", pa.string(), nullable=False),
        pa.field("count", pa.int64(), nullable=False),
    ],
    metadata={b"schema_name": b"ProteinCountTable"},
)


@dataclass(frozen=True)
class FastaRecord:
    """One entry in the proteins FASTA: ``>label`` then sequence."""

    label: str
    sequence: str


def _is_complete_non_fragment(status: str, is_fragment: bool) -> bool:
    """NA's filter: literal ``cDNA status == 'Complete'``. The Fragment
    suffix would break the literal-string match in NA, so we exclude
    is_fragment=True here too."""
    return status == "Complete" and not is_fragment


def build_protein_count_matrix(
    demux_table: pa.Table,
    orf_table: pa.Table,
    samples: Samples,
    *,
    min_protein_count: int = 2,
) -> tuple[pa.Table, list[FastaRecord], str]:
    """Aggregate per-read demux + ORF outputs into a Protein × Sample
    count matrix.

    Returns ``(quant_table, fasta_records, tsv_text)`` where:

    - ``quant_table`` conforms to ``PROTEIN_COUNT_TABLE`` and holds
      one row per ``(protein, sample)`` pair with non-zero count.
    - ``fasta_records`` is a list of :class:`FastaRecord` for every
      unique protein, ordered as ``P0..PN`` matches NA's labelling
      (count-descending across all samples).
    - ``tsv_text`` is the ``protein_counts.tab`` content as a string,
      ready to write directly. Columns are ``Protein`` + sample names
      (count-descending) + ``Sequence``; rows are ``P0..PN`` in the
      same order. NA's pandas-emitted leading index column is
      reproduced for byte-equivalence with the baseline.

    Reads are filtered to ``cDNA status == 'Complete' AND
    is_fragment == False AND sample_id IS NOT NULL``; reads without a
    valid ORF (no row in ``orf_table`` for that ``read_id``) are
    silently dropped.

    ``min_protein_count`` (default 2) drops singleton proteins from
    the FASTA / TSV outputs. The ``_fixed1`` baseline is generated
    with the singleton filter enabled (proteins with total-read-count
    < 2 are excluded), so we default to that for parity. Set to ``1``
    to keep singletons; set to ``0`` to keep everything (including
    proteins seen 0 times, which doesn't happen in practice).
    """
    # 1) Build (read_id -> protein) map from ORF table.
    orf_by_read: dict[str, str] = {}
    for row in orf_table.to_pylist():
        orf_by_read[row["read_id"]] = row["protein"]

    # 2) Build (sample_id -> sample_name) map.
    sample_name_by_id: dict[int, str] = {
        sid: sn
        for sid, sn in zip(
            samples.samples.column("sample_id").to_pylist(),
            samples.samples.column("sample_name").to_pylist(),
        )
    }

    # 3) Walk demux rows; for each Complete + non-Fragment + sample-
    #    assigned + ORF-having read, accumulate (protein, sample) count.
    #    NanoporeAnalysis's record-iteration order is stable insertion
    #    order across the parquet; preserving that lets the count-
    #    descending sort produce the same column order on ties.
    pair_counts: Counter[tuple[str, int]] = Counter()
    protein_counts: Counter[str] = Counter()
    sample_counts: Counter[int] = Counter()
    protein_first_seen: dict[str, int] = {}
    sample_first_seen: dict[int, int] = {}

    for i, row in enumerate(demux_table.to_pylist()):
        if not _is_complete_non_fragment(row["status"], row["is_fragment"]):
            continue
        sample_id = row["sample_id"]
        if sample_id is None:
            continue
        protein = orf_by_read.get(row["read_id"])
        if protein is None:
            continue
        pair_counts[(protein, sample_id)] += 1
        protein_counts[protein] += 1
        sample_counts[sample_id] += 1
        protein_first_seen.setdefault(protein, i)
        sample_first_seen.setdefault(sample_id, i)

    # 4) Order samples by their pre-filter Sample_ID counts —
    #    NanoporeAnalysis's ``samples = list(counts['Sample_ID'])`` is
    #    computed from ALL Complete+Protein reads regardless of the
    #    protein-singleton filter. Every sample with ≥1 such read
    #    appears as a column even if all its surviving proteins are
    #    zero (singleton-only samples become all-zero columns).
    samples_ordered_full = sorted(
        sample_counts.keys(),
        key=lambda s: (-sample_counts[s], sample_first_seen[s]),
    )

    # 5) Singleton-protein filter (NA's _fixed1 enables this). Drop
    #    proteins below the threshold from the pair_counts / labelling,
    #    but keep the full sample column set.
    if min_protein_count > 0:
        keep = {p for p, n in protein_counts.items() if n >= min_protein_count}
        pair_counts = Counter(
            {(p, s): n for (p, s), n in pair_counts.items() if p in keep}
        )
        protein_counts = Counter({p: n for p, n in protein_counts.items() if p in keep})

    # 6) Order proteins by their (filtered) total count (descending),
    #    tie-break by first-seen.
    proteins_ordered = sorted(
        protein_counts.keys(),
        key=lambda p: (-protein_counts[p], protein_first_seen[p]),
    )
    samples_ordered = samples_ordered_full

    protein_labels: dict[str, str] = {
        p: f"P{idx}" for idx, p in enumerate(proteins_ordered)
    }

    # 5) FASTA records (P0..PN in count-descending order).
    fasta = [
        FastaRecord(label=protein_labels[p], sequence=p)
        for p in proteins_ordered
    ]

    # 6) Long-form quant table (one row per non-zero (protein, sample)).
    quant_rows = []
    for (protein, sample_id), count in pair_counts.items():
        quant_rows.append(
            {
                "protein_label": protein_labels[protein],
                "protein_sequence": protein,
                "sample_id": sample_id,
                "sample_name": sample_name_by_id.get(sample_id, f"sample_{sample_id}"),
                "count": count,
            }
        )
    if quant_rows:
        quant_table = pa.Table.from_pylist(quant_rows, schema=PROTEIN_COUNT_TABLE)
    else:
        quant_table = PROTEIN_COUNT_TABLE.empty_table()

    # 7) Wide-form TSV (mirrors NA's pandas to_csv output).
    sample_names_ordered = [sample_name_by_id[s] for s in samples_ordered]
    header = "\t".join(["", "Protein", *sample_names_ordered, "Sequence"])
    lines = [header]
    for idx, protein in enumerate(proteins_ordered):
        row_cells = [str(idx), protein_labels[protein]]
        for sample_id in samples_ordered:
            n = pair_counts.get((protein, sample_id), 0)
            # NA's pandas writes integer counts as floats ('0.0'), so
            # match that for byte-equivalence.
            row_cells.append(f"{float(n):.1f}")
        row_cells.append(protein)
        lines.append("\t".join(row_cells))
    tsv_text = "\n".join(lines) + "\n"

    return quant_table, fasta, tsv_text


def fasta_records_to_text(records: list[FastaRecord]) -> str:
    """Render ``FastaRecord``s to a FASTA string. One sequence per
    line (no wrapping) — matches the format NanoporeAnalysis writes
    via ``write_fastx``.
    """
    return "".join(f">{r.label}\n{r.sequence}\n" for r in records)


__all__ = [
    "PROTEIN_COUNT_TABLE",
    "FastaRecord",
    "build_protein_count_matrix",
    "fasta_records_to_text",
]
