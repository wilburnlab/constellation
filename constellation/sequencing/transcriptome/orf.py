"""ORF prediction over per-read transcript windows.

Mirrors NanoporeAnalysis ``utils.return_best_orf`` semantics:

    - start codon set = ``{ATG}`` only (NA's regex hardcodes ATG;
      alternative starts CTG/TTG are NOT considered, so this module
      wraps a custom CodonTable with ``starts`` narrowed to ``{ATG}``).
    - both_strands=False — strand has already been resolved by demux.
    - min_aa_length=31 — NA's check is ``len(protein) <= 30``, which
      requires strictly more than 30 AA (so ≥31).
    - longest_per_stop=True — collapse alternative starts that share a
      stop codon to the longest version.
    - Returns the LONGEST ORF as the chosen one.

Only reads whose status is ``Complete`` or ``"3' Only"`` are considered
for ORF prediction; NanoporeAnalysis explicitly skips others. The
filter is applied here so the orchestrator hands every demuxed read in
and gets back a sparse table of ORFs.
"""

from __future__ import annotations

from typing import Iterable

import pyarrow as pa

from constellation.core.sequence.nucleic import STANDARD, CodonTable, find_orfs
from constellation.sequencing.transcriptome.classify import ReadStatus
from constellation.sequencing.transcriptome.demux import ReadDemuxResult


# NanoporeAnalysis's ORF regex hardcodes ATG as the only start codon.
# constellation's STANDARD allows the canonical NCBI alternative
# starts (CTG, TTG) which would over-call ORFs at non-ATG starts.
# Build a narrowed table for parity.
_NA_PARITY_CODON_TABLE: CodonTable = CodonTable(
    transl_table=STANDARD.transl_table,
    name=f"{STANDARD.name} (ATG-only starts, NanoporeAnalysis parity)",
    forward=STANDARD.forward,
    starts=frozenset({"ATG"}),
    stops=STANDARD.stops,
)

# NanoporeAnalysis original: ``len(protein) <= 30: continue`` (keeps
# ≥31 AA). The ``_fixed1`` reprocess raised the threshold materially —
# the baseline's minimum stored Protein_length is 60, so the effective
# filter is ≥60 AA (consistent with a ``min_length=60`` + strict ``<``
# semantics). Match that here for parity with the baseline parquet.
_NA_PARITY_MIN_AA: int = 60


# Eligible read statuses for ORF prediction. NanoporeAnalysis literally
# checks ``cDNA_status in ['Complete', "3' Only"]``.
_DEFAULT_ELIGIBLE_STATUSES: frozenset[ReadStatus] = frozenset(
    {ReadStatus.COMPLETE, ReadStatus.THREE_PRIME_ONLY}
)


# Output schema — one row per ORF found per read. ``read_id`` may
# repeat in future when chimera handling lands; in S1 it's exactly
# one row per eligible read with a qualifying ORF.
ORF_TABLE: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        # 0-based half-open offsets into the chosen-orientation
        # transcript window where the ORF (start..stop inclusive,
        # half-open end = position past the stop) lives.
        pa.field("orf_start", pa.int32(), nullable=False),
        pa.field("orf_end", pa.int32(), nullable=False),
        # '+' = same orientation as the chosen demux orientation;
        # NanoporeAnalysis runs both_strands=False so this is always '+'.
        pa.field("orf_strand", pa.string(), nullable=False),
        # The nucleotide ORF (start codon through stop codon, inclusive),
        # upper-case, U → T normalised.
        pa.field("orf_nucleotide", pa.string(), nullable=False),
        # Translated protein (stop codon excluded), 1-letter AA.
        pa.field("protein", pa.string(), nullable=False),
        pa.field("protein_length", pa.int32(), nullable=False),
        # NCBI transl_table number used (1 = Standard).
        pa.field("codon_table", pa.int32(), nullable=False),
    ],
    metadata={b"schema_name": b"OrfTable"},
)


def predict_orfs(
    demux_results: Iterable[ReadDemuxResult],
    *,
    eligible_statuses: frozenset[ReadStatus] = _DEFAULT_ELIGIBLE_STATUSES,
    min_aa_length: int = _NA_PARITY_MIN_AA,
    codon_table: CodonTable = _NA_PARITY_CODON_TABLE,
) -> pa.Table:
    """Predict ORFs for each eligible read.

    ``demux_results`` is the per-read result list returned by
    :func:`constellation.sequencing.transcriptome.demux.locate_segments`
    — carrying the chosen-orientation sequence so we don't have to
    re-RC anything here. Only reads whose classification status is in
    ``eligible_statuses`` are considered; reads with no qualifying ORF
    are silently dropped from the output.

    Returns an ``ORF_TABLE``-shaped Arrow table.

    Note on internal stops. constellation's
    :func:`core.sequence.nucleic.find_orfs` regex allows in-frame stop
    codons inside the matched ORF (the lazy ``{30,}?`` quantifier
    doesn't exclude stops in the bulk). NanoporeAnalysis filters those
    post-translation by requiring ``protein.count(".") == 1`` (exactly
    the trailing stop). We replicate that filter here: any ORF whose
    translated protein contains an internal ``*`` is rejected, and we
    pick the longest survivor across all candidate ORFs.
    """
    rows: list[dict[str, object]] = []
    for r in demux_results:
        # NA-parity eligibility: exact-string match on status AND no
        # Fragment suffix. NA's check is ``cDNA_status in ['Complete',
        # "3' Only"]`` which only matches the literal short strings —
        # reads with a Fragment suffix don't get ORF prediction.
        if r.annotation.classification.status not in eligible_statuses:
            continue
        if r.annotation.classification.is_fragment:
            continue
        ts = r.annotation.transcript_start
        te = r.annotation.transcript_end
        if ts is None or te is None or te <= ts:
            continue
        transcript = r.chosen_sequence[ts:te]
        if len(transcript) < min_aa_length * 3:
            continue
        # We must filter internal-stop ORFs BEFORE constellation's
        # ``longest_per_stop`` grouping, because that grouping picks
        # the longest ORF per (frame, end) regardless of internal
        # stops — a long dirty ORF would mask a short clean one
        # ending at the same stop. So enumerate all candidates first.
        all_orfs = find_orfs(
            transcript,
            codon_table=codon_table,
            min_aa_length=min_aa_length,
            both_strands=False,
            longest_per_stop=False,
        )
        # Filter to ORFs with NO internal stops (NanoporeAnalysis-
        # parity: protein.count('.') == 1 in their convention; for us
        # the trailing stop is already stripped via translate's
        # ``[:-3]``, so a clean ORF has no '*' in its protein).
        clean_orfs = [o for o in all_orfs if "*" not in o.protein]
        if not clean_orfs:
            continue
        # Per-stop longest (replicates NA's
        # ``longest_orfs[(strand, end)] = max-by-Protein_length``).
        # When multiple clean ORFs share the same stop position,
        # earliest-start wins (longest protein).
        per_stop_best: dict[tuple[str, int], object] = {}
        for o in clean_orfs:
            key = (o.strand, o.end)
            prior = per_stop_best.get(key)
            if prior is None or o.length > prior.length:  # type: ignore[attr-defined]
                per_stop_best[key] = o
        # Overall longest survivor.
        orf = max(per_stop_best.values(), key=lambda o: o.length)  # type: ignore[arg-type]
        rows.append(
            {
                "read_id": r.read_id,
                "orf_start": orf.start,
                "orf_end": orf.end,
                "orf_strand": orf.strand,
                "orf_nucleotide": orf.nucleotide,
                "protein": orf.protein,
                "protein_length": orf.length,
                "codon_table": orf.transl_table,
            }
        )

    if not rows:
        return ORF_TABLE.empty_table()
    return pa.Table.from_pylist(rows, schema=ORF_TABLE)


__all__ = [
    "ORF_TABLE",
    "predict_orfs",
]
