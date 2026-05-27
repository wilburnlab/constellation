"""Detection-results schemas — peptide-level and protein-level scores.

Sibling to :mod:`massspec.library` (theoretical / sample-agnostic) and
:mod:`massspec.quant` (empirical / per-acquisition abundances).

A search engine produces *scores* — q-values, posterior error
probabilities, percolator scores — for the entities it identified in a
given acquisition. Constellation models that with two Arrow tables:

    PEPTIDE_SCORE_TABLE   one row per (peptide, acquisition, engine)
    PROTEIN_SCORE_TABLE   one row per (protein, acquisition, engine)

PSM-level (one row per spectrum) is **deliberately deferred** — the
encyclopedia file format aggregates at the peptide/protein level, not
per spectrum, so manufacturing PSM rows from it would invent data.
``PSM_TABLE`` lands when an actual PSM-emitting reader (mzIdentML,
the Counter port) drives the schema design.

``acquisition_id`` is nullable to support run-agnostic scores
(e.g. inference-time peptide priors that don't tie to a specific
acquisition).
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema


PEPTIDE_SCORE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("peptide_id", pa.int64(), nullable=False),
        # Nullable: a missing acquisition_id means run-agnostic score.
        pa.field("acquisition_id", pa.int64(), nullable=True),
        pa.field("score", pa.float64(), nullable=True),
        pa.field("qvalue", pa.float64(), nullable=True),
        pa.field("pep", pa.float64(), nullable=True),  # posterior error probability
        # Engine name: "encyclopedia", "scribe", "msfragger", "counter", ...
        # Disambiguates when a downstream consumer aggregates results
        # from multiple search engines on the same library.
        pa.field("engine", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"PeptideScoreTable"},
)


PROTEIN_SCORE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("protein_id", pa.int64(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=True),
        pa.field("score", pa.float64(), nullable=True),
        pa.field("qvalue", pa.float64(), nullable=True),
        pa.field("engine", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"ProteinScoreTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Novel peptide classification output
# ──────────────────────────────────────────────────────────────────────


# Output of ``classify_novel_peptides`` — one row per unique detected
# novel peptide sequence, classified against an mmseqs2 alignment of
# the novel proteome vs the reference proteome (+ optionally
# Swiss-Prot). Per-peptide deduplication keeps the most-canonical
# classification (priority ordering from ``massspec.search.novel.
# _CLASSIFICATION_PRIORITY``: snp < insertion < deletion < complex <
# truncations < trypsin_cutsite_mutation < deviations < unknown <
# non_reference).
#
# Within the winning class, the row whose source ORF has the highest
# abundance (``protein_abundance`` map, typically mean TPM) is chosen
# as the representative — see ``transcript_id`` and the supporting-
# transcripts pair below for the full multi-mapping context.
NOVEL_PEPTIDE_TABLE: pa.Schema = pa.schema(
    [
        # Canonical AA sequence (no modifications) — the cartographer
        # algorithm operates on canonical sequences only.
        pa.field("peptide_sequence", pa.string(), nullable=False),
        # ProForma 2.0 string, when the peptide is bound to a Library
        # row that carries modseq info. Null otherwise.
        pa.field("modified_sequence", pa.string(), nullable=True),
        # One of the 11 classes from _CLASSIFICATION_PRIORITY.
        pa.field("classification", pa.string(), nullable=False),
        # Aligned reference substring at the peptide's locus
        # (CIGAR-walk derived). Empty string when not applicable
        # (e.g. n_term_deviation, non_reference, unknown).
        pa.field("ref_seq", pa.string(), nullable=True),
        # Novel protein accession (from novel_proteins input) — the
        # ORF identifier emitted by the upstream sequencing module.
        pa.field("protein_id", pa.string(), nullable=False),
        # Reference protein accession the alignment found (mmseqs2 hit
        # target). Null when there's no hit for the novel protein.
        pa.field("ref_protein_id", pa.string(), nullable=True),
        # Gene symbol, populated from gene_map if provided.
        pa.field("gene", pa.string(), nullable=True),
        # mmseqs2 CIGAR string for the hit (query-centric, 1-indexed
        # inclusive coordinates). Empty when no hit.
        pa.field("cigar", pa.string(), nullable=True),
        # Result of the classifier's membership check: ``"reference"``
        # when the alignment target is in the reference proteome the
        # classifier was handed (CIGAR-walked into snp/indel/...);
        # ``"non_reference"`` when it isn't (the peptide's
        # ``classification`` short-circuits to ``"non_reference"``).
        # Null only on rows assembled outside the classifier. NOTE:
        # the values intentionally use cartographer's vocabulary
        # rather than ``ALIGNMENT_HIT_TABLE``'s upstream
        # ``"refseq"`` / ``"swissprot"`` — the classifier always
        # recomputes from membership, so foreign tier labels are
        # replaced. The ``[aligned_to=...]`` tag in the
        # transcriptome→proteome combined.fasta header is computed
        # separately at the orchestrator level using
        # ``refseq``/``swissprot`` membership; the two columns are
        # complementary, not identical.
        pa.field("aligned_to", pa.string(), nullable=True),
        # Full novel protein sequence — present for downstream
        # validation / re-classification without needing the
        # novel-proteins FASTA at hand.
        pa.field("novel_protein_seq", pa.string(), nullable=True),
        # Full reference protein sequence at the aligned target.
        pa.field("ref_protein_seq", pa.string(), nullable=True),
        # The ORF / transcript whose representative was kept after
        # the most-abundant tie-break across all ORFs that produce
        # this peptide. Typically equal to ``protein_id``; differs
        # when the long-read workflow renames per-cluster ORFs.
        pa.field("transcript_id", pa.string(), nullable=True),
        # Number of distinct source ORFs / transcripts the peptide
        # could have come from (rows considered before the most-
        # abundant pick). Always >= 1.
        pa.field("n_transcripts_supporting", pa.int64(), nullable=True),
        # Semicolon-delimited list of all source ORFs the peptide
        # could have come from, e.g. ``"P213;P788"``. Includes the
        # winning ``transcript_id``. Consumers `.split(";")` to
        # recover the set. Preserves the full multi-mapping info.
        pa.field("supporting_transcripts", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"NovelPeptideTable"},
)


register_schema("PeptideScoreTable", PEPTIDE_SCORE_TABLE)
register_schema("ProteinScoreTable", PROTEIN_SCORE_TABLE)
register_schema("NovelPeptideTable", NOVEL_PEPTIDE_TABLE)


__all__ = [
    "NOVEL_PEPTIDE_TABLE",
    "PEPTIDE_SCORE_TABLE",
    "PROTEIN_SCORE_TABLE",
]
