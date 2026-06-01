"""Detection-results schemas — peptide-level and protein-level scores.

Sibling to :mod:`massspec.library` (theoretical / sample-agnostic) and
:mod:`massspec.quant` (empirical / per-acquisition abundances).

A search engine produces *scores* — q-values, posterior error
probabilities, percolator scores — for the entities it identified in a
given acquisition. Constellation models that with two Arrow tables:

    PEPTIDE_SCORE_TABLE   one row per (peptide, acquisition, engine)
    PROTEIN_SCORE_TABLE   one row per (protein, acquisition, engine)
    PSM_TABLE             one row per peptide-spectrum match

PSM-level scores (one row per spectrum) live in ``PSM_TABLE``. It is
populated by readers whose source genuinely emits per-spectrum matches —
MaxQuant ``msms.txt`` (via ``massspec.io.maxquant``) is the first such
driver. Aggregating-only formats — e.g. the encyclopedia file format,
which records peptide/protein-level scores rather than per-spectrum
matches — leave ``Search.psms`` empty rather than manufacture PSM rows
that would invent data.

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
# Per-spectrum matches (PSMs)
# ──────────────────────────────────────────────────────────────────────


# One row per peptide-spectrum match, as emitted by a search engine that
# reports per-spectrum results (MaxQuant ``msms.txt`` is the first
# driver). The primary key is ``(raw_file, psm_id)``: ``psm_id`` is the
# engine's per-export row id (MaxQuant's ``id`` is contiguous 0..N-1
# *within one export*, so it collides across the one-export-per-run
# merges), and ``(raw_file, scan)`` is NOT unique because a single MS/MS
# scan can yield more than one match (MaxQuant ``Type=MULTI-SECPEP``).
#
# The table is built to enable **scan-level join-back** to a converted
# acquisition bundle: ``(raw_file, scan)`` joins ``SCAN_METADATA_TABLE``
# (same int32 ``scan``), and the ``mass_analyzer`` / ``fragmentation``
# columns cross-validate ``SCAN_METADATA_TABLE.analyzer`` /
# ``activation_type``.
#
# NOTE on ``peptide_id``: this is the *engine's internal* peptide index
# (e.g. MaxQuant's ``Peptide ID`` joining its own ``peptides.txt``). It
# is NOT a Constellation ``Library.peptides.peptide_id`` and must NEVER
# be fed into ``Search.validate_against(library)`` as a library FK.
PSM_TABLE: pa.Schema = pa.schema(
    [
        # identity / acquisition join-back
        pa.field("psm_id", pa.int64(), nullable=False),
        pa.field("raw_file", pa.string(), nullable=False),
        # nullable pre-link: resolved against the Acquisitions table.
        pa.field("acquisition_id", pa.int64(), nullable=True),
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("precursor_scan", pa.int32(), nullable=True),
        # peptide identity
        pa.field("sequence", pa.string(), nullable=False),
        # ProForma 2.0 (incl. any reconstructed fixed mods); null when the
        # source modseq couldn't be resolved.
        pa.field("modified_sequence", pa.string(), nullable=True),
        # engine-internal peptide index — see NOTE above (NOT a library FK).
        pa.field("peptide_id", pa.int64(), nullable=True),
        pa.field("mod_peptide_id", pa.int64(), nullable=True),
        pa.field("evidence_id", pa.int64(), nullable=True),
        # ";"-delimited protein accessions, verbatim from the source.
        pa.field("proteins", pa.string(), nullable=True),
        pa.field("charge", pa.int8(), nullable=False),
        # measured vs theoretical
        pa.field("mz", pa.float64(), nullable=True),
        pa.field("mass", pa.float64(), nullable=True),
        pa.field("mass_error_ppm", pa.float64(), nullable=True),
        # SECONDS (normalized from MaxQuant's minutes) to match
        # SCAN_METADATA_TABLE.rt / MS_PEAK_TABLE.rt.
        pa.field("retention_time_s", pa.float64(), nullable=True),
        # acquisition context (cross-check vs SCAN_METADATA_TABLE)
        # stored verbatim: fragmentation is UPPER (HCD/CID/ETD), whereas
        # SCAN_METADATA_TABLE.activation_type is lowercase.
        pa.field("fragmentation", pa.string(), nullable=True),
        pa.field("mass_analyzer", pa.string(), nullable=True),
        pa.field("psm_type", pa.string(), nullable=True),
        # scores / confidence
        pa.field("score", pa.float64(), nullable=True),
        pa.field("delta_score", pa.float64(), nullable=True),
        pa.field("pep", pa.float64(), nullable=True),  # posterior error probability
        pa.field("is_decoy", pa.bool_(), nullable=False),
        # derived: any "CON__" accession in ``proteins`` (MaxQuant's
        # contaminant convention; there is no dedicated msms.txt column).
        pa.field("is_contaminant", pa.bool_(), nullable=False),
        # engine name: "maxquant", ... (matches PEPTIDE_SCORE_TABLE.engine).
        pa.field("engine", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"PsmTable", b"schema_version": b"1"},
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
register_schema("PsmTable", PSM_TABLE)
register_schema("NovelPeptideTable", NOVEL_PEPTIDE_TABLE)


__all__ = [
    "NOVEL_PEPTIDE_TABLE",
    "PEPTIDE_SCORE_TABLE",
    "PROTEIN_SCORE_TABLE",
    "PSM_TABLE",
]
