"""Theoretical / sample-agnostic spectral-library schemas.

A spectral library models the protein → peptide → precursor → fragment
DAG. These five schemas hold *theoretical* structure plus *predicted*
entity-level properties (RT, CCS, fragment intensities) that are tied
to entity identity rather than any single acquisition. Empirical /
sample-specific data — abundances, calibrated transmission efficiencies,
search scores — lives in ``massspec.quant`` and ``massspec.search``.

    PROTEIN_TABLE              one row per protein (sequence-level)
    PEPTIDE_TABLE              one row per (sequence, modification) pair
    PRECURSOR_TABLE            one row per (peptide, charge); has
                               peptide_id FK + predicted RT/CCS
    LIBRARY_FRAGMENT_TABLE     one row per (precursor, ion_type,
                               position, charge, loss); has predicted
                               intensity. Column names align with
                               ``massspec.schemas.FRAGMENT_ION_TABLE``
                               so a library fragment row projects
                               cleanly via ``cast_to_schema``.
    PROTEIN_PEPTIDE_EDGE       pure adjacency for the only true M:N
                               relationship (peptides shared across
                               proteoforms). No efficiency column —
                               calibrated efficiencies live in
                               ``massspec.quant``.

A library built from a FASTA (``massspec.library.search_space``) also
records how each peptide was made, in three tables a library read from a
file leaves empty (2026-11-05):

    PEPTIDE_SPAN_TABLE         one row per place a peptide sequence was cut
                               from a protein: coordinates, missed
                               cleavages, how each end was made, flanks
    SPAN_PEPTIDE_EDGE          which spans can make which peptidoform (a
                               Protein N-term form only from a span at the
                               protein's N-terminus)
    PEPTIDE_MODIFICATION_TABLE each modification of a peptidoform and why it
                               is there (fixed, variable, label; the rule)

``PROTEIN_PEPTIDE_EDGE`` is then the projection of ``SPAN_PEPTIDE_EDGE``
onto (protein, peptide). ``is_decoy`` on proteins and peptides, and a
peptide's ``decoy_of`` / ``decoy_set``, are null where a source says
nothing about decoys.

Sentinel ``-1.0`` for unset float fields (rt_predicted, ccs_predicted,
intensity_predicted) — same convention as the rest of the package.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema

# ──────────────────────────────────────────────────────────────────────
# Tier tables
# ──────────────────────────────────────────────────────────────────────


PROTEIN_TABLE: pa.Schema = pa.schema(
    [
        pa.field("protein_id", pa.int64(), nullable=False),
        pa.field("accession", pa.string(), nullable=False),
        # Nullable: EncyclopeDIA dlib files don't always carry the
        # parent-protein sequence. Constellation-side library construction
        # populates it whenever available; downstream code must tolerate null.
        pa.field("sequence", pa.string(), nullable=True),
        pa.field("description", pa.string(), nullable=True),
        # True for a decoy protein (reversed before digestion); null where
        # the source does not say.
        pa.field("is_decoy", pa.bool_(), nullable=True),
    ],
    metadata={b"schema_name": b"ProteinTable"},
)


PEPTIDE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("peptide_id", pa.int64(), nullable=False),
        # Canonical-stripped (e.g. "PEPTIDE")
        pa.field("sequence", pa.string(), nullable=False),
        # ProForma 2.0 string (HUPO-PSI Final draft 15, Feb 2022). Round-tripped
        # via core.sequence.proforma.parse_proforma / format_proforma. Equals
        # ``sequence`` when unmodified.
        pa.field("modified_sequence", pa.string(), nullable=False),
        # Decoys: whether this peptidoform is one, the target peptidoform a
        # peptide-level decoy was made from, and which decoy set it belongs
        # to (0 the set that competes with the targets, 1.. extra sets such
        # as entrapment). Null where the source does not say.
        pa.field("is_decoy", pa.bool_(), nullable=True),
        pa.field("decoy_of", pa.int64(), nullable=True),
        pa.field("decoy_set", pa.int16(), nullable=True),
    ],
    metadata={b"schema_name": b"PeptideTable"},
)


PRECURSOR_TABLE: pa.Schema = pa.schema(
    [
        pa.field("precursor_id", pa.int64(), nullable=False),
        pa.field("peptide_id", pa.int64(), nullable=False),
        pa.field("charge", pa.int32(), nullable=False),
        pa.field("precursor_mz", pa.float64(), nullable=False),
        # -1.0 sentinel = no prediction available
        pa.field("rt_predicted", pa.float64(), nullable=False),
        pa.field("ccs_predicted", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"PrecursorTable"},
)


# Column names that overlap ``FRAGMENT_ION_TABLE`` (ion_type, position,
# charge, loss_id, mz_theoretical) are deliberately identical — a row
# from this table projects cleanly into a FragmentIonTable view.
LIBRARY_FRAGMENT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("precursor_id", pa.int64(), nullable=False),
        # ion_type / position / charge are NULL when the source carried
        # a raw annotation string we couldn't structurally resolve —
        # e.g. an internal-fragment / immonium / named-loss annotation
        # that mzPAF Phase 1 doesn't decode, or a peak whose
        # (type, position, charge, loss) tuple is biochemistry-masked
        # by LOSS_REGISTRY. The raw ``annotation`` string is still
        # preserved so downstream tools can re-parse with future
        # grammar support without re-reading the source.
        pa.field("ion_type", pa.int8(), nullable=True),
        pa.field("position", pa.int32(), nullable=True),
        pa.field("charge", pa.int32(), nullable=True),
        pa.field("loss_id", pa.string(), nullable=True),
        pa.field("mz_theoretical", pa.float64(), nullable=False),
        # -1.0 sentinel = no prediction. Model-predicted libraries
        # normalise to [0, 1]; empirical-source libraries (e.g. NIST
        # .msp, .dlib) may carry raw observed counts and project to a
        # normalised view downstream. float64 matches mzPeak's choice
        # for intensity arrays — modern Orbitrap/Astral dynamic range
        # exceeds float32's 2**24 exact-integer ceiling, so silent
        # quantisation of large peaks is avoided.
        pa.field("intensity_predicted", pa.float64(), nullable=False),
        pa.field("annotation", pa.string(), nullable=True),
        # An empirical library's word on each fragment: how well its
        # chromatogram correlated with the precursor's in the run it was
        # taken from, and whether the library marks it as one to quantify
        # with (EncyclopeDIA: ``entries.CorrelationArray`` and
        # ``QuantifiedIonsArray``, one value per stored peak). Null where
        # the source says nothing (a predicted library has no
        # correlations). EncyclopeDIA searches a chromatogram library with
        # the correlations: written back without them, the lab's library
        # gave a run 30,682 precursors, 27,035 of them among the 31,895 it
        # gives as the jar wrote it. The flags changed nothing in that
        # search, and the jar sets every one in every file it writes
        # (2026-10-18).
        pa.field("correlation", pa.float64(), nullable=True),
        pa.field("quantified", pa.bool_(), nullable=True),
    ],
    metadata={b"schema_name": b"LibraryFragmentTable"},
)


PROTEIN_PEPTIDE_EDGE: pa.Schema = pa.schema(
    [
        pa.field("protein_id", pa.int64(), nullable=False),
        pa.field("peptide_id", pa.int64(), nullable=False),
    ],
    metadata={b"schema_name": b"ProteinPeptideEdge"},
)


# One row per place a peptide sequence was cut from a protein. Coordinates
# are 0-based, end exclusive. ``n_term`` / ``c_term`` say how each end was
# made: "enzyme" (the protease's rule), "protein" (the protein's own
# terminus: its first residue, after an initiator Met, or a declared mature
# chain; its last residue), "nonspecific" (semi- or unspecific digestion).
# ``preceding`` / ``following`` are the flanking residues, "-" at a protein
# end.
PEPTIDE_SPAN_TABLE: pa.Schema = pa.schema(
    [
        pa.field("protein_id", pa.int64(), nullable=False),
        pa.field("start", pa.int32(), nullable=False),
        pa.field("end", pa.int32(), nullable=False),
        pa.field("sequence", pa.string(), nullable=False),
        pa.field("missed_cleavages", pa.int16(), nullable=False),
        pa.field("n_term", pa.string(), nullable=False),
        pa.field("c_term", pa.string(), nullable=False),
        pa.field("preceding", pa.string(), nullable=False),
        pa.field("following", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"PeptideSpanTable"},
)


# Which spans can make which peptidoform: the span's sequence is the
# peptidoform's, and its context licenses the peptidoform's modifications.
SPAN_PEPTIDE_EDGE: pa.Schema = pa.schema(
    [
        pa.field("protein_id", pa.int64(), nullable=False),
        pa.field("start", pa.int32(), nullable=False),
        pa.field("end", pa.int32(), nullable=False),
        pa.field("peptide_id", pa.int64(), nullable=False),
    ],
    metadata={b"schema_name": b"SpanPeptideEdge"},
)


# Each modification of a peptidoform and why it is there. ``position`` is the
# 0-based residue for a side chain; ``terminus`` is "N" / "C" for a terminal
# modification (position null); both are null for a global isotope label.
# ``modification`` is the UNIMOD accession ("UNIMOD:35") or the isotope
# ("13C"). ``origin`` is "fixed", "variable" or "label"; ``rule`` is the rule
# that placed it, as given ("Oxidation@M", "Acetyl@Protein N-term").
PEPTIDE_MODIFICATION_TABLE: pa.Schema = pa.schema(
    [
        pa.field("peptide_id", pa.int64(), nullable=False),
        pa.field("position", pa.int32(), nullable=True),
        pa.field("terminus", pa.string(), nullable=True),
        pa.field("modification", pa.string(), nullable=False),
        pa.field("origin", pa.string(), nullable=False),
        pa.field("rule", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"PeptideModificationTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────


register_schema("ProteinTable", PROTEIN_TABLE)
register_schema("PeptideTable", PEPTIDE_TABLE)
register_schema("PrecursorTable", PRECURSOR_TABLE)
register_schema("LibraryFragmentTable", LIBRARY_FRAGMENT_TABLE)
register_schema("ProteinPeptideEdge", PROTEIN_PEPTIDE_EDGE)
register_schema("PeptideSpanTable", PEPTIDE_SPAN_TABLE)
register_schema("SpanPeptideEdge", SPAN_PEPTIDE_EDGE)
register_schema("PeptideModificationTable", PEPTIDE_MODIFICATION_TABLE)


__all__ = [
    "PROTEIN_TABLE",
    "PEPTIDE_TABLE",
    "PRECURSOR_TABLE",
    "LIBRARY_FRAGMENT_TABLE",
    "PROTEIN_PEPTIDE_EDGE",
    "PEPTIDE_SPAN_TABLE",
    "SPAN_PEPTIDE_EDGE",
    "PEPTIDE_MODIFICATION_TABLE",
]
