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


register_schema("PeptideScoreTable", PEPTIDE_SCORE_TABLE)
register_schema("ProteinScoreTable", PROTEIN_SCORE_TABLE)


__all__ = [
    "PEPTIDE_SCORE_TABLE",
    "PROTEIN_SCORE_TABLE",
]
