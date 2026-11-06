"""Per-model peptide tokenizers.

Encoding peptides into integer-token tensors is a per-model decision:
each model commits to the characters it reads, the id of each, how a
modified residue or terminus is spelled, and a fixed width. This package
holds those conventions, apart from the chemistry: ``peptide/`` is
unconditional physics, ``tokenize/`` ties it to a model's input contract.

Modules:
    vocabulary    - ``Sites`` (a peptidoform as residues and the UNIMOD
                    accessions at each position, refusing what has no
                    such reading), ``SiteTokens`` (the plain
                    accession-keyed spelling), ``Vocabulary`` (characters
                    to ids, vectorised ``encode`` into padded int64 rows)
                    and the refusal reasons.
    jchronologer  - the tokens JChronologer (EncyclopeDIA's predictor)
                    gives its Chronologer, Cartographer, Electrician and
                    Sculptor models, ported from jchronologer 1.0.10 and
                    checked against the jar's own tokens; built from the
                    models' preprocessing files, which must match the
                    snapshot ``data/jchronologer_models.json``.

Nothing here imports torch.
"""

from constellation.massspec.tokenize.vocabulary import (
    LENGTH_OUT_OF_RANGE,
    REASONS,
    UNKNOWN_RESIDUE,
    UNSUPPORTED_FORM,
    UNSUPPORTED_MODIFICATION,
    Sites,
    SiteTokens,
    TokenizationRefused,
    TokenizedBatch,
    Vocabulary,
)

__all__ = [
    "LENGTH_OUT_OF_RANGE",
    "REASONS",
    "UNKNOWN_RESIDUE",
    "UNSUPPORTED_FORM",
    "UNSUPPORTED_MODIFICATION",
    "SiteTokens",
    "Sites",
    "TokenizationRefused",
    "TokenizedBatch",
    "Vocabulary",
]
