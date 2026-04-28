"""Per-model peptide tokenizers (scaffold).

Encoding peptide sequences into integer-token tensors is a per-model
decision: each downstream MS model (Prosit-style intensity predictors,
Chronologer-style RT, future PSM scorers) commits to a vocabulary tied
to a specific checkpoint. The vocab is a `core.chem.modifications.UNIMOD`
subset plus terminal-token / padding conventions; the tokenizer pairs
with the model.

Lives sibling to `peptide/` rather than inside it because tokenization
is a layered concern above the chemistry — `peptide/` is unconditional
physics, `tokenize/` ties physics to a specific model's input contract.

Modules (TODO):
    vocabulary       - Vocabulary class wrapping a `ModVocab.subset(...)`
                       plus terminal / pad / unknown token policy.
    tokenizer        - Tokenizer base + sequence_to_tensor / tensor_to_sequence.
    prosit_2020      - Vocab + tokenizer paired with Prosit_2020_intensity_HCD.
    chronologer      - Vocab + tokenizer paired with Chronologer_RT.
"""
