"""Primary-sequence math — alphabets, generic ops, nucleic, protein.

Imports `core.chem`. Never imports `core.structure` or `core.io`.
No experimental observables (quality scores, intensities, RTs) live
here — those belong to the sequencing/MS domain modules.

Degenerate-alphabet rule: canonical alphabets expose chem compositions;
degenerate (IUPAC) alphabets do not. Functions that depend on
compositions gate on `alphabet.degenerate` via `requires_canonical`;
functions that tolerate degeneracy declare `degenerate_ok`.

Modules (TODO; scaffolded only):
    alphabets        - Alphabet ABC + DNA/RNA/AA/CODON + IUPAC variants
    ops              - identify_alphabet, tokenize, kmerize (alphabet-aware)
    nucleic          - reverse_complement, translate (pluggable codon tables,
                       degenerate-codon-aware), find_orfs, best_orf
    protein          - digest (trypsin + friends); requires_canonical=True
"""
