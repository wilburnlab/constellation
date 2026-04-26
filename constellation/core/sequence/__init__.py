"""Primary-sequence math — alphabets, generic ops, nucleic, protein.

Imports `core.chem`. Never imports `core.structure` or `core.io`.
No experimental observables (quality scores, intensities, RTs) live
here — those belong to the sequencing/MS domain modules.

Degenerate-alphabet rule: canonical alphabets expose chem compositions;
degenerate (IUPAC) alphabets do not. Functions that depend on
compositions gate on `alphabet.degenerate` via `requires_canonical`;
functions that tolerate degeneracy declare `degenerate_ok`.

Modules:
    alphabets   - Alphabet ABC + DNA/RNA/AA + IUPAC variants, residue
                  compositions, degeneracy expansion + complement tables,
                  `requires_canonical` / `degenerate_ok` decorators.
    ops         - identify_alphabet, validate, normalize, kmerize,
                  sliding_window, hamming_distance,
                  parse_modified_sequence / format_modified_sequence.
    nucleic     - reverse_complement, complement, CodonTable + 7 NCBI
                  tables, translate (degenerate-codon-aware, pluggable
                  table), find_orfs / best_orf, gc_content.
    protein     - Protease + ProteaseRegistry (17 built-in enzymes),
                  cleave / cleave_sites, peptide_composition,
                  peptide_mass, protein_composition.
"""

from constellation.core.sequence.alphabets import (
    AA,
    AA_IUPAC,
    ALPHABETS,
    COMPLEMENT_DNA,
    COMPLEMENT_DNA_IUPAC,
    COMPLEMENT_RNA,
    COMPLEMENT_RNA_IUPAC,
    DEGENERATE_AA,
    DEGENERATE_DNA,
    DEGENERATE_RNA,
    DNA,
    DNA_IUPAC,
    RNA,
    RNA_IUPAC,
    Alphabet,
    canonical_for,
    degenerate_ok,
    expand_token,
    expansion_table,
    requires_canonical,
)
from constellation.core.sequence.nucleic import (
    CODON_TABLES,
    STANDARD,
    CodonTable,
    Orf,
    best_orf,
    complement,
    find_orfs,
    gc_content,
    reverse_complement,
    translate,
)
from constellation.core.sequence.ops import (
    format_modified_sequence,
    hamming_distance,
    identify_alphabet,
    kmerize,
    normalize,
    parse_modified_sequence,
    sliding_window,
    validate,
)
from constellation.core.sequence.protein import (
    PROTEASES,
    Peptide,
    Protease,
    ProteaseRegistry,
    cleave,
    cleave_sites,
    peptide_composition,
    peptide_mass,
    protein_composition,
)

__all__ = [
    # alphabets
    "Alphabet",
    "DNA",
    "RNA",
    "AA",
    "DNA_IUPAC",
    "RNA_IUPAC",
    "AA_IUPAC",
    "ALPHABETS",
    "DEGENERATE_DNA",
    "DEGENERATE_RNA",
    "DEGENERATE_AA",
    "COMPLEMENT_DNA",
    "COMPLEMENT_RNA",
    "COMPLEMENT_DNA_IUPAC",
    "COMPLEMENT_RNA_IUPAC",
    "requires_canonical",
    "degenerate_ok",
    "canonical_for",
    "expansion_table",
    "expand_token",
    # ops
    "identify_alphabet",
    "validate",
    "normalize",
    "kmerize",
    "sliding_window",
    "hamming_distance",
    "parse_modified_sequence",
    "format_modified_sequence",
    # nucleic
    "complement",
    "reverse_complement",
    "CodonTable",
    "CODON_TABLES",
    "STANDARD",
    "translate",
    "Orf",
    "find_orfs",
    "best_orf",
    "gc_content",
    # protein
    "Protease",
    "ProteaseRegistry",
    "PROTEASES",
    "Peptide",
    "cleave",
    "cleave_sites",
    "peptide_composition",
    "peptide_mass",
    "protein_composition",
]
