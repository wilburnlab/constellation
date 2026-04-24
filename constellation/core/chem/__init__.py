"""Elemental primitives — atoms, compositions, isotopes, modifications.

Purely chemical; knows nothing about residues, sequences, or alphabets.
Downstream `core.sequence` builds residue/peptide compositions *from*
`Composition` primitives defined here.

Modules (TODO; scaffolded only):
    atoms            - atom types, monoisotopic masses, natural abundances
    composition      - Composition atom-count tensor (torch); +/*/mass/formula
    isotopes         - isotope_distribution(composition, n_peaks) via FFT
    modifications    - ModVocab + UNIMOD delta compositions
"""
