"""Batched peptide composition and mass, held bit for bit to the
single-peptidoform functions as they were (``tests/_peptide_mass_oracle.py``)."""

from __future__ import annotations

import random

import pytest
import torch

import _peptide_mass_oracle as oracle
from constellation.core.sequence.proforma import MultiPeptidoform, Peptidoform, parse_proforma
from constellation.core.sequence.protein import (
    enumerate_modforms,
    peptide_composition,
    peptide_compositions,
    peptide_mass,
    peptide_masses,
)

EDGE_CASES = [
    "<13C><15N>PEPTIDEK",
    "<13C>PEPTIDE[+10.0]K",
    "PEPTIDEK[UNIMOD:259]",  # SILAC: mass_override
    "[UNIMOD:1]-K[UNIMOD:1]VERPD",
    "PEPT[Formula:C2H2O]IDE",
    "<[UNIMOD:4]@C>PEPCTCIDE",
    "[Phospho]^2?PEPTIDES",
    "{Phospho}PEPTIDE",
    "PEPTIDE-[Methyl]",
    "EM[Oxidation]EVT(ES)[Phospho]PEK",
    "S[Phospho#g1]PEPT[#g1]IDE",  # one group, counted once
    "PEPTIDE[+15.9949]",
    "[Acetyl]-PEPTIDE-[Amidated]",
    "MEMEM[UNIMOD:35]",
    "",
]


def _forms(n: int = 600, seed: int = 7) -> list[Peptidoform]:
    rng = random.Random(seed)
    out: list[Peptidoform] = []
    while len(out) < n:
        seq = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(rng.randint(5, 30)))
        forms = enumerate_modforms(
            seq,
            fixed={"C": "UNIMOD:4"},
            variable={"M": "UNIMOD:35", "S": "UNIMOD:21", "N-term": "UNIMOD:1"},
            max_variable=2,
            validate_specificity=False,
        )
        out.extend(forms[:3])
    return out[:n] + [parse_proforma(s) if s else Peptidoform(sequence="") for s in EDGE_CASES]


def test_masses_are_the_old_bits() -> None:
    forms = _forms()
    expected = [oracle.peptide_mass(f) for f in forms]
    got = peptide_masses(forms)
    assert got.dtype == torch.float64 and got.shape == (len(forms),)
    assert got.tolist() == expected
    assert [peptide_mass(f) for f in forms] == expected


def test_average_masses_are_the_old_bits() -> None:
    forms = [f for f in _forms(200) if not f.global_isotopes]
    expected = [oracle.peptide_mass(f, monoisotopic=False) for f in forms]
    assert peptide_masses(forms, monoisotopic=False).tolist() == expected


def test_compositions_are_the_old_counts() -> None:
    forms = [
        f for f in _forms(300)
        if not any(
            tm.mod is not None and tm.mod.mass_delta is not None and tm.mod.accession is None
            for tms in f.residue_mods.values() for tm in tms
        )
    ]  # fmt: skip
    expected = [oracle.peptide_composition(f).counts.tolist() for f in forms]
    got = peptide_compositions(forms)
    assert got.dtype == torch.int32
    assert got.tolist() == expected
    assert [peptide_composition(f).counts.tolist() for f in forms] == expected


def test_empty_batch() -> None:
    assert peptide_masses([]).shape == (0,)
    assert peptide_compositions([]).shape[0] == 0


def test_same_errors() -> None:
    with pytest.raises(ValueError, match="not in alphabet"):
        peptide_masses([Peptidoform("PEPTIDE"), Peptidoform("PEPXIDE")])
    with pytest.raises(ValueError, match="bare mass-delta"):
        peptide_compositions([parse_proforma("PEPTIDE[+10.0]")])
    with pytest.raises(IndexError):
        peptide_masses([Peptidoform("PEP", residue_mods={9: ()})])
    with pytest.raises(ValueError, match="monoisotopic"):
        peptide_masses([parse_proforma("<13C>PEPTIDE")], monoisotopic=False)
    with pytest.raises(NotImplementedError):
        peptide_masses([MultiPeptidoform(chains=(Peptidoform("PEP"), Peptidoform("TIDE")))])
    with pytest.raises(KeyError):
        peptide_masses([parse_proforma("PEPT[UNIMOD:999999]IDE")])
