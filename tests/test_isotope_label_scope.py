"""What a global isotope label covers (2026-11-05, PI): the peptide chain.

ProForma 2.0 §4.6.1 reads ``<13C>`` as "13C on all residues"; an atom with
no isotope stated takes the natural distribution (Formula Rule 3). So a
label covers the residues and termini, a modification's added atoms are
natural, and the atoms a modification removes come off the labelled chain.
Precursor mass, fragment ladders and isotope envelopes all follow it.
"""

from __future__ import annotations

import pytest
import torch

from constellation.core.chem.elements import ELEMENTS, PROTON_MASS
from constellation.core.sequence.proforma import parse_proforma
from constellation.core.sequence.protein import (
    labelled_compositions,
    peptide_composition,
    peptide_mass,
    peptide_masses,
)
from constellation.massspec.peptide.envelope import neutral_envelope
from constellation.massspec.peptide.ions import IonType, fragment_ladder

C13 = ELEMENTS["C"].isotope_mass(13) - ELEMENTS["C"].monoisotopic_mass
N15 = ELEMENTS["N"].isotope_mass(15) - ELEMENTS["N"].monoisotopic_mass


def _mass(text: str) -> float:
    return peptide_mass(parse_proforma(text))


def test_added_atoms_are_natural():
    # light carbamidomethyl on a labelled peptide: the label's shift is the chain's
    chain = _mass("<13C><15N>ACDEK") - _mass("ACDEK")
    assert _mass("<13C><15N>AC[UNIMOD:4]DEK") - _mass("AC[UNIMOD:4]DEK") == pytest.approx(chain, abs=1e-9)
    assert chain == pytest.approx(21 * C13 + 6 * N15, abs=1e-9)


def test_removed_atoms_come_off_the_labelled_chain():
    # deamidated N is D: the N lost was a 15N
    assert _mass("<15N>GN[UNIMOD:7]K") == pytest.approx(_mass("<15N>GDK"), abs=1e-9)
    # pyro-Glu from Q and from E is one molecule, labelled or not
    assert _mass("<13C><15N>[UNIMOD:28]-QK") == pytest.approx(_mass("<13C><15N>[UNIMOD:27]-EK"), abs=1e-9)


def test_labelled_compositions_counts_the_chain_atoms_of_labelled_elements():
    forms = [parse_proforma(t) for t in ("<15N>AC[UNIMOD:4]N[UNIMOD:7]K", "AC[UNIMOD:4]K", "<13C>K")]
    got = labelled_compositions(forms)
    c, n = ELEMENTS.index("C"), ELEMENTS.index("N")
    # A C N K has 1 + 1 + 2 + 2 = 6 N, deamidation takes one; CAM's N is natural
    assert int(got[0, n]) == 5 and int(got[0, c]) == 0
    assert not got[1].any()
    assert int(got[2, c]) == 6 and int(got[2, n]) == 0
    assert torch.equal(
        peptide_masses(forms), torch.tensor([peptide_mass(f) for f in forms], dtype=torch.float64)
    )


def _ladder(pf, losses=None):
    table, _ = fragment_ladder(
        pf, ion_types=(IonType.B, IonType.Y), neutral_losses=losses, return_tensor=False
    )
    return table.select(["position", "ion_type", "loss_id", "mz_theoretical"]).to_pylist()


@pytest.mark.parametrize(
    "text", ["<13C><15N>AC[UNIMOD:4]DEK", "<15N>GN[UNIMOD:7]QK", "<13C><15N>[UNIMOD:28]-QPEC[UNIMOD:4]K"]
)
def test_fragments_add_up_to_the_precursor(text):
    """A b ion and the y ion of the same bond, at 1+, are the precursor's
    mass plus two protons: the ladder and the precursor agree on what the
    label covers."""
    pf = parse_proforma(text)
    rows = [r for r in _ladder(pf) if r["loss_id"] is None]
    b = {r["position"]: r["mz_theoretical"] for r in rows if r["ion_type"] == int(IonType.B)}
    y = {r["position"]: r["mz_theoretical"] for r in rows if r["ion_type"] == int(IonType.Y)}
    assert b.keys() == y.keys() and len(b) == len(pf.sequence) - 1
    for bond in b:
        assert b[bond] + y[bond] == pytest.approx(peptide_mass(pf) + 2 * PROTON_MASS, abs=1e-6)


def test_an_ammonia_loss_from_a_15n_fragment_loses_15n():
    rows = _ladder(parse_proforma("<15N>PEPTIDEK"), losses=["NH3"])
    y = {r["position"]: r["mz_theoretical"] for r in rows if r["ion_type"] == int(IonType.Y) and r["loss_id"] is None}
    lossy = [r for r in rows if r["loss_id"] == "NH3"]
    assert lossy
    for r in lossy:
        assert y[r["position"]] - r["mz_theoretical"] == pytest.approx(
            ELEMENTS["N"].isotope_mass(15) + 3 * ELEMENTS["H"].monoisotopic_mass, abs=1e-6
        )


@pytest.mark.parametrize("mode", ["binned", "exact"])
def test_a_labelled_envelope_is_the_natural_atoms_envelope_moved(mode):
    """With every C and N labelled, only the CAM's atoms (and H, O, S) vary."""
    labelled = parse_proforma("<13C><15N>AC[UNIMOD:4]DEK")
    masses, ab = neutral_envelope(labelled, n_peaks=4, mode=mode)
    assert float(masses[0]) == pytest.approx(peptide_mass(labelled), abs=1e-6)
    light = parse_proforma("AC[UNIMOD:4]DEK")
    _, ab_light = neutral_envelope(light, n_peaks=4, mode=mode)
    assert float(ab[1]) < float(ab_light[1])  # fewer atoms vary
    comp = peptide_composition(labelled)
    assert comp.atoms["C"] == 23  # the composition itself is unchanged


# PEPC[CAM]K: 24 C and 6 N in the chain, carbamidomethyl adds C2 N; SILAC's
# heavy atoms (UNIMOD:259 13C6 15N2, 188 13C6) are lysine's own, TMT6plex's
# (UNIMOD:737, 13C4 15N, twice) atoms it adds.
@pytest.mark.parametrize(
    "text, natural",
    [
        ("PEPC[UNIMOD:4]K", {"C": 26, "N": 7}),  # nothing labelled
        ("<13C>PEPC[UNIMOD:4]K", {"C": 2, "N": 7}),  # the global label alone: the chain's C
        ("PEPC[UNIMOD:4]K[UNIMOD:259]", {"C": 20, "N": 5}),  # SILAC alone
        ("<13C>PEPC[UNIMOD:4]K[UNIMOD:259]", {"C": 2, "N": 5}),  # both: lysine's C once, CAM's C natural
        ("<13C><[UNIMOD:259]@K>PEPC[UNIMOD:4]K", {"C": 2, "N": 5}),  # SILAC as a fixed modification
        ("<13C><15N>PEPC[UNIMOD:4]K[UNIMOD:259]", {"C": 2, "N": 1}),
        ("<15N>PEPC[UNIMOD:4]K[UNIMOD:188]", {"C": 20, "N": 1}),  # label and substitution of other elements
        ("<13C>[UNIMOD:737]-PEPC[UNIMOD:4]K[UNIMOD:737]", {"C": 18, "N": 9}),  # TMT's heavy atoms are not the chain's
    ],
)
def test_an_atom_a_label_and_a_heavy_modification_both_fix_counts_once(text, natural):
    """Codex review of the scorer: a global label, SILAC and carbamidomethyl
    together took SILAC's heavy atoms off twice (once as the label's chain,
    once as the modification's), and the clamp at zero took the deficit from
    carbamidomethyl's natural carbons, so the envelope varied over none. A
    substitution's heavy atoms are its residue's own: of a labelled element,
    the label covers them already."""
    import numpy as np

    from constellation.core.chem.composition import Composition
    from constellation.core.chem.isotopes import isotope_envelope
    from constellation.core.sequence.protein import natural_abundance_compositions
    from constellation.massspec.peptide.envelope import neutral_envelopes

    pf = parse_proforma(text)
    batched = natural_abundance_compositions([parse_proforma("LVNELTEFAK"), pf])[1]
    single = natural_abundance_compositions([pf])[0]
    assert torch.equal(batched, single)
    for symbol, n in natural.items():
        assert int(single[ELEMENTS.index(symbol)]) == n, symbol
    comp = peptide_composition(pf).counts  # what is fixed is left out; the composition is not changed
    assert torch.equal(single[ELEMENTS.index("H")], comp[ELEMENTS.index("H")])
    # The envelope, one and batched, varies over those atoms and sits at the peptide's mass.
    masses, shares = neutral_envelope(pf, n_peaks=4, normalize="full")
    _, want = isotope_envelope(Composition(single), n_peaks=4, normalize="full")
    np.testing.assert_allclose(shares.numpy(), want.numpy(), rtol=1e-12)
    assert float(masses[0]) == pytest.approx(peptide_mass(pf), abs=1e-9)
    many_masses, many_shares = neutral_envelopes([parse_proforma("LVNELTEFAK"), pf], n_peaks=4)
    assert np.array_equal(many_masses[1], masses.numpy())
    np.testing.assert_allclose(many_shares[1], shares.numpy(), atol=1e-6)
