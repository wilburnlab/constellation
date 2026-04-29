"""Tests for constellation.core.sequence.protein."""

from __future__ import annotations

import pytest

from constellation.core.chem.composition import Composition
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


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────


def test_registry_loaded_with_expected_proteases():
    """Standard MS proteomics enzymes plus a few extras."""
    expected = {
        "Trypsin",
        "Trypsin/P",
        "LysC",
        "LysC/P",
        "LysN",
        "ArgC",
        "Chymotrypsin",
        "Chymotrypsin/low",
        "AspN",
        "AspN+N",
        "GluC",
        "GluC_bicarb",
        "Pepsin_pH1.3",
        "Pepsin_pH2",
        "ProteinaseK",
        "Thermolysin",
        "No_enzyme",
    }
    assert set(PROTEASES.ids()) == expected


def test_registry_get_by_id():
    p = PROTEASES["Trypsin"]
    assert isinstance(p, Protease)
    assert p.id == "Trypsin"


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        PROTEASES["NoSuchEnzyme"]


def test_registry_register_custom():
    """Custom proteases land in a separate registry instance —
    the global PROTEASES singleton is shared and shouldn't be mutated
    in tests, so we use a fresh registry."""
    reg = ProteaseRegistry()
    reg.register_custom(
        id="CustomTryp",
        regex_pattern=r"(?<=[KR])(?!P)",
        name="Custom Trypsin clone",
    )
    assert "CustomTryp" in reg


def test_registry_collision_raises():
    reg = ProteaseRegistry()
    reg.register_custom(id="Foo", regex_pattern=r"(?<=K)")
    with pytest.raises(KeyError):
        reg.register_custom(id="Foo", regex_pattern=r"(?<=R)")


def test_registry_save_load_round_trip(tmp_path):
    reg = ProteaseRegistry()
    reg.register_custom(id="X", regex_pattern=r"(?<=K)", description="x")
    out = tmp_path / "p.json"
    reg.save(out)
    loaded = ProteaseRegistry.load(out)
    assert "X" in loaded
    assert loaded["X"].regex_pattern == r"(?<=K)"


# ──────────────────────────────────────────────────────────────────────
# cleave_sites
# ──────────────────────────────────────────────────────────────────────


def test_cleave_sites_anchors():
    """Site list always starts with 0 and ends with len(seq)."""
    sites = cleave_sites("PEPTIDEK", "Trypsin")
    assert sites[0] == 0
    assert sites[-1] == 8


def test_cleave_sites_trypsin_proline_rule():
    """K-P should NOT cleave under Trypsin; K-A should."""
    no_cut = cleave_sites("PEPKPMAGE", "Trypsin")
    assert no_cut == [0, 9]  # K-P doesn't cut; no other K/R
    cut = cleave_sites("PEPKAMAGE", "Trypsin")
    assert 4 in cut


def test_cleave_sites_trypsin_no_proline_variant():
    """Trypsin/P cuts at K even before P."""
    cut = cleave_sites("PEPKPMAGE", "Trypsin/P")
    assert 4 in cut


# ──────────────────────────────────────────────────────────────────────
# cleave
# ──────────────────────────────────────────────────────────────────────


def test_cleave_trypsin_basic():
    seq = "PEPTIDEKMASTERPRVTEINR"
    peps = cleave(seq, "Trypsin", missed_cleavages=0, min_length=3, max_length=None)
    assert "PEPTIDEK" in peps
    assert "MASTERPR" in peps  # R-V cut at position 16
    assert "VTEINR" in peps


def test_cleave_missed_cleavages():
    seq = "PEPTIDEKAALGRWVRK"  # cuts after K, R, R, K
    peps_0 = cleave(seq, "Trypsin", missed_cleavages=0, min_length=3, max_length=None)
    peps_1 = cleave(seq, "Trypsin", missed_cleavages=1, min_length=3, max_length=None)
    # 1-missed must include all 0-missed plus combined spans
    assert set(peps_0).issubset(set(peps_1))
    assert len(peps_1) > len(peps_0)


def test_cleave_min_max_length():
    """Length filtering is inclusive on both ends."""
    seq = "AKMKCKDKEK"
    short_peps = cleave(seq, "Trypsin", missed_cleavages=0, min_length=2, max_length=2)
    # All cleaved fragments are length 2 (e.g. "AK", "MK", "CK", "DK", "EK").
    assert all(len(p) == 2 for p in short_peps)


def test_cleave_n_to_c_ordering():
    """Peptides come back in N→C order along the protein."""
    seq = "AAAAKGGGGRCCCCRDDDD"  # cuts after K@4, R@9, R@14
    peps_with_spans = cleave(
        seq,
        "Trypsin",
        missed_cleavages=0,
        min_length=2,
        max_length=None,
        return_spans=True,
    )
    starts = [p.start for p in peps_with_spans]
    assert starts == sorted(starts)


def test_cleave_dedup_by_first_occurrence():
    """Same sequence at two positions appears once, in first-encounter order."""
    seq = "AKGKAK"  # cuts at 2, 4 → AK, GK, AK
    peps = cleave(seq, "Trypsin", missed_cleavages=0, min_length=2, max_length=None)
    assert peps.count("AK") == 1
    assert peps.index("AK") < peps.index("GK")


def test_cleave_return_spans_keeps_duplicates():
    seq = "AKGKAK"
    spans = cleave(
        seq,
        "Trypsin",
        missed_cleavages=0,
        min_length=2,
        max_length=None,
        return_spans=True,
    )
    ak_spans = [s for s in spans if s.sequence == "AK"]
    assert len(ak_spans) == 2  # both occurrences kept


def test_cleave_peptide_records():
    spans = cleave(
        "AKGR",
        "Trypsin",
        missed_cleavages=0,
        min_length=2,
        max_length=None,
        return_spans=True,
    )
    assert all(isinstance(s, Peptide) for s in spans)
    ak = next(s for s in spans if s.sequence == "AK")
    assert ak.start == 0
    assert ak.end == 2
    assert ak.n_missed == 0


def test_cleave_lysc():
    """LysC cuts only after K."""
    peps = cleave("AKGR", "LysC", missed_cleavages=0, min_length=2, max_length=None)
    assert peps == ["AK", "GR"]


def test_cleave_aspn_cuts_before_d():
    """AspN: N-terminal to D — peptide ends at D-1, next starts at D."""
    spans = cleave(
        "PEPTIDEKDPRVTEIN",
        "AspN",
        missed_cleavages=0,
        min_length=3,
        max_length=None,
        return_spans=True,
    )
    # Should have a span starting at the D positions
    starts = sorted({s.start for s in spans})
    # D appears at index 5 and 8 → cuts at those positions
    assert 5 in starts or any(s.sequence.startswith("D") for s in spans)


def test_cleave_no_enzyme_returns_whole_sequence():
    peps = cleave("PEPTIDE", "No_enzyme", missed_cleavages=0, min_length=3, max_length=None)
    assert peps == ["PEPTIDE"]


def test_cleave_rejects_invalid_residue():
    with pytest.raises(ValueError):
        cleave("PEPTIDEZ", "Trypsin")


def test_cleave_negative_missed_raises():
    with pytest.raises(ValueError):
        cleave("PEPTIDEK", "Trypsin", missed_cleavages=-1)


def test_cleave_semi_specific_emits_n_terminal_truncations():
    """Semi-specific keeps the C-terminal cut, floats the N-terminal."""
    seq = "PEPTIDEK"
    peps = cleave(
        seq,
        "Trypsin",
        missed_cleavages=0,
        min_length=3,
        max_length=None,
        semi_specific=True,
    )
    # Fully-specific PEPTIDEK plus suffixes EPTIDEK, PTIDEK, TIDEK, IDEK, DEK
    assert "PEPTIDEK" in peps
    assert "EPTIDEK" in peps
    assert "DEK" in peps


# ──────────────────────────────────────────────────────────────────────
# peptide_composition / peptide_mass
# ──────────────────────────────────────────────────────────────────────


def _pf(modseq: str):
    """Tiny shim: parse a ProForma string into a Peptidoform for tests."""
    from constellation.core.sequence.proforma import parse_proforma

    return parse_proforma(modseq)


def test_peptide_composition_returns_composition():
    comp = peptide_composition(_pf("PEPTIDE"))
    assert isinstance(comp, Composition)


def test_peptide_composition_includes_terminal_water():
    """Composition should equal sum(residues) + H2O. PEPTIDE with terminal
    H2O is C34H53N7O15."""
    comp = peptide_composition(_pf("PEPTIDE"))
    assert comp.atoms == {"C": 34, "H": 53, "N": 7, "O": 15}


def test_peptide_mass_peptide_anchor():
    """PEPTIDE monoisotopic mass = 799.35996 (ExPASy reference)."""
    assert abs(peptide_mass(_pf("PEPTIDE")) - 799.35996) < 1e-3


def test_peptide_mass_glycine_alone():
    """Single G + H2O = 75.03203 (free glycine)."""
    assert abs(peptide_mass(_pf("G")) - 75.032028) < 1e-3


@pytest.mark.parametrize(
    "seq,expected_mass",
    [
        # ExPASy reference values (sum residue masses + H2O).
        ("PEPTIDE", 799.35996),
        ("MASTER", 693.31157),
        ("GAVLI", 471.30567),
        ("CYSTEINE", 957.37496),
    ],
)
def test_peptide_mass_known_values(seq: str, expected_mass: float):
    """Cross-check against published peptide masses (ExPASy values)."""
    assert abs(peptide_mass(_pf(seq)) - expected_mass) < 1e-2


def test_peptide_mass_modification_oxidation():
    """Adding UNIMOD:35 (oxidation) on M shifts mass by +15.994915."""
    plain = peptide_mass(_pf("PEPTIDEM"))
    oxed = peptide_mass(_pf("PEPTIDEM[UNIMOD:35]"))
    assert abs((oxed - plain) - 15.994915) < 1e-5


def test_peptide_mass_modification_phospho():
    plain = peptide_mass(_pf("PEPTIDES"))
    phos = peptide_mass(_pf("PEPTIDES[UNIMOD:21]"))
    assert abs((phos - plain) - 79.966331) < 1e-5


def test_peptide_mass_tmt10_heavy_isotope():
    """TMT10 (UNIMOD:737) carries 4 13C + 1 15N — mass_override path
    must apply, giving the canonical 229.162932 delta."""
    plain = peptide_mass(_pf("PEPTIDEK"))
    tmt = peptide_mass(_pf("PEPTIDEK[UNIMOD:737]"))
    assert abs((tmt - plain) - 229.162932) < 1e-4


def test_peptide_mass_multiple_modifications():
    """Two mods apply independently — compare same-residue-letter peptidoforms."""
    delta_ox = 15.994915
    delta_ph = 79.966331
    plain = peptide_mass(_pf("PEPTMES"))
    both = peptide_mass(_pf("PEPTM[UNIMOD:35]ES[UNIMOD:21]"))
    assert abs((both - plain) - (delta_ox + delta_ph)) < 1e-5


def test_peptide_mass_alias_resolves():
    """UNIMOD aliases ('Ox', 'Phospho') route through the vocab."""
    by_id = peptide_mass(_pf("PEPTIDEM[UNIMOD:35]"))
    by_alias = peptide_mass(_pf("PEPTIDEM[Oxidation]"))
    assert abs(by_id - by_alias) < 1e-9


def test_peptide_mass_unknown_modification_raises():
    with pytest.raises(KeyError):
        peptide_mass(_pf("PEPTIDEM[UNIMOD:99999999]"))


def test_peptide_mass_modification_index_out_of_range_raises():
    """Manual Peptidoform construction with an out-of-range mod index."""
    from constellation.core.sequence.proforma import (
        ModRef,
        Peptidoform,
        TaggedMod,
    )

    bad = Peptidoform(
        sequence="PEPTIDE",
        residue_mods={
            99: (TaggedMod(mod=ModRef(cv="UNIMOD", accession="35")),),
        },
    )
    with pytest.raises(IndexError):
        peptide_mass(bad)


def test_peptide_mass_average_mass_path():
    """Average-mass path uses Composition.average_mass; for PEPTIDE the
    expected average is 799.83 (1 sig dig past the decimal is enough
    given standard atomic weights)."""
    avg = peptide_mass(_pf("PEPTIDE"), monoisotopic=False)
    assert abs(avg - 799.83) < 0.5


def test_protein_composition_alias_of_peptide_composition():
    a = protein_composition("MASTER")
    b = peptide_composition(_pf("MASTER"))
    assert a == b


# ──────────────────────────────────────────────────────────────────────
# Cross-package handoff to core.chem
# ──────────────────────────────────────────────────────────────────────


def test_peptide_composition_feeds_isotope_envelope():
    """peptide_composition → core.chem.isotope_envelope works end-to-end."""
    from constellation.core.chem.isotopes import isotope_envelope

    comp = peptide_composition(_pf("PEPTIDE"))
    masses, intens = isotope_envelope(comp, n_peaks=5)
    assert masses.shape == (5,)
    assert intens.shape == (5,)
    # Monoisotopic peak should be the largest for a small peptide.
    assert intens[0] == intens.max()
