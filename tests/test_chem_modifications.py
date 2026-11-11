"""Tests for constellation.core.chem.modifications — UNIMOD vocab + subsetting."""

from __future__ import annotations

import pytest

from constellation.core.chem.composition import Composition
from constellation.core.chem.modifications import (
    Modification,
    ModVocab,
    Specificity,
    UNIMOD,
)


# ──────────────────────────────────────────────────────────────────────
# Built-in UNIMOD vocabulary — cartographer cross-check
# ──────────────────────────────────────────────────────────────────────


def test_unimod_loaded():
    assert len(UNIMOD) >= 1500  # ~2000 in the upstream XML


@pytest.mark.parametrize(
    "id,name_substr,delta_mass",
    [
        ("UNIMOD:1", "Acetyl", 42.010565),
        ("UNIMOD:4", "Carbamidomethyl", 57.021464),
        ("UNIMOD:21", "Phospho", 79.966331),
        ("UNIMOD:34", "Methyl", 14.015650),
        ("UNIMOD:35", "Oxidation", 15.994915),
        ("UNIMOD:36", "Dimethyl", 28.031300),
        ("UNIMOD:37", "Trimethyl", 42.046950),
        ("UNIMOD:64", "Succinyl", 100.016044),
        ("UNIMOD:121", "GG", 114.042927),  # GlyGly remnant
    ],
)
def test_unimod_anchor_values(id: str, name_substr: str, delta_mass: float):
    mod = UNIMOD[id]
    assert mod.id == id
    # name_substr appears in either name or title
    assert name_substr.lower() in mod.name.lower() or any(
        name_substr.lower() in a.lower() for a in mod.aliases
    )
    assert abs(mod.delta_mass - delta_mass) < 1e-5
    # Canonical decomposition for these light-atom mods.
    assert mod.is_canonical_decomposition is True


def test_tmt10_heavy_isotope_path():
    """UNIMOD:737 (TMT6/10plex) carries 4 ¹³C + 1 ¹⁵N — light skeleton
    cannot reproduce its mass, so mass_override is set."""
    tmt = UNIMOD["UNIMOD:737"]
    assert tmt.is_canonical_decomposition is False
    assert tmt.mass_override is not None
    assert abs(tmt.delta_mass - 229.162932) < 1e-5


def test_aliases_resolve():
    """Cartographer-era short codes still work via aliases."""
    assert UNIMOD["Ox"] is UNIMOD["UNIMOD:35"]
    assert UNIMOD["Cam"] is UNIMOD["UNIMOD:4"]
    assert UNIMOD["Ac"] is UNIMOD["UNIMOD:1"]
    assert UNIMOD["Phospho"] is UNIMOD["UNIMOD:21"]


def test_unknown_modification_raises():
    with pytest.raises(KeyError):
        UNIMOD["UNIMOD:99999999"]
    with pytest.raises(KeyError):
        UNIMOD["nonsense"]


@pytest.mark.parametrize("key", ["phospho", "PHOSPHO", "unimod:21", "Unimod:21"])
def test_a_name_or_id_is_found_in_any_case(key):
    """ProForma 2.1: readers SHOULD look CV names up case-insensitively
    where no two entries clash (none known), so `[phospho]` is Phospho."""
    assert key in UNIMOD
    assert UNIMOD[key] is UNIMOD["UNIMOD:21"]


def test_unimod_has_no_names_that_clash_once_case_is_ignored():
    folded = {}
    for mod in UNIMOD:
        for key in (mod.id, *mod.aliases):
            folded.setdefault(key.casefold(), set()).add(mod.id)
    assert not {k: ids for k, ids in folded.items() if len(ids) > 1}


def test_a_spelling_two_modifications_share_once_case_is_ignored_is_refused():
    vocab = ModVocab()
    vocab.register_custom("CUSTOM:a", "a", Composition.from_formula("C"), aliases=("Tag",))
    vocab.register_custom("CUSTOM:b", "b", Composition.from_formula("N"), aliases=("TAG",))
    # As written, each is its own.
    assert vocab["Tag"].id == "CUSTOM:a" and vocab["TAG"].id == "CUSTOM:b"
    assert "tag" not in vocab
    with pytest.raises(KeyError, match="more than one modification"):
        vocab["tag"]
    # A subset holding one of them finds it in any case.
    assert vocab.subset(["CUSTOM:a"])["tag"].id == "CUSTOM:a"


# ──────────────────────────────────────────────────────────────────────
# Subsetting — first-class API for per-tool enablement
# ──────────────────────────────────────────────────────────────────────


def test_subset_filters_supported_ids():
    enabled = UNIMOD.subset(["UNIMOD:35", "UNIMOD:4"])
    assert enabled.supports("UNIMOD:35") is True
    assert enabled.supports("UNIMOD:4") is True
    assert enabled.supports("UNIMOD:21") is False  # phospho not in subset


def test_subset_preserves_aliases():
    enabled = UNIMOD.subset(["UNIMOD:35"])
    # Alias resolves through the subset.
    assert enabled.supports("Ox") is True
    assert enabled["Ox"].id == "UNIMOD:35"


def test_subset_accepts_alias_keys():
    """Subsetting should accept either canonical ids or aliases."""
    enabled = UNIMOD.subset(["Ox", "Cam"])
    assert enabled.supports("UNIMOD:35")
    assert enabled.supports("UNIMOD:4")


def test_subset_does_not_mutate_source():
    n_before = len(UNIMOD)
    _ = UNIMOD.subset(["UNIMOD:35"])
    assert len(UNIMOD) == n_before


def test_union_combines_enabled_sets():
    a = UNIMOD.subset(["UNIMOD:35"])
    b = UNIMOD.subset(["UNIMOD:21"])
    combined = a.union(b)
    assert combined.supports("UNIMOD:35")
    assert combined.supports("UNIMOD:21")


# ──────────────────────────────────────────────────────────────────────
# find_by_mass — IO translation helper
# ──────────────────────────────────────────────────────────────────────


def test_find_by_mass_oxidation():
    """Oxidation mass 15.994915 should match within 1 mDa default."""
    matches = UNIMOD.find_by_mass(15.994915, tolerance_da=1e-3)
    ids = [m.id for m in matches]
    assert "UNIMOD:35" in ids


def test_find_by_mass_phosphorylation():
    matches = UNIMOD.find_by_mass(79.9663, tolerance_da=1e-3)
    ids = [m.id for m in matches]
    assert "UNIMOD:21" in ids


def test_find_by_mass_returns_empty_when_no_match():
    matches = UNIMOD.find_by_mass(0.123456, tolerance_da=1e-6)
    assert matches == ()


# ──────────────────────────────────────────────────────────────────────
# register_custom — escape hatch for non-UNIMOD mods
# ──────────────────────────────────────────────────────────────────────


def test_register_custom_round_trip():
    vocab = UNIMOD.subset(["UNIMOD:35"])
    custom = vocab.register_custom(
        id="CUSTOM:my-label",
        name="MyLabel",
        delta_composition=Composition.from_formula("C2H3N"),
    )
    assert vocab["CUSTOM:my-label"] is custom
    assert custom.is_canonical_decomposition is True


def test_register_custom_collision_raises():
    vocab = UNIMOD.subset(["UNIMOD:35"])
    with pytest.raises(KeyError):
        vocab.register_custom(
            id="UNIMOD:35",  # already in vocab
            name="dup",
            delta_composition=Composition.zeros(),
        )


def test_register_custom_with_mass_override():
    """Heavy-isotope custom: mass_override path."""
    vocab = ModVocab()
    vocab.register_custom(
        id="CUSTOM:13C-label",
        name="Custom 13C label",
        delta_composition=Composition.from_dict({"C": 6}),
        mass_override=72.0 + 6 * 1.00336,
    )
    mod = vocab["CUSTOM:13C-label"]
    assert mod.is_canonical_decomposition is False
    assert mod.mass_override is not None


# ──────────────────────────────────────────────────────────────────────
# Modification dataclass behavior
# ──────────────────────────────────────────────────────────────────────


def test_modification_is_frozen():
    mod = UNIMOD["UNIMOD:35"]
    with pytest.raises((AttributeError, TypeError)):
        mod.id = "X"  # type: ignore[misc]


def test_delta_mass_falls_through_to_composition():
    """Without mass_override, delta_mass should equal composition.mass."""
    mod = Modification(
        id="CUSTOM:test",
        name="test",
        delta_composition=Composition.from_formula("CH2"),
    )
    assert abs(mod.delta_mass - mod.delta_composition.mass) < 1e-12
    assert mod.is_canonical_decomposition is True


# ──────────────────────────────────────────────────────────────────────
# Save / load round-trip
# ──────────────────────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path):
    sub = UNIMOD.subset(["UNIMOD:35", "UNIMOD:21", "UNIMOD:737"])
    out = tmp_path / "vocab.json"
    sub.save(out)
    reloaded = ModVocab.load(out)
    assert len(reloaded) == 3
    for mid in ("UNIMOD:35", "UNIMOD:21", "UNIMOD:737"):
        assert reloaded[mid].id == sub[mid].id
        assert abs(reloaded[mid].delta_mass - sub[mid].delta_mass) < 1e-9
    # Heavy-isotope flag survives round-trip.
    assert reloaded["UNIMOD:737"].is_canonical_decomposition is False
    # Specificities survive round-trip.
    assert reloaded["UNIMOD:21"].specificities == sub["UNIMOD:21"].specificities


# ──────────────────────────────────────────────────────────────────────
# Specificity blocks — drive PSI / EncyclopeDIA terminal-mod localization
# ──────────────────────────────────────────────────────────────────────


def test_acetyl_specificities_cover_n_term_and_lysine():
    """Acetyl is the canonical case: distinguishable as side-chain
    K[Acetyl] vs N-terminal [Acetyl]-K via specificity inspection."""
    ac = UNIMOD["UNIMOD:1"]
    sites = {(s.position, s.site) for s in ac.specificities}
    assert ("Anywhere", "K") in sites
    assert ("Any N-term", "N-term") in sites
    assert ("Protein N-term", "N-term") in sites


def test_acetyl_predicates():
    ac = UNIMOD["UNIMOD:1"]
    assert ac.has_n_term_specificity is True
    assert ac.has_residue_specificity("K") is True
    assert ac.has_residue_specificity("R") is False
    # Acetyl is not *only* terminal — it has a K side-chain placement too.
    assert ac.has_only_terminal_specificity is False


def test_phospho_specificities_are_residue_only():
    """Phospho targets S/T/Y side chains, never termini."""
    ph = UNIMOD["UNIMOD:21"]
    sites = {(s.position, s.site) for s in ph.specificities}
    assert ("Anywhere", "S") in sites
    assert ("Anywhere", "T") in sites
    assert ("Anywhere", "Y") in sites
    assert ph.has_n_term_specificity is False
    assert ph.has_c_term_specificity is False


def test_specificity_is_frozen():
    s = Specificity(position="Anywhere", site="K")
    with pytest.raises((AttributeError, TypeError)):
        s.position = "Any N-term"  # type: ignore[misc]


def test_modification_default_specificities_is_empty_tuple():
    """Custom mods don't carry specificities by default."""
    mod = Modification(
        id="CUSTOM:test",
        name="test",
        delta_composition=Composition.from_formula("CH2"),
    )
    assert mod.specificities == ()
    # Empty specificities → predicates report False, but
    # has_only_terminal_specificity is False (empty isn't "all terminal").
    assert mod.has_n_term_specificity is False
    assert mod.has_c_term_specificity is False
    assert mod.has_only_terminal_specificity is False


# ──────────────────────────────────────────────────────────────────────
# Every UNIMOD placement (2026-11-05): hidden ones kept apart, with their
# classification and neutral losses
# ──────────────────────────────────────────────────────────────────────


def test_shown_specificities_are_what_they_were():
    """``specificities`` is still UNIMOD's shown set (what the EncyclopeDIA
    reader disambiguates with); the hidden ones are beside it."""
    shown = [s for m in UNIMOD for s in m.specificities]
    hidden = [s for m in UNIMOD for s in m.hidden_specificities]
    assert (len(shown), len(hidden)) == (79, 2747)
    assert sum(1 for m in UNIMOD if m.specificities) == 43
    assert not any(s.hidden for s in shown) and all(s.hidden for s in hidden)
    assert all(m.all_specificities == m.specificities + m.hidden_specificities for m in UNIMOD)
    # every modification now names somewhere it goes
    assert all(m.all_specificities for m in UNIMOD)


def test_hidden_placements_carry_their_chemistry():
    silac = UNIMOD["UNIMOD:259"]  # Label:13C(6)15N(2), on K
    assert silac.specificities == ()
    assert [(s.site, s.classification) for s in silac.hidden_specificities] == [("K", "Isotopic label")]
    phospho_s = next(s for s in UNIMOD["UNIMOD:21"].all_specificities if s.site == "S")
    assert [loss.mono_mass for loss in phospho_s.neutral_losses] == [0.0, 97.976896]
    h3po4 = phospho_s.neutral_losses[1].composition
    assert h3po4 is not None and h3po4.atoms == {"H": 3, "O": 4, "P": 1}
    assert {s.classification for m in UNIMOD for s in m.all_specificities} >= {
        "Post-translational", "Artefact", "Chemical derivative", "Isotopic label",
    }


def test_licenses_separates_protein_from_peptide_termini():
    # Acetyl lists both "Protein N-term" and "Any N-term": any N-terminus.
    acetyl = UNIMOD["UNIMOD:1"]
    assert acetyl.licenses("N-term", terminal_residue="A")
    assert acetyl.licenses("K") and acetyl.licenses("S")  # S is a hidden placement
    assert not acetyl.licenses("S", include_hidden=False)
    # A protein-terminal-only placement needs the protein's terminus.
    only_protein = Modification(
        id="CUSTOM:pnt",
        name="protein N-terminal only",
        delta_composition=Composition.from_formula("C2H2O"),
        specificities=(Specificity(position="Protein N-term", site="N-term"),),
    )
    assert not only_protein.licenses("N-term", terminal_residue="M")
    assert only_protein.licenses("N-term", terminal_residue="M", protein_terminus=True)
    assert Specificity(position="Protein N-term", site="N-term").is_protein_terminal
    # A terminal placement that names its residue (pyro-Glu from Q).
    pyro = UNIMOD["UNIMOD:28"]
    assert pyro.licenses("N-term", terminal_residue="Q")
    assert not pyro.licenses("N-term", terminal_residue="E")
    assert not pyro.licenses("Q")  # a terminal modification, not a side chain


def test_save_load_keeps_every_placement(tmp_path):
    sub = UNIMOD.subset(["UNIMOD:1", "UNIMOD:21", "UNIMOD:259"])
    out = tmp_path / "vocab.json"
    sub.save(out)
    reloaded = ModVocab.load(out)
    for mid in ("UNIMOD:1", "UNIMOD:21", "UNIMOD:259"):
        assert reloaded[mid].specificities == sub[mid].specificities
        assert reloaded[mid].hidden_specificities == sub[mid].hidden_specificities


def test_schema_1_vocab_files_still_load(tmp_path):
    """A vocab saved before 2026-11-05 lists shown placements only, with no
    hidden flag: they load as shown."""
    import json

    doc = {
        "schema_version": "1",
        "modifications": [
            {
                "id": "CUSTOM:x",
                "name": "x",
                "delta_composition": {"C": 1},
                "specificities": [{"position": "Anywhere", "site": "K"}],
            }
        ],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(doc))
    mod = ModVocab.load(path)["CUSTOM:x"]
    assert mod.specificities == (Specificity(position="Anywhere", site="K"),)
    assert mod.hidden_specificities == ()
    assert dict(mod.heavy_atoms) == {}  # schema 3's field, absent before


def test_unimod_keeps_the_heavy_isotopes_each_modification_names(tmp_path):
    """Review of 2026-11-10, C8: the build folded 13C / 15N into C / N and
    kept no record of them, so an envelope gave them natural variance.
    Every modification whose mass is not its skeleton's names them, and
    no other does."""
    assert dict(UNIMOD["UNIMOD:259"].heavy_atoms) == {"13C": 6, "15N": 2}  # SILAC K +8
    assert dict(UNIMOD["UNIMOD:737"].heavy_atoms) == {"13C": 4, "15N": 1}  # TMT6plex
    assert dict(UNIMOD["UNIMOD:4"].heavy_atoms) == {}
    assert all(bool(m.heavy_atoms) == (m.mass_override is not None) for m in UNIMOD)
    sub = UNIMOD.subset(["UNIMOD:259", "UNIMOD:35"])
    sub.save(tmp_path / "vocab.json")
    reloaded = ModVocab.load(tmp_path / "vocab.json")
    assert reloaded["UNIMOD:259"].heavy_atoms == sub["UNIMOD:259"].heavy_atoms
    assert reloaded["UNIMOD:259"] == sub["UNIMOD:259"] and hash(reloaded["UNIMOD:259"]) == hash(sub["UNIMOD:259"])
