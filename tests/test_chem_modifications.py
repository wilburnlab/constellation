"""Tests for constellation.core.chem.modifications — UNIMOD vocab + subsetting."""

from __future__ import annotations

import pytest

from constellation.core.chem.composition import Composition
from constellation.core.chem.modifications import (
    Modification,
    ModVocab,
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
