"""Tests for `constellation.massspec.peptide.neutral_losses`.

Covers the LOSS_REGISTRY contents, registry mutation (`add`,
`register_custom`), JSON round-trip, subsetting, and the `loss_applies`
biochemical-rule predicate.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from constellation.core.chem.composition import Composition
from constellation.massspec.peptide.neutral_losses import (
    LOSS_REGISTRY,
    LossRegistry,
    NeutralLoss,
    loss_applies,
)


def test_registry_loaded_at_import():
    """Singleton must contain the four shipped losses."""
    ids = LOSS_REGISTRY.ids()
    assert "H2O" in ids
    assert "NH3" in ids
    assert "HPO3" in ids
    assert "H3PO4" in ids
    assert len(LOSS_REGISTRY) == 4


def test_loss_delta_mass():
    h2o = LOSS_REGISTRY["H2O"]
    assert math.isclose(h2o.delta_mass, 18.010565, abs_tol=1e-3)
    nh3 = LOSS_REGISTRY["NH3"]
    assert math.isclose(nh3.delta_mass, 17.026549, abs_tol=1e-3)


def test_subset_returns_new_registry():
    sub = LOSS_REGISTRY.subset(["H2O", "NH3"])
    assert len(sub) == 2
    assert "H2O" in sub
    assert "HPO3" not in sub
    # Source unaffected
    assert "HPO3" in LOSS_REGISTRY


def test_register_custom():
    reg = LossRegistry()
    reg.register_custom(
        "GLUTHATIONE",
        Composition.from_formula("C10H17N3O6S"),
        triggering_residues=["C"],
        description="Glutathione adduct removal",
    )
    assert "GLUTHATIONE" in reg
    g = reg["GLUTHATIONE"]
    assert g.triggering_residues == frozenset({"C"})
    assert g.description == "Glutathione adduct removal"


def test_collision_raises():
    reg = LossRegistry()
    reg.register_custom("FOO", Composition.from_formula("H2O"))
    with pytest.raises(KeyError, match="already registered"):
        reg.register_custom("FOO", Composition.from_formula("H2O"))


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="unknown neutral loss"):
        LOSS_REGISTRY.get("NOT_A_LOSS")


def test_save_and_load_round_trip(tmp_path: Path):
    out = tmp_path / "out.json"
    LOSS_REGISTRY.save(out)
    reloaded = LossRegistry.load(out)
    assert set(reloaded.ids()) == set(LOSS_REGISTRY.ids())
    for loss_id in LOSS_REGISTRY.ids():
        a = LOSS_REGISTRY[loss_id]
        b = reloaded[loss_id]
        assert a.id == b.id
        assert a.name == b.name
        assert a.triggering_residues == b.triggering_residues
        assert a.triggering_mods == b.triggering_mods
        assert a.applies_to_ion_types == b.applies_to_ion_types
        assert math.isclose(a.delta_mass, b.delta_mass, abs_tol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# loss_applies — predicate semantics
# ──────────────────────────────────────────────────────────────────────


def test_h2o_applies_to_serine_containing_fragment():
    h2o = LOSS_REGISTRY["H2O"]
    assert loss_applies(
        h2o,
        ion_type_name="B",
        fragment_residues=["P", "E", "S"],
        fragment_mods={},
    )


def test_h2o_no_apply_when_no_trigger_residue():
    h2o = LOSS_REGISTRY["H2O"]
    assert not loss_applies(
        h2o,
        ion_type_name="B",
        fragment_residues=["P", "G", "A"],
        fragment_mods={},
    )


def test_nh3_applies_to_K():
    nh3 = LOSS_REGISTRY["NH3"]
    assert loss_applies(
        nh3,
        ion_type_name="Y",
        fragment_residues=["I", "K"],
        fragment_mods={},
    )


def test_hpo3_requires_phospho_mod():
    hpo3 = LOSS_REGISTRY["HPO3"]
    # No phospho → not applicable even with S in fragment
    assert not loss_applies(
        hpo3,
        ion_type_name="B",
        fragment_residues=["S", "T"],
        fragment_mods={},
    )
    # With phospho → applicable
    assert loss_applies(
        hpo3,
        ion_type_name="B",
        fragment_residues=["S", "T"],
        fragment_mods={0: "UNIMOD:21"},
    )


def test_h3po4_requires_phospho_on_S_or_T():
    """H3PO4 requires the Phospho mod to actually sit on S or T (not Y)."""
    h3po4 = LOSS_REGISTRY["H3PO4"]
    # Phospho on S (position 0) → applicable.
    assert loss_applies(
        h3po4,
        ion_type_name="B",
        fragment_residues=["S", "A", "M"],
        fragment_mods={0: "UNIMOD:21"},
    )
    # Phospho on Y (no S/T at all) → not applicable.
    assert not loss_applies(
        h3po4,
        ion_type_name="B",
        fragment_residues=["Y", "A", "M"],
        fragment_mods={0: "UNIMOD:21"},
    )
    # S in fragment but phospho on a different residue (G is not phosphorylatable
    # but for the predicate we just check the position) — because phospho is on
    # position 1 which is A (not in S/T), should be invalid.
    assert not loss_applies(
        h3po4,
        ion_type_name="B",
        fragment_residues=["S", "A", "M"],
        fragment_mods={1: "UNIMOD:21"},
    )


def test_ion_type_restriction():
    """A loss with applies_to_ion_types restriction must respect it."""
    custom = NeutralLoss(
        id="B_ONLY",
        name="b-only loss",
        delta_composition=Composition.from_formula("CO"),
        triggering_residues=frozenset(),
        triggering_mods=frozenset(),
        applies_to_ion_types=frozenset({"B"}),
    )
    assert loss_applies(
        custom, ion_type_name="B", fragment_residues=["A"], fragment_mods={}
    )
    assert not loss_applies(
        custom, ion_type_name="Y", fragment_residues=["A"], fragment_mods={}
    )


def test_unconditional_loss_always_applies():
    """A loss with no triggers and no ion-type restriction is always valid."""
    always = NeutralLoss(
        id="UNIVERSAL",
        name="always-on",
        delta_composition=Composition.from_formula("H2"),
    )
    assert loss_applies(
        always, ion_type_name="B", fragment_residues=[], fragment_mods={}
    )
    assert loss_applies(
        always, ion_type_name="Y", fragment_residues=["A"], fragment_mods={}
    )


def test_list_value_in_fragment_mods():
    """`fragment_mods` may carry list values when a position has multiple
    mods (parse_modified_sequence returns lists in that case)."""
    hpo3 = LOSS_REGISTRY["HPO3"]
    assert loss_applies(
        hpo3,
        ion_type_name="B",
        fragment_residues=["S", "T"],
        fragment_mods={0: ["UNIMOD:35", "UNIMOD:21"]},
    )
