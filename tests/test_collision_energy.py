"""The collision energy a fragment predictor is run at, per precursor charge."""

from __future__ import annotations

import json

import numpy as np
import pytest

from constellation.core.stages import stage_key
from constellation.massspec.library.prediction import (
    CollisionEnergy,
    parse_energy_table,
)

# EncyclopeDIA 6.9.23's PrositCSVWriter.convertNCE(nce, z, zref) for z = 1..7,
# printed by the jar itself with Float.toHexString (2026-11-05).
JAVA_CONVERT_NCE = {
    (33.0, 3): ["0x1.c0cccep4", "0x1.f2aaacp4", "0x1.08p5", "0x1.188p5", "0x1.2b3334p5", "0x1.2b3334p5", "0x1.2b3334p5"],
    (33.0, 2): ["0x1.db3332p4", "0x1.08p5", "0x1.178786p5", "0x1.28fffep5", "0x1.3cccccp5", "0x1.3cccccp5", "0x1.3cccccp5"],
    (30.0, 3): ["0x1.98p4", "0x1.c55556p4", "0x1.ep4", "0x1.fep4", "0x1.1p5", "0x1.1p5", "0x1.1p5"],
    (30.0, 2): ["0x1.bp4", "0x1.ep4", "0x1.fc3c3cp4", "0x1.0ep5", "0x1.2p5", "0x1.2p5", "0x1.2p5"],
    (27.5, 3): ["0x1.76p4", "0x1.9f8e3ap4", "0x1.b8p4", "0x1.d38p4", "0x1.f2aaaap4", "0x1.f2aaaap4", "0x1.f2aaaap4"],
    (27.5, 2): ["0x1.8cp4", "0x1.b8p4", "0x1.d1e1e2p4", "0x1.efp4", "0x1.08p5", "0x1.08p5", "0x1.08p5"],
}


@pytest.mark.parametrize(("nce", "ref"), sorted(JAVA_CONVERT_NCE))
def test_scaled_from_reference_is_encyclopedias_float32_rule(nce, ref):
    got = CollisionEnergy.scaled_from_reference(nce, ref).resolve(np.arange(1, 8))
    want = np.array([float.fromhex(h) for h in JAVA_CONVERT_NCE[(nce, ref)]], dtype=np.float32)
    assert got.dtype == np.float32
    assert got.tobytes() == want.tobytes()


def test_reference_charge_takes_the_nce_unchanged_and_others_scale():
    ce = CollisionEnergy.scaled_from_reference(33.0, 3)
    table = ce.by_charge([1, 2, 3, 4])
    assert table[3] == 33.0
    assert table[1] == pytest.approx(28.05, abs=1e-5)
    assert table[2] == pytest.approx(31.1667, abs=1e-4)
    assert table[4] == pytest.approx(35.0625)


def test_resolve_keeps_the_shape_of_the_charges():
    ce = CollisionEnergy.scaled_from_reference(30.0)
    charges = np.array([[2], [3], [2]])
    got = ce.resolve(charges)
    assert got.shape == (3, 1)
    assert got[0, 0] == got[2, 0]


def test_uniform_gives_every_charge_the_same_energy():
    got = CollisionEnergy.uniform(30.0).resolve([1, 2, 5, 9])
    assert got.tolist() == [30.0, 30.0, 30.0, 30.0]


def test_explicit_table_is_used_as_given():
    ce = CollisionEnergy.explicit({2: 28.5, 3: 31.0, 4: 33.25})
    assert ce.resolve([4, 2, 3, 2]).tolist() == [33.25, 28.5, 31.0, 28.5]
    assert ce.by_charge() == {2: 28.5, 3: 31.0, 4: 33.25}


def test_explicit_table_refuses_a_charge_it_lacks_and_names_it():
    ce = CollisionEnergy.explicit({2: 28.0, 3: 31.0})
    with pytest.raises(ValueError, match="charge 1, 5") as exc:
        ce.resolve([1, 2, 3, 5])
    assert "charges 2, 3" in str(exc.value)


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_resolve_refuses_charges_that_are_not_charges(bad):
    with pytest.raises(ValueError):
        CollisionEnergy.uniform(30.0).resolve([2, bad])


@pytest.mark.parametrize(
    "build",
    [
        lambda: CollisionEnergy.uniform(0),
        lambda: CollisionEnergy.uniform(float("nan")),
        lambda: CollisionEnergy.scaled_from_reference(30.0, 0),
        lambda: CollisionEnergy.scaled_from_reference(30.0, 3, factors={}),
        lambda: CollisionEnergy.scaled_from_reference(30.0, 3, factors={1: -1.0}),
        lambda: CollisionEnergy.explicit({}),
        lambda: CollisionEnergy.explicit({2: 0.0}),
        lambda: CollisionEnergy.explicit({0: 30.0}),
    ],
)
def test_constructors_refuse_nonsense(build):
    with pytest.raises(ValueError):
        build()


def test_custom_factor_table_replaces_searles():
    ce = CollisionEnergy.scaled_from_reference(30.0, 2, factors={1: 1.0, 2: 0.5}, outside_factor=0.25)
    assert ce.by_charge([1, 2, 3]) == {1: 15.0, 2: 30.0, 3: 60.0}


@pytest.mark.parametrize(
    "ce",
    [
        CollisionEnergy.uniform(30.0),
        CollisionEnergy.scaled_from_reference(33.0, 3),
        CollisionEnergy.scaled_from_reference(27.5, 2, factors={1: 1.0, 2: 0.9}, outside_factor=0.8),
        CollisionEnergy.explicit({1: 28.0, 2: 31.0, 3: 33.0}),
    ],
)
def test_describe_round_trips_through_json(ce):
    description = json.loads(json.dumps(ce.describe()))
    assert CollisionEnergy.from_description(description) == ce
    assert description["origin"] == ce.origin
    assert set(description["table"]) >= {"1", "2", "3"}


def test_describe_lists_the_table_for_the_charges_named():
    d = CollisionEnergy.scaled_from_reference(33.0).describe([2, 3])
    assert set(d["table"]) == {"2", "3"}
    assert d["table"]["3"] == 33.0
    assert d["reference_charge"] == 3


def test_a_stage_key_moves_when_one_charges_energy_moves():
    a = CollisionEnergy.explicit({2: 28.0, 3: 31.0})
    b = CollisionEnergy.explicit({2: 28.0, 3: 31.5})
    c = CollisionEnergy.explicit({3: 31.0, 2: 28.0})
    assert stage_key(ce=a.describe()) != stage_key(ce=b.describe())
    assert stage_key(ce=a.describe()) == stage_key(ce=c.describe())


def test_a_reference_scaled_table_and_the_same_explicit_table_are_told_apart():
    scaled = CollisionEnergy.scaled_from_reference(33.0)
    same_values = CollisionEnergy.explicit(scaled.by_charge())
    assert scaled.by_charge() == same_values.by_charge()
    assert scaled.describe()["origin"] != same_values.describe()["origin"]


def test_label():
    assert CollisionEnergy.scaled_from_reference(27.5).label == "27.5"
    assert CollisionEnergy.explicit({2: 28.0}).label == "by-charge"


@pytest.mark.parametrize(
    ("text", "table"),
    [
        ("1=28,2=31,3=33", {1: 28.0, 2: 31.0, 3: 33.0}),
        (" 2 = 27.5 , 3=30 ,", {2: 27.5, 3: 30.0}),
    ],
)
def test_parse_energy_table(text, table):
    assert parse_energy_table(text) == table


@pytest.mark.parametrize("text", ["", "2", "2=", "x=30", "2=30,2=31", "0=30", "2=-5", "2.5=30"])
def test_parse_energy_table_refuses_bad_input(text):
    with pytest.raises(ValueError):
        parse_energy_table(text)
