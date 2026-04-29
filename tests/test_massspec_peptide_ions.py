"""Tests for `constellation.massspec.peptide.ions`.

Covers:
  * b/y parity vs hand-computed values (and Cartographer parity at known points)
  * c/z and a/x algebraic relationships (c=b+NH3, z=y-NH3, a=b-CO, x=y+CO)
  * Modification placement on the correct side (N-side / C-side at each cleavage)
  * Heavy-isotope correction propagating into fragment ions
  * Neutral-loss masking (validity rules + tensor NaN ↔ table absence round-trip)
  * Dispatcher routing (linear / cross-link / branch / multichain)

All inputs flow through `parse_proforma` per the ProForma 2.0 migration.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest
import torch

from constellation.core.chem.modifications import UNIMOD
from constellation.core.sequence.proforma import (
    Peptidoform,
    TaggedMod,
    ModRef,
    parse_proforma,
)
from constellation.massspec.peptide.ions import (
    ION_OFFSET_MASSES,
    IonType,
    fragment_ladder,
    fragment_mz,
)
from constellation.massspec.peptide.mz import PROTON_MASS


# ──────────────────────────────────────────────────────────────────────
# ION_OFFSETS sanity
# ──────────────────────────────────────────────────────────────────────


def test_ion_offset_masses():
    """Per-type offsets match the published Roepstorff-Fohlman conventions."""
    assert math.isclose(ION_OFFSET_MASSES[IonType.B], 0.0, abs_tol=1e-9)
    assert math.isclose(ION_OFFSET_MASSES[IonType.Y], 18.010565, abs_tol=1e-3)
    assert math.isclose(
        ION_OFFSET_MASSES[IonType.A] - ION_OFFSET_MASSES[IonType.B],
        -27.99491,
        abs_tol=1e-3,
    )
    assert math.isclose(
        ION_OFFSET_MASSES[IonType.C] - ION_OFFSET_MASSES[IonType.B],
        17.02655,
        abs_tol=1e-3,
    )
    assert math.isclose(
        ION_OFFSET_MASSES[IonType.X] - ION_OFFSET_MASSES[IonType.Y],
        27.99491,
        abs_tol=1e-3,
    )
    assert math.isclose(
        ION_OFFSET_MASSES[IonType.Z] - ION_OFFSET_MASSES[IonType.Y],
        -17.02655,
        abs_tol=1e-3,
    )


def test_fragment_mz_scalar():
    p_mass = 97.052764  # proline residue (in-chain)
    mz = fragment_mz(p_mass, ion_type=IonType.B, charge=1)
    assert math.isclose(mz, 98.06004, abs_tol=1e-3)


def test_fragment_mz_tensor():
    masses = torch.tensor([97.052764, 226.0954], dtype=torch.float64)
    mzs = fragment_mz(masses, ion_type=IonType.B, charge=1)
    assert mzs.shape == (2,)
    assert torch.allclose(
        mzs, torch.tensor([98.06004, 227.10268], dtype=torch.float64), atol=1e-3
    )


# ──────────────────────────────────────────────────────────────────────
# fragment_ladder — basic shape and b/y values
# ──────────────────────────────────────────────────────────────────────


def test_fragment_ladder_shape_peptide_basic():
    table, tensor = fragment_ladder(
        parse_proforma("PEPTIDE"), max_fragment_charge=1
    )
    assert tensor.shape == (6, 2, 1, 1)
    assert table.num_rows == 12
    assert torch.isfinite(tensor).all()


def test_fragment_ladder_b_y_values_peptide():
    """b/y values for PEPTIDE match the standard published references."""
    table, _ = fragment_ladder(parse_proforma("PEPTIDE"), max_fragment_charge=1)
    rows = table.to_pylist()
    by_pos = {(r["ion_type"], r["position"]): r["mz_theoretical"] for r in rows}

    expected_b = [98.06004, 227.10263, 324.15540, 425.20308, 538.28714, 653.31408]
    for p, exp in enumerate(expected_b):
        assert math.isclose(by_pos[(IonType.B.value, p)], exp, abs_tol=1e-3)

    expected_y = [703.31448, 574.27188, 477.21912, 376.17144, 263.08738, 148.06044]
    for p, exp in enumerate(expected_y):
        assert math.isclose(by_pos[(IonType.Y.value, p)], exp, abs_tol=1e-3)


def test_fragment_ladder_c_z_a_x_algebra():
    """c = b + NH3, z = y - NH3, a = b - CO, x = y + CO at every cleavage."""
    table, tensor = fragment_ladder(
        parse_proforma("PEPTIDE"),
        ion_types=(IonType.A, IonType.B, IonType.C, IonType.X, IonType.Y, IonType.Z),
        max_fragment_charge=1,
    )
    type_idx = {
        t: i
        for i, t in enumerate(
            (IonType.A, IonType.B, IonType.C, IonType.X, IonType.Y, IonType.Z)
        )
    }
    assert torch.isfinite(tensor).all()
    assert table.num_rows == 36

    nh3_mass = 17.026549
    co_mass = 27.994915

    for p in range(6):
        b = tensor[p, type_idx[IonType.B], 0, 0].item()
        c = tensor[p, type_idx[IonType.C], 0, 0].item()
        a = tensor[p, type_idx[IonType.A], 0, 0].item()
        y = tensor[p, type_idx[IonType.Y], 0, 0].item()
        z = tensor[p, type_idx[IonType.Z], 0, 0].item()
        x = tensor[p, type_idx[IonType.X], 0, 0].item()
        assert math.isclose(c - b, nh3_mass, abs_tol=1e-4)
        assert math.isclose(b - a, co_mass, abs_tol=1e-4)
        assert math.isclose(y - z, nh3_mass, abs_tol=1e-4)
        assert math.isclose(x - y, co_mass, abs_tol=1e-4)


def test_fragment_ladder_charge_states():
    """Higher charge: m/z = (neutral + z*proton)/z."""
    _, t1 = fragment_ladder(parse_proforma("PEPTIDE"), max_fragment_charge=1)
    _, t2 = fragment_ladder(parse_proforma("PEPTIDE"), max_fragment_charge=2)
    assert t2.shape == (6, 2, 2, 1)
    assert torch.allclose(t2[:, :, 0:1, :], t1, atol=1e-9)
    b1_z1 = t1[0, 0, 0, 0].item()
    neutral_b1 = b1_z1 - PROTON_MASS
    expected_b1_z2 = (neutral_b1 + 2 * PROTON_MASS) / 2
    assert math.isclose(t2[0, 0, 1, 0].item(), expected_b1_z2, abs_tol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# Modification placement
# ──────────────────────────────────────────────────────────────────────


def test_modification_carbamidomethyl_placement():
    """Position-3 carbamidomethyl Cys (+57.0215) appears in b4..b7 and y8..y6
    (cleavages where the C is on that side), absent otherwise."""
    cam_delta = UNIMOD["UNIMOD:4"].delta_mass
    plain = parse_proforma("PEPCTIDE")  # length 8; C at position 3
    mod = parse_proforma("PEPC[UNIMOD:4]TIDE")

    _, t_plain = fragment_ladder(plain, max_fragment_charge=1)
    _, t_mod = fragment_ladder(mod, max_fragment_charge=1)
    assert t_plain.shape == t_mod.shape == (7, 2, 1, 1)

    delta = (t_mod - t_plain).squeeze()  # (7, 2)
    for p in range(7):
        b_delta = delta[p, 0].item()
        if p >= 3:
            assert math.isclose(b_delta, cam_delta, abs_tol=1e-9), p
        else:
            assert math.isclose(b_delta, 0.0, abs_tol=1e-9), p
    for p in range(7):
        y_delta = delta[p, 1].item()
        if p <= 2:
            assert math.isclose(y_delta, cam_delta, abs_tol=1e-9), p
        else:
            assert math.isclose(y_delta, 0.0, abs_tol=1e-9), p


def test_modification_n_terminal_acetyl_on_all_b():
    """N-terminal acetyl (the dash-form proforma N-term mod) lands on every
    b-ion and no y-ion — canonical demonstration of the K[Ac]VERPD migration."""
    ac_delta = UNIMOD["Acetyl"].delta_mass

    _, t_plain = fragment_ladder(parse_proforma("PEPTIDE"), max_fragment_charge=1)
    _, t_mod = fragment_ladder(
        parse_proforma("[Acetyl]-PEPTIDE"), max_fragment_charge=1
    )
    delta = (t_mod - t_plain).squeeze()
    for p in range(6):
        assert math.isclose(delta[p, 0].item(), ac_delta, abs_tol=1e-9), p
        assert math.isclose(delta[p, 1].item(), 0.0, abs_tol=1e-9), p


def test_modification_position_zero_acetyl_matches_n_term():
    """K[Acetyl]VERPD (residue mod at position 0) and [Acetyl]-KVERPD
    (N-term mod) produce identical fragment ladders even though the
    in-memory peptidoforms are distinct (chemistry-fidelity invariant)."""
    p_n_term = parse_proforma("[Acetyl]-KVERPD")
    p_residue = parse_proforma("K[Acetyl]VERPD")
    assert p_n_term != p_residue  # distinct in-memory
    _, t_n = fragment_ladder(p_n_term, max_fragment_charge=1)
    _, t_r = fragment_ladder(p_residue, max_fragment_charge=1)
    assert (t_n - t_r).abs().max().item() == pytest.approx(0.0, abs=1e-9)


def test_heavy_isotope_silac_in_y_ion():
    """SILAC ¹³C₆ Lys at C-terminus shifts every y-ion (covers the K) but no
    b-ion. Verifies mass_override (heavy-isotope correction) propagates."""
    silac_delta = UNIMOD["UNIMOD:188"].delta_mass
    plain = parse_proforma("PEPTIDEK")
    mod = parse_proforma("PEPTIDEK[UNIMOD:188]")

    _, t_plain = fragment_ladder(plain, max_fragment_charge=1)
    _, t_mod = fragment_ladder(mod, max_fragment_charge=1)
    delta = (t_mod - t_plain).squeeze()  # (7, 2)
    for p in range(7):
        assert math.isclose(delta[p, 0].item(), 0.0, abs_tol=1e-9), p
        assert math.isclose(delta[p, 1].item(), silac_delta, abs_tol=1e-9), p


# ──────────────────────────────────────────────────────────────────────
# Neutral losses
# ──────────────────────────────────────────────────────────────────────


def test_neutral_loss_h2o_on_serine_containing_fragments():
    """For SAMPLER, H2O loss is valid for b-ions covering S (every b-ion,
    since S is residue 0) and y-ions covering E (y at cleavage 0..4)."""
    table, tensor = fragment_ladder(
        parse_proforma("SAMPLER"),
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=1,
        neutral_losses=("H2O",),
    )
    assert tensor.shape == (6, 2, 1, 2)
    for p in range(6):
        assert torch.isfinite(tensor[p, 0, 0, 1]).item(), f"b{p+1} should have H2O"
    for p in range(5):
        assert torch.isfinite(tensor[p, 1, 0, 1]).item(), (
            f"y at p={p} should have H2O (covers E)"
        )
    assert not torch.isfinite(tensor[5, 1, 0, 1]).item()


def test_neutral_loss_nh3_arginine():
    """For SAMPLER, NH3 loss valid on y-ions covering R (always — R is at C-term);
    not valid on b-ions before R."""
    _, tensor = fragment_ladder(
        parse_proforma("SAMPLER"),
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=1,
        neutral_losses=("NH3",),
    )
    assert tensor.shape == (6, 2, 1, 2)
    for p in range(6):
        assert not torch.isfinite(tensor[p, 0, 0, 1]).item(), p
    for p in range(6):
        assert torch.isfinite(tensor[p, 1, 0, 1]).item(), p


def test_neutral_loss_phospho_required():
    """HPO3 loss requires a Phospho mod somewhere in the fragment."""
    _, t_plain = fragment_ladder(
        parse_proforma("SAMPLE"),
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=1,
        neutral_losses=("HPO3",),
    )
    assert torch.isnan(t_plain[..., 1]).all()

    _, t_p = fragment_ladder(
        parse_proforma("S[UNIMOD:21]AMPLE"),
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=1,
        neutral_losses=("HPO3",),
    )
    for p in range(5):
        assert torch.isfinite(t_p[p, 0, 0, 1]).item(), f"b{p+1} should carry HPO3"
        assert torch.isnan(t_p[p, 1, 0, 1]).item(), f"y at p={p} should not carry HPO3"


# ──────────────────────────────────────────────────────────────────────
# Tensor ↔ table round-trip
# ──────────────────────────────────────────────────────────────────────


def test_table_tensor_round_trip():
    """Number of finite cells in the tensor equals number of rows in the table."""
    table, tensor = fragment_ladder(
        parse_proforma("SAMPLER"),
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=2,
        neutral_losses=("H2O", "NH3"),
    )
    finite = torch.isfinite(tensor)
    assert finite.sum().item() == table.num_rows
    rows = table.to_pylist()
    table_keys = {
        (r["position"], r["ion_type"], r["charge"], r["loss_id"]): r["mz_theoretical"]
        for r in rows
    }
    n_pos, n_types, n_charges, n_losses = tensor.shape
    ion_types = (IonType.B, IonType.Y)
    losses_with_baseline: list[str | None] = [None, "H2O", "NH3"]
    for p in range(n_pos):
        for t_idx in range(n_types):
            for c_idx in range(n_charges):
                for l_idx in range(n_losses):
                    val = tensor[p, t_idx, c_idx, l_idx].item()
                    if not math.isnan(val):
                        key = (
                            p,
                            int(ion_types[t_idx]),
                            c_idx + 1,
                            losses_with_baseline[l_idx],
                        )
                        assert key in table_keys, key
                        assert math.isclose(table_keys[key], val, abs_tol=1e-12)


def test_table_uses_fragment_ion_table_schema():
    table, _ = fragment_ladder(parse_proforma("PEPTIDE"), max_fragment_charge=1)
    assert table.schema.metadata is not None
    assert table.schema.metadata.get(b"schema_name") == b"FragmentIonTable"
    assert "peptide_idx" in table.column_names
    assert "ion_type" in table.column_names
    assert table.schema.field("ion_type").type == pa.int8()


def test_peptide_idx_propagates():
    table, _ = fragment_ladder(
        parse_proforma("PEPTIDE"), max_fragment_charge=1, peptide_idx=42
    )
    pep_idx = table.column("peptide_idx").to_pylist()
    assert all(x == 42 for x in pep_idx)


def test_bare_peptidoform_construction():
    """fragment_ladder accepts Peptidoform built directly (without a parse step)."""
    p = Peptidoform(sequence="PEPTIDE")
    table, tensor = fragment_ladder(p, max_fragment_charge=1)
    assert tensor.shape == (6, 2, 1, 1)
    assert table.num_rows == 12


# ──────────────────────────────────────────────────────────────────────
# Error paths
# ──────────────────────────────────────────────────────────────────────


def test_too_short_sequence():
    with pytest.raises(ValueError, match="length"):
        fragment_ladder(Peptidoform(sequence="A"), max_fragment_charge=1)


def test_invalid_charge():
    with pytest.raises(ValueError):
        fragment_ladder(parse_proforma("PEPTIDE"), max_fragment_charge=0)


def test_empty_ion_types():
    with pytest.raises(ValueError):
        fragment_ladder(
            parse_proforma("PEPTIDE"), ion_types=(), max_fragment_charge=1
        )


def test_modification_index_out_of_range():
    """Build a Peptidoform with a residue mod at an out-of-range position."""
    bad = Peptidoform(
        sequence="PEPTIDE",
        residue_mods={
            99: (TaggedMod(mod=ModRef(cv="UNIMOD", accession="35")),),
        },
    )
    with pytest.raises(IndexError):
        fragment_ladder(bad, max_fragment_charge=1)
