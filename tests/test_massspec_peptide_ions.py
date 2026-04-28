"""Tests for `constellation.massspec.peptide.ions`.

Covers:
  * b/y parity vs hand-computed values (and Cartographer parity at known points)
  * c/z and a/x algebraic relationships (c=b+NH3, z=y-NH3, a=b-CO, x=y+CO)
  * Modification placement on the correct side (N-side / C-side at each cleavage)
  * Heavy-isotope correction propagating into fragment ions
  * Neutral-loss masking (validity rules + tensor NaN ↔ table absence round-trip)
"""

from __future__ import annotations

import math

import pyarrow as pa
import torch

from constellation.core.chem.modifications import UNIMOD
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
    # a = b - CO  (CO ~27.99491)
    assert math.isclose(
        ION_OFFSET_MASSES[IonType.A] - ION_OFFSET_MASSES[IonType.B],
        -27.99491,
        abs_tol=1e-3,
    )
    # c = b + NH3  (~17.02655)
    assert math.isclose(
        ION_OFFSET_MASSES[IonType.C] - ION_OFFSET_MASSES[IonType.B],
        17.02655,
        abs_tol=1e-3,
    )
    # x = y + CO
    assert math.isclose(
        ION_OFFSET_MASSES[IonType.X] - ION_OFFSET_MASSES[IonType.Y],
        27.99491,
        abs_tol=1e-3,
    )
    # z = y - NH3  (radical form)
    assert math.isclose(
        ION_OFFSET_MASSES[IonType.Z] - ION_OFFSET_MASSES[IonType.Y],
        -17.02655,
        abs_tol=1e-3,
    )


def test_fragment_mz_scalar():
    # b1 of PEPTIDE: P residue mass 97.05276 + 0 + proton = 98.06004
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
    table, tensor = fragment_ladder("PEPTIDE", max_fragment_charge=1)
    # PEPTIDE has 7 residues → 6 cleavage positions; b/y default → 2 ion types;
    # 1 charge; no losses → 1 (baseline only).
    assert tensor.shape == (6, 2, 1, 1)
    # 6 positions × 2 ion types × 1 charge = 12 valid ions, all finite.
    assert table.num_rows == 12
    assert torch.isfinite(tensor).all()


def test_fragment_ladder_b_y_values_peptide():
    """b/y values for PEPTIDE match the standard published references."""
    table, _ = fragment_ladder("PEPTIDE", max_fragment_charge=1)
    rows = table.to_pylist()
    by_pos = {(r["ion_type"], r["position"]): r["mz_theoretical"] for r in rows}

    # b ions (IonType.B = 1)
    expected_b = [98.06004, 227.10263, 324.15540, 425.20308, 538.28714, 653.31408]
    for p, exp in enumerate(expected_b):
        assert math.isclose(by_pos[(IonType.B.value, p)], exp, abs_tol=1e-3)

    # y ions (IonType.Y = 4); Cartographer convention: y at cleavage p covers
    # residues p+1..L-1 (so larger fragment first).
    expected_y = [703.31448, 574.27188, 477.21912, 376.17144, 263.08738, 148.06044]
    for p, exp in enumerate(expected_y):
        assert math.isclose(by_pos[(IonType.Y.value, p)], exp, abs_tol=1e-3)


def test_fragment_ladder_c_z_a_x_algebra():
    """c = b + NH3, z = y - NH3, a = b - CO, x = y + CO at every cleavage."""
    table, tensor = fragment_ladder(
        "PEPTIDE",
        ion_types=(IonType.A, IonType.B, IonType.C, IonType.X, IonType.Y, IonType.Z),
        max_fragment_charge=1,
    )
    # tensor shape: (6, 6, 1, 1)
    type_idx = {t: i for i, t in enumerate((IonType.A, IonType.B, IonType.C, IonType.X, IonType.Y, IonType.Z))}
    # All 36 ions valid, finite
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
    _, t1 = fragment_ladder("PEPTIDE", max_fragment_charge=1)
    _, t2 = fragment_ladder("PEPTIDE", max_fragment_charge=2)
    assert t2.shape == (6, 2, 2, 1)
    # +1 slot of t2 should equal t1 (which has shape (6, 2, 1, 1))
    assert torch.allclose(t2[:, :, 0:1, :], t1, atol=1e-9)
    # +2 m/z for first b-ion: ((neutral_b1) + 2*proton) / 2 where neutral_b1 = b1 - proton
    b1_z1 = t1[0, 0, 0, 0].item()
    neutral_b1 = b1_z1 - PROTON_MASS
    expected_b1_z2 = (neutral_b1 + 2 * PROTON_MASS) / 2
    assert math.isclose(t2[0, 0, 1, 0].item(), expected_b1_z2, abs_tol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# Modification placement
# ──────────────────────────────────────────────────────────────────────


def test_modification_carbamidomethyl_placement():
    """Position-3 carbamidomethyl Cys (+57.0215) appears in b4..b6 and y4..y6
    (cleavages where the C is on that side), absent from b1..b3 and y1..y3."""
    seq = "PEPCTIDE"  # length 8; C is at position 3
    mods = {3: "UNIMOD:4"}
    cam_delta = UNIMOD["UNIMOD:4"].delta_mass

    _, t_plain = fragment_ladder(seq, max_fragment_charge=1)
    _, t_mod = fragment_ladder(seq, max_fragment_charge=1, modifications=mods)
    # shape (7, 2, 1, 1)
    assert t_plain.shape == t_mod.shape == (7, 2, 1, 1)

    delta = (t_mod - t_plain).squeeze()  # (7, 2)
    # b at cleavage p covers residues 0..p. C is at position 3, so b carries
    # the mod when p >= 3 → cleavages 3..6 (b4..b7).
    for p in range(7):
        b_delta = delta[p, 0].item()
        if p >= 3:
            assert math.isclose(b_delta, cam_delta, abs_tol=1e-9), p
        else:
            assert math.isclose(b_delta, 0.0, abs_tol=1e-9), p
    # y at cleavage p covers residues p+1..L-1. C at position 3 is in y when
    # p+1 <= 3, i.e. p <= 2 → cleavages 0..2 (y8..y6).
    for p in range(7):
        y_delta = delta[p, 1].item()
        if p <= 2:
            assert math.isclose(y_delta, cam_delta, abs_tol=1e-9), p
        else:
            assert math.isclose(y_delta, 0.0, abs_tol=1e-9), p


def test_modification_n_terminal_acetyl_on_all_b():
    """N-terminal acetyl (UNIMOD:1 at position 0) lands on every b-ion and
    no y-ion."""
    seq = "PEPTIDE"
    mods = {0: "UNIMOD:1"}
    ac_delta = UNIMOD["UNIMOD:1"].delta_mass

    _, t_plain = fragment_ladder(seq, max_fragment_charge=1)
    _, t_mod = fragment_ladder(seq, max_fragment_charge=1, modifications=mods)
    delta = (t_mod - t_plain).squeeze()
    for p in range(6):
        # All b-ions carry the acetyl.
        assert math.isclose(delta[p, 0].item(), ac_delta, abs_tol=1e-9), p
        # No y-ions carry it.
        assert math.isclose(delta[p, 1].item(), 0.0, abs_tol=1e-9), p


def test_heavy_isotope_silac_in_y_ion():
    """SILAC ¹³C₆ Lys at C-terminus shifts every y-ion (covers the K) but no
    b-ion. Verifies mass_override (heavy-isotope correction) propagates."""
    seq = "PEPTIDEK"  # K at position 7 (last)
    mods = {7: "UNIMOD:188"}
    silac_delta = UNIMOD["UNIMOD:188"].delta_mass

    _, t_plain = fragment_ladder(seq, max_fragment_charge=1)
    _, t_mod = fragment_ladder(seq, max_fragment_charge=1, modifications=mods)
    delta = (t_mod - t_plain).squeeze()  # (7, 2)
    for p in range(7):
        # b-ions never include the C-term K → no shift
        assert math.isclose(delta[p, 0].item(), 0.0, abs_tol=1e-9), p
        # y-ions always include the C-term K → full SILAC shift
        assert math.isclose(delta[p, 1].item(), silac_delta, abs_tol=1e-9), p


# ──────────────────────────────────────────────────────────────────────
# Neutral losses
# ──────────────────────────────────────────────────────────────────────


def test_neutral_loss_h2o_on_serine_containing_fragments():
    """For SAMPLER (S, A, M, P, L, E, R), H2O loss is valid for b-ions covering S
    (every b-ion, since S is residue 0) and y-ions covering E (y at cleavage 0..4,
    where the fragment includes residue 5 = E). y at cleavage 5 covers only R
    (no S/T/D/E) so H2O is invalid there."""
    seq = "SAMPLER"
    table, tensor = fragment_ladder(
        seq,
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=1,
        neutral_losses=("H2O",),
    )
    # tensor shape: (6, 2, 1, 2) — slot 0 baseline, slot 1 = H2O
    assert tensor.shape == (6, 2, 1, 2)
    # b-ions all cover residue 0 (S) → H2O always valid
    for p in range(6):
        assert torch.isfinite(tensor[p, 0, 0, 1]).item(), f"b{p+1} should have H2O"
    # y at cleavage p covers residues p+1..L-1. E is residue 5; y covers E iff p<=4.
    for p in range(5):
        assert torch.isfinite(tensor[p, 1, 0, 1]).item(), f"y at p={p} should have H2O (covers E)"
    # y at p=5 covers only R — no S/T/D/E → H2O invalid.
    assert not torch.isfinite(tensor[5, 1, 0, 1]).item()


def test_neutral_loss_nh3_arginine():
    """For SAMPLER, NH3 loss valid on y-ions covering R (always — R is at C-term);
    not valid on b-ions before R (every b-ion stops before residue 6)."""
    seq = "SAMPLER"
    _, tensor = fragment_ladder(
        seq,
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=1,
        neutral_losses=("NH3",),
    )
    assert tensor.shape == (6, 2, 1, 2)
    # b at cleavage p covers 0..p. R is residue 6, so b never includes R.
    # → b NH3 should all be NaN.
    for p in range(6):
        assert not torch.isfinite(tensor[p, 0, 0, 1]).item(), p
    # y always includes the C-term residue (R), so NH3 always valid.
    for p in range(6):
        assert torch.isfinite(tensor[p, 1, 0, 1]).item(), p


def test_neutral_loss_phospho_required():
    """HPO3 loss requires a Phospho mod somewhere in the fragment. Without it,
    every (pos, type) NaN. With it, only fragments containing the phospho-site
    get the loss."""
    seq = "SAMPLE"  # length 6
    # Plain peptide — no phospho → HPO3 loss impossible everywhere.
    _, t_plain = fragment_ladder(
        seq,
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=1,
        neutral_losses=("HPO3",),
    )
    # slot 1 is HPO3; should be all NaN.
    assert torch.isnan(t_plain[..., 1]).all()
    # Phospho on S (position 0) — HPO3 valid on every b-ion (all cover S),
    # invalid on every y-ion (y never covers S=residue 0).
    _, t_p = fragment_ladder(
        seq,
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=1,
        modifications={0: "UNIMOD:21"},
        neutral_losses=("HPO3",),
    )
    for p in range(5):
        assert torch.isfinite(t_p[p, 0, 0, 1]).item(), f"b{p+1} should carry HPO3"
        assert torch.isnan(t_p[p, 1, 0, 1]).item(), f"y at p={p} should not carry HPO3"


# ──────────────────────────────────────────────────────────────────────
# Tensor ↔ table round-trip
# ──────────────────────────────────────────────────────────────────────


def test_table_tensor_round_trip():
    """Number of finite cells in the tensor equals number of rows in the table,
    and per-cell m/z matches per-row m/z."""
    table, tensor = fragment_ladder(
        "SAMPLER",
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=2,
        neutral_losses=("H2O", "NH3"),
    )
    finite = torch.isfinite(tensor)
    assert finite.sum().item() == table.num_rows
    # Every finite (p, t, c, l) cell must appear in the table with matching m/z.
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
    table, _ = fragment_ladder("PEPTIDE", max_fragment_charge=1)
    assert table.schema.metadata is not None
    assert table.schema.metadata.get(b"schema_name") == b"FragmentIonTable"
    assert "peptide_idx" in table.column_names
    assert "ion_type" in table.column_names
    assert table.schema.field("ion_type").type == pa.int8()


def test_peptide_idx_propagates():
    table, _ = fragment_ladder("PEPTIDE", max_fragment_charge=1, peptide_idx=42)
    pep_idx = table.column("peptide_idx").to_pylist()
    assert all(x == 42 for x in pep_idx)


# ──────────────────────────────────────────────────────────────────────
# Error paths
# ──────────────────────────────────────────────────────────────────────


def test_too_short_sequence():
    import pytest

    with pytest.raises(ValueError, match="length"):
        fragment_ladder("A", max_fragment_charge=1)


def test_invalid_charge():
    import pytest

    with pytest.raises(ValueError):
        fragment_ladder("PEPTIDE", max_fragment_charge=0)


def test_empty_ion_types():
    import pytest

    with pytest.raises(ValueError):
        fragment_ladder("PEPTIDE", ion_types=(), max_fragment_charge=1)


def test_modification_index_out_of_range():
    import pytest

    with pytest.raises(IndexError):
        fragment_ladder("PEPTIDE", modifications={99: "UNIMOD:35"}, max_fragment_charge=1)
