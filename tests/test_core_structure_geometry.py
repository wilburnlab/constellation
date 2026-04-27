"""Unit tests for ``constellation.core.structure.geometry``."""

from __future__ import annotations

import math

import pyarrow as pa
import pytest
import torch

from constellation.core.structure import (
    ATOM_TABLE,
    Ensemble,
    Topology,
    centroid,
    mass_centroid,
    principal_axes,
    radius_of_gyration,
    rotate,
    translate,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _unit_cube() -> torch.Tensor:
    """Eight-corner unit cube centred at origin, side 2."""
    return torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [+1.0, -1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [-1.0, +1.0, +1.0],
            [+1.0, +1.0, +1.0],
        ],
        dtype=torch.float32,
    )


# ──────────────────────────────────────────────────────────────────────
# translate
# ──────────────────────────────────────────────────────────────────────


def test_translate_2d():
    coords = _unit_cube()
    out = translate(coords, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(out, coords + torch.tensor([1.0, 2.0, 3.0]))


def test_translate_3d_broadcasts_over_frames():
    coords = _unit_cube().unsqueeze(0).expand(4, 8, 3).contiguous()
    out = translate(coords, torch.tensor([0.5, 0.0, 0.0]))
    diff = out - coords
    expected = torch.zeros_like(coords)
    expected[..., 0] = 0.5
    assert torch.allclose(diff, expected)


def test_translate_rejects_bad_vector():
    with pytest.raises(ValueError):
        translate(_unit_cube(), torch.zeros(2))


def test_translate_rejects_bad_coords():
    with pytest.raises(ValueError):
        translate(torch.zeros(5), torch.zeros(3))


# ──────────────────────────────────────────────────────────────────────
# rotate
# ──────────────────────────────────────────────────────────────────────


def test_rotate_identity_preserves_coords():
    coords = _unit_cube()
    out = rotate(coords, torch.eye(3))
    assert torch.allclose(out, coords)


def test_rotate_90deg_about_z_swaps_xy():
    coords = torch.tensor([[1.0, 0.0, 0.0]])
    R = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    out = rotate(coords, R)
    # (1, 0, 0) rotated 90° CCW about z → (0, 1, 0).
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_rotate_about_pivot():
    coords = torch.tensor([[2.0, 0.0, 0.0]])
    R = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pivot = torch.tensor([1.0, 0.0, 0.0])
    out = rotate(coords, R, about=pivot)
    # 180° about pivot=(1,0,0): (2,0,0) → (0,0,0).
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 0.0]]), atol=1e-6)


def test_rotate_broadcasts_over_frames():
    coords = torch.stack([_unit_cube(), _unit_cube() * 2])
    out = rotate(coords, torch.eye(3))
    assert torch.allclose(out, coords)


def test_rotate_rejects_bad_R():
    with pytest.raises(ValueError):
        rotate(_unit_cube(), torch.eye(2))


# ──────────────────────────────────────────────────────────────────────
# centroid + mass_centroid
# ──────────────────────────────────────────────────────────────────────


def test_centroid_unit_cube_at_origin():
    c = centroid(_unit_cube())
    assert torch.allclose(c, torch.zeros(3), atol=1e-6)


def test_centroid_weighted_collapses_to_heavy_atom():
    coords = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    weights = torch.tensor([1.0, 99.0])
    c = centroid(coords, weights)
    # Heavy atom dominates.
    assert c[0].item() > 9.5


def test_centroid_per_frame_for_3d_input():
    coords = torch.stack([_unit_cube(), _unit_cube() + 1.0])
    c = centroid(coords)
    assert c.shape == (2, 3)
    assert torch.allclose(c[0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(c[1], torch.ones(3), atol=1e-6)


def test_centroid_rejects_negative_weights():
    with pytest.raises(ValueError):
        centroid(_unit_cube(), -torch.ones(8))


def test_centroid_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        centroid(_unit_cube(), torch.zeros(8))


def _small_atoms() -> pa.Table:
    rows = [
        {
            "serial": 1,
            "name": "C1",
            "res_name": "LIG",
            "chain_id": "A",
            "res_seq": 1,
            "element": "C",
            "is_hetatm": True,
            "model_id": 1,
        },
        {
            "serial": 2,
            "name": "H1",
            "res_name": "LIG",
            "chain_id": "A",
            "res_seq": 1,
            "element": "H",
            "is_hetatm": True,
            "model_id": 1,
        },
    ]
    cols = {
        f.name: pa.array([r.get(f.name, None) for r in rows], type=f.type)
        for f in ATOM_TABLE
    }
    return pa.table(cols, schema=ATOM_TABLE)


def test_mass_centroid_pulls_element_masses():
    """C at origin (mass 12), H at (1, 0, 0) (mass 1) → centroid near 1/13."""
    atoms = _small_atoms()
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    e = Ensemble(Topology(atoms), coords)
    c = mass_centroid(e)
    # Single-frame ensemble → shape (1, 3).
    assert c.shape == (1, 3)
    expected_x = 1.0 / 13.00782503  # H exact mass / (C + H)
    assert abs(c[0, 0].item() - expected_x) < 1e-3


# ──────────────────────────────────────────────────────────────────────
# radius_of_gyration
# ──────────────────────────────────────────────────────────────────────


def test_rg_unit_cube():
    """Eight corners of cube ±1 → Rg = sqrt(3)."""
    rg = radius_of_gyration(_unit_cube())
    assert abs(rg.item() - math.sqrt(3.0)) < 1e-5


def test_rg_per_frame():
    coords = torch.stack([_unit_cube(), _unit_cube() * 2.0])
    rg = radius_of_gyration(coords)
    assert rg.shape == (2,)
    assert abs(rg[0].item() - math.sqrt(3.0)) < 1e-5
    assert abs(rg[1].item() - 2.0 * math.sqrt(3.0)) < 1e-5


def test_rg_invariant_under_translation():
    coords = _unit_cube()
    rg1 = radius_of_gyration(coords)
    rg2 = radius_of_gyration(coords + 100.0)
    assert torch.allclose(rg1, rg2, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# principal_axes
# ──────────────────────────────────────────────────────────────────────


def test_principal_axes_orthonormal():
    """Stretched ellipsoid: clear principal directions."""
    coords = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    axes = principal_axes(coords)
    assert axes.shape == (3, 3)
    # Rows orthonormal.
    eye = axes @ axes.T
    assert torch.allclose(eye, torch.eye(3), atol=1e-5)
    # Right-handed.
    det = torch.det(axes.to(torch.float64))
    assert det.item() > 0


def test_principal_axes_per_frame():
    coords = torch.stack(
        [
            torch.tensor(
                [
                    [4.0, 0.0, 0.0],
                    [-4.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, -2.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ]
            ),
        ]
        * 2
    )
    axes = principal_axes(coords)
    assert axes.shape == (2, 3, 3)


# ──────────────────────────────────────────────────────────────────────
# Non-goals — explicit deferral assertion
# ──────────────────────────────────────────────────────────────────────


def test_non_goals_not_exported():
    """Kabsch / RMSD / superpose / align must not be in the public surface
    until core.stats + core.optim ship."""
    import constellation.core.structure as cs

    public = set(getattr(cs, "__all__", []))
    forbidden = {"kabsch", "rmsd", "superpose", "align", "optimal_rotation"}
    assert public.isdisjoint(forbidden)
