"""Tensor-native geometry operations.

Free-function API operating on coord tensors of shape ``(N, 3)`` or
``(F, N, 3)`` — both shapes work via implicit broadcasting on the
leading frame dimension.

Float64 is used internally for accumulators (centroid sums, inertia
tensors, eigendecompositions) regardless of the input dtype, then
results are cast back to the input dtype. This matches the precision
discipline of ``core.chem.Composition.mass``.

Explicit non-goals — these symbols MUST NOT appear in this module
this session: ``kabsch``, ``rmsd``, ``superpose``, ``align``,
``optimal_rotation``. Build-order deferral: the constraint /
objective-function machinery in ``core.stats`` + ``core.optim`` has
not shipped yet. They land in this file once those modules are in
place; no module-boundary issue, just sequencing.
"""

from __future__ import annotations

import torch

from constellation.core.chem.atoms import ATOMS
from constellation.core.structure.ensemble import Ensemble


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _check_coords(coords: torch.Tensor) -> None:
    if coords.ndim not in (2, 3) or coords.shape[-1] != 3:
        raise ValueError(
            f"coords must have shape (N, 3) or (F, N, 3); got {tuple(coords.shape)}"
        )


def _broadcast_weights(
    coords: torch.Tensor, weights: torch.Tensor | None
) -> torch.Tensor | None:
    """Reshape ``weights`` to match ``coords``' leading non-feature dims.

    Returns float64 weights of shape ``(N,)`` for ``(N, 3)`` coords
    or ``(N,)`` (broadcast over frames) for ``(F, N, 3)`` coords.
    Negative weights raise.
    """
    if weights is None:
        return None
    if weights.ndim != 1:
        raise ValueError(
            f"weights must be 1-D (length N); got shape {tuple(weights.shape)}"
        )
    n_atoms = coords.shape[-2]
    if weights.shape[0] != n_atoms:
        raise ValueError(f"weights length {weights.shape[0]} != n_atoms {n_atoms}")
    w = weights.to(torch.float64)
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    return w


# ──────────────────────────────────────────────────────────────────────
# Translation / rotation
# ──────────────────────────────────────────────────────────────────────


def translate(coords: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Add ``vector`` (shape ``(3,)`` or broadcastable) to every atom.

    Works on ``(N, 3)`` and ``(F, N, 3)`` coords. Returns a new tensor;
    the input is unmodified.
    """
    _check_coords(coords)
    v = vector.to(coords.dtype)
    if v.shape[-1] != 3:
        raise ValueError(f"vector must have last dim 3; got shape {tuple(v.shape)}")
    return coords + v


def rotate(
    coords: torch.Tensor,
    R: torch.Tensor,
    *,
    about: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate coords by a ``(3, 3)`` rotation matrix.

    Convention: ``coords' = (coords - about) @ R.T + about``. With
    ``about=None`` (default), rotation is about the origin. ``R`` is
    not validated to be orthonormal — pass an SVD-derived rotation if
    you want determinant +1 / orthonormality.

    Broadcasts over the leading frame dim of ``(F, N, 3)`` inputs.
    """
    _check_coords(coords)
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3); got {tuple(R.shape)}")
    R_cast = R.to(coords.dtype)
    if about is None:
        return coords @ R_cast.T
    pivot = about.to(coords.dtype)
    if pivot.shape[-1] != 3:
        raise ValueError(f"about must have last dim 3; got shape {tuple(pivot.shape)}")
    return (coords - pivot) @ R_cast.T + pivot


# ──────────────────────────────────────────────────────────────────────
# Centroids and gyration
# ──────────────────────────────────────────────────────────────────────


def centroid(
    coords: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Centre of mass / unweighted centroid of an atom set.

    For ``(N, 3)`` input, returns shape ``(3,)``. For ``(F, N, 3)``,
    returns ``(F, 3)``. ``weights`` is a 1-D ``(N,)`` tensor of
    non-negative weights (typically masses); ``None`` means uniform.
    """
    _check_coords(coords)
    coords64 = coords.to(torch.float64)
    w = _broadcast_weights(coords, weights)
    if w is None:
        c = coords64.mean(dim=-2)
    else:
        # Reshape weights for broadcasting: (N, 1) for (N, 3) coords;
        # (1, N, 1) for (F, N, 3).
        if coords.ndim == 2:
            w_b = w.unsqueeze(-1)  # (N, 1)
        else:
            w_b = w.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        total = w.sum()
        if total <= 0:
            raise ValueError("weights sum to zero")
        c = (coords64 * w_b).sum(dim=-2) / total
    return c.to(coords.dtype)


def mass_centroid(ensemble: Ensemble) -> torch.Tensor:
    """Mass-weighted centroid of an ``Ensemble``, per frame.

    Pulls element symbols from ``ensemble.topology.atoms.element``
    and resolves through ``core.chem.ATOMS[sym].monoisotopic_mass``.
    Atoms whose element is unknown / has no mass data raise.

    Returns shape ``(F, 3)`` for an ``F``-frame ensemble (use
    ``[0]`` for a single-frame structure).
    """
    elements = ensemble.topology.atoms.column("element").to_pylist()
    masses_list: list[float] = []
    for sym in elements:
        try:
            m = ATOMS[sym].monoisotopic_mass
        except KeyError as exc:
            raise ValueError(
                f"unknown element symbol {sym!r}; cannot compute mass centroid"
            ) from exc
        masses_list.append(m)
    weights = torch.tensor(masses_list, dtype=torch.float64)
    return centroid(ensemble.coords, weights)


def radius_of_gyration(
    coords: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Radius of gyration ``sqrt(sum(w_i * |r_i - r_cm|^2) / sum(w_i))``.

    Returns shape ``()`` for ``(N, 3)`` input, ``(F,)`` for
    ``(F, N, 3)``. With ``weights=None``, all atoms get unit weight.
    """
    _check_coords(coords)
    coords64 = coords.to(torch.float64)
    w = _broadcast_weights(coords, weights)
    cm = centroid(coords, weights).to(torch.float64)
    if coords.ndim == 2:
        diff = coords64 - cm.unsqueeze(0)  # (N, 3)
        sq = (diff * diff).sum(dim=-1)  # (N,)
        if w is None:
            rg2 = sq.mean()
        else:
            rg2 = (w * sq).sum() / w.sum()
    else:
        diff = coords64 - cm.unsqueeze(1)  # (F, N, 3)
        sq = (diff * diff).sum(dim=-1)  # (F, N)
        if w is None:
            rg2 = sq.mean(dim=-1)  # (F,)
        else:
            rg2 = (w.unsqueeze(0) * sq).sum(dim=-1) / w.sum()
    return torch.sqrt(rg2).to(coords.dtype)


# ──────────────────────────────────────────────────────────────────────
# Principal axes
# ──────────────────────────────────────────────────────────────────────


def principal_axes(
    coords: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Principal axes of inertia (or unweighted second-moment matrix).

    Returns a ``(3, 3)`` tensor whose **rows** are the principal axes,
    sorted by descending eigenvalue. The returned axes form an
    orthonormal basis (right-handed sign convention enforced via
    cross-product correction). For ``(F, N, 3)`` input, returns
    ``(F, 3, 3)``.

    Numerically degenerate inputs (collinear / coplanar) yield
    eigenvectors that are mathematically valid but the choice of basis
    in the degenerate eigenspace is implementation-defined; tests
    assert orthonormality and not specific axis directions.
    """
    _check_coords(coords)
    coords64 = coords.to(torch.float64)
    w = _broadcast_weights(coords, weights)
    cm = centroid(coords, weights).to(torch.float64)

    if coords.ndim == 2:
        return _principal_axes_single(coords64, cm, w).to(coords.dtype)
    out = torch.stack(
        [
            _principal_axes_single(coords64[f], cm[f], w)
            for f in range(coords64.shape[0])
        ]
    )
    return out.to(coords.dtype)


def _principal_axes_single(
    coords64: torch.Tensor,
    cm: torch.Tensor,
    w: torch.Tensor | None,
) -> torch.Tensor:
    centred = coords64 - cm.unsqueeze(0)  # (N, 3)
    if w is None:
        moment = centred.T @ centred  # (3, 3) — second-moment, unweighted
    else:
        weighted = centred * w.unsqueeze(-1)
        moment = weighted.T @ centred
    # Eigh returns ascending; flip to descending for principal-axis convention.
    eigvals, eigvecs = torch.linalg.eigh(moment)
    order = torch.argsort(eigvals, descending=True)
    axes = eigvecs[:, order].T  # rows are principal axes
    # Enforce right-handedness: axes[2] should be axes[0] × axes[1].
    cross = torch.linalg.cross(axes[0], axes[1])
    if torch.dot(cross, axes[2]) < 0:
        axes = axes.clone()
        axes[2] = -axes[2]
    return axes


__all__ = [
    "translate",
    "rotate",
    "centroid",
    "mass_centroid",
    "radius_of_gyration",
    "principal_axes",
]
