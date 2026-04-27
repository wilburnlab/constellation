"""Ensemble — N conformers / frames over a shared topology.

A single static structure (one PDB model) is just an ensemble with
``n_frames == 1``. NMR multi-model files, cryo-EM heterogeneous-state
ensembles, and MD trajectories all share the same in-memory layout:

* ``topology`` — atom identity + bonds (``Topology``)
* ``coords``   — torch tensor ``(N_frames, N_atoms, 3)``, float32
* ``frame``    — coord-system metadata (``CoordinateFrame``)
* ``frame_metadata`` — per-frame Arrow table (time, energy, weight, ...)
* ``metadata`` — free-form ensemble-level dict

Selection (``select(expr)``) reduces both the topology and the coord
tensor consistently. Frame slicing (``slice_frames(...)``) reduces the
leading dim of coords + frame_metadata. Bonds are remapped through the
atom-index translation in ``Topology.subset``.

Coordinates are assumed complete — every frame holds an entry for
every atom. Missing-coordinate handling (alt-loc atoms, MD frames
that drop atoms) is a reader-layer concern; the in-memory data
structure does not carry NaN sentinels (torch propagates NaN silently
through reductions, which is treacherous).
"""

from __future__ import annotations

from typing import Any, Iterator

import pyarrow as pa
import torch

from constellation.core.structure.atoms import CoordinateFrame
from constellation.core.structure.topology import Topology


# ──────────────────────────────────────────────────────────────────────
# Frame-metadata schema
# ──────────────────────────────────────────────────────────────────────

FRAME_METADATA: pa.Schema = pa.schema(
    [
        pa.field("frame_id", pa.int32(), nullable=False),
        pa.field("time_ps", pa.float64(), nullable=True),
        pa.field("energy_kcal_per_mol", pa.float64(), nullable=True),
        pa.field("weight", pa.float32(), nullable=True),
        pa.field("temperature_K", pa.float32(), nullable=True),
        pa.field("source_label", pa.string(), nullable=True),
    ]
)


def _default_frame_metadata(n_frames: int) -> pa.Table:
    """Build a ``FRAME_METADATA`` table with only ``frame_id`` filled."""
    ids = pa.array(range(n_frames), type=pa.int32())
    nulls_f64 = pa.nulls(n_frames, type=pa.float64())
    nulls_f32 = pa.nulls(n_frames, type=pa.float32())
    nulls_str = pa.nulls(n_frames, type=pa.string())
    return pa.table(
        [ids, nulls_f64, nulls_f64, nulls_f32, nulls_f32, nulls_str],
        schema=FRAME_METADATA,
    )


# ──────────────────────────────────────────────────────────────────────
# Ensemble
# ──────────────────────────────────────────────────────────────────────


class Ensemble:
    """Topology + ``(N_frames, N_atoms, 3)`` coord tensor."""

    __slots__ = (
        "_topology",
        "_coords",
        "_frame",
        "_frame_metadata",
        "_metadata",
    )

    def __init__(
        self,
        topology: Topology,
        coords: torch.Tensor,
        *,
        frame: CoordinateFrame | None = None,
        frame_metadata: pa.Table | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        coords = self._normalize_coords(coords, topology.n_atoms())
        if frame_metadata is None:
            frame_metadata = _default_frame_metadata(coords.shape[0])
        elif frame_metadata.num_rows != coords.shape[0]:
            raise ValueError(
                f"frame_metadata has {frame_metadata.num_rows} rows but "
                f"coords has {coords.shape[0]} frames"
            )
        self._topology: Topology = topology
        self._coords: torch.Tensor = coords
        self._frame: CoordinateFrame = frame if frame is not None else CoordinateFrame()
        self._frame_metadata: pa.Table = frame_metadata
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}

    @staticmethod
    def _normalize_coords(coords: torch.Tensor, n_atoms: int) -> torch.Tensor:
        """Accept ``(N, 3)`` or ``(F, N, 3)``; return ``(F, N, 3)`` float32."""
        if coords.ndim == 2:
            if coords.shape != (n_atoms, 3):
                raise ValueError(
                    f"coords shape {tuple(coords.shape)} doesn't match "
                    f"topology n_atoms={n_atoms}"
                )
            coords = coords.unsqueeze(0)
        elif coords.ndim == 3:
            if coords.shape[1:] != (n_atoms, 3):
                raise ValueError(
                    f"coords shape {tuple(coords.shape)} inconsistent with "
                    f"topology n_atoms={n_atoms} (expected (F, {n_atoms}, 3))"
                )
        else:
            raise ValueError(
                f"coords must be 2-D (N, 3) or 3-D (F, N, 3); got {coords.ndim}-D"
            )
        if coords.dtype != torch.float32:
            coords = coords.to(torch.float32)
        return coords.contiguous()

    # ── views ────────────────────────────────────────────────────────
    @property
    def topology(self) -> Topology:
        return self._topology

    @property
    def coords(self) -> torch.Tensor:
        return self._coords

    @property
    def frame(self) -> CoordinateFrame:
        return self._frame

    @property
    def frame_metadata(self) -> pa.Table:
        return self._frame_metadata

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def n_frames(self) -> int:
        return self._coords.shape[0]

    def n_atoms(self) -> int:
        return self._coords.shape[1]

    def __len__(self) -> int:
        return self.n_frames()

    def __iter__(self) -> Iterator[tuple[int, torch.Tensor]]:
        for i in range(self.n_frames()):
            yield i, self._coords[i]

    def get_frame(self, i: int) -> torch.Tensor:
        if i < 0 or i >= self.n_frames():
            raise IndexError(f"frame {i} out of range [0, {self.n_frames()})")
        return self._coords[i]

    # ── selection / slicing ─────────────────────────────────────────
    def select(self, expr) -> Ensemble:
        """Subset atoms by a ``pa.compute.Expression`` selection.

        Reduces ``topology`` and ``coords`` consistently. Frame
        metadata and ensemble metadata are preserved.
        """
        idx = self._topology.select_indices(expr)
        if idx.numel() == 0:
            raise ValueError("selection matched zero atoms")
        new_topology = self._topology.subset(idx)
        new_coords = self._coords.index_select(1, idx.to(torch.int64))
        return Ensemble(
            new_topology,
            new_coords,
            frame=self._frame,
            frame_metadata=self._frame_metadata,
            metadata=self._metadata,
        )

    def slice_frames(self, sl: slice | torch.Tensor | list[int]) -> Ensemble:
        """Subset frames by slice or index list.

        Topology is unchanged; coords and frame_metadata are reduced.
        """
        if isinstance(sl, slice):
            indices = list(range(*sl.indices(self.n_frames())))
        elif isinstance(sl, torch.Tensor):
            indices = sl.to(torch.int64).tolist()
        else:
            indices = list(sl)
        for i in indices:
            if i < 0 or i >= self.n_frames():
                raise IndexError(f"frame {i} out of range [0, {self.n_frames()})")
        idx_tensor = torch.tensor(indices, dtype=torch.int64)
        new_coords = self._coords.index_select(0, idx_tensor)
        new_frame_meta = self._frame_metadata.take(pa.array(indices, type=pa.int64()))
        return Ensemble(
            self._topology,
            new_coords,
            frame=self._frame,
            frame_metadata=new_frame_meta,
            metadata=self._metadata,
        )

    # ── metadata ────────────────────────────────────────────────────
    def with_metadata(self, extras: dict[str, Any]) -> Ensemble:
        merged = dict(self._metadata)
        merged.update(extras)
        return Ensemble(
            self._topology,
            self._coords,
            frame=self._frame,
            frame_metadata=self._frame_metadata,
            metadata=merged,
        )

    # ── factories ───────────────────────────────────────────────────
    @classmethod
    def single(
        cls,
        topology: Topology,
        coords: torch.Tensor,
        *,
        frame: CoordinateFrame | None = None,
    ) -> Ensemble:
        """Build a single-frame ``Ensemble``.

        Accepts ``(N_atoms, 3)`` or ``(1, N_atoms, 3)`` — both yield
        an ensemble with ``n_frames == 1``.
        """
        return cls(topology, coords, frame=frame)

    # ── repr ────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"Ensemble(n_frames={self.n_frames()}, "
            f"n_atoms={self.n_atoms()}, units={self._frame.units!r})"
        )


# ──────────────────────────────────────────────────────────────────────
# frame_to_table — coord-augmented Arrow projection
# ──────────────────────────────────────────────────────────────────────


def frame_to_table(ensemble: Ensemble, i: int) -> pa.Table:
    """Materialize a single frame as a coord-augmented Arrow table.

    Returns the ``ATOM_TABLE`` columns from ``ensemble.topology.atoms``
    plus three additional ``x`` / ``y`` / ``z`` ``float32`` columns
    holding the coords for frame ``i``. Useful for export / downstream
    consumers that want a flat tabular view.
    """
    coords = ensemble.get_frame(i).to(torch.float32).cpu().numpy()
    atoms = ensemble.topology.atoms
    table = atoms.append_column("x", pa.array(coords[:, 0], type=pa.float32()))
    table = table.append_column("y", pa.array(coords[:, 1], type=pa.float32()))
    table = table.append_column("z", pa.array(coords[:, 2], type=pa.float32()))
    return table


__all__ = [
    "Ensemble",
    "FRAME_METADATA",
    "frame_to_table",
]
