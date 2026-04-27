"""Selection-predicate helpers for ``STRUCTURE_TABLE`` Arrow tables.

Each helper returns a ``pa.compute.Expression`` over the column names
declared in ``coords.STRUCTURE_TABLE``. Expressions compose with the
standard Python operators ``&`` / ``|`` / ``~``, so callers build
complex selections without a string DSL:

    >>> from constellation.core.structure import (
    ...     select_chain, select_backbone, select_resname,
    ... )
    >>> expr = select_chain("A") & select_backbone() & ~select_resname({"PRO"})

Pass the resulting expression to ``Topology.select_indices`` /
``Ensemble.select`` to get a sub-topology / sub-ensemble. Or apply it
directly via ``table.filter(expr)``.

A VMD-style string-DSL parser is intentionally not provided — once you
have ~8 named predicates, composition with ``&`` / ``|`` / ``~`` reads
better than a mini-language and avoids quoting / escaping headaches.
"""

from __future__ import annotations

from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc


# ──────────────────────────────────────────────────────────────────────
# Constants — canonical residue / atom-name sets
# ──────────────────────────────────────────────────────────────────────

# Standard 20 amino-acid 3-letter codes.
_PROTEIN_RESNAMES: frozenset[str] = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)

# Common water residue names across PDB / Amber / GROMACS / CHARMM.
_WATER_RESNAMES: frozenset[str] = frozenset(
    {"HOH", "WAT", "H2O", "TIP3", "TIP4", "TIP5", "SOL", "DOD"}
)

# Backbone heavy atoms in the PDB convention. Hydrogens (HA, H, ...)
# are intentionally excluded — most use cases want heavy-atom
# backbone; callers wanting H can union with ``select_atom_names``.
_BACKBONE_NAMES: frozenset[str] = frozenset({"N", "CA", "C", "O"})


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _as_set(values: str | Iterable[str]) -> list[str]:
    """Normalize a single string or iterable of strings to a list.

    Strings are wrapped (``"A"`` ≠ list of single chars). Empty
    iterables raise — an empty selection is almost always a bug.
    """
    if isinstance(values, str):
        return [values]
    out = list(values)
    if not out:
        raise ValueError("selection helper requires at least one value")
    return out


# ──────────────────────────────────────────────────────────────────────
# Predicate helpers
# ──────────────────────────────────────────────────────────────────────


def select_chain(chain_id: str | Iterable[str]) -> pc.Expression:
    """Atoms whose ``chain_id`` is in the given set."""
    values = _as_set(chain_id)
    if len(values) == 1:
        return pc.field("chain_id") == values[0]
    return pc.field("chain_id").isin(values)


def select_residues(
    start: int,
    end: int,
    chain_id: str | None = None,
) -> pc.Expression:
    """Atoms with ``res_seq`` in ``[start, end]`` (inclusive on both ends).

    Optionally restrict to a single chain via ``chain_id``.
    """
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")
    expr = (pc.field("res_seq") >= start) & (pc.field("res_seq") <= end)
    if chain_id is not None:
        expr = expr & (pc.field("chain_id") == chain_id)
    return expr


def select_resname(names: str | Iterable[str]) -> pc.Expression:
    """Atoms whose ``res_name`` is in the given set."""
    values = _as_set(names)
    if len(values) == 1:
        return pc.field("res_name") == values[0]
    return pc.field("res_name").isin(values)


def select_atom_names(names: str | Iterable[str]) -> pc.Expression:
    """Atoms whose ``name`` is in the given set."""
    values = _as_set(names)
    if len(values) == 1:
        return pc.field("name") == values[0]
    return pc.field("name").isin(values)


def select_backbone() -> pc.Expression:
    """Protein backbone heavy atoms (N, CA, C, O)."""
    return pc.field("name").isin(list(_BACKBONE_NAMES))


def select_protein() -> pc.Expression:
    """Atoms belonging to one of the standard 20 amino-acid residues."""
    return pc.field("res_name").isin(list(_PROTEIN_RESNAMES))


def select_sidechain() -> pc.Expression:
    """Protein sidechain atoms — protein residues, non-backbone names."""
    return select_protein() & ~select_backbone()


def select_water() -> pc.Expression:
    """Water residues (HOH/WAT/H2O/TIP*/SOL/DOD)."""
    return pc.field("res_name").isin(list(_WATER_RESNAMES))


def select_hetatm() -> pc.Expression:
    """Heteroatoms (PDB ``HETATM`` records)."""
    return pc.field("is_hetatm") == pa.scalar(True)


def select_element(symbol: str | Iterable[str]) -> pc.Expression:
    """Atoms whose element symbol is in the given set."""
    values = _as_set(symbol)
    if len(values) == 1:
        return pc.field("element") == values[0]
    return pc.field("element").isin(values)


__all__ = [
    "select_chain",
    "select_residues",
    "select_resname",
    "select_atom_names",
    "select_backbone",
    "select_sidechain",
    "select_protein",
    "select_water",
    "select_hetatm",
    "select_element",
]
