"""``Quant`` container — empirical / sample-specific MS observations.

Holds five Arrow tables — three per-tier abundance tables and two
per-edge transmission tables — plus an ``Acquisitions`` table for run
provenance. References a ``Library`` by ``library_id`` metadata so that
search-engine outputs and downstream inference know which theoretical
structure these observations were derived from.

Validation:
    PK uniqueness on each (entity_id, acquisition_id) pair (or just
    (entity_id, acquisition_id) for transmissions when acquisition_id
    is non-null; run-agnostic transmission rows are unique on the
    (src, dst) endpoint pair).
    FK closure for acquisition_id values into the held ``Acquisitions``.
    FK closure for entity ids into a provided ``Library`` (optional —
    callers do this when they have the library at hand; if no library
    is passed, only intra-Quant integrity is checked).
    Strict ``efficiency`` ∈ (0, 1] on populated transmission rows
    (-1.0 sentinel is exempt and means "uncalibrated").
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import pyarrow as pa

from constellation.core.io.schemas import cast_to_schema
from constellation.massspec.acquisitions import (
    Acquisitions,
    validate_acquisitions,
)
from constellation.massspec.library.library import Library
from constellation.massspec.quant.schemas import (
    PEPTIDE_QUANT,
    PRECURSOR_QUANT,
    PROTEIN_QUANT,
    TRANSMISSION_PEPTIDE_PRECURSOR,
    TRANSMISSION_PROTEIN_PEPTIDE,
)

# ──────────────────────────────────────────────────────────────────────
# Container
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Quant:
    acquisitions: Acquisitions
    protein_quant: pa.Table
    peptide_quant: pa.Table
    precursor_quant: pa.Table
    transmission_protein_peptide: pa.Table
    transmission_peptide_precursor: pa.Table
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protein_quant",
            cast_to_schema(self.protein_quant, PROTEIN_QUANT),
        )
        object.__setattr__(
            self,
            "peptide_quant",
            cast_to_schema(self.peptide_quant, PEPTIDE_QUANT),
        )
        object.__setattr__(
            self,
            "precursor_quant",
            cast_to_schema(self.precursor_quant, PRECURSOR_QUANT),
        )
        object.__setattr__(
            self,
            "transmission_protein_peptide",
            cast_to_schema(
                self.transmission_protein_peptide, TRANSMISSION_PROTEIN_PEPTIDE
            ),
        )
        object.__setattr__(
            self,
            "transmission_peptide_precursor",
            cast_to_schema(
                self.transmission_peptide_precursor,
                TRANSMISSION_PEPTIDE_PRECURSOR,
            ),
        )
        self.validate()

    # ── empty constructor ───────────────────────────────────────────
    @classmethod
    def empty(cls, acquisitions: Acquisitions | None = None) -> Quant:
        return cls(
            acquisitions=acquisitions or Acquisitions.empty(),
            protein_quant=PROTEIN_QUANT.empty_table(),
            peptide_quant=PEPTIDE_QUANT.empty_table(),
            precursor_quant=PRECURSOR_QUANT.empty_table(),
            transmission_protein_peptide=TRANSMISSION_PROTEIN_PEPTIDE.empty_table(),
            transmission_peptide_precursor=TRANSMISSION_PEPTIDE_PRECURSOR.empty_table(),
        )

    # ── intra-Quant validation ──────────────────────────────────────
    def validate(self) -> None:
        for table, name in (
            (self.protein_quant, "protein_quant"),
            (self.peptide_quant, "peptide_quant"),
            (self.precursor_quant, "precursor_quant"),
        ):
            validate_acquisitions(table, self.acquisitions, nullable=False)
            _check_no_quant_duplicates(table, name)

        for table, name in (
            (self.transmission_protein_peptide, "transmission_protein_peptide"),
            (
                self.transmission_peptide_precursor,
                "transmission_peptide_precursor",
            ),
        ):
            validate_acquisitions(table, self.acquisitions, nullable=True)
            _check_no_transmission_duplicates(table, name)
            _check_efficiency_bounds(table, name)

    # ── cross-validation against a Library ──────────────────────────
    def validate_against(self, library: Library) -> None:
        protein_ids = set(library.proteins.column("protein_id").to_pylist())
        peptide_ids = set(library.peptides.column("peptide_id").to_pylist())
        precursor_ids = set(library.precursors.column("precursor_id").to_pylist())

        _check_fk_in(self.protein_quant, "protein_id", protein_ids, "Library.proteins")
        _check_fk_in(self.peptide_quant, "peptide_id", peptide_ids, "Library.peptides")
        _check_fk_in(
            self.precursor_quant, "precursor_id", precursor_ids,
            "Library.precursors",
        )

        _check_fk_in(
            self.transmission_protein_peptide, "protein_id", protein_ids,
            "Library.proteins",
        )
        _check_fk_in(
            self.transmission_protein_peptide, "peptide_id", peptide_ids,
            "Library.peptides",
        )
        _check_fk_in(
            self.transmission_peptide_precursor, "peptide_id", peptide_ids,
            "Library.peptides",
        )
        _check_fk_in(
            self.transmission_peptide_precursor, "precursor_id", precursor_ids,
            "Library.precursors",
        )

    # ── views ───────────────────────────────────────────────────────
    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_extras)

    def with_metadata(self, extras: dict[str, Any]) -> Quant:
        merged = dict(self.metadata_extras)
        merged.update(extras)
        return replace(self, metadata_extras=merged)

    def with_library_id(self, library_id: str) -> Quant:
        return self.with_metadata({"library_id": library_id})


# ──────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────


_QUANT_PK_COLUMNS = {
    "protein_quant": ("protein_id", "acquisition_id"),
    "peptide_quant": ("peptide_id", "acquisition_id"),
    "precursor_quant": ("precursor_id", "acquisition_id"),
}


def _check_no_quant_duplicates(table: pa.Table, name: str) -> None:
    cols = _QUANT_PK_COLUMNS[name]
    pairs = list(
        zip(*(table.column(c).to_pylist() for c in cols), strict=True)
    )
    if len(set(pairs)) != len(pairs):
        seen: set[tuple[Any, ...]] = set()
        dups: list[tuple[Any, ...]] = []
        for p in pairs:
            if p in seen:
                dups.append(p)
            else:
                seen.add(p)
        raise ValueError(
            f"{name} contains duplicate {cols} pairs: "
            f"{dups[:5]}{'...' if len(dups) > 5 else ''}"
        )


_TRANSMISSION_ENDPOINTS = {
    "transmission_protein_peptide": ("protein_id", "peptide_id"),
    "transmission_peptide_precursor": ("peptide_id", "precursor_id"),
}


def _check_no_transmission_duplicates(table: pa.Table, name: str) -> None:
    src_col, dst_col = _TRANSMISSION_ENDPOINTS[name]
    triples = list(
        zip(
            table.column(src_col).to_pylist(),
            table.column(dst_col).to_pylist(),
            table.column("acquisition_id").to_pylist(),
            strict=True,
        )
    )
    if len(set(triples)) != len(triples):
        seen: set[tuple[Any, ...]] = set()
        dups: list[tuple[Any, ...]] = []
        for t in triples:
            if t in seen:
                dups.append(t)
            else:
                seen.add(t)
        raise ValueError(
            f"{name} has duplicate ({src_col}, {dst_col}, acquisition_id) "
            f"triples: {dups[:5]}{'...' if len(dups) > 5 else ''}"
        )


def _check_efficiency_bounds(table: pa.Table, name: str) -> None:
    """Strict ``(0, 1]`` on populated rows. ``-1.0`` is exempt — it is
    the uncalibrated sentinel and means "no calibration available."
    """
    values = table.column("efficiency").to_pylist()
    bad: list[tuple[int, float]] = []
    for i, v in enumerate(values):
        if v == -1.0:
            continue
        if not (0.0 < v <= 1.0):
            bad.append((i, v))
    if bad:
        raise ValueError(
            f"{name}.efficiency must be in (0, 1] (or -1.0 sentinel); "
            f"violations: {bad[:5]}{'...' if len(bad) > 5 else ''}"
        )


def _check_fk_in(
    table: pa.Table,
    column: str,
    valid_ids: set[Any],
    target_name: str,
) -> None:
    values = table.column(column).to_pylist()
    missing = {v for v in values if v not in valid_ids}
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{column} references ids not present in {target_name}: "
            f"{sample}{'...' if len(missing) > 5 else ''}"
        )


__all__ = ["Quant"]


def assemble_quant(
    *,
    acquisitions: Acquisitions,
    protein_quant: Iterable[dict[str, Any]] = (),
    peptide_quant: Iterable[dict[str, Any]] = (),
    precursor_quant: Iterable[dict[str, Any]] = (),
    transmission_protein_peptide: Iterable[dict[str, Any]] = (),
    transmission_peptide_precursor: Iterable[dict[str, Any]] = (),
    metadata: dict[str, Any] | None = None,
) -> Quant:
    """Build a ``Quant`` from record lists keyed by id, applying defaults
    for nullable / unset fields. Convenience for tests and ad-hoc use."""
    def _table(rows: Iterable[dict[str, Any]], schema: pa.Schema) -> pa.Table:
        rows = list(rows)
        if not rows:
            return schema.empty_table()
        return pa.Table.from_pylist(rows, schema=schema)

    return Quant(
        acquisitions=acquisitions,
        protein_quant=_table(protein_quant, PROTEIN_QUANT),
        peptide_quant=_table(peptide_quant, PEPTIDE_QUANT),
        precursor_quant=_table(precursor_quant, PRECURSOR_QUANT),
        transmission_protein_peptide=_table(
            transmission_protein_peptide, TRANSMISSION_PROTEIN_PEPTIDE
        ),
        transmission_peptide_precursor=_table(
            transmission_peptide_precursor, TRANSMISSION_PEPTIDE_PRECURSOR
        ),
        metadata_extras=dict(metadata or {}),
    )


__all__.append("assemble_quant")
