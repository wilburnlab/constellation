"""``Search`` container — peptide-level and protein-level detection scores.

Holds the peptide/protein score tables (``peptide_scores``,
``protein_scores``) and an optional per-spectrum match table (``psms``,
default empty) plus an ``Acquisitions`` table for run provenance.
References a ``Library`` by ``library_id`` metadata so downstream tooling
can reach the peptide / protein identities the score rows refer to.

Validation:
    PK uniqueness on ``(entity_id, acquisition_id, engine)`` for the
        score tables — one score row per entity per run per engine.
    PK uniqueness on ``(raw_file, psm_id)`` for ``psms`` — a single
        MS/MS scan may yield more than one match, so ``(raw_file, scan)``
        is not a key.
    FK closure for ``acquisition_id`` values into the held
        ``Acquisitions`` (when non-null) — applies to all three tables.
    FK closure for entity ids into a provided ``Library`` (optional —
        callers do this when they have the library at hand). Does NOT
        cover ``psms.peptide_id``, which is an engine-internal index,
        not a library FK.
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
from constellation.massspec.search.schemas import (
    PEPTIDE_SCORE_TABLE,
    PROTEIN_SCORE_TABLE,
    PSM_TABLE,
)


@dataclass(frozen=True, slots=True)
class Search:
    acquisitions: Acquisitions
    peptide_scores: pa.Table
    protein_scores: pa.Table
    metadata_extras: dict[str, Any] = field(default_factory=dict)
    # Appended last so the existing positional signature
    # ``Search(acquisitions, peptide_scores, protein_scores, metadata_extras)``
    # is unchanged. Default empty: aggregating-only readers leave it be.
    psms: pa.Table = field(default_factory=PSM_TABLE.empty_table)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "peptide_scores",
            cast_to_schema(self.peptide_scores, PEPTIDE_SCORE_TABLE),
        )
        object.__setattr__(
            self,
            "protein_scores",
            cast_to_schema(self.protein_scores, PROTEIN_SCORE_TABLE),
        )
        object.__setattr__(self, "psms", cast_to_schema(self.psms, PSM_TABLE))
        self.validate()

    @classmethod
    def empty(cls, acquisitions: Acquisitions | None = None) -> Search:
        return cls(
            acquisitions=acquisitions or Acquisitions.empty(),
            peptide_scores=PEPTIDE_SCORE_TABLE.empty_table(),
            protein_scores=PROTEIN_SCORE_TABLE.empty_table(),
            psms=PSM_TABLE.empty_table(),
        )

    # ── intra-Search validation ─────────────────────────────────────
    def validate(self) -> None:
        for table, name in (
            (self.peptide_scores, "peptide_scores"),
            (self.protein_scores, "protein_scores"),
            (self.psms, "psms"),
        ):
            # acquisition_id is nullable for run-agnostic / pre-link rows.
            validate_acquisitions(table, self.acquisitions, nullable=True)
        _check_no_duplicates(self.peptide_scores, "peptide_scores", "peptide_id")
        _check_no_duplicates(self.protein_scores, "protein_scores", "protein_id")
        _check_no_psm_duplicates(self.psms)

    # ── cross-validation against a Library ──────────────────────────
    def validate_against(self, library: Library) -> None:
        peptide_ids = set(library.peptides.column("peptide_id").to_pylist())
        protein_ids = set(library.proteins.column("protein_id").to_pylist())
        _check_fk_in(
            self.peptide_scores, "peptide_id", peptide_ids, "Library.peptides"
        )
        _check_fk_in(
            self.protein_scores, "protein_id", protein_ids, "Library.proteins"
        )

    # ── views ───────────────────────────────────────────────────────
    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_extras)

    def with_metadata(self, extras: dict[str, Any]) -> Search:
        merged = dict(self.metadata_extras)
        merged.update(extras)
        return replace(self, metadata_extras=merged)

    def with_library_id(self, library_id: str) -> Search:
        return self.with_metadata({"library_id": library_id})


# ──────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────


def _check_no_duplicates(table: pa.Table, name: str, entity_col: str) -> None:
    triples = list(
        zip(
            table.column(entity_col).to_pylist(),
            table.column("acquisition_id").to_pylist(),
            table.column("engine").to_pylist(),
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
            f"{name} has duplicate ({entity_col}, acquisition_id, engine) "
            f"triples: {dups[:5]}{'...' if len(dups) > 5 else ''}"
        )


def _check_no_psm_duplicates(table: pa.Table) -> None:
    """PK uniqueness on ``(raw_file, psm_id)`` for the ``psms`` table.

    A single MS/MS scan can produce more than one match (e.g. MaxQuant
    ``Type=MULTI-SECPEP``), so ``(raw_file, scan)`` is NOT a key; the
    engine's per-export row id paired with ``raw_file`` is.
    """
    pairs = list(
        zip(
            table.column("raw_file").to_pylist(),
            table.column("psm_id").to_pylist(),
            strict=True,
        )
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
            f"psms has duplicate (raw_file, psm_id) pairs: "
            f"{dups[:5]}{'...' if len(dups) > 5 else ''}"
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


def assemble_search(
    *,
    acquisitions: Acquisitions,
    peptide_scores: Iterable[dict[str, Any]] = (),
    protein_scores: Iterable[dict[str, Any]] = (),
    psms: Iterable[dict[str, Any]] = (),
    metadata: dict[str, Any] | None = None,
) -> Search:
    """Build a ``Search`` from record lists. Convenience for tests / readers."""

    def _table(rows: Iterable[dict[str, Any]], schema: pa.Schema) -> pa.Table:
        rows = list(rows)
        if not rows:
            return schema.empty_table()
        return pa.Table.from_pylist(rows, schema=schema)

    return Search(
        acquisitions=acquisitions,
        peptide_scores=_table(peptide_scores, PEPTIDE_SCORE_TABLE),
        protein_scores=_table(protein_scores, PROTEIN_SCORE_TABLE),
        psms=_table(psms, PSM_TABLE),
        metadata_extras=dict(metadata or {}),
    )


__all__ = ["Search", "assemble_search"]
