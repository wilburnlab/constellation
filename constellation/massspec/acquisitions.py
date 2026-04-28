"""Per-acquisition provenance for MS data.

An *acquisition* is one LC-MS injection / run. The ``ACQUISITION_TABLE``
schema is the lookup that downstream Search and Quant tables reference
via ``acquisition_id`` foreign keys; the ``Acquisitions`` container
wraps an Arrow table with PK-uniqueness enforcement and a cross-check
helper that other containers call from their ``validate()`` paths.

Lives at the ``massspec/`` top level (rather than inside library/quant/
search) because every empirical sibling references it; one schema, one
container, one helper.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pyarrow as pa

from constellation.core.io.schemas import (
    cast_to_schema,
    pack_metadata,
    register_schema,
    unpack_metadata,
)

# ──────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────


ACQUISITION_TABLE: pa.Schema = pa.schema(
    [
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_kind", pa.string(), nullable=False),
        pa.field("acquisition_datetime", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"AcquisitionTable"},
)


register_schema("AcquisitionTable", ACQUISITION_TABLE)


# ──────────────────────────────────────────────────────────────────────
# Container
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Acquisitions:
    """Container around an ``ACQUISITION_TABLE``-shaped Arrow table.

    Cast on construction (so callers can hand in a slightly wider table
    and have extras dropped); duplicate ``acquisition_id`` values raise.
    """

    table: pa.Table

    def __post_init__(self) -> None:
        cast = cast_to_schema(self.table, ACQUISITION_TABLE)
        object.__setattr__(self, "table", cast)
        ids = cast.column("acquisition_id").to_pylist()
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(
                f"acquisition_id contains duplicates: {duplicates[:5]}"
                f"{'...' if len(duplicates) > 5 else ''}"
            )

    @classmethod
    def empty(cls) -> Acquisitions:
        return cls(ACQUISITION_TABLE.empty_table())

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, object]],
    ) -> Acquisitions:
        rows = list(records)
        if not rows:
            return cls.empty()
        table = pa.Table.from_pylist(rows, schema=ACQUISITION_TABLE)
        return cls(table)

    @property
    def ids(self) -> list[int]:
        return self.table.column("acquisition_id").to_pylist()

    @property
    def metadata(self) -> dict[str, object]:
        return unpack_metadata(self.table.schema.metadata)

    def with_metadata(self, extras: dict[str, object]) -> Acquisitions:
        existing = unpack_metadata(self.table.schema.metadata)
        existing.update(extras)
        new_table = self.table.replace_schema_metadata(pack_metadata(existing))
        return Acquisitions(new_table)

    def __len__(self) -> int:
        return self.table.num_rows


# ──────────────────────────────────────────────────────────────────────
# Cross-check helper
# ──────────────────────────────────────────────────────────────────────


def validate_acquisitions(
    table: pa.Table,
    acquisitions: Acquisitions,
    *,
    column: str = "acquisition_id",
    nullable: bool = False,
) -> None:
    """Raise if ``table[column]`` references an unknown ``acquisition_id``.

    Used by Search/Quant containers from their ``validate()`` paths.
    ``nullable=True`` permits a column where missing values mean
    "applies to all acquisitions" (e.g. run-agnostic transmission
    calibration).
    """
    if column not in table.column_names:
        raise ValueError(f"table missing FK column {column!r}")
    col = table.column(column)
    known = set(acquisitions.ids)
    values = col.to_pylist()
    unknown: set[int] = set()
    for v in values:
        if v is None:
            if not nullable:
                raise ValueError(
                    f"column {column!r} contains nulls but nullable=False"
                )
            continue
        if v not in known:
            unknown.add(v)
    if unknown:
        sample = sorted(unknown)[:5]
        raise ValueError(
            f"{column} references unknown acquisition_ids: "
            f"{sample}{'...' if len(unknown) > 5 else ''}"
        )


__all__ = [
    "ACQUISITION_TABLE",
    "Acquisitions",
    "validate_acquisitions",
]
