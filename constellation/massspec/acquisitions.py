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


# Bumps on any additive column change. v1 = first versioned form (adds
# per-instrument identity + chronological order to the original 4 columns).
# Additive + nullable, so older 1-version-behind parquet still loads:
# ``cast_to_schema`` fills the missing columns with nulls and merges this
# (newer) metadata in, so a file read by newer code is treated as v1-shaped.
ACQUISITION_SCHEMA_VERSION: int = 1


ACQUISITION_TABLE: pa.Schema = pa.schema(
    [
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_kind", pa.string(), nullable=False),
        pa.field("acquisition_datetime", pa.string(), nullable=True),
        # Per-instrument grouping + chronological position. All nullable:
        # non-Thermo / older sources may lack instrument identity, and
        # ``acquisition_order`` is a derived rank (see with_acquisition_order)
        # left null until computed.
        pa.field("instrument_serial", pa.string(), nullable=True),
        pa.field("instrument_model", pa.string(), nullable=True),
        pa.field("acquisition_order", pa.int32(), nullable=True),
    ],
    metadata={
        b"schema_name": b"AcquisitionTable",
        b"schema_version": str(ACQUISITION_SCHEMA_VERSION).encode("utf-8"),
    },
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

    def with_acquisition_order(self) -> Acquisitions:
        """Return a new ``Acquisitions`` with ``acquisition_order`` filled.

        Chronological rank **within each instrument**: group by
        ``instrument_serial``, order by ``acquisition_datetime`` ascending,
        and assign a **1-based** ``acquisition_order`` (rank 1 = earliest run
        on that instrument). ``acquisition_datetime`` is compared as a string —
        correct for ISO-8601 (``YYYY-MM-DDT…``), which sorts lexicographically.

        Semantics:

        - Ties on ``(instrument_serial, acquisition_datetime)`` break by
          ``acquisition_id`` ascending (the unique PK) → total, deterministic.
        - ``instrument_serial`` null → ``acquisition_order`` null (not part of
          any instrument's chronology).
        - ``instrument_serial`` present but ``acquisition_datetime`` null →
          sorts **last** within its instrument but still receives a rank
          (invariant: serial present ⇒ order present).
        - Empty table → returned unchanged. Idempotent (overwrites any existing
          ``acquisition_order``).

        The returned table's rows are in **sorted order** (by the keys above),
        not the input order; downstream ``acquisition_id`` joins are
        order-insensitive.
        """
        if self.table.num_rows == 0:
            return self
        # nulls sort last on ascending — desired for both null serial and null
        # datetime; acquisition_id (unique, non-null) makes the order total.
        sorted_t = self.table.sort_by(
            [
                ("instrument_serial", "ascending"),
                ("acquisition_datetime", "ascending"),
                ("acquisition_id", "ascending"),
            ]
        )
        serials = sorted_t.column("instrument_serial").to_pylist()
        order: list[int | None] = []
        current: object = object()  # sentinel distinct from any serial incl. None
        rank = 0
        for serial in serials:
            if serial is None:
                order.append(None)
                continue
            if serial != current:
                current = serial
                rank = 1
            else:
                rank += 1
            order.append(rank)
        idx = sorted_t.schema.get_field_index("acquisition_order")
        out = sorted_t.set_column(
            idx, "acquisition_order", pa.array(order, type=pa.int32())
        )
        return Acquisitions(out)

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
    "ACQUISITION_SCHEMA_VERSION",
    "ACQUISITION_TABLE",
    "Acquisitions",
    "validate_acquisitions",
]
