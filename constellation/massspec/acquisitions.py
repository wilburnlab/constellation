"""Per-acquisition provenance for MS data.

An *acquisition* is one LC-MS injection / run. The ``ACQUISITION_TABLE``
schema is the lookup that downstream Search and Quant tables reference
via ``acquisition_id`` foreign keys; the ``Acquisitions`` container
wraps an Arrow table with PK-uniqueness enforcement and a cross-check
helper that other containers call from their ``validate()`` paths.

Lives at the ``massspec/`` top level (rather than inside library/quant/
search) because every empirical sibling references it; one schema, one
container, one helper.

Identity
--------

``acquisition_id`` is a small sequential number, scoped to one
experiment: it is assigned when a run is first registered
(:meth:`Acquisitions.register`), written down once
(:func:`save_acquisitions`), and never reassigned, so every table of the
experiment can be partitioned and joined on it. It says nothing across
experiments; what does is ``source_sha256``, the hash of the instrument
file, which a converted bundle's manifest carries.

Tools name a run their own way. A search engine reports the run it
searched by file name (EncyclopeDIA copies the ``.raw`` name into every
result row as ``SourceFile``), not by any id of ours, so results are
bound back with :meth:`Acquisitions.id_for_source`, which matches on the
file name and falls back to the name without its extension (the same
run may be known as ``run.raw``, ``run.dia`` or ``run.mzML``).

That makes file names part of identity, with one rule: **a file name
means one run**. A run may have several names (the original, a renamed
copy, a file a workflow wrote for it); they are kept as
``source_aliases``. :meth:`Acquisitions.register` checks every name of
an incoming record against every run before it accepts anything, a hash
match included, and refuses a name that already belongs to a different
run, because a result reported under that name would be bound to the
wrong one. A workflow that hands an engine a file of its own making
should not rely on names at all: it says which run the file stands for
(:meth:`Acquisitions.with_source_alias`) and records the binding in its
manifest.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

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
# v2 adds ``source_sha256`` (identity across experiments) and
# ``source_aliases`` (the other file names a run is known by).
ACQUISITION_SCHEMA_VERSION: int = 2


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
        # SHA-256 of the instrument file (hex). The identity of a run
        # across experiments; null when the source was never hashed.
        pa.field("source_sha256", pa.string(), nullable=True),
        # Other file names (no directories) this run is known by, beyond
        # the name in ``source_file``. Null when there are none.
        pa.field("source_aliases", pa.list_(pa.string()), nullable=True),
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


def _chrono_key(value: str | None) -> str | None:
    """Map an ``acquisition_datetime`` string to a chronological sort key.

    Returns a key whose lexicographic order matches true chronological order.
    A raw-string sort is wrong for ISO-8601 values that carry differing UTC
    offsets (e.g. ``2026-01-01T00:30:00+01:00`` is earlier than
    ``2025-12-31T23:45:00Z`` but sorts after it as text). So we parse and, when
    the value is timezone-aware, normalize to UTC before formatting.

    - ``None`` → ``None`` (sorts last; "undated").
    - tz-aware → converted to UTC, formatted ``YYYY-MM-DDTHH:MM:SS.ffffff`` (no
      offset), so all aware values share one comparable reference.
    - naive (no offset) → formatted to the same fixed width. Naive values are
      compared as written (no timezone is assumed — inventing one would be
      wrong); a corpus that mixes naive and offset-aware datetimes is
      inherently ambiguous and the naive values are ordered by their literal
      wall-clock time relative to the UTC-normalized aware ones.

    Raises ``ValueError`` on a non-ISO-8601 value — the field is contractually
    ISO-8601, so a mismatch is a real signal, not something to silently misrank.
    """
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"acquisition_datetime {value!r} is not ISO-8601; cannot order "
            "chronologically"
        ) from exc
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    # Fixed-width microsecond format → lexicographic order == chronological order.
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


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
        on that instrument). Ordering is by a **parsed** UTC-normalized key
        (see :func:`_chrono_key`), not the raw string — so ISO-8601 values
        with differing UTC offsets rank by true instant, not text.

        Semantics:

        - Ties on ``(instrument_serial, chronological time)`` break by
          ``acquisition_id`` ascending (the unique PK) → total, deterministic.
        - ``instrument_serial`` null → ``acquisition_order`` null (not part of
          any instrument's chronology).
        - ``instrument_serial`` present but ``acquisition_datetime`` null →
          sorts **last** within its instrument but still receives a rank
          (invariant: serial present ⇒ order present).
        - Empty table → returned unchanged. Idempotent (overwrites any existing
          ``acquisition_order``).
        - A non-ISO-8601 ``acquisition_datetime`` raises ``ValueError``.

        The returned table's rows are in **sorted order** (by the keys above),
        not the input order; downstream ``acquisition_id`` joins are
        order-insensitive.
        """
        if self.table.num_rows == 0:
            return self
        # Sort on a parsed, UTC-normalized key rather than the raw datetime
        # string: lexicographic order of mixed-offset ISO-8601 text is NOT
        # chronological. Carry the key as a transient column, sort, then drop it.
        keyed = self.table.append_column(
            "_chrono_key",
            pa.array(
                [_chrono_key(v) for v in self.table.column("acquisition_datetime").to_pylist()],
                type=pa.string(),
            ),
        )
        # nulls sort last on ascending — desired for both null serial and null
        # datetime; acquisition_id (unique, non-null) makes the order total.
        sorted_t = keyed.sort_by(
            [
                ("instrument_serial", "ascending"),
                ("_chrono_key", "ascending"),
                ("acquisition_id", "ascending"),
            ]
        ).drop_columns(["_chrono_key"])
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

    # ── identity ────────────────────────────────────────────────────

    def id_for_source(self, name: str) -> int:
        """The ``acquisition_id`` of the run a tool calls ``name``.

        ``name`` is a file name or path as some tool reported it. Tried
        in order: the recorded ``source_file`` exactly; any of the run's
        file names (``source_file`` or an alias); the same name without
        its extension. Raises ``KeyError`` when nothing matches or when
        more than one run does (two runs that differ only by directory
        or extension cannot be told apart by name).
        """
        rows = self.table.to_pylist()
        wanted, wanted_stem = _file_name(name), _stem(name)
        for describe, key in (
            ("source file", lambda r: r["source_file"] == name),
            ("file name", lambda r: wanted in _names_of(r)),
            ("name without extension", lambda r: wanted_stem in {_stem(n) for n in _names_of(r)}),
        ):
            hits = [int(r["acquisition_id"]) for r in rows if key(r)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise KeyError(
                    f"{name!r} matches {len(hits)} acquisitions by {describe} "
                    f"(ids {hits}); it does not identify one run"
                )
        known = ", ".join(sorted(n for r in rows for n in _names_of(r))[:5]) or "none"
        raise KeyError(f"no acquisition is named {name!r} (known: {known})")

    def register(
        self, records: Iterable[dict[str, object]]
    ) -> tuple[Acquisitions, list[int]]:
        """Add runs, keeping every id already handed out.

        Each record needs ``source_file`` and ``source_kind`` and may
        carry the other columns (``source_sha256`` above all, and
        ``source_aliases`` for further names). A record for a run that
        is already here gets that run's id and adds no row; a new one
        gets the next id. Returns the extended container and the id of
        each record, in order.

        Which run a record is, is decided by content when it can be:
        equal ``source_sha256``. When the record or the stored run has
        no hash, it is the file name. Either way every name the record
        brings is first checked against every run:

        - a name that already belongs to a *different* run is refused
          (also when the hash matched some other row: a copy of run 0
          arriving under run 1's name would have results bound to run
          1);
        - two different files (different hashes) with one name are
          refused for the same reason;
        - a name that is new for its run is kept as an alias, so that
          results reported under it are found.
        """
        rows = self.table.to_pylist()
        next_id = max(self.ids, default=-1) + 1
        out: list[int] = []
        for record in records:
            if "source_file" not in record or "source_kind" not in record:
                raise ValueError("an acquisition record needs source_file and source_kind")
            sha = record.get("source_sha256") or None
            names = [_file_name(str(record["source_file"]))]
            names += [_file_name(str(a)) for a in (record.get("source_aliases") or ())]

            same_content = [r for r in rows if sha and r["source_sha256"] == sha]
            if len(same_content) > 1:
                raise ValueError(
                    f"acquisitions {[r['acquisition_id'] for r in same_content]} share "
                    f"sha256 {str(sha)[:12]}...; the table holds one run twice"
                )
            target = same_content[0] if same_content else None
            if target is None:
                # No run with this content is known. A run already using one
                # of the names is this run only if hashes cannot say otherwise.
                holders = _unique_rows(r for n in names for r in rows if n in _names_of(r))
                if len(holders) > 1:
                    raise ValueError(
                        f"the names {names} belong to different acquisitions "
                        f"({[r['acquisition_id'] for r in holders]})"
                    )
                if holders:
                    target = holders[0]
                    held = target["source_sha256"]
                    if sha and held and held != sha:
                        name = next(n for n in names if n in _names_of(target))
                        raise ValueError(
                            f"two different files are both named {name!r} "
                            f"(sha256 {str(held)[:12]}... and {str(sha)[:12]}...); "
                            f"results are bound to runs by file name, so one experiment "
                            f"cannot hold both"
                        )
            if target is None:
                row = {field.name: record.get(field.name) for field in ACQUISITION_TABLE}
                row["acquisition_id"] = next_id
                row["source_sha256"] = sha
                row["source_aliases"] = None
                rows.append(row)
                target = row
                next_id += 1
            _claim_names(rows, target, names)
            if sha and not target["source_sha256"]:
                target["source_sha256"] = sha  # learn the hash; the id stands
            out.append(int(target["acquisition_id"]))  # type: ignore[arg-type]
        return self._rebuilt(rows), out

    def with_source_alias(self, acquisition_id: int, name: str) -> Acquisitions:
        """Say outright that the file ``name`` stands for a run.

        For a workflow that gives an engine a file it wrote itself (a
        spectra file under a unit directory, say): the engine will
        report results under that file's name, and this is what binds
        them. Refused if the name already belongs to another run;
        a no-op if the run already has it.
        """
        rows = self.table.to_pylist()
        target = next((r for r in rows if r["acquisition_id"] == acquisition_id), None)
        if target is None:
            raise KeyError(f"no acquisition {acquisition_id}")
        _claim_names(rows, target, [_file_name(name)])
        return self._rebuilt(rows)

    def _rebuilt(self, rows: list[dict[str, object]]) -> Acquisitions:
        table = pa.Table.from_pylist(rows, schema=ACQUISITION_TABLE)
        return Acquisitions(table.replace_schema_metadata(self.table.schema.metadata))


def _names_of(row: dict[str, object]) -> set[str]:
    """Every file name a run is known by."""
    names = {_file_name(str(row["source_file"]))}
    names.update(row.get("source_aliases") or ())  # type: ignore[arg-type]
    return names


def _unique_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return list({id(r): r for r in rows}.values())


def _claim_names(
    rows: list[dict[str, object]], target: dict[str, object], names: Iterable[str]
) -> None:
    """Give ``target`` each of ``names``, refusing any that another run has.

    Checked against every row, whatever led to ``target`` being chosen.
    """
    for name in names:
        for row in rows:
            if row is not target and name in _names_of(row):
                raise ValueError(
                    f"{name!r} already names acquisition {row['acquisition_id']}, so it cannot "
                    f"also name acquisition {target['acquisition_id']}: results reported under "
                    f"that name could not be bound to one run. Rename the file, or keep the "
                    f"two runs in separate experiments."
                )
    new = [n for n in dict.fromkeys(names) if n not in _names_of(target)]
    if new:
        target["source_aliases"] = sorted([*(target.get("source_aliases") or ()), *new])  # type: ignore[misc]


def _file_name(source: str) -> str:
    """Last path component, for POSIX or Windows spellings alike (a
    result written on Windows names its run with backslashes)."""
    return re.split(r"[\\/]", source)[-1]


def _stem(source: str) -> str:
    name = _file_name(source)
    return name.rsplit(".", 1)[0] if "." in name else name


# ──────────────────────────────────────────────────────────────────────
# Persistence + bundle bridge
# ──────────────────────────────────────────────────────────────────────

ACQUISITIONS_FILENAME = "acquisitions.parquet"


def save_acquisitions(acquisitions: Acquisitions, directory: Path | str) -> Path:
    """Write the table as ``<directory>/acquisitions.parquet`` (atomic:
    a temporary file, then a rename) and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ACQUISITIONS_FILENAME
    tmp = directory / f".{ACQUISITIONS_FILENAME}.tmp"
    try:
        pq.write_table(acquisitions.table, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def load_acquisitions(directory: Path | str) -> Acquisitions:
    """Read what :func:`save_acquisitions` wrote. A file from an older
    schema version loads with the newer columns null."""
    path = Path(directory) / ACQUISITIONS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {ACQUISITIONS_FILENAME} in {Path(directory)}")
    return Acquisitions(pq.read_table(path))


def _iso_datetime(value: object) -> str | None:
    """A run date as ISO-8601, or ``None`` when it is absent or in no
    form met so far. Thermo run headers read ``MM/DD/YYYY HH:MM:SS``."""
    if not value:
        return None
    text = str(value)
    for fmt in ("%m/%d/%Y %H:%M:%S",):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return None
    return text


def acquisition_record_from_bundle(bundle_dir: Path | str) -> dict[str, object]:
    """The registration record for a converted run.

    ``bundle_dir`` is what ``massspec convert`` or ``massspec
    encyclopedia import-spectra`` writes; its ``manifest.json`` names
    the source file, its hash (when it was computed) and the instrument.
    Feed the result to :meth:`Acquisitions.register`.
    """
    manifest_path = Path(bundle_dir) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{Path(bundle_dir)} has no manifest.json: not a converted acquisition bundle"
        ) from None
    source = manifest.get("source_file")
    if not source:
        raise ValueError(f"{manifest_path} names no source_file")
    run = manifest.get("run_metadata") or {}
    return {
        "source_file": str(source),
        "source_kind": str(manifest.get("kind") or "unknown"),
        "source_sha256": manifest.get("source_sha256"),
        "acquisition_datetime": _iso_datetime(run.get("creation_date")),
        "instrument_serial": run.get("instrument_serial"),
        "instrument_model": run.get("instrument_model"),
    }


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
    "ACQUISITIONS_FILENAME",
    "ACQUISITION_SCHEMA_VERSION",
    "ACQUISITION_TABLE",
    "Acquisitions",
    "acquisition_record_from_bundle",
    "load_acquisitions",
    "save_acquisitions",
    "validate_acquisitions",
]
