"""Tests for ``constellation.massspec.acquisitions``."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.core.io.schemas import get_schema
from constellation.massspec.acquisitions import (
    ACQUISITION_SCHEMA_VERSION,
    ACQUISITION_TABLE,
    Acquisitions,
    validate_acquisitions,
)


def _records():
    return [
        {
            "acquisition_id": 0,
            "source_file": "run_01.raw",
            "source_kind": "thermo_raw",
            "acquisition_datetime": "2026-04-28T10:00:00Z",
        },
        {
            "acquisition_id": 1,
            "source_file": "run_02.mzML",
            "source_kind": "mzml",
            "acquisition_datetime": None,
        },
    ]


def test_schema_registered() -> None:
    assert get_schema("AcquisitionTable") is ACQUISITION_TABLE


def test_from_records_round_trip() -> None:
    acq = Acquisitions.from_records(_records())
    assert acq.ids == [0, 1]
    assert acq.table.column("source_kind").to_pylist() == ["thermo_raw", "mzml"]


def test_empty() -> None:
    acq = Acquisitions.empty()
    assert len(acq) == 0
    assert acq.ids == []


def test_duplicate_acquisition_id_rejected() -> None:
    rows = _records()
    rows[1]["acquisition_id"] = 0
    with pytest.raises(ValueError, match="duplicates"):
        Acquisitions.from_records(rows)


def test_extra_columns_dropped_on_cast() -> None:
    rows = _records()
    rows[0]["extra_column"] = "ignored"
    acq = Acquisitions.from_records(rows)
    assert "extra_column" not in acq.table.column_names


def test_with_metadata_round_trip() -> None:
    acq = Acquisitions.from_records(_records()).with_metadata(
        {"x.library.source": "test_set"}
    )
    assert acq.metadata["x.library.source"] == "test_set"


def test_validate_acquisitions_passes_for_known_ids() -> None:
    acq = Acquisitions.from_records(_records())
    table = pa.table({"acquisition_id": [0, 1, 0]})
    validate_acquisitions(table, acq)


def test_validate_acquisitions_rejects_unknown_id() -> None:
    acq = Acquisitions.from_records(_records())
    table = pa.table({"acquisition_id": [0, 99]})
    with pytest.raises(ValueError, match="unknown acquisition_ids"):
        validate_acquisitions(table, acq)


def test_validate_acquisitions_rejects_null_when_not_nullable() -> None:
    acq = Acquisitions.from_records(_records())
    table = pa.table({"acquisition_id": pa.array([0, None], type=pa.int64())})
    with pytest.raises(ValueError, match="nullable=False"):
        validate_acquisitions(table, acq)


def test_validate_acquisitions_allows_null_when_nullable() -> None:
    acq = Acquisitions.from_records(_records())
    table = pa.table({"acquisition_id": pa.array([0, None], type=pa.int64())})
    validate_acquisitions(table, acq, nullable=True)


def test_validate_acquisitions_missing_column_raises() -> None:
    acq = Acquisitions.from_records(_records())
    table = pa.table({"other_column": [0, 1]})
    with pytest.raises(ValueError, match="missing FK column"):
        validate_acquisitions(table, acq)


# ──────────────────────────────────────────────────────────────────────
# Schema v1: instrument identity + per-instrument chronological order
# ──────────────────────────────────────────────────────────────────────


def _instrument_records():
    """Two instruments interleaved in input order; a tie; a null datetime;
    a null-serial row. ``acquisition_id`` is the stable key for assertions."""
    return [
        # instrument A, later run (input first to prove ordering, not input pos)
        {"acquisition_id": 10, "source_file": "b.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2016-06-02T00:00:00",
         "instrument_serial": "A", "instrument_model": "Lumos"},
        # instrument A, earlier run
        {"acquisition_id": 11, "source_file": "a.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2016-06-01T00:00:00",
         "instrument_serial": "A", "instrument_model": "Lumos"},
        # instrument B, tie on datetime with id 13 → id breaks (12 < 13)
        {"acquisition_id": 12, "source_file": "c.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2016-06-01T00:00:00",
         "instrument_serial": "B", "instrument_model": "Lumos"},
        {"acquisition_id": 13, "source_file": "d.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2016-06-01T00:00:00",
         "instrument_serial": "B", "instrument_model": "Lumos"},
        # instrument A, null datetime → sorts last in A but still ranked
        {"acquisition_id": 14, "source_file": "e.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": None,
         "instrument_serial": "A", "instrument_model": "Lumos"},
        # null instrument_serial → null order
        {"acquisition_id": 15, "source_file": "f.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2016-06-01T00:00:00",
         "instrument_serial": None, "instrument_model": None},
    ]


def _order_by_id(acq: Acquisitions) -> dict[int, int | None]:
    rows = acq.table.to_pylist()
    return {r["acquisition_id"]: r["acquisition_order"] for r in rows}


def test_schema_has_version_and_instrument_fields() -> None:
    assert ACQUISITION_TABLE.metadata[b"schema_version"] == b"2"
    assert ACQUISITION_SCHEMA_VERSION == 2
    assert ACQUISITION_TABLE.field("source_sha256").nullable
    names = set(ACQUISITION_TABLE.names)
    assert {"instrument_serial", "instrument_model", "acquisition_order"} <= names
    assert ACQUISITION_TABLE.field("instrument_serial").type == pa.string()
    assert ACQUISITION_TABLE.field("acquisition_order").type == pa.int32()


def test_instrument_fields_round_trip() -> None:
    acq = Acquisitions.from_records(_instrument_records())
    by_id = {r["acquisition_id"]: r for r in acq.table.to_pylist()}
    assert by_id[10]["instrument_serial"] == "A"
    assert by_id[10]["instrument_model"] == "Lumos"


def test_backward_compat_four_column_table() -> None:
    # The literal old-parquet shape: only the original 4 columns, no metadata.
    old = pa.table(
        {
            "acquisition_id": pa.array([0, 1], type=pa.int64()),
            "source_file": ["run_01.raw", "run_02.mzML"],
            "source_kind": ["thermo_raw", "mzml"],
            "acquisition_datetime": ["2016-01-01T00:00:00", None],
        }
    )
    acq = Acquisitions(old)
    assert acq.table.column("instrument_serial").to_pylist() == [None, None]
    assert acq.table.column("instrument_model").to_pylist() == [None, None]
    assert acq.table.column("acquisition_order").to_pylist() == [None, None]


def test_with_acquisition_order_basic() -> None:
    by_id = _order_by_id(Acquisitions.from_records(_instrument_records()).with_acquisition_order())
    # instrument A: id 11 earliest → 1, id 10 → 2
    assert by_id[11] == 1
    assert by_id[10] == 2


def test_with_acquisition_order_tie_break_by_id() -> None:
    by_id = _order_by_id(Acquisitions.from_records(_instrument_records()).with_acquisition_order())
    # instrument B: 12 and 13 share a datetime → lower id ranks first
    assert by_id[12] == 1
    assert by_id[13] == 2


def test_with_acquisition_order_null_datetime_sorts_last_but_ranked() -> None:
    by_id = _order_by_id(Acquisitions.from_records(_instrument_records()).with_acquisition_order())
    # instrument A has 3 rows (11, 10, 14-null-dt) → null-dt gets the last rank
    assert by_id[14] == 3


def test_with_acquisition_order_null_serial_gets_null_order() -> None:
    by_id = _order_by_id(Acquisitions.from_records(_instrument_records()).with_acquisition_order())
    assert by_id[15] is None


def test_with_acquisition_order_idempotent() -> None:
    once = Acquisitions.from_records(_instrument_records()).with_acquisition_order()
    twice = once.with_acquisition_order()
    assert _order_by_id(once) == _order_by_id(twice)


def test_with_acquisition_order_empty() -> None:
    acq = Acquisitions.empty().with_acquisition_order()
    assert len(acq) == 0


def test_with_acquisition_order_preserves_uniqueness() -> None:
    acq = Acquisitions.from_records(_instrument_records()).with_acquisition_order()
    assert len(set(acq.ids)) == len(acq.ids)


def test_with_acquisition_order_ranks_by_utc_instant_not_string() -> None:
    # Same instrument, ISO-8601 with differing offsets: +01:00 row is 23:30 UTC
    # (earlier) and the Z row is 23:45 UTC (later), but raw-string sort would
    # place the 2025…Z row first. The parsed UTC key must rank the +01:00 first.
    recs = [
        {"acquisition_id": 1, "source_file": "later.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2025-12-31T23:45:00Z",
         "instrument_serial": "A", "instrument_model": "M"},
        {"acquisition_id": 2, "source_file": "earlier.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2026-01-01T00:30:00+01:00",
         "instrument_serial": "A", "instrument_model": "M"},
    ]
    by_file = {
        r["source_file"]: r["acquisition_order"]
        for r in Acquisitions.from_records(recs).with_acquisition_order().table.to_pylist()
    }
    assert by_file["earlier.raw"] == 1
    assert by_file["later.raw"] == 2


def test_with_acquisition_order_offset_tie_broken_by_id() -> None:
    # Same UTC instant via different representations (23:45Z == 00:45+01:00) →
    # a true tie, broken by acquisition_id ascending.
    recs = [
        {"acquisition_id": 5, "source_file": "z.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2025-12-31T23:45:00Z",
         "instrument_serial": "A", "instrument_model": "M"},
        {"acquisition_id": 4, "source_file": "off.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "2026-01-01T00:45:00+01:00",
         "instrument_serial": "A", "instrument_model": "M"},
    ]
    by_file = {
        r["source_file"]: r["acquisition_order"]
        for r in Acquisitions.from_records(recs).with_acquisition_order().table.to_pylist()
    }
    assert by_file["off.raw"] == 1  # id 4 < 5
    assert by_file["z.raw"] == 2


def test_with_acquisition_order_rejects_non_iso_datetime() -> None:
    recs = [
        {"acquisition_id": 1, "source_file": "x.raw", "source_kind": "thermo_raw",
         "acquisition_datetime": "not-a-date",
         "instrument_serial": "A", "instrument_model": "M"},
    ]
    with pytest.raises(ValueError, match="not ISO-8601"):
        Acquisitions.from_records(recs).with_acquisition_order()


# ── identity: ids fixed once, runs bound by name ───────────────────────


def _raw(name: str, sha: str | None = None, **extra) -> dict[str, object]:
    return {"source_file": name, "source_kind": "thermo_acquisition", "source_sha256": sha, **extra}


def test_register_hands_out_ids_once_and_keeps_them() -> None:
    from constellation.massspec.acquisitions import Acquisitions

    acq, ids = Acquisitions.empty().register([_raw("/data/a.raw", "aa"), _raw("/data/b.raw", "bb")])
    assert ids == [0, 1] and acq.ids == [0, 1]

    # A later call with one known run (found by content, under another
    # path) and one new run: the known id stands, the new one is next.
    acq2, ids2 = acq.register([_raw("/elsewhere/a.raw", "aa"), _raw("/data/c.raw", "cc")])
    assert ids2 == [0, 2] and acq2.ids == [0, 1, 2]
    assert acq2.table.column("source_file").to_pylist()[0] == "/data/a.raw"
    assert acq.ids == [0, 1]  # the original is untouched

    # Without a hash, the file name decides; the hash is learned later.
    bare, _ = Acquisitions.empty().register([_raw("/data/a.raw")])
    learned, ids3 = bare.register([_raw("/copy/a.raw", "aa")])
    assert ids3 == [0] and learned.table.column("source_sha256").to_pylist() == ["aa"]


def test_register_refuses_two_files_with_one_name() -> None:
    from constellation.massspec.acquisitions import Acquisitions

    acq, _ = Acquisitions.empty().register([_raw("/monday/run.raw", "aa")])
    with pytest.raises(ValueError, match="two different files are both named 'run.raw'"):
        acq.register([_raw("/tuesday/run.raw", "bb")])
    with pytest.raises(ValueError, match="needs source_file and source_kind"):
        acq.register([{"source_file": "x.raw"}])


def test_id_for_source_binds_what_a_tool_reports() -> None:
    from constellation.massspec.acquisitions import Acquisitions

    acq, _ = Acquisitions.empty().register(
        [_raw("/data/run_01.raw", "aa"), _raw("/data/run_02.raw", "bb")]
    )
    assert acq.id_for_source("/data/run_02.raw") == 1  # exactly as recorded
    assert acq.id_for_source("run_01.raw") == 0  # EncyclopeDIA's SourceFile
    assert acq.id_for_source("E:\\somewhere\\run_02.raw") == 1  # a Windows path
    assert acq.id_for_source("run_01.dia") == 0  # same run, another extension
    with pytest.raises(KeyError, match="no acquisition is named 'run_03.raw'"):
        acq.id_for_source("run_03.raw")

    # Two runs that differ only by extension cannot be bound by bare name.
    both, _ = Acquisitions.empty().register([_raw("run.raw", "aa"), _raw("run.mzML", "bb")])
    assert both.id_for_source("run.raw") == 0
    with pytest.raises(KeyError, match="matches 2 acquisitions by name without extension"):
        both.id_for_source("run.dia")


def test_acquisitions_persist_and_older_files_load(tmp_path) -> None:
    import pyarrow.parquet as pq

    from constellation.massspec.acquisitions import (
        Acquisitions,
        load_acquisitions,
        save_acquisitions,
    )

    acq, _ = Acquisitions.empty().register([_raw("/data/a.raw", "aa", instrument_model="Stellar")])
    path = save_acquisitions(acq, tmp_path / "acquisitions")
    assert path.name == "acquisitions.parquet" and not list(path.parent.glob(".*tmp"))
    again = load_acquisitions(tmp_path / "acquisitions")
    assert again.table.to_pylist() == acq.table.to_pylist()
    # Reloaded, a known run still gets the id it had.
    assert again.register([_raw("/moved/a.raw", "aa")])[1] == [0]

    # A version-1 file (no source_sha256) loads with the column null.
    old = acq.table.drop_columns(["source_sha256"])
    (tmp_path / "old").mkdir()
    pq.write_table(old, tmp_path / "old" / "acquisitions.parquet")
    assert load_acquisitions(tmp_path / "old").table.column("source_sha256").to_pylist() == [None]
    with pytest.raises(FileNotFoundError):
        load_acquisitions(tmp_path / "absent")


def test_record_from_a_converted_bundle(tmp_path) -> None:
    import json

    from constellation.massspec.acquisitions import Acquisitions, acquisition_record_from_bundle

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "thermo_acquisition",
                "source_file": "/data/run_01.raw",
                "source_sha256": "a5766964",
                "run_metadata": {
                    "instrument_model": "Stellar",
                    "instrument_serial": "QLT-S-10199",
                    "creation_date": "10/06/2026 22:19:53",
                },
            }
        )
    )
    record = acquisition_record_from_bundle(tmp_path)
    assert record == {
        "source_file": "/data/run_01.raw",
        "source_kind": "thermo_acquisition",
        "source_sha256": "a5766964",
        "acquisition_datetime": "2026-10-06T22:19:53",
        "instrument_serial": "QLT-S-10199",
        "instrument_model": "Stellar",
    }
    acq, ids = Acquisitions.empty().register([record])
    assert ids == [0] and acq.with_acquisition_order().table.column("acquisition_order").to_pylist() == [1]
    with pytest.raises(FileNotFoundError, match="not a converted acquisition bundle"):
        acquisition_record_from_bundle(tmp_path / "nowhere")


# ── names are checked against every run, and a run keeps all its names ─
# (Regressions from the review of b1fe1ca: a hash match used to end
# validation before the remaining file-name conflicts were looked at.)


def test_a_copy_registered_under_another_runs_name_is_refused() -> None:
    """a.raw and b.raw are two runs. A copy of a.raw, named b.raw, is run
    0 by content, but "b.raw" already means run 1: a result reported
    under that name would bind to the wrong run, so it is refused."""
    from constellation.massspec.acquisitions import Acquisitions

    acq, ids = Acquisitions.empty().register([_raw("a.raw", "hash-A"), _raw("b.raw", "hash-B")])
    assert ids == [0, 1]
    with pytest.raises(ValueError, match="'b.raw' already names acquisition 1"):
        acq.register([_raw("/copies/b.raw", "hash-A")])
    # Order of the existing rows must not matter.
    swapped, _ = Acquisitions.empty().register([_raw("b.raw", "hash-B"), _raw("a.raw", "hash-A")])
    with pytest.raises(ValueError, match="'b.raw' already names acquisition 0"):
        swapped.register([_raw("/copies/b.raw", "hash-A")])


def test_a_renamed_copy_keeps_its_name_as_an_alias(tmp_path) -> None:
    """The same run under a new file name gets the run's id, and results
    reported under the new name find it."""
    from constellation.massspec.acquisitions import (
        Acquisitions,
        load_acquisitions,
        save_acquisitions,
    )

    acq, _ = Acquisitions.empty().register([_raw("/data/a.raw", "hash-A"), _raw("b.raw", "hash-B")])
    renamed, ids = acq.register([_raw("/copies/c.raw", "hash-A")])
    assert ids == [0] and renamed.ids == [0, 1]
    assert renamed.id_for_source("c.raw") == 0
    assert renamed.id_for_source("a.raw") == 0 and renamed.id_for_source("b.raw") == 1
    assert renamed.table.column("source_aliases").to_pylist() == [["c.raw"], None]
    # Registering it again adds nothing; the alias survives a round trip.
    again, ids2 = renamed.register([_raw("/elsewhere/c.raw", "hash-A")])
    assert ids2 == [0] and again.table.column("source_aliases").to_pylist() == [["c.raw"], None]
    save_acquisitions(renamed, tmp_path)
    assert load_acquisitions(tmp_path).id_for_source("c.raw") == 0
    # An alias is a name like any other: it cannot be taken by another run.
    with pytest.raises(ValueError, match="two different files are both named 'c.raw'"):
        renamed.register([_raw("c.raw", "hash-C")])


def test_with_source_alias_binds_a_name_explicitly() -> None:
    """What a workflow does when it hands an engine a file it wrote
    itself: say outright which run that file name stands for."""
    from constellation.massspec.acquisitions import Acquisitions

    acq, _ = Acquisitions.empty().register([_raw("run_01.raw", "aa"), _raw("run_02.raw", "bb")])
    bound = acq.with_source_alias(1, "/work/units/u7/input.dia")
    assert bound.id_for_source("input.dia") == 1
    assert acq.table.column("source_aliases").to_pylist() == [None, None]  # original untouched
    assert bound.with_source_alias(1, "input.dia").table.equals(bound.table)  # idempotent
    with pytest.raises(ValueError, match="'run_01.raw' already names acquisition 0"):
        acq.with_source_alias(1, "run_01.raw")
    with pytest.raises(KeyError, match="no acquisition 9"):
        acq.with_source_alias(9, "x.dia")


def test_identity_columns_survive_reordering() -> None:
    from constellation.massspec.acquisitions import Acquisitions

    acq, _ = Acquisitions.empty().register(
        [_raw("a.raw", "aa", instrument_serial="S", acquisition_datetime="2026-10-06T22:19:53")]
    )
    ordered = acq.with_source_alias(0, "a.dia").with_acquisition_order()
    row = ordered.table.to_pylist()[0]
    assert (row["source_sha256"], row["source_aliases"], row["acquisition_order"]) == ("aa", ["a.dia"], 1)
