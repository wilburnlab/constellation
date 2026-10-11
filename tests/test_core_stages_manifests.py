"""``core.stages`` (is this stage done, with these inputs?) and
``core.manifests`` (what kind of directory is this?)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from constellation.core import manifests, stages
from constellation.core.manifests import (
    Manifest,
    ManifestError,
    manifest_kind,
    read_manifest,
    write_manifest,
)

# ── stages ─────────────────────────────────────────────────────────────


def test_stage_key_depends_on_values_not_on_order() -> None:
    a = stages.stage_key(threads=4, inputs=["x", "y"], upstream="abc")
    assert a == stages.stage_key(upstream="abc", inputs=["x", "y"], threads=4)
    assert a != stages.stage_key(threads=8, inputs=["x", "y"], upstream="abc")
    assert a != stages.stage_key(threads=4, inputs=["y", "x"], upstream="abc")
    # Chaining: a different upstream key is a different key.
    assert a != stages.stage_key(threads=4, inputs=["x", "y"], upstream="abd")
    assert len(a) == 32 and stages.stage_key(path=Path("/a/b")) == stages.stage_key(path="/a/b")


def test_input_identity_notices_a_changed_or_missing_file(tmp_path: Path) -> None:
    import os

    one, two = tmp_path / "b.raw", tmp_path / "a.raw"
    one.write_text("first")
    two.write_text("second")
    before = stages.input_identity([one, two])
    assert [e.split(":")[0] for e in before] == [str(two), str(one)]  # sorted
    assert stages.input_identity([two, one]) == before

    one.write_text("first, but longer")
    assert stages.input_identity([one, two]) != before
    one.write_text("first")
    os.utime(one, ns=(1, 1))  # same size, different time
    assert stages.input_identity([one, two]) != before
    one.unlink()
    assert stages.input_identity([one])[0] == f"{one}:missing"


def test_a_stage_is_done_only_with_the_key_it_recorded(tmp_path: Path) -> None:
    stage = tmp_path / "units" / "u1"
    assert not stages.is_done(stage) and stages.recorded_key(stage) is None

    key = stages.stage_key(inputs=["x"], tolerance=10)
    stages.mark_done(stage, key)  # creates the directory
    assert stages.is_done(stage, key) and stages.is_done(stage)
    assert stages.recorded_key(stage) == key
    # Different settings, or a marker that carries no key: not done.
    assert not stages.is_done(stage, stages.stage_key(inputs=["x"], tolerance=20))
    stages.mark_done(stage)
    assert stages.is_done(stage) and not stages.is_done(stage, key)
    assert stages.recorded_key(stage) == ""

    stages.invalidate(stage)
    assert not stages.is_done(stage)
    stages.invalidate(stage)  # idempotent


# ── manifests ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ToyManifest(Manifest):
    KIND = "test_toy"
    SCHEMA_VERSION = 2
    name: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class _OtherManifest(Manifest):
    KIND = "test_other"
    SCHEMA_VERSION = 1


def test_a_manifest_round_trips_as_its_own_class(tmp_path: Path) -> None:
    written = _ToyManifest(name="run", outputs={"elib": "jar/run.elib"}, ids=[0, 1])
    path = write_manifest(tmp_path / "out", written)
    assert path == tmp_path / "out" / "manifest.json"
    raw = json.loads(path.read_text())
    assert list(raw)[:2] == ["kind", "schema_version"]
    assert raw["kind"] == "test_toy" and raw["schema_version"] == 2
    assert raw["created_at"].endswith("Z")  # filled in

    read = read_manifest(tmp_path / "out")
    assert isinstance(read, _ToyManifest)
    assert (read.name, read.outputs, read.ids) == ("run", {"elib": "jar/run.elib"}, [0, 1])
    assert read_manifest(path, expect=_ToyManifest) == read
    assert manifest_kind(tmp_path / "out") == "test_toy"
    assert not list((tmp_path / "out").glob(".manifest*"))  # no temporary file left


def test_reading_says_what_is_wrong(tmp_path: Path) -> None:
    assert manifest_kind(tmp_path) is None
    with pytest.raises(ManifestError, match="no manifest.json"):
        read_manifest(tmp_path)

    write_manifest(tmp_path, _ToyManifest(name="run"))
    with pytest.raises(ManifestError, match="'test_other' one was expected"):
        read_manifest(tmp_path, expect=_OtherManifest)

    path = tmp_path / "manifest.json"
    raw = json.loads(path.read_text())

    path.write_text(json.dumps({**raw, "schema_version": 1}))
    with pytest.raises(ManifestError, match="schema_version 1; this Constellation reads version 2"):
        read_manifest(tmp_path)

    path.write_text(json.dumps({**raw, "kind": "from_the_future"}))
    with pytest.raises(ManifestError, match="'from_the_future', which nothing loaded here reads"):
        read_manifest(tmp_path)

    untyped = {k: v for k, v in raw.items() if k != "kind"}
    path.write_text(json.dumps(untyped))
    assert manifest_kind(tmp_path) is None
    with pytest.raises(ManifestError, match="declares no kind"):
        read_manifest(tmp_path)

    path.write_text("{not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        read_manifest(tmp_path)


def test_fields_added_later_do_not_break_older_or_newer_files(tmp_path: Path) -> None:
    write_manifest(tmp_path, _ToyManifest(name="run"))
    path = tmp_path / "manifest.json"
    raw = json.loads(path.read_text())
    # A newer writer added a field this class does not have; an older one lacked `ids`.
    raw["added_next_year"] = True
    del raw["ids"]
    path.write_text(json.dumps(raw))
    read = read_manifest(tmp_path, expect=_ToyManifest)
    assert read.name == "run" and read.ids == []


def test_a_kind_is_defined_once() -> None:
    assert {"test_toy", "test_other"} <= set(manifests.registered_kinds())
    with pytest.raises(TypeError, match="already registered"):

        @dataclass(frozen=True)
        class _Duplicate(Manifest):
            KIND = "test_toy"
            SCHEMA_VERSION = 1

    with pytest.raises(TypeError, match="SCHEMA_VERSION"):

        @dataclass(frozen=True)
        class _Unversioned(Manifest):
            KIND = "test_unversioned"

    with pytest.raises(TypeError, match="no KIND"):
        write_manifest(Path("."), Manifest())


def test_massspec_kinds_are_registered_and_round_trip(tmp_path: Path) -> None:
    from constellation.massspec.manifest import (
        ExperimentManifest,
        QuantReportManifest,
        SearchManifest,
    )

    assert {"massspec_experiment", "massspec_search", "massspec_quant_report"} <= set(
        manifests.registered_kinds()
    )
    experiment = ExperimentManifest(
        name="u2os", units={"a1": "units/a1"}, tables={"precursor_score": "tables/precursor_score"}
    )
    write_manifest(tmp_path, experiment)
    search = SearchManifest(
        engine="encyclopedia",
        engine_version="6.9.23",
        program="encyclopedia",
        search_mode="peptide-centric",
        acquisition_ids=[0],
        stage_key=stages.stage_key(inputs=["run.dia"]),
        outputs={"result": "jar/run.elib"},
        caveats=["an example"],
    )
    write_manifest(tmp_path / "units" / "a1", search)
    write_manifest(tmp_path / "report", QuantReportManifest(units=["a1"], acquisition_ids=[0]))

    # A reader walks the layout by kind, with no knowledge of who wrote it.
    root = read_manifest(tmp_path)
    assert isinstance(root, ExperimentManifest) and root.acquisitions == "acquisitions"
    unit = read_manifest(tmp_path / root.units["a1"])
    assert isinstance(unit, SearchManifest) and unit == read_manifest(
        tmp_path / "units" / "a1", expect=SearchManifest
    )
    assert unit.search_mode == "peptide-centric" and unit.caveats == ["an example"]
    assert manifest_kind(tmp_path / "report") == "massspec_quant_report"
