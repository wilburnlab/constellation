"""The kinds of directory a mass-spec workflow produces.

One layout, discoverable from the ``manifest.json`` in each directory
(:mod:`constellation.core.manifests`)::

    <experiment>/manifest.json            kind = massspec_experiment
    <experiment>/acquisitions/            the experiment's acquisitions, ids fixed once
    <experiment>/library/                 the library searched, when there is one
    <experiment>/spectra/<run>/           the engine's input for an acquisition, written once
    <experiment>/units/<key>/             kind = massspec_search: one engine run
        manifest.json  _SUCCESS  logs/
        jar/                              the run's own directory: a link to its input,
                                          and what the engine wrote beside it
    <experiment>/tables/<table>/unit=<key>/acquisition_id=<id>/part-0.parquet
    <experiment>/report/                  kind = massspec_quant_report

A standalone search is the same thing with one acquisition and one unit,
so a tool that reads an experiment reads a single search too.

**A unit is a search, not an acquisition.** Its key names what was
searched and carries a digest of how (inputs, library, FASTA, settings,
engine version): ``<run>-<8 hex digits>``. The same acquisition
searched with other settings is another unit beside the first, never a
replacement for it, and its rows sit in their own ``unit=`` partition of
every table. So anything that reads "the result for acquisition N" has
to say which search when there is more than one
(:func:`open_table`, or a unit directory).

These classes fix the field names. What fills them is the search and
orchestration code: the experiment and search manifests are written by
:mod:`constellation.massspec.workflows.encyclopedia`; the quant report
has no writer yet. Paths inside a manifest are relative to the
directory the manifest is in, so an experiment can be moved or copied
whole.

The manifests the current ``massspec search`` / ``predict-library`` /
``library-export`` commands write predate this and declare no kind;
:func:`constellation.core.manifests.manifest_kind` returns ``None`` for
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from constellation.core.manifests import Manifest

if TYPE_CHECKING:
    import pyarrow.dataset as ds

#: The partition levels of an experiment's tables, outermost first.
TABLE_PARTITIONS = ("unit", "acquisition_id")


def table_partition_dir(
    experiment_dir: Path | str, table: str, unit: str, acquisition_id: int | None = None
) -> Path:
    """Where one unit's rows of ``table`` go (for one acquisition, or the
    unit's whole partition when ``acquisition_id`` is left out).

    The unit key is percent-encoded in the directory name, as hive
    partition values are, so a run name with a space or an ``=`` in it
    reads back as itself.
    """
    directory = Path(experiment_dir) / "tables" / table / f"unit={quote(unit, safe='')}"
    return directory if acquisition_id is None else directory / f"acquisition_id={acquisition_id}"


def open_table(experiment_dir: Path | str, table: str) -> ds.Dataset:
    """One table of an experiment as a dataset over all its searches and
    acquisitions, with ``unit`` and ``acquisition_id`` columns from the
    partition directories (filter on them to read one search).

    Raises ``FileNotFoundError`` when the experiment has no such table.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    directory = Path(experiment_dir) / "tables" / table
    if not directory.is_dir():
        tables = Path(experiment_dir) / "tables"
        known = sorted(p.name for p in tables.glob("*") if p.is_dir())
        raise FileNotFoundError(
            f"{experiment_dir} has no table {table!r} (it has: {', '.join(known) or 'none'})"
        )
    partitioning = ds.partitioning(
        pa.schema([("unit", pa.string()), ("acquisition_id", pa.int64())]), flavor="hive"
    )
    return ds.dataset(directory, partitioning=partitioning)


@dataclass(frozen=True)
class ExperimentManifest(Manifest):
    """The root of an experiment: which acquisitions, which library,
    which engine runs, which tables."""

    KIND = "massspec_experiment"
    SCHEMA_VERSION = 1

    name: str = ""
    #: Directory of the persisted ``Acquisitions`` (``acquisitions``).
    acquisitions: str = "acquisitions"
    #: Directory of the library searched, or ``None`` for a search that
    #: used none.
    library: str | None = None
    #: Unit key -> its directory (``units/<key>``). One per search.
    units: dict[str, str] = field(default_factory=dict)
    #: Table name -> its dataset directory (``tables/<table>``).
    tables: dict[str, str] = field(default_factory=dict)
    #: How entity ids were made (``massspec.library.ids.ID_SCHEME``), or
    #: ``None`` for sequential ids that mean nothing outside this experiment.
    id_scheme: str | None = None
    #: Directory of the quantitative report, once one has been made.
    report: str | None = None


@dataclass(frozen=True)
class SearchManifest(Manifest):
    """One run of a search engine over one or more acquisitions.

    Written twice: when the search starts (``status="running"``, with
    what it was asked to do) and when it ends (``"finished"``,
    ``"failed"`` or ``"cancelled"``, with what it did). The unit's
    ``_SUCCESS`` marker, not ``status``, is what says the tables can be
    used; a manifest still saying ``"running"`` with no live process
    belongs to a search that was stopped without a chance to say so.
    """

    KIND = "massspec_search"
    SCHEMA_VERSION = 1

    #: ``"running"``, ``"finished"``, ``"failed"`` or ``"cancelled"``.
    status: str = ""
    #: What went wrong, when ``status`` is ``"failed"`` or ``"cancelled"``.
    error: str = ""

    #: ``"encyclopedia"``.
    engine: str = ""
    engine_version: str = ""
    #: The engine's program, as named in its captured command line
    #: (``"encyclopedia"``, ``"scribetwo"``).
    program: str = ""
    #: ``"peptide-centric"`` or ``"spectrum-centric"``.
    search_mode: str = ""
    #: The acquisitions searched (ids of the experiment's ``Acquisitions``).
    acquisition_ids: list[int] = field(default_factory=list)
    #: Run name as the engine wrote it into its result -> acquisition id.
    #: The binding of the result's rows to acquisitions, recorded where
    #: it was established instead of re-derived from names later.
    sources: dict[str, int] = field(default_factory=dict)
    #: Digest of what makes this search this search: the spectra
    #: searched, library, FASTA, settings, engine and its version. Its
    #: first eight digits end the unit's key.
    search_key: str = ""
    #: :func:`constellation.core.stages.stage_key` of the search and of
    #: how its tables were written; also in the unit's ``_SUCCESS``.
    stage_key: str = ""
    #: Name -> ``{"path", "sha256", "size_bytes"}`` of each input file.
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The search settings in Constellation's spelling (tolerances with
    #: units, enzyme, ...), as the user gave or defaulted them.
    parameters: dict[str, Any] = field(default_factory=dict)
    #: The exact command line the engine was started with (its own
    #: arguments only while the search is running).
    argv: list[str] = field(default_factory=list)
    #: The engine's flags as this search passed them, by the engine's own
    #: names, values as the engine reads them. Settings left to the
    #: engine's defaults are not here.
    engine_flags: dict[str, str] = field(default_factory=dict)
    #: Every setting the engine recorded in its result, defaults included:
    #: what the search actually ran with. Empty until it has finished.
    engine_settings: dict[str, str] = field(default_factory=dict)
    #: Name -> path of each output.
    outputs: dict[str, str] = field(default_factory=dict)
    #: Elapsed time, exit status, Java, rows written; ``finished_at``
    #: (``created_at`` is when the search started).
    runtime: dict[str, Any] = field(default_factory=dict)
    #: Sentences saying where the parsed tables are not the whole of
    #: what the engine wrote. Empty when they are.
    caveats: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QuantReportManifest(Manifest):
    """A quantitative report across the experiment's acquisitions."""

    KIND = "massspec_quant_report"
    SCHEMA_VERSION = 1

    engine: str = ""
    engine_version: str = ""
    acquisition_ids: list[int] = field(default_factory=list)
    #: The search units the report was built from.
    units: list[str] = field(default_factory=list)
    stage_key: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    argv: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)


__all__ = [
    "TABLE_PARTITIONS",
    "ExperimentManifest",
    "QuantReportManifest",
    "SearchManifest",
    "open_table",
    "table_partition_dir",
]
