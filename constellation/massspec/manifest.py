"""The kinds of directory a mass-spec workflow produces.

One layout, discoverable from the ``manifest.json`` in each directory
(:mod:`constellation.core.manifests`)::

    <experiment>/manifest.json            kind = massspec_experiment
    <experiment>/acquisitions/            the experiment's acquisitions, ids fixed once
    <experiment>/library/                 the library searched, when there is one
    <experiment>/units/<key>/             kind = massspec_search: one engine run
        manifest.json  _SUCCESS  jar/     (what the engine read and wrote, its logs)
    <experiment>/tables/<table>/acquisition_id=<id>/part-0.parquet
    <experiment>/report/                  kind = massspec_quant_report

A standalone search is the same thing with one acquisition and one unit,
so a tool that reads an experiment reads a single search too.

These classes fix the field names. What fills them is the search and
orchestration code (phases 4 and 5 of the EncyclopeDIA integration);
nothing writes them yet. Paths inside a manifest are relative to the
directory the manifest is in, so an experiment can be moved or copied
whole.

The manifests the current ``massspec search`` / ``predict-library`` /
``library-export`` commands write predate this and declare no kind;
:func:`constellation.core.manifests.manifest_kind` returns ``None`` for
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from constellation.core.manifests import Manifest


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
    #: Unit key -> its directory (``units/<key>``).
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
    """One run of a search engine over one or more acquisitions."""

    KIND = "massspec_search"
    SCHEMA_VERSION = 1

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
    #: :func:`constellation.core.stages.stage_key` of inputs and
    #: parameters; also in the unit's ``_SUCCESS``.
    stage_key: str = ""
    #: Name -> ``{"path", "sha256", "size_bytes"}`` of each input file.
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The search settings in Constellation's spelling (tolerances with
    #: units, enzyme, ...), as the user gave or defaulted them.
    parameters: dict[str, Any] = field(default_factory=dict)
    #: The exact command line the engine was started with.
    argv: list[str] = field(default_factory=list)
    #: Name -> path of each output.
    outputs: dict[str, str] = field(default_factory=dict)
    #: Elapsed time, exit status, host, Java.
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


__all__ = ["ExperimentManifest", "QuantReportManifest", "SearchManifest"]
