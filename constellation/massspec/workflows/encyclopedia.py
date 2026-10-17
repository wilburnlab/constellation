"""Search one acquisition with EncyclopeDIA and write the result as an experiment.

``search_acquisition`` is the whole single-acquisition path::

    bundle | .raw | .dia
        -> a registered acquisition            <experiment>/acquisitions/
        -> an EncyclopeDIA spectra file        <experiment>/spectra/<run>/<run>.dia
        -> the search, in a directory of its own
                                               <experiment>/units/<unit>/jar/<run>.dia  (a link)
                                               <experiment>/units/<unit>/jar/<run>.dia.elib + sidecars
        -> Constellation tables, by search and acquisition
                                               <experiment>/tables/<table>/unit=<unit>/acquisition_id=<id>/
        -> manifests that say what all of it is
                                               <experiment>/manifest.json, units/<unit>/manifest.json

Constellation does every conversion itself: a ``.raw`` goes through the
Thermo reader into a bundle, and the bundle through ``write_dia`` into
the spectra file the jar reads. The jar is only asked to search. (It can
read a ``.raw`` on its own; that path is not used.) mzML and Bruker
``.d`` inputs are refused until Constellation has readers for them.

The layout is the one in :mod:`constellation.massspec.manifest`. A
standalone search is an experiment with one acquisition and one unit, so
whatever reads an experiment reads this.

Four things this module is careful about:

**Every search has a directory of its own.** ``<unit>`` is
``<run>-<8 hex digits>``, the digits being a digest of what makes the
search this search: the spectra, the library, the FASTA, the settings,
the jar's version. Search an acquisition again with anything changed
and the result goes into another directory and another partition of the
tables, beside the first; nothing already there is touched. Only the
very same search maps to the same directory, which is what lets
``resume`` find it.

**The jar sees a new search each time.** It writes its outputs beside
its input, named for it, and takes a result it finds there for a
finished search: it logs "Previously found N peptides" and returns that
result whatever the settings now say. That is its design (the result is
its ``_SUCCESS`` flag, bound to a location), and it is the half of its
behaviour not copied here. A unit's ``jar/`` holds a link to the
spectra file and, before the run, nothing else; the jar is given the
link and no ``-o``, so everything it writes lands there under its own
names (:mod:`constellation.massspec.search.encyclopedia.outputs` has
the measurements). The spectra file itself is written once per
acquisition and shared by every search of it, so another search costs
a search, not a conversion. As a second check, the settings a result
records about itself are compared with the ones asked for, and a
difference is reported in the unit's caveats.

**Binding.** A result names its run by file name. The unit knows which
acquisition it searched, so it checks that the result names that run and
no other, and records the pairing (``SearchManifest.sources``). Nothing
downstream has to guess from a name.

**Ids.** The tables use content-derived ids
(:mod:`constellation.massspec.library.ids`), so the same peptide has the
same id under every search and acquisition of the experiment, and of
any other.

``resume=True`` returns a unit that already finished as it is. Without
it the same search is run again, in its own directory emptied first;
no other search's directory is involved either way.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import signal
import sqlite3
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from constellation import __version__ as CONSTELLATION_VERSION
from constellation.core import stages
from constellation.core.locks import LockHeld, file_lock
from constellation.core.manifests import (
    MANIFEST_FILENAME,
    manifest_kind,
    read_manifest,
    write_manifest,
)
from constellation.massspec.acquisition_structure import infer_acquisition_structure
from constellation.massspec.acquisitions import (
    MERGED_KIND,
    Acquisitions,
    acquisition_record_from_bundle,
    load_acquisitions,
    save_acquisitions,
)
from constellation.massspec.io.encyclopedia import DiaFile, read_encyclopedia, write_dia
from constellation.massspec.io.encyclopedia._mirror import open_library, read_metadata
from constellation.massspec.io.encyclopedia._project import RUN_AGNOSTIC_SOURCE
from constellation.massspec.io.encyclopedia._read import EncyclopediaReadResult
from constellation.massspec.library.ids import ID_SCHEME, with_content_ids
from constellation.massspec.manifest import (
    ALL_ACQUISITIONS,
    ExperimentManifest,
    QuantReportManifest,
    SearchManifest,
    open_table,
    table_partition_dir,
)
from constellation.massspec.search.encyclopedia._common import sha256_file
from constellation.massspec.search.encyclopedia.outputs import MissingOutputError, locate_outputs
from constellation.massspec.search.encyclopedia.params import (
    EXECUTION_ONLY_FLAGS,
    PEPTIDE_CENTRIC,
    SPECTRUM_CENTRIC,
    ResolvedSearch,
    SearchParameters,
    resolve_search_mode,
)
from constellation.massspec.search.encyclopedia.percolator import percolator_blocker
from constellation.massspec.search.encyclopedia.runner import (
    FlagValue,
    compose_argv,
    format_value,
    resolve_jar,
    run_program,
)
from constellation.thirdparty.jvm import (
    JvmRunError,
    cancellation_requested,
    cancellation_scope,
    request_stop,
)

ENGINE = "encyclopedia"

#: Bumped when what a unit writes changes (tables, their columns, how
#: ids are made), so that a unit written by older code is not taken for
#: finished by newer code.
UNIT_LAYOUT = 1

#: Bumped when the spectra file written for a bundle changes, so that
#: one written by older code is written again (and searched again).
SPECTRA_LAYOUT = 1

#: Where a search's collision-filtered result library goes, inside its
#: directory (:func:`collision_filter`).
COLLISION_DIR = "collision"

_REFUSED_INPUTS = {
    ".mzml": "mzML",
    ".mzxml": "mzXML",
    ".d": "Bruker .d",
    ".wiff": "Sciex .wiff",
}


class WorkflowError(ValueError):
    """The search cannot be run as asked. The message says why."""


@dataclass(frozen=True)
class SearchResult:
    """What :func:`search_acquisition` did."""

    experiment_dir: Path
    unit_dir: Path
    #: The search's key, and the name of its directory under ``units/``.
    unit: str
    acquisition_id: int
    manifest: SearchManifest
    #: Rows written per table (empty when the unit was reused).
    table_rows: dict[str, int]
    #: The unit was already finished with these inputs and settings.
    reused: bool = False
    #: Settings the result records differently from the ones asked for
    #: (also in the manifest's caveats). Empty is what one expects.
    disagreements: tuple[str, ...] = ()
    #: The acquisition's run name (the unit key without its digest).
    run: str = ""


# ──────────────────────────────────────────────────────────────────────
# the input
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Source:
    kind: str  # "bundle" | "raw" | "spectra_file"
    path: Path
    run: str  # the name outputs are given
    record: dict[str, object]  # for Acquisitions.register
    acquisition_type: str | None
    bundle: Path | None = None


def _bundle_source(bundle: Path) -> _Source:
    record = acquisition_record_from_bundle(bundle)
    scans = pq.read_table(
        bundle / "scan_metadata.parquet", columns=["level", "isolation_lower", "isolation_upper"]
    )
    structure = infer_acquisition_structure(scans)
    run = Path(_file_name(str(record["source_file"]))).stem
    return _Source("bundle", bundle, run, record, structure.acquisition_type, bundle=bundle)


def _file_name(source: str) -> str:
    return source.replace("\\", "/").rsplit("/", 1)[-1]


def _classify(source: Path, experiment_dir: Path, convert_raw: bool) -> _Source:
    source = Path(source)
    suffix = source.suffix.lower()
    if suffix in _REFUSED_INPUTS:
        raise WorkflowError(
            f"{source.name}: {_REFUSED_INPUTS[suffix]} input is not supported yet. "
            f"Constellation searches Thermo .raw files (through its own reader), converted "
            f"bundles, and EncyclopeDIA spectra files (.dia)."
        )
    if source.is_dir():
        if not (source / "manifest.json").is_file():
            raise WorkflowError(
                f"{source} is a directory but not a converted acquisition bundle "
                f"(`constellation massspec convert` writes one)"
            )
        return _bundle_source(source)
    if not source.is_file():
        raise WorkflowError(f"no such input: {source}")
    if suffix == ".raw":
        bundle = experiment_dir / "bundles" / source.stem
        # The converter writes the manifest last, so a bundle without
        # one is a conversion that did not finish: do it again.
        if not (bundle / "manifest.json").is_file():
            if not convert_raw:
                raise WorkflowError(f"{source.name} has not been converted (dry run)")
            from constellation.massspec.readers.thermo import convert

            try:
                convert(source, bundle, force=True)
            except ImportError as exc:
                raise WorkflowError(f"cannot read {source.name}: {exc}") from exc
        return _bundle_source(bundle)
    if suffix == ".dia":
        with DiaFile(source) as dia:
            metadata = dia.metadata
            fractions = dia.fraction_ids()
        if len(fractions) > 1:
            raise WorkflowError(
                f"{source.name} is a merged spectra file ({len(fractions)} fractions). "
                f"Searching several runs as one is not part of the single-acquisition path."
            )
        named = str(metadata.get("sourcename") or source.name)
        kind = metadata.get("rawFileStructure.dataAcquisitionType")
        record = {
            "source_file": named,
            "source_kind": "encyclopedia_spectra_file",
            "source_sha256": None,
            "source_aliases": [source.name] if source.name != _file_name(named) else None,
        }
        return _Source(
            "spectra_file", source, source.name[: -len(".dia")], record,
            str(kind).upper() if kind else None,
        )
    raise WorkflowError(
        f"{source.name}: not a Thermo .raw, a converted bundle directory or a .dia spectra file"
    )


# ──────────────────────────────────────────────────────────────────────
# tables
# ──────────────────────────────────────────────────────────────────────


def _remap(column: pa.ChunkedArray, new_ids: pa.ChunkedArray) -> pa.ChunkedArray:
    """Sequential ids (0..n-1, the row they index) to the ids at those rows."""
    return pc.take(new_ids, column)


def content_keyed_tables(read: EncyclopediaReadResult) -> dict[str, pa.Table]:
    """The reader's tables under content-derived ids. Those that are per
    acquisition keep their ``acquisition_id`` column, which
    :func:`_write_tables` turns into partition directories."""
    keyed = with_content_ids(read.library)
    peptide_ids = keyed.peptides.column("peptide_id")
    precursor_ids = keyed.precursors.column("precursor_id")
    protein_ids = keyed.proteins.column("protein_id")
    tables: dict[str, pa.Table] = {
        "protein": keyed.proteins,
        "peptide": keyed.peptides,
        "precursor": keyed.precursors,
        "fragment": keyed.fragments,
        "protein_peptide": keyed.protein_peptide,
    }

    def rekey(table: pa.Table, column: str, new_ids: pa.ChunkedArray) -> pa.Table:
        index = table.schema.get_field_index(column)
        return table.set_column(index, table.schema.field(column), _remap(table.column(column), new_ids))

    if read.quant is not None:
        tables["precursor_quant"] = rekey(read.quant.precursor_quant, "precursor_id", precursor_ids)
        tables["peptide_quant"] = rekey(read.quant.peptide_quant, "peptide_id", peptide_ids)
    if read.search is not None:
        tables["precursor_score"] = rekey(read.search.precursor_scores, "precursor_id", precursor_ids)
        tables["peptide_score"] = rekey(read.search.peptide_scores, "peptide_id", peptide_ids)
        tables["protein_score"] = rekey(read.search.protein_scores, "protein_id", protein_ids)
    return tables


def _write_tables(
    experiment_dir: Path, unit: str, tables: dict[str, pa.Table], *, acquisition_id: int
) -> dict[str, int]:
    """Write a unit's tables, one partition per acquisition.

    A table with an ``acquisition_id`` column is split on it (the column
    becomes the partition directory). One without (the library-level
    tables: protein, peptide, precursor, fragment, edges), and rows whose
    ``acquisition_id`` is null (a score over every run, such as a
    report's protein scores), go under ``acquisition_id``: the unit's own
    acquisition, or :data:`~constellation.massspec.manifest.ALL_ACQUISITIONS`
    for a unit whose result spans several.
    """
    rows: dict[str, int] = {}

    def write(name: str, table: pa.Table, partition: int) -> None:
        directory = table_partition_dir(experiment_dir, name, unit, partition)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / ".part-0.parquet.tmp"
        pq.write_table(table, tmp)
        tmp.replace(directory / "part-0.parquet")

    for name, table in tables.items():
        rows[name] = table.num_rows
        if "acquisition_id" not in table.column_names:
            write(name, table, acquisition_id)
            continue
        column = table.column("acquisition_id")
        unbound = table.filter(pc.is_null(column))
        ids = [i for i in pc.unique(column).to_pylist() if i is not None]
        if unbound.num_rows or not ids:
            write(name, unbound.drop_columns(["acquisition_id"]), acquisition_id)
        for value in sorted(ids):
            part = table.filter(pc.equal(column, value))
            write(name, part.drop_columns(["acquisition_id"]), int(value))
    return rows


def read_experiment_table(
    experiment_dir: Path | str,
    table: str,
    *,
    unit: str | None = None,
    acquisition_id: int | None = None,
) -> pa.Table:
    """One table of an experiment: all of it, or one search's rows
    (``unit``), or one acquisition's (``acquisition_id``), or both.

    The ``unit`` and ``acquisition_id`` columns come back from the
    partition directories. An acquisition searched more than once has
    rows under each of its units, so filter on ``unit`` before treating
    a table as one result.
    """
    expression = None
    for column, value in (("unit", unit), ("acquisition_id", acquisition_id)):
        if value is not None:
            term = ds.field(column) == value
            expression = term if expression is None else expression & term
    return open_table(experiment_dir, table).to_table(filter=expression)


# ──────────────────────────────────────────────────────────────────────
# the experiment's shared records
# ──────────────────────────────────────────────────────────────────────

_LOCK_NAME = ".lock"


def _experiment_lock(experiment_dir: Path) -> contextlib.AbstractContextManager[None]:
    """Held around every read-modify-write of the experiment's manifest
    and acquisitions table, so that searches running at once (threads of
    one command, or separate commands) cannot overwrite each other's
    entries."""
    return file_lock(experiment_dir / _LOCK_NAME)


@contextlib.contextmanager
def _sole_run(directory: Path, what: str) -> Iterator[None]:
    """Held while a search or report runs in ``directory``, so a second
    command running the same one (a detached run still going, then the
    same command again with ``--resume``) is refused at once instead of
    clearing the directory under the first one's JVM."""
    stack = contextlib.ExitStack()
    try:
        stack.enter_context(file_lock(directory.parent / f".{directory.name}.lock", wait=False))
    except LockHeld:
        raise WorkflowError(
            f"{what} is running in another command now; wait for it to end (or stop it) "
            f"before running it again"
        ) from None
    with stack:
        yield


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


#: What may sit in a directory that is not an experiment yet without it
#: being refused: converted bundles, the lock, detached-run records.
_ALLOWED_IN_A_NEW_EXPERIMENT = frozenset({"bundles", _LOCK_NAME, "runs"})


def _load_experiment(experiment_dir: Path) -> tuple[ExperimentManifest, Acquisitions]:
    kind = manifest_kind(experiment_dir)
    if kind is None:
        if experiment_dir.is_dir() and any(
            p.name not in _ALLOWED_IN_A_NEW_EXPERIMENT for p in experiment_dir.iterdir()
        ) and not (experiment_dir / "acquisitions").is_dir():
            raise WorkflowError(
                f"{experiment_dir} is not empty and is not an experiment directory; "
                f"give a new directory, or one a previous search made"
            )
        manifest = ExperimentManifest(name=experiment_dir.name, id_scheme=ID_SCHEME)
    else:
        manifest = read_manifest(experiment_dir, expect=ExperimentManifest)
    try:
        acquisitions = load_acquisitions(experiment_dir / manifest.acquisitions)
    except FileNotFoundError:
        acquisitions = Acquisitions.empty()
    return manifest, acquisitions


def _update_experiment(experiment_dir: Path, **changes: dict[str, str]) -> None:
    """Add entries to the experiment manifest's ``units`` / ``tables`` /
    ``reports``, under the lock and against the manifest as it is now
    (another search may have added its own since this one started)."""
    with _experiment_lock(experiment_dir):
        manifest = read_manifest(experiment_dir, expect=ExperimentManifest)
        merged = {name: {**getattr(manifest, name), **entries} for name, entries in changes.items()}
        write_manifest(
            experiment_dir,
            replace(manifest, constellation_version=CONSTELLATION_VERSION, **merged),
        )


# ──────────────────────────────────────────────────────────────────────
# planning: every search settled before any runs
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SearchPlan:
    """One search, settled before it runs: what it searches, with what,
    how, and so where its directory is. Made by :func:`plan_searches`,
    run by :func:`run_search`."""

    #: The acquisition searched, or the fractions searched as one.
    sources: tuple[_Source, ...]
    #: The spectra file's name without ``.dia`` (and the result's run name).
    run: str
    #: The acquisition the result is bound to: the run itself, or the
    #: merged file's own acquisition.
    acquisition_id: int
    #: For a merged search, the fractions' acquisition ids, in order.
    merged_from: tuple[int, ...]
    resolved: ResolvedSearch
    parameters: SearchParameters
    #: The jar's flags for settings, by its own names (no input files).
    settings: dict[str, FlagValue]
    fasta: Path
    library: Path | None
    spectra_key: str | None
    search_key: str
    stage_key: str
    #: ``<run>-<first 8 digits of search_key>``: the directory under units/.
    unit: str
    engine_version: str
    allow_undocumented: tuple[str, ...]
    annotate_fragments: bool

    @property
    def merged(self) -> bool:
        return bool(self.merged_from)


def _merged_name(runs: Sequence[str]) -> str:
    """``<common prefix>merged<common suffix>``, each cut back to a
    separator: ``gpf-400-500_100min`` and ``gpf-500-600_100min`` give
    ``gpf-merged_100min``."""
    prefix = os.path.commonprefix(list(runs))
    rests = [r[len(prefix) :] for r in runs]
    suffix = os.path.commonprefix([r[::-1] for r in rests])[::-1]
    separators = "_-."
    while prefix and prefix[-1] not in separators:
        prefix = prefix[:-1]
    while suffix and suffix[0] not in separators:
        suffix = suffix[1:]
    return f"{prefix}merged{suffix}"


def plan_searches(
    sources: Sequence[Path | str],
    *,
    experiment_dir: Path | str,
    fasta: Path | str,
    library: Path | str | None = None,
    parameters: SearchParameters | None = None,
    search_mode: str | None = None,
    search_type: str | None = None,
    allow_undocumented: tuple[str, ...] = (),
    annotate_fragments: bool = True,
    merge: bool = False,
    merged_name: str | None = None,
) -> list[SearchPlan]:
    """Settle the searches of ``sources`` and register their acquisitions.

    One plan per source, or with ``merge`` one plan for all of them
    searched as a single spectra file (gas-phase fractions, typically),
    which is registered as an acquisition of its own that names the
    fractions. Acquisitions are registered in the order given, so their
    ids do not depend on which search happens to finish first.

    Everything that can be known before the JVM starts is checked here
    and raises :class:`WorkflowError` (or a more specific error): an
    unsupported input, a mode that does not suit the data, a flag the
    jar does not document, a Percolator that cannot run on this machine,
    a missing library or FASTA. Converting a Thermo ``.raw`` happens
    here too, since the acquisition type comes from its scans.
    """
    experiment_dir = Path(experiment_dir).resolve()
    fasta = Path(fasta).resolve()
    library = Path(library).resolve() if library is not None else None
    parameters = parameters or SearchParameters()
    for label, path in (("--fasta", fasta), ("--library", library)):
        if path is not None and not path.is_file():
            raise WorkflowError(f"{label} not found: {path}")
    if not sources:
        raise WorkflowError("nothing to search: give at least one acquisition")
    if merge and len(sources) < 2:
        raise WorkflowError("a merged search needs at least two acquisitions")
    if merged_name is not None and not merge:
        raise WorkflowError("a merged name only applies to a merged search")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    with _experiment_lock(experiment_dir):
        _load_experiment(experiment_dir)  # refuse a foreign directory before converting anything
    srcs = [_classify(Path(s).resolve(), experiment_dir, convert_raw=True) for s in sources]
    seen: dict[str, Path] = {}
    for src in srcs:
        if src.run in seen:
            raise WorkflowError(f"{src.run} is given twice ({seen[src.run]} and {src.path})")
        seen[src.run] = src.path

    if merge:
        given = [s.path.name for s in srcs if s.bundle is None]
        if given:
            raise WorkflowError(
                f"a merged search is built from converted acquisitions (bundles or .raw "
                f"files); these are spectra files: {', '.join(given)}"
            )
        kinds = sorted({str(s.acquisition_type) for s in srcs})
        if len(kinds) != 1:
            raise WorkflowError(f"the acquisitions to merge are of different types ({kinds})")
        groups: list[tuple[_Source, ...]] = [tuple(srcs)]
    else:
        groups = [(s,) for s in srcs]

    # ── which engine, and can it run ────────────────────────────────
    resolved = [
        resolve_search_mode(search_mode, acquisition_type=g[0].acquisition_type, search_type=search_type)
        for g in groups
    ]
    for r in resolved:
        if r.mode == SPECTRUM_CENTRIC:
            raise WorkflowError(
                "spectrum-centric search (ScribeTwo) is not available through this command yet"
            )
        if r.library_required and library is None:
            raise WorkflowError(f"a {r.mode} search needs --library")
    # Only the peptide-centric program runs the Percolator executable
    # (ScribeTwo scores with its own), so this waits for the mode, even
    # though for an unconverted .raw that means after the conversion.
    extra_argv = [str(x) for pair in parameters.extra.items() for x in pair]
    blocker = percolator_blocker(parameters.percolator_version, extra_argv)
    if blocker is not None:
        raise WorkflowError(blocker)
    files: dict[str, FlagValue] = {"-f": fasta}
    if library is not None:
        files["-l"] = library
    for r, g in zip(resolved, groups, strict=True):
        settings = {**r.flags, **parameters.flags()}
        if "-o" in settings:
            raise WorkflowError(
                "-o is not passed through: the jar's outputs belong beside its input, in the "
                "search's own directory, where the rest of the workflow looks for them"
            )
        # A flag the jar does not document, or a jar older than the pin.
        compose_argv(
            r.program, {"-i": g[0].path, **files, **settings}, allow_undocumented=allow_undocumented
        )
    handle = resolve_jar()

    # ── the acquisitions, registered in the order given ─────────────
    with _experiment_lock(experiment_dir):
        manifest, acquisitions = _load_experiment(experiment_dir)
        acquisitions, ids = acquisitions.register([s.record for s in srcs])
        for src, acquisition_id in zip(srcs, ids, strict=True):
            if src.kind != "spectra_file":
                acquisitions = acquisitions.with_source_alias(acquisition_id, f"{src.run}.dia")
        if merge:
            run = merged_name or _merged_name([s.run for s in srcs])
            acquisitions, (merged_id,) = acquisitions.register(
                [{"source_file": f"{run}.dia", "source_kind": MERGED_KIND, "merged_from": ids}]
            )
            bound = [(run, merged_id, tuple(ids))]
        else:
            bound = [(s.run, i, ()) for s, i in zip(srcs, ids, strict=True)]
        save_acquisitions(acquisitions, experiment_dir / manifest.acquisitions)
        # The directory is an experiment from here on, so it can be read
        # (``massspec summarize``) whatever becomes of these searches. A
        # search is added to ``units`` once it has finished.
        write_manifest(
            experiment_dir,
            replace(manifest, constellation_version=CONSTELLATION_VERSION, id_scheme=ID_SCHEME),
        )

    plans = []
    for group, r, (run, acquisition_id, members) in zip(groups, resolved, bound, strict=True):
        settings = {**r.flags, **parameters.flags()}
        spectra_key = _spectra_key(experiment_dir, group, run)
        search_key = stages.stage_key(
            engine=ENGINE,
            engine_version=handle.version,
            program=r.program,
            settings={k: str(v) for k, v in settings.items() if k not in EXECUTION_ONLY_FLAGS},
            spectra=spectra_key if spectra_key is not None else stages.input_identity([group[0].path]),
            library=stages.input_identity([library]) if library else None,
            fasta=stages.input_identity([fasta]),
        )
        plans.append(
            SearchPlan(
                sources=group,
                run=run,
                acquisition_id=acquisition_id,
                merged_from=members,
                resolved=r,
                parameters=parameters,
                settings=settings,
                fasta=fasta,
                library=library,
                spectra_key=spectra_key,
                search_key=search_key,
                stage_key=stages.stage_key(
                    search=search_key,
                    layout=UNIT_LAYOUT,
                    fragments=annotate_fragments,
                    id_scheme=ID_SCHEME,
                ),
                unit=f"{run}-{search_key[:8]}",
                engine_version=str(handle.version),
                allow_undocumented=tuple(allow_undocumented),
                annotate_fragments=annotate_fragments,
            )
        )
    return plans


# ──────────────────────────────────────────────────────────────────────
# running one search
# ──────────────────────────────────────────────────────────────────────


def _source_record(src: _Source, acquisition_id: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(src.path),
        "kind": src.kind,
        "sha256": src.record.get("source_sha256"),
        "acquisition_type": src.acquisition_type,
    }
    if acquisition_id is not None:
        out["acquisition_id"] = acquisition_id
    return out


def run_search(
    plan: SearchPlan,
    *,
    experiment_dir: Path | str,
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    resume: bool = False,
    stream_to_stderr: bool = True,
) -> SearchResult:
    """Run one planned search and write its result into the experiment.

    One command at a time per search: if another is running this same
    search, this raises :class:`WorkflowError` without touching it.
    """
    experiment_dir = Path(experiment_dir).resolve()
    unit_dir = experiment_dir / "units" / plan.unit
    if resume and stages.is_done(unit_dir, plan.stage_key):
        return _reused_search(experiment_dir, unit_dir, plan)
    with _sole_run(unit_dir, f"the search {plan.unit}"):
        return _run_search(
            plan,
            experiment_dir=experiment_dir,
            jvm_heap_max=jvm_heap_max,
            jvm_heap_min=jvm_heap_min,
            jvm_tmpdir=jvm_tmpdir,
            resume=resume,
            stream_to_stderr=stream_to_stderr,
        )


def _reused_search(experiment_dir: Path, unit_dir: Path, plan: SearchPlan) -> SearchResult:
    done = read_manifest(unit_dir, expect=SearchManifest)
    return SearchResult(
        experiment_dir, unit_dir, plan.unit, plan.acquisition_id, done, {},
        reused=True, run=plan.run,
    )  # fmt: skip


def _run_search(
    plan: SearchPlan,
    *,
    experiment_dir: Path,
    jvm_heap_max: str,
    jvm_heap_min: str | None,
    jvm_tmpdir: Path | None,
    resume: bool,
    stream_to_stderr: bool,
) -> SearchResult:
    unit_dir = experiment_dir / "units" / plan.unit
    jar_dir = unit_dir / "jar"
    spectra_name = f"{plan.run}.dia"
    # What the jar is given: a name for the spectra file inside the
    # search's own directory. Its outputs go beside it, named for it.
    jar_input = jar_dir / spectra_name
    files: dict[str, FlagValue] = {"-f": plan.fasta}
    if plan.library is not None:
        files["-l"] = plan.library
    flags: dict[str, FlagValue] = {"-i": jar_input, **files, **plan.settings}
    program = plan.resolved.program

    if resume and stages.is_done(unit_dir, plan.stage_key):
        # Finished by another command between the check and the lock.
        return _reused_search(experiment_dir, unit_dir, plan)

    # ── the spectra file, and the search's directory made new ───────
    if plan.spectra_key is not None and all(s.bundle is not None for s in plan.sources):
        spectra_file, wrote_spectra_s = _spectra_file(experiment_dir, plan)
    else:
        spectra_file, wrote_spectra_s = plan.sources[0].path, 0.0
    _clear_unit(experiment_dir, unit_dir, plan.unit)
    linked_as = _link(spectra_file, jar_input, within=experiment_dir)

    # ── the record, written before anything runs ────────────────────
    # So that a search that fails or is stopped still says what it was
    # (``massspec summarize``). Rewritten when the search ends.
    inputs: dict[str, Any] = {"fasta": _file_record(plan.fasta)}
    if plan.library is not None:
        inputs["library"] = _file_record(plan.library)
    if plan.merged:
        inputs["sources"] = [
            _source_record(s, i) for s, i in zip(plan.sources, plan.merged_from, strict=True)
        ]
    else:
        inputs["source"] = _source_record(plan.sources[0])
    inputs["spectra_file"] = {
        "path": _relative(spectra_file, experiment_dir),
        "written_by": "constellation" if plan.spectra_key is not None else "given",
        "link": _relative(jar_input, unit_dir),
        "linked_as": linked_as,
    }
    record = SearchManifest(
        created_at=_utc_now(),
        constellation_version=CONSTELLATION_VERSION,
        engine=ENGINE,
        engine_version=plan.engine_version,
        program=program,
        search_mode=plan.resolved.mode,
        status="running",
        acquisition_ids=[plan.acquisition_id],
        search_key=plan.search_key,
        stage_key=plan.stage_key,
        inputs=inputs,
        parameters=plan.parameters.as_given(),
        argv=compose_argv(program, flags, allow_undocumented=plan.allow_undocumented),
        engine_flags={flag: format_value(value) for flag, value in flags.items()},
        runtime={"spectra_file_seconds": round(wrote_spectra_s, 3)},
    )
    write_manifest(unit_dir, record)

    try:
        # ── the search ──────────────────────────────────────────────
        run = run_program(
            program,
            flags,
            log_dir=unit_dir / "logs",
            cwd=jar_dir,
            jvm_heap_max=jvm_heap_max,
            jvm_heap_min=jvm_heap_min,
            jvm_tmpdir=jvm_tmpdir,
            allow_undocumented=plan.allow_undocumented,
            stream_to_stderr=stream_to_stderr,
        )
        outputs = locate_outputs(jar_dir, spectra_name, program)

        # ── the tables ──────────────────────────────────────────────
        with _experiment_lock(experiment_dir):
            _, acquisitions = _load_experiment(experiment_dir)
        read = read_encyclopedia(
            outputs.result_library,
            acquisitions=acquisitions,
            annotate_fragments=plan.annotate_fragments,
        )
        sources = _bound_sources(read, acquisitions, {plan.acquisition_id}, outputs.result_library)
        con = open_library(outputs.result_library)
        try:
            recorded = read_metadata(con)
        finally:
            con.close()
        disagreements = parameter_disagreements(recorded, flags)
        table_rows = _write_tables(
            experiment_dir, plan.unit, content_keyed_tables(read), acquisition_id=plan.acquisition_id
        )
    except BaseException as exc:
        _record_stop(unit_dir, record, exc)
        raise

    # ── the record, as it ended ─────────────────────────────────────
    write_manifest(
        unit_dir,
        replace(
            record,
            status="finished",
            sources=sources,
            argv=[str(a) for a in run.argv],
            engine_settings={k: v for k, v in sorted(recorded.items()) if k.startswith("-")},
            outputs={
                "result_library": _relative(outputs.result_library, unit_dir),
                "logs": "logs",
                **{name: _relative(path, unit_dir) for name, path in sorted(outputs.files.items())},
            },
            runtime={
                **record.runtime,
                "finished_at": _utc_now(),
                "elapsed_seconds": round(run.elapsed_seconds, 3),
                "returncode": run.returncode,
                "java": run.java_version,
                "jar_sha256": run.jar_sha256,
                "table_rows": table_rows,
                "missing_outputs": list(outputs.missing),
            },
            caveats=[*read.caveats, *disagreements],
        ),
    )
    _update_experiment(
        experiment_dir,
        units={plan.unit: f"units/{plan.unit}"},
        tables={name: f"tables/{name}" for name in table_rows},
    )
    stages.mark_done(unit_dir, plan.stage_key)
    return SearchResult(
        experiment_dir, unit_dir, plan.unit, plan.acquisition_id,
        read_manifest(unit_dir, expect=SearchManifest), table_rows,
        disagreements=tuple(disagreements),
        run=plan.run,
    )  # fmt: skip


def _record_stop(directory: Path, record: Any, exc: BaseException) -> None:
    """Rewrite a running record as failed or cancelled, saying why."""
    signum = cancellation_requested()
    stopped = isinstance(exc, (KeyboardInterrupt, SystemExit)) or signum is not None
    if isinstance(exc, JvmRunError) and signum is not None:
        # The jar was told to stop; what it printed on the way out is not
        # why the search ended.
        error = f"stopped by {signal.Signals(signum).name} (the jar exited {exc.returncode})"
    else:
        error = _describe_failure(exc)
    write_manifest(
        directory,
        replace(
            record,
            status="cancelled" if stopped else "failed",
            error=error,
            runtime={**record.runtime, "finished_at": _utc_now()},
        ),
    )


def search_acquisition(
    source: Path | str,
    *,
    experiment_dir: Path | str,
    fasta: Path | str,
    library: Path | str | None = None,
    parameters: SearchParameters | None = None,
    search_mode: str | None = None,
    search_type: str | None = None,
    allow_undocumented: tuple[str, ...] = (),
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    annotate_fragments: bool = True,
    resume: bool = False,
    stream_to_stderr: bool = True,
) -> SearchResult:
    """Search one acquisition and write the result into ``experiment_dir``.

    ``source`` is a converted bundle directory, a Thermo ``.raw`` (it is
    converted into ``<experiment>/bundles/``), or a single-run ``.dia``.
    ``search_mode`` defaults to what suits the data (module
    :mod:`constellation.massspec.search.encyclopedia.params`). The
    refusals before anything runs are :func:`plan_searches`'s.
    """
    (plan,) = plan_searches(
        [source],
        experiment_dir=experiment_dir,
        fasta=fasta,
        library=library,
        parameters=parameters,
        search_mode=search_mode,
        search_type=search_type,
        allow_undocumented=allow_undocumented,
        annotate_fragments=annotate_fragments,
    )
    return run_search(
        plan,
        experiment_dir=experiment_dir,
        jvm_heap_max=jvm_heap_max,
        jvm_heap_min=jvm_heap_min,
        jvm_tmpdir=jvm_tmpdir,
        resume=resume,
        stream_to_stderr=stream_to_stderr,
    )


# ──────────────────────────────────────────────────────────────────────
# several searches
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FanOutResult:
    """What :func:`search_acquisitions` did, in the order planned."""

    plans: tuple[SearchPlan, ...]
    #: Finished searches (``reused`` ones included).
    results: tuple[SearchResult, ...]
    #: Searches that failed, with what stopped them.
    failures: tuple[tuple[SearchPlan, Exception], ...]

    @property
    def ok(self) -> bool:
        return not self.failures


#: ``progress(event, plan, outcome)``: ``"start"`` before a search runs
#: (outcome ``None``), ``"done"`` with its :class:`SearchResult`,
#: ``"failed"`` with the exception.
SearchProgress = Callable[[str, SearchPlan, Any], None]


def search_acquisitions(
    sources: Sequence[Path | str],
    *,
    experiment_dir: Path | str,
    fasta: Path | str,
    library: Path | str | None = None,
    parameters: SearchParameters | None = None,
    search_mode: str | None = None,
    search_type: str | None = None,
    allow_undocumented: tuple[str, ...] = (),
    merge: bool = False,
    merged_name: str | None = None,
    jobs: int = 1,
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    annotate_fragments: bool = True,
    resume: bool = False,
    stream_to_stderr: bool = True,
    progress: SearchProgress | None = None,
) -> FanOutResult:
    """Search several acquisitions into one experiment, ``jobs`` at a time.

    Each search is planned first (:func:`plan_searches`, which also
    registers every acquisition, in the order given), then run. A search
    that fails does not stop the others; it is reported in
    ``failures``, and with ``resume`` a second call runs only what did
    not finish. With ``merge`` the acquisitions are one search, of the
    spectra file made by merging them.

    Several at once: each search is a JVM of its own, given
    ``parameters.threads`` threads and ``jvm_heap_max`` of heap, so the
    machine needs ``jobs`` times both. The jar's own output is not
    streamed when more than one runs (it would interleave); each
    search's log is in its directory. A SIGTERM or SIGHUP stops every
    running search in order and leaves them recorded as cancelled.
    """
    plans = plan_searches(
        sources,
        experiment_dir=experiment_dir,
        fasta=fasta,
        library=library,
        parameters=parameters,
        search_mode=search_mode,
        search_type=search_type,
        allow_undocumented=allow_undocumented,
        annotate_fragments=annotate_fragments,
        merge=merge,
        merged_name=merged_name,
    )
    jobs = max(1, min(int(jobs), len(plans)))
    stream = stream_to_stderr and jobs == 1

    def one(plan: SearchPlan) -> SearchResult:
        if progress is not None:
            progress("start", plan, None)
        return run_search(
            plan,
            experiment_dir=experiment_dir,
            jvm_heap_max=jvm_heap_max,
            jvm_heap_min=jvm_heap_min,
            jvm_tmpdir=jvm_tmpdir,
            resume=resume,
            stream_to_stderr=stream,
        )

    done: dict[int, SearchResult] = {}
    failures: list[tuple[int, Exception]] = []

    def settle(index: int, outcome: SearchResult | Exception) -> None:
        if isinstance(outcome, Exception):
            failures.append((index, outcome))
            if progress is not None:
                progress("failed", plans[index], outcome)
        else:
            done[index] = outcome
            if progress is not None:
                progress("done", plans[index], outcome)

    # Signal handlers can only be set in the main thread, so the scope
    # that turns SIGTERM into an orderly stop is entered here, around
    # every search, whichever thread runs it.
    with cancellation_scope():
        if jobs == 1:
            for index, plan in enumerate(plans):
                try:
                    outcome: SearchResult | Exception = one(plan)
                except Exception as exc:  # noqa: BLE001 - reported, the rest go on
                    outcome = exc
                settle(index, outcome)
        else:
            pool = ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="search")
            try:
                futures = {pool.submit(one, plan): index for index, plan in enumerate(plans)}
                for future in as_completed(futures):
                    try:
                        outcome = future.result()
                    except Exception as exc:  # noqa: BLE001
                        outcome = exc
                    settle(futures[future], outcome)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    # Ctrl-C reaches this thread only; the JVMs, each in a
                    # session of its own, have to be told.
                    request_stop(signal.SIGINT)
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            pool.shutdown(wait=True)
    return FanOutResult(
        plans=tuple(plans),
        results=tuple(done[i] for i in sorted(done)),
        failures=tuple((plans[i], exc) for i, exc in sorted(failures, key=lambda f: f[0])),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _describe_failure(exc: BaseException) -> str:
    """One paragraph for a manifest: what stopped the search."""
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    if isinstance(exc, SystemExit):
        return f"stopped by a signal (exit status {exc.code})"
    if isinstance(exc, JvmRunError):
        tail = exc.stderr_tail.strip().splitlines()[-12:]
        if not tail:
            return f"the jar exited {exc.returncode} and printed nothing to stderr"
        return f"the jar exited {exc.returncode}; the end of its stderr:\n" + "\n".join(tail)
    message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
    return f"{type(exc).__name__}: {message}"


def _spectra_dir(experiment_dir: Path, run: str) -> Path:
    return experiment_dir / "spectra" / run


def _spectra_key(experiment_dir: Path, sources: tuple[_Source, ...], run: str) -> str | None:
    """What identifies the spectra file Constellation writes for a
    bundle, or merges from several: the bundles it comes from, in order,
    and how it is written.

    Also answers for that file handed back as the input (it is found by
    where it lies, and the key by what was recorded when it was
    written), so a search is the same search whichever way its input is
    named. ``None`` for any other spectra file, which is identified by
    the file itself.
    """
    if all(s.bundle is not None for s in sources):
        return stages.stage_key(
            layout=SPECTRA_LAYOUT,
            sources=[
                {
                    "bundle": stages.input_identity([s.bundle / "manifest.json"]),  # type: ignore[operator]
                    "sha256": s.record.get("source_sha256"),
                }
                for s in sources
            ],
        )
    (src,) = sources  # a merge is always of bundles
    own = _spectra_dir(experiment_dir, run)
    if src.path.parent == own:
        return stages.recorded_key(own) or None
    return None


def _spectra_file(experiment_dir: Path, plan: SearchPlan) -> tuple[Path, float]:
    """The spectra file for a plan's bundles, and the seconds spent
    writing it.

    Written once, under ``<experiment>/spectra/<run>/``, and reused by
    every later search of the same acquisition (0.0 seconds then): it
    depends on the bundles alone, not on any search setting. Two
    searches that need the same file at once take turns; the second
    finds it written.
    """
    directory = _spectra_dir(experiment_dir, plan.run)
    path = directory / f"{plan.run}.dia"
    with file_lock(directory.parent / f".{plan.run}.lock"):
        if stages.is_done(directory, plan.spectra_key) and path.is_file():
            return path, 0.0
        stages.invalidate(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # The writer builds the file under a temporary name and renames it
        # into place; one killed outright (SIGKILL) leaves that behind.
        # Every writer holds this lock, so any such file is stale.
        for stale in directory.glob(f".{path.name}.tmp-*"):
            stale.unlink(missing_ok=True)
        started = time.perf_counter()
        write_dia(
            [s.bundle for s in plan.sources],  # type: ignore[misc]
            path,
            overwrite=True,
            # A worker thread is not interrupted by a signal; it asks.
            should_stop=lambda: cancellation_requested() is not None,
        )
        stages.mark_done(directory, plan.spectra_key)
        return path, time.perf_counter() - started


def _link(target: Path, link: Path, *, within: Path) -> str:
    """Make ``link`` name the file ``target``; returns how.

    A symlink where the filesystem allows one (relative when the target
    is inside ``within``, so the experiment can be moved as a whole),
    else a hard link, else a copy.
    """
    link.parent.mkdir(parents=True, exist_ok=True)
    inside = target.is_relative_to(within)
    try:
        link.symlink_to(os.path.relpath(target, link.parent) if inside else target)
        return "symlink"
    except OSError:
        pass
    try:
        os.link(target, link)
        return "hardlink"
    except OSError:
        shutil.copy2(target, link)
        return "copy"


def _clear_unit(experiment_dir: Path, unit_dir: Path, unit: str) -> None:
    """Remove what an earlier run of this same search left, marker first.

    Not tidiness: a result in the directory is the jar's sign that the
    search is finished, and it would return that one (module docstring).
    The unit's rows in the tables go too, so a run that then fails
    leaves none rather than those of the run it replaced. Only this
    unit's own files are touched; no other search's.
    """
    stages.invalidate(unit_dir)
    (unit_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
    for directory in (unit_dir / "jar", unit_dir / "logs", unit_dir / COLLISION_DIR):
        if directory.is_dir():
            shutil.rmtree(directory)
    for table in (experiment_dir / "tables").glob("*"):
        partition = table_partition_dir(experiment_dir, table.name, unit)
        if partition.is_dir():
            shutil.rmtree(partition)


def parameter_disagreements(
    recorded: Mapping[str, str], flags: Mapping[str, FlagValue]
) -> list[str]:
    """Settings the result records differently from the ones asked for.

    A result library lists the effective value of most flags in its
    ``metadata`` table under the flag's own name. ``flags`` is what the
    jar was given. A difference would mean the result is not of this
    search (the jar took an existing one for it) or that the jar adjusted
    the setting itself. Numbers are compared as numbers (the jar writes
    ``10`` for ``10.0``) and text without regard to case; a flag the
    result does not record is not compared.
    """

    def same(asked: str, got: str) -> bool:
        try:
            return math.isclose(float(asked), float(got), rel_tol=1e-9, abs_tol=1e-12)
        except ValueError:
            return asked.casefold() == got.casefold()

    out = []
    for flag, value in flags.items():
        asked = format_value(value)
        if flag in recorded and not same(asked, recorded[flag]):
            out.append(
                f"the result records {flag} {recorded[flag]!r} but the search asked for {asked!r}"
            )
    return out


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)  # an input used in place, outside the unit


def _bound_sources(
    read: EncyclopediaReadResult,
    acquisitions: Acquisitions,
    allowed: set[int],
    result_library: Path,
) -> dict[str, int]:
    """``name the result uses for a run -> acquisition id``, checked.

    The result must name the run (or runs) this unit searched and
    nothing else. A result that names another run (a spectra file whose
    recorded source is some other acquisition of the experiment) would
    otherwise have its rows filed under the wrong acquisition without
    complaint.
    """
    used: set[int] = set()
    for holder, pieces in (
        (read.quant, ("precursor_quant", "peptide_quant")),
        (read.search, ("precursor_scores", "peptide_scores", "protein_scores")),
    ):
        for name in pieces if holder is not None else ():
            piece = getattr(holder, name)
            if piece.num_rows:
                used.update(
                    i for i in pc.unique(piece.column("acquisition_id")).to_pylist() if i is not None
                )
    what = (
        f"acquisition {next(iter(allowed))}"
        if len(allowed) == 1
        else f"acquisitions {sorted(allowed)}"
    )
    stray = used - allowed
    if stray:
        raise WorkflowError(
            f"{result_library.name} holds results for acquisition(s) {sorted(stray)}, "
            f"but this unit searched {what}"
        )
    con = sqlite3.connect(f"file:{result_library}?mode=ro&immutable=1", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        names: set[str] = set()
        # A report across runs names a run in its quantities that its
        # entries (one per precursor, under one run) may not.
        for table in ("entries", "peptidescores", "peptidequants"):
            if table in tables:
                names |= {str(r[0]) for r in con.execute(f"SELECT DISTINCT SourceFile FROM {table}")}
    finally:
        con.close()
    names.discard(RUN_AGNOSTIC_SOURCE)  # a score over every run names no run
    bound = {name: acquisitions.id_for_source(name) for name in sorted(names)}
    wrong = {name: i for name, i in bound.items() if i not in allowed}
    if wrong:
        raise WorkflowError(f"{result_library.name} names run(s) {wrong}, not {what}")
    return bound


# ──────────────────────────────────────────────────────────────────────
# the collision filter, on a finished search's result
# ──────────────────────────────────────────────────────────────────────

#: Bumped when what the collision step writes changes.
COLLISION_LAYOUT = 1


@dataclass(frozen=True)
class CollisionSettings:
    """``search.collision.apply_collision_filter``'s parameters.

    Off unless asked for (PI, 2026-10-17: EncyclopeDIA's own workflow
    does not filter, and neither did the lab's GUI runs). The fragment
    tolerance is in ppm only, as the filter was written for Orbitrap
    fragments; on ion-trap data widen it.
    """

    rt_threshold_s: float = 5.0
    fragment_ppm: float = 20.0
    min_shared_ions: int = 4


def _unit_record(experiment_dir: Path, unit: str) -> tuple[Path, SearchManifest]:
    unit_dir = experiment_dir / "units" / unit
    if not stages.is_done(unit_dir):
        raise WorkflowError(f"search {unit} has not finished; there is no result to use yet")
    return unit_dir, read_manifest(unit_dir, expect=SearchManifest)


def _input_path(experiment_dir: Path, recorded: str) -> Path:
    """A path a manifest records relative to the experiment, or absolute."""
    path = Path(recorded)
    return path if path.is_absolute() else experiment_dir / path


def collision_filter(
    experiment_dir: Path | str,
    unit: str,
    settings: CollisionSettings | None = None,
    *,
    resume: bool = False,
) -> Path:
    """Filter a finished search's result library for co-eluting
    identifications that share fragments; return the filtered library.

    The filter (:func:`constellation.massspec.search.collision.apply_collision_filter`)
    reads the result and the spectra file it came from, keeps the best
    of each cluster of collisions and drops the rest. The filtered copy
    goes to ``units/<unit>/collision/`` with ``collision.json`` saying
    what was dropped and why; the search's own result is not touched.
    """
    from constellation.massspec.search.collision import (
        apply_collision_filter,
        filter_elib_by_losers,
    )

    experiment_dir = Path(experiment_dir).resolve()
    settings = settings or CollisionSettings()
    unit_dir, record = _unit_record(experiment_dir, unit)
    library = unit_dir / record.outputs["result_library"]
    spectra = _input_path(experiment_dir, record.inputs["spectra_file"]["path"])
    out_dir = unit_dir / COLLISION_DIR
    out = out_dir / f"{library.name.removesuffix('.elib')}.collision-filtered.elib"
    key = stages.stage_key(
        search=record.stage_key,
        layout=COLLISION_LAYOUT,
        rt_threshold_s=settings.rt_threshold_s,
        fragment_ppm=settings.fragment_ppm,
        min_shared_ions=settings.min_shared_ions,
    )
    if resume and stages.is_done(out_dir, key) and out.is_file():
        return out
    stages.invalidate(out_dir)
    if out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    started = time.perf_counter()
    losers, meta = apply_collision_filter(
        library,
        spectra,
        rt_threshold_s=settings.rt_threshold_s,
        frag_ppm_tol=settings.fragment_ppm,
        min_shared_ions=settings.min_shared_ions,
        return_metadata=True,
    )
    tmp = out.with_name(f".{out.name}.tmp")
    filtered = filter_elib_by_losers(library, losers, tmp)
    tmp.replace(out)
    summary = {
        "library": _relative(library, experiment_dir),
        "spectra_file": _relative(spectra, experiment_dir),
        "settings": {
            "rt_threshold_s": settings.rt_threshold_s,
            "fragment_ppm": settings.fragment_ppm,
            "min_shared_ions": settings.min_shared_ions,
        },
        "n_losers": len(losers),
        "n_windows": meta.get("n_windows"),
        "n_entries": meta.get("n_entries"),
        "n_clusters": len(meta.get("clusters", ())),
        "rows_removed": filtered,
        "seconds": round(time.perf_counter() - started, 3),
        "losers": sorted(losers),
    }
    (out_dir / "collision.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    stages.mark_done(out_dir, key)
    return out


# ──────────────────────────────────────────────────────────────────────
# the quantitative report across searches (-libexport)
# ──────────────────────────────────────────────────────────────────────

#: Bumped when what a report writes changes.
REPORT_LAYOUT = 1
REPORT_PROGRAM = "libexport"

#: Flags a report takes from the searches it is made from when they agree
#: (the jar re-integrates every run under them). Paths and the scorer are
#: not among them: the report names its own inputs, and it scores with
#: its built-in JavaPot whatever Percolator the searches used.
_REPORT_INHERITS = (
    "-ftol", "-ftolunits", "-fixed", "-foffset", "-minNumOfQuantitativePeaks",
    "-numberOfQuantitativePeaks", "-integratePrecursors", "-maskBadIntegrations",
    "-smoothIntegrations", "-subtractBackground", "-normalizeByTIC",
    "-localizationModification", "-percolatorThreshold", "-percolatorProteinThreshold",
)  # fmt: skip


@dataclass(frozen=True)
class ReportResult:
    """What :func:`report_searches` did."""

    experiment_dir: Path
    report_dir: Path
    #: ``report-<8 hex digits>``: the directory under reports/ and the
    #: ``unit=`` partition of its tables.
    key: str
    manifest: QuantReportManifest
    table_rows: dict[str, int]
    reused: bool = False


def _finished_searches(experiment_dir: Path) -> dict[str, SearchManifest]:
    root = read_manifest(experiment_dir, expect=ExperimentManifest)
    out = {}
    for unit, relative in sorted(root.units.items()):
        if stages.is_done(experiment_dir / relative):
            out[unit] = read_manifest(experiment_dir / relative, expect=SearchManifest)
    return out


def _choose_searches(
    experiment_dir: Path, searches: Sequence[str] | None, library: Path | None
) -> dict[str, SearchManifest]:
    finished = _finished_searches(experiment_dir)
    if searches:
        unknown = [s for s in searches if s not in finished]
        if unknown:
            raise WorkflowError(
                f"not finished searches of {experiment_dir.name}: {', '.join(unknown)} "
                f"(`massspec summarize` lists them)"
            )
        chosen = {s: finished[s] for s in searches}
    else:
        _, acquisitions = _load_experiment(experiment_dir)
        merged = {
            r["acquisition_id"]
            for r in acquisitions.table.to_pylist()
            if r["source_kind"] == MERGED_KIND
        }
        chosen = {
            unit: record
            for unit, record in finished.items()
            if record.search_mode == PEPTIDE_CENTRIC
            and not set(record.acquisition_ids) & merged
            and (library is None or Path(record.inputs.get("library", {}).get("path", "")) == library)
        }
        if not chosen:
            raise WorkflowError(
                f"{experiment_dir.name} has no finished peptide-centric search of a single "
                f"acquisition{' against ' + library.name if library else ''} to report on"
            )
    libraries = {str(r.inputs.get("library", {}).get("path")) for r in chosen.values()}
    if len(libraries) != 1:
        counts = defaultdict(list)
        for unit, record in chosen.items():
            counts[str(record.inputs.get("library", {}).get("path"))].append(unit)
        listing = "; ".join(f"{Path(lib).name}: {len(units)} searches" for lib, units in counts.items())
        raise WorkflowError(
            f"a report is made from searches against one library, and these used "
            f"{len(libraries)} ({listing}); choose with --library or name the searches"
        )
    return chosen


def report_searches(
    experiment_dir: Path | str,
    *,
    fasta: Path | str,
    searches: Sequence[str] | None = None,
    library: Path | str | None = None,
    align: bool = True,
    extra: Mapping[str, FlagValue] | None = None,
    allow_undocumented: tuple[str, ...] = (),
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    annotate_fragments: bool = True,
    resume: bool = False,
    stream_to_stderr: bool = True,
) -> ReportResult:
    """Make EncyclopeDIA's quantitative report across finished searches.

    The jar's ``-libexport``: every run's result is re-read with its
    spectra, retention times are aligned across the runs (``align``),
    every precursor is quantified in every run, and the result is
    scored again as a whole. ``searches`` are unit keys; by default,
    every finished peptide-centric search of a single acquisition (not a
    merged one), which must all have used one library (``library``
    narrows them to the searches against it). Settings the searches
    agree on and the report takes are passed on (``_REPORT_INHERITS``);
    one they disagree on is refused, since the report would integrate
    every run under one value.

    Measured on 6.9.23 (2026-10-17): the jar refuses to start without
    ``-f`` though its help says it needs one only for the Pecan and
    XCorDIA exports; it scores with its built-in JavaPot, not the
    Percolator executable, so a kernel the bundled Percolator cannot run
    on does not stop it; it writes Percolator files into its input
    directory (``report_concatenated_*``), so that directory belongs to
    the report and holds links to each search's files, which it reads
    and does not change; and it names a protein score's run ``global``.

    Written to ``reports/<key>/``: the jar's report (``report.elib``,
    with its peptide and protein matrices and Diffacto input beside it),
    and tables under ``unit=<key>``, quantities and per-run scores by
    acquisition, the library-level tables and the protein scores (which
    span every run) under ``acquisition_id=-1``.
    """
    experiment_dir = Path(experiment_dir).resolve()
    fasta = Path(fasta).resolve()
    if not fasta.is_file():
        raise WorkflowError(f"--fasta not found: {fasta}")
    library_filter = Path(library).resolve() if library is not None else None
    chosen = _choose_searches(experiment_dir, searches, library_filter)
    (library_path,) = {Path(r.inputs["library"]["path"]) for r in chosen.values()}
    if not library_path.is_file():
        raise WorkflowError(f"the searched library is no longer at {library_path}")

    runs: dict[str, str] = {}
    by_acquisition: dict[int, str] = {}
    for unit, record in sorted(chosen.items()):
        for acquisition_id in record.acquisition_ids:
            if acquisition_id in by_acquisition:
                raise WorkflowError(
                    f"{by_acquisition[acquisition_id]} and {unit} are both searches of "
                    f"acquisition {acquisition_id}; a report takes each acquisition once "
                    f"(name the searches to use)"
                )
            by_acquisition[acquisition_id] = unit
        name = Path(record.inputs["spectra_file"]["link"]).name
        if name in runs:
            raise WorkflowError(
                f"{runs[name]} and {unit} both searched a file named {name}; a report "
                f"takes each run once, under its own name"
            )
        runs[name] = unit

    settings: dict[str, FlagValue] = {}
    for flag in _REPORT_INHERITS:
        values = {r.engine_settings.get(flag, r.engine_flags.get(flag)) for r in chosen.values()}
        values.discard(None)
        if len(values) > 1:
            raise WorkflowError(
                f"the searches disagree on {flag} ({sorted(map(str, values))}); a report "
                f"integrates every run under one value"
            )
        if values:
            settings[flag] = values.pop()
    for flag, value in (extra or {}).items():
        settings[flag] = value
    settings["-a"] = align
    handle = resolve_jar()

    key_digest = stages.stage_key(
        layout=REPORT_LAYOUT,
        engine=ENGINE,
        engine_version=handle.version,
        program=REPORT_PROGRAM,
        # A search run again under the same key (without --resume) is a
        # new result in the same place; the result file tells them apart.
        searches=sorted(
            (r.stage_key, stages.input_identity([experiment_dir / "units" / u / r.outputs["result_library"]]))
            for u, r in chosen.items()
        ),
        settings={k: format_value(v) for k, v in sorted(settings.items())},
        library=stages.input_identity([library_path]),
        fasta=stages.input_identity([fasta]),
        fragments=annotate_fragments,
        id_scheme=ID_SCHEME,
    )
    key = f"report-{key_digest[:8]}"
    report_dir = experiment_dir / "reports" / key
    inputs_dir = report_dir / "inputs"
    output = report_dir / "report.elib"
    flags: dict[str, FlagValue] = {
        "-i": inputs_dir, "-o": output, "-l": library_path, "-f": fasta, **settings,
    }  # fmt: skip
    argv = compose_argv(REPORT_PROGRAM, flags, allow_undocumented=allow_undocumented)

    def reused() -> ReportResult:
        done = read_manifest(report_dir, expect=QuantReportManifest)
        return ReportResult(experiment_dir, report_dir, key, done, {}, reused=True)

    if resume and stages.is_done(report_dir, key_digest):
        return reused()
    with _sole_run(report_dir, f"the report {key}"):
        if resume and stages.is_done(report_dir, key_digest):
            return reused()  # finished by another command between the check and the lock

        # ── the report's directory, made new, with the runs linked in ───
        _clear_report(experiment_dir, report_dir, key)
        inputs_dir.mkdir(parents=True)
        acquisition_ids: list[int] = []
        for name, unit in sorted(runs.items()):
            unit_dir = experiment_dir / "units" / unit
            record = chosen[unit]
            acquisition_ids.extend(record.acquisition_ids)
            _link(_input_path(experiment_dir, record.inputs["spectra_file"]["path"]), inputs_dir / name, within=experiment_dir)
            for path in sorted((unit_dir / "jar").iterdir()):
                if path.name == name or path.suffix == ".pdf":
                    continue
                _link(path, inputs_dir / path.name, within=experiment_dir)

        record = QuantReportManifest(
            created_at=_utc_now(),
            constellation_version=CONSTELLATION_VERSION,
            status="running",
            engine=ENGINE,
            engine_version=str(handle.version),
            program=REPORT_PROGRAM,
            acquisition_ids=sorted(acquisition_ids),
            units=sorted(chosen),
            stage_key=key_digest,
            inputs={"library": _file_record(library_path), "fasta": _file_record(fasta)},
            parameters={"align": align, "extra": {k: format_value(v) for k, v in (extra or {}).items()}},
            argv=argv,
            engine_flags={flag: format_value(value) for flag, value in flags.items()},
        )
        write_manifest(report_dir, record)
        try:
            run = run_program(
                REPORT_PROGRAM,
                flags,
                log_dir=report_dir / "logs",
                cwd=report_dir,
                jvm_heap_max=jvm_heap_max,
                jvm_heap_min=jvm_heap_min,
                jvm_tmpdir=jvm_tmpdir,
                allow_undocumented=allow_undocumented,
                stream_to_stderr=stream_to_stderr,
            )
            if not output.is_file():
                raise MissingOutputError(
                    f"{REPORT_PROGRAM} finished but wrote no {output.name}; see {report_dir / 'logs'}"
                )
            with _experiment_lock(experiment_dir):
                _, acquisitions = _load_experiment(experiment_dir)
            read = read_encyclopedia(output, acquisitions=acquisitions, annotate_fragments=annotate_fragments)
            sources = _bound_sources(read, acquisitions, set(acquisition_ids), output)
            con = open_library(output)
            try:
                recorded = read_metadata(con)
            finally:
                con.close()
            table_rows = _write_tables(
                experiment_dir, key, content_keyed_tables(read), acquisition_id=ALL_ACQUISITIONS
            )
        except BaseException as exc:
            _record_stop(report_dir, record, exc)
            raise

        extras = sorted(p.name for p in report_dir.glob("report.elib.*"))
        write_manifest(
            report_dir,
            replace(
                record,
                status="finished",
                sources=sources,
                argv=[str(a) for a in run.argv],
                engine_settings={k: v for k, v in sorted(recorded.items()) if k.startswith("-")},
                outputs={"report": output.name, "logs": "logs", **{name: name for name in extras}},
                runtime={
                    "finished_at": _utc_now(),
                    "elapsed_seconds": round(run.elapsed_seconds, 3),
                    "returncode": run.returncode,
                    "java": run.java_version,
                    "jar_sha256": run.jar_sha256,
                    "table_rows": table_rows,
                },
                caveats=list(read.caveats),
            ),
        )
        _update_experiment(
            experiment_dir,
            reports={key: f"reports/{key}"},
            tables={name: f"tables/{name}" for name in table_rows},
        )
        stages.mark_done(report_dir, key_digest)
        return ReportResult(
            experiment_dir, report_dir, key, read_manifest(report_dir, expect=QuantReportManifest), table_rows
        )


def _clear_report(experiment_dir: Path, report_dir: Path, key: str) -> None:
    """Remove what an earlier run of this same report left, marker first."""
    stages.invalidate(report_dir)
    if report_dir.is_dir():
        for child in report_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    for table in (experiment_dir / "tables").glob("*"):
        partition = table_partition_dir(experiment_dir, table.name, key)
        if partition.is_dir():
            shutil.rmtree(partition)


# ──────────────────────────────────────────────────────────────────────
# the whole chain
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QuantifyResult:
    """What :func:`quantify` did."""

    #: The merged search of the fractions, when fractions were given.
    library_search: SearchResult | None
    #: What the runs were searched against: the merged search's result
    #: (collision-filtered when asked), or the library given.
    library: Path
    #: The collision-filtered library, when the filter ran.
    collision_filtered: Path | None
    runs: FanOutResult
    report: ReportResult | None


def quantify(
    experiment_dir: Path | str,
    *,
    runs: Sequence[Path | str],
    fasta: Path | str,
    library: Path | str,
    fractions: Sequence[Path | str] = (),
    merged_name: str | None = None,
    collision: CollisionSettings | None = None,
    parameters: SearchParameters | None = None,
    jobs: int = 1,
    align: bool = True,
    report: bool = True,
    allow_undocumented: tuple[str, ...] = (),
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    annotate_fragments: bool = True,
    resume: bool = False,
    stream_to_stderr: bool = True,
    progress: SearchProgress | None = None,
) -> QuantifyResult:
    """The chromatogram-library workflow, end to end, into one experiment.

    1. With ``fractions`` (gas-phase fractions of a pooled sample): they
       are merged into one spectra file and searched as one against
       ``library`` (a predicted library, typically). The result is the
       chromatogram library: empirical retention times and fragments for
       everything it identified. ``collision`` filters it first (off
       unless given).
    2. Each of ``runs`` is searched against the chromatogram library (or
       against ``library`` itself without fractions), ``jobs`` at a time.
    3. A quantitative report across those searches (``report=False``
       skips it).

    Every step is the same function the standalone commands call, with
    the same keys, so a step already done by hand, or by an earlier
    ``quantify``, is found and reused under ``resume``. A run whose
    search fails does not stop the others, but the report waits for all
    of them: it is skipped, and the failures are in ``runs``.
    """
    experiment_dir = Path(experiment_dir).resolve()
    common = dict(
        experiment_dir=experiment_dir,
        fasta=fasta,
        parameters=parameters,
        allow_undocumented=allow_undocumented,
        jvm_heap_max=jvm_heap_max,
        jvm_heap_min=jvm_heap_min,
        jvm_tmpdir=jvm_tmpdir,
        annotate_fragments=annotate_fragments,
        resume=resume,
        stream_to_stderr=stream_to_stderr,
        progress=progress,
    )
    if not runs:
        raise WorkflowError("nothing to quantify: give at least one run")
    library_search = None
    filtered = None
    searched_against = Path(library).resolve()
    if fractions:
        merged = search_acquisitions(
            fractions, library=library, merge=True, merged_name=merged_name, **common
        )
        if not merged.ok:
            ((plan, exc),) = merged.failures
            # Raised as it is, so the command exits with the jar's status.
            exc.add_note(f"the search of the merged fractions ({plan.unit}) failed")
            raise exc
        (library_search,) = merged.results
        searched_against = library_search.unit_dir / library_search.manifest.outputs["result_library"]
        if collision is not None:
            filtered = collision_filter(experiment_dir, library_search.unit, collision, resume=resume)
            searched_against = filtered
    elif collision is not None:
        raise WorkflowError("the collision filter applies to a chromatogram library: give fractions")
    run_searches = search_acquisitions(runs, library=searched_against, jobs=jobs, **common)
    report_result = None
    if report and run_searches.ok:
        report_result = report_searches(
            experiment_dir,
            fasta=fasta,
            searches=[r.unit for r in run_searches.results],
            align=align,
            allow_undocumented=allow_undocumented,
            jvm_heap_max=jvm_heap_max,
            jvm_heap_min=jvm_heap_min,
            jvm_tmpdir=jvm_tmpdir,
            annotate_fragments=annotate_fragments,
            resume=resume,
            stream_to_stderr=stream_to_stderr,
        )
    return QuantifyResult(
        library_search=library_search,
        library=searched_against,
        collision_filtered=filtered,
        runs=run_searches,
        report=report_result,
    )


__all__ = [
    "COLLISION_DIR",
    "COLLISION_LAYOUT",
    "ENGINE",
    "REPORT_LAYOUT",
    "SPECTRA_LAYOUT",
    "UNIT_LAYOUT",
    "CollisionSettings",
    "FanOutResult",
    "QuantifyResult",
    "ReportResult",
    "SearchPlan",
    "SearchProgress",
    "SearchResult",
    "WorkflowError",
    "collision_filter",
    "content_keyed_tables",
    "parameter_disagreements",
    "plan_searches",
    "quantify",
    "read_experiment_table",
    "report_searches",
    "run_search",
    "search_acquisition",
    "search_acquisitions",
]
