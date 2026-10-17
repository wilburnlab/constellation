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

import math
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from constellation import __version__ as CONSTELLATION_VERSION
from constellation.core import stages
from constellation.core.manifests import (
    MANIFEST_FILENAME,
    manifest_kind,
    read_manifest,
    write_manifest,
)
from constellation.massspec.acquisition_structure import infer_acquisition_structure
from constellation.massspec.acquisitions import (
    Acquisitions,
    acquisition_record_from_bundle,
    load_acquisitions,
    save_acquisitions,
)
from constellation.massspec.io.encyclopedia import DiaFile, read_encyclopedia, write_dia
from constellation.massspec.io.encyclopedia._mirror import open_library, read_metadata
from constellation.massspec.io.encyclopedia._read import EncyclopediaReadResult
from constellation.massspec.library.ids import ID_SCHEME, with_content_ids
from constellation.massspec.manifest import (
    ExperimentManifest,
    SearchManifest,
    open_table,
    table_partition_dir,
)
from constellation.massspec.search.encyclopedia._common import sha256_file
from constellation.massspec.search.encyclopedia.outputs import RunOutputs, locate_outputs
from constellation.massspec.search.encyclopedia.params import (
    EXECUTION_ONLY_FLAGS,
    SPECTRUM_CENTRIC,
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
from constellation.thirdparty.jvm import JvmRunError

ENGINE = "encyclopedia"

#: Bumped when what a unit writes changes (tables, their columns, how
#: ids are made), so that a unit written by older code is not taken for
#: finished by newer code.
UNIT_LAYOUT = 1

#: Bumped when the spectra file written for a bundle changes, so that
#: one written by older code is written again (and searched again).
SPECTRA_LAYOUT = 1

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
    """The reader's tables under content-derived ids, without their
    ``acquisition_id`` columns (the partition directory carries it)."""
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
    return {
        name: (table.drop_columns(["acquisition_id"]) if "acquisition_id" in table.column_names else table)
        for name, table in tables.items()
    }


def _write_tables(
    experiment_dir: Path, unit: str, acquisition_id: int, tables: dict[str, pa.Table]
) -> dict[str, int]:
    rows: dict[str, int] = {}
    for name, table in tables.items():
        directory = table_partition_dir(experiment_dir, name, unit, acquisition_id)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / ".part-0.parquet.tmp"
        pq.write_table(table, tmp)
        tmp.replace(directory / "part-0.parquet")
        rows[name] = table.num_rows
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
# the workflow
# ──────────────────────────────────────────────────────────────────────


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _load_experiment(experiment_dir: Path) -> tuple[ExperimentManifest, Acquisitions]:
    kind = manifest_kind(experiment_dir)
    if kind is None:
        if experiment_dir.is_dir() and any(
            p.name != "bundles" for p in experiment_dir.iterdir()
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
    :mod:`constellation.massspec.search.encyclopedia.params`).

    Raises :class:`WorkflowError` (or one of the more specific errors of
    the pieces it calls) before the JVM starts for anything that can be
    known beforehand: an unsupported input, a mode that does not suit
    the data, a flag the jar does not document, a Percolator that cannot
    run on this machine, a missing library.
    """
    experiment_dir = Path(experiment_dir).resolve()
    fasta = Path(fasta).resolve()
    library = Path(library).resolve() if library is not None else None
    parameters = parameters or SearchParameters()
    for label, path in (("--fasta", fasta), ("--library", library)):
        if path is not None and not path.is_file():
            raise WorkflowError(f"{label} not found: {path}")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest, acquisitions = _load_experiment(experiment_dir)
    src = _classify(Path(source).resolve(), experiment_dir, convert_raw=True)

    # ── which engine, and can it run ────────────────────────────────
    resolved = resolve_search_mode(
        search_mode, acquisition_type=src.acquisition_type, search_type=search_type
    )
    if resolved.mode == SPECTRUM_CENTRIC:
        raise WorkflowError(
            "spectrum-centric search (ScribeTwo) is not available through this command yet"
        )
    if resolved.library_required and library is None:
        raise WorkflowError(f"a {resolved.mode} search needs --library")
    # Only the peptide-centric program runs the Percolator executable
    # (ScribeTwo scores with its own), so this waits for the mode, even
    # though for an unconverted .raw that means after the conversion.
    extra_argv = [str(x) for pair in parameters.extra.items() for x in pair]
    blocker = percolator_blocker(parameters.percolator_version, extra_argv)
    if blocker is not None:
        raise WorkflowError(blocker)

    # ── the acquisition ─────────────────────────────────────────────
    acquisitions, (acquisition_id,) = acquisitions.register([src.record])
    spectra_name = f"{src.run}.dia"
    if src.kind != "spectra_file":
        acquisitions = acquisitions.with_source_alias(acquisition_id, spectra_name)
    save_acquisitions(acquisitions, experiment_dir / manifest.acquisitions)
    # The directory is an experiment from here on, so that it can be read
    # (``massspec summarize``) whatever becomes of this search. A search
    # is added to ``units`` only once it has finished.
    manifest = replace(manifest, constellation_version=CONSTELLATION_VERSION, id_scheme=ID_SCHEME)
    write_manifest(experiment_dir, manifest)

    # ── the search, and so its directory ────────────────────────────
    settings: dict[str, FlagValue] = {**resolved.flags, **parameters.flags()}
    if "-o" in settings:
        raise WorkflowError(
            "-o is not passed through: the jar's outputs belong beside its input, in the "
            "search's own directory, where the rest of the workflow looks for them"
        )
    files: dict[str, FlagValue] = {"-f": fasta}
    if library is not None:
        files["-l"] = library
    # Checked now, before anything is written: a flag the jar does not
    # document, or a jar older than the pin.
    compose_argv(
        resolved.program, {"-i": src.path, **files, **settings}, allow_undocumented=allow_undocumented
    )
    handle = resolve_jar()

    spectra_key = _spectra_key(experiment_dir, src)
    search_key = stages.stage_key(
        engine=ENGINE,
        engine_version=handle.version,
        program=resolved.program,
        settings={k: str(v) for k, v in settings.items() if k not in EXECUTION_ONLY_FLAGS},
        spectra=spectra_key if spectra_key is not None else stages.input_identity([src.path]),
        library=stages.input_identity([library]) if library else None,
        fasta=stages.input_identity([fasta]),
    )
    unit = f"{src.run}-{search_key[:8]}"
    unit_dir = experiment_dir / "units" / unit
    jar_dir = unit_dir / "jar"
    # What the jar is given: a name for the spectra file inside the
    # search's own directory. Its outputs go beside it, named for it.
    jar_input = jar_dir / spectra_name
    flags: dict[str, FlagValue] = {"-i": jar_input, **files, **settings}

    key = stages.stage_key(
        search=search_key,
        layout=UNIT_LAYOUT,
        fragments=annotate_fragments,
        id_scheme=ID_SCHEME,
    )
    if resume and stages.is_done(unit_dir, key):
        done = read_manifest(unit_dir, expect=SearchManifest)
        return SearchResult(
            experiment_dir, unit_dir, unit, acquisition_id, done, {}, reused=True, run=src.run
        )

    # ── the spectra file, and the search's directory made new ───────
    if src.bundle is not None:
        spectra_file, wrote_spectra_s = _spectra_file(experiment_dir, src, spectra_key)
    else:
        spectra_file, wrote_spectra_s = src.path, 0.0
    _clear_unit(experiment_dir, unit_dir, unit)
    linked_as = _link(spectra_file, jar_input, within=experiment_dir)

    # ── the record, written before anything runs ────────────────────
    # So that a search that fails or is stopped still says what it was
    # (``massspec summarize``). Rewritten when the search ends.
    inputs: dict[str, dict[str, Any]] = {"fasta": _file_record(fasta)}
    if library is not None:
        inputs["library"] = _file_record(library)
    inputs["source"] = {
        "path": str(src.path),
        "kind": src.kind,
        "sha256": src.record.get("source_sha256"),
        "acquisition_type": src.acquisition_type,
    }
    inputs["spectra_file"] = {
        "path": _relative(spectra_file, experiment_dir),
        "written_by": "constellation" if spectra_key is not None else "given",
        "link": _relative(jar_input, unit_dir),
        "linked_as": linked_as,
    }
    record = SearchManifest(
        created_at=_utc_now(),
        constellation_version=CONSTELLATION_VERSION,
        engine=ENGINE,
        engine_version=str(handle.version),
        program=resolved.program,
        search_mode=resolved.mode,
        status="running",
        acquisition_ids=[acquisition_id],
        search_key=search_key,
        stage_key=key,
        inputs=inputs,
        parameters=parameters.as_given(),
        argv=compose_argv(resolved.program, flags, allow_undocumented=allow_undocumented),
        engine_flags={flag: format_value(value) for flag, value in flags.items()},
        runtime={"spectra_file_seconds": round(wrote_spectra_s, 3)},
    )
    write_manifest(unit_dir, record)

    try:
        # ── the search ──────────────────────────────────────────────
        run = run_program(
            resolved.program,
            flags,
            log_dir=unit_dir / "logs",
            cwd=jar_dir,
            jvm_heap_max=jvm_heap_max,
            jvm_heap_min=jvm_heap_min,
            jvm_tmpdir=jvm_tmpdir,
            allow_undocumented=allow_undocumented,
            stream_to_stderr=stream_to_stderr,
        )
        outputs = locate_outputs(jar_dir, spectra_name, resolved.program)

        # ── the tables ──────────────────────────────────────────────
        read = read_encyclopedia(
            outputs.result_library,
            acquisitions=acquisitions,
            annotate_fragments=annotate_fragments,
        )
        sources = _bound_sources(read, acquisitions, acquisition_id, outputs)
        con = open_library(outputs.result_library)
        try:
            recorded = read_metadata(con)
        finally:
            con.close()
        disagreements = parameter_disagreements(recorded, flags)
        table_rows = _write_tables(experiment_dir, unit, acquisition_id, content_keyed_tables(read))
    except BaseException as exc:
        stopped = isinstance(exc, (KeyboardInterrupt, SystemExit))
        write_manifest(
            unit_dir,
            replace(
                record,
                status="cancelled" if stopped else "failed",
                error=_describe_failure(exc),
                runtime={**record.runtime, "finished_at": _utc_now()},
            ),
        )
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
    manifest = replace(
        manifest,
        constellation_version=CONSTELLATION_VERSION,
        id_scheme=ID_SCHEME,
        units={**manifest.units, unit: f"units/{unit}"},
        tables={**manifest.tables, **{name: f"tables/{name}" for name in table_rows}},
    )
    write_manifest(experiment_dir, manifest)
    stages.mark_done(unit_dir, key)
    return SearchResult(
        experiment_dir, unit_dir, unit, acquisition_id,
        read_manifest(unit_dir, expect=SearchManifest), table_rows,
        disagreements=tuple(disagreements),
        run=src.run,
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
        return f"the jar exited {exc.returncode}; the end of its stderr:\n" + "\n".join(tail)
    message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
    return f"{type(exc).__name__}: {message}"


def _spectra_dir(experiment_dir: Path, run: str) -> Path:
    return experiment_dir / "spectra" / run


def _spectra_key(experiment_dir: Path, src: _Source) -> str | None:
    """What identifies the spectra file Constellation writes for a
    bundle: the bundle it comes from and how it is written.

    Also answers for that file handed back as the input (it is found by
    where it lies, and the key by what was recorded when it was
    written), so a search is the same search whichever way its input is
    named. ``None`` for any other spectra file, which is identified by
    the file itself.
    """
    if src.bundle is not None:
        return stages.stage_key(
            layout=SPECTRA_LAYOUT,
            source=stages.input_identity([src.bundle / "manifest.json"]),
            source_sha256=src.record.get("source_sha256"),
        )
    own = _spectra_dir(experiment_dir, src.run)
    if src.path.parent == own:
        return stages.recorded_key(own) or None
    return None


def _spectra_file(experiment_dir: Path, src: _Source, key: str | None) -> tuple[Path, float]:
    """The spectra file for a bundle, and the seconds spent writing it.

    Written once per acquisition, under ``<experiment>/spectra/<run>/``,
    and reused by every later search of that acquisition (0.0 seconds
    then). It depends on the bundle alone, not on any search setting.
    """
    assert src.bundle is not None
    directory = _spectra_dir(experiment_dir, src.run)
    path = directory / f"{src.run}.dia"
    if stages.is_done(directory, key) and path.is_file():
        return path, 0.0
    stages.invalidate(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_dia([src.bundle], path, overwrite=True)
    stages.mark_done(directory, key)
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
    for directory in (unit_dir / "jar", unit_dir / "logs"):
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
    acquisition_id: int,
    outputs: RunOutputs,
) -> dict[str, int]:
    """``name the result uses for its run -> acquisition id``, checked.

    The result must name the run this unit searched and nothing else. A
    result that names another run (a spectra file whose recorded source
    is some other acquisition of the experiment) would otherwise have
    its rows filed under the wrong acquisition without complaint.
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
    stray = used - {acquisition_id}
    if stray:
        raise WorkflowError(
            f"{outputs.result_library.name} holds results for acquisition(s) {sorted(stray)}, "
            f"but this unit searched acquisition {acquisition_id}"
        )
    con = sqlite3.connect(f"file:{outputs.result_library}?mode=ro&immutable=1", uri=True)
    try:
        names = {str(r[0]) for r in con.execute("SELECT DISTINCT SourceFile FROM entries")}
        names |= {str(r[0]) for r in con.execute("SELECT DISTINCT SourceFile FROM peptidescores")}
    finally:
        con.close()
    bound = {name: acquisitions.id_for_source(name) for name in sorted(names)}
    wrong = {name: i for name, i in bound.items() if i != acquisition_id}
    if wrong:
        raise WorkflowError(
            f"{outputs.result_library.name} names run(s) {wrong}, not acquisition {acquisition_id}"
        )
    return bound


__all__ = [
    "ENGINE",
    "SPECTRA_LAYOUT",
    "UNIT_LAYOUT",
    "SearchResult",
    "WorkflowError",
    "content_keyed_tables",
    "parameter_disagreements",
    "read_experiment_table",
    "search_acquisition",
]
