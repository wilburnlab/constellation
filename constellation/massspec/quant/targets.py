"""Adapters that normalize extraction sources into ``XIC_TARGET_TABLE``.

The chromatogram extractor consumes one shape — ``XIC_TARGET_TABLE`` (a
target list of what to extract). Every real source is adapted into it
here, so "general m/z+RT target" (a theoretical library) and
"search-results" (identified PSMs) are *sources*, not separate
operations:

  ``targets_from_library``        PRECURSOR_TABLE (+ PEPTIDE_TABLE) — the
                                  general theoretical target: precursor
                                  m/z + predicted RT, sample-agnostic.
  ``targets_from_search``         PSM_TABLE — identification-anchored;
                                  the only source carrying ``scan`` (so
                                  ``assigned_scans_only`` lights up).
  ``targets_from_precursor_quant``PRECURSOR_QUANT joined to a Library —
                                  re-extract already-quantified precursors
                                  at their observed RT.
  ``targets_from_table``          a bare hand-authored table validated
                                  against the contract (non-peptide m/z
                                  lists / ad-hoc targets).
  ``targets_from_experiment``     a search experiment directory (or one
                                  of its units): the precursors a search
                                  identified in one acquisition, at their
                                  observed RT with the window the engine
                                  integrated over.

``load_targets(path)`` is the convenience used by the CLI: it sniffs a
search experiment or unit directory (by its manifest's ``kind``), a
ParquetDir bundle (Library / Search) or a bare ``.parquet`` / ``.tsv``
file and dispatches to the right adapter.

The library/quant adapters map the ``-1.0`` "no prediction" sentinel
(``rt_predicted`` / ``rt_observed``) to null ``rt_center`` so absent RT
falls through to all-RT extraction rather than gating on a bogus 0-ish
window.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import cast_to_schema
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE

__all__ = [
    "OBSERVED",
    "RT_SCALE_KEY",
    "TARGET_RUN_KEY",
    "library_rt_scale",
    "same_run",
    "target_run",
    "targets_from_library",
    "targets_from_search",
    "targets_from_precursor_quant",
    "targets_from_table",
    "targets_from_experiment",
    "load_targets",
]


def _sentinel_to_null(
    arr: pa.Array | pa.ChunkedArray, sentinel: float = -1.0
) -> pa.Array:
    """Map a float sentinel (and any existing nulls) to null."""
    return pc.if_else(pc.equal(arr, sentinel), pa.scalar(None, pa.float64()), arr)


def _attach_modseq(precursors: pa.Table, peptides: pa.Table | None) -> pa.Table:
    """Left-join ``modified_sequence`` (and ``is_decoy`` / ``decoy_set``
    where the peptide table has them) from a peptide table onto precursors, keeping the
    precursors' order: query numbers downstream are row positions, and
    ``Table.join`` does not promise an order."""
    if peptides is None or "modified_sequence" not in peptides.column_names:
        return precursors.append_column(
            "modified_sequence", pa.nulls(precursors.num_rows, pa.string())
        )
    cols = ["peptide_id", "modified_sequence"] + [
        c for c in ("is_decoy", "decoy_set") if c in peptides.column_names
    ]
    pep = peptides.select(cols)
    row = "__row__"
    joined = precursors.append_column(row, pa.array(np.arange(precursors.num_rows, dtype=np.int64))).join(
        pep, keys="peptide_id", join_type="left outer"
    )
    return joined.take(pc.sort_indices(joined.column(row))).drop_columns([row])


#: Where an adapter records, in the target table's schema metadata, the run
#: a target list was made from (a PSM table's raw file, an experiment's
#: acquisition source file): a source of one run's identifications is for
#: that run's bundle (:func:`target_run`).
TARGET_RUN_KEY = "x.massspec.targets.run"


def target_run(targets: pa.Table) -> str | None:
    """The run ``targets`` were made from, as its source names it (a file
    name or path), or None for a source of no run (a library, a bare
    table). Read raw, so a run named by digits stays a string."""
    value = (targets.schema.metadata or {}).get(TARGET_RUN_KEY.encode())
    return value.decode("utf-8") if value else None


def _with_run(targets: pa.Table, run: str | None) -> pa.Table:
    if not run:
        return targets
    meta = dict(targets.schema.metadata or {})
    meta[TARGET_RUN_KEY.encode()] = str(run).encode("utf-8")
    return targets.replace_schema_metadata(meta)


def same_run(a: str, b: str) -> bool:
    """Whether two names of runs (file names or paths, as tools report them)
    name one run: the same file name, or the same name once its extensions
    are taken off (``run.raw``, ``run.dia``, ``run`` and ``/data/run.raw``
    are one run; ``run_2.raw`` is another), for POSIX and Windows paths."""

    def file_name(name: str) -> str:
        return re.split(r"[\\/]", str(name))[-1]

    def bare(name: str) -> str:
        out = file_name(name)
        while "." in out and out.rsplit(".", 1)[1].lower() in _RUN_EXTENSIONS:
            out = out.rsplit(".", 1)[0]
        return out

    return file_name(a) == file_name(b) or bare(a) == bare(b)


#: Extensions a run's files carry, taken off when two names are compared.
_RUN_EXTENSIONS = frozenset({"raw", "dia", "mzml", "mzxml", "mgf", "d", "wiff", "wiff2", "elib", "dlib", "gz"})


#: Where a library states the clock of its ``rt_predicted``.
RT_SCALE_KEY = "x.massspec.prediction.rt_scale"
#: The scale of a run's own seconds (a chromatogram library's times).
OBSERVED = "observed"


def library_rt_scale(metadata: dict) -> str | None:
    """The clock a library's ``rt_predicted`` is on, as it says: a model's
    scale (``chronologer_hi_seconds``, ``koina:<model>:irt``, ``irt``),
    ``observed`` (seconds of the runs a chromatogram library came from),
    or None when it does not say (taken as run seconds, which is what the
    libraries that say nothing hold). Koina's older key is read too."""
    scale = metadata.get(RT_SCALE_KEY)
    if scale:
        return str(scale)
    legacy = metadata.get("x.koina.rt_scale")
    return f"koina:{metadata.get('x.koina.rt_model', '')}:{legacy}" if legacy else None


def targets_from_library(library: object) -> pa.Table:
    """Normalize a ``Library`` (its precursors + peptides) into targets.

    The general theoretical-target path: each precursor becomes a target
    keyed by ``precursor_id``, anchored on ``precursor_mz``. Its library
    retention time is ``rt_library``, on the library's ``rt_scale``; it is
    also the gating centre (``rt_center``) only when that scale is a run's
    own seconds (``observed``, or unstated). A model's time is no run's
    clock: until 2026-11-07 it was used as one, so a window around it
    gated scans in the wrong place.
    """
    precursors: pa.Table = library.precursors  # type: ignore[attr-defined]
    peptides: pa.Table = library.peptides  # type: ignore[attr-defined]
    joined = _attach_modseq(precursors, peptides)
    scale = library_rt_scale(getattr(library, "metadata_extras", {}) or {})
    rt = _sentinel_to_null(joined.column("rt_predicted"))
    on_run_clock = scale in (None, OBSERVED)
    out = pa.table(
        {
            "target_id": joined.column("precursor_id"),
            "modified_sequence": joined.column("modified_sequence"),
            "precursor_charge": joined.column("charge"),
            "precursor_mz": joined.column("precursor_mz"),
            "rt_center": rt if on_run_clock else pa.nulls(joined.num_rows, pa.float64()),
            "precursor_id": joined.column("precursor_id"),
            "peptide_id": joined.column("peptide_id"),
            **{c: joined.column(c) for c in ("is_decoy", "decoy_set") if c in joined.column_names},
            "rt_library": rt,
            "rt_scale": pa.array([scale] * joined.num_rows, pa.string()),
        }
    )
    return cast_to_schema(out, XIC_TARGET_TABLE)


def _one_run(table: pa.Table, what: str, acquisition_id: int | None, *, raw_file: bool = False) -> pa.Table:
    """``table``'s rows of one run: the one it holds, or ``acquisition_id``.
    A run is its ``acquisition_id``, or (``raw_file``, for PSMs not yet
    linked to an acquisition) its raw file. A target list is for one run:
    rows of another run's scans, swept against this one, are matched at
    the wrong times and pinned to the wrong scans. A table linked in part
    (ids beside nulls) is refused unless ``acquisition_id`` chooses the
    run: its unlinked rows may be any run's, and until 2026-11-11 they
    passed as the linked run's (review of 2026-11-10, C9). An unlinked
    table of one raw file is that run's whatever ``acquisition_id`` says:
    the id then names the acquisition it is swept as (which is how
    ``chromatogram extract`` accepts one run's identifications against
    another run's bundle; C7)."""
    ids = table.column("acquisition_id")
    linked = pc.any(pc.is_valid(ids)).as_py() if table.num_rows else False
    if acquisition_id is not None:
        if not linked:
            files = set(table.column("raw_file").to_pylist()) if raw_file and table.num_rows else set()
            if len(files) == 1:
                # One run's rows, named by their raw file: the id says which
                # acquisition they are swept as, and chooses nothing.
                return table
            raise ValueError(f"the {what} are not linked to acquisitions, so none can be chosen by its id")
        out = table.filter(pc.fill_null(pc.equal(ids, acquisition_id), False))
        if out.num_rows == 0:
            held = sorted(set(pc.drop_null(ids).to_pylist()))
            raise ValueError(f"the {what} hold no rows of acquisition {acquisition_id} (they hold {held})")
        return out
    unlinked = ids.null_count
    if linked and unlinked:
        listing = ""
        if raw_file and "raw_file" in table.column_names:
            files = sorted(set(pc.drop_null(table.column("raw_file").filter(pc.is_null(ids))).to_pylist()))
            listing = f" (raw files {', '.join(files[:6])}{', ...' if len(files) > 6 else ''})" if files else ""
        raise ValueError(
            f"{unlinked} of the {table.num_rows} {what} are linked to no acquisition{listing}, beside rows that "
            f"are: they may be another run's. Link every row, or pass acquisition_id (--acquisition-id) to take "
            f"the rows of one acquisition"
        )
    runs = sorted(set(pc.drop_null(ids).to_pylist())) if linked else (
        sorted(set(table.column("raw_file").to_pylist())) if raw_file else [])  # fmt: skip
    if len(runs) > 1:
        kind = "acquisitions" if linked else "raw files"
        listing = ", ".join(str(r) for r in runs[:6]) + (", ..." if len(runs) > 6 else "")
        raise ValueError(
            f"the {what} are of {len(runs)} {kind} ({listing}); a target list is for one run: "
            f"pass acquisition_id (--acquisition-id) to say which"
        )
    return table


def targets_from_search(search: object, *, acquisition_id: int | None = None) -> pa.Table:
    """Normalize a ``Search``'s ``psms`` (PSM_TABLE) into targets of one run.

    The identification-anchored path: each PSM becomes a target keyed by
    ``psm_id``, anchored on the measured precursor ``mz`` at its scan +
    retention time. ``scan`` is carried (the only source that does), so
    ``assigned_scans_only`` extraction is available. Its ``is_decoy`` and
    ``precursor_id`` are carried (a decoy has ``decoy_set`` 0, the search's
    own decoys), so a decoy stays a decoy and a library's abundances reach
    the targets; until 2026-11-10 they were dropped, and a search's decoys
    came back as targets. A row with no ``precursor_id`` (a reader that
    derives none, a decoy of a reader before 2026-11-11) gets the
    content-derived id of its ``modified_sequence`` at its charge, so a
    decoy finds a library's decoy rows as its target finds its own (review
    of 2026-11-10, C5). The PSMs must be of one run, or ``acquisition_id``
    says which (``psm_id`` is unique only within one); the run's raw file is
    recorded (:func:`target_run`). NOTE the PSM ``peptide_id`` is
    engine-internal — NOT a library FK — so it is deliberately dropped.
    """
    psms: pa.Table = search.psms  # type: ignore[attr-defined]
    # The columns a run, a decoy and a precursor are read from, null where a
    # table (a hand-made one, a reader of an engine without them) has none.
    for name, kind in (("acquisition_id", pa.int64()), ("raw_file", pa.string()), ("precursor_id", pa.int64()),
                       ("is_decoy", pa.bool_())):  # fmt: skip
        if name not in psms.column_names:
            psms = psms.append_column(name, pa.nulls(psms.num_rows, kind))
    psms = _one_run(psms, "PSMs", acquisition_id, raw_file=True)
    decoy = psms.column("is_decoy")
    out = pa.table(
        {
            "target_id": psms.column("psm_id"),
            "modified_sequence": psms.column("modified_sequence"),
            "precursor_charge": psms.column("charge"),
            "precursor_mz": psms.column("mz"),
            "rt_center": psms.column("retention_time_s"),
            "scan": psms.column("scan"),
            "precursor_id": _derived_precursor_ids(psms),
            "is_decoy": decoy,
            "decoy_set": pc.if_else(pc.fill_null(decoy, False), pa.scalar(0, pa.int16()),
                                    pa.scalar(None, pa.int16())),  # fmt: skip
        }
    )
    files = sorted(set(pc.drop_null(psms.column("raw_file")).to_pylist())) if psms.num_rows else []
    return _with_run(cast_to_schema(out, XIC_TARGET_TABLE), files[0] if len(files) == 1 else None)


def _derived_precursor_ids(psms: pa.Table) -> pa.Array:
    """``psms.precursor_id``, a null filled with the content-derived id of
    the row's ``modified_sequence`` at its charge where it has both (each
    distinct pair keyed once); a sequence that has no key stays null."""
    from constellation.core.sequence.proforma import ProFormaError
    from constellation.massspec.library.ids import canonical_precursor_key, content_ids

    ids = psms.column("precursor_id").combine_chunks() if psms.num_rows else pa.array([], pa.int64())
    if ids.null_count == 0:
        return ids
    modseq = psms.column("modified_sequence").to_pylist()
    charge = psms.column("charge").to_pylist()
    values = ids.to_pylist()
    keys: dict[tuple[str, int], int | None] = {}
    for i, value in enumerate(values):
        if value is not None or not modseq[i] or charge[i] is None or int(charge[i]) < 1:
            continue
        pair = (modseq[i], int(charge[i]))
        if pair not in keys:
            try:
                keys[pair] = content_ids([canonical_precursor_key(pair[0], pair[1])])[0].as_py()
            except (ValueError, KeyError, ProFormaError):  # an unparsable or adduct form has no key
                keys[pair] = None
        values[i] = keys[pair]
    return pa.array(values, pa.int64())


def targets_from_precursor_quant(quant: object, library: object, *, acquisition_id: int | None = None) -> pa.Table:
    """Normalize ``PRECURSOR_QUANT`` (joined to a Library) into targets of one run.

    Re-extract already-quantified precursors at their *observed* RT, inside
    the window the quantity was integrated over (``rt_start`` / ``rt_end``)
    where it says one. The library supplies m/z + charge + modseq and the
    decoy flags (quant rows carry only the ``precursor_id`` FK + observed
    RT). The quantities must be of one run, or ``acquisition_id`` says which:
    a table over several runs made one target per run under one id (until
    2026-11-10), and dropped the window and the decoy flags. Rows come in
    the quantities' order.
    """
    pq_tbl: pa.Table = quant.precursor_quant  # type: ignore[attr-defined]
    for name, kind in (("acquisition_id", pa.int64()), ("rt_start", pa.float64()), ("rt_end", pa.float64())):
        if name not in pq_tbl.column_names:
            pq_tbl = pq_tbl.append_column(name, pa.nulls(pq_tbl.num_rows, kind))
    pq_tbl = _one_run(pq_tbl, "quantities", acquisition_id)
    precursors: pa.Table = library.precursors  # type: ignore[attr-defined]
    peptides: pa.Table = library.peptides  # type: ignore[attr-defined]
    prec = _attach_modseq(precursors, peptides)
    keep = ["precursor_id", "peptide_id", "charge", "precursor_mz", "modified_sequence"]
    keep += [c for c in ("is_decoy", "decoy_set") if c in prec.column_names]
    row = "__row__"
    rows = pq_tbl.select(["precursor_id", "rt_observed", "rt_start", "rt_end"])
    rows = rows.append_column(row, pa.array(np.arange(rows.num_rows, dtype=np.int64)))
    joined = rows.join(prec.select(keep), keys="precursor_id", join_type="left outer")
    joined = joined.take(pc.sort_indices(joined.column(row)))  # a join promises no order
    out = pa.table(
        {
            "target_id": joined.column("precursor_id"),
            "modified_sequence": joined.column("modified_sequence"),
            "precursor_charge": joined.column("charge"),
            "precursor_mz": joined.column("precursor_mz"),
            "rt_center": _sentinel_to_null(joined.column("rt_observed")),
            "rt_start": _sentinel_to_null(joined.column("rt_start")),
            "rt_end": _sentinel_to_null(joined.column("rt_end")),
            "precursor_id": joined.column("precursor_id"),
            "peptide_id": joined.column("peptide_id"),
            **{c: joined.column(c) for c in ("is_decoy", "decoy_set") if c in joined.column_names},
        }
    )
    return cast_to_schema(out, XIC_TARGET_TABLE)


def targets_from_table(table: pa.Table | str | Path) -> pa.Table:
    """Validate a hand-authored target table against the contract.

    Accepts a ``pa.Table`` or a path to a ``.parquet`` / ``.tsv`` /
    ``.csv`` file. Raises if the required ``target_id`` column is absent;
    missing optional columns fill with null.
    """
    if not isinstance(table, pa.Table):
        path = Path(table)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(path)
        elif path.suffix in (".tsv", ".csv", ".txt"):
            from pyarrow import csv

            delimiter = "," if path.suffix == ".csv" else "\t"
            table = csv.read_csv(
                path, parse_options=csv.ParseOptions(delimiter=delimiter)
            )
        else:
            raise ValueError(
                f"unsupported target file {path.suffix!r}; use .parquet/.tsv/.csv"
            )
    return cast_to_schema(table, XIC_TARGET_TABLE)


def targets_from_experiment(
    experiment_dir: str | Path, *, unit: str | None = None, acquisition_id: int | None = None
) -> pa.Table:
    """The precursors one search identified in one acquisition, as targets.

    ``experiment_dir`` is a search experiment
    (:mod:`constellation.massspec.manifest`). Each precursor with a
    quantity row becomes a target at its observed retention time, with
    the window the engine integrated over as ``rt_start`` / ``rt_end``
    where it reported one.

    A target list is for one run as one search saw it. ``unit`` (the
    search's key) and ``acquisition_id`` may be left out while the
    experiment holds a single result; with several searches or several
    acquisitions, enough has to be given to leave one, and the error
    lists the choices.
    """
    from constellation.massspec.manifest import ALL_ACQUISITIONS, open_table

    root = Path(experiment_dir)

    def table(name: str) -> pa.Table:
        try:
            dataset = open_table(root, name)
        except FileNotFoundError:
            raise ValueError(
                f"{root} has no {name} table; it is not a searched experiment"
            ) from None
        return dataset.to_table()

    quant = table("precursor_quant")
    held = sorted(
        {
            (u, a)
            for u, a in zip(
                quant.column("unit").to_pylist(),
                quant.column("acquisition_id").to_pylist(),
                strict=True,
            )
        }
    )
    wanted = [
        (u, a)
        for u, a in held
        if (unit is None or u == unit) and (acquisition_id is None or a == acquisition_id)
    ]
    listing = "; ".join(f"search {u}, acquisition {a}" for u, a in held)
    if not wanted:
        asked = " and ".join(
            part
            for part in (
                f"search {unit}" if unit is not None else "",
                f"acquisition {acquisition_id}" if acquisition_id is not None else "",
            )
            if part
        )
        raise ValueError(f"{root} has no results for {asked} (it has: {listing})")
    if len(wanted) > 1:
        raise ValueError(
            f"{root} holds {len(wanted)} results that fit ({listing}); say which one the "
            f"targets are for, or pass that search's directory under units/"
        )
    unit, acquisition_id = wanted[0]

    def of(tbl: pa.Table, *, shared: bool = False) -> pa.Table:
        # A result across several runs (a quantitative report) keeps its
        # precursors and peptides once, under ALL_ACQUISITIONS, not under
        # each run.
        ids = [acquisition_id, ALL_ACQUISITIONS] if shared else [acquisition_id]
        return tbl.filter(
            pc.and_(
                pc.equal(tbl.column("unit"), unit),
                pc.is_in(tbl.column("acquisition_id"), pa.array(ids, tbl.column("acquisition_id").type)),
            )
        )

    quant = of(quant).select(["precursor_id", "rt_observed", "rt_start", "rt_end"])
    precursors = of(table("precursor"), shared=True).select(
        ["precursor_id", "peptide_id", "charge", "precursor_mz"]
    )
    peptides = of(table("peptide"), shared=True).select(["peptide_id", "modified_sequence"])
    joined = quant.join(precursors, keys="precursor_id", join_type="inner").join(
        peptides, keys="peptide_id", join_type="left outer"
    )
    out = pa.table(
        {
            "target_id": joined.column("precursor_id"),
            "modified_sequence": joined.column("modified_sequence"),
            "precursor_charge": joined.column("charge"),
            "precursor_mz": joined.column("precursor_mz"),
            "rt_center": _sentinel_to_null(joined.column("rt_observed")),
            "rt_start": joined.column("rt_start"),
            "rt_end": joined.column("rt_end"),
            "precursor_id": joined.column("precursor_id"),
            "peptide_id": joined.column("peptide_id"),
        }
    )
    return _with_run(cast_to_schema(out, XIC_TARGET_TABLE), _source_file(root, acquisition_id))


def _source_file(experiment_dir: Path, acquisition_id: int) -> str | None:
    """The source file the experiment registered ``acquisition_id`` as, or
    None when it records none."""
    from constellation.massspec.acquisitions import load_acquisitions

    try:
        acquisitions = load_acquisitions(experiment_dir / "acquisitions")
    except FileNotFoundError:
        return None
    rows = acquisitions.table.filter(pc.equal(acquisitions.table.column("acquisition_id"), acquisition_id))
    return rows.column("source_file")[0].as_py() if rows.num_rows else None


def load_targets(path: str | Path, *, library: object | None = None, acquisition_id: int | None = None) -> pa.Table:
    """Load a target source from disk and normalize it.

    Auto-detects by what is on disk: a search experiment or one of its
    unit directories (by the ``kind`` in its ``manifest.json``), a
    Library ParquetDir bundle (``precursors.parquet``, read without its
    fragments) or a ``predict-library`` output directory holding one
    (``library_pqdir/``), a Search bundle
    (``psms.parquet``), a ``PRECURSOR_QUANT`` bundle
    (``precursor_quant.parquet``, requires ``library=``), or a bare
    ``.parquet`` / ``.tsv`` target file. ``acquisition_id`` chooses the run
    of a source that holds several (an experiment, PSMs, quantities). A
    source of one run's identifications (PSMs, an experiment) says which
    run in the table's metadata (:func:`target_run`), so a caller can hold
    it to the bundle swept (``chromatogram extract`` does; review of
    2026-11-10, C7).
    """
    p = Path(path)
    if p.is_dir():
        from constellation.core.manifests import manifest_kind

        kind = manifest_kind(p)
        if kind == "massspec_experiment":
            return targets_from_experiment(p, acquisition_id=acquisition_id)
        if kind == "massspec_search":
            from constellation.core.manifests import read_manifest
            from constellation.massspec.manifest import SearchManifest

            unit = read_manifest(p, expect=SearchManifest)
            if len(unit.acquisition_ids) != 1:
                raise ValueError(
                    f"{p} searched acquisitions {unit.acquisition_ids}; a target list is "
                    f"for one run"
                )
            # <experiment>/units/<unit>/
            return targets_from_experiment(
                p.parent.parent, unit=p.name, acquisition_id=unit.acquisition_ids[0]
            )
        if not (p / "precursors.parquet").exists() and (p / "library_pqdir" / "precursors.parquet").exists():
            p = p / "library_pqdir"  # a predict-library output directory
        if (p / "precursors.parquet").exists():
            from constellation.massspec.library.io import load_library

            # Targets are precursors: the fragments (the bulk of a predicted
            # library) are not read (library.fragment_abundance streams them).
            return targets_from_library(load_library(p, format="parquet_dir", fragments=False))
        if (p / "psms.parquet").exists():
            from constellation.massspec.search.io import load_search

            return targets_from_search(load_search(p, format="parquet_dir"), acquisition_id=acquisition_id)
        if (p / "precursor_quant.parquet").exists():
            if library is None:
                raise ValueError(
                    "a PRECURSOR_QUANT source needs library= for m/z + charge"
                )
            from constellation.massspec.quant.io import load_quant

            return targets_from_precursor_quant(
                load_quant(p, format="parquet_dir"), library, acquisition_id=acquisition_id
            )
        raise ValueError(f"unrecognized target bundle at {p}")
    return targets_from_table(p)
