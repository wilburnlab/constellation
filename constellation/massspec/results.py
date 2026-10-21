"""What the searches of an experiment found, for browsing it.

An experiment directory (``massspec search``, see
:mod:`constellation.massspec.manifest`) keeps each search's result as
tables partitioned by search (``unit``) and acquisition. This module
reads them the way a browser asks:

- :func:`search_units` — the finished searches, with their mode, the
  acquisitions they searched, their library and tolerances;
- :func:`identifications` — one row per identified precursor (a
  peptide-centric search) or per accepted match (a spectrum-centric one,
  with its scan), joined with peptide, scores and proteins;
- :func:`acquisition_bundles` — which converted acquisition (bundle) holds
  each searched run's peaks;
- :func:`precursor_detail`, :func:`library_spectrum`,
  :func:`searched_library_spectrum` — one precursor's quantitation, the
  spectrum the result stored for it, and the spectrum of the library that
  was searched; :func:`trace_targets`, the fragments worth tracing for it;
- :func:`rt_mapping` — a search's alignment of the run's retention times to
  its library's.

Rows are read through partition filters, so asking for one search of one
acquisition reads only its files. Tables a report shares across runs sit
under ``acquisition_id = ALL_ACQUISITIONS`` and are read with each run's.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from constellation.massspec.manifest import ALL_ACQUISITIONS, open_table

IDENTIFICATION_SCHEMA = pa.schema(
    [
        pa.field("unit", pa.string(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("precursor_id", pa.int64()),
        pa.field("peptide_id", pa.int64()),
        pa.field("modified_sequence", pa.string()),
        pa.field("sequence", pa.string()),
        pa.field("charge", pa.int32()),
        pa.field("precursor_mz", pa.float64()),
        #: Seconds: the apex a peptide-centric search reports, or the
        #: retention time of a spectrum-centric match's scan.
        pa.field("rt", pa.float64()),
        #: The window the engine integrated over (peptide-centric).
        pa.field("rt_start", pa.float64()),
        pa.field("rt_end", pa.float64()),
        pa.field("intensity", pa.float64()),
        pa.field("n_quant_ions", pa.int32()),
        pa.field("score", pa.float64()),
        pa.field("qvalue", pa.float64()),
        pa.field("pep", pa.float64()),
        #: Accessions, separated by ";".
        pa.field("proteins", pa.string()),
        #: The matched scan (spectrum-centric); null for peptide-centric.
        pa.field("scan", pa.int32()),
        pa.field("is_decoy", pa.bool_()),
    ],
    metadata={b"schema_name": b"Identifications"},
)

LIBRARY_SPECTRUM_SCHEMA = pa.schema(
    [
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        #: mzPAF; null for a peak the library does not annotate.
        pa.field("annotation", pa.string()),
        pa.field("ion_type", pa.string()),
        pa.field("ion_number", pa.int16()),
        pa.field("fragment_charge", pa.int8()),
        pa.field("loss", pa.string()),
        pa.field("correlation", pa.float64()),
        pa.field("quantified", pa.bool_()),
    ],
    metadata={b"schema_name": b"LibrarySpectrum"},
)

_LETTERS = "abcxyz"


# ── reading partitions ────────────────────────────────────────────────────


def _read(
    experiment_dir: Path,
    table: str,
    *,
    unit: str | None,
    acquisition_id: int | None,
    shared: bool = False,
    columns: list[str] | None = None,
    where: ds.Expression | None = None,
) -> pa.Table | None:
    """One table's rows for a search and acquisition, or ``None`` when the
    experiment has no such table. ``shared`` also takes the rows a report
    keeps for all its runs; ``where`` narrows the scan further."""
    try:
        dataset = open_table(experiment_dir, table)
    except FileNotFoundError:
        return None
    condition = None
    if unit is not None:
        condition = ds.field("unit") == unit
    if acquisition_id is not None:
        ids = [acquisition_id, ALL_ACQUISITIONS] if shared else [acquisition_id]
        part = ds.field("acquisition_id").isin(pa.array(ids, pa.int64()))
        condition = part if condition is None else condition & part
    if where is not None:
        condition = where if condition is None else condition & where
    if columns is not None:
        columns = [c for c in dict.fromkeys(["unit", "acquisition_id", *columns]) if c in dataset.schema.names]
    return dataset.to_table(columns=columns, filter=condition)


def _first_per(table: pa.Table, keys: list[str]) -> pa.Table:
    """One row per ``keys``: the first of each group (rows a report and a
    search both hold, or one entity under two acquisitions)."""
    others = [c for c in table.column_names if c not in keys]
    if not others:
        return table.group_by(keys, use_threads=False).aggregate([])
    grouped = table.group_by(keys, use_threads=False).aggregate([(c, "first") for c in others])
    return grouped.rename_columns([c.removesuffix("_first") for c in grouped.column_names])


def _unit_manifest(experiment_dir: Path, unit: str) -> dict[str, Any]:
    path = experiment_dir / "units" / unit / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{experiment_dir} has no search {unit!r}")
    return json.loads(path.read_text(encoding="utf-8"))


# ── searches ──────────────────────────────────────────────────────────────


#: The jar's tolerance flags: (value flag, unit flag) by tolerance.
_TOLERANCE_FLAGS = {"fragment_tolerance": ("-ftol", "-ftolunits"), "precursor_tolerance": ("-ptol", "-ptolunits")}
_TOLERANCE_UNITS = {"ppm": "ppm", "amu": "Da", "da": "Da"}


def _tolerance(manifest: dict[str, Any], name: str) -> dict[str, Any] | None:
    """A search's ``name`` tolerance as ``{"value", "unit"}``: the one the
    engine recorded running with when it recorded one (ScribeTwo adjusts
    what it is given, and records the adjusted value), else as given."""
    recorded = manifest.get("engine_settings") or {}
    value_flag, unit_flag = _TOLERANCE_FLAGS[name]
    unit = _TOLERANCE_UNITS.get(str(recorded.get(unit_flag, "")).lower())
    try:
        value = float(recorded[value_flag])
    except (KeyError, TypeError, ValueError):
        value = None
    if value is not None and unit is not None:
        return {"value": value, "unit": unit}
    return (manifest.get("parameters") or {}).get(name)


def search_units(experiment_dir: Path | str) -> list[dict[str, Any]]:
    """The experiment's finished searches, as plain data: ``unit``,
    ``search_mode``, ``engine``, ``program``, ``acquisition_ids``,
    ``library`` (path or None), ``fragment_tolerance`` /
    ``precursor_tolerance`` (``{"value", "unit"}``, as the engine recorded
    running with them, else as given), and for a merged search
    ``merged_from`` (the fractions' acquisition ids)."""
    root = Path(experiment_dir)
    out = []
    for manifest_path in sorted((root / "units").glob("*/manifest.json")):
        unit_dir = manifest_path.parent
        if not (unit_dir / "_SUCCESS").exists():
            continue
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs = data.get("inputs") or {}
        merged = inputs.get("sources")
        out.append(
            {
                "unit": unit_dir.name,
                "search_mode": data.get("search_mode") or "",
                "engine": data.get("engine") or "",
                "program": data.get("program") or "",
                "acquisition_ids": [int(a) for a in data.get("acquisition_ids") or ()],
                "library": (inputs.get("library") or {}).get("path"),
                "fragment_tolerance": _tolerance(data, "fragment_tolerance"),
                "precursor_tolerance": _tolerance(data, "precursor_tolerance"),
                "merged_from": [int(s["acquisition_id"]) for s in merged] if merged else None,
            }
        )
    return out


# ── identifications ───────────────────────────────────────────────────────


def _proteins(root: Path, unit: str | None, acquisition_id: int | None) -> pa.Table | None:
    edges = _read(root, "protein_peptide", unit=unit, acquisition_id=acquisition_id, shared=True,
                  columns=["protein_id", "peptide_id"])
    proteins = _read(root, "protein", unit=unit, acquisition_id=acquisition_id, shared=True,
                     columns=["protein_id", "accession"])
    if edges is None or proteins is None or edges.num_rows == 0:
        return None
    edges = _first_per(edges.select(["unit", "protein_id", "peptide_id"]), ["unit", "protein_id", "peptide_id"])
    proteins = _first_per(proteins.select(["unit", "protein_id", "accession"]), ["unit", "protein_id"])
    # Sorted first: a single-threaded group keeps each group's rows in order.
    named = edges.join(proteins, keys=["unit", "protein_id"], join_type="inner").sort_by("accession")
    listed = named.group_by(["unit", "peptide_id"], use_threads=False).aggregate([("accession", "list")])
    return pa.table(
        {
            "unit": listed.column("unit"),
            "peptide_id": listed.column("peptide_id"),
            "proteins": pc.binary_join(listed.column("accession_list"), ";"),
        }
    )


def _peptide_centric(
    root: Path, unit: str | None, acquisition_id: int | None, where: ds.Expression | None
) -> pa.Table | None:
    quant = _read(root, "precursor_quant", unit=unit, acquisition_id=acquisition_id, where=where,
                  columns=["precursor_id", "intensity", "rt_observed", "rt_start", "rt_end", "n_quant_ions"])
    if quant is None or quant.num_rows == 0:
        return None
    precursors = _read(root, "precursor", unit=unit, acquisition_id=acquisition_id, shared=True, where=where,
                       columns=["precursor_id", "peptide_id", "charge", "precursor_mz"])
    peptides = _read(root, "peptide", unit=unit, acquisition_id=acquisition_id, shared=True,
                     columns=["peptide_id", "sequence", "modified_sequence"])
    scores = _read(root, "precursor_score", unit=unit, acquisition_id=acquisition_id, where=where,
                   columns=["precursor_id", "score", "qvalue", "pep"])
    joined = quant.join(
        _first_per(precursors.drop_columns(["acquisition_id"]), ["unit", "precursor_id"]),
        keys=["unit", "precursor_id"],
        join_type="inner",
    )
    if peptides is not None:
        joined = joined.join(
            _first_per(peptides.drop_columns(["acquisition_id"]), ["unit", "peptide_id"]),
            keys=["unit", "peptide_id"],
            join_type="left outer",
        )
    if scores is not None and scores.num_rows:
        scores = scores.filter(pc.is_valid(scores.column("precursor_id")))
        joined = joined.join(
            _first_per(scores, ["unit", "acquisition_id", "precursor_id"]),
            keys=["unit", "acquisition_id", "precursor_id"],
            join_type="left outer",
        )
    proteins = _proteins(root, unit, acquisition_id)
    if proteins is not None:
        joined = joined.join(proteins, keys=["unit", "peptide_id"], join_type="left outer")
    rt = joined.column("rt_observed")
    rt = pc.if_else(pc.less(rt, 0), pa.scalar(None, pa.float64()), rt)  # -1 = not observed
    return _shape(joined.drop_columns(["rt_observed"]).append_column("rt", rt), is_decoy=False)


def _spectrum_centric(
    root: Path, unit: str, acquisition_id: int | None, where: ds.Expression | None, max_qvalue: float | None
) -> pa.Table | None:
    psm = _read(root, "psm", unit=unit, acquisition_id=acquisition_id, where=where,
                columns=["precursor_id", "modified_sequence", "sequence", "charge", "mz", "retention_time_s",
                         "scan", "score", "qvalue", "pep", "proteins", "is_decoy"])
    if psm is None or psm.num_rows == 0:
        return None
    if max_qvalue is not None:
        # ScribeTwo scores the best match of each precursor; its other
        # matches carry no q-value and are accepted, or not, with it.
        keys = ["unit", "acquisition_id", "precursor_id"]
        scored = psm.filter(pc.is_valid(psm.column("qvalue")))
        best = scored.group_by(keys, use_threads=False).aggregate([("qvalue", "min")])
        psm = psm.join(best, keys=keys, join_type="left outer")
        q = pc.coalesce(psm.column("qvalue"), psm.column("qvalue_min"))
        psm = psm.filter(pc.fill_null(pc.less_equal(q, max_qvalue), False)).drop_columns(["qvalue_min"])
        if psm.num_rows == 0:
            return None
    precursors = _read(root, "precursor", unit=unit, acquisition_id=acquisition_id, shared=True,
                       columns=["precursor_id", "peptide_id", "precursor_mz"])
    table = psm.rename_columns(
        ["rt" if c == "retention_time_s" else c for c in psm.column_names]
    )
    if precursors is not None and precursors.num_rows:
        table = table.join(
            _first_per(precursors.drop_columns(["acquisition_id"]), ["unit", "precursor_id"]),
            keys=["unit", "precursor_id"],
            join_type="left outer",
        )
        table = table.set_column(
            table.column_names.index("precursor_mz"),
            "precursor_mz",
            pc.coalesce(table.column("precursor_mz"), table.column("mz")),
        )
    else:
        table = table.append_column("precursor_mz", table.column("mz"))
    return _shape(table.drop_columns(["mz"]))


def _shape(table: pa.Table, is_decoy: bool | None = None) -> pa.Table:
    columns = {}
    for field in IDENTIFICATION_SCHEMA:
        if field.name in table.column_names:
            columns[field.name] = table.column(field.name).cast(field.type)
        elif field.name == "is_decoy" and is_decoy is not None:
            columns[field.name] = pa.array([is_decoy] * table.num_rows, pa.bool_())
        else:
            columns[field.name] = pa.nulls(table.num_rows, field.type)
    return pa.table(columns, schema=IDENTIFICATION_SCHEMA)


def identifications(
    experiment_dir: Path | str,
    *,
    unit: str | None = None,
    acquisition_id: int | None = None,
    max_qvalue: float | None = 0.01,
    include_decoys: bool = False,
    precursor_id: int | None = None,
) -> pa.Table:
    """What the experiment's searches identified (``IDENTIFICATION_SCHEMA``).

    A peptide-centric search gives one row per precursor it quantified, at
    its reported apex (``rt``) and integration window. A spectrum-centric
    search gives one row per match it scored, with the matched ``scan``.
    ``unit`` and ``acquisition_id`` narrow the read; rows say which search
    and run they belong to either way. Only finished searches are read: a
    quantitative report (``reports/``) re-quantifies runs under a key of
    its own, and is read only when ``unit`` names it. Rows above
    ``max_qvalue`` are left out (None keeps every row), and so are decoy
    matches unless ``include_decoys``. A peptide-centric row without a
    q-value is kept; a spectrum-centric match without one (ScribeTwo
    scores only the best match of each precursor) is kept when its
    precursor's best match is within ``max_qvalue``. ``precursor_id``
    reads the rows of one precursor only.
    """
    root = Path(experiment_dir)
    where = None if precursor_id is None else ds.field("precursor_id") == precursor_id
    modes = {u["unit"]: u["search_mode"] for u in search_units(root)}
    parts: list[pa.Table] = []
    centric = [u for u, m in modes.items() if m == "spectrum-centric" and unit in (None, u)]
    if not (unit is not None and unit in centric):
        found = _peptide_centric(root, unit, acquisition_id, where)
        if found is not None:
            if unit is None:  # the searches' rows, not a report's
                peptide_centric = [u for u, m in modes.items() if m != "spectrum-centric"]
                found = found.filter(pc.is_in(found.column("unit"), pa.array(peptide_centric, pa.string())))
            parts.append(found)
    for u in centric:
        found = _spectrum_centric(root, u, acquisition_id, where, max_qvalue)
        if found is not None:
            parts.append(found)
    table = pa.concat_tables(parts) if parts else IDENTIFICATION_SCHEMA.empty_table()
    if max_qvalue is not None:
        q = table.column("qvalue")
        table = table.filter(pc.or_kleene(pc.is_null(q), pc.less_equal(q, max_qvalue)))
    if not include_decoys:
        table = table.filter(pc.invert(pc.fill_null(table.column("is_decoy"), False)))
    return table.sort_by([("unit", "ascending"), ("acquisition_id", "ascending"), ("rt", "ascending")])


# ── where each searched run's peaks are ───────────────────────────────────


def _bundle_manifest(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None
    if data.get("kind") not in ("thermo_acquisition", "encyclopedia_dia_acquisition"):
        return None
    return data


def acquisition_bundles(
    experiment_dir: Path | str, *, candidates: Iterable[Path | str] = ()
) -> list[dict[str, Any]]:
    """Each acquisition of the experiment and the bundle holding its peaks.

    One dict per acquisition: ``acquisition_id``, ``source_file``,
    ``bundle`` (a Path, or None), ``via`` (how it was found), ``units``
    (searches of it) and ``merged_from``. A bundle is found, in order:

    1. as the input a search recorded, when that input was a bundle;
    2. converted from a ``.raw`` into ``<experiment>/bundles/<name>``;
    3. among ``candidates`` (bundles attached by the caller), by the run's
       name: for a run searched from an EncyclopeDIA spectra file there is
       no other way, since such a search keeps no bundle.

    A bundle whose recorded SHA-256 differs from the run's is not taken.
    A merged acquisition (fractions searched as one) has no bundle of its
    own; its ``merged_from`` names the fractions, which are found each in
    the same way.
    """
    from constellation.massspec.acquisitions import load_acquisitions

    root = Path(experiment_dir)
    acquisitions = load_acquisitions(root / "acquisitions")
    rows = {int(r["acquisition_id"]): r for r in acquisitions.table.to_pylist()}
    found: dict[int, dict[str, Any]] = {
        aid: {
            "acquisition_id": aid,
            "source_file": r["source_file"],
            "bundle": None,
            "via": None,
            "units": [],
            "merged_from": [int(i) for i in r["merged_from"]] if r.get("merged_from") else None,
        }
        for aid, r in rows.items()
    }

    def take(aid: int, bundle: Path, via: str) -> None:
        entry = found.get(aid)
        if entry is None or entry["bundle"] is not None:
            return
        manifest = _bundle_manifest(bundle)
        if manifest is None:
            return
        want, have = rows[aid].get("source_sha256"), manifest.get("source_sha256")
        if want and have and want != have:
            return
        entry["bundle"], entry["via"] = bundle, via

    for path in sorted((root / "units").glob("*/manifest.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        inputs = data.get("inputs") or {}
        for aid in data.get("acquisition_ids") or ():
            if int(aid) in found:
                found[int(aid)]["units"].append(path.parent.name)
        sources = inputs.get("sources") or ([inputs["source"]] if inputs.get("source") else [])
        ids = [s.get("acquisition_id") for s in sources]
        if len(sources) == 1 and ids[0] is None:
            ids = [int((data.get("acquisition_ids") or [None])[0])] if data.get("acquisition_ids") else [None]
        for source, aid in zip(sources, ids):
            if aid is None:
                continue
            kind, where = source.get("kind"), Path(str(source.get("path") or ""))
            if kind == "bundle":
                take(int(aid), where, "search input")
            elif kind == "raw":
                take(int(aid), root / "bundles" / where.stem, "experiment bundles")

    for candidate in candidates:
        bundle = Path(candidate)
        manifest = _bundle_manifest(bundle)
        if manifest is None:
            continue
        try:
            aid = acquisitions.id_for_source(str(manifest.get("source_file") or bundle.name))
        except KeyError:
            continue
        take(aid, bundle, "attached")
    return [found[aid] for aid in sorted(found)]


# ── one precursor ─────────────────────────────────────────────────────────


def precursor_detail(
    experiment_dir: Path | str, precursor_id: int, *, unit: str, acquisition_id: int, scan: int | None = None
) -> dict[str, Any]:
    """Everything a search recorded about one precursor in one run, as
    plain data: its identification row (``None`` if it has none; for a
    spectrum-centric search, which matches a precursor in several scans,
    the match in ``scan``, else the best-scored match),
    ``quantified_ions`` (m/z, intensity, correlation of the ions summed),
    ``chromatogram`` (the engine's median chromatogram: rt, intensity),
    and the search's ``library``, ``fragment_tolerance`` and
    ``precursor_tolerance`` (as the engine recorded running with them,
    else as given)."""
    root = Path(experiment_dir)
    rows = identifications(
        root, unit=unit, acquisition_id=acquisition_id, max_qvalue=None, include_decoys=True,
        precursor_id=precursor_id,
    ).to_pylist()
    if scan is not None:
        rows = [r for r in rows if r["scan"] == scan]
    row = sorted(rows, key=lambda r: (r["qvalue"] is None, r["qvalue"] or 0.0))[:1]
    mine = ds.field("precursor_id") == precursor_id

    def rows_of(table: str, columns: list[str]) -> list[dict[str, Any]]:
        t = _read(root, table, unit=unit, acquisition_id=acquisition_id, columns=columns, where=mine)
        return [] if t is None else t.select(columns).to_pylist()

    manifest = _unit_manifest(root, unit)
    return {
        "identification": row[0] if row else None,
        "quantified_ions": rows_of("fragment_quant", ["mz", "intensity", "correlation"]),
        "chromatogram": sorted(rows_of("precursor_chromatogram", ["rt", "intensity"]), key=lambda r: r["rt"]),
        "library": ((manifest.get("inputs") or {}).get("library") or {}).get("path"),
        "fragment_tolerance": _tolerance(manifest, "fragment_tolerance"),
        "precursor_tolerance": _tolerance(manifest, "precursor_tolerance"),
    }


def _peptide_of(root: Path, precursor_id: int, unit: str, acquisition_id: int | None) -> dict[str, Any] | None:
    precursors = _read(root, "precursor", unit=unit, acquisition_id=acquisition_id, shared=True,
                       columns=["precursor_id", "peptide_id", "charge", "precursor_mz"],
                       where=ds.field("precursor_id") == precursor_id)
    hit = [] if precursors is None else precursors.slice(0, 1).to_pylist()
    if not hit:
        return None
    peptides = _read(root, "peptide", unit=unit, acquisition_id=acquisition_id, shared=True,
                     columns=["peptide_id", "sequence", "modified_sequence"],
                     where=ds.field("peptide_id") == hit[0]["peptide_id"])
    pep = [] if peptides is None else peptides.slice(0, 1).to_pylist()
    return {**hit[0], **(pep[0] if pep else {})}


def library_spectrum(
    experiment_dir: Path | str, precursor_id: int, *, unit: str, acquisition_id: int
) -> pa.Table:
    """The spectrum the search's result stored for a precursor
    (``LIBRARY_SPECTRUM_SCHEMA``, in m/z order): for a peptide-centric
    search the peaks it scored, b/y-annotated where they match."""
    from constellation.massspec.annotation.mzpaf import fragment_row_to_mzpaf
    from constellation.massspec.peptide.ions import ion_number

    root = Path(experiment_dir)
    peptide = _peptide_of(root, precursor_id, unit, acquisition_id)
    fragments = _read(root, "fragment", unit=unit, acquisition_id=acquisition_id, shared=True,
                      columns=["precursor_id", "ion_type", "position", "charge", "loss_id", "mz_theoretical",
                               "intensity_predicted", "annotation", "correlation", "quantified"],
                      where=ds.field("precursor_id") == precursor_id)
    if fragments is None or peptide is None:
        return LIBRARY_SPECTRUM_SCHEMA.empty_table()
    rows = fragments.to_pylist()
    length = len(peptide.get("sequence") or "")
    out = []
    for r in rows:
        typed = r["ion_type"] is not None and r["position"] is not None and length > 1
        out.append(
            {
                "mz": r["mz_theoretical"],
                "intensity": r["intensity_predicted"],
                "annotation": fragment_row_to_mzpaf(r, length=length) if typed else r["annotation"],
                "ion_type": _LETTERS[r["ion_type"]] if typed else None,
                "ion_number": ion_number(r["ion_type"], r["position"], length) if typed else None,
                "fragment_charge": r["charge"],
                "loss": r["loss_id"],
                "correlation": r["correlation"],
                "quantified": r["quantified"],
            }
        )
    out.sort(key=lambda r: r["mz"])
    return pa.Table.from_pylist(out, schema=LIBRARY_SPECTRUM_SCHEMA)


def searched_library_spectrum(
    experiment_dir: Path | str, precursor_id: int, *, unit: str
) -> pa.Table | None:
    """The searched library's spectrum of a precursor, or ``None`` (no
    library recorded, the file gone or replaced, not a library, or the
    precursor not in it).

    Columns ``mz`` and ``intensity`` (m/z order); the schema's metadata
    carries the entry's retention time (``x.massspec.rt_seconds``), its
    precursor m/z and the library path. The entry is found through the
    library's own index on peptide sequence and charge, and recognised by
    its content-derived precursor id, so a modification spelled with
    another mass precision (``C[+57.0214635]``) still matches.
    """
    from constellation.massspec.io.encyclopedia._codec import decode_f4, decode_f8
    from constellation.massspec.io.encyclopedia._modseq import parse_encyclopedia_modseq
    from constellation.massspec.library.ids import precursor_id as content_precursor_id

    root = Path(experiment_dir)
    recorded = (_unit_manifest(root, unit).get("inputs") or {}).get("library") or {}
    library = recorded.get("path")
    if not library or not Path(library).is_file():
        return None
    # Not re-hashed (a predicted library is gigabytes); a file of another
    # size is not the one that was searched.
    if recorded.get("size_bytes") is not None and Path(library).stat().st_size != recorded["size_bytes"]:
        return None
    peptide = _peptide_of(root, precursor_id, unit, None)
    if peptide is None or not peptide.get("sequence"):
        return None
    try:
        con = sqlite3.connect(f"file:{library}?mode=ro&immutable=1", uri=True)
        try:
            candidates = con.execute(
                "SELECT PeptideModSeq, PrecursorCharge, PrecursorMz, RTInSeconds, MassArray, "
                "MassEncodedLength, IntensityArray, IntensityEncodedLength FROM entries "
                "WHERE PeptideSeq = ? AND PrecursorCharge = ?",
                (peptide["sequence"], int(peptide["charge"])),
            ).fetchall()
        finally:
            con.close()
    except sqlite3.DatabaseError:
        return None
    for modseq, charge, mz, rt, masses, mass_len, intensities, int_len in candidates:
        try:
            if content_precursor_id(parse_encyclopedia_modseq(modseq), int(charge)) != precursor_id:
                continue
        except ValueError:
            continue
        mzs = decode_f8(masses, mass_len)
        ints = decode_f4(intensities, int_len).astype(np.float64)
        order = np.argsort(mzs, kind="stable")
        return pa.table(
            {"mz": mzs[order], "intensity": ints[order]},
            schema=pa.schema(
                [pa.field("mz", pa.float64()), pa.field("intensity", pa.float64())],
                metadata={
                    b"x.massspec.rt_seconds": str(rt).encode(),
                    b"x.massspec.precursor_mz": str(mz).encode(),
                    b"x.massspec.library": str(library).encode(),
                },
            ),
        )
    return None


TRACE_TARGET_SCHEMA = pa.schema(
    [
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("ion_type", pa.string()),
        pa.field("ion_number", pa.int16()),
        pa.field("fragment_charge", pa.int8()),
        #: The stored intensity (null for a theoretical target).
        pa.field("intensity", pa.float64()),
        #: Whether the engine summed this ion into the precursor's quantity.
        pa.field("quantified", pa.bool_()),
    ],
    metadata={b"schema_name": b"TraceTargets"},
)


def trace_targets(
    experiment_dir: Path | str,
    precursor_id: int,
    *,
    unit: str,
    acquisition_id: int,
    top_n: int = 8,
) -> pa.Table:
    """The fragment ions to trace for a precursor (``TRACE_TARGET_SCHEMA``):
    the ``top_n`` most intense b/y ions of the spectrum the search stored,
    each flagged when the engine quantified the precursor from it. Without
    a stored spectrum, the theoretical choice of
    :func:`massspec.spectra.annotate.theoretical_fragment_targets`."""
    from constellation.massspec.spectra.annotate import theoretical_fragment_targets

    root = Path(experiment_dir)
    stored = library_spectrum(root, precursor_id, unit=unit, acquisition_id=acquisition_id)
    stored = stored.filter(pc.is_in(stored.column("ion_type"), pa.array(["b", "y"])))
    if stored.num_rows:
        rows = sorted(stored.to_pylist(), key=lambda r: -r["intensity"])[:top_n]
        quant = _read(root, "fragment_quant", unit=unit, acquisition_id=acquisition_id, columns=["mz"],
                      where=ds.field("precursor_id") == precursor_id)
        quantified = np.sort(quant.column("mz").to_numpy()) if quant is not None and quant.num_rows else np.empty(0)

        def was_quantified(mz: float) -> bool:
            i = np.searchsorted(quantified, mz)
            return any(0 <= j < quantified.size and abs(quantified[j] - mz) < 1e-4 for j in (i - 1, i))

        return pa.Table.from_pylist(
            [
                {
                    "mz": r["mz"],
                    "label": r["annotation"],
                    "ion_type": r["ion_type"],
                    "ion_number": r["ion_number"],
                    "fragment_charge": r["fragment_charge"],
                    "intensity": r["intensity"],
                    "quantified": was_quantified(r["mz"]),
                }
                for r in rows
            ],
            schema=TRACE_TARGET_SCHEMA,
        )
    peptide = _peptide_of(root, precursor_id, unit, acquisition_id)
    if peptide is None or not peptide.get("modified_sequence"):
        return TRACE_TARGET_SCHEMA.empty_table()
    theoretical = theoretical_fragment_targets(peptide["modified_sequence"], top_n=top_n)
    return pa.Table.from_pylist(
        [
            {
                "mz": r["mz"],
                "label": r["annotation"],
                "ion_type": r["ion_type"],
                "ion_number": r["ion_number"],
                "fragment_charge": r["fragment_charge"],
                "intensity": None,
                "quantified": None,
            }
            for r in theoretical.to_pylist()
        ],
        schema=TRACE_TARGET_SCHEMA,
    )


# ── retention-time alignment ─────────────────────────────────────────────

#: The alignment pass each engine settles on (peptide-centric EncyclopeDIA,
#: then ScribeTwo).
_FINAL_PASSES = ("final", "main")


def rt_mapping(experiment_dir: Path | str, *, unit: str, acquisition_id: int) -> pa.Table:
    """A search's fit of library retention times to the run's: columns
    ``observed_rt`` (the fit's value) and ``library_rt``, both seconds,
    sorted and non-decreasing, so ``np.interp(t, observed_rt, library_rt)``
    puts a time of the run on the library's scale. Taken from the engine's
    final pass, without decoys. Empty when the search recorded no fit."""
    root = Path(experiment_dir)
    empty = pa.table({"observed_rt": pa.array([], pa.float64()), "library_rt": pa.array([], pa.float64())})
    table = _read(root, "rt_alignment", unit=unit, acquisition_id=acquisition_id,
                  columns=["pass", "library_rt", "fitted_rt", "is_decoy"])
    if table is None or table.num_rows == 0:
        return empty
    table = table.filter(pc.invert(pc.fill_null(table.column("is_decoy"), False)))
    passes = set(table.column("pass").to_pylist())
    chosen = next((p for p in _FINAL_PASSES if p in passes), sorted(passes)[-1] if passes else None)
    table = table.filter(pc.equal(table.column("pass"), chosen))
    library = table.column("library_rt").to_numpy(zero_copy_only=False)
    fitted = table.column("fitted_rt").to_numpy(zero_copy_only=False)
    keep = np.isfinite(library) & np.isfinite(fitted)
    library, fitted = library[keep], fitted[keep]
    order = np.argsort(library, kind="stable")
    library = library[order]
    fitted = np.maximum.accumulate(fitted[order])  # a fit of time to time keeps its order
    _, first = np.unique(fitted, return_index=True)
    return pa.table({"observed_rt": fitted[first], "library_rt": library[first]})


__all__ = [
    "IDENTIFICATION_SCHEMA",
    "LIBRARY_SPECTRUM_SCHEMA",
    "TRACE_TARGET_SCHEMA",
    "acquisition_bundles",
    "identifications",
    "library_spectrum",
    "precursor_detail",
    "rt_mapping",
    "search_units",
    "searched_library_spectrum",
    "trace_targets",
]
