"""MaxQuant ``combined/txt/`` reader → :class:`Search` (PSM table).

v1 ingests ``msms.txt`` (one row per peptide-spectrum match) into
``PSM_TABLE`` and captures ``parameters.txt`` as provenance. The
``msms.txt`` location varies — at the export root (Zolg/Wilhelm style) or
under a ``txt/`` subdirectory (Gessulat style) — so the reader locates it
rather than assuming a fixed path.

Parsing is pyarrow-native (no pandas): the big ``;``-delimited blob
columns (``Masses``, ``Mass Deviations``, ``Matches``, ``Intensities``)
are skipped via ``include_columns``. Modseq → ProForma translation is
batched over the distinct ``Modified sequence`` values (far fewer than
PSM rows) and mapped back with ``pc.index_in`` / ``pc.take``.

evidence.txt → Quant, peptides/proteinGroups rollups, and fragment
annotations are deferred (see the module CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.io.schemas import cast_to_schema
from constellation.core.sequence.proforma import format_proforma
from constellation.massspec.acquisitions import Acquisitions
from constellation.massspec.search.schemas import (
    PEPTIDE_SCORE_TABLE,
    PROTEIN_SCORE_TABLE,
    PSM_TABLE,
)
from constellation.massspec.search.search import Search

from constellation.massspec.io.maxquant._modseq import (
    MaxQuantModResolutionError,
    parse_maxquant_modseq,
)
from constellation.massspec.io.maxquant._params import (
    parse_fixed_modifications,
    parse_parameters_txt,
)

# MaxQuant source column → (PSM column name, arrow read type). Columns
# copied straight through to PSM_TABLE.
_PASSTHROUGH: dict[str, tuple[str, pa.DataType]] = {
    "Raw file": ("raw_file", pa.string()),
    "Scan number": ("scan", pa.int32()),
    "Sequence": ("sequence", pa.string()),
    "Proteins": ("proteins", pa.string()),
    "Charge": ("charge", pa.int8()),
    "Fragmentation": ("fragmentation", pa.string()),
    "Mass analyzer": ("mass_analyzer", pa.string()),
    "Type": ("psm_type", pa.string()),
    "m/z": ("mz", pa.float64()),
    "Mass": ("mass", pa.float64()),
    "Mass Error [ppm]": ("mass_error_ppm", pa.float64()),
    "PEP": ("pep", pa.float64()),
    "Score": ("score", pa.float64()),
    "Delta score": ("delta_score", pa.float64()),
    "Precursor Full ScanNumber": ("precursor_scan", pa.int32()),
    "id": ("psm_id", pa.int64()),
    "Peptide ID": ("peptide_id", pa.int64()),
    "Mod. peptide ID": ("mod_peptide_id", pa.int64()),
    "Evidence ID": ("evidence_id", pa.int64()),
}

# Columns read only to *derive* PSM columns (not copied verbatim).
_DERIVE_SOURCE: dict[str, pa.DataType] = {
    "Modified sequence": pa.string(),
    "Retention time": pa.float64(),
    "Reverse": pa.string(),
}

_COLUMN_TYPES: dict[str, pa.DataType] = {
    **{src: typ for src, (_, typ) in _PASSTHROUGH.items()},
    **_DERIVE_SOURCE,
}

# Empty strings and MaxQuant's "NaN" sentinels read as null.
_NULL_VALUES = ["", "NaN", "nan", "NA"]


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────


def read_maxquant_search(
    path: str | Path,
    *,
    acquisitions: Acquisitions | None = None,
    vocab: ModVocab = UNIMOD,
    engine: str = "maxquant",
) -> Search:
    """Read a MaxQuant ``combined/txt/`` export into a :class:`Search`.

    Parameters
    ----------
    path
        The export directory. ``msms.txt`` is located at the root, under
        ``txt/``, or one subdirectory deep.
    acquisitions
        Optional pre-existing :class:`Acquisitions`. When given, each PSM's
        ``acquisition_id`` is resolved by matching its ``Raw file`` to the
        stem of an acquisition's ``source_file`` (unmatched → null,
        recorded in ``metadata_extras["x.maxquant.unmatched_raw_files"]``),
        and the returned ``Search`` carries *these* acquisitions. When
        ``None`` (default), a minimal ``Acquisitions`` is synthesised from
        the distinct ``Raw file`` values (``source_kind="maxquant"``).
    vocab
        Modification vocabulary for modseq resolution (default UNIMOD).
    engine
        Value stamped into ``PSM_TABLE.engine`` (default ``"maxquant"``).
    """
    root = Path(path)
    msms_path = _locate(root, "msms.txt")
    if msms_path is None:
        raise FileNotFoundError(f"no msms.txt found under {root}")
    params_path = _locate(root, "parameters.txt")
    params = parse_parameters_txt(params_path) if params_path is not None else {}
    fixed_mods = parse_fixed_modifications(params.get("Fixed modifications"))

    metadata: dict[str, object] = {
        f"x.maxquant.parameters.{key}": value for key, value in params.items()
    }
    metadata["x.maxquant.msms_path"] = str(msms_path)
    if params_path is None:
        metadata["x.maxquant.parameters_missing"] = True

    raw = _read_msms_table(msms_path)
    if raw.num_rows == 0:
        # Header-only (corrupt-raw acquisitions) or empty file → 0 PSMs.
        acq = acquisitions if acquisitions is not None else Acquisitions.empty()
        return _make_search(acq, PSM_TABLE.empty_table(), metadata)

    psms, acq, unresolved, unmatched = _build_psms(
        raw,
        fixed_mods=fixed_mods,
        vocab=vocab,
        acquisitions=acquisitions,
        engine=engine,
    )
    if unresolved:
        metadata["x.maxquant.unresolved_mods"] = sorted(set(unresolved))
    if unmatched:
        metadata["x.maxquant.unmatched_raw_files"] = sorted(set(unmatched))
    return _make_search(acq, psms, metadata)


# ──────────────────────────────────────────────────────────────────────
# Locating + reading the TSV
# ──────────────────────────────────────────────────────────────────────


def _locate(root: Path, filename: str) -> Path | None:
    """Find ``filename`` at the root, under ``txt/``, or one level deep.

    Bounded and predictable — does not recurse arbitrarily and never
    follows ``..`` (filesystem iteration cannot yield it); the
    path-traversal entry MaxQuant zips sometimes ship (``../mqpar.xml``)
    is therefore a non-issue on the extracted tree.
    """
    candidates = [
        root / filename,
        root / "txt" / filename,
        root / "combined" / "txt" / filename,
        root / "combined" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if root.is_dir():
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for candidate in (sub / filename, sub / "txt" / filename):
                if candidate.is_file():
                    return candidate
    return None


def _read_header(path: Path) -> list[str] | None:
    """Return the tab-split header of the first non-empty line, or None."""
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                if stripped:
                    return stripped.split("\t")
    except OSError:
        return None
    return None


def _read_msms_table(path: Path) -> pa.Table:
    """Read the needed ``msms.txt`` columns (typed) via ``pyarrow.csv``.

    Selects only the columns the PSM table needs (skipping the big
    ``;``-delimited blob columns). Tolerates MaxQuant version drift by
    intersecting the wanted columns with those actually present. A
    header-only file yields a 0-row table.
    """
    header = _read_header(path)
    if header is None:
        return pa.table({})  # empty / unreadable → 0 rows, handled upstream
    wanted = set(_PASSTHROUGH) | set(_DERIVE_SOURCE)
    present = [name for name in header if name in wanted]
    if not present:
        return pa.table({})
    column_types = {name: _COLUMN_TYPES[name] for name in present}
    return pacsv.read_csv(
        str(path),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            include_columns=present,
            column_types=column_types,
            null_values=_NULL_VALUES,
            strings_can_be_null=True,
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Building the PSM table
# ──────────────────────────────────────────────────────────────────────


def _build_psms(
    raw: pa.Table,
    *,
    fixed_mods: list[tuple[str, str]],
    vocab: ModVocab,
    acquisitions: Acquisitions | None,
    engine: str,
) -> tuple[pa.Table, Acquisitions, list[str], list[str]]:
    n = raw.num_rows
    cols: dict[str, pa.Array | pa.ChunkedArray] = {}

    for src, (dst, _typ) in _PASSTHROUGH.items():
        if src in raw.column_names:
            cols[dst] = raw.column(src)

    cols["modified_sequence"], unresolved = _build_modified_sequences(
        raw, fixed_mods=fixed_mods, vocab=vocab
    )

    if "Retention time" in raw.column_names:
        cols["retention_time_s"] = pc.multiply(raw.column("Retention time"), 60.0)

    cols["is_decoy"] = (
        pc.fill_null(pc.equal(raw.column("Reverse"), "+"), False)
        if "Reverse" in raw.column_names
        else pa.array([False] * n, type=pa.bool_())
    )
    cols["is_contaminant"] = (
        pc.fill_null(pc.match_substring(raw.column("Proteins"), "CON__"), False)
        if "Proteins" in raw.column_names
        else pa.array([False] * n, type=pa.bool_())
    )

    acq, acquisition_id, unmatched = _resolve_acquisitions(raw, acquisitions)
    cols["acquisition_id"] = acquisition_id
    cols["engine"] = pa.array([engine] * n, type=pa.string())

    table = pa.table(cols)
    return cast_to_schema(table, PSM_TABLE), acq, unresolved, unmatched


def _build_modified_sequences(
    raw: pa.Table,
    *,
    fixed_mods: list[tuple[str, str]],
    vocab: ModVocab,
) -> tuple[pa.Array | pa.ChunkedArray, list[str]]:
    """Map ``Modified sequence`` → ProForma 2.0, batched over distinct values.

    Per-unique-modseq Python work (the only unavoidable Python in the
    parse) scales with the modform count, not the PSM count; the map-back
    onto every row is vectorised via ``pc.index_in`` + ``pc.take``. An
    unresolved modseq maps to null (the PSM is kept) and is recorded for
    the caller to surface in ``metadata_extras``.
    """
    n = raw.num_rows
    if "Modified sequence" not in raw.column_names:
        return pa.nulls(n, pa.string()), []

    col = raw.column("Modified sequence")
    uniques = pc.unique(col)
    unique_values = uniques.to_pylist()

    proforma_by_unique: list[str | None] = []
    unresolved: list[str] = []
    for modseq in unique_values:
        if modseq is None:
            proforma_by_unique.append(None)
            continue
        try:
            peptidoform = parse_maxquant_modseq(
                modseq, fixed_mods=fixed_mods, vocab=vocab
            )
            proforma_by_unique.append(format_proforma(peptidoform))
        except MaxQuantModResolutionError:
            proforma_by_unique.append(None)
            unresolved.append(modseq)

    proforma_array = pa.array(proforma_by_unique, type=pa.string())
    indices = pc.index_in(col, value_set=uniques)
    return pc.take(proforma_array, indices), unresolved


def _resolve_acquisitions(
    raw: pa.Table,
    acquisitions: Acquisitions | None,
) -> tuple[Acquisitions, pa.Array | pa.ChunkedArray, list[str]]:
    """Map each PSM's ``Raw file`` to an ``acquisition_id``.

    Returns ``(acquisitions, acquisition_id_column, unmatched_raw_files)``.
    With a provided ``Acquisitions``, raw files match on
    ``Path(source_file).stem``; otherwise a minimal one is synthesised
    from the distinct raw files.
    """
    n = raw.num_rows
    if "Raw file" not in raw.column_names:
        acq = acquisitions if acquisitions is not None else Acquisitions.empty()
        return acq, pa.nulls(n, pa.int64()), []

    rf_col = raw.column("Raw file")
    rf_uniques = pc.unique(rf_col)
    rf_values = rf_uniques.to_pylist()
    unmatched: list[str] = []

    if acquisitions is not None:
        stem_to_id = {
            Path(source_file).stem: acq_id
            for source_file, acq_id in zip(
                acquisitions.table.column("source_file").to_pylist(),
                acquisitions.table.column("acquisition_id").to_pylist(),
                strict=True,
            )
        }
        id_for_unique: list[int | None] = []
        for raw_file in rf_values:
            acq_id = stem_to_id.get(raw_file) if raw_file is not None else None
            if raw_file is not None and acq_id is None:
                unmatched.append(raw_file)
            id_for_unique.append(acq_id)
        acq = acquisitions
    else:
        sorted_rf = sorted(rf for rf in rf_values if rf is not None)
        raw_to_id = {raw_file: idx for idx, raw_file in enumerate(sorted_rf)}
        acq = Acquisitions.from_records(
            [
                {
                    "acquisition_id": raw_to_id[raw_file],
                    "source_file": raw_file,
                    "source_kind": "maxquant",
                    "acquisition_datetime": None,
                }
                for raw_file in sorted_rf
            ]
        )
        id_for_unique = [
            raw_to_id.get(raw_file) if raw_file is not None else None
            for raw_file in rf_values
        ]

    id_array = pa.array(id_for_unique, type=pa.int64())
    indices = pc.index_in(rf_col, value_set=rf_uniques)
    return acq, pc.take(id_array, indices), unmatched


def _make_search(
    acq: Acquisitions, psms: pa.Table, metadata: dict[str, object]
) -> Search:
    return Search(
        acquisitions=acq,
        peptide_scores=PEPTIDE_SCORE_TABLE.empty_table(),
        protein_scores=PROTEIN_SCORE_TABLE.empty_table(),
        psms=psms,
        metadata_extras=metadata,
    )


__all__ = ["read_maxquant_search"]
