"""The text files EncyclopeDIA writes beside a result, as tables.

A search writes more than its result library. Beside a peptide-centric
result ``run.dia.elib`` (captured inventory: ``outputs/encyclopedia.json``):

``run.dia.encyclopedia.txt`` / ``run.dia.encyclopedia.decoy.txt``
    Percolator's output: every precursor it scored, target or decoy, one
    row each, with score, q-value and posterior error probability. The
    id is ``<source>:<retention time, s>:<modseq>+<charge>``, a decoy's
    modseq spelled ``decoy<modseq>``; the first file ends with a
    one-field ``pi_0=...`` line. The result library's ``peptidescores``
    keeps only the precursors within the FDR threshold, and on those the
    two agree exactly (the lab's wide-window search, 2026-10-18: all
    31,895 of them, of 58,817 targets scored).
``run.dia.features.txt``
    Percolator's input: the features of each scored precursor. Its
    ``ScanNr`` is a running number of the jar's, not a scan.
``run.dia.encyclopedia.txt.{first,final}.rt_fit.txt``
    The retention-time alignment fitted after the first and the final
    scoring pass: per identified peptide (EncyclopeDIA modseq, no charge;
    a peptide-centric result has one precursor per peptide), library and
    observed time, where the warp puts it and the residual, in minutes.
``run.dia.encyclopedia.txt.{first,final}.ms{1,2}_fit.txt``
    The mass error along the run, measured, modelled and corrected, in
    the unit the header names (``Mass Error (AMU)``).

ScribeTwo keeps the same kinds of thing in ``run.dia.scribe2/``
(``_scribe2``): its ``.scribe2.{prealign,main}.txt`` hold the alignment
and both calibrations together, per identification with its charge (the
mass errors in no named unit: the search's tolerance units, Da for the
lab's ion-trap runs); ``.scribe2.txt`` / ``.decoy.txt`` its scores in
Percolator's layout; ``.scribe2.features.txt`` its features.

Times become seconds. Peptide and precursor ids are content-derived
(``massspec.library.ids``), null for decoys, as in ``_scribe2``. These
files name no run except in their ids, so a fit is filed under the run
of the result it sits beside.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as csv

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.io.schemas import cast_to_schema
from constellation.massspec.acquisitions import Acquisitions
from constellation.massspec.io.encyclopedia._project import _peptidoforms
from constellation.massspec.library.ids import (
    canonical_peptide_key,
    canonical_precursor_key,
    content_ids,
)
from constellation.massspec.search.schemas import (
    MASS_CALIBRATION,
    PRECURSOR_SCORE_TABLE,
    RT_ALIGNMENT,
)

_DECOY_PREFIX = "decoy"
_PASSES = ("first", "final")
_UNITS = {"amu": "Da", "da": "Da", "ppm": "ppm"}


class SidecarError(ValueError):
    """A sidecar is not laid out as captured."""


# ──────────────────────────────────────────────────────────────────────
# reading a tab-separated sidecar
# ──────────────────────────────────────────────────────────────────────


def read_tsv(path: Path, columns: Sequence[str] | None = None) -> pa.Table:
    """A sidecar as Arrow: ``true`` / ``false`` as booleans, ``?`` and an
    empty field as null,
    the results file's ``pi_0=...`` trailer skipped, any other row whose
    fields do not match the header an error."""
    skipped: list[object] = []

    def skip(row: object) -> str:
        if not str(getattr(row, "text", "")).strip().startswith("pi_0="):
            skipped.append(row)
        return "skip"

    try:
        table = csv.read_csv(
            path,
            read_options=csv.ReadOptions(block_size=1 << 26),
            parse_options=csv.ParseOptions(delimiter="\t", invalid_row_handler=skip),
            convert_options=csv.ConvertOptions(
                include_columns=list(columns) if columns else None,
                null_values=["", "?"],
                true_values=["true"],
                false_values=["false"],
                strings_can_be_null=True,
            ),
        )
    except (pa.ArrowInvalid, KeyError) as exc:
        raise SidecarError(f"{path.name}: {exc}") from exc
    if skipped:
        raise SidecarError(f"{path.name}: {len(skipped)} rows do not have the header's columns")
    return table


def parse_ids(ids: Sequence[str]) -> tuple[list[str], np.ndarray, list[str], np.ndarray, np.ndarray]:
    """``<source>:<rt s>:<modseq>+<charge>`` ids as ``(sources, rts,
    modseqs, charges, is_decoy)``, the decoy prefix taken off."""
    n = len(ids)
    sources: list[str] = []
    modseqs: list[str] = []
    rts = np.empty(n)
    charges = np.empty(n, dtype=np.int64)
    decoy = np.zeros(n, dtype=bool)
    for i, value in enumerate(ids):
        try:
            source, rt, rest = value.rsplit(":", 2)
            modseq, charge = rest.rsplit("+", 1)
            rts[i] = float(rt)
            charges[i] = int(charge)
        except ValueError:
            raise SidecarError(f"an id is not <source>:<rt>:<sequence>+<charge>: {value!r}") from None
        sources.append(source)
        decoy[i] = modseq.startswith(_DECOY_PREFIX)
        modseqs.append(modseq.removeprefix(_DECOY_PREFIX))
    return sources, rts, modseqs, charges, decoy


def _proforma(modseqs: Sequence[str], vocab: ModVocab) -> list[str]:
    _, proforma_of = _peptidoforms(sorted(set(modseqs)), vocab)
    return [proforma_of[m] for m in modseqs]


def _ids(keys: list[str | None]) -> pa.Array:
    """Content ids for the keys given, null where the key is ``None``."""
    present = [k for k in keys if k is not None]
    values = iter(content_ids(present).to_pylist()) if present else iter(())
    return pa.array([next(values) if k is not None else None for k in keys], type=pa.int64())


def precursor_ids(modified: Sequence[str], charges: np.ndarray, skip: np.ndarray, vocab: ModVocab) -> pa.Array:
    """Content-derived precursor ids of ProForma ``modified`` at ``charges``,
    null where ``skip`` (a decoy, which names no real precursor, or a row
    that has its id already)."""
    return _ids([
        None if s else canonical_precursor_key(m, int(z), vocab=vocab)
        for m, z, s in zip(modified, charges, skip, strict=True)
    ])  # fmt: skip


def peptide_ids(modified: Sequence[str], skip: Sequence[bool | None], vocab: ModVocab) -> pa.Array:
    """Content-derived peptide ids, null where ``skip`` (as for
    :func:`precursor_ids`)."""
    return _ids([
        None if s else canonical_peptide_key(m, vocab=vocab)
        for m, s in zip(modified, skip, strict=True)
    ])  # fmt: skip


# ──────────────────────────────────────────────────────────────────────
# where they are
# ──────────────────────────────────────────────────────────────────────


def peptide_centric_sidecars(result_library: Path | str) -> dict[str, Path]:
    """The sidecars beside a peptide-centric result that are there:
    ``scores`` / ``decoy_scores`` / ``features`` / ``rt_fit:<pass>`` /
    ``ms<level>_fit:<pass>`` -> path."""
    path = Path(result_library)
    stem = path.with_name(path.name.removesuffix(path.suffix))
    wanted = {
        "scores": f"{stem}.encyclopedia.txt",
        "decoy_scores": f"{stem}.encyclopedia.decoy.txt",
        "features": f"{stem}.features.txt",
    }
    for kind in ("rt_fit", "ms1_fit", "ms2_fit"):
        for name in _PASSES:
            wanted[f"{kind}:{name}"] = f"{stem}.encyclopedia.txt.{name}.{kind}.txt"
    return {key: Path(p) for key, p in wanted.items() if Path(p).is_file()}


# ──────────────────────────────────────────────────────────────────────
# scores, alignment, calibration, features
# ──────────────────────────────────────────────────────────────────────


def read_scored_precursors(
    paths: Sequence[Path], *, acquisitions: Acquisitions, engine: str, vocab: ModVocab = UNIMOD
) -> pa.Table:
    """Percolator-layout scores (targets and decoys) as
    ``PRECURSOR_SCORE_TABLE`` rows, ``precursor_id`` left null for the
    caller to fill. A ProForma that two of the jar's spellings share (an
    N-terminal modification written two ways) keeps its best-scored row."""
    parts = [read_tsv(p, ["PSMId", "score", "q-value", "posterior_error_prob"]) for p in paths]
    table = pa.concat_tables(parts) if parts else pa.table({"PSMId": pa.array([], pa.string())})
    sources, _, modseqs, charges, decoy = parse_ids(table.column("PSMId").to_pylist())
    by_source = {s: acquisitions.id_for_source(s) for s in set(sources)}
    out = pa.table(
        {
            "precursor_id": pa.nulls(table.num_rows, type=pa.int64()),
            "acquisition_id": pa.array([by_source[s] for s in sources], type=pa.int64()),
            "modified_sequence": pa.array(_proforma(modseqs, vocab), type=pa.string()),
            "charge": pa.array(charges, type=pa.int32()),
            "score": table.column("score").cast(pa.float64()),
            "qvalue": table.column("q-value").cast(pa.float64()),
            "pep": table.column("posterior_error_prob").cast(pa.float64()),
            "is_decoy": pa.array(decoy, type=pa.bool_()),
            "engine": pa.array([engine] * table.num_rows, type=pa.string()),
        }
    )
    keys = ["modified_sequence", "charge", "acquisition_id", "engine", "is_decoy"]
    out = out.sort_by([("qvalue", "ascending")])
    first = (
        out.append_column("_row", pa.array(np.arange(out.num_rows)))
        .group_by(keys, use_threads=False)
        .aggregate([("_row", "min")])
        .column("_row_min")
        .to_numpy()
    )
    return cast_to_schema(out.take(pa.array(np.sort(first))), PRECURSOR_SCORE_TABLE)


def _alignment(
    *, engine: str, pass_names: Sequence[str], modseqs: Sequence[str], charges: Sequence[int | None],
    library: np.ndarray, observed: np.ndarray, fitted: np.ndarray, residual: np.ndarray,
    within: pa.Array, decoy: Sequence[bool | None], acquisition_id: int | None, vocab: ModVocab,
) -> pa.Table:  # fmt: skip
    modified = _proforma(modseqs, vocab)
    n = len(modified)
    return pa.table(
        {
            "acquisition_id": pa.array([acquisition_id] * n, type=pa.int64()),
            "engine": pa.array([engine] * n, type=pa.string()),
            "pass": pa.array(list(pass_names), type=pa.string()),
            "modified_sequence": pa.array(modified, type=pa.string()),
            "peptide_id": peptide_ids(modified, decoy, vocab),
            "charge": pa.array(list(charges), type=pa.int32()),
            "library_rt": pa.array(library * 60.0, type=pa.float64()),
            "observed_rt": pa.array(observed * 60.0, type=pa.float64()),
            "fitted_rt": pa.array(fitted * 60.0, type=pa.float64()),
            "residual": pa.array(residual * 60.0, type=pa.float64()),
            "within_95": within.cast(pa.bool_()),
            "is_decoy": pa.array(list(decoy), type=pa.bool_()),
        },
        schema=RT_ALIGNMENT,
    )


def _f(table: pa.Table, column: str) -> np.ndarray:
    return table.column(column).cast(pa.float64()).to_numpy(zero_copy_only=False)


def read_rt_fits(
    files: dict[str, Path], *, acquisition_id: int | None, engine: str, vocab: ModVocab = UNIMOD
) -> pa.Table:
    """A peptide-centric search's ``rt_fit`` files (``rt_fit:<pass>`` ->
    path) as ``RT_ALIGNMENT``."""
    parts = []
    for name in _PASSES:
        path = files.get(f"rt_fit:{name}")
        if path is None:
            continue
        t = read_tsv(path)
        parts.append(
            _alignment(
                engine=engine, pass_names=[name] * t.num_rows,
                modseqs=t.column("sequence").cast(pa.string()).to_pylist(),
                charges=[None] * t.num_rows, library=_f(t, "library"), observed=_f(t, "actual"),
                fitted=_f(t, "warpToactual"), residual=_f(t, "delta"), within=t.column("within95p"),
                decoy=t.column("isDecoy").cast(pa.bool_()).to_pylist(), acquisition_id=acquisition_id,
                vocab=vocab,
            )
        )  # fmt: skip
    return pa.concat_tables(parts) if parts else RT_ALIGNMENT.empty_table()


def _unit_of(header: str) -> str:
    inside = header.partition("(")[2].partition(")")[0].strip().lower()
    unit = _UNITS.get(inside)
    if unit is None:
        raise SidecarError(f"a mass error in an unknown unit: {header!r}")
    return unit


def read_mass_fits(files: dict[str, Path], *, acquisition_id: int | None, engine: str) -> pa.Table:
    """A peptide-centric search's ``ms1_fit`` / ``ms2_fit`` files as
    ``MASS_CALIBRATION`` (no peptide: the jar writes ``?``)."""
    parts = []
    for level in (1, 2):
        for name in _PASSES:
            path = files.get(f"ms{level}_fit:{name}")
            if path is None:
                continue
            t = read_tsv(path)
            error = next(c for c in t.column_names if c.startswith("Mass Error"))
            n = t.num_rows
            parts.append(
                pa.table(
                    {
                        "acquisition_id": pa.array([acquisition_id] * n, type=pa.int64()),
                        "engine": pa.array([engine] * n, type=pa.string()),
                        "pass": pa.array([name] * n, type=pa.string()),
                        "ms_level": pa.array([level] * n, type=pa.int8()),
                        "rt": pa.array(_f(t, "Retention Time (Min)") * 60.0, type=pa.float64()),
                        "mass_error": pa.array(_f(t, error), type=pa.float64()),
                        "predicted_mass_error": pa.array(_f(t, "predictedMassError"), type=pa.float64()),
                        "corrected_mass_error": pa.array(_f(t, "correctedMassError"), type=pa.float64()),
                        "unit": pa.array([_unit_of(error)] * n, type=pa.string()),
                        "within_95": t.column("within95p").cast(pa.bool_()),
                    }
                )
            )
    return cast_to_schema(pa.concat_tables(parts), MASS_CALIBRATION) if parts else MASS_CALIBRATION.empty_table()


def read_scribe2_fits(
    reports_dir: Path, *, acquisition_id: int | None, units: tuple[str, str], vocab: ModVocab = UNIMOD
) -> tuple[pa.Table, pa.Table]:
    """ScribeTwo's ``.scribe2.{prealign,main}.txt`` as ``(RT_ALIGNMENT,
    MASS_CALIBRATION)``. ``units`` are the MS1 and MS2 mass-error units
    (the search's precursor and fragment tolerance units)."""
    from constellation.massspec.io.encyclopedia._project import SCRIBETWO_ENGINE

    stem = reports_dir.name
    alignment, calibration = [], []
    for name in ("prealign", "main"):
        path = reports_dir / f"{stem}.{name}.txt"
        if not path.is_file():
            continue
        t = read_tsv(path)
        modseqs = t.column("sequence").cast(pa.string()).to_pylist()
        charges = t.column("charge").cast(pa.int64()).to_pylist()
        passes = t.column("pass").cast(pa.string()).to_pylist()
        alignment.append(
            _alignment(
                engine=SCRIBETWO_ENGINE, pass_names=passes, modseqs=modseqs, charges=charges,
                library=_f(t, "rtLibrary"), observed=_f(t, "rtActual"), fitted=_f(t, "rtPredicted"),
                residual=_f(t, "rtDelta"), within=t.column("rtWithin95p"), decoy=[False] * t.num_rows,
                acquisition_id=acquisition_id, vocab=vocab,
            )
        )  # fmt: skip
        modified = _proforma(modseqs, vocab)
        ids = peptide_ids(modified, [False] * t.num_rows, vocab)
        for level, unit in ((1, units[0]), (2, units[1])):
            p = f"ms{level}"
            # An identification with no measurement at this level (empty
            # fields in ScribeTwo's report) has no row for it.
            measured = t.column(f"{p}MassError").is_valid().to_numpy(zero_copy_only=False)
            calibration.append(
                cast_to_schema(
                    pa.table(
                        {
                            "acquisition_id": pa.array([acquisition_id] * t.num_rows, type=pa.int64()),
                            "engine": pa.array([SCRIBETWO_ENGINE] * t.num_rows, type=pa.string()),
                            "pass": pa.array(passes, type=pa.string()),
                            "ms_level": pa.array([level] * t.num_rows, type=pa.int8()),
                            "rt": pa.array(_f(t, f"{p}RT") * 60.0, type=pa.float64()),
                            "mass_error": pa.array(_f(t, f"{p}MassError"), type=pa.float64()),
                            "predicted_mass_error": pa.array(_f(t, f"{p}PredictedMassError"), type=pa.float64()),
                            "corrected_mass_error": pa.array(_f(t, f"{p}Delta"), type=pa.float64()),
                            "unit": pa.array([unit] * t.num_rows, type=pa.string()),
                            "within_95": t.column(f"{p}Within95p").cast(pa.bool_()),
                            "modified_sequence": pa.array(modified, type=pa.string()),
                            "peptide_id": ids,
                            "charge": pa.array(charges, type=pa.int32()),
                            "is_decoy": pa.array([False] * t.num_rows, type=pa.bool_()),
                        }
                    ).filter(pa.array(measured)),
                    MASS_CALIBRATION,
                )
            )
    return (
        pa.concat_tables(alignment) if alignment else RT_ALIGNMENT.empty_table(),
        pa.concat_tables(calibration) if calibration else MASS_CALIBRATION.empty_table(),
    )


#: Feature-file columns that are not features.
_NOT_FEATURES = {"id", "Label", "ScanNr", "sequence", "Peptide", "peptide", "Proteins", "proteinIds"}


def read_features(
    path: Path, *, acquisitions: Acquisitions, engine: str, vocab: ModVocab = UNIMOD
) -> pa.Table:
    """Percolator's input as the jar wrote it: one row per scored match,
    key columns first (run, modified sequence, charge, retention time in
    seconds, decoy flag, content-derived precursor id, the jar's
    ``ScanNr``), then every feature under its own name as float64, then
    the proteins. No fixed schema: the features are the engine's
    (35 columns from a peptide-centric search, 59 from ScribeTwo)."""
    t = read_tsv(path)
    sources, rts, modseqs, charges, _ = parse_ids(t.column("id").cast(pa.string()).to_pylist())
    decoy = t.column("Label").cast(pa.int64()).to_numpy(zero_copy_only=False) < 0
    modified = _proforma(modseqs, vocab)
    by_source = {s: acquisitions.id_for_source(s) for s in set(sources)}
    columns: dict[str, pa.Array] = {
        "acquisition_id": pa.array([by_source[s] for s in sources], type=pa.int64()),
        "engine": pa.array([engine] * t.num_rows, type=pa.string()),
        "modified_sequence": pa.array(modified, type=pa.string()),
        "charge": pa.array(charges, type=pa.int32()),
        "rt": pa.array(rts, type=pa.float64()),
        "is_decoy": pa.array(decoy, type=pa.bool_()),
        "precursor_id": precursor_ids(modified, charges, decoy, vocab),
        "scan_nr": t.column("ScanNr").cast(pa.int64()),
    }
    for name in t.column_names:
        if name not in _NOT_FEATURES and name not in columns:
            columns[name] = t.column(name).cast(pa.float64())
    for name in ("Proteins", "proteinIds"):
        if name in t.column_names:
            columns["proteins"] = t.column(name).cast(pa.string())
            break
    return pa.table(columns)


__all__ = [
    "SidecarError",
    "parse_ids",
    "peptide_centric_sidecars",
    "read_features",
    "read_mass_fits",
    "read_rt_fits",
    "read_scored_precursors",
    "read_scribe2_fits",
    "read_tsv",
]
