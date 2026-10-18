"""ScribeTwo's peptide-spectrum matches, read from the reports beside its result.

A ScribeTwo search of ``run.dia`` writes its result library beside the
input (``run.dia.dlib``) and its reports into ``run.dia.scribe2/`` (the
captured inventory is ``data/_raw/encyclopedia/<version>/outputs/
scribetwo.json``). The result library holds one entry per identified
precursor and no scan; two of the reports hold the matches:

``run.dia.scribe2.features.txt``
    Every peptide-spectrum match ScribeTwo scored, target or decoy, one
    row each, in Percolator's input layout. ``id`` is
    ``<source file>:<retention time, s>:<modified sequence>+<charge>``
    (a decoy's sequence is spelled ``decoy<sequence>``); ``Label`` is 1
    for a target and -1 for a decoy; ``ScanNr`` is the spectrum's index
    in the spectra file, which for a single run is the raw file's scan
    number; ``ExpMass`` is the precursor's observed m/z, beside the
    identified peptide's ``precursorMz`` and ``precursorMass``. Checked
    on the lab's DDA run (88,490 matches): every match's retention time
    is its scan's, and ``ExpMass`` is within 0.01 Da of ``precursorMz``
    at the median, not at the isolation target. On a DIA run (82,254
    matches) the scan's isolation window always holds the precursor, but
    the retention time in the id is the scan's for only 84% of matches
    and minutes away for some others (perhaps the peptide's apex; not
    established), so ``retention_time_s`` is the id's and ``scan`` is
    ``ScanNr``, each as ScribeTwo wrote it.
``run.dia.scribe2.txt`` / ``run.dia.scribe2.decoy.txt``
    The best match of each precursor after ScribeTwo's re-scoring
    (JavaPot): score, q-value and posterior error probability, under the
    same ``id``. The first file ends with a one-field trailer line,
    ``pi_0=...``.

:func:`read_scribe2_psms` gives one ``PSM_TABLE`` row per scored match.
The best match of each precursor (``psm_type`` ``"best"``) carries its
score, q-value and posterior error probability; the precursor's other
matches (``"other"``: the same precursor matched in other scans) carry
none, since ScribeTwo reports none for them. ``precursor_id`` is the
content-derived id of the target's precursor, the one the precursor
tables of a content-keyed result use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.io.schemas import cast_to_schema
from constellation.massspec.acquisitions import Acquisitions
from constellation.massspec.io.encyclopedia._project import SCRIBETWO_ENGINE, _peptidoforms
from constellation.massspec.library.ids import canonical_precursor_key, content_ids
from constellation.massspec.search.schemas import PSM_TABLE

#: ScribeTwo's report directory and files, after ``<input>``.
REPORTS_SUFFIX = ".scribe2"
_FEATURES = ".scribe2.features.txt"
_RESULTS = (".scribe2.txt", ".scribe2.decoy.txt")
_DECOY_PREFIX = "decoy"


class Scribe2ReportError(ValueError):
    """ScribeTwo's reports are missing or not as captured."""


def scribe2_reports(result_library: Path | str) -> Path | None:
    """The report directory of a ScribeTwo result (``run.dia.dlib`` ->
    ``run.dia.scribe2/``), or ``None`` when it is not there."""
    path = Path(result_library)
    directory = path.with_name(path.name.removesuffix(path.suffix) + REPORTS_SUFFIX)
    return directory if (directory / (directory.name + ".features.txt")).is_file() else None


def _read_tsv(path: Path, columns: list[str]) -> pa.Table:
    skipped: list[object] = []

    def skip(row: object) -> str:
        # The results file ends with a one-field trailer (``pi_0=null`` on
        # the lab's DDA run); anything else malformed is an error.
        if not str(getattr(row, "text", "")).strip().startswith("pi_0="):
            skipped.append(row)
        return "skip"

    try:
        table = csv.read_csv(
            path,
            read_options=csv.ReadOptions(block_size=1 << 26),
            parse_options=csv.ParseOptions(delimiter="\t", invalid_row_handler=skip),
            convert_options=csv.ConvertOptions(include_columns=columns),
        )
    except (pa.ArrowInvalid, KeyError) as exc:
        raise Scribe2ReportError(f"{path.name}: {exc}") from exc
    if skipped:
        raise Scribe2ReportError(f"{path.name}: {len(skipped)} rows do not have the header's columns")
    return table


def read_scribe2_psms(
    reports_dir: Path | str,
    *,
    acquisitions: Acquisitions,
    vocab: ModVocab = UNIMOD,
) -> pa.Table:
    """Every match ScribeTwo scored, as ``PSM_TABLE`` (module docstring).

    ``reports_dir`` is the ``<input>.scribe2/`` directory. Each match's
    source file is bound to ``acquisitions`` by name, as the result
    library's rows are.
    """
    reports_dir = Path(reports_dir)
    stem = reports_dir.name.removesuffix(REPORTS_SUFFIX)
    features = _read_tsv(
        reports_dir / f"{stem}{_FEATURES}",
        ["id", "Label", "ScanNr", "ExpMass", "precursorMz", "precursorMass", "Proteins"],
    )
    scored: dict[str, tuple[float, float, float]] = {}
    for suffix in _RESULTS:
        results = _read_tsv(
            reports_dir / f"{stem}{suffix}", ["PSMId", "score", "q-value", "posterior_error_prob"]
        )
        for psm, score, q, pep in zip(
            results.column("PSMId").to_pylist(),
            results.column("score").to_pylist(),
            results.column("q-value").to_pylist(),
            results.column("posterior_error_prob").to_pylist(),
            strict=True,
        ):
            scored[psm] = (score, q, pep)

    ids = features.column("id").to_pylist()
    n = len(ids)
    sources: list[str] = []
    rts = np.empty(n)
    spelled: list[str] = []
    charges = np.empty(n, dtype=np.int64)
    for i, psm in enumerate(ids):
        try:
            source, rt, rest = psm.rsplit(":", 2)
            modseq, charge = rest.rsplit("+", 1)
            rts[i] = float(rt)
            charges[i] = int(charge)
        except ValueError:
            raise Scribe2ReportError(f"a match id is not <source>:<rt>:<sequence>+<charge>: {psm!r}") from None
        sources.append(source)
        spelled.append(modseq.removeprefix(_DECOY_PREFIX))
    is_decoy = features.column("Label").to_numpy(zero_copy_only=False) < 0

    forms, proforma_of = _peptidoforms(sorted(set(spelled)), vocab)
    modified = [proforma_of[m] for m in spelled]
    plain = [forms[m].sequence for m in spelled]

    # Content-derived precursor ids, for targets.
    precursor_id = np.zeros(n, dtype=np.int64)
    targets = np.flatnonzero(~is_decoy)
    keys = [canonical_precursor_key(modified[i], int(charges[i]), vocab=vocab) for i in targets]
    precursor_id[targets] = content_ids(keys).to_numpy(zero_copy_only=False)

    by_source = {s: acquisitions.id_for_source(s) for s in set(sources)}
    best = [scored.get(psm) for psm in ids]
    observed = features.column("ExpMass").to_numpy(zero_copy_only=False).astype(np.float64)
    theoretical = features.column("precursorMz").to_numpy(zero_copy_only=False).astype(np.float64)
    proteins = features.column("Proteins")
    table = pa.table(
        {
            "psm_id": pa.array(np.arange(n), type=pa.int64()),
            "raw_file": pa.array(sources, type=pa.string()),
            "acquisition_id": pa.array([by_source[s] for s in sources], type=pa.int64()),
            "scan": features.column("ScanNr").cast(pa.int32()),
            "sequence": pa.array(plain, type=pa.string()),
            "modified_sequence": pa.array(modified, type=pa.string()),
            "precursor_id": pa.array(precursor_id, type=pa.int64(), mask=is_decoy),
            "proteins": proteins.cast(pa.string()),
            "charge": pa.array(charges, type=pa.int8()),
            "mz": pa.array(observed, type=pa.float64()),
            "mass": features.column("precursorMass").cast(pa.float64()),
            "mass_error_ppm": pa.array((observed - theoretical) / theoretical * 1e6, type=pa.float64()),
            "retention_time_s": pa.array(rts, type=pa.float64()),
            "psm_type": pa.array(["best" if b is not None else "other" for b in best], type=pa.string()),
            "score": pa.array([b[0] if b else None for b in best], type=pa.float64()),
            "pep": pa.array([b[2] if b else None for b in best], type=pa.float64()),
            "qvalue": pa.array([b[1] if b else None for b in best], type=pa.float64()),
            "is_decoy": pa.array(is_decoy, type=pa.bool_()),
            "is_contaminant": pc.fill_null(
                pc.match_substring(proteins.cast(pa.string()), "CON__"), False
            ),
            "engine": pa.array([SCRIBETWO_ENGINE] * n, type=pa.string()),
        }
    )
    missing = len(scored) - sum(b is not None for b in best)
    if missing:
        raise Scribe2ReportError(
            f"{missing} scored matches in {stem}{_RESULTS[0]} / {_RESULTS[1]} are not in "
            f"{stem}{_FEATURES}"
        )
    return cast_to_schema(table, PSM_TABLE)


__all__ = ["REPORTS_SUFFIX", "Scribe2ReportError", "read_scribe2_psms", "scribe2_reports"]
