"""EncyclopeDIA ``.dlib`` / ``.elib`` reader — produces ``(Library, Quant?, Search?)``.

DLIB and ELIB share one SQLite container schema; the ``Quant`` and
``Search`` halves get built only when the corresponding tables /
columns are populated. The reader is one piece of code that returns
all three projections; ``adapters.py`` slices out whichever one a
caller asked for via the Library / Quant / Search Reader Protocols.

The work is done in two layers underneath: :mod:`._mirror` turns the
file's tables into Arrow as they are, and :mod:`._project` says what
they mean. This module is the stable entry point over them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pyarrow as pa

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.massspec.acquisitions import Acquisitions
from constellation.massspec.io.encyclopedia._project import (
    SCRIBETWO_ENGINE,
    _acquisitions_for,
    engine_of,
    project_library_file,
)
from constellation.massspec.io.encyclopedia._scribe2 import read_scribe2_psms, scribe2_reports
from constellation.massspec.io.encyclopedia._sidecar import (
    peptide_centric_sidecars,
    read_features,
    read_mass_fits,
    read_rt_fits,
    read_scored_precursors,
    read_scribe2_fits,
)
from constellation.massspec.library.library import Library
from constellation.massspec.quant.quant import Quant
from constellation.massspec.search.search import Search


@dataclass(frozen=True, slots=True)
class EncyclopediaReadResult:
    """All three projections of one ``.dlib`` / ``.elib`` file.

    ``library`` is always populated. ``quant`` is populated when the
    file carries any sample-specific data (chromatogram blob in
    ``entries`` rows, populated ``peptidequants`` / ``fragmentquants``
    / ``retentiontimes`` tables). ``search`` is populated when
    ``peptidescores`` or ``proteinscores`` carry any rows.

    ``caveats`` says, in sentences with counts, where the projection is
    not the whole file: quantity or score rows that could not be tied
    to one precursor, precursors with no quantity row, or a file that
    holds no quantities at all. Empty when none applies, which is the
    normal case. Callers that write the projection to disk should show
    them and record them.
    """

    library: Library
    quant: Quant | None
    search: Search | None
    caveats: tuple[str, ...] = ()
    #: Tables from the text files the jar wrote beside the result
    #: (``_sidecar``): ``rt_alignment`` (RT_ALIGNMENT), ``mass_calibration``
    #: (MASS_CALIBRATION), ``percolator_feature`` (the engine's features,
    #: no fixed schema). Only those it found; empty for a file read on
    #: its own.
    sidecars: dict[str, pa.Table] = field(default_factory=dict)


def read_encyclopedia(
    path: Path | str,
    *,
    annotate_fragments: bool = True,
    fragment_tolerance_ppm: float = 20.0,
    enabled_modifications: ModVocab = UNIMOD,
    include_decoys: bool = False,
    acquisitions: Acquisitions | None = None,
    psms: bool = True,
    sidecars: bool = True,
) -> EncyclopediaReadResult:
    """Read an EncyclopeDIA ``.dlib`` / ``.elib`` file.

    Parameters
    ----------
    path
        SQLite file on disk.
    annotate_fragments
        Whether to build the fragment table: one row per stored peak,
        annotated as a b/y ion or an intact-precursor ion where it is
        one. When ``False``, ``Library.fragments`` is empty, and the
        read is several times faster.
    fragment_tolerance_ppm
        Tolerance for that annotation. Stored fragment m/z are library
        values, so 20 ppm is generous whatever the instrument.
    enabled_modifications
        UNIMOD vocabulary used for parsing ``PeptideModSeq``. Use a
        subset to constrain modification interpretation.
    include_decoys
        When ``False`` (default), rows where ``peptidetoprotein.isDecoy``
        is true are dropped from the protein/peptide tables.
        ``Search.precursor_scores`` keeps decoy rows either way, flagged.
    acquisitions
        An experiment's acquisitions to bind the file's runs to, by
        source name. Without it the file's own source files are numbered
        in sorted order.
    psms
        For a ScribeTwo result with its reports beside it
        (``run.dia.dlib`` and ``run.dia.scribe2/``): read every scored
        peptide-spectrum match into ``Search.psms`` (:mod:`._scribe2`).
        The result library alone has no scans.
    sidecars
        Read the text files the jar wrote beside the result
        (:mod:`._sidecar`): Percolator's scores make
        ``Search.precursor_scores`` complete (every precursor scored, not
        only those within the FDR threshold), and the calibration fits
        and features land in ``EncyclopediaReadResult.sidecars``.

    What each output column holds, and why, is in :mod:`._project`.
    """
    projection = project_library_file(
        path,
        annotate_fragments=annotate_fragments,
        fragment_tolerance_ppm=fragment_tolerance_ppm,
        vocab=enabled_modifications,
        include_decoys=include_decoys,
        acquisitions=acquisitions,
    )
    search = projection.search
    metadata = projection.library.metadata_extras.get("x.encyclopedia.metadata", {})
    engine = engine_of(metadata)
    vocab = enabled_modifications
    tables: dict[str, pa.Table] = {}
    scribetwo = engine == SCRIBETWO_ENGINE
    reports = scribe2_reports(path) if scribetwo and (psms or sidecars) else None
    files = peptide_centric_sidecars(path) if sidecars and not scribetwo else {}
    # A result with nothing within the FDR threshold has no score rows,
    # and so no Search from the result library; what the jar scored is
    # still in the files beside it.
    files_only = search is None and (reports is not None or bool(files))
    if files_only:
        search = Search.empty(_result_acquisitions(metadata, projection, acquisitions))
    if reports is not None:
        if psms:
            search = replace(
                search, psms=read_scribe2_psms(reports, acquisitions=search.acquisitions, vocab=vocab)
            )
        if sidecars:
            stem = reports.name
            run = _single_run(metadata, search.acquisitions)
            units = (_unit(metadata, "-ptolunits"), _unit(metadata, "-ftolunits"))
            tables["rt_alignment"], tables["mass_calibration"] = read_scribe2_fits(
                reports, acquisition_id=run, units=units, vocab=vocab
            )
            scores = [reports / f"{stem}.txt", reports / f"{stem}.decoy.txt"]
            features = reports / f"{stem}.features.txt"
            search = _with_scores(search, [p for p in scores if p.is_file()], engine, vocab)
            if features.is_file():
                tables["percolator_feature"] = read_features(
                    features, acquisitions=search.acquisitions, engine=engine, vocab=vocab
                )
    elif files:
        run = _single_run(metadata, search.acquisitions)
        if any(k.startswith("rt_fit:") for k in files):
            tables["rt_alignment"] = read_rt_fits(files, acquisition_id=run, engine=engine, vocab=vocab)
        if any(k.startswith("ms") for k in files):
            tables["mass_calibration"] = read_mass_fits(files, acquisition_id=run, engine=engine)
        scores = [files[k] for k in ("scores", "decoy_scores") if k in files]
        search = _with_scores(search, scores, engine, vocab)
        if "features" in files:
            tables["percolator_feature"] = read_features(
                files["features"], acquisitions=search.acquisitions, engine=engine, vocab=vocab
            )
    if files_only and not any(
        t.num_rows for t in (search.precursor_scores, search.peptide_scores, search.protein_scores, search.psms)
    ):
        search = None
    return EncyclopediaReadResult(
        library=projection.library,
        quant=projection.quant,
        search=search,
        caveats=projection.caveats,
        sidecars={k: v for k, v in tables.items() if v.num_rows},
    )


def _result_acquisitions(
    metadata: dict[str, str], projection: Any, bind: Acquisitions | None
) -> Acquisitions:
    """The acquisitions of a result that has no score rows: the ones to
    bind to, else the quantities', else its runs as its metadata names
    them (``sourcefile``, ``a.raw|b.raw`` for several)."""
    if bind is not None:
        return bind
    if projection.quant is not None:
        return projection.quant.acquisitions
    sources = [s for s in str(metadata.get("sourcefile") or "").split("|") if s]
    return _acquisitions_for(sorted(sources), "encyclopedia.scores", None)[0]


def _single_run(metadata: dict[str, str], acquisitions: Acquisitions) -> int | None:
    """The acquisition of a single-run result (its ``sourcefile``), or
    ``None`` for a result across runs (``a.raw|b.raw``)."""
    source = str(metadata.get("sourcefile") or "")
    if not source or "|" in source:
        return None
    try:
        return acquisitions.id_for_source(source)
    except KeyError:
        return None


def _unit(metadata: dict[str, str], flag: str) -> str:
    return "ppm" if str(metadata.get(flag, "PPM")).strip().upper() == "PPM" else "Da"


def _with_scores(search: Search, paths: list[Path], engine: str, vocab: ModVocab) -> Search:
    """``search`` with its precursor scores made complete from the
    Percolator-layout scores in ``paths``: every precursor scored, the
    result library's own rows keeping their precursor ids (they agree on
    the values: the scores, q-values and PEPs of the precursors within
    the threshold are the same in both)."""
    if not paths:
        return search
    complete = read_scored_precursors(paths, acquisitions=search.acquisitions, engine=engine, vocab=vocab)
    keys = ["modified_sequence", "charge", "acquisition_id", "is_decoy"]
    known = search.precursor_scores.select([*keys, "precursor_id"]).rename_columns([*keys, "_known"])
    joined = complete.join(known, keys=keys, join_type="left outer")
    joined = joined.set_column(
        joined.schema.get_field_index("precursor_id"), "precursor_id", joined.column("_known")
    ).drop_columns(["_known"])
    # Rows of the result library the sidecar does not have (none, in the
    # files measured) are kept as they are.
    rest = search.precursor_scores.join(complete.select(keys), keys=keys, join_type="left anti")
    return replace(search, precursor_scores=pa.concat_tables([joined.select(rest.column_names), rest]))


__all__ = ["EncyclopediaReadResult", "read_encyclopedia"]
