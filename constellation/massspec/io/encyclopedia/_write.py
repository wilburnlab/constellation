"""EncyclopeDIA ``.dlib`` / ``.elib`` writer — the inverse of ``_read``.

Writes the container EncyclopeDIA 6.9.23 writes, version 0.1.18: the
tables and indexes of the captured schema (``_schema.LIBRARY_TABLES``,
10 tables and 22 indexes), filled the way the jar fills them, as read off
its own results. The extension is the caller's choice; the two are one
format.

What each table holds:

``entries``
    One row per precursor. A result's entry is the precursor in the
    first acquisition it has a quantity in; a library's (no ``Quant``) is
    the precursor alone, under the library's own source name. Its stored
    spectrum is the precursor's fragment rows by m/z; ``RTInSeconds`` the
    acquisition's observed time (else the library's); ``Score`` that
    acquisition's posterior error probability (else 0, as in the jar's
    predicted libraries); the integration window and the median
    chromatogram come from the quant tables. Each stored peak's
    correlation (``CorrelationArray``) is the fragment rows' own
    ``correlation``, and left NULL where the library has none (as in the
    jar's predicted libraries); its quantified flag
    (``QuantifiedIonsArray``) is the rows' own ``quantified``, and where
    the library has none, set on every peak, as the jar sets it in every
    file it writes. The jar searches a chromatogram library with the
    correlations: written back without them, the lab's library gave a
    run 30,682 precursors, 27,035 of them among the 31,895 it gives as
    the jar wrote it, and with them the same 31,895. The flags made no
    difference to that search, all set, a subset or NULL (2026-10-18).
    N-terminal modifications are spelled before the first residue, as the
    jar spells them in this table.
``peptidequants``
    One row per precursor per acquisition with a quantity: its window,
    intensity, the ions it was summed from (``FRAGMENT_QUANT``) and its
    chromatogram (``PRECURSOR_CHROMATOGRAM``); N-terminal modifications
    on the first residue, as the jar spells them here.
``peptidescores``
    The precursor scores within the FDR threshold the metadata names
    (``-percolatorThreshold``; all of them when it names none), decoys
    included: what the jar keeps there. Spelled as ``entries``.
``proteinscores``
    The protein scores, the protein's ``score`` as ``MinimumPeptidePEP``.
``peptidetoprotein``
    Every peptide's proteins; a peptide with none gets an accession of
    its own (``UNMAPPED_<sequence>``).
``metadata``
    The container version, this writer, and the metadata the library
    carries from the file it was read from.
``decoyentries``, ``fragmentquants``, ``retentiontimes``,
``peptidelocalizations``
    Created and left empty: Constellation has nothing that fills them
    (and the jar's own search results leave them empty too).

Not carried, because nothing in Constellation holds them: a precursor's
ion mobility and the jar's protein groups.
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from constellation.massspec.io.encyclopedia import _sql
from constellation.massspec.io.encyclopedia._codec import encode_f4, encode_f8
from constellation.massspec.io.encyclopedia._modseq import proforma_to_encyclopedia
from constellation.massspec.io.encyclopedia._schema import (
    LIBRARY_CONTAINER_VERSION,
    LIBRARY_INDEXES,
    LIBRARY_TABLES,
)
from constellation.massspec.library.library import Library
from constellation.massspec.quant.quant import Quant
from constellation.massspec.search.search import Search

WRITER_VERSION = "constellation/2026.10"

#: The source name of a library's entries when nothing names one.
LIBRARY_SOURCE = "constellation"


def write_encyclopedia(
    path: Path | str,
    library: Library,
    *,
    quant: Quant | None = None,
    search: Search | None = None,
    overwrite: bool = False,
) -> None:
    """Write a ``Library`` (+ optional ``Quant`` / ``Search``) as a
    ``.dlib`` / ``.elib`` (module docstring for what goes where).

    Written to a temporary file beside ``path`` and moved into place
    when complete, so a write that fails leaves nothing behind (and an
    existing file as it was)."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing {p}")

    upstream = dict(library.metadata_extras.get("x.encyclopedia.metadata", {}))
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    tmp.unlink(missing_ok=True)
    try:
        with _sql.open_rw(tmp) as con:
            for _name, ddl in LIBRARY_TABLES:
                con.execute(ddl)
            names = _Names(library)
            sources = _sources(quant, search)
            _write_entries(con, library, names, sources, quant, search, upstream)
            _write_peptidetoprotein(con, library)
            _write_metadata(con, upstream)
            if search is not None:
                _write_peptidescores(con, library, names, sources, search, upstream)
                _write_proteinscores(con, library, sources, search)
            if quant is not None:
                _write_peptidequants(con, library, names, sources, quant)
            for stmt in LIBRARY_INDEXES:
                con.execute(stmt)
            con.commit()
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, p)


# ──────────────────────────────────────────────────────────────────────
# lookups shared by the tables
# ──────────────────────────────────────────────────────────────────────


class _Names:
    """A peptide's two EncyclopeDIA spellings and its sequence, made once."""

    def __init__(self, library: Library) -> None:
        self.sequence = dict(
            zip(library.peptides.column("peptide_id").to_pylist(),
                library.peptides.column("sequence").to_pylist(), strict=True)
        )  # fmt: skip
        self._modified = dict(
            zip(library.peptides.column("peptide_id").to_pylist(),
                library.peptides.column("modified_sequence").to_pylist(), strict=True)
        )  # fmt: skip
        self._spelled: dict[tuple[str, bool], tuple[str, str]] = {}

    def spell(self, proforma: str, terminal_first: bool) -> str:
        return self._spell(proforma, terminal_first)[0]

    def plain(self, proforma: str) -> str:
        return self._spell(proforma, True)[1]

    def _spell(self, proforma: str, terminal_first: bool) -> tuple[str, str]:
        key = (proforma, terminal_first)
        if key not in self._spelled:
            self._spelled[key] = proforma_to_encyclopedia(proforma, terminal_first=terminal_first)
        return self._spelled[key]

    def of(self, peptide_id: int, terminal_first: bool) -> str:
        return self.spell(self._modified[peptide_id], terminal_first)


def _sources(*containers: Quant | Search | None) -> dict[int, str]:
    """``acquisition_id -> source name`` (the jar's ``SourceFile``)."""
    out: dict[int, str] = {}
    for c in containers:
        if c is not None:
            t = c.acquisitions.table
            out.update(zip(t.column("acquisition_id").to_pylist(), t.column("source_file").to_pylist(), strict=True))
    return out


def _source_name(sources: dict[int, str], acquisition_id: int | None) -> str:
    if acquisition_id is None:
        return "global"  # the jar's name for a score across every run
    name = sources.get(acquisition_id)
    if name is None:
        raise ValueError(f"acquisition {acquisition_id} has no source name to write")
    return Path(name).name


def _groups(columns: Iterable[np.ndarray]) -> dict[tuple[int, ...], tuple[int, int]]:
    """``key -> (start, stop)`` of each run of equal keys in rows already
    sorted by those columns."""
    cols = [np.asarray(c) for c in columns]
    n = len(cols[0]) if cols else 0
    if not n:
        return {}
    change = np.zeros(n, dtype=bool)
    change[0] = True
    for c in cols:
        change[1:] |= c[1:] != c[:-1]
    starts = np.flatnonzero(change)
    stops = np.append(starts[1:], n)
    return {tuple(int(c[s]) for c in cols): (int(s), int(e)) for s, e in zip(starts, stops, strict=True)}


def _sorted(table: pa.Table, keys: list[str]) -> pa.Table:
    return table.sort_by([(k, "ascending") for k in keys])


# ──────────────────────────────────────────────────────────────────────
# entries
# ──────────────────────────────────────────────────────────────────────


_ENTRIES_COLUMNS = (
    "PrecursorMz", "PrecursorCharge", "PeptideModSeq", "PeptideSeq", "Copies", "RTInSeconds",
    "Score", "MassEncodedLength", "MassArray", "IntensityEncodedLength", "IntensityArray",
    "CorrelationEncodedLength", "CorrelationArray", "QuantifiedIonsArray", "RTInSecondsStart",
    "RTInSecondsStop", "IonMobility", "MedianChromatogramEncodedLength", "MedianChromatogramArray",
    "SourceFile",
)  # fmt: skip


def _write_entries(
    con: Any, library: Library, names: _Names, sources: dict[int, str],
    quant: Quant | None, search: Search | None, upstream: dict[str, Any],
) -> None:  # fmt: skip
    fragments = _sorted(library.fragments, ["precursor_id", "mz_theoretical"])
    frag_slices = _groups([fragments.column("precursor_id").to_numpy()])
    frag_mz = fragments.column("mz_theoretical").to_numpy(zero_copy_only=False).astype(np.float64)
    frag_intensity = fragments.column("intensity_predicted").to_numpy(zero_copy_only=False).astype(np.float64)
    frag_correlation = fragments.column("correlation").to_numpy(zero_copy_only=False).astype(np.float64)
    has_correlations = fragments.column("correlation").null_count < fragments.num_rows
    quantified_column = fragments.column("quantified")
    frag_quantified = quantified_column.fill_null(True).to_numpy(zero_copy_only=False)
    frag_quantified_known = quantified_column.is_valid().to_numpy(zero_copy_only=False)

    # The acquisition each precursor's entry belongs to: the first it has
    # a quantity in.
    quant_row: dict[int, dict[str, Any]] = {}
    chrom_slices: dict[tuple[int, ...], tuple[int, int]] = {}
    chrom_intensity = np.zeros(0)
    if quant is not None and quant.precursor_quant.num_rows:
        pq = _sorted(quant.precursor_quant, ["precursor_id", "acquisition_id"])
        for row in pq.to_pylist():
            quant_row.setdefault(row["precursor_id"], row)
        chrom = _sorted(quant.precursor_chromatogram, ["precursor_id", "acquisition_id", "rt"])
        chrom_slices = _groups([chrom.column("precursor_id").to_numpy(), chrom.column("acquisition_id").to_numpy()])
        chrom_intensity = chrom.column("intensity").to_numpy(zero_copy_only=False)
    pep: dict[tuple[int, int], float] = {}
    if search is not None:
        ps = search.precursor_scores
        for pid, aid, value in zip(ps.column("precursor_id").to_pylist(), ps.column("acquisition_id").to_pylist(),
                                   ps.column("pep").to_pylist(), strict=True):  # fmt: skip
            if pid is not None and aid is not None and value is not None:
                pep[(pid, aid)] = float(value)
    library_source = str(upstream.get("sourcefile") or LIBRARY_SOURCE)
    if "|" in library_source:  # a report's sourcefile lists its runs
        library_source = LIBRARY_SOURCE

    precursors = library.precursors
    rows: list[tuple[Any, ...]] = []
    for pid, peptide_id, charge, mz, rt in zip(
        precursors.column("precursor_id").to_pylist(), precursors.column("peptide_id").to_pylist(),
        precursors.column("charge").to_pylist(), precursors.column("precursor_mz").to_pylist(),
        precursors.column("rt_predicted").to_pylist(), strict=True,
    ):  # fmt: skip
        start, stop = frag_slices.get((pid,), (0, 0))
        mzs, intensities = frag_mz[start:stop], frag_intensity[start:stop]
        mz_blob, mz_len = encode_f8(mzs)
        intensity_blob, intensity_len = encode_f4(intensities.astype(np.float32))
        # Each peak's correlation, when the library has them and this
        # precursor has one for every peak (an entry with no peaks gets
        # empty arrays, as in the jar's files).
        correlation = frag_correlation[start:stop]
        correlation_blob, correlation_len = (
            encode_f4(correlation.astype(np.float32))
            if has_correlations and not np.isnan(correlation).any()
            else (None, None)
        )
        # Each peak's quantified flag: the library's own when every peak
        # has one, else every peak, which is what the jar writes in every
        # file seen (its predictions, searches, chromatogram libraries and
        # reports); one byte each, compressed like the other arrays.
        own = frag_quantified_known[start:stop].all()
        flags = zlib.compress(
            (frag_quantified[start:stop] if own else np.ones(stop - start, dtype=bool)).astype(np.uint8).tobytes()
        )
        q = quant_row.get(pid)
        if q is not None:
            aid = q["acquisition_id"]
            source = _source_name(sources, aid)
            rt_seconds = q["rt_observed"] if q["rt_observed"] is not None and q["rt_observed"] >= 0 else rt
            window = (q.get("rt_start"), q.get("rt_end"))
            c0, c1 = chrom_slices.get((pid, aid), (0, 0))
            chrom_blob, chrom_len = encode_f4(chrom_intensity[c0:c1].astype(np.float32)) if c1 > c0 else (None, None)
            score = pep.get((pid, aid), 0.0)
        else:
            source, rt_seconds, window = library_source, rt, (None, None)
            chrom_blob = chrom_len = None
            score = 0.0
        rows.append((
            float(mz), int(charge), names.of(peptide_id, True), names.sequence[peptide_id], 1,
            float(rt_seconds), float(score), int(mz_len), mz_blob, int(intensity_len), intensity_blob,
            correlation_len, correlation_blob, flags, window[0], window[1], None, chrom_len, chrom_blob,
            source,
        ))  # fmt: skip
    _sql.insert_many(con, "entries", _ENTRIES_COLUMNS, rows)


# ──────────────────────────────────────────────────────────────────────
# peptidetoprotein
# ──────────────────────────────────────────────────────────────────────


#: Accession prefix for a peptide with no protein mapping. A DLIB's
#: reader gates ``entries`` on ``peptidetoprotein``, so an unmapped
#: peptide would be written and then read back as nothing at all. That
#: is a real case, not a malformed library: a synthetic-peptide panel
#: has no protein of origin.
#:
#: One pseudo-accession PER PEPTIDE, not one shared bucket. EncyclopeDIA
#: runs protein-level inference and FDR over accessions, so collapsing N
#: synthetic peptides onto a single accession would present them as one
#: protein with N supporting peptides — the shape of a confidently
#: identified protein. Per-peptide accessions keep them the independent
#: single-peptide entities they actually are, and the prefix keeps every
#: synthetic target greppable downstream.
UNMAPPED_ACCESSION_PREFIX = "UNMAPPED_"


def _write_peptidetoprotein(con: Any, library: Library) -> None:
    accession_for = dict(
        zip(
            library.proteins.column("protein_id").to_pylist(),
            library.proteins.column("accession").to_pylist(),
            strict=True,
        )
    )
    seq_for = dict(
        zip(
            library.peptides.column("peptide_id").to_pylist(),
            library.peptides.column("sequence").to_pylist(),
            strict=True,
        )
    )
    mapped_peptide_ids = set(library.protein_peptide.column("peptide_id").to_pylist())
    rows = {
        (seq_for[k], False, accession_for[p])
        for p, k in zip(
            library.protein_peptide.column("protein_id").to_pylist(),
            library.protein_peptide.column("peptide_id").to_pylist(),
            strict=True,
        )
    }
    rows |= {
        (seq, False, f"{UNMAPPED_ACCESSION_PREFIX}{seq}")
        for pid, seq in seq_for.items()
        if pid not in mapped_peptide_ids
    }
    _sql.insert_many(
        con, "peptidetoprotein", ("PeptideSeq", "isDecoy", "ProteinAccession"), sorted(rows)
    )


# ──────────────────────────────────────────────────────────────────────
# metadata
# ──────────────────────────────────────────────────────────────────────


_OWN_KEYS = ("version", "constellation_writer_version", "created_iso")


def _write_metadata(con: Any, upstream: dict[str, Any]) -> None:
    rows: list[tuple[str, str]] = [
        ("version", LIBRARY_CONTAINER_VERSION),
        ("constellation_writer_version", WRITER_VERSION),
        ("created_iso", datetime.now(timezone.utc).isoformat()),
    ]
    # The metadata of the file the library was read from (the settings a
    # search recorded, among them), kept as it was.
    rows += [(str(k), str(v)) for k, v in upstream.items() if k not in _OWN_KEYS]
    _sql.insert_many(con, "metadata", ("Key", "Value"), rows)


# ──────────────────────────────────────────────────────────────────────
# peptidescores / proteinscores
# ──────────────────────────────────────────────────────────────────────


def _threshold(upstream: dict[str, Any]) -> float | None:
    try:
        return float(upstream["-percolatorThreshold"])
    except (KeyError, TypeError, ValueError):
        return None


def _write_peptidescores(
    con: Any, library: Library, names: _Names, sources: dict[int, str], search: Search,
    upstream: dict[str, Any],
) -> None:  # fmt: skip
    threshold = _threshold(upstream)
    rows: list[tuple[Any, ...]] = []
    for proforma, charge, aid, qvalue, pep, decoy in _scored_precursors(library, search):
        if threshold is not None and (qvalue is None or qvalue > threshold):
            continue
        rows.append((
            int(charge), names.spell(proforma, True), names.plain(proforma),
            _source_name(sources, aid), 1.0 if qvalue is None else float(qvalue),
            1.0 if pep is None else float(pep), bool(decoy),
        ))  # fmt: skip
    _sql.insert_many(
        con, "peptidescores",
        ("PrecursorCharge", "PeptideModSeq", "PeptideSeq", "SourceFile", "QValue",
         "PosteriorErrorProbability", "IsDecoy"),
        rows,
    )  # fmt: skip


def _scored_precursors(library: Library, search: Search) -> Iterator[tuple[Any, ...]]:
    """``(modified_sequence, charge, acquisition_id, qvalue, pep,
    is_decoy)`` per precursor score, from ``precursor_scores``; for a
    search that has only peptide-level scores (one built by hand, or
    read from a bundle written before precursor scores existed), each
    peptide's score under every precursor of it in the library, since
    the table the jar reads is per precursor."""
    scores = search.precursor_scores
    if scores.num_rows:
        yield from zip(
            scores.column("modified_sequence").to_pylist(), scores.column("charge").to_pylist(),
            scores.column("acquisition_id").to_pylist(), scores.column("qvalue").to_pylist(),
            scores.column("pep").to_pylist(), scores.column("is_decoy").to_pylist(), strict=True,
        )  # fmt: skip
        return
    peptides = search.peptide_scores
    if not peptides.num_rows:
        return
    charges: dict[int, list[int]] = {}
    for peptide_id, charge in zip(
        library.precursors.column("peptide_id").to_pylist(),
        library.precursors.column("charge").to_pylist(), strict=True,
    ):  # fmt: skip
        charges.setdefault(peptide_id, []).append(int(charge))
    modified = dict(
        zip(library.peptides.column("peptide_id").to_pylist(),
            library.peptides.column("modified_sequence").to_pylist(), strict=True)
    )  # fmt: skip
    without = sorted({p for p in peptides.column("peptide_id").to_pylist() if p not in charges})
    if without:
        raise ValueError(
            f"{len(without)} scored peptide(s) have no precursor in the library (peptide ids "
            f"{without[:5]}); EncyclopeDIA's scores are per precursor, so their scores could "
            f"not be written"
        )
    for peptide_id, aid, qvalue, pep in zip(
        peptides.column("peptide_id").to_pylist(), peptides.column("acquisition_id").to_pylist(),
        peptides.column("qvalue").to_pylist(), peptides.column("pep").to_pylist(), strict=True,
    ):  # fmt: skip
        for charge in sorted(charges[peptide_id]):
            yield modified[peptide_id], charge, aid, qvalue, pep, False


def _write_proteinscores(con: Any, library: Library, sources: dict[int, str], search: Search) -> None:
    accession_for = dict(
        zip(library.proteins.column("protein_id").to_pylist(),
            library.proteins.column("accession").to_pylist(), strict=True)
    )  # fmt: skip
    t = search.protein_scores
    rows = [
        (group, accession_for[p], _source_name(sources, aid), 1.0 if q is None else float(q),
         1.0 if score is None else float(score), False)
        for group, (p, aid, q, score) in enumerate(zip(
            t.column("protein_id").to_pylist(), t.column("acquisition_id").to_pylist(),
            t.column("qvalue").to_pylist(), t.column("score").to_pylist(), strict=True,
        ))
    ]  # fmt: skip
    _sql.insert_many(
        con, "proteinscores",
        ("ProteinGroup", "ProteinAccession", "SourceFile", "QValue", "MinimumPeptidePEP", "IsDecoy"),
        rows,
    )  # fmt: skip


# ──────────────────────────────────────────────────────────────────────
# peptidequants
# ──────────────────────────────────────────────────────────────────────


_PEPTIDEQUANTS_COLUMNS = (
    "PrecursorCharge", "PeptideModSeq", "PeptideSeq", "SourceFile", "RTInSecondsCenter",
    "RTInSecondsStart", "RTInSecondsStop", "TotalIntensity", "NumberOfQuantIons",
    "QuantIonMassLength", "QuantIonMassArray", "QuantIonIntensityLength", "QuantIonIntensityArray",
    "QuantIonCorrelationLength", "QuantIonCorrelationArray", "BestFragmentCorrelation",
    "BestFragmentDeltaMassPPM", "MedianChromatogramEncodedLength", "MedianChromatogramArray",
    "MedianChromatogramRTEncodedLength", "MedianChromatogramRTArray", "IdentifiedTICRatio",
)  # fmt: skip


def _write_peptidequants(
    con: Any, library: Library, names: _Names, sources: dict[int, str], quant: Quant
) -> None:
    if not quant.precursor_quant.num_rows:
        return
    of_precursor = dict(
        zip(library.precursors.column("precursor_id").to_pylist(),
            zip(library.precursors.column("peptide_id").to_pylist(),
                library.precursors.column("charge").to_pylist(), strict=True), strict=True)
    )  # fmt: skip
    ions = _sorted(quant.fragment_quant, ["precursor_id", "acquisition_id"])
    ion_slices = _groups([ions.column("precursor_id").to_numpy(), ions.column("acquisition_id").to_numpy()])
    ion_mz = ions.column("mz").to_numpy(zero_copy_only=False).astype(np.float64)
    ion_intensity = ions.column("intensity").to_numpy(zero_copy_only=False).astype(np.float64)
    ion_correlation = ions.column("correlation").fill_null(np.nan).to_numpy(zero_copy_only=False)
    chrom = _sorted(quant.precursor_chromatogram, ["precursor_id", "acquisition_id", "rt"])
    chrom_slices = _groups([chrom.column("precursor_id").to_numpy(), chrom.column("acquisition_id").to_numpy()])
    chrom_rt = chrom.column("rt").to_numpy(zero_copy_only=False)
    chrom_intensity = chrom.column("intensity").to_numpy(zero_copy_only=False)

    rows: list[tuple[Any, ...]] = []
    for row in quant.precursor_quant.to_pylist():
        pid, aid = row["precursor_id"], row["acquisition_id"]
        peptide_id, charge = of_precursor[pid]
        i0, i1 = ion_slices.get((pid, aid), (0, 0))
        mz_blob, mz_len = encode_f8(ion_mz[i0:i1])
        intensity_blob, intensity_len = encode_f4(ion_intensity[i0:i1].astype(np.float32))
        correlation = ion_correlation[i0:i1]
        has_correlation = i1 > i0 and not np.isnan(correlation).any()
        correlation_blob, correlation_len = (
            encode_f4(correlation.astype(np.float32)) if has_correlation else (None, None)
        )
        c0, c1 = chrom_slices.get((pid, aid), (0, 0))
        chrom_blob, chrom_len = encode_f4(chrom_intensity[c0:c1].astype(np.float32))
        rt_blob, rt_len = encode_f4(chrom_rt[c0:c1].astype(np.float32))
        n_ions = row["n_quant_ions"] if row["n_quant_ions"] is not None else i1 - i0
        # The window is required; unknown, it is the apex alone.
        apex = float(row["rt_observed"])
        start = apex if row["rt_start"] is None else float(row["rt_start"])
        stop = apex if row["rt_end"] is None else float(row["rt_end"])
        rows.append((
            int(charge), names.of(peptide_id, False), names.sequence[peptide_id],
            _source_name(sources, aid), apex, start, stop,
            float(row["intensity"]), int(n_ions), int(mz_len), mz_blob, int(intensity_len),
            intensity_blob, correlation_len, correlation_blob,
            float(np.nanmax(correlation)) if has_correlation else 0.0, 0.0, int(chrom_len),
            chrom_blob, int(rt_len), rt_blob, 0.0,
        ))  # fmt: skip
    _sql.insert_many(con, "peptidequants", _PEPTIDEQUANTS_COLUMNS, rows)


__all__ = ["LIBRARY_SOURCE", "UNMAPPED_ACCESSION_PREFIX", "WRITER_VERSION", "write_encyclopedia"]
