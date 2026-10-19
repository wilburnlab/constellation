"""Project an EncyclopeDIA library file onto Constellation's tables.

:mod:`._mirror` turns the file's tables into Arrow as they are. This
module says what they *mean*: which rows are peptides, precursors and
proteins, what a stored spectrum's peaks are, which number is a
precursor's quantity, and which rows are scores. One function,
:func:`project_library_file`, produces the ``(Library, Quant?, Search?)``
triple; ``read_encyclopedia`` is a thin wrapper over it.

It replaces a reader that built every row as a Python dict, and whose
output, checked against the first real 6.9.23 result, was partial and in
places wrong (``docs/plans/encyclopedia-6.9.23-survey.md``). The rules
below are the corrections. Each is a statement about the file, read off
real files, not a preference.

Identity
    A peptide is what its ``entries`` spelling names, as ProForma 2.0
    (:func:`._modseq.parse_encyclopedia_modseq`). Other tables are
    joined to ``entries`` through the residue-form spelling
    (:func:`._modseq.encyclopedia_residue_form`), which is a join key
    and never an identity; a row that fits two precursors is counted as
    ambiguous, not assigned.

Quantity
    A precursor's quantity is ``peptidequants.TotalIntensity``, the
    jar's own number (the sum of the ions it chose to quantify with),
    with ``NumberOfQuantIons`` and the integration window beside it. A
    precursor the jar identified but could not quantify has quantity 0
    and 0 quant ions, and stays that way. Only a file with no
    ``peptidequants`` at all falls back to the sum of the stored
    spectrum, and says so. Each quantity row's arrays become rows of their
    own, under the same precursor and run: the ions it was summed from
    (``fragment_quant``: m/z, intensity, correlation) and its median
    chromatogram (``precursor_chromatogram``: retention time, intensity).

Fragments
    One row per stored peak, no more and no fewer, at the m/z the file
    stores (so a library written back out is the library read in; phase
    7). A peak within tolerance of a b or y ion (fragment charge up to the
    precursor's) is annotated as that ion: the nearest one, and on a tie
    the lower charge. A peak that is the intact precursor, with or without a water
    or ammonia loss, at any charge up to the precursor's, is kept with an
    mzPAF precursor annotation (``p``, ``p-H2O^2``, ...) and null ion
    columns. Anything else is kept with null ion columns and no
    annotation. Nothing is dropped and nothing is listed twice.

    Except in a ScribeTwo result (spectrum-centric). Searched without a
    library its stored spectra are observed spectra: about a thousand
    peaks per identification, almost none of them ions of the peptide
    (with a chromatogram library, 3% of the peaks were not). There the tolerance is the
    one the result records (``-ftol``, after ScribeTwo's own
    adjustment), only peaks that are ions of the peptide are kept, one
    per ion (the most intense; on a tie the nearer), and the rest are
    counted in a caveat (PI, 2026-10-17). The whole spectrum stays in
    the acquisition, reached through the identification's scan.

Scores
    ``precursor_scores`` holds every row of ``peptidescores``, decoys
    included. The peptide-level table is its roll-up (best q-value
    across charge states) over targets.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import Peptidoform, format_proforma
from constellation.massspec.acquisitions import ACQUISITION_TABLE, Acquisitions
from constellation.massspec.io.encyclopedia import _mirror
from constellation.massspec.io.encyclopedia._modseq import (
    encyclopedia_residue_form,
    parse_encyclopedia_modseq,
)
from constellation.massspec.library.library import Library
from constellation.massspec.library.schemas import (
    LIBRARY_FRAGMENT_TABLE,
    PEPTIDE_TABLE,
    PRECURSOR_TABLE,
    PROTEIN_PEPTIDE_EDGE,
    PROTEIN_TABLE,
)
from constellation.massspec.peptide.ions import IonType, fragment_ladder_indices_batch
from constellation.massspec.quant.quant import Quant
from constellation.massspec.quant.schemas import (
    FRAGMENT_QUANT,
    PEPTIDE_QUANT,
    PRECURSOR_CHROMATOGRAM,
    PRECURSOR_QUANT,
    PROTEIN_QUANT,
    TRANSMISSION_PEPTIDE_PRECURSOR,
    TRANSMISSION_PROTEIN_PEPTIDE,
)
from constellation.massspec.search.schemas import (
    PEPTIDE_SCORE_TABLE,
    PRECURSOR_SCORE_TABLE,
    PROTEIN_SCORE_TABLE,
)
from constellation.massspec.search.search import Search

ENGINE = "encyclopedia"
#: The engine name of a ScribeTwo (spectrum-centric) result.
SCRIBETWO_ENGINE = "scribetwo"

_PROTON = 1.007276466812
#: Neutral losses a stored intact-precursor peak may carry, as
#: ``(mass, mzPAF suffix)``. Seen in real results at every charge up to
#: the precursor's; together they are 6.7% of stored peaks and a tenth
#: of stored intensity.
_PRECURSOR_LOSSES: tuple[tuple[float, str], ...] = (
    (0.0, ""),
    (18.010564684, "-H2O"),
    (17.026549101, "-NH3"),
    (36.021129368, "-H2O-H2O"),
    (35.037113785, "-H2O-NH3"),
)
# Entries are annotated a batch at a time; within a batch each precursor's
# m/z values are shifted into their own band so one sorted search matches
# all of them at once. The band must exceed any m/z; the batch size keeps
# the shifted values exact to well under a ppm in float64.
_BAND = 100_000.0
_FRAGMENT_BATCH = 4096


@dataclass(frozen=True, slots=True)
class Projection:
    """What :func:`project_library_file` returns."""

    library: Library
    quant: Quant | None
    search: Search | None
    #: Sentences, with counts, saying where the tables are not the whole
    #: of the file. Empty when they are.
    caveats: tuple[str, ...] = ()


# ──────────────────────────────────────────────────────────────────────
# peptides
# ──────────────────────────────────────────────────────────────────────


def _peptidoforms(
    modseqs: Sequence[str], vocab: ModVocab
) -> tuple[dict[str, Peptidoform], dict[str, str]]:
    """Parse each distinct EncyclopeDIA modseq once.

    Returns ``modseq -> Peptidoform`` and ``modseq -> ProForma``. A
    string with no bracket is its own ProForma and needs no parsing,
    which is most of a library.
    """
    forms: dict[str, Peptidoform] = {}
    proforma: dict[str, str] = {}
    for modseq in modseqs:
        if "[" not in modseq:
            forms[modseq] = Peptidoform(sequence=modseq)
            proforma[modseq] = modseq
        else:
            form = parse_encyclopedia_modseq(modseq, vocab=vocab)
            forms[modseq] = form
            proforma[modseq] = format_proforma(form)
    return forms, proforma


def _first_occurrence_ids(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Number distinct ``keys`` 0, 1, 2, ... in order of first appearance.

    Returns ``(id of each row, row index of each id's first appearance)``.
    """
    _values, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order), dtype=np.int64)
    return rank[inverse], first[order]


def _codes(values: pa.ChunkedArray | pa.Array) -> tuple[np.ndarray, list[Any]]:
    """Dictionary-encode a column: ``(code of each row, distinct values)``."""
    encoded = pc.dictionary_encode(
        values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    )
    return encoded.indices.to_numpy(zero_copy_only=False).astype(np.int64), encoded.dictionary.to_pylist()


# ──────────────────────────────────────────────────────────────────────
# fragments
# ──────────────────────────────────────────────────────────────────────


class _Ladders:
    """Theoretical b/y ladders, computed once per distinct peptidoform."""

    def __init__(self, forms: dict[str, Peptidoform], vocab: ModVocab) -> None:
        self._forms = forms
        self._vocab = vocab
        self._cache: dict[tuple[str, int], tuple[np.ndarray, ...]] = {}

    def ensure(self, wanted: Sequence[tuple[str, int]]) -> None:
        """Compute the ladders of ``(modseq, max fragment charge)`` pairs
        not seen yet, one batched call per charge."""
        missing: dict[int, list[str]] = {}
        for modseq, charge in wanted:
            if (modseq, charge) not in self._cache:
                missing.setdefault(charge, []).append(modseq)
        for charge, modseqs in missing.items():
            modseqs = list(dict.fromkeys(modseqs))
            ladders = fragment_ladder_indices_batch(
                [self._forms[m] for m in modseqs],
                ion_types=(IonType.B, IonType.Y),
                max_fragment_charge=charge,
                vocab=self._vocab,
            )
            for modseq, ladder in zip(modseqs, ladders, strict=True):
                n = len(ladder)
                mz = np.fromiter(ladder.values(), dtype=np.float64, count=n)
                keys = np.array([k[:3] for k in ladder], dtype=np.int64).reshape(n, 3)
                # Sorted by m/z, then charge, so that on an exact tie the
                # lower charge is the one a search lands on first.
                order = np.lexsort((keys[:, 2], mz))
                self._cache[(modseq, charge)] = (
                    mz[order],
                    keys[order, 0].astype(np.int8),
                    keys[order, 1].astype(np.int32),
                    keys[order, 2].astype(np.int32),
                )

    def get(self, modseq: str, charge: int) -> tuple[np.ndarray, ...]:
        return self._cache[(modseq, charge)]

    def forget(self) -> None:
        self._cache.clear()


def _annotate_batch(
    obs_mz: np.ndarray,
    obs_owner: np.ndarray,
    theoretical: list[tuple[np.ndarray, ...]],
    precursor_mz: np.ndarray,
    precursor_charge: np.ndarray,
    tolerance: tuple[float, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str | None]]:
    """Annotate the stored peaks of one batch of entries.

    ``obs_owner`` is the entry (0-based within the batch) each peak
    belongs to; ``theoretical[i]`` is entry ``i``'s ladder. Returns, per
    peak, ``ion_type``, ``position``, ``charge`` (each -1 where the peak
    is not a b/y ion), the m/z to record, and an mzPAF annotation for
    intact-precursor peaks. ``tolerance`` is ``(value, "ppm" | "da")``.
    """
    n_obs = len(obs_mz)
    value, unit = tolerance
    window = obs_mz * value * 1e-6 if unit == "ppm" else np.full(n_obs, float(value))
    ion_type = np.full(n_obs, -1, dtype=np.int8)
    position = np.full(n_obs, -1, dtype=np.int32)
    charge = np.full(n_obs, -1, dtype=np.int32)
    recorded_mz = obs_mz.copy()
    annotation: list[str | None] = [None] * n_obs
    if n_obs == 0:
        return ion_type, position, charge, recorded_mz, annotation

    sizes = np.fromiter((len(t[0]) for t in theoretical), dtype=np.int64, count=len(theoretical))
    if sizes.sum():
        t_owner = np.repeat(np.arange(len(theoretical), dtype=np.int64), sizes)
        t_mz = np.concatenate([t[0] for t in theoretical])
        t_ion = np.concatenate([t[1] for t in theoretical])
        t_pos = np.concatenate([t[2] for t in theoretical])
        t_charge = np.concatenate([t[3] for t in theoretical])
        # Each entry's ladder is already sorted and entries are in order,
        # so the shifted values are sorted as a whole.
        t_key = t_owner * _BAND + t_mz
        o_key = obs_owner * _BAND + obs_mz
        right = np.searchsorted(t_key, o_key, side="left")
        left = right - 1
        best = np.full(n_obs, -1, dtype=np.int64)
        best_error = np.full(n_obs, np.inf)
        for candidate in (left, right):  # left first: it wins an exact tie
            valid = (candidate >= 0) & (candidate < len(t_key))
            index = np.where(valid, candidate, 0)
            same_entry = valid & (t_owner[index] == obs_owner)
            error = np.where(same_entry, np.abs(t_mz[index] - obs_mz), np.inf)
            better = error < best_error
            best = np.where(better, index, best)
            best_error = np.where(better, error, best_error)
        matched = best_error <= window
        hit = best[matched]
        ion_type[matched] = t_ion[hit]
        position[matched] = t_pos[hit]
        charge[matched] = t_charge[hit]
        recorded_mz[matched] = t_mz[hit]
    else:
        matched = np.zeros(n_obs, dtype=bool)

    # What is left: is it the intact precursor?
    rest = np.flatnonzero(~matched)
    if len(rest):
        owner = obs_owner[rest]
        neutral = (precursor_mz[owner] - _PROTON) * precursor_charge[owner]
        max_charge = int(precursor_charge.max())
        found = np.zeros(len(rest), dtype=bool)
        for k in range(1, max_charge + 1):
            for loss, suffix in _PRECURSOR_LOSSES:
                expected = (neutral - loss + k * _PROTON) / k
                is_it = (
                    ~found
                    & (precursor_charge[owner] >= k)
                    & (np.abs(expected - obs_mz[rest]) <= window[rest])
                )
                if is_it.any():
                    label = f"p{suffix}" + (f"^{k}" if k > 1 else "")
                    for i in rest[is_it]:
                        annotation[i] = label
                    recorded_mz[rest[is_it]] = expected[is_it]
                    found |= is_it
    return ion_type, position, charge, recorded_mz, annotation


def _fragment_tables(
    con: Any,
    keep: np.ndarray,
    precursor_of_row: np.ndarray,
    modseq_of_row: Sequence[str],
    precursor_mz: np.ndarray,
    precursor_charge: np.ndarray,
    ladders: _Ladders,
    tolerance: tuple[float, str],
    matched_only: bool = False,
    dropped: list[int] | None = None,
) -> Iterator[pa.Table]:
    """One LIBRARY_FRAGMENT_TABLE chunk per batch of ``entries`` rows.

    ``matched_only`` keeps only the peaks that are ions of the peptide,
    one per ion, and adds the number of peaks left out to ``dropped[0]``.
    """
    declared = {name for name, _ in _mirror._declared(con, "entries")}
    extras = [c for c in ("CorrelationArray", "QuantifiedIonsArray") if c in declared]
    row = 0
    for batch in _mirror.iter_table(
        con, "entries", columns=["MassArray", "IntensityArray", *extras], batch_rows=_FRAGMENT_BATCH
    ):
        rows = np.arange(row, row + batch.num_rows)
        row += batch.num_rows
        rows = rows[keep[rows]]
        if not len(rows):
            continue
        local = rows - (row - batch.num_rows)
        masses = batch.column("MassArray").take(pa.array(local))
        intensities = batch.column("IntensityArray").take(pa.array(local))
        if masses.null_count or intensities.null_count:
            raise _mirror.MirrorError("an entry has no stored spectrum (null MassArray)")
        sizes = pc.list_value_length(masses).to_numpy(zero_copy_only=False).astype(np.int64)
        if not np.array_equal(
            sizes, pc.list_value_length(intensities).to_numpy(zero_copy_only=False)
        ):
            raise _mirror.MirrorError("an entry's MassArray and IntensityArray differ in length")
        obs_mz = masses.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
        obs_intensity = intensities.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
        obs_owner = np.repeat(np.arange(len(rows), dtype=np.int64), sizes)
        correlation = _per_peak(batch, "CorrelationArray", local, sizes, np.float64)
        quantified = _per_peak(batch, "QuantifiedIonsArray", local, sizes, np.bool_)

        charges = precursor_charge[rows]
        wanted = [(modseq_of_row[r], int(z)) for r, z in zip(rows, charges, strict=True)]
        ladders.ensure(wanted)
        theoretical = [ladders.get(m, z) for m, z in wanted]
        ion_type, position, charge, recorded_mz, annotation = _annotate_batch(
            obs_mz, obs_owner, theoretical, precursor_mz[rows], charges.astype(np.int64), tolerance
        )
        ladders.forget()
        owner_precursor = np.repeat(precursor_of_row[rows], sizes)
        if matched_only:
            keep_peak = _one_peak_per_ion(
                obs_owner, ion_type, position, charge, annotation, obs_intensity,
                np.abs(recorded_mz - obs_mz),
            )  # fmt: skip
            if dropped is not None:
                dropped[0] += int(len(keep_peak) - keep_peak.sum())
            index = np.flatnonzero(keep_peak)
            ion_type, position, charge = ion_type[index], position[index], charge[index]
            obs_mz, obs_intensity = obs_mz[index], obs_intensity[index]
            correlation, quantified = correlation.take(pa.array(index)), quantified.take(pa.array(index))
            owner_precursor = owner_precursor[index]
            annotation = [annotation[i] for i in index]
        unassigned = ion_type < 0
        yield pa.table(
            {
                "precursor_id": pa.array(owner_precursor, type=pa.int64()),
                "ion_type": pa.array(ion_type, type=pa.int8(), mask=unassigned),
                "position": pa.array(position, type=pa.int32(), mask=unassigned),
                "charge": pa.array(charge, type=pa.int32(), mask=unassigned),
                "loss_id": pa.nulls(len(obs_mz), type=pa.string()),
                # The file's own m/z: the library's value, or for a
                # ScribeTwo result the observed peak. The annotation says
                # which ion it is.
                "mz_theoretical": pa.array(obs_mz, type=pa.float64()),
                "intensity_predicted": pa.array(obs_intensity, type=pa.float64()),
                "annotation": pa.array(annotation, type=pa.string()),
                "correlation": correlation,
                "quantified": quantified,
            },
            schema=LIBRARY_FRAGMENT_TABLE,
        )


def _per_peak(
    batch: pa.RecordBatch, column: str, rows: np.ndarray, sizes: np.ndarray, dtype: Any
) -> pa.Array:
    """A per-peak array column of the entries ``rows`` (positions in the
    batch), aligned with their stored peaks: null for an entry that has
    none, or whose array does not have one value per peak."""
    arrow_type = pa.float64() if dtype is np.float64 else pa.bool_()
    total = int(sizes.sum())
    if column not in batch.schema.names:
        return pa.nulls(total, type=arrow_type)
    values = batch.column(column).take(pa.array(rows))
    lengths = pc.list_value_length(values).fill_null(-1).to_numpy(zero_copy_only=False)
    if np.array_equal(lengths, sizes):
        flat = values.flatten().to_numpy(zero_copy_only=False)
        return pa.array(flat.astype(dtype), type=arrow_type)
    parts = []
    for value, length, size in zip(values, lengths, sizes, strict=True):
        if length == size:
            parts.append(pa.array(np.asarray(value.as_py()).astype(dtype), type=arrow_type))
        else:
            parts.append(pa.nulls(int(size), type=arrow_type))
    return pa.concat_arrays(parts) if parts else pa.nulls(0, type=arrow_type)


def _one_peak_per_ion(
    owner: np.ndarray,
    ion_type: np.ndarray,
    position: np.ndarray,
    charge: np.ndarray,
    annotation: list[str | None],
    intensity: np.ndarray,
    error: np.ndarray,
) -> np.ndarray:
    """Which peaks to keep when only ions are kept: of the peaks one
    entry's spectrum has for one ion (a b/y ion, or an intact-precursor
    annotation), the most intense, and on a tie the nearer."""
    n = len(owner)
    if n == 0:
        return np.zeros(0, dtype=bool)
    precursor_label = np.array([a or "" for a in annotation], dtype=object)
    is_ion = (ion_type >= 0) | (precursor_label != "")
    rows = np.flatnonzero(is_ion)
    keep = np.zeros(n, dtype=bool)
    if not len(rows):
        return keep
    _, label_code = np.unique(precursor_label[rows].astype(str), return_inverse=True)
    # Most intense first, then nearest; the first row of each ion wins.
    order = np.lexsort((error[rows], -intensity[rows], label_code, charge[rows], position[rows],
                        ion_type[rows], owner[rows]))  # fmt: skip
    ranked = rows[order]
    key = np.stack(
        [owner[ranked], ion_type[ranked], position[ranked], charge[ranked], label_code[order]],
        axis=1,
    )
    first = np.ones(len(ranked), dtype=bool)
    first[1:] = np.any(key[1:] != key[:-1], axis=1)
    keep[ranked[first]] = True
    return keep


def engine_of(metadata: dict[str, str]) -> str:
    """Which engine wrote a result: ScribeTwo records its ``search type``
    as ``ScribeTwo...`` (``ScribeTwoDDA``), the peptide-centric search as
    ``EncyclopeDIA``."""
    kinds = {v for k, v in metadata.items() if k.endswith(" search type")}
    return SCRIBETWO_ENGINE if any(str(v).startswith("ScribeTwo") for v in kinds) else ENGINE


def _recorded_tolerance(metadata: dict[str, str], value_flag: str, unit_flag: str) -> tuple[float, str] | None:
    """A tolerance as the result records it, ``(value, "ppm" | "da")``."""
    try:
        value = float(metadata[value_flag])
    except (KeyError, ValueError):
        return None
    unit = str(metadata.get(unit_flag, "PPM")).strip().lower()
    return value, "ppm" if unit == "ppm" else "da"


# ──────────────────────────────────────────────────────────────────────
# the projection
# ──────────────────────────────────────────────────────────────────────


#: How EncyclopeDIA names the run of a score computed across every run of
#: a report (``-libexport`` writes its protein scores under it): no single
#: acquisition. Such rows get a null ``acquisition_id``, which the score
#: schemas define as a run-agnostic score.
RUN_AGNOSTIC_SOURCE = "global"


def _acquisitions_for(
    sources: Sequence[str], kind: str, bind: Acquisitions | None
) -> tuple[Acquisitions, dict[str, int | None]]:
    """The acquisitions a file's rows belong to, and ``SourceFile -> id``.

    With ``bind`` (an experiment's acquisitions) every source must be one
    of them, found by name; an unknown or ambiguous source is an error,
    not a new acquisition. Without it, the file's own sources are
    numbered in sorted order, which is all a standalone read can do.
    :data:`RUN_AGNOSTIC_SOURCE` maps to ``None`` either way.
    """
    agnostic = {RUN_AGNOSTIC_SOURCE: None} if RUN_AGNOSTIC_SOURCE in sources else {}
    sources = [s for s in sources if s != RUN_AGNOSTIC_SOURCE]
    if bind is not None:
        return bind, {**{s: bind.id_for_source(s) for s in sources}, **agnostic}
    ordered = sorted(sources)
    table = pa.Table.from_pylist(
        [
            {"acquisition_id": i, "source_file": s, "source_kind": kind, "acquisition_datetime": None}
            for i, s in enumerate(ordered)
        ],
        schema=ACQUISITION_TABLE,
    )
    return Acquisitions(table), {**{s: i for i, s in enumerate(ordered)}, **agnostic}


def project_library_file(
    path: Path | str,
    *,
    annotate_fragments: bool = True,
    fragment_tolerance_ppm: float = 20.0,
    vocab: ModVocab = UNIMOD,
    include_decoys: bool = False,
    acquisitions: Acquisitions | None = None,
) -> Projection:
    """Read a ``.dlib`` / ``.elib`` as ``(Library, Quant?, Search?)``.

    Ids are sequential, in order of first appearance in the file, and
    mean nothing outside the result (re-key with
    :func:`constellation.massspec.library.ids.with_content_ids` to get
    ids that do). ``acquisitions`` binds the file's runs to an
    experiment's acquisitions by source name instead of numbering them
    afresh. With ``annotate_fragments=False`` the fragment table is left
    empty, which is much the fastest way to read identifications,
    quantities and scores.
    """
    path = Path(path)
    caveats: list[str] = []
    con = _mirror.open_library(path)
    try:
        tables = set(_mirror.table_names(con))
        metadata = _mirror.read_metadata(con)

        # ── proteins and the peptides they license ──────────────────
        links = _mirror.read_table(con, "peptidetoprotein")
        if not include_decoys:
            links = links.filter(pc.invert(pc.fill_null(links.column("isDecoy"), False)))
        accession_code, accessions = _codes(links.column("ProteinAccession"))
        protein_of_link, first_link = _first_occurrence_ids(accession_code)
        accessions_in_order = [accessions[accession_code[i]] for i in first_link]
        proteins = pa.table(
            {
                "protein_id": pa.array(np.arange(len(accessions_in_order)), type=pa.int64()),
                "accession": pa.array(accessions_in_order, type=pa.string()),
                "sequence": pa.nulls(len(accessions_in_order), type=pa.string()),
                "description": pa.nulls(len(accessions_in_order), type=pa.string()),
            },
            schema=PROTEIN_TABLE,
        )

        # ── entries: peptides and precursors ────────────────────────
        entries = _mirror.read_table(
            con,
            "entries",
            columns=["PrecursorMz", "PrecursorCharge", "PeptideModSeq", "PeptideSeq",
                     "RTInSeconds", "Score", "SourceFile"],
        )
        n_entries = entries.num_rows
        if include_decoys:
            keep = np.ones(n_entries, dtype=bool)
        else:
            # An entry whose sequence is only ever a decoy has no target
            # protein and is left out with the decoys.
            licensed = pc.unique(links.column("PeptideSeq"))
            keep = pc.is_in(entries.column("PeptideSeq"), value_set=licensed).to_numpy(
                zero_copy_only=False
            )
        modseq_code, modseqs = _codes(entries.column("PeptideModSeq"))
        forms, proforma_of = _peptidoforms(modseqs, vocab)
        modseq_of_row = [modseqs[c] for c in modseq_code]
        proforma_code, proformas = _codes(pa.array([proforma_of[m] for m in modseqs]))
        peptide_key = proforma_code[modseq_code]  # same ProForma -> same peptide

        kept_rows = np.flatnonzero(keep)
        peptide_of_kept, first_peptide = _first_occurrence_ids(peptide_key[kept_rows])
        peptide_of_row = np.full(n_entries, -1, dtype=np.int64)
        peptide_of_row[kept_rows] = peptide_of_kept
        peptide_rows = kept_rows[first_peptide]
        sequences = entries.column("PeptideSeq").to_pylist()
        peptides = pa.table(
            {
                "peptide_id": pa.array(np.arange(len(peptide_rows)), type=pa.int64()),
                "sequence": pa.array([sequences[r] for r in peptide_rows], type=pa.string()),
                "modified_sequence": pa.array(
                    [proforma_of[modseq_of_row[r]] for r in peptide_rows], type=pa.string()
                ),
            },
            schema=PEPTIDE_TABLE,
        )

        charges = entries.column("PrecursorCharge").to_numpy(zero_copy_only=False).astype(np.int64)
        source_code, sources = _codes(entries.column("SourceFile"))
        # (peptide, charge, source) as one integer; charge < 256.
        precursor_key = (peptide_of_row * 256 + charges) * max(len(sources), 1) + source_code
        precursor_of_kept, first_precursor = _first_occurrence_ids(precursor_key[kept_rows])
        precursor_of_row = np.full(n_entries, -1, dtype=np.int64)
        precursor_of_row[kept_rows] = precursor_of_kept
        precursor_rows = kept_rows[first_precursor]
        n_precursors = len(precursor_rows)
        repeated = len(kept_rows) - n_precursors
        if repeated:
            caveats.append(
                f"{repeated:,} entries repeat a precursor (same peptide, charge and source file); "
                f"the first is the precursor, and the stored peaks of all are kept"
            )
        precursor_mz = entries.column("PrecursorMz").to_numpy(zero_copy_only=False).astype(np.float64)
        rt = entries.column("RTInSeconds").to_numpy(zero_copy_only=False).astype(np.float64)
        precursors = pa.table(
            {
                "precursor_id": pa.array(np.arange(n_precursors), type=pa.int64()),
                "peptide_id": pa.array(peptide_of_row[precursor_rows], type=pa.int64()),
                "charge": pa.array(charges[precursor_rows], type=pa.int32()),
                "precursor_mz": pa.array(precursor_mz[precursor_rows], type=pa.float64()),
                "rt_predicted": pa.array(rt[precursor_rows], type=pa.float64()),
                "ccs_predicted": pa.array(np.full(n_precursors, -1.0), type=pa.float64()),
            },
            schema=PRECURSOR_TABLE,
        )

        # ── protein-peptide edges: by sequence, every modified form ──
        link_sequence_code, link_sequences = _codes(links.column("PeptideSeq"))
        peptides_by_sequence: dict[str, list[int]] = {}
        for pid, sequence in enumerate(peptides.column("sequence").to_pylist()):
            peptides_by_sequence.setdefault(sequence, []).append(pid)
        edges: set[tuple[int, int]] = set()
        for protein, code in zip(protein_of_link.tolist(), link_sequence_code.tolist(), strict=True):
            for pid in peptides_by_sequence.get(link_sequences[code], ()):
                edges.add((protein, pid))
        ordered_edges = sorted(edges)
        protein_peptide = pa.table(
            {
                "protein_id": pa.array([e[0] for e in ordered_edges], type=pa.int64()),
                "peptide_id": pa.array([e[1] for e in ordered_edges], type=pa.int64()),
            },
            schema=PROTEIN_PEPTIDE_EDGE,
        )

        # ── fragments ───────────────────────────────────────────────
        engine = engine_of(metadata)
        if annotate_fragments and len(kept_rows):
            tolerance: tuple[float, str] = (fragment_tolerance_ppm, "ppm")
            matched_only = engine == SCRIBETWO_ENGINE
            if matched_only:
                tolerance = _recorded_tolerance(metadata, "-ftol", "-ftolunits") or tolerance
            dropped = [0]
            chunks = list(
                _fragment_tables(
                    con, keep, precursor_of_row, modseq_of_row, precursor_mz, charges,
                    _Ladders(forms, vocab), tolerance, matched_only, dropped,
                )
            )
            fragments = (
                pa.concat_tables(chunks) if chunks else LIBRARY_FRAGMENT_TABLE.empty_table()
            )
            if dropped[0]:
                unit = "ppm" if tolerance[1] == "ppm" else "Da"
                caveats.append(
                    f"{dropped[0]:,} of the {dropped[0] + fragments.num_rows:,} peaks this ScribeTwo "
                    f"result stores are not ions of the peptide within {tolerance[0]:g} {unit} (or "
                    f"repeat an ion) and are not fragment rows"
                )
        else:
            fragments = LIBRARY_FRAGMENT_TABLE.empty_table()

        library = Library(
            proteins=proteins,
            peptides=peptides,
            precursors=precursors,
            fragments=fragments,
            protein_peptide=protein_peptide,
            metadata_extras={"x.encyclopedia.metadata": metadata},
        )

        # ── join key of every precursor, for the tables that use it ──
        residue_form = {m: encyclopedia_residue_form(m) for m in modseqs}
        by_join_key: dict[tuple[str, int, str], list[int]] = {}
        # ...and by the entries spelling itself, for `peptidescores`,
        # which spells N-terminal modifications the way `entries` does.
        by_spelling: dict[tuple[str, int, str], int] = {}
        for r in precursor_rows.tolist():
            source, charge = sources[source_code[r]], int(charges[r])
            key = (residue_form[modseq_of_row[r]], charge, source)
            by_join_key.setdefault(key, []).append(int(precursor_of_row[r]))
            by_spelling[(modseq_of_row[r], charge, source)] = int(precursor_of_row[r])

        quant = _project_quant(
            con, tables, entries, kept_rows, precursor_rows, precursor_of_row, sources,
            source_code, by_join_key, peptide_of_row, rt, acquisitions, caveats,
        )
        search = _project_search(
            con, tables, library, by_spelling, by_join_key, vocab, include_decoys,
            acquisitions, caveats, engine,
        )
    finally:
        con.close()
    return Projection(library=library, quant=quant, search=search, caveats=tuple(caveats))


def _project_quant(
    con: Any,
    tables: set[str],
    entries: pa.Table,
    kept_rows: np.ndarray,
    precursor_rows: np.ndarray,
    precursor_of_row: np.ndarray,
    sources: list[Any],
    source_code: np.ndarray,
    by_join_key: dict[tuple[str, int, str], list[int]],
    peptide_of_row: np.ndarray,
    rt: np.ndarray,
    bind: Acquisitions | None,
    caveats: list[str],
) -> Quant | None:
    n_quant_rows = _mirror.count_rows(con, "peptidequants") if "peptidequants" in tables else 0
    has_chromatogram = (
        con.execute(
            "SELECT 1 FROM entries WHERE MedianChromatogramArray IS NOT NULL LIMIT 1"
        ).fetchone()
        is not None
    )
    populated = any(
        t in tables and _mirror.count_rows(con, t) for t in ("fragmentquants", "retentiontimes")
    )
    if not (n_quant_rows or has_chromatogram or populated) or not len(precursor_rows):
        return None
    quants = (
        _mirror.read_table(
            con,
            "peptidequants",
            columns=["PrecursorCharge", "PeptideModSeq", "SourceFile", "RTInSecondsCenter",
                     "RTInSecondsStart", "RTInSecondsStop", "TotalIntensity", "NumberOfQuantIons"],
        )
        if n_quant_rows
        else None
    )
    present = {sources[c] for c in source_code[precursor_rows] if sources[c]}
    if quants is not None:
        present.update(s for s in pc.unique(quants.column("SourceFile")).to_pylist() if s)
    if not present:
        return None
    acquisitions, acquisition_of = _acquisitions_for(sorted(present), "encyclopedia.entries", bind)

    n = len(precursor_rows)
    precursor_ids = precursor_of_row[precursor_rows]
    acquisition_ids = np.array(
        [acquisition_of[sources[c]] for c in source_code[precursor_rows]], dtype=np.int64
    )
    intensity = np.zeros(n, dtype=np.float64)
    rt_observed = rt[precursor_rows].astype(np.float64)
    rt_start = np.full(n, np.nan)
    rt_end = np.full(n, np.nan)
    n_ions = np.full(n, -1, dtype=np.int64)
    position_of = {int(pid): i for i, pid in enumerate(precursor_ids.tolist())}

    if quants is not None:
        # A single-run result has one quantity row per entry, under the
        # entry's own run. A report across runs (``-libexport``) has one
        # entry per precursor but a quantity row per precursor PER RUN, so
        # a row for a run the entry does not name is found by the
        # precursor alone and becomes a row of its own.
        by_form: dict[tuple[str, int], set[int]] = {}
        for (form_key, charge_key, _source), ids in by_join_key.items():
            by_form.setdefault((form_key, charge_key), set()).update(ids)
        extra: dict[tuple[int, int], tuple[float, float, float, float, int]] = {}
        seen = np.zeros(n, dtype=bool)
        unmatched = ambiguous = repeated = 0
        # Which precursor and run each quantity row was given to (-1: none),
        # for the rows of its arrays.
        row_precursor = np.full(quants.num_rows, -1, dtype=np.int64)
        row_acquisition = np.full(quants.num_rows, -1, dtype=np.int64)
        rows = zip(
            quants.column("PeptideModSeq").to_pylist(),
            quants.column("PrecursorCharge").to_pylist(),
            quants.column("SourceFile").to_pylist(),
            quants.column("TotalIntensity").to_pylist(),
            quants.column("RTInSecondsCenter").to_pylist(),
            quants.column("RTInSecondsStart").to_pylist(),
            quants.column("RTInSecondsStop").to_pylist(),
            quants.column("NumberOfQuantIons").to_pylist(),
            strict=True,
        )
        spelled: dict[str, str] = {}
        for r, (modseq, charge, source, total, center, start, stop, ions) in enumerate(rows):
            form = spelled.get(modseq)
            if form is None:
                form = spelled[modseq] = encyclopedia_residue_form(modseq)
            candidates = by_join_key.get((form, int(charge), source), ())
            if len(candidates) > 1:
                ambiguous += 1
                continue
            if candidates:
                i = position_of[candidates[0]]
                if seen[i]:
                    repeated += 1
                    continue
                seen[i] = True
                row_precursor[r], row_acquisition[r] = candidates[0], acquisition_of[source]
                intensity[i] = float(total)
                rt_start[i] = np.nan if start is None else float(start)
                rt_end[i] = np.nan if stop is None else float(stop)
                n_ions[i] = -1 if ions is None else int(ions)
                continue
            elsewhere = by_form.get((form, int(charge)), set())
            if not elsewhere:
                unmatched += 1
                continue
            if len(elsewhere) > 1:
                ambiguous += 1
                continue
            key = (next(iter(elsewhere)), acquisition_of[source])
            if key in extra:
                repeated += 1
                continue
            row_precursor[r], row_acquisition[r] = key
            extra[key] = (
                float(total),
                np.nan if center is None else float(center),
                np.nan if start is None else float(start),
                np.nan if stop is None else float(stop),
                -1 if ions is None else int(ions),
            )
        missing = int((~seen).sum())
        for count, what in (
            (unmatched, "matched no precursor of the entries table and are left out"),
            (ambiguous, "fit more than one precursor (the table cannot tell an N-terminal "
                        "modification from one on the first residue) and are left out"),
            (repeated, "repeat a precursor already given a quantity and are left out"),
        ):
            if count:
                caveats.append(f"{count:,} peptidequants rows {what}")
        if missing:
            caveats.append(
                f"{missing:,} precursors have no peptidequants row; their quantity is "
                f"recorded as 0 with the number of quant ions unknown"
            )
        if extra:
            keys = list(extra)
            values = np.array(list(extra.values()), dtype=np.float64)
            precursor_ids = np.concatenate([precursor_ids, np.array([k[0] for k in keys], dtype=np.int64)])
            acquisition_ids = np.concatenate([acquisition_ids, np.array([k[1] for k in keys], dtype=np.int64)])
            intensity = np.concatenate([intensity, values[:, 0]])
            rt_observed = np.concatenate([rt_observed, values[:, 1]])
            rt_start = np.concatenate([rt_start, values[:, 2]])
            rt_end = np.concatenate([rt_end, values[:, 3]])
            n_ions = np.concatenate([n_ions, values[:, 4].astype(np.int64)])
            n = len(precursor_ids)
    else:
        # No quantities in the file at all (a library with chromatograms):
        # the sum of the stored spectrum is the only number there is.
        caveats.append(
            "the file has no peptidequants table rows; precursor_quant.intensity is the "
            "sum of each entry's stored spectrum, not a quantity EncyclopeDIA computed"
        )
        position_in_entries = {int(r): i for i, r in enumerate(precursor_rows.tolist())}
        row = 0
        for batch in _mirror.iter_table(con, "entries", columns=["IntensityArray"]):
            column = batch.column("IntensityArray")
            values = column.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
            sizes = pc.list_value_length(column).fill_null(0).to_numpy(zero_copy_only=False)
            ends = np.cumsum(sizes)
            running = np.concatenate([[0.0], np.cumsum(values)])
            sums = running[ends] - running[ends - sizes]
            for offset, value in enumerate(sums):
                i = position_in_entries.get(row + offset)
                if i is not None:
                    intensity[i] = float(value)
            row += batch.num_rows

    precursor_quant = pa.table(
        {
            "precursor_id": pa.array(precursor_ids, type=pa.int64()),
            "acquisition_id": pa.array(acquisition_ids, type=pa.int64()),
            "intensity": pa.array(intensity, type=pa.float64()),
            "rt_observed": pa.array(rt_observed, type=pa.float64()),
            "ccs_observed": pa.array(np.full(n, -1.0), type=pa.float64()),
            "score": pa.nulls(n, type=pa.float64()),
            "rt_start": pa.array(rt_start, type=pa.float64(), mask=np.isnan(rt_start)),
            "rt_end": pa.array(rt_end, type=pa.float64(), mask=np.isnan(rt_end)),
            "n_quant_ions": pa.array(n_ions, type=pa.int32(), mask=n_ions < 0),
        },
        schema=PRECURSOR_QUANT,
    )
    if n_quant_rows:
        # A peptide's abundance is the sum over its charge states.
        peptide_of = dict(
            zip(
                precursor_of_row[precursor_rows].tolist(),
                peptide_of_row[precursor_rows].tolist(),
                strict=True,
            )
        )
        peptide_ids = np.array([peptide_of[int(p)] for p in precursor_ids], dtype=np.int64)
        rollup = (
            pa.table({"peptide_id": peptide_ids, "acquisition_id": acquisition_ids, "abundance": intensity})
            .group_by(["peptide_id", "acquisition_id"], use_threads=False)
            .aggregate([("abundance", "sum")])
        )
        peptide_quant = pa.table(
            {
                "peptide_id": rollup.column("peptide_id"),
                "acquisition_id": rollup.column("acquisition_id"),
                "abundance": rollup.column("abundance_sum"),
                "score": pa.nulls(rollup.num_rows, type=pa.float64()),
            },
            schema=PEPTIDE_QUANT,
        )
    else:
        peptide_quant = PEPTIDE_QUANT.empty_table()
    if quants is not None:
        chromatogram, fragment_quant = _quant_arrays(con, row_precursor, row_acquisition)
    else:
        chromatogram, fragment_quant = PRECURSOR_CHROMATOGRAM.empty_table(), FRAGMENT_QUANT.empty_table()
    return Quant(
        acquisitions=acquisitions,
        protein_quant=PROTEIN_QUANT.empty_table(),
        peptide_quant=peptide_quant,
        precursor_quant=precursor_quant,
        transmission_protein_peptide=TRANSMISSION_PROTEIN_PEPTIDE.empty_table(),
        transmission_peptide_precursor=TRANSMISSION_PEPTIDE_PRECURSOR.empty_table(),
        precursor_chromatogram=chromatogram,
        fragment_quant=fragment_quant,
    )


def _flat(batch: pa.RecordBatch, column: str, wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(values, sizes)`` of a list column, over the rows ``wanted``
    (a boolean mask); a null list counts as empty."""
    values = batch.column(column)
    sizes = pc.list_value_length(values).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    flat = values.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
    keep_values = np.repeat(wanted, sizes)
    return flat[keep_values], sizes[wanted]


def _quant_arrays(
    con: Any, row_precursor: np.ndarray, row_acquisition: np.ndarray
) -> tuple[pa.Table, pa.Table]:
    """The quantity rows' arrays as long tables: the ions each precursor's
    intensity was summed from, and its median chromatogram."""
    declared = {name for name, _ in _mirror._declared(con, "peptidequants")}
    columns = [c for c in ("QuantIonMassArray", "QuantIonIntensityArray", "QuantIonCorrelationArray",
                           "MedianChromatogramArray", "MedianChromatogramRTArray") if c in declared]  # fmt: skip
    chromatograms: list[pa.Table] = []
    ions: list[pa.Table] = []
    row = 0
    for batch in _mirror.iter_table(con, "peptidequants", columns=columns):
        owner = row_precursor[row : row + batch.num_rows]
        run = row_acquisition[row : row + batch.num_rows]
        row += batch.num_rows
        wanted = owner >= 0
        if not wanted.any():
            continue
        if "QuantIonMassArray" in columns and "QuantIonIntensityArray" in columns:
            mz, sizes = _flat(batch, "QuantIonMassArray", wanted)
            intensity, intensity_sizes = _flat(batch, "QuantIonIntensityArray", wanted)
            if not np.array_equal(sizes, intensity_sizes):
                raise _mirror.MirrorError("a quantity row's ion masses and intensities differ in length")
            if "QuantIonCorrelationArray" in columns:
                correlation, correlation_sizes = _flat(batch, "QuantIonCorrelationArray", wanted)
                if not np.array_equal(sizes, correlation_sizes):
                    raise _mirror.MirrorError("a quantity row's ion masses and correlations differ in length")
                correlation_array = pa.array(correlation, type=pa.float64())
            else:
                correlation_array = pa.nulls(len(mz), type=pa.float64())
            ions.append(
                pa.table(
                    {
                        "precursor_id": pa.array(np.repeat(owner[wanted], sizes), type=pa.int64()),
                        "acquisition_id": pa.array(np.repeat(run[wanted], sizes), type=pa.int64()),
                        "mz": pa.array(mz, type=pa.float64()),
                        "intensity": pa.array(intensity, type=pa.float64()),
                        "correlation": correlation_array,
                    },
                    schema=FRAGMENT_QUANT,
                )
            )
        if "MedianChromatogramArray" in columns and "MedianChromatogramRTArray" in columns:
            intensity, sizes = _flat(batch, "MedianChromatogramArray", wanted)
            rts, rt_sizes = _flat(batch, "MedianChromatogramRTArray", wanted)
            if not np.array_equal(sizes, rt_sizes):
                raise _mirror.MirrorError("a quantity row's chromatogram and its times differ in length")
            chromatograms.append(
                pa.table(
                    {
                        "precursor_id": pa.array(np.repeat(owner[wanted], sizes), type=pa.int64()),
                        "acquisition_id": pa.array(np.repeat(run[wanted], sizes), type=pa.int64()),
                        "rt": pa.array(rts, type=pa.float64()),
                        "intensity": pa.array(intensity, type=pa.float64()),
                    },
                    schema=PRECURSOR_CHROMATOGRAM,
                )
            )
    return (
        pa.concat_tables(chromatograms) if chromatograms else PRECURSOR_CHROMATOGRAM.empty_table(),
        pa.concat_tables(ions) if ions else FRAGMENT_QUANT.empty_table(),
    )


def _project_search(
    con: Any,
    tables: set[str],
    library: Library,
    by_spelling: dict[tuple[str, int, str], int],
    by_join_key: dict[tuple[str, int, str], list[int]],
    vocab: ModVocab,
    include_decoys: bool,
    bind: Acquisitions | None,
    caveats: list[str],
    engine: str = ENGINE,
) -> Search | None:
    n_peptide_scores = _mirror.count_rows(con, "peptidescores") if "peptidescores" in tables else 0
    n_protein_scores = _mirror.count_rows(con, "proteinscores") if "proteinscores" in tables else 0
    if not (n_peptide_scores or n_protein_scores):
        return None
    peptide_scores = _mirror.read_table(con, "peptidescores") if n_peptide_scores else None
    protein_scores = _mirror.read_table(con, "proteinscores") if n_protein_scores else None
    sources: set[str] = set()
    for table in (peptide_scores, protein_scores):
        if table is not None:
            sources.update(s for s in pc.unique(table.column("SourceFile")).to_pylist() if s)
    acquisitions, acquisition_of = _acquisitions_for(sorted(sources), "encyclopedia.scores", bind)

    precursor_score_table = PRECURSOR_SCORE_TABLE.empty_table()
    peptide_score_table = PEPTIDE_SCORE_TABLE.empty_table()
    if peptide_scores is not None:
        modseqs = peptide_scores.column("PeptideModSeq").to_pylist()
        charges = peptide_scores.column("PrecursorCharge").to_pylist()
        row_sources = peptide_scores.column("SourceFile").to_pylist()
        decoy = peptide_scores.column("IsDecoy").to_pylist()
        _forms, proforma_of = _peptidoforms(list(dict.fromkeys(modseqs)), vocab)
        residue_form: dict[str, str] = {}
        precursor_ids: list[int | None] = []
        ambiguous = 0
        for modseq, charge, source, is_decoy in zip(modseqs, charges, row_sources, decoy, strict=True):
            if is_decoy:
                precursor_ids.append(None)
                continue
            # This table spells a precursor as `entries` does, so the
            # exact spelling finds it. The residue-form key is only a
            # fallback for a file that spells the two tables differently.
            exact = by_spelling.get((modseq, int(charge), source))
            if exact is not None:
                precursor_ids.append(exact)
                continue
            form = residue_form.get(modseq)
            if form is None:
                form = residue_form[modseq] = encyclopedia_residue_form(modseq)
            candidates = by_join_key.get((form, int(charge), source), ())
            if len(candidates) > 1:
                ambiguous += 1
                precursor_ids.append(None)
            else:
                precursor_ids.append(candidates[0] if candidates else None)
        if ambiguous:
            caveats.append(
                f"{ambiguous:,} peptidescores rows fit more than one precursor by spelling and "
                f"carry no precursor_id (their modified_sequence and charge identify them)"
            )
        n = len(modseqs)
        precursor_score_table = pa.table(
            {
                "precursor_id": pa.array(precursor_ids, type=pa.int64()),
                "acquisition_id": pa.array([acquisition_of[s] for s in row_sources], type=pa.int64()),
                "modified_sequence": pa.array([proforma_of[m] for m in modseqs], type=pa.string()),
                "charge": pa.array(charges, type=pa.int32()),
                "score": pa.nulls(n, type=pa.float64()),
                "qvalue": peptide_scores.column("QValue"),
                "pep": peptide_scores.column("PosteriorErrorProbability"),
                "is_decoy": pa.array([bool(d) for d in decoy], type=pa.bool_()),
                "engine": pa.array([engine] * n, type=pa.string()),
            },
            schema=PRECURSOR_SCORE_TABLE,
        )
        # Peptide level: the best q-value across charge states, over the
        # rows that have a precursor in the library.
        peptide_of_precursor = dict(
            zip(
                library.precursors.column("precursor_id").to_pylist(),
                library.precursors.column("peptide_id").to_pylist(),
                strict=True,
            )
        )
        best: dict[tuple[int, int], tuple[float, float]] = {}
        for pid, source, qvalue, pep, is_decoy in zip(
            precursor_ids,
            row_sources,
            peptide_scores.column("QValue").to_pylist(),
            peptide_scores.column("PosteriorErrorProbability").to_pylist(),
            decoy,
            strict=True,
        ):
            if pid is None or (is_decoy and not include_decoys):
                continue
            key = (peptide_of_precursor[pid], acquisition_of[source])
            if key not in best or qvalue < best[key][0]:
                best[key] = (float(qvalue), float(pep))
        peptide_score_table = pa.table(
            {
                "peptide_id": pa.array([k[0] for k in best], type=pa.int64()),
                "acquisition_id": pa.array([k[1] for k in best], type=pa.int64()),
                "score": pa.nulls(len(best), type=pa.float64()),
                "qvalue": pa.array([v[0] for v in best.values()], type=pa.float64()),
                "pep": pa.array([v[1] for v in best.values()], type=pa.float64()),
                "engine": pa.array([engine] * len(best), type=pa.string()),
            },
            schema=PEPTIDE_SCORE_TABLE,
        )

    protein_score_table = PROTEIN_SCORE_TABLE.empty_table()
    if protein_scores is not None:
        protein_id_of = dict(
            zip(
                library.proteins.column("accession").to_pylist(),
                library.proteins.column("protein_id").to_pylist(),
                strict=True,
            )
        )
        # The protein's best q-value in each run, with that row's
        # MinimumPeptidePEP as its score (the jar's protein-level number).
        best_protein: dict[tuple[int, int], tuple[float, float | None]] = {}
        minimum_pep = (
            protein_scores.column("MinimumPeptidePEP").to_pylist()
            if "MinimumPeptidePEP" in protein_scores.column_names
            else [None] * protein_scores.num_rows
        )
        for accession, source, qvalue, is_decoy, score in zip(
            protein_scores.column("ProteinAccession").to_pylist(),
            protein_scores.column("SourceFile").to_pylist(),
            protein_scores.column("QValue").to_pylist(),
            protein_scores.column("IsDecoy").to_pylist(),
            minimum_pep,
            strict=True,
        ):
            if (is_decoy and not include_decoys) or accession not in protein_id_of:
                continue
            key = (protein_id_of[accession], acquisition_of[source])
            if key not in best_protein or qvalue < best_protein[key][0]:
                best_protein[key] = (float(qvalue), None if score is None else float(score))
        protein_score_table = pa.table(
            {
                "protein_id": pa.array([k[0] for k in best_protein], type=pa.int64()),
                "acquisition_id": pa.array([k[1] for k in best_protein], type=pa.int64()),
                "score": pa.array([v[1] for v in best_protein.values()], type=pa.float64()),
                "qvalue": pa.array([v[0] for v in best_protein.values()], type=pa.float64()),
                "engine": pa.array([engine] * len(best_protein), type=pa.string()),
            },
            schema=PROTEIN_SCORE_TABLE,
        )
    return Search(
        acquisitions=acquisitions,
        peptide_scores=peptide_score_table,
        protein_scores=protein_score_table,
        precursor_scores=precursor_score_table,
    )


__all__ = ["ENGINE", "SCRIBETWO_ENGINE", "Projection", "engine_of", "project_library_file"]
