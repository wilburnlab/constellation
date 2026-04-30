"""EncyclopeDIA ``.dlib`` / ``.elib`` writer — the inverse of ``_read``.

Single shared writer; output extension is purely a naming convention
chosen by the caller (DLIB and ELIB are the same SQLite schema). When
``quant`` and / or ``search`` are passed, the corresponding tables /
columns are populated; otherwise they're left empty so the file is a
valid bare DLIB.

Lossy on the chemical-fidelity axis: terminal modifications collapse
onto residue 0 in EncyclopeDIA's grammar — see ``format_encyclopedia_modseq``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.io.encyclopedia import _sql
from constellation.massspec.io.encyclopedia._codec import (
    compress_intensity,
    compress_mz,
)
from constellation.massspec.io.encyclopedia._modseq import (
    format_encyclopedia_modseq,
)
from constellation.massspec.io.encyclopedia._schema import (
    LIBRARY_INDEXES,
    LIBRARY_TABLES,
)
from constellation.massspec.library.library import Library
from constellation.massspec.quant.quant import Quant
from constellation.massspec.search.search import Search

WRITER_VERSION = "constellation/2026.04"


def write_encyclopedia(
    path: Path | str,
    library: Library,
    *,
    quant: Quant | None = None,
    search: Search | None = None,
    overwrite: bool = False,
) -> None:
    """Write a ``Library`` (+ optional ``Quant`` / ``Search``) as a SQLite
    ``.dlib``/``.elib`` file.

    The caller picks the file extension; the on-disk format is identical.
    All 9 tables are created (matching EncyclopeDIA's own behaviour);
    those without provided source data are simply left empty.
    """
    p = Path(path)
    if p.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing {p}")
        p.unlink()

    with _sql.open_rw(p) as con:
        # Build full schema.
        for _name, ddl in LIBRARY_TABLES:
            con.execute(ddl)

        _write_entries(con, library, quant)
        _write_peptidetoprotein(con, library)
        _write_metadata(con, library)
        if search is not None:
            _write_peptidescores(con, library, search)
            _write_proteinscores(con, library, search)
        if quant is not None:
            _write_peptidequants(con, library, quant)

        for stmt in LIBRARY_INDEXES:
            con.execute(stmt)
        con.commit()


# ──────────────────────────────────────────────────────────────────────
# entries — the bulk of the file
# ──────────────────────────────────────────────────────────────────────


_ENTRIES_INSERT_COLS = (
    "PrecursorMz",
    "PrecursorCharge",
    "PeptideModSeq",
    "PeptideSeq",
    "Copies",
    "RTInSeconds",
    "Score",
    "MassEncodedLength",
    "MassArray",
    "IntensityEncodedLength",
    "IntensityArray",
    "CorrelationEncodedLength",
    "CorrelationArray",
    "QuantifiedIonsArray",
    "RTInSecondsStart",
    "RTInSecondsStop",
    "MedianChromatogramEncodedLength",
    "MedianChromatogramArray",
    "SourceFile",
)


def _write_entries(con, library: Library, quant: Quant | None) -> None:
    # Index Library by precursor_id → fragments table view, peptide row,
    # precursor row.
    fragments = library.fragments
    precursors = library.precursors
    peptides = library.peptides

    pid_to_peptide_id = dict(
        zip(
            precursors.column("precursor_id").to_pylist(),
            precursors.column("peptide_id").to_pylist(),
            strict=True,
        )
    )
    pid_to_charge = dict(
        zip(
            precursors.column("precursor_id").to_pylist(),
            precursors.column("charge").to_pylist(),
            strict=True,
        )
    )
    pid_to_mz = dict(
        zip(
            precursors.column("precursor_id").to_pylist(),
            precursors.column("precursor_mz").to_pylist(),
            strict=True,
        )
    )
    pid_to_rt = dict(
        zip(
            precursors.column("precursor_id").to_pylist(),
            precursors.column("rt_predicted").to_pylist(),
            strict=True,
        )
    )
    kid_to_seq = dict(
        zip(
            peptides.column("peptide_id").to_pylist(),
            peptides.column("sequence").to_pylist(),
            strict=True,
        )
    )
    kid_to_modseq = dict(
        zip(
            peptides.column("peptide_id").to_pylist(),
            peptides.column("modified_sequence").to_pylist(),
            strict=True,
        )
    )

    # Cache encyclopedia-modseq strings per peptide_id so we don't
    # parse + format the same ProForma string ``n_charges`` times.
    encyc_modseq: dict[int, str] = {}

    # Group fragment-table rows by precursor_id — each entries row
    # aggregates one precursor's fragments into MassArray/IntensityArray.
    frag_groups: dict[int, list[tuple[float, float]]] = {}
    for prec_id, mz, intensity in zip(
        fragments.column("precursor_id").to_pylist(),
        fragments.column("mz_theoretical").to_pylist(),
        fragments.column("intensity_predicted").to_pylist(),
        strict=True,
    ):
        frag_groups.setdefault(prec_id, []).append((float(mz), float(intensity)))

    # Optional Quant lookup: (precursor_id, source_file) → row dict.
    pq_lookup: dict[tuple[int, str], dict[str, float]] = {}
    sf_for_acq: dict[int, str] = {}
    if quant is not None:
        sf_for_acq = dict(
            zip(
                quant.acquisitions.table.column("acquisition_id").to_pylist(),
                quant.acquisitions.table.column("source_file").to_pylist(),
                strict=True,
            )
        )
        pq = quant.precursor_quant
        for prec_id, acq_id, intensity, rt in zip(
            pq.column("precursor_id").to_pylist(),
            pq.column("acquisition_id").to_pylist(),
            pq.column("intensity").to_pylist(),
            pq.column("rt_observed").to_pylist(),
            strict=True,
        ):
            sf = sf_for_acq.get(acq_id, "")
            pq_lookup[(prec_id, sf)] = {"intensity": intensity, "rt_observed": rt}

    rows: list[tuple[Any, ...]] = []
    for prec_id, peptide_id in pid_to_peptide_id.items():
        # Sort fragments by m/z so the on-disk MassArray is monotonic
        # (matches encyclopedia's own writer).
        frags = sorted(frag_groups.get(prec_id, []), key=lambda x: x[0])
        if not frags:
            mz_arr = torch.zeros(0, dtype=torch.float64)
            int_arr = torch.zeros(0, dtype=torch.float32)
        else:
            mz_arr = torch.tensor([f[0] for f in frags], dtype=torch.float64)
            int_arr = torch.tensor([f[1] for f in frags], dtype=torch.float32)
        mz_blob, mz_len = compress_mz(mz_arr)
        int_blob, int_len = compress_intensity(int_arr)

        if peptide_id not in encyc_modseq:
            modseq_proforma = kid_to_modseq[peptide_id]
            encyc_modseq[peptide_id] = format_encyclopedia_modseq(
                parse_proforma(modseq_proforma)
            )
        peptide_modseq = encyc_modseq[peptide_id]
        peptide_seq = kid_to_seq[peptide_id]

        # Pick the SourceFile from the first matching quant row, or
        # fall back to empty string. A precursor can show up in multiple
        # acquisitions; we emit one entries row per (precursor,
        # source_file) pair the quant table mentions, falling back to
        # one empty-source row otherwise.
        source_files = [sf for (pid, sf) in pq_lookup.keys() if pid == prec_id]
        if not source_files:
            source_files = [""]

        for source_file in source_files:
            qrow = pq_lookup.get((prec_id, source_file), {})
            # Score, RTInSecondsStart/Stop, etc. left as defaults / NULL.
            rows.append(
                (
                    float(pid_to_mz[prec_id]),
                    int(pid_to_charge[prec_id]),
                    peptide_modseq,
                    peptide_seq,
                    1,  # Copies — encyclopedia uses 1 for predicted
                    float(qrow.get("rt_observed", pid_to_rt[prec_id])),
                    0.0,  # Score
                    int(mz_len),
                    mz_blob,
                    int(int_len),
                    int_blob,
                    None,  # CorrelationEncodedLength
                    None,  # CorrelationArray
                    None,  # QuantifiedIonsArray
                    None,  # RTInSecondsStart
                    None,  # RTInSecondsStop
                    None,  # MedianChromatogramEncodedLength
                    None,  # MedianChromatogramArray
                    source_file,
                )
            )

    _sql.insert_many(con, "entries", _ENTRIES_INSERT_COLS, rows)


# ──────────────────────────────────────────────────────────────────────
# peptidetoprotein
# ──────────────────────────────────────────────────────────────────────


def _write_peptidetoprotein(con, library: Library) -> None:
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
    rows = [
        (seq_for[k], False, accession_for[p])
        for p, k in zip(
            library.protein_peptide.column("protein_id").to_pylist(),
            library.protein_peptide.column("peptide_id").to_pylist(),
            strict=True,
        )
    ]
    _sql.insert_many(
        con, "peptidetoprotein", ("PeptideSeq", "isDecoy", "ProteinAccession"), rows
    )


# ──────────────────────────────────────────────────────────────────────
# metadata
# ──────────────────────────────────────────────────────────────────────


def _write_metadata(con, library: Library) -> None:
    rows: list[tuple[str, str]] = [
        ("version", "0.1.15"),  # encyclopedia's own format-version key
        ("constellation_writer_version", WRITER_VERSION),
        ("created_iso", datetime.now(timezone.utc).isoformat()),
    ]
    # Pass through any preserved encyclopedia-side metadata.
    upstream = library.metadata_extras.get("x.encyclopedia.metadata", {})
    for k, v in upstream.items():
        # Avoid clobbering our own keys.
        if k in {"version", "constellation_writer_version", "created_iso"}:
            continue
        rows.append((str(k), str(v)))
    _sql.insert_many(con, "metadata", ("Key", "Value"), rows)


# ──────────────────────────────────────────────────────────────────────
# peptidescores / proteinscores
# ──────────────────────────────────────────────────────────────────────


def _write_peptidescores(con, library: Library, search: Search) -> None:
    if search.peptide_scores.num_rows == 0:
        return
    sf_for = dict(
        zip(
            search.acquisitions.table.column("acquisition_id").to_pylist(),
            search.acquisitions.table.column("source_file").to_pylist(),
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
    modseq_for = dict(
        zip(
            library.peptides.column("peptide_id").to_pylist(),
            library.peptides.column("modified_sequence").to_pylist(),
            strict=True,
        )
    )
    # Encyclopedia's peptidescores has a charge column but our schema
    # is charge-agnostic. Use precursors to enumerate (peptide, charge)
    # pairs and assign each peptide score row per matching charge.
    charges_for: dict[int, list[int]] = {}
    for k, c in zip(
        library.precursors.column("peptide_id").to_pylist(),
        library.precursors.column("charge").to_pylist(),
        strict=True,
    ):
        charges_for.setdefault(k, []).append(int(c))

    rows: list[tuple[Any, ...]] = []
    for k, acq_id, qvalue, pep in zip(
        search.peptide_scores.column("peptide_id").to_pylist(),
        search.peptide_scores.column("acquisition_id").to_pylist(),
        search.peptide_scores.column("qvalue").to_pylist(),
        search.peptide_scores.column("pep").to_pylist(),
        strict=True,
    ):
        sf = sf_for.get(acq_id, "")
        peptide_modseq = format_encyclopedia_modseq(parse_proforma(modseq_for[k]))
        peptide_seq = seq_for[k]
        for charge in charges_for.get(k, [2]):  # default to 2+ if no precursor known
            rows.append(
                (
                    int(charge),
                    peptide_modseq,
                    peptide_seq,
                    sf,
                    float(qvalue) if qvalue is not None else 1.0,
                    float(pep) if pep is not None else 1.0,
                    False,
                )
            )
    _sql.insert_many(
        con,
        "peptidescores",
        (
            "PrecursorCharge",
            "PeptideModSeq",
            "PeptideSeq",
            "SourceFile",
            "QValue",
            "PosteriorErrorProbability",
            "IsDecoy",
        ),
        rows,
    )


def _write_proteinscores(con, library: Library, search: Search) -> None:
    if search.protein_scores.num_rows == 0:
        return
    sf_for = dict(
        zip(
            search.acquisitions.table.column("acquisition_id").to_pylist(),
            search.acquisitions.table.column("source_file").to_pylist(),
            strict=True,
        )
    )
    accession_for = dict(
        zip(
            library.proteins.column("protein_id").to_pylist(),
            library.proteins.column("accession").to_pylist(),
            strict=True,
        )
    )
    rows = [
        (
            int(p),  # ProteinGroup — round-trippable as the protein_id
            accession_for[p],
            sf_for.get(acq, ""),
            float(qvalue) if qvalue is not None else 1.0,
            1.0,  # MinimumPeptidePEP — round-tripping at the protein level requires search-side data we don't model yet
            False,
        )
        for p, acq, qvalue in zip(
            search.protein_scores.column("protein_id").to_pylist(),
            search.protein_scores.column("acquisition_id").to_pylist(),
            search.protein_scores.column("qvalue").to_pylist(),
            strict=True,
        )
    ]
    _sql.insert_many(
        con,
        "proteinscores",
        (
            "ProteinGroup",
            "ProteinAccession",
            "SourceFile",
            "QValue",
            "MinimumPeptidePEP",
            "IsDecoy",
        ),
        rows,
    )


# ──────────────────────────────────────────────────────────────────────
# peptidequants
# ──────────────────────────────────────────────────────────────────────


def _write_peptidequants(con, library: Library, quant: Quant) -> None:
    if quant.peptide_quant.num_rows == 0:
        return
    sf_for = dict(
        zip(
            quant.acquisitions.table.column("acquisition_id").to_pylist(),
            quant.acquisitions.table.column("source_file").to_pylist(),
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
    modseq_for = dict(
        zip(
            library.peptides.column("peptide_id").to_pylist(),
            library.peptides.column("modified_sequence").to_pylist(),
            strict=True,
        )
    )
    charges_for: dict[int, list[int]] = {}
    for k, c in zip(
        library.precursors.column("peptide_id").to_pylist(),
        library.precursors.column("charge").to_pylist(),
        strict=True,
    ):
        charges_for.setdefault(k, []).append(int(c))

    # Empty quant-mass / quant-int blobs (we don't preserve those at
    # round-trip parity yet).
    empty_mz_blob, empty_mz_len = compress_mz(torch.zeros(0, dtype=torch.float64))
    empty_int_blob, empty_int_len = compress_intensity(
        torch.zeros(0, dtype=torch.float32)
    )

    rows: list[tuple[Any, ...]] = []
    for k, acq_id, abundance in zip(
        quant.peptide_quant.column("peptide_id").to_pylist(),
        quant.peptide_quant.column("acquisition_id").to_pylist(),
        quant.peptide_quant.column("abundance").to_pylist(),
        strict=True,
    ):
        sf = sf_for.get(acq_id, "")
        peptide_modseq = format_encyclopedia_modseq(parse_proforma(modseq_for[k]))
        peptide_seq = seq_for[k]
        for charge in charges_for.get(k, [2]):
            rows.append(
                (
                    int(charge),
                    peptide_modseq,
                    peptide_seq,
                    sf,
                    0.0,  # RTInSecondsCenter
                    0.0,  # Start
                    0.0,  # Stop
                    float(abundance),
                    0,  # NumberOfQuantIons
                    int(empty_mz_len),
                    empty_mz_blob,
                    int(empty_int_len),
                    empty_int_blob,
                    0.0,  # BestFragmentCorrelation
                    0.0,  # BestFragmentDeltaMassPPM
                    int(empty_int_len),
                    empty_int_blob,  # MedianChromatogramArray
                    None,  # MedianChromatogramRTEncodedLength
                    None,  # MedianChromatogramRTArray
                    1.0,  # IdentifiedTICRatio
                )
            )
    _sql.insert_many(
        con,
        "peptidequants",
        (
            "PrecursorCharge",
            "PeptideModSeq",
            "PeptideSeq",
            "SourceFile",
            "RTInSecondsCenter",
            "RTInSecondsStart",
            "RTInSecondsStop",
            "TotalIntensity",
            "NumberOfQuantIons",
            "QuantIonMassLength",
            "QuantIonMassArray",
            "QuantIonIntensityLength",
            "QuantIonIntensityArray",
            "BestFragmentCorrelation",
            "BestFragmentDeltaMassPPM",
            "MedianChromatogramEncodedLength",
            "MedianChromatogramArray",
            "MedianChromatogramRTEncodedLength",
            "MedianChromatogramRTArray",
            "IdentifiedTICRatio",
        ),
        rows,
    )


__all__ = ["write_encyclopedia", "WRITER_VERSION"]
