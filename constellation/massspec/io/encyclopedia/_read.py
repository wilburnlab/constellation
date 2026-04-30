"""EncyclopeDIA ``.dlib`` / ``.elib`` reader — produces ``(Library, Quant?, Search?)``.

DLIB and ELIB share one SQLite container schema; the ``Quant`` and
``Search`` halves get built only when the corresponding tables /
columns are populated. The reader is one piece of code that returns
all three projections; ``adapters.py`` slices out whichever one a
caller asked for via the Library / Quant / Search Reader Protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import format_proforma
from constellation.massspec.acquisitions import (
    ACQUISITION_TABLE,
    Acquisitions,
)
from constellation.massspec.io.encyclopedia import _sql
from constellation.massspec.io.encyclopedia._annotate import annotate_peaks
from constellation.massspec.io.encyclopedia._codec import (
    decompress_intensity,
    decompress_mz,
)
from constellation.massspec.io.encyclopedia._modseq import (
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
from constellation.massspec.quant.quant import Quant
from constellation.massspec.quant.schemas import (
    PEPTIDE_QUANT,
    PRECURSOR_QUANT,
    PROTEIN_QUANT,
    TRANSMISSION_PEPTIDE_PRECURSOR,
    TRANSMISSION_PROTEIN_PEPTIDE,
)
from constellation.massspec.search.schemas import (
    PEPTIDE_SCORE_TABLE,
    PROTEIN_SCORE_TABLE,
)
from constellation.massspec.search.search import Search


@dataclass(frozen=True, slots=True)
class EncyclopediaReadResult:
    """All three projections of one ``.dlib`` / ``.elib`` file.

    ``library`` is always populated. ``quant`` is populated when the
    file carries any sample-specific data (chromatogram blob in
    ``entries`` rows, populated ``peptidequants`` / ``fragmentquants``
    / ``retentiontimes`` tables). ``search`` is populated when
    ``peptidescores`` or ``proteinscores`` carry any rows.
    """

    library: Library
    quant: Quant | None
    search: Search | None


def read_encyclopedia(
    path: Path | str,
    *,
    annotate_fragments: bool = True,
    fragment_tolerance_ppm: float = 20.0,
    enabled_modifications: ModVocab = UNIMOD,
    include_decoys: bool = False,
) -> EncyclopediaReadResult:
    """Read an EncyclopeDIA ``.dlib`` / ``.elib`` file.

    Parameters
    ----------
    path
        SQLite file on disk.
    annotate_fragments
        Whether to re-annotate decoded peaks against each peptidoform's
        theoretical b/y ladder (cheap, recommended). When ``False``,
        the ``Library.fragments`` table is empty.
    fragment_tolerance_ppm
        Tolerance for the fragment annotation. 20 ppm is the default
        EncyclopeDIA tolerance.
    enabled_modifications
        UNIMOD vocabulary used for parsing ``PeptideModSeq``. Use a
        subset to constrain modification interpretation.
    include_decoys
        When ``False`` (default), rows where ``peptidetoprotein.isDecoy``
        is true are dropped from the protein/peptide tables.
    """
    p = Path(path)
    with _sql.open_ro(p) as con:
        meta = _sql.fetch_metadata(con)

        # ── Phase A: Library ───────────────────────────────────────
        library, peptide_id_for, precursor_id_for = _build_library(
            con,
            annotate_fragments=annotate_fragments,
            fragment_tolerance_ppm=fragment_tolerance_ppm,
            vocab=enabled_modifications,
            include_decoys=include_decoys,
            metadata_extras={"x.encyclopedia.metadata": meta},
        )

        # ── Phase B: Quant (conditional) ───────────────────────────
        quant = _build_quant(con, peptide_id_for, precursor_id_for)

        # ── Phase C: Search (conditional) ──────────────────────────
        search = _build_search(con, library, peptide_id_for, include_decoys)

    return EncyclopediaReadResult(library=library, quant=quant, search=search)


# ──────────────────────────────────────────────────────────────────────
# Library construction
# ──────────────────────────────────────────────────────────────────────


def _build_library(
    con,
    *,
    annotate_fragments: bool,
    fragment_tolerance_ppm: float,
    vocab: ModVocab,
    include_decoys: bool,
    metadata_extras: dict[str, Any],
) -> tuple[Library, dict, dict]:
    """Build the Library + return id-mapping dicts the Quant/Search build need."""
    # 1. peptidetoprotein → distinct accessions + edges.
    accession_to_id: dict[str, int] = {}
    pp_pairs: list[tuple[str, str]] = []  # (accession, peptide_seq)
    for row in _sql.iter_peptidetoprotein(con):
        if not include_decoys and row["isDecoy"]:
            continue
        acc = row["ProteinAccession"]
        if acc not in accession_to_id:
            accession_to_id[acc] = len(accession_to_id)
        pp_pairs.append((acc, row["PeptideSeq"]))

    proteins_table = pa.Table.from_pylist(
        [
            {
                "protein_id": pid,
                "accession": acc,
                "sequence": None,
                "description": None,
            }
            for acc, pid in sorted(accession_to_id.items(), key=lambda kv: kv[1])
        ],
        schema=PROTEIN_TABLE,
    )

    # 2. entries → peptides + precursors + fragments.
    # Walk entries once, keep parsed Peptidoform objects so the fragment
    # re-annotation step doesn't have to re-parse.
    peptide_id_for: dict[tuple[str, str], int] = {}
    peptidoform_for: dict[int, Any] = {}
    peptide_rows: list[dict[str, Any]] = []
    precursor_id_for: dict[tuple[int, int, str], int] = {}  # (peptide_id, charge, source) → precursor_id
    precursor_rows: list[dict[str, Any]] = []
    fragment_rows: list[dict[str, Any]] = []

    valid_peptide_seqs = {seq for _acc, seq in pp_pairs}

    for row in _sql.iter_entries(con):
        seq = row["PeptideSeq"]
        if not include_decoys and seq not in valid_peptide_seqs:
            # Entry references a peptide that's only present as a decoy
            # in peptidetoprotein → skip in lockstep.
            continue
        modseq_raw = row["PeptideModSeq"]
        peptidoform = parse_encyclopedia_modseq(modseq_raw, vocab=vocab)
        modseq_proforma = format_proforma(peptidoform)
        key = (seq, modseq_proforma)
        if key not in peptide_id_for:
            pid = len(peptide_id_for)
            peptide_id_for[key] = pid
            peptidoform_for[pid] = peptidoform
            peptide_rows.append(
                {
                    "peptide_id": pid,
                    "sequence": seq,
                    "modified_sequence": modseq_proforma,
                }
            )
        peptide_id = peptide_id_for[key]
        charge = int(row["PrecursorCharge"])
        source_file = row["SourceFile"]
        prec_key = (peptide_id, charge, source_file)
        if prec_key not in precursor_id_for:
            cid = len(precursor_id_for)
            precursor_id_for[prec_key] = cid
            precursor_rows.append(
                {
                    "precursor_id": cid,
                    "peptide_id": peptide_id,
                    "charge": charge,
                    "precursor_mz": float(row["PrecursorMz"]),
                    "rt_predicted": float(row["RTInSeconds"]),
                    "ccs_predicted": -1.0,
                }
            )
        else:
            cid = precursor_id_for[prec_key]

        if annotate_fragments:
            mz_t = decompress_mz(row["MassArray"], int(row["MassEncodedLength"]))
            int_t = decompress_intensity(
                row["IntensityArray"], int(row["IntensityEncodedLength"])
            )
            for frag in annotate_peaks(
                peptidoform,
                precursor_charge=charge,
                obs_mz=mz_t,
                obs_intensity=int_t,
                tolerance_ppm=fragment_tolerance_ppm,
            ):
                frag["precursor_id"] = cid
                fragment_rows.append(frag)

    peptides_table = pa.Table.from_pylist(peptide_rows, schema=PEPTIDE_TABLE) if peptide_rows else PEPTIDE_TABLE.empty_table()
    precursors_table = pa.Table.from_pylist(precursor_rows, schema=PRECURSOR_TABLE) if precursor_rows else PRECURSOR_TABLE.empty_table()
    fragments_table = pa.Table.from_pylist(fragment_rows, schema=LIBRARY_FRAGMENT_TABLE) if fragment_rows else LIBRARY_FRAGMENT_TABLE.empty_table()

    # 3. Build PROTEIN_PEPTIDE_EDGE; deduplicate (acc, peptide_seq) pairs
    # since a peptide can appear in peptidetoprotein with both decoy and
    # target rows for the same accession.
    edges: set[tuple[int, int]] = set()
    for acc, seq in pp_pairs:
        # A peptide may appear under multiple charge / modified states;
        # protein↔peptide adjacency is at the canonical-sequence level.
        for (pseq, _modseq), pid in peptide_id_for.items():
            if pseq == seq:
                edges.add((accession_to_id[acc], pid))
    pp_table = pa.Table.from_pylist(
        [{"protein_id": p, "peptide_id": k} for p, k in sorted(edges)],
        schema=PROTEIN_PEPTIDE_EDGE,
    ) if edges else PROTEIN_PEPTIDE_EDGE.empty_table()

    library = Library(
        proteins=proteins_table,
        peptides=peptides_table,
        precursors=precursors_table,
        fragments=fragments_table,
        protein_peptide=pp_table,
        metadata_extras=metadata_extras,
    )
    return library, peptide_id_for, precursor_id_for


# ──────────────────────────────────────────────────────────────────────
# Quant construction
# ──────────────────────────────────────────────────────────────────────


def _build_quant(
    con,
    peptide_id_for: dict[tuple[str, str], int],
    precursor_id_for: dict[tuple[int, int, str], int],
) -> Quant | None:
    """Build a Quant if any sample-specific data is present, else None."""
    has_chrom = _entries_have_chromatograms(con)
    has_pq = _sql.has_rows(con, "peptidequants")
    has_fq = _sql.has_rows(con, "fragmentquants")
    has_rt = _sql.has_rows(con, "retentiontimes")
    if not (has_chrom or has_pq or has_fq or has_rt):
        return None

    # Acquisitions: distinct entries.SourceFile values.
    source_files = sorted(
        {src for (_pid, _ch, src) in precursor_id_for.keys() if src}
    )
    if not source_files:
        return None
    acq_rows = [
        {
            "acquisition_id": i,
            "source_file": sf,
            "source_kind": "encyclopedia.entries",
            "acquisition_datetime": None,
        }
        for i, sf in enumerate(source_files)
    ]
    source_to_acq: dict[str, int] = {sf: i for i, sf in enumerate(source_files)}
    acq_table = pa.Table.from_pylist(acq_rows, schema=ACQUISITION_TABLE)
    acquisitions = Acquisitions(acq_table)

    # PRECURSOR_QUANT: one row per (precursor_id, acquisition_id),
    # intensity = sum of decoded IntensityArray, rt_observed = RTInSeconds.
    pq_rows: list[dict[str, Any]] = []
    seen_pq: set[tuple[int, int]] = set()  # (precursor_id, acquisition_id)
    for row in _sql.iter_entries(con):
        seq = row["PeptideSeq"]
        # Reconstruct the same peptide_id mapping: we need modseq_proforma.
        # Re-parse here is fine — encyclopedia files are small enough that
        # the cost is dominated by blob decompression, not modseq parsing.
        modseq_proforma = format_proforma(parse_encyclopedia_modseq(row["PeptideModSeq"]))
        key = (seq, modseq_proforma)
        if key not in peptide_id_for:
            # Decoy that we filtered out in the library build.
            continue
        peptide_id = peptide_id_for[key]
        charge = int(row["PrecursorCharge"])
        source_file = row["SourceFile"]
        prec_key = (peptide_id, charge, source_file)
        if prec_key not in precursor_id_for:
            continue
        precursor_id = precursor_id_for[prec_key]
        acq_id = source_to_acq[source_file]
        if (precursor_id, acq_id) in seen_pq:
            continue
        seen_pq.add((precursor_id, acq_id))
        # Intensity = sum of the decoded IntensityArray (peak-list TIC).
        int_t = decompress_intensity(
            row["IntensityArray"], int(row["IntensityEncodedLength"])
        )
        pq_rows.append(
            {
                "precursor_id": precursor_id,
                "acquisition_id": acq_id,
                "intensity": float(int_t.sum().item()),
                "rt_observed": float(row["RTInSeconds"]),
                "ccs_observed": -1.0,
                "score": None,
            }
        )

    pq_table = pa.Table.from_pylist(pq_rows, schema=PRECURSOR_QUANT) if pq_rows else PRECURSOR_QUANT.empty_table()

    # PEPTIDE_QUANT: from peptidequants table when populated, else
    # rolled up from PRECURSOR_QUANT.
    if has_pq:
        peptide_quant_rows = []
        seen_pkq: set[tuple[int, int]] = set()
        for row in _sql.iter_peptidequants(con):
            modseq_proforma = format_proforma(
                parse_encyclopedia_modseq(row["PeptideModSeq"])
            )
            key = (row["PeptideSeq"], modseq_proforma)
            if key not in peptide_id_for:
                continue
            peptide_id = peptide_id_for[key]
            source_file = row["SourceFile"]
            if source_file not in source_to_acq:
                continue
            acq_id = source_to_acq[source_file]
            if (peptide_id, acq_id) in seen_pkq:
                # peptidequants can have multiple rows per (peptide,
                # source) when the peptide appears at multiple charge
                # states. Sum intensities across charges.
                # Find the existing row and accumulate.
                for r in peptide_quant_rows:
                    if r["peptide_id"] == peptide_id and r["acquisition_id"] == acq_id:
                        r["abundance"] += float(row["TotalIntensity"])
                        break
                continue
            seen_pkq.add((peptide_id, acq_id))
            peptide_quant_rows.append(
                {
                    "peptide_id": peptide_id,
                    "acquisition_id": acq_id,
                    "abundance": float(row["TotalIntensity"]),
                    "score": None,
                }
            )
        peptide_quant_table = pa.Table.from_pylist(peptide_quant_rows, schema=PEPTIDE_QUANT) if peptide_quant_rows else PEPTIDE_QUANT.empty_table()
    else:
        peptide_quant_table = PEPTIDE_QUANT.empty_table()

    return Quant(
        acquisitions=acquisitions,
        protein_quant=PROTEIN_QUANT.empty_table(),
        peptide_quant=peptide_quant_table,
        precursor_quant=pq_table,
        transmission_protein_peptide=TRANSMISSION_PROTEIN_PEPTIDE.empty_table(),
        transmission_peptide_precursor=TRANSMISSION_PEPTIDE_PRECURSOR.empty_table(),
    )


def _entries_have_chromatograms(con) -> bool:
    """Cheap probe: any entries.MedianChromatogramArray non-NULL?"""
    cur = con.execute(
        "SELECT 1 FROM entries WHERE MedianChromatogramArray IS NOT NULL LIMIT 1"
    )
    return cur.fetchone() is not None


# ──────────────────────────────────────────────────────────────────────
# Search construction
# ──────────────────────────────────────────────────────────────────────


def _build_search(
    con,
    library: Library,
    peptide_id_for: dict[tuple[str, str], int],
    include_decoys: bool,
) -> Search | None:
    has_pep_scores = _sql.has_rows(con, "peptidescores")
    has_prot_scores = _sql.has_rows(con, "proteinscores")
    if not (has_pep_scores or has_prot_scores):
        return None

    # Gather distinct source files for Acquisitions.
    source_files: set[str] = set()
    pep_score_rows: list[dict[str, Any]] = []
    prot_score_rows: list[dict[str, Any]] = []

    if has_pep_scores:
        for row in _sql.iter_peptidescores(con):
            if not include_decoys and row["IsDecoy"]:
                continue
            source_files.add(row["SourceFile"])
        # Need acquisitions before we can FK; build it after gathering.

    if has_prot_scores:
        for row in _sql.iter_proteinscores(con):
            if not include_decoys and row["IsDecoy"]:
                continue
            source_files.add(row["SourceFile"])

    sf_list = sorted(source_files)
    source_to_acq = {sf: i for i, sf in enumerate(sf_list)}
    acq_table = pa.Table.from_pylist(
        [
            {
                "acquisition_id": i,
                "source_file": sf,
                "source_kind": "encyclopedia.scores",
                "acquisition_datetime": None,
            }
            for i, sf in enumerate(sf_list)
        ],
        schema=ACQUISITION_TABLE,
    )
    acquisitions = Acquisitions(acq_table)

    # peptide_scores: one row per (peptide_id, acquisition_id, engine).
    # peptidescores has one row per (peptide, charge, source) — collapse
    # across charges by keeping the best (lowest q-value) score.
    if has_pep_scores:
        best: dict[tuple[int, int], dict[str, Any]] = {}
        for row in _sql.iter_peptidescores(con):
            if not include_decoys and row["IsDecoy"]:
                continue
            modseq_proforma = format_proforma(
                parse_encyclopedia_modseq(row["PeptideModSeq"])
            )
            key = (row["PeptideSeq"], modseq_proforma)
            if key not in peptide_id_for:
                continue
            peptide_id = peptide_id_for[key]
            acq_id = source_to_acq[row["SourceFile"]]
            qvalue = float(row["QValue"])
            pep = float(row["PosteriorErrorProbability"])
            existing = best.get((peptide_id, acq_id))
            if existing is None or qvalue < existing["qvalue"]:
                best[(peptide_id, acq_id)] = {
                    "peptide_id": peptide_id,
                    "acquisition_id": acq_id,
                    "score": None,
                    "qvalue": qvalue,
                    "pep": pep,
                    "engine": "encyclopedia",
                }
        pep_score_rows = list(best.values())

    pep_score_table = (
        pa.Table.from_pylist(pep_score_rows, schema=PEPTIDE_SCORE_TABLE)
        if pep_score_rows
        else PEPTIDE_SCORE_TABLE.empty_table()
    )

    # protein_scores: collapse across protein groups by accession; keep
    # best q-value.
    if has_prot_scores:
        accession_to_id = {
            acc: pid
            for acc, pid in zip(
                library.proteins.column("accession").to_pylist(),
                library.proteins.column("protein_id").to_pylist(),
                strict=True,
            )
        }
        best_prot: dict[tuple[int, int], dict[str, Any]] = {}
        for row in _sql.iter_proteinscores(con):
            if not include_decoys and row["IsDecoy"]:
                continue
            acc = row["ProteinAccession"]
            if acc not in accession_to_id:
                continue
            protein_id = accession_to_id[acc]
            acq_id = source_to_acq[row["SourceFile"]]
            qvalue = float(row["QValue"])
            existing = best_prot.get((protein_id, acq_id))
            if existing is None or qvalue < existing["qvalue"]:
                best_prot[(protein_id, acq_id)] = {
                    "protein_id": protein_id,
                    "acquisition_id": acq_id,
                    "score": None,
                    "qvalue": qvalue,
                    "engine": "encyclopedia",
                }
        prot_score_rows = list(best_prot.values())

    prot_score_table = (
        pa.Table.from_pylist(prot_score_rows, schema=PROTEIN_SCORE_TABLE)
        if prot_score_rows
        else PROTEIN_SCORE_TABLE.empty_table()
    )

    return Search(
        acquisitions=acquisitions,
        peptide_scores=pep_score_table,
        protein_scores=prot_score_table,
    )


__all__ = ["EncyclopediaReadResult", "read_encyclopedia"]
