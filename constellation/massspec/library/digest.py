"""FASTA / peptide list → the precursor grid a predictor works from.

Backend-agnostic on purpose: this sits beside ``library.py`` rather than
inside ``koina/`` because an in-process PyTorch predictor, or any future
prediction backend, needs exactly the same enumeration. Nothing here
knows a network service exists.

The heavy lifting is delegated, not reimplemented — ``cleave`` for
digestion, ``enumerate_modforms`` for the fixed/variable mod grid, and
``precursor_mz`` for the charge sweep.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.protein import cleave, enumerate_modforms
from constellation.core.sequence.proforma import format_proforma, parse_proforma
from constellation.massspec.peptide.mz import precursor_mz

#: EncyclopeDIA's defaults, so the two backends enumerate the same grid
#: unless the caller says otherwise.
DEFAULT_MIN_MZ = 396.4
DEFAULT_MAX_MZ = 1002.7


@dataclass(frozen=True, slots=True)
class PrecursorSpec:
    """One (peptidoform, charge) the predictor will be asked about."""

    modified_sequence: str  # ProForma 2.0
    sequence: str  # bare residues
    charge: int
    precursor_mz: float
    proteins: tuple[str, ...] = ()  # accessions; empty for orphan peptides


def _specs_for_modforms(
    modforms: Iterable,
    charges: Sequence[int],
    *,
    min_mz: float,
    max_mz: float,
    proteins: tuple[str, ...],
    vocab: ModVocab,
) -> list[PrecursorSpec]:
    out: list[PrecursorSpec] = []
    for form in modforms:
        modseq = format_proforma(form)
        for z in charges:
            mz = precursor_mz(form, charge=z, vocab=vocab)
            if not (min_mz <= mz <= max_mz):
                continue
            out.append(
                PrecursorSpec(
                    modified_sequence=modseq,
                    sequence=form.sequence,
                    charge=z,
                    precursor_mz=mz,
                    proteins=proteins,
                )
            )
    return out


def precursors_from_fasta(
    fasta: Path | str,
    *,
    protease: str = "Trypsin",
    missed_cleavages: int = 1,
    min_missed_cleavages: int = 0,
    min_length: int = 6,
    max_length: int = 30,
    excise_initiator_met: bool = True,
    fixed_mods: Mapping[str, str] | None = None,
    variable_mods: Mapping[str, str | Sequence[str]] | None = None,
    max_variable_mods: int = 1,
    charges: Sequence[int] = (2, 3),
    min_mz: float = DEFAULT_MIN_MZ,
    max_mz: float = DEFAULT_MAX_MZ,
    vocab: ModVocab = UNIMOD,
) -> list[PrecursorSpec]:
    """Digest a FASTA into a deduplicated precursor grid.

    Peptides shared between proteins collapse to one entry carrying every
    parent accession, matching ``PROTEIN_PEPTIDE_EDGE``'s M:N semantics —
    predicting the same peptidoform once per protein would be wasted
    inference and would break the Library's PK uniqueness.

    ``missed_cleavages`` is the *maximum*; ``min_missed_cleavages`` is the
    matching floor, so ``min_missed_cleavages=1, missed_cleavages=1``
    yields only the singly-missed peptides. A peptide reachable at more
    than one missed-cleavage count (the same residues spanning different
    cut sites) is kept if any of its spans satisfies the floor.
    """
    if min_missed_cleavages < 0:
        raise ValueError(
            f"min_missed_cleavages must be >= 0, got {min_missed_cleavages}"
        )
    if min_missed_cleavages > missed_cleavages:
        raise ValueError(
            f"min_missed_cleavages ({min_missed_cleavages}) exceeds "
            f"missed_cleavages ({missed_cleavages})"
        )
    from constellation.massspec.search.novel import read_fasta_proteins

    table = read_fasta_proteins(fasta)
    accessions = table.column("protein_id").to_pylist()
    sequences = table.column("sequence").to_pylist()

    # peptide sequence → the accessions it came from, first-seen order.
    peptide_proteins: dict[str, list[str]] = {}
    for accession, protein_seq in zip(accessions, sequences, strict=True):
        # Spans rather than bare sequences so `n_missed` is available to
        # filter on; `cleave` sorts them by (start, end) and the
        # setdefault below dedups on sequence, reproducing the bare-string
        # path's first-seen ordering exactly.
        peptides = cleave(
            protein_seq,
            protease,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            excise_initiator_met=excise_initiator_met,
            validate_alphabet=False,
            return_spans=True,
        )
        for peptide in peptides:
            if peptide.n_missed < min_missed_cleavages:
                continue
            owners = peptide_proteins.setdefault(peptide.sequence, [])
            if accession not in owners:
                owners.append(accession)

    specs: list[PrecursorSpec] = []
    for peptide, owners in peptide_proteins.items():
        modforms = enumerate_modforms(
            peptide,
            fixed=fixed_mods,
            variable=variable_mods,
            max_variable=max_variable_mods,
            vocab=vocab,
        )
        specs.extend(
            _specs_for_modforms(
                modforms,
                charges,
                min_mz=min_mz,
                max_mz=max_mz,
                proteins=tuple(owners),
                vocab=vocab,
            )
        )
    return specs


def precursors_from_peptide_list(
    path: Path | str,
    *,
    charges: Sequence[int] = (2, 3),
    min_mz: float = 0.0,
    max_mz: float = float("inf"),
    vocab: ModVocab = UNIMOD,
) -> list[PrecursorSpec]:
    """Read an explicit peptide list — no digestion, no mod enumeration.

    Accepts TSV/CSV or parquet with a ``modified_sequence`` column (a
    ProForma 2.0 string) and an optional ``charge`` column; rows without
    a charge are expanded across ``charges``. m/z bounds default to
    unbounded, because a caller naming exact peptides means them — this
    is the path for "predict these 167 targets", where silently dropping
    some to an m/z window would defeat the purpose.
    """
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        table = pq.read_table(p)
    else:
        from pyarrow import csv as pa_csv

        delimiter = "," if p.suffix.lower() == ".csv" else "\t"
        table = pa_csv.read_csv(
            p, parse_options=pa_csv.ParseOptions(delimiter=delimiter)
        )

    if "modified_sequence" not in table.column_names:
        raise ValueError(
            f"{p} has no 'modified_sequence' column; found "
            f"{table.column_names}"
        )

    modseqs = table.column("modified_sequence").to_pylist()
    if "charge" in table.column_names:
        raw_charges = table.column("charge").to_pylist()
    else:
        raw_charges = [None] * len(modseqs)

    specs: list[PrecursorSpec] = []
    seen: set[tuple[str, int]] = set()
    for modseq, charge in zip(modseqs, raw_charges, strict=True):
        if modseq is None:
            continue
        form = parse_proforma(modseq)
        wanted = (int(charge),) if charge is not None else tuple(charges)
        for z in wanted:
            if (modseq, z) in seen:
                continue
            seen.add((modseq, z))
            mz = precursor_mz(form, charge=z, vocab=vocab)
            if not (min_mz <= mz <= max_mz):
                continue
            specs.append(
                PrecursorSpec(
                    modified_sequence=format_proforma(form),
                    sequence=form.sequence,
                    charge=z,
                    precursor_mz=mz,
                )
            )
    return specs


def precursors_from_library(path: Path | str) -> list[PrecursorSpec]:
    """Take an existing Library's exact precursors, for re-prediction.

    The point of this path is swapping the *model* while holding the
    peptide set fixed — e.g. re-predicting an EncyclopeDIA-built HCD
    library's precursors under a CID model, which is the only way to get
    CID reference spectra for a panel that was defined under HCD.
    """
    from constellation.massspec.library.io import load_library

    lib = load_library(path)
    peptides = {
        row["peptide_id"]: row["modified_sequence"]
        for row in lib.peptides.select(["peptide_id", "modified_sequence"]).to_pylist()
    }
    edges: dict[int, list[str]] = {}
    accession_by_id = {
        row["protein_id"]: row["accession"]
        for row in lib.proteins.select(["protein_id", "accession"]).to_pylist()
    }
    for row in lib.protein_peptide.to_pylist():
        edges.setdefault(row["peptide_id"], []).append(
            accession_by_id[row["protein_id"]]
        )

    specs: list[PrecursorSpec] = []
    for row in lib.precursors.select(
        ["peptide_id", "charge", "precursor_mz"]
    ).to_pylist():
        modseq = peptides[row["peptide_id"]]
        specs.append(
            PrecursorSpec(
                modified_sequence=modseq,
                sequence=parse_proforma(modseq).sequence,
                charge=int(row["charge"]),
                precursor_mz=float(row["precursor_mz"]),
                proteins=tuple(edges.get(row["peptide_id"], ())),
            )
        )
    return specs


def specs_to_table(specs: Sequence[PrecursorSpec]) -> pa.Table:
    """Precursor grid as Arrow, for logging and manifest summaries."""
    return pa.table(
        {
            "modified_sequence": [s.modified_sequence for s in specs],
            "sequence": [s.sequence for s in specs],
            "charge": pa.array([s.charge for s in specs], pa.int32()),
            "precursor_mz": pa.array([s.precursor_mz for s in specs], pa.float64()),
            "proteins": [list(s.proteins) for s in specs],
        }
    )


__all__ = [
    "DEFAULT_MAX_MZ",
    "DEFAULT_MIN_MZ",
    "PrecursorSpec",
    "precursors_from_fasta",
    "precursors_from_library",
    "precursors_from_peptide_list",
    "specs_to_table",
]
