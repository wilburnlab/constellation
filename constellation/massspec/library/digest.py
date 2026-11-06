"""FASTA / peptide list → the precursor grid a predictor works from.

Backend-agnostic on purpose: this sits beside ``library.py`` rather than
inside ``koina/`` because an in-process PyTorch predictor, or any future
prediction backend, needs exactly the same enumeration. Nothing here
knows a network service exists.

A FASTA is enumerated by :mod:`constellation.massspec.library.search_space`,
the package's one enumerator (2026-11-05); :func:`precursors_from_fasta` is
a projection of the :class:`~constellation.massspec.library.Library` it
builds into the flat ``PrecursorSpec`` list a predictor works from. The
peptide-list and library paths parse what they are given and compute m/z
with ``precursor_mz``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.core.chem.modifications import UNIMOD, ModVocab
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


def precursors_from_fasta(
    fasta: Path | str,
    *,
    protease: str = "Trypsin",
    missed_cleavages: int = 1,
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

    The search space of ``fasta`` with no decoys and no budget
    (:func:`~constellation.massspec.library.search_space.build_search_space`),
    flattened: one spec per precursor, in the order the search space holds
    them, ``proteins`` each peptide's accessions in FASTA order. A
    modification is placed only where UNIMOD lists it, and a peptide with a
    residue outside the 20 is left out (it has no mass).
    """
    from constellation.massspec.library.search_space import (
        Digestion,
        SearchSpaceSettings,
        build_search_space,
        rules_from_mapping,
    )

    settings = SearchSpaceSettings(
        digestion=Digestion(
            protease=protease,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            excise_initiator_met=excise_initiator_met,
        ),
        rules=rules_from_mapping(fixed_mods, variable_mods, vocab=vocab),
        max_variable=max_variable_mods,
        charges=tuple(charges),
        min_mz=min_mz,
        max_mz=max_mz,
        decoys=(),
        max_peptidoforms=None,
        max_precursors=None,
    )
    library = build_search_space([fasta], settings, vocab=vocab)
    assert not isinstance(library, dict)
    return _specs_from_library(library)


def _specs_from_library(library) -> list[PrecursorSpec]:
    accession_of = dict(
        zip(library.proteins.column("protein_id").to_pylist(), library.proteins.column("accession").to_pylist(), strict=True)
    )
    proteins_of: dict[int, list[str]] = {}
    for protein, peptide in zip(
        library.protein_peptide.column("protein_id").to_pylist(),
        library.protein_peptide.column("peptide_id").to_pylist(),
        strict=True,
    ):
        owners = proteins_of.setdefault(peptide, [])
        acc = accession_of[protein]
        if acc not in owners:
            owners.append(acc)
    peptide_rows = {
        r["peptide_id"]: r for r in library.peptides.select(["peptide_id", "sequence", "modified_sequence"]).to_pylist()
    }
    specs: list[PrecursorSpec] = []
    for r in library.precursors.select(["peptide_id", "charge", "precursor_mz"]).to_pylist():
        pep = peptide_rows[r["peptide_id"]]
        specs.append(
            PrecursorSpec(
                modified_sequence=pep["modified_sequence"],
                sequence=pep["sequence"],
                charge=int(r["charge"]),
                precursor_mz=float(r["precursor_mz"]),
                proteins=tuple(proteins_of.get(r["peptide_id"], ())),
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
