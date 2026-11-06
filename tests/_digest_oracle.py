"""``massspec.library.digest.precursors_from_fasta`` as it was before
2026-11-05, when it digested and enumerated on its own; frozen as the
oracle its replacement (a projection of ``build_search_space``) is held
to. It reads with the old FASTA reader, inlined here. Do not edit; it is
a record."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pyarrow as pa

from _cleave_oracle import cleave, enumerate_modforms
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import format_proforma
from constellation.massspec.library.digest import DEFAULT_MAX_MZ, DEFAULT_MIN_MZ, PrecursorSpec
from constellation.massspec.peptide.mz import precursor_mz

__all__ = ["precursors_from_fasta"]


def read_fasta_proteins(path):
    p = Path(path)
    protein_ids, sequences = [], []
    current_id, current_seq = None, []
    with p.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    protein_ids.append(current_id)
                    sequences.append("".join(current_seq).upper())
                current_id = line[1:].split()[0] if len(line) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            protein_ids.append(current_id)
            sequences.append("".join(current_seq).upper())
    return pa.table({"protein_id": pa.array(protein_ids, type=pa.string()), "sequence": pa.array(sequences, type=pa.string())})


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
    """
    table = read_fasta_proteins(fasta)
    accessions = table.column("protein_id").to_pylist()
    sequences = table.column("sequence").to_pylist()

    # peptide sequence → the accessions it came from, first-seen order.
    peptide_proteins: dict[str, list[str]] = {}
    for accession, protein_seq in zip(accessions, sequences, strict=True):
        peptides = cleave(
            protein_seq,
            protease,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            excise_initiator_met=excise_initiator_met,
            validate_alphabet=False,
        )
        for peptide in peptides:
            owners = peptide_proteins.setdefault(peptide, [])
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
