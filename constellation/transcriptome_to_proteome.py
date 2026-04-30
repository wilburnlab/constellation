"""Top-level cross-modality script: transcriptome → spectral library.

Bridge between :mod:`constellation.sequencing` (de novo transcriptome
assembly producing predicted proteins) and :mod:`constellation.massspec`
(spectral library prediction + DIA / DDA search). The lab's matched
genome + transcriptome + proteome workflow exits here.

Pipeline:

    1. Source: a finalized ``TRANSCRIPT_CLUSTER_TABLE`` from
       :mod:`sequencing.transcriptome` (predicted proteins per
       consensus transcript).
    2. Filter on ORF length / start-codon / minimum cluster size to
       arrive at a high-confidence predicted-proteome FASTA.
    3. Run massspec.peptide.cleave + filter for tryptic peptides
       within mass range, charge envelope, etc. — the front half of
       library construction.
    4. Either pipe into Koina (if checkpoints available) for spectral
       prediction, or emit a peptide list for an external prediction
       tool (Prosit, MS²PIP).
    5. Build a ``Library`` container; save as ParquetDir or .dlib
       for use by Cartographer-style search workflows.

Placement: top-level ``constellation/`` per CLAUDE.md's "≥2 workflows
before a folder" rule. When a second cross-modality workflow lands
(structure → spectral, genome → spectral library directly bypassing
transcriptome), this and that workflow co-locate into a dedicated
subpackage with a name TBD (not ``bridges/`` or ``pipelines/``;
candidates: ``asterism/``, ``confluence/``, ``weft/``, ``synthesis/``,
``composer/``, ``lattice/`` — defer until the second workflow appears).

Status: STUB. Pending Phase 12.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa


_PHASE = "Phase 12 (transcriptome → proteome bridge)"


def predict_library_from_transcriptome(
    cluster_table: pa.Table,         # TRANSCRIPT_CLUSTER_TABLE
    *,
    output_path: Path,
    min_protein_length_aa: int = 50,
    min_cluster_size: int = 3,
    enzyme: str = "Trypsin",
    missed_cleavages: int = 2,
    charge_range: tuple[int, int] = (2, 4),
    mass_range_da: tuple[float, float] = (600.0, 6000.0),
    koina_model: str | None = None,
):
    """Build a spectral ``Library`` from a transcriptome's predicted
    proteins.

    If ``koina_model`` is provided, route the peptide list through
    Koina for spectral prediction (uses :mod:`massspec.koina` —
    currently scaffold). Otherwise emit a peptide list and let an
    external prediction tool finish the library.

    Returns the populated ``Library`` (also writes to ``output_path``
    in ParquetDir form).
    """
    raise NotImplementedError(
        f"predict_library_from_transcriptome pending {_PHASE}"
    )


def write_predicted_proteome_fasta(
    cluster_table: pa.Table,
    output_path: Path,
    *,
    min_length_aa: int = 50,
    min_cluster_size: int = 3,
) -> int:
    """Project ``TRANSCRIPT_CLUSTER_TABLE.predicted_protein`` to a FASTA
    file (one record per cluster). Returns the number of records
    written. Cheap pre-step before full library construction — useful
    for downstream tools that just want the protein list.
    """
    raise NotImplementedError(
        f"write_predicted_proteome_fasta pending {_PHASE}"
    )


__all__ = [
    "predict_library_from_transcriptome",
    "write_predicted_proteome_fasta",
]
