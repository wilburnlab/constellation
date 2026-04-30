"""Project directory layout — where POD5s, BAMs, assemblies live.

The standard Constellation project tree::

    project_root/
    ├── manifest.toml          # Acquisitions, Samples, artifact map
    ├── pod5/                  # Raw signal inputs (one subdir per acquisition)
    │   ├── flowcell-FAQ12345/
    │   │   ├── *.pod5
    │   ├── flowcell-FAQ23456/
    ├── bam/                   # Dorado output (basecalled + optionally aligned)
    │   ├── FAQ12345.bam
    │   ├── FAQ12345.bam.bai
    │   ├── FAQ23456.bam
    ├── fastx/                 # Optional: cached FASTA/FASTQ projections
    ├── transcriptome/         # cDNA pipeline artifacts
    │   ├── demuxed.parquet    # READ_DEMUX_TABLE
    │   ├── clusters.parquet   # cluster membership
    │   ├── consensus.parquet  # TRANSCRIPT_CLUSTER_TABLE
    │   ├── proteins.fasta     # predicted proteins → bridges to massspec
    ├── assembly/              # Genomic assembly artifacts
    │   ├── primary.fasta      # HiFiASM primary contigs
    │   ├── primary.gfa
    │   ├── polished.fasta
    │   ├── scaffolded.fasta   # post-RagTag
    ├── annotation/            # Annotation features for the assembly
    │   ├── busco/             # BUSCO output dir
    │   ├── repeats.gff3
    │   ├── telomeres.gff3
    │   ├── transcript_features.gff3
    └── work/                  # Scratch dir for intermediate files
                               # (mmseqs indexes, minimap2 .mmi, etc.)

Users can also place pre-existing BAMs / FASTAs into the appropriate
subdirectories and skip the steps that produced them — the manifest
tracks which artifacts exist independently of which steps ran.

Status: STUB. Pending Phase 11.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_PHASE = "Phase 11 (projects/layout, projects/manifest)"


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    """Path bundle for a project tree.

    All paths are relative to a ``project_root`` supplied at
    instantiation; methods return absolute paths. Defaults match the
    standard layout above; ``standard_layout(root)`` is the
    convenience factory for the default tree.
    """

    project_root: Path
    pod5_subdir: str = "pod5"
    bam_subdir: str = "bam"
    fastx_subdir: str = "fastx"
    transcriptome_subdir: str = "transcriptome"
    assembly_subdir: str = "assembly"
    annotation_subdir: str = "annotation"
    work_subdir: str = "work"
    manifest_filename: str = "manifest.toml"

    @property
    def manifest_path(self) -> Path:
        return self.project_root / self.manifest_filename

    @property
    def pod5_dir(self) -> Path:
        return self.project_root / self.pod5_subdir

    @property
    def bam_dir(self) -> Path:
        return self.project_root / self.bam_subdir

    @property
    def fastx_dir(self) -> Path:
        return self.project_root / self.fastx_subdir

    @property
    def transcriptome_dir(self) -> Path:
        return self.project_root / self.transcriptome_subdir

    @property
    def assembly_dir(self) -> Path:
        return self.project_root / self.assembly_subdir

    @property
    def annotation_dir(self) -> Path:
        return self.project_root / self.annotation_subdir

    @property
    def work_dir(self) -> Path:
        return self.project_root / self.work_subdir

    def ensure_dirs(self) -> None:
        """Create the subdirectories if missing. Idempotent."""
        raise NotImplementedError(f"ProjectLayout.ensure_dirs pending {_PHASE}")


def standard_layout(project_root: Path | str) -> ProjectLayout:
    """Convenience factory — returns ``ProjectLayout`` with default
    subdirectory names rooted at ``project_root``."""
    return ProjectLayout(project_root=Path(project_root))


__all__ = [
    "ProjectLayout",
    "standard_layout",
]
