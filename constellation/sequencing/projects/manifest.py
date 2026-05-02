"""``Project`` and ``Manifest`` — orchestrator over the standard layout.

``Manifest`` is the on-disk TOML record of what's in a project:
acquisitions, samples, derived artifacts (BAM paths, assembly paths,
annotation paths). It's the single source of truth for "has this step
completed?" checks; the actual filesystem is the source of truth for
the underlying bytes, and the manifest records the binding.

``Project`` wraps the ``Manifest`` + ``ProjectLayout`` and exposes the
verbs that delegate to the pure-functional core:

    proj.basecall(model)       → calls DoradoRunner.basecaller
    proj.demux(design)         → calls transcriptome.demux.locate_segments
    proj.cluster()             → calls transcriptome.cluster.cluster_reads
    proj.assemble()            → calls assembly.hifiasm.HiFiAsmRunner.run
    proj.polish(rounds=2)      → calls assembly.polish.PolishRunner.run
    proj.scaffold(reference)   → calls assembly.ragtag.RagTagRunner.run
    proj.annotate(...)         → calls annotation.* runners

Each verb is idempotent — re-running a completed step short-circuits
unless ``force=True``. The manifest's ``with_artifact`` updater is the
*only* place project state mutates.

Status: STUB. Pending Phase 11.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constellation.sequencing.acquisitions import Acquisitions
from constellation.sequencing.projects.layout import (
    ProjectLayout,
    standard_layout,
)
from constellation.sequencing.samples import Samples


_PHASE = "Phase 11 (projects/manifest)"


@dataclass(frozen=True, slots=True)
class Manifest:
    """On-disk project manifest — TOML, append-most-mutations friendly.

    Lives at ``layout.manifest_path``. The TOML schema is documented
    in :doc:`/sequencing/project-format` (Phase 11).
    """

    acquisitions: Acquisitions
    samples: Samples
    # Free-form artifact registry: key is a pipeline-stage tag
    # ('bam:FAQ12345', 'assembly:primary', 'annotation:busco', ...);
    # value is a ``Path | str`` resolved relative to the project root.
    artifacts: dict[str, str]

    @classmethod
    def empty(cls) -> "Manifest":
        raise NotImplementedError(f"Manifest.empty pending {_PHASE}")

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        raise NotImplementedError(f"Manifest.load pending {_PHASE}")

    def save(self, path: Path) -> None:
        raise NotImplementedError(f"Manifest.save pending {_PHASE}")

    def with_artifact(self, tag: str, path: Path | str) -> "Manifest":
        """Return a new Manifest with the given artifact registered."""
        raise NotImplementedError(f"Manifest.with_artifact pending {_PHASE}")

    def has_artifact(self, tag: str) -> bool:
        raise NotImplementedError(f"Manifest.has_artifact pending {_PHASE}")


@dataclass(frozen=True, slots=True)
class Project:
    """Project orchestrator — manifest + layout + verbs.

    Use ``Project.open(root)`` to load an existing project,
    ``Project.create(root)`` to bootstrap a new one (creates the
    standard subdirectory tree and an empty manifest).
    """

    layout: ProjectLayout
    manifest: Manifest

    # ── construction ───────────────────────────────────────────
    @classmethod
    def open(cls, root: Path | str) -> "Project":
        raise NotImplementedError(f"Project.open pending {_PHASE}")

    @classmethod
    def create(cls, root: Path | str) -> "Project":
        raise NotImplementedError(f"Project.create pending {_PHASE}")

    # ── input registration ──────────────────────────────────────
    def add_acquisitions(
        self,
        pod5_paths: list[Path],
        *,
        flow_cell_id: str | None = None,
        sample_kit: str | None = None,
        experiment_type: str | None = None,
    ) -> "Project":
        """Register one or more POD5 inputs as acquisitions; returns a
        new Project with an updated manifest."""
        raise NotImplementedError(f"Project.add_acquisitions pending {_PHASE}")

    def add_samples(
        self,
        sample_definitions: list[dict],
    ) -> "Project":
        """Register samples + their acquisition / barcode mappings."""
        raise NotImplementedError(f"Project.add_samples pending {_PHASE}")

    # ── pipeline verbs ──────────────────────────────────────────
    def basecall(self, *, model, force: bool = False) -> "Project":
        raise NotImplementedError(f"Project.basecall pending {_PHASE}")

    def demux(self, *, design, force: bool = False) -> "Project":
        raise NotImplementedError(f"Project.demux pending {_PHASE}")

    def cluster(self, *, force: bool = False) -> "Project":
        raise NotImplementedError(f"Project.cluster pending {_PHASE}")

    def assemble(self, *, force: bool = False) -> "Project":
        raise NotImplementedError(f"Project.assemble pending {_PHASE}")

    def polish(self, *, rounds: int = 1, force: bool = False) -> "Project":
        raise NotImplementedError(f"Project.polish pending {_PHASE}")

    def scaffold(self, *, reference, force: bool = False) -> "Project":
        raise NotImplementedError(f"Project.scaffold pending {_PHASE}")

    def annotate(self, *, busco_lineage: str | None = None,
                 force: bool = False) -> "Project":
        raise NotImplementedError(f"Project.annotate pending {_PHASE}")


def open_project(root: Path | str) -> Project:
    """Convenience function — same as :meth:`Project.open`."""
    return Project.open(root)


def create_project(root: Path | str) -> Project:
    """Convenience function — same as :meth:`Project.create`. Creates
    the standard layout subdirectories under ``root``."""
    layout = standard_layout(root)
    layout.ensure_dirs()
    return Project.create(root)


__all__ = [
    "Manifest",
    "Project",
    "open_project",
    "create_project",
]
