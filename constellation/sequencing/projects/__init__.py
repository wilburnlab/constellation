"""Project orchestrator — directory-layout-aware verb dispatcher.

A ``Project`` is the lab-friendly handle around a tree of nanopore
data + derived artifacts. Knows the directory layout (where POD5
inputs live, where to write BAMs, where assemblies land, etc.),
maintains a manifest of acquisitions / samples / artifacts, and
exposes verbs that delegate to the pure-functional core.

The pure-functional layer (``bam_dir_to_assembly`` etc.) works
without a Project — labs that produce BAMs externally and just want
the Constellation analysis layer can skip the project layer entirely.
The Project layer is for users who want one-stop pipeline orchestration.

Status: STUB. Pending Phase 11.
"""

from __future__ import annotations

from constellation.sequencing.projects.layout import (
    ProjectLayout,
    standard_layout,
)
from constellation.sequencing.projects.manifest import Manifest, Project

__all__ = [
    "Project",
    "Manifest",
    "ProjectLayout",
    "standard_layout",
]
