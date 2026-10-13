"""Genome-browser modality.

Tracks over a reference genome: the reference sequence and annotation
from the per-user reference cache, plus coverage, read pile-ups,
transcript clusters and splice junctions from attached
``transcriptome align`` / ``cluster`` output directories.

Importing this package registers the ``genome`` modality and its track
kernels. Nothing here needs the ``[viz]`` extras: the FastAPI router is
imported only when ``routers()`` is called.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from constellation.viz.modalities import Modality, register_modality
from constellation.viz.modalities.genome import tracks  # noqa: F401
from constellation.viz.modalities.genome.requests import (
    GenomeOpenRequest,
    GenomeSaveRequest,
)
from constellation.viz.modalities.genome.session import (
    GenomeSession,
    inspect_source,
    session_from_saved,
)

#: Every name this modality imports from a domain module, as
#: ``(module, name)``. The viz core imports none; a modality may import a
#: small pure-function surface if it declares it here and does the import
#: inside a function body. ``tests/test_viz_import_boundary.py`` checks
#: the declaration against the code in both directions.
DOMAIN_IMPORTS = (
    ("constellation.sequencing.align.cigar", "parse_cs_long_mismatch_positions"),
    ("constellation.sequencing.reference.handle", "ReferenceNotInstalledError"),
    ("constellation.sequencing.reference.handle", "list_installed"),
    ("constellation.sequencing.reference.handle", "parse_handle"),
    ("constellation.sequencing.reference.handle", "read_defaults"),
    ("constellation.sequencing.reference.handle", "read_meta_toml"),
    ("constellation.sequencing.reference.handle", "resolve"),
    ("constellation.sequencing.transcriptome.manifest", "read_manifest_dir"),
)


def _open_session(request: GenomeOpenRequest) -> GenomeSession:
    return GenomeSession.open(
        reference_handle=request.reference_handle,
        sources=[asdict(source) for source in request.sources],
        label=request.label,
        saved_as=request.saved_as,
    )


def _routers() -> Sequence[Any]:
    from constellation.viz.modalities.genome.endpoints import router

    return (router,)


MODALITY = register_modality(
    Modality(
        name=GenomeSession.modality,
        open_request=GenomeOpenRequest,
        open_session=_open_session,
        inspect_source=inspect_source,
        session_from_saved=session_from_saved,
        save_request=GenomeSaveRequest,
        routers=_routers,
    )
)
