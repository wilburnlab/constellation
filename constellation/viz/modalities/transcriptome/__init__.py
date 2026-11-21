"""Transcriptome browser modality.

The de novo EM clusterer's runs (``transcriptome cluster --mode em-*``):
its clusters, and the support behind any one of them — the read matrix of
its node against its consensus, with the M-step's column statistics and
covariance arcs, re-created by verified replay. The plan is
``docs/plans/transcriptome-browser.md`` (this is its PR2, the support view).

Importing this package registers the ``transcriptome`` modality. It has no
track kernels: the support view is one view of six tables served by its own
routes, not a stack of panels. Nothing here needs the ``[viz]`` extras or a
domain module at import time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from constellation.viz.modalities import Modality, register_modality
from constellation.viz.modalities.transcriptome.requests import (
    TranscriptomeOpenRequest,
    TranscriptomeSaveRequest,
)
from constellation.viz.modalities.transcriptome.session import (
    TranscriptomeSession,
    inspect_source,
    session_from_saved,
)

#: Every name this modality imports from a domain module, as
#: ``(module, name)``; see ``tests/test_viz_import_boundary.py``.
DOMAIN_IMPORTS = (
    ("constellation.sequencing.transcriptome.cluster.denovo.em.results", "ReplayError"),
    ("constellation.sequencing.transcriptome.cluster.denovo.em.results", "cluster_index"),
    ("constellation.sequencing.transcriptome.cluster.denovo.em.results", "find_clusters"),
    ("constellation.sequencing.transcriptome.cluster.denovo.em.results", "node_differences"),
    ("constellation.sequencing.transcriptome.cluster.denovo.em.results", "replay_cluster"),
    ("constellation.sequencing.transcriptome.cluster.denovo.em.results", "replay_summary"),
    ("constellation.sequencing.transcriptome.cluster.denovo.em.results", "replay_template"),
    ("constellation.sequencing.transcriptome.cluster.denovo.em.results", "run_info"),
    ("constellation.sequencing.transcriptome.cluster.denovo.em.results", "support_tables"),
)


def _open_session(request: TranscriptomeOpenRequest) -> TranscriptomeSession:
    return TranscriptomeSession.open(
        sources=[asdict(source) for source in request.sources],
        label=request.label,
        saved_as=request.saved_as,
    )


def _routers() -> Sequence[Any]:
    from constellation.viz.modalities.transcriptome.endpoints import router

    return (router,)


MODALITY = register_modality(
    Modality(
        name=TranscriptomeSession.modality,
        open_request=TranscriptomeOpenRequest,
        open_session=_open_session,
        inspect_source=inspect_source,
        session_from_saved=session_from_saved,
        save_request=TranscriptomeSaveRequest,
        routers=_routers,
    )
)
