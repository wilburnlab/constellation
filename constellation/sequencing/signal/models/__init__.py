"""Custom basecaller models — physics-based research direction.

Where the lab's ambitions for physically-grounded nanopore basecallers
live. The model contract follows the Constellation pattern: each
basecaller is a ``Parametric(nn.Module)`` (per :mod:`core.stats`) so
fitting / log-likelihood / inference all flow through the unified
``core.optim`` machinery.

A ``BASECALLER_MODEL`` registry tracks the available models — analog
to ``OPTIMIZER_REGISTRY``. Concrete models register at import time;
``find_model(name)`` resolves by name.

Status: scaffold-only. Concrete models defer to the lab's planned
synthetic-library + DC-analysis research — this module is here so
the directory structure documents the research direction even though
no implementations land for some time.
"""

from __future__ import annotations

from constellation.sequencing.signal.models.physical import (
    BASECALLER_MODEL,
    BasecallerModel,
    find_model,
    register_model,
)

__all__ = [
    "BasecallerModel",
    "BASECALLER_MODEL",
    "register_model",
    "find_model",
]
