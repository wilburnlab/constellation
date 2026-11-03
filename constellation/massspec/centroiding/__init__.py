"""Peak lists made from a profile bundle, with no vendor library.

A centroid is a compressed view of the profile, so a run can have several, each
a complete ordinary peak bundle that every consumer reads like any other
(:mod:`constellation.massspec.bundle`). :func:`centroid_bundle` makes one with a
registered algorithm; :data:`ALGORITHMS` is the registry. Thermo's centroider is
not here: it takes a .NET ``Scan``, so it lives in ``readers.thermo`` and a
conversion runs it. This package imports nothing from ``readers``.
"""

from __future__ import annotations

from constellation.massspec.centroiding.apex import pick_apex
from constellation.massspec.centroiding.driver import (
    ALGORITHMS,
    Algorithm,
    centroid_bundle,
    peak_list_key,
)

__all__ = ["ALGORITHMS", "Algorithm", "centroid_bundle", "peak_list_key", "pick_apex"]
