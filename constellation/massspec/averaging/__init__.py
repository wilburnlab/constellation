"""One spectrum per kind of scan from a direct-infusion profile run.

:func:`average_bundle` reads a run's profile bundle (``<run>.profile``, from
``massspec convert``) and writes ``<run>.average``: an ordinary profile bundle
holding the mean of every repeat of each kind of scan, point by point on the
ion trap's m/z grid, with each point's standard deviation and replicate count
beside it. This package imports nothing from ``readers``.
"""

from __future__ import annotations

from constellation.massspec.averaging.driver import (
    POINTS_DIR,
    SUMMARY_FILENAME,
    average_bundle,
    group_scans,
)

__all__ = ["POINTS_DIR", "SUMMARY_FILENAME", "average_bundle", "group_scans"]
