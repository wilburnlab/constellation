"""Query models for the mass-spec kernels.

A binding is one acquisition, so a query says only what to read from it:
which MS level and isolation window for a chromatogram, which scan for a
spectrum. Retention times are seconds and m/z values Thomson, as in the
bundle.

Stdlib dataclasses only — this module is imported by
``import constellation.viz``, which must work without pydantic. Range
constraints ride as ``field(metadata=...)`` (HTTP 422 when violated);
rules that span fields are in :meth:`check` (HTTP 400). No ``slots=True``:
subclasses call ``super().check()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from constellation.viz.tracks.base import TrackQuery


def _check_range(start: float | None, end: float | None, name: str) -> None:
    if (start is None) != (end is None):
        raise ValueError(f"{name}_start and {name}_end go together")
    if start is not None and end is not None and end <= start:
        raise ValueError(f"{name}_end must be greater than {name}_start")


@dataclass(frozen=True, kw_only=True)
class ChromatogramQuery(TrackQuery):
    """A trace over retention time for one MS level.

    ``trace`` is ``tic`` (total ion current), ``bpc`` (base peak) or
    ``scans`` (one point per scan at its TIC, carrying the scan's precursor
    m/z: the scan events of a DDA level). ``window`` restricts a DIA level
    to one isolation window (the ids of the ``windows`` route). The whole
    trace is sent when it is short enough to draw at full resolution, and
    the client zooms it locally; a longer one is reduced to what
    ``viewport_px`` columns show over ``rt_start``..``rt_end`` (the whole
    run when they are omitted), and its rows say so (``decimated``).
    """

    level: int = field(default=1, metadata={"ge": 1, "le": 10})
    trace: Literal["tic", "bpc", "scans"] = "tic"
    window: int | None = field(default=None, metadata={"ge": 0})
    rt_start: float | None = None
    rt_end: float | None = None

    def check(self) -> None:
        super().check()
        _check_range(self.rt_start, self.rt_end, "rt")


@dataclass(frozen=True, kw_only=True)
class SpectrumQuery(TrackQuery):
    """The peaks of one scan, optionally within ``mz_start``..``mz_end``."""

    scan: int = field(metadata={"ge": 0})
    mz_start: float | None = None
    mz_end: float | None = None

    def check(self) -> None:
        super().check()
        _check_range(self.mz_start, self.mz_end, "mz")


__all__ = ["ChromatogramQuery", "SpectrumQuery"]
