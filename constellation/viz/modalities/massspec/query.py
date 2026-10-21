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
    """The peaks of one scan, optionally within ``mz_start``..``mz_end``.

    With ``modified_sequence`` (ProForma 2.0) and ``charge``, the peaks are
    annotated with that peptide's ions at ``tolerance`` / ``tolerance_unit``:
    ``ion_types`` (b and y when empty), fragment charges up to
    ``max_fragment_charge`` (one below the precursor's, at most 3, when
    omitted) and the ``neutral_losses`` named (none when empty).
    """

    scan: int = field(metadata={"ge": 0})
    mz_start: float | None = None
    mz_end: float | None = None
    modified_sequence: str | None = None
    charge: int | None = field(default=None, metadata={"ge": 1, "le": 10})
    tolerance: float | None = field(default=None, metadata={"gt": 0})
    tolerance_unit: Literal["ppm", "Da"] | None = None
    ion_types: tuple[Literal["a", "b", "c", "x", "y", "z"], ...] = ()
    max_fragment_charge: int | None = field(default=None, metadata={"ge": 1, "le": 6})
    neutral_losses: tuple[str, ...] = ()

    def check(self) -> None:
        super().check()
        _check_range(self.mz_start, self.mz_end, "mz")
        if self.modified_sequence and (self.charge is None or self.tolerance is None or self.tolerance_unit is None):
            raise ValueError("annotating a spectrum needs charge, tolerance and tolerance_unit")


#: Traces one request may ask for.
MAX_XIC_TRACES = 64


@dataclass(frozen=True, kw_only=True)
class XicQuery(TrackQuery):
    """Extracted-ion chromatograms of the m/z values in ``mz`` (repeated),
    over one MS level (and DIA ``window``) within ``rt_start``..``rt_end``
    (the whole run when omitted). ``label`` names each trace (repeated,
    one per m/z, or none)."""

    level: int = field(default=1, metadata={"ge": 1, "le": 10})
    window: int | None = field(default=None, metadata={"ge": 0})
    mz: tuple[float, ...] = ()
    label: tuple[str, ...] = ()
    tolerance: float = field(metadata={"gt": 0})
    tolerance_unit: Literal["ppm", "Da"]
    rt_start: float | None = None
    rt_end: float | None = None

    def check(self) -> None:
        super().check()
        _check_range(self.rt_start, self.rt_end, "rt")
        if not self.mz:
            raise ValueError("give at least one m/z")
        if len(self.mz) > MAX_XIC_TRACES:
            raise ValueError(f"at most {MAX_XIC_TRACES} traces per request")
        if self.label and len(self.label) != len(self.mz):
            raise ValueError("give one label per m/z, or none")


__all__ = ["MAX_XIC_TRACES", "ChromatogramQuery", "SpectrumQuery", "XicQuery"]
