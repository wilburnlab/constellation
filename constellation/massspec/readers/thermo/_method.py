"""What a Thermo instrument-method text says about the scans it made.

The method text (RawFileReader's instrument-method dump, carried by a
bundle's acquisition metadata and by a ``.dia``'s ``thermo.instrument_method.*``
keys) defines each scan of the method in a block headed ``Scan <kind>``,
with its settings, among them the m/z range it acquired::

    Scan MasterScan
        Scan Range (m/z) = 350-1250
    ...
    Scan DIAScan                          (or ddMSnScan, with MSn Level = 2)
        Scan Range (m/z) = 200-1500

A ``.dia`` keeps an MS1 scan's range (as its isolation bounds) but not an
MS2 scan's, so this is where an imported run's MS2 range comes from.
Pure Python, no .NET.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

__all__ = ["scan_ranges_from_method"]

_SCAN_RE = re.compile(r"^\s*Scan\s+(?P<kind>[A-Za-z0-9]+)\s*$")
_LEVEL_RE = re.compile(r"^\s*MSn Level\s*=\s*(?P<level>\d+)\s*$")
_RANGE_RE = re.compile(r"^\s*Scan Range \(m/z\)\s*=\s*(?P<lo>[0-9.]+)\s*-\s*(?P<hi>[0-9.]+)\s*$")
# A heading that starts something other than a scan's settings.
_OTHER_RE = re.compile(r"^\s*(Experiment\s+\d+|Scan Event\s+\d+|Filter\s+\w+|Data Dependent Properties)\s*$")
#: Scan kinds that isolate nothing: full MS1 scans.
_MS1_KINDS = {"masterscan", "fullscan", "fullms", "ms1scan", "msscan"}


def scan_ranges_from_method(texts: Iterable[str]) -> dict[int, tuple[float, float]]:
    """Each MS level's acquired m/z range, from method texts: the level's
    one range if every scan of it the method defines names the same, else
    no entry (a level of several ranges is not one range to apply). A scan
    block's level is its ``MSn Level``, else 1 for a full scan
    (``MasterScan``) and 2 for any other kind."""
    found: dict[int, set[tuple[float, float]]] = {}
    for text in texts:
        if not text:
            continue
        kind: str | None = None
        level: int | None = None
        span: tuple[float, float] | None = None

        def close() -> None:
            if kind is None or span is None:
                return
            lvl = level if level is not None else (1 if kind.lower() in _MS1_KINDS else 2)
            found.setdefault(lvl, set()).add(span)

        for line in str(text).splitlines():
            m = _SCAN_RE.match(line)
            if m and not _OTHER_RE.match(line):
                close()
                kind, level, span = m.group("kind"), None, None
                continue
            if _OTHER_RE.match(line):
                close()
                kind, level, span = None, None, None
                continue
            if kind is None:
                continue
            m = _LEVEL_RE.match(line)
            if m:
                level = int(m.group("level"))
                continue
            m = _RANGE_RE.match(line)
            if m and span is None:
                span = (float(m.group("lo")), float(m.group("hi")))
        close()
    return {lvl: next(iter(r)) for lvl, r in found.items() if len(r) == 1}
