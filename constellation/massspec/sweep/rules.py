"""How an ion's tolerance window becomes a match.

- :class:`Nearest`: the scan's nearest peak, if within tolerance (the
  scan-major extractor's rule, with its tie to the lower m/z).
- :class:`AllWithin`: every peak within tolerance, one match each.
- :class:`SumWithin`: every peak within tolerance as one match, its
  intensity their sum and its m/z their intensity-weighted mean
  (EncyclopeDIA's extraction rule).

A rule is a reduction over the peaks a window holds, so a rule over
profile points can be added later without changing how scans are swept.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MatchRule", "Nearest", "AllWithin", "SumWithin", "rule_named"]


@dataclass(frozen=True)
class Nearest:
    name: str = "nearest"


@dataclass(frozen=True)
class AllWithin:
    name: str = "all"


@dataclass(frozen=True)
class SumWithin:
    name: str = "sum"


MatchRule = Nearest | AllWithin | SumWithin

_RULES = {r.name: r for r in (Nearest(), AllWithin(), SumWithin())}


def rule_named(name: str) -> MatchRule:
    """The rule called ``name`` (``nearest``, ``all`` or ``sum``)."""
    try:
        return _RULES[name]
    except KeyError:
        raise ValueError(f"unknown match rule {name!r}; use one of {', '.join(_RULES)}") from None
