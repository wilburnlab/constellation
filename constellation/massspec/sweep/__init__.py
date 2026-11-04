"""The sweep: matching what queries expect against an acquisition, scan by scan.

One pass over a converted acquisition's scans in acquisition order. Each
query (a peptidoform at a charge, or a bare m/z) expects signal at its
channels, and is matched in the scans inside its retention-time interval
and, for fragments, the isolation windows its precursor falls in; no
interval means every scan, a global XIC. All the hypotheses that could
claim a scan's points are matched against it together, which is what lets
a search see which points are shared. Queries are compiled from
extraction targets by :func:`compile_queries`.

Vocabulary (``docs/plans/sweep.md``): a *point* is one stored (m/z,
intensity) value of a scan, a *peak* a centroid (a peak list's points are
peaks), a *channel* one m/z a query expects signal at with what it is
(an isotope, a fragment ion), a *cell* one channel in one scan, and a
*match* a cell with the point or points its rule gives it.
"""

from constellation.massspec.sweep.queries import (
    FRAGMENTS,
    ISOTOPES,
    Channels,
    ChannelSettings,
    QueryPlan,
    compile_queries,
)

__all__ = [
    "FRAGMENTS",
    "ISOTOPES",
    "ChannelSettings",
    "Channels",
    "QueryPlan",
    "compile_queries",
]
