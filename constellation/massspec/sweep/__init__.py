"""The sweep: matching what queries expect against an acquisition, scan by scan.

One pass over a converted acquisition's scans in acquisition order. Each
query (a peptidoform at a charge, or a bare m/z) is matched in the scans
inside its retention-time interval and, for fragments, the isolation
windows its precursor falls in; no interval means every scan, a global
XIC. All the hypotheses that could claim a scan's peaks are matched
against it together, which is what lets a search see which peaks are
shared. Queries are compiled from extraction targets by
:func:`compile_queries`.
"""

from constellation.massspec.sweep.queries import (
    FRAGMENTS,
    ISOTOPES,
    IonSettings,
    Ions,
    QueryPlan,
    compile_queries,
)

__all__ = [
    "FRAGMENTS",
    "ISOTOPES",
    "IonSettings",
    "Ions",
    "QueryPlan",
    "compile_queries",
]
