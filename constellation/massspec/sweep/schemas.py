"""Arrow schemas of a sweep: its plan, and what its map hands its reduce.

A sweep's queries and their ions are written out as
:data:`SWEEP_QUERY_TABLE` and :data:`SWEEP_ION_TABLE` so that worker
processes read the plan rather than receive it pickled. The map writes
the scans it saw (:data:`SWEEP_SCAN_TABLE`), the peaks it matched
(:data:`SWEEP_MATCH_TABLE`) and, with a gate, its fragment counts
(:data:`SWEEP_EVIDENCE_TABLE`); the reduce reads them back per bucket of
queries. Ion coordinates use :data:`~constellation.massspec.quant.schemas.XIC_TRACE_TABLE`'s
types, so a trace is a join and a cast, not a conversion.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema

SWEEP_SCHEMA_VERSION: int = 1


def _meta(name: str) -> dict[bytes, bytes]:
    return {b"schema_name": name.encode(), b"schema_version": str(SWEEP_SCHEMA_VERSION).encode()}


# One row per query: an extraction target as the sweep resolved it. `query`
# is the target's position in the input, which is also the order output
# comes in.
SWEEP_QUERY_TABLE: pa.Schema = pa.schema(
    [
        pa.field("query", pa.int32(), nullable=False),
        pa.field("target_id", pa.int64(), nullable=False),
        pa.field("modified_sequence", pa.string(), nullable=True),
        pa.field("precursor_charge", pa.int8(), nullable=True),
        # Null: no isolation gate (the query is matched in every window).
        pa.field("precursor_mz", pa.float64(), nullable=True),
        # True where the target gave no precursor m/z and it was computed.
        pa.field("precursor_mz_derived", pa.bool_(), nullable=False),
        # Inclusive retention-time bounds (s); ±inf when unbounded.
        pa.field("rt_lo", pa.float64(), nullable=False),
        pa.field("rt_hi", pa.float64(), nullable=False),
        # The one scan its fragments may be matched in, if any.
        pa.field("assigned_scan", pa.int32(), nullable=True),
        pa.field("precursor_id", pa.int64(), nullable=True),
        pa.field("peptide_id", pa.int64(), nullable=True),
    ],
    metadata=_meta("SweepQuery"),
)

# One row per ion a query looks for. `kind` 1 is a precursor isotope
# (matched in MS1 scans), 2 a fragment (matched in MS2 and higher). `ion`
# numbers a query's ions of one kind in output order.
SWEEP_ION_TABLE: pa.Schema = pa.schema(
    [
        pa.field("kind", pa.int8(), nullable=False),
        pa.field("query", pa.int32(), nullable=False),
        pa.field("ion", pa.int32(), nullable=False),
        pa.field("mz_theoretical", pa.float64(), nullable=False),
        pa.field("precursor_charge", pa.int8(), nullable=False),
        pa.field("isotope", pa.int8(), nullable=True),
        pa.field("ion_type", pa.int8(), nullable=True),
        pa.field("position", pa.int32(), nullable=True),
        pa.field("fragment_charge", pa.int32(), nullable=True),
        pa.field("loss_id", pa.string(), nullable=True),
        # The isotope's share of its envelope; null where not modelled.
        pa.field("abundance", pa.float32(), nullable=True),
        # What a match counts as evidence for: the isotope, or the rank of
        # the fragment's (ion type, position) within the query, so that one
        # bond seen at two charges counts once.
        pa.field("evidence_key", pa.int32(), nullable=False),
    ],
    metadata=_meta("SweepIon"),
)

# One row per scan the map read, peakless scans included.
SWEEP_SCAN_TABLE: pa.Schema = pa.schema(
    [
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("rt", pa.float64(), nullable=False),
        pa.field("isolation_lower", pa.float64(), nullable=True),
        pa.field("isolation_upper", pa.float64(), nullable=True),
        pa.field("n_peaks", pa.int32(), nullable=False),
    ],
    metadata=_meta("SweepScan"),
)

# One row per (query ion, scan, peak) match. Only what the map knows that
# the reduce cannot look up again: everything else is joined from the plan
# and the scan ledger.
SWEEP_MATCH_TABLE: pa.Schema = pa.schema(
    [
        pa.field("query", pa.int32(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("ion", pa.int32(), nullable=False),
        # The peak's position in its scan's stored order; null when a rule
        # aggregates several peaks into one match.
        pa.field("peak_index", pa.int32(), nullable=True),
        pa.field("mz_observed", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        # Peaks the match stands for (1 except for aggregating rules).
        pa.field("n_peaks", pa.int32(), nullable=False),
    ],
    metadata=_meta("SweepMatch"),
)

# One row per (query, evidence group, scan) with any evidence: distinct
# evidence keys matched among the scan's evidence peaks, the distinct peaks
# that matched them, the ions involved, and those peaks' summed intensity.
SWEEP_EVIDENCE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("query", pa.int32(), nullable=False),
        # 0 for fragments; the precursor charge for isotopes.
        pa.field("group", pa.int8(), nullable=False),
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("n_keys", pa.int16(), nullable=False),
        pa.field("n_peaks", pa.int16(), nullable=False),
        pa.field("n_ions", pa.int16(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
    ],
    metadata=_meta("SweepEvidence"),
)


# One row per (scan, peak, claiming ion) for each peak more than one cell
# matched, as the map found them: the hand-off to the claims table.
SWEEP_CLAIM_TABLE: pa.Schema = pa.schema(
    [
        pa.field("level", pa.int8(), nullable=False),
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("peak_index", pa.int32(), nullable=False),
        pa.field("query", pa.int32(), nullable=False),
        pa.field("ion", pa.int32(), nullable=False),
        # Distinct queries and (query, ion) cells claiming the peak.
        pa.field("n_queries", pa.int32(), nullable=False),
        pa.field("n_pairs", pa.int32(), nullable=False),
        pa.field("mz_observed", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
    ],
    metadata=_meta("SweepClaim"),
)

# Terminal: one row per query, what the sweep saw of it.
SWEEP_QUERY_SUMMARY_TABLE: pa.Schema = pa.schema(
    [
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("target_id", pa.int64(), nullable=False),
        pa.field("query", pa.int32(), nullable=False),
        pa.field("modified_sequence", pa.string(), nullable=True),
        pa.field("precursor_charge", pa.int8(), nullable=True),
        # The evidence gate's verdict; null without a gate.
        pa.field("gate_applicable", pa.bool_(), nullable=True),
        pa.field("passed", pa.bool_(), nullable=True),
        pa.field("n_evidence_scans", pa.int32(), nullable=True),
        pa.field("longest_run", pa.int32(), nullable=True),
        pa.field("best_scan", pa.int32(), nullable=True),
        pa.field("best_rt", pa.float64(), nullable=True),
        # Distinct bonds (or isotopes) at the best scan.
        pa.field("best_n_keys", pa.int32(), nullable=True),
        # Scans each level could have matched it in, and ion cells matched.
        pa.field("n_eligible_ms1", pa.int32(), nullable=False),
        pa.field("n_eligible_ms2", pa.int32(), nullable=False),
        pa.field("n_matched_ms1", pa.int32(), nullable=False),
        pa.field("n_matched_ms2", pa.int32(), nullable=False),
        # Its matches whose peak another query also claimed; null without claims.
        pa.field("n_shared_matches", pa.int32(), nullable=True),
    ],
    metadata=_meta("SweepQuerySummary"),
)

# Terminal: the claims, with what each claimant is.
SWEEP_PEAK_CLAIM_TABLE: pa.Schema = pa.schema(
    [
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("rt", pa.float64(), nullable=False),
        pa.field("peak_index", pa.int32(), nullable=False),
        pa.field("mz_observed", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        pa.field("n_queries", pa.int32(), nullable=False),
        pa.field("n_pairs", pa.int32(), nullable=False),
        pa.field("target_id", pa.int64(), nullable=False),
        pa.field("query", pa.int32(), nullable=False),
        pa.field("modified_sequence", pa.string(), nullable=True),
        pa.field("precursor_charge", pa.int8(), nullable=False),
        pa.field("isotope", pa.int8(), nullable=True),
        pa.field("ion_type", pa.int8(), nullable=True),
        pa.field("position", pa.int32(), nullable=True),
        pa.field("fragment_charge", pa.int32(), nullable=True),
        pa.field("loss_id", pa.string(), nullable=True),
        pa.field("mz_theoretical", pa.float64(), nullable=False),
        pa.field("mz_error_ppm", pa.float64(), nullable=False),
    ],
    metadata=_meta("SweepPeakClaim"),
)


register_schema("SweepQuery", SWEEP_QUERY_TABLE)
register_schema("SweepIon", SWEEP_ION_TABLE)
register_schema("SweepScan", SWEEP_SCAN_TABLE)
register_schema("SweepMatch", SWEEP_MATCH_TABLE)
register_schema("SweepEvidence", SWEEP_EVIDENCE_TABLE)
register_schema("SweepClaim", SWEEP_CLAIM_TABLE)
register_schema("SweepQuerySummary", SWEEP_QUERY_SUMMARY_TABLE)
register_schema("SweepPeakClaim", SWEEP_PEAK_CLAIM_TABLE)

__all__ = [
    "SWEEP_SCHEMA_VERSION",
    "SWEEP_QUERY_TABLE",
    "SWEEP_ION_TABLE",
    "SWEEP_SCAN_TABLE",
    "SWEEP_MATCH_TABLE",
    "SWEEP_EVIDENCE_TABLE",
    "SWEEP_CLAIM_TABLE",
    "SWEEP_QUERY_SUMMARY_TABLE",
    "SWEEP_PEAK_CLAIM_TABLE",
]
