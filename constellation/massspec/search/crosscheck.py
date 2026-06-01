"""Cross-validate PSM scan assignments against converted scan metadata.

A ``PSM_TABLE`` carries ``mass_analyzer`` (e.g. ``"FTMS"``) and
``fragmentation`` (e.g. ``"HCD"``) verbatim from the search engine. The
``.raw → parquet`` converter writes one ``scan_metadata.parquet`` bundle
per acquisition (``SCAN_METADATA_TABLE``, keyed by ``scan``) carrying
``analyzer`` and ``activation_type``. Joining a PSM to its scan and
comparing these is the "validate the join, don't assume it" check the
MaxQuant reader's scan-level join-back relies on.

This helper is engine-agnostic — it works on any ``PSM_TABLE`` (or
``Search``), not just MaxQuant. It is intentionally OUTSIDE the reader's
read path: the reader has no ``scan_metadata`` at hand, and the check is
opt-in QC.

Casing note: ``SCAN_METADATA_TABLE.activation_type`` is lowercase while
``PSM_TABLE.fragmentation`` is upper (``HCD``/``CID``/``ETD``); the
comparator lowercases the PSM value. A comparison where either side is
null (unconfirmable) counts as a mismatch, conservatively.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from constellation.massspec.search.search import Search


def cross_validate_against_scan_metadata(
    source: Search | pa.Table,
    scan_metadata: pa.Table,
    *,
    raw_file: str | None = None,
) -> pa.Table:
    """Per-PSM agreement of analyzer / fragmentation vs ``scan_metadata``.

    Parameters
    ----------
    source
        A :class:`Search` (its ``psms`` is used) or a ``PSM_TABLE``-shaped
        ``pa.Table``.
    scan_metadata
        ``SCAN_METADATA_TABLE`` for **one** acquisition (the converter
        writes one bundle per acquisition, keyed by ``scan``).
    raw_file
        Restrict to PSMs from this ``Raw file`` before the join. Required
        when the PSM table spans more than one acquisition — otherwise
        ``scan`` numbers from different runs would cross-match this single
        acquisition's metadata.

    Returns
    -------
    pa.Table
        One row per (scoped) PSM, sorted by ``psm_id``::

            psm_id, raw_file, scan, psm_analyzer, scan_analyzer,
            psm_fragmentation, scan_activation, status

        ``status`` ∈ ``{matched, analyzer_mismatch, activation_mismatch,
        both_mismatch, scan_absent}``.
    """
    psms = source.psms if isinstance(source, Search) else source

    if raw_file is not None:
        psms = psms.filter(pc.equal(psms.column("raw_file"), raw_file))
    else:
        distinct = pc.unique(psms.column("raw_file"))
        if len(distinct) > 1:
            raise ValueError(
                "scan_metadata is per-acquisition, but the PSM table spans "
                f"{len(distinct)} raw files {distinct.to_pylist()!r}; pass "
                "raw_file= to scope the check to one acquisition."
            )

    left = psms.select(
        ["psm_id", "raw_file", "scan", "mass_analyzer", "fragmentation"]
    )
    right = scan_metadata.select(["scan", "analyzer", "activation_type"]).rename_columns(
        ["scan", "scan_analyzer", "scan_activation"]
    )
    right = right.append_column(
        "_scan_present", pa.array([True] * right.num_rows, type=pa.bool_())
    )
    joined = left.join(right, keys="scan", join_type="left outer")

    present = pc.fill_null(joined.column("_scan_present"), False)
    psm_analyzer = joined.column("mass_analyzer")
    scan_analyzer = joined.column("scan_analyzer")
    psm_frag = joined.column("fragmentation")
    scan_activation = joined.column("scan_activation")

    analyzer_ok = pc.fill_null(pc.equal(psm_analyzer, scan_analyzer), False)
    activation_ok = pc.fill_null(
        pc.equal(pc.utf8_lower(psm_frag), scan_activation), False
    )

    status = pc.if_else(
        pc.invert(present),
        "scan_absent",
        pc.if_else(
            pc.and_(analyzer_ok, activation_ok),
            "matched",
            pc.if_else(
                activation_ok,  # analyzer bad, activation ok
                "analyzer_mismatch",
                pc.if_else(
                    analyzer_ok,  # activation bad, analyzer ok
                    "activation_mismatch",
                    "both_mismatch",
                ),
            ),
        ),
    )

    result = pa.table(
        {
            "psm_id": joined.column("psm_id"),
            "raw_file": joined.column("raw_file"),
            "scan": joined.column("scan"),
            "psm_analyzer": psm_analyzer,
            "scan_analyzer": scan_analyzer,
            "psm_fragmentation": psm_frag,
            "scan_activation": scan_activation,
            "status": status,
        }
    )
    return result.sort_by("psm_id")


__all__ = ["cross_validate_against_scan_metadata"]
