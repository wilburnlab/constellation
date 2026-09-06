"""Per-cluster quantification → ``FEATURE_QUANT(feature_origin='cluster_id')``.

Each read is mapped to its cluster (via ``read_map.uniq_id`` →
``cluster_of``), grouped by ``(cluster_id, sample_id)``, and counted.
TPM follows the long-read convention (one read = one transcript, no
length normalisation): ``count × 1e6 / sum_in_sample(count)``. This is
the "quant value propagated forward" the cluster carries alongside its
consensus.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.schemas.quant import FEATURE_QUANT


CLUSTER_ENGINE = "constellation_denovo"


def cluster_feature_quant(
    read_map: pa.Table,
    cluster_of: np.ndarray,
) -> pa.Table:
    """Per-(cluster, sample) counts + TPM as a ``FEATURE_QUANT`` table.

    ``read_map`` carries ``(read_id, uniq_id, sample_id)``; ``cluster_of``
    maps ``uniq_id`` → ``cluster_id`` (``-1`` = dropped).
    """
    if read_map.num_rows == 0:
        return FEATURE_QUANT.empty_table()

    uniq_id = read_map.column("uniq_id").to_numpy(zero_copy_only=False)
    cl = cluster_of[uniq_id]
    keep = cl >= 0
    sample_np = pc.fill_null(read_map.column("sample_id"), -1).to_numpy(
        zero_copy_only=False
    )

    tbl = pa.table(
        {
            "cluster_id": pa.array(cl[keep].astype(np.int64)),
            "sample_id": pa.array(sample_np[keep].astype(np.int64)),
        }
    )
    grouped = tbl.group_by(["cluster_id", "sample_id"]).aggregate(
        [("cluster_id", "count")]
    )
    cluster_col = grouped.column("cluster_id")
    sample_col = grouped.column("sample_id")
    count = pc.cast(grouped.column("cluster_id_count"), pa.float64())

    # Per-sample totals → TPM.
    per_sample = grouped.group_by("sample_id").aggregate([("cluster_id_count", "sum")])
    totals = {
        int(s): float(t)
        for s, t in zip(
            per_sample.column("sample_id").to_pylist(),
            per_sample.column("cluster_id_count_sum").to_pylist(),
        )
    }
    sample_np = sample_col.to_numpy(zero_copy_only=False)
    count_np = count.to_numpy(zero_copy_only=False)
    tpm_np = np.array(
        [
            (c * 1e6 / totals[int(s)]) if totals.get(int(s), 0) else 0.0
            for c, s in zip(count_np, sample_np)
        ],
        dtype=np.float64,
    )

    n = grouped.num_rows
    return pa.table(
        {
            "feature_id": cluster_col,
            "sample_id": sample_col,
            "engine": pa.array([CLUSTER_ENGINE] * n, type=pa.string()),
            "feature_origin": pa.array(["cluster_id"] * n, type=pa.string()),
            "count": count,
            "tpm": pa.array(tpm_np),
            "cpm": pa.nulls(n, pa.float64()),
            "coverage_mean": pa.nulls(n, pa.float32()),
            "coverage_median": pa.nulls(n, pa.float32()),
            "coverage_fraction": pa.nulls(n, pa.float32()),
            "multimap_fraction": pa.nulls(n, pa.float32()),
        },
        schema=FEATURE_QUANT,
    )


__all__ = ["cluster_feature_quant", "CLUSTER_ENGINE"]
