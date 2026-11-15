"""Which of a round's M-step shards are the round's.

``run_batched`` writes ``rNN/mstep/<key>/part-NNNNN.parquet``, one shard per
unit, and never clears the directory; the number of units is the worker count
times ``units_per_worker``, capped by the live templates. So a round that ran
once with more units than its re-run leaves the extra shards behind, and
anything that GLOBS the directory reads them as the round's. The loop itself
never did (it reads the shard paths its own call returned), but the resume
that rebuilds a finished round's successor did — and extending a finished run
with ``--resume --rounds N+k`` always takes that path, because a final round
never writes its successor's templates — and so did the diagnostics. Stale
node rows become duplicate templates.

Since this module, ``_run_mstep`` clears ``mstep/`` before it writes and
records ``mstep.n_units`` and ``mstep.n_membership_rows`` in ``round.json``;
every reader goes through :func:`round_mstep_shards`, which takes exactly the
recorded shards. A round written before that has only ``n_nodes``: its shards
are the shortest prefix whose node rows add up to it with every
``(parent_template_id, haplotype_id)`` once — the attempt that finished wrote
its shards from index zero — and anything else is refused rather than
guessed. A directory with no ``round.json`` at all is not a round the loop
finished (resume reads only those); it is read as the contiguous run of
shards from ``part-00000``, the most that can be said of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
    NODE_MEMBERSHIP_TABLE,
    REFINED_NODE_TABLE,
    in_membership_order,
    in_node_order,
)

MSTEP_DIR = "mstep"
NODES_KEY = "nodes"
MEMBERSHIP_KEY = "node_membership"


class StaleShardsError(ValueError):
    """The M-step shards on disk are not the ones the round recorded."""


def round_record(rd: Path) -> dict:
    """``rNN/round.json``, or ``{}``."""
    path = Path(rd) / "round.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _shard_index(path: Path) -> int:
    return int(path.stem.split("-", 1)[1])


def _all_shards(rd: Path, key: str) -> list[Path]:
    d = Path(rd) / MSTEP_DIR / key
    return sorted(d.glob("part-*.parquet"), key=_shard_index)


def _named(rd: Path, key: str, n: int) -> list[Path]:
    """Shards ``part-00000 .. part-{n-1}`` of ``key``, all required."""
    want = [Path(rd) / MSTEP_DIR / key / f"part-{i:05d}.parquet" for i in range(n)]
    missing = [p.name for p in want if not p.exists()]
    if missing:
        raise StaleShardsError(
            f"{Path(rd).name}: {n} M-step units, but {key}/ is missing {missing[:3]}"
        )
    return want


def _rows(paths: list[Path]) -> np.ndarray:
    return np.array([pq.read_metadata(p).num_rows for p in paths], dtype=np.int64)


def _keys_unique(paths: list[Path]) -> bool:
    if not paths:
        return True
    t = pa_ds.dataset(paths).to_table(columns=["parent_template_id", "haplotype_id"])
    key = (
        t.column("parent_template_id").to_numpy(zero_copy_only=False).astype(np.int64)
        << np.int64(16)
    ) | t.column("haplotype_id").to_numpy(zero_copy_only=False).astype(np.int64)
    return np.unique(key).size == key.size


def round_mstep_shards(rd: Path, key: str) -> list[Path]:
    """The shards of ``rd/mstep/<key>`` that the round wrote, in unit order.

    Raises :class:`StaleShardsError` when the directory does not hold what the
    round recorded, or (for a round written before the record) when no prefix
    of it is consistent with the round's node count.
    """
    rd = Path(rd)
    record = round_record(rd)
    mstep = record.get("mstep") or {}
    if "n_units" in mstep:
        return _named(rd, key, int(mstep["n_units"]))
    # Written before the record: find the prefix the finished attempt wrote.
    nodes = _all_shards(rd, NODES_KEY)
    if not nodes:
        return []
    # Only a run from part-00000 can be what one attempt wrote; whatever lies
    # past a gap is stale by construction.
    run = 0
    while run < len(nodes) and _shard_index(nodes[run]) == run:
        run += 1
    nodes = nodes[:run]
    if "n_nodes" not in record:
        # No round.json at all: not a round the loop finished (it writes the
        # record before _SUCCESS, and resume reads only finished rounds) —
        # a directory assembled by hand or by a tool. Nothing says which
        # shards are its, so the contiguous run is the whole of the answer.
        return _named(rd, key, run)
    want_nodes = int(record["n_nodes"])
    if want_nodes == 0:
        return []
    cum = np.cumsum(_rows(nodes))
    for k in (np.flatnonzero(cum == want_nodes) + 1).tolist():
        if _keys_unique(nodes[:k]):
            return _named(rd, key, k)
    raise StaleShardsError(
        f"{rd.name}: its M-step shards hold {int(cum[-1]) if cum.size else 0} "
        f"node rows in {len(nodes)} files and no prefix of them is the "
        f"{want_nodes} unique nodes round.json records — left by an earlier "
        f"attempt with more units. Remove {rd / MSTEP_DIR} and re-run the round."
    )


def read_round_nodes(rd: Path, columns: list[str] | None = None) -> pa.Table:
    """A finished round's M-step nodes, in node order, checked.

    The order is ``(parent_template_id, haplotype_id)``, the order the next
    round's template ids are assigned in.
    """
    rd = Path(rd)
    shards = round_mstep_shards(rd, NODES_KEY)
    if not shards:
        return REFINED_NODE_TABLE.empty_table()
    table = in_node_order(pa_ds.dataset(shards, schema=REFINED_NODE_TABLE).to_table())
    record = round_record(rd)
    if "n_nodes" in record and table.num_rows != int(record["n_nodes"]):
        raise StaleShardsError(
            f"{rd.name}: {table.num_rows} node rows on disk, round.json "
            f"records {record['n_nodes']}"
        )
    if not _keys_unique(shards):
        raise StaleShardsError(f"{rd.name}: duplicate (parent, haplotype) nodes")
    return table.select(columns) if columns else table


def read_round_membership(rd: Path) -> pa.Table:
    """A finished round's per-node read membership, in membership order."""
    rd = Path(rd)
    shards = round_mstep_shards(rd, MEMBERSHIP_KEY)
    if not shards:
        return NODE_MEMBERSHIP_TABLE.empty_table()
    table = in_membership_order(
        pa_ds.dataset(shards, schema=NODE_MEMBERSHIP_TABLE).to_table()
    )
    want = (round_record(rd).get("mstep") or {}).get("n_membership_rows")
    if want is not None and table.num_rows != int(want):
        raise StaleShardsError(
            f"{rd.name}: {table.num_rows} membership rows on disk, round.json "
            f"records {want}"
        )
    return table


__all__ = [
    "MEMBERSHIP_KEY",
    "MSTEP_DIR",
    "NODES_KEY",
    "StaleShardsError",
    "read_round_membership",
    "read_round_nodes",
    "round_mstep_shards",
    "round_record",
]
