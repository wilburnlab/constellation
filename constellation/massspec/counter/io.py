"""`CounterResult` container + native ParquetDir round-trip.

A lean bundle of the three Counter tables (`COUNTER_N_TABLE` always; the
calibration + per-peptide-params tables optionally). Persisted as a
directory of Parquet files + a `manifest.json`, mirroring
`massspec.quant.io`'s ParquetDir convention. (The full Reader/Writer
Protocol + registry — as in `quant.io` — is deferred; a follow-up adds it
if a second on-disk format appears.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["CounterResult", "save_counter", "load_counter"]


@dataclass(frozen=True)
class CounterResult:
    """Counter output bundle. `counter_n` is the deliverable; the calibration
    and per-peptide-params tables capture the fitted model state;
    `peak_attribution` is the optional sparse ion→progenitor soft-attribution map
    (`COUNTER_PEAK_ATTRIBUTION_TABLE`) — the "what's left" foundation."""

    counter_n: pa.Table
    global_calibration: pa.Table | None = None
    peptide_params: pa.Table | None = None
    peak_attribution: pa.Table | None = None
    metadata_extras: dict[str, Any] = field(default_factory=dict)


_OPTIONAL = ("global_calibration", "peptide_params", "peak_attribution")


def save_counter(result: CounterResult, path: str | Path) -> None:
    """Write a `CounterResult` as a ParquetDir bundle + manifest."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    pq.write_table(result.counter_n, p / "counter_n.parquet")
    written = ["counter_n"]
    for name in _OPTIONAL:
        tbl = getattr(result, name)
        if tbl is not None:
            pq.write_table(tbl, p / f"{name}.parquet")
            written.append(name)
    manifest = {
        "format": "counter_parquet_dir",
        "tables": written,
        "metadata": result.metadata_extras,
    }
    (p / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_counter(path: str | Path) -> CounterResult:
    """Read a `CounterResult` ParquetDir bundle."""
    p = Path(path)
    manifest: dict[str, Any] = {}
    if (p / "manifest.json").exists():
        manifest = json.loads((p / "manifest.json").read_text())
    counter_n = pq.read_table(p / "counter_n.parquet")
    optional: dict[str, pa.Table | None] = {}
    for name in _OPTIONAL:
        fp = p / f"{name}.parquet"
        optional[name] = pq.read_table(fp) if fp.exists() else None
    return CounterResult(
        counter_n=counter_n,
        global_calibration=optional["global_calibration"],
        peptide_params=optional["peptide_params"],
        peak_attribution=optional["peak_attribution"],
        metadata_extras=dict(manifest.get("metadata", {})),
    )
