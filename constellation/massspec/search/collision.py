"""DIA collision filter — drop co-eluting peptide identifications.

Port of cartographer's ``filter_elib_by_collision``. DIA expands the
search space by querying many isobaric / near-isobaric precursors at
once, so peptides that share fragment ions and elute at the same time
can be co-assigned the same chromatographic signal. The four-step
filter:

    1. Group EncyclopeDIA ``entries`` rows by their GPF isolation window
       (windows come from the ``ranges`` table of a ``combined.dia``,
       deduplicated by rounding to 0.1 m/z).
    2. Within each window, pair rows whose ``RTInSeconds`` differs by
       less than ``rt_threshold_s``.
    3. For each pair, count observed fragment-ion m/z values within
       ``frag_ppm_tol`` ppm; pairs at or above ``min_shared_ions`` are
       flagged as collision pairs.
    4. Build the collision graph, take connected components, and keep
       the lowest-Score peptide per cluster (lower = better in
       EncyclopeDIA). All other modseqs in the cluster are returned as
       collision losers.

Defaults match cartographer's validated 260402 / 260429 parameters:
``rt_threshold_s=5.0``, ``frag_ppm_tol=20.0``, ``min_shared_ions=4``.

Opt-in only: shipped now so it's ready when the v6.5.15 sweep
validates whether EncyclopeDIA handles collisions internally. Wired
behind ``constellation massspec classify-novel-peptides
--collision-filter --dia <combined.dia>``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from constellation.massspec.io.encyclopedia import _sql
from constellation.massspec.io.encyclopedia._codec import decompress_mz


def _merge_isolation_windows(
    starts: list[float], stops: list[float]
) -> list[tuple[float, float]]:
    """Deduplicate ``ranges`` Start/Stop pairs and merge near-duplicates.

    The EncyclopeDIA ``ranges`` table contains one row per acquisition
    cycle so each isolation window appears many times with
    floating-point noise. Cartographer's approach: round to 0.1 m/z,
    group, take the mean.
    """
    seen: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for lo, hi in zip(starts, stops, strict=True):
        key = (round(lo, 1), round(hi, 1))
        seen.setdefault(key, []).append((lo, hi))
    merged: list[tuple[float, float]] = []
    for vals in seen.values():
        lo_mean = sum(v[0] for v in vals) / len(vals)
        hi_mean = sum(v[1] for v in vals) / len(vals)
        merged.append((lo_mean, hi_mean))
    merged.sort()
    return merged


def _assign_window(
    mz: float, windows: list[tuple[float, float]]
) -> tuple[float, float] | None:
    """First window where Start <= mz < Stop; ``None`` if no match."""
    for lo, hi in windows:
        if lo <= mz < hi:
            return (lo, hi)
    return None


def _count_shared_ions(
    mz_a: torch.Tensor, mz_b: torch.Tensor, frag_ppm_tol: float
) -> int:
    """Count m/z values in ``mz_a`` matching at least one in ``mz_b``
    within ``frag_ppm_tol`` ppm."""
    if mz_a.numel() == 0 or mz_b.numel() == 0:
        return 0
    ppm_diff = torch.abs(mz_a[:, None] - mz_b[None, :]) / mz_b[None, :] * 1e6
    return int((ppm_diff < frag_ppm_tol).any(dim=1).sum().item())


def _connected_components(
    pairs: list[tuple[str, str]],
) -> list[set[str]]:
    """Union-find on (a, b) edges → list of component sets."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)
    components: dict[str, set[str]] = {}
    for node in parent:
        root = find(node)
        components.setdefault(root, set()).add(node)
    return list(components.values())


def apply_collision_filter(
    elib_path: str | Path,
    dia_path: str | Path,
    *,
    rt_threshold_s: float = 5.0,
    frag_ppm_tol: float = 20.0,
    min_shared_ions: int = 4,
    return_metadata: bool = False,
) -> set[str] | tuple[set[str], dict]:
    """Identify collision-loser PeptideModSeq strings in an ``.elib``.

    Walks the elib ``entries`` table + the ``combined.dia`` ``ranges``
    table; clusters co-eluting peptide IDs that share observed fragment
    ions in the same GPF window; returns the set of losers (everything
    not the lowest-Score representative per cluster).

    Parameters
    ----------
    elib_path
        Path to the EncyclopeDIA ``.elib`` SQLite search-result file.
    dia_path
        Path to the ``combined.dia`` SQLite file holding the ``ranges``
        table that defines GPF isolation windows.
    rt_threshold_s
        Max |ΔRT| (seconds) for a candidate co-elution pair.
    frag_ppm_tol
        Fragment m/z match tolerance in ppm.
    min_shared_ions
        Minimum shared observed fragment ions to flag a pair.
    return_metadata
        When ``True``, returns ``(losers, metadata)`` where ``metadata``
        is ``{"pairs": [...], "clusters": [{"winner": ..., "losers":
        [...]}], "n_windows": N, "n_entries": M}``.

    Returns
    -------
    set[str]
        ``PeptideModSeq`` strings flagged as collision losers. The
        caller drops these before downstream consumption (e.g. novel
        peptide classification).
    """
    elib = Path(elib_path)
    dia = Path(dia_path)

    # 1. Load isolation windows from .dia ranges table.
    with _sql.open_ro(dia) as con:
        ranges_rows = list(_sql.iter_dia_ranges(con))
    starts = [float(r["Start"]) for r in ranges_rows]
    stops = [float(r["Stop"]) for r in ranges_rows]
    windows = _merge_isolation_windows(starts, stops)

    # 2. Load entries + decompress observed fragment m/z arrays.
    with _sql.open_ro(elib) as con:
        rows = list(_sql.iter_entries(con))

    obs_mz: list[torch.Tensor] = []
    for row in rows:
        blob = row.get("MassArray")
        n = row.get("MassEncodedLength") or 0
        if blob is None or not n:
            obs_mz.append(torch.empty(0, dtype=torch.float64))
            continue
        obs_mz.append(decompress_mz(blob, int(n)))

    precursor_mz = [float(r["PrecursorMz"]) for r in rows]
    rt = [float(r["RTInSeconds"]) for r in rows]
    modseqs: list[str] = [r["PeptideModSeq"] for r in rows]
    scores: list[float | None] = [
        (float(r["Score"]) if r.get("Score") is not None else None)
        for r in rows
    ]
    window_idx = [_assign_window(mz, windows) for mz in precursor_mz]

    # 3. Group rows by window, pair-test within each window.
    win_to_rows: dict[tuple[float, float], list[int]] = {}
    for i, w in enumerate(window_idx):
        if w is None:
            continue
        win_to_rows.setdefault(w, []).append(i)

    pairs: list[tuple[str, str]] = []
    for row_idxs in win_to_rows.values():
        if len(row_idxs) < 2:
            continue
        for ai in range(len(row_idxs)):
            i = row_idxs[ai]
            for bi in range(ai + 1, len(row_idxs)):
                j = row_idxs[bi]
                if abs(rt[i] - rt[j]) >= rt_threshold_s:
                    continue
                if modseqs[i] == modseqs[j]:
                    # Same modseq across SourceFiles is never a collision pair.
                    continue
                shared = _count_shared_ions(
                    obs_mz[i], obs_mz[j], frag_ppm_tol
                )
                if shared >= min_shared_ions:
                    pairs.append((modseqs[i], modseqs[j]))

    # 4. Best-Score-per-cluster wins (lower = better in EncyclopeDIA).
    score_lookup: dict[str, float] = {}
    for ms, sc in zip(modseqs, scores, strict=True):
        if sc is None:
            continue
        if ms not in score_lookup or sc < score_lookup[ms]:
            score_lookup[ms] = sc

    losers: set[str] = set()
    clusters_info: list[dict] = []
    for cluster in _connected_components(pairs):
        scored = [
            (s, score_lookup.get(s))
            for s in cluster
            if score_lookup.get(s) is not None
        ]
        if not scored:
            continue
        scored.sort(key=lambda x: x[1])
        winner = scored[0][0]
        cluster_losers = [s for s, _ in scored[1:]]
        losers.update(cluster_losers)
        clusters_info.append({"winner": winner, "losers": cluster_losers})

    if return_metadata:
        metadata = {
            "pairs": pairs,
            "clusters": clusters_info,
            "n_windows": len(windows),
            "n_entries": len(rows),
            "rt_threshold_s": rt_threshold_s,
            "frag_ppm_tol": frag_ppm_tol,
            "min_shared_ions": min_shared_ions,
        }
        return losers, metadata
    return losers


__all__ = ["apply_collision_filter"]
