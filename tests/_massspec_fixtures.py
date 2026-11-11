"""Synthetic converted acquisitions (bundles) shared by the mass-spec tests.

Small DIA and DDA runs shaped like real ones (scan order, isolation
windows, precursor links, filter strings), and a writer that lays them out
as ``massspec convert`` does: ``peaks/`` (or ``profile/``),
``scan_metadata.parquet``, ``acquisition_metadata.parquet`` and, with
:func:`write_bundle_manifest`, the manifest that makes the directory a bundle.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from constellation.massspec.bundle import BUNDLE_SCHEMA_VERSION, PROFILE_BUNDLE_KIND
from constellation.massspec.schemas import (
    ACQUISITION_METADATA_TABLE,
    MS_PEAK_TABLE,
    SCAN_METADATA_TABLE,
)

MS1_FILTER = "ITMS + c NSI r Full ms [350.0000-1250.0000]"


def dia_scans(
    *,
    cycles: int = 12,
    windows: tuple[tuple[float, float], ...] = ((400.0, 402.0), (402.0, 404.0), (404.0, 406.0)),
    seed: int = 0,
    empty_scan: int | None = None,
) -> list[dict[str, Any]]:
    """A small DIA run: ``cycles`` × (1 MS1 + one MS2 per window)."""
    rng = np.random.default_rng(seed)
    scans: list[dict[str, Any]] = []
    number = 0
    for _cycle in range(cycles):
        for level, window in [(1, None), *[(2, w) for w in windows]]:
            number += 1
            n_peaks = 0 if number == empty_scan else int(rng.integers(3, 9))
            mz = np.sort(rng.uniform(200.0, 1500.0, n_peaks))
            scan = {
                "scan": number,
                "level": level,
                # Not representable in float32, so rounding is exercised.
                "rt": number * 0.731 + 0.0123456789,
                "iit": 31.043 if level == 2 else 0.753,
                "mz": mz,
                "intensity": rng.uniform(10.0, 5e5, n_peaks),
                "filter_string": MS1_FILTER,
            }
            if window is not None:
                lo, hi = window
                target = (lo + hi) / 2.0 + 0.4324
                scan.update(
                    isolation_lower=target - 1.0,
                    isolation_upper=target + 1.0,
                    precursor_mz=target,
                    ms2_isolation_offset=0.0,
                    filter_string=f"ITMS + c NSI r Full ms2 {target:.4f}@hcd26.00 [200.0000-1500.0000]",
                    activation_type="hcd",
                )
            scans.append(scan)
    return scans


def dda_scans(*, n_ms1: int = 4, per_ms1: int = 5, seed: int = 1) -> list[dict[str, Any]]:
    """A small DDA run: every MS2 isolates a different precursor."""
    rng = np.random.default_rng(seed)
    scans: list[dict[str, Any]] = []
    number = 0
    for _ in range(n_ms1):
        number += 1
        parent = number
        mz = np.sort(rng.uniform(350.0, 1250.0, 6))
        scans.append(
            {"scan": number, "level": 1, "rt": number * 0.4, "iit": 1.0, "mz": mz,
             "intensity": rng.uniform(1e3, 1e6, 6), "filter_string": MS1_FILTER}
        )  # fmt: skip
        for _ in range(per_ms1):
            number += 1
            # Five decimals of real precision; the filter string keeps four.
            target = round(float(rng.uniform(400.0, 1000.0)), 5)
            offset = 0.25
            mz = np.sort(rng.uniform(200.0, 1500.0, 5))
            scans.append(
                {"scan": number, "level": 2, "rt": number * 0.4, "iit": 11.333, "mz": mz,
                 "intensity": rng.uniform(10.0, 1e5, 5),
                 "isolation_lower": target + offset - 0.8,
                 "isolation_upper": target + offset + 0.8,
                 "precursor_mz": target, "precursor_charge": 2, "precursor_scan": float(parent),
                 "ms2_isolation_offset": offset, "activation_type": "hcd",
                 "filter_string": f"ITMS + c NSI r d Full ms2 {target:.4f}@hcd26.00 [200.0000-1500.0000]"}
            )  # fmt: skip
    return scans


def write_bundle(
    root: Path,
    name: str,
    scans: list[dict[str, Any]],
    *,
    profile: bool = False,
    acquisition: dict[str, Any] | None = None,
    row_group_size: int = 17,
) -> Path:
    """Write ``scans`` as a bundle shaped like ``massspec convert`` output.

    The points go in two parts, the scans split between them, under
    ``peaks/``, or with ``profile`` under ``profile/`` (a profile bundle).
    ``acquisition`` adds or replaces acquisition-metadata fields.
    """
    bundle = root / name
    bundle.mkdir(parents=True)
    peak_rows: dict[str, list[Any]] = {f.name: [] for f in MS_PEAK_TABLE}
    meta_rows: list[dict[str, Any]] = []
    for s in scans:
        n = int(s["mz"].size)
        for f in MS_PEAK_TABLE:
            if f.name == "mz":
                peak_rows["mz"].extend(s["mz"].tolist())
            elif f.name == "intensity":
                peak_rows["intensity"].extend(s["intensity"].tolist())
            else:
                peak_rows[f.name].extend([s.get(f.name)] * n)
        row = {k: v for k, v in s.items() if k not in ("mz", "intensity")}
        row["peak_count"] = n
        if profile:
            row["spectrum_type"] = "profile"
        if s["level"] == 1 and "scan_window_lower" not in s:
            row["scan_window_lower"], row["scan_window_upper"] = 350.0, 1250.0
        meta_rows.append(row)
    peaks = pa.table(
        {k: pa.array(v, MS_PEAK_TABLE.field(k).type) for k, v in peak_rows.items()},
        schema=MS_PEAK_TABLE,
    )
    # Split at a scan boundary near the middle; within each part, scans
    # still straddle row groups.
    scan_col = peaks.column("scan").to_numpy()
    cut = int(np.searchsorted(scan_col, scan_col[len(scan_col) // 2])) if len(scan_col) else 0
    listing = bundle / ("profile" if profile else "peaks")
    listing.mkdir()
    for i, part in enumerate((peaks.slice(0, cut), peaks.slice(cut))):
        pq.write_table(part, listing / f"part-{i:05d}.parquet", row_group_size=row_group_size)
    scan_table = pa.Table.from_pylist(meta_rows, schema=SCAN_METADATA_TABLE)
    pq.write_table(scan_table, bundle / "scan_metadata.parquet")
    acquisition = {
        "source_format": "thermo_raw",
        "source_path": f"/data/{name}.raw",
        "constellation_version": "0.0.0",
        "schema_version": 1,
        "instrument_model": "Stellar",
        "instrument_serial": "QLT-S-10199",
        "software_version": "1.0.4354",
        "creation_date": "10/07/2026 12:15:40",
        "analyzers": ["ITMS"],
        "n_scans": len(scans),
        **(acquisition or {}),
    }
    pq.write_table(
        pa.Table.from_pylist([acquisition], schema=ACQUISITION_METADATA_TABLE),
        bundle / "acquisition_metadata.parquet",
    )
    return bundle


def with_scan_stats(scans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the TIC and base peak a Thermo conversion records per scan."""
    for s in scans:
        if s["mz"].size:
            top = int(np.argmax(s["intensity"]))
            s["tic"] = float(s["intensity"].sum())
            s["base_peak_mz"] = float(s["mz"][top])
            s["base_peak_intensity"] = float(s["intensity"][top])
    return scans


def write_bundle_manifest(
    bundle: Path,
    *,
    kind: str | None = None,
    sha256: str | None = None,
    run: str | None = None,
) -> Path:
    """Give a :func:`write_bundle` directory the manifest a real bundle has.

    ``run`` is the run's name when it is not the directory's (a profile
    bundle, a derived peak list); ``sha256`` what ties a run's bundles.
    """
    name = run or bundle.name
    profile = (bundle / "profile").is_dir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "kind": kind or (PROFILE_BUNDLE_KIND if profile else "thermo_acquisition"),
                "source_file": f"/data/{name}.raw",
                "source_sha256": sha256 or f"sha-of-{name}",
                "parameters": {"rt_bin_width_s": 60.0},
                "outputs": {
                    "profile" if profile else "peaks": "profile" if profile else "peaks",
                    "scan_metadata": "scan_metadata.parquet",
                    "acquisition_metadata": "acquisition_metadata.parquet",
                },
                "run_metadata": {
                    "instrument_model": "Stellar",
                    "instrument_serial": "QLT-S-10199",
                    "creation_date": "10/07/2026 12:15:40",
                },
            }
        )
    )
    return bundle


# ── a run with peptides planted in it ─────────────────────────────────

#: Planted by default: a pair one missed cleavage apart, a peptide and its
#: oxidised form, and two unrelated peptides, each at its own apex (s).
PLANTED_PEPTIDES: tuple[tuple[str, int, float], ...] = (
    ("LVNELTEFAK", 2, 60.0),
    ("LVNELTEFAKTCVADESHAGCEK", 3, 64.0),
    ("PEPTMIDEK", 2, 90.0),
    ("PEPTM[Oxidation]IDEK", 2, 84.0),
    ("HLVDEPQNLIK", 2, 120.0),
    ("YICDNQDTISSK", 2, 61.0),
)


@dataclass
class PlantedRun:
    """A synthetic run and the targets for what was planted in it."""

    scans: list[dict[str, Any]]
    targets: pa.Table
    #: (scan, m/z) of a scan holding two peaks at exactly the same m/z.
    duplicate: tuple[int, float]
    #: (scan, m/z, theoretical m/z) of a peak placed exactly 20 ppm from an ion.
    edge: tuple[int, float, float]
    #: What was planted, per peptide (in ``targets`` order): ``frag_mz`` and
    #: ``frag_ab``, its fragments and their relative abundances, and the rest.
    planted: list[dict[str, Any]] = field(default_factory=list)


def drop_unacquired(trace: pa.Table, bundle: Path) -> pa.Table:
    """An XIC trace (``XIC_TRACE_TABLE``) without its zero-filled rows of
    channels outside their scan's acquired m/z range, as a sweep of
    ``bundle`` gives it (trace v2, 2026-11-10): the frozen oracle's output
    held to that rule. A matched row is always kept."""
    meta = pq.read_table(Path(bundle) / "scan_metadata.parquet",
                         columns=["scan", "scan_window_lower", "scan_window_upper"])  # fmt: skip
    scans = meta.column("scan").to_numpy().astype(np.int64)
    lo = meta.column("scan_window_lower").to_numpy(zero_copy_only=False).astype(np.float64)
    hi = meta.column("scan_window_upper").to_numpy(zero_copy_only=False).astype(np.float64)
    by = np.argsort(scans)
    at = by[np.searchsorted(scans[by], trace.column("scan").to_numpy().astype(np.int64))]
    mz = trace.column("mz_theoretical").to_numpy().astype(np.float64)
    observed = trace.column("mz_observed").to_numpy(zero_copy_only=False).astype(np.float64)
    outside = (mz < lo[at]) | (mz > hi[at])
    return trace.filter(pa.array(~(np.isnan(observed) & outside)))


def _gauss(rt: float, apex: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((rt - apex) / sigma) ** 2))


def planted_run(
    *,
    acquisition: str = "dia",
    peptides: Sequence[tuple[str, int, float]] = PLANTED_PEPTIDES,
    cycles: int = 60,
    cycle_s: float = 3.0,
    sigma_s: float = 5.0,
    window_width: float = 50.0,
    window_range: tuple[float, float] = (350.0, 1250.0),
    staggered: bool = False,
    noise_peaks: int = 25,
    error_ppm: float = 2.0,
    seed: int = 0,
) -> PlantedRun:
    """A DIA or DDA run in which ``peptides`` (modseq, charge, apex RT) elute
    as Gaussians: precursor isotopes in MS1, b/y fragments in the MS2 scans
    that isolate them, each peak ``error_ppm`` from theory, over uniform
    noise. One scan carries a duplicated peak and one a peak exactly 20 ppm
    from an ion, so tie and tolerance-edge handling are exercised. DIA
    windows tile ``window_range``; ``staggered`` offsets alternate cycles by
    half a window. DDA isolates each peptide in the cycles near its apex,
    plus one off-target MS2 per cycle."""
    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.peptide.envelope import peptide_envelope
    from constellation.massspec.peptide.ions import fragment_ladder
    from constellation.massspec.peptide.mz import precursor_mz
    from constellation.massspec.quant.schemas import XIC_TARGET_TABLE

    rng = np.random.default_rng(seed)
    planted = []
    for modseq, z, apex in peptides:
        pf = parse_proforma(modseq)
        iso_mz, iso_ab = peptide_envelope(pf, charge=z, n_peaks=3)
        ladder, _ = fragment_ladder(pf, max_fragment_charge=min(2, z), return_tensor=False)
        frag_mz = np.asarray(ladder.column("mz_theoretical").to_numpy(), dtype=np.float64)
        frag_mz = frag_mz[(frag_mz > 200.0) & (frag_mz < 1500.0)]
        planted.append(
            {
                "modseq": modseq,
                "z": z,
                "apex": apex,
                "pmz": precursor_mz(pf, charge=z),
                "iso_mz": iso_mz.numpy().astype(np.float64),
                "iso_ab": iso_ab.numpy().astype(np.float64),
                "frag_mz": frag_mz,
                "frag_ab": rng.uniform(0.1, 1.0, frag_mz.size),
                "scale": float(rng.uniform(2e5, 2e6)),
            }
        )

    lo_mz, hi_mz = window_range
    edges = np.arange(lo_mz, hi_mz + 1e-9, window_width)
    base = list(zip(edges[:-1], edges[1:]))

    def peaks(rt, ions):
        mz = [rng.uniform(200.0, 1500.0, noise_peaks)]
        inten = [rng.uniform(1e2, 3e3, noise_peaks)]
        if ions:
            theo, ab, scale = (np.asarray(v, dtype=np.float64) for v in zip(*ions))
            mz.append(theo * (1.0 + error_ppm * 1e-6))
            inten.append(ab * scale)
        mz_a, int_a = np.concatenate(mz), np.concatenate(inten)
        keep = int_a > 50.0
        order = np.argsort(mz_a[keep], kind="stable")
        return mz_a[keep][order], int_a[keep][order]

    scans: list[dict[str, Any]] = []
    number = 0

    def add(level, rt, ions, window=None, target=None, charge=None):
        nonlocal number
        number += 1
        mz, inten = peaks(rt, ions)
        scan = {"scan": number, "level": level, "rt": rt, "iit": 10.0, "mz": mz,
                "intensity": inten, "filter_string": MS1_FILTER}  # fmt: skip
        if level == 2:
            lo, hi = window
            scan.update(
                isolation_lower=lo, isolation_upper=hi, precursor_mz=target,
                precursor_charge=charge, ms2_isolation_offset=0.0, activation_type="cid",
                filter_string=f"ITMS + c NSI r Full ms2 {target:.4f}@cid30.00 [200.0000-1500.0000]",
            )  # fmt: skip
        scans.append(scan)

    for cycle in range(cycles):
        t0 = cycle * cycle_s
        ms1_ions = []
        for p in planted:
            g = _gauss(t0, p["apex"], sigma_s)
            if g > 1e-3:
                ms1_ions.extend(zip(p["iso_mz"], p["iso_ab"], [p["scale"] * g] * 3))
        add(1, t0 + 0.0123456789, ms1_ions)
        if acquisition == "dia":
            shift = window_width / 2.0 if staggered and cycle % 2 else 0.0
            for k, (lo, hi) in enumerate(base):
                lo, hi = lo + shift, hi + shift
                rt = t0 + (k + 1) * cycle_s / (len(base) + 2)
                ions = []
                for p in planted:
                    if lo <= p["pmz"] <= hi:
                        g = _gauss(rt, p["apex"], sigma_s)
                        if g > 1e-3:
                            ions.extend(zip(p["frag_mz"], p["frag_ab"], [p["scale"] * g] * p["frag_mz"].size))
                add(2, rt, ions, (lo, hi), (lo + hi) / 2.0)
        else:
            chosen = [p for p in planted if abs(t0 - p["apex"]) < 2.5 * sigma_s]
            for k, p in enumerate(chosen):
                rt = t0 + (k + 1) * cycle_s / (len(chosen) + 3)
                g = _gauss(rt, p["apex"], sigma_s)
                ions = list(zip(p["frag_mz"], p["frag_ab"], [p["scale"] * g] * p["frag_mz"].size))
                add(2, rt, ions, (p["pmz"] - 0.8, p["pmz"] + 0.8), p["pmz"], p["z"])
            off = float(rng.uniform(400.0, 1000.0))
            add(2, t0 + cycle_s * 0.95, [], (off - 0.8, off + 0.8), off, 2)

    # A duplicated peak, and a peak exactly 20 ppm above a fragment, in the
    # MS2 scan nearest the first peptide's apex that isolates it.
    first = planted[0]
    ms2 = [s for s in scans if s["level"] == 2
           and s["isolation_lower"] <= first["pmz"] <= s["isolation_upper"]]  # fmt: skip
    host = min(ms2, key=lambda s: abs(s["rt"] - first["apex"]))
    dup = float(host["mz"][host["mz"].size // 2])
    theo = float(first["frag_mz"][first["frag_mz"].size // 2])
    edge_mz = theo + 20.0 * theo * 1e-6
    mz = np.concatenate((host["mz"], [dup, edge_mz]))
    inten = np.concatenate((host["intensity"], [7.5e4, 3.3e4]))
    order = np.argsort(mz, kind="stable")
    host["mz"], host["intensity"] = mz[order], inten[order]

    targets = pa.Table.from_pylist(
        [
            {
                "target_id": 1000 + i,
                "modified_sequence": p["modseq"],
                "precursor_charge": p["z"],
                "precursor_mz": p["pmz"],
                "rt_center": p["apex"],
                "precursor_id": 5000 + i,
            }
            for i, p in enumerate(planted)
        ],
        schema=XIC_TARGET_TABLE,
    )
    return PlantedRun(
        scans=scans,
        targets=targets,
        duplicate=(host["scan"], dup),
        edge=(host["scan"], edge_mz, theo),
        planted=planted,
    )
