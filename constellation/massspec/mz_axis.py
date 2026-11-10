"""The m/z axis of a bundle, measured from the data with no identifications.

Peptide masses cluster on a lattice about 1.0005 Da apart, because the 20
residues have nearly the same mass excess. Folding a run's signal onto that
period says where its signal clusters; how far that is from where theoretical
peptides cluster is the axis's error (Egertson et al. 2012, FineTune). The
result is a curve, a shift in Th against m/z: an ion-trap's error depends on
m/z (on one Stellar run -0.19 to -0.10 Th over 200-700 m/z) far more than on
retention time (at most 0.025 Th at fixed m/z), so one curve per MS level.

As FineTune, it compares the phase of the fold with a reference's, every 20 Th
over a window of at least +-20 Th, interpolated linearly. Three differences:

- the coefficient is summed directly (``core.signal.phase``). FineTune's
  2**18-point FFT of a binned map is a way of getting that one coefficient;
  a test holds the two equal;
- a window widens until the phase's standard error is small enough, where
  FineTune widens to a total ion current suited to an LTQ's intensity scale.
  A knot that never gets there is left out, and interpolated across;
- MS1 uses the 2+ lattice at half the period, where FineTune treats only MS2.
  That wraps at +-0.25 Th, which an axis has been seen to exceed, so its branch
  is taken from the MS2 curve; without one the level is reported as ambiguous.

Weighting by intensity matters far more than how peaks were picked: on the
run above, 0.009 Th RMS against the identified axis on raw profile points and
0.011 on their local maxima, against 0.057 and 0.059 with every point weighted
alike. A symmetric peak shape does not move the phase at all, so this needs no
centroider. A peak list has a curve of its own, since its centroider adds its
own bias, and the difference from the profile's curve measures that bias.

A curve is stored beside the bundle's points (``<bundle>/mz_axes/<key>.parquet``)
and never written into them: a bundle holds the instrument's values. It applies
only where it is named. Reading (:func:`corrected`) subtracts the shift;
interpolating the shift rather than the mapped axis keeps m/z strictly
increasing past the last knot, where the shift simply continues flat.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from constellation.core import stages
from constellation.core.signal.phase import PhaseFold
from constellation.massspec.bundle import MZ_AXES_DIR, list_mz_axes, point_paths

logger = logging.getLogger(__name__)

#: "The newest curve this bundle has."
LATEST = "latest"
METHOD_VERSION = 1
REFERENCE_FILE = "mass_lattice_reference.parquet"

#: One row per knot of a curve: at ``mz`` the instrument reads ``shift_th`` high.
MZ_AXIS_TABLE = pa.schema(
    [
        pa.field("level", pa.int8(), nullable=False),
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("shift_th", pa.float64(), nullable=False),
        pa.field("se_th", pa.float64(), nullable=False),
        pa.field("coherence", pa.float64(), nullable=False),
        #: Half the window the knot was measured over.
        pa.field("half_window_th", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"MzAxis"},
)

#: The reference: per bin of m/z, the coefficient of theoretical ions, for each
#: kind of ion a level is folded on.
MZ_REFERENCE_TABLE = pa.schema(
    [
        pa.field("kind", pa.string(), nullable=False),
        pa.field("mz_bin", pa.int32(), nullable=False),
        pa.field("re", pa.float64(), nullable=False),
        pa.field("im", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"MzAxisReference"},
)
REFERENCE_FRAGMENTS = "fragment_1plus"
REFERENCE_PRECURSORS = "precursor_2plus"


@dataclass(frozen=True)
class MzAxisSettings:
    """How a curve is measured."""

    #: Knot spacing and the narrowest half window, as FineTune.
    step_th: float = 20.0
    min_half_window_th: float = 20.0
    max_half_window_th: float = 100.0
    half_window_step_th: float = 10.0
    #: A window widens until its phase is known this well.
    target_se_th: float = 0.01
    #: Enough scans per level: about 650 MS2 scans kept within 0.02 Th through
    #: 1200 m/z on a Stellar run, and 234 MS1 scans within 0.014 Th to 1100.
    min_scans: dict[int, int] = field(default_factory=lambda: {1: 200, 2: 5000})
    #: Blocks of the data for the standard error, and the seed of which row
    #: groups are read.
    n_blocks: int = 32
    seed: int = 20261030
    #: Points this close to the isolation window are unfragmented precursor.
    isolation_margin_th: float = 1.0


@dataclass(frozen=True)
class MzAxis:
    """A measured curve, per MS level."""

    key: str
    table: pa.Table
    parameters: dict[str, Any]

    def knots(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        at = np.flatnonzero(self.table.column("level").to_numpy() == level)
        return (
            self.table.column("mz").to_numpy()[at],
            self.table.column("shift_th").to_numpy()[at],
        )

    def shift(self, level: int, mz: np.ndarray) -> np.ndarray:
        """How far the instrument reads high at each ``mz`` of this level:
        interpolated between knots, flat past the last. Zero for a level
        with no curve. A level whose branch the measurement left undetermined
        (``ambiguous_levels``: MS1 with no MS2 curve to take it from, so its
        shift is known only to within half a period, ±0.25 Th) is refused:
        until 2026-11-10 it was applied without a word, and could move every
        point by half a Th."""
        self.check_level(level)
        xs, ys = self.knots(level)
        if xs.size == 0:
            return np.zeros_like(np.asarray(mz, dtype=np.float64))
        return np.interp(np.asarray(mz, dtype=np.float64), xs, ys)

    def check_level(self, level: int) -> None:
        """Refuse a level whose branch is undetermined (see :meth:`shift`)."""
        if int(level) in {int(v) for v in self.parameters.get("ambiguous_levels", ())}:
            raise ValueError(
                f"m/z axis {self.key!r}: its MS{level} curve's branch is undetermined (no MS2 curve to take it "
                f"from, so its shift is known only to within half a period, ±0.25 Th); it cannot be applied "
                f"at MS{level}. Measure the axis on a run with MS2, or apply it to MS2 only"
            )

    def corrected(self, level: int, mz: np.ndarray) -> np.ndarray:
        """Instrument m/z of this level to what it should have read."""
        mz = np.asarray(mz, dtype=np.float64)
        return mz - self.shift(level, mz)

    def observed(self, level: int, mz: np.ndarray) -> np.ndarray:
        """True m/z to what the instrument reads, for querying an index built
        on instrument values. The shift changes slowly, so one step of the
        fixed point is enough."""
        mz = np.asarray(mz, dtype=np.float64)
        return mz + self.shift(level, mz + self.shift(level, mz))


def load_reference(path: Path | str | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The reference coefficients by kind, each indexed by ``mz_bin``, and its
    metadata (``period``, ``bin_width``, ``lo``, ``source``). The packaged one,
    built by ``scripts/build-mass-lattice-reference.py``, unless a path is given."""
    if path is None:
        source = resources.files("constellation.data").joinpath(REFERENCE_FILE)
        with resources.as_file(source) as file:
            table = pq.read_table(file)
    else:
        table = pq.read_table(path)
    meta = json.loads(table.schema.metadata[b"x.massspec.mz_reference"].decode())
    n_bins = int(meta["n_bins"])
    coefficients: dict[str, np.ndarray] = {}
    for kind in set(table.column("kind").to_pylist()):
        rows = table.filter(pc.equal(table.column("kind"), kind))
        values = np.zeros(n_bins, dtype=np.complex128)
        bins = rows.column("mz_bin").to_numpy()
        values[bins] = rows.column("re").to_numpy() + 1j * rows.column("im").to_numpy()
        coefficients[kind] = values
    return coefficients, meta


def _row_groups(paths: list[Path], seed: int) -> Iterator[tuple[Path, int]]:
    """Every row group of the files, in an order fixed by ``seed``. Contiguous
    stretches of scans rather than scattered scans, since a scattered sample
    of a run touches every row group anyway."""
    groups = [
        (path, i) for path in paths for i in range(pq.ParquetFile(path).metadata.num_row_groups)
    ]
    for at in np.random.default_rng(seed).permutation(len(groups)):
        yield groups[int(at)]


def _folds_of(
    bundle: Path, settings: MzAxisSettings, reference_meta: dict[str, Any]
) -> tuple[dict[int, PhaseFold], dict[int, int], dict[int, np.ndarray]]:
    """Fold enough of the bundle's points, level by level, in one pass."""
    period = float(reference_meta["period"])
    lo = float(reference_meta["lo"])
    hi = lo + float(reference_meta["bin_width"]) * int(reference_meta["n_bins"])
    fold_of = {
        level: PhaseFold(
            period / harmonic(level), lo, hi,
            bin_width=float(reference_meta["bin_width"]), n_blocks=settings.n_blocks,
        )
        for level in (1, 2)
    }
    seen: dict[int, set[int]] = {1: set(), 2: set()}
    rt_sum = {level: np.zeros(settings.n_blocks) for level in fold_of}
    rt_weight = {level: np.zeros(settings.n_blocks) for level in fold_of}
    columns = ["scan", "level", "rt", "mz", "intensity", "isolation_lower", "isolation_upper"]
    for read, (path, i) in enumerate(_row_groups(point_paths(bundle), settings.seed)):
        if all(len(seen[lv]) >= settings.min_scans.get(lv, 0) for lv in seen):
            break
        table = pq.ParquetFile(path).read_row_group(i, columns=columns)
        levels = table.column("level").to_numpy()
        for level, fold in fold_of.items():
            at = np.flatnonzero(levels == level)
            if at.size == 0:
                continue
            taken = table.take(pa.array(at))
            mz = taken.column("mz").to_numpy()
            intensity = taken.column("intensity").to_numpy()
            keep = intensity > 0
            if level >= 2:
                low = taken.column("isolation_lower").to_numpy(zero_copy_only=False)
                high = taken.column("isolation_upper").to_numpy(zero_copy_only=False)
                margin = settings.isolation_margin_th
                keep &= ~((mz >= low - margin) & (mz <= high + margin))
            block = read % settings.n_blocks
            fold.add(torch.from_numpy(mz[keep]), torch.from_numpy(intensity[keep]), block)
            seen[level].update(taken.column("scan").to_numpy().tolist())
            rts = taken.column("rt").to_numpy()
            rt_sum[level][block] += float(rts.sum())
            rt_weight[level][block] += rts.size
    block_rt = {
        level: np.where(rt_weight[level] > 0, rt_sum[level] / np.maximum(rt_weight[level], 1), np.nan)
        for level in fold_of
    }
    return fold_of, {level: len(s) for level, s in seen.items()}, block_rt


def harmonic(level: int) -> int:
    """The lattice a level is folded on, as a fraction of the period: fragments
    are mostly 1+, survey ions mostly 2+."""
    return 2 if level == 1 else 1


def _reference_kind(level: int) -> str:
    return REFERENCE_PRECURSORS if level == 1 else REFERENCE_FRAGMENTS


def _knots(
    fold: PhaseFold,
    reference: np.ndarray,
    level: int,
    meta: dict[str, Any],
    settings: MzAxisSettings,
) -> list[dict[str, float]]:
    """A knot every step, each from the narrowest window that pins it down."""
    extent = fold.occupied()
    if extent is None:
        return []
    bin_width = float(meta["bin_width"])
    lo_axis = float(meta["lo"])
    in_turns = fold.period / (2.0 * math.pi)
    first = math.ceil(extent[0] / settings.step_th) * settings.step_th

    def reference_sum(lo: float, hi: float) -> complex:
        a = max(int(math.floor((lo - lo_axis) / bin_width)), 0)
        b = int(math.ceil((hi - lo_axis) / bin_width))
        return complex(reference[a:b].sum())

    knots = []
    for alpha in np.arange(first, extent[1], settings.step_th):
        half = settings.min_half_window_th
        while half <= settings.max_half_window_th:
            # As FineTune: a window off the edge of the data moves inward.
            lo = float(alpha) - half
            hi = float(alpha) + half
            if lo < extent[0]:
                lo, hi = extent[0], extent[0] + 2 * half
            if hi > extent[1]:
                lo, hi = max(extent[1] - 2 * half, extent[0]), extent[1]
            ref = reference_sum(lo, hi)
            window = fold.window(lo, hi, reference=ref) if ref != 0 else None
            if window is not None and window.se * in_turns <= settings.target_se_th:
                knots.append(
                    {
                        "mz": float(alpha),
                        "shift_th": window.phase * in_turns,
                        "se_th": window.se * in_turns,
                        "coherence": window.coherence,
                        "half_window_th": half,
                    }
                )
                break
            half += settings.half_window_step_th
    return knots


def _unwrapped(knots: list[dict[str, float]], period: float) -> None:
    """Make a curve continuous, in place: a phase only says where within the
    period, so each knot takes the branch nearest its neighbour, working out
    from the best-determined one."""
    if not knots:
        return
    start = int(np.argmin([k["se_th"] for k in knots]))
    for order in (range(start + 1, len(knots)), range(start - 1, -1, -1)):
        previous = knots[start]["shift_th"]
        for i in order:
            knots[i]["shift_th"] += period * round((previous - knots[i]["shift_th"]) / period)
            previous = knots[i]["shift_th"]


def estimate_mz_axis(
    bundle_dir: Path | str,
    *,
    settings: MzAxisSettings | None = None,
    reference: tuple[dict[str, np.ndarray], dict[str, Any]] | None = None,
) -> MzAxis:
    """Measure a bundle's m/z axis (see the module's docstring). Reads what it
    needs of the bundle's points and writes nothing; :func:`save_mz_axis` keeps
    the result."""
    settings = settings or MzAxisSettings()
    coefficients, meta = reference or load_reference()
    bundle = Path(bundle_dir)
    folds, scans, block_rt = _folds_of(bundle, settings, meta)
    period = float(meta["period"])
    knots: dict[int, list[dict[str, float]]] = {}
    for level, fold in folds.items():
        knots[level] = _knots(fold, coefficients[_reference_kind(level)], level, meta, settings)
        _unwrapped(knots[level], fold.period)

    # MS1's branch from the MS2 curve at the same m/z: the two levels agree to
    # hundredths of a Th, and its own period only says within +-0.25 Th.
    ambiguous: list[int] = []
    if knots[1]:
        if knots[2]:
            xs = np.array([k["mz"] for k in knots[2]])
            ys = np.array([k["shift_th"] for k in knots[2]])
            gaps = [float(np.interp(k["mz"], xs, ys)) - k["shift_th"] for k in knots[1]]
            turns = round(float(np.median(gaps)) / folds[1].period)
            for k in knots[1]:
                k["shift_th"] += turns * folds[1].period
        else:
            ambiguous.append(1)

    # Early against late in the run, for the record; not corrected. A median
    # over the knots, since half the data at each is noisy.
    drift: dict[str, float] = {}
    for level, fold in folds.items():
        rts = block_rt[level]
        if not knots[level] or np.isnan(rts).all():
            continue
        middle = np.nanmedian(rts)
        early = torch.from_numpy(np.nan_to_num(rts, nan=np.inf) <= middle)
        late = torch.from_numpy(np.nan_to_num(rts, nan=-np.inf) > middle)
        reach = fold.period / (2.0 * math.pi)
        gaps: list[float] = []
        for k in knots[level]:
            lo, hi = k["mz"] - k["half_window_th"], k["mz"] + k["half_window_th"]
            a = fold.window(lo, hi, blocks=early)
            b = fold.window(lo, hi, blocks=late)
            if a.blocks and b.blocks:
                gaps.append(abs(math.remainder(a.phase - b.phase, 2.0 * math.pi)) * reach)
        if gaps:
            drift[str(level)] = float(np.median(gaps))

    rows = [
        {"level": level, **knot}
        for level in sorted(knots)
        for knot in sorted(knots[level], key=lambda k: k["mz"])
    ]
    table = pa.Table.from_pylist(rows, schema=MZ_AXIS_TABLE)
    settings_record = {**asdict(settings), "min_scans": {str(k): v for k, v in settings.min_scans.items()}}
    parameters = {
        "method": "mass_lattice",
        "method_version": METHOD_VERSION,
        "reference": meta.get("source"),
        "period": period,
        "settings": settings_record,
        "n_scans": {str(k): v for k, v in scans.items()},
        "ambiguous_levels": ambiguous,
        "drift_th": drift,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    key = stages.stage_key(
        method="mass_lattice",
        method_version=METHOD_VERSION,
        reference=meta.get("source"),
        settings=settings_record,
    )[:8]
    return MzAxis(key=key, table=table, parameters=parameters)


def save_mz_axis(bundle_dir: Path | str, axis: MzAxis) -> Path:
    """Keep a curve beside the bundle's points; returns its file. The same
    measurement again replaces itself."""
    directory = Path(bundle_dir) / MZ_AXES_DIR
    directory.mkdir(exist_ok=True)
    path = directory / f"{axis.key}.parquet"
    table = axis.table.replace_schema_metadata(
        {
            **(axis.table.schema.metadata or {}),
            b"x.massspec.mz_axis": json.dumps(axis.parameters).encode(),
        }
    )
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    return path


def load_mz_axis(bundle_dir: Path | str, key: str = LATEST) -> MzAxis | None:
    """A curve by key, or the newest with ``latest``; None when the bundle has
    no curve at all. An unknown key is an error, since it was named."""
    if key == LATEST:
        keys = list_mz_axes(bundle_dir)
        if not keys:
            return None
        key = keys[-1]
    path = Path(bundle_dir) / MZ_AXES_DIR / f"{key}.parquet"
    if not path.is_file():
        raise ValueError(f"{Path(bundle_dir).name} has no m/z axis {key!r}: it has {list_mz_axes(bundle_dir)}")
    table = pq.read_table(path)
    parameters = json.loads(table.schema.metadata[b"x.massspec.mz_axis"].decode())
    return MzAxis(key=key, table=table, parameters=parameters)


def scale_error_ppm(axis: MzAxis, level: int) -> float | None:
    """The straight line through a level's curve as an error in scale, in ppm:
    what a frequency offset of the fold would be. None with under 2 knots."""
    xs, ys = axis.knots(level)
    if xs.size < 2:
        return None
    return float(np.polyfit(xs, ys, 1)[0] * 1e6)


def describe(axis: MzAxis) -> list[str]:
    """Lines saying what a curve holds."""
    lines = []
    for level in sorted(set(axis.table.column("level").to_pylist())):
        xs, ys = axis.knots(level)
        drift = axis.parameters.get("drift_th", {}).get(str(level))
        scale = scale_error_ppm(axis, level)
        lines.append(
            f"MS{level}: {xs.size} knots over {xs.min():.0f}-{xs.max():.0f} m/z, shift "
            f"{ys.min():+.3f} to {ys.max():+.3f} Th (median {np.median(ys):+.3f})"
            + (f", overall {scale:+.0f} ppm" if scale is not None else "")
            + (f", typically {drift:.3f} Th between early and late scans" if drift is not None else "")
        )
    for level in axis.parameters.get("ambiguous_levels", ()):
        lines.append(
            f"MS{level}: no MS2 curve to take its branch from, so it is known only within "
            f"half a period"
        )
    return lines


__all__ = [
    "LATEST",
    "METHOD_VERSION",
    "MZ_AXES_DIR",
    "MZ_AXIS_TABLE",
    "MZ_REFERENCE_TABLE",
    "MzAxis",
    "MzAxisSettings",
    "describe",
    "estimate_mz_axis",
    "harmonic",
    "list_mz_axes",
    "load_mz_axis",
    "load_reference",
    "save_mz_axis",
    "scale_error_ppm",
]
