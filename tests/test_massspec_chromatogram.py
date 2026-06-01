"""Tests for XIC extraction (Mode A scan-major + Mode B mass-index)."""

from __future__ import annotations

import collections

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.peptide.envelope import peptide_envelope
from constellation.massspec.peptide.ions import IonType, fragment_ladder_indices_batch
from constellation.massspec.peptide.mz import precursor_mz
from constellation.massspec.quant.chromatogram import (
    extract_xic_indexed,
    extract_xic_scan_major,
    load_xic,
    save_xic,
)
from constellation.massspec.quant.peak_index import build_peak_index
from constellation.massspec.quant.schemas import XIC_TRACE_TABLE

_PK = {
    "scan": pa.int32(),
    "level": pa.int8(),
    "rt": pa.float64(),
    "mz": pa.float64(),
    "intensity": pa.float64(),
    "isolation_lower": pa.float64(),
    "isolation_upper": pa.float64(),
    "precursor_mz": pa.float64(),
}
_PEP = "PEPTIDEK"
_NOISE_PPM = 1.5
_TOL_PPM = 10.0


def _peaks(rows):
    cols = collections.defaultdict(list)
    for r in rows:
        for k in _PK:
            cols[k].append(r.get(k))
    return pa.table({k: pa.array(cols[k], _PK[k]) for k in _PK})


def _pk(scan, level, rt, mz, inten, lo=None, hi=None, prec=None):
    return dict(
        scan=scan,
        level=level,
        rt=rt,
        mz=mz,
        intensity=inten,
        isolation_lower=lo,
        isolation_upper=hi,
        precursor_mz=prec,
    )


def _target(
    target_id=1,
    modseq=_PEP,
    charge=2,
    prec_mz=None,
    rt_center=None,
    rt_start=None,
    rt_end=None,
    scan=None,
):
    return pa.table(
        {
            "target_id": pa.array([target_id], pa.int64()),
            "modified_sequence": pa.array([modseq]),
            "precursor_charge": pa.array([charge], pa.int8()),
            "precursor_mz": pa.array([prec_mz], pa.float64()),
            "rt_center": pa.array([rt_center], pa.float64()),
            "rt_start": pa.array([rt_start], pa.float64()),
            "rt_end": pa.array([rt_end], pa.float64()),
            "scan": pa.array([scan], pa.int32()),
            "precursor_id": pa.array([None], pa.int64()),
            "peptide_id": pa.array([None], pa.int64()),
        }
    )


def _ms1_dataset(signal_scans=(1, 2, 3)):
    """MS1 peaks: PEPTIDEK 2+ isotopes (with +1.5 ppm error) in signal scans,
    plus a noise peak in every scan."""
    mz2, _ = peptide_envelope(parse_proforma(_PEP), charge=2, n_peaks=3)
    mz2 = mz2.tolist()
    rows = []
    for s in range(5):
        rt = 100.0 + s
        rows.append(_pk(s, 1, rt, 123.456, 5.0))
        if s in signal_scans:
            for k, m in enumerate(mz2):
                rows.append(
                    _pk(s, 1, rt, m * (1 + _NOISE_PPM / 1e6), 1000.0 * s + 10 * k)
                )
    return _peaks(rows), mz2


def _charge2(table):
    return table.filter(pc.equal(table["precursor_charge"], 2))


# ── MS1 ───────────────────────────────────────────────────────────────


def test_ms1_schema_and_signal_recovery():
    peaks, mz2 = _ms1_dataset()
    out = extract_xic_scan_major(
        peaks,
        _target(prec_mz=mz2[0], rt_center=102.0),
        acquisition_id=0,
        level=1,
        n_isotopes=3,
        rt_window=2.0,
        tolerance=_TOL_PPM,
    )
    assert out.schema.equals(XIC_TRACE_TABLE)
    sub = _charge2(out)
    assert sub.num_rows == 9  # 3 isotopes × 3 signal scans
    errs = sub.column("mz_error_ppm").to_pylist()
    assert all(1.0 < e < 2.0 for e in errs)  # observed - theoretical ~ +1.5 ppm


def test_ms1_artifact_unmatched_and_isotope_axis():
    peaks, mz2 = _ms1_dataset()
    out = _charge2(
        extract_xic_scan_major(
            peaks,
            _target(prec_mz=mz2[0], rt_center=102.0),
            acquisition_id=0,
            level=1,
            n_isotopes=3,
            rt_window=2.0,
            tolerance=_TOL_PPM,
        )
    )
    assert set(out.column("isotope").to_pylist()) == {0, 1, 2}
    assert out.column("ion_type").null_count == out.num_rows  # MS1 → no fragment coords


def test_ms1_rt_window_excludes_outside():
    peaks, mz2 = _ms1_dataset(signal_scans=(1, 2, 3))
    # window ±0 around rt 102 (scan 2) → only scan 2
    out = _charge2(
        extract_xic_scan_major(
            peaks,
            _target(prec_mz=mz2[0], rt_center=102.0),
            acquisition_id=0,
            level=1,
            n_isotopes=3,
            rt_window=0.0,
            tolerance=_TOL_PPM,
        )
    )
    assert set(out.column("scan").to_pylist()) == {2}


def test_ms1_rt_window_none_uses_all_scans():
    peaks, mz2 = _ms1_dataset(signal_scans=(1, 2, 3))
    out = _charge2(
        extract_xic_scan_major(
            peaks,
            _target(prec_mz=mz2[0]),
            acquisition_id=0,
            level=1,
            n_isotopes=3,
            rt_window=None,
            tolerance=_TOL_PPM,
        )
    )
    assert set(out.column("scan").to_pylist()) == {1, 2, 3}


def test_ms1_charge_range_expansion_adds_charges():
    peaks, mz2 = _ms1_dataset()
    out = extract_xic_scan_major(
        peaks,
        _target(prec_mz=mz2[0], rt_center=102.0),
        acquisition_id=0,
        level=1,
        n_isotopes=1,
        charge_range=(1, 4),
        rt_window=2.0,
        tolerance=_TOL_PPM,
    )
    # all four charges generated as queries even though only z=2 matches
    out_all = extract_xic_scan_major(
        peaks,
        _target(prec_mz=mz2[0], rt_center=102.0),
        acquisition_id=0,
        level=1,
        n_isotopes=1,
        charge_range=(1, 4),
        rt_window=2.0,
        tolerance=_TOL_PPM,
        drop_unmatched=False,
    )
    assert set(out_all.column("precursor_charge").to_pylist()) == {1, 2, 3, 4}
    # only z=2 actually matches the injected peaks
    assert set(out.column("precursor_charge").to_pylist()) == {2}


def test_ms1_no_peptide_bare_mz_target():
    rows = [_pk(0, 1, 10.0, 500.0, 1000.0), _pk(0, 1, 10.0, 123.0, 9.0)]
    out = extract_xic_scan_major(
        _peaks(rows),
        _target(modseq=None, charge=None, prec_mz=500.0),
        acquisition_id=0,
        level=1,
        n_isotopes=1,
    )
    assert out.num_rows == 1
    assert abs(out.column("mz_observed")[0].as_py() - 500.0) < 1e-9


def test_ms1_explicit_rt_bounds_override_window():
    peaks, mz2 = _ms1_dataset(signal_scans=(1, 2, 3))
    # rt_start/end select scans 1-2 (rt 101-102); rt_window would be ignored
    out = _charge2(
        extract_xic_scan_major(
            peaks,
            _target(prec_mz=mz2[0], rt_center=103.0, rt_start=100.5, rt_end=102.5),
            acquisition_id=0,
            level=1,
            n_isotopes=1,
            rt_window=0.0,
            tolerance=_TOL_PPM,
        )
    )
    assert set(out.column("scan").to_pylist()) == {1, 2}


def test_ms1_drop_unmatched_default_vs_keep():
    rows = [_pk(0, 1, 10.0, 999.0, 5.0)]  # nothing near the target m/z
    dropped = extract_xic_scan_major(
        _peaks(rows),
        _target(modseq=None, charge=None, prec_mz=500.0),
        acquisition_id=0,
        level=1,
        n_isotopes=1,
    )
    kept = extract_xic_scan_major(
        _peaks(rows),
        _target(modseq=None, charge=None, prec_mz=500.0),
        acquisition_id=0,
        level=1,
        n_isotopes=1,
        drop_unmatched=False,
    )
    assert dropped.num_rows == 0
    assert kept.num_rows == 1
    assert kept.column("intensity")[0].as_py() == 0.0


# ── MS2 ───────────────────────────────────────────────────────────────


def _ms2_dataset(covering_scans=(10, 11)):
    pf = parse_proforma(_PEP)
    P = precursor_mz(pf, charge=2)
    ladder = fragment_ladder_indices_batch(
        [pf], ion_types=(IonType.B, IonType.Y), max_fragment_charge=1
    )[0]
    frags = [(c, m) for c, m in ladder.items() if c[2] == 1][:4]
    rows = []
    for s, rt in zip(covering_scans, (50.0, 51.0)):
        for _c, m in frags:
            rows.append(
                _pk(s, 2, rt, m * (1 + _NOISE_PPM / 1e6), 500.0, P - 1.0, P + 1.0, P)
            )
    # a non-covering window scan carrying the same fragment m/z (must be excluded)
    rows.append(_pk(99, 2, 52.0, frags[0][1], 999.0, P + 50.0, P + 52.0, P + 51.0))
    return _peaks(rows), P, frags


def test_ms2_isolation_window_gating():
    peaks, P, frags = _ms2_dataset()
    out = extract_xic_scan_major(
        peaks,
        _target(charge=2, prec_mz=P, rt_center=50.5),
        acquisition_id=0,
        level=2,
        rt_window=5.0,
        max_fragment_charge=1,
        tolerance=_TOL_PPM,
    )
    assert out.num_rows == len(frags) * 2  # only the two covering scans
    assert set(out.column("scan").to_pylist()) == {10, 11}
    assert out.column("ion_type").null_count == 0
    assert out.column("isotope").null_count == out.num_rows


def test_ms2_null_isolation_bounds_pass_all():
    pf = parse_proforma(_PEP)
    P = precursor_mz(pf, charge=2)
    ladder = fragment_ladder_indices_batch([pf], max_fragment_charge=1)[0]
    c, m = next((c, m) for c, m in ladder.items() if c[2] == 1)
    rows = [_pk(1, 2, 50.0, m, 100.0, None, None, P)]  # null isolation bounds
    out = extract_xic_scan_major(
        _peaks(rows),
        _target(charge=2, prec_mz=P, rt_center=50.0),
        acquisition_id=0,
        level=2,
        rt_window=5.0,
        max_fragment_charge=1,
        tolerance=_TOL_PPM,
    )
    assert out.num_rows >= 1


def test_ms2_bare_mz_target_errors():
    peaks, P, _ = _ms2_dataset()
    with pytest.raises(ValueError):
        extract_xic_scan_major(
            _peaks([_pk(1, 2, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0)]),
            _target(modseq=None, charge=2, prec_mz=P),
            acquisition_id=0,
            level=2,
        )


def test_ms2_assigned_scans_only():
    peaks, P, frags = _ms2_dataset(covering_scans=(10, 11))
    out = extract_xic_scan_major(
        peaks,
        _target(charge=2, prec_mz=P, rt_center=50.5, scan=10),
        acquisition_id=0,
        level=2,
        rt_window=5.0,
        max_fragment_charge=1,
        tolerance=_TOL_PPM,
        assigned_scans_only=True,
    )
    assert set(out.column("scan").to_pylist()) == {10}


def test_mode_a_vs_mode_b_parity_binned_ms2_wide_window(tmp_path):
    """A binned (DDA) MS2 level where a wide-window covering scan lands 2+
    bins from the target's precursor bin must still be found by Mode B —
    the bin-neighbor span derives from the manifest's max isolation
    half-width, not a hardcoded ±1."""
    pf = parse_proforma(_PEP)
    P = precursor_mz(pf, charge=2)
    ladder = fragment_ladder_indices_batch([pf], max_fragment_charge=1)[0]
    frags = [(c, m) for c, m in ladder.items() if c[2] == 1][:4]
    rows = []
    # Many narrow windows centered far from P → force binning + none cover P.
    for i in range(6):
        c = 600.0 + i * 0.5
        rows.append(_pk(200 + i, 2, float(i), 123.0 + i, 10.0, c - 0.5, c + 0.5, c))
    # Covering scan: wide window [P-5, P+5] covers P, but its precursor m/z
    # (the bin value) is P-4 → bins ~2 cells away from floor(P / 2).
    for _c, m in frags:
        rows.append(
            _pk(
                300,
                2,
                50.0,
                m * (1 + _NOISE_PPM / 1e6),
                500.0,
                P - 5.0,
                P + 5.0,
                P - 4.0,
            )
        )
    peaks = _peaks(rows)
    tgt = _target(charge=2, prec_mz=P, rt_center=50.0)
    kw = dict(
        acquisition_id=0,
        level=2,
        rt_window=100.0,
        max_fragment_charge=1,
        tolerance=_TOL_PPM,
    )

    a = extract_xic_scan_major(peaks, tgt, **kw)
    src = tmp_path / "bundle"
    src.mkdir()
    pq.write_table(peaks, src / "peaks.parquet")
    man = build_peak_index(
        src, tmp_path / "idx", max_isolation_windows=2, dda_window_width_mz=2.0
    )
    assert man.window_scheme.get("2") == "binned"
    assert man.dda_max_halfwidth_mz.get("2") == 5.0
    b = extract_xic_indexed(tmp_path / "idx", tgt, **kw)

    assert a.num_rows == b.num_rows == len(frags)
    assert set(a.column("scan").to_pylist()) == {300}
    assert set(b.column("scan").to_pylist()) == {300}
    assert sorted(round(x, 6) for x in a.column("mz_observed").to_pylist()) == sorted(
        round(x, 6) for x in b.column("mz_observed").to_pylist()
    )


# ── all_in_window ─────────────────────────────────────────────────────


def test_all_in_window_emits_multiple_peaks():
    rows = [
        _pk(0, 1, 10.0, 500.000, 1000.0),
        _pk(0, 1, 10.0, 500.004, 500.0),
        _pk(0, 1, 10.0, 123.0, 9.0),
    ]
    tgt = _target(modseq=None, charge=None, prec_mz=500.0)
    nearest = extract_xic_scan_major(
        _peaks(rows), tgt, acquisition_id=0, level=1, n_isotopes=1
    )
    allwin = extract_xic_scan_major(
        _peaks(rows), tgt, acquisition_id=0, level=1, n_isotopes=1, all_in_window=True
    )
    assert nearest.num_rows == 1
    assert allwin.num_rows == 2


# ── Mode A vs Mode B parity ───────────────────────────────────────────


def _sortkey(t):
    d = t.sort_by(
        [
            ("precursor_charge", "ascending"),
            ("scan", "ascending"),
            ("isotope", "ascending"),
        ]
    )
    return list(
        zip(
            d["scan"].to_pylist(),
            d["isotope"].to_pylist(),
            [round(x, 6) for x in d["intensity"].to_pylist()],
            [round(x, 6) for x in d["mz_observed"].to_pylist()],
        )
    )


def test_mode_a_vs_mode_b_parity_ms1(tmp_path):
    peaks, mz2 = _ms1_dataset()
    tgt = _target(prec_mz=mz2[0], rt_center=102.0)
    a = _charge2(
        extract_xic_scan_major(
            peaks,
            tgt,
            acquisition_id=0,
            level=1,
            n_isotopes=3,
            rt_window=2.0,
            tolerance=_TOL_PPM,
        )
    )
    src = tmp_path / "bundle"
    src.mkdir()
    pq.write_table(peaks, src / "peaks.parquet")
    build_peak_index(src, tmp_path / "idx")
    b = _charge2(
        extract_xic_indexed(
            tmp_path / "idx",
            tgt,
            acquisition_id=0,
            level=1,
            n_isotopes=3,
            rt_window=2.0,
            tolerance=_TOL_PPM,
        )
    )
    assert a.num_rows == b.num_rows == 9
    assert _sortkey(a) == _sortkey(b)


def test_mode_a_vs_mode_b_parity_drop_unmatched_false(tmp_path):
    """Mode B must honour drop_unmatched=False: the rectangular grid of
    unmatched (charge × isotope × scan) zero rows that Mode A emits."""
    peaks, mz2 = _ms1_dataset(signal_scans=(1, 2, 3))
    tgt = _target(prec_mz=mz2[0], rt_center=102.0)
    kw = dict(
        acquisition_id=0,
        level=1,
        n_isotopes=3,
        charge_range=(1, 4),
        rt_window=2.0,
        tolerance=_TOL_PPM,
        drop_unmatched=False,
    )
    a = extract_xic_scan_major(peaks, tgt, **kw)
    src = tmp_path / "bundle"
    src.mkdir()
    pq.write_table(peaks, src / "peaks.parquet")
    build_peak_index(src, tmp_path / "idx")
    b = extract_xic_indexed(tmp_path / "idx", tgt, **kw)
    # rt window [100,104] gates all 5 scans (each carries a noise peak):
    # 4 charges × 3 isotopes × 5 candidate scans = 60 rectangular cells.
    assert a.num_rows == b.num_rows == 60

    def _full_key(t):
        d = t.sort_by(
            [
                ("precursor_charge", "ascending"),
                ("isotope", "ascending"),
                ("scan", "ascending"),
            ]
        )
        obs = d["mz_observed"].to_pylist()
        return list(
            zip(
                d["precursor_charge"].to_pylist(),
                d["isotope"].to_pylist(),
                d["scan"].to_pylist(),
                [round(x, 6) for x in d["intensity"].to_pylist()],
                [None if o is None or o != o else round(o, 6) for o in obs],
            )
        )

    assert _full_key(a) == _full_key(b)
    # 9 matched cells (charge 2 × 3 isotopes × 3 signal scans); the rest zero.
    assert pc.sum(pc.equal(a["intensity"], 0.0)).as_py() == 51
    assert pc.sum(pc.equal(b["intensity"], 0.0)).as_py() == 51


def test_use_index_exact_error_restores_f64(tmp_path):
    peaks, mz2 = _ms1_dataset()
    tgt = _target(prec_mz=mz2[0], rt_center=102.0)
    src = tmp_path / "bundle"
    src.mkdir()
    pq.write_table(peaks, src / "peaks.parquet")
    build_peak_index(src, tmp_path / "idx32", downcast=["mz"])
    default = _charge2(
        extract_xic_indexed(
            tmp_path / "idx32",
            tgt,
            acquisition_id=0,
            level=1,
            n_isotopes=3,
            rt_window=2.0,
            tolerance=_TOL_PPM,
        )
    )
    exact = _charge2(
        extract_xic_indexed(
            tmp_path / "idx32",
            tgt,
            acquisition_id=0,
            level=1,
            n_isotopes=3,
            rt_window=2.0,
            tolerance=_TOL_PPM,
            exact_error=True,
            peaks_dir=src,
        )
    )
    a = _charge2(
        extract_xic_scan_major(
            peaks,
            tgt,
            acquisition_id=0,
            level=1,
            n_isotopes=3,
            rt_window=2.0,
            tolerance=_TOL_PPM,
        )
    )
    # f32-index reported mz differs from the f64 truth; exact_error restores it
    f32_obs = sorted(default.column("mz_observed").to_pylist())
    exact_obs = sorted(exact.column("mz_observed").to_pylist())
    truth = sorted(a.column("mz_observed").to_pylist())
    assert exact_obs == pytest.approx(truth, abs=1e-9)
    assert f32_obs != pytest.approx(truth, abs=1e-9)


def test_exact_error_without_peaks_dir_raises_on_f32_index(tmp_path):
    peaks, mz2 = _ms1_dataset()
    src = tmp_path / "bundle"
    src.mkdir()
    pq.write_table(peaks, src / "peaks.parquet")
    build_peak_index(src, tmp_path / "idx32", downcast=["mz"])
    with pytest.raises(ValueError, match="peaks_dir"):
        extract_xic_indexed(
            tmp_path / "idx32",
            _target(prec_mz=mz2[0], rt_center=102.0),
            acquisition_id=0,
            level=1,
            n_isotopes=3,
            rt_window=2.0,
            tolerance=_TOL_PPM,
            exact_error=True,
            peaks_dir=None,
        )


# ── integration-readiness (Phase 8 guard) ─────────────────────────────


def test_trace_is_reduction_ready():
    peaks, mz2 = _ms1_dataset()
    out = _charge2(
        extract_xic_scan_major(
            peaks,
            _target(prec_mz=mz2[0], rt_center=102.0),
            acquisition_id=0,
            level=1,
            n_isotopes=3,
            rt_window=2.0,
            tolerance=_TOL_PPM,
        )
    )
    # group by (target_id, precursor_charge) → a per-precursor trace whose
    # scans are RT-ordered and retain the isotope axis (stage 1-2 substrate).
    d = out.sort_by([("scan", "ascending")])
    rts = d.column("rt").to_pylist()
    assert rts == sorted(rts)
    assert d.column("isotope").null_count == 0
    assert {"target_id", "precursor_charge", "rt", "intensity", "isotope"} <= set(
        out.column_names
    )


# ── I/O ───────────────────────────────────────────────────────────────


def test_save_load_xic_roundtrip(tmp_path):
    peaks, mz2 = _ms1_dataset()
    out = extract_xic_scan_major(
        peaks,
        _target(prec_mz=mz2[0], rt_center=102.0),
        acquisition_id=0,
        level=1,
        n_isotopes=3,
        rt_window=2.0,
        tolerance=_TOL_PPM,
    )
    save_xic(out, tmp_path / "xic", metadata={"level": 1})
    back = load_xic(tmp_path / "xic")
    assert back.schema.equals(XIC_TRACE_TABLE)
    assert back.num_rows == out.num_rows
    assert (
        back.column("mz_error_da").to_pylist() == out.column("mz_error_da").to_pylist()
    )
