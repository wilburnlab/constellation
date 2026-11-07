"""Compiling extraction targets into a sweep's queries (`massspec.sweep.queries`).

The channels must be the m/z values the scan-major extractor queried, value
for value and in the same order, so they are checked against the frozen copy of its
query builders in `tests/_xic_scan_major_oracle.py`.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import _xic_scan_major_oracle as oracle
from constellation.core.chem.elements import ELEMENTS
from constellation.core.sequence.proforma import parse_proforma
from constellation.core.sequence.protein import peptide_composition
from constellation.core.stats.units import ISOTOPE_MASS_DIFF
from constellation.massspec.peptide.ions import IonType
from constellation.massspec.peptide.mz import precursor_mz
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.sweep import FRAGMENTS, ISOTOPES, ChannelSettings, QueryPlan, compile_queries

_SEQS = ["PEPTIDEK", "LVNELTEFAK", "PEPTM[Oxidation]IDEK", "<13C><15N>ACDEFGHIK",
         "HLVDEPQNLIK", "PEPTIDEK"]  # fmt: skip


def _targets(rows) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE)


def _peptide_targets(**over):
    rows = []
    for i, s in enumerate(_SEQS):
        z = 2 + (i % 2)
        rows.append(dict(target_id=10 + i, modified_sequence=s, precursor_charge=z,
                         precursor_mz=precursor_mz(parse_proforma(s), charge=z), **over))  # fmt: skip
    return _targets(rows)


def _bits(x):
    return np.asarray(x, dtype=np.float64).view(np.int64)


def _isotope_rows(plan, q):
    iso = plan.isotopes
    a, b = iso.offsets[q], iso.offsets[q + 1]
    return list(zip(_bits(iso.mz[a:b]).tolist(), iso.precursor_charge[a:b].tolist(),
                    iso.isotope[a:b].tolist()))  # fmt: skip


@pytest.mark.parametrize("charge_range", [(1, 4), None, (2, 3)])
@pytest.mark.parametrize("n_isotopes", [1, 3, 5])
def test_isotopes_match_the_extractor(charge_range, n_isotopes):
    targets = _peptide_targets()
    plan = compile_queries(
        targets, levels=[1], channels=ChannelSettings(n_isotopes=n_isotopes, charge_range=charge_range)
    )
    for q, rec in enumerate(oracle._targets_to_records(targets)):
        want = oracle._ms1_queries(rec, n_isotopes=n_isotopes, charge_range=charge_range)
        got = _isotope_rows(plan, q)
        assert got == [(_bits([w.mz])[0], w.precursor_charge, w.isotope) for w in want]


@pytest.mark.parametrize("charge", [None, 3])
def test_bare_mz_isotopes_match_the_extractor(charge):
    targets = _targets([dict(target_id=1, precursor_mz=612.3456789, precursor_charge=charge)])
    plan = compile_queries(targets, levels=[1])
    want = oracle._ms1_queries(oracle._targets_to_records(targets)[0], n_isotopes=3,
                               charge_range=(1, 4))  # fmt: skip
    assert _isotope_rows(plan, 0) == [(_bits([w.mz])[0], w.precursor_charge, w.isotope) for w in want]


@pytest.mark.parametrize(
    "settings",
    [
        ChannelSettings(),
        ChannelSettings(ion_types=(IonType.A, IonType.B, IonType.Y), max_fragment_charge=1),
        ChannelSettings(max_fragment_charge=3, neutral_losses=("H2O", "NH3")),
    ],
)
def test_fragments_match_the_extractor(settings):
    targets = _peptide_targets()
    plan = compile_queries(targets, levels=[2], channels=settings)
    want = oracle._ms2_queries_batch(
        oracle._targets_to_records(targets), ion_types=settings.ion_types,
        max_fragment_charge=settings.max_fragment_charge, neutral_losses=settings.neutral_losses,
    )  # fmt: skip
    f = plan.fragments
    names = [None, *f.loss_ids]
    for q in range(targets.num_rows):
        a, b = f.offsets[q], f.offsets[q + 1]
        got = list(zip(_bits(f.mz[a:b]).tolist(), f.precursor_charge[a:b].tolist(),
                       f.ion_type[a:b].tolist(), f.position[a:b].tolist(),
                       f.fragment_charge[a:b].tolist(), [names[k] for k in f.loss[a:b]]))  # fmt: skip
        assert got == [
            (_bits([w.mz])[0], w.precursor_charge, w.ion_type, w.position, w.fragment_charge,
             w.loss_id)
            for w in want[q]
        ]  # fmt: skip


def test_evidence_keys_count_fragment_ions_not_charges_or_losses():
    plan = compile_queries(_peptide_targets(), levels=[2], channels=ChannelSettings(max_fragment_charge=2,
                           neutral_losses=("H2O",)))  # fmt: skip
    f = plan.fragments
    for q in range(plan.n_queries):
        a, b = f.offsets[q], f.offsets[q + 1]
        ions = list(zip(f.ion_type[a:b].tolist(), f.position[a:b].tolist()))
        keys = f.evidence_key[a:b].tolist()
        assert sorted(set(keys)) == list(range(len(set(ions))))
        assert {k: ion for k, ion in zip(keys, ions)} == dict(zip(keys, ions))  # one fragment ion per key
        assert len({ion: k for k, ion in zip(keys, ions)}) == len(set(keys))


def test_errors_are_the_extractors():
    with pytest.raises(ValueError, match="requires modified_sequence"):
        compile_queries(_targets([dict(target_id=1, precursor_mz=500.0)]), levels=[2])
    with pytest.raises(ValueError, match="requires precursor_charge"):
        compile_queries(_targets([dict(target_id=1, modified_sequence="PEPTIDEK")]), levels=[2])
    # A sequence without a charge is looked for in MS1 at the fallback
    # charges (it was an error under charge_range=None until 2026-11-07).
    plan = compile_queries(_targets([dict(target_id=1, modified_sequence="PEPTIDEK")]), levels=[1])
    assert sorted(set(plan.isotopes.precursor_charge.tolist())) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="needs modified_sequence or precursor_mz"):
        compile_queries(_targets([dict(target_id=1)]), levels=[1])
    with pytest.raises(ValueError, match="linear peptidoforms"):
        compile_queries(_targets([dict(target_id=1, modified_sequence="PEPTIDEK//EMEVK",
                                       precursor_charge=2)]), levels=[2])  # fmt: skip


def test_with_ms1_a_target_without_fragments_is_not_an_error():
    targets = _targets([
        dict(target_id=1, precursor_mz=500.0, precursor_charge=2),
        dict(target_id=2, modified_sequence="PEPTIDEK"),
        dict(target_id=3, modified_sequence="LVNELTEFAK", precursor_charge=2),
    ])  # fmt: skip
    plan = compile_queries(targets, levels=[1, 2])
    assert plan.fragments.counts().tolist()[:2] == [0, 0]
    assert plan.fragments.counts()[2] > 0
    assert (plan.isotopes.counts() > 0).all()


_RT = [
    dict(),
    dict(rt_center=100.0),
    dict(rt_start=90.0),
    dict(rt_start=90.0, rt_end=110.0),
    dict(rt_start=90.0, rt_end=110.0, rt_center=500.0),
    dict(rt_end=110.0, rt_center=100.0),
    dict(rt_center=float("nan")),
]


@pytest.mark.parametrize("rt_window", [None, 7.5])
def test_rt_bounds_match_the_extractor(rt_window):
    targets = _targets([dict(target_id=i, precursor_mz=500.0, **r) for i, r in enumerate(_RT)])
    plan = compile_queries(targets, levels=[1], rt_window=rt_window)
    for q, rec in enumerate(oracle._targets_to_records(targets)):
        lo, hi = oracle._rt_bounds(rec, rt_window)
        # The extractor bisected on these, so a NaN bound let every scan through.
        lo = -np.inf if np.isnan(lo) else lo
        hi = np.inf if np.isnan(hi) else hi
        assert _bits([plan.rt_lo[q], plan.rt_hi[q]]).tolist() == _bits([lo, hi]).tolist()


def test_precursor_mz_is_derived_when_missing():
    targets = _targets([
        dict(target_id=1, modified_sequence="PEPTIDEK", precursor_charge=2),
        dict(target_id=2, modified_sequence="PEPTIDEK", precursor_charge=2, precursor_mz=123.0),
        dict(target_id=3, precursor_mz=456.0),
    ])  # fmt: skip
    plan = compile_queries(targets, levels=[1, 2])
    assert plan.precursor_mz[0] == precursor_mz(parse_proforma("PEPTIDEK"), charge=2)
    assert plan.precursor_mz_derived.tolist() == [True, False, False]
    assert plan.precursor_mz[1:].tolist() == [123.0, 456.0]
    kept = compile_queries(targets, levels=[2, 1], derive_precursor_mz=False)
    assert np.isnan(kept.precursor_mz[0]) and not kept.precursor_mz_derived.any()


def test_assigned_scan_only_when_asked():
    targets = _targets([dict(target_id=1, precursor_mz=500.0, scan=42), dict(target_id=2, precursor_mz=1.0)])
    assert compile_queries(targets, levels=[1]).assigned_scan.tolist() == [-1, -1]
    on = compile_queries(targets, levels=[1], assigned_scans_only=True)
    assert on.assigned_scan.tolist() == [42, -1]


def test_heavy_labels_shift_isotopes_and_fragments():
    light, heavy = "PEPTIDEK", "<13C><15N>PEPTIDEK"
    targets = _targets([dict(target_id=i, modified_sequence=s, precursor_charge=2)
                        for i, s in enumerate((light, heavy))])  # fmt: skip
    plan = compile_queries(targets, levels=[1, 2], channels=ChannelSettings(charge_range=None))
    atoms = peptide_composition(parse_proforma(light)).atoms
    shift = atoms["C"] * (ELEMENTS["C"].isotope_mass(13) - ELEMENTS["C"].monoisotopic_mass) + atoms[
        "N"
    ] * (ELEMENTS["N"].isotope_mass(15) - ELEMENTS["N"].monoisotopic_mass)
    iso = plan.isotopes
    assert iso.mz[iso.offsets[1]] - iso.mz[iso.offsets[0]] == pytest.approx(shift / 2, abs=1e-6)
    frag = plan.fragments
    n = frag.counts()
    assert n[0] == n[1]
    assert (frag.mz[frag.offsets[1]:] > frag.mz[: frag.offsets[1]]).all()


def test_negative_isotope_offsets_for_labelled_peptides():
    targets = _targets([dict(target_id=1, modified_sequence="<13C><15N>PEPTIDEK", precursor_charge=2)])
    plan = compile_queries(
        targets, levels=[1], channels=ChannelSettings(isotope_offsets=(-2, -1, 0, 1), charge_range=None)
    )
    iso = plan.isotopes
    assert iso.isotope.tolist() == [-2, -1, 0, 1]
    m0 = iso.mz[2]
    assert iso.mz[1] == pytest.approx(m0 - ISOTOPE_MASS_DIFF / 2)
    assert iso.mz[0] == pytest.approx(m0 - 2 * ISOTOPE_MASS_DIFF / 2)
    assert np.isnan(iso.abundance[:2]).all() and np.isfinite(iso.abundance[2:]).all()


def test_plan_round_trips_through_its_tables():
    settings = ChannelSettings(max_fragment_charge=3, neutral_losses=("H2O",))
    targets = _peptide_targets(rt_center=100.0, scan=7)
    plan = compile_queries(targets, levels=[1, 2], channels=settings, rt_window=5.0, assigned_scans_only=True)
    queries, channels = plan.to_tables()
    back = QueryPlan.from_tables(queries, channels)
    assert back.levels == plan.levels
    for name in ("precursor_charge", "precursor_mz", "precursor_mz_derived", "rt_lo", "rt_hi",
                 "assigned_scan"):  # fmt: skip
        np.testing.assert_array_equal(getattr(back, name), getattr(plan, name))
    for kind, a, b in ((ISOTOPES, plan.isotopes, back.isotopes), (FRAGMENTS, plan.fragments, back.fragments)):
        assert a.kind == b.kind == kind
        for name in ("offsets", "mz", "precursor_charge", "ion_type", "position",
                     "fragment_charge", "loss", "abundance", "evidence_key"):  # fmt: skip
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        assert a.loss_ids == b.loss_ids
    assert back.targets.column("target_id").equals(plan.targets.column("target_id"))


def test_a_written_plan_keeps_its_targets_and_settings(tmp_path):
    """A plan on disk (what workers and a spilled sweep read) dropped the
    target table's other columns (rt_center, scan, ...) and reverted the
    channel settings to their defaults (review of 2026-11-06)."""
    settings = ChannelSettings(n_isotopes=4, charge_range=(2, 3), max_fragment_charge=3,
                               ion_types=(IonType.A, IonType.B, IonType.Y), neutral_losses=("H2O",))  # fmt: skip
    targets = _peptide_targets(rt_center=100.0, rt_start=None, scan=7, rt_library=55.0, rt_scale="observed")
    plan = compile_queries(targets, levels=[1, 2], channels=settings, rt_window=5.0, assigned_scans_only=True)
    plan.write(tmp_path / "plan")
    back = QueryPlan.read(tmp_path / "plan")
    assert back.targets.equals(plan.targets)
    assert back.settings == plan.settings == settings
    assert ChannelSettings.from_dict(settings.to_dict()) == settings


def test_ms1_is_at_each_targets_own_charge_unless_a_range_is_asked():
    """The default charge_range (1, 4) replaced a target's own charge: a 5+
    precursor had no MS1 at 5+, and a peptide's 2+ and 3+ precursors got
    identical channels, each claiming the other's matches."""
    rows = [dict(target_id=1, modified_sequence="PEPTIDEK", precursor_charge=5),
            dict(target_id=2, modified_sequence="LVNELTEFAK", precursor_charge=2),
            dict(target_id=3, modified_sequence="LVNELTEFAK", precursor_charge=3)]  # fmt: skip
    plan = compile_queries(_targets(rows), levels=[1])
    iso = plan.isotopes
    charges = [sorted(set(iso.precursor_charge[iso.offsets[q]:iso.offsets[q + 1]].tolist())) for q in range(3)]
    assert charges == [[5], [2], [3]]
    assert not np.array_equal(iso.mz[iso.offsets[1]:iso.offsets[2]], iso.mz[iso.offsets[2]:iso.offsets[3]])
    ranged = compile_queries(_targets(rows), levels=[1], channels=ChannelSettings(charge_range=(1, 4))).isotopes
    q2 = ranged.mz[ranged.offsets[1]:ranged.offsets[2]]
    q3 = ranged.mz[ranged.offsets[2]:ranged.offsets[3]]
    assert np.array_equal(q2, q3)  # asked for: the analyte's evidence across charges


@pytest.mark.parametrize("bad", [dict(isotope_offsets=()), dict(n_isotopes=0), dict(charge_range=(3, 2)),
                                 dict(max_fragment_charge=0), dict(ion_types=())])  # fmt: skip
def test_impossible_channel_settings_are_refused(bad):
    """An empty isotope selection crashed in max() of nothing."""
    with pytest.raises(ValueError, match="isotope selection is empty|must be|must name"):
        ChannelSettings(**bad)


def test_an_rt_window_is_refused_around_a_model_clock():
    """A library's time on a model's scale is not a run's seconds; a window
    around it gated scans in the wrong place (review of 2026-11-06)."""
    rows = [dict(target_id=1, modified_sequence="PEPTIDEK", precursor_charge=2, rt_library=1200.0,
                 rt_scale="chronologer_hi_seconds")]  # fmt: skip
    with pytest.raises(ValueError, match="chronologer_hi_seconds, not this run's seconds"):
        compile_queries(_targets(rows), levels=[2], rt_window=60.0)
    assert np.isinf(compile_queries(_targets(rows), levels=[2]).rt_lo).all()  # no window: the whole run
    rows[0].update(rt_start=100.0, rt_end=200.0)  # explicit bounds are the caller's to give
    assert compile_queries(_targets(rows), levels=[2], rt_window=60.0).rt_lo.tolist() == [100.0]
    observed = [dict(target_id=2, modified_sequence="PEPTIDEK", precursor_charge=2, rt_center=900.0,
                     rt_library=900.0, rt_scale="observed")]  # fmt: skip
    assert compile_queries(_targets(observed), levels=[2], rt_window=60.0).rt_lo.tolist() == [840.0]


# ── one generator, one abundance convention (2026-11-07) ──────────────


def test_isotope_abundance_is_the_share_of_the_whole_envelope():
    """An isotope's abundance was its share of the peaks kept, so it changed
    with n_isotopes; it is now its share of the whole envelope."""
    targets = _peptide_targets()
    three = compile_queries(targets, levels=[1], channels=ChannelSettings(n_isotopes=3)).isotopes
    five = compile_queries(targets, levels=[1], channels=ChannelSettings(n_isotopes=5)).isotopes
    for q in range(targets.num_rows):
        a = three.abundance[three.offsets[q]:three.offsets[q + 1]]
        b = five.abundance[five.offsets[q]:five.offsets[q + 1]]
        assert np.allclose(a, b[:3], rtol=1e-6) and b.sum() <= 1 + 1e-6 and a.sum() < b.sum()


def test_the_top_isotopes_of_a_heavy_peptide_can_leave_out_m0():
    long = "HLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTPTLVEVSR"
    targets = _targets([dict(target_id=1, modified_sequence=long, precursor_charge=4),
                        dict(target_id=2, modified_sequence="PEPTIDEK", precursor_charge=2)])  # fmt: skip
    iso = compile_queries(targets, levels=[1], channels=ChannelSettings(top_isotopes=2)).isotopes
    heavy = iso.isotope[iso.offsets[0]:iso.offsets[1]].tolist()
    light = iso.isotope[iso.offsets[1]:iso.offsets[2]].tolist()
    assert 0 not in heavy and len(heavy) == 2 and heavy == sorted(heavy)
    assert light == [0, 1]
    full = compile_queries(targets, levels=[1], channels=ChannelSettings(n_isotopes=8)).isotopes
    ab = full.abundance[full.offsets[0]:full.offsets[1]]
    assert sorted(np.argsort(-ab)[:2].tolist()) == heavy


def test_top_fragments_needs_abundances_that_rank_ions():
    from constellation.massspec.peptide.channels import UNMODELLED, Uniform

    for source in (UNMODELLED, Uniform()):
        with pytest.raises(ValueError, match="top_fragments needs abundances that rank"):
            compile_queries(_peptide_targets(), levels=[2], channels=ChannelSettings(top_fragments=10),
                            fragment_abundance=source)  # fmt: skip


def test_uniform_fragment_abundance_is_the_classic_equal_series():
    """Equal intensity for every ion of the series asked, as the pre-Prosit
    DDA scores assume: each query's abundances sum to 1; weights tilt it."""
    from constellation.massspec.peptide.channels import Uniform

    settings = ChannelSettings(ion_types=(IonType.A, IonType.B, IonType.Y), neutral_losses=("H2O",))
    plain = compile_queries(_peptide_targets(), levels=[2], channels=settings, fragment_abundance=Uniform())
    f = plain.fragments
    for q in range(plain.n_queries):
        a = f.abundance[f.offsets[q]:f.offsets[q + 1]]
        assert np.isclose(a.sum(), 1.0) and np.allclose(a, a[0])
    tilted = compile_queries(_peptide_targets(), levels=[2], channels=settings,
                             fragment_abundance=Uniform(ion_type_weights={IonType.A: 0.1}, loss_weight=0.2)).fragments  # fmt: skip
    a = tilted.abundance
    plain_ion = (tilted.ion_type == IonType.B) & (tilted.loss == 0)
    a_ions = (tilted.ion_type == IonType.A) & (tilted.loss == 0)
    for q in range(tilted.offsets.size - 1):
        sl = slice(tilted.offsets[q], tilted.offsets[q + 1])
        assert np.isclose(a[sl].sum(), 1.0)
        assert np.allclose(a[sl][a_ions[sl]] / a[sl][plain_ion[sl]].max(), 0.1, rtol=1e-5)
    assert (a[a_ions].max() < a[plain_ion].min())
    assert (a[tilted.loss > 0].max() < a[plain_ion].min())
    assert plain.fragment_abundance["source"] == "uniform"


def test_compiling_many_targets_gives_each_its_own_channels():
    """Vectorized over targets: compiling them together gives each target
    the channels compiling it alone gives (no per-target loop to lean on)."""
    targets = _peptide_targets()
    together = compile_queries(targets, levels=[1, 2])
    for q in range(targets.num_rows):
        alone = compile_queries(targets.slice(q, 1), levels=[1, 2])
        for kind in ("isotopes", "fragments"):
            a, b = getattr(together, kind), getattr(alone, kind)
            sl = slice(a.offsets[q], a.offsets[q + 1])
            for name in ("mz", "precursor_charge", "isotope", "ion_type", "position", "fragment_charge", "loss",
                         "evidence_key"):  # fmt: skip
                np.testing.assert_array_equal(getattr(a, name)[sl], getattr(b, name), err_msg=f"{kind}.{name}")
            np.testing.assert_allclose(a.abundance[sl], b.abundance, rtol=1e-6)
        assert together.precursor_mz[q] == alone.precursor_mz[0]
