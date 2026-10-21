"""Annotating an observed spectrum with a peptide's ions (massspec.spectra.annotate)."""

from __future__ import annotations

import numpy as np
import pytest

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.peptide.mz import precursor_mz
from constellation.massspec.spectra.annotate import (
    annotate_spectrum,
    default_fragment_charge,
    ion_ladder,
    precursor_isotopes,
    theoretical_fragment_targets,
)


def test_the_ladder_is_numbered_as_labels_are() -> None:
    ladder = ion_ladder("PEPTIDEK")
    rows = {r["annotation"]: r for r in ladder.to_pylist()}
    # y1 = K + H2O + H+, b2 = PE + H+; bond 0 is b1 and y7.
    assert rows["y1"]["mz"] == pytest.approx(147.112804, abs=1e-5)
    assert rows["b2"]["mz"] == pytest.approx(227.102633, abs=1e-5)
    assert rows["b1"]["position"] == rows["y7"]["position"] == 0
    assert len(ladder) == 14  # b1..b7 and y1..y7 at charge 1


def test_peaks_at_ions_are_labelled_and_the_rest_left_alone() -> None:
    ladder = ion_ladder("PEPTIDEK", max_fragment_charge=2)
    picks = {r["annotation"]: r["mz"] for r in ladder.to_pylist() if r["annotation"] in ("y3", "b4", "y5^2")}
    noise = [173.0, 501.5, 999.9]
    shift = 5e-6  # 5 ppm high
    mz = np.array(sorted([m * (1 + shift) for m in picks.values()] + noise))
    intensity = np.arange(1.0, mz.size + 1)

    table = annotate_spectrum(mz, intensity, "PEPTIDEK", charge=3, tolerance=10, tolerance_unit="ppm")
    assert table.column("mz").to_pylist() == mz.tolist()  # one row per peak, as given
    labelled = {r["annotation"]: r for r in table.to_pylist() if r["annotation"]}
    assert set(labelled) == {"y3", "b4", "y5^2"}
    assert labelled["y5^2"]["ion_type"] == "y"
    assert labelled["y5^2"]["ion_number"] == 5
    assert labelled["y5^2"]["fragment_charge"] == 2
    assert labelled["y3"]["error_ppm"] == pytest.approx(5.0, abs=0.01)
    assert labelled["y3"]["error_da"] > 0  # observed minus theoretical
    assert sum(r["annotation"] is None for r in table.to_pylist()) == len(noise)


def test_the_intact_precursor_is_labelled() -> None:
    target = precursor_mz(parse_proforma("PEPTIDEK"), charge=2)
    table = annotate_spectrum([target + 0.1, 600.0], [5.0, 1.0], "PEPTIDEK", charge=2,
                              tolerance=0.35, tolerance_unit="Da")  # fmt: skip
    first = table.to_pylist()[0]
    assert first["annotation"] == "p^2" and first["ion_type"] == "p" and first["ion_number"] is None
    assert first["error_da"] == pytest.approx(0.1, abs=1e-9)


def test_a_modified_peptide_shifts_its_ions() -> None:
    plain = {r["annotation"]: r["mz"] for r in ion_ladder("PEPTMIDEK").to_pylist()}
    oxidised = {r["annotation"]: r["mz"] for r in ion_ladder("PEPTM[Oxidation]IDEK").to_pylist()}
    assert oxidised["b4"] == pytest.approx(plain["b4"])
    assert oxidised["b5"] - plain["b5"] == pytest.approx(15.994915, abs=1e-5)


def test_empty_spectra_and_bad_input() -> None:
    assert annotate_spectrum([], [], "PEPTIDEK", charge=2, tolerance=10, tolerance_unit="ppm").num_rows == 0
    with pytest.raises(ValueError, match="differ in length"):
        annotate_spectrum([1.0], [], "PEPTIDEK", charge=2, tolerance=10, tolerance_unit="ppm")
    with pytest.raises(ValueError, match="unknown ion type"):
        ion_ladder("PEPTIDEK", ion_types=("q",))


@pytest.mark.parametrize(("charge", "expected"), [(1, 1), (2, 1), (3, 2), (4, 3), (6, 3)])
def test_default_fragment_charge(charge, expected) -> None:
    assert default_fragment_charge(charge) == expected


def test_precursor_isotopes_step_by_a_neutron_over_the_charge() -> None:
    iso = precursor_isotopes("PEPTIDEK", charge=2, n_isotopes=3)
    mz = iso.column("mz").to_pylist()
    assert iso.column("label").to_pylist() == ["M", "M+1", "M+2"]
    assert mz[0] == pytest.approx(precursor_mz(parse_proforma("PEPTIDEK"), charge=2), abs=1e-4)
    assert mz[1] - mz[0] == pytest.approx(1.00335 / 2, abs=2e-3)
    assert max(iso.column("abundance").to_pylist()) == 1.0


def test_theoretical_fragment_targets_prefer_long_y_ions() -> None:
    targets = theoretical_fragment_targets("PEPTIDEKAAR", top_n=4)
    assert targets.column("annotation").to_pylist() == ["y10", "y9", "y8", "y7"]
    narrow = theoretical_fragment_targets("PEPTIDEKAAR", top_n=20, mz_range=(300.0, 600.0))
    assert all(300.0 <= m <= 600.0 for m in narrow.column("mz").to_pylist())
    assert {r["ion_type"] for r in narrow.to_pylist()} <= {"b", "y"}
