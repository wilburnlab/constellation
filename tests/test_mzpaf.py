"""Tests for constellation.massspec.annotation.mzpaf — mzPAF v1.0 parser/formatter."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.massspec.annotation import (
    MzPAFSyntaxError,
    PeakAnnotation,
    format_mzpaf,
    fragment_row_to_mzpaf,
    fragment_table_to_mzpaf,
    parse_mzpaf,
)
from constellation.massspec.annotation._grammar import IonClass, MassErrorUnit


# ──────────────────────────────────────────────────────────────────────
# Single-annotation parsing
# ──────────────────────────────────────────────────────────────────────


def test_basic_peptide_ion():
    p = parse_mzpaf("y3")
    assert len(p.annotations) == 1
    a = p.annotations[0]
    assert a.ion_class is IonClass.PEPTIDE
    assert a.ion_letter == "y"
    assert a.position == 3
    assert a.charge is None
    assert a.losses == ()


def test_b_ion_with_charge():
    a = parse_mzpaf("b5^2").annotations[0]
    assert a.ion_letter == "b"
    assert a.position == 5
    assert a.charge == 2


def test_y_ion_with_water_loss():
    a = parse_mzpaf("y3-H2O").annotations[0]
    assert a.ion_letter == "y"
    assert len(a.losses) == 1
    assert a.losses[0].sign == -1
    assert a.losses[0].token == "H2O"
    assert a.losses[0].is_named is False


def test_y_ion_with_named_loss_phospho():
    a = parse_mzpaf("y5-[Phospho]").annotations[0]
    assert a.losses[0].is_named is True
    assert a.losses[0].token == "Phospho"


def test_chained_losses():
    """+CO-H2O is a real mzPAF v1.0 example (x→y conversion)."""
    a = parse_mzpaf("y3+CO-H2O").annotations[0]
    assert len(a.losses) == 2
    assert a.losses[0].sign == 1 and a.losses[0].token == "CO"
    assert a.losses[1].sign == -1 and a.losses[1].token == "H2O"


def test_mass_error_ppm():
    a = parse_mzpaf("b2-H2O/3.2ppm").annotations[0]
    assert a.mass_error is not None
    assert a.mass_error.value == pytest.approx(3.2)
    assert a.mass_error.unit is MassErrorUnit.PPM


def test_mass_error_da():
    a = parse_mzpaf("y1/-1.4").annotations[0]
    assert a.mass_error is not None
    assert a.mass_error.value == pytest.approx(-1.4)
    assert a.mass_error.unit is MassErrorUnit.DA


def test_isotope_offset():
    a = parse_mzpaf("y4+2i").annotations[0]
    assert a.isotope == 2


def test_isotope_negative():
    a = parse_mzpaf("y4-1i").annotations[0]
    assert a.isotope == -1


def test_adduct():
    a = parse_mzpaf("y4[M+H]").annotations[0]
    assert a.adduct == "M+H"


def test_confidence():
    a = parse_mzpaf("b2*0.75").annotations[0]
    assert a.confidence == pytest.approx(0.75)


def test_unknown_peak():
    a = parse_mzpaf("?").annotations[0]
    assert a.ion_class is IonClass.UNKNOWN
    assert a.ion_letter is None
    assert a.position is None


def test_full_decoration_chain():
    """All decorations in canonical order."""
    a = parse_mzpaf("y3-H2O+1i[M+H]^2/3.2ppm*0.85").annotations[0]
    assert a.ion_letter == "y"
    assert a.position == 3
    assert len(a.losses) == 1 and a.losses[0].token == "H2O"
    assert a.isotope == 1
    assert a.adduct == "M+H"
    assert a.charge == 2
    assert a.mass_error is not None
    assert a.confidence == pytest.approx(0.85)


# ──────────────────────────────────────────────────────────────────────
# Alternates and multi-analyte
# ──────────────────────────────────────────────────────────────────────


def test_alternates_comma_separated():
    p = parse_mzpaf("y3/-0.001,b6-H2O*0.5")
    assert len(p.annotations) == 2
    assert p.annotations[0].ion_letter == "y"
    assert p.annotations[1].ion_letter == "b"
    assert p.annotations[1].confidence == pytest.approx(0.5)


def test_multi_analyte_prefix():
    p = parse_mzpaf("1@b2,2@y3")
    assert len(p.annotations) == 2
    assert p.annotations[0].analyte_idx == 1
    assert p.annotations[1].analyte_idx == 2


def test_alternates_with_confidence_sum():
    p = parse_mzpaf("b2-H2O/3.2ppm*0.75,b4-H2O^2/3.2ppm*0.25")
    assert len(p.annotations) == 2
    assert sum(a.confidence for a in p.annotations) == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────
# Round-trip
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "annotation",
    [
        "y3",
        "b5^2",
        "y3-H2O",
        "y3-[Phospho]",
        "y3+CO-H2O",
        "y3-H2O+1i[M+H]^2/3.2ppm*0.85",
        "?",
        "y3/-0.001,b6-H2O*0.5",
        "1@b2,2@y3",
    ],
)
def test_round_trip(annotation: str):
    """parse → format must be lossless for spec-compliant inputs."""
    parsed = parse_mzpaf(annotation)
    re_rendered = format_mzpaf(parsed)
    re_parsed = parse_mzpaf(re_rendered)
    assert parsed == re_parsed


# ──────────────────────────────────────────────────────────────────────
# Deferred ion classes raise NotImplementedError (spec-valid, future work)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "annotation",
    [
        "IY",  # immonium
        "IY[Phospho]",  # immonium with mod
        "m3:6",  # internal
        "p^2",  # precursor
        "r[TMT127N]",  # reporter
        "_{Urocanic Acid}",  # named
        "f{C16H22O}",  # formula
        "s{CN=C=O}",  # SMILES
    ],
)
def test_deferred_ion_classes(annotation: str):
    with pytest.raises(NotImplementedError):
        parse_mzpaf(annotation)


# ──────────────────────────────────────────────────────────────────────
# Syntax errors
# ──────────────────────────────────────────────────────────────────────


def test_empty_input_rejected():
    with pytest.raises(MzPAFSyntaxError, match="empty"):
        parse_mzpaf("")


def test_unrecognized_prefix_rejected():
    with pytest.raises(MzPAFSyntaxError, match="unrecognized"):
        parse_mzpaf("Q5")  # not an ion-series letter


def test_unbalanced_bracket_rejected():
    with pytest.raises(MzPAFSyntaxError):
        parse_mzpaf("y3-[Phospho")


# ──────────────────────────────────────────────────────────────────────
# Projection: fragment row -> mzPAF
# ──────────────────────────────────────────────────────────────────────


def test_fragment_row_to_mzpaf_basic():
    s = fragment_row_to_mzpaf(
        {"ion_type": 4, "position": 3, "charge": 1, "loss_id": None}
    )
    assert s == "y3"


def test_fragment_row_to_mzpaf_with_charge():
    s = fragment_row_to_mzpaf(
        {"ion_type": 1, "position": 5, "charge": 2, "loss_id": None}
    )
    assert s == "b5^2"


def test_fragment_row_to_mzpaf_with_loss():
    s = fragment_row_to_mzpaf(
        {"ion_type": 4, "position": 3, "charge": 2, "loss_id": "H2O"}
    )
    assert s == "y3-[H2O]^2"


def test_fragment_row_round_trip_through_parser():
    """A row → mzPAF string → parse → the parsed annotation should
    recover the same ion_type/position/charge/loss."""
    row = {"ion_type": 4, "position": 3, "charge": 2, "loss_id": "H2O"}
    s = fragment_row_to_mzpaf(row)
    a = parse_mzpaf(s).annotations[0]
    assert a.ion_letter == "y"
    assert a.position == 3
    assert a.charge == 2
    assert a.losses[0].token == "H2O"
    assert a.losses[0].is_named is True


def test_fragment_table_vectorized():
    table = pa.table(
        {
            "ion_type": pa.array([1, 4, 4], type=pa.int8()),
            "position": pa.array([2, 3, 5], type=pa.int32()),
            "charge": pa.array([1, 2, 1], type=pa.int32()),
            "loss_id": pa.array([None, "H2O", None], type=pa.string()),
        }
    )
    arr = fragment_table_to_mzpaf(table)
    assert arr.to_pylist() == ["b2", "y3-[H2O]^2", "y5"]


def test_fragment_table_handles_missing_loss_column():
    """Tables without a loss_id column should still vectorize cleanly."""
    table = pa.table(
        {
            "ion_type": pa.array([1, 4], type=pa.int8()),
            "position": pa.array([2, 3], type=pa.int32()),
            "charge": pa.array([1, 1], type=pa.int32()),
        }
    )
    arr = fragment_table_to_mzpaf(table)
    assert arr.to_pylist() == ["b2", "y3"]


# ──────────────────────────────────────────────────────────────────────
# Empty / pathological annotations
# ──────────────────────────────────────────────────────────────────────


def test_empty_peak_annotation_construction():
    """A PeakAnnotation built from an empty tuple is valid (zero alternates)."""
    p = PeakAnnotation()
    assert p.annotations == ()
    assert format_mzpaf(p) == ""
