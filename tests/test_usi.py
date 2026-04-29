"""Tests for constellation.massspec.annotation.usi — USI v1.0 parser + formatter."""

from __future__ import annotations

import pytest

from constellation.core.sequence.proforma import Peptidoform, parse_proforma
from constellation.massspec.annotation import (
    USI,
    USISyntaxError,
    parse_usi,
)


# ──────────────────────────────────────────────────────────────────────
# Round-trip on canonical spec examples
# ──────────────────────────────────────────────────────────────────────


CANONICAL_USIS = [
    # No interpretation (5-field form)
    "mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555",
    # With ProForma interpretation
    "mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555:VLHPLEGAVVIIFK/2",
    # Modified peptide (Oxidation on M)
    "mzspec:PXD002255:ES2_F47:scan:33829:LHFFM[Oxidation]PGFAPLTSR/3",
    # N-terminal mod (the canonical disambiguation case)
    "mzspec:PXD000561:run01:scan:1:[Acetyl]-KVERPD/2",
    # Index-type variation
    "mzspec:PXD000561:run01:index:42:VLHPLEGAVVIIFK/2",
    # nativeId form
    "mzspec:PXD000561:run01:nativeId:controllerType=0 controllerNumber=1 scan=4242:VLHPLEGAVVIIFK/2",
]


@pytest.mark.parametrize("usi_str", CANONICAL_USIS)
def test_round_trip(usi_str: str):
    parsed = parse_usi(usi_str)
    assert parsed.format() == usi_str


def test_parse_basic_no_interpretation():
    u = parse_usi("mzspec:PXD000561:run01:scan:17555")
    assert u.collection == "PXD000561"
    assert u.run_name == "run01"
    assert u.index_type == "scan"
    assert u.index == "17555"
    assert u.interpretation is None


def test_parse_with_proforma_interpretation():
    u = parse_usi("mzspec:PXD000561:run01:scan:17555:VLHPLEGAVVIIFK/2")
    assert u.interpretation is not None
    assert u.interpretation.sequence == "VLHPLEGAVVIIFK"
    assert u.interpretation.charge == 2


def test_parse_modified_peptide_in_interpretation():
    u = parse_usi(
        "mzspec:PXD002255:ES2_F47:scan:33829:LHFFM[Oxidation]PGFAPLTSR/3"
    )
    assert u.interpretation is not None
    assert u.interpretation.sequence == "LHFFMPGFAPLTSR"
    assert u.interpretation.charge == 3
    assert 4 in u.interpretation.residue_mods  # M is 0-indexed at position 4


def test_parse_n_terminal_acetyl():
    """The K[Acetyl]V vs [Acetyl]-KV distinction must round-trip in USI."""
    u = parse_usi("mzspec:PXD000561:run01:scan:1:[Acetyl]-KVERPD/2")
    assert u.interpretation is not None
    assert u.interpretation.sequence == "KVERPD"
    # ProForma puts the Acetyl on n_term_mods, not residue_mods[0]
    assert len(u.interpretation.n_term_mods) == 1
    assert 0 not in u.interpretation.residue_mods


# ──────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────


def test_from_spectrum_with_scan():
    u = USI.from_spectrum(
        collection="PXD000561",
        run_name="run01",
        scan=4242,
    )
    assert u.format() == "mzspec:PXD000561:run01:scan:4242"


def test_from_spectrum_with_peptidoform():
    pep = parse_proforma("VLHPLEGAVVIIFK/2")
    assert isinstance(pep, Peptidoform)
    u = USI.from_spectrum(
        collection="PXD000561",
        run_name="run01",
        scan=4242,
        peptidoform=pep,
    )
    assert u.format() == "mzspec:PXD000561:run01:scan:4242:VLHPLEGAVVIIFK/2"


def test_from_spectrum_requires_exactly_one_index():
    with pytest.raises(ValueError):
        USI.from_spectrum(collection="X", run_name="Y")  # none provided
    with pytest.raises(ValueError):
        USI.from_spectrum(
            collection="X", run_name="Y", scan=1, index=2
        )  # both provided


# ──────────────────────────────────────────────────────────────────────
# Syntax errors
# ──────────────────────────────────────────────────────────────────────


def test_empty_input_rejected():
    with pytest.raises(USISyntaxError):
        parse_usi("")


def test_missing_mzspec_prefix():
    with pytest.raises(USISyntaxError, match="must start with"):
        parse_usi("foo:PXD:run:scan:1")


def test_too_few_fields():
    with pytest.raises(USISyntaxError, match="at least 5"):
        parse_usi("mzspec:PXD000561:run01:scan")


def test_invalid_index_type():
    with pytest.raises(USISyntaxError, match="index_type must be"):
        parse_usi("mzspec:PXD000561:run01:weird:42")


def test_empty_collection_rejected():
    with pytest.raises(USISyntaxError, match="collection is empty"):
        parse_usi("mzspec::run01:scan:42")


def test_interpretation_without_charge_rejected():
    """USI spec REQUIRES a charge in the interpretation slot."""
    with pytest.raises(USISyntaxError, match="charge"):
        parse_usi("mzspec:PXD000561:run01:scan:1:VLHPLEGAVVIIFK")


def test_format_without_charge_rejected():
    """Building a USI with a peptidoform missing charge → format error."""
    pep = parse_proforma("VLHPLEGAVVIIFK")  # no /N
    assert isinstance(pep, Peptidoform)
    u = USI(
        collection="PXD",
        run_name="run01",
        index_type="scan",
        index="1",
        interpretation=pep,
    )
    with pytest.raises(USISyntaxError, match="charge"):
        u.format()


# ──────────────────────────────────────────────────────────────────────
# Implicit string conversion
# ──────────────────────────────────────────────────────────────────────


def test_str_dunder_calls_format():
    u = parse_usi("mzspec:PXD000561:run01:scan:1")
    assert str(u) == "mzspec:PXD000561:run01:scan:1"
