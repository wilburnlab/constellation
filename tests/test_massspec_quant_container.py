"""Container-level tests for ``constellation.massspec.quant.Quant``."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.massspec.acquisitions import Acquisitions
from constellation.massspec.library import Library, assign_ids
from constellation.massspec.peptide.ions import IonType
from constellation.massspec.quant import Quant, assemble_quant


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _library() -> Library:
    return assign_ids(
        proteins=[{"accession": "P1", "sequence": "M"}],
        peptides=[{"modified_sequence": "PEPTIDE"}],
        precursors=[
            {"modified_sequence": "PEPTIDE", "charge": 2, "precursor_mz": 400.0}
        ],
        fragments=[
            {
                "modified_sequence": "PEPTIDE",
                "precursor_charge": 2,
                "ion_type": int(IonType.Y),
                "position": 5,
                "charge": 1,
                "mz_theoretical": 600.0,
                "intensity_predicted": 1.0,
            }
        ],
        protein_peptide=[("P1", "PEPTIDE")],
    )


def _acquisitions() -> Acquisitions:
    return Acquisitions.from_records(
        [
            {
                "acquisition_id": 0,
                "source_file": "run_0.raw",
                "source_kind": "thermo_raw",
                "acquisition_datetime": None,
            },
            {
                "acquisition_id": 1,
                "source_file": "run_1.raw",
                "source_kind": "thermo_raw",
                "acquisition_datetime": None,
            },
        ]
    )


def _basic_quant() -> Quant:
    return assemble_quant(
        acquisitions=_acquisitions(),
        protein_quant=[
            {"protein_id": 0, "acquisition_id": 0, "abundance": 1.2e6, "score": 0.9},
            {"protein_id": 0, "acquisition_id": 1, "abundance": 1.5e6, "score": 0.85},
        ],
        peptide_quant=[
            {"peptide_id": 0, "acquisition_id": 0, "abundance": 9.0e5, "score": None},
        ],
        precursor_quant=[
            {
                "precursor_id": 0,
                "acquisition_id": 0,
                "intensity": 5.0e5,
                "rt_observed": 1234.5,
                "ccs_observed": -1.0,
                "score": None,
            },
        ],
        transmission_protein_peptide=[
            # Run-agnostic calibration
            {
                "protein_id": 0,
                "peptide_id": 0,
                "acquisition_id": None,
                "efficiency": 0.42,
            },
        ],
        transmission_peptide_precursor=[
            {
                "peptide_id": 0,
                "precursor_id": 0,
                "acquisition_id": None,
                "efficiency": 0.78,
            },
        ],
        metadata={"library_id": "abc123"},
    )


# ──────────────────────────────────────────────────────────────────────
# Construction + validation
# ──────────────────────────────────────────────────────────────────────


def test_minimal_quant_constructs() -> None:
    q = _basic_quant()
    assert q.protein_quant.num_rows == 2
    assert q.peptide_quant.num_rows == 1
    assert q.precursor_quant.num_rows == 1
    assert q.transmission_protein_peptide.num_rows == 1
    assert q.transmission_peptide_precursor.num_rows == 1


def test_empty_quant_constructs() -> None:
    q = Quant.empty(_acquisitions())
    assert q.protein_quant.num_rows == 0


def test_validate_against_library() -> None:
    lib = _library()
    q = _basic_quant()
    q.validate_against(lib)


def test_validate_against_library_rejects_unknown_protein() -> None:
    lib = _library()
    q = assemble_quant(
        acquisitions=_acquisitions(),
        protein_quant=[
            {"protein_id": 999, "acquisition_id": 0, "abundance": 1.0, "score": None},
        ],
    )
    with pytest.raises(ValueError, match="not present in Library.proteins"):
        q.validate_against(lib)


def test_validate_rejects_unknown_acquisition_id() -> None:
    with pytest.raises(ValueError, match="unknown acquisition_ids"):
        assemble_quant(
            acquisitions=_acquisitions(),
            protein_quant=[
                {
                    "protein_id": 0,
                    "acquisition_id": 999,  # not in acquisitions
                    "abundance": 1.0,
                    "score": None,
                },
            ],
        )


def test_validate_rejects_null_acquisition_id_on_quant() -> None:
    with pytest.raises(ValueError, match="nullable=False"):
        assemble_quant(
            acquisitions=_acquisitions(),
            protein_quant=[
                {
                    "protein_id": 0,
                    "acquisition_id": None,
                    "abundance": 1.0,
                    "score": None,
                },
            ],
        )


def test_validate_rejects_duplicate_quant_pk() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        assemble_quant(
            acquisitions=_acquisitions(),
            protein_quant=[
                {"protein_id": 0, "acquisition_id": 0, "abundance": 1.0, "score": None},
                {"protein_id": 0, "acquisition_id": 0, "abundance": 2.0, "score": None},
            ],
        )


# ──────────────────────────────────────────────────────────────────────
# Efficiency bounds
# ──────────────────────────────────────────────────────────────────────


def test_efficiency_at_one_accepted() -> None:
    """Upper bound is inclusive — efficiency = 1.0 is valid."""
    q = assemble_quant(
        acquisitions=_acquisitions(),
        transmission_protein_peptide=[
            {
                "protein_id": 0,
                "peptide_id": 0,
                "acquisition_id": None,
                "efficiency": 1.0,
            }
        ],
    )
    assert q.transmission_protein_peptide.num_rows == 1


def test_efficiency_above_one_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        assemble_quant(
            acquisitions=_acquisitions(),
            transmission_protein_peptide=[
                {
                    "protein_id": 0,
                    "peptide_id": 0,
                    "acquisition_id": None,
                    "efficiency": 1.05,
                }
            ],
        )


def test_efficiency_zero_rejected() -> None:
    """Lower bound is exclusive — 0.0 is rejected (use the -1.0
    sentinel for uncalibrated rows instead)."""
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        assemble_quant(
            acquisitions=_acquisitions(),
            transmission_peptide_precursor=[
                {
                    "peptide_id": 0,
                    "precursor_id": 0,
                    "acquisition_id": None,
                    "efficiency": 0.0,
                }
            ],
        )


def test_efficiency_negative_one_sentinel_accepted() -> None:
    """-1.0 sentinel is exempt from bounds checking."""
    q = assemble_quant(
        acquisitions=_acquisitions(),
        transmission_protein_peptide=[
            {
                "protein_id": 0,
                "peptide_id": 0,
                "acquisition_id": None,
                "efficiency": -1.0,
            }
        ],
    )
    assert q.transmission_protein_peptide.num_rows == 1


def test_efficiency_other_negative_rejected() -> None:
    """Negative values that aren't the -1.0 sentinel still fail."""
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        assemble_quant(
            acquisitions=_acquisitions(),
            transmission_peptide_precursor=[
                {
                    "peptide_id": 0,
                    "precursor_id": 0,
                    "acquisition_id": None,
                    "efficiency": -0.5,
                }
            ],
        )


# ──────────────────────────────────────────────────────────────────────
# Run-agnostic + per-acquisition transmission rows coexist
# ──────────────────────────────────────────────────────────────────────


def test_transmission_can_mix_run_agnostic_and_per_acquisition() -> None:
    q = assemble_quant(
        acquisitions=_acquisitions(),
        transmission_protein_peptide=[
            {
                "protein_id": 0,
                "peptide_id": 0,
                "acquisition_id": None,
                "efficiency": 0.5,
            },
            {
                "protein_id": 0,
                "peptide_id": 0,
                "acquisition_id": 0,
                "efficiency": 0.42,
            },
            {
                "protein_id": 0,
                "peptide_id": 0,
                "acquisition_id": 1,
                "efficiency": 0.55,
            },
        ],
    )
    assert q.transmission_protein_peptide.num_rows == 3


def test_transmission_duplicate_endpoint_pair_with_same_acquisition_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        assemble_quant(
            acquisitions=_acquisitions(),
            transmission_protein_peptide=[
                {
                    "protein_id": 0,
                    "peptide_id": 0,
                    "acquisition_id": 0,
                    "efficiency": 0.42,
                },
                {
                    "protein_id": 0,
                    "peptide_id": 0,
                    "acquisition_id": 0,
                    "efficiency": 0.55,
                },
            ],
        )


# ──────────────────────────────────────────────────────────────────────
# Metadata / library_id linkage
# ──────────────────────────────────────────────────────────────────────


def test_with_library_id_round_trip() -> None:
    q = _basic_quant().with_library_id("library-hash-xyz")
    assert q.metadata["library_id"] == "library-hash-xyz"


def test_with_metadata_merges() -> None:
    q = _basic_quant().with_metadata({"x.quant.fitter": "Counter-v1"})
    assert q.metadata["library_id"] == "abc123"
    assert q.metadata["x.quant.fitter"] == "Counter-v1"
