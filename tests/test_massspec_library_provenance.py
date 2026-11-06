"""The Library's span, span→peptide and modification tables, and its decoy
columns (2026-11-05)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.massspec.library import Library
from constellation.massspec.library.io import load_library, save_library


def _library(**overrides) -> Library:
    proteins = pa.table({"protein_id": [1, 2], "accession": ["A", "B"], "sequence": ["MPEPK", "XXPEPK"]})
    peptides = pa.table(
        {
            "peptide_id": [10, 11, 12],
            "sequence": ["PEPK", "PEPK", "PEPK"],
            "modified_sequence": ["PEPK", "[UNIMOD:1]-PEPK", "PEPK"],
            "is_decoy": [False, False, True],
            "decoy_of": pa.array([None, None, 10], pa.int64()),
            "decoy_set": pa.array([None, None, 0], pa.int16()),
        }
    )
    precursors = pa.table(
        {
            "precursor_id": [100, 101, 102],
            "peptide_id": [10, 11, 12],
            "charge": pa.array([2, 2, 2], pa.int32()),
            "precursor_mz": [1.0, 2.0, 1.0],
            "rt_predicted": [-1.0] * 3,
            "ccs_predicted": [-1.0] * 3,
        }
    )
    spans = pa.table(
        {
            "protein_id": [1, 2],
            "start": pa.array([1, 2], pa.int32()),
            "end": pa.array([5, 6], pa.int32()),
            "sequence": ["PEPK", "PEPK"],
            "missed_cleavages": pa.array([0, 0], pa.int16()),
            "n_term": ["protein", "enzyme"],
            "c_term": ["protein", "protein"],
            "preceding": ["M", "X"],
            "following": ["-", "-"],
        }
    )
    span_peptide = pa.table(
        {
            "protein_id": [1, 2, 1],
            "start": pa.array([1, 2, 1], pa.int32()),
            "end": pa.array([5, 6, 5], pa.int32()),
            "peptide_id": [10, 10, 11],  # the acetyl form only from the protein N-terminus
        }
    )
    tables = dict(
        proteins=proteins,
        peptides=peptides,
        precursors=precursors,
        fragments=pa.table({"precursor_id": pa.array([], pa.int64()), "mz_theoretical": pa.array([], pa.float64()),
                            "intensity_predicted": pa.array([], pa.float64())}),
        protein_peptide=pa.table({"protein_id": [1, 2, 1], "peptide_id": [10, 10, 11]}),
        peptide_spans=spans,
        span_peptide=span_peptide,
        peptide_modifications=pa.table(
            {"peptide_id": [11], "position": pa.array([None], pa.int32()), "terminus": ["N"],
             "modification": ["UNIMOD:1"], "origin": ["variable"], "rule": ["Acetyl@Protein N-term"]}
        ),
    )
    tables.update(overrides)
    return Library(**tables)


def test_valid_library() -> None:
    lib = _library()
    assert lib.peptide_spans.num_rows == 2 and lib.span_peptide.num_rows == 3
    assert lib.peptides.column("is_decoy").to_pylist() == [False, False, True]


def test_edge_must_name_a_span() -> None:
    bad = _library().span_peptide.to_pydict()
    bad["start"][2] = 0
    with pytest.raises(ValueError, match="names spans not in"):
        _library(span_peptide=pa.table(bad))


def test_span_sequence_must_be_the_peptides() -> None:
    spans = _library().peptide_spans.to_pydict()
    spans["sequence"][1] = "PEPR"
    with pytest.raises(ValueError, match="not its peptide's"):
        _library(peptide_spans=pa.table(spans))


def test_protein_peptide_must_be_the_projection() -> None:
    with pytest.raises(ValueError, match="projection"):
        _library(protein_peptide=pa.table({"protein_id": [1, 2], "peptide_id": [10, 10]}))


def test_decoy_of_must_be_a_target() -> None:
    peptides = _library().peptides.to_pydict()
    peptides["decoy_of"][2] = 99
    with pytest.raises(ValueError, match="references ids not present"):
        _library(peptides=pa.table(peptides))
    peptides["decoy_of"] = [None, 12, 10]
    peptides["is_decoy"] = [False, True, True]
    with pytest.raises(ValueError, match="itself a decoy"):
        _library(peptides=pa.table(peptides))


def test_duplicate_spans_refused() -> None:
    spans = pa.concat_tables([_library().peptide_spans, _library().peptide_spans.slice(0, 1)])
    with pytest.raises(ValueError, match="duplicate"):
        _library(peptide_spans=spans)


def test_subset_keeps_spans_and_decoys_of_kept_targets() -> None:
    sub = _library().subset_proteins([2])
    assert sub.peptide_spans.column("protein_id").to_pylist() == [2]
    assert sorted(sub.peptides.column("peptide_id").to_pylist()) == [10, 12]  # 12 is 10's decoy
    assert sub.peptide_modifications.num_rows == 0  # the acetyl form went with protein 1


def test_parquet_dir_round_trip(tmp_path: Path) -> None:
    lib = _library()
    save_library(lib, tmp_path / "lib", format="parquet_dir")
    back = load_library(tmp_path / "lib", format="parquet_dir")
    for name in ("peptide_spans", "span_peptide", "peptide_modifications", "peptides"):
        assert getattr(back, name).equals(getattr(lib, name))


def test_a_library_without_provenance_writes_no_new_files(tmp_path: Path) -> None:
    lib = _library()
    plain = Library(lib.proteins, lib.peptides, lib.precursors, lib.fragments, lib.protein_peptide)
    save_library(plain, tmp_path / "plain", format="parquet_dir")
    assert sorted(p.name for p in (tmp_path / "plain").glob("*.parquet")) == [
        "fragments.parquet", "peptides.parquet", "precursors.parquet", "protein_peptide.parquet", "proteins.parquet",
    ]
    back = load_library(tmp_path / "plain", format="parquet_dir")
    assert back.peptide_spans.num_rows == 0
