"""Tests for :mod:`massspec.io.maxquant` — MaxQuant ``combined/txt/`` reader.

Synthetic fixtures under ``tests/data/maxquant/`` (real ProteomeTools
exports are gitignored and never committed). An opt-in smoke test runs
against a real export when ``MAXQUANT_SMOKE_PATH`` is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

# Importing the io package triggers adapter registration in SEARCH_READERS.
import constellation.massspec.io  # noqa: F401
from constellation.core.sequence.proforma import format_proforma, parse_proforma
from constellation.massspec.acquisitions import ACQUISITION_TABLE, Acquisitions
from constellation.massspec.io.maxquant import (
    MaxQuantModResolutionError,
    parse_fixed_modifications,
    parse_maxquant_modseq,
    read_maxquant_search,
)
from constellation.massspec.search.io import (
    SEARCH_READERS,
    load_search,
    save_search,
)
from constellation.massspec.search.schemas import PSM_TABLE

DATA = Path(__file__).resolve().parent / "data" / "maxquant"
ROOT = DATA / "root_layout"
NESTED = DATA / "nested_layout"
HEADER_ONLY = DATA / "header_only"


def _by_id(psms: pa.Table) -> dict[int, dict]:
    """psm_id → row dict (fixtures are tiny; to_pylist is fine here)."""
    return {r["psm_id"]: r for r in psms.to_pylist()}


# ──────────────────────────────────────────────────────────────────────
# Registration + dispatch (format-name only)
# ──────────────────────────────────────────────────────────────────────


def test_registered_with_empty_extension():
    assert "maxquant" in SEARCH_READERS
    assert SEARCH_READERS["maxquant"].extension == ""


def test_load_search_by_format_name():
    direct = read_maxquant_search(ROOT)
    via_registry = load_search(ROOT, format="maxquant")
    assert via_registry.psms.num_rows == direct.psms.num_rows == 10


def test_bare_dir_path_does_not_dispatch_to_maxquant():
    # extension="" never auto-resolves; a bare directory resolves to the
    # ParquetDir reader, which fails on the missing acquisitions.parquet.
    with pytest.raises(Exception):  # noqa: B017 - any error, just not a maxquant Search
        load_search(ROOT)


# ──────────────────────────────────────────────────────────────────────
# Locator (root vs nested layout)
# ──────────────────────────────────────────────────────────────────────


def test_locator_root_layout():
    s = read_maxquant_search(ROOT)
    assert s.psms.num_rows == 10
    assert s.metadata_extras["x.maxquant.msms_path"].endswith("root_layout/msms.txt")


def test_locator_nested_layout_ignores_sibling():
    # msms.txt is under txt/; the _meta/mqpar.xml sibling must not confuse it.
    s = read_maxquant_search(NESTED)
    assert s.psms.num_rows == 2
    assert s.metadata_extras["x.maxquant.msms_path"].endswith("nested_layout/txt/msms.txt")


# ──────────────────────────────────────────────────────────────────────
# PSM table shape + primary key
# ──────────────────────────────────────────────────────────────────────


def test_psm_schema_matches():
    s = read_maxquant_search(ROOT)
    assert s.psms.schema == PSM_TABLE


def test_primary_key_unique_and_validates():
    # The MULTI-SECPEP pair shares (raw_file, scan) but has distinct psm_id;
    # the Search container constructs without raising → PK is (raw_file, psm_id).
    s = read_maxquant_search(ROOT)
    pairs = list(
        zip(
            s.psms.column("raw_file").to_pylist(),
            s.psms.column("psm_id").to_pylist(),
            strict=True,
        )
    )
    assert len(set(pairs)) == len(pairs) == 10


def test_multi_secpep_duplicate_scan_preserved():
    s = read_maxquant_search(ROOT)
    scan200 = s.psms.filter(pc.equal(s.psms.column("scan"), 200))
    assert scan200.num_rows == 2
    assert set(scan200.column("psm_id").to_pylist()) == {6, 7}
    assert set(scan200.column("raw_file").to_pylist()) == {"runA"}


# ──────────────────────────────────────────────────────────────────────
# modseq → ProForma (incl. fixed-mod reconstruction)
# ──────────────────────────────────────────────────────────────────────


def test_modseq_translation():
    s = read_maxquant_search(ROOT)
    rows = _by_id(s.psms)
    assert rows[0]["modified_sequence"] == "PEPTIDEK"  # unmodified
    assert rows[1]["modified_sequence"] == "AC[UNIMOD:4]DEFGHK"  # fixed Cam reconstructed
    assert rows[2]["modified_sequence"] == "M[UNIMOD:35]PEPTIDEK"  # variable ox
    assert rows[3]["modified_sequence"] == "M[UNIMOD:35]C[UNIMOD:4]PEPTIK"  # ox + fixed Cam
    assert rows[9]["modified_sequence"] == "[UNIMOD:1]-PEPTIDEK"  # N-term acetyl


def test_modseqs_round_trip_through_proforma():
    s = read_maxquant_search(ROOT)
    for modseq in s.psms.column("modified_sequence").to_pylist():
        if modseq is None:
            continue
        assert format_proforma(parse_proforma(modseq)) == modseq


# ──────────────────────────────────────────────────────────────────────
# decoy / contaminant flags
# ──────────────────────────────────────────────────────────────────────


def test_decoy_and_contaminant_flags():
    s = read_maxquant_search(ROOT)
    rows = _by_id(s.psms)
    decoys = {i for i, r in rows.items() if r["is_decoy"]}
    contaminants = {i for i, r in rows.items() if r["is_contaminant"]}
    assert decoys == {4}
    assert contaminants == {5}
    assert rows[5]["proteins"] == "P2;CON__Q9XYZ1"  # verbatim, not split
    # both flags are non-nullable: blanks resolve to False, never null
    assert s.psms.column("is_decoy").null_count == 0
    assert s.psms.column("is_contaminant").null_count == 0


# ──────────────────────────────────────────────────────────────────────
# units + dtypes
# ──────────────────────────────────────────────────────────────────────


def test_retention_time_normalized_to_seconds():
    s = read_maxquant_search(ROOT)
    # fixture RT is 10.0 minutes everywhere → 600.0 seconds
    assert set(s.psms.column("retention_time_s").to_pylist()) == {600.0}


def test_dtypes():
    s = read_maxquant_search(ROOT)
    types = {f.name: f.type for f in s.psms.schema}
    assert types["psm_id"] == pa.int64()
    assert types["scan"] == pa.int32()
    assert types["precursor_scan"] == pa.int32()
    assert types["charge"] == pa.int8()
    assert types["retention_time_s"] == pa.float64()
    assert types["is_decoy"] == pa.bool_()


def test_analyzer_and_fragmentation_carried_verbatim():
    s = read_maxquant_search(ROOT)
    rows = _by_id(s.psms)
    assert rows[0]["mass_analyzer"] == "FTMS" and rows[0]["fragmentation"] == "HCD"
    assert rows[8]["mass_analyzer"] == "ITMS" and rows[8]["fragmentation"] == "CID"


# ──────────────────────────────────────────────────────────────────────
# acquisition association
# ──────────────────────────────────────────────────────────────────────


def test_acquisition_synthesis_default():
    s = read_maxquant_search(ROOT)
    # two distinct Raw file values → two synthesised acquisitions
    acq = s.acquisitions.table.to_pylist()
    assert {a["source_file"] for a in acq} == {"runA", "runB"}
    assert all(a["source_kind"] == "maxquant" for a in acq)
    assert s.psms.column("acquisition_id").null_count == 0
    rows = _by_id(s.psms)
    # sorted raw files: runA→0, runB→1
    assert rows[0]["acquisition_id"] == 0  # runA
    assert rows[8]["acquisition_id"] == 1  # runB


def test_acquisition_binding_to_provided():
    provided = Acquisitions(
        pa.Table.from_pylist(
            [
                {"acquisition_id": 7, "source_file": "runA.raw", "source_kind": "thermo", "acquisition_datetime": None},
                {"acquisition_id": 9, "source_file": "runB.raw", "source_kind": "thermo", "acquisition_datetime": None},
            ],
            schema=ACQUISITION_TABLE,
        )
    )
    s = read_maxquant_search(ROOT, acquisitions=provided)
    rows = _by_id(s.psms)
    assert rows[0]["acquisition_id"] == 7  # runA.raw stem == runA
    assert rows[8]["acquisition_id"] == 9  # runB.raw stem == runB
    assert s.acquisitions.table.num_rows == 2


def test_acquisition_binding_unmatched_recorded():
    provided = Acquisitions(
        pa.Table.from_pylist(
            [
                {"acquisition_id": 7, "source_file": "runA.raw", "source_kind": "thermo", "acquisition_datetime": None},
            ],
            schema=ACQUISITION_TABLE,
        )
    )
    s = read_maxquant_search(ROOT, acquisitions=provided)
    rows = _by_id(s.psms)
    assert rows[0]["acquisition_id"] == 7  # runA matched
    assert rows[8]["acquisition_id"] is None  # runB unmatched → null (allowed)
    assert s.metadata_extras["x.maxquant.unmatched_raw_files"] == ["runB"]


# ──────────────────────────────────────────────────────────────────────
# header-only + provenance
# ──────────────────────────────────────────────────────────────────────


def test_header_only_is_zero_psms_not_error():
    s = read_maxquant_search(HEADER_ONLY)
    assert s.psms.num_rows == 0
    assert s.psms.schema == PSM_TABLE
    # parameters provenance is still captured
    assert s.metadata_extras["x.maxquant.parameters.Version"] == "1.5.3.30"


def test_parameters_provenance_captured():
    s = read_maxquant_search(ROOT)
    assert s.metadata_extras["x.maxquant.parameters.Version"] == "1.5.3.30"
    assert (
        s.metadata_extras["x.maxquant.parameters.Fixed modifications"]
        == "Carbamidomethyl (C)"
    )


# ──────────────────────────────────────────────────────────────────────
# unresolved modification handling
# ──────────────────────────────────────────────────────────────────────


def test_unresolved_inline_mod_raises_in_parser():
    with pytest.raises(MaxQuantModResolutionError):
        parse_maxquant_modseq("_A(zz)K_")


def test_unresolved_mod_keeps_psm_with_null_modseq(tmp_path):
    # A modseq with an unknown inline code → PSM kept, modified_sequence null,
    # the offending modseq recorded in metadata.
    header = (
        "Raw file\tScan number\tSequence\tModified sequence\tCharge\tReverse\tid\n"
    )
    body = "runZ\t5\tAK\t_A(zz)K_\t2\t\t0\n"
    (tmp_path / "msms.txt").write_text(header + body, encoding="utf-8")
    s = read_maxquant_search(tmp_path)
    assert s.psms.num_rows == 1
    assert s.psms.column("modified_sequence").to_pylist() == [None]
    assert s.metadata_extras["x.maxquant.unresolved_mods"] == ["_A(zz)K_"]


# ──────────────────────────────────────────────────────────────────────
# parameters / fixed-mod parsing units
# ──────────────────────────────────────────────────────────────────────


def test_parse_fixed_modifications():
    assert parse_fixed_modifications("Carbamidomethyl (C)") == [("Carbamidomethyl", "C")]
    assert parse_fixed_modifications(
        "Carbamidomethyl (C);Acetyl (Protein N-term)"
    ) == [("Carbamidomethyl", "C"), ("Acetyl", "Protein N-term")]
    assert parse_fixed_modifications("") == []
    assert parse_fixed_modifications(None) == []


# ──────────────────────────────────────────────────────────────────────
# ParquetDir round-trip of the produced Search.psms
# ──────────────────────────────────────────────────────────────────────


def test_psms_parquet_round_trip(tmp_path):
    s = read_maxquant_search(ROOT)
    out = tmp_path / "search"
    save_search(s, out, format="parquet_dir")
    s2 = load_search(out, format="parquet_dir")
    assert s2.psms.num_rows == 10
    assert s2.psms.schema == PSM_TABLE
    assert s2.psms.equals(s.psms)


# ──────────────────────────────────────────────────────────────────────
# Opt-in smoke against a real export (gitignored data)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    "MAXQUANT_SMOKE_PATH" not in os.environ,
    reason="set MAXQUANT_SMOKE_PATH to a real MaxQuant combined/txt export dir",
)
def test_real_export_smoke():
    s = read_maxquant_search(os.environ["MAXQUANT_SMOKE_PATH"])
    assert s.psms.num_rows > 100
    assert s.psms.column("acquisition_id").null_count == 0
    for modseq in s.psms.column("modified_sequence").to_pylist():
        if modseq is not None:
            assert format_proforma(parse_proforma(modseq)) == modseq
