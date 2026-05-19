"""Tests for the NIST .msp library-view reader.

Covers the synthetic fixture committed at ``tests/data/msp/synthetic.msp``
plus the raise-on-raw path at ``tests/data/msp/raw_only.msp``. Opt-in
big-file smoke test at the bottom runs against the ProteomeTools
FTMS_HCD_20 distribution when present.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pytest

from constellation.massspec.io.msp import read_msp_library
from constellation.massspec.library import (
    LIBRARY_READERS,
    load_library,
    save_library,
)
from constellation.massspec.peptide.ions import IonType, fragment_ladder
from constellation.massspec.peptide.mz import precursor_mz
from constellation.core.sequence.proforma import (
    format_proforma,
    parse_proforma,
)


FIXTURE = Path(__file__).parent / "data" / "msp" / "synthetic.msp"
RAW_FIXTURE = Path(__file__).parent / "data" / "msp" / "raw_only.msp"


# ──────────────────────────────────────────────────────────────────────
# Registration + dispatch
# ──────────────────────────────────────────────────────────────────────


def test_msp_reader_is_registered() -> None:
    assert "nist.msp" in LIBRARY_READERS
    reader = LIBRARY_READERS["nist.msp"]
    assert reader.extension == ".msp"


def test_load_library_dispatches_by_suffix() -> None:
    lib_via_dispatch = load_library(FIXTURE)
    lib_direct = read_msp_library(FIXTURE)
    assert lib_via_dispatch.peptides.num_rows == lib_direct.peptides.num_rows


# ──────────────────────────────────────────────────────────────────────
# Library shape + FK closure
# ──────────────────────────────────────────────────────────────────────


def test_synthetic_library_shape() -> None:
    lib = read_msp_library(FIXTURE)
    # 10 entries → 10 unique peptidoforms (no dupes in fixture) → 10 precursors
    assert lib.metadata_extras["x.msp.n_entries"] == 10
    assert lib.peptides.num_rows == 10
    assert lib.precursors.num_rows == 10
    assert lib.fragments.num_rows > 0
    # Library __post_init__ validated PK + FK; reaching this point proves
    # closure holds. Spot-check the protein side too.
    assert lib.proteins.num_rows == 2  # ACC1, ACC2 from REGNANTA
    assert lib.protein_peptide.num_rows == 2


# ──────────────────────────────────────────────────────────────────────
# Modification translation through the canonical ProForma path
# ──────────────────────────────────────────────────────────────────────


def test_mods_field_translates_to_proforma_unimod_accession() -> None:
    lib = read_msp_library(FIXTURE)
    modseqs = [r["modified_sequence"] for r in lib.peptides.to_pylist()]
    assert "AAAAAAAAQM[UNIMOD:35]HAK" in modseqs
    assert "AC[UNIMOD:4]ST[UNIMOD:21]PK" in modseqs


def test_modified_sequence_round_trips_through_parse_proforma() -> None:
    lib = read_msp_library(FIXTURE)
    for row in lib.peptides.to_pylist():
        modseq = row["modified_sequence"]
        pep = parse_proforma(modseq)
        # ProForma 2.0 stable canonical form should round-trip exactly
        assert format_proforma(pep) == modseq


# ──────────────────────────────────────────────────────────────────────
# m/z is theoretical, not observed
# ──────────────────────────────────────────────────────────────────────


def test_mz_theoretical_comes_from_fragment_ladder() -> None:
    """For each fragment row with structured columns populated, the
    stored ``mz_theoretical`` must equal an independent
    ``fragment_ladder`` recomputation. Proves the reader is not
    silently copying the observed peak m/z into the theoretical
    column."""
    lib = read_msp_library(FIXTURE)
    # Build {(precursor_id) -> peptidoform_with_charge} once
    pid_to_pep: dict[int, tuple[object, int]] = {}
    peptide_by_id = {r["peptide_id"]: r for r in lib.peptides.to_pylist()}
    for prec in lib.precursors.to_pylist():
        pep_row = peptide_by_id[prec["peptide_id"]]
        modseq = pep_row["modified_sequence"] + f"/{prec['charge']}"
        pep = parse_proforma(modseq)
        pid_to_pep[prec["precursor_id"]] = (pep, prec["charge"])

    for row in lib.fragments.to_pylist():
        if row["ion_type"] is None:
            continue  # unparseable — m/z is observed pass-through; tested below
        pep, charge = pid_to_pep[row["precursor_id"]]
        table, _ = fragment_ladder(
            pep,
            ion_types=tuple(IonType),
            max_fragment_charge=max(charge, 1),
            neutral_losses=["H2O", "NH3", "HPO3", "H3PO4"],
            return_tensor=False,
        )
        # Find matching row by (ion_type, position, charge, loss_id)
        key = (
            row["ion_type"],
            row["position"],
            row["charge"],
            row["loss_id"],
        )
        ladder_rows = table.to_pylist()
        candidates = [
            r
            for r in ladder_rows
            if (
                int(r["ion_type"]) == key[0]
                and int(r["position"]) == key[1]
                and int(r["charge"]) == key[2]
                and r["loss_id"] == key[3]
            )
        ]
        assert candidates, f"no ladder row for {key}"
        assert candidates[0]["mz_theoretical"] == pytest.approx(
            row["mz_theoretical"], abs=1e-6
        )


def test_precursor_mz_is_computed_from_peptidoform() -> None:
    """``PRECURSOR_TABLE.precursor_mz`` is theoretical (derived from
    the peptidoform), not the ``Parent=`` value from the file."""
    lib = read_msp_library(FIXTURE)
    peptide_by_id = {r["peptide_id"]: r for r in lib.peptides.to_pylist()}
    for prec in lib.precursors.to_pylist():
        pep_row = peptide_by_id[prec["peptide_id"]]
        modseq = pep_row["modified_sequence"] + f"/{prec['charge']}"
        pep = parse_proforma(modseq)
        expected = precursor_mz(pep)
        assert prec["precursor_mz"] == pytest.approx(expected, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────
# Intensity handling — raw pass-through by default
# ──────────────────────────────────────────────────────────────────────


def test_intensity_raw_pass_through_by_default() -> None:
    """A hand-picked peak in the fixture has raw intensity 9999.0 (the
    b/y peak in entry 1). Verify it survives unscaled."""
    lib = read_msp_library(FIXTURE)
    intensities = lib.fragments.column("intensity_predicted").to_pylist()
    assert max(intensities) == pytest.approx(9999.0)


def test_intensity_normalize_max_brings_per_spectrum_max_to_one() -> None:
    lib = read_msp_library(FIXTURE, intensity_normalize="max")
    # Per-precursor max should now be 1.0
    by_prec: dict[int, list[float]] = {}
    for row in lib.fragments.to_pylist():
        by_prec.setdefault(row["precursor_id"], []).append(
            row["intensity_predicted"]
        )
    for ints in by_prec.values():
        assert max(ints) == pytest.approx(1.0, abs=1e-9)


def test_intensity_predicted_dtype_is_float64() -> None:
    lib = read_msp_library(FIXTURE)
    assert lib.fragments.field("intensity_predicted").type == pa.float64()


# ──────────────────────────────────────────────────────────────────────
# Comment fields → side-table + Library columns
# ──────────────────────────────────────────────────────────────────────


def test_irt_na_maps_to_neg_one_sentinel() -> None:
    lib = read_msp_library(FIXTURE)
    rt_values = lib.precursors.column("rt_predicted").to_pylist()
    # PEPTID and NAMEDPATH entries carry iRT=NA / no iRT
    assert -1.0 in rt_values
    # Numeric iRTs survive
    assert 42.5 in rt_values
    assert -24.61 in rt_values


def test_protein_field_populates_proteins_and_edges() -> None:
    lib = read_msp_library(FIXTURE)
    accessions = sorted(
        r["accession"] for r in lib.proteins.to_pylist()
    )
    assert accessions == ["ACC1", "ACC2"]
    edges = lib.protein_peptide.to_pylist()
    assert len(edges) == 2  # both proteins → same peptide (REGNANTA)


def test_side_table_carries_only_non_redundant_comment_keys() -> None:
    lib = read_msp_library(FIXTURE)
    side = lib.metadata_extras["x.msp.precursor_comments"]
    cols = set(side.column_names)
    assert "precursor_id" in cols
    # Extended keys from HCDREADY entry
    for key in ("Inst", "Collision_energy", "Scan", "RTInSeconds"):
        assert key in cols, f"expected {key} in side-table columns"
    # Library-redundant keys MUST NOT appear as side-table columns
    for redundant in (
        "iRT",
        "Charge",
        "Parent",
        "Mods",
        "Modstring",
        "Fullname",
        "Pep",
        "Name",
        "Protein",
    ):
        assert redundant not in cols, (
            f"library-redundant key {redundant} leaked into side-table"
        )


def test_side_table_pk_aligns_with_precursors() -> None:
    lib = read_msp_library(FIXTURE)
    side = lib.metadata_extras["x.msp.precursor_comments"]
    pids = set(side.column("precursor_id").to_pylist())
    precursor_ids = set(
        lib.precursors.column("precursor_id").to_pylist()
    )
    # Side-table PKs are a subset of precursor IDs (we only emit rows
    # for entries that carried extra Comment keys).
    assert pids.issubset(precursor_ids)


# ──────────────────────────────────────────────────────────────────────
# Unparseable / unsupported annotations
# ──────────────────────────────────────────────────────────────────────


def test_unparseable_annotation_preserved_with_null_structured_columns() -> None:
    """The SAMPLED entry carries a peak annotated ``IS-ATC/0.0ppm``
    (internal-fragment shape, not supported by mzPAF Phase 1). The
    row must exist with NULL structured columns and the raw
    annotation string preserved."""
    lib = read_msp_library(FIXTURE)
    null_rows = [
        r for r in lib.fragments.to_pylist() if r["ion_type"] is None
    ]
    assert null_rows, "expected at least one unparseable-annotation row"
    raw = [r["annotation"] for r in null_rows]
    assert any("IS-ATC" in (a or "") for a in raw)
    # Counters reflect the same
    assert lib.metadata_extras["x.msp.unparseable_annotations"] >= 1


# ──────────────────────────────────────────────────────────────────────
# Raw-style MSP raises
# ──────────────────────────────────────────────────────────────────────


def test_raw_style_msp_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no parseable peptide"):
        read_msp_library(RAW_FIXTURE)


# ──────────────────────────────────────────────────────────────────────
# ParquetDir round-trip preserves the side-table
# ──────────────────────────────────────────────────────────────────────


def test_parquet_round_trip_preserves_side_table(tmp_path: Path) -> None:
    lib = read_msp_library(FIXTURE)
    out = tmp_path / "round_trip"
    save_library(lib, out, format="parquet_dir")
    rehydrated = load_library(out, format="parquet_dir")
    original = lib.metadata_extras["x.msp.precursor_comments"]
    recovered = rehydrated.metadata_extras["x.msp.precursor_comments"]
    assert isinstance(recovered, pa.Table)
    assert original.equals(recovered)


def test_parquet_round_trip_preserves_scalar_metadata(
    tmp_path: Path,
) -> None:
    lib = read_msp_library(FIXTURE)
    out = tmp_path / "rt_scalar"
    save_library(lib, out, format="parquet_dir")
    rehydrated = load_library(out, format="parquet_dir")
    for k in (
        "x.msp.source_path",
        "x.msp.format_name",
        "x.msp.n_entries",
        "x.msp.unparseable_annotations",
        "x.msp.dropped_peaks",
    ):
        assert rehydrated.metadata_extras[k] == lib.metadata_extras[k]


# ──────────────────────────────────────────────────────────────────────
# Opt-in big-file smoke (skipped unless env var points at the file)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("MSP_SMOKE_PATH"),
    reason="MSP_SMOKE_PATH not set — opt-in smoke test against a real "
    "ProteomeTools MSP. Set MSP_SMOKE_PATH=/path/to/FTMS_HCD_20_annotated_*.msp",
)
def test_proteome_tools_smoke() -> None:
    path = Path(os.environ["MSP_SMOKE_PATH"]).expanduser()
    if not path.exists():
        pytest.skip(f"MSP_SMOKE_PATH points at missing file: {path}")
    lib = read_msp_library(path)
    assert lib.precursors.num_rows > 1000
    assert all(
        mz > 0
        for mz in lib.precursors.column("precursor_mz").to_pylist()[:1000]
    )
    # Biognosys iRT range (loose bounds — the file does have iRT=NA rows)
    rts = [
        r
        for r in lib.precursors.column("rt_predicted").to_pylist()
        if r != -1.0
    ]
    assert rts, "expected some numeric iRT values in ProteomeTools file"
    assert -60.0 < min(rts) < max(rts) < 200.0
