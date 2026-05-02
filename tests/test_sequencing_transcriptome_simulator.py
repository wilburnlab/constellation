"""Tests for ``constellation.sequencing.transcriptome.simulator``.

Two tiers:

  (a) unit tests on ``mutate`` and ``assemble_sequence`` — fast,
      no I/O.
  (b) end-to-end test that runs the simulator → demux pipeline and
      asserts ground-truth status + barcode_id agreement on the
      deterministic-clean subset (must be 100%).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.samples import Samples
from constellation.sequencing.transcriptome.classify import ReadStatus
from constellation.sequencing.transcriptome.designs import CDNA_WILBURN_V1
from constellation.sequencing.transcriptome.simulator import (
    GROUND_TRUTH_TABLE,
    ReadSpec,
    assemble_sequence,
    generate_stress_test_specs,
    is_deterministic_clean,
    mutate,
    simulate_panel,
)


# ──────────────────────────────────────────────────────────────────────
# Tier (a) — unit tests on the mutation primitive
# ──────────────────────────────────────────────────────────────────────


def test_mutate_zero_edits_is_identity() -> None:
    rng = np.random.default_rng(0)
    seq = "ACGTACGTAC"
    assert mutate(seq, 0, rng=rng) == seq


def test_mutate_deterministic_under_seed() -> None:
    seq = "ACGTACGTAC"
    a = mutate(seq, 5, rng=np.random.default_rng(42))
    b = mutate(seq, 5, rng=np.random.default_rng(42))
    assert a == b


def test_mutate_substitution_only_preserves_length() -> None:
    seq = "ACGTACGTAC"
    out = mutate(seq, 3, kinds=("sub",), rng=np.random.default_rng(7))
    assert len(out) == len(seq)


def test_mutate_insertion_only_grows_length() -> None:
    seq = "ACGTACGTAC"
    out = mutate(seq, 4, kinds=("ins",), rng=np.random.default_rng(7))
    assert len(out) == len(seq) + 4


def test_mutate_deletion_only_shrinks_length() -> None:
    seq = "A" * 20
    out = mutate(seq, 3, kinds=("del",), rng=np.random.default_rng(7))
    assert len(out) == 17


# ──────────────────────────────────────────────────────────────────────
# Tier (a) — assemble_sequence structural tests
# ──────────────────────────────────────────────────────────────────────


_SSP = "AAGCAGTGGTATCAACGCAGAGTACGCGGG"
_PRIMER3 = "GTACTCTGCGTTGATACCACTGCTT"


def _bare_complete_spec(rid: str = "test") -> ReadSpec:
    return ReadSpec(
        read_id=rid,
        expected_status=ReadStatus.COMPLETE,
        orientation="+",
        transcript_length=300,
        transcript_id=rid,
        polyA_length=22,
        polyA_artifact="clean",
        barcode_index=0,
    )


def test_assemble_clean_complete_contains_all_anchors() -> None:
    """Zero-mutation Complete spec → SSP, polyA run, primer3 all
    present in order, with the expected transcript length between."""
    rng = np.random.default_rng(0)
    spec = _bare_complete_spec()
    seq, qual = assemble_sequence(spec, CDNA_WILBURN_V1, rng=rng)
    assert _SSP in seq
    assert "A" * 22 in seq
    assert _PRIMER3 in seq
    # Order: SSP comes before polyA, polyA before primer3.
    assert seq.index(_SSP) < seq.index("A" * 22) < seq.index(_PRIMER3)
    # Quality string matches sequence length.
    assert len(qual) == len(seq)


def test_assemble_missing_barcode_omits_barcode_segment() -> None:
    """Missing-Barcode spec emits SSP+T+polyA+primer3 but no barcode
    region between polyA and primer3."""
    rng = np.random.default_rng(0)
    spec = ReadSpec(
        read_id="mb",
        expected_status=ReadStatus.MISSING_BARCODE,
        orientation="+",
        transcript_length=300,
        transcript_id="mb",
        polyA_length=22,
        polyA_artifact="clean",
        barcode_index=None,
        include_ssp=True,
        include_primer3=True,
        include_polyA=True,
        include_barcode=False,
    )
    seq, _ = assemble_sequence(spec, CDNA_WILBURN_V1, rng=rng)
    polya_end = seq.index("A" * 22) + 22
    primer3_start = seq.index(_PRIMER3)
    # The window between polyA end and primer3 start should be exactly
    # zero — no barcode segment was emitted.
    assert primer3_start == polya_end


def test_assemble_five_prime_only_omits_primer3_and_barcode() -> None:
    rng = np.random.default_rng(0)
    spec = ReadSpec(
        read_id="fp",
        expected_status=ReadStatus.FIVE_PRIME_ONLY,
        orientation="+",
        transcript_length=300,
        transcript_id="fp",
        polyA_length=22,
        polyA_artifact="clean",
        barcode_index=None,
        include_ssp=True,
        include_primer3=False,
        include_polyA=True,
        include_barcode=False,
    )
    seq, _ = assemble_sequence(spec, CDNA_WILBURN_V1, rng=rng)
    assert _SSP in seq
    assert _PRIMER3 not in seq
    assert "A" * 22 in seq


def test_assemble_polyA_artifact_expanded_runs_long() -> None:
    """``expanded`` artifact yields a polyA run longer than nominal."""
    rng = np.random.default_rng(0)
    spec = _bare_complete_spec()
    spec_expanded = ReadSpec(
        **{**spec.__dict__, "polyA_artifact": "expanded"}
    )
    seq, _ = assemble_sequence(spec_expanded, CDNA_WILBURN_V1, rng=rng)
    # Find the longest A-run.
    longest = max(
        (run for run in seq.split("A") if run == ""),
        default="",
        key=len,
    )
    # Longest A-run = transcript may have stretches; check that we have
    # at least nominal+8 = 30 consecutive A's somewhere.
    assert "A" * 30 in seq


def test_generate_stress_test_specs_meets_size_floor() -> None:
    specs = generate_stress_test_specs(CDNA_WILBURN_V1, n_per_category=5, seed=42)
    assert len(specs) >= 200


def test_generate_stress_test_specs_covers_all_statuses() -> None:
    specs = generate_stress_test_specs(CDNA_WILBURN_V1, n_per_category=5, seed=42)
    statuses = {s.expected_status for s in specs}
    expected = {
        ReadStatus.COMPLETE,
        ReadStatus.THREE_PRIME_ONLY,
        ReadStatus.FIVE_PRIME_ONLY,
        ReadStatus.MISSING_BARCODE,
        ReadStatus.UNKNOWN,
        ReadStatus.COMPLEX,
    }
    assert expected.issubset(statuses)


def test_generate_stress_test_specs_deterministic() -> None:
    a = generate_stress_test_specs(CDNA_WILBURN_V1, n_per_category=3, seed=42)
    b = generate_stress_test_specs(CDNA_WILBURN_V1, n_per_category=3, seed=42)
    assert [s.read_id for s in a] == [s.read_id for s in b]
    assert [s.barcode_index for s in a] == [s.barcode_index for s in b]


# ──────────────────────────────────────────────────────────────────────
# Tier (a) — simulate_panel SAM + parquet file emission
# ──────────────────────────────────────────────────────────────────────


def test_simulate_panel_writes_sam_and_parquet(tmp_path: Path) -> None:
    specs = generate_stress_test_specs(CDNA_WILBURN_V1, n_per_category=2, seed=7)
    sam_path = tmp_path / "sim.sam"
    gt_path = tmp_path / "ground_truth.parquet"
    simulate_panel(specs, CDNA_WILBURN_V1, sam_path=sam_path, ground_truth_path=gt_path)
    assert sam_path.exists()
    assert gt_path.exists()

    gt = pq.read_table(gt_path)
    assert gt.schema.equals(GROUND_TRUTH_TABLE)
    assert gt.num_rows == len(specs)

    # SAM file: one header line per @, plus one body line per spec.
    body_lines = [
        ln for ln in sam_path.read_text().splitlines() if ln and not ln.startswith("@")
    ]
    assert len(body_lines) == len(specs)


# ──────────────────────────────────────────────────────────────────────
# Tier (b) — end-to-end: simulator → demux pipeline round-trip
# ──────────────────────────────────────────────────────────────────────


def test_simulator_to_demux_clean_round_trip(tmp_path: Path) -> None:
    """Deterministic-clean specs (zero mutations, in-range polyA, status
    in {Complete, 3' Only, 5' Only, Missing Barcode}) must round-trip
    with 100% status agreement against ground truth.

    This is the regression fixture for the future ProbabilisticScorer:
    if the hard scorer can't get 100% on the clean subset, the test
    setup itself is wrong.
    """
    from constellation.sequencing.transcriptome.stages import run_demux_pipeline

    all_specs = generate_stress_test_specs(
        CDNA_WILBURN_V1, n_per_category=3, seed=123
    )
    clean = [s for s in all_specs if is_deterministic_clean(s)]
    assert len(clean) >= 30, f"clean subset too small ({len(clean)})"

    sim_dir = tmp_path / "simulator"
    sam_path = sim_dir / "synthetic.sam"
    gt_path = sim_dir / "ground_truth.parquet"
    simulate_panel(clean, CDNA_WILBURN_V1, sam_path=sam_path, ground_truth_path=gt_path)

    n_barcodes = len(CDNA_WILBURN_V1.layout[3].barcodes)
    samples = Samples.from_records(
        samples=[
            {
                "sample_id": i + 1,
                "sample_name": f"BC{i + 1:02d}",
                "description": None,
            }
            for i in range(n_barcodes)
        ],
        edges=[
            {"sample_id": i + 1, "acquisition_id": 1, "barcode_id": i}
            for i in range(n_barcodes)
        ],
    )

    out_dir = tmp_path / "demux"
    run_demux_pipeline(
        sam_path,
        library_design="cdna_wilburn_v1",
        samples=samples,
        acquisition_id=1,
        output_dir=out_dir,
        batch_size=1000,
        n_workers=1,
    )

    demux_table = pq.read_table(out_dir / "read_demux.parquet")
    rows = demux_table.to_pylist()
    by_id = {r["read_id"]: r for r in rows}

    # Status agreement
    gt = pq.read_table(gt_path).to_pylist()
    mismatches: list[tuple[str, str, str]] = []
    for g in gt:
        rid = g["read_id"]
        actual = by_id.get(rid)
        if actual is None:
            mismatches.append((rid, g["expected_status"], "MISSING"))
            continue
        if actual["status"] != g["expected_status"]:
            mismatches.append(
                (rid, g["expected_status"], actual["status"])
            )
    assert not mismatches, (
        f"status mismatches on {len(mismatches)} clean reads (showing first 5): "
        f"{mismatches[:5]}"
    )

    # Barcode agreement on Complete + 3'-Only specs (these are the
    # statuses that emit a barcode_id). Read partitioned segments via
    # the parquet dataset API since shards are split per-batch.
    import pyarrow.dataset as ds

    seg_ds = ds.dataset(str(out_dir / "read_segments"), format="parquet")
    seg_table = seg_ds.to_table()
    seg_by_read: dict[str, int | None] = {}
    for r in seg_table.to_pylist():
        if r["segment_kind"] == "barcode":
            seg_by_read[r["read_id"]] = r["barcode_id"]
    bc_mismatches: list[tuple[str, int | None, int | None]] = []
    for g in gt:
        if g["expected_status"] not in ("Complete", "3' Only"):
            continue
        rid = g["read_id"]
        actual_bc = seg_by_read.get(rid)
        expected_bc = g["expected_barcode_index"]
        if actual_bc != expected_bc:
            bc_mismatches.append((rid, expected_bc, actual_bc))
    assert not bc_mismatches, (
        f"barcode_id mismatches on {len(bc_mismatches)} reads "
        f"(showing first 5): {bc_mismatches[:5]}"
    )
