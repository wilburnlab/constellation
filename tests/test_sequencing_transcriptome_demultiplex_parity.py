"""Parity gate — `constellation transcriptome demultiplex` against
NanoporeAnalysis ``_fixed1`` baseline.

Runs the full pipeline (SAM → segments → demux → ORF → count matrix)
on the lab's ``test_simplex.sam`` fixture and asserts the output
matches the supplied baseline:

    1. Per-read ``sample_id`` agreement: ≥99.7% match. Divergences
       are NanoporeAnalysis bugs we deliberately don't replicate
       (see ``transcriptome.demux._pick_orientation`` for the
       Unknown-Fragment bug we corrected, and the empty-subject
       short-circuit in ``align.locate.locate_substring``).
    2. Per-read ``status`` agreement: ≥99.4%. The remaining ~20 reads
       are reads where the upstream Unknown-Fragment bug was hiding
       a real annotation; constellation surfaces it (e.g. baseline's
       ``"Unknown Fragment"`` becomes our ``"5' Only Fragment"``).
    3. Count matrix: every ``(protein_sequence, sample_name)`` cell
       in the baseline TSV has the matching count in our output.
       This is the gate that actually matters for downstream analysis;
       100% parity is required.
    4. Protein FASTA: same set of protein sequences.

Both deviations are strict accuracy improvements over the baseline:
the baseline pipeline silently drops real annotations and silently
fabricates spurious ``Complete + BC01`` assignments on edge-case
reads. constellation does neither.

The test is gated on the lab fixture being present; CI without that
fixture skips the slow gate. Marked ``slow`` so it can be opted out
of via ``pytest -m 'not slow'``.
"""

from __future__ import annotations

import pathlib

import pyarrow.parquet as pq
import pytest

from constellation.sequencing.readers.sam_bam import SamReader
from constellation.sequencing.samples import Samples
from constellation.sequencing.transcriptome.demux import (
    locate_segments,
    resolve_demux,
)
from constellation.sequencing.transcriptome.orf import predict_orfs
from constellation.sequencing.transcriptome.panels import CDNA_WILBURN_V1
from constellation.sequencing.transcriptome.quant import (
    build_protein_count_matrix,
)


_FIXTURE_DIR = pathlib.Path(
    "/home/dbwilburn/WilburnLab/Coding/Constellation/test_nanopore_data/"
    "pod5_cDNA/Constellation_test_data"
)
_FIXTURE_SAM = _FIXTURE_DIR / "test_simplex.sam"
_BASELINE_PARQUET = _FIXTURE_DIR / "test_simplex_0_fixed1.parquet"
_BASELINE_TSV = _FIXTURE_DIR / "qJS00x_fixed1_protein-counts.tab"
_BASELINE_FASTA = _FIXTURE_DIR / "qJS00x_fixed1_proteins.fasta"


pytestmark = pytest.mark.skipif(
    not _FIXTURE_SAM.exists(),
    reason="lab test fixture not present — parity gate is local-only",
)


@pytest.fixture(scope="module")
def pipeline_outputs():
    """Run the full S1 pipeline once and yield its outputs."""
    samples = Samples.from_records(
        samples=[
            {"sample_id": i + 1, "sample_name": f"test_simplex_BC{i + 1:02d}", "description": None}
            for i in range(12)
        ],
        edges=[
            {"sample_id": i + 1, "acquisition_id": 1, "barcode_id": i}
            for i in range(12)
        ],
    )
    reads = SamReader().read(_FIXTURE_SAM, acquisition_id=1).primary
    segments, demux, results = locate_segments(reads, CDNA_WILBURN_V1)
    demux = resolve_demux(demux, segments, samples=samples, acquisition_id=1)
    orfs = predict_orfs(results)
    quant, fasta_records, tsv_text = build_protein_count_matrix(
        demux, orfs, samples
    )
    return {
        "samples": samples,
        "reads": reads,
        "segments": segments,
        "demux": demux,
        "orfs": orfs,
        "quant": quant,
        "fasta_records": fasta_records,
        "tsv_text": tsv_text,
    }


@pytest.fixture(scope="module")
def baseline():
    """Load the NanoporeAnalysis _fixed1 baseline outputs."""
    parquet = pq.read_table(_BASELINE_PARQUET)
    tsv_text = _BASELINE_TSV.read_text()
    fasta_text = _BASELINE_FASTA.read_text()
    return {
        "parquet": parquet,
        "tsv_text": tsv_text,
        "fasta_text": fasta_text,
    }


# ──────────────────────────────────────────────────────────────────────
# Per-read parity (status + sample_id)
# ──────────────────────────────────────────────────────────────────────


def _baseline_status_map(parquet) -> dict[str, str]:
    return {r["QNAME"]: r["cDNA status"] for r in parquet.to_pylist()}


def _baseline_umi_map(parquet) -> dict[str, str | None]:
    return {r["QNAME"]: r["Best UMI"] for r in parquet.to_pylist()}


def _mine_status_map(demux) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in demux.to_pylist():
        s = r["status"]
        if r["is_fragment"]:
            s += " Fragment"
        out[r["read_id"]] = s
    return out


def _mine_umi_map(segments, construct) -> dict[str, str | None]:
    barcodes = construct.layout[3].barcodes
    out: dict[str, str | None] = {}
    for r in segments.to_pylist():
        if r["segment_kind"] == "barcode" and r["barcode_id"] is not None:
            out[r["read_id"]] = barcodes[r["barcode_id"]].name
    return out


@pytest.mark.slow
def test_status_agreement_floor(pipeline_outputs, baseline):
    """≥99.4% per-read status agreement.

    Divergences (~20 reads on test_simplex.sam) are reads where
    NanoporeAnalysis's Unknown-Fragment string-equality bug discarded
    a real forward-orientation annotation; constellation surfaces it.
    See ``transcriptome.demux._pick_orientation`` for the rationale.
    """
    base = _baseline_status_map(baseline["parquet"])
    mine = _mine_status_map(pipeline_outputs["demux"])
    agree = sum(1 for k in base if mine.get(k) == base[k])
    rate = agree / len(base)
    assert rate >= 0.994, (
        f"status agreement {rate:.4f} below 99.4% floor "
        f"({agree}/{len(base)})"
    )


@pytest.mark.slow
def test_barcode_agreement_floor(pipeline_outputs, baseline):
    """≥99.7% per-read barcode agreement."""
    base = _baseline_umi_map(baseline["parquet"])
    mine = _mine_umi_map(pipeline_outputs["segments"], CDNA_WILBURN_V1)
    # Compare on baseline's read set, treating missing-from-mine as None.
    agree = sum(1 for k in base if mine.get(k) == base[k])
    rate = agree / len(base)
    assert rate >= 0.997, (
        f"barcode agreement {rate:.4f} below 99.7% floor "
        f"({agree}/{len(base)})"
    )


# ──────────────────────────────────────────────────────────────────────
# Count-matrix parity (the gate that actually matters)
# ──────────────────────────────────────────────────────────────────────


def _parse_counts_tsv(text: str) -> dict[str, dict[str, float]]:
    """Parse a NanoporeAnalysis-format protein-counts TSV into a
    ``protein_sequence -> {sample_name: count}`` dict (zero-cells
    omitted)."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    header = lines[0].split("\t")
    sample_cols = header[2:-1]
    out: dict[str, dict[str, float]] = {}
    for line in lines[1:]:
        cells = line.split("\t")
        seq = cells[-1]
        per_sample = {}
        for i, sample_name in enumerate(sample_cols):
            cell = cells[2 + i]
            count = float(cell)
            if count != 0.0:
                per_sample[sample_name] = count
        out[seq] = per_sample
    return out


@pytest.mark.slow
def test_count_matrix_protein_set_matches(pipeline_outputs, baseline):
    """Same set of protein sequences in mine + baseline."""
    mine = _parse_counts_tsv(pipeline_outputs["tsv_text"])
    base = _parse_counts_tsv(baseline["tsv_text"])
    assert set(mine.keys()) == set(base.keys()), (
        f"protein-sequence sets diverge: "
        f"mine_only={list(set(mine) - set(base))[:3]}, "
        f"base_only={list(set(base) - set(mine))[:3]}"
    )


@pytest.mark.slow
def test_count_matrix_per_cell_parity(pipeline_outputs, baseline):
    """Every (protein, sample) → count cell matches exactly."""
    mine = _parse_counts_tsv(pipeline_outputs["tsv_text"])
    base = _parse_counts_tsv(baseline["tsv_text"])
    assert set(mine.keys()) == set(base.keys())
    for seq in base:
        assert mine[seq] == base[seq], (
            f"count-cell mismatch on {seq[:30]!r}: "
            f"mine={mine[seq]}, base={base[seq]}"
        )


@pytest.mark.slow
def test_protein_fasta_set_equality(pipeline_outputs, baseline):
    """Mine and baseline FASTA contain the same set of (label-free)
    sequences — labels (``P0..P4``) may differ on count ties but
    sequences must match."""
    mine_seqs = {r.sequence for r in pipeline_outputs["fasta_records"]}
    base_seqs = set()
    for line in baseline["fasta_text"].split("\n"):
        line = line.strip()
        if line and not line.startswith(">"):
            base_seqs.add(line)
    assert mine_seqs == base_seqs, (
        f"protein-FASTA sequence sets diverge: "
        f"mine_only={list(mine_seqs - base_seqs)[:1]}, "
        f"base_only={list(base_seqs - mine_seqs)[:1]}"
    )
