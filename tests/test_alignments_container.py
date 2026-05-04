"""Tests for ``constellation.sequencing.alignments``.

Hand-crafted small fixtures (no pysam required); cover container
construction, validation, filter step semantics, ParquetDir round-trip,
and CIGAR-derived aligned-fraction.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.sequencing.acquisitions import Acquisitions
from constellation.sequencing.alignments import (
    Alignments,
    load_alignments,
    save_alignments,
)
from constellation.sequencing.alignments.alignments import (
    _aligned_fraction_array,
)
from constellation.sequencing.reference.reference import GenomeReference


def _alignment_row(**overrides):
    base = {
        "alignment_id": 0,
        "read_id": "r0",
        "acquisition_id": 1,
        "ref_name": "chr1",
        "ref_start": 100,
        "ref_end": 200,
        "strand": "+",
        "mapq": 60,
        "flag": 0,
        "cigar_string": "100M",
        "nm_tag": None,
        "as_tag": None,
        "read_group": None,
        "is_secondary": False,
        "is_supplementary": False,
    }
    base.update(overrides)
    return base


def _make_alignments(rows=None, tags=None, **kwargs):
    if rows is None:
        rows = [_alignment_row()]
    table = pa.Table.from_pylist(rows)
    tags_table = pa.Table.from_pylist(tags or [])
    return Alignments(alignments=table, tags=tags_table, **kwargs)


def test_construction_validates_pk_uniqueness() -> None:
    rows = [_alignment_row(alignment_id=0), _alignment_row(alignment_id=0)]
    with pytest.raises(ValueError, match="alignment_id contains duplicate"):
        _make_alignments(rows)


def test_tags_fk_closure() -> None:
    rows = [_alignment_row(alignment_id=0)]
    bad_tags = [{"alignment_id": 999, "tag": "MD", "type": "Z", "value": "100"}]
    with pytest.raises(ValueError, match="absent from ALIGNMENT_TABLE"):
        _make_alignments(rows, tags=bad_tags)


def test_validate_against_genome_ok() -> None:
    a = _make_alignments()
    contigs = pa.Table.from_pylist(
        [
            {"contig_id": 0, "name": "chr1", "length": 1000, "topology": None, "circular": None},
        ]
    )
    sequences = pa.Table.from_pylist([{"contig_id": 0, "sequence": "A" * 1000}])
    genome = GenomeReference(contigs=contigs, sequences=sequences)
    a.validate_against(genome)


def test_validate_against_genome_rejects_unknown_contig() -> None:
    rows = [_alignment_row(ref_name="chrZ")]
    a = _make_alignments(rows)
    contigs = pa.Table.from_pylist(
        [
            {"contig_id": 0, "name": "chr1", "length": 1000, "topology": None, "circular": None},
        ]
    )
    sequences = pa.Table.from_pylist([{"contig_id": 0, "sequence": "A" * 1000}])
    genome = GenomeReference(contigs=contigs, sequences=sequences)
    with pytest.raises(ValueError, match="absent from GenomeReference"):
        a.validate_against(genome)


def test_primary_filters_secondaries_and_supplementaries() -> None:
    rows = [
        _alignment_row(alignment_id=0),
        _alignment_row(alignment_id=1, is_secondary=True),
        _alignment_row(alignment_id=2, is_supplementary=True),
    ]
    a = _make_alignments(rows)
    assert a.n_alignments == 3
    assert a.primary().n_alignments == 1


def test_filter_records_per_stage_audit() -> None:
    rows = [
        _alignment_row(alignment_id=0, mapq=60),
        _alignment_row(alignment_id=1, mapq=10),
        _alignment_row(alignment_id=2, mapq=60, ref_start=100, ref_end=150),
    ]
    a = _make_alignments(rows)
    out = a.filter(min_mapq=30, min_length=80)
    audit = out.metadata_extras["filter_steps"]
    # Fixed stage order in Alignments.filter: primary_only, min_length,
    # min_mapq, min_aligned_fraction (independent of kwarg order).
    assert audit == [
        {"stage": "min_length>=80", "kept": 2, "dropped": 1},
        {"stage": "min_mapq>=30", "kept": 1, "dropped": 1},
    ]
    assert out.n_alignments == 1


def test_filter_aligned_fraction_drops_high_indel_reads() -> None:
    rows = [
        # 100M → 100/100 = 1.0
        _alignment_row(alignment_id=0, cigar_string="100M"),
        # 50M50I → aligned_bp=50, read_length=100, fraction=0.5
        _alignment_row(alignment_id=1, cigar_string="50M50I"),
        # 80M20S → aligned_bp=80, read_length=100, fraction=0.8
        _alignment_row(alignment_id=2, cigar_string="80M20S"),
    ]
    a = _make_alignments(rows)
    out = a.filter(min_aligned_fraction=0.85)
    assert out.n_alignments == 1  # only the 100M passes


def test_aligned_fraction_handles_degenerate_cigars() -> None:
    rows = [
        _alignment_row(alignment_id=0, cigar_string="100M"),
        _alignment_row(alignment_id=1, cigar_string="*"),  # null fraction
        _alignment_row(alignment_id=2, cigar_string=""),  # null
    ]
    a = _make_alignments(rows)
    af = _aligned_fraction_array(a.alignments)
    py = af.to_pylist()
    assert py[0] == pytest.approx(1.0)
    assert py[1] is None
    assert py[2] is None


def test_filter_subsets_tags_to_surviving_alignments() -> None:
    rows = [
        _alignment_row(alignment_id=0, mapq=60),
        _alignment_row(alignment_id=1, mapq=10),
    ]
    tags = [
        {"alignment_id": 0, "tag": "MD", "type": "Z", "value": "100"},
        {"alignment_id": 1, "tag": "MD", "type": "Z", "value": "100"},
    ]
    a = _make_alignments(rows, tags=tags)
    out = a.filter(min_mapq=30)
    assert out.tags.num_rows == 1
    assert out.tags.column("alignment_id").to_pylist() == [0]


def test_parquet_dir_round_trip(tmp_path: Path) -> None:
    rows = [_alignment_row(alignment_id=0), _alignment_row(alignment_id=1)]
    tags = [{"alignment_id": 0, "tag": "MD", "type": "Z", "value": "100"}]
    a = _make_alignments(rows, tags=tags, metadata_extras={"engine": "test"})

    out = tmp_path / "aln"
    save_alignments(a, out)
    loaded = load_alignments(out)
    assert loaded.n_alignments == 2
    assert loaded.tags.num_rows == 1
    assert loaded.metadata_extras == {"engine": "test"}


def test_parquet_dir_round_trip_with_acquisitions(tmp_path: Path) -> None:
    acq_table = pa.Table.from_pylist(
        [
            {
                "acquisition_id": 1,
                "source_path": "/tmp/x.bam",
                "source_kind": "bam",
                "acquisition_datetime": None,
                "instrument_id": None,
                "flow_cell_id": None,
                "flow_cell_type": None,
                "sample_kit": None,
                "basecaller_model": None,
                "experiment_type": None,
            }
        ]
    )
    a = _make_alignments(acquisitions=Acquisitions(acq_table))
    out = tmp_path / "aln"
    save_alignments(a, out)
    loaded = load_alignments(out)
    assert loaded.acquisitions is not None
    assert loaded.acquisitions.ids == [1]


def test_acquisitions_fk_validation() -> None:
    rows = [_alignment_row(acquisition_id=99)]
    acq_table = pa.Table.from_pylist(
        [
            {
                "acquisition_id": 1,
                "source_path": "/tmp/x.bam",
                "source_kind": "bam",
                "acquisition_datetime": None,
                "instrument_id": None,
                "flow_cell_id": None,
                "flow_cell_type": None,
                "sample_kit": None,
                "basecaller_model": None,
                "experiment_type": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="unknown acquisition_ids"):
        _make_alignments(rows, acquisitions=Acquisitions(acq_table))
