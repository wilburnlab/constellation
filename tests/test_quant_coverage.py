"""Tests for ``constellation.sequencing.quant.coverage.build_pileup``.

Sweep-line is the algorithmically delicate piece here — exact-position
ties, depth transitions through zero, and the splice-aware "depth = 0
in introns" property all need explicit checks.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.sequencing.quant.coverage import build_pileup
from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_TABLE,
)
from constellation.sequencing.schemas.quant import COVERAGE_TABLE
from constellation.sequencing.schemas.reference import CONTIG_TABLE


_UNSTRATIFIED = -1


def _alignment(**overrides) -> dict:
    base = {
        "alignment_id": 0,
        "read_id": "r0",
        "acquisition_id": 1,
        "ref_name": "chr1",
        "ref_start": 0,
        "ref_end": 100,
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


def _block(**overrides) -> dict:
    base = {
        "alignment_id": 0,
        "block_index": 0,
        "ref_start": 0,
        "ref_end": 100,
        "query_start": 0,
        "query_end": 100,
        "n_match": None,
        "n_mismatch": None,
        "n_insert": 0,
        "n_delete": 0,
    }
    base.update(overrides)
    return base


def _contigs() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"contig_id": 1, "name": "chr1", "length": 100_000,
             "topology": None, "circular": None},
            {"contig_id": 2, "name": "chr2", "length": 50_000,
             "topology": None, "circular": None},
        ],
        schema=CONTIG_TABLE,
    )


def _alignments(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_TABLE)


def _blocks(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_BLOCK_TABLE)


def _by_contig_sample(table: pa.Table) -> dict[tuple[int, int], list[dict]]:
    out: dict[tuple[int, int], list[dict]] = {}
    for row in table.to_pylist():
        out.setdefault((row["contig_id"], row["sample_id"]), []).append(row)
    return out


# ──────────────────────────────────────────────────────────────────────


def test_single_block_yields_one_rle_row() -> None:
    al = _alignments([_alignment(alignment_id=1)])
    blocks = _blocks(
        [_block(alignment_id=1, ref_start=100, ref_end=200)]
    )
    out = build_pileup(blocks, al, _contigs())
    assert out.num_rows == 1
    row = out.to_pylist()[0]
    assert row["contig_id"] == 1
    assert row["sample_id"] == _UNSTRATIFIED
    assert row["start"] == 100 and row["end"] == 200
    assert row["depth"] == 1


def test_two_overlapping_reads_produce_three_segments() -> None:
    """Reads at [100,200) + [150,250) → 3 RLE rows: depth 1 / 2 / 1."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=100, ref_end=200),
            _block(alignment_id=2, ref_start=150, ref_end=250),
        ]
    )
    out = build_pileup(blocks, al, _contigs())
    rows = out.to_pylist()
    assert len(rows) == 3
    assert rows[0] == {
        "contig_id": 1, "sample_id": _UNSTRATIFIED,
        "start": 100, "end": 150, "depth": 1,
    }
    assert rows[1] == {
        "contig_id": 1, "sample_id": _UNSTRATIFIED,
        "start": 150, "end": 200, "depth": 2,
    }
    assert rows[2] == {
        "contig_id": 1, "sample_id": _UNSTRATIFIED,
        "start": 200, "end": 250, "depth": 1,
    }


def test_intron_gap_produces_zero_depth_between_blocks() -> None:
    """A spliced read covers only its blocks — depth in the intron is
    0 (and zero-depth segments are NOT emitted; callers compute the
    gap from adjacent rows).
    """
    al = _alignments([_alignment(alignment_id=1)])
    blocks = _blocks(
        [
            _block(alignment_id=1, block_index=0, ref_start=100, ref_end=200),
            _block(alignment_id=1, block_index=1, ref_start=500, ref_end=600),
        ]
    )
    out = build_pileup(blocks, al, _contigs())
    rows = out.to_pylist()
    # Two non-zero RLE rows, the intron [200, 500) is implicit
    assert len(rows) == 2
    assert rows[0]["start"] == 100 and rows[0]["end"] == 200
    assert rows[1]["start"] == 500 and rows[1]["end"] == 600


def test_exact_endpoint_tie_does_not_drop_a_segment() -> None:
    """One read ends exactly where another begins → one continuous
    RLE row at depth 1 (the half-open intervals don't overlap).
    """
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=100, ref_end=200),
            _block(alignment_id=2, ref_start=200, ref_end=300),
        ]
    )
    out = build_pileup(blocks, al, _contigs())
    rows = out.to_pylist()
    # Half-open: [100,200) and [200,300) don't overlap; ties cancel.
    # We get a single depth-1 row covering [100, 300) since the +1/-1
    # at pos 200 sum to zero.
    assert len(rows) == 1
    assert rows[0]["start"] == 100 and rows[0]["end"] == 300
    assert rows[0]["depth"] == 1


def test_three_overlap_layers() -> None:
    """Three reads with nested overlap: [100,300), [150,250), [180,220)."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
            _alignment(alignment_id=3, read_id="rC"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=100, ref_end=300),
            _block(alignment_id=2, ref_start=150, ref_end=250),
            _block(alignment_id=3, ref_start=180, ref_end=220),
        ]
    )
    out = build_pileup(blocks, al, _contigs())
    rows = out.to_pylist()
    # Expected RLE: [100,150)→1, [150,180)→2, [180,220)→3, [220,250)→2, [250,300)→1
    assert len(rows) == 5
    assert [(r["start"], r["end"], r["depth"]) for r in rows] == [
        (100, 150, 1),
        (150, 180, 2),
        (180, 220, 3),
        (220, 250, 2),
        (250, 300, 1),
    ]


def test_per_sample_stratification() -> None:
    """When a read_to_sample mapping is provided, output is per
    (contig, sample) partition.
    """
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=100, ref_end=200),
            _block(alignment_id=2, ref_start=100, ref_end=200),
        ]
    )
    out = build_pileup(
        blocks, al, _contigs(),
        read_to_sample={"rA": 10, "rB": 11},
    )
    rows = out.to_pylist()
    # Two partitions, each one RLE row at depth 1.
    assert len(rows) == 2
    by_sid = {r["sample_id"]: r for r in rows}
    assert by_sid[10]["depth"] == 1
    assert by_sid[11]["depth"] == 1


def test_unmapped_sample_id_drops_read() -> None:
    """When read_to_sample is provided but a read isn't in it, that
    read's blocks contribute zero to depth.
    """
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB_unknown"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=100, ref_end=200),
            _block(alignment_id=2, ref_start=100, ref_end=200),
        ]
    )
    out = build_pileup(
        blocks, al, _contigs(),
        read_to_sample={"rA": 10},  # rB_unknown deliberately absent
    )
    rows = out.to_pylist()
    assert len(rows) == 1
    assert rows[0]["sample_id"] == 10
    assert rows[0]["depth"] == 1


def test_secondary_alignments_excluded_by_default() -> None:
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB", is_secondary=True),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=100, ref_end=200),
            _block(alignment_id=2, ref_start=150, ref_end=250),
        ]
    )
    out = build_pileup(blocks, al, _contigs())
    rows = out.to_pylist()
    assert len(rows) == 1
    assert rows[0]["start"] == 100 and rows[0]["end"] == 200
    assert rows[0]["depth"] == 1


def test_per_contig_partitioning() -> None:
    """Reads on chr1 and chr2 produce separate per-contig RLE runs."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA", ref_name="chr1"),
            _alignment(alignment_id=2, read_id="rB", ref_name="chr2"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=100, ref_end=200),
            _block(alignment_id=2, ref_start=300, ref_end=400),
        ]
    )
    out = build_pileup(blocks, al, _contigs())
    rows = out.to_pylist()
    assert len(rows) == 2
    by_contig = {r["contig_id"]: r for r in rows}
    assert by_contig[1]["start"] == 100 and by_contig[1]["end"] == 200
    assert by_contig[2]["start"] == 300 and by_contig[2]["end"] == 400


def test_alignment_to_unknown_contig_is_dropped() -> None:
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA", ref_name="chrX_unknown"),
            _alignment(alignment_id=2, read_id="rB", ref_name="chr1"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=100, ref_end=200),
            _block(alignment_id=2, ref_start=300, ref_end=400),
        ]
    )
    out = build_pileup(blocks, al, _contigs())
    rows = out.to_pylist()
    assert len(rows) == 1
    assert rows[0]["contig_id"] == 1


def test_empty_inputs_return_empty_schema() -> None:
    out = build_pileup(
        ALIGNMENT_BLOCK_TABLE.empty_table(),
        ALIGNMENT_TABLE.empty_table(),
        _contigs(),
    )
    assert out.schema.equals(COVERAGE_TABLE)
    assert out.num_rows == 0


def test_zero_length_block_is_skipped() -> None:
    """A block with ref_start == ref_end shouldn't emit an event pair."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=100, ref_end=100),  # degenerate
            _block(alignment_id=2, ref_start=200, ref_end=300),
        ]
    )
    out = build_pileup(blocks, al, _contigs())
    rows = out.to_pylist()
    assert len(rows) == 1
    assert rows[0]["start"] == 200 and rows[0]["end"] == 300


def test_output_sorted_by_contig_sample_start() -> None:
    """Output ordering: ``(contig_id, sample_id, start)`` ascending."""
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA", ref_name="chr2"),
            _alignment(alignment_id=2, read_id="rB", ref_name="chr1"),
            _alignment(alignment_id=3, read_id="rC", ref_name="chr1"),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=100, ref_end=200),
            _block(alignment_id=2, ref_start=300, ref_end=400),
            _block(alignment_id=3, ref_start=100, ref_end=200),
        ]
    )
    out = build_pileup(blocks, al, _contigs())
    rows = out.to_pylist()
    # chr1 first (contig_id=1), then chr2 (contig_id=2); within chr1,
    # the [100,200) row comes before [300,400).
    assert [(r["contig_id"], r["start"]) for r in rows] == [
        (1, 100), (1, 300), (2, 100),
    ]
