"""Tests for ``extract_alignment_blocks`` — the worker-layer
orchestrator that lifts (filtered alignments + cs tags) into
``ALIGNMENT_BLOCK_TABLE`` rows.

The CIGAR + cs:long parsers themselves are covered in
``test_align_cigar.py``. This file covers:
    - cs:long preferred when present (n_match / n_mismatch populated)
    - CIGAR fallback when cs is absent (n_match / n_mismatch null)
    - soft clip → query_start advance
    - empty inputs → empty output, schema-shaped
    - unmapped / "*" CIGAR → no rows
"""

from __future__ import annotations

import pyarrow as pa

from constellation.sequencing.quant import extract_alignment_blocks
from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_TABLE,
    ALIGNMENT_TAG_TABLE,
)


def _alignment(**overrides) -> dict:
    base = {
        "alignment_id": 0,
        "read_id": "r0",
        "acquisition_id": 1,
        "ref_name": "chr1",
        "ref_start": 1000,
        "ref_end": 1100,
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


def _alignments(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_TABLE)


def _tags(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_TAG_TABLE)


def test_empty_alignments_returns_empty_block_table() -> None:
    out = extract_alignment_blocks(
        ALIGNMENT_TABLE.empty_table(), ALIGNMENT_TAG_TABLE.empty_table()
    )
    assert out.schema.equals(ALIGNMENT_BLOCK_TABLE)
    assert out.num_rows == 0


def test_cigar_only_path_leaves_match_counts_null() -> None:
    """When no cs tag is available, CIGAR-derived blocks have null
    n_match / n_mismatch.
    """
    al = _alignments(
        [
            _alignment(
                alignment_id=1,
                ref_start=1000,
                ref_end=1300,
                cigar_string="100M500N200M",
            )
        ]
    )
    out = extract_alignment_blocks(al, ALIGNMENT_TAG_TABLE.empty_table())
    assert out.num_rows == 2
    rows = out.to_pylist()
    assert rows[0]["n_match"] is None and rows[0]["n_mismatch"] is None
    assert rows[1]["n_match"] is None and rows[1]["n_mismatch"] is None
    assert rows[0]["ref_start"] == 1000 and rows[0]["ref_end"] == 1100
    assert rows[1]["ref_start"] == 1600 and rows[1]["ref_end"] == 1800


def test_cs_long_path_populates_match_counts() -> None:
    """When cs is present, n_match / n_mismatch are populated."""
    al = _alignments(
        [
            _alignment(
                alignment_id=1,
                ref_start=1000,
                ref_end=1109,
                cigar_string="100M500N9M",
            )
        ]
    )
    tags = _tags(
        [
            {
                "alignment_id": 1,
                "tag": "cs",
                "type": "Z",
                "value": ":100~gt500ag=AAAA*at=AAAA",
            }
        ]
    )
    out = extract_alignment_blocks(al, tags)
    rows = out.to_pylist()
    assert len(rows) == 2
    assert rows[0]["n_match"] == 100 and rows[0]["n_mismatch"] == 0
    assert rows[1]["n_match"] == 8 and rows[1]["n_mismatch"] == 1


def test_soft_clip_advances_query_start() -> None:
    """Leading 50S → blocks start at query_start=50."""
    al = _alignments(
        [
            _alignment(
                alignment_id=1,
                ref_start=1000,
                ref_end=1100,
                cigar_string="50S100M30S",
            )
        ]
    )
    out = extract_alignment_blocks(al, ALIGNMENT_TAG_TABLE.empty_table())
    rows = out.to_pylist()
    assert len(rows) == 1
    assert rows[0]["query_start"] == 50
    assert rows[0]["query_end"] == 150


def test_unmapped_cigar_produces_no_rows() -> None:
    """``"*"`` CIGAR (unmapped) and empty CIGAR are dropped silently."""
    al = _alignments(
        [
            _alignment(alignment_id=1, cigar_string="*"),
            _alignment(alignment_id=2, cigar_string=""),
        ]
    )
    out = extract_alignment_blocks(al, ALIGNMENT_TAG_TABLE.empty_table())
    assert out.num_rows == 0


def test_mixed_alignments_resolve_per_alignment_id() -> None:
    """Two alignments, one with cs and one without — each gets its
    matching path.
    """
    al = _alignments(
        [
            _alignment(
                alignment_id=10,
                ref_start=1000,
                ref_end=1100,
                cigar_string="100M",
            ),
            _alignment(
                alignment_id=20,
                ref_start=2000,
                ref_end=2100,
                cigar_string="100M",
            ),
        ]
    )
    tags = _tags(
        [
            # Only alignment 20 has cs
            {
                "alignment_id": 20,
                "tag": "cs",
                "type": "Z",
                "value": ":100",
            }
        ]
    )
    out = extract_alignment_blocks(al, tags)
    rows = out.to_pylist()
    assert len(rows) == 2
    by_aid = {r["alignment_id"]: r for r in rows}
    assert by_aid[10]["n_match"] is None       # CIGAR fallback
    assert by_aid[20]["n_match"] == 100         # cs:long path


def test_block_index_is_monotonic_within_alignment() -> None:
    al = _alignments(
        [
            _alignment(
                alignment_id=1,
                ref_start=0,
                ref_end=350,
                cigar_string="100M50N100M50N100M",
            )
        ]
    )
    out = extract_alignment_blocks(al, ALIGNMENT_TAG_TABLE.empty_table())
    block_indexes = out.column("block_index").to_pylist()
    assert block_indexes == [0, 1, 2]


def test_unrelated_tags_in_table_do_not_match_cs() -> None:
    """Tag rows where ``tag != 'cs'`` must not be misread as cs."""
    al = _alignments(
        [
            _alignment(
                alignment_id=1,
                ref_start=0,
                ref_end=100,
                cigar_string="100M",
            )
        ]
    )
    tags = _tags(
        [
            {"alignment_id": 1, "tag": "MD", "type": "Z", "value": "100"},
            {"alignment_id": 1, "tag": "SA", "type": "Z", "value": "junk"},
        ]
    )
    out = extract_alignment_blocks(al, tags)
    rows = out.to_pylist()
    assert len(rows) == 1
    # No cs available — n_match must remain null.
    assert rows[0]["n_match"] is None
