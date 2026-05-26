"""Unit tests for the shared `_alignment_view` block-attach helpers.

The read_pileup kernel and (after PR 5) cluster_pileup's member-view
kernel both attach two nested-list columns to alignment-shaped tables:

- ``blocks``: list<struct{ref_start, ref_end, n_match, n_mismatch}>
  built by joining ALIGNMENT_BLOCK_TABLE rows in
- ``mismatch_positions``: list<int64> of per-base substitution
  positions, parsed from ALIGNMENT_CS_TABLE's ``cs_string`` column

Both helpers must:
  - emit the exact wire schema (so the kernel can ``cast`` to
    ``READ_PILEUP_VECTOR_SCHEMA`` without surprise)
  - return empty lists (not nulls) when the corresponding input is
    absent — the renderer's per-row loop expects uniform shapes
  - tolerate an empty alignment table without raising
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_CS_TABLE,
)
from constellation.viz.tracks._alignment_view import (
    BLOCKS_LIST_TYPE,
    MISMATCH_POSITIONS_TYPE,
    attach_blocks,
    attach_mismatch_positions,
)


def _alignments(rows: list[dict]) -> pa.Table:
    """Three-column slim alignments table — only what attach_blocks
    actually reads."""
    return pa.table(
        {
            "alignment_id": [r["alignment_id"] for r in rows],
            "ref_start": [r["ref_start"] for r in rows],
            "ref_end": [r["ref_end"] for r in rows],
        },
        schema=pa.schema(
            [
                pa.field("alignment_id", pa.int64(), nullable=False),
                pa.field("ref_start", pa.int64(), nullable=False),
                pa.field("ref_end", pa.int64(), nullable=False),
            ]
        ),
    )


def _blocks(rows: list[dict]) -> pa.Table:
    full_rows: list[dict] = []
    for r in rows:
        base = {
            "alignment_id": 0,
            "block_index": 0,
            "ref_start": 0,
            "ref_end": 0,
            "query_start": 0,
            "query_end": 0,
            "n_match": None,
            "n_mismatch": None,
            "n_insert": 0,
            "n_delete": 0,
        }
        base.update(r)
        full_rows.append(base)
    return pa.Table.from_pylist(full_rows, schema=ALIGNMENT_BLOCK_TABLE)


# ──────────────────────────────────────────────────────────────────────
# attach_blocks
# ──────────────────────────────────────────────────────────────────────


def test_attach_blocks_assembles_struct_list_per_alignment() -> None:
    """Each alignment's per-CIGAR block rows roll into a single
    list<struct> cell. Block order follows block_index ascending."""
    alignments = _alignments(
        [
            {"alignment_id": 1, "ref_start": 0, "ref_end": 300},
            {"alignment_id": 2, "ref_start": 500, "ref_end": 800},
        ]
    )
    blocks = _blocks(
        [
            {"alignment_id": 1, "block_index": 1, "ref_start": 200, "ref_end": 300},
            {"alignment_id": 1, "block_index": 0, "ref_start": 0, "ref_end": 100,
             "n_match": 95, "n_mismatch": 5},
            {"alignment_id": 2, "block_index": 0, "ref_start": 500, "ref_end": 800,
             "n_match": 300, "n_mismatch": 0},
        ]
    )
    out = attach_blocks(alignments, blocks)
    assert out.schema.field("blocks").type == BLOCKS_LIST_TYPE
    rows = out.to_pylist()
    # Alignment 1 got both its blocks, in block_index order.
    a1_blocks = rows[0]["blocks"]
    assert [(b["ref_start"], b["ref_end"]) for b in a1_blocks] == [
        (0, 100),
        (200, 300),
    ]
    assert a1_blocks[0]["n_match"] == 95
    assert a1_blocks[0]["n_mismatch"] == 5
    # Alignment 2 got its one block.
    a2_blocks = rows[1]["blocks"]
    assert [(b["ref_start"], b["ref_end"]) for b in a2_blocks] == [(500, 800)]


def test_attach_blocks_synthetic_fallback_when_no_block_row() -> None:
    """An alignment with no ALIGNMENT_BLOCK_TABLE row gets a single
    synthetic block spanning its own ref range, with null match/mismatch
    counts — keeps the renderer's per-row loop shape uniform."""
    alignments = _alignments(
        [{"alignment_id": 99, "ref_start": 10, "ref_end": 40}]
    )
    out = attach_blocks(alignments, ALIGNMENT_BLOCK_TABLE.empty_table())
    [row] = out.to_pylist()
    [block] = row["blocks"]
    assert (block["ref_start"], block["ref_end"]) == (10, 40)
    assert block["n_match"] is None
    assert block["n_mismatch"] is None


def test_attach_blocks_mixed_real_and_synthetic() -> None:
    """When some alignments have blocks and others don't, each path
    is taken independently."""
    alignments = _alignments(
        [
            {"alignment_id": 1, "ref_start": 0, "ref_end": 50},
            {"alignment_id": 2, "ref_start": 200, "ref_end": 250},
        ]
    )
    blocks = _blocks(
        [
            {"alignment_id": 2, "block_index": 0, "ref_start": 200, "ref_end": 250,
             "n_match": 50, "n_mismatch": 0},
        ]
    )
    out = attach_blocks(alignments, blocks)
    rows = out.to_pylist()
    # Alignment 1: synthetic fallback.
    assert rows[0]["blocks"][0]["n_match"] is None
    # Alignment 2: real block.
    assert rows[1]["blocks"][0]["n_match"] == 50


def test_attach_blocks_empty_alignments_returns_empty_column() -> None:
    """Zero alignments yields a zero-row blocks column of the right type."""
    alignments = _alignments([])
    out = attach_blocks(alignments, ALIGNMENT_BLOCK_TABLE.empty_table())
    assert out.num_rows == 0
    assert out.schema.field("blocks").type == BLOCKS_LIST_TYPE


# ──────────────────────────────────────────────────────────────────────
# attach_mismatch_positions
# ──────────────────────────────────────────────────────────────────────


def test_attach_mismatch_positions_skip_emits_empty_lists(
    tmp_path: Path,
) -> None:
    """skip=True (zoom too coarse) bypasses the cs parse entirely;
    every row gets an empty list. No cs file even has to exist."""
    alignments = _alignments(
        [{"alignment_id": 1, "ref_start": 0, "ref_end": 100}]
    )
    out = attach_mismatch_positions(alignments, tmp_path / "missing", skip=True)
    [row] = out.to_pylist()
    assert row["mismatch_positions"] == []
    assert out.schema.field("mismatch_positions").type == MISMATCH_POSITIONS_TYPE


def test_attach_mismatch_positions_missing_path_emits_empty_lists(
    tmp_path: Path,
) -> None:
    """No alignment_cs/ on disk → empty lists, no error. Defensive
    against partial source dirs (kernel's discover() should already
    have caught this via the hard-require gate)."""
    alignments = _alignments(
        [{"alignment_id": 1, "ref_start": 0, "ref_end": 100}]
    )
    out = attach_mismatch_positions(alignments, tmp_path / "missing", skip=False)
    [row] = out.to_pylist()
    assert row["mismatch_positions"] == []


def test_attach_mismatch_positions_parses_cs_when_present(
    tmp_path: Path,
) -> None:
    """Real cs:long path → positions extracted via the cs grammar."""
    cs_dir = tmp_path / "alignment_cs"
    cs_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"alignment_id": 1, "cs_string": ":10*ac:4"},
                {"alignment_id": 2, "cs_string": ""},
            ],
            schema=ALIGNMENT_CS_TABLE,
        ),
        cs_dir / "part-00000.parquet",
    )
    alignments = _alignments(
        [
            {"alignment_id": 1, "ref_start": 100, "ref_end": 115},
            {"alignment_id": 2, "ref_start": 200, "ref_end": 215},
            # Alignment with no matching cs_string row → empty positions.
            {"alignment_id": 3, "ref_start": 300, "ref_end": 315},
        ]
    )
    out = attach_mismatch_positions(alignments, cs_dir, skip=False)
    rows = out.to_pylist()
    # Substitution at offset 10 from ref_start=100 → position 110.
    assert rows[0]["mismatch_positions"] == [110]
    assert rows[1]["mismatch_positions"] == []
    assert rows[2]["mismatch_positions"] == []


def test_attach_mismatch_positions_malformed_cs_falls_back_to_empty(
    tmp_path: Path,
) -> None:
    """A malformed cs string for one alignment doesn't fail the whole
    request — that alignment gets empty positions; others still parse."""
    cs_dir = tmp_path / "alignment_cs"
    cs_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"alignment_id": 1, "cs_string": "@@@bogus"},
                {"alignment_id": 2, "cs_string": ":5*at:5"},
            ],
            schema=ALIGNMENT_CS_TABLE,
        ),
        cs_dir / "part-00000.parquet",
    )
    alignments = _alignments(
        [
            {"alignment_id": 1, "ref_start": 0, "ref_end": 10},
            {"alignment_id": 2, "ref_start": 0, "ref_end": 10},
        ]
    )
    out = attach_mismatch_positions(alignments, cs_dir, skip=False)
    rows = out.to_pylist()
    assert rows[0]["mismatch_positions"] == []
    assert rows[1]["mismatch_positions"] == [5]


def test_attach_mismatch_positions_empty_alignments_returns_empty_column(
    tmp_path: Path,
) -> None:
    alignments = _alignments([])
    out = attach_mismatch_positions(alignments, tmp_path / "missing", skip=True)
    assert out.num_rows == 0
    assert out.schema.field("mismatch_positions").type == MISMATCH_POSITIONS_TYPE
