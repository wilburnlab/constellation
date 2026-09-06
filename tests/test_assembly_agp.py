"""AGP parser — ``parse_agp`` → SCAFFOLD_TABLE rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.sequencing.assembly.agp import parse_agp


def test_parse_agp_pairs_contig_with_following_gap(tmp_path: Path):
    agp = tmp_path / "s.agp"
    agp.write_text(
        "## AGP version 2.0\n"
        "chr1_RagTag\t1\t1000\t1\tW\tctg1\t1\t1000\t+\n"
        "chr1_RagTag\t1001\t1100\t2\tN\t100\tscaffold\tyes\talign_genus\n"
        "chr1_RagTag\t1101\t2100\t3\tW\tctg2\t1\t1000\t-\n"
    )
    rows = parse_agp(agp, {"ctg1": 0, "ctg2": 1})
    assert len(rows) == 2
    assert rows[0]["contig_id"] == 0
    assert rows[0]["position"] == 0
    assert rows[0]["orientation"] == "+"
    assert rows[0]["gap_size"] == 100  # gap to the next contig
    assert rows[0]["gap_type"] == "estimated"
    assert rows[1]["contig_id"] == 1
    assert rows[1]["orientation"] == "-"
    assert rows[1]["gap_size"] == -1  # terminal contig
    assert rows[0]["scaffold_id"] == rows[1]["scaffold_id"] == 0
    assert rows[1]["position"] == 1


def test_parse_agp_multiple_scaffolds_get_distinct_ids(tmp_path: Path):
    agp = tmp_path / "s.agp"
    agp.write_text(
        "chrA\t1\t10\t1\tW\tc1\t1\t10\t+\n"
        "chrB\t1\t10\t1\tW\tc2\t1\t10\t+\n"
    )
    rows = parse_agp(agp, {"c1": 0, "c2": 1})
    assert rows[0]["scaffold_id"] == 0
    assert rows[1]["scaffold_id"] == 1
    assert rows[0]["gap_size"] == -1  # single-contig scaffold, no gap


def test_parse_agp_u_gap_is_unknown(tmp_path: Path):
    agp = tmp_path / "s.agp"
    agp.write_text(
        "chrA\t1\t10\t1\tW\tc1\t1\t10\t+\n"
        "chrA\t11\t110\t2\tU\t100\tscaffold\tyes\tna\n"
        "chrA\t111\t120\t3\tW\tc2\t1\t10\t+\n"
    )
    rows = parse_agp(agp, {"c1": 0, "c2": 1})
    assert rows[0]["gap_type"] == "unknown"
    assert rows[0]["gap_size"] == 100


def test_parse_agp_unknown_component_raises(tmp_path: Path):
    agp = tmp_path / "s.agp"
    agp.write_text("chrA\t1\t10\t1\tW\tmystery\t1\t10\t+\n")
    with pytest.raises(ValueError, match="not found"):
        parse_agp(agp, {"c1": 0})
