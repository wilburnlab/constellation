"""GFA segment parser — ``parse_gfa_contigs``."""

from __future__ import annotations

import gzip
from pathlib import Path

from constellation.sequencing.assembly.gfa import GfaContig, parse_gfa_contigs


def test_parse_gfa_segments(tmp_path: Path):
    gfa = tmp_path / "primary.gfa"
    gfa.write_text(
        "H\tVN:Z:1.0\n"
        "S\tptg000001l\tACGTACGT\tLN:i:8\trd:i:35\n"
        "S\tptg000002c\tTTTTGGGGCCCC\tLN:i:12\trd:i:40\n"
        "L\tptg000001l\t+\tptg000002c\t+\t0M\n"
    )
    recs = parse_gfa_contigs(gfa)
    assert len(recs) == 2
    assert isinstance(recs[0], GfaContig)
    assert recs[0].name == "ptg000001l"
    assert recs[0].sequence == "ACGTACGT"
    assert recs[0].length == 8
    assert recs[0].read_coverage == 35.0
    assert recs[1].name == "ptg000002c"
    assert recs[1].length == 12
    assert recs[1].read_coverage == 40.0
    # circular is not set from GFA itself (assembler convention applied upstream)
    assert recs[0].circular is None


def test_parse_gfa_star_sequence_uses_ln_tag(tmp_path: Path):
    gfa = tmp_path / "p.gfa"
    gfa.write_text("S\tseg1\t*\tLN:i:100\n")
    recs = parse_gfa_contigs(gfa)
    assert recs[0].sequence == ""
    assert recs[0].length == 100
    assert recs[0].read_coverage is None


def test_parse_gfa_length_falls_back_to_sequence(tmp_path: Path):
    gfa = tmp_path / "p.gfa"
    gfa.write_text("S\tseg1\tACGTA\n")  # no LN tag
    recs = parse_gfa_contigs(gfa)
    assert recs[0].length == 5


def test_parse_gfa_gzip(tmp_path: Path):
    p = tmp_path / "p.gfa.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write("S\ts1\tACGT\tLN:i:4\trd:i:10\n")
    recs = parse_gfa_contigs(p)
    assert len(recs) == 1
    assert recs[0].read_coverage == 10.0


def test_parse_gfa_float_coverage(tmp_path: Path):
    gfa = tmp_path / "p.gfa"
    gfa.write_text("S\ts1\tACGT\tLN:i:4\trd:f:12.5\n")
    recs = parse_gfa_contigs(gfa)
    assert recs[0].read_coverage == 12.5
