"""BUSCO — pure summary/full-table parsers + mock-binary BuscoRunner."""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pytest

from constellation.sequencing.annotation.busco import (
    BuscoRunner,
    busco_summary_row,
    parse_busco_full_table,
    parse_busco_summary,
)
from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.assembly.stats import apply_busco, assembly_stats


# ── pure parsers ──────────────────────────────────────────────────────


def test_parse_busco_summary():
    text = "\tC:95.0%[S:93.0%,D:2.0%],F:3.0%,M:2.0%,n:100\n"
    v = parse_busco_summary(text)
    assert v["complete"] == pytest.approx(0.95)
    assert v["single"] == pytest.approx(0.93)
    assert v["duplicated"] == pytest.approx(0.02)
    assert v["fragmented"] == pytest.approx(0.03)
    assert v["missing"] == pytest.approx(0.02)


def test_parse_busco_summary_no_line_raises():
    with pytest.raises(ValueError, match="summary line"):
        parse_busco_summary("nothing useful here")


def test_parse_busco_full_table():
    text = (
        "# Busco id\tStatus\tSequence\tGene Start\tGene End\tStrand\tScore\tLength\n"
        "1at2759\tComplete\tctg0\t10\t100\t+\t250.5\t90\n"
        "2at2759\tDuplicated\tctg1\t5\t55\t-\t120.0\t50\n"
        "3at2759\tMissing\n"
        "4at2759\tComplete\tunknown_ctg\t1\t9\t+\t10\t9\n"
    )
    rows = parse_busco_full_table(text, {"ctg0": 0, "ctg1": 1})
    assert len(rows) == 2  # Missing skipped, unknown_ctg skipped
    assert rows[0]["name"] == "1at2759"
    assert rows[0]["contig_id"] == 0
    assert rows[0]["start"] == 9  # 1-based 10 → 0-based 9
    assert rows[0]["end"] == 100
    assert rows[0]["strand"] == "+"
    assert rows[0]["type"] == "gene"
    assert rows[0]["source"] == "busco"
    assert rows[1]["contig_id"] == 1
    assert rows[1]["strand"] == "-"


def test_apply_busco_merges_into_assembly_stats():
    contigs = pa.table(
        {
            "contig_id": pa.array([0], type=pa.int64()),
            "name": ["c0"],
            "length": pa.array([100], type=pa.int64()),
        }
    )
    sequences = pa.table(
        {"contig_id": pa.array([0], type=pa.int64()), "sequence": ["ACGT" * 25]}
    )
    asm = Assembly.from_tables(contigs, sequences)
    base = assembly_stats(asm)
    assert base.to_pylist()[0]["busco_complete"] is None
    busco = busco_summary_row(
        {"complete": 0.9, "single": 0.88, "duplicated": 0.02,
         "fragmented": 0.05, "missing": 0.05},
        "eukaryota_odb10",
    )
    merged = apply_busco(base, busco)
    row = merged.to_pylist()[0]
    assert row["busco_complete"] == pytest.approx(0.9)
    assert row["busco_lineage"] == "eukaryota_odb10"
    assert row["n_contigs"] == 1  # base columns preserved


# ── mock-binary runner ────────────────────────────────────────────────

_BUSCO_STUB = r"""#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "BUSCO 5.7.1"; exit 0; fi
out_path=""; run_name="busco"; lineage="testlin"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --out_path) out_path="${args[$((i+1))]}" ;;
    -o) run_name="${args[$((i+1))]}" ;;
    -l) lineage="${args[$((i+1))]}" ;;
  esac
done
rundir="${out_path}/${run_name}"
mkdir -p "${rundir}/run_${lineage}"
printf '\tC:95.0%%[S:93.0%%,D:2.0%%],F:3.0%%,M:2.0%%,n:100\n' \
  > "${rundir}/short_summary.specific.${lineage}.busco.txt"
{
  printf '# Busco id\tStatus\tSequence\tGene Start\tGene End\tStrand\tScore\tLength\n'
  printf '1at2759\tComplete\tctg0\t10\t100\t+\t250.5\t90\n'
  printf '2at2759\tMissing\n'
} > "${rundir}/run_${lineage}/full_table.tsv"
"""


def test_busco_runner_mock(tmp_path: Path, monkeypatch):
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash unavailable")
    home = tmp_path / "busco_home"
    home.mkdir()
    stub = home / "busco"
    stub.write_text(_BUSCO_STUB)
    stub.chmod(0o755)
    monkeypatch.setenv("CONSTELLATION_BUSCO_HOME", str(home))

    contigs = pa.table(
        {
            "contig_id": pa.array([0], type=pa.int64()),
            "name": ["ctg0"],
            "length": pa.array([100], type=pa.int64()),
        }
    )
    sequences = pa.table(
        {"contig_id": pa.array([0], type=pa.int64()), "sequence": ["ACGT" * 25]}
    )
    asm = Assembly.from_tables(contigs, sequences)

    stats_row, features = BuscoRunner(lineage="testlin").run(asm, tmp_path / "busco_out")
    assert stats_row.to_pylist()[0]["busco_complete"] == pytest.approx(0.95)
    assert stats_row.to_pylist()[0]["busco_lineage"] == "testlin"
    assert features.num_rows == 1
    assert features.to_pylist()[0]["name"] == "1at2759"
    assert features.to_pylist()[0]["contig_id"] == 0
