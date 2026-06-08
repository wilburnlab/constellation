"""``Assembly`` ParquetDir round-trip (with and without scaffolds)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.assembly.io import load_assembly, save_assembly


def _toy(scaffolds: pa.Table | None = None) -> Assembly:
    contigs = pa.table(
        {
            "contig_id": pa.array([0, 1, 2], type=pa.int64()),
            "name": ["c0", "c1", "c2"],
            "length": pa.array([6, 5, 4], type=pa.int64()),
        }
    )
    sequences = pa.table(
        {
            "contig_id": pa.array([0, 1, 2], type=pa.int64()),
            "sequence": ["GGCCCC", "ATATA", "ACGT"],
        }
    )
    return Assembly.from_tables(contigs, sequences, scaffolds=scaffolds)


def _scaffolds() -> pa.Table:
    return pa.table(
        {
            "scaffold_id": pa.array([0, 0], type=pa.int64()),
            "name": ["scf0", "scf0"],
            "contig_id": pa.array([0, 1], type=pa.int64()),
            "position": pa.array([0, 1], type=pa.int32()),
            "orientation": ["+", "-"],
            "gap_size": pa.array([100, -1], type=pa.int64()),
            "gap_type": ["estimated", None],
        }
    )


def test_roundtrip_no_scaffolds(tmp_path: Path):
    a = _toy().with_metadata({"assembler": "test"})
    save_assembly(a, tmp_path / "asm")
    a2 = load_assembly(tmp_path / "asm")
    assert a2.n_contigs == 3
    assert a2.total_length == 15
    assert a2.scaffolds is None
    assert a2.metadata_extras["assembler"] == "test"
    assert a2.stats.to_pylist()[0]["n50"] == a.stats.to_pylist()[0]["n50"]
    assert not (tmp_path / "asm" / "scaffolds.parquet").exists()


def test_roundtrip_with_scaffolds(tmp_path: Path):
    a = _toy(scaffolds=_scaffolds())
    save_assembly(a, tmp_path / "asm")
    a2 = load_assembly(tmp_path / "asm")
    assert a2.scaffolds is not None
    assert a2.scaffolds.num_rows == 2
    assert a2.stats.to_pylist()[0]["n_scaffolds"] == 1


def test_sequences_one_row_group_per_contig(tmp_path: Path):
    a = _toy()
    save_assembly(a, tmp_path / "asm")
    pf = pq.ParquetFile(tmp_path / "asm" / "sequences.parquet")
    assert pf.num_row_groups == a.n_contigs
