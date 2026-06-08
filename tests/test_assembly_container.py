"""``Assembly`` container — construction, validation, projection."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.reference.reference import GenomeReference


def _contigs(ids: list[int], names: list[str], lengths: list[int]) -> pa.Table:
    return pa.table(
        {
            "contig_id": pa.array(ids, type=pa.int64()),
            "name": names,
            "length": pa.array(lengths, type=pa.int64()),
        }
    )


def _sequences(ids: list[int], seqs: list[str]) -> pa.Table:
    return pa.table(
        {
            "contig_id": pa.array(ids, type=pa.int64()),
            "sequence": seqs,
        }
    )


def _toy() -> Assembly:
    return Assembly.from_tables(
        _contigs([0, 1, 2], ["c0", "c1", "c2"], [6, 5, 4]),
        _sequences([0, 1, 2], ["GGCCCC", "ATATA", "ACGT"]),
    )


def test_construct_and_properties():
    a = _toy()
    assert a.n_contigs == 3
    assert a.total_length == 15
    assert a.scaffolds is None


def test_to_genome_reference_projection():
    a = _toy()
    g = a.to_genome_reference()
    assert isinstance(g, GenomeReference)
    assert g.n_contigs == 3
    assert g.sequence_of(0) == "GGCCCC"
    # assembly-only provenance columns are dropped in the projection
    assert "read_coverage" not in g.contigs.column_names
    assert "topology" in g.contigs.column_names


def test_duplicate_contig_id_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Assembly.from_tables(
            _contigs([0, 0], ["a", "b"], [1, 1]),
            _sequences([0], ["A"]),
        )


def test_orphan_sequence_rejected():
    with pytest.raises(ValueError, match="references ids"):
        Assembly.from_tables(
            _contigs([0], ["x"], [4]),
            _sequences([0, 9], ["ACGT", "TTTT"]),
        )


def test_scaffold_contig_ids_are_provenance_not_fk_checked():
    # SCAFFOLD_TABLE.contig_id lives in the pre-scaffold draft id space,
    # so a value absent from the (post-scaffold) contigs is accepted as
    # provenance rather than rejected.
    scaffolds = pa.table(
        {
            "scaffold_id": pa.array([0], type=pa.int64()),
            "name": ["scf0"],
            "contig_id": pa.array([99], type=pa.int64()),  # draft-space id
            "position": pa.array([0], type=pa.int32()),
            "orientation": ["+"],
            "gap_size": pa.array([-1], type=pa.int64()),
            "gap_type": [None],
        }
    )
    a = Assembly.from_tables(
        _contigs([0], ["scf0"], [4]),
        _sequences([0], ["ACGT"]),
        scaffolds=scaffolds,
    )
    assert a.scaffolds is not None
    assert a.stats.to_pylist()[0]["n_scaffolds"] == 1


def test_with_metadata():
    a = _toy().with_metadata({"assembler": "hifiasm"})
    assert a.metadata_extras["assembler"] == "hifiasm"
    assert a.n_contigs == 3  # survives the replace()
