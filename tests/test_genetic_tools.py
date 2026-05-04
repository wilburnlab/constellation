"""``GeneticTools`` container + bundled genetic_tools.json + FASTA emission."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.sequencing.genetic_tools import (
    GeneticTools,
    load_default_genetic_tools,
)
from constellation.sequencing.readers.fastx import read_fasta_genetic_tools
from constellation.sequencing.schemas.reference import (
    GENETIC_TOOL_CATEGORIES,
    GENETIC_TOOL_TABLE,
)


def _make_tools() -> GeneticTools:
    rows = [
        {
            "tool_id": 0,
            "name": "AmpR",
            "category": "antibiotic_resistance",
            "sequence_type": "protein",
            "sequence": "MIRRABLA",
            "source": "manual_curation",
            "source_url": None,
            "references_json": None,
        },
        {
            "tool_id": 1,
            "name": "T7",
            "category": "promoter",
            "sequence_type": "nucleotide",
            "sequence": "TAATACGACTCACTATAG",
            "source": "manual_curation",
            "source_url": None,
            "references_json": None,
        },
    ]
    table = pa.Table.from_pylist(rows, schema=GENETIC_TOOL_TABLE)
    return GeneticTools(tools=table)


def test_construct_validates():
    g = _make_tools()
    assert g.n_tools == 2


def test_unknown_category_rejected():
    rows = [
        {
            "tool_id": 0,
            "name": "X",
            "category": "made_up",
            "sequence_type": "protein",
            "sequence": "AAA",
            "source": "x",
            "source_url": None,
            "references_json": None,
        }
    ]
    table = pa.Table.from_pylist(rows, schema=GENETIC_TOOL_TABLE)
    with pytest.raises(ValueError, match="unknown category"):
        GeneticTools(tools=table)


def test_unknown_sequence_type_rejected():
    rows = [
        {
            "tool_id": 0,
            "name": "X",
            "category": "antibiotic_resistance",
            "sequence_type": "rna",
            "sequence": "AAA",
            "source": "x",
            "source_url": None,
            "references_json": None,
        }
    ]
    table = pa.Table.from_pylist(rows, schema=GENETIC_TOOL_TABLE)
    with pytest.raises(ValueError, match="unknown sequence_type"):
        GeneticTools(tools=table)


def test_empty_sequence_rejected():
    rows = [
        {
            "tool_id": 0,
            "name": "X",
            "category": "antibiotic_resistance",
            "sequence_type": "protein",
            "sequence": "",
            "source": "x",
            "source_url": None,
            "references_json": None,
        }
    ]
    table = pa.Table.from_pylist(rows, schema=GENETIC_TOOL_TABLE)
    with pytest.raises(ValueError, match="sequence is empty"):
        GeneticTools(tools=table)


def test_of_category_filter():
    g = _make_tools()
    abr = g.of_category("antibiotic_resistance")
    assert abr.num_rows == 1
    assert abr.column("name")[0].as_py() == "AmpR"


def test_to_fasta_protein_only():
    g = _make_tools()
    fasta = g.to_fasta(sequence_type="protein")
    assert ">tool:0 AmpR category=antibiotic_resistance" in fasta
    assert "MIRRABLA" in fasta
    assert "TAATACGACTCACTATAG" not in fasta


def test_to_fasta_all():
    g = _make_tools()
    fasta = g.to_fasta()
    # Both records present
    assert ">tool:0" in fasta and ">tool:1" in fasta


def test_load_default_genetic_tools_bundle():
    g = load_default_genetic_tools()
    assert g.n_tools >= 50  # bundled DB has 67+ entries
    # Spans multiple categories
    cats = set(g.tools.column("category").to_pylist())
    assert cats.issubset(GENETIC_TOOL_CATEGORIES)
    assert "antibiotic_resistance" in cats
    assert "fluorescent_protein" in cats
    assert g.metadata_extras.get("version")


def test_read_fasta_genetic_tools(tmp_path: Path):
    p = tmp_path / "tools.fa"
    p.write_text(">FLAG\nDYKDDDDK\n>HA\nYPYDVPDYA\n")
    g = read_fasta_genetic_tools(
        p, category="epitope_tag", sequence_type="protein"
    )
    assert g.n_tools == 2
    rows = g.tools.to_pylist()
    assert rows[0]["name"] == "FLAG"
    assert rows[0]["sequence"] == "DYKDDDDK"
    assert rows[0]["category"] == "epitope_tag"


def test_read_fasta_genetic_tools_validates_category(tmp_path: Path):
    p = tmp_path / "x.fa"
    p.write_text(">x\nA\n")
    with pytest.raises(ValueError, match="unknown category"):
        read_fasta_genetic_tools(p, category="bogus", sequence_type="protein")
