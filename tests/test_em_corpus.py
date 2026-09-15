"""The EM read corpus — streaming write, row-indexed FASTA, mmapped access.

Two contracts are load-bearing enough to pin here:

* **FASTA names are corpus row indices.** The E-step's whole rewrite rests on
  ``q_name`` being an integer that parses straight out of the PAF byte buffer,
  so if row ``i`` of the Arrow file is not the read named ``>i``, every read's
  sequence lookup silently returns someone else's.
* **Rows are dense.** Reads dropped by the length filter (or for an empty
  window) must not leave a hole, or the same aliasing happens downstream.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
    ReadStore,
    write_corpus,
)


def _write_demux_dir(
    tmp_path: Path, rows: list[tuple], *, with_quality: bool = True
) -> Path:
    """rows = [(read_id, window_sequence, sample_id, dorado_quality), ...]"""
    lead, trail = "GGGGGGGGGG", "CCCCCCCCCC"
    reads_cols = {
        "read_id": [r[0] for r in rows],
        "sequence": [lead + r[1] + trail for r in rows],
        "quality": ["I" * (len(r[1]) + 20) for r in rows],
    }
    if with_quality:
        reads_cols["dorado_quality"] = pa.array([r[3] for r in rows], pa.float32())
    demux_tbl = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "transcript_segment_index": [0] * len(rows),
            "sample_id": pa.array([r[2] for r in rows], pa.int64()),
            "orientation": ["+"] * len(rows),
            "transcript_start": pa.array([len(lead)] * len(rows), pa.int32()),
            "transcript_end": pa.array(
                [len(lead) + len(r[1]) for r in rows], pa.int32()
            ),
            "score": pa.array([1.0] * len(rows), pa.float32()),
            "is_chimera": [False] * len(rows),
            "status": ["Complete"] * len(rows),
            "is_fragment": [False] * len(rows),
            "artifact": ["none"] * len(rows),
        }
    )
    demux = tmp_path / "demux"
    (demux / "reads").mkdir(parents=True)
    (demux / "read_demux").mkdir(parents=True)
    pq.write_table(pa.table(reads_cols), demux / "reads" / "part-00000.parquet")
    pq.write_table(demux_tbl, demux / "read_demux" / "part-00000.parquet")
    return demux


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    out, name, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(buf)))
            name, buf = line[1:], []
        else:
            buf.append(line)
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def test_fasta_names_are_corpus_row_indices(tmp_path):
    rows = [
        ("r0", "A" * 100, 0, 30.0),
        ("r1", "C" * 200, 1, 12.0),
        ("r2", "G" * 300, 0, 25.5),
    ]
    demux = _write_demux_dir(tmp_path, rows)
    corpus = write_corpus(demux, tmp_path / "corpus", max_window_length=None)

    assert corpus.n_reads == 3
    records = _parse_fasta(corpus.fasta_path)
    assert [name for name, _ in records] == ["0", "1", "2"]

    store = ReadStore.open(corpus.arrow_path)
    try:
        # Row i of the Arrow file IS the read named ">i" in the FASTA.
        for row, (_, seq) in enumerate(records):
            assert store.take_sequences([row]) == [seq]
    finally:
        store.close()


def test_dropped_reads_do_not_leave_holes(tmp_path):
    """A filtered read must renumber the rest, not leave a gap."""
    rows = [
        ("keep_a", "A" * 100, 0, 30.0),
        ("too_long", "T" * 5000, 0, 30.0),
        ("keep_b", "C" * 120, 0, 30.0),
    ]
    demux = _write_demux_dir(tmp_path, rows)
    corpus = write_corpus(demux, tmp_path / "corpus", max_window_length=1000)

    assert corpus.n_reads == 2
    assert corpus.stats["n_dropped_long"] == 1
    assert corpus.stats["max_input_length"] == 5000

    records = _parse_fasta(corpus.fasta_path)
    assert [name for name, _ in records] == ["0", "1"]

    store = ReadStore.open(corpus.arrow_path)
    try:
        assert store.read_id.to_pylist() == ["keep_a", "keep_b"]
        assert store.take_sequences([0, 1]) == ["A" * 100, "C" * 120]
    finally:
        store.close()


def test_dorado_quality_rides_through(tmp_path):
    rows = [("r0", "A" * 100, 0, 31.5), ("r1", "C" * 100, 1, 9.25)]
    demux = _write_demux_dir(tmp_path, rows)
    corpus = write_corpus(demux, tmp_path / "corpus", max_window_length=None)

    store = ReadStore.open(corpus.arrow_path)
    try:
        assert store.dorado_quality.tolist() == pytest.approx([31.5, 9.25])
        assert store.sample_id.tolist() == [0, 1]
        assert store.lengths().tolist() == [100, 100]
    finally:
        store.close()


def test_missing_dorado_quality_is_null_not_an_error(tmp_path):
    """Demux dirs written before the qs:f tag landed must still load."""
    rows = [("r0", "A" * 100, 0, 0.0)]
    demux = _write_demux_dir(tmp_path, rows, with_quality=False)
    corpus = write_corpus(demux, tmp_path / "corpus", max_window_length=None)

    store = ReadStore.open(corpus.arrow_path)
    try:
        assert store.n_reads == 1
        assert store.table.column("dorado_quality").null_count == 1
    finally:
        store.close()


def test_resume_skips_a_completed_corpus(tmp_path):
    rows = [("r0", "A" * 100, 0, 30.0), ("r1", "C" * 100, 0, 30.0)]
    demux = _write_demux_dir(tmp_path, rows)
    out = tmp_path / "corpus"
    first = write_corpus(demux, out, max_window_length=None)

    # Make the FASTA detectably stale; a resumed call must not rewrite it.
    sentinel = first.fasta_path.read_text() + "\n# sentinel\n"
    first.fasta_path.write_text(sentinel)

    second = write_corpus(demux, out, max_window_length=None, resume=True)
    assert second.n_reads == first.n_reads
    assert second.fasta_path.read_text() == sentinel
