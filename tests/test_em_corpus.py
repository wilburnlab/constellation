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

import numpy as np

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


# ── residency, not just output size ───────────────────────────────────


def _chunked_corpus(tmp_path, n_batches=8, per_batch=50):
    """A corpus written as several record batches, as a real one is."""
    from constellation.sequencing.transcriptome.cluster.denovo._io import _READS_SCHEMA

    path = tmp_path / "chunked.arrow"
    rng = np.random.default_rng(0)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, _READS_SCHEMA) as writer:
            for b in range(n_batches):
                writer.write_table(
                    pa.table(
                        {
                            "read_id": pa.array(
                                [f"b{b}_r{i}" for i in range(per_batch)], pa.string()
                            ),
                            "sequence": pa.array(
                                [
                                    "".join(rng.choice(list("ACGT"), 60))
                                    for _ in range(per_batch)
                                ],
                                pa.large_string(),
                            ),
                            "sample_id": pa.array(
                                np.full(per_batch, b, np.int64)
                            ),
                            "dorado_quality": pa.array(
                                np.full(per_batch, 30.0, np.float32)
                            ),
                        },
                        schema=_READS_SCHEMA,
                    )
                )
    return path


def test_take_never_concatenates_the_chunked_corpus(tmp_path, monkeypatch):
    """`pc.take` on a ChunkedArray concatenates it ALL before indexing.

    So fetching one row off a 232 MB / 48-chunk mmapped corpus peaked at
    +444.8 MB — about twice the corpus — against +5.1 MB taking chunk-locally.
    At 9.4M reads the corpus is ~25 GB, which is ~47 GB per worker on first
    access, in every worker: a lazy file-backed corpus made fully resident.

    Asserted structurally rather than by measuring memory, so it cannot go
    quiet on a different allocator: `pc.take` must never be handed the
    chunked column at all.
    """
    import pyarrow.compute as pc

    store = ReadStore.open(_chunked_corpus(tmp_path))
    try:
        assert store.sequence.num_chunks > 1, "the fixture must be chunked"

        real_take = pc.take
        offenders = []

        def spy(data, indices, **kw):
            if isinstance(data, pa.ChunkedArray) and data.num_chunks > 1:
                offenders.append(data.num_chunks)
            return real_take(data, indices, **kw)

        monkeypatch.setattr(pc, "take", spy)
        store.take_sequences(np.array([3, 199, 45]))
        store.take_read_ids(np.array([3, 199, 45]))
        assert not offenders, (
            f"pc.take was handed a {offenders[0]}-chunk column — that "
            "concatenates the whole corpus"
        )
    finally:
        store.close()


def test_chunked_take_matches_pc_take_exactly(tmp_path):
    """Cheaper is only useful if it is also identical."""
    import pyarrow.compute as pc

    from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
        chunked_take,
    )

    store = ReadStore.open(_chunked_corpus(tmp_path))
    try:
        rng = np.random.default_rng(3)
        for _ in range(50):
            rows = rng.integers(0, store.n_reads, int(rng.integers(1, 40)))
            want = pc.take(store.sequence, pa.array(rows)).to_pylist()
            got = chunked_take(store.sequence, rows, store.chunk_starts).to_pylist()
            assert want == got
        # order, duplicates and the boundaries between chunks
        edges = np.array([0, 49, 50, 51, 99, 100, store.n_reads - 1])
        assert (
            chunked_take(store.sequence, edges, store.chunk_starts).to_pylist()
            == pc.take(store.sequence, pa.array(edges)).to_pylist()
        )
        rev = np.array([120, 7, 120, 3, 7])
        assert (
            chunked_take(store.sequence, rev, store.chunk_starts).to_pylist()
            == pc.take(store.sequence, pa.array(rev)).to_pylist()
        )
        assert chunked_take(store.sequence, np.array([]), store.chunk_starts).to_pylist() == []
    finally:
        store.close()


def test_rows_resolve_to_the_right_chunk(tmp_path):
    """A row must come back as itself, not as its neighbour in another chunk."""
    store = ReadStore.open(_chunked_corpus(tmp_path, n_batches=4, per_batch=10))
    try:
        assert store.take_read_ids(np.arange(40)).to_pylist() == [
            f"b{b}_r{i}" for b in range(4) for i in range(10)
        ]
        assert store.sample_id.tolist() == [b for b in range(4) for _ in range(10)]
    finally:
        store.close()
