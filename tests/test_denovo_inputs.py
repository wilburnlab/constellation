"""The input length filter at the corpus boundary.

An oversized window is a concatemer, not a transcript. The measured input
carried a 361,908 nt "cDNA"; 2,220 windows exceed 15 kb (0.024% of reads) and
they seed giant templates directly — round 1's maximum template length equals
the input maximum.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.transcriptome.cluster.denovo._io import (
    load_demux_windows,
)


def _write_demux_dir(tmp_path: Path, rows: list[tuple]) -> Path:
    """rows = [(read_id, window_sequence, sample_id), ...]"""
    lead, trail = "GGGGGGGGGG", "CCCCCCCCCC"
    reads_tbl = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sequence": [lead + r[1] + trail for r in rows],
            "quality": ["I" * (len(r[1]) + 20) for r in rows],
        }
    )
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
    pq.write_table(reads_tbl, demux / "reads" / "part-00000.parquet")
    pq.write_table(demux_tbl, demux / "read_demux" / "part-00000.parquet")
    return demux


def _corpus(tmp_path):
    rows = [
        ("short", "A" * 500, 0),
        ("at_limit", "C" * 1000, 0),
        ("over", "G" * 1001, 1),
        ("way_over", "T" * 40_000, 1),
    ]
    return _write_demux_dir(tmp_path, rows)


def test_filter_drops_above_the_limit_and_keeps_the_boundary(tmp_path):
    reads, stats = load_demux_windows(_corpus(tmp_path), max_window_length=1000)
    assert reads.column("read_id").to_pylist() == ["short", "at_limit"]
    assert stats["n_input"] == 4
    assert stats["n_dropped_long"] == 2
    assert stats["max_input_length"] == 40_000


def test_none_is_a_byte_identical_no_op(tmp_path):
    """The library default must not change what existing callers see — this is
    the components-path parity guard."""
    demux = _corpus(tmp_path)
    unfiltered, stats = load_demux_windows(demux)
    assert unfiltered.num_rows == 4
    assert stats["n_dropped_long"] == 0
    # A limit above everything present is the same table.
    generous, _ = load_demux_windows(demux, max_window_length=1_000_000)
    assert generous.equals(unfiltered)


def test_zero_disables_the_filter(tmp_path):
    """The CLI uses 0 as 'off', so the library must not read it as 'drop
    everything'."""
    reads, stats = load_demux_windows(_corpus(tmp_path), max_window_length=0)
    assert reads.num_rows == 4
    assert stats["n_dropped_long"] == 0


def test_stats_are_reported_even_when_nothing_is_dropped(tmp_path):
    """0.024% of reads is a number a user has to be able to audit — it moves
    per-sample quant totals."""
    _reads, stats = load_demux_windows(_corpus(tmp_path), max_window_length=50_000)
    assert stats == {
        "n_input": 4,
        "n_dropped_long": 0,
        "max_input_length": 40_000,
    }


def test_oversized_windows_are_dropped_not_trimmed(tmp_path):
    """Truncating a 40 kb concatemer to the limit would manufacture a read that
    was never sequenced."""
    reads, _stats = load_demux_windows(_corpus(tmp_path), max_window_length=1000)
    lengths = [len(s) for s in reads.column("sequence").to_pylist()]
    assert 1000 not in lengths[2:], "nothing was trimmed down to the limit"
    assert sorted(lengths) == [500, 1000]


def test_empty_corpus_is_not_an_error(tmp_path):
    reads, stats = load_demux_windows(
        _write_demux_dir(tmp_path, [("only", "A" * 9_000, 0)]),
        max_window_length=1000,
    )
    assert reads.num_rows == 0
    assert stats["n_dropped_long"] == 1


@pytest.mark.parametrize("limit", [499, 500, 501])
def test_the_boundary_is_inclusive(tmp_path, limit):
    reads, _ = load_demux_windows(
        _write_demux_dir(tmp_path, [("r", "A" * 500, 0)]), max_window_length=limit
    )
    assert reads.num_rows == (0 if limit < 500 else 1)


def test_dropping_a_window_does_not_crash_a_quiet_run(tmp_path, monkeypatch):
    """The length-filter report called a logger that only exists under
    `verbose` or a progress callback — so the report crashed on exactly the
    runs where a window was actually dropped and the number mattered."""
    from constellation.sequencing.transcriptome.cluster.denovo import pipeline

    demux = _corpus(tmp_path)
    seen = {}

    def _fake(reads, **kw):
        seen["n"] = reads.num_rows
        raise _Stop

    class _Stop(Exception):
        pass

    monkeypatch.setattr(pipeline, "assemble_clusters", _fake)
    with pytest.raises(_Stop):
        pipeline.cluster_transcripts(
            demux_dir=demux,
            output_dir=tmp_path / "out",
            max_window_length=1000,
            verbose=False,
            progress_cb=None,
        )
    assert seen["n"] == 2, "the filter ran; the report must not have crashed"
