"""Tests for ``constellation.sequencing.align.map``.

Two test groups:
  * Pure-Python tests for ``_iter_demux_read_batches`` +
    ``_format_fastq_bytes`` — run anywhere with pyarrow.
  * minimap2 / samtools end-to-end tests — gated on tool availability.

The full pipeline test exercises ``map_to_genome`` against a tiny
synthetic genome FASTA + a synthetic S1 demux output dir.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.align.map import (
    _format_fastq_bytes,
    _iter_demux_read_batches,
)
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.reads import READ_TABLE
from constellation.sequencing.schemas.transcriptome import READ_DEMUX_TABLE


_TOOLS_AVAILABLE = bool(shutil.which("minimap2") and shutil.which("samtools"))
_skip_no_tools = pytest.mark.skipif(
    not _TOOLS_AVAILABLE,
    reason=(
        "minimap2 / samtools not on PATH; install via "
        "`conda install -c bioconda minimap2 samtools`"
    ),
)


def _make_genome():
    contigs = pa.Table.from_pylist(
        [
            {"contig_id": 0, "name": "chr1", "length": 1000, "topology": None, "circular": None},
        ]
    )
    sequences = pa.Table.from_pylist(
        [
            {"contig_id": 0, "sequence": "ACGTACGTAC" * 100},
        ]
    )
    return GenomeReference(contigs=contigs, sequences=sequences)


def _make_demux_dir(
    tmp_path: Path,
    *,
    reads: list[dict],
    demux: list[dict],
) -> Path:
    """Stand up a minimal S1 demux output: ``reads/`` + ``read_demux/``
    partitioned datasets, each with one parquet shard."""
    demux_dir = tmp_path / "demux"
    (demux_dir / "reads").mkdir(parents=True)
    (demux_dir / "read_demux").mkdir(parents=True)

    reads_table = pa.Table.from_pylist(reads, schema=READ_TABLE)
    pq.write_table(reads_table, demux_dir / "reads" / "part-00000.parquet")

    demux_table = pa.Table.from_pylist(demux, schema=READ_DEMUX_TABLE)
    pq.write_table(demux_table, demux_dir / "read_demux" / "part-00000.parquet")
    return demux_dir


def test_iter_demux_read_batches_filters_and_slices(tmp_path: Path) -> None:
    """Filter set + transcript-window slicing produce the right FASTQ.

    Synthetic dataset:
      r0: status='Complete', sample_id=1, transcript window [10:50] → emit
      r1: status='Unknown'   → drop (status filter)
      r2: status='Complete', sample_id=null → drop (sample_id filter)
      r3: status='Complete', sample_id=1, is_fragment=True → drop
    """
    seq = "ACGT" * 100  # 400 bp marker sequence
    qual = "I" * 400
    reads = [
        {"read_id": f"r{i}", "acquisition_id": 1, "sequence": seq, "quality": qual,
         "length": 400, "mean_quality": 40.0, "channel": None, "start_time_s": None,
         "duration_s": None, "dorado_quality": None, "read_group": None,
         "duplex_class": None}
        for i in range(4)
    ]

    def _demux_row(read_id: str, *, status: str, sample_id, is_fragment: bool):
        return {
            "read_id": read_id,
            "transcript_segment_index": 0,
            "sample_id": sample_id,
            "orientation": "+",
            "transcript_start": 10,
            "transcript_end": 50,
            "score": 0.0,
            "is_chimera": False,
            "status": status,
            "is_fragment": is_fragment,
            "artifact": "none",
        }

    demux = [
        _demux_row("r0", status="Complete", sample_id=1, is_fragment=False),
        _demux_row("r1", status="Unknown", sample_id=1, is_fragment=False),
        _demux_row("r2", status="Complete", sample_id=None, is_fragment=False),
        _demux_row("r3", status="Complete", sample_id=1, is_fragment=True),
    ]
    demux_dir = _make_demux_dir(tmp_path, reads=reads, demux=demux)

    batches = list(_iter_demux_read_batches(demux_dir))
    assert len(batches) == 1
    batch = batches[0]
    # The batch carries the full sequence + the transcript_{start,end}
    # offsets — slicing happens inside _format_fastq_bytes.
    assert batch.column("read_id").to_pylist() == ["r0"]
    assert batch.column("sample_id").to_pylist() == [1]
    assert batch.column("sequence").to_pylist() == [seq]
    assert batch.column("quality").to_pylist() == [qual]
    assert batch.column("transcript_start").to_pylist() == [10]
    assert batch.column("transcript_end").to_pylist() == [50]

    fastq_bytes, n_reads = _format_fastq_bytes(batch)
    assert n_reads == 1
    text = fastq_bytes.decode("ascii")
    assert text.startswith("@r0\n")
    lines = text.strip().split("\n")
    assert len(lines) == 4
    assert lines[1] == seq[10:50]
    assert lines[2] == "+"
    assert lines[3] == qual[10:50]


def test_iter_demux_read_batches_skips_invalid_window(tmp_path: Path) -> None:
    """transcript_start=-1 / transcript_end<=transcript_start are dropped."""
    seq = "ACGT" * 50
    reads = [
        {"read_id": "r0", "acquisition_id": 1, "sequence": seq, "quality": "I" * 200,
         "length": 200, "mean_quality": 40.0, "channel": None, "start_time_s": None,
         "duration_s": None, "dorado_quality": None, "read_group": None,
         "duplex_class": None},
        {"read_id": "r1", "acquisition_id": 1, "sequence": seq, "quality": "I" * 200,
         "length": 200, "mean_quality": 40.0, "channel": None, "start_time_s": None,
         "duration_s": None, "dorado_quality": None, "read_group": None,
         "duplex_class": None},
    ]
    demux = [
        {"read_id": "r0", "transcript_segment_index": 0, "sample_id": 1,
         "orientation": "+",
         "transcript_start": -1, "transcript_end": -1, "score": 0.0,
         "is_chimera": False, "status": "Complete", "is_fragment": False,
         "artifact": "none"},
        {"read_id": "r1", "transcript_segment_index": 0, "sample_id": 1,
         "orientation": "+",
         "transcript_start": 50, "transcript_end": 50, "score": 0.0,
         "is_chimera": False, "status": "Complete", "is_fragment": False,
         "artifact": "none"},
    ]
    demux_dir = _make_demux_dir(tmp_path, reads=reads, demux=demux)
    batches = list(_iter_demux_read_batches(demux_dir))
    assert batches == []


@_skip_no_tools
def test_map_to_genome_end_to_end(tmp_path: Path) -> None:
    """Full pipeline: synthetic demux dir → minimap2 → sorted+indexed BAM.

    The read carries a 30 bp 'adapter' prefix + a 100 bp transcript
    window matching the genome + a 30 bp 'adapter' suffix. The demux
    table locates the transcript window at [30, 130). Streaming should
    feed only the 100 bp window to minimap2 — if it fed the full read
    (or sliced wrongly), the alignment would either soft-clip the
    adapters or fail to map cleanly.
    """
    genome = _make_genome()
    transcript = "ACGTACGTAC" * 10  # 100 bp matches the genome
    full_read = "N" * 30 + transcript + "N" * 30  # 160 bp total
    full_qual = "I" * len(full_read)

    reads = [
        {
            "read_id": "r0",
            "acquisition_id": 1,
            "sequence": full_read,
            "quality": full_qual,
            "length": len(full_read),
            "mean_quality": 40.0,
            "channel": None,
            "start_time_s": None,
            "duration_s": None,
            "dorado_quality": None,
            "read_group": None,
            "duplex_class": None,
        },
    ]
    demux = [
        {
            "read_id": "r0",
            "transcript_segment_index": 0,
            "sample_id": 1,
            "orientation": "+",
            "transcript_start": 30,
            "transcript_end": 130,
            "score": 0.0,
            "is_chimera": False,
            "status": "Complete",
            "is_fragment": False,
            "artifact": "none",
        },
    ]
    demux_dir = _make_demux_dir(tmp_path, reads=reads, demux=demux)

    from constellation.sequencing.align.map import map_to_genome
    from constellation.sequencing.io.sam_bam import read_bam

    output_dir = tmp_path / "out"
    bam = map_to_genome(demux_dir, genome, output_dir=output_dir, threads=1)
    assert bam.exists()
    assert (output_dir / "genome.fa").exists()
    assert (output_dir / "genome.mmi").exists()
    assert (output_dir / "bam" / "aligned.bam.bai").exists()

    aln = read_bam(bam, genome=genome, acquisition_id=1)
    assert aln.n_alignments >= 1
    assert "chr1" in set(aln.alignments.column("ref_name").to_pylist())
    # If we fed minimap2 the trimmed window, the CIGAR is clip-free
    # (e.g. "100M"). If we'd fed the full raw read, we'd see "30S100M30S"
    # — direct proof that the demux slicing took effect.
    cigars = aln.alignments.column("cigar_string").to_pylist()
    assert all("S" not in c for c in cigars), (
        f"expected no soft-clipping in CIGAR (window already trimmed); got {cigars}"
    )


@_skip_no_tools
def test_map_to_genome_caches_fasta(tmp_path: Path) -> None:
    """Genome FASTA is regenerated only on contig-count change."""
    from constellation.sequencing.align.map import map_to_genome

    genome = _make_genome()
    transcript = "ACGTACGTAC" * 10
    reads = [
        {
            "read_id": "r0", "acquisition_id": 1, "sequence": transcript,
            "quality": "I" * len(transcript), "length": len(transcript),
            "mean_quality": 40.0, "channel": None, "start_time_s": None,
            "duration_s": None, "dorado_quality": None, "read_group": None,
            "duplex_class": None,
        },
    ]
    demux = [
        {
            "read_id": "r0", "transcript_segment_index": 0, "sample_id": 1,
            "orientation": "+",
            "transcript_start": 0, "transcript_end": len(transcript),
            "score": 0.0, "is_chimera": False, "status": "Complete",
            "is_fragment": False, "artifact": "none",
        },
    ]
    demux_dir = _make_demux_dir(tmp_path, reads=reads, demux=demux)
    output_dir = tmp_path / "out"
    map_to_genome(demux_dir, genome, output_dir=output_dir, threads=1)
    fa_mtime = (output_dir / "genome.fa").stat().st_mtime
    map_to_genome(demux_dir, genome, output_dir=output_dir, threads=1)
    assert (output_dir / "genome.fa").stat().st_mtime == fa_mtime


@_skip_no_tools
def test_minimap2_build_index_skips_when_fresh(tmp_path: Path) -> None:
    from constellation.sequencing.align.minimap2 import minimap2_build_index

    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr1\n" + "ACGTACGTAC" * 100 + "\n")
    mmi = tmp_path / "g.mmi"
    minimap2_build_index(fasta, mmi, threads=1)
    first = mmi.stat().st_mtime
    minimap2_build_index(fasta, mmi, threads=1)
    assert mmi.stat().st_mtime == first


@_skip_no_tools
def test_minimap2_run_accepts_arbitrary_args(tmp_path: Path) -> None:
    """Generic runner stays use-case-agnostic — verify by passing
    map-ont (DNA) flags rather than splice."""
    from constellation.sequencing.align.minimap2 import (
        minimap2_build_index,
        minimap2_run,
    )

    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr1\n" + "ACGTACGTAC" * 100 + "\n")
    mmi = tmp_path / "g.mmi"
    minimap2_build_index(fasta, mmi, threads=1)
    fastq = tmp_path / "r.fq"
    fastq.write_text(f"@r0\n{'ACGTACGTAC' * 10}\n+\n{'I' * 100}\n")
    sam = tmp_path / "out.sam"
    out = minimap2_run(
        target=mmi,
        queries=[fastq],
        output_path=sam,
        args=("-ax", "map-ont"),
        threads=1,
    )
    assert out == sam
    assert sam.exists()
    assert sam.stat().st_size > 0


# ── orientation-aware transcript windows ───────────────────────────────
#
# transcript_start/end index the CHOSEN-orientation frame, but reads/
# stores the original strand (stages.py builds the reads shard from the
# raw parse, then runs demux on it). Slicing the stored bytes directly
# took the wrong strand over the wrong interval for every '-' read, so
# reverse-oriented reads clustered apart from the forward reads of the
# same transcript instead of joining them.


def _demux_one(seq: str, qual: str, read_id: str):
    """Run demux on a single read; return its resolved annotation."""
    import pyarrow as pa

    from constellation.sequencing.schemas.reads import READ_TABLE
    from constellation.sequencing.transcriptome.demux.demux import locate_segments
    from constellation.sequencing.transcriptome.demux.designs import CDNA_WILBURN_V1

    reads = pa.Table.from_pylist(
        [
            {
                "read_id": read_id,
                "acquisition_id": 0,
                "sequence": seq,
                "quality": qual,
            }
        ],
        schema=READ_TABLE,
    )
    _segments, _demux, results = locate_segments(reads, CDNA_WILBURN_V1)
    return results[0]


def test_reverse_oriented_read_yields_the_same_window_as_its_forward_twin():
    """The same molecule, sequenced both ways, must give one window.

    This is the whole point: if the two disagree, the reverse read forms
    its own cluster and the transcript's abundance is split in two.
    """
    import numpy as np
    import pyarrow as pa

    from constellation.sequencing.align.map import transcript_window_buffers
    from constellation.sequencing.transcriptome.demux import simulator as sim
    from constellation.sequencing.transcriptome.demux.demux import (
        _reverse_complement,
    )
    from constellation.sequencing.transcriptome.demux.designs import CDNA_WILBURN_V1

    rng = np.random.default_rng(7)
    seq, qual = sim.assemble_sequence(
        sim.ReadSpec(
            read_id="x",
            expected_status=sim.ReadStatus.COMPLETE,
            orientation="+",
            transcript_length=400,
            transcript_id="T1",
            polyA_length=30,
            polyA_artifact="clean",
            barcode_index=2,
        ),
        CDNA_WILBURN_V1,
        rng=rng,
    )
    rows = [("fwd", seq, qual), ("rev", _reverse_complement(seq), qual[::-1])]
    annotated = [(rid, s, q, _demux_one(s, q, rid)) for rid, s, q in rows]

    # Precondition: the two really did resolve to opposite orientations,
    # otherwise this test proves nothing.
    orientations = [a.annotation.orientation for _r, _s, _q, a in annotated]
    assert orientations == ["+", "-"], orientations

    batch = pa.RecordBatch.from_pylist(
        [
            {
                "read_id": rid,
                "sequence": s,
                "quality": q,
                "orientation": a.annotation.orientation,
                "transcript_start": a.annotation.transcript_start,
                "transcript_end": a.annotation.transcript_end,
            }
            for rid, s, q, a in annotated
        ]
    )
    buf, off = transcript_window_buffers(batch)
    windows = [bytes(buf[off[i] : off[i + 1]]).decode() for i in range(2)]
    assert windows[0] == windows[1]

    # And it is demux's own notion of the window, not merely self-consistent.
    fwd = annotated[0][3]
    assert windows[0] == fwd.chosen_sequence[
        fwd.annotation.transcript_start : fwd.annotation.transcript_end
    ]


def test_quality_window_is_reversed_but_not_complemented():
    """Quality travels with the bases but has no complement."""
    import numpy as np
    import pyarrow as pa

    from constellation.sequencing.align.map import transcript_window_buffers

    batch = pa.RecordBatch.from_pylist(
        [
            {
                "read_id": "r",
                "sequence": "AAACCCGGGTTT",
                "quality": "0123456789ab",
                "orientation": "-",
                "transcript_start": 0,
                "transcript_end": 12,
            }
        ]
    )
    seq_buf, seq_off = transcript_window_buffers(batch)
    q_buf, q_off = transcript_window_buffers(batch, column="quality", complement=False)
    assert bytes(seq_buf).decode() == "AAACCCGGGTTT"  # revcomp of itself here
    assert bytes(q_buf).decode() == "ba9876543210"
    assert int(seq_off[-1]) == int(q_off[-1]) == 12
    del np


def test_missing_orientation_column_is_refused(tmp_path):
    """A pre-orientation demux dir must error, not silently mis-slice."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from constellation.sequencing.align.map import _iter_demux_read_batches

    (tmp_path / "reads").mkdir(parents=True)
    (tmp_path / "read_demux").mkdir(parents=True)
    pq.write_table(
        pa.table({"read_id": ["a"], "sequence": ["ACGT"]}),
        tmp_path / "reads" / "part-00000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "read_id": ["a"],
                "sample_id": pa.array([1], pa.int64()),
                "transcript_start": pa.array([0], pa.int32()),
                "transcript_end": pa.array([4], pa.int32()),
                "status": ["Complete"],
                "is_fragment": [False],
                "is_chimera": [False],
            }
        ),
        tmp_path / "read_demux" / "part-00000.parquet",
    )
    with pytest.raises(ValueError, match="predates the `orientation` column"):
        list(_iter_demux_read_batches(tmp_path))
