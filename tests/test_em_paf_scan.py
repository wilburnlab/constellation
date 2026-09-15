"""The deferred-decode PAF scanner.

The scanner exists to avoid building twelve Arrow string columns for ~480M
rows per round when one row per read is kept. That is only worth doing if it
decodes the *same numbers* as the general reader, so the headline test here is
differential against ``readers/paf.py`` rather than against hand-written
expectations.

The second contract is chunk-independence: ``iter_hit_blocks`` cuts blocks
back to a read-group boundary and carries the remainder as raw bytes, so the
grouping it emits must not depend on how the pipe happened to split the
stream.
"""

from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
import pytest

from constellation.sequencing.readers.paf import read_paf
from constellation.sequencing.transcriptome.cluster.denovo.em.paf_scan import (
    iter_hit_blocks,
    scan_paf_block,
)


def _row(
    q,
    t,
    *,
    strand="+",
    as_score=2000,
    q_len=1200,
    q_start=0,
    q_end=1198,
    t_len=1300,
    t_start=5,
    t_end=1203,
    n_match=1150,
    aln_len=1200,
    mapq=60,
    cigar="100=1X99=",
    tags=True,
):
    fixed = "\t".join(
        str(x)
        for x in (
            q,
            q_len,
            q_start,
            q_end,
            strand,
            t,
            t_len,
            t_start,
            t_end,
            n_match,
            aln_len,
            mapq,
        )
    )
    if not tags:
        return fixed
    extra = ["tp:A:P", "NM:i:50", f"AS:i:{as_score}"]
    if cigar is not None:
        extra.append(f"cg:Z:{cigar}")
    return fixed + "\t" + "\t".join(extra)


def _block(rows: list[str]) -> np.ndarray:
    return np.frombuffer(("\n".join(rows) + "\n").encode(), dtype=np.uint8)


def test_matches_the_general_reader_field_for_field():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(200):
        cg = "".join(
            f"{int(rng.integers(3, 40))}{op}" for op in rng.choice(list("=XID"), 8)
        )
        rows.append(
            _row(
                i // 4,
                int(rng.integers(0, 50)),
                as_score=int(rng.integers(-500, 4000)),
                q_start=int(rng.integers(0, 30)),
                q_end=int(rng.integers(900, 1200)),
                t_start=int(rng.integers(0, 40)),
                t_end=int(rng.integers(900, 1300)),
                n_match=int(rng.integers(800, 1200)),
                aln_len=int(rng.integers(900, 1300)),
                cigar=cg,
            )
        )
    text = ("\n".join(rows) + "\n").encode()

    ref = read_paf(io.BytesIO(text))
    hb = scan_paf_block(np.frombuffer(text, dtype=np.uint8), n_templates=50)

    assert len(hb) == ref.num_rows  # all '+' and all templates in range
    assert hb.read_row.tolist() == [int(x) for x in ref.column("q_name").to_pylist()]
    assert hb.template_row.tolist() == [
        int(x) for x in ref.column("t_name").to_pylist()
    ]
    assert hb.as_score.tolist() == ref.column("as_score").to_pylist()

    all_hits = np.arange(len(hb))
    got = hb.int_fields(
        all_hits,
        (
            "q_len",
            "q_start",
            "q_end",
            "t_len",
            "t_start",
            "t_end",
            "n_match",
            "aln_len",
        ),
    )
    for name, values in got.items():
        assert values.tolist() == ref.column(name).to_pylist(), name

    assert hb.cigars(all_hits).to_pylist() == ref.column("cigar").to_pylist()


def test_reverse_strand_hits_are_dropped():
    """A '-' hit pairs a reverse CIGAR with a forward read; it cannot be used."""
    hb = scan_paf_block(
        _block([_row(0, 1), _row(0, 2, strand="-"), _row(1, 3)]), n_templates=10
    )
    assert hb.read_row.tolist() == [0, 1]
    assert hb.template_row.tolist() == [1, 3]
    assert hb.n_dropped_strand == 1


def test_out_of_range_template_is_dropped():
    hb = scan_paf_block(_block([_row(0, 1), _row(0, 99)]), n_templates=10)
    assert hb.template_row.tolist() == [1]
    assert hb.n_dropped_template == 1


def test_negative_as_score_parses():
    hb = scan_paf_block(_block([_row(0, 1, as_score=-317)]), n_templates=5)
    assert hb.as_score.tolist() == [-317]


def test_missing_cigar_is_null_not_empty():
    hb = scan_paf_block(_block([_row(0, 1, cigar=None), _row(0, 2)]), n_templates=5)
    cig = hb.cigars(np.arange(2))
    assert cig.to_pylist() == [None, "100=1X99="]


def test_row_without_tags_still_parses_its_fixed_fields():
    hb = scan_paf_block(_block([_row(3, 4, tags=False)]), n_templates=5)
    assert hb.read_row.tolist() == [3]
    assert hb.as_score.tolist() == [0]
    got = hb.int_fields(np.array([0]), ("mapq", "aln_len"))
    assert got["mapq"].tolist() == [60]
    assert got["aln_len"].tolist() == [1200]


@pytest.mark.parametrize("chunk", [7, 64, 512, 4096, 1 << 20])
def test_grouping_is_independent_of_chunk_size(chunk):
    """Blocks must carry partial groups as raw bytes, not decoded state."""
    rows = []
    for read in range(40):
        for k in range(3):
            rows.append(_row(read, read * 3 + k, as_score=1000 + k))
    text = ("\n".join(rows) + "\n").encode()

    chunks = [text[i : i + chunk] for i in range(0, len(text), chunk)]
    blocks = list(iter_hit_blocks(chunks, n_templates=200, block_bytes=chunk))

    reads, templates, scores = [], [], []
    for hb in blocks:
        # Every emitted block holds only complete groups.
        for r in np.unique(hb.read_row):
            assert (hb.read_row == r).sum() == 3
        reads.extend(hb.read_row.tolist())
        templates.extend(hb.template_row.tolist())
        scores.extend(hb.as_score.tolist())

    assert reads == [r for r in range(40) for _ in range(3)]
    assert templates == [r * 3 + k for r in range(40) for k in range(3)]
    assert scores == [1000 + k for _ in range(40) for k in range(3)]


def test_deferred_decode_after_a_carried_boundary():
    """int_fields/cigars must stay correct on a block built from carried bytes."""
    rows = []
    for read in range(12):
        for k in range(2):
            rows.append(_row(read, k, t_start=read * 10 + k, cigar=f"{read + 1}="))
    text = ("\n".join(rows) + "\n").encode()

    seen: list[tuple[int, int, str]] = []
    for hb in iter_hit_blocks(
        [text[i : i + 37] for i in range(0, len(text), 37)],
        n_templates=5,
        block_bytes=37,
    ):
        fields = hb.int_fields(np.arange(len(hb)), ("t_start",))
        cigs = hb.cigars(np.arange(len(hb))).to_pylist()
        seen.extend(zip(hb.read_row.tolist(), fields["t_start"].tolist(), cigs))

    assert seen == [
        (read, read * 10 + k, f"{read + 1}=") for read in range(12) for k in range(2)
    ]


def test_int_fields_on_a_subset_picks_the_right_rows():
    rows = [_row(0, i, t_start=100 + i, n_match=900 + i) for i in range(6)]
    hb = scan_paf_block(_block(rows), n_templates=10)
    sel = np.array([1, 4, 5])
    got = hb.int_fields(sel, ("t_start", "n_match"))
    assert got["t_start"].tolist() == [101, 104, 105]
    assert got["n_match"].tolist() == [901, 904, 905]
    assert hb.cigars(sel).to_pylist() == ["100=1X99="] * 3
