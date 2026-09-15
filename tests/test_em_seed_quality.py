"""Quality-aware seed selection.

The seed read is the frame the whole M-step is built on, and seeding ignored a
free 4x accuracy signal. Over 8.3M reads the top decile by ``dorado_quality``
has median error 0.00114 against 0.00501 for the top decile by *length* — which
is 1.10x, slightly worse than choosing at random. Length and accuracy are
independent (Spearman +0.015 to +0.049 in every length stratum), so quality can
be a floor and length the objective without trading one against the other.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from constellation.sequencing.transcriptome.cluster.denovo.em.seed import (
    SEED_QUALITY_FLOOR,
    RepCandidates,
    extract_seed_orfs,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.seed import (
    REPRESENTATIVE_POLICIES,
)

_POLICY = REPRESENTATIVE_POLICIES["longest-above-quality"]


def _cand(lengths, quality):
    n = len(lengths)
    return RepCandidates(
        template_length=np.asarray(lengths, dtype=np.int64),
        orf_start=np.zeros(n, dtype=np.int64),
        abundance=np.ones(n, dtype=np.int64),
        median_length=np.full(n, int(np.median(lengths)), dtype=np.int64),
        quality=np.asarray(quality, dtype=np.float64),
    )


def test_longest_read_clearing_the_floor_wins():
    rank = _POLICY(_cand([500, 3000, 1200], [30.0, 25.0, 28.0]))
    assert int(np.argmin(rank)) == 1


def test_a_long_read_below_the_floor_loses_to_a_shorter_clean_one():
    """The floor is the point: length selects accuracy not at all."""
    rank = _POLICY(_cand([3000, 900], [9.0, 26.0]))
    assert int(np.argmin(rank)) == 1


def test_when_nothing_clears_the_floor_the_best_read_wins():
    """Fall back to highest quality — with no group-level branch."""
    q = [SEED_QUALITY_FLOOR - 10, SEED_QUALITY_FLOOR - 1, SEED_QUALITY_FLOOR - 5]
    rank = _POLICY(_cand([3000, 400, 1500], q))
    assert int(np.argmin(rank)) == 1


def test_every_clearing_row_outranks_every_failing_row():
    rank = _POLICY(_cand([10, 100_000], [SEED_QUALITY_FLOOR, SEED_QUALITY_FLOOR - 0.1]))
    assert rank[0] < rank[1], "a 10 nt clean read still beats a 100 kb dirty one"


def test_missing_quality_degrades_to_the_existing_tie_breaks():
    """A demux dir predating the qs:f tag must not error or invert."""
    rank = _POLICY(_cand([500, 3000], [-1.0, -1.0]))
    assert rank[0] == rank[1], "all rows fail equally; length breaks it downstream"


# ── through extract_seed_orfs ─────────────────────────────────────────


def _orf(aa: int, *, stop: str = "TAA") -> str:
    """An ATG-started sense ORF of `aa` residues plus a stop."""
    return "ATG" + "GCT" * (aa - 1) + stop


def _reads_table(rows):
    """rows = [(read_id, sequence, quality)]"""
    return pa.table(
        {
            "read_id": pa.array([r[0] for r in rows], pa.string()),
            "sequence": pa.array([r[1] for r in rows], pa.large_string()),
            "sample_id": pa.array([0] * len(rows), pa.int64()),
            "dorado_quality": pa.array([r[2] for r in rows], pa.float32()),
        }
    )


def test_seeding_elects_the_longest_clean_cdna_for_an_orf():
    """Same ORF, three cDNAs: the longest one clearing Q22 becomes the frame."""
    orf = _orf(40)
    reads = _reads_table(
        [
            ("short_clean", "AAAA" + orf + "TTTT", 30.0),
            ("long_dirty", "A" * 300 + orf + "T" * 300, 8.0),
            ("mid_clean", "A" * 60 + orf + "T" * 60, 27.0),
        ]
    )
    seed, _ = extract_seed_orfs(
        reads, min_aa_length=30, representative="longest-above-quality"
    )
    assert seed.num_rows == 1
    assert seed.column("representative_read_id").to_pylist() == ["mid_clean"]
    assert seed.column("n_reads").to_pylist() == [3]


def test_seeding_falls_back_when_no_read_clears_the_floor():
    orf = _orf(40)
    reads = _reads_table(
        [
            ("dirty_long", "A" * 300 + orf + "T" * 300, 7.0),
            ("less_dirty_short", "AA" + orf + "TT", 19.0),
        ]
    )
    seed, _ = extract_seed_orfs(
        reads, min_aa_length=30, representative="longest-above-quality"
    )
    assert seed.column("representative_read_id").to_pylist() == ["less_dirty_short"]


def test_seeding_without_a_quality_column_still_works():
    orf = _orf(40)
    reads = pa.table(
        {
            "read_id": pa.array(["a", "b"], pa.string()),
            "sequence": pa.array(
                ["AA" + orf + "TT", "A" * 100 + orf + "T" * 100], pa.large_string()
            ),
            "sample_id": pa.array([0, 0], pa.int64()),
        }
    )
    seed, _ = extract_seed_orfs(
        reads, min_aa_length=30, representative="longest-above-quality"
    )
    assert seed.num_rows == 1
    # Nothing clears; the length tie-break picks the longer cDNA.
    assert seed.column("representative_read_id").to_pylist() == ["b"]


def test_the_named_representative_is_the_best_quality_read_of_its_cdna():
    """Reads sharing an exact cDNA are identical, so name the cleanest one."""
    orf = _orf(40)
    same = "AA" + orf + "TT"
    reads = _reads_table([("noisy", same, 12.0), ("clean", same, 33.0)])
    seed, _ = extract_seed_orfs(
        reads, min_aa_length=30, representative="longest-above-quality"
    )
    assert seed.column("representative_read_id").to_pylist() == ["clean"]


def test_existing_policies_are_unaffected():
    orf = _orf(40)
    reads = _reads_table(
        [
            ("a", "A" * 300 + orf + "T" * 300, 5.0),
            ("b", "AA" + orf + "TT", 35.0),
        ]
    )
    seed, _ = extract_seed_orfs(
        reads, min_aa_length=30, representative="longest-template"
    )
    assert seed.column("representative_read_id").to_pylist() == ["a"]
