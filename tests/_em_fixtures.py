"""Synthetic corpora for end-to-end EM runs that need no minimap2.

``three_gene_panel`` is built to exercise, in one small run, the three things
the evidence and replay tests have to see happen — a covariance split, a
merge between rounds and a final merge whose survivor is rebuilt — under ORF
seeding and the native E-step, neither of which launches minimap2. The tests
that use it assert those preconditions themselves, so the fixture cannot
quietly stop exercising them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODONS = ["GCT", "TGC", "GAT", "GAA", "TTT", "GGT", "CAT", "ATT", "AAA", "CTT"]


def random_orf(rng, n_codons: int) -> str:
    body = "".join(rng.choice(CODONS) for _ in range(n_codons - 1))
    return "ATG" + body + "TAA"


def random_utr(rng, n: int) -> str:
    """Stop-free in every frame would be overkill; the ORF caller takes the
    longest ATG→stop, which the long coding body wins."""
    return "".join(rng.choice(list("ACGT"), size=n))


def mutate(rng, seq: str, rate: float) -> str:
    """Substitutions, deletions and insertions at a combined ``rate``."""
    out = []
    for base in seq:
        r = rng.random()
        if r < rate * 0.6:
            out.append(rng.choice([b for b in "ACGT" if b != base]))
        elif r < rate * 0.8:
            continue
        elif r < rate:
            out.append(base)
            out.append(rng.choice(list("ACGT")))
        else:
            out.append(base)
    return "".join(out)


def write_demux(tmp_path: Path, rows, *, n_samples: int = 1) -> Path:
    """A demux directory the EM corpus reader accepts.

    ``rows`` are ``(read_id, window, dorado_quality)``; reads are spread over
    ``n_samples`` sample ids round-robin so per-sample quant is non-trivial.
    """
    lead, trail = "GGGGGGGGGG", "CCCCCCCCCC"
    reads = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sequence": [lead + r[1] + trail for r in rows],
            "quality": ["I" * (len(r[1]) + 20) for r in rows],
            "dorado_quality": pa.array([r[2] for r in rows], pa.float32()),
        }
    )
    demux = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "transcript_segment_index": [0] * len(rows),
            "sample_id": pa.array(
                [i % n_samples for i in range(len(rows))], pa.int64()
            ),
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
    d = Path(tmp_path) / "demux"
    (d / "reads").mkdir(parents=True)
    (d / "read_demux").mkdir(parents=True)
    pq.write_table(reads, d / "reads" / "part-00000.parquet")
    pq.write_table(demux, d / "read_demux" / "part-00000.parquet")
    return d


def three_gene_panel(tmp_path: Path, *, seed: int = 20261114) -> Path:
    """Three genes; gene C carries two haplotypes that differ only in its
    3' UTR, at three substitutions at least 60 nt apart (24:12 reads).

    Under ORF seeding each read's error-variant ORF seeds a template, so a
    transcript starts fragmented across several near-identical templates —
    which is what gives the merges something to do. Gene C's haplotypes share
    an ORF, so they land on one template, and only the M-step's covariance can
    separate them.
    """
    rng = np.random.default_rng(seed)
    genes = []
    for n_codons in (140, 160, 150):
        genes.append(
            random_utr(rng, 60) + random_orf(rng, n_codons) + random_utr(rng, 220)
        )
    c = genes[2]
    utr3 = len(c) - 220
    sites = [utr3 + 30, utr3 + 100, utr3 + 170]
    alt = list(c)
    for s in sites:
        alt[s] = {"A": "C", "C": "G", "G": "T", "T": "A"}[alt[s]]
    c_alt = "".join(alt)

    rows = []
    for g, (truth, n) in enumerate(zip(genes[:2], (40, 34))):
        rows += [
            (f"g{g}_r{i:02d}", mutate(rng, truth, 0.01), float(rng.uniform(18, 32)))
            for i in range(n)
        ]
    rows += [
        (f"gC_major_r{i:02d}", mutate(rng, c, 0.01), float(rng.uniform(18, 32)))
        for i in range(24)
    ]
    rows += [
        (f"gC_minor_r{i:02d}", mutate(rng, c_alt, 0.01), float(rng.uniform(18, 32)))
        for i in range(12)
    ]
    return write_demux(tmp_path, rows, n_samples=3)


#: The variant sites of gene C's minor haplotype, for tests that look for them.
GENE_C_SITE_SPACING = (30, 100, 170)
