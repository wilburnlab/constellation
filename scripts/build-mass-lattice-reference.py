#!/usr/bin/env python
"""Build the reference the m/z axis compares a run's fold with.

Writes constellation/data/mass_lattice_reference.parquet: per 1 Th bin of m/z, the
Fourier coefficient at the lattice's period of theoretical peptide ions, as
fragments (1+ b and y ions, for MS2) and precursors (2+, at half the period, for
MS1). A run's phase minus this one's is the error of its axis. Peptide mass excess
is close to universal, which is why FineTune's C. elegans map served yeast, so one
table serves any run and no FASTA is needed at run time.

    python scripts/build-mass-lattice-reference.py --fasta UP000005640_9606.fasta

From a FASTA the ions are a tryptic digest (no missed cleavages, 7-30 residues,
carbamidomethyl C), y ions weighted 1 and b ions 0.5. With --library the fragments
come from an EncyclopeDIA library instead, each entry's peaks at their stored m/z
and intensity, normalised per entry.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from constellation.core.chem.elements import PROTON_MASS
from constellation.core.sequence.alphabets import _AA_RESIDUE_COMPS
from constellation.core.signal.phase import PhaseFold
from constellation.massspec.mz_axis import (
    MZ_REFERENCE_TABLE,
    REFERENCE_FILE,
    REFERENCE_FRAGMENTS,
    REFERENCE_PRECURSORS,
)

#: The mean spacing of peptide masses (Da), from a tryptic human digest; FineTune
#: used 1.00045475 from a C. elegans library.
PERIOD = 1.000495
LO, HI, BIN_WIDTH = 100.0, 2100.0, 1.0
WATER = 18.0105646837
CARBAMIDOMETHYL = 57.0214637

RESIDUES = {k: v.mass for k, v in _AA_RESIDUE_COMPS.items() if k in "GASPVTCLINDQKEMHFRYW"}
RESIDUES["C"] += CARBAMIDOMETHYL


def tryptic_peptides(fasta: Path) -> set[str]:
    seqs = [s.replace("\n", "") for s in re.split(r">.*\n", fasta.read_text())[1:]]
    return {
        p
        for s in seqs
        for p in re.sub(r"(?<=[KR])(?!P)", " ", s).split()
        if 7 <= len(p) <= 30 and set(p) <= RESIDUES.keys()
    }


def _rows(fold: PhaseFold, kind: str) -> list[dict]:
    rows = []
    for b in range(fold.n_bins):
        lo = LO + b * BIN_WIDTH
        c = fold.coefficient(lo, lo + BIN_WIDTH)
        if c != 0:
            rows.append({"kind": kind, "mz_bin": b, "re": c.real, "im": c.imag})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", type=Path, required=True, help="Tryptic digest source; always gives the precursors.")
    ap.add_argument("--library", type=Path, default=None, help="An EncyclopeDIA library for the fragments.")
    ap.add_argument("-o", "--output", type=Path, default=None)
    a = ap.parse_args()
    out = a.output or Path(__file__).resolve().parent.parent / "constellation" / "data" / REFERENCE_FILE

    peptides = tryptic_peptides(a.fasta)
    print(f"{len(peptides):,} tryptic peptides")
    fragments = PhaseFold(PERIOD, LO, HI, bin_width=BIN_WIDTH, n_blocks=1)
    precursors = PhaseFold(PERIOD / 2, LO, HI, bin_width=BIN_WIDTH, n_blocks=1)
    frag, weight, prec = [], [], []
    for p in peptides:
        c = np.cumsum([RESIDUES[r] for r in p])
        mass = c[-1] + WATER
        prec.append((mass + 2 * PROTON_MASS) / 2)
        frag += [c[1:-1] + PROTON_MASS, mass - c[:-1] + PROTON_MASS]
        weight += [np.full(len(p) - 2, 0.5), np.ones(len(p) - 1)]
    x = torch.tensor(prec, dtype=torch.float64)
    precursors.add(x, torch.ones_like(x))
    source = f"tryptic digest of {a.fasta.name}"
    if a.library is None:
        fragments.add(torch.from_numpy(np.concatenate(frag)), torch.from_numpy(np.concatenate(weight)))
    else:
        from constellation.massspec.io.encyclopedia._mirror import iter_table, open_library

        n = 0
        with open_library(a.library) as connection:
            for batch in iter_table(connection, "entries", columns=["MassArray", "IntensityArray"]):
                mz = batch.column("MassArray").to_pylist()
                intensity = batch.column("IntensityArray").to_pylist()
                xs, ws = [], []
                for m, i in zip(mz, intensity):
                    i = np.asarray(i, dtype=np.float64)
                    total = i.sum()
                    if total > 0:
                        xs.append(np.asarray(m, dtype=np.float64))
                        ws.append(i / total)
                        n += 1
                if xs:
                    fragments.add(torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ws)))
        print(f"{n:,} library entries")
        source = f"fragments of {a.library.name}, precursors of a tryptic digest of {a.fasta.name}"

    rows = _rows(fragments, REFERENCE_FRAGMENTS) + _rows(precursors, REFERENCE_PRECURSORS)
    table = pa.Table.from_pylist(rows, schema=MZ_REFERENCE_TABLE)
    meta = {"period": PERIOD, "lo": LO, "bin_width": BIN_WIDTH, "n_bins": fragments.n_bins, "source": source}
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"x.massspec.mz_reference": json.dumps(meta).encode()}
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")
    print(f"wrote {out}: {table.num_rows} rows, {out.stat().st_size / 1024:.0f} KiB ({source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
