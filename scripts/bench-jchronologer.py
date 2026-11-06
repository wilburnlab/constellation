"""Check Constellation's JChronologer predictors against a library EncyclopeDIA made.

Samples entries of an EncyclopeDIA ``.dlib`` predicted by JChronologer
(EncyclopeDIA's ``-convert -fastaToLibrary`` path, for example the human
proteome library on the lab's data drive), predicts the same precursors in
PyTorch through ``library.prediction`` and reports:

* retention time: the largest difference, in seconds;
* fragments: each stored peak matched to a predicted ion at 10 ppm, the
  peaks with no match, and the largest difference of max-normalised
  intensities (a peak where two ions share an m/z within 10 ppm is
  ambiguous and counted apart);
* throughput of each model at ``--threads``.

The library is opened read-only (``mode=ro&immutable=1``); the jar is not
run. The energy is EncyclopeDIA's: ``--nce`` scaled from
``--default-charge`` (``CollisionEnergy.scaled_from_reference``).

    python3 scripts/bench-jchronologer.py LIBRARY.dlib [--sample 3000] [--nce 30] [--threads 8]
"""

from __future__ import annotations

import argparse
import sqlite3
import time

import numpy as np

from constellation.core.sequence.proforma import format_proforma
from constellation.massspec.io.encyclopedia._codec import decode_f4, decode_f8
from constellation.massspec.io.encyclopedia._modseq import parse_encyclopedia_modseq
from constellation.massspec.library.prediction.assemble import Ladders
from constellation.massspec.library.prediction.base import Requests
from constellation.massspec.library.prediction.collision_energy import CollisionEnergy
from constellation.massspec.library.prediction.registry import default_predictors
from constellation.massspec.library.prediction.requests import unlabelled_sites
from constellation.massspec.peptide.ions import IonType, bond_position


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("library")
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--nce", type=float, default=30.0)
    parser.add_argument("--default-charge", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--throughput", type=int, default=20000, help="requests timed per model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.library}?mode=ro&immutable=1", uri=True)
    total = con.execute("select max(rowid) from entries").fetchone()[0]
    rng = np.random.default_rng(args.seed)
    ids = sorted(set(rng.integers(1, total + 1, max(args.sample, args.throughput)).tolist()))
    rows = []
    for k in range(0, len(ids), 900):
        chunk = ids[k : k + 900]
        rows += con.execute(
            "select PeptideModSeq, PrecursorCharge, RTInSeconds, MassArray, IntensityArray from entries "
            f"where rowid in ({','.join('?' * len(chunk))})",
            chunk,
        ).fetchall()
    print(f"{len(rows)} entries sampled of {total}")

    proforma = [format_proforma(parse_encyclopedia_modseq(r[0])) for r in rows]
    peptides = sorted(set(proforma))
    index = {p: i for i, p in enumerate(peptides)}
    sites = tuple(unlabelled_sites(p)[0] for p in peptides)
    charge = np.array([int(r[1]) for r in rows], np.int32)
    energy = CollisionEnergy.scaled_from_reference(args.nce, args.default_charge).resolve(charge)
    requests = Requests(sites, np.array([index[p] for p in proforma]), charge, energy)
    predictors = default_predictors("jchronologer", ("retention_time", "fragments"), threads=args.threads)
    for prop, predictor in predictors.items():
        refused = predictor.check(requests).refused
        if refused.any():
            print(f"{prop}: {int(refused.sum())} entries refused")

    check = np.arange(min(args.sample, len(rows)))
    sub = requests.take(check)
    rt = predictors["retention_time"].predict(sub).values
    stored_rt = np.array([float(rows[i][2]) for i in check])
    print(f"RT: max |Constellation - library| {np.max(np.abs(rt - stored_rt)):.3g} s over {len(check)} entries")

    frags = predictors["fragments"].predict(sub)
    used = sorted({int(p) for p in sub.peptide})
    local = {p: i for i, p in enumerate(used)}
    ladders = Ladders([sites_peptidoform(peptides[p]) for p in used], (IonType.B, IonType.Y), 3)
    unmatched = ambiguous = exact = 0
    worst = 0.0
    for e, i in enumerate(check.tolist()):
        masses, stored = decode_f8(rows[i][3]), decode_f4(rows[i][4])
        sel = frags.request == e
        pep = local[int(sub.peptide[e])]
        length = len(sites[int(sub.peptide[e])].residues)
        t, n, c, v = frags.ion_type[sel], frags.ion_number[sel], frags.fragment_charge[sel], frags.intensity[sel]
        if not len(v):
            unmatched += len(masses)
            continue
        position = np.asarray(bond_position(t, n.astype(np.int64), length))
        mz = ladders.mz(np.full(len(t), pep), t, position, c)
        ours = v / v.max()
        theirs = stored / stored.max()
        diffs = []
        for m, s in zip(masses, theirs):
            near = np.flatnonzero(np.abs(mz - m) / m < 10e-6)
            if not len(near):
                unmatched += 1
            elif len(near) > 1:
                ambiguous += 1
            else:
                diffs.append(abs(ours[near[0]] - s))
        if diffs:
            worst = max(worst, max(diffs))
            exact += max(diffs) < 1e-5
    print(
        f"fragments: worst normalised intensity difference {worst:.3g}; {exact}/{len(check)} entries within 1e-5; "
        f"{unmatched} stored peaks unmatched, {ambiguous} ambiguous (two ions within 10 ppm)"
    )

    n = min(args.throughput, len(rows))
    timed = requests.take(np.arange(n))
    for prop, predictor in default_predictors("jchronologer", threads=args.threads).items():
        start = time.perf_counter()
        predictor.check(timed)
        predictor.predict(timed)
        seconds = time.perf_counter() - start
        print(f"{prop:15s} {n / seconds:9.0f} requests/s ({args.threads} threads, tokenizing included)")
    return 0


def sites_peptidoform(text: str):
    from constellation.core.sequence.proforma import parse_proforma

    return parse_proforma(text)


if __name__ == "__main__":
    raise SystemExit(main())
