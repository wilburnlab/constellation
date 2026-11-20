#!/usr/bin/env python
"""Time and memory of replaying one EM template, by size.

Replay (``em/replay.py``) re-runs the M-step for one template and the support
view then reads every member's alleles at every PWM column
(``consensus.member_alleles``). Both are per-template costs the transcriptome
browser pays when a node is opened, and nothing measured them: the
transcriptome-browser plan's "a few seconds, about 2 GiB at 20k members x
2 kb" was extrapolated from frame_consensus's +95 MiB per 1,000 x 2 kb.

Each size runs in a fresh subprocess so its peak RSS is its own. Members are
reads drawn from a template at 1% error (substitutions, deletions,
insertions), a quarter of them carrying a second haplotype at three linked
sites so the covariance pass has a split to find, aligned to the template
with the E-step's own two-pass aligner (as the final rebuild does). The
alignment is set up before timing starts: replay reads CIGARs the E-step
already wrote.

    python scripts/bench-em-replay.py                      # 1k / 5k x 1.5 / 3 kb
    python scripts/bench-em-replay.py --members 20000 --lengths 2000
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time


def _one(n_members: int, length: int, seed: int) -> dict:
    import numpy as np

    from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
        MemberSpec,
        member_alleles,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep import (
        refine_template_with_internals,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.rebuild import (
        align_to_survivor,
    )

    rng = np.random.default_rng(seed)
    template = "".join(rng.choice(list("ACGT"), size=length))
    alt = list(template)
    for site in (length // 4, length // 2, 3 * length // 4):
        alt[site] = {"A": "C", "C": "G", "G": "T", "T": "A"}[alt[site]]
    alt = "".join(alt)

    def mutate(seq: str) -> str:
        out = []
        for base in seq:
            r = rng.random()
            if r < 0.006:
                out.append(rng.choice([b for b in "ACGT" if b != base]))
            elif r < 0.008:
                continue
            elif r < 0.01:
                out.append(base)
                out.append(rng.choice(list("ACGT")))
            else:
                out.append(base)
        return "".join(out)

    members = []
    t0 = time.perf_counter()
    for i in range(n_members):
        seq = mutate(alt if i % 4 == 0 else template)
        placed = align_to_survivor(seq, template, identity_floor=0.9)
        if placed is None:
            continue
        cigar, t_start, q_start = placed
        members.append(
            MemberSpec(
                member_seq=seq,
                weight=1.0,
                cigar=cigar,
                centroid_is_query=False,
                ref_start=t_start,
                member_start=q_start,
                member_id=len(members),
            )
        )
    setup_s = time.perf_counter() - t0
    rss_setup = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t1 = time.perf_counter()
    nodes, internals = refine_template_with_internals(
        template,
        members,
        template_id=1,
        n_assigned=float(len(members)),
        **MStepParams().kernel_kwargs(),
    )
    kernel_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    alleles = member_alleles(internals.pooled, np.arange(internals.pooled.plan.n_columns))
    alleles_s = time.perf_counter() - t2
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "members": len(members),
        "length": length,
        "nodes": len(nodes),
        "setup_s": round(setup_s, 2),
        "kernel_s": round(kernel_s, 2),
        "alleles_s": round(alleles_s, 2),
        "alleles_mib": round(alleles.nbytes / 2**20, 1),
        # ru_maxrss is KiB on Linux.
        "rss_after_setup_mib": round(rss_setup / 1024, 0),
        "rss_peak_mib": round(rss_peak / 1024, 0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--members", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--lengths", type=int, nargs="+", default=[1500, 3000])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--one", nargs=2, type=int, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.one:
        print(json.dumps(_one(args.one[0], args.one[1], args.seed)))
        return 0
    cols = [
        "members", "length", "nodes", "kernel_s", "alleles_s", "alleles_mib",
        "rss_after_setup_mib", "rss_peak_mib",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---:|" * len(cols))
    for n in args.members:
        for length in args.lengths:
            proc = subprocess.run(
                [sys.executable, __file__, "--one", str(n), str(length), "--seed", str(args.seed)],
                capture_output=True,
                text=True,
                check=True,
            )
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            print("| " + " | ".join(str(row[c]) for c in cols) + " |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
