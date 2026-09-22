"""ORF-anchored EM refinement for de novo transcriptome clustering.

The shipped de novo path groups whole reads by connected components over an
edit-distance-verified graph. That works, but it has nothing to anchor on:
exact replication of whole reads is 0.15%, so abundance ≈ 1 everywhere and
the roadmap's abundance-anchored greedy set-cover has no hubs. At the **ORF**
level exact replication is 30% of reads.

So ORFs are used as large, informed k-mers for *seeding only*, and the
clustering itself is an EM loop over whole reads.

**Seeding has two implementations, and only the first stage differs.** Both
emit the same ``TEMPLATE_TABLE``, so everything downstream is identical:

===============  =======================================================
stage            what it does
===============  =======================================================
``seed`` +       ``--mode em-orf``: longest sense ORF per read →
``fold``         dereplicate ORF nucleotides → one representative read
                 per distinct ORF, then tight clustering of the ORFs
                 themselves (identity ~0.97, ±3 codons) by
                 abundance-ordered radius-1 greedy set cover
``seed_kmer``    ``--mode em-kmer``: the ``--mode kmer`` kernels on the
                 READS — dereplicate → minimizers → candidates → verify
                 (5' unbounded, 3' ≤ 100) → connected components — then
                 one elected read per cluster. No ORF: this partition
                 does not consult one, and asserting a seed ORF would
                 relax the M-step's support gate for nothing
``elect``        shared by both: rank a seed group's distinct cDNAs by
                 the ``--seed-representative`` policy and take the first
``estep``        every read aligned to every template (minimap2),
                 assigned by an absolute identity floor with a ranked
                 tie-break
``mstep``        per-template folded PWM → variants → haplotypes → per-
                 haplotype consensus → ORF re-predicted under a support
                 gate, with **no minimum protein length** — the floor
                 leaked into the next round's certified interval
``refine``       carry recruiting templates forward, iterate
===============  =======================================================

Measured at 9.39M reads, kmer seeding is **4.4x fewer templates** (850,450 vs
3,778,760) in **4.9x fewer bases**, turning round 1's E-step from 13.95 h /
127 GB into 2.89 h / 26 GB while assigning *more* reads (98.6% vs 96.2%) at
the same gene agreement (0.980 vs 0.981). It also takes the ``-N`` cap-hit
rate from 11.0% to 0.05% — the condition ``rounds.py`` itself calls unsound
rankings. It gives up 2 points of transcript agreement. The default is still
``em-orf`` until the multi-round comparison that justifies a flip has run.

The product is reference transcripts carrying ORF nodes with per-sample read
support — a transcriptome. Proteoforms are a heuristic *inside* clustering,
not its target.
"""

from __future__ import annotations


__all__: list[str] = []
