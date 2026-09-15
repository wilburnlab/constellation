"""ORF-anchored EM refinement for de novo transcriptome clustering.

The shipped de novo path groups whole reads by connected components over an
edit-distance-verified graph. That works, but it has nothing to anchor on:
exact replication of whole reads is 0.15%, so abundance ≈ 1 everywhere and
the roadmap's abundance-anchored greedy set-cover has no hubs. At the **ORF**
level exact replication is 30% of reads.

So ORFs are used as large, informed k-mers for *seeding only*, and the
clustering itself is an EM loop over whole reads:

===========  ===========================================================
stage        what it does
===========  ===========================================================
``seed``     longest sense ORF per read → dereplicate ORF nucleotides →
             one representative read per distinct ORF, whose full cDNA
             becomes that group's **template**
``fold``     tight clustering of the ORFs themselves (identity ~0.97,
             ±3 codons) by abundance-ordered radius-1 greedy set cover,
             classifying each verified pair from the verify columns
``estep``    every read aligned to every template (minimap2), assigned
             by score inside an absolute band with a ranked tie-break
``mstep``    per-template folded PWM → variants → haplotypes → per-
             haplotype consensus → ORF re-predicted under a support gate
``refine``   merge converged templates, prune unsupported ones, iterate
===========  ===========================================================

The product is reference transcripts carrying ORF nodes with per-sample read
support — a transcriptome. Proteoforms are a heuristic *inside* clustering,
not its target.
"""

from __future__ import annotations


__all__: list[str] = []
