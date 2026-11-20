# The EM clusterer: matching extents, and its hot spots (PR3)

**Status:** in progress on `feat/em-extent-speed` (2026-11-20). PR3 of the transcriptome-browser plan ([transcriptome-browser.md](transcriptome-browser.md)), renumbered by the PI on 2026-11-20. One commit per phase, one PR.

## Why

**Extent.** The transcriptome browser showed template #697 of the Nov2026 run (Pebp1, round 12) full of 3′-anchored reads. Their starts were 113 nt into exon 1 and 78 nt into exon 3, each right after a G run: the template-switching oligo priming inside the transcript.

The PI's principle (2026-11-20): **the EM's only task is to remove sequencing error.**
- A template is a representative transcript.
- A read joins one only when their extents match.
- Whether a shorter molecule is library prep, biology or degradation cannot be told from the read. It is answered after clustering, between templates, by the graph's `contained` relation.

The code had drifted from this in four places:
- **E-step admission** asked only for identity and an alignment covering half the *shorter* sequence. That guard was meant against slivers.
- **M-step node consensus** covered the union of its reads' spans, extended wherever 3 reads overhung it.
- **em-kmer seeding** left the 5′ overhang unbounded.
- **The likelihood** never charged a read for the template it leaves uncovered.

On round 12 of the Nov2026 run, 64% of the 9.42M assignments have an end more than 30 nt from the template's:

| | Share of assignments |
|---|---|
| Read starts more than 30 nt into the template | 57% |
| Read ends more than 30 nt before the template's end | 16% |
| Read reaches past the template | about 1% |

**Speed.** [docs/reviews/em-performance-2026-11-15.md](../reviews/em-performance-2026-11-15.md) ranks the loop's remaining hot spots:
- **Tier A:** exact, and large.
- **Tier B:** exact, and moderate.
- **Tier C:** changes results.

The exact items come first, while the golden digests can still show them bit-identical. The extent fix changes results on purpose.

## Decisions (PI, 2026-11-20: "a good first version")

1. **One end tolerance, 30 nt at each end, used everywhere:** seeding overhangs, E-step admission, the final rebuild, and the graph's `equivalent` relation and merge.
   - Read overhang and uncovered template both count, so the rule is symmetric.
2. **A node's ends are the median start and median end of its reads.** It is no longer the union of their spans plus 3-read extensions.
   - Under ±30 nt admission this seeks the modes: template ends settle on start-site and cleavage-site modes instead of creeping outward and losing their own reads.
3. **A read with no template of its extent seeds a new one** in the next round, so no read leaves the output for being a different length.
4. **Truncation is answered after clustering,** by `contained` edges. A flag for strand invasion (a 5′ end right after a G run) may annotate them; it is not part of the loop.

## Measuring

**`scripts/bench-em-loop.py`:**
- `build`: a synthetic corpus shaped like real data (80 genes, long-tailed depth, alternative first exons, haplotypes, random and sharp 5′ truncation, 3′ jitter, homopolymer-biased indels).
- `run`: the whole loop under cProfile, writing `bench.json` (stage seconds, top functions) and `digests.json` (every table the run wrote).
- `compare`: timings side by side, and every digest that differs.

**Baseline, `main` at `77f540f`:** 4,997 reads, 3 rounds, native E-step, one thread.

| Seeding | Wall | Top costs |
|---|---|---|
| kmer | 38.1 s | edlib 5.3 s, seeding verify 4.2 s, minimizer encoding 2.5 s, the native join 1.2 s, per-read projection 1.0 s |
| orf | 128.4 s | **the native join's expansion, 22.5 s** (not in the review, which profiled em-orf under minimap2); translation inside `best_sense_orf`, 36 s cumulative; covariance `_reestimate`, 4.4 s |

## Phases

### Exact (golden digests unchanged; each with an equivalence test against the code it replaces)

The whole exact part, every digest identical: **em-kmer 38.1 → 29.7 s (−22%), em-orf 128.4 → 49.7 s (−61%)**.

| Phase | What | Review item |
|---|---|---|
| P3.0 | `scripts/bench-em-loop.py`, and the baseline above | — |
| P3.1 | **The ORF finder:** one pass over the codons (`core.sequence.nucleic.longest_orf`) for `best_sense_orf` and demux's ORF calling, and a canonical fast path in `translate`. Seeding's finder is 12.8× faster (3.50 → 0.27 ms a transcript), demux's 7.2×. Whole loop: em-kmer 38.1 → 33.0 s, em-orf 128.4 → 78.1 s, digests identical | A1, A2 |
| P3.2 | **The finalist alignment** in fewer passes over its CIGAR: pieces merged only where they meet, the score trim's minima from `accumulate`, the statistics from the trim's contributions. Python share 0.29 → 0.22 s per 2,000 finalists; the rest is edlib. Arrays (the review's proposal) were measured twice as slow at ~100 ops a CIGAR | A3 |
| P3.3 | **The native join** sorts its expanded rows on one packed int64 key instead of a four-key `lexsort`: a 16M-row sub-chunk 10–27 → ~4 s. With P3.2: em-kmer 33.0 → 30.2 s, em-orf 78.1 → 58.0 s | new |
| P3.4 | **Split children reuse the pool's projections** (16,561 → 14,952 on em-orf). Projecting every member at once with arrays was measured slower and dropped | A4 |
| P3.5 | **Tier B:** covariance `_reestimate` (one weighted `bincount`) and `dense_states` (no per-entry dict); a block's `sample_id` / `dorado_quality` taken chunk-locally instead of copying the whole column (#14); the PAF tail gathered in a list, not re-concatenated per chunk (#15); the final outputs' read ids, representatives and roles in Arrow (#17). Seeding verify proved edlib-bound. With P3.4: em-kmer 29.7 s, em-orf 49.7 s | B |

### Extent (changes results; new golden digests)

| Phase | What |
|---|---|
| P3.6 | **`EmParams.end_tolerance` (30)** in every E-step reducer (minimap2, edlib, native):<br>• a read is admitted only with both ends within tolerance;<br>• refusals are `unassigned_reason = 'extent'`;<br>• candidates whose implied placement is already out of tolerance are dropped before the shortlist;<br>• the final rebuild admits the same way;<br>• the rule is stamped in `estep.json`, so an older run is refused on resume, not continued under another rule. |
| P3.7 | **Node ends** = median member start and end (`MStepParams.node_extent`: `median`, the new default; `union`, the old). Recorded in `params.json`, so replaying an older run still verifies. |
| P3.8 | **New templates from unmatched reads:** the reads a round leaves without a template of their extent are seeded (the run's own seeder, over those reads alone) into templates for the next round. Seeding's overhangs default to the tolerance. |
| P3.9 | **The graph's `equivalent` tolerance and the merge** read the same value. |
| P3.10 | **Tests and docs:**<br>• a gene with full-length, strand-invasion and short-3′-UTR reads ends as three clusters joined by `contained` edges;<br>• every assignment of every fixture run lies within tolerance;<br>• the golden digests are re-captured;<br>• `sequencing/CLAUDE.md` is corrected where it argued for 5′-truncated recruitment. |

## Not in this PR

- **Tier C.** Windowing the template as well as the read in the finalist's core alignment, and aligning fewer finalists per read, both change results and wait for real data.
- **`find_orfs`'s one-pass form.** Demux no longer calls it; `best_orf` and the public API still do.
- **The per-round ε refit.** Still deferred by decision.
