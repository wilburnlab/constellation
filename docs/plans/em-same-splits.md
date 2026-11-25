# The EM clusterer: a split the algorithm calls "the same" is never made, never protected, never written

**Status:** built on `fix/em-same-splits` (2026-11-24/25), one commit per phase. Prompted by the bench handoff `handoff_mstep_terminal_columns_and_invisible_splits.md` (Denovo_Transcriptome_Clustering/docs, 2026-11-23); P3 and P4 go beyond it, at the PI's decision. Real-data validation (a re-run of em_261120's configuration) is the PI's.

## Why

The PI opened byte-identical twin clusters in the transcriptome browser: em_261120 (main `6e47177`, PR #116's extent rule, 9.43M reads, 12 rounds) has 449 twin groups over 907 clusters. The handoff traced Tubb4b 1378/1379 and Prm2 1263/1272 to M-step splits on molecule ends.

**What "the same" means in the EM.** The graph's `equivalent` relation with **no edits**: identical sequence over the shared span, ends within the tolerance (`graph.effective_tolerance`: 30 nt, a tenth of the length below 300 nt). Two templates that are the same cannot be told apart by any read the extent rule admits to either.

Measured on em_261120 (read-only, the local copy):

| | |
|---|---|
| Round-12 split templates with at least one pair of same children | 17,109 of 23,893 (72%) |
| Round-12 same pairs refused by the kin guard alone | 114,781 (0 refused for any other reason) |
| — of which siblings of this round's split | 60,183 |
| — of which kin through earlier rounds | 54,598 |
| Reads in the nodes involved | 4.72M (half the corpus) |
| Output twin pairs that were same-but-kin-refused at round 12 | 223 of 467 |
| Output twin pairs that arose in the final stage | 244 of 467 (202 left apart by the single star pass, 42 that differed by 1–2 edits until the rebuild) |

Tubb4b `…5092` was not byte-identical at round 12: node 1 (1,548 nt) is a prefix of node 0 (1,550 nt), and 7 of its 8 nodes are identical over their overlap with ends at most 13 nt apart. Only the final rebuild made 1378 and 1379 byte-identical.

## Where the defects sit

```
 seed (em-kmer): reads ─► kmer clusters ─► one template per cluster
 ┌───────────────────────────── round r = 1 … R ─────────────────────────────┐
 │ E-step  each read → ONE template: identity ≥ 0.97, both ends within 30 nt  │
 │ M-step  pooled PWM → candidate columns (end columns got in here) ── [1]    │
 │         → covariance → SPLIT: one node per state-tuple (≤ 8)               │
 │         → each node's consensus, its reads' median ends ──────────── [2]   │
 │ graph   every pair of nodes: equivalent / contained, edits, end offsets    │
 │ merge   star groups; kin guard: nodes of one split never rejoin ───── [3]  │
 └───────────────────────────────────────────────────────────────────────────┘
 final:  graph → merge ONCE → REBUILD survivors (an M-step without the split)
         → compare clusters again (was a report only) ───────────────── [4]
```

- **[1]** The E-step decides which ends count as the same, but the reads it admits still end up to 30 nt apart. At an extension column a read anchored at that end votes a gap and a read that runs past votes its bases (`consensus._covers_junction`), so the column reads as a base-versus-gap allele and enters by the allelic route, coverage route off or not. Reads sharing an alternative start or cleavage site form a perfect clique.
- **[2]** A node keeps its reads' median extent, so the columns a split used can lie outside every child's emitted consensus.
- **[3]** The kin guard (each node's split origin, walked through the lineage; kin = same origin) refuses kin pairs so the edit-count merge cannot undo the M-step's allele splits, the split/merge cycle of 2026-09-24. Its only exemption was byte-identity, so same kin a few nt apart stayed apart in every later round, and the next M-step re-split them.
- **[4]** The final merge is one pass and the rebuild after it changes consensus; nothing merged after that.

## What was built

| Phase | Commit | Rule | Recorded as | Old runs |
|---|---|---|---|---|
| P1 | `7fbd6c8` | Terminal extension columns (`COL_EXT_5P` / `COL_EXT_3P`) are not candidates; interior insertion columns, first/last junction included, stay. `call_variants` already applied it. | `MStepParams.candidate_extensions = False`; `n_excluded_extension` | replay with extensions admitted |
| P2 | `94e0a96` | Children that are the same (consensus calls agree on every column both emit, ends within the merge's tolerance capped as the graph caps it) are one node, grouped as the merge groups (`guarded_set_cover`, heaviest first, group ends within the tolerance: sameness does not chain). | `MStepParams.fold_same = True`, `same_tol_5p/3p`, `same_short_tol_div` (set from `EmParams`); `n_folded_same`; `KernelInternals.node_of_survivor` | replay without the fold |
| P3 | `d0e1e9e` | Kin that are the same are exempt from the kin guard; kin an edit apart stay protected; `--merge-siblings` unchanged. | `MergePredicate.kin_exempt = "same"` (`EmParams.merge_kin_exempt`); `n_kin_refused_same` (graph stats, refine.json, final.json) | a predicate recorded without the field reads as `"identical"`, so resuming an old run is refused by name |
| P4 | `84437ce` | After the rebuild, the clusters are related as written (once, before the outputs) and those that are now the same are folded into the deepest, its consensus, ends and ORF kept, unless the survivor's ends refuse one of the absorbed reads (`rebuild.extent_refusals`). One pass is final because no consensus changes. | `EmParams.final_fold_same = True`; `folded_final.parquet`; final.json `same_fold`, `n_same_pairs`, `n_same_folded`, `n_same_refused_extent`, `n_same_remaining`; `cluster_nodes.parquet` `n_folded`, `folded_node_template_ids` | null columns; nothing to fold |
| fix | `ada5dd9` | Replay skips a counter a round did not record (`evidence.KERNEL_COUNTERS_ADDED`); found replaying em_261120 locally. | — | — |

Also in P4: the cluster-edge export re-decides each edge's `same_split_origin` and `mergeable` for the clusters as written (a survivor inherits the origins of what it absorbed and holds its reads); replay re-creates a cluster's consensus without its folded reads and lists them beside it; the support view shows them as members never aligned to its frame, as it does for a survivor whose rebuild failed; the report says what the fold did. The report flags a kin merge as the split/merge cycle only under `--merge-siblings`, and flags same kin kept apart under the old rule.

Repeating merge → rebuild passes at the end was considered and rejected (PI, 2026-11-24): it would be a third copy of the M-step's consensus step, each pass able to make new same pairs.

## On real data: the three forensic templates, replayed

Round 12 of em_261120, the stored inputs re-run through the kernel (`replay_template`), under the recorded rules and under P1 + P2:

| Template | Recorded rules | P1 + P2 |
|---|---|---|
| Tubb4b `…5092` (6,564 reads) | 8 nodes; signatures: 12 `EXT_5P` + template 585; 4 `EXT_3P`; an insertion at 486 | **1 node**; 19 extension columns excluded; 1 candidate (G6/G5 at template 492), no partner, no edge |
| Prm2 `…4923` (3,225 reads) | 8 nodes; one signature of 4 `EXT_5P`, template 38 and 43, 4 `EXT_3P` | **2 nodes** (2,867 / 358), on template 38 + 43 only |
| Prm2 `…4916` (6,967 reads) | 8 nodes; signatures: 6 `EXT_3P`; template 41 + 46; 2 `EXT_5P` | **1 node**: no candidate (below) |

Replays under the recorded rules verify (`…5092`, `…4923`); `…4916` differs only in one p-value at 1.6e-8 relative, numpy 2.4.4 here against 2.4.6 on OSC (below).

## Decisions

- **The kin guard spares same pairs** (PI, 2026-11-24), although the handoff had scoped a kin-guard change out: 54,598 of the refused pairs were kin through earlier rounds, which no M-step change reaches.
- **The final stage ends on a fold that changes no consensus** (PI, 2026-11-24), not on repeated merge → rebuild passes.
- **The candidacy family is the eligible columns** (PI, 2026-11-25). Extension columns leave before Benjamini–Hochberg, as the handoff prescribed. BH is step-up, and the extension columns' p ≈ 0 had been lowering the bar for marginal interior columns. Measured: `…4916`'s linked pair (template 41 / 46, minor 3.7% / 5.2%, p = 0.0043 / 0.0054, 176 carrier reads) passed at a cut of 0.00535 only because 8 extension columns ranked ahead of it; among the 4 interior testable columns alone nothing clears q = 0.01, so `…4916` does not split (`…4923` still splits on the same pair). Tubb4b keeps G6/G5 (p 1.4e-7) and loses the insertion at 486 (p 0.0017) and template 585 (p 0.0034). Candidacy no longer depends on how ragged a template's ends are.

## Open

- **Low power of the candidacy pre-filter for linked low-frequency variants.** A per-column test with overdispersion ρ = 0.01 caps the effective depth near 100, so 176 reads carrying two linked deletions at 2.5% give p ≈ 0.005 per column. The linkage is the real evidence, and the covariance test sees only candidates. A separate question from this work.
- **Tubb4b's G6/G5 column** (template 492, 15% minor, 41% vs 9% between 1379 and 1378, uniform across mice) is a candidate with no partner, so it cannot split: a lone column never earns a node, by design. Whether a second expressed copy should be separable is the handoff's open question, unchanged.
- **Replay across machines.** `FLOAT_RTOL` (1e-9) is tighter than the drift of a small beta-binomial p-value between numpy/scipy builds (1.6e-8 relative on `…4916`), so such a template reports a mismatch when replayed off the machine that ran it.
- **`match_rate` / `indel_rate`** are written as nulls in EM `cluster_membership.parquet` (`em/outputs.py`): a one-line gap the browser will meet.
- **Expected on the PI's re-run** (handoff §5, `slurm/138_em_extent.sbatch` with a new tag): twins ≈ 0; `n_kin_refused_same` 0 every round; `n_same_remaining` small (group-span refusals, inherent to a tolerance relation); fewer templates at the 8-node cap; fewer clusters and more reads per cluster, RefSeq proteins not lower. Also worth a look: `n_merge_undone_extent`, `unresolved`, and whether the fold changes the number of rebuilt survivors.
