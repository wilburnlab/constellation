# The EM transcriptome clusterer, reviewed for a browser (2026-11-14)

**Scope.** `constellation/sequencing/transcriptome/cluster/denovo/em/` on `main` at `a633080`, read for one purpose: to set down what the loop does, what each stage produces, what reaches disk and what does not, before a transcriptome browser is designed on top of it ([docs/plans/transcriptome-browser.md](../plans/transcriptome-browser.md)). The bench notebook (`/mnt/e/Constellation_Optimization/Denovo_Transcriptome_Clustering/docs/log.md`, entries 2026-09-05 → 2026-10-07, and the ledger) was read for the questions the browser has to answer; §5 lists them.

This is not a correctness review of the statistics. The findings in §4 are about what the code makes visible, plus four defects found along the way. Nothing here has been changed.

---

## 1. The loop in one picture

```
demux windows ──► corpus/ ──► seed/ ──► rounds/r01 … rNN ──► final merge + rebuild ──► outputs
                  (sorted      (orf |     │
                   by read_id)  kmer)     │  templates ─► E-step ─► assignments ─► M-step ─► nodes + node membership
                                          │      ▲                                              │
                                          │      └──── refine (lineage; merge) ◄─── graph ◄─────┘
```

Each round is `_SUCCESS`-gated. The loop stops at `--rounds` (default 6) or when `churn.frac_unsettled < 0.005`.

## 2. Stage by stage: what it computes, what it produces, and whether that is kept

Each stage below lists its computation, its data structures, and what reaches disk. "Kept" means written under the run directory.

### 2.0 Corpus ([em/corpus.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/corpus.py))

- **Input.** The demux Complete windows, with windows longer than `--max-window-length` (15 kb) dropped.
- **Order.** Sorted by `read_id`, then by `sequence` and `sample_id`. This is the determinism fix of ledger #59 and #61.
- **Identity.** The **corpus row (`read_row`) is the read's identity everywhere downstream.** FASTA records are named by row, and PAF names parse as integers.
- **Kept.**
  - `corpus/reads.arrow`, mmapped by every worker, with columns `read_id`, `sequence`, `sample_id`, `dorado_quality`.
  - `reads.fa`, `stats.json` and `settings.json`.
  - Under `--estep-aligner native`, also `corpus/minimizers/`.
- **Not carried:** per-base qualities. The log (2026-10-03) noted this.

### 2.1 Seeding: the round-1 templates

Two seeders produce the same `TEMPLATE_TABLE`.

**em-orf** ([seed.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/seed.py), [fold.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/fold.py)):
1. Find the longest sense ORF of each distinct cDNA.
2. Dereplicate the ORF nucleotides exactly.
3. Elect one read per distinct ORF, using `em/elect.py`.
4. Fold the ORFs by radius-1 greedy set cover.

This writes `SEED_ORF_TABLE` and a read → ORF map.

**em-kmer** ([seed_kmer.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/seed_kmer.py)):
1. Run the `--mode kmer` kernels on the reads: dereplicate, minimizers, anchor-star candidates, then edlib verify. Ends are 5′ unbounded and 3′ ≤ 100.
2. Group by connected components, or by greedy set cover.
3. Elect one read per cluster.

This writes `READ_CLUSTER_MAP` (read → seed cluster).

**Output:** `TEMPLATE_TABLE`. Its columns are:
- `template_id = (round << 40) | row`
- `sequence`
- `node_weight`
- `orf_replication`
- `seed_read_quality`
- `seed_read_row`
- `declared_variants`

**Kept:** `seed/templates.parquet`, then either `seed_orfs.parquet` or `read_cluster.parquet`, then `stats.json` and `params.json`.

The seed clusters are the **round-0 genealogy**: the ancestors of every round-1 template. Under em-kmer the map from read to seed cluster is kept. The history view needs it (plan §5.4).

### 2.2 E-step: assign every read to at most one template

The E-step code is [assign.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/assign.py), [scheduler.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/scheduler.py), [likelihood.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/likelihood.py), [native.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/native.py) and [realign.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/realign.py).

**1. Candidates.** There are three ways to get them:

| Aligner | How candidates are produced |
|---|---|
| minimap2 (default) | `-N 500`, `-c` CIGAR on every hit |
| `--estep-aligner edlib` | minimap2 chains form a shortlist (k = 32); edlib aligns the shortlist |
| `--estep-aligner native` | Constellation's own minimizer join supplies the shortlist; edlib aligns it |

**2. Pool (admission).** A candidate enters the pool when `n_match / aln_len ≥ p_floor`.
- `p_floor` is 0.97. With `--p-floor-quality-scale` it is eased per read.
- Under the two-pass aligners, the alignment must also cover half the shorter of the read and the template.

**3. Ranking.**
- **Round 1 follows the seeder's rule.**
  - em-orf ranks on replication, then on seed quality.
  - em-kmer ranks on identity first. A better-replicated seed can take the read only within the noise band `z·sqrt(L·e(1−e))/L`.
- **Round 2 onward** ranks on a log-likelihood over the *differing* positions only.
  - ε depends on context.
  - Indels are charged per run.
  - Abundance can override this only inside a narrow window: Δlogl ≤ 5 nats **and** 20× the support.

**Output:** `EM_ASSIGNMENT_TABLE`, one row per corpus read. A read with no admitted candidate gets `template_id = −1` and an `unassigned_reason`. The row carries:
- **The winner's alignment:** `cigar`, `t_start/t_end`, `q_start/q_end`.
- **Scores:** `as_score/as_delta`, `logl/logl_delta`.
- **Candidate-pool counters:** `n_hits`, `n_admitted`, `candidate_cap_hit`, `shortlist_truncated`, `chain_score`.
- **Fit:** `identity`, `aligned_len`.
- **The read's `sample_id`.**

**Kept:** `rNN/assignments/part-*.parquet` and `round.json` (E-step counters, `n_newly_lost`).

**Not kept: who came second.** The row does not say which template the read nearly went to, and the shortlist is not stored either. Finding F6. (`as_delta` and `logl_delta` are not margins to a runner-up: each is the winner's *shortfall from the best* candidate on that score, zero whenever the winner is the best. Corrected after review; see F6.)

### 2.3 M-step: split each template on covarying columns and rebuild its consensus

**Orchestration** is [mstep_pool.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/mstep_pool.py). Templates are bin-packed into units, largest cost first. Each worker mmaps the corpus and the templates itself, and the units run in a fork pool with no torch below the fork.

**Kernel:** `refine_template` ([mstep.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/mstep.py)). For one template it runs:

| Step | Function | Data structure | Shape |
|---|---|---|---|
| a. members | `specs_from_assignment_slice` | `MemberSpec` per read (the E-step CIGAR, offsets) | ≤ 20,000, a uniform sample seeded by `template_id`; the unsampled tail is given to the major node |
| b. pooled consensus | `consensus.frame_consensus` | `ConsensusResult`: `ColumnPlan` (template T → frame F, insertion blocks reserved from the members' own CIGARs), `pwm` (F × 5 float64: A, C, G, T, gap), `winner` (F,), the maps `cons_of_frame` / `frame_of_cons` / `template_of_frame`, `column_kind`, per-member projections | F ≈ template length + insertion columns |
| c. column statistics | `columns.column_stats` | `ColumnStats`, all (F,): depth, base coverage, major/minor allele, minor fraction, uncovered fraction (global and local), homopolymer run, ε, boundary mass and its locally uniform expectation | — |
| d. candidates | `columns.candidate_columns` | `CandidateSet` (V,): PWM column, route (allelic / coverage), exact run collapse (`run_len`), effect, ε, major allele, BH p-value, cap counts | V ≤ 512 |
| e. read states | `consensus.member_allele_events` → `covariance.read_states` | `AlleleEvents`, then `ReadStates`: sparse CSR of each read's departures from the major allele over V, with the read's observed range | nnz ≈ 0–17 per read |
| f. covariance | `covariance.covariance_graph` | `CovarianceGraph`: edges (u, v) with **p-values only**, a `keep` mask, BH cut, budget drops. The test is two-sided: χ² for dense pairs, exact hypergeometric for sparse ones. Coverage × coverage pairs are never tested. | E edges |
| g. signatures | `covariance.quasi_cliques(gamma=0.6)` | candidate-index sets (edge-seeded greedy quasi-cliques) | — |
| h. patterns | `covariance.resolve_signature` | `Assignment` per signature: `patterns` (K × S), `mass`, `labels` (a pattern per read), **`margin` (the best minus second-best log-likelihood per read)**. Patterns are grown from the worst-explained read under a BIC penalty `(S+1)·ln N`. | — |
| i. nodes | state tuples, `_fold_small_tuples` | each read's tuple of labels; tuples with ≥ `min_node_reads` (2) are kept, at most `max_nodes` (8); the rest go to the nearest survivor by ε-weighted Hamming distance | — |
| j. per-node consensus | `frame_consensus(plan=pooled.plan)`, `_support_span`, `best_sense_orf` | the node's consensus trimmed to the span its reads cover; `declared_variants` in the node's own coordinates; `allele_string` (`=` major, `.` unobserved, `~` uncovered, base/gap); ORF annotation | — |

**Output** (`REFINED_NODE_TABLE` and `NODE_MEMBERSHIP_TABLE`):
- **Nodes:** `consensus`, `n_reads`, `node_weight`, `protein`, `orf_*`, `allele_string`, `declared_variants`, the extension/trim/insertion counts, `n_members_used`, `subsample_fraction`.
- **Membership:** `(round, parent_template_id, haplotype_id, read_row)`.

**Kept:** `rNN/mstep/nodes/` and `rNN/mstep/node_membership/`.

**Everything in steps b–i is discarded** when the worker returns. That includes the PWM, the column plan, the column statistics, the candidates, the read states, the covariance edges, the signatures, the patterns, the labels and margins, and the per-template diagnostic dictionary (findings F1, F2).

### 2.4 Template graph: how one round's nodes relate to each other

Code: [graph.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/graph.py).

**Candidates.** An uncapped k19/w19 containment probe join:
- ≤ 16 probes per sequence, stratified by position.
- A windowed diagonal count decides which pairs to keep.
- An antisense filter removes reverse-complement pairs.

**Kernel: `relate_pair`.**
1. Two edlib infix passes.
2. A fold: what both sequences overhang at one end is charged as edits.
3. One global (NW) alignment of the two cores.

**Classification** stops at the first rule a pair fails:

no placement → terminal divergence → staggered → below floor → internal variant → **equivalent** / **contained** (5′ / 3′ / both).

Only the last two are written as edges. The other outcomes appear only as counts in `stats.json`.

**Output:** `TEMPLATE_EDGE_TABLE`:
- Relation and truncation.
- The lengths, overhangs and deltas at each end.
- `n_edits` and its split into mismatch / insert / delete.
- Identity and shared probes.
- `same_split_origin`.
- `mergeable`.

**Kept:** `rNN/graph/{edges.parquet, stats.json}`, cached under a stamp.

### 2.5 Refine: next round's templates, lineage and merge

Code: [refine.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/refine.py).

**Next templates.** `next_templates` makes one template per node.
- Ids are `((r+1) << 40) | row`, assigned over the node rows.
- Zero-length consensuses are dropped *after* the ids are assigned.

**Lineage.** `LINEAGE_TABLE` records `(round, child, parent, rule, n_reads)`. `rule` is one of `carry`, `split`, `unrecruited` or `merge`.

**Merge.** On by default, at K = 2. Two nodes merge when all of these hold:
- the pair is `equivalent`;
- `n_edits ≤ 2`;
- both ends are within tolerance;
- they are not kin by split origin (walked back through the lineage);
- each has at least `merge_min_reads`.

Groups are formed by a guarded set cover, and the best-supported member survives. Merges are recorded in `MERGED_TABLE`, with the survivor's signed reach at each end.

**Churn.** `measure_churn` scatters both rounds' assignments by `read_row`. A move from a parent to its own child is *inherited*, not counted as a switch. `frac_unsettled` is what the stopping rule reads.

**Kept:** `rNN/lineage.parquet`, `merged.parquet`, `refine.json`, `round.json` (churn).

### 2.6 The final stage

Code: [rounds.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/rounds.py) `_final_graph_and_merge`, [rebuild.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/rebuild.py) and [outputs.py](../../constellation/sequencing/transcriptome/cluster/denovo/em/outputs.py).

**Steps:**
1. Build the last round's graph.
2. Merge under the same predicate as between rounds.
3. Rebuild each survivor that absorbed something:
   - re-align the pooled reads with `align_finalist`;
   - build one consensus with `pooled_node`;
   - call the ORF on it.
4. Relate the clusters again into `graph_final/` whenever anything was rebuilt.

**Outputs.**
- `clusters.parquet` (`TRANSCRIPT_CLUSTER_TABLE`, `mode='em'`). `cluster_id` is the row among the kept nodes.
- `cluster_membership.parquet`.
- `feature_quant.parquet`: per-sample counts and TPM, with sample ids taken from the assignments.
- `cluster.fa` and `proteins.fasta`, with records named `cluster_<row>`.
- `cluster_edges.parquet`.
- `final.json`, `merged_final.parquet` and `manifest.json`.
- `diagnostics/report.md`, with sections convergence, seeding, candidate pool, assignment rule, reference drift, cluster sizes and template graph.

## 3. Identities and coordinate systems

A browser has to translate between all of these, and some of the joins are implicit today:

| Identity | Scope | Defined by | Joins to |
|---|---|---|---|
| `read_row` | run | corpus row | `read_id` (corpus), every per-round table |
| `template_id` | run | `(round << 40) \| row` | `template_row` per round; lineage; edges |
| node key `(round, parent_template_id, haplotype_id)` | round | M-step emission | next round's `template_id` by **row in node order** (`in_node_order`) |
| `cluster_id` | run | row among the final nodes the merge kept | **nothing persisted** (F3) |
| seed cluster | run | `read_cluster.parquet` (em-kmer) | round-1 templates via the elected read |
| T / F / C columns | node | template string / PWM frame (with insertion blocks) / consensus (gap-winning frame columns removed, then flank-trimmed by `lo`) | `ColumnPlan`, `cons_of_frame`, `n_trimmed_5p`; **only in memory** |
| `sample_id` | experiment | demux `samples/` | assignments, `feature_quant` |

## 4. Findings

Severity follows the repo's review convention: **P1** gives wrong results, **P2** makes something wrong invisible, **gap** is a missing capability the browser needs.

**Status** (branch `feat/em-evidence`, PR1 of [the plan](../plans/transcriptome-browser.md)):

| # | Status |
|---|---|
| F1 | Fixed: compact evidence always (`mstep/columns`, `mstep/covariance`, `mstep/reads`); dense evidence by verified replay (`em/replay.py`, `em/support.py`) |
| F2 | Fixed: `mstep/templates` and `mstep/disagreements`, every round |
| F3 | Fixed: `cluster_nodes.parquet` |
| F4 | Fixed: the support tables place every read on the final consensus through replay, rebuilt survivors included |
| F5 | Fixed: a failing kernel falls back to a pooled node, then to the template itself, and is recorded; a raising rebuild is counted |
| F6 | Fixed: `runner_up_*` columns; the description below is corrected |
| F7 | Fixed: weighted 2×2 counts, φ and log odds on every significant edge |
| F8 | Fixed: `core.ids` content ids in `cluster_nodes.parquet`; FASTA headers unchanged |
| F9 | Deferred by decision: the refit is wanted once the browser makes quality assessable; the docs now say the loop does not refit, and `mstep/disagreements` records the substrate |
| F10 | Fixed |
| F11 | Fixed |
| F12 | Found while planning; fixed (`em/round_io.py`) |

**F1 (gap, blocks the support view). The M-step's evidence is computed and thrown away.**

The PI's suspicion is correct, and the gap is wider than the PWM. Steps b–i of §2.3 live only inside `mstep_worker` ([mstep_pool.py:351](../../constellation/sequencing/transcriptome/cluster/denovo/em/mstep_pool.py#L351)), which returns nothing but nodes and membership. None of the following is on disk:
- the PWM (and so the frame, the plan and the coordinate maps);
- the candidate columns and their route, effect, ε and p-value;
- the covariance edges;
- the signatures;
- the per-read patterns and margins.

The bench notes confirm the cost. There is no column or covariance dump in ~50 forensic scripts. Per-column counts for the Ldhc poly-A split and for the C/T near-duplicate pair were pulled by hand.

**F2 (P2). The per-template diagnostic record is built and then dropped.**

`refine_template` assembles a dict on the major node ([mstep.py:366-403](../../constellation/sequencing/transcriptome/cluster/denovo/em/mstep.py#L366); `stats=diag if hid == 0`, line 459). It holds:
- the column count;
- candidates by route, collapsed and capped;
- pairs seen, tested and significant;
- columns retained and dropped by budget, and `Σk²`;
- signatures, observed tuples and nodes;
- `disagreement_stats`.

Its own docstring says it exists because *"the method returned one template" has to be visible rather than inferred from a quiet result*. But `_node_row` ([mstep_pool.py:528](../../constellation/sequencing/transcriptome/cluster/denovo/em/mstep_pool.py#L528)) never serialises `node.stats`, so this record never reaches disk. It is small (one row per template per round), and it is the cheapest first step toward fixing F1.

**F3 (gap). A final cluster cannot be joined to its history.**

- `cluster_id` is `np.arange(n)` over the nodes the final merge kept ([outputs.py:111](../../constellation/sequencing/transcriptome/cluster/denovo/em/outputs.py#L111)).
- The `keep_rows` that would map it back to the last round's node rows is deliberately left out of `final.json` ([rounds.py:1621](../../constellation/sequencing/transcriptome/cluster/denovo/em/rounds.py#L1621)).
- `clusters.parquet` carries no template id, no node key and no round.

The mapping can be re-derived: take the node order, drop the ids `merged_final.parquet` lists as absorbed, and rank what is left. Every consumer would have to repeat that, and it breaks silently if any step changes. The cluster → lineage join is the spine of the history view.

**F4 (gap). No read is aligned to the consensus the run outputs.**

- Assignments align reads to the last round's *templates*.
- The emitted consensus is that template's PWM frame with gap-winning columns removed and the flanks trimmed.
- For a rebuilt survivor, the consensus comes from `align_finalist` alignments that are not written either.

So "reads under the final consensus" needs one of two things:
- the M-step's coordinate maps (F1), to project the E-step CIGARs from T through F to C; or
- a fresh alignment.

The components path has a schema for exactly this, `CLUSTER_ALIGNMENT_TABLE` (`--emit-alignments`). The EM path has nothing equivalent.

**F5 (P2). A template whose M-step raises vanishes without a trace.**

`mstep_worker` catches every exception per template and `continue`s ([mstep_pool.py:412](../../constellation/sequencing/transcriptome/cluster/denovo/em/mstep_pool.py#L412)). There is no counter, no log line and no record. The consequences:
- The template's reads are in the round's assignments but in no node's membership.
- In a middle round they come back at the next E-step.
- In the **final** round they are missing from `cluster_membership.parquet` and from quant, and nothing reports it.
- Nothing compares assigned reads with membership rows (`em/diagnostics.py` has no such check).

Isolating one template's failure is right. Doing so silently is the class of defect the 2026-09-15 reviews were about. A per-round `n_mstep_failed` and the failing template ids belong in the F2 record.

**F6 (gap). The runner-up is not recorded.**

The assignment row does not say *which* template came second. (As first written this finding said the row keeps "the margin to the second-best candidate" in `as_delta` / `logl_delta`. It does not: each is the winner's shortfall from the best candidate on that score, zero whenever the winner is the best, so it measures the rule's departure from the score, not a margin.) Three of the newest investigations needed it, and each rebuilt it by replaying the E-step outside the pipeline:
- **ledger #72:** the shortlist ties of Gsg1, Hnrnpk and Tex35, via `shortlist_explain.py` and `probe_count_replay.py`;
- **handoff_scheduler_v3 §9:** the near-tie dump;
- **ledger #71:** the shortlisted candidates of unassigned reads.

What it would take:
- **For the browser:** a `runner_up_template_id` (int64, nullable), the "why here and not there" of every read.
- **For the forensics:** an opt-in per-read shortlist record.

**F7 (gap). Covariance edges carry significance, not effect.**

`CovarianceGraph` keeps the p-values. The 2×2 counts that produced them (`n11`, `n1x`, `nx1`, `n` in `_pair_pvalues`) are discarded. A p-value conflates depth with association: a φ of 0.3 gives χ² ≈ 4.5 (p ≈ 0.03) at 50 reads and χ² ≈ 450 (p < 1e-90) at 5,000. An arc diagram coloured by p would mostly show depth. Keep the counts (or φ and log-OR) with each edge.

**F8 (gap). Output ids are row numbers.**

`cluster_id`, `cluster_<row>` in both FASTAs, and `template_id` are positional. As a result:
- Two runs, or a re-run after a seeder change, share no transcript id.
- `massspec.library.ids.protein_id` hashes the FASTA *accession*. A proteomics search of `proteins.fasta` would therefore key its proteins on `cluster_<row>`, which means nothing outside that run.

The project's convention for an id that independent producers must agree on is xxh64 of content. A content-derived transcript id (of the consensus) and protein id (of the protein sequence) would let a cluster be:
- compared across runs and with the components path;
- joined to proteomics on peptide or protein sequence, with no domain importing the other.

**F9 (P2). The error model is never refit, and the "annealing" knobs are never scheduled.**

The documentation says otherwise in three places:
- `refine_template` returns `disagreement_stats` "so the round loop can refit ε from round *r* and pass it in as round *r+1*'s prior";
- `sequencing/CLAUDE.md` says the same;
- the knobs are described as per-round arguments "so the round loop can schedule them".

In the code:
- `_run_mstep` passes no `error_model` ([rounds.py:1139](../../constellation/sequencing/transcriptome/cluster/denovo/em/rounds.py#L1139)), and the E-step gets none either.
- The disagreement statistics are dropped (F2).
- The same `MStepParams` is used every round.

So every round of every run uses the default `ErrorModel()`. Its homopolymer rate is the one the likelihood module itself records as 21× below the bench's measurement. This may be a deliberate deferral. Either way it should be written down, because the browser's column shading (ε per column) will show the default, not a fitted model.

**F10 (minor). One read-cardinality Python dict per template.**

`mstep_worker` builds `pos_of_row = {int(r): k for ...}` over **every** read assigned to a template ([mstep_pool.py:435](../../constellation/sequencing/transcriptome/cluster/denovo/em/mstep_pool.py#L435)), not just the ≤ 20k sampled. A 500k-read mega-template builds a 500k-entry dict per round. `np.searchsorted` on a sorted copy does the same job. This is small next to the consensus, but it is the pattern `sequencing/CLAUDE.md` forbids.

**F12 (P2, found while planning PR1). A resume could read stale M-step shards.**

`run_batched` writes one shard per unit and never clears its output, and the unit count follows the worker count. A round re-run with fewer units than an earlier attempt leaves that attempt's extra shards beside its own. The live loop reads only the paths its call returned, but `_resume_point` — the path every `--resume --rounds N+k` extension of a finished run takes, since a final round never writes its successor's templates — globbed `mstep/nodes/part-*.parquet`, as did the diagnostics. The stale node rows became duplicate templates (the default merge collapses exact copies; under `--no-merge` they survive).

**F11 (doc drift).**
- The paragraph that opens the M-step description in `sequencing/CLAUDE.md` (`*em/mstep.py + em/columns.py + em/covariance.py* — … ORF re-predicted under a support gate`) predates the gate's removal on 2026-09-30, which a later paragraph of the same file describes.
- `RefinedTemplate.stats` says a "bench driver" merges the statistics; the in-tree loop does not.

## 5. What the bench has needed to see

These are recurring needs from the notebook and ledger. They are the browser's acceptance tests, because each was answered by an ad hoc script.

| Question | Example (ledger / log) | Answered today by | Browser view |
|---|---|---|---|
| Which reads of gene *G* went where, round by round? | Gsg1 / Hnrnpk / Tex35 lost after 3–6 rounds (#72) | `gene_focus.py`, `gene_partition_report.py`, `lost_reads.py` | History: read flow restricted to a selection |
| Is this template one bad read? | Tuba1a t302020, 21 in-frame stops (#34) | identity recomputed from `--eqx` CIGARs | Support: identity / quality per read, PWM agreement |
| Why did this template split? | Ldhc `2199023680430`, 8 children, two ending in A-runs (second #48) | per-column counts pulled by hand | Support: candidate columns, arcs, reads ordered by pattern |
| Are two near-identical nodes a real variant or noise? | the 99.92 % pair, 100 % C vs 85 % T | hand counts | Support: two-node comparison at the differing columns |
| Why did a read pick X over Y? | Tcp1 t261991: 78% of its reads pulled off a better-scoring template by the node-weight override | the round-6 scheduler audit | Inspector: winner, runner-up (F6), margins |
| Did the merge pick the right form? | Usp39, Kif2b, Uchl3, Col16a1 (#52, #55) | `merge_forensics.py`, counterfactual reruns | History: merge events with both forms' support |
| Where did the unassigned reads come from? | seed orphans: Ppp4c, BC048671, Mlf1, Ppil6, Gm5646 (#71, #73, #76) | `unassigned_fate.py`, `seed_cluster_audit.py` | Unassigned as a first-class population (plan §5.6) |
| Is the elected seed read a good template? | Gsg1 elected 2,500 nt vs p90 1,413 | `seed_member_template_diff.py` | Seed view: members vs elected read, block vs scattered |
| Does the consensus match the reference transcript? | everything in `refseq_compare.py` / `fidelity_recall.py` | scripts | Annotation layer: RefSeq hit per node |

Two observations follow from this table.
- **The bench works gene-first.** Nearly every investigation starts from a gene name and needs the reads of that gene, which the run does not know. The genome-guided `transcriptome align` output does: a read → gene assignment for the same reads, joined on `read_id`. Colouring reads by their genome-assigned gene or isoform turns "purity" from a statistic into something visible.
- **The bench compares runs** (`compare_runs.py`, `table_diff.py`, `edge_set_diff.py`), which F8 blocks.
