# The EM clusterer's remaining hot spots (2026-11-15)

**Scope.** The de novo EM clusterer on `feat/em-evidence`, after the consensus-kernel fix of `d267120`, reviewed for time that Python spends doing what arrays could. Three sources:

- the bench's own timings at scale (`/mnt/e/.../docs/log.md`, `results/*churn*.tsv`, the ledger);
- two cProfile runs of the whole loop here, on a 5,408-read synthetic corpus built to look like real data: 80 genes, long-tailed depth, isoforms, haplotypes, 5′ truncation, homopolymer-biased indels;
- a static sweep of every loop whose length grows with the data.

**Already fixed on this branch:**
- `d267120`: the PWM scatter and `member_alleles`. One 5,000 × 3 kb template went from 849 s to 28 s.
- `6932406`: the empty evidence tables, which were rebuilt per template.

Nothing else here is changed yet.

---

## 1. Where the time goes at scale

Every timing at 9.4M reads that exists is an E-step / M-step split taken from `round.json`. Nothing at scale times the corpus, the graph, the merges, the rebuild or the outputs inside an EM run.

| Run (bench) | E-step | M-step | Other |
|---|---|---|---|
| 9.4M em-kmer, native, 12 rounds, 48 workers (`r12_native…261004`) | ~2,000–2,250 s / round (70% of E+M) | ~770–1,210 s / round | wall 10 h 41 m – 11 h 40 m; read minimizers 1,438 s once |
| 9.4M em-kmer, minimap2, 20 rounds (`r20…261001b`) | 12,126 s r1, then 2,050–4,380 s | 640–1,600 s | wall 28–35 h |
| 9.4M kmer seeding (sweep driver) | — | — | minimizers 1,271 s, candidates 354 s, verify 157 s, at 145 GB |
| 1M em-kmer, native, current defaults | 344–405 s / round | 55–90 s / round | — |
| graph, 9.4M | — | — | 4–8 min / round (minimap2 runs, ~450k nodes) |

**The E-step dominates at every scale.** The M-step is the next largest, a third to a half of the E-step under the native aligner.

## 2. Where the time goes here

**em-kmer, native E-step, 3 rounds: 74 s in total.**

| Stage | Time |
|---|---|
| Seeding | 13 s |
| E-step (rounds 1–3) | 17 / 9 / 9 s |
| M-step (rounds 1–3) | 7–9 s each |
| Graph | < 1 s / round |

Inside them:

- **E-step finalist alignment:** 30.3 s of the run.
  - 20.0 s is edlib, a C call that is inherent.
  - **10.3 s is Python around it:** parsing the CIGAR into a list, the anchor trim, merging, score trimming, and statistics. That runs once for each of the 47k finalist alignments, about seven passes over the CIGAR operations each time.
- **M-step ORF annotation:** 11.4 s, about 44% of all M-step time.
  - `best_sense_orf` costs 20 ms per node.
  - Its regex finds one match per ATG, overlapping, and every match is translated in full by `translate`.
  - `translate` makes a Python call per codon, each with an upper/replace/`all()` check for degenerate bases. That is 3.6M calls here.
- **M-step CIGAR projection:** about 1.4 s, with ~50 small `np.arange` calls per read in `project_member_events`.

**em-orf seeding, minimap2 E-step.**
- Seeding spent **64.5 s predicting ORFs** for 5,408 unique reads, 12 ms each. In this pipeline that is once per unique cDNA, 9.15M at full scale.
- The E-step is minimap2 itself (the 376 s of pipe reads are minimap2 running), a cost that can only be cut by having fewer templates, which kmer seeding does.
- **Covariance work** shows up in this run: `_reestimate` took 6.2 s and `dense_states` 3.0 s.

## 3. Opportunities

**Tier A: exact (bit-identical output), large, low risk.**

| # | Where | What | Evidence | Expected effect |
|---|---|---|---|---|
| A1 | `denovo/orf.py::best_sense_orf` | One pass instead of a translation per ATG. Codon codes per frame, each ATG's next in-frame stop by `searchsorted`, the longest per stop, and only the winner translated. The regex's lazy match, its `[ACGT]` class and the original tie-break (longest, then earliest start) are reproduced exactly. | A prototype matched the shipped function on **8,408 sequences at two floors with 0 mismatches**, including Ns, lowercase, low-complexity and short sequences, and was 9–12× faster. | ~40% of every M-step round; most of em-orf seeding; the final rebuild |
| A2 | `core/sequence/nucleic.py::translate`, `find_orfs` | Look up canonical ACGT codons directly and expand degenerate codons only when one appears; give `find_orfs` the same one-pass treatment as A1. | **Demux's ORF calling (`demux/orf.py` → `find_orfs`) has the same overlapped-regex-then-translate-each pattern on every raw read** (~154M in a PromethION run). Not yet measured. | Every `translate` caller; demux could gain the most of anything here |
| A3 | `em/realign.py::align_finalist` and helpers | Keep the CIGAR as integer arrays from a single parse; do the anchor trim, score trim and statistics with cumsum/argmax; build the CIGAR string once. | 10.3 of 30.3 s of alignment is Python. | ~⅓ of the E-step's alignment time, in the stage that is 60–85% of the wall |
| A4 | `consensus.py::project_member_events`, `_cigar.parse_cigar`; `mstep._node` | Project every member of a template at once (parse the CIGARs from the Arrow buffer, segmented cumsum for positions, `repeat` for runs). Split children reuse `pooled.projections` instead of re-running `frame_consensus` from the member specs. | ~50 `np.arange` calls per read; split templates pay the projection twice. | The M-step's remaining per-read cost |

**Tier B: exact, moderate or local.** (Item numbers are the static sweep's.)

- **`plan_columns` / `_member_events`:** recomputed three times per member per template (sweep #4). Build one event table and reuse it.
- **`member_allele_events` outer loop and `member_spans`:** scalar Python per member (#5). Both vectorise from the frame start/end arrays.
- **`covariance`:**
  - `read_states` runs per member (#6).
  - `_reestimate` does a bincount per (pattern, column) (#7; 6.2 s in the em-orf profile).
  - `dense_states` does a dict lookup per (member, state) (#8; 3.0 s).
  - `quasi_cliques` works with Python sets, up to O(V³) on hairballs (#9).
  - `epsilon_homopolymer` is a Python call per column (#12).
- **Corpus columns copied whole on every block:** `ReadStore.sample_id` / `dorado_quality` re-copy the full multi-chunk corpus column on each access, twice per E-step block. That is ~75 MB × ~1,700 blocks a round under minimap2 (#14). Cache them per process.
- **Quadratic PAF tail:** it is rebuilt by concatenation on every pipe chunk until 128 MB (`paf_scan.iter_hit_blocks`), roughly 8× copy amplification over ~216 GB of PAF a round (#15).
- **Seeding verify:** materialises ~9.15M Python strings (~14 GB) that forked workers then dirty, and pickles an 11-tuple per accepted pair back to the parent (#16).
- **Final outputs:** two `to_pylist` calls over 9.4M read ids and a per-row representative comparison (#17).
- **Structured-dtype sorts:** structured `(int64, int64)` keys are sorted and searched over membership-sized arrays in the rebuild, the merge and churn (#21). Packed int64 keys are already used elsewhere.
- **Smaller items:**
  - `_classify_columns`: O(blocks × V) (#22).
  - `plan_mstep_units`: `argmin` over ~800 loads per template (#24).
  - Per-template Arrow/`to_pylist` overhead, which dominates the many 1–3-read templates (#10).

**Tier C: change results, so they need validating on real data.**

- **Window the template as well as the read** in the finalist's core alignment. edlib's cost grows faster than length (0.095 ms at 800 nt, 0.52 ms at 1.5 kb, 1.15 ms at 3 kb), and partial reads on long templates are common. Placement could in principle differ, for example on tandem repeats.
- **Align fewer finalists per read.** Native rounds align 8–24 per read. The ranking could stop once the remaining candidates cannot win, but that needs a bound on the likelihood, which does not exist yet.

**Measured and dropped: bounding edlib's edit distance.** With `task="path"` it gives no speedup here, at 800, 1,500 and 3,000 nt. It would also have changed admissions, since the floor is judged after end trimming.

**Not covered by Python fixes:**
- minimap2 itself under em-orf;
- seeding's minimizer extraction (1,271 s at 9.4M). It is vectorised torch, but makes 15 full passes per k-mer position plus two hash passes. A rolling formulation might help, but would need a benchmark first.

## 4. Recommended order

1. **A1 and A2 together,** one ORF finder serving both seeding and demux, kept bit-identical by tests that compare against the replaced functions.
2. **A3,** the largest share of the dominant stage. Exactness is checked by replaying finalists before and after on the profiling corpus and on the evidence fixture.
3. **A4,** then **Tier B** by measured share.

Each step comes with:
- a profile before and after on the same corpus;
- the golden digests of `tests/test_em_evidence.py` unchanged;
- an equivalence test against the replaced code where the function is local.

Tier C waits for a real dataset, because it needs the effect on assignments measured, not just its speed.
