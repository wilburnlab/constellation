# EM round-1 seeding by read-level kmer clustering (`--mode em-kmer`)

## Status

**Phases 0–3 SHIPPED** (2026-09-19). `--mode em-kmer` runs end to end; `--mode em-orf` is unchanged
and remains the default. 3,173 tests pass (the 4 `test_proforma.py` pyteomics-interop failures are
pre-existing and unrelated). **Phase 4 — the `generate_candidates` RSS fix — is NOT done**, so the
measured-best `--minimizers-per-seq 100` still needs a ~260 GB node; 50 is the default and runs at
~145 GB.

### What the implementation changed from this plan

**The plan was wrong about ORFs, and the PI caught it.** It specified predicting an ORF per elected
template "so round 1's `orf_aa_length` is meaningful and `gated_orf`'s certified interval is right".
That reasoning is backwards. `gated_orf` treats the seed ORF interval as *certified by construction*
and declines to judge it, so predicting one at seed time **relaxes the M-step's support gate** — and
under kmer seeding nothing asserted that ORF, because the cluster earned its template on read
support. ORF detection was steering the algorithm on a claim the seeder never made. Under ORF
seeding the certification is earned (the template was *selected because* it carried that ORF); here
it is not. Templates now ship `orf_start == orf_end == 0`, `seed_by_kmer_clustering` takes no
`min_aa_length`, and that field is out of the kmer resume stamp. Proteins are predicted per node on
its own consensus, which is where annotation belongs.

**And the M-step's length floor came out with it, on the PI's call:** it "should have no floor and
never should have". `gated_orf`'s `min_aa_length` and `MStepParams.min_aa_length` are gone. The floor
was not merely an annotation preference — each node's ORF interval carries forward as the next
round's *certified* interval, so it was setting how conservatively later rounds gated their own
ORFs. Removal is monotone (the longest ATG→stop per `(frame, stop)` does not depend on the floor), and
every M-step test passed unchanged, which is the evidence. `--min-aa-length` is now the em-orf seeding
key alone at default **30** — at 60 the seeder cannot make a template for Prm1 (51 aa) — stays 60 for
`--mode kmer`'s consensus annotation, and is refused under `--mode em-kmer`.

Then four things found while building:

1. **The chaining guard needed an absolute floor, not just a fraction.** As specified it fired on
   every small corpus — three transcripts at 20x each put 33% of the reads in each cluster,
   correctly. A fraction needs a corpus to be a fraction *of*, so both thresholds now also require
   `--min-chain-cluster-reads` (10,000, settable) reads in the cluster. That floor is **not a
   measured constant** — it is chosen so a guard whose whole evidence base is at 9.4M reads stays
   quiet where it has nothing to say.

2. **The resume stamp had to be checked in `_loop`, not in `_round_one_templates`.** `_resume_point`
   short-circuits past the seed stage whenever a completed round exists, so as planned the check
   would have covered only resumes that rebuild `seed/` — and a resume restarting at round 4 is just
   as committed to the original seeder. It now runs on every resume.

3. **`--max-len-delta` has no CLI flag**, so it is not in the inapplicable-flag list. `--fold-identity`
   moved to `default=None` so "was it passed?" means something.

4. **The ORF seeder now writes the same `seed/stats.json` shape** as the kmer one, so
   `section_seeding` renders either without branching and the two are directly comparable. Not in the
   plan; it is what makes the comparison this work exists for readable from the report.

Two deliberate extras beyond the plan's Phase 0: `seed.py`'s election was unified onto
`elect_representatives` rather than merely sharing its helpers (the ORF hash is densified with
`np.unique`, which is sort-order-identical to the inline lexsort it replaced, so `orf_id` assignment
is unchanged), and `denovo/orf.py` gained `predict_orfs_parallel` so the fork-pool is shared rather
than imported across a private name.

### Original plan follows

## Context — what the experiments established

Source: `results/kmer_read_clustering.md` and ledger items #48–51 in the Denovo_Transcriptome_Clustering
bench (`/mnt/e/Constellation_Optimization/Denovo_Transcriptome_Clustering`), jobs 14613263 / 14613264 /
14613408 / 14615094 / 14615395, 2026-09-17.

The EM's round 1 is bottlenecked by **template count**, and the ORF seeder produces one seed per
distinct ORF — which scales ~linearly with reads (ledger #35), so it compresses only ~1.4×. Running
Constellation's *own* Layer-0/1 kernels (`dereplicate → extract_minimizers → generate_candidates →
verify_candidates → connected_components`) on the **reads themselves** produces a far smaller
template set without merging genes:

| | ORF seed + fold | kmer `0.93 / inf:100 / m100` |
|---|---|---|
| templates | 3,778,760 | **850,450** (4.4× fewer) |
| template bases | 6,122 Mb | **1,258 Mb** (4.9× fewer) |
| round-1 E-step wall | 13.95 h (extrapolated; 14.33 h observed) | **2.89 h** at `m50 / 0.90` |
| round-1 E-step RSS | 127 GB | **26 GB** |
| reads assigned | 0.962 | **0.986** |
| gene agreement | 0.981 | 0.980 |
| tx agreement | 0.783 | 0.761 |
| `-N` cap-hit rate | **11.0%** | **0.05%** |

Three findings drive the design, and two of them are counter-intuitive enough to be worth stating
before any code is written:

1. **The ORF path is not on the cost/quality frontier at all.** The *shipped* kmer gate
   (`0.98 / 30:30`) costs the same as ORF seeding — 14.07 h, 4.3M templates, 6.4 Gb — and beats it on
   every outcome (0.971 vs 0.962 assigned, gene agreement 0.990 vs 0.981, tx agreement 0.812 vs 0.783,
   tx template recall 0.998 vs 0.978). Relaxing identity/ends then buys the 4.8× cost cut. So this is
   not a "cheaper but worse" trade at the operating point; it dominates at equal cost and the cost
   reduction is bought separately, with 2 points of transcript agreement.

2. **The `-N` cap-hit rate is a correctness fix, not a cost one.** Against the ORF set the median read
   has 659 candidate templates and 11.0% of reads hit the cap — the exact condition
   [rounds.py:328](../../constellation/sequencing/transcriptome/cluster/denovo/em/rounds.py#L328)
   warns makes "this round's rankings unsound for them", because minimap2 truncates by score among
   near-clone templates. Against the kmer set: 84 candidates, 0.05%. The redundancy ORF seeding
   creates is what the E-step then has to arbitrate over.

3. **A 3′ cap is what makes 5′-unbounded recruitment safe, and only at depth does that show.** At
   9.39M reads, `components` + `inf:inf` puts **7,096,457 reads (76% of the corpus) into one
   component at purity 0.154**; `500:500` still yields a 410,660-read component. The same settings at
   1M peak at 328k, so the failure is **invisible below full scale** and the existing
   `_MEGA_CLUSTER_UNIQUES = 2_000_000` guard never fires on the read count. `inf:100` holds purity at
   0.983–0.988 while keeping the 5′-truncation tolerance `results/cds_integrity_v2.md` says is needed
   (30% of Complete windows are 5′-partial). This is ledger #14 + #50.

Two further results shape the interface rather than the defaults:

- **linclust's longest-read bucket centre is the wrong rule here** (ledger #51, a negative result).
  Swept as bucket anchor *and* claim/centroid order, the shipped `abundance` rule wins decisively at
  9.39M (gene dominant 0.866 vs 0.815 `length` vs 0.806 `qlen`) because anchoring on the longest read
  pairs the most length-discordant reads in each bucket — precisely the pairs the overhang gate
  rejects. At 1M the three converge, so a 1M-only experiment would have called it a tie. **Do not port
  it.** Radius-1 `greedy_set_cover` remains the chain-proof fallback (largest cluster 60k vs 7.1M) at
  ~1.4× the templates and 10 points of contiguity.

- **Seeding only — no consensus after round 1.** Median read↔best-template identity is 0.992–0.994 for
  every kmer set, so admission at `p_floor = 0.97` is nowhere near binding and template *error* is not
  what limits recruitment. The real gap is template *extent* (median coverage of the reference
  transcript by the elected template is 0.31–0.44), and a consensus over cluster members would not fix
  that either — it needs the M-step's extension machinery, which already exists.

### The one place this plan departs from what was measured

In the sweep, a single "rule" drove **three** things at once: the minimizer bucket's star anchor, the
greedy claim / component centroid order, **and** which read became the template. That conflation is
why `abundance` (best partition, elected rep at median error 0.0050, 72% clearing Q22) and `qlen`
(worse partition at 1.7× the templates, but elected rep at median error 0.0040–0.0045, 97% clearing
Q22) had to be chosen between.

They do not have to be. **This plan decouples them**: the grouping order stays the shipped `abundance`
rule (candidate anchor + component centroid tie-break, already
[cluster_graph.py:65](../../constellation/sequencing/transcriptome/cluster/denovo/cluster_graph.py#L65)),
and the per-cluster template election runs the EM's existing `--seed-representative` policy —
`longest-above-quality` by default — over the cluster's member reads. That is what ledger #48 asks for
in as many words ("with the representative elected per cluster by `longest-above-quality`, the existing
policy"), and it should recover most of `qlen`'s seed-accuracy gain at `abundance`'s template count.

**This combination was not measured.** It is the first thing the full-scale run should check: compare
`rep_err_median_ge2` / `rep_q_ge_floor_ge2` against the sweep's 0.0050 / 0.721 for `abundance` and
0.0040 / 0.969 for `qlen` at the same gate.

---

## Design

### The interface the seeder must satisfy

`em/rounds.py::_round_one_templates`
([rounds.py:347](../../constellation/sequencing/transcriptome/cluster/denovo/em/rounds.py#L347))
returns a `TEMPLATE_TABLE`
([templates.py:34](../../constellation/sequencing/transcriptome/cluster/denovo/em/templates.py#L34)).
That is the whole contract. Field by field, for the kmer path:

| field | ORF path | kmer path |
|---|---|---|
| `template_id` | `arange(n)` | `arange(n)` — unchanged |
| `sequence` | elected read's full cDNA | elected read's full cDNA (the cluster member, **not** the component centroid) |
| `orf_start` / `orf_end` / `orf_aa_length` | from the seed ORF | **`0`** — no ORF is predicted; see the Status section |
| `node_weight` | fold group read count | cluster read count |
| `orf_replication` | fold group read count | cluster read count |
| `seed_read_quality` | elected read's `dorado_quality` | same |
| `seed_read_row` | corpus row of the elected read | same |
| `declared_variants` | `[]` | `[]` |

`orf_replication` keeps its name. Round 1's ranker
([scheduler.py:108](../../constellation/sequencing/transcriptome/cluster/denovo/em/scheduler.py#L108))
sorts on `(orf_replication desc, seed_read_quality desc, template_idx asc)`; under kmer seeding the
first key becomes *cluster support*, which is a strictly better round-1 prior than ORF replication and
needs no code change. Renaming the column would break resume across rounds for no gain — the
docstrings get the dual meaning instead. From round 2 `refine.py` zeroes it anyway
([refine.py:135](../../constellation/sequencing/transcriptome/cluster/denovo/em/refine.py#L135)).

**ORF prediction is kept**, at one call per elected template. It is cheap at that cardinality and it
matters: `mstep.py::gated_orf`
([mstep.py:134](../../constellation/sequencing/transcriptome/cluster/denovo/em/mstep.py#L134))
treats `(seed_orf_start, seed_orf_end)` as the interval certified *by construction* and judges only
ground outside it. Passing `(0, 0)` would be merely conservative, not wrong — but it would make every
round-1 ORF call in the M-step gate against an empty certificate. A template with genuinely no
qualifying ORF gets `(0, 0)` and is kept (there is no ORF filter on the kmer path — the cluster earned
its template by read support, not by carrying a protein).

### New module: `em/seed_kmer.py`

```python
@dataclass(frozen=True, slots=True)
class KmerSeedResult:
    templates: pa.Table      # TEMPLATE_TABLE
    read_cluster: pa.Table   # (read_id, cluster_id, sample_id) — the ORF path's read_orf analogue
    stats: dict              # stage counts + the #50 guard, straight into the manifest

def seed_by_kmer_clustering(
    reads: pa.Table, *,
    identity: float = 0.93,
    max_5p_overhang: int = UNBOUNDED_OVERHANG,
    max_3p_overhang: int = 100,
    kmer: int = 15, window: int = 10, minimizers_per_seq: int | None = 50,
    min_shared: int = 2, diag_span_max: int = 20,
    grouping: Literal["components", "greedy"] = "components",
    min_seed_reads: int = 1,
    representative: str = "longest-above-quality",
    min_aa_length: int = 30,
    max_cluster_read_frac: float = 0.25,
    threads: int = 1,
    progress: Callable[[str], None] | None = None,
) -> KmerSeedResult
```

Flow — every stage an existing shipped kernel, no new algorithms:

1. `dereplicate(reads)` → `uniq, read_map`. The corpus table carries `dorado_quality` as a fourth
   column; `dereplicate` projects what it needs, so it passes through unchanged.
2. `extract_minimizers(uniq.sequence, k=kmer, w=window, max_per_seq=minimizers_per_seq)`.
3. `generate_candidates(index, abundance, min_shared=..., diag_span_max=...)`; `del index` before verify
   (the kmer pipeline already does this —
   [pipeline.py:569](../../constellation/sequencing/transcriptome/cluster/denovo/pipeline.py#L569)).
4. `verify_candidates(candidates, seqs, identity=..., max_5p=..., max_3p=..., threads=...)`.
5. `connected_components(...)` (or `greedy_set_cover` behind `grouping="greedy"`, the ledger-#51
   chain-proof fallback) → `cluster_of` per uniq; **the returned `centroid_uniq` is discarded** — step 7
   elects the template instead.
6. Read→cluster: `cluster_of[uniq_id_of_read]`; cluster read counts by `np.add.at` over `abundance`.
7. **Elect one read per cluster** via the shared policy helper (below).
8. `min_seed_reads` filter on cluster read count (the `ge2` rows of the bench), then dense remap.
9. ORF prediction over the elected template sequences only, via `seed.py::_predict_orfs`.
10. Assemble `TEMPLATE_TABLE`.

### Shared representative election — `em/elect.py` (Phase 0 refactor)

`seed.py` already contains exactly the election machinery the kmer path needs, as three private
helpers plus an inline two-pass lexsort
([seed.py:214–345](../../constellation/sequencing/transcriptome/cluster/denovo/em/seed.py#L214)):
`_group_bounds`, `_weighted_median_per_group`, `_best_quality_per_uniq`, and the pass-A/pass-B
group-statistics + rank + first-of-group pattern. The only difference between the two seeders is the
**group key**: ORF hash there, `cluster_of` here.

Extract it into a public `em/elect.py`:

```python
def elect_representatives(
    group_of_uniq: np.ndarray,          # int64, dense cluster/group id per uniq
    n_groups: int, *,
    template_length: np.ndarray,        # int64 per uniq
    abundance: np.ndarray,              # int64 per uniq
    quality: np.ndarray,                # float64 per uniq, -1 where absent
    orf_start: np.ndarray | None = None,
    policy: str | RepresentativePolicy = "longest-above-quality",
) -> tuple[np.ndarray, np.ndarray]:     # (elected uniq per group, group read count)
```

`REPRESENTATIVE_POLICIES` and `RepCandidates` move here; `seed.py` imports them back (no cycle —
`elect` is a leaf). `tests/test_em_seed_quality.py` is the regression net for the move: it must pass
unchanged.

**`most-5p-flank` is not available under kmer seeding.** It ranks on `orf_start`, which does not exist
before election on this path (predicting ORFs over all 9.15M uniques just to rank is the cost the kmer
seeder exists to avoid). `seed_by_kmer_clustering` raises on it with a message naming the four
supported policies rather than silently substituting one.

**Drive-by simplification, verified in the read:** `_best_quality_per_uniq` recovers each read's row in
`reads` with `pc.index_in(read_map.read_id, value_set=reads.read_id.combine_chunks())` — a 9.4M-string
hash per run. `dereplicate` builds `read_map` row-aligned with `reads` by construction
([dereplicate.py:128](../../constellation/sequencing/transcriptome/cluster/denovo/dereplicate.py#L128)),
so this is `np.arange(n)`. Worth taking with an explicit test asserting the alignment, since the
refactor is already touching the function.

### Unbounded overhangs

`verify_candidates` gates with a plain `overhang_5p > max_5p` comparison
([verify.py:99](../../constellation/sequencing/transcriptome/cluster/denovo/verify.py#L99)), so a large
sentinel *is* "unbounded" with no kernel change — which is what the bench prototype relied on. Add

```python
UNBOUNDED_OVERHANG = 1 << 30   # verify.py, exported
```

and a CLI parser that accepts `inf` / `none` / a negative integer for
`--max-5p-overhang` / `--max-3p-overhang` and maps them to it. Keep it a named constant rather than a
magic number so the manifest records something legible and `inf:100` round-trips.

### The mega-component guard (ledger #50a)

After grouping, compute `largest_cluster_reads / n_reads` and put it in `stats`. Two responses:

- **Warn loudly above 1%**, in the same register as `_warn_on_saturation` — named, with the cause
  ("connected-components chaining; a 3′ cap is what prevents it") and the lever (`--max-3p-overhang`).
- **Fail the seed stage above `--max-seed-cluster-frac` (default 0.25).** A component holding a
  quarter of the corpus is never a transcript, and the failure mode this catches costs a 14-hour job:
  the E-step would index one 7M-read template and the M-step's `--max-members-per-template` would cap
  its PWM while the partition stays garbage. Set `0` to disable for a deliberate sweep.

The existing `_MEGA_CLUSTER_UNIQUES` warning in `pipeline.py` counts *uniques*, which is why it never
fired at 7.1M reads; this guard counts reads, and belongs in the seeder so `--mode kmer` and
`--mode em-kmer` both get it.

### CLI surface

`--mode` gains two spellings and keeps one:

```
--mode {genome, kmer, em-orf, em-kmer}
```

- `em-orf` — today's `--mode em`, unchanged.
- `em-kmer` — kmer seeding.
- `em` — accepted, resolves to `em-orf`, prints a one-line note (not a deprecation: it is
  under-specified, not wrong). **The default does not flip in this PR.** ORF seeding is dominated on
  every measured axis, but the point of this work is the multi-round comparison, and flipping the
  default before that runs would destroy the baseline it is measured against.
- `genome-guided` / `de-novo` keep normalising as they do now.

Implementation: `_normalise_cluster_mode` returns `"em"` for all three EM spellings, and a new
`_em_seeding(mode) -> Literal["orf", "kmer"]` carries the choice into `EmParams.seeding`. **The
`mode` column of `clusters.parquet` stays `"em"`** — the mechanism is still EM, the seeder is a
parameter — so `CLUSTER_MODES`, the viz colour maps
([cluster_pileup.ts:39](../../constellation/viz/frontend/src/track_renderers/cluster_pileup.ts#L39),
[TrackSettingsPanel.ts:412](../../constellation/viz/frontend/src/widgets/TrackSettingsPanel.ts#L412))
and `test_em_cli_modes.py::test_the_frontend_colour_maps_keep_the_old_keys` are untouched. It also
makes the eventual consolidation to `--mode em --seeding kmer` a pure CLI change.

#### Flag resolution — one flag, three right answers

`--identity`, `--max-5p-overhang` and `--max-3p-overhang` already exist for `--mode kmer` with
defaults that are wrong for `em-kmer`. This is the problem `--overdispersion` already solved in this
parser: `default=None`, resolved per handler
([__main__.py:1451](../../constellation/cli/__main__.py#L1451)). Same treatment:

| flag | `--mode kmer` | `--mode em-kmer` | `--mode em-orf` |
|---|---|---|---|
| `--identity` | 0.98 | **0.93** | error if passed → `--fold-identity` |
| `--max-5p-overhang` | 30 | **inf** | error if passed |
| `--max-3p-overhang` | 30 | **100** | error if passed |
| `--kmer` / `--window` | 15 / 10 | 15 / 10 (seed sketch) | 15 / 10 (fold sketch) |
| `--minimizers-per-seq` | 50 | 50 (see below) | 50 |
| `--min-shared` *(new)* | 2 | 2 | 2 |
| `--diag-span-max` *(new)* | 20 | 20 | 20 |
| `--seed-grouping` *(new)* | n/a | `components` | n/a |
| `--max-seed-cluster-frac` *(new)* | n/a | 0.25 | n/a |
| `--min-seed-reads` | n/a | 1 | 1 |
| `--seed-representative` | n/a | `longest-above-quality` (no `most-5p-flank`) | `longest-above-quality` |
| `--fold-identity` / `--max-len-delta` | n/a | error if passed | 0.97 / 9 |

`--kmer` / `--window` / `--minimizers-per-seq` are reused rather than duplicated because the ORF-fold
sketch and the read-seed sketch are never both live: `em-kmer` replaces seed+fold outright. Flags that
do not apply **error rather than silently no-op**, matching the precedent set by
`massspec predict-library`'s EncyclopeDIA-only flags under `--backend koina`.

`--min-shared` and `--diag-span-max` are ledger #4: hidden constants in `generate_candidates` today.
The sweep settles their shape (`min_shared=3` costs 2–10 points of gene contiguity; `diag_span_max=60`
buys ~1 point over 20), so `(2, 20)` stays the default and the diagonal span is the one worth relaxing.

#### On `--minimizers-per-seq`: default 50, measured best 100

The headline operating point is m=100. Sketch density buys compression and contiguity *at the same
time* at scale: m50 → m100 at `0.93 / inf:100` takes the set from 987,925 templates at 0.855 gene
dominance to **850,450 at 0.866** (purity 0.986 → 0.985, ARI 0.928 → 0.931) — 14% fewer templates *and*
better contiguity, because the denser sketch finds the pairs that were splitting genes. The 1M grid
says m=100 → 200 is worth ~1 more point again.

It costs **242 GB peak RSS in `generate_candidates`** (m=50: 145 GB), which puts m=100 on a 450 GB node
and m=200 out of reach entirely. So: **default stays 50**, and the help text plus the plan record that
100 is the measured-best where RAM allows. Phase 4 removes the ceiling.

### Resume safety

`_round_one_templates` short-circuits on `seed/_SUCCESS` and reads `seed/templates.parquet`. With two
seeders that is a silent-wrong-answer path: resuming an ORF run under `--mode em-kmer` reuses the ORF
templates and reports kmer parameters in the manifest.

Write `seed/params.json` stamping `{seeding, identity, max_5p, max_3p, kmer, window,
minimizers_per_seq, min_shared, diag_span_max, grouping, min_seed_reads, representative,
min_aa_length}` beside `_SUCCESS`, and refuse a resume whose stamp differs — naming the fields that
changed, as the `--resume` guards elsewhere do.

### Diagnostics and manifest

`parameters={"mode": "em", **asdict(params)}` in `write_em_manifest`
([rounds.py:246](../../constellation/sequencing/transcriptome/cluster/denovo/em/rounds.py#L246)) means
the new `EmParams` fields flow into `manifest.json` for free. `stats` from the seeder goes into
`stages` under a `seed` key.

New `em/diagnostics.py::section_seeding(em_dir)`, reading `seed/stats.json`, reporting whichever seeder
ran:

- stage counts and timings (uniques, minimizers, candidate pairs, verified edges, clusters);
- template count, template Mb, reads per template, singleton fraction, `n_clusters_ge2`;
- **largest cluster as a fraction of reads** — the #50 guard, flagged above 1%;
- elected-representative quality: median `dorado_quality`, fraction clearing Q22. This is the number
  that tests the decoupling described above, so it belongs in the report rather than in a one-off
  script.

---

## Phases

### Phase 0 — shared election + parameter exposure

1. `em/elect.py`: move `RepCandidates`, `REPRESENTATIVE_POLICIES`, `SEED_QUALITY_FLOOR`,
   `_group_bounds`, `_weighted_median_per_group`, `_best_quality_per_uniq`, and the pass-A/pass-B
   election out of `seed.py`; add `elect_representatives(...)`. `seed.py` re-imports and keeps its
   public names.
2. Simplify `_best_quality_per_uniq`'s row recovery to positional, with a test asserting `read_map` is
   row-aligned with `reads`.
3. `UNBOUNDED_OVERHANG` in `verify.py`.
4. `min_shared` / `diag_span_max` threaded through `assemble_clusters` and `cluster_transcripts` as
   parameters (ledger #4, the two that the sweep measured).

Net behaviour change: none. `tests/test_em_seed_quality.py`, `test_denovo_cluster.py` and
`test_em_rounds.py` must pass untouched.

### Phase 1 — `em/seed_kmer.py` + the round-loop dispatch

1. `seed_by_kmer_clustering` as specified, with `KmerSeedResult`.
2. `EmParams` gains `seeding: Literal["orf","kmer"] = "orf"`, `seed_identity`, `seed_max_5p`,
   `seed_max_3p`, `seed_grouping`, `min_shared`, `diag_span_max`, `max_seed_cluster_frac`.
   (`kmer` / `window` / `minimizers_per_seq` / `min_seed_reads` / `seed_representative` /
   `min_aa_length` already exist and are reused.)
3. `_round_one_templates` dispatches on `params.seeding`; the kmer branch writes
   `seed/templates.parquet`, `seed/read_cluster.parquet`, `seed/stats.json`, `seed/params.json`.
4. Resume stamp check.

### Phase 2 — CLI

1. `--mode {genome, kmer, em-orf, em-kmer}` + `em` alias; `_em_seeding`.
2. `--identity` / `--max-5p-overhang` / `--max-3p-overhang` to `default=None`, resolved per handler;
   `inf` / `none` / negative accepted for the overhangs.
3. New flags: `--min-shared`, `--diag-span-max`, `--seed-grouping`, `--max-seed-cluster-frac`.
4. Inapplicable-flag errors, both directions (EncyclopeDIA/koina precedent).
5. `_cmd_transcriptome_cluster_em` builds `EmParams` with the seeding fields.

### Phase 3 — diagnostics, manifest, docs

1. `section_seeding` + wiring into `build_em_report`.
2. `seed` stats into the manifest `stages`.
3. Docs: the EM section of [constellation/sequencing/CLAUDE.md](../../constellation/sequencing/CLAUDE.md),
   the `sequencing` row of the root [CLAUDE.md](../../CLAUDE.md) module index, the status block of
   [de-novo-clustering-roadmap.md](de-novo-clustering-roadmap.md), and `em/__init__.py`'s stage table
   (which currently documents `seed` as ORF-only).

### Phase 4 — optional, but required before m=100 at full scale (ledger #49)

`generate_candidates` peaks at **145 GB (m=50) / 242 GB (m=100)** on 9.39M reads — ~20 simultaneously
live int64 intermediates over the minimizer occurrence list (bucket ids, sizes, the lexsort order,
anchor rows, diagonals, pair keys, the argsort), none freed before the return
([candidates.py:45](../../constellation/sequencing/transcriptome/cluster/denovo/candidates.py#L45)).

Fix: walk the hash-sorted index in **bucket-aligned chunks** and accumulate pairs per chunk. The
anchor-star is already linear per bucket, so chunking changes no output — which makes it testable by
bit-identity against the current implementation on a fixture, at several chunk sizes.

Same pass should widen the `(uniq_a << 31) | uniq_b` pair key, which caps `n_uniq` at 2^31. 9.15M
uniques is far inside that today, but the cap is invisible and silent when it is reached.

Not required for the first full-scale run at m=50.

---

## Defaults to ship

From the sweep's frontier, for `--mode em-kmer`:

```
--identity 0.93  --max-5p-overhang inf  --max-3p-overhang 100
--kmer 15  --window 10  --minimizers-per-seq 50
--min-shared 2  --diag-span-max 20  --seed-grouping components
--min-seed-reads 1  --seed-representative longest-above-quality
--max-seed-cluster-frac 0.25
```

Expected at 9.39M reads, m=50: ~988k templates / ~1,464 Mb / ~3.4 h round-1 E-step at ~29 GB.
`--identity 0.90` is the cheapest usable point at m=50 (874k templates, 2.89 h, gene dominance 0.867 —
slightly *better* than 0.93 at this density, 3 points worse on transcript template recall).
`--minimizers-per-seq 100 --identity 0.93` is the measured-best partition (850k templates, gene
dominance 0.866, ARI 0.931) on a node with ≥ ~260 GB.

`k=17–19 ≈ k=15 > k=13`, and `w` does not matter at fixed `m` (the bottom-m cap, not the window, sets
density) — so neither is worth sweeping again.

---

## What the full-scale run should measure

The point of shipping both seeders is the comparison the bench could only do for round 1:

1. **Does the decoupled election recover `qlen`'s seed accuracy at `abundance`'s template count?**
   `rep_err_median_ge2` / `rep_q_ge_floor_ge2` against 0.0050 / 0.721 (`abundance`) and 0.0040 / 0.969
   (`qlen`). This is the one untested piece of the design.
2. **Does the transcript-agreement gap close over rounds?** Round 1 gives up 2 points of tx agreement
   (0.761 vs 0.783) and 11 of tx template recall (0.869 vs 0.978). §7 of the writeup says transcript
   purity sits at 0.74–0.76 for *every* seeding including ORF-after-E-step, i.e. isoforms are merged by
   any seeder and separating them is the M-step's job. Rounds 2+ are where that claim gets tested.
3. **Does the cap-hit rate stay at 0.05%** through rounds 2+, where refine splits templates and
   re-introduces near-clones.
4. **Total wall clock to convergence**, against the ORF path's 6 rounds — the number that decides
   whether the default flips.

## Risks

- **`verify_candidates` materialises `seqs: list[str]` over all uniques** — ~9.15M Python strings,
  ~6 GB of payload, in the parent before a fork pool. `--mode kmer` already does exactly this at this
  scale, so it is a known cost rather than a new one; the bench prototype used a numpy sequence buffer
  with `.tobytes()` slices instead and is the reference implementation if it bites. Watch RSS at the
  verify stage on the first full run.
- **Chaining is depth-dependent and invisible in small tests.** The 76%-in-one-component failure only
  appears at 9.39M. No synthetic fixture will catch a regression here; the `--max-seed-cluster-frac`
  guard and the diagnostics line are the only defence, which is why they are in Phase 1/3 rather than
  deferred.
- **`generate_candidates` is not bit-reproducible run to run** (±0.3% candidate pairs on identical
  input — ledger #13, likely thread-order dependence in the bucket walk). It already affects
  `--mode kmer`; it now also affects round-1 template counts, so a rerun comparison should not expect
  exact equality.

## Observed in passing, not in scope

`--min-aa-length` defaults to **60** on the parser and `_cmd_transcriptome_cluster_em` passes it
straight into both `EmParams.min_aa_length` (whose own default is 30) and `MStepParams`
([__main__.py:3313](../../constellation/cli/__main__.py#L3313)). So the EM path runs at 60, not the 30
its dataclass and docstrings describe, and ledger #6 notes 60 drops Prm1 (51 aa), the most abundant
testis transcript. Worth resolving deliberately — it changes ORF calls on both seeding paths — but not
inside this change.
