# Transcriptome browser: design foundations

**Status:** design approved for its work order, 2026-11-14 (§12). PR1, the EM evidence contract, is built on `feat/em-evidence` (§13); PRs 2–6 are scoped, and each gets a detailed plan when it starts.

**Companion:** [docs/reviews/em-clusterer-2026-11-14.md](../reviews/em-clusterer-2026-11-14.md), referred to here as *the review*. It covers what the EM clusterer computes, what it writes, and the findings (F1–F12) this design depends on, with their status.

**Sources for the platform.**
- The viz modality contract: [constellation/viz/CLAUDE.md](../../constellation/viz/CLAUDE.md), [viz-modality-generalization.md](viz-modality-generalization.md) and [ms-browser.md](ms-browser.md).
- The bench notebook, for what investigations have actually needed.

---

## 1. What it is for

There are two jobs, in this order.

**1. Diagnose the clusterer (now).** Show how the reads support a consensus, why a template split, which columns co-varied, and how stars split and merged across rounds.
- The bench has ~50 ad hoc forensic scripts and no plots. Every case in the review's §5 was answered by one.
- The browser's first acceptance test is that those cases become clicks.

**2. Explore a transcriptome (later, and for longer).** The final nodes are stars:
- connected by any similarity;
- coloured by any attribute;
- filtered by any criterion;
- tied to samples, proteins, peptides and annotation.

**These are one data model at different depths, not two tools.** The drill-down path is:

star map → node → its support → one read.

The diagnostic views are the deep end of the exploratory one. The only difference is that the history (rounds) matters to the first job and is mostly noise to the second.

**What has no precedent.** I know of no tool that combines all of this:
- graph exploration of a transcriptome;
- a per-node read matrix (MSA) with covariance;
- the algorithm's own history.

Each piece has a good analogue:
- **Graph and embedding exploration:** Cytoscape, Gephi, the cellxgene / UMAP scatter idiom, GNPS molecular networks.
- **Read pileup:** IGV.
- **Contact arcs over a sequence:** RNA arc diagrams, cross-link maps.
- **Alluvial flows between clusterings:** clustree, Sankey diagrams.

The design work is in how these interlock (§4), not in any one view.

## 2. Principles carried over from Constellation

- **Every number shown is a number the pipeline wrote, or a pure function of them.**
  - The support view shows the M-step's own ε, p-values, patterns and margins. It shows no re-derivation that could disagree with what the run did.
  - Where the run threw a quantity away (F1), it is written or replayed (§7). It is never approximated in the browser.
- **Nothing is silently truncated.** The genome browser's `cluster_pileup` members view keeps the first 6,000 by `alignment_id` and says nothing (survey; `cluster_pileup.py:441-446`). Every cap here is reported where it bites.
- **Structure is tested, not eyeballed.** "Do the colours cluster on this layout?" is a statistic: assortativity or Moran's I against a permutation null, over the edge layers that made the layout (§6.4). The browser offers the number beside the picture. This is invariant 1 applied to exploration.
- **The generic parts are generic.** The graph, layout, 2D interaction, picking and selection pieces land in `core` and in the viz engine, and are tested on non-transcriptome inputs. GNPS-style spectral networks and protein groups in mass spec have the same shape as the star map.
- **Notebook first, then browser** (invariant 10). Each kernel calls a domain read API that is also the notebook API, as `massspec.{scans, results}` is for the mass-spec browser.

## 3. The data model

### 3.1 Entities

| Entity | Identity | Count at 9.4M reads | Source |
|---|---|---|---|
| Read | `read_row` (run), `read_id` | 9.45M | `corpus/reads.arrow` |
| Template / node, per round | `template_id`; node key `(round, parent, haplotype)` | ~0.85M round-1 templates (em-kmer), then the round's nodes | `rounds/rNN/` |
| Cluster (final transcript) | `cluster_id`, plus a content id (review F8) | about 0.37–0.46M at r6 in older bench runs; current-code figure unmeasured | `clusters.parquet` |
| Sample | `sample_id` | 12 (WT / KI / KO) | demux `samples/` |
| Protein | content id of the sequence | ≤ clusters | `proteins.fasta` |
| Peptide (later) | `massspec.library.ids.peptide_id` | — | a link table (§9.5) |
| Reference feature (optional) | RefSeq transcript / gene id | — | an annotation table (§9.4) |

### 3.2 Two roles: relations position the stars, attributes paint them

This is the PI's distinction between *associative* and *predictive* features, written down so that the design can depend on it.

**An edge layer** is a typed relation between two entities of one kind. It is a table of `(src, dst, layer, weight, attributes…)`.
- Layers are chosen to build the layout or to group stars.
- Layers are also drawn as lines, but **which layers position the stars and which are drawn are separate choices**.

Candidate layers for clusters:

| Layer | Meaning | Exists? |
|---|---|---|
| `equivalent` / `contained` | the run's own template graph | yes (`cluster_edges.parquet`) |
| `staggered` / `internal_variant` / `divergent_5p/3p` | alternative ends, skipped segments: the "shared exon" relations | **computed and discarded** by `graph.py`; only counts are kept. Writing them is cheap (review §2.4) |
| lineage kin | descend from one round-1 template or seed cluster | derivable from lineage |
| protein similarity / homology | predicted proteins align | new (in-house k-mer sketch, or MMseqs2 as a thirdparty tool) |
| shared domain | same Pfam / InterPro domain | new (HMMER as thirdparty), later |
| same reference gene | best RefSeq hit, or genome-guided gene of the reads | new (§9.4) |
| shares a detected peptide | proteomics evidence | new (§9.5) |
| co-expression | correlated across samples | derivable from `feature_quant` |

**An attribute** is a column of the entity's table, and it drives colour, size, opacity and filters.
- **Continuous:** `n_reads`, length, ORF / UTR lengths, per-sample TPM, fold change, purity, noise conformity (§5.3).
- **Categorical:** annotation gene, "has protein", flags, sample group.
- **Set membership:** any named selection becomes an in/out attribute.

**The question the browser is built to ask** is whether attribute *A* clusters on the layout made from layers *L*. §6.4 answers it with a statistic as well as a picture.

## 4. The interaction grammar: what is a window, what is a tooltip, how things interlock

Everything follows one rule set:

| Surface | Triggered by | Holds | Lifetime | Fetches |
|---|---|---|---|---|
| **Tooltip** | hover | ≤ 6 fields and at most one sparkline; never anything clickable | while hovering | nothing beyond what is drawn (the attribute row is already on the client) |
| **Inspector** (side panel) | click / selection change | tabbed modules about the *current selection* (§5.2) | follows the selection | lazily, per open tab |
| **Panel** (dock tab) | "open" from an inspector or a context menu | a full view: star map, support, history, expression | until closed. **Pinned** panels stay on their entity; **following** panels track the selection | per view |
| **Selection set** | click, shift-click, lasso, filter expression, search | a typed id set with a name and provenance | session (savable) | — |

**Hover never changes state. Click selects. Open pins.**

**Selections are sets.** A lasso on the star map, "the reads of this node", "the nodes of gene *Tex35*" and "the round-3 ancestors of these clusters" are all sets of some entity kind. They are what flows between views:
- the history view shows a set's read flow;
- the support view lists a set's nodes;
- the expression panel compares a set with the rest.

A set can be:
- named and saved;
- used as a filter;
- used as a colour attribute (in/out).

**Linking** uses one typed bus per browser (`selection:set`, `focus:entity`, `round:changed`, `layout:changed`). The viz CLAUDE.md already prescribes this; the survey found it unbuilt in the mass-spec browser, which links imperatively. Each panel has a pin toggle: following panels re-target on `focus:entity`, pinned ones do not. This is IDEs' "link with editor".

**Layout** is dockview, which is already a dependency and already runs the dashboard. It is chosen over a fixed grid because the views are heterogeneous (map, matrix, tree, charts) and the PI wants several of them open at once.

A default arrangement:

```
┌──────────────────────────────────────────────┬──────────────────────┐
│  Star map                     [layers ▾]     │  Inspector           │
│   · · ✦ ·   ✦✦                [colour ▾]     │  ┌Summary┬Expr┬Prot┬…┐│
│     ✦──✦  ·   · ·             [filter ▾]     │  │ cluster 18342     ││
│   ·  ✦  ·     ✦─✦─✦           [sets ▾]       │  │ 1,204 reads · 1.6kb││
│        · ·                    A on L: I=0.41 │  │ ORF 412 aa  …     ││
│                                (p<0.001)     │  └───────────────────┘│
├──────────────────────────────────────────────┴──────────────────────┤
│  [Support: cluster 18342 📌] [History: Tex35 set] [Expression]       │
└─────────────────────────────────────────────────────────────────────┘
```

## 5. The views

### 5.1 Star map (the primary view)

- **Nodes.** Final clusters by default.
  - A round scrubber switches the map to any round's nodes (§5.4).
  - A default filter (e.g. `n_reads ≥ 2`) keeps the first view readable. The filter is visible and its count is shown, never implied.
- **Position.** A layout over the chosen layers with user weights (§6.3).
- **Visual channels.**
  - Colour: any attribute.
  - Size: `log n_reads`.
  - Opacity: the filter.
- **"Blink" has three modes:** dim non-matching, hide non-matching, or pulse matching (for finding).
- **Edges.**
  - Drawn per layer, each with its own style.
  - By default only for the selection's neighbourhood, or when zoomed in past a density threshold. 10⁵ nodes with all their edges drawn is an unreadable hairball.
- **Semantic zoom on the map.**
  - Zoomed out: density, plus *constellation* labels, i.e. components or communities named by their dominant annotation.
  - Zoomed in: individual stars and their edges.
- **Structure readout:** the §6.4 statistic for the current colour attribute on the current layout layers, with its permutation p.
- **2D or 3D:** decision D2 (§12).
  - The layout engine takes the dimension as a parameter, so 3D costs a camera, not a different algorithm.
  - The recommendation is 2D first, with 2D rotation. 3D adds occlusion and depth ambiguity, and publication export is 2D. Add an orbit view if the PI wants it after using 2D.

### 5.2 The inspector, a registry of modules

Each tab is a module: `{entity kind, title, applies(entity), fetch, render}`. "Modules could be added to do any level of analysis on the sequence" then means adding one without touching the shell. First modules:

- **Summary:**
  - lengths;
  - ORF and the 5′/3′ UTR lengths (from `orf_start/orf_end` on the consensus);
  - reads, rounds survived, rebuild/merge flags;
  - content id.
- **Expression:** per-sample counts and TPM grouped by condition, from `feature_quant` and `samples/`. This needs sample metadata (§9.3).
- **Protein:** the sequence, with domains and observed peptides later.
- **Neighbours:** the edges in each layer, sortable, each row selectable.
- **Support (compact):** the coverage, PWM agreement and arc tracks with no reads, plus "open support view".
- **History (compact):** the ancestry chain and reads per round.
- **Annotation:** the RefSeq hit, and the genome-guided gene/isoform make-up of the node's reads (§9.4).

### 5.3 Support view: the read matrix of one node

**Priority:** the PI's highest, and the view the bench needs most.

**Coordinates.**
- The x axis is the **PWM frame of the M-step that built the node**, the columns `F` of the review's §3. This is the only space where every read's every base, gap and insertion has a column.
- Two displays of it:
  - **consensus space** (the default): gap-winning columns hidden, insertions shown as IGV-style markers with counts;
  - **frame space**: every insertion column shown.
- The rulers give consensus coordinates (and template coordinates in frame space).

**Rows** are reads:
- grouped by node (one parent's children side by side when the view is opened on a split);
- ordered within a node by signature pattern, then start, then identity. Ordering by pattern is what makes the covariance *visible* when the rows are aggregated.

**Tracks above the matrix:**
- consensus sequence;
- coverage, as depth and base coverage;
- per-column minor-allele fraction, shaded by the column's ε. "Observed above noise" shows directly against what the error model allows;
- candidate columns, marked by route;
- **covariance arcs**:
  - each arc joins two candidate columns, above the axis, with height proportional to distance;
  - **colour = effect (φ or log-OR)**, opacity = −log₁₀ q, thickness = co-observed reads;
  - only significant arcs by default, filterable by signature;
  - signatures shown as coloured bands under the arcs;
  - hovering an arc highlights both columns and shows the 2×2 table;
  - clicking an arc re-sorts the reads by their joint state at the pair.

**Per-read side columns:**
- sample;
- identity and Dorado quality;
- the pattern-assignment margin (how sure the M-step was);
- `logl_delta` and the runner-up template (review F6);
- the genome-guided gene/isoform, when an align run is attached.
- Any of these can colour the row.

**Level of detail is a function of both axes, independently** (the PI's point):

| px / column ↓ \ px / row → | ≥ 10 (rows legible) | 1–10 (thin rows) | < 1 (rows aggregated) |
|---|---|---|---|
| **≥ 8** | letters at departures, faint letters at matches | coloured departure ticks | per row-bin × column allele fractions |
| **1–8** | coloured departure ticks, grey span | ticks | heat of departure density |
| **< 1** | per-pixel departure density per row | density | 2-D density image (rows sorted by pattern, so block structure shows) |

So a user can zoom x onto one domain with all reads as coloured ticks, or zoom both axes to read bases. Zoomed fully out, the transcript is a density image whose bands are the signatures.

**The payload is sparse, and that decides the architecture.** A read agrees with its consensus at ~98–99% of columns. Two things are sent:
- each read's span;
- its *departures*: `(column, allele)` where it differs, plus the insertion runs.

Size estimate (my arithmetic, not measured):
- At ~1.2% error over 1.5 kb that is ~18 departures a read.
- The M-step's cap of 20,000 members gives ~360k departures, a few MB of Arrow.
- So the whole matrix renders client-side at every LOD, with no server raster.

This is exactly the M-step's own `ProjectedMember` / `AlleleEvents` content (review §2.3), so it comes from replay (§7.2). Nothing new is computed.

**Above the cap.** A node with more reads than the M-step sampled is shown on the sample the M-step used, labelled with `subsample_fraction`. The unsampled tail can be added explicitly, by projecting reads through the E-step CIGAR with no PWM involvement, and labelled as such.

**Comparison mode.** Two nodes (siblings of a split, or a merge pair) are shown on their shared parent frame, restricted to the columns where they differ. This is the 99.92% C/T pair and the Ldhc poly-A split from the review's §5, as a view.

**Noise conformity** is a physically grounded purity check. A node of one transcript should be "thermal": each read's departure count should follow its expected error. That expectation comes from length × the error model, or from Dorado quality.
- The view plots observed against expected departures per read, plus the node's overdispersion.
- Reads hotter than thermal are candidate mis-assignments or unresolved variants.
- This is the bench's finding #76b (73% of unassigned reads carry 2–4× their predicted error) turned into a per-node attribute that can colour the star map.

**Round stepping.** For a lineage, the same reads are shown under each round's template. This shows how a split was reached.

### 5.4 History view

**Genealogy.**
- The lineage forest runs from round 0 (seed clusters, em-kmer) through round 1 … round N to the final merge, with x = round.
- Event types: `carry`, `split`, `merge` (which makes it a DAG), `unrecruited`.
- Chains of `carry` collapse by default, so only events show.
- It is always shown **for a set** (a gene, a family, a lasso), never globally. Millions of lineage rows have no useful global picture.

**Read flow.** An alluvial plot of where a set's reads sat in each round.
- Bands are coloured by whether a move was inherited (parent → own child) or a genuine switch, the two halves of `measure_churn`.
- This is Gsg1 / Hnrnpk / Tex35 (ledger #72) made visible.

**Per-round dashboard:** small multiples of the `round.json` counters (churn, newly lost, cap hits, nodes, merges).

**The star map in history mode.**
- Round *r*'s nodes start at their parent's position from round *r−1*, plus a small offset, and then relax under the round's own layers. Stars visibly split and merge as the scrubber moves.
- This is the PI's "initial spacing as the ancestral phylogeny", made exact: the lineage is the layout's prior, a warm start with anchoring (§6.3).
- The same mechanism keeps the final map stable when layers or filters change.

**Merge events** show both forms' support and the survivor rule's choice, the merge forensics of ledger #52 and #55.

### 5.5 Seed view (em-kmer)

For a seed cluster, show its members against the elected read, with each difference classified as *block* (one indel ≥ 15 nt) or *scattered*. This is `seed_member_template_diff.py`, the orphan cases of ledger #71, #73 and #76.

### 5.6 Unassigned reads as a population

Reads with `template_id = −1`, grouped by `unassigned_reason`, shown with:
- their best identity against Dorado quality (the floor-calibration plot);
- their runner-up or best-identity template, once F6 lands;
- their fate across rounds.

These reads are not on the star map, because they belong to no node, so they need their own entry point.

## 6. The generic engine: what `core` gains

### 6.1 What exists

From the survey:
- **`core.graph`** is a pure-Python `Network` (a dict of lists, DFS components) and a `Tree`. No weights, CSR, kNN or layout.
- **`core/optim/_layout.py`** is *parameter* layout (flattening an `nn.Module`), not graph layout.
- **Fast graph code** lives in sequencing: scipy CSR components in `cluster_graph.py`.
- **Union-finds:** two hand-written ones in massspec.
- **No dependency** on networkx, igraph, UMAP, pynndescent or scikit-learn.

### 6.2 `core.graph`: a layered sparse graph

**`LayeredGraph`** (name to settle). Arrow at the edges and CSR inside:
- **Tables:** a node table and an edge table `(src, dst, layer: dictionary, weight, attrs…)`, both registered schemas.
- **Indices:** CSR per layer, as numpy / `torch.sparse_csr`.
- **Operations:**
  - weighted mixing of layers;
  - attribute predicates on edges;
  - induced subgraphs;
  - components (scipy `csgraph`);
  - top-k sparsification;
  - symmetrisation.

**kNN from vectors** (expression profiles, protein embeddings):
- blocked exact `torch.cdist`, so no new dependency;
- an approximate method only if measured necessary.

**`Network`** keeps its API. Moving its internals onto CSR is a separate change.

### 6.3 Layout as an optimisation problem (invariant 6)

**The objective.** A layout is positions `X ∈ ℝ^{N×d}`, held as an `nn.Parameter`, that minimise a divergence between the graph's affinities and a low-dimensional kernel.
- **UMAP's objective:** fuzzy cross-entropy with negative sampling, under the kernel `(1 + a‖x‖^{2b})⁻¹`.
- **t-SNE's:** the Student-t kernel, which `core.stats` already has as a distribution.
- **Stress / force-directed:** for small subgraphs.

Each is an `nn.Module` loss optimised by `core.optim`'s registry (Adam). This is the same machinery that fits peaks and Counter, not a vendored UMAP.

**What the browser needs from it:**
- **Warm start and anchoring:** a penalty pulling each node towards a prior position. This is the history animation and layout stability.
- **Pinning.**
- **Component packing:** lay out each component, then place components by a weak global layer (expression, protein sketch), or pack them. Most clusters are singletons or small families, and a pure force layout of a disconnected graph is a scatter of debris.
- **Determinism:** seeded.
- **Progress:** positions streamed every *k* epochs.

**Placement in the import DAG.** The module needs the graph, the optimisers and the stats kernels. It cannot sit in `core.graph`, which is upstream of `stats`. The recommendation is a new module after `stats`, named for what it does (e.g. `core.embed`); the name is decision D6.

**Where it runs.** Server-side, in torch.
- A rough guess: ~10⁶ edges per epoch vectorised is tens of ms on CPU. This is to be measured, not assumed.
- The client only renders, plus a small local relaxation while dragging.

### 6.4 Statistics of an attribute on a graph

These are the "predictive" readout:
- **categorical:** assortativity;
- **continuous:** Moran's I / Geary's C;
- **enrichment of a selection set** against the rest.

Each comes with a permutation null. They are generic and small, and they belong beside the layout.

### 6.5 Generality check

Before the transcriptome uses any of this, B's tests (§11) build a spectral-similarity network from mass-spec spectra (cosine on binned peaks, kNN, layout). If the abstraction only fits transcripts, that test will show it.

## 7. The data contract: what the sequencing side must write or replay

These are the review's findings turned into work. All of it is useful in a notebook before any browser exists.

### 7.1 Always written (small)

| # | Table / change | Fixes | Size |
|---|---|---|---|
| A1 | `mstep/templates/`: one row per template per round. Holds the `diag` dict `refine_template` already builds, a `status` (ok / failed + error class), and the error-model ε actually used | F2, F5, F9 (recording) | templates per round |
| A2 | `cluster_nodes.parquet`: `cluster_id → (final round, node key, graph template id, rebuilt, absorbed ids)`, plus content ids for transcripts (xxh64 of the consensus) and proteins. A content id is an *attribute*, not a key, because byte-identical twins exist | F3, F8 | clusters |
| A3 | `runner_up_template_id` on `EM_ASSIGNMENT_TABLE` (nullable) | F6 | one int64 per read per round |
| A4 | `mstep/columns/` (candidate columns: frame and template position, route, run length, major/minor, depth, minor fraction, ε, effect, p, kept, signature) and `mstep/covariance/` (u, v, n11, n1x, nx1, n, p, significant). Per-read `margin` on node membership | F1 (compact), F7 | most templates have no candidates; **to be measured on a real round before it is made default-on** |
| A6 | Opt-in: the relations `graph.py` classifies and discards (`staggered`, `internal_variant`, `divergent_*`) as edges | the "shared segment" layer | edges |

### 7.2 Replayed, never all written

**What replay is.** `replay_template(run_dir, round, template_id)` re-runs `refine_template` for one template, from:
- the round's templates;
- its assignment slice;
- the corpus;
- the run's recorded parameters.

It returns the full evidence: PWM, plan, coordinate maps, read projections, states, covariance and patterns. The subsample is already seeded by template id, and the loop is deterministic (ledger #56–#62), so a replay reproduces the run.

**Verification is built in.** The replayed nodes' consensus strings must equal the stored ones. If the code has changed since the run and they differ, the replay is refused or flagged, never shown as if it were the run.

**Why replay rather than a full dump.**
1. **Size.** A full PWM per template per round is (my arithmetic) ~10⁶ templates × ~1.5 k columns × 5 × 4 bytes ≈ 30 GB a round at 9.4M reads.
2. **It works on the runs already on the bench node** (if the consensus check passes), without re-running 9.4M reads.
3. **One template costs seconds to a minute or two** (measured after the P1.7a fix: 1,000 reads × 1.5 kb in 5 s, 20,000 × 2 kb in 76 s plus 13 s for the alleles; it was an unmeasured "seconds" here, and 50 min before the fix). Acceptable for "open support view" with a cache and a progress indicator; not instant.

**Where the dense data comes from:**
- **Opt-in full dump:** `--emit-mstep-detail {none, summary, full}` with a template or gene filter, for offline work.
- **Final rebuilt survivors** (review F4): replay of `rebuild.py`'s pooled path.
- **Reads under the final consensus:** reads are projected through the replayed coordinate maps, T → F → C. That projection is the support view's payload.

### 7.3 A read API

`sequencing.transcriptome.cluster.denovo.em.results` (name to settle) is the notebook and browser surface: open a run, look up nodes, clusters, lineage, a read's history, and replay. Its pure functions are the transcriptome modality's `DOMAIN_IMPORTS`, as `massspec.{scans, results}` are for mass spec.

## 8. The viz platform: generic work that the other browsers also gain

From the survey, the engine today has:
- SVG only;
- pan/zoom on x only, with the wheel always zooming x;
- no hit-testing;
- no selection model;
- no side panel;
- Arrow responses decoded only after the whole body arrives;
- a `SavedSession` with fixed keys only.

| # | Change | Also serves |
|---|---|---|
| C1 | 2-D pan/zoom with independent x and y zoom (`attachPanZoom2D`; `Interval` per axis) | the mass-spec RT × m/z map (plan step H3, unbuilt) |
| C2 | A raster layer: Canvas2D for the read matrix and **WebGL for the star map** (10⁵–10⁶ points: SVG cannot), with picking (a spatial index or colour-id buffer) | the genome pileup, whose reads today get no letters and mismatch *positions* only |
| C3 | Selection sets, a typed bus used in practice, the tooltip / inspector / pin components of §4 | both browsers (the mass-spec drawer is a special case) |
| C4 | An opaque per-modality block in `SavedSession` (layers, weights, colour-by, filters, named sets, pinned panels) | mass spec (whose saved views are a deferred item) |
| C5 | A **generic alignment-matrix renderer** (rows × columns from spans + sparse departures, with the LOD table of §5.3) | the genome read pileup could adopt it as-is |

**Export.** The viz CLAUDE.md says `Save SVG` "always offers a vector export", and the survey found that untrue today: export serialises the hybrid PNGs. With canvas and WebGL in play, the policy has to be written down:
- re-render the *visible* LOD as vectors under a glyph budget;
- otherwise embed a raster under vector overlays and say so in the file.

**WebGL library:** decision D3. Two options:
- **deck.gl** (`@deck.gl/core` + `layers`, usable without React): picking, orthographic and orbit views, transitions;
- **regl** with hand-written picking: small and explicit.

The recommendation is a one-day spike rendering 500k points with picking in each, and choosing on the result.

## 9. The transcriptome modality

### 9.1 Session

**Sources:**
- an EM run directory (required);
- the demux directory and `samples/`, from its manifest;
- optional: an align directory (genome-guided read labels), a reference handle, annotation tables, a proteomics link table.

**Kinds** are read from the manifests, as mass spec does.

### 9.2 Kernels (`tx_*`)

- Star map nodes and edges.
- Layout: compute and stream.
- Node detail.
- Support (spans + departures + tracks, from replay).
- Covariance.
- Lineage and read flow.
- Expression.
- Unassigned.

Every kernel has a query model that declares its caps, and every response says when a cap was reached.

### 9.3 Samples and conditions

`Samples` has `sample_name` and `description` but no structured condition. The 12 testis samples are WT / KI / KO, and that grouping is only in someone's head. A sample-metadata table, `sample_id` → arbitrary typed columns, joined at session open, makes conditions attributes and fold changes computable. It belongs in sequencing's `Samples`, not in the browser.

### 9.4 Reference-aware layers (optional inputs)

- **Genome-guided read labels:** `transcriptome align` on the same demux gives each read a gene and an isoform fingerprint. Joined by `read_id` and projected onto `read_row`, this colours the support view's rows and gives each node a purity. This is how the bench judges partitions (`gene_partition_report.py`).
- **RefSeq hits:** a cluster → reference transcript/protein table from a mapping step. This is a sequencing verb, not browser code. The bench's `refseq_compare.py` and `fidelity_recall.py` are its prototype.

### 9.5 Proteomics

**The import rule.** The domain modules never cross-import, and a viz modality may import from one domain (`CLAUDE.md`).

**Recommended route.** A top-level cross-modality workflow, the pattern `transcriptome_to_proteome.py` already sets, writes a **link table**: cluster content id ↔ protein ↔ peptide (`massspec.library.ids.peptide_id`) ↔ evidence per sample. The browser reads it as a plain table. Peptide ids are content-derived, so they agree across runs, which is the point of F8.

**What it gives:**
- per-cluster evidence attributes;
- a "shares a detected peptide" layer. On the star map that layer *is* protein inference's indistinguishable groups, drawn.

## 10. Scale budget

These are bench numbers at 9.4M reads, from the review and ledger.

| Quantity | Size | Consequence |
|---|---|---|
| Final clusters | ~4 × 10⁵ | WebGL map; the default filter; density when zoomed out |
| Reads per node | up to 10⁵+ (Prm1 etc.); M-step sample ≤ 20k | sparse payload; aggregated LOD above ~2k rows |
| Departures per support view | ~10⁵–10⁶ | a few MB of Arrow, rendered client-side |
| Lineage rows | ~rounds × nodes, 10⁶–10⁷ | always per set, never global |
| Assignment rows per round | 9.45M | read flow per set, by `read_row` scatter (as `measure_churn` does) |

## 11. Orchestration

There are four tracks. **A** and **B** have no dependency on each other or on the frontend.

```
A  sequencing data contract ──┬──────────────► D2 support view ──┐
                              │                                   ├─► D3 history ─► D4 layers ─► D5 proteomics
C1/C2/C5 viz 2D + raster ─────┘                                   │
B  core.graph + embed + graph stats ──► C3/C4 ──► D1 star map ────┘
```

**Proposed PRs.** Each PR has one commit per phase.

1. **A: the EM evidence contract.**
   - **Contents:** A1, A2, A3, A4 behind a flag until measured, the replay API, the read API, and fixes for F5 and F10.
   - **Use on day one:** the bench's forensic scripts can be rewritten against it.
   - **Compatibility:** all additive. Older runs simply lack the new tables, and replay still serves them.
2. **C1 + C2 (Canvas2D) + C5, plus D2 on a minimal session.**
   - The support view, entered from a node list or search rather than the star map.
   - It is the PI's top priority, and it validates the LOD and arc design on real data before the larger investment.
3. **B: `core.graph` layered CSR, `core.embed`, graph statistics.**
   - Tested on synthetic graphs and a mass-spec spectral network.
   - Can run in parallel with 1–2.
4. **C2 (WebGL) + C3 + C4 + D1:** the star map, inspector, selection sets and saved views.
5. **D3:** history (genealogy, read flow, round scrubber, history-mode map).
6. **D4 / D5:** reference layers and protein similarity, then proteomics links. Each needs a pipeline verb first.

**Development data.** The bench's run directories are on the bench node, not on this workstation, so development needs a local run:
- the 172,411-read / 323-gene panel or the 100k smoke test, run end to end here;
- plus one copied round of a full run for scale checks of A4's sizes.

Real-scale timings are the PI's to run.

**Bench targets for acceptance** (the review's §5):
- Gsg1 / Hnrnpk / Tex35 (read flow);
- Tuba1a t302020 (support);
- Ldhc `2199023680430` (split comparison);
- the 99.92% C/T pair (comparison);
- Usp39 / Kif2b / Uchl3 (merges);
- Ppp4c / BC048671 / Gm5646 (seed view);
- H3f3b codon 133 and Akap4 840/849 (minority proteoforms).

## 12. Decisions

Decided by the PI on 2026-11-14 unless marked open.

| # | Decision | Outcome |
|---|---|---|
| D1 | Where the M-step evidence comes from | **Decided:** compact tables always, dense evidence by verified replay, a full dump opt-in |
| D2 | 2D or 3D star map | **Decided:** 2D first, possibly only. Build fundamentals generically where it costs nothing; 3D must not drive choices |
| D3 | WebGL dependency | **Open until PR4:** a spike of in-house WebGL2 against deck.gl at 500k points (§14) |
| D4 | First additional edge layers | Open (PR6): the discarded graph relations and genome-guided gene recommended first |
| D5 | Proteomics link | Open (PR6): a top-level workflow writing link tables recommended |
| D6 | Name and DAG position of the layout module | Open (PR3): a new `core` module after `stats` recommended |
| D7 | The error-model refit (F9) | **Decided:** wanted, deliberately deferred until the browser makes clustering quality assessable. PR1 records the substrate (`mstep/disagreements`) and the docs say the loop does not refit |
| D8 | Sample metadata | Open (PR6): a typed sample-metadata table in `Samples` recommended |
| D9 | Order of work | **Decided:** PR1 evidence contract → PR2 support view; PR3 engine in parallel; PR4 star map after it |
| D10 | A local development run | **Decided:** the PI downloads a dataset after PR1 lands; PR1 is developed on synthetic fixtures |

## 13. PR1: the EM evidence contract (built)

One commit per phase on `feat/em-evidence`:

| Phase | What |
|---|---|
| P1.0 | EM tests skip per test, not per module, so minimap2-free tests run everywhere; `tests/_em_fixtures.three_gene_panel` (ORF seeding + native E-step, no minimap2) that splits, merges and rebuilds, asserting its own preconditions; golden digests of nodes, membership and clusters |
| P1.1 | `rNN/params.json` at round start; the kernel's hidden defaults made `MStepParams` fields |
| P1.2 | F12: a round's M-step shards are the ones it recorded (`em/round_io.py`) |
| P1.3 | One per-template code path (`refine_one_template`); fallbacks instead of lost reads; `mstep/templates`, `mstep/disagreements`; the report's M-step section |
| P1.4 | Compact evidence: `mstep/columns`, `mstep/covariance` (2×2 counts, φ), `mstep/reads` (margins); `--mstep-evidence` |
| P1.5 | The E-step runner-up; `logl` null, not NaN |
| P1.6 | `cluster_nodes.parquet`; `core.ids` |
| P1.7a | `frame_consensus`'s scatter and `member_alleles` vote a read's empty insertion blocks at once (bit-identical): one template at 5,000 reads × 3 kb, 849 s / 6.6 GB → 28 s / 1.8 GB — the run's own M-step too |
| P1.7 | Verified replay, the support tables, the read API (`em/results.py`), `scripts/bench-em-replay.py` |
| P1.8 | Documentation |

Replay verified every template and cluster of a run made by the code of `a633080`, before any of these records existed. **For PR2:** replay is not instant — 20,000 reads × 2 kb take 76 s plus 13 s to read every allele, at 3.8 GB (`scripts/bench-em-replay.py`) — so the support view needs an on-disk replay cache per session, a progress indicator, and one replay at a time; the opt-in full dump (D1) is the way to precompute the templates a session will open. Real-data checks are the PI's: the evidence tables' sizes, the fallback rate, and the share of templates that verify, on the downloaded dataset.

## 14. WebGL: the trade-offs (decided at PR4's first step)

Only the star map needs it — 10⁵–10⁶ points, edges, picking, animation. The support view is Canvas2D (its level-of-detail table bounds what is drawn per frame), so nothing before PR4 waits on this.

| Option | For | Against |
|---|---|---|
| Canvas2D + semantic zoom (no WebGL) | no dependency; a density image zoomed out, points only where few are visible | full-scale redraws ~50–200 ms; split/merge animation over many stars impractical; edges expensive |
| Thin in-house WebGL2 (`engine/gl/`) | no dependency, matching the frontend's minimal-deps history; the star map needs a narrow set (instanced points, lines, id-buffer picking, a 2D camera, a filter uniform, position interpolation); camera and picking transfer to later 3D | we own shaders, buffers, context loss, HiDPI and anti-aliasing (~1–2k lines); rendering testable only in headless Chrome |
| regl | small, less boilerplate, proven in scientific viz | WebGL1 API, little recent maintenance; picking, camera and text still ours |
| deck.gl (`@deck.gl/core` + `layers`, no React) | GPU picking with events, orthographic and orbit views, transitions, typed-array attributes, GPU attribute filtering ("blink by filter"), maintained | the largest dependency (several hundred kB); its own view-state model; major-version churn (v8 → v9); geospatial-first |
| three.js | the general 3D engine, a huge ecosystem | 2D data-vis ergonomics (camera, lasso, labels, fast point picking) all ours; frequent breaking releases |

On 3D and molecules: what transfers is renderer-independent — selection sets, the bus, the inspector, picking, camera math. Molecular rendering (cartoons, surfaces, impostor spheres, selection algebra) is a specialty in itself, and embedding Mol\* (the PDB's MIT-licensed viewer) is very likely right for a structure modality whatever the star map uses. So the star map is chosen for the star map: spike in-house WebGL2 against deck.gl at 500k points (frame time while panning, picking latency, filter toggling, bundle size, lines of our code) and lean in-house unless deck.gl's built-ins clearly save more than their weight.
