# Search space: FASTA → a provenance-keeping target graph → sweep targets

Status: built on `feat/search-space` (2026-11-05). The browser's typed-peptide path is the next
phase (end of this file); scoring is deferred.

## Why

The sweep (`massspec.sweep`) matches queries against an acquisition in one pass, and the only
thing that made queries from a FASTA was a bench script. The PI's framing (2026-11-05):

- FASTA is the primary input.
- The generative chain protein → peptide → peptidoform → precursor → fragment is a graph with many
  paths to the same signal, and what each step did is information a scorer may need later. It must
  stay visible rather than be collapsed early, and the protein → peptide collapse the old digest
  made is exactly the thing to avoid.
- Purified proteins come first, so the enumeration has to go wide (most conceivable modifications
  and cleavages) without exploding silently.
- Decoys are an engine that several methods pass through.

Decided with the PI: extend `Library` rather than add a container; the command is `massspec
search-space`; decoys default to EncyclopeDIA's rule; generator first, browser next.

## What it is

`massspec.library.search_space.build_search_space(fastas, SearchSpaceSettings) -> Library`,
behind `constellation massspec search-space`. The output is a library directory that
`massspec chromatogram extract --targets` and `predict-library --from-library` already read.

### One table per stage

| Stage | Table | Key | Holds |
|---|---|---|---|
| input | `PROTEIN_TABLE` (+ `is_decoy`) | accession (content id) | sequence, description |
| digestion | `PEPTIDE_SPAN_TABLE` (new) | (protein_id, start, end) | sequence, missed cleavages, `n_term` / `c_term` ∈ {enzyme, protein, nonspecific}, flanking residues |
| modification | `SPAN_PEPTIDE_EDGE` (new) | (span, peptide_id) | which spans can make which peptidoform |
| | `PEPTIDE_TABLE` (+ `is_decoy`, `decoy_of`, `decoy_set`) | canonical ProForma (content id) | sequence, modified_sequence |
| | `PEPTIDE_MODIFICATION_TABLE` (new) | (peptide_id, site, modification, origin, rule) | position or terminus, UNIMOD accession or isotope, origin (fixed / variable / label), the rule; a form two paths make by different rules has a row for each |
| charge | `PRECURSOR_TABLE` | peptide key + charge (content id) | charge, m/z |
| fragment | none | — | compiled by the sweep (`compile_queries`) into `SWEEP_CHANNEL_TABLE` |

The search space stops at precursors and records nothing about fragmentation, so one search space
serves any fragmentation settings.
- A molecule appears once, under its content-derived id, with an edge to every span that makes it.
- `PROTEIN_PEPTIDE_EDGE` is the projection of the span edges, checked equal.
- A Protein N-term modification is linked only from spans at a protein N-terminus. These are the
  protein's first residue, the residue after an initiator Met, or a declared mature-chain start
  (`--protein-terminus P02769=25-607`).
- Isotope labels are a modification axis: light and heavy forms of a protein's peptides are
  linked to the same spans. A label (`<13C><15N>`) covers the peptide chain, its residues and
  termini: atoms a modification adds are natural (light iodoacetamide on a labelled protein) and
  atoms it removes come off the labelled chain (PI, 2026-11-05; "Review by Codex" below).

### Rules

- **Spelling:** `MOD[@SITE,...][:max=N]`. MOD is a UNIMOD accession or name. A site is a residue,
  `N-term`, `C-term`, `Protein N-term` or `Protein C-term`.
  - With no `@`, the sites UNIMOD shows are used; for a modification shown nowhere (SILAC K), its
    hidden sites.
  - `@*` means every placement UNIMOD lists.
- **Placement:** a rule places its modification only where UNIMOD lists it
  (`enumerate_modforms(skip_unlicensed=True)`), so one rule set applies across every peptide:
  pyro-Glu goes on an N-terminal Q and nowhere else.
- **One modification per slot, and what may share one:** a residue carries one chemical
  modification and, beside it, at most one isotopic substitution (SILAC lysine or arginine, which
  makes the residue's own atoms heavier and leaves its amine free), so heavy lysine has its
  acetylated form. An N-terminal cyclization of the first residue's side chain (UNIMOD:26 from C,
  27 from E, 28 from Q, each defined from the unmodified residue) replaces that residue's chemical
  modification rather than adding to it: with carbamidomethyl fixed, the cyclized form is
  `[UNIMOD:26]-C…`, not `[UNIMOD:26]-C[UNIMOD:4]…`.
- **Refusals:**
  - a rule naming a site UNIMOD never lists (`Oxidation@A`);
  - two fixed rules that would put different modifications on one slot of some peptide (an
    `N-term` and a `Protein N-term` rule both licensed for its first residue), which is a
    contradiction, not a precedence; rules on different residues pass (pyro-Glu from Q and from E);
  - a modification that is one end of a link between residues (half a disulfide, UNIMOD:374; an
    intact crosslinker, "Intact … crosslinker"), since a search space holds linear peptidoforms
    and linked species are not generated; quenched monolinks pass;
  - isotopic-substitution rules together with whole-protein labels, which would label one
    residue's atoms twice;
  - a label with two isotopes of one element (`33S,34S`) or an isotope the element lacks; `D`
    and `2H` are one label (written `2H`);
  - a `--protein-terminus` naming no protein given.

### Decoys (`massspec.library.decoys`, `DECOY_METHODS`)

| Method | Level | What |
|---|---|---|
| `reverse-interior` (default) | peptide | first and last residue kept, the interior reversed (EncyclopeDIA 6.9.23) |
| `shuffle-interior` | peptide | both termini kept, the interior shuffled, seeded by the target's sequence, the set and the seed; for extra and entrapment sets |
| `reverse-c-fixed` | peptide | pseudo-reverse, for comparison |
| `reverse-protein` | protein | proteins reversed before digestion (FragPipe-style), for comparison |

**Peptide-level decoys** permute a target's residues with `permute_peptidoform`: modifications
move with their residues, while terminal modifications and labels stay put. A decoy therefore has
the target's composition, precursor m/z and isolation windows.

**Collisions and seeding.** A candidate whose I/L-normalised sequence is a target's falls back as
EncyclopeDIA does: to seeded terminal-preserving shuffles, and the pair is dropped if none
qualifies. Seeding by sequence, and settling sequences in sorted order (two can propose one
decoy), make a set independent of input order and give all forms of a peptide one decoy sequence.
A reversed protein's peptide that is a target's under I/L is dropped too, and counted.

**Sets.** Set 0 competes with the targets; sets 1 and up are extra or entrapment sets.
`--decoys METHOD[:sets=N][:seed=S]` is repeatable.

**What carries the decoy flag.** `is_decoy` is carried into `XIC_TARGET_TABLE` (v2), the sweep's
queries and its query summary (sweep schemas v3). This replaces the bench's `target_id` offset,
which gave modified and labelled targets no decoys.

### Budgets

- The number of peptidoforms is counted before any is built (`count_modforms`, a DP over the
  enumeration's slots, polynomial in their number: thirty methionines with oxidation count to
  2^30 at once; an upper bound, since a form made by two contexts counts twice). Enumeration
  under a per-modification limit refuses a choice before extending it, so its work is
  proportional to what it returns.
- The number of precursors is counted before any table is built.
- Over either budget, the build raises with the counts rather than truncating.
- `--count` prints every stage's count and writes nothing.

## Review of the machinery under it (2026-11-05)

### Fixed on the way

**Digestion (`core.sequence.protein.cleave`)**
- Spans carry how each end was made and their flanking residues.
- Semi-specific digestion now floats either end; it used to float only the N-terminus.
- New: `specificity="unspecific"` (every span in the length bounds) and declared protein termini.
  `No_enzyme` still means "never cut".
- A Met-clipped span is no longer dropped when its unclipped parent is one residue over
  `max_length`. This is the one difference from the old digest: 1, 12 and 1 extra precursors on 300
  human proteins under three settings, all such peptides.

**Modification enumeration (`enumerate_modforms`)**
- A `SiteContext` says whether the peptide sits at its protein's terminus, with sentinels
  `Protein N-term` / `Protein C-term`.
- Per-modification limits (`max_per_mod`).
- `skip_unlicensed`.
- `return_origin`, giving each modification's slot, key, fixed/variable and the rule site.
- Validation against every UNIMOD placement, hidden included.
- `count_modforms`.

**Mass (`peptide_mass` / `peptide_composition`)**
- These were 104–133 µs per form, one torch add per residue.
- `peptide_masses` / `peptide_compositions` are batched: residue counts in one pass, each distinct
  `ModRef` resolved once, then `torch` `batched_mass`.
  - Batched: 7–10 µs per form.
  - Single item, as its one-item case: about 60–80 µs.
  - Both bit-identical to the old functions on 33,750 forms, every modification kind included.
- Callers that benefit: `precursor_mz`, `neutral_envelope` (the sweep's MS1 channels, Counter,
  `spectra.annotate`).

**UNIMOD build (`scripts/build-unimod-json.py`)**
- The build skipped every specificity marked `hidden` (2,747 of 2,826), but that is a Mascot
  display flag, not a judgement on the chemistry: SILAC K/R and GlyGly-K are hidden. The build also
  dropped classification and all 1,713 neutral losses.
- Everything is kept now. `Modification.specificities` is still exactly the shown set, held for
  all 1,560 modifications, so the EncyclopeDIA reader's disambiguation is unchanged.
- `hidden_specificities`, `all_specificities`, `Specificity.classification` and
  `Specificity.neutral_losses` are new, along with `Modification.licenses(site, terminal_residue,
  protein_terminus)`, which separates Any from Protein termini.

**FASTA I/O (`core.io.fasta`, `core.io.text`)**
- The package had 4 parsers, 2 header scanners, a counter and a dozen writers. One of them (the one
  the digest used) read no gzip and kept `\r` from CRLF files.
- One module now reads (`iter_fasta`, `read_fasta`, `iter_fasta_batches`, `count_fasta_records`),
  writes (`write_fasta`, `format_fasta`) and parses headers (`parse_header`: UniProt, NCBI
  brackets, bare tags).
- **Migrated, each held to its old output:**
  - `sequencing.readers.fastx`, and the reader copy in `transcriptome_to_proteome.py`;
  - `search.novel.read_fasta_proteins` and both gene maps;
  - the de novo, EM corpus, template, E-step, demux, genome-FASTA and genome-guided `cluster.fa`
    writers.
- `open_text`, which detects gzip from the magic bytes, replaces seven private copies of
  `_open_text`.
- The FASTA `RawReader` stubs are gone; a FASTA is not instrument data.

**Library**
- New: the span, edge and modification tables, the decoy columns, and Arrow-side validation
  (`pc.is_in`, group_by) in place of Python sets for every check.
- `subset_proteins` cascades through spans, and a decoy follows its target.
- ParquetDir writes the new tables only when they have rows, so other libraries' directories are
  unchanged.

**Sweep hook-up**
- `targets_from_library` keeps the precursors' order through its join (query numbers are row
  positions) and carries `is_decoy`.
- `chromatogram extract --own-charge` matches MS1 at each target's own charge. The default
  `charge_range=(1, 4)` gave each of a peptide's precursors copies of the same MS1 channels; it
  stays the default because Counter models a peptide across charges.

**`digest.precursors_from_fasta`** is now a projection of the search space: one enumerator. On
300 human proteins it gives the same output in the same order, the Met-clipped boundary aside, at
1.4 / 3.5 / 8.3 s against 3.4 / 11.6 / 24.1 s. The Koina path's hard-coded C+57 / M+16 pass
through unchanged.

### Found, not fixed here

- **Counter CLI** builds `Peptidoform(sequence=modseq)` from ProForma strings
  (`massspec/cli/counter.py:175,225,290,516`), so a modified target breaks `counter estimate`.
  Counter is frozen pending its revision.
- **ProForma:**
  - adduct charges do not parse;
  - UNIMOD titles containing `>` cannot be parsed (the generator writes accessions);
  - a MultiPeptidoform with global labels formats to invalid text;
  - a multi-CV reference keeps its first entry only;
  - mass deltas format as `+-0.0` / `+1e-05`, and the second re-parses as a name;
  - lowercase and whitespace are accepted;
  - `Peptidoform` is unhashable;
  - `ModRef` equality depends on the producer.
- **Chemistry and sequence:**
  - `ModVocab.find_by_mass` is a linear scan of about 15 ms;
  - U and O have no mass;
  - `nucleic.find_orfs` can return ORFs with internal stops.
- **Neutral losses:**
  - `tests/test_biochem_mask.py` checks a mask production does not call;
  - a terminal modification is treated as residue 0's side chain by loss triggers;
  - a bare-mass phospho never triggers H3PO4.
- **Sweep and library internals:**
  - `isotope_offsets=()` crashes;
  - the gate counts (ion type, position), not bonds;
  - `assign_ids` mishandles duplicates;
  - `Network` claims a CSR index it does not build.
- **Channel generation exists five times:** the sweep, Mode B, `spectra.annotate`, Counter
  candidates and `results.trace_targets`. The search space adds no sixth.
- **SHA-256 of a file is written eight times** across the package; `core.stages.file_sha256` is
  the one new code should use.
- **The FASTQ readers and writers** are the next candidate for the FASTA treatment.
- **The frozen pipeline's own FASTA writers** in `transcriptome_to_proteome.py` are left to its
  revision.

## Review by Codex (2026-11-05), and what was done

The branch was reviewed independently ([docs/reviews/search-space-2026-11-05.md](../reviews/search-space-2026-11-05.md)),
with a regression suite of 125 cases (`tests/test_search_space_review.py`), 21 of them strict
expected failures reproducing twelve defects and one inherited limitation. All 21 are fixed and the
markers removed.

| | Finding | Done |
|---|---|---|
| R1 | A protein's ¹³C/¹⁵N label also labelled the atoms a reagent added (light carbamidomethyl came out 3.0037 Da heavy per cysteine) | A global label covers the chain (the PI's choice of three options, 2026-11-05). ProForma 2.0 §4.6 reads `<13C>` as "13C on all residues" and Formula Rule 3 gives an atom with no isotope stated the natural distribution, so a modification's added atoms are natural; the atoms it removes come off the labelled chain (deamidation and pyro-Glu lose a ¹⁵N). `core.sequence.protein.labelled_compositions` says which atoms a label covers, and precursor mass, fragment ladders (neutral losses included: an NH₃ lost from a ¹⁵N fragment is ¹⁵NH₃) and envelopes all use it. Limitation: a PTM made in vivo from labelled metabolites is computed light; such a form would need an explicit labelled formula. |
| R2 | Fixed SILAC lysine hid variable lysine modifications | `Modification.is_isotopic_substitution` (a mass override over an empty skeleton): a residue holds one chemical modification and one isotopic substitution beside it |
| R3 | Fixed carbamidomethyl plus UNIMOD:26 counted the carbamidomethyl twice | `SIDE_CHAIN_CYCLIZATIONS` (UNIMOD:26, 27, 28) replace the first residue's chemical modification; a deamidated Q is no longer also Gln→pyro-Glu |
| R4 | Half disulfides and intact crosslinkers were placed as one-residue modifications | Refused at the rule, monolinks kept |
| R5 | Counting without limits visited every combination (2^30 for thirty methionines); a limit filtered after enumerating | The count is always the DP; enumeration under a limit (or a cyclization guard) is a depth-first walk that refuses a choice before extending it, sorted into the old order; with neither it is `itertools` as before. The same speed per form as before on BSA (4.6 µs at two modifications), and whole builds take the same time |
| R6 | `predict-library --from-library` rebuilt a flat library, losing decoy flags and provenance | `predict_library(library=...)` and the CLI put the predictions onto the source library (`library.attach_predictions`, `Library.subset_precursors`): decoys, spans and modification rows survive; a precursor the model cannot take is left out with what only it needed (a decoy with its target), and counted |
| R7 | `Acetyl@Protein N-term` and `Dimethyl@N-term`, both fixed, overwrote each other in rule order | Refused where both are licensed for a residue; refused in `enumerate_modforms` too if two fixed modifications meet on a slot |
| R8 | Reversed-protein decoys could equal targets under I/L | Dropped and counted, as peptide-level collisions are |
| R9 | Shuffle collisions were resolved in FASTA order | Settled in sorted sequence order |
| R10 | An unmatched `--protein-terminus` was ignored | Refused |
| R11 | `D,2H` applied deuterium twice; `33S,34S` both applied | Labels normalised (`D` → `2H`), deduplicated, one isotope per element, checked against the element's isotopes |
| R12 | A form two paths make by different rules kept only the first path's origin | A modification row per (site, modification, origin, rule) |
| G1 | A labelled molecule's envelope gave its labelled atoms natural variance | The envelope varies over the unlabelled atoms only and is moved to the labelled mass, in binned and exact mode (exact mode used to refuse labels) |

Also from the review: the comment calling UNIMOD:259 the ¹³C₆-only label is corrected (that is
UNIMOD:188; 259 is ¹³C₆¹⁵N₂ lysine). The review's remaining-coverage list (partial incorporation,
modification-dependent cleavage, linked species, non-canonical residues, adducts, empirical
small-panel controls) is carried into "Still to do". Found while fixing it, not fixed: the
canonical peptide key keeps `<D>` and `<2H>` apart, so one molecule spelled both ways gets two
ids (the search space now writes `2H`).

## Measurements (2026-11-05, this machine)

- **Per form, after this work:**
  - human RefSeq (136,807 proteins, trypsin, 2 missed cleavages, about 17M spans): enumeration
    25 µs per peptide, keys about 4 µs per form, mass about 10 µs per form;
  - a wide purified enumeration (albumin, unspecific 7–30 aa, 7 variable modification types, at
    most 2 per peptide): 309k forms in seconds.
- **Writing the corpus FASTA through the shared writer:** byte-identical, at 0.56 s against
  0.59 s for 200k × 1.5 kb.

**End to end on the purified run.** The run is BSA plus ¹³C/¹⁵N lysin, ion-trap DIA, 100 × 8 m/z
windows, 60 min (`/mnt/e/Kirby/261031_EMT-Lysin-XL-test-samples/261101_sample1_…_CID24.raw`).

*Conversion and search space*
- **Converting:** `massspec convert --no-profile --threads 6` took 3 min 44 s.
- **The search space:** `--fasta bsa.fasta --fasta lysin.fasta --label lysin.fasta=13C,15N
  --variable Oxidation@M --variable Carbamidomethyl@C --max-variable-mods 2 --missed-cleavages 2
  --min-mz 350 --max-mz 1250`. It holds 254 spans, 740 peptidoforms and 1,756 precursors, plus
  1,756 `reverse-interior` decoys, with none falling back and none dropped.
- **Against the bench's set:** it has every one of the bench's 1,745 targets, plus 11 labelled
  lysin precursors the bench had cut by applying its m/z window to their light mass.

*The gated MS2 sweep* (0.35 Da, the gate's defaults, 4 workers): 3 min 0 s, at most 3.0 GB a
process.

| | targets passing | decoys passing |
|---|---|---|
| all | 230 / 1,756 (13.1%) | 34 / 1,756 (1.9%) |
| light | 198 / 1,277 | 26 / 1,277 |
| ¹³C¹⁵N | 32 / 479 | 8 / 479 |
| unmodified | 90 / 588 | 10 / 588 (1.7%) |
| modified | 140 / 1,168 | 24 / 1,168 (2.1%) |

- **Targets:** the 230 are sweep.md's, with the same long runs (RHPYFYAPELLYYANK 67 scans,
  KVPQVSTPTLVEVSR 66, LGEYGFQNALIVR 40) and the same 32 labelled lysin peptides.
- **Decoys:** sweep.md's 11 of 442 (2.5%) came from unmodified light decoys only, the bench's
  pseudo-reverse with the C-terminus fixed. Every target now has its EncyclopeDIA-style decoy, and
  modified and labelled decoys pass at about the rate unmodified ones do.

*After the review's fixes* (same command, 2 min 48 s, at most 3.0 GB a process): the search space
has the same counts, and 231 targets and 34 decoys pass. Seven verdicts changed, all on labelled
lysin peptides with a carbamidomethylated cysteine, which are now 3.0037 Da lighter per cysteine
(R1): one target (`<13C><15N>KLLVLC[UNIMOD:4]IFAMMATLAM[UNIMOD:35]SR/3`, from lysin's signal
peptide) newly passes, and three decoys pass in place of three others. Nothing else moved.

*Both levels with `--own-charge`* (MS1 at 0.5 Da): 5 min 33 s.
- It gives the same verdicts, since the gate reads MS2 only.
- MS1 adds 10.8M trace rows, one charge per precursor. With the default `charge_range` (1, 4),
  each precursor would carry its peptide's four charges.

## Still to do

- **Browser typed peptide (next phase):**
  - a run-scoped `/resolve` that builds the precursor through the search space (content id, all
    charges), with channels from the sweep compiler behind a declared function;
  - `locate_precursor` for the window, and an apex from a quick index XIC;
  - a charge picker, DDA fragments over the isolating scans, and identifications matched by
    `precursor_id`;
  - a 400 instead of a 500 for unparseable mass deltas;
  - then a protein box: accession or sequence → digest → pick a peptide → trace.
- **Scoring:**
  - predicted intensities (`LIBRARY_FRAGMENT_TABLE`) → channel `abundance`;
  - a channel-overlap graph over the compiled plan;
  - a scorer as a gated reducer.
- **UNIMOD neutral losses → `LOSS_REGISTRY`**, and which site licensed a loss.
- **Sequence variants and isoforms** as protein rows with a parent link; PEFF, and UniProt headers
  carried into the protein table.
- **Open searches:** unlocalised mass-offset targets (`[+Δ]?SEQ`, MS1 only); precursor-level losses
  and adducts.
- **Linked species:** intrachain disulfides (pairs of UNIMOD:374 within one peptide), cross-links
  and linked chains, each with its fragmentation. Refused until then.
- **Partial isotope incorporation** (a distribution of labelled atoms; ProForma 2.0 §5.6 does not
  represent it either), and SILAC's heavy atoms taken out of an envelope's variance (UNIMOD stores
  no per-atom isotopes for them).
- **Modification-dependent cleavage** (an acetylated lysine trypsin does not cut): digestion
  precedes modification, and missed cleavages count unmodified sites.
- **Proteome scale:** partitioned search-space output, a change to `Library`'s storage contract,
  wanted together with the sweep's P7 (query compilation without the per-target loop, which could
  read these tables directly).
- **`--reference HANDLE`** for the reference cache's `protein.faa`, resolved outside `massspec`.
