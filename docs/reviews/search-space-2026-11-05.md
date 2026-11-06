**Review of `feat/search-space`, 2026-11-05**

Reviewed `aec8bc722db32ab73882c514355b9940561be2e8` against local `main`
(`e9bc1da3af5ee640999f71918bb6e17938e2bd92`). This covers the five branch
commits, including shared chemistry, digestion, FASTA, library, decoy and
consumer changes. Existing uncommitted work was left untouched. No production
code was changed by the review.

**Recommendation: request changes before relying on this for comprehensive
purified-protein identification.** The digestion changes have strong independent
support. The modification/label model can both omit real precursors and emit
incorrect masses, and the advertised prediction handoff loses decoy identity.
Passing comparisons with the previous implementation do not establish chemical
correctness: several limitations below are inherited and newly exposed by the
search-space API.

The accompanying [regression suite](../../tests/test_search_space_review.py)
contains 125 cases: **104 pass and 21 are strict expected failures**. They cover
12 findings and one inherited isotope-envelope limitation. Strict xfails are
temporary review markers, not fixes: `--runxfail` exposes the defects as ordinary
failures, and a fix that starts passing requires removing its marker. Unexpected
exception types are not swallowed by the markers.

| Finding | Priority | Main location |
|---|---|---|
| R1: metabolic labels also label atoms added by reagents | P1 | `search_space.py:514–516` |
| R2: fixed SILAC suppresses compatible lysine PTMs | P1 | `protein.py:641–646` |
| R3: pyro-carbamidomethyl double-counts fixed carbamidomethyl | P1 | `protein.py:634–649`, `search_space.py:206–212` |
| R4: bridge chemistry is accepted without paired attachment sites | P2 | `search_space.py:176–193`, `protein.py:799–808` |
| R5: budgets and per-mod limits do not bound combinatorial work | P1 | `protein.py:669–672`, `789–795` |
| R6: prediction loses decoys and provenance | P1 | `digest.py:212–239`, `koina/assemble.py:142–148` |
| R7: overlapping terminal fixed rules silently overwrite each other | P2 | `search_space.py:298–301`, `protein.py:636–639` |
| R8: reversed-protein decoys can be I/L-equivalent to targets | P2 | `search_space.py:542–551` |
| R9: shuffle collision resolution depends on FASTA order | P2 | `decoys.py:215–225` |
| R10: unmatched mature-chain declarations are silently ignored | P2 | `search_space.py:364–370` |
| R11: isotope aliases and contradictory labels are not validated | P2 | `search_space.py:246–250` |
| R12: merging contexts loses modification provenance | P2 | `search_space.py:518–526` |

Paths in the table are relative to `constellation/massspec/library/`, except
`protein.py`, which is `constellation/core/sequence/protein.py`.

**R1 — Keep protein-derived isotope labels separate from added reagent atoms.**
With protein `ACDEK`, `--label P=13C,15N` and fixed `Carbamidomethyl@C`, the
builder emits `<13C><15N>AC[UNIMOD:4]DEK`. The mass implementation applies the
global label to the final composition, including CAM's two carbons and one
nitrogen. Ordinary light iodoacetamide therefore contributes **3.003745 Da too
much** per cysteine: expected 2+ m/z **325.155020**, observed **326.656892**.
This affects the exact labelled/alkylated-protein use case documented in the
branch and can remove a real precursor at the m/z filter, shift isolation-window
selection, and shift modified fragments. Applying labels to a fully specified
ProForma molecule already behaved this way on main; the new bug is treating a
protein's metabolic label as that molecule-wide instruction. Preserve isotope
composition by atom origin, or explicitly distinguish metabolic labels from
whole-molecule replacement. The distinction between metabolic labels and
preparation artefacts is also documented in the
[Mascot isotope configuration](https://www.matrixscience.com/help/quant_config_help.html).
Tests exercise 13C, 15N, and both together, with independent atom counts.

**R2 — Permit chemically compatible labels and PTMs on the same residue.**
`--fixed UNIMOD:259@K --variable Acetyl@K` on `PEPTIDEK` produces only the
SILAC form. The fixed-slot exclusion silently removes the acetylated heavy
lysine form regardless of the modification budget. This is an isotopic
substitution and an independent chemical modification, not two competing
alkylations. Moving SILAC to variable makes them alternatives and still omits
their combination. The newly admitted hidden SILAC specificity makes this
reachable with the normal validated API; main rejected that rule. Model isotope
occupancy separately from reactive-group occupancy, or supply an explicit
combined-modification mechanism. The expected label is **+8.014199 Da** for
[UNIMOD:259](https://www.unimod.org/modifications_view.php?editid1=259), with
another +42.010565 Da for acetylation. Some existing source comments incorrectly
describe 259 as the 13C6-only label; that is UNIMOD:188. The general stacking
requirement is described by
[Matrix Science](https://www.matrixscience.com/blog/modifications-round-up-part-2.html).

**R3 — Resolve or reject overlapping terminal/residue chemistry.**
Fixed `Carbamidomethyl@C` plus variable `UNIMOD:26` on `CPEPTIDEK` produces
`[UNIMOD:26]-C[UNIMOD:4]PEPTIDEK`: +97.016378 Da relative to the unmodified
peptide. Pyro-carbamidomethyl's +39.994915 Da is already relative to **unmodified
cysteine**, so this double-counts CAM by +57.021464 Da. The N-terminal slot and
residue-zero slot are treated independently even though this reaction consumes
both functional groups. Either substitute the correct final chemistry or reject
this incompatible rule combination and explain the appropriate ammonia-loss
representation. UNIMOD explicitly explains the reference state in
[record 26](https://www.unimod.org/modifications_view.php?editid1=26). The slot
limitation exists on main too, but the new generator currently offers no guard
against it. The regression accepts either correct enumeration or an explicit
rejection.

**R4 — Reject unsupported bridge modifications instead of generating ordinary
linear peptides.** `UNIMOD:374@C` on single-cysteine `ACDEK` emits a standalone
half-disulfide (-1.007825 Da). `UNIMOD:1898@K` on `GKKG` places an intact DSS
bridge independently on either or both lysines, without shared anchors; the
double placement adds two bridges rather than one link between the residues.
The ordinary linear fragment machinery cannot interpret these as the intended
molecules. UNIMOD placement licensing alone does not establish a complete
chemical structure. Until linked-peptide enumeration and fragmentation exist,
reject known unsupported intact bridges/half-bridge rules; retain supported
quenched monolinks. Do not rely solely on the classification field: the bundled
intact DSS entry is a `Chemical derivative`, and the half-disulfide is `Multiple`.
See [UNIMOD:374](https://www.unimod.org/modifications_view.php?editid1=374) and
[UNIMOD:1898](https://www.unimod.org/modifications_view.php?editid1=1898).

**R5 — Make preflight and enumeration proportional to useful work.** Without
per-modification limits, `_count_planned` visits every slot combination instead
of using dynamic programming. Thirty methionines with oxidation and
`max_variable=30` require visiting **2^30** combinations just to calculate one
integer. A peptidoform budget of 1 does not prevent this. Adding a per-mod limit
of one oxidation makes the DP count cheap, but the actual enumerator still
visits all combinations and rejects them afterward, to return only 31 forms.
All three reproductions exceeded a two-second in-process deadline after imports.
Use a polynomial count when limits are absent, budget-aware saturation where
exact totals are unnecessary, and prune enumeration before choosing disallowed
modifications. This matters when widening the search for multiply modified
purified proteins, even when the final requested result is small.

**R6 — Preserve the new graph through `predict-library --from-library`.** The
CLI's Koina path converts a library to `PrecursorSpec` objects, whose fields
cannot carry `is_decoy`, `decoy_of`, `decoy_set`, spans, or modification origins.
Assembly then constructs a fresh library. A two-peptidoform library containing
`PEPTIDEK` and its default decoy `PEDITPEK` comes back with **both decoy flags
null**, and its provenance tables empty. The loss is reproduced offline with
synthetic model responses; no remote prediction is involved. This is a new
integration gap between the expanded library schema and an existing consumer
explicitly advertised by the new command. Populate predictions onto the source
library, with explicit handling for any unsupported or dropped precursors,
rather than rebuilding identities from the lossy projection. At minimum,
refuse an operation that cannot preserve target/decoy identity.

**R7 — Check fixed-rule conflicts after mapping to physical slots.** Fixed
`Acetyl@Protein N-term` and fixed `Dimethyl@N-term` pass settings validation
because their site strings differ. Both map to the same N-terminal slot in a
protein-terminal peptide. Last assignment wins: reversing rule order changes
the sole product from dimethylated to acetylated without any error. Validate
overlap in the context where both apply; the same principle applies to
C-terminal aliases. Tests cover both input orders.

**R8 — Apply the target-collision policy to protein-level decoys too.** For
`AILA` with `No_enzyme` and `reverse-protein`, `ALIA` is emitted as a decoy.
Both normalize to `ALLA`, so their ordinary b/y mass ladders and precursor
masses are indistinguishable. Peptide-level decoys use `il_key` to reject this,
but reversed-protein decoys are classified using exact peptidoform identity
only. Remove or otherwise resolve these target/decoy collisions before emitting
the decoy flag. This is particularly consequential in small protein panels,
where one indistinguishable decoy can dominate the decoy evidence.

**R9 — Resolve shuffle collisions in a stable sequence order.** With
`shuffle-interior`, seed 0, `No_enzyme`, targets `PADCEK` and `PAECDK` both
initially propose `PCEDAK`. In that input order the resulting pairings are
`PADCEK→PCEDAK`, `PAECDK→PEDCAK`; reverse FASTA order and they become
`PAECDK→PCEDAK`, `PADCEK→PEDACK`. Per-sequence RNG seeding is insufficient
because the shared `taken` set is updated in input order. The existing order
test has no competing proposals. Resolve unique sequences in a deterministic
order before restoring the requested output order.

**R10 — Validate every requested mature-chain accession.** A declaration such
as `--protein-terminus TYPO=3` is silently ignored if the FASTA has only `P`.
The build succeeds with no mature-chain peptide and no correctly licensed
protein-terminal modification at that position. Unlike unmatched labels, no
global check detects the unused declaration. Check all terminus targets before
digestion and report unmatched identifiers, especially because FASTA ids and
bare accessions are both accepted.

**R11 — Normalize isotope aliases and enforce one substitution per element.**
`P=D,2H` passes validation and applies the deuterium shift twice; deduplication
is by spelling only. `P=33S,34S` on `MPEPTIDEK` likewise adds both sulfur
substitution shifts to the same atom. These are not valid whole-element label
states. Normalize aliases to `(element, mass number)`, deduplicate equivalent
labels, reject conflicting mass numbers, and validate against the available
isotope table before generating forms. This also prevents meaningless labels
from surviving merely because a peptide happens to contain zero of that element.

**R12 — Retain modification origins on the paths that produce a molecule.**
Use proteins A=`GPEPTIDEK` and B=`AAAKGPEPTIDEK`, zero missed cleavages, fixed
`Acetyl@Protein N-term`, and variable `Acetyl@N-term`. The shared
`[UNIMOD:1]-GPEPTIDEK` correctly has two span edges, but its modification table
records only the first context's origin. A-first says `fixed`; B-first says
`variable`. The other origin is discarded because modification rows are emitted
only when the peptide key is first inserted. A scorer cannot recover the
correct per-path rule from this graph. Preserve path-specific provenance in an
edge/placement relation while still deduplicating the molecule. The tests
require both origins to survive and cover both FASTA orders.

**Coverage and remaining readiness gaps**

The new passing tests independently verify all substrings for 54 combinations
of protease, specificity, missed-cleavage limit and Met handling on 64 sequences
each (**3,456 scenarios**), with declared N- and C-termini, length limits,
flanks and cleavage counts. These expectations do not call `cleave_sites`, use
its regex, or compare with the frozen implementation. They also check 36
modification/limit combinations against a Cartesian oracle, tabulated mass
deltas against Pyteomics residue masses, exact C/N label counts, mature
C-terminal amidation, separate heavy/light m/z filtering, and complex default
decoy/Parquet round trips.

The following limitations are distinct from the branch findings above. They
prevent a claim that all relevant species in a sensitive purified preparation
are covered:

| Area | Current behavior and additional validation needed |
|---|---|
| G1: isotope envelopes | Existing `peptide.envelope.neutral_envelope` shifts a natural-abundance envelope without removing labelled C/N from its variance; exact mode explicitly refuses global labels. The new strict-xfail checks a fully 13C/15N-labelled molecule against the natural distribution of only its H/O/S atoms. SILAC likewise needs fewer stochastic C/N atoms, not only a mass shift. This implementation predates the branch. |
| Partial incorporation and isotope selection | `--label` represents whole-element states, with no enrichment/purity distribution. Sweep isotope offsets are separate from search-space precursors and do not model positional partial incorporation or its fragment distribution. Add known enrichment mixtures, M−1/M−2 heavy satellites, M+1/M+2 selection, isotope-window edge cases, and exact 15N versus 13C spacing. |
| Modification-dependent cleavage | Digestion occurs before modifications. The missed-cleavage count is explicitly the count of unmodified-sequence enzyme sites; it is not conditioned on acetylated/glycated lysine or other cleavage-blocking chemistry. Add known blocked-site substrates and a policy for modification timing. Simply raising the missed-cleavage limit includes more spans but does not make their provenance chemistry-aware. |
| Modification compatibility and context | UNIMOD site licensing is not a reaction model. Additional tests should cover loss of the N-terminal amine after cyclization, cumulative oxidation states, glycosylation sequons, chemical-label exclusion, and dependent modifications. Known combined UNIMOD entries can help some cases but do not solve general compatibility. |
| Linked species | Interpeptide disulfides, intact crosslinks, linked mature chains and their fragment topologies have no generator here. Explicit rejection is the safe supported behavior until their molecular identity and fragmentation are implemented. |
| Noncanonical residues | U/O/B/Z/X/J-containing spans are intentionally skipped and counted. Selenoproteins, pyrrolysine, ambiguity resolution and substituted peptides are therefore not exhaustively represented. The existing test asserts this omission; it does not establish mass support. |
| Other precursor species | Charges are protonated positive integers; adduct composition, in-source products, isotope-purity mixtures and open/unknown-mass modifications are not independent generator axes. Add explicit supported/unsupported contracts before treating an empty result as absence. |
| Small-panel identification | Test empirical purified-protein controls, contaminants, overlapping isotope/charge species, positional isomers, shared peptides, absent proteins and low-intensity artefacts. Generating a larger candidate list is not itself validation of identification or error-rate calibration. |

For reference, deamidation (+0.984016 Da) differs from one 13C substitution
(+1.003355 Da) by only about **0.019339 Da neutral**, or **0.009670 Th at 2+**.
Independent tests should preserve these as different hypotheses at suitable
resolution; broad enumeration must not merge them by a loose mass tolerance.

**Validation record**

- Across the existing targeted suites, **849 tests passed**; four additional
  tests remain blocked by the missing `lxml` dependency, with identical failures
  reproduced on `main`. Reruns are not counted as additional tests.
- Existing targeted tests: 216/216 passed for search-space/core chemistry,
  digestion, provenance, FASTA, identities and sweep targets.
- Wider chemistry/library/reader run: 300 tests. Two initial EncyclopeDIA
  integration failures were sandbox restrictions (read-only cache and local
  helper sockets). All six tests in that integration file passed after the
  approved rerun outside the sandbox.
- Consumer/ProForma/sweep run: 270 passed, four failed because the environment
  lacks `lxml`. The same four fail on an isolated copy of `main`; they are not
  attributed to this branch. No dependencies were installed or upgraded.
- FASTA consumer regressions: 63/63 passed for transcriptome-to-proteome and
  reference annotation, genome and transcript readers.
- New review suite: 104 passed, 21 strict xfails; rerunning with `--runxfail`
  confirms the failures rather than suppressing test execution.
- Ruff on the new tests and `git diff --check` passed. The full repository
  suite and new empirical instrument acquisitions were not run.

Reproduce the new coverage with:

```bash
/home/dbwilburn/miniconda3/envs/constellation/bin/python -m pytest \
  tests/test_search_space_review.py -q -rx --tb=short
```

Add `--runxfail` to treat the known defects as normal failures. The three
combinatorial regression tests use a two-second POSIX timer and skip on systems
without `setitimer`; all other new tests are offline and deterministic.

**Resolution (2026-11-05)**

All twelve findings and G1 were fixed on `feat/search-space`, and the 21 strict-xfail markers were
removed from `tests/test_search_space_review.py`, whose tests now hold the fixes. R1 was settled
by the PI choosing among three readings: a global label covers the peptide chain, a
modification's added atoms are natural, and its removed atoms come off the labelled chain. The
R6 test now goes through `predict_library(library=...)`, the hand-off the fix provides. What was
done for each finding, and what was found on the way, is in
[docs/plans/search-space.md](../plans/search-space.md), "Review by Codex".
