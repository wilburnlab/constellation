# Library prediction with interchangeable predictors

Status: approved 2026-11-05 (PI); P0–P4 built on `feat/library-prediction` (2026-11-06), one commit per phase in one PR (PI).

## Why

The native search needs fragment intensities, retention times and (for later scoring) charge-state
and CCS predictions on a search space's precursors, targets and decoys alike, before the sweep can
score anything. Koina is the only in-Python predictor today: it needs the network, and Prosit 2020
refuses terminal modifications. EncyclopeDIA 6.9.23 ships its own predictor, JChronologer, as five
TorchScript models inside `lib/jchronologer-1.0.10.jar`. They load in plain PyTorch. Checked
2026-11-05 against the jar's own output:

- all 126 entries of a local ScribeTwo cache reproduce exactly: MS2 to 1e-5, RT float32-identical;
- 3,000 entries sampled from a 2.5M-entry human library built on another machine reproduce to
  float noise: RT within 0.5 ms; MS2 exact for all but 18, 17 of which are the checker's own matching
  of coincident b/y ions.

Running them in-process, read from the installed jar, gives a local, deterministic predictor whose
output is EncyclopeDIA's own. A comparison against EncyclopeDIA then isolates the scorer. The
contract they run behind is Koina's decomposition done locally: prepare inputs → run the model →
decode into a common form. Koina itself becomes one predictor, with its own input formatting and
output parsing around a network call, and an in-house model joins the same way later.

PI decisions (2026-11-05):
- the models are read from the installed jar, never vendored;
- HCD only for now: the exported Cartographer has a `cid_conv` head whose weights are untouched
  default initialisation (max |w| = the init bound, kurtosis 1.80). An in-house model will follow;
- collision energy is a property of each precursor charge. It can be scaled from a reference charge
  (EncyclopeDIA's rule, a first approximation) or given explicitly per charge, as tuned per
  instrument;
- a precursor that any requested predictor cannot take is dropped, together with its target/decoy
  partner at the same charge, and counted by reason;
- charge-state probabilities and CCS are predicted by default.

## Words

A *predictor* produces one *property*: `fragments`, `retention_time`, `ccs` or `charge_state`.
"Engine" is taken: the `Search` tables are keyed by it. The CLI keeps `--backend`.

Model ids are `jchronologer:{chronologer,cartographer,electrician,sculptor}` and `koina:<model>`.
Documentation says "JChronologer's Cartographer model", because the lab's own package is also
called Cartographer.

## The JChronologer models, as read from the jar

| Model | Property | Inputs | Output |
|---|---|---|---|
| `Chronologer_20220601193755` | RT | tokens int64[52] | × 60 = seconds on Chronologer's scale (not a run's) |
| `Cartographer_multistart_best_run01` | fragments | tokens[33], charge one-hot float32[6], NCE/100 float32[1] | float32[174], clamped at 0 and max-normalised |
| `Electrician_20260225110528` | charge state | tokens[33] | float32[6] → finite positives normalised to sum 1 |
| `Sculptor_20260311095327` | CCS | tokens[52], charge one-hot[6] | `ccs_norm·train_ccs_std + train_ccs_mean` (Å²; the `.dlib`'s `IonMobility`) |
| `Scout_20260427110352_run01` | all four (the jar's fast mode) | tokens[33], charge, NCE | ms2[116], irt, ccs, charge_dist; later |

**Which jar.** The EncyclopeDIA jar's manifest Class-Path names `lib/jchronologer-1.0.10.jar`. The
1.0.9 jar beside it is a leftover, and its Sculptor JSON differs.

**Tokens.**
- Layout: N-terminal token, then residue tokens, then `_`, zero-padded to `max_peptide_len + 2`.
- Token *characters* are shared across the family (`c` Cys CAM, `m` Met ox, `s t y` phospho, `^`
  N-terminal acetyl, `-` free N-terminus, `(` `)` pyro-Glu Q / E). Each model maps characters to
  its own ints and supports its own subset; Chronologer has no oxidised Trp, deamidation or HexNAc.
- Chronologer, Cartographer and Electrician key modifications by regular expressions over
  EncyclopeDIA's mass spelling. Scout and Sculptor declare UNIMOD maps.
- Pyro-Glu is written two ways:
  - Cartographer: N-terminal `(` / `)` and the plain residue;
  - Chronologer: N-terminal `(` / `)` *and* residue `e`. Any other spelling misses RT by minutes.
    Koina's own Chronologer preprocessor makes the same rewrite.

**Decoding fragments** (`CartographerSpectrumDecoder`, `PeptideMassCalculator.isFragmentPossible`):
- For ion number n = 1..29 and fragment charge c = 1..3: y at `6(n−1)+(c−1)`, b at `6(n−1)+3+(c−1)`.
- An ion is kept when its value is finite, ≥ 0 and ≥ `minimumReportedIntensity` (0.01), and
  L > 1, 0 < n < L, 1 ≤ c ≤ 3 and c ≤ the precursor charge.
- b/y only; no losses.

**Collision energy** (`PrositCSVWriter.convertNCE(float nce, byte z, byte zdef)`):
- In float32: `(nce · f(zdef)) / f(z)`, with f = {1: 1.0, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75} and
  0.75 outside 1–5.
- EncyclopeDIA calls it with the library's default charge (3), widens the result to double, and
  JChronologer divides by 100.0 before the model takes it as float32.

**JChronologer's source.** searlelab/jchronologer is public (Apache-2.0); its HEAD is 1.0.10 (pom
version; last commit 2026-09-21, "small fix to handle pyroglu in Sculptor correctly"), the version
EncyclopeDIA 6.9.23 loads. What follows was read from it, and the tokens were checked against the
jar's own preprocessing code (P1).

**Charge selection** (`JChronologerLibraryPredictionClient`; to be confirmed against the golden
data before it is coded):
- A charge is eligible when it lies in the charge range and its m/z falls in the m/z window.
- Among eligible charges, those above 1% are kept, and so is any charge whose probability is at least
  the second-highest eligible probability (the highest, when two tie for it). So at least the two
  most likely eligible charges survive.
- That rule is EncyclopeDIA's. JChronologer's own automatic selection
  (`DefaultChronologerLibraryPredictor.predict`) keeps every charge 1–6 with `p ≥` the request's
  minimum; its distribution is Electrician's finite positive outputs divided by their sum (a double
  sum, each quotient cast to float).
- In the jar, RT seconds are `float32(pred_hi × 60f)` and CCS is `out × (float)std + (float)mean`. A
  peptide Sculptor cannot tokenize gets no CCS; it is not refused.

**Threads.** The jar runs one inference thread per worker. In torch 2.12, the same thread count and
batch size give bit-identical output; thread count alone moves values by up to 4.5e-6, batch size
by up to 2.2e-5.

## Design

`constellation/massspec/library/prediction/`:

| Module | Holds |
|---|---|
| `base` | `Property`, `Capabilities` (only what the orchestration reads), and the `Predictor` protocol: `name`, `provides`, `capabilities`, `identity()`, `check(batch)` (a reason or None per row, no model call) and `predict(batch)`, whose outputs are plain numpy dataclasses |
| `collision_energy` | `CollisionEnergy`: a resolved charge → NCE table and how it was made (`uniform`, `scaled_from_reference`, `explicit`); `resolve(charges)` in float32; `describe()` for metadata and stage keys |
| `requests` | `build_requests(library, settings)`: label stripping (global isotopes plus isotopic-substitution modifications; TMT kept), dedup on (stripped canonical key, charge) in sorted order, fan-out to every source precursor, the pairing map on (`decoy_of`, charge) |
| `torch_runner` | fixed batches over the canonical order under `inference_mode`, one intra-op thread per worker, parallel across batches; reused by the in-house model |
| `assemble` | vectorised: ion number → bond (`bond_position`); m/z from the *labelled* source ladder; a backend's own m/z checked against the stripped peptide that was sent; fragments under the source `precursor_id` with `annotation` null; RecordBatches for streaming |
| `run` | `predict_onto(library, {property: predictor}, settings) -> (Library, PredictionReport)`: every `check` first; a rejection drops the precursor and its partner; one Library build |
| `registry` | `backend:model` → predictor, imported lazily |

The tokenizer is `constellation/massspec/tokenize/`:
- `vocabulary.Vocabulary`: characters ↔ ids, `(residue, accession) → char`, N-terminal map, vectorised `encode`;
- `jchronologer`: one vocabulary per model, plus the terminal rules.

Input is keyed by UNIMOD accession: a modification outside JChronologer's fifteen (or, for Sculptor,
outside its file's map) is refused, never matched by mass, so Ethyl-K is not taken for dimethyl-K.
The regular-expression models apply the file's rules to the jar's mass spelling, cached per (residue,
modification set), which reproduces the jar exactly, including sums at one position; a derived table
would only have been a second copy of the rules, so the snapshot holds hashes, not maps.
`constellation/data/jchronologer_models.json` (`scripts/build-jchronologer-models-json.py`) records
the SHA-256 of the jar and every model member by version and role, and a preprocessing file whose
hash is not in it is refused.

`constellation/massspec/library/jchronologer/`:
- `bundle`: locate through `thirdparty.registry.find("encyclopedia")` and the manifest Class-Path;
  read members with `zipfile`; `torch.jit.load` from bytes, cached per process;
- `models`: the four adapters.

Library: `PRECURSOR_TABLE.charge_probability` (nullable float64; null = not predicted).
`attach_predictions` carries `ccs_predicted` and `charge_probability`.

## Phases

- **P0.** Fixes that stand on their own:
  - collision energy per charge, with Koina moved onto it (`--default-charge` now reaches Koina; new
    `--collision-energy-by-charge`);
  - the `.dlib` writer stops writing decoys as targets;
  - `attach_predictions` carries `ccs_predicted`;
  - the fragment ladder costs under 30 µs a peptide (about 256 µs today).
- **P1.** The tokenizer, tested in CI against the real preprocessing JSONs copied to `tests/data`
  (Apache-2.0, credited in `NOTICE.md`).
- **P2.** The four JChronologer predictors through the contract, by Python API:
  - unit tests on synthetic outputs;
  - a golden test against a small `.dlib` made once by the jar from a crafted FASTA with modification
    switches on (read raw through `_mirror.iter_table`);
  - a bench script for the large human library.
- **P3.** `predict-library --backend jchronologer` (`--from-library` or `--peptides`; `--fasta` points
  to `search-space`), resume through `core.stages`, an envelope manifest, streamed fragments, and a
  `doctor` row.
- **P4.** Koina as a predictor, and models mixed per property (`--ms2-model koina:…`, `none` to skip).

Later:
- calibrating collision energy per charge from identifications (survey JChronologer's
  `CalibrateNceMain` first);
- Scout;
- the in-house model (settle `constellation.models` in the import DAG first);
- decoys in a `.dlib`'s `decoyentries`;
- `IonMobility` from `ccs_predicted`;
- CID.

## Progress

### P0 (2026-11-05)

- **Collision energy per charge.** `library.prediction.collision_energy.CollisionEnergy` has three
  constructors (`uniform`, `scaled_from_reference`, `explicit`), with `resolve` (float32),
  `by_charge`, `describe` / `from_description` and `parse_energy_table`.
  - `scaled_from_reference` equals the jar's `convertNCE` bit for bit. Checked on NCE 33 / 30 / 27.5,
    reference 3 and 2, charges 1–7, against values the jar printed itself (`Float.toHexString`), now
    in `tests/test_collision_energy.py`.
  - Koina moved onto it:
    - `predict_fragments` / `predict_library` take a `CollisionEnergy` or a number, plus
      `nce_reference_charge`;
    - the library records `x.massspec.prediction.collision_energy` (table and origin over the
      charges predicted);
    - `koina/nce.py:adjust_nce` is a shim over the corrected rule.
  - **Behaviour change.** Koina libraries made with the DIA adjustment now use EncyclopeDIA's
    energies: at NCE 30, a 3+ precursor goes from 25.5 to 30 and a 2+ from 27.0 to 28.33. Libraries
    made before have `x.koina.adjust_nce_for_dia` but no `x.massspec.prediction.collision_energy`.
  - CLI:
    - `--default-charge` (default 3) now reaches `--backend koina`, where it was silently ignored.
    - New `--collision-energy-by-charge Z=NCE,...`. It refuses `--collision-energy`,
      `--default-charge` and `--no-adjust-nce-for-dia`, a table missing a charge being predicted, a
      model with no energy input, and `--backend encyclopedia`.
    - The Koina manifest records the reference charge and the table.
- **`.dlib` writer.** A library's decoy peptides (`is_decoy`) are left out of `entries` and
  `peptidetoprotein`, with a `UserWarning` giving the counts. Before, they were written as targets
  (`isDecoy` false), so the jar searched them and made decoys of them. Quantities of the dropped
  precursors are skipped; `peptidescores` keeps the search's own decoys.
- **`attach_predictions` carries `ccs_predicted`.** It goes onto the source library like
  `rt_predicted`; an unset value (−1) keeps the source's.
- **Fragment ladder.**
  - Residue masses are cached per labelling, and modification masses per batch call.
  - Bit-identical to the previous code on 10,000 random peptidoforms (3.64M ions, every batch entry
    point; isotope labels, terminal, fixed, SILAC, mass-delta modifications, losses).
  - Speed: 275 → 36–40 µs a tryptic peptide (about 1.5 min for 2.5M peptides instead of 11), and
    590 → 30–38 µs on a modification-heavy mix. That misses the 30 µs aimed for. What remains is
    per-peptide Python preparation and the numpy scatter of each group's ions, which scales with the
    ions.

### P1 (2026-11-06)

- **`massspec.tokenize`.**
  - `vocabulary`: `Sites` (residues plus the sorted UNIMOD accessions at the N-terminus, each residue
    and the C-terminus), `SiteTokens` (the accession-keyed spelling) and `Vocabulary` (characters to
    ids; `encode` pads spellings into int64 rows with a lookup table, in a few numpy operations).
    Refusals carry one of four stable reasons (`length_out_of_range`, `unknown_residue`,
    `unsupported_modification`, `unsupported_form`) and a detail.
  - `jchronologer`: a port of the jar's three tokenization paths. Chronologer gets
    `unimodToMassEncoded`, then `ChronologerPreprocessor`; Cartographer gets the mass spelling
    `predict` rebuilds after `foldFirstResiduePyrogluToNterm`; Sculptor gets `SculptorTokenizer`.
    Electrician reads Cartographer's tokens, as in the jar, and its file must agree with
    Cartographer's.
  - Constellation's `[UNIMOD:26]-C` (S-carbamoylmethylcysteine cyclization) is not one of the jar's
    fifteen. It is given to the models the way the jar writes that molecule: UNIMOD:28 on the
    N-terminus of a carbamidomethyl C (Chronologer `)d`, Cartographer and Sculptor `(c`).
- **Checked against the jar.** `scripts/survey-jchronologer-tokens.py` compiles
  `scripts/jchronologer/TokenOracle.java` and runs it on EncyclopeDIA's own classpath. The corpus is
  274 peptides: every (residue, accession) Scout or Sculptor names, the N-terminal forms, length
  limits, refusals and 200 random peptidoforms. The jar's tokens are in
  `tests/data/jchronologer/1.0.10/tokens.tsv`, beside the four preprocessing files.
  - Constellation's tokens equal the jar's on 804 of 822 rows.
  - The other 18 fall into four deliberate departures, each pinned by the test:
    1. Sculptor takes accessions in its own map that the jar cannot name, so never sends it (13 rows).
    2. Pyro-Glu written on the first residue goes to the N-terminus for every model; the jar does this
       for Cartographer only, so its Sculptor gives such a peptide no CCS (2 rows).
    3. A side-chain acetyl lysine at residue 0 stays `a`. The jar's `patchModSequence` moves any
       residue-0 `[+42.010565]` to the N-terminus, so it tells Chronologer and Cartographer the
       peptide is N-terminally acetylated (2 rows).
    4. A C-terminal modification is refused. The jar leaves it out of Cartographer's tokens; its
       Chronologer path refuses it (1 row).
- **Which jar.** The first oracle run used a `lib/*` classpath and loaded classes from the leftover
  `jchronologer-1.0.9.jar`: Chronologer refused N-terminal pyro-Glu and Sculptor gave it no tokens.
  Anything run on the jar must use the jars EncyclopeDIA's manifest `Class-Path` names
  (`thirdparty.encyclopedia.manifest_class_path` / `jchronologer_jar`). P2 reads members from the
  named jar with `zipfile`, so it is not exposed.
- **Speed** (200,000 random tryptic peptides, carbamidomethyl C, some oxidised M, phospho and
  N-terminal acetyl):
  - canonical ProForma to `Sites` takes 5.5 µs a peptide, read without building a `Peptidoform`
    (`parse_proforma` takes 14 µs), and equals the parser's result on the whole corpus;
  - encoding takes 3.2–3.9 µs a peptide per model;
  - a million peptides through all three models take about 15 s, against the 30 s aimed for.


### P2 (2026-11-06)

- **The contract** (`library.prediction.base`). A predictor has `check(requests) -> Refusals` (no model
  run) and `predict(requests) -> FragmentArrays | ValueArray`. Requests are unique (unlabelled
  `Sites`, charge, float32 energy). Fragments are named by ion type, ion number and fragment charge;
  the orchestration adds bonds, m/z and precursor ids. Refusal reasons are the tokenizer's four plus
  `charge_out_of_range` and `collision_energy_out_of_range`.
- **Requests** (`requests.build_requests`).
  - Labels come off: global labels and isotopic-substitution modifications; TMT stays. Heavy and
    light precursors share a request.
  - Requests are sorted by the peptide's canonical string, then charge, so they do not depend on the
    library's row order (tested by shuffling it).
  - Groups are a target and every decoy of it (`decoy_of`) at one charge. A decoy with no `decoy_of`
    stands alone.
  - A peptide with no reading as sites (a modification named by mass, say) is refused for every
    predictor, under the property `input`.
- **Orchestration** (`run.predict_onto`).
  - Every check runs first. A refused precursor drops with its whole group: counted by property and
    reason, with partners counted apart (`dropped_with_partner`, which also takes decoys left
    without a target by `subset_precursors`).
  - Kept requests are predicted in chunks of 100,000.
  - Values go onto `rt_predicted`, `ccs_predicted` and the new nullable
    `PRECURSOR_TABLE.charge_probability` (also carried by `attach_predictions`); NaN keeps the
    source's value. Fragments replace the source's.
  - Metadata: `x.massspec.prediction.{predictors, collision_energy, rt_scale, report}`. The report
    carries the deuterium caveat whenever labels were removed.
- **Assembly** (`assemble`).
  - Each fragment's m/z is the source peptide's own (labelled) ladder entry, found by index (bond,
    type, charge order) without a join. A labelled precursor gets labelled m/z beside its light form's
    intensities (tested).
  - A backend's reported m/z can be checked against the peptide sent (`reported_mz_error_ppm`, for
    Koina in P4).
- **Runner** (`torch_runner`). Fixed batches, each on one intra-op thread, parallel across batches. One
  thread and four give the same bits (tested), so the batch size is a setting to record and the
  thread count is not.
- **JChronologer** (`library.jchronologer`). `bundle` locates the jar the manifest names and checks
  each member's SHA-256 against the snapshot on read. `models` holds the four predictors with the
  jar's float32 arithmetic and its decoder. CI tests them on a fake bundle (the real preprocessing
  files and stand-in models): their inputs (tokens, charge one-hot, NCE/100) and their decoding.
- **Golden test, changed from the plan.** The plan had EncyclopeDIA predict a crafted FASTA through
  `run_predict_library`. JChronologer's public API (`ChronologerLibraryPredictor.predict` on explicit
  (peptide, charges, NCE) requests, or its automatic charge selection) gives the jar's predictions for
  chosen peptides directly, with CCS and charge probabilities, and no digestion or EncyclopeDIA
  wrapping in between. `scripts/jchronologer/PredictionOracle.java` runs it (on the manifest's
  classpath) and `scripts/survey-jchronologer-predictions.py` keeps 80 requests (292 entries, 3
  refused) in `tests/data/jchronologer/1.0.10/predictions.tsv`. `tests/test_jchronologer_golden.py`
  (skipped without EncyclopeDIA) runs the real models on them:

  | Property | Agreement with the jar |
  |---|---|
  | RT | within 0.14 ms |
  | Fragments | the same 4,601 ions; intensities within 3.4e-6; m/z within 8e-7 Da |
  | CCS | within 6e-5 Å²; none exactly where the jar gives none (TMT) |
  | Charge probabilities | within 9e-8 |

  The jar refuses deamidation, HexNAc and oxidised Trp; so does Chronologer here. EncyclopeDIA's own
  wrapping (energy conversion and charge selection) is checked by the bench instead, against a library
  it made.
- **Bench** (`scripts/bench-jchronologer.py`, the 2.5M-entry human `.dlib`, 3,000 entries):
  - every entry agrees within 1e-5 of max-normalised intensity, with every stored peak matched (48
    peaks ambiguous, two ions within 10 ppm). The 18 "misses" of the first check were the checker's;
  - RT agrees within 0.6 ms;
  - throughput at 8 threads, tokenizing included: fragments 4,100 requests/s, RT 9,000, CCS 10,000,
    charge states 46,000. A 5M-precursor search space would take about 20 min for fragments, against
    EncyclopeDIA's 86 min for 2.79M predictions.
- **JChronologer's own rules, read from its source.**
  - Its automatic charge selection keeps charges 1–6 with `p ≥` the request's minimum. The two-best
    rule above is EncyclopeDIA's.
  - In the jar, a peptide Sculptor cannot tokenize (TMT) gets no CCS and stays in the library.
- **Open: TMT and CCS by default.** Sculptor has no TMT tokens. With CCS on by default and the PI's
  drop-paired rule, every TMT precursor would be dropped unless CCS is turned off. P3's CLI needs a
  choice: keep the rule and point to `--no-ccs` in the summary, or let CCS (and charge state) leave the
  value unset instead of dropping, as the jar does.
- **Risks.** torch 2.12 deprecates `torch.jit.load`. JChronologer ships TorchScript, so the warning is
  silenced at the one load site; if TorchScript loading is removed, the models will need converting.
  `predict_onto` builds the whole fragment table in memory (about 150M rows for 5M precursors); P3
  streams it.

### P3 (2026-11-06)

- **`massspec predict-library --backend jchronologer`** (`cli/_predict_jchronologer.py`).
  - Input is `--from-library` (a search space) or `--peptides`. `--fasta` is refused with the
    `search-space` command to run instead.
  - Collision energy: one per run, `--collision-energy` (default `--default-nce`) scaled from
    `--default-charge`, or `--collision-energy-by-charge`, which must cover every charge in the
    library. A sweep is refused.
  - Fragments and RT are always predicted. Charge states and CCS are predicted unless
    `--no-charge-states` / `--no-ccs`.
  - `--min-charge-probability P` runs Electrician first and refuses target precursors below P,
    counted as `below_min_charge_probability`; their decoys follow by group.
  - `--threads` (default all cores), `--batch-size` (default 2048) and `--device {cpu,cuda}`.
    `--min-intensity` now defaults per backend (0.01 here, 1e-4 for Koina).
  - Each backend refuses the others' flags: Koina's and EncyclopeDIA's here, and `--threads` and the
    rest of this backend's flags there.
  - The `--backend encyclopedia` help no longer names 6.5.15.
- **Output.** `<output-dir>/library_pqdir`: fragments are streamed chunk by chunk
  (`predict_onto(fragments_sink=)`) to a side file, which replaces the empty table the library writer
  leaves. With `--output-dlib`, a `.dlib` too. Beside it:
  - `manifest.json` (`build_manifest_envelope`): the prediction report, every model's identity and
    SHA-256, the upstream library's identity, and the runtime (threads, batch size, device, torch);
  - `_SUCCESS` carrying a `core.stages` key over the upstream library (its recorded key, else its
    files' SHA-256), the predictors' identities (batch size and minimum intensity among them), the
    energy per charge and `--min-charge-probability`. Threads and device are not keyed.
  - `--resume` reuses only the same prediction. A different one in the directory is refused without
    `--resume`, and redone with it.
- **`doctor`:** a `jchronologer` row (`ok` / `unverified` / `missing`; none without EncyclopeDIA). It
  reads and hashes the jar's members without loading a model.
- **End to end on the purified run.** The BSA + ¹³C¹⁵N lysin search space (the search-space plan's
  settings: 1,756 targets, 1,756 decoys) predicts in 6.6 s at 1.2 GB peak: 3,512 precursors and
  110,292 fragments, none dropped. `chromatogram extract --targets` on the predicted library (MS2,
  0.35 Da, the gate, 4 workers; 3 min 38 s, at most 3.0 GB) passes 231 of 1,756 targets (33 of 479
  lysin) and 34 decoys, as the search space alone did (230 / 32 / 34). The gate counts fragments and
  does not read intensities yet, so this checks the wiring, not the predictions.
- **Scale** (human UniProt, the first 1,000 and 2,000 proteins; trypsin, one missed cleavage,
  charges 2–4, fixed carbamidomethyl, variable oxidation, interior-reversal decoys; 12 threads):

  | proteins | precursors (targets + decoys) | fragments | wall | peak memory |
  |---|---|---|---|---|
  | 1,000 | 471,908 | 11.5M | 3 min 0 s | 4.9 GB |
  | 2,000 | 963,700 | 23.6M | 6 min 4 s | 5.4 GB |

  That is about 2,650 precursors a second through all four models, assembly and writing included;
  memory is about 4.3 GB fixed plus about 1.2 KB a precursor. For all 20,652 proteins at these
  settings (about 9.9M precursors) that extrapolates to about 63 min and 16 GB. Not run: the
  in-memory search space would add about 10 GB at that size (proteome-scale search-space output is
  still to do), and long full-machine runs have crashed this machine before.

### P4 (2026-11-06)

- **Koina as predictors** (`library.koina.predictor`).
  - `KoinaFragments(model)` and `KoinaRetentionTime(model)` meet the contract. Construction reads the
    model's declared inputs from the server, or from a client given, so whether it takes an energy is
    known up front.
  - `check` is local. Koina's sequence is written from the request's unlabelled sites, and the
    overlay refuses the rest with a reason before any request is sent: N-terminal modifications for
    the 2020 models, modifications outside the model's set, length, charge, a missing energy, and
    non-standard residues.
  - `predict` parses each distinct annotation once and leaves out Koina's padding. A fragment the
    peptide cannot have (an ion number at or beyond its length) is left out and counted, where the
    old assembly kept it as a partial-ID row. Koina's m/z ride as `mz_reported`; `predict_onto` checks
    them against the peptide sent (`max_reported_mz_error_ppm` in the report) and never uses them.
- **Registry.** `koina:<model>` builds any Koina model. The property comes from the overlay, or is
  given (`property=`) for a model the overlay does not know, and is checked against it.
- **`koina.api.predict_library(library=...)`** goes through `predict_onto`.
  - Labelled precursors are predicted as their light form, with labelled m/z, instead of refused.
  - `on_unsupported="error"` refuses before any request (`predict_onto(strict=True)`, which raises
    `PredictionRefused`); `"skip"` drops refused precursors with their partners.
  - The FASTA / specs path is unchanged. Retention time is now sent Koina's dialect on both paths;
    `predict_rt` had been sent canonical ProForma.
- **Koina's Chronologer against JChronologer's** (live, `tests/test_koina_live.py`). It is the same
  checkpoint: the same seconds to 1e-4 on seven peptides, N-terminal acetyl and both pyro-Glu forms
  included, once pyro-Glu is written on its residue.
  - `Q[UNIMOD:28]TALVELLK` gives 1247.45 s, as the jar does; `[UNIMOD:28]-QTALVELLK` gives 1073.44 s.
    `RtModel.pyro_glu_on_first_residue` records this, and `KoinaRetentionTime` sends pyro-Glu that way
    for `Chronologer_RT`.
  - Two forms differ in every spelling tried and are left as measured: TMT (LVNELTEFAK with TMT6plex
    on the N-terminus and K: Koina 1180.20 s, jar 1144.29 s) and S-carbamoylmethylcysteine (jar
    577.50 s; Koina 310–758 s).
- **CLI.** With `--backend jchronologer`, each property's model is a flag (`--ms2-model`,
  `--rt-model`, `--charge-model`, `--ccs-model`) taking `backend:model` or `none`, defaulting to
  JChronologer's.
  - A bare model name is refused there, `--koina-url` only applies when a Koina model is chosen, and
    `--no-ccs` / `--no-charge-states` contradict an explicit model for the same property.
  - `--ms2-model` / `--rt-model` default to nothing in the parser and resolve per backend.
    `--backend koina` still defaults to `Prosit_2020_intensity_HCD` / `Chronologer_RT` and accepts a
    `koina:` prefix.
- **The Koina backend's run directory.**
  - `manifest.json` is the shared envelope (`build_manifest_envelope`), with its `koina` block and
    `runs` unchanged.
  - `_SUCCESS` carries a `core.stages` key over the source (SHA-256, or a search space's key), the
    digestion, the models and server, the energies and the settings. Each energy of a sweep is keyed
    under it.
  - `--resume` reuses only a matching key and redoes another. A marker with no key, written before
    keys, is still reused, with a note saying so.
- **Not done here.** The FASTA path of `--backend koina` still digests through `library.digest` and
  assembles with `assemble_library`; `massspec search-space` plus `--from-library` is the path that
  keeps provenance.

## Review by Codex (2026-11-06), and what was done

The PR (#107) was reviewed independently by Codex. It found five defects, all fixed in a commit after
P4, each with a test that fails on the code before it.

| | Finding | Done |
|---|---|---|
| C1 (P1) | With `--ms2-model none`, `predict-library --backend jchronologer` replaced the library's fragments with an empty streamed file, so predicting only RT, CCS or charge states onto a predicted library deleted its spectra, and emptied any `.dlib` export | Fragments are streamed only when a fragment model runs; otherwise the library is written with the fragments it came with |
| C2 (P2) | Resuming with changed settings replaced the Parquet library, then stopped at the `.dlib` export on `FileExistsError` (the export writer does not overwrite by default), leaving the new library beside the old `.dlib`. The Koina backend did the same once its resume was keyed (P4) | A run replaces its own exports (ones its manifests record, or inside its directory) and refuses any other file at that path. The check comes before anything is predicted and before `_SUCCESS` is removed, so a clash leaves the finished run as it was |
| C3 (P2) | `--resume` of a finished prediction with a new `--output-dlib` said "already complete" without writing it, since the export path is not part of the key | Each export records the stage key of the prediction it holds (`exports` in the manifest). Resuming a finished prediction writes an export that is missing or holds an earlier prediction, and leaves a current one alone. The Koina backend keys its output paths, so a new path there is a new run, as before |
| C4 (P2) | A Koina model returning a neutral loss (`y3-H2O+1`) crashed `predict_library(library=...)` with `NotImplementedError` in the new assembler; the old path had kept such ions | Loss-aware assembly: a loss ion is a typed row with its `loss_id`, its m/z the source peptide's own ion less the loss (labels included: a ¹⁵N chain loses ¹⁵NH₃), looked up by index in a ladder of only the peptides with one (`fragment_ladder_arrays_batch(loss_rules=False)`, every loss at every bond). The loss rules are not applied to a loss a model names: they say which losses to expect where nobody has said. `reported_mz_error_ppm` checks loss ions too. The old path kept such ions as partial-ID rows (no ion identity, Koina's m/z), wrong for a labelled precursor. No model the overlay registers today returns losses |
| C5 (P2) | Predicting one property onto a predicted library (fragments, then RT alone) replaced the record of which predictor made each property, so the spectra lost their model and settings | `x.massspec.prediction.predictors` is merged property by property. A property predicted again takes this run's collision energy or time scale, or drops the old one if the new model has none; one left alone keeps its record, as its values are kept |
