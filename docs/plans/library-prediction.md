# Library prediction with interchangeable predictors

Status: approved 2026-11-05 (PI); being built on `feat/library-prediction`, one PR per phase.

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

**Charge selection** (`JChronologerLibraryPredictionClient`; to be confirmed against the golden
data before it is coded):
- A charge is eligible when it lies in the charge range and its m/z falls in the m/z window.
- Among eligible charges, those above 1% are kept, and so is any charge whose probability is at least
  the second-highest eligible probability (the highest, when two tie for it). So at least the two
  most likely eligible charges survive.

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

Maps are keyed by UNIMOD accession. A jar mass prefix that would admit a second modification is
refused rather than guessed. They are generated, with every jar member's SHA-256, into
`constellation/data/jchronologer_models.json` by `scripts/build-jchronologer-models-json.py`: the
captured-not-transcribed rule the jar CLI follows. A model member whose hash is not in the snapshot
is refused.

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
