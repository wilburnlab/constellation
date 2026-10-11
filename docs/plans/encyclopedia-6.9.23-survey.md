# EncyclopeDIA 6.5.15 — utility inventory

Authoritative catalogue of every jar entry point exposed by EncyclopeDIA 6.5.15, with the flag table per utility and a "what changed since 2.12.30" diff. Each Python wrapper under [constellation/massspec/search/encyclopedia/](../../constellation/massspec/search/encyclopedia/) translates a typed kwarg surface to the flag table for one utility; new utilities land as new wrapper files PR by PR.

Status: **stub** — populated by PR 1 (which surveys the jar end-to-end) and extended by each follow-on PR. Until then the high-level invocation summaries below are the contract PR 0's CLI parsers and stub wrappers anchor on.

## How to (re)survey

`scripts/install-encyclopedia.sh --installer <install4j.sh>` installs into the per-user home cache `~/.constellation/encyclopedia/<version>/`, and `constellation.thirdparty.registry.find("encyclopedia")` discovers it there (env var → home cache → `third_party/`). `$CONSTELLATION_ENCYCLOPEDIA_HOME` overrides the location — point it at any install dir containing `encyclopedia-<version>.jar`.

> The earlier dev build was placed at an **in-repo `constellation/encyclopedia/` path** with `$CONSTELLATION_ENCYCLOPEDIA_HOME` pointed at it — *not* `~/.constellation/encyclopedia/6.5.15-dev/` (a stale assumption that previously appeared here). `~/.constellation/encyclopedia/` is the canonical home the installer now establishes.

The commands below resolve the jar through the env var (or the home-cache `current` symlink) so they work regardless of where the build lives:

```bash
ENC="${CONSTELLATION_ENCYCLOPEDIA_HOME:-$HOME/.constellation/encyclopedia/current}"

# Top-level help — lists all -<program> entry points
"$ENC"/jre/bin/java -jar "$ENC"/encyclopedia-*.jar --help

# Per-program help (e.g. -walnut, -convert, -libexport, -scribetwodda)
"$ENC"/jre/bin/java -jar "$ENC"/encyclopedia-*.jar <program> -h

# Nested -convert subcommands
"$ENC"/jre/bin/java -jar "$ENC"/encyclopedia-*.jar -convert -<subprogram> -h
```

When system Java is available, the runner picks `shutil.which("java")`; the bundled JRE in the install dir is the fallback for boxes without one.

## Top-level programs (verified from `--help`)

| Program | Role | Wrapped? | Notes |
|---|---|---|---|
| (default) | DIA library search — `-i <mzml/dia> -l <library>` | PR 1 → [library_search.py](../../constellation/massspec/search/encyclopedia/library_search.py) | Headline workflow. Default Percolator is `v3-01` (bundled). |
| `-walnut` | DDA FASTA-search (PECAN-style scoring) | not yet | Wrap on lab demand. |
| `-thesaurus` | Phospho-site localization | not yet | Wrap on lab demand. |
| `-xcordia` | Cross-correlation DIA scoring | not yet | Alternative scoring engine. |
| `-scribe` | Scribe library search (DDA) | not yet | Older scoring engine. |
| `-scribetwodda` | ScribeTwo, DDA mode (fragment-indexed) | not yet | Optional FASTA prediction baked in (`-nce`, `-defaultCharge`, `-predictionCache`). |
| `-scribetwodia` | ScribeTwo, DIA mode | not yet | Same as `-scribetwodda` flag surface. |
| `-browser` | Interactive ELIB browser | never (GUI) | Not a CLI-driven workflow. |
| `-libexport` | Combine searches → quant-report library | PR 4 → [library_export.py](../../constellation/massspec/search/encyclopedia/library_export.py) | Nested sub-modes `-pecan` / `-xcordia` / `-phospho` for non-default scoring pipelines, deferred. |
| `-convert` | File-format converters and preprocessing (nested — see below) | partial (PR 2 + PR 3) | One nested verb per row in the next table. |
| `-batch` | XML-driven batch driver | not yet | Wrap if a batch use case appears. |

## `-convert` sub-utilities (verified from `-convert -h`)

| Sub-utility | Role | Wrapped? | Notes |
|---|---|---|---|
| `-fastaToJChronologerLibrary` | **FASTA → predicted .dlib via JChronologer** | PR 2 → [predict_library.py](../../constellation/massspec/search/encyclopedia/predict_library.py) | **New in 6.5.15.** JChronologer (RT) + Sculptor (CCS/IMS) + Electrician (charge prob). In-process PyTorch; replaces Prosit/Koina. PTM toggles are per-PTM `off\|var\|fix` enums. Peptide length fixed 7-31. |
| `-processDIA` | Preprocess one mzML or merge multiple → `.dia` | PR 3 → [process_dia.py](../../constellation/massspec/search/encyclopedia/process_dia.py) | Accepts `.mzML`, `.raw`, `.d`, `.DIA`. Multi-file input is colon-delimited (`-i a:b:c`). |
| `-prositCSVToLibrary` | Prosit / Spectronaut / DIA-NN CSV → DLIB | not yet | |
| `-blibToLib` | BLIB → DLIB | not yet | |
| `-mspToLib` | MSP / SPTXT → DLIB | not yet | |
| `-openswathTSVToLibrary` | OpenSwath TSV → DLIB | not yet | |
| `-libraryToBlib` | DLIB → BLIB | not yet | |
| `-mergeLibraries` | merge N DLIBs → one DLIB | not yet | |
| `-fastaToPrositCSV` | FASTA → Prosit input CSV | not yet | Only needed if the lab keeps a Koina path in parallel. |
| `-adjustLibraryForPTMs` | add PTM / SILAC masses to a library | not yet | |

## Detailed flag tables

Filled by PR 1 (top-level search + `-libexport`) and PR 2 (predict-library) and PR 3 (process-dia). One subsection per utility, format:

> ### `<utility>`
>
> Invocation: `java -jar encyclopedia-6.5.15.jar <flags>`
>
> | Flag | Type | Required | Default (6.5.15) | Notes |
> |---|---|---|---|---|
>
> Diff vs 2.12.30:
> - **Added**: ...
> - **Removed**: ...
> - **Renamed**: ...

Stubbed below — populated as the PRs land.

### (default) library search

Invocation: `java -jar encyclopedia-6.5.15.jar -i <input> -l <library> [...]`

Verified from `java -jar encyclopedia-6.5.15.jar --help` on 6.5.15.

**Required flags:**

| Flag | Type | Notes |
|---|---|---|
| `-i` | path | input `.DIA` or `.mzML` (or vendor `.raw`/`.d` via bundled MSRawJava) |
| `-l` | path | library `.dlib` (chromatogram-free predicted) or `.elib` (chromatogram library) |
| `-f` | path | background proteome FASTA |

**`-f` is mandatory in 6.5.15**, which is a change from older versions. Omitting it aborts the run before any file is opened with `You are required to specify an input file (-i), a library file (-l), and a fasta file (-f)` — and it does so *even when the library already carries decoys*, contradicting the older "only needed for decoy generation" guidance. Verified empirically: `-i` + `-l` alone produces that message, while adding `-f` proceeds to the library read. `--fasta` is therefore `required=True` on `massspec search` so the failure surfaces at argparse time instead of as a jar exit code after JVM startup.

**Common optional flags (wrapped as typed CLI args):**

| Flag | Default | Constellation CLI flag |
|---|---|---|
| `-o` | `<input>.encyclopedia.txt` | always set by handler to `<output-dir>/<input-stem>.encyclopedia.txt` |
| `-f` | — | `--fasta` (**required** — see note below) |
| `-ptol` / `-ptolunits` | `10 ppm` | `--precursor-tolerance` + `--precursor-tolerance-unit {ppm,Da}` |
| `-ftol` / `-ftolunits` | `10 ppm` | `--fragment-tolerance` + `--fragment-tolerance-unit {ppm,Da}` |
| `-lftol` / `-lftolunits` | `10 ppm` | `--library-fragment-tolerance` + `--library-fragment-tolerance-unit {ppm,Da}` |
| `-acquisition` | `DIA` | `--acquisition` |
| `-enzyme` | `trypsin` | `--enzyme` |
| `-frag` | `CID` | `--fragmentation` |
| `-percolatorVersion` | `v3-01` | `--percolator-version` |
| `-percolatorThreshold` | `0.01` | `--percolator-threshold` |
| `-percolatorProteinThreshold` | `0.01` | `--percolator-protein-threshold` |
| `-numberOfThreadsUsed` | `20` | `--threads` |

### Mass-tolerance units

EncyclopeDIA's `MassErrorUnitType` enum is `{PPM, AMU, RESOLUTION}`, matched case-insensitively. Three things follow, all verified against the 6.5.15 jar:

* **Daltons are spelled `AMU`.** The literal `Da` is rejected — `EncyclopediaException: Error parsing fragment mass error unit type from [Da]`. Constellation's CLI takes `Da` (the spelling used by `massspec chromatogram extract` and `peptide.match.match_mz`) and translates in `library_search._ENCYCLOPEDIA_UNITS`.
* **`AMU` is the only absolute window.** From `MassTolerance`'s constructor + `getTolerance(mass)`: `PPM` stores `percent = v/1e6` and returns `percent × mass`; `AMU` stores `amuTolerance = v` and returns it unchanged. `isRelativeTolerance()` is false only for `AMU`. This is what ion-trap data wants — ±0.8 Da at m/z 300 and at m/z 1500 alike.
* **`RESOLUTION` is `PPM` re-parameterized**, not a third kind of tolerance: the constructor stores `ppmTolerance = 500000 / v`, i.e. a window half-width of `mass / (2R)` (half the peak FWHM at resolving power `R`). `-ftolunits Resolution -ftol 30000` ≡ `-ftolunits ppm -ftol 16.67`. It adds no capability, so it is not exposed as a typed choice; reach it via `--encyclopedia-arg '-ftolunits=Resolution'`.

`-ftol` and `-lftol` are deliberately **not** coupled. `DotProduct.getIndividualPeakScores` consults both: `-ftol` is the window applied to the acquired spectrum, `-lftol` the one applied to the library peak list. Against a *predicted* library the library m/z are exact theoretical values and want the tight ppm default even when `-ftol` is widened to 0.8 Da for ion-trap data; only widen `-lftol` when searching a *measured* chromatogram `.elib` whose peaks carry the instrument's own mass error.

Unit parsing happens in `SearchParameterParser.parseParameters` **before** any input file is opened, which makes the translation cheap to smoke-test — a run with nonexistent paths still validates the units and then fails on the library read. `tests/test_massspec_library_search.py` uses exactly that (Tier B, gated on a resolved encyclopedia install).

**Anything not wrapped** passes through via `--encyclopedia-arg FLAG=VALUE` (repeatable). The full flag surface (~50 options including `-adjustInferredRTBoundaries`, `-expectedPeakWidth`, `-filterPeaklists`, `-fixed C=57.0214635`, `-foffset`, `-integratePrecursors`, `-localizationModification`, `-maskBadIntegrations`, `-maxWindowWidth`, `-minIntensity`, `-minIntensityNumIons`, `-minNumIntegratedRTPoints`, `-minNumOfQuantitativePeaks`, `-normalizeByTIC`, `-numberOfExtraDecoyLibrariesSearche`, `-numberOfQuantitativePeaks`, `-percolatorTrainingFDR`, `-percolatorTrainingSetSize`, `-poffset`, `-precursorIsolationMargin`, `-precursorWindowSize`, `-rtWindowInMin`, `-scoringBreadthType`, `-skipLibraryRetentionTime`, `-smoothIntegrations`, `-subtractBackground`, `-topNTargetsUsed`, `-usePercolator`, `-verifyModificationIons`) is documented in the jar's `--help`; expose typed flags here as lab use cases demand them.

**Output convention quirk worth knowing:** the chromatogram `.elib` lands **next to the input file** as `<input>.elib` (EncyclopeDIA convention; not redirectable via `-o`, which controls only the `.encyclopedia.txt` report). The Constellation handler runs the jar with `cwd=output_dir` to keep incidental files in the run-dir, then locates the produced `.elib` via filename convention for the auto-ingest step.

Diff vs 2.12.30:
- **Default Percolator version changed**: was `v3-01` in 2.12.30 too; the `v3-05` Linux binary is also bundled in 6.5.15 for users who want to opt into newer Percolator.
- **Vendor-raw input support**: 6.5.15 accepts `.raw`/`.d` directly via bundled MSRawJava (2.12.30 required msconvert).

### `-convert -fastaToJChronologerLibrary`

Invocation: `java -jar encyclopedia-6.5.15.jar -convert -fastaToJChronologerLibrary <flags>`

Verified from `java -jar encyclopedia-6.5.15.jar -convert -fastaToJChronologerLibrary -h` on 6.5.15.

| Flag | Type | Required | Default (6.5.15) | Notes |
|---|---|---|---|---|
| `-i` | path | yes | — | input `.FASTA` |
| `-o` | path | no | — | output `.dlib` path |
| `-addDecoys` | bool | no | `true` | predict reversed decoys + write to DLIB decoy table (roughly doubles prediction time) |
| `-adjustNCEForDIA` | bool | no | `true` | DIA-mode NCE adjustment |
| `-defaultCharge` | int | no | `3` | charge used for FASTA-prediction NCE adjustment |
| `-defaultNCE` | int | no | `33` | normalized collision energy |
| `-entrapmentSeed` | int | no | `1` | RNG seed for protein-entrapment generation |
| `-enzyme` | string | no | `Trypsin` | enzyme name |
| `-generateProteinEntrapments` | bool | no | `false` | one shuffled protein-level entrapment per target |
| `-maxCharge` | int | no | `6` | max precursor charge filter |
| `-maxMissedCleavage` | int | no | `1` | |
| `-maxMz` | float | no | `1002.7` | max precursor m/z filter |
| `-maxVariableForms` | int | no | `1000` | max number of variable-modification combinations per peptide |
| `-maxVariableMods` | int | no | `1` | max number of variable mods per peptide |
| `-minCharge` | int | no | `1` | |
| `-minMz` | float | no | `396.4` | |
| `-predictionCache` | path | no | OS user-data dir | shared FASTA-prediction cache dir |
| `-raggedNTerm` | bool | no | `false` | enumerate N-terminal ragged variants |
| `-ptmAcetyl` | enum | no | `off` | `off\|var\|fix` |
| `-ptmProteinNTermAcetyl` | enum | no | `var` | `off\|var\|fix` |
| `-ptmCarbamidomethyl` | enum | no | `fix` | `off\|var\|fix` |
| `-ptmDeamidation` | enum | no | `off` | `off\|var\|fix` |
| `-ptmDimethyl` | enum | no | `off` | `off\|var\|fix` |
| `-ptmGlyGly` | enum | no | `off` | `off\|var\|fix` |
| `-ptmHexNAc` | enum | no | `off` | `off\|var\|fix` |
| `-ptmMethyl` | enum | no | `off` | `off\|var\|fix` |
| `-ptmOxidation` | enum | no | `off` | `off\|var\|fix` |
| `-ptmPhospho` | enum | no | `off` | `off\|var\|fix` |
| `-ptmPyroGluQ` | enum | no | `var` | `off\|var\|fix` |
| `-ptmSuccinyl` | enum | no | `off` | `off\|var\|fix` |
| `-ptmTrimethyl` | enum | no | `off` | `off\|var\|fix` |
| `-ptmTMT` | enum | no | `off` | `off\|var\|fix` |

Constraints noted in `-h`:
- Peptide length fixed at 7-31 residues for JChronologer prediction.
- Charge states selected from Electrician probabilities (>1%), then filtered by charge range + m/z range.
- JChronologer always runs Sculptor CCS/IMS prediction and stores it in the library `IonMobility` column.

Diff vs 2.12.30:
- **Added**: entire utility is new in 6.5.15 (the v2.12.30 family has no in-jar FASTA-prediction path — Prosit CSV via `-convert -fastaToPrositCSV` then Koina round-trip).

### `-convert -processDIA`

Invocation: `java -jar encyclopedia-6.5.15.jar -convert -processDIA <flags>`

Verified from `java -jar encyclopedia-6.5.15.jar -convert -processDIA -h` on 6.5.15.

| Flag | Type | Required | Default (6.5.15) | Notes |
|---|---|---|---|---|
| `-i` | path(s) | yes | — | input `.mzML`, `.raw`, `.d`, or `.DIA` files; merge multiple by deliminating with `:` |
| `-o` | path | yes (merge mode) | — | output merged `.DIA` file (only used when merging multiple inputs) |

Single-input mode preprocesses one acquisition into its `.DIA` cache (the cache lands next to the input). Multi-input mode merges N acquisitions into one consolidated `.DIA` (`-o`) — this is the gas-phase-fractionation workflow: one `.raw` per GPF fraction → one merged `.DIA` covering the full m/z range.

Window-metadata preservation: per-MS2 isolation-window info (target m/z + lower/upper offsets) is read from the input via MSRawJava for vendor formats or directly from mzML and propagated to the `.DIA` cache. Overlapping / staggered DIA window schemes are preserved intact; demultiplexing happens at the search stage, not the preprocess stage.

Diff vs 2.12.30:
- **Added**: direct vendor-raw support (`.raw`, `.d`) via the bundled MSRawJava. The v2.12.30 path required msconvert (ProteoWizard) to produce mzML first.

### `-libexport`

*Pending PR 4 — default mode only; `-libexport -pecan` / `-xcordia` / `-phospho` deferred unless a use case appears.*

## v2.12.30 is no longer supported

Constellation pins EncyclopeDIA to **>= 6.5.15** (`MINIMUM_ENCYCLOPEDIA_VERSION` in `massspec.search.encyclopedia._common`). The old public 2.12.30 release is hand-rolled CLI (not picocli), single-program-per-invocation, and — fatally for the transcriptome→proteome pipeline — has no `-convert -fastaToJChronologerLibrary` predict-library utility (new in 6.5.15). Resolving a sub-6.5.15 jar now **hard-errors** (was: warn-and-proceed) via `require_min_encyclopedia`, naming the offending version and pointing at `scripts/install-encyclopedia.sh --installer`. The install script no longer fetches 2.12.30.

## When a public >= 6.5.15 release ships

No adapter code edit is needed — [constellation/thirdparty/encyclopedia.py](../../constellation/thirdparty/encyclopedia.py) globs `encyclopedia-*.jar` and picks the highest version (`pick="highest"`), so 6.5.15 / 6.6.0 / 7.0 all resolve as-is. Just fill in the `RELEASE_URL` + `RELEASE_SHA256` `__UNSET__` sentinels in [scripts/install-encyclopedia.sh](../../scripts/install-encyclopedia.sh); the default (no-`--installer`) mode then downloads + verifies + installs to `~/.constellation/encyclopedia/<version>/`, auto-detecting whether the artifact is a bare `.jar` or an install4j `.sh`. Until then, `--installer <install4j.sh>` is the working path. Wrapper code is unaffected — `run_jar` is artifact-shape-agnostic.
