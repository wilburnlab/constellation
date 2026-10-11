# EncyclopeDIA 6.9.23 — command survey

Constellation pins EncyclopeDIA to **>= 6.9.23**
(`constellation.thirdparty.encyclopedia.MINIMUM_VERSION`). This note records what that jar offers
and how Constellation tracks it. It replaces the hand-maintained 6.5.15 flag tables, which had
drifted: they listed `-scribetwodda` / `-scribetwodia` for a jar whose ScribeTwo is one program.

Surveyed 2026-10-10. Everything in "Programs" and "Changes since 6.5.15" comes from running the
jar's own `-h`. Everything in "What help does not show" comes from reading the outputs of real
6.9.23 runs (a GUI-driven peptide-centric DIA search and a ScribeTwo DDA search on a Stellar);
nothing in that section has been reproduced from Constellation's CLI yet.

## Where the facts live

The flag tables are data, not prose, so they are not repeated here.

| Artefact | What | Written by |
|---|---|---|
| `constellation/data/_raw/encyclopedia/6.9.23/help/<program>.txt` | normalised `-h` output, one file per program | `scripts/survey-encyclopedia.py` (needs the jar) |
| `constellation/data/_raw/encyclopedia/6.9.23/programs.json` | the programs probed, the jar's SHA-256, the Java used | same |
| `constellation/data/encyclopedia_cli.json` | programs and flags parsed from the help text; packaged | `scripts/build-encyclopedia-cli-json.py` (no jar needed) |

`constellation.massspec.search.encyclopedia.cli_spec` loads the JSON and answers "is this flag
documented for this program" (`is_known_flag`, `unknown_flags`). `tests/test_encyclopedia_cli_spec.py`
holds three things in place: the JSON is in step with the raw text, every flag the hand-written
wrappers emit is documented, and, when the pinned jar is installed, a fresh capture equals the
snapshot.

### Re-surveying (a new pin, or a suspicion that help changed)

```bash
constellation doctor                          # encyclopedia must be `ok`
python3 scripts/survey-encyclopedia.py        # jar -> _raw/encyclopedia/<version>/
python3 scripts/build-encyclopedia-cli-json.py
git diff constellation/data                   # the changes, flag by flag
pytest tests/test_encyclopedia_cli_spec.py
```

The capture is normalised so it is the same on every machine and every run: `[HH:MM:SS]` prefixes
are stripped, the MSRawJava "Thermo server" line is dropped (another thread writes it, so it lands
somewhere different each time), and the `-numberOfThreadsUsed` default, which is the host's core
count, is replaced with `<host cores>`.

## Installing and running the jar

- **The jar is not standalone.** Its manifest `Class-Path` points at `lib/` beside it (108 jars;
  MSRawJava is loaded at startup). A copy of `encyclopedia-6.9.23.jar` on its own fails with
  `NoClassDefFoundError`. Install the whole directory:
  `bash scripts/install-encyclopedia.sh --from-dir <extracted install>` or
  `--installer <install4j .sh>`, into `~/.constellation/encyclopedia/<version>/`.
- **Java.** The jar warns that execution "may be unstable" outside Java 11–21 and bundles
  Temurin 21. The conda env's `openjdk` is unpinned (25 at the time of the survey), so the registry
  spec sets `prefer_bundled_jre` and `run_jar` uses `jre/bin/java` beside the jar.
- **Side effect at startup.** Every invocation, `-h` included, makes MSRawJava unpack "Thermo
  server" runtime files. Where they go was not traced. Relevant to parallel runs.
- **The install4j `.vmoptions`** (`-Xmx…`) applies only to the `EncyclopeDIA` launcher script, not
  to `java -jar`, which is what Constellation runs.

## Programs

`listed` = advertised by a parent program's help. "Constellation" = how it is reached today.

| Program | argv | Listed | Constellation |
|---|---|---|---|
| peptide-centric search (the original EncyclopeDIA algorithm; DIA only) | none (also `-encyclopedia`) | yes | `massspec search` |
| ScribeTwo, spectrum-centric: DDA, DDA+ (chimeric) and DIA | `-scribetwo -searchType auto\|dia\|dda_chimeric\|dda` | yes | not yet (planned as `--search-mode spectrum-centric`) |
| Scribe, the original (DIA-style scoring on spectrum-centric analysis) | `-scribe` | yes | no |
| Walnut, FASTA search with PECAN-style scoring | `-walnut` (alias `-pecan`) | yes | no |
| Thesaurus, phospho positional isomers | `-thesaurus` | yes | no |
| XCorDIA | `-xcordia` | yes | no |
| Library export / quant report | `-libexport` (sub-modes `-pecan`, `-xcordia`, `-phospho` print the same help) | yes | `massspec library-export` |
| Batch, XML-driven | `-batch` | yes | no (`-batch -h` prints two lines, then throws `NullPointerException`) |
| ELIB browser | `-browser` | yes | never; it opens the GUI, so it is not surveyed |
| Converters | `-convert -<name>` | yes | see below |

Converters (`-convert -h`): `-fastaToJChronologerLibrary` (`massspec predict-library`),
`-processDIA` (legacy wrapper kept only for the transcriptome pipeline; Constellation writes
`.dia` itself), `-prositCSVToLibrary`, `-blibToLib`, `-mspToLib`, `-openswathTSVToLibrary`,
`-libraryToBlib`, `-mergeLibraries`, `-fastaToPrositCSV`, `-adjustLibraryForPTMs`. Accepted but
not listed: `-mergeDIA` (prints the `-processDIA` help) and `-fastaToKoinaLibrary` (jar-side Koina
prediction; `-models`, `-url`).

### ScribeTwo in particular

- Requires `-i` and `-f`. `-l` is **optional**: without a library it predicts from the FASTA on
  the fly, using its own `-nce`, `-defaultCharge`, `-minLength/-maxLength`, `-minCharge/-maxCharge`,
  `-fixed` / `-variable`, `-ptm…` and `-predictionCache`.
- It also takes the whole shared search-parameter set (`-ptol`, `-ftol`, `-enzyme`, …).
- `-mzIdentML` writes mzIdentML 1.2.0 plus an adjusted mzML, DDA mode only.
- Default report name `[input].scribe2.txt`. The GUI reference run wrote no text reports, so what
  the CLI writes besides the result library is unconfirmed.

## Changes since 6.5.15

- ScribeTwo is `-scribetwo -searchType …`; `-scribetwodda` and `-scribetwodia` are gone.
- New shared search flags: `-peakPickTimsIMS`, `-predictRawSettings`.
- `-libexport` gains `-percolatorLocation`; its help says `-f` is needed only for the Pecan /
  XCorDIA export modes. (Constellation's wrapper comments say 6.5.15 required `-f` for the default
  mode too. Unresolved until a `-libexport` run without `-f`.)
- `-convert -fastaToJChronologerLibrary`: flags and defaults unchanged.
- `-convert -processDIA` still documents `.mzML`, `.raw`, `.d`, `.DIA` input. Search programs
  document `.DIA` or `.MZML` only; whether search still takes a `.raw` directly is unconfirmed, and
  the transcriptome pipeline's per-injection stage relies on it.
- Every flag the four existing wrappers emit is still documented.

## What help does not show

- **Truncated names.** The shared-parameter table cuts names at 35 characters: it prints
  `-numberOfExtraDecoyLibrariesSearche` for `-numberOfExtraDecoyLibrariesSearched`.
  `cli_spec.is_known_flag` treats a documented name of exactly that width as a prefix.
- **Undocumented parameters.** A result library's `metadata` table records every effective
  parameter under its flag name. Recorded but not in help, peptide-centric: `-instrument`,
  `-proteinGroupScoreMode`, `-quantifyAcrossSamples`, `-percolatorTrainingIterations`,
  `-percolatorModelFile`, `-precursorIsolationRangeFile`, `-targetWindowCenter`,
  `-percolatorVersionNumber`. ScribeTwo adds `-chimericSearching`, `-adjustTolerances`,
  `-isomerSearch`, `-acceptWeakPrecursors`, `-scribeTwoBucketSize`, `-scribeTwoMinFragMz`,
  `-scribeTwoTopCandidates`, `-scribeTwoUseProteinContextFeatures`,
  `-scribeTwoUseSharedChargeStateSupportFeatures`. Also present in the jar and unlisted for
  `-libexport`: `-alignOnly`, `-alignmentFrom`. So "not in the snapshot" means "not documented",
  not "rejected". The top-level `-enableAdvancedOptions` flag may be what exposes them. **Decision
  (PI, 2026-10-11): keep a record, do not use them.** The record is
  `UNDOCUMENTED_PARAMETERS` in `constellation/massspec/search/encyclopedia/cli_spec.py`, with where
  each name came from; none has been tried on the command line and none is needed near-term.
- **`EncyclopediaTwo` is not a program to use.** A class of that name exists in the jar with a
  `-p` pre-alignment-library option and a `.encyclopedia2.txt` report, and it is in no help text.
  Per the PI (2026-10-11) nothing by that name is working or available: it is most likely
  unfinished code that shipped in the jar. Constellation ignores it. (The default program's log
  does print "Using EncyclopeDIA 2.X Scoring System"; what that refers to was not looked into.)
- **Reference runs came from the GUI**, so there are no command lines to copy. Rebuild a run's
  parameters from its result library's metadata and pass them explicitly; GUI and CLI defaults may
  differ.

### File formats written by 6.9.23-era tools

- **`.dia` is format `version 0.8.0`** (Constellation's schema is 0.5.0, from 2.12.30). New: a
  `fractions` table; `spectra.IsolationWindowTarget` (was `IsolationWindowCenter`); `spectra.TIC`;
  ion-mobility array columns on `spectra` and `precursor`; `ranges.RtStart`, `RtStop`,
  `IonMobilityStart`, `IonMobilityStop`.
- **The jar reads `.dia` metadata to decide how to search** (log: "predicted as DIA
  staggered=false (metadata)"): `rawFileStructure.dataAcquisitionType`,
  `rawFileStructure.isStaggered`, `rawFileStructure.precursorMarginSize`, among about fifty
  `file.*`, `run.*`, `instrument.*`, `acq.*`, `thermo.*`, `conversion.*` keys.
- **Single file:** `SpectrumIndex` is the raw scan number, `SpectrumName` is `scan=N`; scan and
  injection times are in seconds; DIA `PrecursorCharge` is 0; MS1 isolation bounds are the scan
  range; `ranges.DutyCycle = (RtStop − RtStart) / (NumWindows − 1)`.
- **Merged file:** `fractions` names each source; `SpectrumIndex` is renumbered 0-based and
  continuous, each fraction's MS1 block before its MS2 block; each fraction's MS1 bounds are set to
  its MS2 span; metadata shrinks to 13 keys.
- **Result libraries are container `version 0.1.18`** (Constellation's schema is 0.1.15): a
  `decoyentries` table, `entries.IonMobility`, `peptidequants.QuantIonCorrelation*`, 22 indexes.
  `SourceFile` is the `.dia`'s `sourcename`, i.e. the `.raw` name. Metadata carries
  `<source> search type` (`EncyclopeDIA`, `ScribeTwoDDA`) and `PercolatorVersion` (`3.01`,
  `JavaPot`).
- **ScribeTwo writes its result as `<input>.dia.dlib`**, in the same container.
- **Peptide-centric sidecars**, all `<input>.dia.*` beside the input: `.features.txt` (Percolator
  input, 35 columns), `.encyclopedia.txt` and `.encyclopedia.decoy.txt` (Percolator output),
  `.encyclopedia.txt.model`, `first` / `final` `rt_fit`, `ms1_fit`, `ms2_fit` tables with PDFs,
  `.encyclopedia.txt.log`.

## Open items

- Whether search accepts a `.raw` directly in 6.9.23.
- Whether `-libexport` needs `-f` in its default mode.
- What ScribeTwo writes when run from the CLI.
- Which tool and version wrote the reference `.dia` files (`conversion.outputType = EncyclopeDIA`).
