# EncyclopeDIA 6.9.23 — command survey

Constellation pins EncyclopeDIA to **>= 6.9.23**
(`constellation.thirdparty.encyclopedia.MINIMUM_VERSION`). This note records what that jar offers
and how Constellation tracks it. It replaces the hand-maintained 6.5.15 flag tables, which had
drifted: they listed `-scribetwodda` / `-scribetwodia` for a jar whose ScribeTwo is one program.

Surveyed 2026-10-10. Everything in "Programs" and "Changes since 6.5.15" comes from running the
jar's own `-h`. Everything in "What help does not show" comes from reading the outputs of real
6.9.23 runs (a GUI-driven peptide-centric DIA search and a ScribeTwo DDA search on a Stellar).
"Running a search" (2026-10-11) is the peptide-centric search reproduced from Constellation's CLI;
ScribeTwo has not been run from the command line yet.

## Where the facts live

The flag tables are data, not prose, so they are not repeated here.

| Artefact | What | Written by |
|---|---|---|
| `constellation/data/_raw/encyclopedia/6.9.23/help/<program>.txt` | normalised `-h` output, one file per program | `scripts/survey-encyclopedia.py` (needs the jar) |
| `constellation/data/_raw/encyclopedia/6.9.23/programs.json` | the programs probed, the jar's SHA-256, the Java used | same |
| `constellation/data/encyclopedia_cli.json` | programs and flags parsed from the help text; packaged | `scripts/build-encyclopedia-cli-json.py` (no jar needed) |
| `constellation/data/_raw/encyclopedia/6.9.23/schema/{result_library,spectra_file}.sql` | the `CREATE` statements of a result library (`.elib` / `.dlib`, container 0.1.18) and of a spectra file (`.dia`, format 0.8.0) | `scripts/survey-encyclopedia-outputs.py` (needs a finished run) |
| `constellation/data/_raw/encyclopedia/6.9.23/outputs/encyclopedia.json` | the files a peptide-centric search leaves behind, the columns of the tab-separated ones, the tables the result library fills, its metadata keys | same |

`constellation.massspec.search.encyclopedia.cli_spec` loads the JSON and answers "is this flag
documented for this program" (`is_known_flag`, `unknown_flags`). `tests/test_encyclopedia_cli_spec.py`
holds three things in place: the JSON is in step with the raw text, every flag the hand-written
wrappers emit is documented, and, when the pinned jar is installed, a fresh capture equals the
snapshot. `tests/test_encyclopedia_output_capture.py` holds Constellation's own schemas to the
captured table definitions: the `.dia` writer matches exactly, and the result-library writer
(still the 0.1.15 layout) differs by a stated list (a `decoyentries` table, `entries.IonMobility`,
two `peptidequants.QuantIonCorrelation*` columns, 16 indexes).

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
- **A helper process and temp files on every invocation.** MSRawJava unpacks a .NET "Thermo
  server" (about 200 MB of runtime files) into `java.io.tmpdir` as `msraw-thermo-<n>/` and starts
  it, for every run, `-h` included. The jar does not reliably tidy up. Measured: every run leaves
  its directory behind, empty after a clean exit and holding 65-80 MB after an error exit; and the
  server process is usually stopped but not always (several were found running hours after quick
  error exits made while the machine was busy). The test and survey runs of the first two days
  had left 237 directories (1.4 GB) in `/tmp` and six orphan servers. So `thirdparty.jvm` trusts
  nothing: `run_jar` and `capture_jar` start the JVM in its own session, stop whatever is left in
  its process group when it is done (also on Ctrl-C), and give each run a private
  `java.io.tmpdir` that is removed afterwards. Anything that starts the jar should go through one
  of the two. A side benefit: two searches at once cannot collide on temp files, and each run's
  Percolator executable is unpacked into its own directory. The cost of the separate session is
  that a signal to the wrapper no longer reaches the JVM, so SIGTERM and SIGHUP are handled while
  one is running (`jvm.cancellation_scope`): a real search cancelled with SIGTERM twenty seconds
  in exits with status 143 in about a second, JVM, helper server and temp directory gone.
  SIGKILL of the wrapper cannot be intercepted.
- **Percolator is an external executable, and the default one does not run under WSL2.** See
  "Running a search" below.
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
- **The reference `.dia` files were written by MSRawJava** (its `EncyclopeDIAFile` class; the
  library is bundled in both EncyclopeDIA and MSForest). Found while reproducing them:
  - Scalars are stored float32-rounded; `TIC` is the sequential float32 sum of the stored
    intensities; `ranges.DutyCycle` is computed from unrounded scan times.
  - **DDA:** `spectra.PrecursorName` is the parent MS1 scan number (0 in DIA, which has none);
    `IsolationWindowTarget` is the full-precision isolation m/z, not the filter string's four
    decimals; metadata carries `rawFileStructure.dataAcquisitionType` only, with no `isStaggered`
    or `precursorMarginSize`.
  - **DDA `ranges` are a sample.** 134,180 rows for 159,789 distinct windows, and `NumWindows`
    sums to 149,158 of 179,532 MS2 scans. What is sampled is not recoverable from the file.
  - **Four of the eight reference `.dia` files are each missing one MS2 scan.** The `.raw`
    contains it, `massspec convert` reads it, and the file's own `acq.ms2_count` counts it, but
    the `spectra` table has one row fewer: scan 156345 of `…gpf-500-600…`, 88035 of
    `…gpf-800-900…`, 225323 of `…350-1200_8mz…`, 139436 of `…dda…-1`. Nothing about those scans
    looks unusual (thousands of peaks, ordinary injection times). The merged GPF file inherits two
    of them. Constellation keeps every scan, so its files differ from the reference by exactly
    these.
- **Peptide-centric sidecars**: `.features.txt` (Percolator input, 35 columns),
  `.encyclopedia.txt` and `.encyclopedia.decoy.txt` (Percolator output), `.encyclopedia.txt.model`,
  `first` / `final` `rt_fit`, `ms1_fit`, `ms2_fit` tables with PDFs, `.encyclopedia.txt.log`. The
  GUI writes them all beside the input as `<run>.dia.*`, and so does the command line when no
  `-o` is given: 24 files, the result library as `<run>.dia.elib`. With
  `-o <dir>/<run>.encyclopedia.txt` the same 24 land in `<dir>` as `<run>.*` instead and nothing
  is written beside the input (how `massspec search` ran until 2026-10-12; it now runs the jar on
  a link in the search's own directory with no `-o`, see "A result beside the input is the jar's
  completion marker" below). The inventory of the default arrangement is checked in
  (`outputs/encyclopedia.json`, names as `<input>.*` where `<input>` is the input's file name).

## Running a search (first real run, 2026-10-11)

The wide-window DIA run of the reference set (`261008_u2os_dia_100min-IOa_HCD26_350-1200_8mz`,
106 windows of 8 m/z, 237,652 MS2 scans) searched against the GPF chromatogram library, from
`constellation massspec search` on WSL2, 12 cores. Parameters rebuilt from the reference result's
metadata: all three tolerances 0.35 Da (`AMU` to the jar), and `--encyclopedia-arg=-fixed=`,
because the reference recorded an empty `-fixed` where the command line defaults to
`C=57.0214635`. The CLI run's result metadata then equals the GUI run's in 56 of 59 keys; the
other three are the thread count, the Percolator path and `pi0`.

| Run | Input | Percolator | Peptides at 1% FDR | Against the GUI reference |
|---|---|---|---|---|
| reference | reference `.dia`, GUI on Windows | 3.01 | 31,833 | |
| A | copy of the reference `.dia` | 3.01 | 31,830 | 31,826 shared; 7 and 4 unshared |
| A2 | the same file again | 3.01 | 31,833 | identical set |
| **B** | **`.dia` written by Constellation from the `.raw`** | 3.01 | **31,833** | **identical set** |
| C | copy of the reference `.dia` | 3.05 | 31,836 | 31,829 shared; 4 and 7 unshared |

- **A Constellation-written spectra file searches the same as the reference file.** B identifies
  exactly the reference's 31,833 precursors and 5,168 proteins; over all of them retention times
  and quantities are identical and the largest score difference is 0.0011.
- **Run-to-run noise is real and small.** Of two repeat searches of one file, one matched the
  reference exactly and one differed by 11 precursors (Jaccard 0.99965). The scoring itself is
  reproducible: the Percolator input has the same 76,400 rows in the same order every time, and
  between any two runs 40 to 190 of them differ, only in `deltaRT` (by at most 0.035), which
  comes from the retention-time fit. That is enough to move a handful of precursors across the
  FDR line. For every pair of runs, shared precursors have identical retention times and, for
  the 26,130 the jar quantifies, identical quantities (it gives the other 18% of its
  identifications a quantity of zero). `massspec encyclopedia compare-results` is the tool; a repeat search is the
  yardstick. The reproduction is kept as an opt-in test,
  `tests/test_encyclopedia_search_reference.py` (it needs the jar, the data and minutes of CPU,
  so it runs only when `CONSTELLATION_SEARCH_REFERENCE_*` name them).
- **The jar reads a `.raw` directly, with no intermediate file.** `-i <run>.raw` is accepted by
  the search (help documents `.DIA` or `.MZML` only), is read through the Thermo server, and
  nothing is written beside the input. So the transcriptome pipeline's per-injection stage, which
  passes `.raw` straight to search, still works on 6.9.23. For `…gpf-900-1000…` the direct `.raw`
  search and a search of the Constellation-written `.dia` gave the same Percolator input, row for
  row (3,771 rows, none differing), and the same 3,600 precursors with zero difference in score,
  retention time or quantity.
- **Time and memory.** 8.4 to 9.4 minutes for the wide-window run on 12 cores with other work
  going on (the GUI reference: 3.0 minutes on 96 threads); about a minute for one GPF fraction.
  Resident memory stayed under 4 GB with `-Xmx10g` and `-Xmx16g`.
- **Two searches at once work.** A2 and C ran concurrently (10 GB heap each, 12 threads each on
  12 cores), about 15 minutes each, with separate output directories and the private temp
  directories described above. Nothing collided.
- **JVM flags.** An older lab project carried `-XX:TieredStopAtLevel=1` ("prevents OpenJDK 11 C2
  JIT segfaults on WSL2"), `--enable-native-access=ALL-UNNAMED` and two
  `-Djdk.xml.*EntitySizeLimit=0` properties as must-keep flags. None is passed here and none was
  needed: six complete searches under the bundled Temurin 21 on WSL2 with only `-Xmx` and
  `-Djava.awt.headless=true`, no JVM crash, nothing on stderr. (The XML limits concern mzML
  input, which was not exercised. Whether that old WSL2 segfault was really the JIT, or the
  Percolator crash below seen from further away, is not known.)

### A result beside the input is the jar's completion marker

By design (PI, 2026-10-12): the jar writes its outputs beside its input, named for it, and a
result already there tells it the search is finished. It works as the jar's own `_SUCCESS` flag,
and it does not compare settings. Noticed when a test of re-searching one acquisition with
different settings would not fail the way it should have; then measured on a GPF fraction, first
searched at `-ptol 0.35` (AMU):

| Second search, asked for at `-ptol 0.2` | Wall | What the jar did | Precursors | `-ptol` in the result's metadata |
|---|---|---|---|---|
| into the directory holding the first search's files | 36 s | logged "Previously found 3600 peptides identified at 1.0% FDR", exited 0 | 3,600 | 0.35 |
| in a fresh directory holding only a symlink to the `.dia`, no `-o`; the first search's files sit beside the link's *target* | 56 s | a full search; outputs beside the link as `<run>.dia.*`; nothing written beside the target | 3,583 | 0.2 |
| the real `.dia` used where it is, the first search's files beside it, `-o` naming a report in another fresh directory | 52 s | a full search; outputs as the `-o` name's `.*` in that directory | 3,583 | 0.2 |

So: the marker is looked for where the outputs go, which is beside the input unless `-o` moves
them; the jar does not follow a symlink back to its target's directory; and a `.dia` opens through
a symlink. (The frozen pipeline's notes record that a Thermo `.raw` does *not*, which is why it
hard-links; Constellation never hands the jar a `.raw`.) The class files carry the same message
for ScribeTwo.

What Constellation does with this (PI's arrangement): every search gets a directory of its own
holding a link to the spectra file and nothing else, and the jar is run on the link with no `-o`.
Its outputs then carry the jar's own names (`<run>.dia.elib`), which is the layout `-libexport`
pairs up, and the spectra file is written once per acquisition however many times it is searched.
`outputs/encyclopedia.json` is the inventory of such a directory. As a second check the workflow
compares the settings a result records about itself (the result library's `metadata` table lists
most flags under their own names) with the ones it passed.

### Percolator on a `vsyscall=none` kernel (WSL2)

The first search here failed after 6.5 minutes with
`Percolator exited with non-zero status: 139`, having scored the whole run.

- The jar carries its Percolator builds as executables (`bin/percolator-v{2-10,3-01,3-05}.*`),
  unpacks the selected one into `java.io.tmpdir` and runs it twice per search (after the first
  pass with `--maxiter 3`, then for the final result).
- The default on Linux, `percolator-v3-01.lin`, is a statically linked build from May 2017
  ("for GNU/Linux 2.6.18"). Its C library implements `time()` by calling the legacy vsyscall page
  at `0xffffffffff600400`. A kernel booted with `vsyscall=none` does not map that page, so the
  process dies with SIGSEGV before printing anything. The kernel log says so directly:
  `vsyscall attempted with vsyscall=none ip:ffffffffff600400`. The WSL2 kernel (6.6.87 here) is
  booted that way. It is one call site in the binary; a copy with that one call replaced by the
  real `time` system call runs normally, which is how runs A, A2 and B above were made with
  Percolator 3.01.
- `percolator-v2-10.lin` and `percolator-v3-05.lin` are dynamically linked and have no such
  call. 3.05 needs `libgomp.so.1`, which this machine has only in the conda env; Constellation's
  CLI puts the env's `lib` on `LD_LIBRARY_PATH` for its children, and run C used it that way.
- **`-percolatorVersion` also accepts the path of a Percolator executable.** Not in the help;
  read from the jar's `PercolatorVersion` class (`parseFilePath`, then `parseUri`) and confirmed
  by runs A, A2 and B. The result's metadata then records the path under `-percolatorVersion`
  and the version the executable reports under `PercolatorVersion`.
- **What Constellation does.** `search.encyclopedia.percolator.percolator_blocker` checks, before
  the JVM starts, whether the selected build needs the vsyscall page and whether this kernel maps
  it (`[vsyscall]` in `/proc/self/maps`). `massspec search` refuses with the three ways out
  (boot with `vsyscall=emulate`, which on WSL2 is `kernelCommandLine = vsyscall=emulate` under
  `[wsl2]` in `.wslconfig`; `--percolator-version v3-05`; `--percolator-version <path>`), and
  `constellation doctor` has a `percolator` row. The kernel-parameter route is the documented
  one for old static binaries and was **not** tried on this machine (it needs a WSL restart).
  Constellation does not patch or replace the binary itself.
- **Not covered.** The transcriptome pipeline calls the search wrapper directly, so it gets no
  pre-flight and would fail at its first search on such a kernel. `-libexport` also runs
  Percolator and takes `-percolatorLocation` rather than `-percolatorVersion`; it has not been
  run here.

### What the existing ingest does with a 6.9.23 result

`massspec search` follows the jar with `read_encyclopedia(<run>.elib)` and writes Library, Quant
and Search ParquetDir bundles. The `.elib` is not changed and remains the complete record; the
bundles are a derived copy. Checked on run A (31,830 precursors, container 0.1.18). What the copy
gets right: the 31,830 precursors with their retention times, 31,830 peptide scores, 5,167
protein scores, and peptide-level quantities taken from the jar's `peptidequants.TotalIntensity`.
What it gets wrong or leaves out, most serious first:

1. **`precursor_quant.intensity` is not EncyclopeDIA's quantity.** It is the sum of every
   intensity in the entry's stored spectrum (about 36 peaks), where the jar's own number is the
   sum over the few ions it chose to quantify with. Against `TotalIntensity` for the same
   precursor: median ratio 1.98, 5th to 95th percentile 1.02 to 219, 15.8% within 10%, log-log
   r = 0.70. The 5,608 precursors the jar identified but could not quantify
   (`NumberOfQuantIons` = 0, `TotalIntensity` = 0) get a non-zero value.
2. **188 peptides have no peptide-level quantity**: every pyro-Glu peptide. The jar spells an
   N-terminal modification two ways in one file (see the format note below), and the reader
   resolves the two spellings of `-17.026549` to different ProForma strings (`[-17.026549]-Q…`
   and `Q[-17.026549]…`; neither is resolved to the UNIMOD entry), so the quantity row finds no
   peptide. N-terminal acetyl (570 peptides) survives because both spellings resolve to
   `[UNIMOD:1]-`.
3. **8.5% of stored fragment peaks are not carried over** (100,450 of 1,181,605). The reader
   keeps a stored peak only if it lies within 20 ppm of a b or y ion at fragment charge 1 or 2.
   (A further 2,499 rows of the fragment table repeat a stored peak: it fits a singly and a
   doubly charged ion of the same m/z and is listed under both, which is the matcher's
   documented behaviour.) On an 8,000-entry sample the dropped peaks are: the intact precursor
   or its water / ammonia loss, at any charge up to the precursor's (6.7% of all peaks, 10.4% of
   stored intensity; EncyclopeDIA stores these in the spectrum, they are not sequence fragments);
   b or y ions at charge 3 or more (1.6% of peaks, 0.6% of intensity); everything else 0.1%. No
   entry lost all its peaks; one of 8,000 was left with fewer than three.
4. **Nothing else in `entries` or `peptidequants` is carried**: which ions were quantified
   (`QuantifiedIonsArray`, `QuantIon*Array`), per-fragment correlations, integration boundaries,
   chromatograms. Decoy rows of `peptidescores` (317) and `proteinscores` (27) are dropped, by
   the reader's default. `protein_quant` is empty.
5. **It was slow**: 223 s on this machine (8 to 15 minutes alongside other searches), 99 s of
   it a loop comparing every protein link with every peptide (35,767 x 31,830 = 1.1 billion
   string comparisons). On the human predicted library (2.5 million entries, 2.7 million protein
   links) the same loop extrapolates to about a week, so `predict-library` with its default
   ingest, or loading that `.dlib`, could not finish.

None of this affects the search or anything measured in this note: `compare-results` reads the
`.elib` directly.

**What was done about it (2026-10-11, PI decision):** only what costs nothing later. The
linking loop is replaced by an index (same output, 37 s for run A; all nine Parquet tables are
identical to what the old code wrote). That removes one wall, not the other: the reader still
builds every fragment row as a Python object in memory (about 1 GB for run A's 32,000
precursors, so roughly 85 GB extrapolated for the 2.5-million-entry predicted library), so a
proteome-scale `.dlib` still cannot be ingested with fragments; that needs the streaming phase 4
reader. And the reader no longer keeps quiet:
`EncyclopediaReadResult.caveats` states items 1 to 3 in sentences with counts, and every command
that writes the Parquet copy prints them and records them under `ingest.caveats` in its
manifest. The numbers themselves are unchanged. Item 2 is fixed in phase 3 (the modified-sequence
parser, which the content-derived ids need anyway); items 1, 3 and 4 are fixed once, in the
phase 4 reader, on top of which `read_encyclopedia` is to be rebuilt so that there is one
reader. Nobody depends on the current Parquet numbers, the frozen pipeline's included.

**Format note found on the way: N-terminal modifications are spelled two ways in one result
file.** `entries` and `peptidescores` write the modification before the first residue
(`[+42.010565]MDFQHRPGGK`); `peptidequants` writes it on the residue (`M[+42.010565]DFQHRPGGK`),
ahead of the residue's own modification if it has one (`C[+42.010565][+57.0214635]DF…`). 758 of
31,830 precursors here (570 acetyl, 188 pyro-Glu). A join between the two tables on
`PeptideModSeq` loses them; `compare_results` normalises to one spelling
(`_result_diff._precursor_key`), and the phase 4 reader must do the same.

### The reader that replaced it (phase 4, 2026-10-12)

`read_encyclopedia` is now two layers. `io/encyclopedia/_mirror.py` copies any table of a
`.dlib` / `.elib` to Arrow in batches, decoding the array columns from one codec map; it opens
the file read-only and immutable. `_project.py` builds `Library` / `Quant` / `Search` from the
tables Constellation models. Checked field by field against SQL on three real files:

| File | Precursors | Result |
|---|---|---|
| Run A, peptide-centric result (`.elib`) | 31,830 | Row counts equal SQL. Every precursor's quantity, retention time, integration window and ion count, and every score row, equal the file's. 1,181,605 fragment rows = stored peaks: 93.3% b/y, 6.7% intact precursor (`p`, `p-H2O`, `p-NH3` at any charge to the precursor's), none unannotated. 9.5 s with fragments, 1.5 s without (the old reader: 223 s, then 37 s) |
| GPF chromatogram library (`.elib`) | 76,448 | Same checks pass |
| ScribeTwo DDA result (`.dia.dlib`) | 17,640 | Same checks pass; see below for its stored spectra |
| Predicted human library (`.dlib`) | 2,499,798 | Read without fragments in 190 s at 4.2 GB; 20,597 proteins, 1,390,718 peptides. The old reader could not finish it |

The five points above, in the new reader: (1) `precursor_quant.intensity` is `TotalIntensity`;
`n_quant_ions` 0 marks identified-not-quantified. (2) Fixed in phase 3. (3) Every stored peak
is a fragment row: b/y at any charge up to the precursor's (the nearer ion, the lower charge on
a tie), the intact precursor as an mzPAF `p` ion, anything else kept unannotated. (4)
Integration boundaries and decoy score rows are carried; chromatograms, correlations and the
quantified-ion flags are reachable through the mirror and get their own tables in phase 7. (5)
See the table.

**A ScribeTwo DDA result stores whole observed spectra, not matched ions.** The reference DDA
result holds 19.1 million peaks for 17,640 entries (about 1,080 per entry, against 37 in a
peptide-centric result), and 99.6% of them match no b/y or precursor ion at 20 ppm. So its
fragment table is nearly all unannotated rows, and annotating it properly means using the
tolerance the result itself records (it is ion-trap data searched at 0.35 Da), and probably
not writing the unmatched peaks as library fragments at all. That is phase 6's to decide.

### The quantitative report (`-libexport`), first run 2026-10-17

Run by hand over a directory of symlinks to two searches' files (the spectra file, `.dia.elib`,
`.features.txt`, the Percolator reports, the fit tables; PDFs left out), every linked file
checksummed before and after:

- **`-f` is required.** Without it: "You are required to specify an input file or directory (-i),
  an input library file (-l), a fasta database (-f), and an output library file (-o)", although
  the help lists `-f` as needed only for the Pecan and XCorDIA exports. (The frozen pipeline had
  noted the same of 6.5.15.)
- **It re-scores with JavaPot.** The report's metadata says `PercolatorVersion JavaPot`; the
  Percolator executable is not run, so a kernel the bundled Percolator cannot run on is no
  obstacle, and `-percolatorLocation` is not needed.
- **It writes into its input directory**: `<output stem>_concatenated_features.txt`, `_results.txt`,
  `_decoy.txt`, `_results.txt.model`. It changed none of the linked files (all 26 checksums equal).
- **Beside the report** it writes `<output>.peptides.txt` and `.proteins.txt` (matrices, one column
  per run), `.diffacto.csv` and `.diffacto.samples.lst`.
- **The report**: `entries` one row per peptide across all runs, each under one run; `peptidequants`
  a row per peptide per run that quantified it; `peptidescores` per run, decoys included;
  `proteinscores` under the run `global`; metadata `sourcefile` lists the runs joined by `|` and
  records `RT align between samples true` with `-a true`.
- 21 s for two fractions (14,507 + 3,552 entries). With no shared peptides between the two
  fractions the alignment had nothing to fit ("Not enough points ... to compute regression between
  samples"), as expected.
- **Its quantities are its own, not the searches'** (measured on the full chain, one wide-window
  run): the report re-quantifies every precursor on `-numberOfQuantitativePeaks` ions (5), the
  same ions in every run, where the search summed every quantitative ion it found (17-22 for
  typical peptides). Identifications, scores and retention times equal the search's; quantities
  do not (median |log2 ratio| 1.26, log-scale Pearson r 0.66), and 765 of 31,895 precursors get
  no quantity row at all.
- **It records `-ptol` and `-lftol` without using them.** Neither is in its help, and the report's
  metadata shows the jar defaults (10 ppm) whatever the searches used. Passing the searches'
  0.35 Da (with `--allow-undocumented`) left every one of the 31,130 quantities identical, so
  Constellation does not pass them.

### Reproducing a search: Percolator's training subsample (2026-10-17)

A search reproduces exactly only while Percolator trains on every PSM. The jar hands Percolator
`-percolatorTrainingSetSize` PSMs at most (default 500,000) and Percolator draws them by their
order in its input, which follows the jar's thread count. Measured on the six-fraction
chromatogram library rebuilt from the lab's GPF files at the GUI reference's settings (12 threads
here, 96 in the GUI's run): the same 2,007,934 PSMs went to Percolator, the first pass trained on
different subsamples (291,345 vs 290,790 positives), the retention-time and mass calibrations fitted
from that pass shifted a little, and those calibration features (`deltaRT`, `ms1MassError`,
`ms2MassError`) are the only features that differ on the 2,007,884 PSMs both runs share. The
result: 76,775 against 76,448 precursors at 1% FDR, 75,929 shared (Jaccard 0.982), with identical
retention times and quantities on every shared one. The wide-window search (76,399 PSMs, no
subsample) matched its GUI reference exactly. A search with more PSMs than the training-set size
is reproducible only with that size raised above its PSM count.

## Open items

- What ScribeTwo writes when run from the CLI.
- Whether ScribeTwo's DDA search reads the `ranges` table (Constellation writes every window; the
  reference writes a sample).
- Whether `vsyscall=emulate` in `.wslconfig` makes the bundled Percolator 3.01 run under WSL2
  (expected; untested).
- Which of its flags the jar rewrites before recording them in a result's metadata (ScribeTwo's
  `-adjustTolerances` does; `-predictRawSettings` might). Until that is known, a recorded
  setting that differs from the one passed is reported as a caveat, not treated as an error.

Closed on 2026-10-11: search accepts a `.raw` directly (yes, and writes no `.dia`); where the
Thermo server's files go (`java.io.tmpdir`).
