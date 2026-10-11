# EncyclopeDIA integration: convert, search, parse (pinned to 6.9.23)

## Status

Plan approved by the PI on 2026-10-10. Eight PR-sized phases (P0–P7).

- **P0 (command survey and pin): done, 2026-10-10.** The jar is pinned to 6.9.23, installs as a
  whole app directory and runs under its bundled Java; its command line is checked in as data
  (`constellation/data/encyclopedia_cli.json`, raw help under `data/_raw/encyclopedia/6.9.23/`);
  `doctor` reports a too-old jar. Survey note: [`encyclopedia-6.9.23-survey.md`](encyclopedia-6.9.23-survey.md).
- **P1 (EncyclopeDIA spectra file, the `.dia`): done, 2026-10-11.**
  `massspec encyclopedia {export-spectra, import-spectra, compare-spectra}`;
  `io/encyclopedia/{_dia.py, _dia_write.py, _dia_import.py, _dia_diff.py}`;
  `massspec/acquisition_structure.py`; `SCAN_METADATA_TABLE` v3 (`scan_window_lower/upper`).
  Measured against the reference files in `/mnt/e/261010_u2os`, starting from each `.raw`
  (`massspec convert` → `massspec encyclopedia export-spectra` →
  `massspec encyclopedia compare-spectra`, no tolerances):
  - **Single-run DIA** (`…gpf-900-1000…`): identical. 81,850 scans, 109,001,282 peaks, the 49
    isolation windows and every metadata value both files carry.
  - **Merged GPF** (six fractions): identical in all 1,000,398 shared scans, 1,147,820,018 peaks,
    the 294 windows, the `fractions` table and all metadata. The only difference is two MS2 scans
    the reference lacks, and the `SpectrumIndex` shift that follows from them.
  - **DDA**: identical in all 188,509 shared scans and 195,219,042 peaks. Differences: one MS2
    scan the reference lacks, and the `ranges` table (the reference lists a sample of the
    windows; Constellation lists all of them, by design).
  - **Round trip** (`.dia` → bundle → `.dia`): every scan and peak exact; only values derived
    from scan times at float32 move, by one unit in the last place.
  - **The reference writer drops scans.** Four of the eight reference files are each missing
    one MS2 scan that the `.raw` contains and the file's own `acq.ms2_count` counts.
    Constellation keeps them.
  - Writing took 30 s for the 1 GB DIA run, about a minute for the 1.8 GB DDA run, and about
    8 minutes for the merged 9.6 GB file (12 cores, with other jobs running). The writer reads the MS1 scan range from the new columns, or from `filter_string`
    for bundles converted before they existed, so nothing needs re-converting.
  - Not done here, and still the first job of P2: an actual search on a Constellation-written
    file. Equal content makes that very likely to work, but it has not been run.
  - 15 provenance metadata keys the reference carries (instrument-method text, sample vial,
    integrated-intensity summaries) are not written, because a bundle does not hold them. The
    search jar reads none of them.
- **P2–P7: not started.** `process-dia` (jar-side conversion) is still a CLI verb; the plan
  removes it in P4, once the new search path takes bundles directly.
- **Naming (PI, 2026-10-11).** No command or flag is named `dia`: DIA is an acquisition method,
  and `.dia` is one tool's file extension. The P1 commands were first built as
  `massspec dia {export,import,diff}` and renamed to the `massspec encyclopedia …-spectra` group
  above; the plan text below has been updated to match. Later phases follow the same rule
  (`massspec encyclopedia {run,flags}` was already tool-named).
- **PI decisions on P0's open items (2026-10-11).** `EncyclopediaTwo` is not a working or
  available program (probably unfinished code in the jar) and is of no interest: ignore it.
  Undocumented parameters are to be recorded, not used: they are listed in
  `cli_spec.UNDOCUMENTED_PARAMETERS` and are not needed near-term. Both supersede the "to
  confirm" wording about them further down.

The rest of this file is the plan as approved.

## Context

EncyclopeDIA is becoming the lab's main search engine for more than the transcriptome work, and
6.9.23 adds features Constellation does not reach (ScribeTwo spectrum-centric search; Chronologer
prediction is wrapped already). Today the jar is driven through four hand-written wrappers, the
only complete run is the transcriptome pipeline's, file conversion is left to the jar, and most of
what the jar writes is not parsed (see `docs/plans/massspec-pre-revision-review.md`).

Goal: general mass-spec wiring to **process files** (Constellation's own `.raw` → parquet → `.dia`),
**submit searches** (peptide-centric or spectrum-centric as a toggle), and **parse every output**
into parquet with stable ids, as the base for later analysis and a mass-spec browser.

## Decisions already made

- **Pin:** 6.9.23 is the single minimum for every jar-backed command, the transcriptome pipeline
  included (two version strings in its tests change; its code does not).
- **Jar access:** the 6.9.23 install is on this machine (`/mnt/d/EncyclopeDIA`) and its commands
  have been surveyed. Reference data and outputs are at `/mnt/e/261010_u2os` and have been
  inspected (both in the next two sections). Nothing has been run through Constellation yet.
- **Execution:** Constellation runs the jar inside the user's allocation: one search per
  acquisition, parallel with a thread/heap budget, resume, detach/follow. No SLURM submission.
- **Transcriptome pipeline:** untouched. The four legacy wrapper modules it imports stay as they
  are (its tests patch them by module attribute), so new code gets a new runner beside them.
- **Inputs:** Thermo `.raw` only. The standalone `process-dia` verb is removed; mzML and `.d` are
  refused with a clear message. Constellation never calls MSForest.

## What the 6.9.23 jar says (surveyed 2026-10-10)

From running `-h` with the bundled Java in the install at `/mnt/d/EncyclopeDIA`
(top level, `-scribetwo`, `-convert`, `-convert -processDIA`, `-convert -fastaToJChronologerLibrary`,
`-libexport`), plus strings read from the jar's classes for things help does not list. Nothing has
been searched yet, so behaviour marked "to confirm" needs a real run.

- **Programs:** default (peptide-centric EncyclopeDIA), `-scribetwo`, `-scribe`, `-walnut`,
  `-thesaurus`, `-xcordia`, `-libexport`, `-convert`, `-browser`, `-batch`. An `EncyclopediaTwo`
  class also exists (`-p` pre-alignment library, report `.encyclopedia2.txt`) but is not in the
  help; how it is invoked is to confirm (the default program's log already reports "EncyclopeDIA
  2.X Scoring System").
- **Search inputs** are documented as `.DIA` or `.MZML` for both the default program and ScribeTwo.
  Whether a `.raw` is still accepted directly by search is to confirm; the frozen pipeline's
  per-injection stage passes `.raw` straight to search.
- **ScribeTwo:** `-searchType auto|dia|dda_chimeric|dda` (default `auto`); requires `-i` and `-f`.
  `-l` is optional: without a library it predicts from the FASTA on the fly (`-nce`,
  `-defaultCharge`, `-minLength/-maxLength`, `-minCharge/-maxCharge`, `-ptm…`, `-predictionCache`).
  It also takes the whole shared search-parameter set (`-ptol`, `-ftol`, `-enzyme`, …). It scores
  with a built-in "JavaPot" and can write mzIdentML (`-mzIdentML`, DDA only). The jar names
  `.scribe2.txt`, `.scribe2.features.txt`, decoy and protein reports as CLI outputs; the GUI
  reference run produced none of them, so what the CLI actually writes is to confirm.
- **Default program:** 45 shared parameters with printed defaults (threads 12, Percolator
  `v3-01`); new since 6.5.15: `-peakPickTimsIMS`, `-predictRawSettings`.
- **`-convert`:** help lists `-processDIA` (still documents `.mzML`, `.raw`, `.d`, `.DIA` input),
  `-fastaToJChronologerLibrary` (flags and defaults unchanged from 6.5.15), `-prositCSVToLibrary`,
  `-blibToLib`, `-mspToLib`, `-openswathTSVToLibrary`, `-libraryToBlib`, `-mergeLibraries`,
  `-fastaToPrositCSV`, `-adjustLibraryForPTMs`. Present in the jar but not listed: `-mergeDIA`,
  `-fastaToKoinaLibrary`.
- **`-libexport`:** help says `-f` is required only for the Pecan / XCorDIA export modes; adds
  `-percolatorLocation`. Present but unlisted: `-alignOnly`, `-alignmentFrom`,
  `-quantifyAcrossSamples`.
- **Existing wrappers:** every flag the four current wrappers emit still exists (acceptance needs a
  run).
- **Outputs named in the jar:** `.encyclopedia.txt`, `.encyclopedia.decoy.txt`,
  `.encyclopedia.protein.txt`, `.encyclopedia.protein_decoy.txt`, `.features.txt`, the `.scribe2.*`
  equivalents, `.demux.dia`, `.log`.
- **Runtime:** the jar is not standalone; it needs the install directory's `lib/` (108 jars,
  MSRawJava loaded at startup, which unpacks "Thermo server" runtime files even for `-h`). It warns
  outside Java 11–21: the conda env's OpenJDK is 25, the bundled JRE is Temurin 21. The runner
  currently prefers `java` on `PATH`, so for this tool it must prefer the bundled JRE.

## What the reference data shows (`/mnt/e/261010_u2os`, inspected read-only)

Six GPF `.raw` files with a `.dia` each, their merged `.dia`, a peptide-centric search of the
merge, one wide-window DIA run searched against that result, and one DDA run searched with
ScribeTwo. All from a Stellar (ion trap; searched at 0.35 AMU).

- **`.dia` is format `version 0.8.0`,** not the 0.5.0 of the old local sample and of the repo's
  schema. Differences: a `fractions` table; `spectra.IsolationWindowCenter` is now
  `IsolationWindowTarget`; `spectra.TIC`; ion-mobility array columns on `spectra` and `precursor`;
  `ranges` gains `RtStart`, `RtStop`, `IonMobilityStart/Stop`.
- **The jar reads the `.dia` metadata to decide how to search** (the log says "predicted as DIA
  staggered=false (metadata)"). The keys include `rawFileStructure.dataAcquisitionType`,
  `rawFileStructure.isStaggered`, `rawFileStructure.precursorMarginSize`, plus ~50 `file.*`,
  `run.*`, `instrument.*`, `acq.*`, `thermo.*`, `conversion.*` entries. A Constellation-written
  `.dia` has to supply the ones the jar acts on.
- **Single-file conventions:** `SpectrumIndex` is the raw scan number and `SpectrumName` is
  `scan=N`; times and injection times are in seconds; DIA `PrecursorCharge` is 0; MS1 isolation
  bounds are the scan range (350–1250); `DutyCycle` = `(RtStop − RtStart) / (NumWindows − 1)`
  (1.728 s here), confirming the formula.
- **Merged-file conventions:** `fractions` names each source `.dia`; `SpectrumIndex` is renumbered
  0-based and continuous, each fraction's MS1 block followed by its MS2 block; each fraction's MS1
  bounds are set to that fraction's MS2 span; metadata shrinks to 13 keys.
- **Result libraries are container `version 0.1.18`** (the repo's schema is 0.1.15): a
  `decoyentries` table, `entries.IonMobility`, `peptidequants.QuantIonCorrelation*`, 22 indexes.
  `SourceFile` is the `.dia`'s `sourcename`, i.e. the `.raw` name, which is the key outputs bind
  on. Metadata records `search type` (`EncyclopeDIA`, `ScribeTwoDDA`), `PercolatorVersion`
  (`3.01`, `JavaPot`), the library searched, and every effective parameter, including some help
  does not list (`-proteinGroupScoreMode`, `-instrument`, `-percolatorTrainingIterations`).
- **ScribeTwo writes its result as `<input>.dia.dlib`,** same container, with scores and quant
  tables filled. The reference run was made from the GUI, which produced no text reports, so there
  is no command line and no `.scribe2.*` file to parse. The result's metadata records all 88
  effective parameters (for example `-chimericSearching true`, `-adjustTolerances true` with the
  adjusted tolerances, `-instrument Iontrap/Iontrap`, the PTM states), so the run can be rebuilt
  as a CLI call.
- **Peptide-centric sidecars,** all beside the input as `<input>.dia.*`: `.features.txt`
  (Percolator input, 35 columns), `.encyclopedia.txt` and `.encyclopedia.decoy.txt` (Percolator
  output), `.encyclopedia.txt.model`, `first` / `final` `rt_fit`, `ms1_fit`, `ms2_fit` tables with
  PDFs, and `.encyclopedia.txt.log`. The log names the scoring as "EncyclopeDIA 2.X".
- **Reference numbers:** the wide-window run gave 31,833 peptides at 1% FDR against the GPF result
  library.

## Design choices

- **The jar's CLI is captured, not hand-copied.** A survey script dumps every program's `--help`,
  the SQLite DDL of every file the jar writes, and where each output lands. A parsed JSON of
  programs and flags is packaged; one generic runner validates flag names against it before
  starting Java. Typed parameters exist only where Constellation adds meaning (tolerance units,
  PTM toggles, search mode, budgets). A version bump becomes a diff.
- **Search mode is a toggle between two engines,** `--search-mode {peptide-centric,spectrum-centric}`:

  | `--search-mode` | Jar program | DIA data | DDA data |
  |---|---|---|---|
  | `peptide-centric` | the default program (the original EncyclopeDIA algorithm, Searle et al.); needs a library | yes, and the default | refused: there is no peptide-centric DDA mode |
  | `spectrum-centric` | ScribeTwo: `-scribetwo -searchType …`; library optional (predicts from the FASTA without one) | yes (`dia`) | yes, and the default (`dda`, or `dda_chimeric` for DDA+) |

  Under `spectrum-centric`, `--search-type {auto,dia,dda,dda-chimeric}` passes straight to
  `-searchType` (default `auto`, the jar's own). Constellation detects the acquisition type from
  scan metadata to pick the default mode and to refuse mismatches (peptide-centric on DDA; `dia`
  on DDA data). The original Scribe (`-scribe`, DIA-style scoring on spectrum-centric analysis) is
  not part of the toggle; it stays reachable through the generic `massspec encyclopedia run`. One
  resolver function holds the table; flags that belong to one engine error under the other, as
  `predict-library --backend` already does; the engine is read back from each output's
  `search type` metadata so both kinds of result can sit side by side. (The repo's 6.5.15 notes
  list `-scribetwodda` / `-scribetwodia`; both are gone in 6.9.23.)
- **`.dia` is written by streaming a Thermo bundle,** driven by `scan_metadata.parquet` (it has
  every scan, including empty ones) with peaks read in batches. The writer targets format 0.8.0
  and reproduces the single-file and merged-file conventions listed above, including the metadata
  keys the jar acts on; the reader accepts 0.5.0 and 0.8.0. The reference `.dia` files are the
  acceptance target.
- **Stable ids.** Acquisitions get experiment-scoped sequential ids, persisted once, with
  `source_sha256` as the cross-experiment identity; outputs bind to them through the `sourcename`
  Constellation writes into the `.dia`, which the jar copies into every result as `SourceFile`. Peptides, precursors and proteins get content-derived ids
  (xxh64 of a canonical ProForma key), opt-in, so parallel searches and library-less searches agree
  without a shared library.
- **One layout, discoverable by `kind`.** A standalone search is a one-acquisition experiment:

  ```
  <experiment>/manifest.json          kind=massspec_experiment
  <experiment>/acquisitions/  library/
  <experiment>/units/<key>/           manifest.json (kind=massspec_search), _SUCCESS, jar/ (.dia, .elib, sidecars, logs)
  <experiment>/tables/<table>/acquisition_id=<id>/part-0.parquet
  <experiment>/report/                kind=massspec_quant_report
  ```

- **Read/write bound.** Read: every table and column, through a schema-driven SQLite → Arrow
  mirror, plus the text sidecars. Write: the three things the jar consumes from Constellation:
  `.dia` (search input), `.dlib` (library), and `.elib` (quant-report input).

## Phases

**The command survey comes first.** Its headline facts are above (gathered while planning); P0
turns them into checked-in artefacts for every program and sub-program, so later phases and tests
read the jar's CLI from the repo instead of from memory. P1 and P3 need no jar and can run
alongside.

| Phase | Goal | Main files | Proves |
|---|---|---|---|
| **P0** Command survey and pin | Capture `-h` for the top level and every program and sub-program of 6.9.23; build the flag JSON; raise the floor; diff against the 6.5.15 notes | `scripts/{survey-encyclopedia.py*, build-encyclopedia-cli-json.py*}`; `constellation/data/_raw/encyclopedia/6.9.23/help/`*; `constellation/data/encyclopedia_cli.json`*; `thirdparty/{registry.py, encyclopedia.py, jvm.py}` (`ToolSpec.min_version`, doctor "too old", bundled JRE preferred for this tool); `search/encyclopedia/_common.py`; `scripts/install-encyclopedia.sh` (`--from-dir`); `docs/plans/encyclopedia-6.9.23-survey.md`* replacing the 6.5.15 notes | The full flag set per program is in the repo and tested against; the jar is installed off the Windows drive and resolved with its bundled Java; `doctor` flags a too-old jar. Open items for the PI: how `EncyclopediaTwo` is invoked, and whether the unlisted flags should be used |
| **P1** EncyclopeDIA spectra file (`.dia`) | Streaming reader (0.5.0 and 0.8.0), writer (0.8.0, single and merged), `.dia` → bundle import, structural diff; `massspec encyclopedia {export-spectra,import-spectra,compare-spectra}` | `massspec/io/encyclopedia/{_dia.py, _dia_write.py*, _dia_diff.py*, _codec.py, _schema.py}`; `massspec/schemas.py` + `readers/thermo/_read.py` (additive `scan_window_lower/upper`) | Each reference `.dia` → import → export → diff is clean, the merged one included; synthetic round trip runs in CI |
| **P2** First real run | With today's verbs, run end to end on the reference data; record the DDL and location of every output; settle the JVM flags | `constellation/data/_raw/encyclopedia/6.9.23/{schema,outputs}/`*; `thirdparty/jvm.py` | Re-running the reference search on the reference `.dia` reproduces it (and measures run-to-run noise). raw → bundle → Constellation `.dia` → the same search lands within that noise. Constellation `.dia` vs reference `.dia` by `encyclopedia compare-spectra` |
| **P3** Foundations | Split the 3,848-line CLI module (pure move); shared stage and manifest helpers; identity | `massspec/cli.py` → `massspec/cli/`; `core/{stages.py*, manifests.py*}`; `massspec/manifest.py*`; `massspec/acquisitions.py`; `massspec/library/ids.py*` | Suite unchanged after the split; canonical-key tests over every sample modseq |
| **P4** Single-acquisition path | New runner, mode resolver, mirror reader, typed manifest, experiment layout; `search` takes bundles / `.raw` / `.dia`; `process-dia` removed; `massspec encyclopedia {run,flags}` | `search/encyclopedia/{cli_spec.py*, runner.py*, params.py*, outputs.py*}`; `io/encyclopedia/{_mirror.py*, _project.py*}`; `massspec/workflows/encyclopedia.py*`; `search/schemas.py`, `quant/schemas.py`, `quant/targets.py` | Parquet row counts equal SQL counts; no orphan ids; `chromatogram extract` accepts the result as targets |
| **P5** Orchestration | Fan-out, resume, per-unit status, detach/follow; fraction merge; chromatogram-library step with the collision filter; quant report | `workflows/encyclopedia.py`; `core/detach.py*`; `_dia_write.py`; `cli/encyclopedia.py` | Kill + `--resume` reruns only unfinished units; chained standalone equals orchestrated |
| **P6** ScribeTwo | `--search-mode spectrum-centric` (until then that choice exits with "not yet available"): `-searchType` dia / dda / dda_chimeric, with or without a library; parse its `.dia.dlib` result, plus whatever reports the CLI turns out to write and the optional mzIdentML, into `PSM_TABLE`. Independent of P5 and can swap with it | `params.py`, `_project.py`, `search/schemas.py` | The reference DDA run re-run from the CLI with its recorded parameters lands near its 17,640 entries; ScribeTwo also runs on DIA; PSM scans join to scan metadata |
| **P7** Breadth and write fidelity | Chromatograms, fragment quants, RT alignment, sidecars (Percolator input/output, calibration fit tables), ion mobility; writer on container 0.1.18 (`decoyentries`, `IonMobility`, 22 indexes) | `_project.py`, `_write.py`, `_schema.py`, `io/encyclopedia/_sidecar.py*` | The jar searches a Constellation-written `.dlib` and accepts a rewritten `.elib` |

`*` = new file. Paths are under `constellation/` unless they start with `scripts/` or `docs/`.

Reused rather than rebuilt: `readers/thermo/_filter.parse_filter_string`, `thirdparty/jvm.run_jar`,
`_common.{build_manifest_envelope, available_memory_gib, sha256_file}`, `search/collision.py`,
`search/crosscheck.py`, `quant/targets.py`, the stage-key design in
`sequencing/assembly/pipeline.py:72-134` and the detach design in
`sequencing/basecall/dorado_run.py` (both re-implemented in `core`; `sequencing` is not migrated).

New tables (additive columns elsewhere): `PRECURSOR_SCORE_TABLE` with decoy flag (P4);
`PRECURSOR_CHROMATOGRAM`, `FRAGMENT_QUANT`, `RT_ALIGNMENT` (P7, after a real 6.9.23 output is seen).

Docs each phase: `massspec/CLAUDE.md`, root `CLAUDE.md`; P0 replaces
`docs/plans/encyclopedia-6.5.15-utilities.md` with the 6.9.23 survey note and adds one newly found
pipeline defect to the review note.

## Needed from the PI

- **Nothing more is needed to start any phase.** In place as of 2026-10-10: the install at
  `/mnt/d/EncyclopeDIA` (P0 copies it into `~/.constellation/encyclopedia/6.9.23/` so Java does
  not run off the Windows drive); the reference data, `UP000005640_9606.fasta` and its predicted
  library `…trypsin.z1-6_nce30_DIA.dlib` at `/mnt/e/261010_u2os`; the Thermo DLLs (`doctor` reports
  1.4.5).
- **Reference runs were made in the GUI,** so there are no command lines. P2 and P6 rebuild each
  run's parameters from the result library's metadata and pass them explicitly, since GUI and CLI
  defaults may differ.
- **A question, not a blocker:** which tool and version wrote the reference `.dia` files (their
  metadata says `conversion.outputType = EncyclopeDIA`).
- **Optional:** a second wide-window DIA replicate, so the orchestrated quant report in P5 has
  more than one injection to align.

## Verification

| Tier | Needs | Checks |
|---|---|---|
| CI | nothing | Flags vs snapshot JSON; `.dia` round trip on a generated bundle; reader on SQLite built from checked-in 6.9.23 DDL; resume with a fake jar |
| Samples | `$CONSTELLATION_ENCYCLOPEDIA_SAMPLES` (old 2.12.30 files) and the reference folder | `.dia` round trip for 0.5.0 and 0.8.0; row-count invariants on the reference `.elib` and ScribeTwo `.dlib` |
| Jar | 6.9.23 installed | Flag acceptance; re-survey equals snapshot |
| Jar + data | reference folder, FASTA, library | Reproduce the reference search; Constellation `.dia` vs reference `.dia` (`encyclopedia compare-spectra`, then search results within measured noise) |
| + Thermo DLLs | | The same starting from `.raw` |

Comparisons are on logical content sorted by natural key, never SQLite bytes; search results are
compared against the noise measured by running the same input twice.

## Main risks and the early check for each

- The jar rejects or misreads a Constellation `.dia` (a wrong or missing metadata key would change how it searches, silently): the P2 run, plus `encyclopedia compare-spectra` against the reference `.dia`, metadata included.
- Thermo centroids differ from the reference `.dia`'s peak lists: per-scan peak count and intensity sum on 100 scans.
- 6.9.23 changed something the frozen pipeline relies on (for example `.raw` passed straight to search): snapshot test over its four arg builders in P0, one direct-`.raw` search in P2; reported, not patched.
- The jar's commands drift again or differ from what help shows (the repo's notes were already wrong for ScribeTwo): P0 checks the full snapshot in, and tests compare every flag Constellation emits against it.
- ScribeTwo from the CLI behaves differently from the GUI run we have (reports, file names, defaults): run it once from the CLI at the start of P6, before sizing the parser work.
- Parallel jars collide (prediction cache, Percolator temp files, heap × jobs): two concurrent searches in P2; per-unit working and temp directories.
- Java: the jar wants 11–21 and conda has 25, so P0 switches this tool to its bundled Temurin 21; the old WSL2 JIT crash is checked on the first real run in P2.

## Out of scope

Changes to the transcriptome pipeline or Counter; mzML and Bruker readers; SLURM submission; the
mass-spec browser itself; migrating `sequencing` onto the new `core` helpers.
