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
  - An actual search on a Constellation-written file was left to P2 (below).
  - 15 provenance metadata keys the reference carries (instrument-method text, sample vial,
    integrated-intensity summaries) are not written, because a bundle does not hold them. The
    search jar reads none of them.
- **P2 (first real run): done, 2026-10-11.** Details and numbers are in the survey note's
  "Running a search" section; the short version:
  - **The acceptance test passed.** The reference wide-window run, `.raw` → `massspec convert` →
    `massspec encyclopedia export-spectra` → `massspec search` with the reference run's
    parameters, identifies exactly the 31,833 precursors and 5,168 proteins of the GUI reference
    result, with identical retention times and quantities. Two repeat searches of the reference
    `.dia` itself gave 31,833 (identical) and 31,830 (11 precursors different), which is the
    run-to-run noise. For one GPF fraction, the jar's own direct read of the `.raw` and a search
    of the Constellation-written `.dia` produced the same Percolator input row for row.
  - **Search takes a `.raw` directly in 6.9.23** and writes no intermediate file, so the frozen
    pipeline's per-injection stage still works. The risk item about it is closed.
  - **The jar's default Percolator does not run under WSL2** (a 2017 static binary that needs
    the legacy vsyscall page; the kernel is booted with `vsyscall=none`). The search died after
    scoring the whole run. Now detected before the JVM starts
    (`search/encyclopedia/percolator.py`; `massspec search` refuses with the fixes; `doctor` has a
    `percolator` row). `--percolator-version` takes `v3-05` or the path of any Percolator
    executable. Not handled: the frozen pipeline, which calls the wrapper directly.
  - **The jar leaks a helper server and temp files.** `thirdparty/jvm.py` now starts every JVM in
    its own session, stops what is left of its process group afterwards, and gives each run a
    private `java.io.tmpdir`; `capture_jar` is the same for short calls (`-h`, argument probes).
  - **JVM flags settled:** bundled Temurin 21, `-Xmx` and headless only. None of the older
    project's WSL2 / XML flags was needed across six complete searches. Memory stayed under 4 GB.
  - **Two searches at once** run without colliding (about 1.7x slower each on 12 cores).
  - **Outputs recorded:** `data/_raw/encyclopedia/6.9.23/schema/{result_library,spectra_file}.sql`
    and `outputs/encyclopedia.json` (`scripts/survey-encyclopedia-outputs.py`), with a test that
    holds Constellation's schemas to them.
  - **New verb:** `massspec encyclopedia compare-results` (`io/encyclopedia/_result_diff.py`).
  - **A CLI defect fixed on the way:** the libstdc++ defence in `cli/__main__.py` took "the env"
    from `$CONDA_PREFIX` (the activated env) instead of the interpreter's own, so calling
    `constellation` by full path printed a false skew note, re-executed every command, and put
    another env's `lib` first for child processes.
  - **For P4:** the existing `.elib` ingest gives a partial and in places wrong Parquet copy of
    a 0.1.18 result (survey note, "What the existing ingest does"): its precursor-level
    intensity is not the jar's quantity, pyro-Glu peptides lose their peptide-level quantity,
    8.5% of stored fragment peaks (mostly intact-precursor ions) are dropped, and it took
    about four minutes. Also found: the jar spells N-terminal modifications differently in
    `entries` and `peptidequants`. The `.elib` is unaffected. **Decided with the PI
    (2026-10-11):** now, only the free parts: the quadratic loop is gone (37 s, output
    unchanged; a proteome-scale `.dlib` still cannot be ingested with fragments, because every
    row is built in memory, which is the P4 reader's job) and the reader reports
    what its copy leaves out (`EncyclopediaReadResult.caveats`, printed and written to
    `ingest.caveats`). The modified-sequence fix is P3's (the ids need it). Quantity and peak
    handling are fixed once in P4, and **`read_encyclopedia` is rebuilt on the P4 reader** so
    there is one reader; that changes the frozen pipeline's Parquet numbers, which the PI has
    said is fine (nobody depends on them).
  - **Not done:** `-libexport` on WSL2 (it runs Percolator too, selected by a different flag);
    the `vsyscall=emulate` kernel setting, which needs a WSL restart.
- **P3 (foundations): done, 2026-10-11.** Each part with tests:
  - **Modified sequences.** The EncyclopeDIA parser resolves terminal masses against the first
    residue, so both of the jar's N-terminal spellings give one peptide (pyro-Glu is now
    `[UNIMOD:28]-`), and `encyclopedia_residue_form` is the shared key for joining a result
    file's tables. In the old reader every peptide of run A now has its peptide-level quantity
    and the total equals the jar's exactly; the only other change to its output is
    `mz_theoretical` of the 188 pyro-Glu precursors' b ions, by under 1e-7 Da.
  - **Ids.** `massspec/library/ids.py`: canonical peptide / precursor / protein keys and
    content ids (`content-v1`), `with_content_ids(library)`. The hash is xxh64, as the plan
    said and as the read fingerprints of `sequencing` use: one hash for content-derived ids
    across Constellation (PI, 2026-10-11). It was first committed as BLAKE2b (standard library,
    no dependency) and switched before anything was written with it; the `ms` extra now
    installs `xxhash`. Checked over every modified sequence in the old sample files and the two
    6.9.23 reference results.
  - **Stage and manifest helpers.** `core/stages.py`, `core/manifests.py`, and the three
    mass-spec manifest kinds in `massspec/manifest.py` (field names fixed, nothing writes them
    until P4).
  - **Acquisition identity.** `ACQUISITION_TABLE` v2 (`source_sha256`),
    `Acquisitions.register` / `id_for_source`, `save_acquisitions` / `load_acquisitions`,
    `acquisition_record_from_bundle`. On the reference set the eight converted runs register in
    order and the run names inside the peptide-centric and ScribeTwo results bind to the right
    ones. A result of a *merged* spectra file names the merged file, which is no acquisition:
    P5 has to carry that binding in the unit's manifest.
  - **The CLI split.** `massspec/cli.py` (4,065 lines) is now the `massspec/cli/` package, one
    module per verb plus `_common.py`. A pure move, checked three ways: each of the 70
    definitions is byte-identical to the original, the whole `massspec` parser tree (help text,
    defaults, types, handlers) is identical before and after, and the test suite passes
    unchanged with no test edited. `__init__` re-exports the private names the frozen pipeline
    and the tests import.
- **Review of the foundations before P4 (Codex on `b1fe1ca`, 2026-10-11): three defects, all
  reproduced, each fixed with a regression test written first.**
  - **Acquisition registration could bind results to the wrong run.** A hash match ended
    validation before the other rows' file names were checked, so a copy of run 0 arriving under
    run 1's name was registered as run 0 while `id_for_source` resolved the name to run 1; and a
    renamed copy got the right id but lost its name, so results under it found nothing. Now
    every name of a record is checked against every run before anything is accepted, a run
    keeps its other names (`source_aliases`), and `with_source_alias` binds a name outright.
  - **SIGTERM to the wrapper left the JVM running.** It runs in its own session, and the default
    SIGTERM action kills Python without any cleanup. `jvm.cancellation_scope` now handles SIGTERM
    and SIGHUP while a JVM runs. The worker-thread case needed more than raising from the
    handler: CPython 3.12 marks a thread as finished when an exception from a signal handler
    lands in `Thread.join`, so the process exited with that thread's cleanup half done; the
    handler now waits for runs in other threads before unwinding. Checked with a fake Java
    (main thread and worker thread, SIGTERM and SIGHUP) and by cancelling a real search.
  - **`compare-results` merged distinct modification sites.** It used the residue-form join key
    as precursor identity, so N-terminal and lysine acetylation on a leading K were one
    precursor, and one overwrote the other when both occurred. Identity is now the peptide the
    `entries` spelling names; the join key is used only to find quantities, and a quantity row
    that fits two precursors is counted and reported. Every comparison in the survey note was
    re-run and is unchanged.
  - Also, mine: `massspec.library.ids` imports `xxhash` on first use, because the P4 reader will
    import it and the CLI loads the readers at start-up, which a base install must survive.
  - Also, found by the full test run: the help-snapshot test was flaky (about one capture in
    thirty). The jar's helper thread sometimes prints into the middle of another line, and the
    normalisation dropped the whole line. It now keeps what stood before the interruption; sixty
    consecutive captures of the affected program match the snapshot.
  - The review's two requirements for P4 are added to its acceptance criteria below.
- **P4 (single-acquisition path): done, 2026-10-12.**
  - **What `massspec search` is now.** `--input` is one acquisition (a `massspec convert`
    bundle, a Thermo `.raw`, or a single-run spectra file), `--output-dir` an *experiment
    directory*. `massspec/workflows/encyclopedia.py` registers the acquisition, writes the
    spectra file itself (once per acquisition, `spectra/<run>/<run>.dia`), runs the jar in the
    search's own directory on a link to it, reads the result, and writes
    `units/<unit>/{manifest.json, _SUCCESS, jar/, logs/}` and
    `tables/<table>/unit=<unit>/acquisition_id=<id>/part-0.parquet` under content-derived ids,
    with the experiment manifest at the root. `<unit>` is `<run>-<8 hex digits>`: every search
    has a directory of its own (below). The jar's files in `jar/` carry its own names
    (`<run>.dia.elib`, `<run>.dia.encyclopedia.txt`, …). Another search into the same
    experiment, of a new acquisition or of one searched before, adds a unit and a partition
    beside what is there, with the same ids for whatever they share. `--search-mode` defaults
    from the data; `spectrum-centric` is refused as not yet available (P6). `process-dia` is removed.
    Refused before the JVM starts: mzML / `.d` / `.wiff` input, a merged spectra file,
    peptide-centric search of DDA data, a missing library, an undocumented flag, a Percolator
    that cannot run here.
  - **The pieces.** `search/encyclopedia/runner.py` (any program, flags checked against the
    captured help; also `massspec encyclopedia run` and `flags`), `params.py` (the mode
    resolver and the typed settings), `outputs.py` (a finished run held to the captured file
    inventory), `io/encyclopedia/_mirror.py` (SQLite → Arrow for any table, streaming) and
    `_project.py` (the tables Constellation models). `read_encyclopedia` is a wrapper over the
    last two; the old annotator is deleted. New: `PRECURSOR_SCORE_TABLE` (decoys included),
    `PRECURSOR_QUANT.{rt_start, rt_end, n_quant_ions}`, `SearchManifest.sources`,
    `quant.targets.targets_from_experiment`.
  - **Acceptance, on the reference wide-window run through the new command**
    (`massspec search --input <bundle> --output-dir <experiment>`, the reference parameters):
    - Against the GUI reference result (`compare-results`): the same 31,833 precursors and
      5,168 proteins, retention times identical, quantities identical for all 26,130 quantified
      precursors.
    - Tables against the jar's result library, by SQL: every row count equal (31,833
      precursors, 1,181,711 fragment rows = stored peaks, 32,150 score rows with their 317
      decoys, 5,168 protein scores, 35,443 protein links).
    - Field level, with the ids recomputed independently from the `entries` spelling: all
      31,833 precursors agree with the jar on id, m/z, retention time, quantity, integration
      window, quantified-ion count, peak count and peak intensities; all 32,150 score rows on
      q-value and posterior error probability.
    - No orphan id in any of nine foreign-key directions; every quantity and score row under
      acquisition 0, which the unit manifest records as the binding of the run name the result
      uses; no caveats.
    - 688 s in all: 113 s writing the 2.3 GB spectra file (once; a later search of the
      acquisition skips it), 544 s in the jar, the rest reading the result and writing 25 MB of
      tables. `--resume` afterwards returns in 2.4 s.
    - These are the numbers of the final layout (below). The same run under the two earlier
      layouts gave the same result, in 617 s and 649 s (the jar's share was 489 to 544 s
      across the three).
  - **From a `.raw`, and a second acquisition.** `--input <the GPF 900–1000 .raw>` converted
    it into `<experiment>/bundles/`, searched it and wrote the tables in 212 s; the result is
    identical to the search from the already-converted bundle (3,600 precursors, 2,524
    proteins; retention times, scores and quantities equal). Searching the 800–900 fraction
    into the same directory added acquisition 1 (7,308 precursors) beside it, both passing the
    same field-level check; the 1,629 proteins the two share have the same id in both (the two
    fractions share no peptide, being disjoint m/z ranges). The opt-in reference test
    (`tests/test_encyclopedia_search_reference.py`) was rewritten for the new command and
    passes on the same fraction. (These three were run under the first layout; conversion,
    registration and the tables are the same code in the final one.)
  - **Binding is explicit.** The unit knows which acquisition it searched; it checks that the
    result names that run and no other, and records the pairing in its manifest. A result that
    names a different run is an error (tested), not rows filed under the wrong acquisition.
  - **`chromatogram extract --targets`** takes the experiment (while it holds one acquisition)
    or a unit directory: given the unit directory of the GPF 900–1000 fraction it extracted
    227,765 trace rows for all 3,600 identified precursors.
  - **The reader, checked against SQL on three real results** (survey note, "The reader that
    replaced it"): row counts and every precursor's quantity, retention time, integration
    window and ion count agree; every stored peak is a fragment row; 9.5 s for the result the
    old reader took 223 s over. The 2.5-million-entry predicted library reads without
    fragments in 190 s at 4.2 GB.
  - **Every search has its own directory (PI, 2026-10-12); what is not copied from the jar.**
    EncyclopeDIA writes its outputs beside its input and, by design, takes a result it finds
    there for a finished search: the result is its `_SUCCESS` flag, bound to a location, and
    settings are not compared. I met it as a surprise (a test of re-searching with different
    settings would not fail; the real jar, asked for 0.2 Da over a finished 0.35 Da search,
    logged "Previously found 3600 peptides" and exited 0 in 36 s). My first two layouts each
    kept half of that model: one unit per acquisition, emptied and searched again, so that a
    new search *destroyed* the earlier one where the jar would have *returned* it. The PI
    corrected both halves:
    - **The spectra file is linked, not rewritten.** It is written once per acquisition, and
      each search's `jar/` holds a link to it and, before the run, nothing else; the jar is
      run on the link with no `-o`, so it writes there under its own names and always sees a
      new search. Checked on the real jar first (survey note, "A result beside the input is
      the jar's completion marker"): it does not follow the symlink back, so nothing is
      written beside the target and a finished search sitting there is not seen.
    - **A unit is a search, not an acquisition.** Its key is the run name plus a digest of the
      spectra, library, FASTA, settings and jar version. The same acquisition searched with
      anything changed goes into another directory and another `unit=` partition of every
      table; nothing already there is touched. Only the very same search maps to the same
      directory, which is what `--resume` finds; without `--resume` that one search is run
      again in its own directory, emptied first. How the tables are written (`--no-fragments`)
      and how many threads the jar is given are not part of the digest.
    Consequence for readers: "the result for acquisition N" has to name the search once there
    is more than one. `read_experiment_table(..., unit=)`, `targets_from_experiment(..., unit=)`
    and a unit directory given to `chromatogram extract --targets` all do; the bare experiment
    is refused with the choices listed.
    On real data, through the command, into one experiment: the GPF 900–1000 fraction at
    0.35 Da (3,600 precursors, 91 s) and then the same acquisition at 0.2 Da (3,583, 58 s, no
    time spent on the spectra file). Two directories and two partitions; all 30 files of the
    first search unchanged afterwards; each result library records its own `-ptol`; both pass
    the field-level check against their own result; `--resume` finds either in 2.3 s; the
    experiment as chromatogram targets is refused with both listed, and each search's
    directory gives its own 3,600 or 3,583. A spectra file given where it lies, with a
    finished search beside it, is searched properly through an absolute link and nothing is
    written next to it.
    The link is a relative symlink, falling back to a hard link and then a copy where the
    filesystem has no symlinks. The settings a result records are still compared with the ones
    passed (a difference is a warning and a caveat in the manifest). The output inventory was
    re-captured from such a directory: the same 24 files as before, named `<input>.*`. The
    four legacy wrappers pass the caller's directory straight through, so for them the jar's
    marker remains the caller's to know (review note, "A result beside the input means the
    search is finished").
  - **For P6:** a ScribeTwo DDA result stores whole observed spectra (about 1,080 peaks per
    entry, 99.6% matching no fragment at 20 ppm), so its fragment table needs the result's own
    tolerance and probably a rule about unmatched peaks.
  - **`massspec summarize` (PI, 2026-10-17).** The other half of naming search directories
    `<run>-<digest>`: the name keeps searches apart and never invites an overwrite, and the
    summary says what each one is, rather than trying to make the name informative. Given an
    experiment it lists the acquisitions by name and every directory under `units/`, finished
    or not, with status, library, FASTA, settings in Constellation's spelling, what it found, and
    what sets it apart from the other searches of the same acquisition; given one search's
    directory, every setting the jar recorded, the flags given, the command line, inputs and
    caveats. `--json` for the data (`massspec.experiment.summarize`). To make failed and stopped
    searches describable, the search manifest is now written when a search starts (`running`,
    with the flags passed) as well as when it ends (`finished` / `failed` / `cancelled`, with the
    error, and with every setting the jar recorded in its result, defaults included), and the
    experiment's own manifest is written as soon as the acquisition is registered, so even an
    experiment whose only search failed can be read.
    On real data (one GPF fraction searched at 0.35 Da and at 0.2 Da into one experiment, and a
    third search at 0.25 Da stopped with SIGTERM after 25 s): the summary lists the two
    finished searches with their 3,600 and 3,583 precursors and the cancelled one with
    "stopped by a signal (exit status 143)" and its settings marked as given; each search's
    difference from the others is its precursor tolerance and nothing else (the jar's default
    `trypsin` and its recorded `Trypsin` are not counted as a difference); no JVM or helper
    server was left running after the cancel.
  - **Not done in P4:** `<experiment>/library/` (nothing writes a shared library yet; each
    acquisition's partition carries its own rows), text sidecars and the remaining result
    tables (P7), and anything across acquisitions (P5). For P5: `-libexport` pairs each spectra
    file with the `<file>.elib` beside it, which is what a unit's `jar/` now holds, but it has
    not been run on symlinked `.dia` files yet.
- **P5–P7: not started.**
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
  <experiment>/spectra/<run>/         <run>.dia, written once per acquisition (added in P4)
  <experiment>/units/<key>/           manifest.json (kind=massspec_search), _SUCCESS, jar/ (a link to the .dia, .dia.elib, sidecars), logs/
  <experiment>/tables/<table>/unit=<key>/acquisition_id=<id>/part-0.parquet   (the unit level added in P4)
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
| **P4** Single-acquisition path | New runner, mode resolver, mirror reader, typed manifest, experiment layout; `search` takes bundles / `.raw` / `.dia`; `process-dia` removed; `massspec encyclopedia {run,flags}` | `search/encyclopedia/{cli_spec.py*, runner.py*, params.py*, outputs.py*}`; `io/encyclopedia/{_mirror.py*, _project.py*}`; `massspec/workflows/encyclopedia.py*`; `search/schemas.py`, `quant/schemas.py`, `quant/targets.py` | Parquet row counts equal SQL counts; no orphan ids; `chromatogram extract` accepts the result as targets. **Added after review (2026-10-11):** field-level agreement, not only counts (each precursor's parsed quantity, retention time and score equal the jar's own values for that precursor); and every result bound to its acquisition explicitly, through a binding the unit's manifest records, never by matching names after the fact |
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
