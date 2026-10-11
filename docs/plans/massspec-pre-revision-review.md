# Mass spec, Counter and transcriptome→proteome: state before the revisions

## Status

Review snapshot taken 2026-10-10 at `9e42224`, ahead of planned major revisions to Counter and to
the transcriptome→proteome pipeline. It records what exists, how the pieces connect, and the
defects found. **The defects in §5 were deliberately left unfixed** (PI decision, 2026-10-10): both
subsystems are about to be reworked, so they are listed to be designed out rather than patched.

How it was checked: code reading throughout, plus the mass-spec, Counter and bridge test files run
locally (936 passed, 16 skipped). Nothing was run on real data, and the EncyclopeDIA jar was not
installed on the review machine, so every statement about the jar's behaviour comes from the code,
its comments and the test suite. Line numbers are as of `9e42224`.

Module-level detail lives in [`constellation/massspec/CLAUDE.md`](../../constellation/massspec/CLAUDE.md);
this note covers the seams between modules, which that file does not.

**Since this snapshot:** the version floor was raised from 6.5.15 to 6.9.23 and the jar's command
line is now tracked as data; see [`encyclopedia-6.9.23-survey.md`](encyclopedia-6.9.23-survey.md),
which replaces the 6.5.15 utilities notes this review cites. Statements below about "6.5.15" and
about what the jar accepts describe the state at `9e42224`.

## 1. What is in `massspec`

About 28k lines: nine subpackages and one CLI module.

| Subpackage | Lines | Contents |
|---|---|---|
| `io/` | 5,087 | EncyclopeDIA `.dlib` / `.elib` / `.dia`, NIST `.msp`, MaxQuant `combined/txt` |
| `counter/` | 3,922 | Ion-count model, 15 modules |
| `cli.py` | 3,848 | All nine `massspec` verbs: parsers and handlers |
| `search/` | 3,509 | `Search` container, EncyclopeDIA jar wrappers, novel-peptide classifier, collision filter |
| `quant/` | 2,593 | `Quant` container, XIC extraction, peak index, target adapters |
| `library/` | 2,572 | `Library` container, digest, Koina backend |
| `readers/` | 2,334 | Thermo `.raw` only |
| `peptide/` | 1,906 | m/z, fragment ladders, envelopes, neutral losses |
| `annotation/` | 959 | mzPAF, USI |
| `spectra/` | 533 | MS2 similarity, consensus, fragmentation noise model; nothing in the package consumes it yet |
| `tokenize/` | 20 | Docstring only |

**Containers:** `Acquisitions`, `Library`, `Quant`, `Search`, `CounterResult`. **Registered Arrow
schemas:** 26 (raw peaks and scan metadata, the library tier, the quant tier including
`XIC_TARGET_TABLE` / `XIC_TRACE_TABLE`, the search tier including `PSM_TABLE` and
`NOVEL_PEPTIDE_TABLE`, and four Counter tables).

**Formats:**

| Format | Direction | Produces |
|---|---|---|
| Thermo `.raw` | read, plus streaming convert | `MS_PEAK_TABLE` + `SCAN_METADATA_TABLE` + `ACQUISITION_METADATA_TABLE` bundle |
| EncyclopeDIA `.dia` | read | three private list-column schemas, **not** `MS_PEAK_TABLE` |
| EncyclopeDIA `.dlib` / `.elib` | read + write | `Library`, optional `Quant`, optional `Search` (no PSMs) |
| NIST `.msp` | read | `Library` (annotated entries only) |
| MaxQuant `combined/txt` | read | `Search` with `PSM_TABLE` |
| ParquetDir | read + write | all four containers |
| Koina | producer | `Library` |

No mzML, Bruker `.d`, mzSpecLib, mzIdentML or mzTab reader exists. Because `.dia` does not map to
`MS_PEAK_TABLE`, Thermo `.raw` is the only input that can feed XIC extraction and Counter.

## 2. Running EncyclopeDIA

### How the jar is reached

- **Discovery** ([`thirdparty/encyclopedia.py`](../../constellation/thirdparty/encyclopedia.py)):
  `$CONSTELLATION_ENCYCLOPEDIA_HOME`, then `~/.constellation/encyclopedia/current`, then
  `third_party/`. The highest `encyclopedia-*.jar` wins.
- **Version floor:** 6.5.15, enforced by `require_min_encyclopedia` in every handler and once at the
  top of the pipeline.
- **Install:** `scripts/install-encyclopedia.sh --installer <path>` works. The default (download)
  mode errors, because no public release at or above 6.5.15 has a URL in the script. A machine
  without the lab's installer cannot run any jar-backed command.
- **Java** ([`thirdparty/jvm.py`](../../constellation/thirdparty/jvm.py)): `java` on `PATH` is tried
  first and the JRE bundled with the jar is only a fallback. `environment.yml` installs `openjdk`,
  so inside the conda env the jar runs under conda's Java, not its own.

### Standalone verbs

Four jar programs are wrapped, each as a pure `build_*_args` function plus a `run_*` runner in
[`massspec/search/encyclopedia/`](../../constellation/massspec/search/encyclopedia/), and each
exposed as a `constellation massspec` verb:

| Verb | Jar invocation | After the jar |
|---|---|---|
| `predict-library` | `-convert -fastaToJChronologerLibrary` | ingests the `.dlib` to a `Library` ParquetDir |
| `process-dia` | `-convert -processDIA` | nothing (the `.dia` is for the jar) |
| `search` | default mode | locates the `.elib`, ingests to Library / Quant / Search ParquetDirs |
| `library-export` | `-libexport` | ingests the consolidated `.elib` |

Every handler writes a `manifest.json` (jar SHA, Java version, argv, tolerances) and `_SUCCESS`, and
honours `--resume`. `--encyclopedia-arg FLAG=VALUE` passes through any flag that is not wrapped.
`predict-library --backend koina` is the second predictor and can also write a `.dlib`. The other
jar programs (`-walnut`, `-thesaurus`, `-scribe*`, `-xcordia`, the rest of
`-convert`) are not wrapped; see the 6.5.15 utilities notes (since replaced by
[`encyclopedia-6.9.23-survey.md`](encyclopedia-6.9.23-survey.md)).

Two further verbs touch `.elib` files without the jar: `collision-filter` (rewrites an `.elib` at
the SQLite level) and `classify-novel-peptides`.

### Inside the pipeline

`constellation transcriptome-to-proteome` calls the `run_*` runners directly, not the handlers:

| Stage | Runner |
|---|---|
| 5 `predict_library` | `run_predict_library` |
| 6 `process_dia` | `run_process_dia` |
| 7 `gpf_search` | `run_library_search`, then the collision filter |
| 9 `per_injection` | `run_library_search` per file, thread pool |
| 10 `quant_report` | `run_library_export` |

Stages 0–4 and 8 do not use the jar.

### Where the two paths differ

- **Only the pipeline can produce a quant report unaided.** Stage 10 builds a flat directory holding
  each raw file (hard-linked, because the jar's Thermo reader fails through a symlink), its `.elib`
  renamed to `<raw>.elib`, and the Percolator / feature sidecars the search left beside the source
  raw ([`transcriptome_to_proteome.py:1436-1505`](../../constellation/transcriptome_to_proteome.py#L1436-L1505)).
  The standalone `library-export` passes `--search-dir` straight to the jar and does none of this,
  and the standalone `search` additionally redirects its report with `-o` to a stem-named file in
  the output directory. Chaining standalone `search` × N into `library-export` therefore very
  likely needs that arrangement done by hand. Not verified against the jar.
- **`library-export --fasta` is optional in the CLI but required by the jar.** The parser's help
  calls it "not used by the default ELIB-to-ELIB pathway"; the wrapper's own comment and the
  pipeline say 6.5.15 requires `-f`.
- **The pipeline cannot start from a FASTA.** `--demux-dir` is required and there is no stage
  selection, only `--resume`. A DIA experiment with no transcriptome has no orchestrated path:
  `predict-library` → `process-dia` → `search` → `collision-filter` → `search` per injection →
  `library-export`, chained by hand.
- **The pipeline exposes fewer knobs.** Stage 5 passes only the PTM toggles (charge range, m/z
  range and NCE are the jar's defaults). Stages 7 and 9 pass tolerances, Percolator version and
  threshold, and threads. `--encyclopedia-arg` reaches the rest but is applied to every jar stage,
  so a flag meant for one stage is sent to all of them.
- **The pipeline only uses the EncyclopeDIA predictor.** There is no `--backend`.
- **Defaults differ.** Standalone `search` leaves tolerances and Percolator version to the jar
  (10 ppm, `v3-01`); the pipeline sets 10 ppm explicitly and `v3-05`.
- **Bookkeeping is duplicated.** The pipeline writes its own lighter stage manifests
  (`_write_stage_manifest`) and has its own ingest helper (`_maybe_auto_ingest_elib`) rather than
  sharing the handlers' `build_manifest_envelope` and ingest code.

### What is tested against a real jar

- **Arg builders:** covered for all four programs, no JVM needed.
- **Real-jar tests:** six, gated on an installed jar (three for `predict-library`, three for search
  flag acceptance). None runs `process-dia`, `library-export`, or a search that produces an `.elib`.
  The test data directory holds no spectra fixture.
- **Pipeline:** the end-to-end tests replace all five runners (four jar, one mmseqs2) with fakes, so
  they exercise stage wiring, naming and resume, not the jar.

The evidence that the pipeline runs for real is the lab's own runs and the fixes they produced
(`--fasta` required by search, the single-input `.dia` relocation, hard-link instead of symlink).

## 3. Transcriptome→proteome: where the code lives

| Location | Lines | Role |
|---|---|---|
| [`transcriptome_to_proteome.py`](../../constellation/transcriptome_to_proteome.py) | 1,811 | FASTA dedup / filter / combine helpers and the 10-stage orchestrator |
| [`cli/transcriptome_to_proteome.py`](../../constellation/cli/transcriptome_to_proteome.py) | 584 | Parser, reference-handle resolution, spectra-input expansion |
| [`massspec/search/novel.py`](../../constellation/massspec/search/novel.py), its CLI, `NOVEL_PEPTIDE_TABLE` | ~1,530 | CIGAR-walking classifier, gene maps from FASTA headers, a FASTA reader |
| [`massspec/search/collision.py`](../../constellation/massspec/search/collision.py), its CLI | ~630 | MS-only logic, written for this pipeline |
| [`sequencing/quant/protein_counts.py`](../../constellation/sequencing/quant/protein_counts.py) | 352 | Stage 1 counts and TPM |
| `core/io/schemas.py`, `thirdparty/mmseqs2_run.py` | — | `ALIGNMENT_HIT_TABLE`, mmseqs2 runner |

`massspec` does not import `sequencing`; the only mentions are in docstrings. About 8% of `massspec`
exists for this pipeline. The coupling is by contract:

- `NOVEL_PEPTIDE_TABLE` carries `transcript_id`, `n_transcripts_supporting`, `supporting_transcripts`.
- `classify-novel-peptides --protein-counts` expects a `tpm` column.
- `build_gene_map_from_fasta_headers` (in `massspec`) parses the `[gene=…]` tags that
  `build_combined_fasta` (in the bridge) writes.

Layering loose ends in the bridge:

- It imports two private helpers from `massspec.cli`: `_resolve_tolerance_args`
  ([`transcriptome_to_proteome.py:1698`](../../constellation/transcriptome_to_proteome.py#L1698))
  and `_add_tolerance_args` ([`cli/transcriptome_to_proteome.py:36`](../../constellation/cli/transcriptome_to_proteome.py#L36)).
- `run_transcriptome_to_proteomics` takes an `argparse.Namespace`, so it is not callable from a
  notebook without building one.
- There are three FASTA parsers: the bridge's `_iter_fasta`, `sequencing.readers.fastx`, and
  `read_fasta_proteins` in `novel.py`. The `novel.py` readers do not handle gzip.

## 4. Counter and how it meets the rest

Counter imports only `core.stats`, `core.optim`, `core.sequence.proforma` and
`massspec.peptide.envelope`. Its only consumer is `massspec/cli.py`. It is MS1-only.

The two data paths do not meet:

- **Counter:** `massspec convert` → `chromatogram extract` → `counter calibrate` →
  `counter estimate` → `COUNTER_N_TABLE`.
- **Pipeline:** raw file → EncyclopeDIA jar → `.elib` → `-libexport` quant report.

The pipeline never calls `convert`, `chromatogram` or `counter`. `COUNTER_N_TABLE` is not reduced to
`PRECURSOR_QUANT` or attached to a `Quant`.

The intended seam is `XIC_TARGET_TABLE`, via [`quant/targets.py`](../../constellation/massspec/quant/targets.py).
Starting from the ParquetDir bundles the pipeline ingests from an `.elib`, only one of the three
adapters works from the CLI (see defects 3 and 4).

## 5. Known defects, left unfixed

Each is marked **run** (reproduced) or **read** (from the code only).

1. **Stage 8 drops the abundance tie-break** (read). The orchestrator has `counts_tpm` in scope but
   calls `classify_novel_peptides` without `protein_abundance`
   ([`transcriptome_to_proteome.py:1363`](../../constellation/transcriptome_to_proteome.py#L1363)).
   For a peptide that maps to several ORFs, the representative `transcript_id` is iteration order;
   the standalone command with `--protein-counts` picks the most abundant ORF.
2. **`counter calibrate --trace` fails on a `chromatogram extract` bundle directory** (run:
   `pq.read_table` on a directory holding `xic_trace.parquet` + `manifest.json` raises
   `ArrowInvalid`). It uses `pq.read_table`
   ([`massspec/cli.py:3805`](../../constellation/massspec/cli.py#L3805)) where `estimate` uses
   `load_xic`.
3. **A Search bundle ingested from an `.elib` yields zero XIC targets** (read). `targets_from_search`
   reads only `psms`; the EncyclopeDIA reader leaves `psms` empty and the ParquetDir writer always
   writes the empty `psms.parquet`, so `load_targets` picks that path and returns nothing, silently.
4. **A Quant bundle cannot be used as XIC targets from the CLI** (read). `load_targets` needs
   `library=` for that source and `chromatogram extract` has no `--library` flag
   ([`massspec/cli.py:3270`](../../constellation/massspec/cli.py#L3270)), so it raises. This is the
   observed-RT source, the one a Counter run on pipeline output would want.
5. **`modified_sequence` does not join across outputs** (read). Stage 8 and `_load_detected_peptides`
   read `PeptideModSeq` from SQLite directly, so `NOVEL_PEPTIDE_TABLE` carries EncyclopeDIA
   mass-delta strings, while the ingested `peptides.parquet` carries the UNIMOD-resolved form from
   `parse_encyclopedia_modseq`.
6. **Stage 8 reads the raw reference FASTA**
   ([`transcriptome_to_proteome.py:1332`](../../constellation/transcriptome_to_proteome.py#L1332)),
   against the comment at line 903 that later stages must use the deduplicated one (read). It looks
   equivalent in effect, since deduplication is by sequence.
7. **Counter's batch driver is not importable API** (read). Co-fit partitioning and the worker pool
   live in private CLI functions
   ([`massspec/cli.py:3432-3693`](../../constellation/massspec/cli.py#L3432-L3693)).
8. **`library-export --fasta` optional vs required** — see §2.
9. **A pipeline run without `--protein-fasta` fails after all ten stages** (found 2026-10-10,
   after this review; the helper call was run, the end-to-end consequence was read).
   `_input_meta(None)` raises `AttributeError`, and the orchestrator passes `protein_fasta`
   to it unguarded when writing the top-level manifest
   ([`transcriptome_to_proteome.py:1552`](../../constellation/transcriptome_to_proteome.py#L1552));
   that value is `None` unless the flag is given. The mocked end-to-end test always passes the
   flag, so it does not see this.
10. **The pipeline cannot finish a search under WSL2** (found 2026-10-11; the standalone search
    was run, the pipeline consequence was read). EncyclopeDIA's default Linux Percolator (3.01)
    segfaults on a kernel booted with `vsyscall=none`, which is how WSL2 boots, after the run has
    been scored. `massspec search` now checks for this before starting the jar; the orchestrator
    calls `run_library_search` directly, with no `percolator_version` argument, so it would get as
    far as its first search and fail there with `Percolator exited with non-zero status: 139`.
    Native Linux hosts with the usual kernel settings (OSC) are not affected.
11. **The `.elib` → ParquetDir ingest gives a partial and in places wrong copy of a search
    result** (run, 2026-10-11, on a 6.9.23 result with 31,830 precursors; numbers and method in
    the survey note, "What the existing ingest does with a 6.9.23 result"). The `.elib` itself
    is untouched. In the Parquet copy that `read_encyclopedia` produces:
    - `precursor_quant.intensity` is the sum of all stored peak intensities, not the jar's
      `TotalIntensity` (median 2x it, 5th to 95th percentile 1.02x to 219x), and the 5,608
      precursors the jar could not quantify get a non-zero value;
    - the 188 pyro-Glu peptides have no `peptide_quant` row, because the jar spells N-terminal
      modifications differently in `entries` and `peptidequants` and `parse_encyclopedia_modseq`
      maps the two spellings of `-17.026549` to different strings;
    - 8.5% of stored fragment peaks are dropped: mostly intact-precursor ions (6.7% of peaks,
      10.4% of stored intensity), plus b/y ions at charge 3 or more (1.6%);
    - quantified-ion flags, correlations, integration boundaries, chromatograms and decoy score
      rows are not carried;
    - it took 223 s, 99 s of it in a quadratic protein-peptide linking loop (which would not
      finish at all on a proteome-scale predicted library).
    Both the standalone handlers and the pipeline's `_maybe_auto_ingest_elib` go through this.
    **Partly addressed 2026-10-11:** the loop is replaced (same output, 37 s), and the reader now
    reports the first three points as `caveats` with counts, which the standalone commands print
    and write to their manifests (the pipeline's helper does not show them). The numbers are
    unchanged until the phase 4 reader of the EncyclopeDIA integration replaces this path.

## 6. Stale docstrings, also left as they are

- `massspec/__init__.py` lists `library/`, `chromatogram`, `scoring` and `search/` as TODO scaffolds.
- `massspec/library/io.py` says the `.dlib` reader and writer are `NotImplementedError` stubs.
- `massspec/search/collision.py` says the filter is opt-in and wired only behind
  `classify-novel-peptides`; it has its own verb and is on by default in the pipeline.
- The `NOVEL_PEPTIDE_TABLE` header comment says one row per peptide sequence; the classifier emits
  one per `(peptide_sequence, modified_sequence)`.
