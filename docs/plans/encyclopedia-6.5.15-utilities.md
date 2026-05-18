# EncyclopeDIA 6.5.15 — utility inventory

Authoritative catalogue of every jar entry point exposed by EncyclopeDIA 6.5.15, with the flag table per utility and a "what changed since 2.12.30" diff. Each Python wrapper under [constellation/massspec/search/encyclopedia/](../../constellation/massspec/search/encyclopedia/) translates a typed kwarg surface to the flag table for one utility; new utilities land as new wrapper files PR by PR.

Status: **stub** — populated by PR 1 (which surveys the jar end-to-end) and extended by each follow-on PR. Until then the high-level invocation summaries below are the contract PR 0's CLI parsers and stub wrappers anchor on.

## How to (re)survey

Local dev install at `~/.constellation/encyclopedia/6.5.15-dev/` (the install4j installer was run with `-q -dir`).

```bash
# Top-level help — lists all -<program> entry points
~/.constellation/encyclopedia/6.5.15-dev/jre/bin/java \
  -jar ~/.constellation/encyclopedia/6.5.15-dev/encyclopedia-6.5.15.jar --help

# Per-program help
~/.../jre/bin/java -jar ~/.../encyclopedia-6.5.15.jar <program> -h
# e.g. -walnut, -convert, -libexport, -scribetwodda, ...

# Nested -convert subcommands
~/.../jre/bin/java -jar ~/.../encyclopedia-6.5.15.jar -convert -<subprogram> -h
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

*Pending PR 1.*

### `-convert -fastaToJChronologerLibrary`

*Pending PR 2 — full flag table captured during the survey, including the per-PTM `-ptm<Name>` enum surface.*

### `-convert -processDIA`

*Pending PR 3.*

### `-libexport`

*Pending PR 4 — default mode only; `-libexport -pecan` / `-xcordia` / `-phospho` deferred unless a use case appears.*

## v2.12.30 compatibility notes

The publicly-available release (`scripts/install-encyclopedia.sh` pins this) is hand-rolled CLI (not picocli), with a different default-CLI shape — single-program-per-invocation, no `-program` selector mode. The wrappers warn (not error) when `find("encyclopedia").version` falls outside `SUPPORTED_VERSIONS = {"2.12.30", "6.5.15"}`. For v2.12.30 calls, drop the `-convert` prefix (sub-utilities became their own top-level flags in 6.5.15 only) and expect a smaller flag surface; the `--encyclopedia-arg FLAG=VALUE` CLI escape hatch is the path of least resistance until per-version dispatch lands.

## When 6.5.15 ships publicly

Bump `ENCYCLOPEDIA_VERSION` in [constellation/thirdparty/encyclopedia.py](../../constellation/thirdparty/encyclopedia.py) to `6.5.15`, set `artifact` to `"encyclopedia-6.5.15.jar"`, and extend [scripts/install-encyclopedia.sh](../../scripts/install-encyclopedia.sh) with a v6.5.15 install path (download the install4j `.sh`, run `-q -dir third_party/encyclopedia/6.5.15/`, repoint the `current` symlink). Wrapper code is unaffected — the runner is artifact-shape-agnostic.
