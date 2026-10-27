# Thermo `.raw` reader: speed, centroiding, fidelity

## Status

- **Plan approved** by the PI on 2026-10-26. The approved original is
  `~/.claude/plans/effervescent-conjuring-raven.md`. Work is on `feat/thermo-reader`.
- **Decisions (PI, 2026-10-26):**
  - Profile data is centroided with Thermo's algorithm by default; `--profile` keeps the
    profile points; a Constellation centroider comes later (Stellar bench ledger #15).
  - Peaks become a directory of part files (`peaks/part-NNNNN.parquet`).
  - Thermo RawFileReader is upgraded to 8.0.42, Thermo's current .NET 8 build.
  - Bundles hold the instrument's exact values, not Thermo's display-rounded strings.
  - The profile path is checked on the Kirby profile Stellar runs (`/mnt/e/Kirby`).
- **Phase 1** (range reader, vectorized writer, Thermo centroider, fixes): done. Measured
  2026-10-26:
  - **GPF 900–1000** converts in 30–32 s (was 145 s).
    - Its bundle equals the old reader's in every peak (109 M) and every scan-metadata
      column, with the same row groups. The exceptions are the intended ones: `peak_count`,
      and the new `spectrum_type` / `centroider`.
    - Its `.dia` is still identical to MSRawJava's (`test_export_matches_a_reference_dia`).
  - **Stellar DDA, 1.8 GB:** 123 s through the CLI, hashing and exit included. The old
    reader crashed after 580 s.
  - **Orbitrap files.** The Q Exactive HF phospho run (profile MS1 with a centroid stream)
    and an Exploris run report `AGC PS Mode: -1`. That overflowed the unsigned column, so the
    old reader could not convert either. The column is now signed, and any negative value in
    an unsigned trailer column is stored as null.
  - **Old against new on scan ranges** of those files and the DDA: every peak equal, and
    every scan-metadata column apart from the intended ones.
  - **Kirby RF30 (profile ion trap):** every scan is centroided by `thermo`, at about 1,220
    peaks per MS2 scan.
    - 50 random scans equal a direct `Scan.ToCentroid`.
    - `--profile` points equal `GetSegmentedScanFromScanNumber`.
    - About 1 ms per scan of DLL work warm; reading cold from `/mnt/e` is latency-bound.
  - **Crashes seen on the old reader:** it died with signal 11 partway through two of the
    three larger files on this machine. The new reader has not crashed.
  - **One exit hang:** once, under memory pressure, a finished conversion spent 2.5 h at
    21 GB in pythonnet's shutdown (`pythonnet.unload` at exit). On a quiet machine that
    shutdown takes about 6 s.
- **Phase 2** (peak parts, `massspec/bundle.py`, workers): done. Measured 2026-10-26 on the
  workstation (i7-8700: 6 cores, 12 threads):
  - **What it does.**
    - The driver plans contiguous scan ranges: at least 2,000 scans, about 4 per worker.
      They run in spawned worker processes, one part per range.
    - Every worker is started, and opens the file, while the probe runs. CommonCore's first
      `GetScanEvents` call costs about 1.5 s in every process.
    - Workers skip pythonnet's CLR shutdown at exit.
    - `convert_batch` gives every file's ranges to one pool.
    - `--threads` defaults to the cores, up to 8. The search workflow converts a `.raw` with
      workers too.
  - **GPF 900–1000:** 145 s → 20–25 s with 4–8 workers (single process: 30–48 s; this
    machine's timings vary run to run).
    - Eight workers are no faster than four here: total CPU time triples at 8, the cost of
      hyperthreading.
    - The 8-worker bundle equals the old reader's in every peak (17 parts, 172 row groups
      against 167).
  - **Kirby RF30** (18.7 GB, profile ion trap, 236,340 scans): 134–147 s with 8 workers,
    476 s in one process.
    - The first attempt took 613 s. The cause was NumPy views over .NET arrays, which pin
      them: a block of pinned arrays made every .NET collection slower and more frequent, and
      on these scans Thermo's centroider allocates large arrays. Arrays are now copied out at
      once: 6.5 → 1.1 ms per scan.
    - **Per-scan agreement:** an 8-worker run equals the single-process run on every scan
      (count, sum of m/z, sum of intensity).
    - **Other runs differed in one or two scans** (±1–3 peaks). The cause is this
      workstation's memory, not the reader:
      - The local copy of the file hashed to `7e8a668a…` once and to the original's
        `41af474e…` three times later. `cmp` finds no differing byte.
      - A scan's profile points hashed differently at different times, and Thermo's
        centroider followed its input.
      - On the same input bytes it gives the same peaks in every process. That held with
        tiered compilation, dynamic PGO and hardware intrinsics each turned off.
      - The machine's earlier notes record the same kind of transient read errors.
    - **To check before relying on it:** a run on reliable hardware (OSC) comparing the
      worker run with a single-process run.
- **Phase 3** (exact values): done.
  - **What the bundle now holds:**
    - Isolation width comes from the scan event.
    - Typed trailer columns come from `GetTrailerExtraValues`, aligned with the trailer header.
      Which entries a scan has still comes from its own display list.
    - `trailer_extras` keeps Thermo's display strings. It used to re-format them: `100.000`
      became `100.0`.
    - The header (label, type, display precision) is in `acquisition_metadata.trailer_header`
      (v2).
  - **The `.dia` writer** rounds injection time and MS2 isolation width as Thermo displays them,
    for bundles that record the header.
    - The rounding is ties to even, as Thermo's is. The first try rounded half away from zero,
      and `diff_dia` caught 2 of 81,850 scans whose injection time was an exact tie (75.5625 ms
      shown as `75.562`).
  - **Parity:** GPF 900–1000 identical to MSRawJava. Stellar DDA identical in every per-scan
    value; its only differences are the two documented departures (every DDA window listed; one
    MS2 scan MSRawJava drops).
  - **Orbitrap runs:** monoisotopic m/z gains up to 0.05 mDa, injection times up to 0.0005 ms,
    and isolation widths are the instrument's float32 values. Peaks are unchanged.
- **Phases 4–5:** not started.

## Why

`massspec convert` was reported to be far slower than MSForest and to be missing features.
Measured on 2026-10-26 on the WSL workstation:

| File | Before | Notes |
|---|---|---|
| Stellar GPF DIA 900–1000: 1.02 GB, 81,850 scans, 109 M peaks | `convert` 145 s (under cProfile), plus 30 s for `write_dia` | MSRawJava 26.7.31 (MSForest's converter, bundled with EncyclopeDIA 6.9.23) writes the `.dia` in 101 s. Ours is value-identical to it (`diff_dia` reports nothing) |
| Kirby profile Stellar DIA: 18.7 GB, 236,340 scans | Unusable | Centroid mode wrote no peaks, because ion-trap profile scans have no centroid stream. `--profile` would push about 4.6 billion points through Python lists |

**Where the 145 s went:**
- About 65 s building per-peak Python lists and making Arrow arrays from them (`_PeakBuffer`).
- 20–28 s encoding one Parquet file.
- About 40 s on per-scan DLL calls and metadata, of which 16 s is the floor of a minimal loop.

Two avoidable costs sit inside that 40 s:
- `GetFilterForScanNumber().ToString()` takes 52 µs per scan, while `ScanStatistics.ScanType`
  holds the same string.
- The trailer's labels are marshalled again for every scan.

**Measurements behind the design:**
- Writing the same 109 M rows as 4–8 part files from parallel threads takes 4–5 s.
- .NET calls through pythonnet hold the GIL, so threads do not speed up reading; separate
  processes do, about 3.5× at 8.
- Reading the profile file: about 19,500 points per MS2 scan. `Scan.ToCentroid` turns one into
  about 1,200 peaks in about 1.1 ms.

**Defects found along the way:**
1. Ion-trap profile scans got no peaks in centroid mode, without a warning (bench ledger #13).
2. A symlinked `.raw` failed in `SelectInstrument` (ledger #12).
3. `peak_count` held `PacketCount`, which is not a peak count: 9,644 for a scan of 1,180 peaks.
4. The typed trailer columns held Thermo's display strings rounded to the header's precision.
   - The MS2 isolation width was 2.00 where the instrument used 2.0009, so every isolation
     window was 0.0009 Th too narrow.
   - Injection times were rounded to 3 decimals.
   - MSRawJava reads the same rounded values, which is why our `.dia` matched it.
5. No instrument method text: CommonCore 5.0.0.93 returns None on Linux; 8.0.6 returns it.
6. `acquisition_metadata.profile_mode` recorded the CLI flag, not what was written.

## Design

### A. Bundle layout v2
- **Peaks:** `peaks/part-NNNNN.parquet`.
  - The zero-padded names sort in scan order.
  - Each part is a contiguous scan range, with each scan's peaks in stored (ascending m/z) order.
  - Row groups are cut at the RT bin (`rt_bin_width_s`) or at 1 Mi rows.
  - Parts are written under a temporary name and renamed.
  - `MS_PEAK_TABLE` is unchanged.
- **Manifest:** `schema_version` 2.
  - `outputs.peaks = "peaks"`.
  - `parameters` gain `peaks` (`centroid` | `profile`), `centroider`, `workers`, and the DLL
    assembly version.
  - `read_manifest` accepts 1 and 2.
- **`massspec/bundle.py`** is the one definition of a bundle's peaks, and every consumer goes
  through it:
  - `peak_paths`: the v2 parts in order, or the legacy `peaks.parquet`.
  - `peaks_dataset`: built from that explicit file list, so it never picks up
    `scan_metadata.parquet` or `mz_index/`.
  - `iter_peak_batches`: batches in scan order across parts.
  - `has_peaks`.
- **Re-converting** (`convert(force=True)`, `import_dia(overwrite=True)`) clears the previous
  outputs first.

### B. Reading a scan range
- One `GetScanEvents(lo, hi)` per range. Then, per scan:
  - `GetScanStatsForScanNumber`: retention time from `StartTime`, the filter string from
    `ScanType`, TIC, base peak, centroid flag, mass range.
  - The trailer, with its column mapping cached per label set.
  - The peaks (C).
- Per-scan values collect into numpy arrays. Per-peak columns are `np.repeat` of them, and Arrow
  arrays are made from numpy, so no Python object exists per peak.
- A writer thread owns the Parquet writer. pyarrow releases the GIL, so encoding overlaps the
  next scans' DLL calls.

### C. How each scan's peaks are made
| Mode | Scan acquired as | Peaks | `centroider` |
|---|---|---|---|
| centroid (default) | centroid | the stored centroids | `instrument` |
| centroid | profile with a centroid stream (Orbitrap label data) | the stream, with resolution, noise and baseline | `instrument` |
| centroid | profile without a stream (ion trap) | `Scan.ToCentroid`, as ThermoRawFileParser and msconvert do | `thermo` |
| profile (`--profile`) | profile | the profile points | null |
| profile | centroid | the stored centroids | `instrument` |

- **Recorded per scan:** in `spectrum_type` (`centroid` | `profile`) and `centroider`, in
  `SCAN_METADATA_TABLE` v4.
- **Logged:** `convert` prints one line per MS level saying how its peaks were made.
- **No silent empty levels:** a level with scans but no peaks fails the conversion with a message.
- **Choosing a centroider:** `convert(centroider="thermo")` checks the name against a registry,
  which the Constellation centroider will join. There is no CLI flag until a second choice exists.
- **`profile_mode`** (acquisition metadata) means peaks *stored* as profile. So the `.dia` writer
  takes a Thermo-centroided profile acquisition and still refuses a `--profile` bundle.

### D. Exact values
- **From the scan event's reaction:** precursor target, isolation width and offset, collision
  energy and activation. Bounds are target + offset ± width / 2 at the exact width. The filter
  string stays the fallback.
- **Typed trailer columns** take typed values. Which keys a scan has comes from the scan's own
  label list. `trailer_extras` keeps Thermo's display strings verbatim.
- **`peak_count`** is the number of peaks written.
- **The trailer header** (label, type, display precision) is recorded in the acquisition
  metadata.
- **The `.dia` writer** reproduces MSRawJava's rounding explicitly from that header: injection
  time to 3 decimals, isolation width to its display precision. `diff_dia` against MSRawJava
  stays empty.

### E. Workers within a file
- **The parent:**
  - plans contiguous scan ranges (at least 5,000 scans, about 4 per worker, in scan order, so
    disk access stays near-sequential);
  - runs them in a spawn pool, where each worker loads the CLR once and writes one part per
    range, returning that range's scan-metadata table;
  - writes the scan and acquisition metadata, and the manifest last;
  - hashes the file in a thread meanwhile.
- **`convert_batch`** feeds every file's ranges to one pool.
- **The CLI:** `--threads N` means worker processes for one file or many. It defaults to
  `min(8, cpu_count)`; it was 1.

### F. RawFileReader 8.0.42 and instrument metadata
- **What the install script fetches:**
  - Thermo's three nupkgs (Data, RawFileReader, RandomAccessReaderPlugin) from
    `github.com/thermofisherlsms/RawFileReader`. They are pinned to commit
    `cd0a429a9894c13cc8bc0a5a10d54f5b7eb991bb`, because Thermo removes old versions from that
    repository.
  - Their dependencies, OpenMcdf 3.1.4 and CommandLineParser 2.9.1, from nuget.org.
  - Every file is SHA-256 pinned.
- **Fallback:** the ThermoRawFileParser 1.4.5 bundle (CommonCore 5.0.0.93) stays available
  behind a flag.
- **New acquisition metadata:** the instrument method text per device, sample information
  (vial, injection volume, method file, raw file name) and the run header (expected run time,
  spectra count, maximum integrated intensity, start time).
- **The `.dia` writer** then writes the same keys MSRawJava does: `thermo.instrument_method.*`,
  `thermo.sample.*`, `thermo.run.*`, `run.start_time_iso8601`, `run.tic_total`.
- **Known cost:** 8.0.6 formats filter strings 4–7× slower than 5.0 (140–210 µs per scan). The
  workers share that cost.
- **Values:** 8.0.6 gave identical peaks, statistics, filter strings and trailers to 5.0.0.93 on
  9,000 sampled scans of three files (Stellar DIA, Stellar DDA, Q Exactive HF with profile MS1).

## Phases
0. This document.
1. Range reader, vectorized writer, Thermo centroider and per-scan provenance, `peak_count`,
   symlinks, the empty-level guard, hashing in a thread. Still writes the single-file layout.
2. Layout v2, `massspec/bundle.py`, consumers, fixtures, workers, `convert_batch`'s shared pool,
   the CLI.
3. Exact values, and the `.dia` writer's explicit MSRawJava rounding.
4. RawFileReader 8.0.42, method text, sample and run metadata, and the matching `.dia` keys.
5. Documentation (`massspec/CLAUDE.md`, the root `CLAUDE.md`).

## Verification
- **Unit tests:** a fake raw source (statistics, events, trailer, segmented scan, centroid stream,
  a centroider seam) covers every peak path, both modes, exact isolation, trailer presence, the
  empty-level guard, range planning and part order, both layouts and both manifest versions.
- **Identity, old against new**, column by column: Stellar GPF 900–1000, Stellar DDA, the
  Q Exactive HF phospho run, and an Exploris run. Only the intended columns may differ.
- **`.dia` parity** against MSRawJava's output for GPF 900–1000 and DDA. `compare-spectra` must
  report nothing, and after phase 4 across every metadata key.
- **Kirby RF30:**
  - every scan's peaks are made by `thermo`, at about 1,200 per scan;
  - 50 scans checked against a direct `Scan.ToCentroid`;
  - `--profile` points on a subset checked against `GetSegmentedScanFromScanNumber`.
- **Timings:** before, after at 1/4/8 workers, and MSRawJava.
  - Targets for GPF 900–1000: at most 30 s with one worker and 10 s with eight.
  - Kirby is bounded by `/mnt/e` read speed.
- **Browser access** (`scripts/bench-ms-browser-access.py`), before and after, decides whether
  `mz` and `intensity` use `BYTE_STREAM_SPLIT`: 27 % smaller files at about the same write time.

## Not in this work
- A Constellation centroider (ledger #15), which goes in through the `centroider` seam.
- Per-peak charge and flags from the Orbitrap centroid stream.
- Moving the per-scan columns out of the peak table. Profile data pays for 15 of them on every
  point.
- The ion-trap scan rate as a column.
- Other devices (UV, pumps) and the status log.
- A lazy `torch` import in `core.io.schemas`, which costs each worker 1.25 s of its 1.8 s
  import.
