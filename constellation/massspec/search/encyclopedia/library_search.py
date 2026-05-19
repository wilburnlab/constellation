"""Wrapper for EncyclopeDIA library search (default-mode invocation).

Maps to the jar's default entry point::

    java -jar encyclopedia-<ver>.jar -i <mzml|dia> -l <library.elib> [...]

EncyclopeDIA's default search writes its outputs in two places:

  * The chromatogram ``.elib`` (the main result — what downstream
    ``libexport`` / individual-sample searches consume) is written
    alongside the input file as ``<input>.elib``.
  * The ``-o`` flag controls the ``.encyclopedia.txt`` *report* file —
    a human-readable peptide-level summary — defaulting to
    ``<input>.encyclopedia.txt`` when omitted.

This wrapper handles the side-effect-write quirk by running the jar
with ``cwd=output_dir`` and locating the produced ``.elib`` after the
run via the CLI handler in ``massspec.cli``.

Full flag surface lives in
[docs/plans/encyclopedia-6.5.15-utilities.md]. PR 1 (this commit)
wraps the headline params + the common Percolator / tolerance knobs;
the escape-hatch ``--encyclopedia-arg FLAG=VALUE`` passthrough covers
everything else without code changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from constellation.thirdparty.jvm import JvmResult, run_jar


def build_library_search_args(
    *,
    input_file: Path,
    library: Path,
    fasta: Path | None = None,
    report_output: Path | None = None,
    fragment_tolerance_ppm: float | None = None,
    precursor_tolerance_ppm: float | None = None,
    acquisition: str | None = None,
    enzyme: str | None = None,
    fragmentation: str | None = None,
    percolator_version: str | None = None,
    percolator_threshold: float | None = None,
    percolator_protein_threshold: float | None = None,
    threads: int | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Translate typed kwargs to the EncyclopeDIA CLI argv (no JVM call).

    Only emits flags the caller supplied — anything left as ``None`` is
    omitted from the command line so EncyclopeDIA's built-in defaults
    apply. Pure function — exists so the Tier A test can exercise the
    flag layout without spawning Java.
    """
    args: list[str] = [
        "-i",
        str(input_file),
        "-l",
        str(library),
    ]
    if fasta is not None:
        args.extend(["-f", str(fasta)])
    if report_output is not None:
        args.extend(["-o", str(report_output)])
    if fragment_tolerance_ppm is not None:
        args.extend(["-ftol", str(float(fragment_tolerance_ppm)), "-ftolunits", "ppm"])
    if precursor_tolerance_ppm is not None:
        args.extend(["-ptol", str(float(precursor_tolerance_ppm)), "-ptolunits", "ppm"])
    if acquisition is not None:
        args.extend(["-acquisition", str(acquisition)])
    if enzyme is not None:
        args.extend(["-enzyme", str(enzyme)])
    if fragmentation is not None:
        args.extend(["-frag", str(fragmentation)])
    if percolator_version is not None:
        args.extend(["-percolatorVersion", str(percolator_version)])
    if percolator_threshold is not None:
        args.extend(["-percolatorThreshold", str(float(percolator_threshold))])
    if percolator_protein_threshold is not None:
        args.extend(
            ["-percolatorProteinThreshold", str(float(percolator_protein_threshold))]
        )
    if threads is not None:
        args.extend(["-numberOfThreadsUsed", str(int(threads))])
    args.extend(str(a) for a in extra_args)
    return args


def run_library_search(
    *,
    input_file: Path,
    library: Path,
    fasta: Path | None = None,
    report_output: Path | None = None,
    output_dir: Path,
    fragment_tolerance_ppm: float | None = None,
    precursor_tolerance_ppm: float | None = None,
    acquisition: str | None = None,
    enzyme: str | None = None,
    fragmentation: str | None = None,
    percolator_version: str | None = None,
    percolator_threshold: float | None = None,
    percolator_protein_threshold: float | None = None,
    threads: int | None = None,
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    extra_args: Sequence[str] = (),
    extra_jvm_args: Sequence[str] = (),
    stream_to_stderr: bool = True,
) -> JvmResult:
    """Run an EncyclopeDIA DIA library search.

    Required: ``input_file`` (.mzML / .dia / .raw / .d), ``library``
    (.dlib chromatogram-free or .elib chromatogram-library). Optional
    ``fasta`` is the background proteome — required by some scoring
    pathways but not by the default chromatogram-library search when
    the library already carries decoys.

    The chromatogram ``.elib`` output is written alongside the input
    file by EncyclopeDIA's convention. Locating it post-run is the
    caller's responsibility (the CLI handler in ``massspec.cli``
    does it).

    The jar is run with ``cwd=output_dir`` so any incidental files it
    drops in the working directory land in the run-dir rather than
    polluting the user's shell cwd. Stderr / stdout stream to
    ``<output_dir>/logs/``.
    """
    args = build_library_search_args(
        input_file=input_file,
        library=library,
        fasta=fasta,
        report_output=report_output,
        fragment_tolerance_ppm=fragment_tolerance_ppm,
        precursor_tolerance_ppm=precursor_tolerance_ppm,
        acquisition=acquisition,
        enzyme=enzyme,
        fragmentation=fragmentation,
        percolator_version=percolator_version,
        percolator_threshold=percolator_threshold,
        percolator_protein_threshold=percolator_protein_threshold,
        threads=threads,
        extra_args=extra_args,
    )
    return run_jar(
        "encyclopedia",
        args=args,
        jvm_heap_max=jvm_heap_max,
        jvm_heap_min=jvm_heap_min,
        jvm_tmpdir=jvm_tmpdir,
        extra_jvm_args=extra_jvm_args,
        log_dir=output_dir / "logs",
        stream_to_stderr=stream_to_stderr,
        cwd=output_dir,
    )


def find_search_elib(input_file: Path) -> Path | None:
    """Locate the chromatogram ``.elib`` EncyclopeDIA's default search
    produces.

    Convention: EncyclopeDIA writes ``<input>.elib`` next to the input
    file. Returns the path if it exists, ``None`` otherwise so the
    caller can report a clear error.
    """
    candidate = input_file.parent / f"{input_file.name}.elib"
    if candidate.is_file():
        return candidate
    # Fallback: try stripping any extension and tacking on `.elib`
    stem_candidate = input_file.parent / f"{input_file.stem}.elib"
    if stem_candidate.is_file():
        return stem_candidate
    return None


__all__ = [
    "build_library_search_args",
    "find_search_elib",
    "run_library_search",
]
