"""Read-group harmonization for multi-flow-cell Dorado BAMs.

Genomic ONT runs span several flow cells; merging their BAMs yields a
file with multiple ``@RG`` read groups. ``dorado polish`` resolves its
polishing model from the ``@RG`` ``DS`` (``basecall_model=``) tag and
**rejects a BAM whose read groups name more than one basecaller model**.
The fix (applied EARLY — before FASTQ conversion, assembly, and polish):

1. read every input BAM's ``@RG`` headers and extract the basecaller
   model from each ``DS`` field,
2. validate that they all agree (error otherwise, unless overridden),
3. merge the inputs and collapse every read onto ONE unified ``@RG``
   that preserves the model ``DS`` — so the harmonized BAM feeds both
   ``samtools fastq`` (→ hifiasm) and ``dorado aligner`` / ``dorado
   polish`` with a single, model-tagged read group.

The header inspection + validation are pure functions (testable against
synthetic ``@RG`` dicts); the merge/collapse shells out to samtools.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from constellation.core.progress import ProgressCallback, ProgressEvent
from constellation.thirdparty.registry import ToolNotFoundError, find


_SAMTOOLS_HINT = (
    "samtools not found; install via bioconda "
    "(`conda install -c bioconda samtools`) or set $CONSTELLATION_SAMTOOLS_HOME"
)

DEFAULT_UNIFIED_RG = "constellation_unified"


# ──────────────────────────────────────────────────────────────────────
# Pure header inspection
# ──────────────────────────────────────────────────────────────────────


def extract_basecall_model(ds: str | None) -> str | None:
    """Pull the ``basecall_model=`` value out of a Dorado ``@RG`` ``DS``.

    Dorado writes ``DS`` as space-separated ``key=value`` tokens, e.g.
    ``"basecall_model=dna_r10.4.1_e8.2_400bps_sup@v5.0.0 runid=abc..."``.
    Returns ``None`` when ``ds`` is empty or carries no model token.
    """
    if not ds:
        return None
    for token in ds.split():
        if token.startswith("basecall_model="):
            return token[len("basecall_model=") :] or None
    return None


def _models_from_header(header: Mapping) -> set[str]:
    """Set of basecaller models named across a BAM header's ``@RG`` lines."""
    models: set[str] = set()
    for rg in header.get("RG", []):
        model = extract_basecall_model(rg.get("DS"))
        if model is not None:
            models.add(model)
    return models


def read_basecaller_models(bam_paths: Iterable[Path]) -> dict[Path, set[str]]:
    """Map each input BAM/SAM → the set of basecaller models in its ``@RG``.

    Header-only read (cheap, no record decode) via pysam — the sanctioned
    BAM dependency. An input with no ``@RG`` (or no model token) maps to an
    empty set.
    """
    import pysam  # type: ignore[import-not-found]

    out: dict[Path, set[str]] = {}
    for raw in bam_paths:
        path = Path(raw)
        mode = "r" if path.suffix.lower() == ".sam" else "rb"
        with pysam.AlignmentFile(str(path), mode, check_sq=False) as fh:
            header = fh.header.to_dict()
        out[path] = _models_from_header(header)
    return out


def validate_single_model(
    models: Mapping[Path, set[str]],
    *,
    allow_multi: bool = False,
) -> str | None:
    """Collapse per-file model sets to the single shared model.

    Returns the one model string, or ``None`` when no input named a model
    (harmonization still proceeds — but ``dorado polish`` will need a
    model supplied another way). Raises ``ValueError`` naming the
    divergent models + their files when more than one distinct model is
    present and ``allow_multi`` is false.
    """
    distinct: set[str] = set()
    for model_set in models.values():
        distinct |= model_set
    if len(distinct) <= 1:
        return next(iter(distinct)) if distinct else None
    if allow_multi:
        return sorted(distinct)[0]
    lines = [
        f"  {path}: {', '.join(sorted(model_set))}"
        for path, model_set in models.items()
        if model_set
    ]
    raise ValueError(
        "multiple basecaller models across inputs — `dorado polish` requires "
        "exactly one:\n"
        + "\n".join(lines)
        + "\nre-basecall the flow cells uniformly, or pass --allow-multi-model "
        "to override (polish will then need an explicit model)."
    )


# ──────────────────────────────────────────────────────────────────────
# Merge + collapse (samtools)
# ──────────────────────────────────────────────────────────────────────


def _resolve_samtools() -> Path:
    try:
        return find("samtools").path
    except ToolNotFoundError as exc:
        raise FileNotFoundError(_SAMTOOLS_HINT) from exc


def _emit(progress_cb: ProgressCallback | None, stage: str, message: str) -> None:
    if progress_cb is not None:
        progress_cb(ProgressEvent(kind="stage_progress", stage=stage, message=message))


def harmonize_read_group(
    in_bams: Sequence[Path],
    out_bam: Path,
    *,
    unified_rg_id: str = DEFAULT_UNIFIED_RG,
    model_ds: str | None = None,
    threads: int = 8,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Concatenate inputs and collapse them onto a single model-tagged ``@RG``.

    Steps: ``samtools cat`` (concatenate the unaligned Dorado BAMs — no
    coordinate-sort requirement, unlike ``merge``) → ``samtools
    addreplacerg`` (every read's ``RG:Z`` becomes ``unified_rg_id``) →
    header rewrite dropping the now-orphaned source ``@RG`` lines (so the
    header lists *only* the unified group — what ``dorado polish``
    requires) → ``samtools index``. Returns ``out_bam``. Works for a
    single input too (the read group is still canonicalized).
    """
    samtools = str(_resolve_samtools())
    out_bam = Path(out_bam)
    work = out_bam.parent
    work.mkdir(parents=True, exist_ok=True)
    merged = work / f"{out_bam.stem}.merged.bam"
    tagged = work / f"{out_bam.stem}.tagged.bam"
    header_file = work / f"{out_bam.stem}.header.sam"

    rg_tokens = ["@RG", f"ID:{unified_rg_id}", "PL:ONT"]
    if model_ds:
        rg_tokens.append(f"DS:basecall_model={model_ds}")
    rg_line = "\t".join(rg_tokens)

    try:
        _emit(progress_cb, "harmonize-rg", "samtools cat")
        subprocess.run(
            [samtools, "cat", "-o", str(merged), *(str(b) for b in in_bams)],
            check=True,
        )
        _emit(progress_cb, "harmonize-rg", "samtools addreplacerg")
        subprocess.run(
            [samtools, "addreplacerg", "-@", str(int(threads)),
             "-m", "overwrite_all", "-r", rg_line, "-o", str(tagged), str(merged)],
            check=True,
        )
        # Drop the orphaned source @RG header lines so only the unified
        # group survives (reheader swaps the header; reads already carry
        # the unified RG:Z from addreplacerg).
        header_text = subprocess.run(
            [samtools, "view", "-H", str(tagged)],
            check=True, capture_output=True, text=True,
        ).stdout
        kept = [ln for ln in header_text.splitlines() if not ln.startswith("@RG")]
        kept.append(rg_line)
        header_file.write_text("\n".join(kept) + "\n")
        _emit(progress_cb, "harmonize-rg", "samtools reheader")
        with out_bam.open("wb") as out:
            subprocess.run(
                [samtools, "reheader", str(header_file), str(tagged)],
                stdout=out, check=True,
            )
        subprocess.run([samtools, "index", str(out_bam)], check=True)
    finally:
        for tmp in (merged, tagged, header_file):
            tmp.unlink(missing_ok=True)
    return out_bam


__all__ = [
    "DEFAULT_UNIFIED_RG",
    "extract_basecall_model",
    "read_basecaller_models",
    "validate_single_model",
    "harmonize_read_group",
]
