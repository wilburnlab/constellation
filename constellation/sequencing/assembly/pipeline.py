"""``run_assembly_pipeline`` — the ``genome assemble`` orchestrator.

Drives the genome-assembly spine, each stage gated by a ``_SUCCESS``
marker + ``--resume`` and emitting ``ProgressEvent``s (never raw stdout):

    (--pod5) basecall  → basecall/<fc>.bam
    ingest + harmonize → bam/harmonized.bam   (single model-tagged @RG)
    samtools fastq     → reads/reads.fastq.gz
    hifiasm --ont      → assembly/  (+ draft report)
    (opt) ragtag       → scaffold/  (+ report)
    (opt) dorado polish→ polish/    (+ report)
    comparative report + manifest

Standalone re-entry (``genome scaffold`` / ``polish`` / ``diagnose``) read
the per-stage ``Assembly`` bundles + the manifest this writes.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pyarrow as pa

from constellation.core.progress import ProgressCallback, ProgressEvent
from constellation.sequencing.assembly._samtools import samtools_fastq
from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.assembly.diagnostics import (
    generate_assembly_report,
    generate_comparative_report,
)
from constellation.sequencing.assembly.hifiasm import HiFiAsmRunner
from constellation.sequencing.assembly.io import load_assembly, save_assembly
from constellation.sequencing.assembly.manifest import write_assembly_manifest
from constellation.sequencing.assembly.polish import PolishRunner
from constellation.sequencing.assembly.ragtag import RagTagRunner
from constellation.sequencing.assembly.stats import apply_busco
from constellation.sequencing.basecall.dorado import DoradoRunner
from constellation.sequencing.basecall.dorado_run import dorado_version
from constellation.sequencing.basecall.models import DoradoModel
from constellation.sequencing.basecall.readgroup import (
    DEFAULT_UNIFIED_RG,
    harmonize_read_group,
    read_basecaller_models,
    validate_single_model,
)
from constellation.sequencing.reference.reference import GenomeReference
from constellation.thirdparty.registry import ToolNotFoundError, find

_BAM_SUFFIXES = (".bam", ".sam")


def _emit(cb: ProgressCallback | None, kind: str, stage: str, message: str = "") -> None:
    if cb is not None:
        cb(ProgressEvent(kind=kind, stage=stage, message=message))


def _collect_inputs(paths: list[Path], suffixes: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for suffix in suffixes:
                out.extend(sorted(p.glob(f"*{suffix}")))
        else:
            out.append(p)
    return out


def _done(stage_dir: Path) -> bool:
    return (stage_dir / "_SUCCESS").exists()


def _mark_done(stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "_SUCCESS").touch()


# Map the user-facing --reads-compression vocabulary to the codec understood
# by ``samtools_fastq`` ("auto" picks the multithreaded BGZF path).
_FASTQ_CODEC = {"auto": "bgzf", "bgzf": "bgzf", "gzip": "gzip", "none": "none"}


def _resolve_scratch_root(scratch_dir: str | Path | None, output_dir: Path) -> Path:
    """Where the (throwaway) reads FASTQ lives.

    Scratch routing is opt-in so the default keeps the legacy in-tree layout
    (``<output_dir>/reads/``). ``--scratch-dir auto`` consults
    ``$SLURM_TMPDIR`` then ``$TMPDIR``; an explicit path is used as-is. When a
    node-local scratch is used, the FASTQ goes in a job-unique subdir so
    concurrent jobs sharing the mount don't collide.
    """
    if scratch_dir is None:
        return output_dir
    if str(scratch_dir) == "auto":
        raw = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
        if not raw:
            return output_dir
        root = Path(raw)
    else:
        root = Path(scratch_dir)
    root = root.expanduser()
    if root == output_dir:
        return output_dir
    job = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    return root / f"constellation_asm_{job}"


def _rel_or_abs(path: Path, base: Path) -> str:
    """Path relative to ``base`` when possible, else absolute (off-tree scratch)."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _tool_version(name: str) -> str | None:
    try:
        return find(name).version
    except ToolNotFoundError:
        return None


def _stage_stats_dict(asm: Assembly) -> dict:
    s = asm.stats.to_pylist()[0]
    return {
        "n_contigs": s["n_contigs"],
        "n_scaffolds": s["n_scaffolds"],
        "total_length": s["total_length"],
        "n50": s["n50"],
        "largest_contig": s["largest_contig"],
        "gc_content": s["gc_content"],
        "busco_complete": s.get("busco_complete"),
    }


def _finalize_stage(
    asm: Assembly,
    *,
    stage_dir: Path,
    stage_label: str,
    busco_lineage: str | None,
    threads: int,
    emit_report: bool,
) -> Assembly:
    """Optionally enrich with BUSCO, persist the bundle, emit a report."""
    if busco_lineage:
        from constellation.sequencing.annotation.busco import BuscoRunner

        busco_row, _features = BuscoRunner(lineage=busco_lineage, threads=threads).run(
            asm, stage_dir / "busco"
        )
        asm = dataclasses.replace(asm, stats=apply_busco(asm.stats, busco_row))
    save_assembly(asm, stage_dir / "assembly")
    if emit_report:
        generate_assembly_report(
            asm, output_dir=stage_dir / "diagnostics", stage_label=stage_label
        )
    return asm


def run_assembly_pipeline(
    *,
    output_dir: Path,
    reads: list[Path] | None = None,
    pod5: list[Path] | None = None,
    basecall_model: str | None = None,
    modified_bases: tuple[str, ...] = (),
    device: str = "cuda:0",
    duplex: bool = False,
    emit_moves: bool = True,
    read_group: str = DEFAULT_UNIFIED_RG,
    allow_multi_model: bool = False,
    hifiasm_mode: str = "ont",
    hifiasm_extra: tuple[str, ...] = (),
    scaffold_reference: GenomeReference | None = None,
    scaffold_reference_handle: str | None = None,
    scaffold_reference_path: str | None = None,
    assembly_accession: str | None = None,
    ragtag_extra: tuple[str, ...] = (),
    polish_rounds: int = 0,
    dorado_polish_extra: tuple[str, ...] = (),
    busco_lineage: str | None = None,
    threads: int = 1,
    reads_compression: str = "auto",
    scratch_dir: str | Path | None = None,
    keep_intermediates: bool = True,
    resume: bool = False,
    emit_report: bool = True,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Run the genome-assembly pipeline; return ``output_dir``.

    ``reads_compression`` (``auto``/``bgzf``/``gzip``/``none``) controls how the
    intermediate reads FASTQ is written: ``auto``/``bgzf`` use the multithreaded
    ``bgzip`` pipe, ``none`` writes plain uncompressed FASTQ (fastest, ~3x the
    disk). ``scratch_dir`` (a path, or ``"auto"`` for ``$SLURM_TMPDIR``/``$TMPDIR``)
    routes that throwaway FASTQ to node-local storage instead of the output tree.
    ``keep_intermediates=False`` unlinks the reads FASTQ once assembly succeeds.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(reads) == bool(pod5):
        raise ValueError("provide exactly one of `reads` (BAM/SAM) or `pod5`")

    stage_stats: dict[str, pa.Table] = {}
    stages: dict[str, dict] = {}

    # ── 1. basecall (pod5 mode only) ────────────────────────────────
    if pod5:
        if not basecall_model:
            raise ValueError("--pod5 mode requires --model")
        input_mode = "pod5"
        input_files = [str(p) for p in pod5]
        model = DoradoModel.parse(basecall_model)
        basecall_dir = output_dir / "basecall"
        bam_inputs: list[Path] = []
        runner = DoradoRunner(device=device, threads=threads)
        # Each --pod5 argument is one flow cell → its own basecall BAM
        # (a directory of .pod5 is passed whole to dorado).
        for i, source in enumerate(Path(p) for p in pod5):
            out_bam = basecall_dir / f"flowcell_{i:03d}.bam"
            if resume and out_bam.exists() and _done(basecall_dir):
                bam_inputs.append(out_bam)
                continue
            _emit(progress_cb, "stage_start", "basecall", str(source))
            if duplex:
                handle = runner.duplex(model, [source], out_bam, device=device)
            else:
                handle = runner.basecaller(
                    model, [source], out_bam,
                    modified_bases=modified_bases, device=device,
                    emit_moves=emit_moves,
                )
            rc = handle.wait()
            if rc != 0:
                raise RuntimeError(f"dorado basecaller failed (rc={rc}) for {source}")
            bam_inputs.append(out_bam)
        _mark_done(basecall_dir)
        _emit(progress_cb, "stage_done", "basecall", f"{len(bam_inputs)} BAM(s)")
    else:
        input_mode = "bam"
        bam_inputs = _collect_inputs([Path(p) for p in (reads or [])], _BAM_SUFFIXES)
        input_files = [str(p) for p in bam_inputs]
        if not bam_inputs:
            raise ValueError("no BAM/SAM inputs found under --reads")

    # ── 2. ingest + read-group harmonization ────────────────────────
    bam_dir = output_dir / "bam"
    harmonized = bam_dir / "harmonized.bam"
    models = read_basecaller_models(bam_inputs)
    model_ds = validate_single_model(models, allow_multi=allow_multi_model)
    if resume and harmonized.exists() and _done(bam_dir):
        _emit(progress_cb, "stage_progress", "harmonize", "resume: using existing")
    else:
        _emit(progress_cb, "stage_start", "harmonize", str(harmonized))
        harmonize_read_group(
            bam_inputs, harmonized,
            unified_rg_id=read_group, model_ds=model_ds,
            threads=threads, progress_cb=progress_cb,
        )
        _mark_done(bam_dir)
        _emit(progress_cb, "stage_done", "harmonize", str(harmonized))

    # ── 3. fastq ────────────────────────────────────────────────────
    # The reads FASTQ is a throwaway hifiasm input; once assembly is done it's
    # never read again (polish reuses the harmonized BAM). Skip regeneration on
    # a post-assembly resume so a vanished node-local-scratch FASTQ doesn't
    # force needless rework.
    assembly_dir = output_dir / "assembly"
    assembly_done = (
        resume and _done(assembly_dir) and (assembly_dir / "assembly").exists()
    )
    reads_dir = _resolve_scratch_root(scratch_dir, output_dir) / "reads"
    fastq_name = "reads.fastq" if reads_compression == "none" else "reads.fastq.gz"
    fastq = reads_dir / fastq_name
    if assembly_done or (resume and fastq.exists() and _done(reads_dir)):
        pass
    else:
        _emit(progress_cb, "stage_start", "fastq", str(fastq))
        samtools_fastq(
            harmonized, fastq, threads=threads,
            codec=_FASTQ_CODEC.get(reads_compression, "bgzf"),
        )
        _mark_done(reads_dir)
        _emit(progress_cb, "stage_done", "fastq", str(fastq))

    # ── 4. assemble ─────────────────────────────────────────────────
    if assembly_done:
        draft = load_assembly(assembly_dir / "assembly")
    else:
        _emit(progress_cb, "stage_start", "assemble", "hifiasm")
        draft = HiFiAsmRunner(
            threads=threads, mode=hifiasm_mode, extra_args=hifiasm_extra
        ).run([fastq], assembly_dir / "primary", progress_cb=progress_cb)
        draft = _finalize_stage(
            draft, stage_dir=assembly_dir, stage_label="draft",
            busco_lineage=busco_lineage, threads=threads, emit_report=emit_report,
        )
        _mark_done(assembly_dir)
        _emit(progress_cb, "stage_done", "assemble", f"{draft.n_contigs} contigs")
    # Throwaway FASTQ cleanup once assembly is complete.
    if not keep_intermediates:
        Path(fastq).unlink(missing_ok=True)
    stage_stats["draft"] = draft.stats
    stages["assemble"] = _stage_stats_dict(draft)
    working = draft
    outputs: dict[str, str] = {
        "harmonized_bam": _rel_or_abs(harmonized, output_dir),
        "reads_fastq": _rel_or_abs(fastq, output_dir),
        "assembly_bundle": "assembly/assembly",
        "primary_gfa": "assembly/primary.bp.p_ctg.gfa",
    }

    # ── 5. scaffold (optional) ──────────────────────────────────────
    if scaffold_reference is not None:
        scaffold_dir = output_dir / "scaffold"
        if resume and _done(scaffold_dir) and (scaffold_dir / "assembly").exists():
            scaffolded = load_assembly(scaffold_dir / "assembly")
        else:
            _emit(progress_cb, "stage_start", "scaffold", "ragtag")
            scaffolded = RagTagRunner(threads=threads, extra_args=ragtag_extra).run(
                working, scaffold_reference, scaffold_dir, progress_cb=progress_cb
            )
            scaffolded = _finalize_stage(
                scaffolded, stage_dir=scaffold_dir, stage_label="scaffold",
                busco_lineage=busco_lineage, threads=threads, emit_report=emit_report,
            )
            _mark_done(scaffold_dir)
            _emit(progress_cb, "stage_done", "scaffold", f"{scaffolded.n_contigs} scaffolds")
        stage_stats["scaffold"] = scaffolded.stats
        stages["scaffold"] = _stage_stats_dict(scaffolded)
        outputs["scaffold_bundle"] = "scaffold/assembly"
        working = scaffolded

    # ── 6. polish (optional) ────────────────────────────────────────
    if polish_rounds > 0:
        polish_dir = output_dir / "polish"
        if resume and _done(polish_dir) and (polish_dir / "assembly").exists():
            polished = load_assembly(polish_dir / "assembly")
        else:
            _emit(progress_cb, "stage_start", "polish", f"{polish_rounds} round(s)")
            polished = PolishRunner(
                rounds=polish_rounds, threads=threads, device=device,
                dorado_polish_extra=dorado_polish_extra,
            ).run(working, [harmonized], polish_dir, progress_cb=progress_cb)
            polished = _finalize_stage(
                polished, stage_dir=polish_dir, stage_label="polish",
                busco_lineage=busco_lineage, threads=threads, emit_report=emit_report,
            )
            _mark_done(polish_dir)
            _emit(progress_cb, "stage_done", "polish", "done")
        stage_stats["polish"] = polished.stats
        stages["polish"] = _stage_stats_dict(polished)
        outputs["polish_bundle"] = "polish/assembly"
        working = polished

    # ── 7. comparative report + manifest ────────────────────────────
    if emit_report and len(stage_stats) >= 2:
        generate_comparative_report(stage_stats, output_dir=output_dir / "diagnostics")
        outputs["comparison"] = "diagnostics/comparison.md"

    write_assembly_manifest(
        output_dir / "manifest.json",
        input_mode=input_mode,
        input_files=input_files,
        unified_read_group=read_group,
        polish_rounds=polish_rounds,
        parameters={
            "threads": threads,
            "hifiasm_mode": hifiasm_mode,
            "reads_compression": reads_compression,
            "device": device if input_mode == "pod5" else None,
            "duplex": duplex,
            "emit_moves": emit_moves if input_mode == "pod5" else None,
        },
        stages=stages,
        outputs=outputs,
        basecall_model=basecall_model,
        modified_bases=list(modified_bases) or None,
        device=device if input_mode == "pod5" else None,
        basecaller_model_ds=model_ds,
        scaffold_reference_handle=scaffold_reference_handle,
        scaffold_reference_path=scaffold_reference_path,
        assembly_accession=assembly_accession,
        tool_versions={
            "hifiasm": _tool_version("hifiasm"),
            "dorado": dorado_version(),
            "ragtag": _tool_version("ragtag") if scaffold_reference else None,
            "samtools": _tool_version("samtools"),
            "busco": _tool_version("busco") if busco_lineage else None,
        },
        tool_args={
            "hifiasm": [f"--{hifiasm_mode}" if hifiasm_mode == "ont" else "", *hifiasm_extra],
            "ragtag": list(ragtag_extra),
            "dorado_polish": list(dorado_polish_extra),
        },
        busco_lineage=busco_lineage,
    )
    _mark_done(output_dir)
    return output_dir


__all__ = ["run_assembly_pipeline"]
