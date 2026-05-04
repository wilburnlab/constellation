"""Mapping verbs — use-case orchestrators on top of ``minimap2_run``.

Each verb is a thin composition: materialise/locate the target,
construct the use-case-specific minimap2 arg list, call
:func:`constellation.sequencing.align.minimap2.minimap2_run`, sort+index
via samtools when a BAM is requested. The minimap2 subprocess wrapper
itself stays generic (see :mod:`.minimap2`).

Shipped:

    map_to_genome      splice-aware full-length cDNA → genome alignment
                       (matches ``cdna_wilburn_v1`` forward-stranded
                       library; flags hard-coded here, not in the
                       runner)

Pending:

    map_assembly       assembly-vs-assembly (asm5/asm10/asm20). Stub
                       carries a TODO flagging the same use-case
                       orchestrator pattern as ``map_to_genome``.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from constellation.sequencing.align.minimap2 import (
    minimap2_build_index,
    minimap2_run,
)
from constellation.sequencing.progress import ProgressCallback, ProgressEvent
from constellation.sequencing.reference.reference import GenomeReference
from constellation.thirdparty.registry import ToolNotFoundError, find


# Flag set for forward-stranded full-length cDNA → genome (cdna_wilburn_v1).
# Lives at the orchestrator layer, not in the minimap2 runner.
_GENOME_MODE_ARGS: tuple[str, ...] = (
    "-ax",
    "splice",
    "-uf",
    "--cs=long",
    "--secondary=no",
)


def _resolve_samtools() -> Path:
    try:
        return find("samtools").path
    except ToolNotFoundError as exc:
        raise FileNotFoundError(
            "samtools not on $PATH; install via bioconda: "
            "`conda install -c bioconda samtools`"
        ) from exc


def _materialise_genome_fasta(
    genome: GenomeReference,
    fasta_path: Path,
    meta_path: Path,
) -> Path:
    """Write a GenomeReference's contigs to FASTA, caching by contig count.

    Skips rewrite if ``fasta_path`` exists and ``meta_path`` records a
    matching contig count. Coarse cache key — adequate for the lab's
    "import once, align many" workflow; if a future caller mutates a
    GenomeReference between align invocations, drop the meta file.
    """
    fasta_path = Path(fasta_path)
    meta_path = Path(meta_path)
    expected_n = int(genome.contigs.num_rows)
    if fasta_path.exists() and meta_path.exists():
        try:
            stamp = json.loads(meta_path.read_text())
            if int(stamp.get("n_contigs", -1)) == expected_n:
                return fasta_path
        except (OSError, json.JSONDecodeError):
            pass

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    contig_ids = genome.contigs.column("contig_id").to_pylist()
    names = genome.contigs.column("name").to_pylist()
    with fasta_path.open("w", encoding="utf-8") as fh:
        for contig_id, name in zip(contig_ids, names):
            fh.write(f">{name}\n")
            fh.write(genome.sequence_of(int(contig_id)))
            fh.write("\n")
    meta_path.write_text(
        json.dumps({"n_contigs": expected_n, "fasta": fasta_path.name}, indent=2)
    )
    return fasta_path


def _samtools_sort_index(
    inputs: Sequence[Path],
    output_bam: Path,
    *,
    threads: int = 8,
) -> Path:
    """Sort + index ``inputs`` into a single coordinate-sorted ``output_bam``.

    Uses ``samtools cat | samtools sort`` for multi-input merge; sort
    maintains coordinate order across the merge. Indexes via
    ``samtools index``. Returns the indexed BAM path.
    """
    samtools_bin = _resolve_samtools()
    output_bam = Path(output_bam)
    output_bam.parent.mkdir(parents=True, exist_ok=True)

    if len(inputs) == 1:
        sort_input = [str(samtools_bin), "view", "-b", str(inputs[0])]
    else:
        sort_input = [str(samtools_bin), "cat", *(str(p) for p in inputs)]
    sort_cmd = [
        str(samtools_bin),
        "sort",
        "-@",
        str(int(threads)),
        "-o",
        str(output_bam),
    ]

    cat_proc = subprocess.Popen(sort_input, stdout=subprocess.PIPE)
    try:
        sort_proc = subprocess.Popen(sort_cmd, stdin=cat_proc.stdout)
        if cat_proc.stdout is not None:
            cat_proc.stdout.close()
        sort_rc = sort_proc.wait()
        cat_rc = cat_proc.wait()
    finally:
        if cat_proc.poll() is None:
            cat_proc.kill()
    if cat_rc != 0:
        raise subprocess.CalledProcessError(cat_rc, sort_input)
    if sort_rc != 0:
        raise subprocess.CalledProcessError(sort_rc, sort_cmd)

    subprocess.run(
        [str(samtools_bin), "index", str(output_bam)],
        check=True,
    )
    return output_bam


def map_to_genome(
    reads_paths: Sequence[Path],
    genome: GenomeReference,
    *,
    output_dir: Path,
    threads: int = 8,
    extra_minimap2_args: tuple[str, ...] = (),
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Splice-aware full-length cDNA → genome alignment via minimap2.

    Materialises ``genome`` to a cached FASTA at ``output_dir/genome.fa``
    and a ``.mmi`` at ``output_dir/genome.mmi``; both reuse on subsequent
    calls. Each input file is mapped separately, then ``samtools cat |
    samtools sort`` concatenates and coordinate-sorts the per-input BAMs
    into ``output_dir/bam/aligned.bam`` (indexed via ``samtools index``).

    Returns the final sorted, indexed BAM path. The CLI handler then
    feeds it into :func:`sequencing.io.sam_bam.read_bam` (or the
    chunked decoder for the pipeline).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bam_dir = output_dir / "bam"
    bam_dir.mkdir(parents=True, exist_ok=True)

    fasta = _materialise_genome_fasta(
        genome,
        output_dir / "genome.fa",
        output_dir / ".genome_meta.json",
    )
    mmi = minimap2_build_index(
        fasta, output_dir / "genome.mmi", threads=threads
    )

    args = _GENOME_MODE_ARGS + tuple(extra_minimap2_args)

    per_input: list[Path] = []
    n_inputs = len(reads_paths)
    if progress_cb is not None:
        progress_cb(
            ProgressEvent(
                kind="stage_start",
                stage="align",
                total=n_inputs,
                message=f"minimap2 splice-aware mapping of {n_inputs} input(s)",
            )
        )
    for idx, reads_path in enumerate(reads_paths):
        per_input_bam = bam_dir / f"aligned.{idx:05d}.bam"
        minimap2_run(
            target=mmi,
            queries=[Path(reads_path)],
            output_path=per_input_bam,
            args=args,
            threads=threads,
            progress_cb=None,  # don't double-emit at runner level
        )
        per_input.append(per_input_bam)
        if progress_cb is not None:
            progress_cb(
                ProgressEvent(
                    kind="stage_progress",
                    stage="align",
                    completed=idx + 1,
                    total=n_inputs,
                    message=str(per_input_bam),
                )
            )

    final_bam = _samtools_sort_index(
        per_input, bam_dir / "aligned.bam", threads=threads
    )

    # Cleanup intermediates — single sorted BAM is the artefact
    for p in per_input:
        try:
            p.unlink()
        except OSError:
            pass

    if progress_cb is not None:
        progress_cb(
            ProgressEvent(
                kind="stage_done",
                stage="align",
                completed=n_inputs,
                total=n_inputs,
                message=str(final_bam),
            )
        )
    return final_bam


def map_assembly(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Map one assembly against another (asm5 / asm10 / asm20).

    TODO: ship as a thin orchestrator over :func:`minimap2_run` mirroring
    :func:`map_to_genome`. Same use-case-specific-flags pattern.
    """
    raise NotImplementedError(
        "map_assembly: pending — orchestrator on top of minimap2_run "
        "with asm5/asm10/asm20 preset"
    )


__all__ = ["map_to_genome", "map_assembly"]
