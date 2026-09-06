"""Dorado-based polishing — dorado aligner + dorado polish.

Per round: align the harmonized reads back to the current draft with
``dorado aligner`` (NOT plain minimap2 — ``dorado polish`` accepts only
Dorado-aligned BAMs), sort + index, then ``dorado polish`` emits a
consensus FASTA that becomes the next round's draft. Returns a new
``Assembly`` with ``polish_rounds`` stamped per contig.

The input reads must be the single-``@RG`` harmonized BAM (see
:func:`sequencing.basecall.readgroup.harmonize_read_group`) — ``dorado
polish`` resolves its model from that read group's ``DS`` tag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from constellation.core.progress import ProgressCallback, ProgressEvent
from constellation.sequencing.assembly._samtools import samtools_index, samtools_sort
from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.basecall.dorado import DoradoRunner
from constellation.sequencing.basecall.dorado_run import dorado_version
from constellation.sequencing.readers.fastx import read_fasta_genome
from constellation.sequencing.reference.materialize import materialise_genome_fasta


@dataclass(frozen=True)
class PolishRunner:
    """Polish a draft ``Assembly`` with ``dorado aligner`` + ``dorado polish``.

    ``run(assembly, read_paths, output_dir, rounds=N)`` runs N rounds; the
    reads are the single-``@RG`` harmonized BAM (``read_paths[0]``).
    """

    rounds: int = 1
    threads: int = 8
    device: str = "cuda:0"
    dorado_polish_extra: tuple[str, ...] = field(default_factory=tuple)

    def run(
        self,
        assembly: Assembly,
        read_paths: list[Path],
        output_dir: Path,
        *,
        rounds: int | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> Assembly:
        n = rounds if rounds is not None else self.rounds
        if n < 0:
            # 0 intentionally disables polishing; a negative count is a
            # typo that silently returned the input unchanged while the
            # standalone command printed "polished" and the pipeline
            # wrote a successful manifest recording polish_rounds: -1.
            raise ValueError(f"polish rounds must be >= 0; got {n}")
        if n < 1:
            return assembly
        if not read_paths:
            raise ValueError("PolishRunner.run needs the harmonized reads BAM")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(read_paths) == 1:
            reads_bam = Path(read_paths[0])
        else:
            # `genome polish --reads` takes nargs="+". Using only the
            # first silently dropped every other flow cell from alignment
            # and consensus, materially cutting coverage and biasing the
            # polished assembly. Harmonize them into one model-tagged @RG,
            # exactly as the assemble pipeline does — dorado polish
            # rejects multi-model BAMs, so this is also what it needs.
            from constellation.sequencing.basecall.readgroup import (
                harmonize_read_group,
            )

            reads_bam = harmonize_read_group(
                [Path(p) for p in read_paths],
                output_dir / "harmonized.bam",
                threads=self.threads,
            )

        dorado = DoradoRunner(device=self.device, threads=self.threads)
        # Baseline polish count carried in from a prior polish run, if any.
        prior = [
            x
            for x in assembly.contigs.column("polish_rounds").to_pylist()
            if x is not None
        ]
        base = max(prior) if prior else 0

        working = assembly
        for r in range(1, n + 1):
            round_dir = output_dir / f"round_{r}"
            round_dir.mkdir(parents=True, exist_ok=True)
            if progress_cb is not None:
                progress_cb(
                    ProgressEvent(
                        kind="stage_progress",
                        stage="polish",
                        message=f"round {r}/{n}",
                        completed=r,
                        total=n,
                    )
                )

            draft_fasta = materialise_genome_fasta(
                working.to_genome_reference(),
                round_dir / "draft.fasta",
                round_dir / "draft.fasta.meta.json",
            )
            aligned = round_dir / "aligned.bam"
            rc = dorado.aligner(draft_fasta, reads_bam, aligned).wait()
            if rc != 0:
                raise RuntimeError(
                    f"dorado aligner exited {rc} in polish round {r} "
                    f"({round_dir}); refusing to continue"
                )
            sorted_bam = round_dir / "aligned.sorted.bam"
            samtools_sort(aligned, sorted_bam, threads=self.threads)
            samtools_index(sorted_bam, threads=self.threads)

            consensus = round_dir / "consensus.fasta"
            rc = dorado.polish(
                sorted_bam,
                draft_fasta,
                consensus,
                extra=tuple(self.dorado_polish_extra),
            ).wait()
            # The exit code was discarded and the stdout file parsed
            # regardless. read_fasta_genome accepts an empty or truncated
            # FASTA, so an OOM or model mismatch produced a persisted,
            # _SUCCESS-marked, possibly EMPTY assembly that looked like a
            # completed polish.
            if rc != 0:
                raise RuntimeError(
                    f"dorado polish exited {rc} in round {r} ({round_dir}); "
                    f"refusing to read {consensus.name} as a polished assembly"
                )

            polished_genome = read_fasta_genome(consensus)
            if polished_genome.contigs.num_rows == 0:
                raise RuntimeError(
                    f"dorado polish exited 0 in round {r} but produced no "
                    f"contigs at {consensus}; refusing to persist an empty "
                    f"assembly"
                )
            provenance = json.dumps(
                {
                    "stage": "polish",
                    "tool": "dorado polish",
                    "version": dorado_version(),
                    "round": r,
                }
            )
            working = Assembly.from_genome_reference(
                polished_genome,
                scaffolds=working.scaffolds,
                haplotype="polished",
                polish_rounds=base + r,
                provenance_json=provenance,
                metadata_extras={
                    "stage": "polish",
                    "polish_rounds": base + r,
                    "polished_fasta": str(consensus),
                },
            )
        return working


__all__ = ["PolishRunner"]
