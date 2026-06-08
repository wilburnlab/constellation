"""HiFiASM subprocess orchestrator.

HiFiASM (Cheng et al. 2021) is the lab's primary genome assembler for
nanopore long reads. The ``--ont`` mode (stable since hifiasm 0.25.0)
runs the built-in ONT corrector for R10.4.1 simplex reads; ``hifi`` mode
(no ``--ont``) is for PacBio HiFi. Primary contigs land in
``<prefix>.bp.p_ctg.gfa`` with the assembled sequence + an ``rd:i``
read-coverage tag per segment, which we parse into an :class:`Assembly`.

Composes the generic :func:`sequencing.assembly.hifiasm_run.hifiasm_run`
wrapper (mirrors the ``map_to_genome`` / ``minimap2_run`` split). The
binary is resolved via :func:`constellation.thirdparty.find('hifiasm')`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from constellation.core.progress import ProgressCallback
from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.assembly.gfa import GfaContig, parse_gfa_contigs
from constellation.sequencing.assembly.hifiasm_run import hifiasm_run
from constellation.sequencing.schemas.assembly import ASSEMBLY_CONTIG_TABLE
from constellation.sequencing.schemas.reference import SEQUENCE_TABLE
from constellation.thirdparty.registry import ToolNotFoundError, find


@dataclass(frozen=True)
class HiFiAsmRunner:
    """Thin orchestrator around the ``hifiasm`` binary.

    ``run(read_paths, output_prefix)`` invokes hifiasm, parses the
    primary-contig GFA, and returns an ``Assembly``. ``run_diploid``
    returns the two haplotype assemblies. ``mode='ont'`` (default) adds
    ``--ont`` for R10 simplex reads; ``mode='hifi'`` omits it.
    """

    threads: int = 16
    mode: str = "ont"
    extra_args: tuple[str, ...] = ()

    # ── public verbs ────────────────────────────────────────────────
    def run(
        self,
        read_paths: list[Path],
        output_prefix: Path,
        *,
        threads: int | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> Assembly:
        prefix = self._invoke(read_paths, output_prefix, threads, progress_cb)
        gfa = _find_gfa(prefix, ("bp.p_ctg", "p_ctg"))
        records = parse_gfa_contigs(gfa)
        return self._records_to_assembly(records, haplotype="primary", gfa=gfa)

    def run_diploid(
        self,
        read_paths: list[Path],
        output_prefix: Path,
        *,
        threads: int | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> tuple[Assembly, Assembly]:
        prefix = self._invoke(read_paths, output_prefix, threads, progress_cb)
        gfa1 = _find_gfa(prefix, ("bp.hap1.p_ctg", "hap1.p_ctg"))
        gfa2 = _find_gfa(prefix, ("bp.hap2.p_ctg", "hap2.p_ctg"))
        hap1 = self._records_to_assembly(
            parse_gfa_contigs(gfa1), haplotype="hap1", gfa=gfa1
        )
        hap2 = self._records_to_assembly(
            parse_gfa_contigs(gfa2), haplotype="hap2", gfa=gfa2
        )
        return (hap1, hap2)

    # ── internals ───────────────────────────────────────────────────
    def _mode_args(self) -> tuple[str, ...]:
        return ("--ont",) if self.mode == "ont" else ()

    def _invoke(
        self,
        read_paths: list[Path],
        output_prefix: Path,
        threads: int | None,
        progress_cb: ProgressCallback | None,
    ) -> Path:
        args = self._mode_args() + tuple(self.extra_args)
        return hifiasm_run(
            read_paths,
            Path(output_prefix),
            args=args,
            threads=threads if threads is not None else self.threads,
            progress_cb=progress_cb,
        )

    def _records_to_assembly(
        self,
        records: list[GfaContig],
        *,
        haplotype: str,
        gfa: Path,
    ) -> Assembly:
        provenance = json.dumps(
            {
                "assembler": "hifiasm",
                "version": _hifiasm_version(),
                "mode": self.mode,
                "args": list(self._mode_args() + tuple(self.extra_args)),
                "gfa": str(gfa),
            }
        )
        contig_rows = []
        seq_rows = []
        for i, rec in enumerate(records):
            contig_rows.append(
                {
                    "contig_id": i,
                    "name": rec.name,
                    "length": rec.length,
                    "read_coverage": rec.read_coverage,
                    "polish_rounds": None,
                    "haplotype": haplotype,
                    "circular": _circular_from_name(rec.name),
                    "provenance_json": provenance,
                }
            )
            seq_rows.append({"contig_id": i, "sequence": rec.sequence})
        contigs = pa.Table.from_pylist(contig_rows, schema=ASSEMBLY_CONTIG_TABLE)
        sequences = pa.Table.from_pylist(seq_rows, schema=SEQUENCE_TABLE)
        return Assembly.from_tables(
            contigs,
            sequences,
            metadata_extras={"assembler": "hifiasm", "haplotype": haplotype},
        )


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _find_gfa(prefix: Path, infixes: tuple[str, ...]) -> Path:
    """Locate a hifiasm contig GFA for ``prefix`` among candidate infixes.

    hifiasm names primary contigs ``<prefix>.bp.p_ctg.gfa`` (balanced
    phasing, the default) or ``<prefix>.p_ctg.gfa`` (``--primary``); the
    candidate list covers both. The ``.noseq.gfa`` variants are not
    candidates — they carry no sequence.
    """
    prefix = Path(prefix)
    candidates = [prefix.parent / f"{prefix.name}.{infix}.gfa" for infix in infixes]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no hifiasm contig GFA for prefix {prefix}; looked for "
        f"{[c.name for c in candidates]}"
    )


def _circular_from_name(name: str) -> bool | None:
    """hifiasm marks contig circularity with a trailing ``c`` (circular)
    or ``l`` (linear) on the segment name (e.g. ``ptg000002c``)."""
    if not name:
        return None
    if name[-1] == "c":
        return True
    if name[-1] == "l":
        return False
    return None


def _hifiasm_version() -> str | None:
    try:
        return find("hifiasm").version
    except ToolNotFoundError:
        return None


__all__ = ["HiFiAsmRunner"]
