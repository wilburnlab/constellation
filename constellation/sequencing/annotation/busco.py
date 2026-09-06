"""BUSCO subprocess wrapper — completeness + ortholog gene calls.

BUSCO (Benchmarking Universal Single-Copy Orthologs) is the standard
genome-completeness metric for eukaryotic assemblies. The runner
materializes the assembly FASTA, runs ``busco``, and parses two outputs:

- ``short_summary.*.txt`` → the ``busco_*`` completeness fractions for
  ``ASSEMBLY_STATS`` (the biologically meaningful quality signal the
  draft→scaffold→polish comparative report leans on),
- ``run_<lineage>/full_table.tsv`` → ``FEATURE_TABLE`` rows for the
  Complete + Duplicated orthologs (a paired ``Annotation``).

Lineage-specific ortholog datasets (eukaryota_odb10, vertebrata_odb12,
...) live outside the repo; ``BUSCO_DOWNLOADS_PATH`` points at the
on-disk cache (passed through as ``--download_path --offline``). The
summary / full-table parsers are pure functions → unit-testable without
a BUSCO install.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.reference.materialize import materialise_genome_fasta
from constellation.sequencing.schemas.reference import FEATURE_TABLE
from constellation.thirdparty.registry import ToolNotFoundError, find, venv_safe_env


_INSTALL_HINT = (
    "busco not found; install via bioconda (`conda install -c bioconda busco`) "
    "or `bash scripts/install-busco.sh`, or set $CONSTELLATION_BUSCO_HOME"
)

_SUMMARY_RE = re.compile(
    r"C:([\d.]+)%\[S:([\d.]+)%,D:([\d.]+)%\],F:([\d.]+)%,M:([\d.]+)%"
)


# ──────────────────────────────────────────────────────────────────────
# Pure parsers
# ──────────────────────────────────────────────────────────────────────


def parse_busco_summary(text: str) -> dict[str, float]:
    """Parse a BUSCO ``short_summary`` into completeness fractions (0–1).

    Returns keys ``complete`` / ``single`` / ``duplicated`` /
    ``fragmented`` / ``missing``. Raises ``ValueError`` if the canonical
    ``C:..[S:..,D:..],F:..,M:..`` line is absent.
    """
    m = _SUMMARY_RE.search(text)
    if m is None:
        raise ValueError("no BUSCO C:[S:,D:],F:,M: summary line found")
    c, s, d, f, miss = (float(x) / 100.0 for x in m.groups())
    return {
        "complete": c,
        "single": s,
        "duplicated": d,
        "fragmented": f,
        "missing": miss,
    }


def parse_busco_full_table(
    text: str,
    contig_name_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Parse a BUSCO ``full_table.tsv`` → ``FEATURE_TABLE`` row dicts.

    Only Complete + Duplicated orthologs (which carry a sequence + coords)
    become features. BUSCO coordinates are 1-based inclusive → converted
    to 0-based half-open. Orthologs on a sequence not in
    ``contig_name_to_id`` are skipped.
    """
    rows: list[dict[str, Any]] = []
    fid = 0
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 6:
            continue
        busco_id, status, sequence = f[0], f[1], f[2]
        if status not in ("Complete", "Duplicated"):
            continue
        contig_id = contig_name_to_id.get(sequence)
        if contig_id is None:
            continue
        try:
            g_start = int(float(f[3]))
            g_end = int(float(f[4]))
        except ValueError:
            continue
        strand = f[5] if f[5] in ("+", "-") else "."
        score = None
        if len(f) > 6 and f[6]:
            try:
                score = float(f[6])
            except ValueError:
                score = None
        rows.append(
            {
                "feature_id": fid,
                "contig_id": contig_id,
                "start": min(g_start, g_end) - 1,
                "end": max(g_start, g_end),
                "strand": strand,
                "type": "gene",
                "name": busco_id,
                "parent_id": None,
                "source": "busco",
                "score": score,
                "phase": None,
                "attributes_json": json.dumps({"busco_status": status}),
            }
        )
        fid += 1
    return rows


def busco_summary_row(values: dict[str, float], lineage: str) -> pa.Table:
    """One-row table with the ``busco_*`` + ``busco_lineage`` columns."""
    return pa.table(
        {
            "busco_complete": pa.array([values["complete"]], type=pa.float32()),
            "busco_single": pa.array([values["single"]], type=pa.float32()),
            "busco_duplicated": pa.array([values["duplicated"]], type=pa.float32()),
            "busco_fragmented": pa.array([values["fragmented"]], type=pa.float32()),
            "busco_missing": pa.array([values["missing"]], type=pa.float32()),
            "busco_lineage": pa.array([lineage], type=pa.string()),
        }
    )


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────


def _resolve_busco() -> Path:
    try:
        return find("busco").path
    except ToolNotFoundError as exc:
        raise FileNotFoundError(_INSTALL_HINT) from exc


def _tail(path: Path, n: int = 25) -> str:
    try:
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        )
    except OSError:
        return ""


def _find_one(root: Path, pattern: str, *, recursive: bool = False) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    return matches[0] if matches else None


@dataclass(frozen=True)
class BuscoRunner:
    """Runs ``busco`` against an assembly and emits completeness + features."""

    lineage: str  # 'vertebrata_odb12', 'eukaryota_odb10', ...
    threads: int = 8
    mode: str = "genome"  # 'genome' | 'transcriptome' | 'protein'

    def run(
        self,
        assembly: Assembly,
        output_dir: Path,
    ) -> tuple[pa.Table, pa.Table]:
        """Returns ``(busco_summary_row, feature_rows)``.

        ``busco_summary_row`` is a one-row table of the busco_* columns
        (merge into ASSEMBLY_STATS via
        :func:`sequencing.assembly.stats.apply_busco`). ``feature_rows``
        are ``FEATURE_TABLE`` orthologs for a paired ``Annotation``.
        """
        busco = str(_resolve_busco())
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fasta = materialise_genome_fasta(
            assembly.to_genome_reference(),
            output_dir / "assembly.fasta",
            output_dir / "assembly.fasta.meta.json",
        )
        run_name = "busco"
        cmd = [
            busco,
            "-i", str(fasta),
            "-l", self.lineage,
            "-m", self.mode,
            "-o", run_name,
            "--out_path", str(output_dir),
            "-c", str(int(self.threads)),
            "-f",
        ]
        downloads = os.environ.get("BUSCO_DOWNLOADS_PATH")
        if downloads:
            cmd += ["--download_path", downloads, "--offline"]

        log = output_dir / "busco.log"
        with log.open("wb") as fh:
            # busco runs under its own venv interpreter — strip PYTHONPATH so
            # an HPC-exported path can't shadow the venv's pinned deps.
            result = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=venv_safe_env()
            )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, stderr=_tail(log)
            )

        run_dir = output_dir / run_name
        summary_path = _find_one(run_dir, "short_summary.*.txt")
        if summary_path is None:
            raise FileNotFoundError(f"no BUSCO short_summary under {run_dir}")
        values = parse_busco_summary(summary_path.read_text())
        stats_row = busco_summary_row(values, self.lineage)

        name_to_id = dict(
            zip(
                assembly.contigs.column("name").to_pylist(),
                assembly.contigs.column("contig_id").to_pylist(),
            )
        )
        full_path = _find_one(run_dir, "full_table.tsv", recursive=True)
        feature_rows = (
            parse_busco_full_table(full_path.read_text(), name_to_id)
            if full_path is not None
            else []
        )
        features = (
            pa.Table.from_pylist(feature_rows, schema=FEATURE_TABLE)
            if feature_rows
            else FEATURE_TABLE.empty_table()
        )
        return (stats_row, features)


__all__ = [
    "BuscoRunner",
    "parse_busco_summary",
    "parse_busco_full_table",
    "busco_summary_row",
]
