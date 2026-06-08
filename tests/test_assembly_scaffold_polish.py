"""RagTagRunner + PolishRunner — full orchestrators via mock binaries.

No real ragtag / dorado / samtools needed: tiny shell stubs resolved
through the ``$CONSTELLATION_*_HOME`` env vars stand in. Exercises the
FASTA materialization, AGP parsing, scaffold-sequence ingest, the
align→sort→index→polish loop, and polish-round bookkeeping.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pytest

from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.assembly.polish import PolishRunner
from constellation.sequencing.assembly.ragtag import RagTagRunner
from constellation.sequencing.reference.reference import GenomeReference

_RAGTAG_STUB = r"""#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "RagTag v2.1.0"; exit 0; fi
out=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "-o" ]]; then out="${args[$((i+1))]}"; fi
done
[[ -z "$out" ]] && exit 0
mkdir -p "$out"
printf '>chr1_RagTag\nACGTACGTNNNNTTTTGGGG\n' > "$out/ragtag.scaffold.fasta"
{
  printf 'chr1_RagTag\t1\t8\t1\tW\tctg1\t1\t8\t+\n'
  printf 'chr1_RagTag\t9\t12\t2\tN\t4\tscaffold\tyes\talign_genus\n'
  printf 'chr1_RagTag\t13\t20\t3\tW\tctg2\t1\t8\t-\n'
} > "$out/ragtag.scaffold.agp"
"""

_DORADO_STUB = r"""#!/usr/bin/env bash
verb="${1:-}"
case "$verb" in
  --version) echo "dorado 0.8.3+mock"; exit 0 ;;
  aligner) printf 'MOCKALN'; exit 0 ;;
  polish) draft="${@: -1}"; cat "$draft"; exit 0 ;;
  *) exit 0 ;;
esac
"""

_SAMTOOLS_STUB = r"""#!/usr/bin/env bash
case "${1:-}" in
  --version) echo "samtools 1.21"; exit 0 ;;
  sort)
    shift; out=""; inp=""
    while [[ $# -gt 0 ]]; do
      case "$1" in -@) shift 2 ;; -o) out="$2"; shift 2 ;; *) inp="$1"; shift ;; esac
    done
    cp "$inp" "$out" ;;
  index) bam="${@: -1}"; : > "${bam}.bai" ;;
  *) exit 0 ;;
esac
"""


def _install_stub(home: Path, rel: str, body: str, monkeypatch, env_var: str) -> None:
    path = home / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    monkeypatch.setenv(env_var, str(home))


def _draft_assembly() -> Assembly:
    contigs = pa.table(
        {
            "contig_id": pa.array([0, 1], type=pa.int64()),
            "name": ["ctg1", "ctg2"],
            "length": pa.array([8, 8], type=pa.int64()),
        }
    )
    sequences = pa.table(
        {
            "contig_id": pa.array([0, 1], type=pa.int64()),
            "sequence": ["ACGTACGT", "TTTTGGGG"],
        }
    )
    return Assembly.from_tables(contigs, sequences)


def _reference() -> GenomeReference:
    contigs = pa.table(
        {
            "contig_id": pa.array([0], type=pa.int64()),
            "name": ["chr1"],
            "length": pa.array([20], type=pa.int64()),
        }
    )
    sequences = pa.table(
        {
            "contig_id": pa.array([0], type=pa.int64()),
            "sequence": ["ACGTACGTACGTTTTTGGGG"],
        }
    )
    return GenomeReference(contigs=contigs, sequences=sequences)


@pytest.fixture(autouse=True)
def _need_bash():
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash unavailable for mock binaries")


def test_ragtag_runner_builds_scaffolded_assembly(tmp_path: Path, monkeypatch):
    _install_stub(
        tmp_path / "ragtag_home", "ragtag.py", _RAGTAG_STUB, monkeypatch,
        "CONSTELLATION_RAGTAG_HOME",
    )
    result = RagTagRunner().run(_draft_assembly(), _reference(), tmp_path / "scaffold")

    # contigs are now the scaffold sequences
    assert result.n_contigs == 1
    assert result.contigs.to_pylist()[0]["name"] == "chr1_RagTag"
    assert result.contigs.to_pylist()[0]["haplotype"] == "scaffold"
    # SCAFFOLD_TABLE records the draft-contig composition
    assert result.scaffolds is not None
    assert result.scaffolds.num_rows == 2
    sc = result.scaffolds.to_pylist()
    assert sc[0]["contig_id"] == 0 and sc[0]["gap_size"] == 4
    assert sc[1]["contig_id"] == 1 and sc[1]["orientation"] == "-"
    assert result.stats.to_pylist()[0]["n_scaffolds"] == 1
    # scaffold FASTA path stashed for downstream polish
    assert "scaffolded_fasta" in result.metadata_extras


def test_polish_runner_increments_polish_rounds(tmp_path: Path, monkeypatch):
    _install_stub(
        tmp_path / "dorado_home", "bin/dorado", _DORADO_STUB, monkeypatch,
        "CONSTELLATION_DORADO_HOME",
    )
    _install_stub(
        tmp_path / "samtools_home", "bin/samtools", _SAMTOOLS_STUB, monkeypatch,
        "CONSTELLATION_SAMTOOLS_HOME",
    )
    reads_bam = tmp_path / "harmonized.bam"
    reads_bam.write_bytes(b"")  # mock dorado ignores content

    polished = PolishRunner(rounds=1, device="cpu").run(
        _draft_assembly(), [reads_bam], tmp_path / "polish"
    )
    rows = polished.contigs.to_pylist()
    assert all(r["polish_rounds"] == 1 for r in rows)
    assert all(r["haplotype"] == "polished" for r in rows)
    assert polished.metadata_extras["polish_rounds"] == 1
    # consensus FASTA was produced + parsed (draft echoed back by the mock)
    assert (tmp_path / "polish" / "round_1" / "consensus.fasta").exists()


def test_polish_two_rounds_accumulate(tmp_path: Path, monkeypatch):
    _install_stub(
        tmp_path / "dorado_home", "bin/dorado", _DORADO_STUB, monkeypatch,
        "CONSTELLATION_DORADO_HOME",
    )
    _install_stub(
        tmp_path / "samtools_home", "bin/samtools", _SAMTOOLS_STUB, monkeypatch,
        "CONSTELLATION_SAMTOOLS_HOME",
    )
    reads_bam = tmp_path / "harmonized.bam"
    reads_bam.write_bytes(b"")
    polished = PolishRunner(rounds=2, device="cpu").run(
        _draft_assembly(), [reads_bam], tmp_path / "polish"
    )
    assert polished.contigs.to_pylist()[0]["polish_rounds"] == 2
    assert (tmp_path / "polish" / "round_2" / "consensus.fasta").exists()
