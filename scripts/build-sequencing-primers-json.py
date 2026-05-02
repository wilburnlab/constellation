"""Generate constellation/data/sequencing/cdna_wilburn_v1.json — the lab's
in-house TSO-based PCR-cDNA library construct (5' SSP + 3' adapter +
24-barcode panel).

Chemistry summary. The construct is a template-switching reverse
transcription + PCR amplification protocol — the "SMART" / IsoSeq family
of cDNA preparations, generic name TSO-PCR-cDNA. cDNA is read 5' → 3' as:

    [5' SSP] [transcript] [polyA] [3' adapter] [barcode]

where the 5' SSP is the strand-switch primer, the 3' adapter is shared
across all 24 oligo-dT primers (its bases come from the first 25 nt of
any dT primer, reverse-complemented), and the barcode is bases 25-40
(0-indexed half-open [25:41]) of the dT primer.

Sequences are taken verbatim from
~/projects/NanoporeAnalysis/transcriptome_nf/scripts/primers.txt — the
NanoporeAnalysis primer panel the lab has been running production with.

Run from project root:
    python3 scripts/build-sequencing-primers-json.py
"""

from __future__ import annotations

import json
from pathlib import Path


# 5' strand-switch primer (constant across all reads in this construct).
SSP = "AAGCAGTGGTATCAACGCAGAGTACGCGGG"

# Common 25-nt prefix on every dT primer. The 3' adapter that demux
# anchors against is the reverse-complement of this prefix.
DT_PRIMER_PREFIX = "AAGCAGTGGTATCAACGCAGAGTAC"


def _reverse_complement(seq: str) -> str:
    table = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq.translate(table)[::-1]


# Every dT primer has the form:
#   <prefix:25 nt> <barcode:16 nt> TTTTTT...VN
# We carry the full primer here so the regenerable build matches the
# upstream FASTA byte-for-byte; the barcode and prefix are extracted by
# slicing.
DT_PRIMERS: list[tuple[str, str]] = [
    ("dT_BC1001_PB", "AAGCAGTGGTATCAACGCAGAGTACCACATATCAGAGTGCGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1002_PB", "AAGCAGTGGTATCAACGCAGAGTACACACACAGACTGTGAGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1003_PB", "AAGCAGTGGTATCAACGCAGAGTACACACATCTCGTGAGAGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1004_PB", "AAGCAGTGGTATCAACGCAGAGTACCACGCACACACGCGCGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1005_PB", "AAGCAGTGGTATCAACGCAGAGTACCACTCGACTCTCGCGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1006_PB", "AAGCAGTGGTATCAACGCAGAGTACCATATATATCAGCTGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1007_PB", "AAGCAGTGGTATCAACGCAGAGTACTCTGTATCTCTATGTGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1008_PB", "AAGCAGTGGTATCAACGCAGAGTACACAGTCGAGCGCTGCGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1009_PB", "AAGCAGTGGTATCAACGCAGAGTACACACACGCGAGACAGATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1010_PB", "AAGCAGTGGTATCAACGCAGAGTACACGCGCTATCTCAGAGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1011_PB", "AAGCAGTGGTATCAACGCAGAGTACCTATACGTATATCTATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1012_PB", "AAGCAGTGGTATCAACGCAGAGTACACACTAGATCGCGTGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1013_PB", "AAGCAGTGGTATCAACGCAGAGTACCTCTCGCATACGCGAGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1014_PB", "AAGCAGTGGTATCAACGCAGAGTACCTCACTACGCGCGCGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1015_PB", "AAGCAGTGGTATCAACGCAGAGTACCGCATGACACGTGTGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1016_PB", "AAGCAGTGGTATCAACGCAGAGTACCATAGAGAGATAGTATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1017_PB", "AAGCAGTGGTATCAACGCAGAGTACCACACGCGCGCTATATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1018_PB", "AAGCAGTGGTATCAACGCAGAGTACTCACGTGCTCACTGTGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1019_PB", "AAGCAGTGGTATCAACGCAGAGTACACACACTCTATCAGATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1020_PB", "AAGCAGTGGTATCAACGCAGAGTACCACGACACGACGATGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1021_PB", "AAGCAGTGGTATCAACGCAGAGTACCTATACATAGTGATGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1022_PB", "AAGCAGTGGTATCAACGCAGAGTACCACTCACGTGTGATATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1023_PB", "AAGCAGTGGTATCAACGCAGAGTACCAGAGAGATATCTCTGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
    ("dT_BC1024_PB", "AAGCAGTGGTATCAACGCAGAGTACCATGTAGAGCAGAGAGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTVN"),
]

BARCODE_LENGTH = 16  # bases [25:41] of every dT primer


def _barcode_name(primer_id: str) -> str:
    """`dT_BC1001_PB` -> `BC01`. Mirrors NanoporeAnalysis's slicing
    (process_sam.py: `f"BC{p[7:9]}"`)."""
    return f"BC{primer_id[7:9]}"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "constellation" / "data" / "sequencing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cdna_wilburn_v1.json"

    # Sanity-check the prefix assumption — every dT primer must start
    # with DT_PRIMER_PREFIX. If this ever fails, the primer panel has
    # diverged and the slicing-based barcode extraction needs updating.
    for primer_id, seq in DT_PRIMERS:
        if not seq.startswith(DT_PRIMER_PREFIX):
            raise ValueError(
                f"primer {primer_id!r} does not start with the expected "
                f"prefix {DT_PRIMER_PREFIX!r}"
            )

    primer_3 = _reverse_complement(DT_PRIMER_PREFIX)
    barcodes = []
    for primer_id, seq in DT_PRIMERS:
        bc_name = _barcode_name(primer_id)
        bc_seq = seq[len(DT_PRIMER_PREFIX) : len(DT_PRIMER_PREFIX) + BARCODE_LENGTH]
        if len(bc_seq) != BARCODE_LENGTH:
            raise ValueError(
                f"barcode for {primer_id!r} has length {len(bc_seq)} != {BARCODE_LENGTH}"
            )
        barcodes.append(
            {
                "name": bc_name,
                "sequence": bc_seq,
                "source_primer": primer_id,
            }
        )

    doc = {
        "schema_version": "1",
        "name": "cdna_wilburn_v1",
        "version": 1,
        "chemistry": "tso_pcr_cdna",
        "description": (
            "In-house TSO-based PCR-cDNA construct used by the Wilburn "
            "lab for ONT long-read transcriptomics. Same chemistry "
            "family as IsoSeq (PacBio) and SMARTer (Takara) — generic "
            "term: template-switching reverse transcription + PCR. "
            "24-barcode panel (BC01-BC24) embedded in oligo-dT primers."
        ),
        "source": (
            "Sequences from NanoporeAnalysis/transcriptome_nf/scripts/"
            "primers.txt; verbatim panel the lab has been running in "
            "production with NanoporeAnalysis."
        ),
        "ssp": SSP,
        "primer_3": primer_3,
        "barcode_length": BARCODE_LENGTH,
        "barcodes": barcodes,
    }
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote panel cdna_wilburn_v1 ({len(barcodes)} barcodes) → {out_path}")


if __name__ == "__main__":
    main()
