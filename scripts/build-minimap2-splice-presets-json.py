"""Generate constellation/data/sequencing/minimap2_splice_presets.json.

Hand-curated organism-class presets for minimap2's splice-aware mode
(``-ax splice``). The presets bound the spurious-junction behaviour
that minimap2's stock defaults invite on small, gene-dense genomes —
notably the ``-G 200000`` default that lets the splice DP collapse
adjacent genes into one read with a 200 kb "intron".

Three v1 profiles:

    compact_eukaryote   genome ≤50 Mb, mean intron 100–300 bp
    intermediate_eukaryote  genome 50–500 Mb, mean intron ~250 bp
                            (long-tail to ~10 kb)
    animal              genome ≥1 Gb, mean intron 1–10 kb

Each profile is a small dict of (flag, value) pairs that the resolver
in ``constellation/sequencing/align/presets.py`` composes on top of
the hardcoded base args (``-ax splice -uf --cs=long --secondary=no``).

Run from project root:

    python3 scripts/build-minimap2-splice-presets-json.py
"""

from __future__ import annotations

import json
from pathlib import Path


# Each preset is an ordered list of (flag, value) pairs (a JSON array of
# 2-element arrays), preserving the order minimap2 sees on the command
# line. This makes diffs in the JSON readable and keeps the resolver
# deterministic.
PRESETS = [
    {
        "id": "compact_eukaryote",
        "name": "Compact eukaryote (yeast, fungi, microsporidia)",
        "description": (
            "Genome ≤50 Mb, mean intron ~100–300 bp, gene density "
            "~one ORF every 1–2 kb. Caps spurious intron length at "
            "5 kb (vs. minimap2's default 200 kb) and adds a small "
            "non-canonical splice penalty so the DP prefers GT-AG / "
            "GC-AG / AT-AC junctions over cryptic motifs."
        ),
        "applies_to_examples": [
            "Pichia pastoris",
            "Saccharomyces cerevisiae",
            "Schizosaccharomyces pombe",
            "Candida albicans",
            "Aspergillus nidulans",
            "Neurospora crassa",
        ],
        "minimap2_flags": [
            ["-G", "5000"],
            ["-C", "5"],
        ],
    },
    {
        "id": "intermediate_eukaryote",
        "name": "Intermediate eukaryote (compact plants, invertebrates)",
        "description": (
            "Genome ~50–500 Mb, mean intron ~250 bp with a long tail "
            "to ~10 kb. Caps at 50 kb — enough headroom for the largest "
            "real introns in these organisms while still rejecting "
            "the multi-hundred-kb spurious jumps that minimap2's stock "
            "defaults invite."
        ),
        "applies_to_examples": [
            "Drosophila melanogaster",
            "Caenorhabditis elegans",
            "Arabidopsis thaliana",
            "Oryza sativa",
            "Brachypodium distachyon",
        ],
        "minimap2_flags": [
            ["-G", "50000"],
            ["-C", "5"],
        ],
    },
    {
        "id": "animal",
        "name": "Vertebrate / long-introned animal",
        "description": (
            "Genome ≥1 Gb, mean intron ~1–10 kb with introns up to "
            "~1 Mb. Matches minimap2's stock splice-preset cap (200 kb) "
            "plus a small non-canonical penalty so motif-violating "
            "junctions don't win on score alone."
        ),
        "applies_to_examples": [
            "Homo sapiens",
            "Mus musculus",
            "Rattus norvegicus",
            "Danio rerio",
            "Xenopus tropicalis",
        ],
        "minimap2_flags": [
            ["-G", "200000"],
            ["-C", "5"],
        ],
    },
]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = (
        repo_root
        / "constellation"
        / "data"
        / "sequencing"
        / "minimap2_splice_presets.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": "1",
        "name": "minimap2_splice_presets",
        "source": (
            "Curated by the Wilburn lab from minimap2 docs "
            "(https://lh3.github.io/minimap2/minimap2.html) and per-clade "
            "intron-length distributions in Ensembl / Ensembl Genomes / "
            "RefSeq. Each profile is a small set of flag overrides intended "
            "to compose on top of the splice-mode base args "
            "(-ax splice -uf --cs=long --secondary=no)."
        ),
        "presets": PRESETS,
    }
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {len(PRESETS)} minimap2 splice presets → {out_path}")


if __name__ == "__main__":
    main()
