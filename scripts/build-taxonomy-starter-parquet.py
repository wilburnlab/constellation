#!/usr/bin/env python3
"""Regenerate ``constellation/data/taxonomy_starter.parquet``.

Two modes:

* **bundled v1 hand-curated** (default, no args): a small starter
  (~100 species across the tree of life + their full lineages) that
  unblocks the manual CLI tests and lab-priority taxa without
  requiring a fresh NCBI taxdump. Lab targets + key model organisms
  are baked into the dict literals below.

* **--taxdump <path>**: full v1 curation rule against a fetched
  ``taxdump.tar.gz``. Admits every species T such that:
  (a) T is in any Ensembl release;
  (b) T has a RefSeq reference_genome or representative_genome
      (caller must also pass --refseq-summary <path>);
  (c) T falls under a curated clade in the whitelist below
      (expanded via taxonomy.descendants()).
  Plus every ancestor of any admitted taxon, closure-wise.

  Example workflow:
    constellation taxonomy update           # populates ~/.constellation/taxonomy/
    python scripts/build-taxonomy-starter-parquet.py \\
        --taxdump ~/.cache/taxdump.tar.gz \\
        --refseq-summary assembly_summary_refseq.txt

Output: ``constellation/data/taxonomy_starter.parquet``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa


# ──────────────────────────────────────────────────────────────────────
# Hand-curated v1 starter — lab targets + canonical model organisms
# ──────────────────────────────────────────────────────────────────────


# Each row: (taxid, parent_taxid, rank, scientific_name, division_id)
# parent_taxid=None for root. division_id from NCBI division.dmp; we use
# only a small subset (0=Bacteria, 1=Invertebrates, 2=Mammals,
# 4=Plants and Fungi, 5=Primates, 9=Viruses, 10=Vertebrates, 11=Env).

_V1_NODES: list[tuple[int, int | None, str, str, int | None]] = [
    # Root + cellular organisms backbone
    (1, None, "no rank", "root", None),
    (131567, 1, "no rank", "cellular organisms", None),
    # ── Eukaryota backbone ────────────────────────────────────────
    (2759, 131567, "superkingdom", "Eukaryota", None),
    (33154, 2759, "clade", "Opisthokonta", None),
    (33208, 33154, "kingdom", "Metazoa", 1),
    (6072, 33208, "clade", "Eumetazoa", 1),
    (33213, 6072, "clade", "Bilateria", 1),
    (33511, 33213, "clade", "Deuterostomia", 1),
    (7711, 33511, "phylum", "Chordata", 10),
    (89593, 7711, "subphylum", "Craniata", 10),
    (7742, 89593, "clade", "Vertebrata", 10),
    (7776, 7742, "clade", "Gnathostomata", 10),
    (117570, 7776, "clade", "Teleostomi", 10),
    (117571, 117570, "clade", "Euteleostomi", 10),
    (8287, 117571, "superclass", "Sarcopterygii", 10),
    (1338369, 8287, "clade", "Dipnotetrapodomorpha", 10),
    (32523, 1338369, "clade", "Tetrapoda", 10),
    (32524, 32523, "clade", "Amniota", 10),
    # ── Mammals ───────────────────────────────────────────────────
    (40674, 32524, "class", "Mammalia", 2),
    (32525, 40674, "clade", "Theria", 2),
    (9347, 32525, "clade", "Eutheria", 2),
    (1437010, 9347, "clade", "Boreoeutheria", 2),
    (314146, 1437010, "superorder", "Euarchontoglires", 2),
    (9443, 314146, "order", "Primates", 5),
    (376913, 9443, "suborder", "Haplorrhini", 5),
    (314293, 376913, "infraorder", "Simiiformes", 5),
    (9526, 314293, "parvorder", "Catarrhini", 5),
    (314295, 9526, "superfamily", "Hominoidea", 5),
    (9604, 314295, "family", "Hominidae", 5),
    (207598, 9604, "subfamily", "Homininae", 5),
    (9605, 207598, "genus", "Homo", 5),
    (9606, 9605, "species", "Homo sapiens", 5),
    # Rodents
    (314147, 314146, "clade", "Glires", 6),
    (9989, 314147, "order", "Rodentia", 6),
    (337687, 9989, "suborder", "Myomorpha", 6),
    (10066, 337687, "superfamily", "Muroidea", 6),
    (10088, 10066, "family", "Muridae", 6),
    (39107, 10088, "subfamily", "Murinae", 6),
    (862507, 39107, "genus", "Mus", 6),
    (10090, 862507, "species", "Mus musculus", 6),
    (10114, 39107, "genus", "Rattus", 6),
    (10116, 10114, "species", "Rattus norvegicus", 6),
    # ── Bony fish ─────────────────────────────────────────────────
    (186623, 117570, "class", "Actinopteri", 10),
    (7898, 186623, "clade", "Neopterygii", 10),
    (32443, 7898, "infraclass", "Teleostei", 10),
    (186634, 32443, "clade", "Otomorpha", 10),
    (186626, 186634, "clade", "Otophysi", 10),
    (7952, 186626, "order", "Cypriniformes", 10),
    (7953, 7952, "family", "Cyprinidae", 10),
    (7954, 7953, "genus", "Danio", 10),
    (7955, 7954, "species", "Danio rerio", 10),
    # ── Amphibia + Plethodontidae ────────────────────────────────
    (8292, 32523, "class", "Amphibia", 10),
    (8293, 8292, "clade", "Batrachia", 10),
    (8294, 8293, "order", "Caudata", 10),
    (54541, 8294, "family", "Plethodontidae", 10),
    (54542, 54541, "subfamily", "Plethodontinae", 10),
    (54545, 54542, "genus", "Plethodon", 10),
    (54549, 54545, "species", "Plethodon cinereus", 10),
    (54552, 54545, "species", "Plethodon glutinosus", 10),
    (323754, 54545, "species", "Plethodon shermani", 10),
    # ── Protostomia backbone ─────────────────────────────────────
    (33317, 33213, "clade", "Protostomia", 1),
    (1206794, 33317, "clade", "Ecdysozoa", 1),
    # Arthropoda → Drosophila
    (88770, 1206794, "clade", "Panarthropoda", 1),
    (6656, 88770, "phylum", "Arthropoda", 1),
    (197563, 6656, "clade", "Mandibulata", 1),
    (197562, 197563, "clade", "Pancrustacea", 1),
    (6960, 197562, "superclass", "Hexapoda", 1),
    (50557, 6960, "class", "Insecta", 1),
    (85512, 50557, "clade", "Dicondylia", 1),
    (7496, 85512, "subclass", "Pterygota", 1),
    (33340, 7496, "infraclass", "Neoptera", 1),
    (33392, 33340, "clade", "Endopterygota", 1),
    (7147, 33392, "order", "Diptera", 1),
    (7203, 7147, "suborder", "Brachycera", 1),
    (43733, 7203, "infraorder", "Muscomorpha", 1),
    (43738, 43733, "superfamily", "Ephydroidea", 1),
    (7214, 43738, "family", "Drosophilidae", 1),
    (43845, 7214, "subfamily", "Drosophilinae", 1),
    (7215, 43845, "genus", "Drosophila", 1),
    (7227, 7215, "species", "Drosophila melanogaster", 1),
    # Nematoda → C. elegans
    (1283342, 1206794, "clade", "Nematoida", 1),
    (6231, 1283342, "phylum", "Nematoda", 1),
    (119089, 6231, "class", "Chromadorea", 1),
    (6236, 119089, "order", "Rhabditida", 1),
    (55879, 6236, "family", "Rhabditidae", 1),
    (6237, 55879, "genus", "Caenorhabditis", 1),
    (6239, 6237, "species", "Caenorhabditis elegans", 1),
    # ── Spiralia / Mollusca / Echinodermata ──────────────────────
    (2697495, 33317, "clade", "Spiralia", 1),
    (1206795, 2697495, "clade", "Lophotrochozoa", 1),
    (6447, 1206795, "phylum", "Mollusca", 1),
    (6448, 6447, "class", "Gastropoda", 1),
    (1063539, 6448, "subclass", "Vetigastropoda", 1),
    (216498, 1063539, "order", "Lepetellida", 1),
    (216499, 216498, "family", "Haliotidae", 1),
    (216500, 216499, "genus", "Haliotis", 1),
    (6454, 216500, "species", "Haliotis rufescens", 1),
    (61441, 216500, "species", "Haliotis discus hannai", 1),
    (164618, 216500, "species", "Haliotis sorenseni", 1),
    (164620, 216500, "species", "Haliotis cracherodii", 1),
    (164622, 216500, "species", "Haliotis fulgens", 1),
    (164626, 216500, "species", "Haliotis corrugata", 1),
    (302850, 216500, "species", "Haliotis midae", 1),
    # Tegula
    (216502, 1063539, "order", "Trochida", 1),
    (216510, 216502, "family", "Tegulidae", 1),
    (231223, 216510, "genus", "Tegula", 1),
    (231224, 231223, "species", "Tegula funebralis", 1),
    (231225, 231223, "species", "Tegula brunnea", 1),
    # Aplysia (Heterobranchia)
    (216572, 6448, "subclass", "Heterobranchia", 1),
    (6494, 216572, "order", "Aplysiida", 1),
    (6499, 6494, "genus", "Aplysia", 1),
    (6500, 6499, "species", "Aplysia californica", 1),
    # Cephalopoda → Octopus (sister to Gastropoda within Mollusca)
    (6605, 6447, "class", "Cephalopoda", 1),
    (32577, 6605, "subclass", "Coleoidea", 1),
    (6645, 32577, "order", "Octopoda", 1),
    (6646, 6645, "family", "Octopodidae", 1),
    (6647, 6646, "genus", "Octopus", 1),
    (6648, 6647, "species", "Octopus vulgaris", 1),
    # Echinodermata under Deuterostomia
    (7586, 33511, "phylum", "Echinodermata", 1),
    (7624, 7586, "class", "Echinoidea", 1),
    (133436, 7624, "order", "Camarodonta", 1),
    (7672, 133436, "family", "Strongylocentrotidae", 1),
    (7666, 7672, "genus", "Strongylocentrotus", 1),
    (7668, 7666, "species", "Strongylocentrotus purpuratus", 1),
    (7587, 7586, "class", "Asteroidea", 1),
    (133448, 7587, "order", "Valvatida", 1),
    (133450, 133448, "family", "Asterinidae", 1),
    (133452, 133450, "genus", "Patiria", 1),
    (133454, 133452, "species", "Patiria miniata", 1),
    # ── Fungi (Saccharomyces only — other lineages join via taxonomy update) ─
    (4751, 33154, "kingdom", "Fungi", 4),
    (451864, 4751, "subkingdom", "Dikarya", 4),
    (4890, 451864, "phylum", "Ascomycota", 4),
    (716545, 4890, "clade", "saccharomyceta", 4),
    (147537, 716545, "subphylum", "Saccharomycotina", 4),
    (4891, 147537, "class", "Saccharomycetes", 4),
    (4892, 4891, "order", "Saccharomycetales", 4),
    (4893, 4892, "family", "Saccharomycetaceae", 4),
    (4930, 4893, "genus", "Saccharomyces", 4),
    (4932, 4930, "species", "Saccharomyces cerevisiae", 4),
    # ── Plants — Arabidopsis ─────────────────────────────────────
    (33090, 2759, "kingdom", "Viridiplantae", 4),
    (35493, 33090, "phylum", "Streptophyta", 4),
    (131221, 35493, "subphylum", "Streptophytina", 4),
    (3193, 131221, "clade", "Embryophyta", 4),
    (58023, 3193, "clade", "Tracheophyta", 4),
    (78536, 58023, "clade", "Euphyllophyta", 4),
    (58024, 78536, "clade", "Spermatophyta", 4),
    (3398, 58024, "class", "Magnoliopsida", 4),
    (1437183, 3398, "clade", "Mesangiospermae", 4),
    (71240, 1437183, "clade", "eudicotyledons", 4),
    (91827, 71240, "clade", "Gunneridae", 4),
    (1437201, 91827, "clade", "Pentapetalae", 4),
    (71275, 1437201, "clade", "rosids", 4),
    (91836, 71275, "clade", "malvids", 4),
    (3699, 91836, "order", "Brassicales", 4),
    (3700, 3699, "family", "Brassicaceae", 4),
    (980083, 3700, "tribe", "Camelineae", 4),
    (3701, 980083, "genus", "Arabidopsis", 4),
    (3702, 3701, "species", "Arabidopsis thaliana", 4),
    # ── Bacteria ─────────────────────────────────────────────────
    (2, 131567, "superkingdom", "Bacteria", 0),
    (1224, 2, "phylum", "Pseudomonadota", 0),
    (1236, 1224, "class", "Gammaproteobacteria", 0),
    (91347, 1236, "order", "Enterobacterales", 0),
    (543, 91347, "family", "Enterobacteriaceae", 0),
    (561, 543, "genus", "Escherichia", 0),
    (562, 561, "species", "Escherichia coli", 0),
    # ── Archaea ──────────────────────────────────────────────────
    (2157, 131567, "superkingdom", "Archaea", 0),
    (28890, 2157, "phylum", "Euryarchaeota", 0),
    # ── Viruses (just the root marker so search-by-name finds it) ─
    (10239, 1, "superkingdom", "Viruses", 9),
]


# Each row: (taxid, name, name_class). The build script auto-adds the
# scientific-name row from _V1_NODES.
_V1_NAMES_EXTRA: list[tuple[int, str, str]] = [
    # Common + vernacular names for high-priority species.
    (9606, "human", "genbank common name"),
    (9606, "man", "common name"),
    (10090, "house mouse", "genbank common name"),
    (10090, "mouse", "common name"),
    (10116, "Norway rat", "genbank common name"),
    (10116, "rat", "common name"),
    (7955, "zebrafish", "genbank common name"),
    (7227, "fruit fly", "genbank common name"),
    (7227, "fly", "common name"),
    (6239, "C. elegans", "common name"),
    (6239, "nematode worm", "common name"),
    (6239, "worm", "common name"),
    (4932, "baker's yeast", "genbank common name"),
    (4932, "brewer's yeast", "common name"),
    (4932, "yeast", "common name"),
    (3702, "thale cress", "genbank common name"),
    (3702, "mouse-ear cress", "common name"),
    (562, "E. coli", "common name"),
    (54549, "red-backed salamander", "genbank common name"),
    (54549, "eastern red-backed salamander", "common name"),
    (54552, "northern slimy salamander", "common name"),
    (6454, "red abalone", "genbank common name"),
    (61441, "Pacific abalone", "common name"),
    (61441, "disk abalone", "common name"),
    (164618, "white abalone", "genbank common name"),
    (164620, "black abalone", "genbank common name"),
    (164622, "green abalone", "common name"),
    (164626, "pink abalone", "common name"),
    (231224, "black turban snail", "common name"),
    (231225, "brown turban snail", "common name"),
    (6500, "California sea hare", "genbank common name"),
    (6648, "common octopus", "genbank common name"),
    (7668, "purple sea urchin", "common name"),
    (133454, "bat star", "common name"),
]


# ──────────────────────────────────────────────────────────────────────
# Builders
# ──────────────────────────────────────────────────────────────────────


def build_hand_curated_starter() -> tuple[pa.Table, pa.Table, pa.Table]:
    """Materialise the hand-curated v1 starter as the three Arrow tables.

    Deduplicates entries by taxid (later wins for accidental dupes in
    the dict literal), validates parent references, and emits one
    ``scientific name`` row per node plus the extra common-name rows.
    """
    # Deduplicate nodes by taxid — last entry wins.
    node_by_id: dict[int, tuple[int, int | None, str, str, int | None]] = {}
    for row in _V1_NODES:
        taxid, parent, rank, sci, div = row
        if taxid == parent:
            # Self-parenting only valid for the root (1); silently skip
            # any other accidental self-parent placeholder rows.
            if taxid != 1:
                continue
        node_by_id[taxid] = row

    # Ensure all parents are present.
    node_ids = set(node_by_id)
    for taxid, parent, _, _, _ in node_by_id.values():
        if parent is None or parent == taxid:
            continue
        if parent not in node_ids:
            raise ValueError(
                f"node {taxid} references parent {parent} not in the starter set"
            )

    sorted_rows = sorted(node_by_id.values(), key=lambda r: r[0])
    nodes_tbl = pa.table(
        {
            "taxid": pa.array([r[0] for r in sorted_rows], type=pa.int64()),
            "parent_taxid": pa.array(
                [r[1] if r[1] is not None else r[0] for r in sorted_rows],
                type=pa.int64(),
            ),
            "rank": pa.array([r[2] for r in sorted_rows], type=pa.string()),
            "division_id": pa.array([r[4] for r in sorted_rows], type=pa.int16()),
            "genetic_code_id": pa.array(
                [None] * len(sorted_rows), type=pa.int16()
            ),
            "mito_genetic_code_id": pa.array(
                [None] * len(sorted_rows), type=pa.int16()
            ),
            "scientific_name": pa.array(
                [r[3] for r in sorted_rows], type=pa.string()
            ),
        }
    )

    # Names: one scientific-name row per node + extras.
    name_rows: list[tuple[int, str, str]] = [
        (r[0], r[3], "scientific name") for r in sorted_rows
    ]
    for tid, nm, klass in _V1_NAMES_EXTRA:
        if tid not in node_ids:
            raise ValueError(f"extra name for unknown taxid {tid}")
        name_rows.append((tid, nm, klass))
    names_tbl = pa.table(
        {
            "taxid": pa.array([r[0] for r in name_rows], type=pa.int64()),
            "name": pa.array([r[1] for r in name_rows], type=pa.string()),
            "name_lower": pa.array(
                [r[1].lower() for r in name_rows], type=pa.string()
            ),
            "name_class": pa.array([r[2] for r in name_rows], type=pa.string()),
            "unique_name": pa.array(
                [None] * len(name_rows), type=pa.string()
            ),
        }
    )

    # Merged: empty for v1.
    merged_tbl = pa.table(
        {
            "old_taxid": pa.array([], type=pa.int64()),
            "new_taxid": pa.array([], type=pa.int64()),
        }
    )
    return nodes_tbl, names_tbl, merged_tbl


def build_from_taxdump(
    taxdump_path: Path,
    *,
    refseq_summary: Path | None = None,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Full v1 curation against a fetched NCBI taxdump.

    Not implemented in v1 — left as a documented stub. Callers wanting
    the comprehensive starter should run
    ``constellation taxonomy update`` instead and use the cached
    bundle directly.
    """
    raise NotImplementedError(
        "full-taxdump starter regeneration deferred to a follow-up PR — "
        "for now, run `constellation taxonomy update` to install the "
        "comprehensive cached bundle and let the resolver auto-prefer it"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate constellation/data/taxonomy_starter.parquet."
    )
    parser.add_argument(
        "--taxdump",
        type=Path,
        default=None,
        help="path to taxdump.tar.gz (enables full-curation mode)",
    )
    parser.add_argument(
        "--refseq-summary",
        type=Path,
        default=None,
        help="path to assembly_summary_refseq.txt (used in full-curation mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output parquet path (defaults to constellation/data/taxonomy_starter.parquet)",
    )
    args = parser.parse_args()

    if args.taxdump is not None:
        nodes, names, merged = build_from_taxdump(
            args.taxdump, refseq_summary=args.refseq_summary
        )
        source = "ncbi-taxdump"
    else:
        nodes, names, merged = build_hand_curated_starter()
        source = "constellation-v1-hand-curated"

    out_path = args.output
    if out_path is None:
        here = Path(__file__).resolve().parent.parent
        out_path = here / "constellation" / "data" / "taxonomy_starter.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from constellation.core.taxonomy.bundled import write_bundled_starter

    write_bundled_starter(
        str(out_path),
        nodes=nodes,
        names=names,
        merged=merged,
        meta={"source": source, "schema_version": "1"},
    )
    print(
        f"wrote {out_path} — {nodes.num_rows} nodes, {names.num_rows} names, "
        f"{merged.num_rows} merged ({out_path.stat().st_size / 1024:.1f} KiB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
