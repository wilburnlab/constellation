"""Ensembl (vertebrate) genome catalog.

Source: ``https://ftp.ensembl.org/pub/release-<N>/species.txt``
(tab-delimited; columns documented in the README at that path).

Fields in the file (Ensembl 100+):

    name, species, division, taxonomy_id, assembly, assembly_accession,
    genebuild, variation, microarray, pan_compara, peptide_compara,
    genome_alignments, other_alignments, core_db, species_id

We map this to ``ASSEMBLY_CATALOG_TABLE`` rows with URLs derived from
the Ensembl directory layout (``release-<N>/fasta/<species>/dna/``,
``release-<N>/gff3/<species>/``, ``release-<N>/fasta/<species>/cdna/``,
``release-<N>/fasta/<species>/pep/``). FASTA filenames follow
``<Genus_species>.<AssemblyName>.<dna|cdna.all|pep.all>.fa.gz`` so we
construct the canonical name without per-row HTTP probes.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from constellation.catalog._http import http_get_text
from constellation.catalog.schemas import ASSEMBLY_CATALOG_TABLE
from constellation.catalog.types import CatalogRow


_ENSEMBL_PUB = "https://ftp.ensembl.org/pub"


def species_txt_url(release: int) -> str:
    return f"{_ENSEMBL_PUB}/release-{release}/species_EnsemblVertebrates.txt"


def fetch_species_txt(release: int, *, timeout: int = 120) -> str:
    return http_get_text(species_txt_url(release), timeout=timeout)


def parse_species_txt(body: str, *, release: int) -> pa.Table:
    """Parse an Ensembl ``species_EnsemblVertebrates.txt`` body into Arrow."""
    rows = _parse_rows(body, release=release)
    return _rows_to_table(rows)


def fetch_catalog(release: int, *, timeout: int = 120) -> tuple[pa.Table, dict[str, Any]]:
    body = fetch_species_txt(release, timeout=timeout)
    table = parse_species_txt(body, release=release)
    meta = {
        "source": "ensembl",
        "release": str(release),
        "source_url": species_txt_url(release),
        "n_rows": table.num_rows,
    }
    return table, meta


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


def _parse_rows(body: str, *, release: int) -> list[CatalogRow]:
    rows: list[CatalogRow] = []
    header: list[str] | None = None
    for catalog_id, raw in enumerate(body.splitlines()):
        if not raw.strip():
            continue
        if raw.startswith("#"):
            # Ensembl files have a single ``#name\tspecies\t...`` header.
            header_line = raw.lstrip("# ").strip()
            header = [c.strip() for c in header_line.split("\t")]
            continue
        if header is None:
            continue
        fields = [c.strip() for c in raw.split("\t")]
        if len(fields) < len(header):
            fields = fields + [""] * (len(header) - len(fields))
        rec = dict(zip(header, fields))
        species_path = rec.get("species") or rec.get("name") or ""
        if not species_path:
            continue
        assembly = rec.get("assembly") or ""
        assembly_accession = rec.get("assembly_accession") or None
        taxid = _maybe_int(rec.get("taxonomy_id", ""))
        species_name = _humanize(species_path)
        slug = species_path.lower()
        base = (
            f"{_ENSEMBL_PUB}/release-{release}/fasta/{species_path}/dna/"
            f"{species_path.capitalize()}.{assembly}.dna.toplevel.fa.gz"
        )
        gff = (
            f"{_ENSEMBL_PUB}/release-{release}/gff3/{species_path}/"
            f"{species_path.capitalize()}.{assembly}.{release}.gff3.gz"
        )
        cdna = (
            f"{_ENSEMBL_PUB}/release-{release}/fasta/{species_path}/cdna/"
            f"{species_path.capitalize()}.{assembly}.cdna.all.fa.gz"
        )
        protein = (
            f"{_ENSEMBL_PUB}/release-{release}/fasta/{species_path}/pep/"
            f"{species_path.capitalize()}.{assembly}.pep.all.fa.gz"
        )
        checksums = (
            f"{_ENSEMBL_PUB}/release-{release}/fasta/{species_path}/dna/CHECKSUMS"
        )
        rows.append(
            CatalogRow(
                catalog_id=catalog_id,
                source="ensembl",
                release=str(release),
                taxid=taxid,
                species_name=species_name,
                organism_slug=slug,
                assembly_accession=assembly_accession,
                assembly_name=assembly or None,
                assembly_level=None,
                refseq_category=None,
                annotation_release=str(release),
                fasta_url=base,
                gff_url=gff,
                cdna_url=cdna,
                protein_url=protein,
                checksums_url=checksums,
                checksums_kind="ensembl",
                division=str(rec.get("division") or "EnsemblVertebrates"),
            )
        )
    return rows


def _rows_to_table(rows: list[CatalogRow]) -> pa.Table:
    fields = ASSEMBLY_CATALOG_TABLE.names
    cols: dict[str, list[Any]] = {f: [] for f in fields}
    for row in rows:
        for f in fields:
            cols[f].append(getattr(row, f))
    pa_cols = {
        name: pa.array(values, type=ASSEMBLY_CATALOG_TABLE.field(name).type)
        for name, values in cols.items()
    }
    return pa.table(pa_cols, schema=ASSEMBLY_CATALOG_TABLE)


def _maybe_int(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _humanize(species_path: str) -> str:
    """``homo_sapiens`` → ``Homo sapiens``."""
    parts = species_path.split("_")
    if not parts:
        return species_path
    head = parts[0].capitalize()
    tail = " ".join(parts[1:])
    return f"{head} {tail}".strip()


__all__ = [
    "fetch_catalog",
    "fetch_species_txt",
    "parse_species_txt",
    "species_txt_url",
]
