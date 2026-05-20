"""Ensembl Genomes catalog (fungi / plants / metazoa / protists).

Source: ``http://ftp.ensemblgenomes.org/pub/<division>/release-<N>/
species_Ensembl<Division>.txt`` per division.

Shares the parse path with the vertebrate catalog (same column layout)
but maps to different FTP roots and a per-division ``division`` field.

Plain HTTP (not HTTPS): the host serves a TLS certificate that does not
match ``ftp.ensemblgenomes.org`` (verified 2026-05; CN belongs to EBI's
mirror infrastructure). The data is public read-only, no auth or secrets
in flight, so plain HTTP is the lower-risk choice over disabling TLS
verification — matches the convention used by samtools / wget / the NCBI
datasets CLI for these public FTP archives. Switch to a stable HTTPS
mirror (e.g. ``https://ftp.ebi.ac.uk/ensemblgenomes/pub/``) when one is
identified.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from constellation.catalog._http import http_get_text
from constellation.catalog.schemas import ASSEMBLY_CATALOG_TABLE
from constellation.catalog.types import CatalogRow


_EG_BASE = "http://ftp.ensemblgenomes.org/pub"
_EG_REST_VERSION = (
    "https://rest.ensembl.org/info/eg_version?content-type=application/json"
)
DIVISIONS = ("fungi", "plants", "metazoa", "protists")
_DIVISION_TAG = {
    "fungi": "Fungi",
    "plants": "Plants",
    "metazoa": "Metazoa",
    "protists": "Protists",
}


def species_txt_url(division: str, release: int) -> str:
    tag = _DIVISION_TAG[division]
    return f"{_EG_BASE}/{division}/release-{release}/species_Ensembl{tag}.txt"


def latest_release(*, timeout: int = 60) -> int:
    """Probe Ensembl's REST API for the current Ensembl Genomes release.

    Uses ``rest.ensembl.org/info/eg_version`` — Ensembl's canonical,
    programmatically-stable endpoint for the currently published Ensembl
    Genomes release. Preferred over scraping the FTP directory because:

    - The FTP host (``ftp.ensemblgenomes.org``) serves a TLS cert that
      doesn't match its hostname, so HTTPS probes fail verification.
    - Per-division ``current_README`` files (``pub/<division>/current_README``)
      return 404 — that layout no longer exists on the EG mirror.
    - Scraping the listing for ``release-N`` directories picks up staged
      next-release directories whose species manifests 403.

    All four divisions (fungi/plants/metazoa/protists) ship from the same
    release number, so a single REST call suffices.
    """
    body = http_get_text(_EG_REST_VERSION, timeout=timeout)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"could not parse Ensembl Genomes release from {_EG_REST_VERSION}: "
            f"non-JSON response ({exc})"
        ) from exc
    version = data.get("version")
    if version is None:
        raise RuntimeError(
            f"Ensembl REST {_EG_REST_VERSION} returned no 'version' field"
        )
    return int(version)


def fetch_species_txt(division: str, release: int, *, timeout: int = 120) -> str:
    return http_get_text(species_txt_url(division, release), timeout=timeout)


def parse_species_txt(body: str, *, division: str, release: int) -> pa.Table:
    rows = _parse_rows(body, division=division, release=release)
    return _rows_to_table(rows)


def fetch_catalog(
    release: int,
    *,
    divisions: tuple[str, ...] = DIVISIONS,
    timeout: int = 120,
) -> tuple[pa.Table, dict[str, Any]]:
    """Pull species lists across the requested divisions, concatenated."""
    parts: list[pa.Table] = []
    urls: list[str] = []
    for div in divisions:
        body = fetch_species_txt(div, release, timeout=timeout)
        parts.append(parse_species_txt(body, division=div, release=release))
        urls.append(species_txt_url(div, release))
    if parts:
        table = pa.concat_tables(parts)
    else:
        table = pa.table([], schema=ASSEMBLY_CATALOG_TABLE)
    meta = {
        "source": "ensembl_genomes",
        "release": str(release),
        "source_urls": list(urls),
        "divisions": list(divisions),
        "n_rows": table.num_rows,
    }
    return table, meta


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


def _parse_rows(body: str, *, division: str, release: int) -> list[CatalogRow]:
    rows: list[CatalogRow] = []
    header: list[str] | None = None
    for catalog_id, raw in enumerate(body.splitlines()):
        if not raw.strip():
            continue
        if raw.startswith("#"):
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
        fasta = (
            f"{_EG_BASE}/{division}/release-{release}/fasta/{species_path}/dna/"
            f"{species_path.capitalize()}.{assembly}.dna.toplevel.fa.gz"
        )
        gff = (
            f"{_EG_BASE}/{division}/release-{release}/gff3/{species_path}/"
            f"{species_path.capitalize()}.{assembly}.{release}.gff3.gz"
        )
        cdna = (
            f"{_EG_BASE}/{division}/release-{release}/fasta/{species_path}/cdna/"
            f"{species_path.capitalize()}.{assembly}.cdna.all.fa.gz"
        )
        protein = (
            f"{_EG_BASE}/{division}/release-{release}/fasta/{species_path}/pep/"
            f"{species_path.capitalize()}.{assembly}.pep.all.fa.gz"
        )
        checksums = (
            f"{_EG_BASE}/{division}/release-{release}/fasta/{species_path}/dna/CHECKSUMS"
        )
        rows.append(
            CatalogRow(
                catalog_id=catalog_id,
                source="ensembl_genomes",
                release=str(release),
                taxid=taxid,
                species_name=species_name,
                organism_slug=slug,
                assembly_accession=assembly_accession,
                assembly_name=assembly or None,
                assembly_level=None,
                refseq_category=None,
                annotation_release=str(release),
                fasta_url=fasta,
                gff_url=gff,
                cdna_url=cdna,
                protein_url=protein,
                checksums_url=checksums,
                checksums_kind="ensembl",
                division=division,
            )
        )
    return rows


def _rows_to_table(rows: list[CatalogRow]) -> pa.Table:
    fields = ASSEMBLY_CATALOG_TABLE.names
    cols: dict[str, list[Any]] = {f: [] for f in fields}
    for row in rows:
        for f in fields:
            cols[f].append(getattr(row, f))
    return pa.table(
        {
            name: pa.array(values, type=ASSEMBLY_CATALOG_TABLE.field(name).type)
            for name, values in cols.items()
        },
        schema=ASSEMBLY_CATALOG_TABLE,
    )


def _maybe_int(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _humanize(species_path: str) -> str:
    parts = species_path.split("_")
    if not parts:
        return species_path
    head = parts[0].capitalize()
    tail = " ".join(parts[1:])
    return f"{head} {tail}".strip()


__all__ = [
    "DIVISIONS",
    "fetch_catalog",
    "fetch_species_txt",
    "latest_release",
    "parse_species_txt",
    "species_txt_url",
]
