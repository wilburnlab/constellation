"""RefSeq assembly catalog.

Source: ``https://ftp.ncbi.nlm.nih.gov/genomes/refseq/assembly_summary_refseq.txt``
(tab-delimited; header on line 2 with a ``#`` prefix). For narrower
catalogs that load faster, callers can target a per-clade summary:
``/genomes/refseq/<clade>/assembly_summary.txt`` (e.g.
``vertebrate_mammalian``, ``plant``, ``invertebrate``).

Columns we read (NCBI documents 24 columns total):

    assembly_accession, bioproject, biosample, wgs_master,
    refseq_category, taxid, species_taxid, organism_name,
    infraspecific_name, isolate, version_status, assembly_level,
    release_type, genome_rep, seq_rel_date, asm_name, submitter,
    gbrs_paired_asm, paired_asm_comp, ftp_path, excluded_from_refseq,
    relation_to_type_material, asm_not_live_date, assembly_type,
    group, ...

The ``ftp_path`` column gives the canonical assembly directory; we
derive ``_genomic.fna.gz`` / ``_genomic.gff.gz`` /
``_rna_from_genomic.fna.gz`` / ``_protein.faa.gz`` filenames from the
directory name.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from constellation.catalog._http import http_get_text
from constellation.catalog.schemas import ASSEMBLY_CATALOG_TABLE
from constellation.catalog.types import CatalogRow


_REFSEQ_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq"


def summary_url(clade: str | None = None) -> str:
    if clade is None:
        return f"{_REFSEQ_BASE}/assembly_summary_refseq.txt"
    return f"{_REFSEQ_BASE}/{clade}/assembly_summary.txt"


def fetch_summary(clade: str | None = None, *, timeout: int = 300) -> str:
    return http_get_text(summary_url(clade), timeout=timeout)


def parse_summary(body: str) -> pa.Table:
    rows = _parse_rows(body)
    return _rows_to_table(rows)


def fetch_catalog(
    clade: str | None = None,
    *,
    release_tag: str | None = None,
    timeout: int = 300,
) -> tuple[pa.Table, dict[str, Any]]:
    body = fetch_summary(clade, timeout=timeout)
    table = parse_summary(body)
    from datetime import datetime, timezone

    rel = release_tag or datetime.now(timezone.utc).strftime("%Y%m%d")
    meta = {
        "source": "refseq",
        "release": rel,
        "source_url": summary_url(clade),
        "clade": clade or "all",
        "n_rows": table.num_rows,
    }
    return table, meta


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


def _parse_rows(body: str) -> list[CatalogRow]:
    rows: list[CatalogRow] = []
    header: list[str] | None = None
    catalog_id = 0
    for raw in body.splitlines():
        if not raw.strip():
            continue
        if raw.startswith("##"):
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
        accession = rec.get("assembly_accession") or ""
        if not accession:
            continue
        ftp_path = rec.get("ftp_path") or ""
        if not ftp_path:
            continue
        # FTP paths come in as ``https://ftp.ncbi.nlm.nih.gov/...``; some
        # historical rows use ``ftp://`` — normalise to https.
        if ftp_path.startswith("ftp://"):
            ftp_path = "https://" + ftp_path[len("ftp://"):]
        ftp_path = ftp_path.rstrip("/")
        asm_dir = ftp_path.rsplit("/", 1)[-1]
        species_name = rec.get("organism_name") or ""
        taxid = _maybe_int(rec.get("species_taxid") or rec.get("taxid", ""))
        refseq_category = rec.get("refseq_category") or None
        assembly_level = rec.get("assembly_level") or None
        assembly_name = rec.get("asm_name") or None
        group = rec.get("group") or None
        fasta = f"{ftp_path}/{asm_dir}_genomic.fna.gz"
        gff = f"{ftp_path}/{asm_dir}_genomic.gff.gz"
        rna = f"{ftp_path}/{asm_dir}_rna_from_genomic.fna.gz"
        protein = f"{ftp_path}/{asm_dir}_protein.faa.gz"
        checksums = f"{ftp_path}/md5checksums.txt"
        rows.append(
            CatalogRow(
                catalog_id=catalog_id,
                source="refseq",
                release=accession,
                taxid=taxid,
                species_name=species_name or accession,
                organism_slug=_slugify(species_name) or accession.lower(),
                assembly_accession=accession,
                assembly_name=assembly_name,
                assembly_level=assembly_level,
                refseq_category=refseq_category,
                annotation_release=None,
                fasta_url=fasta,
                gff_url=gff,
                cdna_url=rna,
                protein_url=protein,
                checksums_url=checksums,
                checksums_kind="refseq",
                division=group,
            )
        )
        catalog_id += 1
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


def _slugify(name: str) -> str:
    out: list[str] = []
    prev_underscore = False
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                out.append("_")
                prev_underscore = True
    return "".join(out).strip("_")


__all__ = ["fetch_catalog", "fetch_summary", "parse_summary", "summary_url"]
