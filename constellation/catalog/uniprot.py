"""UniProt reference proteomes catalog.

Source: ``https://ftp.uniprot.org/pub/databases/uniprot/current_release/
knowledgebase/reference_proteomes/README``

The README documents proteome IDs (UP000…) with their organism and
taxid; the per-domain ``Eukaryota/`` / ``Bacteria/`` / ``Archaea/`` /
``Viruses/`` subdirectories hold per-proteome FASTA + GFF + idmapping.

This v1 parser reads UniProt's tabular ``proteomes.txt`` listing (one
row per reference proteome, tab-delimited) when available. Catalog
rows set ``source="uniprot"``, ``gff_url`` to ``null`` for the v1 cut,
and ``protein_url`` to the canonical FASTA path.

Per the PR scope decision: UniProt entries are catalogued and exposed
via ``catalog show`` but ``reference fetch`` does not materialise
protein FASTAs in this PR. Follow-up PR-C designs the
``ProteomeReference`` container that fetches and refolds them.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from constellation.catalog._http import http_get_text
from constellation.catalog.schemas import ASSEMBLY_CATALOG_TABLE
from constellation.catalog.types import CatalogRow


_UNIPROT_BASE = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/reference_proteomes"
)


def proteomes_index_url() -> str:
    """Canonical proteomes README path."""
    return f"{_UNIPROT_BASE}/README"


def fetch_index(timeout: int = 300) -> str:
    return http_get_text(proteomes_index_url(), timeout=timeout)


def parse_index(body: str, *, release: str) -> pa.Table:
    """Parse the proteomes README and return ``ASSEMBLY_CATALOG_TABLE`` rows.

    UniProt's README has a tabular section beginning with a header
    ``Proteome_ID\\tTax_ID\\tOSCODE\\tSUPERREGNUM\\t#(1)\\t#(2)\\t#(3)\\tSpecies Name``
    followed by one row per reference proteome. We scan for that
    header and parse subsequent lines until the next blank line.
    """
    rows = _parse_rows(body, release=release)
    return _rows_to_table(rows)


def fetch_catalog(
    release: str,
    *,
    timeout: int = 300,
) -> tuple[pa.Table, dict[str, Any]]:
    body = fetch_index(timeout=timeout)
    table = parse_index(body, release=release)
    meta = {
        "source": "uniprot",
        "release": release,
        "source_url": proteomes_index_url(),
        "n_rows": table.num_rows,
    }
    return table, meta


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


_HEADER_TOKENS = ("Proteome_ID", "Tax_ID")


def _parse_rows(body: str, *, release: str) -> list[CatalogRow]:
    rows: list[CatalogRow] = []
    in_table = False
    catalog_id = 0
    for raw in body.splitlines():
        if not in_table:
            if all(tok in raw for tok in _HEADER_TOKENS):
                in_table = True
            continue
        if not raw.strip():
            # End of the tabular section.
            in_table = False
            continue
        fields = [c.strip() for c in raw.split("\t")]
        if len(fields) < 4:
            continue
        proteome_id = fields[0]
        if not proteome_id.startswith("UP"):
            continue
        taxid = _maybe_int(fields[1])
        super_regnum = fields[3] if len(fields) > 3 else ""
        species_name = fields[7] if len(fields) > 7 else ""
        slug = _slugify(species_name) or proteome_id.lower()
        domain_dir = _domain_dir(super_regnum)
        fasta = (
            f"{_UNIPROT_BASE}/{domain_dir}/{proteome_id}/"
            f"{proteome_id}_{taxid}.fasta.gz"
        ) if taxid is not None else None
        if fasta is None:
            continue
        rows.append(
            CatalogRow(
                catalog_id=catalog_id,
                source="uniprot",
                release=release,
                taxid=taxid,
                species_name=species_name or proteome_id,
                organism_slug=slug,
                assembly_accession=proteome_id,
                assembly_name=None,
                assembly_level=None,
                refseq_category=None,
                annotation_release=None,
                fasta_url=fasta,
                gff_url=None,
                cdna_url=None,
                protein_url=fasta,
                checksums_url=None,
                checksums_kind=None,
                division=super_regnum.lower() or None,
            )
        )
        catalog_id += 1
    return rows


def _domain_dir(super_regnum: str) -> str:
    s = super_regnum.lower()
    if s.startswith("euk"):
        return "Eukaryota"
    if s.startswith("bact"):
        return "Bacteria"
    if s.startswith("arch"):
        return "Archaea"
    if s.startswith("vir"):
        return "Viruses"
    return super_regnum or "Eukaryota"


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
    prev = False
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
            prev = False
        else:
            if not prev:
                out.append("_")
                prev = True
    return "".join(out).strip("_")


__all__ = [
    "fetch_catalog",
    "fetch_index",
    "parse_index",
    "proteomes_index_url",
]
