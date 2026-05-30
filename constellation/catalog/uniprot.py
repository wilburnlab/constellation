"""UniProt reference proteomes catalog + SwissProt FASTA fetcher.

Two surfaces:

  * ``fetch_catalog`` / ``parse_index`` — the v1 metadata catalog
    (reads UniProt's ``proteomes.txt`` listing). Catalogs the
    reference-proteome universe; does not materialise FASTAs.
  * ``fetch_swissprot`` — back-compat shim that materialises the full
    Swiss-Prot FASTA via the standard reference portal
    (``swissprot@uniprot-<release>`` handle). The orchestrator + viz
    layer no longer call this directly — they go through
    ``Reference.open("swissprot")`` after a ``constellation reference
    fetch uniprot:swissprot`` install. The shim stays so any external
    scripts that imported it keep working.

The catalog source: ``https://ftp.uniprot.org/pub/databases/uniprot/
current_release/knowledgebase/reference_proteomes/README``.

The Swiss-Prot FASTA lands at the standard portal layout:
``~/.constellation/references/swissprot/uniprot-<release>/protein.faa``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path
from typing import Any

import pyarrow as pa

from constellation import __version__ as _CONSTELLATION_VERSION  # noqa: F401  (kept for back-compat)
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


# ──────────────────────────────────────────────────────────────────────
# Swiss-Prot FASTA fetcher (back-compat shim)
# ──────────────────────────────────────────────────────────────────────


_SWISSPROT_BASE = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete"
)
_SWISSPROT_FASTA_NAME = "uniprot_sprot.fasta.gz"
_SWISSPROT_RELDATE_NAME = "reldate.txt"

_RELEASE_RE = re.compile(r"Release\s+(\d{4}_\d{2})")

# Optional override base for unit testing — mirrors the
# ``_TEST_HTTP_BASE_OVERRIDE`` pattern in ``sequencing/reference/fetch.py``.
_TEST_HTTP_BASE_OVERRIDE: str | None = None


def _swissprot_base() -> str:
    if _TEST_HTTP_BASE_OVERRIDE is not None:
        return _TEST_HTTP_BASE_OVERRIDE
    return _SWISSPROT_BASE


@dataclasses.dataclass(frozen=True, slots=True)
class SwissprotHandle:
    """A materialised SwissProt FASTA (back-compat shape).

    ``fasta_path`` is the gunzipped FASTA on disk (the standard
    ``protein.faa`` under the portal cache layout, NOT the legacy
    ``sprot.fasta``). ``release`` is the UniProt release id
    (``YYYY_NN`` format). ``sha256`` is computed on the gunzipped FASTA
    bytes. ``source_url`` is the upstream FASTA URL the bytes came
    from.

    New code should prefer ``Reference.open("swissprot")`` directly;
    this struct stays as the return shape of the ``fetch_swissprot``
    back-compat shim for external callers.
    """

    fasta_path: Path
    release: str
    sha256: str
    source_url: str


def swissprot_fasta_url() -> str:
    return f"{_swissprot_base()}/{_SWISSPROT_FASTA_NAME}"


def swissprot_reldate_url() -> str:
    return f"{_swissprot_base()}/{_SWISSPROT_RELDATE_NAME}"


def _probe_swissprot_release(*, timeout: int = 60) -> str:
    """Parse the UniProt ``reldate.txt`` and extract the release id.

    The file's first content line is
    ``UniProt Knowledgebase Release YYYY_NN consists of:``.
    """
    body = http_get_text(swissprot_reldate_url(), timeout=timeout)
    m = _RELEASE_RE.search(body)
    if m is None:
        raise ValueError(
            f"could not parse UniProt release from {swissprot_reldate_url()}: "
            f"first 200 chars:\n{body[:200]!r}"
        )
    return m.group(1)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_swissprot(
    *,
    release: str | None = None,
    timeout: int = 600,
    force: bool = False,
) -> SwissprotHandle:
    """Back-compat shim: materialise SwissProt via the standard portal.

    Delegates to ``fetch_reference`` against the synthetic
    ``swissprot@uniprot-<release>`` handle, which routes through the
    portal's proteome-only path (writes ``protein.faa`` + ``meta.toml``
    with ``has_genome=false`` / ``has_proteome=true``).

    New code should prefer ``Reference.open("swissprot")`` directly
    against an installed cache (set up once via ``constellation
    reference fetch uniprot:swissprot``). This shim is retained solely
    for back-compat with external scripts.

    Parameters
    ----------
    release
        UniProt release id (``YYYY_NN``). ``None`` (default) auto-detects
        via ``reldate.txt``.
    timeout
        HTTP timeout in seconds.
    force
        When ``True``, re-fetch even if the cache slot is complete.

    Returns
    -------
    SwissprotHandle
        Carries the on-disk FASTA path (``protein.faa`` under the
        portal layout) + release + sha256 + source URL.
    """
    # Late imports to avoid circular deps with the sequencing layer.
    from constellation.sequencing.reference.fetch import (
        _ResolvedSpec,
        _fetch_proteome_only,
    )
    from constellation.sequencing.reference.handle import Handle

    if release is None:
        release = _probe_swissprot_release(timeout=timeout)

    source_url = swissprot_fasta_url()
    handle = Handle(organism="swissprot", source="uniprot", release=release)
    resolved = _ResolvedSpec(
        handle=handle,
        fasta_url="",
        gff_url="",
        checksums_url=None,
        checksums_kind=None,
        assembly_name=None,
        annotation_release=None,
        assembly_accession=None,
        taxid=None,
        scientific_name=None,
        strain=None,
        protein_url=source_url,
        cdna_url=None,
    )

    result = _fetch_proteome_only(
        resolved,
        handle,
        spec="swissprot",
        output_dir=None,
        timeout=timeout,
        use_cache=True,
        force=force,
    )
    fasta_path = result.protein_fasta_path
    if fasta_path is None:
        raise RuntimeError(
            f"internal: SwissProt proteome-only fetch returned no "
            f"protein_fasta_path for handle {handle}"
        )
    return SwissprotHandle(
        fasta_path=fasta_path,
        release=release,
        sha256=_sha256_of(fasta_path),
        source_url=source_url,
    )


__all__ = [
    "SwissprotHandle",
    "fetch_catalog",
    "fetch_index",
    "fetch_swissprot",
    "parse_index",
    "proteomes_index_url",
    "swissprot_fasta_url",
    "swissprot_reldate_url",
]
