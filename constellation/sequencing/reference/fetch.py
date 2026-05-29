"""Stdlib-HTTP fetchers for Ensembl / Ensembl Genomes / RefSeq.

Thin convenience layer for the 90% case: a user wants the genome FASTA
+ GFF3 for a well-known organism / accession and doesn't want to
hand-curate the URLs. Hits the published FTP/HTTPS endpoints directly,
gunzips inline, and writes a ParquetDir bundle ready for downstream
``constellation reference`` consumers.

No third-party tool deps — no NCBI Datasets CLI, no auth, no signed
URLs. If a user hits an organism/accession outside the URL conventions
encoded here, they fall back to manual download + ``constellation
reference import``.

Default destination is the per-user **reference cache** at
``~/.constellation/references/<organism>/<release>/`` (overridable via
``$CONSTELLATION_REFERENCES_HOME``). The ``--output-dir`` flag adds an
additional copy at the requested location; ``--no-cache --output-dir``
preserves the legacy "scratch fetch" behaviour without populating the
cache.

Supported sources (parse syntax: ``<source>:<id>``):

    ensembl:human            → vertebrate Ensembl release (latest)
    ensembl:mouse            → vertebrate Ensembl release (latest)
    ensembl_genomes:saccharomyces_cerevisiae
                             → fungi/protists/plants/metazoa Ensembl
                               Genomes (auto-resolves division)
    refseq:GCF_001708105.1   → NCBI RefSeq assembly accession
    genbank:GCA_001708105.1  → NCBI GenBank assembly accession

Each source has a small URL-builder function. Add new sources as new
``_SOURCES`` entries.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from constellation import __version__ as _CONSTELLATION_VERSION
from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.annotation.io import save_annotation
from constellation.sequencing.reference import handle as ref_handle
from constellation.sequencing.reference.handle import (
    Handle,
    acquire_fetch_lock,
    cache_root,
    clean_partial,
    partial_dir,
    promote_partial,
    set_default,
    update_current_pointer,
    write_meta_toml,
)
from constellation.sequencing.reference.io import save_genome_reference
from constellation.sequencing.reference.reference import GenomeReference


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Result of a successful reference fetch.

    ``protein_fasta_path`` is the path to a cached ``protein.faa`` (the
    gunzipped ``_protein.faa.gz`` for RefSeq, etc.) when one was fetched
    for this assembly. ``None`` when the source doesn't ship a protein
    FASTA, the URL 404'd, or the cache hit was an older entry written
    before protein FASTA fetching was wired in.

    ``cdna_fasta_path`` mirrors ``protein_fasta_path`` for the
    ``_rna_from_genomic.fna.gz`` (RefSeq) / ``cdna.all.fa.gz`` (Ensembl)
    sibling fetch. Both are opportunistic: a failed fetch yields ``None``
    without failing the overall call.

    ``genome`` is ``None`` for proteome-only installs (UniProt source);
    callers must tolerate this when surfacing fetch results to the user.
    """

    genome: GenomeReference | None
    annotation: Annotation | None
    handle: Handle
    cache_path: Path | None
    output_path: Path | None
    sources: dict[str, str]  # name → URL
    skipped_cache: bool = False  # idempotency short-circuit?
    protein_fasta_path: Path | None = None
    cdna_fasta_path: Path | None = None


# ──────────────────────────────────────────────────────────────────────
# Source URL resolvers
# ──────────────────────────────────────────────────────────────────────


_ENSEMBL_FTP_BASE = "https://ftp.ensembl.org/pub/current_fasta"
_ENSEMBL_GFF3_BASE = "https://ftp.ensembl.org/pub/current_gff3"
_ENSEMBL_PUB_BASE = "https://ftp.ensembl.org/pub"
# Plain HTTP: ftp.ensemblgenomes.org serves a TLS certificate that does
# not match its hostname (verified 2026-05). The data is public, so plain
# HTTP is lower-risk than disabling cert verification. See the matching
# comment in constellation/catalog/ensembl_genomes.py.
_ENSEMBL_GENOMES_FTP_BASE = "http://ftp.ensemblgenomes.org/pub"

# Optional override base for unit testing (set by tests, never by users).
_TEST_HTTP_BASE_OVERRIDE: str | None = None


# Curated species → (ensembl_species_path, common_assembly_string).
# Restricted to the lab's likely targets; extend as needed.
_ENSEMBL_VERTEBRATE_SPECIES: dict[str, str] = {
    "human": "homo_sapiens",
    "mouse": "mus_musculus",
    "rat": "rattus_norvegicus",
    "zebrafish": "danio_rerio",
    "fly": "drosophila_melanogaster",
    "worm": "caenorhabditis_elegans",
}


# Species → ("fungi" | "plants" | "metazoa" | "protists", binomial).
_ENSEMBL_GENOMES_SPECIES: dict[str, tuple[str, str]] = {
    "saccharomyces_cerevisiae": ("fungi", "saccharomyces_cerevisiae"),
    "schizosaccharomyces_pombe": ("fungi", "schizosaccharomyces_pombe"),
    "candida_albicans": ("fungi", "_collection/candida_albicans"),
    "pichia_pastoris": ("fungi", "_collection/komagataella_phaffii_gs115"),
    "komagataella_phaffii": ("fungi", "_collection/komagataella_phaffii_gs115"),
    "arabidopsis_thaliana": ("plants", "arabidopsis_thaliana"),
}


def _ensembl_release_for(*, release: int | None) -> tuple[str, str, str]:
    """Resolve (release_id, fasta_base, gff_base) for the vertebrate Ensembl.

    Pinned via ``release`` arg when supplied; otherwise parses the
    ``current_README`` to extract "Ensembl Release N" — keeps the
    fetched-bytes provenance honest even when the URL says
    ``current_fasta/``.
    """
    if release is not None:
        rel = f"release-{release}"
        return (
            str(release),
            f"{_ENSEMBL_PUB_BASE}/{rel}/fasta",
            f"{_ENSEMBL_PUB_BASE}/{rel}/gff3",
        )
    resolved = _probe_ensembl_release_number()
    if resolved is None:
        # Couldn't determine — fall back to the symlinked current_*
        # directories. The release field stays empty (handle will be
        # rejected by the validator below).
        raise ValueError(
            "could not determine current Ensembl release number; "
            "pass --release N explicitly. (probe of current_README "
            f"at {_ENSEMBL_PUB_BASE}/current_README returned no match)"
        )
    return (
        str(resolved),
        f"{_ENSEMBL_PUB_BASE}/release-{resolved}/fasta",
        f"{_ENSEMBL_PUB_BASE}/release-{resolved}/gff3",
    )


def _probe_ensembl_release_number() -> int | None:
    """Read ``pub/current_README`` and extract the Ensembl release."""
    try:
        body = _http_get_text(f"{_ENSEMBL_PUB_BASE}/current_README")
    except (urllib.error.URLError, OSError):
        return None
    # Format includes a line like "Ensembl Release 111 Databases."
    m = re.search(r"Ensembl Release\s+(\d+)", body)
    if m:
        return int(m.group(1))
    return None


def _ensembl_urls(
    species_path: str, *, release: int | None
) -> tuple[str, str, str | None, str, str | None]:
    """Vertebrate Ensembl FASTA + GFF3 + cDNA URLs + resolved release + assembly name.

    ``cdna_url`` is ``None`` when the species' ``cdna/`` directory
    listing doesn't expose a ``.cdna.all.fa.gz`` (unusual for vertebrate
    species but tolerated — cDNA is opportunistic, not required).
    """
    rel_id, fa_base, gff_base = _ensembl_release_for(release=release)
    fa_dir = f"{fa_base}/{species_path}/dna"
    gff_dir = f"{gff_base}/{species_path}"
    fa_url = _resolve_pattern_url(fa_dir, suffix=".dna.toplevel.fa.gz")
    gff_url = _resolve_pattern_url(gff_dir, suffix=".gff3.gz", exclude="abinitio")
    cdna_url: str | None
    try:
        cdna_url = _resolve_pattern_url(
            f"{fa_base}/{species_path}/cdna", suffix=".cdna.all.fa.gz"
        )
    except (FileNotFoundError, urllib.error.URLError, urllib.error.HTTPError, OSError):
        cdna_url = None
    asm = _parse_ensembl_assembly_name(fa_url, species_path)
    return fa_url, gff_url, cdna_url, rel_id, asm


def _ensembl_genomes_release_for(division: str, *, release: int | None) -> tuple[str, str, str]:
    """(release_id, fasta_base, gff_base) for Ensembl Genomes."""
    if release is not None:
        rel = f"release-{release}"
        return (
            str(release),
            f"{_ENSEMBL_GENOMES_FTP_BASE}/{division}/{rel}/fasta",
            f"{_ENSEMBL_GENOMES_FTP_BASE}/{division}/{rel}/gff3",
        )
    # Probe Ensembl's REST API for the current EG release (rest.ensembl.org
    # has valid TLS and a stable JSON contract). FTP-side discovery is
    # broken: per-division ``current_README`` returns 404; the host's own
    # TLS cert doesn't match its name. All divisions share the same release
    # number so one call suffices for any division.
    import json
    try:
        body = _http_get_text(
            "https://rest.ensembl.org/info/eg_version?content-type=application/json"
        )
        version = json.loads(body).get("version")
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        version = None
    if version is not None:
        rel_id = str(int(version))
        return (
            rel_id,
            f"{_ENSEMBL_GENOMES_FTP_BASE}/{division}/release-{rel_id}/fasta",
            f"{_ENSEMBL_GENOMES_FTP_BASE}/{division}/release-{rel_id}/gff3",
        )
    raise ValueError(
        f"could not determine current Ensembl Genomes {division!r} release; "
        "pass --release N explicitly"
    )


def _ensembl_genomes_urls(
    division: str, species_path: str, *, release: int | None
) -> tuple[str, str, str | None, str, str | None]:
    rel_id, fa_base, gff_base = _ensembl_genomes_release_for(division, release=release)
    fa_dir = f"{fa_base}/{species_path}/dna"
    gff_dir = f"{gff_base}/{species_path}"
    fa_url = _resolve_pattern_url(fa_dir, suffix=".dna.toplevel.fa.gz")
    gff_url = _resolve_pattern_url(gff_dir, suffix=".gff3.gz", exclude="abinitio")
    cdna_url: str | None
    try:
        cdna_url = _resolve_pattern_url(
            f"{fa_base}/{species_path}/cdna", suffix=".cdna.all.fa.gz"
        )
    except (FileNotFoundError, urllib.error.URLError, urllib.error.HTTPError, OSError):
        cdna_url = None
    asm = _parse_ensembl_assembly_name(fa_url, species_path)
    return fa_url, gff_url, cdna_url, rel_id, asm


def _parse_ensembl_assembly_name(fa_url: str, species_path: str) -> str | None:
    """Pull the assembly name out of an Ensembl FASTA filename.

    Format: ``Genus_species.AssemblyName.dna.toplevel.fa.gz`` →
    ``AssemblyName``.
    """
    name = fa_url.rsplit("/", 1)[-1]
    if ".dna.toplevel.fa.gz" not in name:
        return None
    head = name.split(".dna.toplevel.fa.gz", 1)[0]
    parts = head.split(".", 1)
    if len(parts) < 2:
        return None
    return parts[1]


def _refseq_urls(
    accession: str,
) -> tuple[str, str, str, str, str, str | None, str | None, dict[str, str | None]]:
    """RefSeq/GenBank FASTA + GFF + protein FASTA + cDNA FASTA + release_slug + ... .

    Returns a tuple ``(fa_url, gff_url, protein_url, cdna_url,
    release_slug, asm_name, ann_release, metadata)``. The release slug is
    ``<accession>`` when an annotation release is unknown, or
    ``<accession>-ar<N>`` when the assembly's ``assembly_report.txt``
    declares one. This is the canonical handle release portion for NCBI
    sources — the assembly accession alone does NOT pin the GFF
    (RefSeq's annotation pipeline versions independently of the
    assembly).

    Both ``protein_url`` and ``cdna_url`` are the standard
    ``_protein.faa.gz`` / ``_rna_from_genomic.fna.gz`` siblings of the
    genomic FASTA. Always derived from the URL pattern; assemblies that
    don't ship one (rare, mostly GenBank-only unannotated draft
    assemblies) will 404 on download and the fetch will treat that as
    "no protein/cDNA FASTA available" rather than failing the whole
    fetch.
    """
    if not (accession.startswith("GCF_") or accession.startswith("GCA_")):
        raise ValueError(f"refseq accession must start with GCF_ or GCA_: {accession!r}")
    prefix = accession.split("_")[0]
    digits = accession.split("_")[1].split(".")[0]
    if len(digits) != 9:
        raise ValueError(
            f"unexpected accession digit-count for {accession!r}; expected 9 digits"
        )
    parts = [digits[0:3], digits[3:6], digits[6:9]]
    base = (
        f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{prefix}/"
        f"{parts[0]}/{parts[1]}/{parts[2]}/"
    )
    asm_dir = _resolve_dir_match(base, prefix=accession)
    asm_url_base = f"{base}{asm_dir}"
    if not asm_url_base.endswith("/"):
        asm_url_base += "/"
    fa_url = f"{asm_url_base}{asm_dir}_genomic.fna.gz"
    gff_url = f"{asm_url_base}{asm_dir}_genomic.gff.gz"
    protein_url = f"{asm_url_base}{asm_dir}_protein.faa.gz"
    cdna_url = f"{asm_url_base}{asm_dir}_rna_from_genomic.fna.gz"
    asm_name = asm_dir.split("_", 2)[-1] if asm_dir.count("_") >= 2 else None
    metadata = _probe_refseq_assembly_metadata(asm_url_base, asm_dir)
    ann_release = metadata.get("annotation_release")
    release_slug = accession if ann_release is None else f"{accession}-ar{ann_release}"
    return fa_url, gff_url, protein_url, cdna_url, release_slug, asm_name, ann_release, metadata


# Keys returned by ``_probe_refseq_assembly_metadata``. All values are
# ``str | None``; missing fields are ``None``.
_ASM_REPORT_FIELDS: tuple[str, ...] = (
    "organism_name",       # "# Organism name:"   e.g. "Komagataella phaffii GS115 (budding yeasts)"
    "infraspecific_name",  # "# Infraspecific name:" e.g. "strain=GS115" → captured raw
    "strain",              # parsed out of infraspecific_name (drop "strain=" prefix when present)
    "taxid",               # "# Taxid:"           e.g. "644223"
    "annotation_release",  # "# Annotation release:" e.g. "103"
)


def _probe_refseq_assembly_metadata(
    asm_url_base: str, asm_dir: str
) -> dict[str, str | None]:
    """Best-effort parse of ``<asm>_assembly_report.txt``.

    Extracts organism name, infraspecific name (strain), taxid, and
    annotation release in a single HTTP fetch. Missing fields → ``None``;
    network failure → all fields ``None``. RefSeq assemblies expose the
    full set; pure GenBank assemblies usually carry organism + taxid but
    not annotation release.
    """
    out: dict[str, str | None] = {k: None for k in _ASM_REPORT_FIELDS}
    url = f"{asm_url_base}{asm_dir}_assembly_report.txt"
    try:
        body = _http_get_text(url)
    except (urllib.error.URLError, OSError):
        return out
    for line in body.splitlines():
        if not line.startswith("#"):
            continue
        key_part, sep, value = line.partition(":")
        if not sep:
            continue
        key = key_part.lstrip("#").strip().lower()
        value = value.strip()
        if not value:
            continue
        if key == "organism name":
            out["organism_name"] = value
        elif key == "infraspecific name":
            out["infraspecific_name"] = value
            # "strain=GS115" → "GS115"; bare "GS115" passes through.
            if value.lower().startswith("strain="):
                out["strain"] = value.split("=", 1)[1].strip() or None
            elif "=" not in value:
                out["strain"] = value
        elif key == "taxid":
            out["taxid"] = value
        elif key == "annotation release":
            out["annotation_release"] = value
    return out


# ──────────────────────────────────────────────────────────────────────
# HTTP helpers (download with ETag + Last-Modified capture)
# ──────────────────────────────────────────────────────────────────────


def _http_get(url: str, *, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"constellation/{_CONSTELLATION_VERSION}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_text(url: str, *, timeout: int = 60) -> str:
    return _http_get(url, timeout=timeout).decode("utf-8", errors="replace")


def _resolve_pattern_url(
    listing_url: str, *, suffix: str, exclude: str | None = None
) -> str:
    """Pick a single file from an HTTP directory listing matching ``suffix``."""
    if not listing_url.endswith("/"):
        listing_url = listing_url + "/"
    body = _http_get_text(listing_url)
    candidates: list[str] = []
    for token in body.split('"'):
        if token.endswith(suffix):
            if exclude is not None and exclude.lower() in token.lower():
                continue
            candidates.append(token)
    if not candidates:
        for line in body.splitlines():
            for tok in line.split():
                if tok.endswith(suffix) and (
                    exclude is None or exclude.lower() not in tok.lower()
                ):
                    candidates.append(tok)
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise FileNotFoundError(
            f"no file matching {suffix!r} found at {listing_url}"
        )
    candidates.sort(key=len)
    return listing_url + candidates[0]


def _resolve_dir_match(base_url: str, *, prefix: str) -> str:
    """Pick a single subdirectory under ``base_url`` matching ``prefix``."""
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    body = _http_get_text(base_url)
    matches: list[str] = []
    for token in body.split('"'):
        if token.startswith(prefix) and "/" not in token.rstrip("/"):
            matches.append(token.rstrip("/"))
    if not matches:
        for line in body.splitlines():
            for tok in line.split():
                tok = tok.rstrip("/")
                if tok.startswith(prefix):
                    matches.append(tok)
    matches = list(dict.fromkeys(matches))
    if not matches:
        raise FileNotFoundError(f"no assembly directory under {base_url}")
    matches.sort()
    return matches[0]


def _download_to(
    url: str, dest: Path, *, timeout: int = 600
) -> tuple[str | None, str | None]:
    """Stream-download a URL to ``dest`` and return (etag, last_modified)."""
    req = urllib.request.Request(
        url, headers={"User-Agent": f"constellation/{_CONSTELLATION_VERSION}"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        etag = resp.headers.get("ETag")
        last_modified = resp.headers.get("Last-Modified")
        shutil.copyfileobj(resp, out)
    return etag, last_modified


def _gunzip_to(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as g, dst.open("wb") as out:
        shutil.copyfileobj(g, out)


def _sha256_of(path: Path) -> str:
    """sha256 hex digest of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bsd_sum_of(path: Path) -> tuple[int, int]:
    """BSD ``sum`` 16-bit checksum + 1024-byte block count.

    Matches what Ensembl's ``CHECKSUMS`` files publish. The algorithm is
    a 16-bit rotated additive checksum; block count is ceil(filesize / 1024).
    """
    checksum = 0
    n_bytes = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            n_bytes += len(chunk)
            for b in chunk:
                checksum = (checksum >> 1) + ((checksum & 1) << 15)
                checksum = (checksum + b) & 0xFFFF
    blocks = (n_bytes + 1023) // 1024
    return checksum, blocks


# ──────────────────────────────────────────────────────────────────────
# Source-checksum verification (best-effort)
# ──────────────────────────────────────────────────────────────────────


def _verify_ensembl_checksums(
    checksums_url: str, files: dict[str, Path], *, timeout: int = 60
) -> bool:
    """Verify each ``<basename, path>`` against the upstream ``CHECKSUMS``.

    Returns True if every file matched a published entry. Mismatches +
    fetch errors return False; callers decide whether to warn-and-proceed.
    """
    try:
        body = _http_get_text(checksums_url, timeout=timeout)
    except (urllib.error.URLError, OSError):
        return False
    # Format: "<sum> <blocks> <filename>\n"
    published: dict[str, tuple[int, int]] = {}
    for line in body.splitlines():
        toks = line.strip().split()
        if len(toks) >= 3:
            try:
                published[toks[2]] = (int(toks[0]), int(toks[1]))
            except ValueError:
                continue
    if not published:
        return False
    all_match = True
    for basename, local_path in files.items():
        expected = published.get(basename)
        if expected is None:
            all_match = False
            continue
        actual = _bsd_sum_of(local_path)
        if actual != expected:
            all_match = False
    return all_match


def _verify_refseq_checksums(
    md5_url: str, files: dict[str, Path], *, timeout: int = 60
) -> bool:
    """Verify each file against the upstream ``md5checksums.txt``.

    Format: ``<md5>  ./<filename>`` per line.
    """
    try:
        body = _http_get_text(md5_url, timeout=timeout)
    except (urllib.error.URLError, OSError):
        return False
    published: dict[str, str] = {}
    for line in body.splitlines():
        toks = line.strip().split(maxsplit=1)
        if len(toks) == 2:
            md5 = toks[0]
            name = toks[1].lstrip("./")
            published[name] = md5
    if not published:
        return False
    all_match = True
    for basename, local_path in files.items():
        expected = published.get(basename)
        if expected is None:
            all_match = False
            continue
        actual = _md5_of(local_path)
        if actual.lower() != expected.lower():
            all_match = False
    return all_match


# ──────────────────────────────────────────────────────────────────────
# Source dispatch
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _ResolvedSpec:
    """A handle + URLs + assembly metadata ready for fetch.

    ``taxid`` / ``scientific_name`` are populated when the spec was
    resolved through the taxonomy + catalog path (species-name dispatch).
    They land in ``meta.toml`` so downstream consumers can join cached
    references against the taxonomy tree.

    ``strain`` carries the infraspecific identifier parsed out of NCBI's
    ``assembly_report.txt`` (``# Infraspecific name: strain=GS115`` →
    ``"GS115"``). The organism slug is genus+species only by design, so
    strain is preserved as a meta.toml field for ``reference summary``.
    """

    handle: Handle
    fasta_url: str
    gff_url: str
    checksums_url: str | None
    checksums_kind: str | None  # 'ensembl' | 'refseq' | None
    assembly_name: str | None
    annotation_release: str | None
    assembly_accession: str | None
    taxid: int | None = None
    scientific_name: str | None = None
    strain: str | None = None
    # Optional protein FASTA URL. RefSeq always populates this (the
    # ``_protein.faa.gz`` sibling of the genomic FASTA); Ensembl rows
    # populate it via the catalog's ``protein_url`` column when known.
    # ``None`` means the source doesn't ship a protein FASTA at the
    # standard location, in which case ``fetch_reference`` skips the
    # protein FASTA materialisation step.
    protein_url: str | None = None
    # Optional cDNA FASTA URL. Mirrors ``protein_url`` semantics — RefSeq
    # ``_rna_from_genomic.fna.gz`` sibling, or Ensembl/Ensembl Genomes
    # ``cdna.all.fa.gz`` under the species ``cdna/`` directory. Catalog
    # rows for all genome-bearing sources populate this; UniProt rows
    # leave it ``None``. Opportunistic fetch — 404 leaves the cache slot
    # without a ``cdna.fna`` and the overall fetch still succeeds.
    cdna_url: str | None = None


def _resolve_spec(
    spec: str,
    *,
    release: int | None,
    source: str | None = None,
) -> _ResolvedSpec:
    """Map a fetch spec to URLs + handle.

    Accepted spec shapes:

      * ``<source>:<id>`` — legacy form. ``source`` arg ignored.
      * ``<species_name>`` — bare species name or vernacular (e.g.
        ``'Haliotis rufescens'``, ``'red abalone'``, ``'human'``).
        Routes through ``TaxonomyResolver`` → ``CatalogResolver``;
        ``source`` arg pins the catalog source (RefSeq-first by default).
      * ``<taxid>`` — integer NCBI taxid. Same routing as species name.
    """
    if ":" not in spec:
        return _resolve_species_query(spec, source=source, release=release)
    src, ident = spec.split(":", 1)
    src = src.strip().lower()
    ident = ident.strip()
    source = src  # backward compat for the legacy branches below

    if source == "ensembl":
        species_path = _ENSEMBL_VERTEBRATE_SPECIES.get(ident, ident)
        fa_url, gff_url, cdna_url, rel_id, asm_name = _ensembl_urls(
            species_path, release=release
        )
        # The CHECKSUMS file always lives in the FASTA directory.
        checksums_url = fa_url.rsplit("/", 1)[0] + "/CHECKSUMS"
        organism = species_path.replace("/", "_")
        return _ResolvedSpec(
            handle=Handle(organism=organism, source="ensembl", release=rel_id),
            fasta_url=fa_url,
            gff_url=gff_url,
            checksums_url=checksums_url,
            checksums_kind="ensembl",
            assembly_name=asm_name,
            annotation_release=rel_id,  # Ensembl annotation rides with release
            assembly_accession=None,
            cdna_url=cdna_url,
        )

    if source == "ensembl_genomes":
        if ident not in _ENSEMBL_GENOMES_SPECIES:
            raise KeyError(
                f"unknown ensembl_genomes species {ident!r}; supported: "
                f"{sorted(_ENSEMBL_GENOMES_SPECIES)}. Use 'constellation "
                f"reference import' for organisms outside this list."
            )
        division, species_path = _ENSEMBL_GENOMES_SPECIES[ident]
        fa_url, gff_url, cdna_url, rel_id, asm_name = _ensembl_genomes_urls(
            division, species_path, release=release
        )
        checksums_url = fa_url.rsplit("/", 1)[0] + "/CHECKSUMS"
        return _ResolvedSpec(
            handle=Handle(organism=ident, source="ensembl_genomes", release=rel_id),
            fasta_url=fa_url,
            gff_url=gff_url,
            checksums_url=checksums_url,
            checksums_kind="ensembl",
            assembly_name=asm_name,
            annotation_release=rel_id,
            assembly_accession=None,
            cdna_url=cdna_url,
        )

    if source in {"refseq", "genbank"}:
        if release is not None:
            raise ValueError(
                f"--release is not applicable to {source!r}; the accession "
                f"already pins the assembly version (e.g. GCF_000001635.27)"
            )
        fa_url, gff_url, protein_url, cdna_url, release_slug, asm_name, ann_release, metadata = _refseq_urls(ident)
        # md5 file lives next to the GFF/FASTA in the assembly dir.
        md5_url = fa_url.rsplit("/", 1)[0] + "/md5checksums.txt"
        # Organism slug precedence:
        #   1. Curated override (_REFSEQ_ORGANISM_SLUGS) for canonical lab targets.
        #   2. Parsed "# Organism name:" from assembly_report.txt — the principled path.
        #   3. Sanitised assembly_name with proper prefix-strip (defense-in-depth
        #      fallback for assemblies whose report can't be fetched).
        #   4. Accession itself — last resort.
        slug_from_report = _slugify_organism_name(metadata.get("organism_name"))
        sanitised_asm = (asm_name or "").lower().replace(".", "_").replace("-", "_")
        # NOTE: ``removeprefix`` (not ``lstrip``!) — ``lstrip(chars)`` strips any
        # character in the argument's set from the left, which silently chewed
        # the leading 'a' off "asm2700v1" via ``.lstrip("gca_")`` before this fix.
        sanitised_asm = sanitised_asm.removeprefix("gcf_").removeprefix("gca_")
        organism = (
            slug_from_report
            or sanitised_asm
            or ident.lower()
        )
        # Curated override wins last so a hand-picked slug overrides any of
        # the auto-derived choices above (matches prior behaviour).
        organism = _organism_for_accession(ident, fallback=organism)
        return _ResolvedSpec(
            handle=Handle(organism=organism, source=source, release=release_slug),
            fasta_url=fa_url,
            gff_url=gff_url,
            checksums_url=md5_url,
            checksums_kind="refseq",
            assembly_name=asm_name,
            annotation_release=ann_release,
            assembly_accession=ident,
            strain=metadata.get("strain"),
            protein_url=protein_url,
            cdna_url=cdna_url,
        )

    raise KeyError(
        f"unknown source {source!r}; supported: 'ensembl', 'ensembl_genomes', "
        f"'refseq', 'genbank'"
    )


def _resolve_species_query(
    query: str,
    *,
    source: str | None,
    release: int | None,
) -> _ResolvedSpec:
    """Species-name / taxid dispatch through taxonomy + catalog.

    Lookup chain: ``TaxonomyResolver.auto().lookup_strict(query)`` →
    ``CatalogResolver.from_cache().best_for(taxid, source=source)`` →
    convert the catalog row into a ``_ResolvedSpec`` that the existing
    download pipeline can consume.

    Raises ``ValueError`` with an actionable message when:
      - the query doesn't resolve to a known taxon (taxonomy unknown)
      - no catalog row matches the resolved taxid (catalog stale / missing)
    """
    from constellation.catalog import (
        CatalogResolver,
    )
    from constellation.core.taxonomy import (
        TaxonomyResolver,
        UnknownTaxonError,
    )

    tax = TaxonomyResolver.auto()
    try:
        node = tax.lookup_strict(query)
    except UnknownTaxonError as exc:
        raise ValueError(
            f"unknown species / taxid {query!r} — not in the taxonomy. "
            "Run `constellation taxonomy update` if this is a species "
            "outside the bundled starter set."
        ) from exc

    catalogs = CatalogResolver.from_cache()
    if catalogs.is_empty():
        raise ValueError(
            f"no catalogs installed — cannot resolve URLs for {node.scientific_name!r}. "
            "Run `constellation catalog update refseq` (and/or ensembl / "
            "ensembl_genomes / uniprot) to populate them."
        )
    row = catalogs.best_for(node.taxid, source=source)
    if row is None:
        sources = sorted({s for s, _ in catalogs.installed_sources()})
        raise ValueError(
            f"no catalog hit for taxid {node.taxid} ({node.scientific_name!r}); "
            f"installed sources: {sources}. Try `constellation catalog update <source>` "
            "to refresh, or pass --source explicitly."
        )
    return _spec_from_catalog_row(row, taxid=node.taxid, scientific_name=node.scientific_name)


def _spec_from_catalog_row(
    row: "CatalogRow",  # noqa: F821 — imported lazily above
    *,
    taxid: int,
    scientific_name: str,
) -> _ResolvedSpec:
    """Convert a catalog row into the ``_ResolvedSpec`` the fetch pipeline expects.

    UniProt rows are proteome-only — ``fasta_url`` and ``gff_url`` come
    back as empty strings (the proteome-only-dispatch signal in
    ``fetch_reference``) and ``protein_url`` carries the FASTA URL.
    """
    src = row.source
    if src in {"refseq", "genbank"}:
        # RefSeq + GenBank use the RefSeq cache convention
        # (release_slug = accession); annotation_release pins the GFF.
        release_slug = row.assembly_accession or row.release
        if row.annotation_release:
            release_slug = f"{row.assembly_accession}-ar{row.annotation_release}"
        handle = Handle(
            organism=row.organism_slug, source=src, release=release_slug
        )
    elif src in {"ensembl", "ensembl_genomes"}:
        handle = Handle(
            organism=row.organism_slug, source=src, release=row.release
        )
    elif src == "uniprot":
        # Proteome-only install — handle uses the UniProt release as the
        # cache release slug (e.g. "2026_02" for SwissProt, or a
        # proteome ID for taxid-specific UP rows).
        handle = Handle(
            organism=row.organism_slug, source="uniprot", release=row.release
        )
        if not row.protein_url:
            raise ValueError(
                f"UniProt catalog row for {row.organism_slug!r} has no "
                "protein_url; cannot materialise a proteome-only install"
            )
        return _ResolvedSpec(
            handle=handle,
            fasta_url="",  # proteome-only dispatch signal
            gff_url="",
            checksums_url=row.checksums_url,
            checksums_kind=row.checksums_kind,
            assembly_name=row.assembly_name,
            annotation_release=None,
            assembly_accession=row.assembly_accession,
            taxid=taxid,
            scientific_name=scientific_name,
            protein_url=row.protein_url,
            cdna_url=None,
        )
    else:
        raise ValueError(f"catalog row has unknown source {src!r}")
    return _ResolvedSpec(
        handle=handle,
        fasta_url=row.fasta_url,
        gff_url=row.gff_url or "",
        checksums_url=row.checksums_url,
        checksums_kind=row.checksums_kind,
        assembly_name=row.assembly_name,
        annotation_release=row.annotation_release,
        assembly_accession=row.assembly_accession,
        taxid=taxid,
        scientific_name=scientific_name,
        protein_url=row.protein_url,
        cdna_url=row.cdna_url,
    )


# Known-good organism slugs for the lab's likely RefSeq targets. Falls
# back to the parsed assembly_name when not in this table — users can
# always rename via `reference link` or by editing the cache directly.
_REFSEQ_ORGANISM_SLUGS: dict[str, str] = {
    "GCF_000001405": "homo_sapiens",
    "GCF_000001635": "mus_musculus",
    "GCF_000001895": "rattus_norvegicus",
    "GCF_000001215": "drosophila_melanogaster",
    "GCF_000146045": "saccharomyces_cerevisiae",
    "GCF_000146885": "candida_albicans",
    "GCF_001708105": "komagataella_phaffii",
}


def _organism_for_accession(accession: str, *, fallback: str) -> str:
    """Map a GCF_/GCA_ accession to a stable organism slug."""
    base = accession.split(".")[0]  # drop version suffix
    return _REFSEQ_ORGANISM_SLUGS.get(base, fallback)


def _slugify_organism_name(organism_name: str | None) -> str | None:
    """Slugify an NCBI ``# Organism name`` line.

    ``"Komagataella phaffii GS115 (budding yeasts)"`` → ``"komagataella_phaffii"``.
    Strategy: drop parenthesised commentary, take the first two whitespace-
    separated tokens (genus + species), lowercase, join with ``_``. Strain /
    substrain tokens beyond position 2 are dropped — they're preserved
    separately via ``# Infraspecific name`` and surface in ``meta.toml``.
    Single-token names pass through. Empty / ``None`` input → ``None``.
    """
    if not organism_name:
        return None
    # Strip "(commentary)" segments such as "(budding yeasts)".
    cleaned = re.sub(r"\([^)]*\)", " ", organism_name)
    tokens = cleaned.split()
    if not tokens:
        return None
    head = tokens[: min(2, len(tokens))]
    slug = "_".join(t.lower() for t in head)
    # Restrict to the [a-z0-9_] alphabet used by ``_ORGANISM_RE`` in
    # handle.py — anything else would fail the validator downstream.
    slug = re.sub(r"[^a-z0-9_]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug or not slug[0].isalpha():
        return None
    return slug


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def fetch_reference(
    spec: str,
    output_dir: str | Path | None = None,
    *,
    release: int | None = None,
    source: str | None = None,
    timeout: int = 600,
    use_cache: bool = True,
    verify_source_checksums: bool = True,
    force: bool = False,
    handle_override: Handle | None = None,
) -> FetchResult:
    """Fetch a genome FASTA + GFF3 by source/id; write to cache and/or ``output_dir``.

    Destination semantics:
        - ``use_cache=True, output_dir=None`` → write only to the cache.
        - ``use_cache=True, output_dir=<path>`` → write BOTH cache and path
          (two copies; ``output_dir`` becomes a portable local snapshot).
        - ``use_cache=False, output_dir=<path>`` → write only to path
          (preserves the legacy "scratch fetch" behaviour).
        - ``use_cache=False, output_dir=None`` → ``ValueError`` (no destination).

    Idempotency: when writing to the cache, a complete prior install of
    the resolved handle short-circuits the fetch (returns the cached
    ``GenomeReference`` + ``Annotation``). Pass ``force=True`` to always
    re-fetch.
    """
    if not use_cache and output_dir is None:
        raise ValueError(
            "fetch_reference requires a destination: either use_cache=True "
            "or pass output_dir=<path>"
        )

    # Late-import the readers to keep the constellation startup fast.
    from constellation.sequencing.readers.fastx import read_fasta_genome
    from constellation.sequencing.readers.gff import read_gff3

    resolved = _resolve_spec(spec, release=release, source=source)
    handle = handle_override or resolved.handle
    if not handle.is_qualified():
        raise RuntimeError(
            f"internal: resolved handle is unqualified ({handle!r}); "
            "should not happen — source resolver returns a release id"
        )

    # Proteome-only path (UniProt rows): empty fasta_url + gff_url and a
    # populated protein_url is the signal. Bypasses the
    # genome+annotation parse pipeline entirely.
    if not resolved.fasta_url and not resolved.gff_url and resolved.protein_url:
        return _fetch_proteome_only(
            resolved,
            handle,
            spec=spec,
            output_dir=output_dir,
            timeout=timeout,
            use_cache=use_cache,
            force=force,
        )

    # ── Idempotency check (cache only) ──────────────────────────────
    root = cache_root() if use_cache else None
    cache_release_dir: Path | None = None
    if use_cache and root is not None:
        cache_release_dir = root / handle.organism / handle.release_slug()

    if (
        use_cache
        and cache_release_dir is not None
        and not force
        and _cache_is_complete(cache_release_dir)
    ):
        genome, annotation = _load_from_cache(cache_release_dir)
        cached_protein = cache_release_dir / "protein.faa"
        cached_cdna = cache_release_dir / "cdna.fna"
        sources = {"genome": resolved.fasta_url, "annotation": resolved.gff_url}
        if resolved.protein_url:
            sources["protein"] = resolved.protein_url
        if resolved.cdna_url:
            sources["cdna"] = resolved.cdna_url
        return FetchResult(
            genome=genome,
            annotation=annotation,
            handle=handle,
            cache_path=cache_release_dir,
            output_path=Path(output_dir).resolve() if output_dir else None,
            sources=sources,
            skipped_cache=True,
            protein_fasta_path=cached_protein if cached_protein.is_file() else None,
            cdna_fasta_path=cached_cdna if cached_cdna.is_file() else None,
        )

    # ── Download + parse (writes happen under fetch lock) ────────────
    with tempfile.TemporaryDirectory(prefix="constellation_fetch_") as scratch:
        scratch_path = Path(scratch)
        fa_gz = scratch_path / "genome.fa.gz"
        gff_gz = scratch_path / "annotation.gff3.gz"
        fa_decoded = scratch_path / "genome.fa"
        gff_decoded = scratch_path / "annotation.gff3"
        protein_gz = scratch_path / "protein.faa.gz"
        protein_decoded = scratch_path / "protein.faa"
        cdna_gz = scratch_path / "cdna.fna.gz"
        cdna_decoded = scratch_path / "cdna.fna"

        fa_etag, fa_lm = _download_to(resolved.fasta_url, fa_gz, timeout=timeout)
        gff_etag, gff_lm = _download_to(resolved.gff_url, gff_gz, timeout=timeout)

        # Optional protein FASTA — failure here is non-fatal: assemblies
        # without a published protein FASTA (rare; mostly GenBank-only
        # unannotated drafts) simply skip the materialisation step. The
        # genome + annotation fetch still succeeds.
        protein_etag: str | None = None
        protein_lm: str | None = None
        protein_fetched = False
        if resolved.protein_url:
            try:
                protein_etag, protein_lm = _download_to(
                    resolved.protein_url, protein_gz, timeout=timeout
                )
                protein_fetched = True
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                print(
                    f"warning: protein FASTA fetch failed for "
                    f"{resolved.protein_url}: {exc}. Continuing without a "
                    f"cached protein.faa for this assembly.",
                    file=sys.stderr,
                )

        # Optional cDNA FASTA — same non-fatal try/except shape as the
        # protein block above. RefSeq's ``_rna_from_genomic.fna.gz`` /
        # Ensembl's ``cdna.all.fa.gz`` sibling lands at ``cdna.fna`` in
        # the cache when present.
        cdna_etag: str | None = None
        cdna_lm: str | None = None
        cdna_fetched = False
        if resolved.cdna_url:
            try:
                cdna_etag, cdna_lm = _download_to(
                    resolved.cdna_url, cdna_gz, timeout=timeout
                )
                cdna_fetched = True
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                print(
                    f"warning: cDNA FASTA fetch failed for "
                    f"{resolved.cdna_url}: {exc}. Continuing without a "
                    f"cached cdna.fna for this assembly.",
                    file=sys.stderr,
                )

        # Local sha256 — always.
        sha256_map = {
            "fasta": _sha256_of(fa_gz),
            "gff3": _sha256_of(gff_gz),
        }
        if protein_fetched:
            sha256_map["protein"] = _sha256_of(protein_gz)
        if cdna_fetched:
            sha256_map["cdna"] = _sha256_of(cdna_gz)

        # Source-checksum verification — best-effort warn-not-fail.
        source_checksum_verified = False
        if verify_source_checksums and resolved.checksums_url and resolved.checksums_kind:
            files = {
                resolved.fasta_url.rsplit("/", 1)[-1]: fa_gz,
                resolved.gff_url.rsplit("/", 1)[-1]: gff_gz,
            }
            if resolved.checksums_kind == "ensembl":
                source_checksum_verified = _verify_ensembl_checksums(
                    resolved.checksums_url, files, timeout=timeout
                )
            elif resolved.checksums_kind == "refseq":
                source_checksum_verified = _verify_refseq_checksums(
                    resolved.checksums_url, files, timeout=timeout
                )
            if not source_checksum_verified:
                print(
                    f"warning: source checksum verification failed for "
                    f"{resolved.checksums_url}; local sha256 still recorded. "
                    f"Bytes still trusted via local sha256 in meta.toml.",
                    file=sys.stderr,
                )

        _gunzip_to(fa_gz, fa_decoded)
        _gunzip_to(gff_gz, gff_decoded)
        if protein_fetched:
            _gunzip_to(protein_gz, protein_decoded)
        if cdna_fetched:
            _gunzip_to(cdna_gz, cdna_decoded)

        genome = read_fasta_genome(fa_decoded)
        contig_name_to_id = {
            row["name"]: row["contig_id"] for row in genome.contigs.to_pylist()
        }
        annotation = read_gff3(gff_decoded, contig_name_to_id=contig_name_to_id)

        # Annotate provenance.
        genome_meta: dict[str, Any] = dict(genome.metadata_extras)
        genome_meta.update(
            {
                "fetch_source": spec,
                "fetch_url_fasta": resolved.fasta_url,
                "handle": str(handle),
                "assembly_accession": resolved.assembly_accession,
                "assembly_name": resolved.assembly_name,
                "annotation_release": resolved.annotation_release,
                "sha256_fasta": sha256_map["fasta"],
            }
        )
        genome = genome.with_metadata(genome_meta)

        annotation_meta: dict[str, Any] = dict(annotation.metadata_extras)
        annotation_meta.update(
            {
                "fetch_source": spec,
                "fetch_url_gff3": resolved.gff_url,
                "handle": str(handle),
                "assembly_accession": resolved.assembly_accession,
                "assembly_name": resolved.assembly_name,
                "annotation_release": resolved.annotation_release,
                "sha256_gff3": sha256_map["gff3"],
            }
        )
        annotation = annotation.with_metadata(annotation_meta)
        annotation.validate_against(genome)

        urls_meta = {
            "fasta": {"url": resolved.fasta_url, "etag": fa_etag, "last_modified": fa_lm},
            "gff3": {"url": resolved.gff_url, "etag": gff_etag, "last_modified": gff_lm},
        }
        if resolved.checksums_url:
            urls_meta["checksums"] = {"url": resolved.checksums_url}
        if protein_fetched:
            urls_meta["protein"] = {
                "url": resolved.protein_url,
                "etag": protein_etag,
                "last_modified": protein_lm,
            }
        if cdna_fetched:
            urls_meta["cdna"] = {
                "url": resolved.cdna_url,
                "etag": cdna_etag,
                "last_modified": cdna_lm,
            }

        # ── Cache write under lock ──────────────────────────────────
        cached_protein_path: Path | None = None
        cached_cdna_path: Path | None = None
        if use_cache and cache_release_dir is not None:
            with acquire_fetch_lock(handle):
                # Re-check idempotency inside the lock — another worker
                # may have populated the cache while we were downloading.
                if not force and _cache_is_complete(cache_release_dir):
                    cached_existing = cache_release_dir / "protein.faa"
                    if cached_existing.is_file():
                        cached_protein_path = cached_existing
                    cached_existing_cdna = cache_release_dir / "cdna.fna"
                    if cached_existing_cdna.is_file():
                        cached_cdna_path = cached_existing_cdna
                else:
                    _write_into_cache(
                        cache_release_dir,
                        genome=genome,
                        annotation=annotation,
                        handle=handle,
                        urls_meta=urls_meta,
                        sha256_map=sha256_map,
                        source_checksum_verified=source_checksum_verified,
                        resolved=resolved,
                        force=force,
                        protein_fasta_src=(
                            protein_decoded if protein_fetched else None
                        ),
                        cdna_fasta_src=(
                            cdna_decoded if cdna_fetched else None
                        ),
                        has_proteome=protein_fetched,
                        has_cdna=cdna_fetched,
                    )
                    if protein_fetched:
                        cached_protein_path = cache_release_dir / "protein.faa"
                    if cdna_fetched:
                        cached_cdna_path = cache_release_dir / "cdna.fna"

        # ── Optional --output-dir write ─────────────────────────────
        if output_dir is not None:
            out_path = Path(output_dir).expanduser().resolve()
            out_path.mkdir(parents=True, exist_ok=True)
            save_genome_reference(genome, out_path / "genome")
            save_annotation(annotation, out_path / "annotation")
            if protein_fetched:
                shutil.copy2(protein_decoded, out_path / "protein.faa")
            if cdna_fetched:
                shutil.copy2(cdna_decoded, out_path / "cdna.fna")
        else:
            out_path = None

    sources = {"genome": resolved.fasta_url, "annotation": resolved.gff_url}
    if resolved.protein_url:
        sources["protein"] = resolved.protein_url
    if resolved.cdna_url:
        sources["cdna"] = resolved.cdna_url
    return FetchResult(
        genome=genome,
        annotation=annotation,
        handle=handle,
        cache_path=cache_release_dir if use_cache else None,
        output_path=out_path,
        sources=sources,
        skipped_cache=False,
        protein_fasta_path=cached_protein_path,
        cdna_fasta_path=cached_cdna_path,
    )


def _cache_is_complete(release_dir: Path) -> bool:
    """Lightweight check that a cache slot looks fully populated.

    Accepts both the genome-bearing layout (``genome/`` + ``annotation/``
    + ``meta.toml``) and the proteome-only layout (``protein.faa`` +
    ``meta.toml`` with ``has_genome=false``). The disambiguating signal
    is the ``[contents]`` block in ``meta.toml`` (v2) — when absent (v1
    caches) the genome-bearing layout is assumed for back-compat.
    """
    if not release_dir.is_dir():
        return False
    if not (release_dir / "meta.toml").exists():
        return False
    meta = ref_handle.read_meta_toml(release_dir)
    if meta is None:
        return False
    contents = meta.get("contents") or {}
    has_genome = bool(contents.get("has_genome", True))
    if not has_genome:
        # Proteome-only slot: protein.faa is the only mandatory artifact.
        return (release_dir / "protein.faa").is_file()
    if not (release_dir / "genome" / "manifest.json").exists():
        return False
    if not (release_dir / "annotation" / "manifest.json").exists():
        return False
    return True


def _load_from_cache(release_dir: Path) -> tuple[GenomeReference, Annotation]:
    """Load (GenomeReference, Annotation) from a populated cache slot."""
    from constellation.sequencing.annotation.io import load_annotation
    from constellation.sequencing.reference.io import load_genome_reference

    genome = load_genome_reference(release_dir / "genome")
    annotation = load_annotation(release_dir / "annotation")
    return genome, annotation


def _write_into_cache(
    release_dir: Path,
    *,
    genome: GenomeReference | None,
    annotation: Annotation | None,
    handle: Handle,
    urls_meta: dict[str, dict[str, Any]],
    sha256_map: dict[str, str],
    source_checksum_verified: bool,
    resolved: _ResolvedSpec,
    force: bool,
    protein_fasta_src: Path | None = None,
    cdna_fasta_src: Path | None = None,
    has_proteome: bool | None = None,
    has_cdna: bool | None = None,
) -> None:
    """Atomic cache write: stage into ``.partial/`` then rename.

    When ``protein_fasta_src`` / ``cdna_fasta_src`` are provided
    (gunzipped files in scratch), copies them into the staged cache as
    ``protein.faa`` / ``cdna.fna`` so they land atomically alongside the
    genome + annotation.

    Proteome-only installs pass ``genome=None`` + ``annotation=None`` +
    ``protein_fasta_src=<path>`` and set ``has_proteome=True``; the
    ``[contents]`` block in ``meta.toml`` records the resulting layout.
    """
    if has_proteome is None:
        has_proteome = protein_fasta_src is not None
    if has_cdna is None:
        has_cdna = cdna_fasta_src is not None

    # Force overwrite: remove the live dir first; the .partial atomic
    # rename below replaces it cleanly.
    if force and release_dir.exists():
        shutil.rmtree(release_dir)

    # Defensive: clean any leftover .partial from a prior crash.
    clean_partial(release_dir)

    stage = partial_dir(release_dir)
    stage.mkdir(parents=True, exist_ok=False)

    if genome is not None:
        save_genome_reference(genome, stage / "genome")
    if annotation is not None:
        save_annotation(annotation, stage / "annotation")
    if protein_fasta_src is not None:
        shutil.copy2(protein_fasta_src, stage / "protein.faa")
    if cdna_fasta_src is not None:
        shutil.copy2(cdna_fasta_src, stage / "cdna.fna")
    write_meta_toml(
        stage,
        handle=handle,
        assembly_accession=resolved.assembly_accession,
        assembly_name=resolved.assembly_name,
        annotation_release=resolved.annotation_release,
        constellation_version=_CONSTELLATION_VERSION,
        urls=urls_meta,
        sha256=sha256_map,
        source_checksum_verified=source_checksum_verified,
        taxid=resolved.taxid,
        scientific_name=resolved.scientific_name,
        strain=resolved.strain,
        has_genome=genome is not None,
        has_annotation=annotation is not None,
        has_proteome=has_proteome,
        has_cdna=has_cdna,
    )

    promote_partial(release_dir)

    # Update the organism's `current` pointer.
    organism_dir = release_dir.parent
    update_current_pointer(organism_dir, handle.release_slug())

    # Auto-default when this is the organism's first install.
    existing = ref_handle._installed_release_slugs(organism_dir)
    defaults_map = ref_handle.read_defaults()
    if handle.organism not in defaults_map and len(existing) == 1:
        set_default(handle.organism, handle.release_slug())


# ──────────────────────────────────────────────────────────────────────
# Proteome-only fetch (UniProt sources)
# ──────────────────────────────────────────────────────────────────────


def _fetch_proteome_only(
    resolved: _ResolvedSpec,
    handle: Handle,
    *,
    spec: str,
    output_dir: str | Path | None,
    timeout: int,
    use_cache: bool,
    force: bool,
) -> FetchResult:
    """Fetch a proteome-only install (UniProt source).

    Downloads ``protein.faa`` only — no genome, no GFF, no cDNA. Writes
    a v2 ``meta.toml`` flagged ``has_genome=false`` + ``has_annotation=false``
    + ``has_proteome=true``. The ``[contents]`` block is what
    ``_cache_is_complete`` keys off to recognise the proteome-only
    layout on subsequent fetches.
    """
    # ── Idempotency check (cache only) ──────────────────────────────
    root = cache_root() if use_cache else None
    cache_release_dir: Path | None = None
    if use_cache and root is not None:
        cache_release_dir = root / handle.organism / handle.release_slug()

    if (
        use_cache
        and cache_release_dir is not None
        and not force
        and _cache_is_complete(cache_release_dir)
    ):
        cached_protein = cache_release_dir / "protein.faa"
        return FetchResult(
            genome=None,
            annotation=None,
            handle=handle,
            cache_path=cache_release_dir,
            output_path=Path(output_dir).resolve() if output_dir else None,
            sources={"protein": resolved.protein_url} if resolved.protein_url else {},
            skipped_cache=True,
            protein_fasta_path=cached_protein if cached_protein.is_file() else None,
        )

    if not resolved.protein_url:
        raise ValueError(
            f"proteome-only fetch requires resolved.protein_url; got None "
            f"for handle {handle!r}"
        )

    # ── Download + cache write ──────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="constellation_fetch_proteome_") as scratch:
        scratch_path = Path(scratch)
        # UniProt FTP serves SwissProt as ``uniprot_sprot.fasta.gz`` — we
        # accept any gz-suffixed protein file via ``_gunzip_to``.
        protein_gz = scratch_path / "protein.faa.gz"
        protein_decoded = scratch_path / "protein.faa"

        protein_etag, protein_lm = _download_to(
            resolved.protein_url, protein_gz, timeout=timeout
        )
        sha256_map = {"protein": _sha256_of(protein_gz)}
        _gunzip_to(protein_gz, protein_decoded)

        urls_meta = {
            "protein": {
                "url": resolved.protein_url,
                "etag": protein_etag,
                "last_modified": protein_lm,
            }
        }

        cached_protein_path: Path | None = None
        if use_cache and cache_release_dir is not None:
            with acquire_fetch_lock(handle):
                if not force and _cache_is_complete(cache_release_dir):
                    cached_existing = cache_release_dir / "protein.faa"
                    if cached_existing.is_file():
                        cached_protein_path = cached_existing
                else:
                    _write_into_cache(
                        cache_release_dir,
                        genome=None,
                        annotation=None,
                        handle=handle,
                        urls_meta=urls_meta,
                        sha256_map=sha256_map,
                        source_checksum_verified=False,
                        resolved=resolved,
                        force=force,
                        protein_fasta_src=protein_decoded,
                        cdna_fasta_src=None,
                        has_proteome=True,
                        has_cdna=False,
                    )
                    cached_protein_path = cache_release_dir / "protein.faa"

        if output_dir is not None:
            out_path = Path(output_dir).expanduser().resolve()
            out_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(protein_decoded, out_path / "protein.faa")
        else:
            out_path = None

    return FetchResult(
        genome=None,
        annotation=None,
        handle=handle,
        cache_path=cache_release_dir if use_cache else None,
        output_path=out_path,
        sources={"protein": resolved.protein_url},
        skipped_cache=False,
        protein_fasta_path=cached_protein_path,
    )


__all__ = [
    "FetchResult",
    "fetch_reference",
]
