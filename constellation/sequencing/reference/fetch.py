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
from collections.abc import Callable
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
    read_meta_toml,
    set_default,
    update_current_pointer,
    write_meta_toml,
)
from constellation.sequencing.reference.io import save_genome_reference
from constellation.sequencing.reference.reference import GenomeReference


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Result of a successful reference fetch."""

    genome: GenomeReference
    annotation: Annotation | None
    handle: Handle
    cache_path: Path | None
    output_path: Path | None
    sources: dict[str, str]  # name → URL
    skipped_cache: bool = False  # idempotency short-circuit?


# ──────────────────────────────────────────────────────────────────────
# Source URL resolvers
# ──────────────────────────────────────────────────────────────────────


_ENSEMBL_FTP_BASE = "https://ftp.ensembl.org/pub/current_fasta"
_ENSEMBL_GFF3_BASE = "https://ftp.ensembl.org/pub/current_gff3"
_ENSEMBL_PUB_BASE = "https://ftp.ensembl.org/pub"
_ENSEMBL_GENOMES_FTP_BASE = "https://ftp.ensemblgenomes.org/pub"

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
) -> tuple[str, str, str, str | None]:
    """Vertebrate Ensembl FASTA + GFF3 URLs + resolved release + assembly name."""
    rel_id, fa_base, gff_base = _ensembl_release_for(release=release)
    fa_dir = f"{fa_base}/{species_path}/dna"
    gff_dir = f"{gff_base}/{species_path}"
    fa_url = _resolve_pattern_url(fa_dir, suffix=".dna.toplevel.fa.gz")
    gff_url = _resolve_pattern_url(gff_dir, suffix=".gff3.gz", exclude="abinitio")
    asm = _parse_ensembl_assembly_name(fa_url, species_path)
    return fa_url, gff_url, rel_id, asm


def _ensembl_genomes_release_for(division: str, *, release: int | None) -> tuple[str, str, str]:
    """(release_id, fasta_base, gff_base) for Ensembl Genomes."""
    if release is not None:
        rel = f"release-{release}"
        return (
            str(release),
            f"{_ENSEMBL_GENOMES_FTP_BASE}/{division}/{rel}/fasta",
            f"{_ENSEMBL_GENOMES_FTP_BASE}/{division}/{rel}/gff3",
        )
    # Probe the division's current README — same shape as vertebrate Ensembl
    try:
        body = _http_get_text(
            f"{_ENSEMBL_GENOMES_FTP_BASE}/{division}/current_README"
        )
    except (urllib.error.URLError, OSError):
        body = ""
    m = re.search(r"Release\s+(\d+)", body)
    if m:
        rel_id = m.group(1)
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
) -> tuple[str, str, str, str | None]:
    rel_id, fa_base, gff_base = _ensembl_genomes_release_for(division, release=release)
    fa_dir = f"{fa_base}/{species_path}/dna"
    gff_dir = f"{gff_base}/{species_path}"
    fa_url = _resolve_pattern_url(fa_dir, suffix=".dna.toplevel.fa.gz")
    gff_url = _resolve_pattern_url(gff_dir, suffix=".gff3.gz", exclude="abinitio")
    asm = _parse_ensembl_assembly_name(fa_url, species_path)
    return fa_url, gff_url, rel_id, asm


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
) -> tuple[str, str, str, str | None, str | None]:
    """RefSeq/GenBank FASTA + GFF + release_slug + assembly_name + annotation_release.

    The release slug returned is ``<accession>`` when an annotation
    release is unknown, or ``<accession>-ar<N>`` when the assembly's
    ``assembly_report.txt`` declares one. This is the canonical handle
    release portion for NCBI sources — the assembly accession alone
    does NOT pin the GFF (RefSeq's annotation pipeline versions
    independently of the assembly).
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
    asm_name = asm_dir.split("_", 2)[-1] if asm_dir.count("_") >= 2 else None
    ann_release = _probe_refseq_annotation_release(asm_url_base, asm_dir)
    release_slug = accession if ann_release is None else f"{accession}-ar{ann_release}"
    return fa_url, gff_url, release_slug, asm_name, ann_release


def _probe_refseq_annotation_release(
    asm_url_base: str, asm_dir: str
) -> str | None:
    """Best-effort: read ``<asm>_assembly_report.txt`` and extract
    ``# Annotation release: <N>``. RefSeq assemblies expose this; pure
    GenBank assemblies usually don't. Failure → ``None`` (caller falls
    back to accession-only release slug)."""
    url = f"{asm_url_base}{asm_dir}_assembly_report.txt"
    try:
        body = _http_get_text(url)
    except (urllib.error.URLError, OSError):
        return None
    for line in body.splitlines():
        if line.lower().startswith("# annotation release"):
            _, _, value = line.partition(":")
            value = value.strip()
            if value:
                return value
    return None


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
    """A handle + URLs + assembly metadata ready for fetch."""

    handle: Handle
    fasta_url: str
    gff_url: str
    checksums_url: str | None
    checksums_kind: str | None  # 'ensembl' | 'refseq' | None
    assembly_name: str | None
    annotation_release: str | None
    assembly_accession: str | None


def _resolve_spec(spec: str, *, release: int | None) -> _ResolvedSpec:
    """Map a ``<source>:<id>`` string to a complete fetch spec."""
    if ":" not in spec:
        raise ValueError(
            f"reference spec must be '<source>:<id>'; got {spec!r}. "
            f"Examples: 'refseq:GCF_001708105.1', 'ensembl_genomes:saccharomyces_cerevisiae'"
        )
    source, ident = spec.split(":", 1)
    source = source.strip().lower()
    ident = ident.strip()

    if source == "ensembl":
        species_path = _ENSEMBL_VERTEBRATE_SPECIES.get(ident, ident)
        fa_url, gff_url, rel_id, asm_name = _ensembl_urls(
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
        )

    if source == "ensembl_genomes":
        if ident not in _ENSEMBL_GENOMES_SPECIES:
            raise KeyError(
                f"unknown ensembl_genomes species {ident!r}; supported: "
                f"{sorted(_ENSEMBL_GENOMES_SPECIES)}. Use 'constellation "
                f"reference import' for organisms outside this list."
            )
        division, species_path = _ENSEMBL_GENOMES_SPECIES[ident]
        fa_url, gff_url, rel_id, asm_name = _ensembl_genomes_urls(
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
        )

    if source in {"refseq", "genbank"}:
        if release is not None:
            raise ValueError(
                f"--release is not applicable to {source!r}; the accession "
                f"already pins the assembly version (e.g. GCF_000001635.27)"
            )
        fa_url, gff_url, release_slug, asm_name, ann_release = _refseq_urls(ident)
        # md5 file lives next to the GFF/FASTA in the assembly dir.
        md5_url = fa_url.rsplit("/", 1)[0] + "/md5checksums.txt"
        # Organism slug: not knowable from the accession alone in v1 —
        # fall back to the lowercased assembly_name (or accession if
        # absent) until the taxonomy layer lands in PR-B.
        organism = (asm_name or ident).lower().replace(".", "_").replace("-", "_")
        # Strip GCF_/GCA_ prefix if it survived (e.g. asm_name was None).
        organism = organism.lstrip("gcf_").lstrip("gca_") or ident.lower()
        # Use a friendlier guess for the well-known accessions; otherwise
        # the user can always re-import with --handle to rename.
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
        )

    raise KeyError(
        f"unknown source {source!r}; supported: 'ensembl', 'ensembl_genomes', "
        f"'refseq', 'genbank'"
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


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def fetch_reference(
    spec: str,
    output_dir: str | Path | None = None,
    *,
    release: int | None = None,
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

    resolved = _resolve_spec(spec, release=release)
    handle = handle_override or resolved.handle
    if not handle.is_qualified():
        raise RuntimeError(
            f"internal: resolved handle is unqualified ({handle!r}); "
            "should not happen — source resolver returns a release id"
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
        return FetchResult(
            genome=genome,
            annotation=annotation,
            handle=handle,
            cache_path=cache_release_dir,
            output_path=Path(output_dir).resolve() if output_dir else None,
            sources={"genome": resolved.fasta_url, "annotation": resolved.gff_url},
            skipped_cache=True,
        )

    # ── Download + parse (writes happen under fetch lock) ────────────
    with tempfile.TemporaryDirectory(prefix="constellation_fetch_") as scratch:
        scratch_path = Path(scratch)
        fa_gz = scratch_path / "genome.fa.gz"
        gff_gz = scratch_path / "annotation.gff3.gz"
        fa_decoded = scratch_path / "genome.fa"
        gff_decoded = scratch_path / "annotation.gff3"

        fa_etag, fa_lm = _download_to(resolved.fasta_url, fa_gz, timeout=timeout)
        gff_etag, gff_lm = _download_to(resolved.gff_url, gff_gz, timeout=timeout)

        # Local sha256 — always.
        sha256_map = {
            "fasta": _sha256_of(fa_gz),
            "gff3": _sha256_of(gff_gz),
        }

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

        # ── Cache write under lock ──────────────────────────────────
        if use_cache and cache_release_dir is not None:
            with acquire_fetch_lock(handle):
                # Re-check idempotency inside the lock — another worker
                # may have populated the cache while we were downloading.
                if not force and _cache_is_complete(cache_release_dir):
                    pass
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
                    )

        # ── Optional --output-dir write ─────────────────────────────
        if output_dir is not None:
            out_path = Path(output_dir).expanduser().resolve()
            out_path.mkdir(parents=True, exist_ok=True)
            save_genome_reference(genome, out_path / "genome")
            save_annotation(annotation, out_path / "annotation")
        else:
            out_path = None

    return FetchResult(
        genome=genome,
        annotation=annotation,
        handle=handle,
        cache_path=cache_release_dir if use_cache else None,
        output_path=out_path,
        sources={"genome": resolved.fasta_url, "annotation": resolved.gff_url},
        skipped_cache=False,
    )


def _cache_is_complete(release_dir: Path) -> bool:
    """Lightweight check that a cache slot looks fully populated."""
    if not release_dir.is_dir():
        return False
    if not (release_dir / "meta.toml").exists():
        return False
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
    genome: GenomeReference,
    annotation: Annotation,
    handle: Handle,
    urls_meta: dict[str, dict[str, Any]],
    sha256_map: dict[str, str],
    source_checksum_verified: bool,
    resolved: _ResolvedSpec,
    force: bool,
) -> None:
    """Atomic cache write: stage into ``.partial/`` then rename."""
    # Force overwrite: remove the live dir first; the .partial atomic
    # rename below replaces it cleanly.
    if force and release_dir.exists():
        shutil.rmtree(release_dir)

    # Defensive: clean any leftover .partial from a prior crash.
    clean_partial(release_dir)

    stage = partial_dir(release_dir)
    stage.mkdir(parents=True, exist_ok=False)

    save_genome_reference(genome, stage / "genome")
    save_annotation(annotation, stage / "annotation")
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


__all__ = [
    "FetchResult",
    "fetch_reference",
]
