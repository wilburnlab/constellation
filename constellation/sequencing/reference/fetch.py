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
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.annotation.io import save_annotation
from constellation.sequencing.reference.io import save_genome_reference
from constellation.sequencing.reference.reference import GenomeReference


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Result of a successful reference fetch."""

    genome: GenomeReference
    annotation: Annotation | None
    output_dir: Path
    sources: dict[str, str]  # name → URL


# ──────────────────────────────────────────────────────────────────────
# Source URL resolvers
# ──────────────────────────────────────────────────────────────────────


_ENSEMBL_VERTEBRATE_LATEST_RELEASE = "current"  # the FTP server symlinks
_ENSEMBL_FTP_BASE = "https://ftp.ensembl.org/pub/current_fasta"
_ENSEMBL_GFF3_BASE = "https://ftp.ensembl.org/pub/current_gff3"
_ENSEMBL_GENOMES_FTP_BASE = "https://ftp.ensemblgenomes.org/pub"


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


def _ensembl_urls(species_path: str) -> tuple[str, str]:
    """Resolve canonical Ensembl vertebrate FASTA + GFF3 URLs for a
    species path (e.g. ``'homo_sapiens'``)."""
    # Vertebrate Ensembl format: e.g.
    # https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/
    # Homo_sapiens.GRCh38.dna.toplevel.fa.gz
    # We don't know the assembly version a priori, so we use a
    # directory-listing approach via a known suffix.
    fa_dir = f"{_ENSEMBL_FTP_BASE}/{species_path}/dna"
    gff_dir = f"{_ENSEMBL_GFF3_BASE}/{species_path}"
    fa_url = _resolve_pattern_url(fa_dir, suffix=".dna.toplevel.fa.gz")
    gff_url = _resolve_pattern_url(gff_dir, suffix=".gff3.gz", exclude="abinitio")
    return fa_url, gff_url


def _ensembl_genomes_urls(division: str, species_path: str) -> tuple[str, str]:
    """Resolve canonical Ensembl Genomes FASTA + GFF3 URLs."""
    fa_dir = f"{_ENSEMBL_GENOMES_FTP_BASE}/{division}/current/fasta/{species_path}/dna"
    gff_dir = f"{_ENSEMBL_GENOMES_FTP_BASE}/{division}/current/gff3/{species_path}"
    fa_url = _resolve_pattern_url(fa_dir, suffix=".dna.toplevel.fa.gz")
    gff_url = _resolve_pattern_url(gff_dir, suffix=".gff3.gz", exclude="abinitio")
    return fa_url, gff_url


def _refseq_urls(accession: str) -> tuple[str, str]:
    """Resolve canonical RefSeq FTP URLs for a GCF_/GCA_ assembly."""
    # NCBI assembly FTP layout:
    # https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/001/708/105/GCF_001708105.1_ASM170810v1/
    if not (accession.startswith("GCF_") or accession.startswith("GCA_")):
        raise ValueError(f"refseq accession must start with GCF_ or GCA_: {accession!r}")
    prefix = accession.split("_")[0]  # GCF or GCA
    digits = accession.split("_")[1].split(".")[0]  # 001708105
    if len(digits) != 9:
        raise ValueError(
            f"unexpected accession digit-count for {accession!r}; expected 9 digits"
        )
    parts = [digits[0:3], digits[3:6], digits[6:9]]
    base = (
        f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{prefix}/"
        f"{parts[0]}/{parts[1]}/{parts[2]}/"
    )
    # Within that folder is a single subdirectory whose name is
    # <accession>_<asmname> — list and pick.
    asm_dir = _resolve_dir_match(base, prefix=accession)
    asm_url_base = f"{base}{asm_dir}"
    if not asm_url_base.endswith("/"):
        asm_url_base += "/"
    fa_url = f"{asm_url_base}{asm_dir}_genomic.fna.gz"
    gff_url = f"{asm_url_base}{asm_dir}_genomic.gff.gz"
    return fa_url, gff_url


# ──────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────


def _http_get(url: str, *, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "constellation/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_text(url: str, *, timeout: int = 60) -> str:
    return _http_get(url, timeout=timeout).decode("utf-8", errors="replace")


def _resolve_pattern_url(
    listing_url: str, *, suffix: str, exclude: str | None = None
) -> str:
    """Pick a single file from an HTTP directory listing matching
    ``suffix``. Excludes filenames containing ``exclude`` (case-
    insensitive) when supplied (e.g. drop ``abinitio`` GFF3s)."""
    if not listing_url.endswith("/"):
        listing_url = listing_url + "/"
    body = _http_get_text(listing_url)
    candidates: list[str] = []
    # Coarse parse: pull out hrefs / plain filenames matching the suffix.
    for token in body.split('"'):
        if token.endswith(suffix):
            if exclude is not None and exclude.lower() in token.lower():
                continue
            candidates.append(token)
    # Fallback: scan whole-line tokens for nginx-style autoindex.
    if not candidates:
        for line in body.splitlines():
            for tok in line.split():
                if tok.endswith(suffix) and (
                    exclude is None or exclude.lower() not in tok.lower()
                ):
                    candidates.append(tok)
    candidates = list(dict.fromkeys(candidates))  # dedup, preserve order
    if not candidates:
        raise FileNotFoundError(
            f"no file matching {suffix!r} found at {listing_url}"
        )
    # Prefer the shortest filename (canonical assembly is usually
    # shorter than chromosome-specific shards).
    candidates.sort(key=len)
    return listing_url + candidates[0]


def _resolve_dir_match(base_url: str, *, prefix: str) -> str:
    """Pick the single subdirectory under ``base_url`` whose name
    starts with ``prefix`` (used for NCBI assembly dirs)."""
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


def _download_to(url: str, dest: Path, *, timeout: int = 600) -> None:
    """Stream-download a URL to ``dest`` (no in-memory accumulation)."""
    req = urllib.request.Request(url, headers={"User-Agent": "constellation/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _gunzip_to(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as g, dst.open("wb") as out:
        shutil.copyfileobj(g, out)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


_SourceResolver = Callable[[str], tuple[str, str]]


def _source_resolver_for(spec: str) -> tuple[_SourceResolver, str]:
    """Map a ``<source>:<id>`` string to (resolver, id)."""
    if ":" not in spec:
        raise ValueError(
            f"reference spec must be '<source>:<id>'; got {spec!r}. "
            f"Examples: 'refseq:GCF_001708105.1', "
            f"'ensembl_genomes:saccharomyces_cerevisiae'"
        )
    source, ident = spec.split(":", 1)
    source = source.strip().lower()
    ident = ident.strip()
    if source == "ensembl":
        species_path = _ENSEMBL_VERTEBRATE_SPECIES.get(ident, ident)
        return (lambda _: _ensembl_urls(species_path)), ident
    if source == "ensembl_genomes":
        if ident not in _ENSEMBL_GENOMES_SPECIES:
            raise KeyError(
                f"unknown ensembl_genomes species {ident!r}; supported: "
                f"{sorted(_ENSEMBL_GENOMES_SPECIES)}. Use 'constellation "
                f"reference import' for organisms outside this list."
            )
        division, species_path = _ENSEMBL_GENOMES_SPECIES[ident]
        return (lambda _: _ensembl_genomes_urls(division, species_path)), ident
    if source in {"refseq", "genbank"}:
        return (lambda i: _refseq_urls(i)), ident
    raise KeyError(
        f"unknown source {source!r}; supported: 'ensembl', 'ensembl_genomes', "
        f"'refseq', 'genbank'"
    )


def fetch_reference(
    spec: str,
    output_dir: str | Path,
    *,
    timeout: int = 600,
) -> FetchResult:
    """Fetch a genome FASTA + GFF3 by source/id, decompress, parse, and
    save as ``<output_dir>/{genome,annotation}/`` ParquetDir bundles.

    ``spec`` is a ``'<source>:<id>'`` string — see module docstring for
    supported sources. Network access is required; this function never
    runs offline. Failures (connection timeout, missing accession,
    listing parse miss) raise the underlying ``urllib`` /
    ``FileNotFoundError`` / ``KeyError``.
    """
    # Late-import the readers to keep the constellation startup fast
    # for users who never invoke ``reference fetch``.
    from constellation.sequencing.readers.fastx import read_fasta_genome
    from constellation.sequencing.readers.gff import read_gff3

    resolver, ident = _source_resolver_for(spec)
    fa_url, gff_url = resolver(ident)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="constellation_fetch_") as scratch:
        scratch_path = Path(scratch)
        fa_gz = scratch_path / "genome.fa.gz"
        gff_gz = scratch_path / "annotation.gff3.gz"
        fa_decoded = scratch_path / "genome.fa"
        gff_decoded = scratch_path / "annotation.gff3"

        _download_to(fa_url, fa_gz, timeout=timeout)
        _download_to(gff_url, gff_gz, timeout=timeout)

        _gunzip_to(fa_gz, fa_decoded)
        _gunzip_to(gff_gz, gff_decoded)

        genome = read_fasta_genome(fa_decoded)
        contig_name_to_id = {
            row["name"]: row["contig_id"]
            for row in genome.contigs.to_pylist()
        }
        annotation = read_gff3(gff_decoded, contig_name_to_id=contig_name_to_id)

    # Annotate provenance
    genome_meta: dict[str, Any] = dict(genome.metadata_extras)
    genome_meta.update(
        {
            "fetch_source": spec,
            "fetch_url_fasta": fa_url,
        }
    )
    genome = genome.with_metadata(genome_meta)

    annotation_meta: dict[str, Any] = dict(annotation.metadata_extras)
    annotation_meta.update(
        {
            "fetch_source": spec,
            "fetch_url_gff3": gff_url,
        }
    )
    annotation = annotation.with_metadata(annotation_meta)

    annotation.validate_against(genome)

    save_genome_reference(genome, out_dir / "genome")
    save_annotation(annotation, out_dir / "annotation")

    return FetchResult(
        genome=genome,
        annotation=annotation,
        output_dir=out_dir,
        sources={"genome": fa_url, "annotation": gff_url},
    )


__all__ = [
    "FetchResult",
    "fetch_reference",
]
