"""UniProt reference proteomes catalog + SwissProt FASTA fetcher.

Two surfaces:

  * ``fetch_catalog`` / ``parse_index`` — the v1 metadata catalog
    (reads UniProt's ``proteomes.txt`` listing). Catalogs the
    reference-proteome universe; does not materialise FASTAs.
  * ``fetch_swissprot`` — direct FASTA fetcher for the full Swiss-Prot
    knowledgebase. Used by the transcriptome→proteomics pipeline as
    the secondary database for novel-protein competitive alignment.

The catalog source: ``https://ftp.uniprot.org/pub/databases/uniprot/
current_release/knowledgebase/reference_proteomes/README``.

The Swiss-Prot FASTA source:
``https://ftp.uniprot.org/pub/databases/uniprot/current_release/
knowledgebase/complete/uniprot_sprot.fasta.gz``. Cache layout
mirrors the reference-genome cache convention:
``~/.constellation/references/swissprot/<release>/sprot.fasta``.

Per the catalog-PR scope decision: UniProt rows are catalogued and
exposed via ``catalog show`` but ``reference fetch`` (the genome verb)
does not materialise protein FASTAs. ``fetch_swissprot`` is the
proteome-specific materialisation path.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa

from constellation import __version__ as _CONSTELLATION_VERSION
from constellation.catalog._http import http_get_text
from constellation.catalog.schemas import ASSEMBLY_CATALOG_TABLE
from constellation.catalog.types import CatalogRow
from constellation.sequencing.reference.handle import cache_root


try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None


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
# Swiss-Prot FASTA fetcher
# ──────────────────────────────────────────────────────────────────────


_SWISSPROT_BASE = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete"
)
_SWISSPROT_FASTA_NAME = "uniprot_sprot.fasta.gz"
_SWISSPROT_RELDATE_NAME = "reldate.txt"

_USER_AGENT = f"constellation/{_CONSTELLATION_VERSION}"
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
    """A materialised SwissProt FASTA.

    ``fasta_path`` is the gunzipped FASTA on disk; pass it directly to
    consumers that want a FASTA path (e.g. mmseqs2 target, mmseqs
    competitive-target builder). ``release`` is the UniProt release id
    (``YYYY_NN`` format). ``sha256`` is computed on the gunzipped
    FASTA bytes. ``source_url`` is the upstream FASTA URL the bytes
    came from.
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


def _swissprot_release_dir(release: str, *, cache_dir: Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    return cache_root() / "swissprot" / release


def _partial_dir(release_dir: Path) -> Path:
    return release_dir.with_name(release_dir.name + ".partial")


@contextmanager
def _swissprot_fetch_lock(release_dir: Path) -> Iterator[Path]:
    """Per-release fetch lock; blocks concurrent invocations.

    Mirrors :func:`constellation.sequencing.reference.handle.acquire_fetch_lock`
    but lock path lives directly under the swissprot cache dir. Uses
    ``fcntl.flock`` on POSIX; degrades to best-effort presence check on
    Windows-native.
    """
    parent = release_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f"{release_dir.name}.lock"

    if _fcntl is None:
        # Windows-native fallback.
        lock_path.touch()
        try:
            yield lock_path
        finally:
            if lock_path.exists():
                lock_path.unlink()
        return

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
        yield lock_path
    finally:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _release_dir_is_complete(release_dir: Path) -> bool:
    if not release_dir.is_dir():
        return False
    if not (release_dir / "_SUCCESS").exists():
        return False
    if not (release_dir / "sprot.fasta").is_file():
        return False
    return True


def _download_to(url: str, dest: Path, *, timeout: int) -> tuple[str | None, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        etag = resp.headers.get("ETag")
        last_modified = resp.headers.get("Last-Modified")
        shutil.copyfileobj(resp, out)
    return etag, last_modified


def _gunzip_to(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as g, dst.open("wb") as out:
        shutil.copyfileobj(g, out)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_meta_toml(
    release_dir: Path,
    *,
    release: str,
    source_url: str,
    sha256: str,
    etag: str | None,
    last_modified: str | None,
) -> None:
    """Write meta.toml with provenance details.

    Hand-rolled TOML (the fetch path is stdlib-only). Mirrors the
    keys of :func:`sequencing.reference.handle.write_meta_toml`'s
    output where applicable.
    """
    lines = [
        f'release = "{release}"',
        f'source_url = "{source_url}"',
        f'sha256 = "{sha256}"',
        f'constellation_version = "{_CONSTELLATION_VERSION}"',
        f'fetched_at = "{datetime.now(timezone.utc).isoformat()}"',
    ]
    if etag is not None:
        lines.append(f'etag = "{etag}"')
    if last_modified is not None:
        lines.append(f'last_modified = "{last_modified}"')
    (release_dir / "meta.toml").write_text("\n".join(lines) + "\n")


def fetch_swissprot(
    *,
    release: str | None = None,
    cache_dir: Path | None = None,
    timeout: int = 600,
    force: bool = False,
) -> SwissprotHandle:
    """Materialise the UniProt Swiss-Prot FASTA into the local cache.

    Lazy-fetches when the requested release isn't cached; idempotent
    re-runs short-circuit on ``_SUCCESS``.

    Parameters
    ----------
    release
        UniProt release id (``YYYY_NN``). ``None`` (default) auto-detects
        via ``reldate.txt`` — the current release at fetch time.
    cache_dir
        Explicit cache directory. ``None`` (default) places the FASTA
        under ``<cache_root>/swissprot/<release>/`` following the
        reference-cache convention (`cache_root()` from
        :mod:`constellation.sequencing.reference.handle`).
    timeout
        HTTP timeout in seconds (default 600 — Swiss-Prot is ~90 MB
        compressed; tight nodes can be slow).
    force
        When ``True``, re-fetch even if the cache slot is complete.

    Returns
    -------
    SwissprotHandle
        Carries the on-disk FASTA path + release + sha256 + source URL.
        Pass ``.fasta_path`` to downstream tools (mmseqs2 target,
        FASTA-merger, etc.).

    Raises
    ------
    ValueError
        Release autodetect failed (``reldate.txt`` unreachable or
        unparseable).
    urllib.error.URLError / OSError
        Network failure during fetch.
    """
    if release is None:
        release = _probe_swissprot_release(timeout=timeout)

    release_dir = _swissprot_release_dir(release, cache_dir=cache_dir)

    # Fast path: cache hit + not forced.
    if not force and _release_dir_is_complete(release_dir):
        sha = _sha256_of(release_dir / "sprot.fasta")
        return SwissprotHandle(
            fasta_path=release_dir / "sprot.fasta",
            release=release,
            sha256=sha,
            source_url=swissprot_fasta_url(),
        )

    with _swissprot_fetch_lock(release_dir):
        # Re-check inside the lock (another worker may have completed
        # the fetch while we were blocked).
        if not force and _release_dir_is_complete(release_dir):
            sha = _sha256_of(release_dir / "sprot.fasta")
            return SwissprotHandle(
                fasta_path=release_dir / "sprot.fasta",
                release=release,
                sha256=sha,
                source_url=swissprot_fasta_url(),
            )

        if force and release_dir.exists():
            shutil.rmtree(release_dir)
        # Clean any leftover .partial from a prior crash.
        stage = _partial_dir(release_dir)
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True, exist_ok=False)

        try:
            fasta_gz = stage / "sprot.fasta.gz"
            fasta = stage / "sprot.fasta"
            source_url = swissprot_fasta_url()
            try:
                etag, last_modified = _download_to(
                    source_url, fasta_gz, timeout=timeout
                )
            except (urllib.error.URLError, OSError) as exc:
                print(
                    f"error: failed to fetch {source_url}: {exc}",
                    file=sys.stderr,
                )
                raise
            _gunzip_to(fasta_gz, fasta)
            sha = _sha256_of(fasta)
            _write_meta_toml(
                stage,
                release=release,
                source_url=source_url,
                sha256=sha,
                etag=etag,
                last_modified=last_modified,
            )
            (stage / "_SUCCESS").write_bytes(b"")
            # Atomic promote.
            os.rename(stage, release_dir)
        except BaseException:
            # Clean staging on any error so future runs aren't blocked
            # by a half-written partial.
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            raise

    return SwissprotHandle(
        fasta_path=release_dir / "sprot.fasta",
        release=release,
        sha256=sha,
        source_url=swissprot_fasta_url(),
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
