"""Download + refold NCBI's ``taxdump.tar.gz`` to Arrow.

Source: ``https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz``

The tarball ships four ``.dmp`` files we care about — ``nodes.dmp``,
``names.dmp``, ``merged.dmp``, ``delnodes.dmp`` — in a custom
tab-pipe-tab field separator. ``parse_taxdump_archive(tar_path)``
returns the three Arrow tables ``TaxonomyResolver`` consumes; the
``fetch_taxdump`` entry point chains download → parse → write.
"""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa

from constellation import __version__ as _CONSTELLATION_VERSION
from constellation.core.taxonomy.schemas import (
    TAXONOMY_MERGED_TABLE,
    TAXONOMY_NAMES_TABLE,
    TAXONOMY_NODES_TABLE,
)
from constellation.core.taxonomy.store import update_current_pointer, write_bundle


NCBI_TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
_FIELD_SEP = "\t|\t"
_LINE_END = "\t|"


# ──────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────


def parse_taxdump_archive(
    tar_path: Path | str,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Read a ``taxdump.tar.gz`` and return ``(nodes, names, merged)``.

    The archive contains plain-text ``.dmp`` members; we read them
    in-memory rather than extracting to disk.
    """
    tar_path = Path(tar_path)
    nodes_lines: list[bytes] | None = None
    names_lines: list[bytes] | None = None
    merged_lines: list[bytes] | None = None
    with tarfile.open(tar_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.split("/")[-1]
            if name == "nodes.dmp":
                with tar.extractfile(member) as fh:  # type: ignore[union-attr]
                    nodes_lines = fh.read().splitlines()
            elif name == "names.dmp":
                with tar.extractfile(member) as fh:  # type: ignore[union-attr]
                    names_lines = fh.read().splitlines()
            elif name == "merged.dmp":
                with tar.extractfile(member) as fh:  # type: ignore[union-attr]
                    merged_lines = fh.read().splitlines()
    if nodes_lines is None:
        raise ValueError(f"nodes.dmp not found in {tar_path!s}")
    if names_lines is None:
        raise ValueError(f"names.dmp not found in {tar_path!s}")
    if merged_lines is None:
        merged_lines = []

    # Parse names first — nodes wants the scientific-name column denormalised.
    names_tbl = _parse_names_dmp(names_lines)
    scientific = _scientific_name_index(names_tbl)
    nodes_tbl = _parse_nodes_dmp(nodes_lines, scientific=scientific)
    merged_tbl = _parse_merged_dmp(merged_lines)
    return nodes_tbl, names_tbl, merged_tbl


def _split_dmp_line(line: bytes) -> list[str]:
    s = line.decode("utf-8", errors="replace")
    # NCBI .dmp lines end with "\t|" then optional trailing whitespace.
    if s.endswith(_LINE_END):
        s = s[: -len(_LINE_END)]
    parts = s.split(_FIELD_SEP)
    return [p.strip() for p in parts]


def _parse_nodes_dmp(
    lines: list[bytes],
    *,
    scientific: dict[int, str],
) -> pa.Table:
    taxids: list[int] = []
    parents: list[int] = []
    ranks: list[str] = []
    div_ids: list[int | None] = []
    gc_ids: list[int | None] = []
    mt_ids: list[int | None] = []
    sci_names: list[str] = []
    for raw in lines:
        if not raw:
            continue
        fields = _split_dmp_line(raw)
        # nodes.dmp columns:
        #   0 tax_id
        #   1 parent_tax_id
        #   2 rank
        #   3 embl_code
        #   4 division_id
        #   5 inherited_div_flag
        #   6 genetic_code_id
        #   7 inherited_GC_flag
        #   8 mitochondrial_genetic_code_id
        #   9 inherited_MGC_flag
        #  10 GenBank_hidden_flag
        #  11 hidden_subtree_root_flag
        #  12 comments
        try:
            taxid = int(fields[0])
            parent = int(fields[1])
        except (ValueError, IndexError):
            continue
        rank = fields[2] if len(fields) > 2 else "no rank"
        div_id = _maybe_int(fields[4]) if len(fields) > 4 else None
        gc_id = _maybe_int(fields[6]) if len(fields) > 6 else None
        mt_id = _maybe_int(fields[8]) if len(fields) > 8 else None
        taxids.append(taxid)
        parents.append(parent)
        ranks.append(rank or "no rank")
        div_ids.append(div_id)
        gc_ids.append(gc_id)
        mt_ids.append(mt_id)
        sci_names.append(scientific.get(taxid, ""))
    return pa.table(
        {
            "taxid": pa.array(taxids, type=pa.int64()),
            "parent_taxid": pa.array(parents, type=pa.int64()),
            "rank": pa.array(ranks, type=pa.string()),
            "division_id": pa.array(div_ids, type=pa.int16()),
            "genetic_code_id": pa.array(gc_ids, type=pa.int16()),
            "mito_genetic_code_id": pa.array(mt_ids, type=pa.int16()),
            "scientific_name": pa.array(sci_names, type=pa.string()),
        },
        schema=TAXONOMY_NODES_TABLE,
    )


def _parse_names_dmp(lines: list[bytes]) -> pa.Table:
    taxids: list[int] = []
    names: list[str] = []
    names_lower: list[str] = []
    name_classes: list[str] = []
    unique_names: list[str | None] = []
    for raw in lines:
        if not raw:
            continue
        fields = _split_dmp_line(raw)
        # names.dmp columns:
        #   0 tax_id
        #   1 name_txt
        #   2 unique_name
        #   3 name_class
        try:
            taxid = int(fields[0])
            name_txt = fields[1]
            unique = fields[2] if len(fields) > 2 else ""
            name_class = fields[3] if len(fields) > 3 else "synonym"
        except (ValueError, IndexError):
            continue
        if not name_txt:
            continue
        taxids.append(taxid)
        names.append(name_txt)
        names_lower.append(name_txt.lower())
        name_classes.append(name_class)
        unique_names.append(unique or None)
    return pa.table(
        {
            "taxid": pa.array(taxids, type=pa.int64()),
            "name": pa.array(names, type=pa.string()),
            "name_lower": pa.array(names_lower, type=pa.string()),
            "name_class": pa.array(name_classes, type=pa.string()),
            "unique_name": pa.array(unique_names, type=pa.string()),
        },
        schema=TAXONOMY_NAMES_TABLE,
    )


def _scientific_name_index(names_tbl: pa.Table) -> dict[int, str]:
    """Pull just the ``scientific name`` rows into a ``taxid → name`` dict."""
    if names_tbl.num_rows == 0:
        return {}
    classes = names_tbl.column("name_class").to_pylist()
    taxids = names_tbl.column("taxid").to_pylist()
    names = names_tbl.column("name").to_pylist()
    out: dict[int, str] = {}
    for c, t, n in zip(classes, taxids, names):
        if c == "scientific name":
            out[t] = n
    return out


def _parse_merged_dmp(lines: list[bytes]) -> pa.Table:
    olds: list[int] = []
    news: list[int] = []
    for raw in lines:
        if not raw:
            continue
        fields = _split_dmp_line(raw)
        try:
            olds.append(int(fields[0]))
            news.append(int(fields[1]))
        except (ValueError, IndexError):
            continue
    return pa.table(
        {
            "old_taxid": pa.array(olds, type=pa.int64()),
            "new_taxid": pa.array(news, type=pa.int64()),
        },
        schema=TAXONOMY_MERGED_TABLE,
    )


def _maybe_int(s: str) -> int | None:
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────────────
# Download + write
# ──────────────────────────────────────────────────────────────────────


def fetch_taxdump(
    *,
    dest_root: Path | None = None,
    timeout: int = 600,
    url: str = NCBI_TAXDUMP_URL,
    progress_cb: Any = None,
) -> Path:
    """Download ``taxdump.tar.gz``, parse, write the Arrow bundle, point
    ``current`` at it. Returns the bundle directory."""
    from constellation.core.taxonomy.store import taxonomy_root

    root = dest_root or taxonomy_root()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    bundle_dir = root / f"ncbi-{stamp}"

    with tempfile.TemporaryDirectory(prefix="constellation-taxdump-") as td:
        tar_path = Path(td) / "taxdump.tar.gz"
        sha = _download(url, tar_path, timeout=timeout, progress_cb=progress_cb)
        nodes, names, merged = parse_taxdump_archive(tar_path)
        meta = {
            "schema_version": 1,
            "source": "ncbi",
            "source_url": url,
            "sha256": sha,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "constellation_version": _CONSTELLATION_VERSION,
            "n_nodes": nodes.num_rows,
            "n_names": names.num_rows,
            "n_merged": merged.num_rows,
        }
        write_bundle(bundle_dir, nodes=nodes, names=names, merged=merged, meta=meta)
    update_current_pointer(bundle_dir, root=root)
    return bundle_dir


def _download(
    url: str, dest: Path, *, timeout: int = 600, progress_cb: Any = None
) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": f"constellation/{_CONSTELLATION_VERSION}"},
    )
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        seen = 0
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            seen += len(chunk)
            if progress_cb is not None and total:
                progress_cb(seen, total)
    return h.hexdigest()


__all__ = [
    "NCBI_TAXDUMP_URL",
    "fetch_taxdump",
    "parse_taxdump_archive",
]
