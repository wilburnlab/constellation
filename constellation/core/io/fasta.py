"""FASTA reading and writing, once for the whole package.

Before this module the package had four FASTA parsers, two header
scanners, a record counter and a dozen writers, most of them private
copies of one another. They agreed on the format only by habit; one of
them read no gzip and kept the ``\\r`` of a CRLF file in its sequences.
This is the one implementation they now call.

Reading
-------
:func:`iter_fasta` yields :class:`FastaRecord` (the header line and the
sequence); :func:`iter_fasta_batches` and :func:`read_fasta` give the same
records as Arrow (:data:`FASTA_SCHEMA`). The rules, which are the ones
``sequencing.readers.fastx`` had:

- gzip is read transparently, recognised by its magic bytes;
- line endings (``\\n`` or ``\\r\\n``) are removed;
- blank lines, and lines starting with ``;`` (the old FASTA comment), are
  skipped;
- a header is the line after ``>``, trailing whitespace removed; its id is
  the text before the first whitespace and its description the rest;
- a sequence is its lines joined, with spaces and tabs removed. Case is
  kept: whether to fold it is the caller's decision.

Writing
-------
:func:`write_fasta` writes one record per sequence, ``>{id}[ {description}]``
then the sequence on one line (no wrapping, which every consumer here
expects). Sequences and ids may be Python sequences or Arrow arrays; Arrow
columns are converted in slices of ``chunk_rows``, which the de novo
corpus measured as ten times faster than a vectorised emitter and bounds
the strings alive at once. :func:`format_fasta` is the same into a string.

Headers
-------
:func:`parse_header` reads the conventions found in the files this package
meets: UniProt (``sp|P02769|ALBU_BOVIN Albumin OS=Bos taurus OX=9913
GN=ALB PE=1 SV=4``), NCBI bracket tags (``[gene=ALB]``,
``[organism=Homo sapiens]``) and bare ``key=value`` tokens (``gene=ALB``,
``avg_tpm=1.5``). Every tag is kept, in header order.
"""

from __future__ import annotations

import itertools
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import IO, Any

import pyarrow as pa

from constellation.core.io.text import open_text

__all__ = [
    "FASTA_SCHEMA",
    "FastaHeader",
    "FastaRecord",
    "count_fasta_records",
    "format_fasta",
    "iter_fasta",
    "iter_fasta_batches",
    "parse_header",
    "read_fasta",
    "split_header",
    "write_fasta",
]

#: One row per record. ``sequence`` is a large string because a genome's
#: contigs together pass the 2 GiB a plain string array can hold.
FASTA_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("sequence", pa.large_string(), nullable=False),
    ],
    metadata={b"schema_name": b"FastaRecord", b"schema_version": b"1"},
)


def split_header(header: str) -> tuple[str, str]:
    """``(id, description)`` of a header: split at the first whitespace."""
    parts = header.split(None, 1)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


@dataclass(frozen=True, slots=True)
class FastaRecord:
    """One record: its header (without ``>``) and its sequence."""

    header: str
    sequence: str

    @property
    def id(self) -> str:
        return split_header(self.header)[0]

    @property
    def description(self) -> str:
        return split_header(self.header)[1]


# ──────────────────────────────────────────────────────────────────────
# Reading
# ──────────────────────────────────────────────────────────────────────


def _joined(chunks: list[str]) -> str:
    return "".join(chunks).replace(" ", "").replace("\t", "")


def iter_fasta(path: str | Path) -> Iterator[FastaRecord]:
    """Yield the records of a FASTA file in order (rules in the module
    docstring). Sequence lines before the first header are ignored."""
    header: str | None = None
    chunks: list[str] = []
    with open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                if header is not None:
                    yield FastaRecord(header, _joined(chunks))
                header = line[1:].rstrip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield FastaRecord(header, _joined(chunks))


def iter_fasta_batches(path: str | Path, *, batch_rows: int = 65_536) -> Iterator[pa.RecordBatch]:
    """The records of a FASTA file as :data:`FASTA_SCHEMA` record batches of
    at most ``batch_rows`` rows."""
    if batch_rows < 1:
        raise ValueError(f"batch_rows must be >= 1, got {batch_rows}")
    ids: list[str] = []
    descriptions: list[str] = []
    sequences: list[str] = []

    def batch() -> pa.RecordBatch:
        return pa.record_batch(
            [
                pa.array(ids, pa.string()),
                pa.array(descriptions, pa.string()),
                pa.array(sequences, pa.large_string()),
            ],
            schema=FASTA_SCHEMA,
        )

    for record in iter_fasta(path):
        rec_id, description = split_header(record.header)
        ids.append(rec_id)
        descriptions.append(description)
        sequences.append(record.sequence)
        if len(ids) == batch_rows:
            yield batch()
            ids, descriptions, sequences = [], [], []
    if ids:
        yield batch()


def read_fasta(path: str | Path) -> pa.Table:
    """Every record of a FASTA file as one :data:`FASTA_SCHEMA` table."""
    return pa.Table.from_batches(list(iter_fasta_batches(path)), schema=FASTA_SCHEMA)


def count_fasta_records(path: str | Path) -> int:
    """The number of header lines (``>``) in a FASTA file."""
    with open_text(path) as fh:
        return sum(1 for line in fh if line.startswith(">"))


# ──────────────────────────────────────────────────────────────────────
# Writing
# ──────────────────────────────────────────────────────────────────────


def _values(values: Any, chunk_rows: int) -> Iterator[Any]:
    """Iterate a Python iterable, or an Arrow array in ``chunk_rows`` slices."""
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        for lo in range(0, len(values), chunk_rows):
            yield from values.slice(lo, chunk_rows).to_pylist()
    else:
        yield from values


def _write_records(
    fh: IO[str],
    sequences: Any,
    ids: Any,
    descriptions: Any,
    *,
    start_row: int,
    skip_empty: bool,
    chunk_rows: int,
) -> int:
    write = fh.write
    seqs = _values(sequences, chunk_rows)
    n = 0
    if ids is None and descriptions is None and not skip_empty:
        # The de novo corpus's form: records named by row, nothing to check.
        for row, seq in enumerate(seqs, start_row):
            write(f">{row}\n{seq if seq is not None else ''}\n")
            n += 1
        return n
    # Sequences drive the iteration; ids and descriptions must match them.
    named = (
        zip(seqs, _values(ids, chunk_rows), strict=True)
        if ids is not None
        else zip(seqs, itertools.count(start_row))
    )
    rows = (
        ((seq, name, d) for (seq, name), d in zip(named, _values(descriptions, chunk_rows), strict=True))
        if descriptions is not None
        else ((seq, name, None) for seq, name in named)
    )
    for seq, name, description in rows:
        if skip_empty and not seq:
            continue
        head = f"{name} {description}" if description else f"{name}"
        if seq is not None and len(seq) > _LONG_SEQUENCE:
            # A contig: write it as it is rather than copy it into one string.
            write(f">{head}\n")
            write(seq)
            write("\n")
        else:
            write(f">{head}\n{seq if seq is not None else ''}\n")
        n += 1
    return n


#: Above this many characters a sequence is written on its own, not
#: formatted into one string with its header (a chromosome is ~250 MB).
_LONG_SEQUENCE = 1 << 20


def write_fasta(
    dest: str | Path | IO[str],
    sequences: Iterable[str | None] | pa.Array | pa.ChunkedArray,
    ids: Iterable[Any] | pa.Array | pa.ChunkedArray | None = None,
    *,
    descriptions: Iterable[str | None] | pa.Array | pa.ChunkedArray | None = None,
    start_row: int = 0,
    skip_empty: bool = False,
    chunk_rows: int = 50_000,
    atomic: bool = False,
) -> int:
    """Write ``sequences`` as FASTA records and return how many were written.

    ``dest`` is a path (created, or replaced) or an open text handle, which
    is written to and left open, so one file can be written across several
    calls. ``ids`` names each record; without it records are named by row,
    counting from ``start_row``. A ``description`` is written after the id
    and one space, and left out when empty or null. With ``skip_empty`` a
    record whose sequence is empty or null is not written (nor counted);
    without it a null sequence is written as an empty one. ``atomic``
    (paths only) writes beside ``dest`` and renames, so a reader never sees
    half a file.
    """
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    kw = dict(start_row=start_row, skip_empty=skip_empty, chunk_rows=chunk_rows)
    if hasattr(dest, "write"):
        if atomic:
            raise ValueError("atomic=True needs a path, not an open handle")
        return _write_records(dest, sequences, ids, descriptions, **kw)  # type: ignore[arg-type]
    path = Path(dest)  # type: ignore[arg-type]
    target = path.with_name(path.name + ".tmp") if atomic else path
    try:
        with target.open("w", encoding="utf-8") as fh:
            n = _write_records(fh, sequences, ids, descriptions, **kw)
        if atomic:
            os.replace(target, path)
    except BaseException:
        if atomic:
            target.unlink(missing_ok=True)
        raise
    return n


def format_fasta(
    sequences: Iterable[str | None] | pa.Array | pa.ChunkedArray,
    ids: Iterable[Any] | pa.Array | pa.ChunkedArray | None = None,
    *,
    descriptions: Iterable[str | None] | pa.Array | pa.ChunkedArray | None = None,
    skip_empty: bool = False,
) -> str:
    """:func:`write_fasta` into a string."""
    buf = StringIO()
    write_fasta(buf, sequences, ids, descriptions=descriptions, skip_empty=skip_empty)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────
# Headers
# ──────────────────────────────────────────────────────────────────────

_UNIPROT_KEYS = "OS|OX|GN|PE|SV"
_TAG = re.compile(
    r"\[(?P<bracket_key>[A-Za-z_][\w.-]*)=(?P<bracket_value>[^\]]*)\]"
    rf"|\b(?P<uniprot_key>{_UNIPROT_KEYS})=(?P<uniprot_value>.*?)(?=\s+(?:{_UNIPROT_KEYS})=|\s*$)"
    r"|\b(?P<key>[A-Za-z_][\w.-]*)=(?P<value>\S*)"
)


@dataclass(frozen=True, slots=True)
class FastaHeader:
    """What a FASTA header says.

    ``accession`` is the middle field of a pipe-delimited id
    (``sp|P02769|ALBU_BOVIN`` → ``P02769``), else the id itself;
    ``database`` and ``entry_name`` are the first and third fields when
    the id has them. ``tags`` holds every ``key=value`` found in the
    description, in order, whichever convention wrote it.
    """

    id: str
    description: str
    database: str | None
    accession: str
    entry_name: str | None
    tags: tuple[tuple[str, str], ...]

    def tag(self, key: str) -> str | None:
        """The first value of ``key``, or None."""
        return next((v for k, v in self.tags if k == key), None)

    @property
    def gene(self) -> str | None:
        """The first gene tag (UniProt ``GN=``, NCBI ``[gene=]`` or ``gene=``),
        up to its first whitespace, since a gene symbol has none; None when
        there is no gene tag or its value is empty."""
        value = next((v for k, v in self.tags if k in ("gene", "GN")), None)
        if not value:
            return None
        return value.split(None, 1)[0]


def parse_header(header: str) -> FastaHeader:
    """Parse a FASTA header line (with or without its ``>``)."""
    rec_id, description = split_header(header[1:] if header.startswith(">") else header)
    database = entry_name = None
    accession = rec_id
    if "|" in rec_id:
        parts = rec_id.split("|")
        if len(parts) >= 2 and parts[1]:
            database = parts[0] or None
            accession = parts[1]
            entry_name = parts[2] if len(parts) >= 3 and parts[2] else None
    tags: list[tuple[str, str]] = []
    for m in _TAG.finditer(description):
        if m.group("bracket_key") is not None:
            tags.append((m.group("bracket_key"), m.group("bracket_value")))
        elif m.group("uniprot_key") is not None:
            tags.append((m.group("uniprot_key"), m.group("uniprot_value")))
        else:
            tags.append((m.group("key"), m.group("value")))
    return FastaHeader(rec_id, description, database, accession, entry_name, tuple(tags))
