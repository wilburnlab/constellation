"""A PAF decoder that reads six fields eagerly and the rest on demand.

:mod:`constellation.sequencing.readers.paf` is the general reader and stays
the right tool for anything that wants a ``PAF_RECORD_SCHEMA`` table. It is
the wrong tool for the E-step, for a reason about shape rather than
implementation quality: with ``--secondary=yes -N 500`` a read emits up to 501
lines, of which **exactly one** is kept. Building all twelve fixed columns as
Arrow string arrays — a gather plus a copy each — and regex-extracting four
tags for every one of ~480M rows per round spends almost all of its time on
rows that are about to be discarded.

What every row genuinely needs is ``q_name``, ``t_name``, ``strand``, and
``n_match`` / ``aln_len`` — the last two because the **admission gate** is
``n_match / aln_len >= p_floor`` and it judges every candidate, not just
winners — plus ``AS:i`` for the shortlist and the per-round diagnostics. Only
the winners need ``q_start / q_end / t_start / t_end / q_len / t_len`` and the
``cg:Z`` CIGAR. So :func:`scan_paf_block` decodes the six and returns a
:class:`HitBlock` holding the raw bytes plus a row/tab offset index, from
which the rest is decoded for whatever subset the caller asks for.

Measured on a 143 MB block of 1.02M rows (20k reads x 51 hits), decoding the
winners only: **42 MB/s -> 122 MB/s, 2.9x** end to end, with the eager half
alone at 152 MB/s. At 216 GB of PAF per round that is 85 min -> 29 min of
single-threaded reduction.

Three things make the eager half cheap:

* **Integer names.** The corpus FASTA names each read by its row index and the
  template FASTA names each template by its row index, so ``q_name`` and
  ``t_name`` parse as integers straight out of the byte buffer — no string
  array, no ``cast``. Read grouping is then ``np.diff`` on int64, and the row
  index needed to look a sequence back up rides along for free.
* **A right-aligned digit matrix** rather than the ragged
  repeat/gather/``reduceat`` form — 3.5x faster, see :func:`_parse_int_field`.
* **Seeded tag search.** ``AS:i:`` is found by locating the rare byte ``S``
  and checking its neighbours, rather than by full-block comparisons or a
  regex over a materialised tag column.

Group contiguity is assumed and is load-bearing: minimap2 emits a query's hits
together, which holds for a **single index part**. ``run_estep``'s ``-I`` guard
is what enforces that, so it protects grouping here as well as the read-mass
double-counting it was written for. Relaxing it silently corrupts this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa


_TAB = ord("\t")
_NL = ord("\n")
_MINUS = ord("-")
_ZERO = ord("0")
_N_FIXED = 12

# Field index within the 12 mandatory PAF columns.
_COL = {
    "q_name": 0,
    "q_len": 1,
    "q_start": 2,
    "q_end": 3,
    "strand": 4,
    "t_name": 5,
    "t_len": 6,
    "t_start": 7,
    "t_end": 8,
    "n_match": 9,
    "aln_len": 10,
    "mapq": 11,
}

# Place values for the digit-parser. PAF integers are far short of this.
_POW10 = (10 ** np.arange(19, dtype=np.int64)).astype(np.int64)


def _parse_int_field(
    buf: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Parse ``buf[starts[i]:ends[i]]`` as base-10 integers, vectorized.

    Right-aligns every field into an ``(n, max_width)`` digit matrix and takes
    one dot product with the place values. The ragged alternative — repeat /
    arange / gather / ``np.add.reduceat`` — is the obvious form and is **3.5x
    slower** here (0.220 s vs 0.062 s on 1.02M fields), because it builds three
    int64 temporaries the length of the *concatenated digits* where this builds
    two the shape of the matrix. PAF integers are narrow, so `max_width` is
    ~5-7 and the matrix is small.

    Handles an optional leading ``-``: the sign sits at ``starts`` and the
    digits are taken from ``ends`` backwards, so it falls outside the matrix
    on its own.
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    n = starts.size
    if n == 0:
        return np.empty(0, dtype=np.int64)

    neg = buf[starts] == _MINUS
    widths = ends - (starts + neg)
    if np.any(widths <= 0):
        bad = int(np.flatnonzero(widths <= 0)[0])
        raise ValueError(f"empty integer field at row {bad}")
    max_width = int(widths.max())
    if max_width > _POW10.size:
        raise ValueError("integer field too wide to parse")

    cols = np.arange(max_width, dtype=np.int64)
    idx = (ends[:, None] - max_width) + cols
    np.clip(idx, 0, buf.size - 1, out=idx)
    digits = buf[idx].astype(np.int64) - _ZERO
    active = cols >= (max_width - widths)[:, None]
    if np.any(active & ((digits < 0) | (digits > 9))):
        bad = int(
            np.flatnonzero((active & ((digits < 0) | (digits > 9))).any(axis=1))[0]
        )
        raise ValueError(f"non-digit byte in integer field at row {bad}")
    digits *= active
    out = digits @ _POW10[max_width - 1 - cols]
    return np.where(neg, -out, out)


@dataclass(frozen=True, slots=True)
class HitBlock:
    """One decoded PAF block: six fields eager, the rest addressable.

    ``read_row`` / ``template_row`` / ``as_score`` are parallel arrays over the
    **surviving** hits — reverse-strand hits and hits to unknown templates are
    dropped during the scan, so a consumer never filters again. ``src_row``
    maps each surviving hit back to its row in the block, which is what the
    deferred decoders index with.
    """

    read_row: np.ndarray  # int64 — q_name, the corpus row index
    template_row: np.ndarray  # int64 — t_name, the TemplateSet row index
    as_score: np.ndarray  # int64 — AS:i, 0 when the tag is absent
    n_match: np.ndarray  # int64 — PAF col 10; the identity gate's numerator
    aln_len: np.ndarray  # int64 — PAF col 11; its denominator
    src_row: np.ndarray  # int64 — index into the block's rows
    n_rows: int  # rows in the block, before dropping
    n_dropped_strand: int
    n_dropped_template: int
    # Deferred-decode handles. `buf` is kept alive by this dataclass.
    buf: np.ndarray
    row_start: np.ndarray
    row_end: np.ndarray
    tab: np.ndarray
    first_tab: np.ndarray
    #: ``s1:i`` chaining score per surviving hit, decoded only when the scan
    #: asked for it (the two-pass E-step's shortlist key); else ``None``.
    chain_score: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.read_row.size)

    def take(self, sel: np.ndarray) -> HitBlock:
        """A view over a subset of the surviving hits."""
        sel = np.asarray(sel, dtype=np.int64)
        return HitBlock(
            read_row=self.read_row[sel],
            template_row=self.template_row[sel],
            as_score=self.as_score[sel],
            n_match=self.n_match[sel],
            aln_len=self.aln_len[sel],
            src_row=self.src_row[sel],
            n_rows=self.n_rows,
            n_dropped_strand=self.n_dropped_strand,
            n_dropped_template=self.n_dropped_template,
            buf=self.buf,
            row_start=self.row_start,
            row_end=self.row_end,
            tab=self.tab,
            first_tab=self.first_tab,
            chain_score=None if self.chain_score is None else self.chain_score[sel],
        )

    def int_fields(
        self, hits: np.ndarray, columns: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        """Decode integer columns for ``hits`` (indices into the survivors)."""
        rows = self.src_row[np.asarray(hits, dtype=np.int64)]
        ft = self.first_tab[rows]
        out: dict[str, np.ndarray] = {}
        for name in columns:
            k = _COL[name]
            starts = self.row_start[rows] if k == 0 else self.tab[ft + (k - 1)] + 1
            ends = (
                self.tab[ft + k] if k < _N_FIXED - 1 else self._last_field_end(rows, ft)
            )
            out[name] = _parse_int_field(self.buf, starts, ends)
        return out

    def _last_field_end(self, rows: np.ndarray, ft: np.ndarray) -> np.ndarray:
        """End of field 11, which is a tab when tags follow and EOL when not."""
        cand = ft + (_N_FIXED - 1)
        in_range = cand < self.tab.size
        end = np.where(
            in_range,
            self.tab[np.clip(cand, 0, max(self.tab.size - 1, 0))],
            self.row_end[rows],
        )
        # A tab belonging to a later row means this row had no tags.
        return np.minimum(end, self.row_end[rows])

    def cigars(self, hits: np.ndarray) -> pa.Array:
        """The ``cg:Z`` CIGAR of each of ``hits``, as a large_string array.

        Vectorized: find every ``cg:Z:`` in the block by seeding on ``Z``,
        map each to its row, then gather the bytes for the requested rows.
        Null where a row carries no CIGAR.
        """
        want = self.src_row[np.asarray(hits, dtype=np.int64)]
        starts, ends, cg_row = self._cigar_spans()
        if cg_row.size == 0:
            return pa.nulls(want.size, pa.large_string())

        pos = np.searchsorted(cg_row, want)
        pos_ok = (pos < cg_row.size) & (
            cg_row[np.clip(pos, 0, cg_row.size - 1)] == want
        )
        sel = np.clip(pos, 0, cg_row.size - 1)
        s = np.where(pos_ok, starts[sel], 0)
        e = np.where(pos_ok, ends[sel], 0)
        return _gather_large_string(self.buf, s, e, valid=pos_ok)

    def _cigar_spans(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        buf = self.buf
        z = np.flatnonzero(buf == ord("Z"))
        if z.size == 0:
            return (np.empty(0, np.int64),) * 3
        z = z[z >= 3]
        if z.size == 0:
            return (np.empty(0, np.int64),) * 3
        ok = (
            (buf[z - 1] == ord(":"))
            & (buf[z - 2] == ord("g"))
            & (buf[z - 3] == ord("c"))
            & (z + 1 < buf.size)
            & (buf[np.clip(z + 1, 0, buf.size - 1)] == ord(":"))
        )
        z = z[ok]
        if z.size == 0:
            return (np.empty(0, np.int64),) * 3
        starts = z + 2
        rows = np.searchsorted(self.row_end, starts, side="left")
        # End at the next tab, or the row's newline, whichever comes first.
        nxt = np.searchsorted(self.tab, starts, side="left")
        tab_end = np.where(
            nxt < self.tab.size,
            self.tab[np.clip(nxt, 0, max(self.tab.size - 1, 0))],
            self.buf.size,
        )
        ends = np.minimum(
            tab_end, self.row_end[np.clip(rows, 0, self.row_end.size - 1)]
        )
        return starts.astype(np.int64), ends.astype(np.int64), rows.astype(np.int64)


def _gather_large_string(
    buf: np.ndarray, starts: np.ndarray, ends: np.ndarray, *, valid: np.ndarray
) -> pa.Array:
    """Build a large_string array from ``buf[starts[i]:ends[i]]`` spans."""
    n = starts.size
    widths = np.where(valid, ends - starts, 0).astype(np.int64)
    widths = np.maximum(widths, 0)
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(widths, out=off[1:])
    total = int(off[-1])
    if total == 0:
        data = b""
    else:
        seg = np.repeat(np.arange(n, dtype=np.int64), widths)
        within = np.arange(total, dtype=np.int64) - off[seg]
        data = buf[np.repeat(starts, widths) + within].tobytes()
    null_bitmap = None
    null_count = 0
    if not valid.all():
        null_count = int((~valid).sum())
        null_bitmap = pa.py_buffer(np.packbits(valid, bitorder="little").tobytes())
    return pa.LargeStringArray.from_buffers(
        length=n,
        value_offsets=pa.py_buffer(off.tobytes()),
        data=pa.py_buffer(data),
        null_bitmap=null_bitmap,
        null_count=null_count,
    )


def scan_paf_block(
    buf: np.ndarray, *, n_templates: int, chain_score: bool = False
) -> HitBlock:
    """Decode one newline-terminated PAF byte block.

    Reverse-strand hits are dropped here rather than downstream, and as a
    correctness requirement rather than a filter: PAF reports ``q_start`` on
    the forward query while ``cg:Z`` describes the alignment of the reverse
    complement, so a ``-`` hit paired with a forward ``member_seq`` votes the
    complement of what it supports into the PWM.

    Hits whose ``t_name`` is outside ``[0, n_templates)`` are dropped too —
    that is a template FASTA / TemplateSet mismatch, not data.

    ``chain_score=True`` also decodes ``s1:i`` (0 where absent). It is off by
    default so the single-pass path pays nothing for it.
    """
    nl = np.flatnonzero(buf == _NL)
    if nl.size == 0:
        raise ValueError("PAF block contains no complete record")
    row_start = np.empty(nl.size, dtype=np.int64)
    row_start[0] = 0
    row_start[1:] = nl[:-1] + 1
    row_end = nl.astype(np.int64)

    tab = np.flatnonzero(buf == _TAB).astype(np.int64)
    first_tab = np.searchsorted(tab, row_start, side="left")
    n_tabs = np.diff(np.append(first_tab, tab.size))
    if np.any(n_tabs < _N_FIXED - 1):
        bad = int(np.flatnonzero(n_tabs < _N_FIXED - 1)[0])
        raise ValueError(
            f"malformed PAF record at row {bad}: expected at least "
            f"{_N_FIXED} fields, found {int(n_tabs[bad]) + 1}"
        )

    n_rows = int(nl.size)
    q = _parse_int_field(buf, row_start, tab[first_tab])
    t = _parse_int_field(buf, tab[first_tab + 4] + 1, tab[first_tab + 5])
    plus = buf[tab[first_tab + 3] + 1] == ord("+")
    # n_match / aln_len are eager because the ADMISSION GATE needs them on
    # every hit, not just winners: a candidate is admitted on
    # `n_match / aln_len >= p_floor`. They are cols 9 and 10, so each ends at
    # its own following tab.
    n_match = _parse_int_field(buf, tab[first_tab + 8] + 1, tab[first_tab + 9])
    aln_len = _parse_int_field(buf, tab[first_tab + 9] + 1, tab[first_tab + 10])
    as_score = _scan_as_tag(buf, nl, tab, n_rows)
    s1 = _scan_int_tag(buf, nl, tab, n_rows, b"s1:i:") if chain_score else None

    keep_strand = plus
    n_dropped_strand = int((~keep_strand).sum())
    in_range = (t >= 0) & (t < n_templates)
    n_dropped_template = int((keep_strand & ~in_range).sum())
    keep = keep_strand & in_range
    src = np.flatnonzero(keep)

    return HitBlock(
        read_row=q[src],
        template_row=t[src],
        as_score=as_score[src],
        n_match=n_match[src],
        aln_len=aln_len[src],
        src_row=src.astype(np.int64),
        n_rows=n_rows,
        n_dropped_strand=n_dropped_strand,
        n_dropped_template=n_dropped_template,
        buf=buf,
        row_start=row_start,
        row_end=row_end,
        tab=tab,
        first_tab=first_tab.astype(np.int64),
        chain_score=None if s1 is None else s1[src],
    )


def _scan_int_tag(
    buf: np.ndarray, nl: np.ndarray, tab: np.ndarray, n_rows: int, tag: bytes
) -> np.ndarray:
    """An integer tag (``tag`` including its ``:i:``) per row, 0 where absent.

    Seeded on the tabs — every tag starts right after one — which is general
    where :func:`_scan_as_tag`'s rare-byte seed is specific to ``AS``.
    """
    out = np.zeros(n_rows, dtype=np.int64)
    width = len(tag)
    pos = tab + 1
    pos = pos[pos + width < buf.size]
    if pos.size == 0:
        return out
    ok = np.ones(pos.size, dtype=bool)
    for i, b in enumerate(tag):
        ok &= buf[pos + i] == b
    starts = pos[ok] + width
    if starts.size == 0:
        return out
    nxt = np.searchsorted(tab, starts, side="left")
    tab_end = np.where(
        nxt < tab.size, tab[np.clip(nxt, 0, max(tab.size - 1, 0))], buf.size
    )
    rows = np.searchsorted(nl, starts, side="left")
    ends = np.minimum(tab_end, nl[np.clip(rows, 0, nl.size - 1)])
    out[rows] = _parse_int_field(buf, starts, ends)
    return out


def _scan_as_tag(
    buf: np.ndarray, nl: np.ndarray, tab: np.ndarray, n_rows: int
) -> np.ndarray:
    """``AS:i:`` per row, 0 where absent.

    Seeded on the byte ``S``, which is rare in this stream: names are
    integers, an ``--eqx`` CIGAR is ``=XID``, and the only other uppercase
    ``S`` minimap2 emits is ``tp:A:S``, whose next byte is a tab.
    """
    out = np.zeros(n_rows, dtype=np.int64)
    s = np.flatnonzero(buf == ord("S"))
    s = s[(s >= 2) & (s + 4 < buf.size)]
    if s.size == 0:
        return out
    ok = (
        (buf[s - 1] == ord("A"))
        & (buf[s - 2] == _TAB)
        & (buf[s + 1] == ord(":"))
        & (buf[s + 2] == ord("i"))
        & (buf[s + 3] == ord(":"))
    )
    s = s[ok]
    if s.size == 0:
        return out
    starts = s + 4
    # `tab` is passed in rather than recomputed: rescanning the block for tabs
    # here cost 0.169 s of this function's 0.460 s.
    nxt = np.searchsorted(tab, starts, side="left")
    tab_end = np.where(
        nxt < tab.size, tab[np.clip(nxt, 0, max(tab.size - 1, 0))], buf.size
    )
    rows = np.searchsorted(nl, starts, side="left")
    ends = np.minimum(tab_end, nl[np.clip(rows, 0, nl.size - 1)])
    out[rows] = _parse_int_field(buf, starts, ends)
    return out


def iter_hit_blocks(
    chunks: Iterable[bytes],
    *,
    n_templates: int,
    block_bytes: int = 128 << 20,
) -> Iterator[HitBlock]:
    """Assemble minimap2's stdout into **group-aligned** :class:`HitBlock`s.

    Every emitted block contains only complete read groups. The mechanism is
    the point: rather than carrying a decoded accumulator across the boundary,
    a block is cut back to its last read-group start and the remaining **raw
    bytes** are prepended to the next block. Nothing decoded crosses, there is
    no state machine, and the output is independent of how the pipe happened
    to chunk (the same property ``iter_paf_batches`` gets one level down).

    Re-decoding cost is at most one read's hits per block boundary.
    """
    tail = b""
    for raw in chunks:
        tail = tail + raw if tail else raw
        if len(tail) < block_bytes:
            continue
        cut = tail.rfind(b"\n") + 1
        if cut == 0:
            continue
        block, rest = tail[:cut], tail[cut:]
        hb = scan_paf_block(
            np.frombuffer(block, dtype=np.uint8), n_templates=n_templates
        )
        keep, carry_from = _split_trailing_group(hb)
        if keep is not None:
            yield keep
        tail = block[carry_from:] + rest
    if tail:
        cut = tail.rfind(b"\n") + 1
        if cut:
            yield scan_paf_block(
                np.frombuffer(tail[:cut], dtype=np.uint8), n_templates=n_templates
            )


def iter_group_blocks(
    chunks: Iterable[bytes], *, block_bytes: int = 128 << 20
) -> Iterator[bytes]:
    """Group-aligned **raw** PAF blocks, for decoding somewhere else.

    The same guarantee as :func:`iter_hit_blocks` — every block holds only
    complete read groups — without decoding anything, so a pool worker can
    receive plain ``bytes`` (cheap to pickle, no parent buffer attached) and
    scan them itself. The cut is found on raw ``q_name`` bytes of the last
    few lines, so it is independent of which hits a scan would drop.
    """
    tail = b""
    for raw in chunks:
        tail = tail + raw if tail else raw
        if len(tail) < block_bytes:
            continue
        cut = tail.rfind(b"\n") + 1
        if cut == 0:
            continue
        group_start = _last_group_start(tail, cut)
        if group_start == 0:
            continue  # one read's hits fill the block; wait for more bytes
        yield tail[:group_start]
        tail = tail[group_start:]
    if tail:
        cut = tail.rfind(b"\n") + 1
        if cut:
            yield tail[:cut]


def _last_group_start(buf: bytes, end: int) -> int:
    """Byte offset of the first line of the read group ending at ``end``."""
    line_start = buf.rfind(b"\n", 0, end - 1) + 1
    name = buf[line_start : buf.index(b"\t", line_start)]
    while line_start > 0:
        prev = buf.rfind(b"\n", 0, line_start - 1) + 1
        if buf[prev : buf.index(b"\t", prev)] != name:
            break
        line_start = prev
    return line_start


def _split_trailing_group(hb: HitBlock) -> tuple[HitBlock | None, int]:
    """Cut ``hb`` before its last read group. Returns ``(block, carry_offset)``.

    The carry offset is a byte position in the block, so the remainder is
    re-scanned from raw bytes with the next chunk appended.
    """
    if len(hb) == 0:
        # Nothing survived (all antisense, say) — carry nothing, the rows are
        # decided. Re-scanning them would re-drop them.
        return None, hb.buf.size
    # Find the last group's start by contiguity, NOT by sortedness: minimap2
    # emits a query's hits together, which is all this may assume. Input order
    # happens to make read_row ascending today, and depending on that would be
    # a silent trap the first time it is not.
    changes = np.flatnonzero(hb.read_row[1:] != hb.read_row[:-1]) + 1
    first_of_last = int(changes[-1]) if changes.size else 0
    if first_of_last == 0:
        # One group fills the whole block: emitting it would be wrong (more
        # hits may follow), so carry everything and wait for more bytes.
        return None, 0
    carry_row = int(hb.src_row[first_of_last])
    return hb.take(np.arange(first_of_last)), int(hb.row_start[carry_row])


__all__ = [
    "HitBlock",
    "iter_group_blocks",
    "iter_hit_blocks",
    "scan_paf_block",
]
