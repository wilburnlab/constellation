"""CIGAR + cs:long parsers — alignment-block extraction.

Per-alignment derivers that lift the data already encoded in the
shipped ``ALIGNMENT_TABLE.cigar_string`` + ``cs`` tag stream into
per-block rows for ``ALIGNMENT_BLOCK_TABLE``. Operate on one alignment
record at a time (per-record cost is small; vectorisation buys
nothing here). Output is :class:`Block` records that the worker layer
projects into Arrow.

Two parsers, one block model:

    parse_cigar_blocks(cigar, ref_start, query_start, *, intron_min_bp)
        Block boundaries from CIGAR alone. ``n_match`` / ``n_mismatch``
        come out None — CIGAR ``M`` doesn't distinguish match from
        mismatch (only ``=`` / ``X`` do, and minimap2 emits ``M``).

    parse_cs_long_blocks(cs, ref_start, query_start)
        Block boundaries + per-block match / mismatch counts from
        minimap2's ``cs:Z`` tag (long form ``=NNNN`` for matches,
        ``*ab`` for substitutions, ``~aa<N>yy`` for splices).

Higher-level orchestration (use cs:long when available, fall back to
CIGAR) lives at the worker boundary in
:mod:`constellation.sequencing.quant.genome_count`, not here — so
this module stays a pure parser surface.

Block-break rule:

    A "block" is a contiguous M/=/X run bounded by N (intron) or any
    deletion ≥ ``intron_min_bp``. Long I (insertion) does NOT break
    a block — insertions add bases to the query within the same
    aligned block. Soft/hard clips at the alignment ends are excluded
    from blocks. Small I/D within a block contribute to
    ``n_insert`` / ``n_delete``.

Coordinates: 0-based half-open throughout, matching
``ALIGNMENT_TABLE.ref_start`` / ``ref_end``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────
# Block record
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Block:
    """One M/=/X-contiguous alignment block (≈ one exon for spliced reads).

    ``n_match`` / ``n_mismatch`` are ``None`` when the producer is
    CIGAR-only (CIGAR ``M`` is undifferentiated); both are populated
    by the cs:long parser. ``n_insert`` / ``n_delete`` count *small*
    I/D operations within the block (i.e. those below the
    ``intron_min_bp`` threshold for the CIGAR parser, or any
    sub-splice deletion for the cs parser).
    """

    block_index: int
    ref_start: int
    ref_end: int
    query_start: int
    query_end: int
    n_match: int | None
    n_mismatch: int | None
    n_insert: int
    n_delete: int


# ──────────────────────────────────────────────────────────────────────
# CIGAR
# ──────────────────────────────────────────────────────────────────────


# CIGAR ops keyed by which axes they consume. SAM spec:
#   M I S = X  consume query
#   M D N = X  consume reference
#   H P        consume neither
_CIGAR_TOKEN_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def parse_cigar_blocks(
    cigar_string: str,
    ref_start: int,
    *,
    query_start: int = 0,
    intron_min_bp: int = 25,
) -> list[Block]:
    """Split a CIGAR string into alignment blocks.

    Block boundaries: any ``N`` operation, or any ``D`` of length
    ``>= intron_min_bp``. Soft/hard clips at the ends are skipped;
    soft clips at the start advance ``query_start``. Small I/D within
    a block contribute to ``n_insert`` / ``n_delete``.

    Returns ``[]`` for an unmapped read (empty / ``"*"`` CIGAR) or for
    a CIGAR consisting entirely of clip operations.

    Raises ``ValueError`` on malformed CIGAR.
    """
    if not cigar_string or cigar_string == "*":
        return []

    tokens: list[tuple[int, str]] = []
    pos = 0
    for m in _CIGAR_TOKEN_RE.finditer(cigar_string):
        if m.start() != pos:
            raise ValueError(f"malformed CIGAR at offset {pos}: {cigar_string!r}")
        tokens.append((int(m.group(1)), m.group(2)))
        pos = m.end()
    if pos != len(cigar_string):
        raise ValueError(f"trailing bytes in CIGAR: {cigar_string!r}")

    blocks: list[Block] = []
    block_index = 0
    ref_pos = ref_start
    query_pos = query_start

    # Per-block accumulators — None means "no block open"
    in_block = False
    block_ref_start = 0
    block_query_start = 0
    n_insert = 0
    n_delete = 0

    def open_block() -> None:
        nonlocal in_block, block_ref_start, block_query_start, n_insert, n_delete
        if in_block:
            return
        in_block = True
        block_ref_start = ref_pos
        block_query_start = query_pos
        n_insert = 0
        n_delete = 0

    def close_block() -> None:
        nonlocal in_block, block_index
        if not in_block:
            return
        if ref_pos > block_ref_start:
            blocks.append(
                Block(
                    block_index=block_index,
                    ref_start=block_ref_start,
                    ref_end=ref_pos,
                    query_start=block_query_start,
                    query_end=query_pos,
                    n_match=None,
                    n_mismatch=None,
                    n_insert=n_insert,
                    n_delete=n_delete,
                )
            )
            block_index += 1
        in_block = False

    for length, op in tokens:
        if op in ("M", "=", "X"):
            open_block()
            ref_pos += length
            query_pos += length
        elif op == "I":
            # Long I never breaks a block (per the rule above) — only
            # shifts query.
            if in_block:
                n_insert += length
            query_pos += length
        elif op == "D":
            if length >= intron_min_bp:
                close_block()
                ref_pos += length
            else:
                if in_block:
                    n_delete += length
                ref_pos += length
        elif op == "N":
            close_block()
            ref_pos += length
        elif op == "S":
            # Soft clip: advances query but is not within an aligned
            # block. Should appear only at the alignment ends (SAM
            # spec); we don't enforce that here.
            close_block()
            query_pos += length
        elif op in ("H", "P"):
            # Hard clip + padding: consume neither axis. H bases are
            # not in the SEQ field, so query_pos is unaffected.
            pass
        else:  # pragma: no cover — covered by the regex character class
            raise ValueError(f"unsupported CIGAR op {op!r}")

    close_block()
    return blocks


# ──────────────────────────────────────────────────────────────────────
# cs:long
# ──────────────────────────────────────────────────────────────────────


# cs grammar (minimap2):
#   :N           short-form match of length N
#   =NNNN        long-form match: literal ref bases (lowercase or upper)
#   *ab          substitution: ref base a, query base b (lowercase)
#   +xxx         insertion in query: lowercase query bases
#   -xxx         deletion from ref: lowercase ref bases
#   ~aa<N>bb     splice/intron of length N with donor aa + acceptor bb
#
# All literals and donor/acceptor dinucleotides may be upper or lower
# case depending on the producer; the regex is case-insensitive.
_CS_TOKEN_RE = re.compile(
    r"""
    (?:
        :(?P<match_short>\d+)
      | =(?P<match_long>[A-Za-z]+)
      | \*(?P<sub_ref>[A-Za-z])(?P<sub_query>[A-Za-z])
      | \+(?P<insert>[A-Za-z]+)
      | -(?P<delete>[A-Za-z]+)
      | ~(?P<splice_donor>[A-Za-z]{2})(?P<splice_len>\d+)(?P<splice_acceptor>[A-Za-z]{2})
    )
    """,
    re.VERBOSE,
)


def parse_cs_long_blocks(
    cs_string: str,
    ref_start: int,
    *,
    query_start: int = 0,
) -> list[Block]:
    """Split minimap2's cs tag into alignment blocks with match/mismatch
    counts.

    Block boundaries come from the cs grammar's ``~`` (splice)
    operator. Within a block: ``=NNNN`` and ``:N`` add to ``n_match``,
    ``*ab`` adds to ``n_mismatch``, ``+xxx`` adds to ``n_insert``,
    ``-xxx`` adds to ``n_delete``. Unlike CIGAR, cs:long never produces
    a "long deletion" intron — the ``~`` operator is the only intron
    indicator — so there's no ``intron_min_bp`` knob here.

    Returns ``[]`` for an empty cs string.

    Raises ``ValueError`` on malformed cs.
    """
    if not cs_string:
        return []

    blocks: list[Block] = []
    block_index = 0
    ref_pos = ref_start
    query_pos = query_start

    in_block = False
    block_ref_start = 0
    block_query_start = 0
    n_match = 0
    n_mismatch = 0
    n_insert = 0
    n_delete = 0

    def open_block() -> None:
        nonlocal in_block, block_ref_start, block_query_start
        nonlocal n_match, n_mismatch, n_insert, n_delete
        if in_block:
            return
        in_block = True
        block_ref_start = ref_pos
        block_query_start = query_pos
        n_match = 0
        n_mismatch = 0
        n_insert = 0
        n_delete = 0

    def close_block() -> None:
        nonlocal in_block, block_index
        if not in_block:
            return
        if ref_pos > block_ref_start:
            blocks.append(
                Block(
                    block_index=block_index,
                    ref_start=block_ref_start,
                    ref_end=ref_pos,
                    query_start=block_query_start,
                    query_end=query_pos,
                    n_match=n_match,
                    n_mismatch=n_mismatch,
                    n_insert=n_insert,
                    n_delete=n_delete,
                )
            )
            block_index += 1
        in_block = False

    pos = 0
    for m in _CS_TOKEN_RE.finditer(cs_string):
        if m.start() != pos:
            raise ValueError(f"malformed cs string at offset {pos}: {cs_string!r}")
        pos = m.end()
        if (g := m.group("match_short")) is not None:
            length = int(g)
            open_block()
            n_match += length
            ref_pos += length
            query_pos += length
        elif (g := m.group("match_long")) is not None:
            length = len(g)
            open_block()
            n_match += length
            ref_pos += length
            query_pos += length
        elif m.group("sub_ref") is not None:
            open_block()
            n_mismatch += 1
            ref_pos += 1
            query_pos += 1
        elif (g := m.group("insert")) is not None:
            length = len(g)
            if in_block:
                n_insert += length
            query_pos += length
        elif (g := m.group("delete")) is not None:
            length = len(g)
            if in_block:
                n_delete += length
            ref_pos += length
        elif (g := m.group("splice_len")) is not None:
            length = int(g)
            close_block()
            ref_pos += length
        else:  # pragma: no cover — covered by the regex alternation
            raise ValueError(f"unrecognised cs token at offset {m.start()}")

    if pos != len(cs_string):
        raise ValueError(f"trailing bytes in cs string: {cs_string!r}")

    close_block()
    return blocks


# ──────────────────────────────────────────────────────────────────────
# Junction extraction
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Junction:
    """An intron implied by adjacent alignment blocks.

    ``donor_pos`` is the previous block's ``ref_end`` (intron 5');
    ``acceptor_pos`` is the next block's ``ref_start`` (intron 3').
    Strand is alignment-strand inherited; canonical-motif resolution
    happens later via the genome lookup in
    :mod:`constellation.sequencing.quant.junctions`.
    """

    donor_pos: int
    acceptor_pos: int


def blocks_to_junctions(blocks: list[Block]) -> list[Junction]:
    """Adjacent-block reduction. Returns one junction per consecutive
    block pair (i.e. ``len(blocks) - 1`` junctions, or zero if the read
    has fewer than two blocks).
    """
    if len(blocks) < 2:
        return []
    return [
        Junction(donor_pos=a.ref_end, acceptor_pos=b.ref_start)
        for a, b in zip(blocks[:-1], blocks[1:])
    ]


# ──────────────────────────────────────────────────────────────────────
# Soft-clip-aware query origin
# ──────────────────────────────────────────────────────────────────────


def query_start_from_cigar(cigar_string: str) -> int:
    """Number of leading soft-clipped bases in the CIGAR.

    Use as the ``query_start`` argument to :func:`parse_cigar_blocks`
    or :func:`parse_cs_long_blocks` so per-block ``query_start`` /
    ``query_end`` index against the SEQ field directly. Hard clips are
    NOT counted (those bases are absent from SEQ).
    """
    if not cigar_string or cigar_string == "*":
        return 0
    m = _CIGAR_TOKEN_RE.match(cigar_string)
    if m is None:
        return 0
    # Skip a leading H if present
    if m.group(2) == "H":
        m = _CIGAR_TOKEN_RE.match(cigar_string, m.end())
        if m is None:
            return 0
    if m.group(2) == "S":
        return int(m.group(1))
    return 0


__all__ = [
    "Block",
    "Junction",
    "parse_cigar_blocks",
    "parse_cs_long_blocks",
    "blocks_to_junctions",
    "query_start_from_cigar",
]
