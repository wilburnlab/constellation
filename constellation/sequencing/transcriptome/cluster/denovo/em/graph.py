"""How two templates relate: a pair kernel, a classifier and the edge tables.

**What this replaces, and why.** The redundancy detector in ``refine.py`` ran
minimap2 ``-c`` all-vs-all and called a pair redundant on two numbers, and
both measured the wrong thing. *Identity* was ``n_match / aln_len`` on a
**local** hit, so whatever the aligner declined to align was free: unaligned
ends cost nothing, asm5 clips a terminal difference instead of charging it,
and a template under ~200 nt never produced a hit at all. *End tolerance* was
``min_mutual_coverage = 0.95``, a proportion standing in for a length — about
100 nt of free end at 2 kb and 25 nt at 500 nt — which passed staggered pairs
and passed two Akap4 proteoforms that differ by an unrelated first exon.
Bench, 9.4M reads, 6 rounds, merge on vs off: clusters carrying an ORF 41.3%
vs 54.4%, and 51.6% of merges re-joined two children the M-step had split from
one parent in the same round.

Here a pair is *measured* once, in nucleotides, and the measurement is kept.
What may merge is a predicate over the stored columns, decided elsewhere; this
module only says what a pair is.

**The kernel** (:func:`relate_pair`). ``src`` is the shorter sequence, ``dst``
the longer::

    tol    = min(tolerance, len(src) // 10)
    k1     = min(budget(len src) + tol5 + tol3, len(src) - 1)
    pass 1   HW(src -> dst), locations, k = k1      -> dst overhangs
    pass 2   HW(dst[start:end+1] -> src), k = d1    -> src overhangs
    fold     min(src overhang, dst overhang) at each end returns to both cores
    trace    NW(src core, dst core), path, only when 0 < edits <= budget

*Why two passes.* An infix alignment frees the ends of its **target** only.
Pass 1 therefore finds how far ``dst`` reaches beyond ``src`` and charges every
base by which ``src`` reaches beyond ``dst`` as an edit — a 25-nt 5' extension
on the shorter template is 25 insertions. Pass 2 turns the pair round so those
bases become a free end too. It is run whenever pass 1 is inexact; an earlier
design skipped it when the target "was not exhausted", and its review found a
5' extension charged as edits in 86 of 3,000 trials.

*Why pass 2 aligns the placed window and not the whole of ``dst``.* The whole
of ``dst`` cannot be placed inside ``src`` when ``src`` is nested in it: every
base of ``dst`` outside the shared span is an edit there, so ``dst -> src``
returns -1 for every pair nested by more than the bound, and a contained
template's own overhang could not be separated from its internal edits. The
window is the part of ``dst`` pass 1 says ``src`` occupies, so ``d2 <= d1``
always (the pass-1 alignment restricted to the window is a valid pass-2
alignment) and the answer is the same as the whole-sequence one whenever the
extents match.

*Why overhangs come from locations and never from a CIGAR.* Unit-cost edit
distance has no preference between one clean gap and the same bases scattered
among chance matches: a 20-nt query overhang came back as
``6I1=2I1=9I1=3I497=``. The start and end of a placement are well defined
where the run structure of its path is not.

*The fold.* After both passes the two sequences can each report an overhang at
the **same** end — ``dst`` starts 2 nt before the placement and ``src`` starts
3 nt before the window — and read literally that makes two differing ends
free. It is not rare: 183 of 498 inexact pairs whose ends carry 0-20 nt of
unrelated sequence. What both overhang is not an overhang, it is sequence they
hold opposite each other. So ``min(src overhang, dst overhang)`` at each end
goes back into both cores and is charged by the global alignment. Afterwards
at most one of the two overhangs at an end is non-zero.

*The tie rule* (:func:`_choose_location`). edlib reports every end position
that reaches the optimum and they are not interchangeable. It returns the
**shortest** of tied ends: a last-base mismatch on a 2 kb pair comes back as
``[(0, 1998), (0, 1999)]`` and the first of those reads the mismatch as a 1-nt
overhang, uncharged. So the longest span wins. But a tandem repeat returns one
placement per copy, and longest-first alone takes the wrong copy — so the
candidates are first restricted to those starting within ``hint_band`` of
where the shared minimizers say ``src`` starts, and only if none is near does
the longest overall win.

*edlib is called on* ``bytes``. A numpy ``uint8`` slice is accepted and
converted element by element, a fixed cost per call that dwarfs the alignment:
at 1.4 kb an infix pass is 438 us against 90 us (4.9x) and a bounded global
distance 354 us against 27 us (13x). A ``memoryview`` is 2-5x slower.

**Classification** runs in a fixed order — placement, terminal divergence,
stagger, identity floor, internal length change, then extent. The terminal and
stagger rules come before the identity floor so that what a pair is counted as
depends as little as possible on how long its template is. Terminal divergence
(``div_5p`` / ``div_3p``) is the number of alignment columns outside the first
/ last exact run of ``anchor_len`` (or of the tolerance, where that is the
shorter), judged against the same nt tolerance as an overhang: sequence both
hold at an end and do not share is as much a difference of extent as sequence
only one holds.

The extent rule is symmetric. ``src`` is the shorter sequence because an
infix alignment needs one, not because the relation does: which of the two is
``contained`` is decided by which one reaches past the other, and the edge is
stored contained member first. Inside the tolerance that member can be the
longer of the two by a few nt — it is cut short by more than the tolerance at
one end and reaches past, by less, at the other.

**What the dropped-by-reason counts are not.** They are the first rule a pair
failed, on what was measured of it, and that depends on how long the template
is:

- A pair over the edit budget is never traced, so it has no ``div`` and its
  cores are not an alignment. Neither the divergence rule nor the length rule
  can fire for it: an alternative end or a 27-nt skipped segment on a
  **short** template is counted ``below_floor``, the same on a long one
  ``divergent`` or ``internal_variant``. The trace is the dearest call in the
  kernel (a bounded global path is ~400 us at 1.4 kb against ~90 us for an
  infix pass) and is spent only on pairs that can still become an edge.
- ``no_placement`` is "pass 1 found nothing within ``k1``", and ``k1`` is the
  budget plus both tolerances. A stagger longer than that is never placed, so
  it is never seen to be a stagger: 80 nt apart is ``no_placement`` at 1.4 kb
  and ``staggered`` at 3 kb. So is an internal skip longer than ``k1`` — 90 nt
  below about 3 kb.

None of these is an edge under any reading; only the label moves.

**Measured.** All four overhangs exact in 4,000 of 4,000 randomized staggers
whose edits sit at least 12 nt from the junctions, and in 3,729 of 4,000 with
half the edits within 4 nt of one, worst error 2 nt.

**Known limits.**

- A junction is ambiguous to about 2 nt when edits sit within 4 nt of it:
  the bases either side can be read as overhang or as alignment at equal cost.
- Two or more differences inside the last 10 nt can be read as a 1-nt overhang
  plus one fewer edit (1.8% of pairs at two differences, 19% at four). Neither
  this nor the junction ambiguity can produce a 0-edit pair, which is what the
  default merge predicate requires.
- In a tandem repeat whose unit is shorter than ``hint_band`` the neighbouring
  copies are inside the band too, and a longer placement there wins.
- An alternative end shorter than the tolerance is ``equivalent`` with its
  edits recorded.
- ``N`` is an ordinary character: an unresolved-N column costs one edit.

**The builder** (:func:`build_graph`) turns a round's nodes into an edge
table. What runs where is part of the design::

    parent      sequences -> one uint8 buffer + int64 offsets
    parent      byte-identical rows -> classes, one representative each
    parent      sketch the representatives            (torch)
    fork pool   probe join -> candidate pairs         (numpy)
    fork pool   relate_pair on every candidate        (edlib)
    parent      representatives -> rows, annotate, write, in task order

*torch runs in the parent and nowhere else.* By the time either pool forks
the sketch has started torch's thread pool, and a torch op in a forked child
deadlocks on OpenMP — a silent hang, not an exception. So the join and the
kernel are numpy and edlib only, and the sketch is imported inside the
builder rather than by this module.

*Sequences reach the workers as a* ``uint8`` *buffer, never as*
``list[str]``. A fork shares pages copy-on-write, and merely reading a
``str`` out of a list writes its reference count onto the page it lives on:
measured, a ``list[str]`` module global is privately copied ~100% by each
worker that touches it. A numpy buffer has no per-element object to count.
The buffer stays private to the builder — several chunks are combined here
and never handed back, because a 2.2 GB ``large_string`` cannot be cast back
to ``string``.

*Tasks are cut by cost, not by count.* A pair costs 0.09-0.57 ms at 1.4 kb
and 16-25 ms at 15 kb x 15 kb, so tasks of equal pair counts differ ~100x in
work and the pool idles behind whichever holds the long family. A task is a
run of candidates whose summed ``len(src) * len(dst)`` is roughly constant.
Results are consumed in submission order, which is what makes the output
independent of the worker count.

*Byte-identical templates are dereplicated before the join.* Exact
duplication is the commonest relation there is — the detector this replaces
found 26.9-29.4% of final templates removable, the median redundant pair
exactly identical, the largest component 331 templates — and the join is the
worst place to discover it: ``c`` identical templates sit together in every
bucket any of them touches, so each probe expands ``c`` rows ``c`` times
over, and copies alone can carry a family past ``bucket_cap``. Dereplicated,
a class is sketched once and measured once, through its lowest row; its own
members are joined by exact edges that need no alignment, and each edge of
the representative is written for every member.

*Cluster edges keep only direct measurements* (:func:`cluster_edges`). When
the final merge absorbs a node, the edges measured against it describe ITS
sequence — its overhangs, its edits. Re-pointed at the survivor they would
be numbers nobody measured, so they are dropped, and the survivor speaks
through the edges it was given itself.
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import re
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, suppress
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from constellation.core.io.schemas import register_schema


#: Part of the graph cache stamp. Bump when :func:`relate_pair` or the
#: classifier can return a different answer for the same pair and parameters.
GRAPH_KERNEL_VERSION = 1

#: Fields that change how the work is cut up and never what it produces, so
#: they stay out of the cache stamp.
_EXECUTION_ONLY = frozenset({"chunk_rows"})


def _is_count(value: object) -> bool:
    """A non-negative, finite ``int`` — ``inf`` and ``True`` are neither."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


#: The least a :class:`GraphParams` count may be, where that is not 1.
_LEAST = {
    "tol_5p": 0,
    "tol_3p": 0,
    "min_budget": 0,
    "hint_band": 0,
    "diag_band": 0,
    # A bucket of one holds no pair.
    "bucket_cap": 2,
}


@dataclass(frozen=True, slots=True)
class GraphParams:
    """Every knob of the template graph: kernel, classifier and candidates.

    The end tolerances are nucleotides, not a proportion, and must be finite:
    the pass-1 bound is built from them, and they are what ``equivalent``
    means.
    """

    identity_floor: float = 0.99
    tol_5p: int = 30
    tol_3p: int = 30
    min_budget: int = 3
    anchor_len: int = 15
    internal_indel_min: int = 9
    hint_band: int = 8
    #: tolerance = min(tol, len(src) // short_tol_div): a fixed 30 nt is a
    #: fifth of a 150-nt template.
    short_tol_div: int = 10
    kmer: int = 19
    window: int = 19
    probes_per_seq: int = 16
    bucket_cap: int = 20_480
    overflow_anchors: int = 32
    max_candidates: int = 20_480
    min_shared: int = 2
    diag_band: int = 64
    chunk_rows: int = 8_000_000  # execution only
    # A backstop, but not execution-only: when it binds, buckets are demoted
    # to the anchor fallback and the edges change.
    max_rows: int = 32_000_000_000

    def __post_init__(self) -> None:
        floor = self.identity_floor
        if (
            isinstance(floor, bool)
            or not isinstance(floor, (int, float))
            or not 0.0 < floor <= 1.0
        ):
            raise ValueError(f"identity_floor must be in (0, 1], got {floor!r}")
        # Every field but the floor is a count, and every one is checked: a
        # value that cannot mean anything does not fail, it answers. With
        # internal_indel_min = 0 every measured pair is an internal variant,
        # exact twins included, and the graph is written without them.
        for f in dataclasses.fields(self):
            if f.name == "identity_floor":
                continue
            least = _LEAST.get(f.name, 1)
            value = getattr(self, f.name)
            if not _is_count(value) or value < least:
                raise ValueError(
                    f"{f.name} must be an integer >= {least}, got {value!r}"
                )
        if self.kmer > 31:
            # A canonical k-mer is packed 2 bits per base into one int64.
            raise ValueError(f"kmer must be in [1, 31], got {self.kmer!r}")

    def semantic(self) -> dict:
        """The fields that decide the result — what a cache stamp records."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in _EXECUTION_ONLY
        }


class Relation(IntEnum):
    """What a measured pair is. Only the first two are ever written as edges;
    the rest are counted by reason and dropped."""

    EQUIVALENT = 0
    CONTAINED = 1
    NO_PLACEMENT = 2
    DIVERGENT_5P = 3
    DIVERGENT_3P = 4
    STAGGERED = 5
    INTERNAL_VARIANT = 6
    BELOW_FLOOR = 7


RELATION_NAMES: dict[int, str] = {int(r): r.name.lower() for r in Relation}

PRODUCED_RELATIONS = (Relation.EQUIVALENT, Relation.CONTAINED)

#: The ``truncation`` vocabulary, in dictionary order.
TRUNCATION_NAMES: tuple[str, ...] = ("5p", "3p", "both")


# ──────────────────────────────────────────────────────────────────────
# Edge tables
# ──────────────────────────────────────────────────────────────────────

_DICT = pa.dictionary(pa.int8(), pa.string())

#: The measurement of one pair, shared by both edge tables. Every column is
#: written from ``src``'s side. On an 'equivalent' edge ``src`` is the
#: shorter sequence (ties: the lower row). On a 'contained' edge it is the
#: CONTAINED one, which is the shorter except inside the tolerance: cut short
#: by more than the tolerance at one end and reaching past by less at the
#: other, it can be the longer by a few nt. Read ``src_len`` / ``dst_len``
#: rather than assume.
EDGE_FEATURE_FIELDS: list[pa.Field] = [
    # 'equivalent' or 'contained' — the names in RELATION_NAMES, of which
    # only PRODUCED_RELATIONS are written. 'contained' is directed: src lies
    # inside dst, to within the tolerance. 'equivalent' is stored ONCE and
    # must be read as symmetric; a consumer walking neighbours has to look
    # at both columns.
    pa.field("relation", _DICT, nullable=False),
    # 'contained' only: the ends of dst that src does not reach — '5p',
    # '3p' or 'both' — i.e. where dst overhang + div exceeds the tolerance.
    # Null for 'equivalent'.
    pa.field("truncation", _DICT, nullable=True),
    # Whole-sequence lengths, nt.
    pa.field("src_len", pa.int32(), nullable=False),
    pa.field("dst_len", pa.int32(), nullable=False),
    # Unshared nt beyond the OTHER sequence's end, after the fold. At each
    # end at most one of the src / dst pair is non-zero.
    pa.field("src_overhang_5p", pa.int32(), nullable=False),
    pa.field("src_overhang_3p", pa.int32(), nullable=False),
    pa.field("dst_overhang_5p", pa.int32(), nullable=False),
    pa.field("dst_overhang_3p", pa.int32(), nullable=False),
    # dst overhang - src overhang, signed: positive where dst is the longer
    # at that end. One number per end, which the invariant above makes
    # lossless.
    pa.field("delta_5p", pa.int32(), nullable=False),
    pa.field("delta_3p", pa.int32(), nullable=False),
    # The shared span (the two cores, overhangs removed). n_edits is their
    # global edit distance and core_len_delta = len(dst core) -
    # len(src core); both are properties of the pair. aligned_len is NOT
    # quite: it is the column count of the path that was traced,
    # len(src core) + n_delete, and two adjacent substitutions traced as an
    # insertion plus a deletion make the same span one column longer.
    pa.field("aligned_len", pa.int32(), nullable=False),
    pa.field("n_edits", pa.int32(), nullable=False),
    pa.field("core_len_delta", pa.int32(), nullable=False),
    # The split of n_edits, src as query: a mismatch column, a src base
    # absent from dst, a dst base absent from src. These come from ONE
    # co-optimal path; only their sum (n_edits) and insert - delete
    # (= -core_len_delta) are path-invariant. All three are 0 on a pair
    # over the edit budget, which is never traced. n_indel_runs is
    # deliberately absent: edlib's traceback does not consolidate gaps, and
    # one clean 90-nt deletion traced as 20 runs.
    pa.field("n_mismatch", pa.int32(), nullable=False),
    pa.field("n_insert", pa.int32(), nullable=False),
    pa.field("n_delete", pa.int32(), nullable=False),
    # (aligned_len - n_edits) / aligned_len, over the shared span only.
    pa.field("identity", pa.float64(), nullable=False),
    # Pass-1 distance: the WHOLE of the shorter sequence placed inside the
    # longer, its overhangs charged. The "terminal gaps count" identity is
    # 1 - this / min(src_len, dst_len). The one column that is not mirrored
    # when an edge is stored the other way round from how it was measured.
    pa.field("edit_distance_placed", pa.int32(), nullable=False),
    # Terminal divergence: alignment columns outside the first / last exact
    # run of min(anchor_len, tolerance). Sequence both hold at an end and do
    # not share, where an overhang is sequence only one holds.
    pa.field("div_5p", pa.int32(), nullable=False),
    pa.field("div_3p", pa.int32(), nullable=False),
    # Probes the pair shared inside one diagonal window — why it was
    # measured at all.
    pa.field("n_shared_probes", pa.int32(), nullable=False),
    # The pair came through a candidate cap (an overflowing bucket or a
    # truncated query), so either endpoint's edge list may be incomplete.
    pa.field("candidate_overflow", pa.bool_(), nullable=False),
    # The two were separated by the same M-step split, this round or an
    # earlier one: merging them undoes a decision the M-step made.
    pa.field("same_split_origin", pa.bool_(), nullable=False),
    # Reads each endpoint holds.
    pa.field("src_n_reads", pa.int64(), nullable=False),
    pa.field("dst_n_reads", pa.int64(), nullable=False),
    # True under THIS run's merge predicate, whether or not the run merged.
    pa.field("mergeable", pa.bool_(), nullable=False),
]


#: One round's edges, over that round's M-step nodes (the pre-merge set).
TEMPLATE_EDGE_TABLE: pa.Schema = pa.schema(
    [
        # The round whose nodes these are.
        pa.field("node_round", pa.int32(), nullable=False),
        # template_id_for(node_round + 1, node_row): the id each node carries
        # as a template of the next round, and the ONLY join key for merging.
        pa.field("src_template_id", pa.int64(), nullable=False),
        pa.field("dst_template_id", pa.int64(), nullable=False),
        # Rows of the node set. Not rows of the next round's templates: a
        # node with an empty consensus is dropped after ids are assigned.
        pa.field("src_row", pa.int64(), nullable=False),
        pa.field("dst_row", pa.int64(), nullable=False),
        *EDGE_FEATURE_FIELDS,
    ],
    metadata={b"schema_name": b"EmTemplateEdgeTable"},
)

register_schema("EmTemplateEdgeTable", TEMPLATE_EDGE_TABLE)


#: The final output's edges, keyed on ``clusters.parquet`` ids.
CLUSTER_EDGE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("src_cluster_id", pa.int64(), nullable=False),
        pa.field("dst_cluster_id", pa.int64(), nullable=False),
        *EDGE_FEATURE_FIELDS,
    ],
    metadata={b"schema_name": b"EmClusterEdgeTable"},
)

register_schema("EmClusterEdgeTable", CLUSTER_EDGE_TABLE)


# ──────────────────────────────────────────────────────────────────────
# Pair kernel
# ──────────────────────────────────────────────────────────────────────

_PPM = 100_000


def edit_budget(length: int, params: GraphParams) -> int:
    """Edits a sequence of ``length`` may carry and still clear the floor.

    Integer arithmetic: ``(1 - 0.93) * 100`` is 6.999… in floating point and
    truncates to 6, one edit short of what the floor says.
    """
    parts = round((1.0 - params.identity_floor) * _PPM)
    return max(params.min_budget, (int(length) * parts) // _PPM)


def effective_tolerance(src_len: int, params: GraphParams) -> tuple[int, int]:
    """``(5', 3')`` end tolerance in nt for a pair whose shorter member is
    ``src_len`` long — the configured one from 300 nt up, a tenth of the
    template below that."""
    cap = int(src_len) // params.short_tol_div
    return min(params.tol_5p, cap), min(params.tol_3p, cap)


@dataclass(frozen=True, slots=True)
class PairMeasurement:
    """One pair, measured. Every count is from ``src``'s side.

    When ``relation`` is ``NO_PLACEMENT`` nothing was measured and every
    numeric field is 0.
    """

    relation: Relation
    truncation: str | None  # '5p' | '3p' | 'both' for CONTAINED, else None
    src_overhang_5p: int
    src_overhang_3p: int
    dst_overhang_5p: int
    dst_overhang_3p: int
    aligned_len: int
    n_edits: int
    n_mismatch: int
    n_insert: int
    n_delete: int
    core_len_delta: int  # len(dst core) - len(src core) = n_delete - n_insert
    edit_distance_placed: int  # pass-1 distance: whole src placed in dst
    div_5p: int
    div_3p: int
    #: CONTAINED only: it is ``dst`` that lies inside ``src``. The counts
    #: stay as measured, from ``src``'s side; whoever stores the edge turns
    #: it round.
    src_is_container: bool = False

    @property
    def identity(self) -> float:
        if self.aligned_len == 0:
            return 0.0
        return (self.aligned_len - self.n_edits) / self.aligned_len


_NO_PLACEMENT = PairMeasurement(
    relation=Relation.NO_PLACEMENT,
    truncation=None,
    src_overhang_5p=0,
    src_overhang_3p=0,
    dst_overhang_5p=0,
    dst_overhang_3p=0,
    aligned_len=0,
    n_edits=0,
    n_mismatch=0,
    n_insert=0,
    n_delete=0,
    core_len_delta=0,
    edit_distance_placed=0,
    div_5p=0,
    div_3p=0,
)

_OP_RE = re.compile(rb"(\d+)([=XID])")


def _choose_location(
    locations: Sequence[tuple[int | None, int]], hint: int | None, band: int
) -> tuple[int, int] | None:
    """Pick one of edlib's tied placements (the module docstring says why).

    Among placements starting within ``band`` of ``hint`` the longest span
    wins; ties go to the one nearest the hint, then the smallest start. With
    none near, or no hint, the longest overall wins by the same tie-breaks.
    Degenerate placements (no start, or an end before it) are never chosen.
    """
    valid = [(s, e) for s, e in locations if s is not None and e >= s]
    if not valid:
        return None
    if hint is None:
        return min(valid, key=lambda loc: (loc[0] - loc[1], loc[0]))
    near = [loc for loc in valid if abs(loc[0] - hint) <= band]
    return min(
        near or valid,
        key=lambda loc: (loc[0] - loc[1], abs(loc[0] - hint), loc[0]),
    )


def _nearest_occurrence(src: bytes, dst: bytes, hint: int | None) -> int:
    """Start of the exact occurrence of ``src`` in ``dst`` nearest ``hint``
    (ties: the smaller start), or -1. Two searches, however many copies."""
    if hint is None:
        return dst.find(src)
    after = dst.find(src, hint)
    before = dst.rfind(src, 0, hint + len(src))  # starts at or before hint
    if before < 0 or after < 0:
        return max(before, after)
    return before if hint - before <= after - hint else after


def _infix(query: bytes, target: bytes, k: int) -> tuple[int, list]:
    """``(distance, locations)`` of ``query`` placed end to end in ``target``."""
    import edlib

    res = edlib.align(query, target, mode="HW", task="locations", k=k)
    return res["editDistance"], res["locations"] or []


def _global(query: bytes, target: bytes, k: int, task: str) -> tuple[int, bytes]:
    """``(distance, extended CIGAR)`` of the end-to-end alignment."""
    import edlib

    res = edlib.align(query, target, mode="NW", task=task, k=k)
    return res["editDistance"], (res.get("cigar") or "").encode("ascii")


def _substitution_ops(cs: bytes, cd: bytes, n_edits: int) -> list | None:
    """The all-diagonal path as ``(n, op)`` runs, when it is an optimal one.

    Two cores of equal length whose Hamming distance IS the known optimum
    differ by substitutions along a co-optimal path, and that path can be
    read off the mismatch mask without a traceback.
    """
    a = np.frombuffer(cs, dtype=np.uint8)
    b = np.frombuffer(cd, dtype=np.uint8)
    at = np.flatnonzero(a != b)
    if at.size != n_edits:
        return None
    ops: list[tuple[int, bytes]] = []
    prev = 0
    for pos in at.tolist():
        if pos > prev:
            ops.append((pos - prev, b"="))
        if ops and ops[-1][1] == b"X":
            ops[-1] = (ops[-1][0] + 1, b"X")
        else:
            ops.append((1, b"X"))
        prev = pos + 1
    if len(cs) > prev:
        ops.append((len(cs) - prev, b"="))
    return ops


def _anchor_runs(tol: tuple[int, int], params: GraphParams) -> tuple[int, int]:
    """The exact run that ends the divergence at each end, ``(5', 3')``.

    ``anchor_len``, but never longer than the tolerance ``div`` is judged
    against. Below 150 nt the tolerance is under 15 nt, and with a 15-nt
    anchor ONE substitution 6-14 nt from an end is a divergence of 7-15
    columns — past the tolerance, so the pair was dropped as an alternative
    end: 10% of the positions of a 100-nt template, 55% of a 40-nt one.
    """
    return (
        min(params.anchor_len, max(1, tol[0])),
        min(params.anchor_len, max(1, tol[1])),
    )


def _read_trace(
    ops: list, anchor: tuple[int, int]
) -> tuple[int, int, int, int, int, int] | None:
    """``(aligned_len, n_mismatch, n_insert, n_delete, div_5p, div_3p)``.

    ``div_5p`` is the number of columns before the first run of
    ``anchor[0]`` exact matches and ``div_3p`` the number after the last run
    of ``anchor[1]``. ``None`` when either run is missing: the pair shares
    nothing that end could be measured from.
    """
    total = n_x = n_i = n_d = 0
    first_anchor = last_anchor_end = -1
    for raw_n, op in ops:
        n = int(raw_n)
        if op == b"=":
            if first_anchor < 0 and n >= anchor[0]:
                first_anchor = total
            if n >= anchor[1]:
                last_anchor_end = total + n
        elif op == b"X":
            n_x += n
        elif op == b"I":
            n_i += n
        else:
            n_d += n
        total += n
    if first_anchor < 0 or last_anchor_end < 0:
        return None
    return total, n_x, n_i, n_d, first_anchor, total - last_anchor_end


def _classify(
    *,
    src_oh: tuple[int, int],
    dst_oh: tuple[int, int],
    div: tuple[int, int],
    core_len_delta: int,
    n_edits: int,
    budget: int,
    tol: tuple[int, int],
    internal_indel_min: int,
) -> tuple[Relation, str | None, bool]:
    """``(relation, truncation, src is the container)`` of a placed,
    anchored pair. Order is part of the rule.

    *The extent rule does not know which sequence is* ``src``. Each one
    either reaches past the other by more than the tolerance at an end, or
    it does not. Neither does: ``equivalent``. Both do, which after the fold
    is at opposite ends: ``staggered``. One does: the other is ``contained``
    in it, **whichever of the two it is**. Asking only whether ``src`` reaches
    past ``dst`` made the answer depend on the order of the pair: 28 nt apart
    at both ends with one substitution 5 nt into the shared span was
    ``contained`` one way round and ``staggered`` the other, and between two
    sequences of EQUAL length the order is the order of their rows.

    *The length rule is for traced pairs.* A pair over the budget has no
    trace, its cores are what two infix placements left, and their lengths
    differ wherever the two passes read an unrelated end unevenly: counted
    as internal variants, 23% of pairs with unrelated ends were named for a
    difference they do not have. So the budget is asked first, and what
    reaches the length rule was aligned end to end.
    """
    if div[0] > tol[0]:
        return Relation.DIVERGENT_5P, None, False
    if div[1] > tol[1]:
        return Relation.DIVERGENT_3P, None, False
    src_past = (src_oh[0] + div[0] > tol[0], src_oh[1] + div[1] > tol[1])
    dst_past = (dst_oh[0] + div[0] > tol[0], dst_oh[1] + div[1] > tol[1])
    if any(src_past) and any(dst_past):
        return Relation.STAGGERED, None, False
    if n_edits > budget:
        return Relation.BELOW_FLOOR, None, False
    if abs(core_len_delta) >= internal_indel_min:
        return Relation.INTERNAL_VARIANT, None, False
    past = src_past if any(src_past) else dst_past
    if not any(past):
        return Relation.EQUIVALENT, None, False
    truncation = "both" if all(past) else ("5p" if past[0] else "3p")
    return Relation.CONTAINED, truncation, any(src_past)


def _mirrored(m: PairMeasurement) -> PairMeasurement:
    """``m`` as it reads from the other sequence's side. ``div`` and the
    shared span belong to the pair; ``edit_distance_placed`` stays what was
    measured, since the other placement never was."""
    if m.relation is Relation.NO_PLACEMENT:
        return m
    return dataclasses.replace(
        m,
        src_overhang_5p=m.dst_overhang_5p,
        src_overhang_3p=m.dst_overhang_3p,
        dst_overhang_5p=m.src_overhang_5p,
        dst_overhang_3p=m.src_overhang_3p,
        n_insert=m.n_delete,
        n_delete=m.n_insert,
        core_len_delta=-m.core_len_delta,
        src_is_container=(m.relation is Relation.CONTAINED and not m.src_is_container),
    )


def relate_pair(
    src: bytes, dst: bytes, diag: int | None, params: GraphParams
) -> PairMeasurement:
    """Measure and classify one pair, from ``src``'s side.

    ``src`` must not be the longer of the two. ``diag`` is the expected
    start of ``src`` on ``dst`` — negative when ``src`` starts before ``dst``
    does — or ``None`` for no hint. Both sequences must be ``bytes``.

    Between two sequences of EQUAL length the answer is the same whichever
    is passed as ``src``, mirrored. The two passes are not symmetric — with
    edits inside the last few nt of a junction, placing ``a`` in ``b`` and
    ``b`` in ``a`` can differ by an edit — and the caller's tie-break is the
    order of two rows, so a tied pair is always measured in the order of
    the sequences themselves.
    """
    ls, ld = len(src), len(dst)
    if ls > ld:
        raise ValueError(f"src must be the shorter sequence ({ls} > {ld})")
    if ls == ld and src > dst:
        turned = None if diag is None else -int(diag)
        return _mirrored(_relate_ordered(dst, src, turned, params))
    return _relate_ordered(src, dst, diag, params)


def _relate_ordered(
    src: bytes, dst: bytes, diag: int | None, params: GraphParams
) -> PairMeasurement:
    ls, ld = len(src), len(dst)
    tol = effective_tolerance(ls, params)
    budget = edit_budget(ls, params)
    anchor = _anchor_runs(tol, params)
    # Where src is expected to start on dst, and the window on src.
    hint_dst = hint_src = None
    if diag is not None:
        diag = int(diag)  # a numpy scalar would ride into the measurement
        hint_dst, hint_src = max(diag, 0), max(-diag, 0)

    src_5p = src_3p = d1 = d2 = m5 = m3 = 0
    if diag is not None and diag >= 0 and dst[diag : diag + ls] == src:
        start = diag
    else:
        start = _nearest_occurrence(src, dst, hint_dst)
    if start >= 0:
        # Exact: src is a substring of dst and no alignment is needed.
        dst_5p, dst_3p = start, ld - start - ls
    else:
        # k >= len(query) lets edlib return an all-insertion placement.
        k1 = min(budget + tol[0] + tol[1], ls - 1)
        d1, locations = _infix(src, dst, k1)
        if d1 < 0:
            return _NO_PLACEMENT
        placed = _choose_location(locations, hint_dst, params.hint_band)
        if placed is None:
            return _NO_PLACEMENT
        start, end = placed
        dst_5p, dst_3p = start, ld - 1 - end

        d2, locations = _infix(dst[start : end + 1], src, d1)
        if d2 < 0:
            return _NO_PLACEMENT
        placed = _choose_location(locations, hint_src, params.hint_band)
        if placed is None:
            return _NO_PLACEMENT
        src_5p, src_3p = placed[0], ls - 1 - placed[1]

        m5 = min(src_5p, dst_5p)
        m3 = min(src_3p, dst_3p)
        src_5p, dst_5p = src_5p - m5, dst_5p - m5
        src_3p, dst_3p = src_3p - m3, dst_3p - m3

    cs = src[src_5p : ls - src_3p]
    cd = dst[dst_5p : ld - dst_3p]
    delta = len(cd) - len(cs)

    n_x = n_i = n_d = div_5p = div_3p = 0
    if cs == cd:
        n_edits, aligned = 0, len(cs)
        if aligned < max(anchor):
            return _NO_PLACEMENT
    else:
        ops = None
        if m5 or m3:
            # The fold lengthened both cores, so d2 is no longer their
            # distance; pairing the restored bases column for column bounds it.
            bound = d2 + m5 + m3
            if bound <= budget:
                n_edits, cigar = _global(cs, cd, bound, "path")
                ops = _OP_RE.findall(cigar)
            else:
                n_edits, _ = _global(cs, cd, bound, "distance")
        else:
            n_edits = d2
        if n_edits < 0:
            return _NO_PLACEMENT
        if n_edits > budget:
            aligned = max(len(cs), len(cd))
        else:
            if ops is None and delta == 0:
                ops = _substitution_ops(cs, cd, n_edits)
            if ops is None:
                _, cigar = _global(cs, cd, n_edits, "path")
                ops = _OP_RE.findall(cigar)
            trace = _read_trace(ops, anchor)
            if trace is None:
                return _NO_PLACEMENT
            aligned, n_x, n_i, n_d, div_5p, div_3p = trace

    relation, truncation, turned = _classify(
        src_oh=(src_5p, src_3p),
        dst_oh=(dst_5p, dst_3p),
        div=(div_5p, div_3p),
        core_len_delta=delta,
        n_edits=n_edits,
        budget=budget,
        tol=tol,
        internal_indel_min=params.internal_indel_min,
    )
    return PairMeasurement(
        relation=relation,
        truncation=truncation,
        src_overhang_5p=src_5p,
        src_overhang_3p=src_3p,
        dst_overhang_5p=dst_5p,
        dst_overhang_3p=dst_3p,
        aligned_len=aligned,
        n_edits=n_edits,
        n_mismatch=n_x,
        n_insert=n_i,
        n_delete=n_d,
        core_len_delta=delta,
        edit_distance_placed=d1,
        div_5p=div_5p,
        div_3p=div_3p,
        src_is_container=turned,
    )


# ──────────────────────────────────────────────────────────────────────
# Merge predicate
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MergePredicate:
    """Which ``equivalent`` edges may merge.

    Every threshold is absolute — edits and nucleotides — so none of them
    loosens with template length: one edit is 0.998 identity at 500 nt and
    0.9998 at 5 kb.
    """

    max_edits: int = 0
    tol_5p: int = 30
    tol_3p: int = 30
    min_reads: int = 0
    #: Merge two templates an M-step split apart. Off, because that is the
    #: split/merge cycle: 51.6% of the removed detector's merges were these.
    merge_siblings: bool = False

    def __post_init__(self) -> None:
        for name in ("max_edits", "tol_5p", "tol_3p", "min_reads"):
            if not _is_count(getattr(self, name)):
                raise ValueError(
                    f"{name} must be a non-negative integer, "
                    f"got {getattr(self, name)!r}"
                )
        if not isinstance(self.merge_siblings, bool):
            raise ValueError(
                f"merge_siblings must be a bool, got {self.merge_siblings!r}"
            )


_RELATION_CODE = {name: code for code, name in RELATION_NAMES.items()}
#: ``-1`` is "no truncation" wherever a truncation is held as an integer.
_TRUNCATION_CODE = {None: -1, **{name: i for i, name in enumerate(TRUNCATION_NAMES)}}


def _column(table: pa.Table | pa.RecordBatch, name: str) -> np.ndarray:
    """A numeric or boolean column as one numpy array."""
    return table.column(name).to_numpy(zero_copy_only=False)


def _codes(column: pa.Array | pa.ChunkedArray, code_of: dict) -> np.ndarray:
    """int8 codes of a dictionary (or plain string) column; ``-1`` for null.

    Read through each chunk's OWN dictionary. The tables written here put the
    enum value in the index, but a table that was filtered, concatenated or
    re-encoded on its way back is under no obligation to have kept it there.
    """
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    out = [np.empty(0, dtype=np.int8)]
    for chunk in chunks:
        if not pa.types.is_dictionary(chunk.type):
            chunk = chunk.dictionary_encode()
        values = chunk.dictionary.to_pylist()
        lut = np.array([code_of[value] for value in values] + [-1], dtype=np.int8)
        index = chunk.indices.to_numpy(zero_copy_only=False)
        if chunk.null_count:
            index = np.where(chunk.is_null().to_numpy(zero_copy_only=False), -1, index)
        out.append(lut[index.astype(np.int64)])
    return np.concatenate(out)


_PREDICATE_COLUMNS = (
    "src_len",
    "dst_len",
    "src_overhang_5p",
    "src_overhang_3p",
    "dst_overhang_5p",
    "dst_overhang_3p",
    "n_edits",
    "div_5p",
    "div_3p",
    "same_split_origin",
    "src_n_reads",
    "dst_n_reads",
)


def _predicate_columns(edges: pa.Table | pa.RecordBatch) -> dict[str, np.ndarray]:
    cols = {name: _column(edges, name) for name in _PREDICATE_COLUMNS}
    cols["relation"] = _codes(edges.column("relation"), _RELATION_CODE)
    return cols


def _mergeable_mask(
    c: dict[str, np.ndarray], predicate: MergePredicate, exact_only: bool
) -> tuple[np.ndarray, np.ndarray]:
    """``(mergeable, byte-identical)`` per edge. ``relation`` is its int8
    code here, and nothing but integer and boolean columns is read."""
    s5, s3 = c["src_overhang_5p"], c["src_overhang_3p"]
    d5, d3 = c["dst_overhang_5p"], c["dst_overhang_3p"]
    equivalent = c["relation"] == Relation.EQUIVALENT
    identical = (
        equivalent
        & (c["n_edits"] == 0)
        & (s5 == 0)
        & (s3 == 0)
        & (d5 == 0)
        & (d3 == 0)
        & (c["src_len"] == c["dst_len"])
    )
    ok = equivalent & (c["n_edits"] <= (0 if exact_only else predicate.max_edits))
    ok &= s5 + c["div_5p"] <= predicate.tol_5p
    ok &= d5 + c["div_5p"] <= predicate.tol_5p
    ok &= s3 + c["div_3p"] <= predicate.tol_3p
    ok &= d3 + c["div_3p"] <= predicate.tol_3p
    ok &= np.minimum(c["src_n_reads"], c["dst_n_reads"]) >= predicate.min_reads
    if not predicate.merge_siblings:
        ok &= ~c["same_split_origin"] | identical
    return ok, identical


def is_mergeable(
    edges: pa.Table, predicate: MergePredicate, *, exact_only: bool = False
) -> np.ndarray:
    """Whether each edge may merge under ``predicate``: one bool per edge.

    ::

        equivalent AND n_edits <= max_edits
        AND at each end   src overhang + div <= tol
                    and   dst overhang + div <= tol
        AND min(src_n_reads, dst_n_reads) >= min_reads
        AND (merge_siblings OR NOT same_split_origin OR byte-identical)

    ``exact_only`` holds ``max_edits`` at 0 whatever the predicate says: the
    final output's merge has no M-step after it to rebuild a consensus, so it
    may join only pairs whose shared span already agrees base for base.
    Byte-identical — no edits, no overhang, equal length — is exempt from the
    split-origin guard, because no split can have meant it.

    Decided on the integer columns and never on ``identity``, whose last
    digits depend on which co-optimal path was traced. The stored
    ``mergeable`` column is not read either: it records the predicate of the
    run that wrote it. Every row is judged alone, so a file may be passed
    through one record batch at a time. Either edge table is accepted.
    """
    return _mergeable_mask(_predicate_columns(edges), predicate, exact_only)[0]


def mergeable_pairs(
    edges: pa.Table, predicate: MergePredicate, *, exact_only: bool = False
) -> dict[str, np.ndarray]:
    """The mergeable edges of a ``TEMPLATE_EDGE_TABLE``, as arrays.

    ``src_template_id`` / ``dst_template_id`` / ``src_row`` / ``dst_row`` are
    int64. ``off_5p`` / ``off_3p`` (int64, signed) are how far ``dst`` reaches
    beyond ``src`` at that end — ``delta_5p`` / ``delta_3p``. ``identical`` is
    True for a byte-identical pair.

    Merge on the template ids. A row is a row of the node set the graph was
    built over, which stops being a row of the next round's templates as soon
    as one node has an empty consensus.
    """
    keep, identical = _mergeable_mask(_predicate_columns(edges), predicate, exact_only)
    out = {
        name: _column(edges, name).astype(np.int64)[keep]
        for name in ("src_template_id", "dst_template_id", "src_row", "dst_row")
    }
    out["off_5p"] = _column(edges, "delta_5p").astype(np.int64)[keep]
    out["off_3p"] = _column(edges, "delta_3p").astype(np.int64)[keep]
    out["identical"] = identical[keep]
    return out


# ──────────────────────────────────────────────────────────────────────
# Split origin
# ──────────────────────────────────────────────────────────────────────


def _int64_column(name: str, values: object, n: int | None) -> np.ndarray:
    """``values`` as a contiguous little-endian int64 ``(n,)`` array."""
    arr = np.asarray(values)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array, got dtype {arr.dtype}")
    if arr.ndim != 1 or (n is not None and arr.shape[0] != n):
        raise ValueError(
            f"{name} must be one value per row ({n} rows), got shape {arr.shape}"
        )
    return np.ascontiguousarray(arr, dtype="<i8")


class _LineageRule(dict):
    """The rows a line is walked through. Every other rule reads as -1: an
    ``unrecruited`` row has no child at all."""

    def __missing__(self, key: object) -> int:
        return -1


_LINEAGE_RULE = _LineageRule(carry=0, split=1, merge=2)


def _read_lineage(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(child id ascending, its parent id, it came of a split)`` over a
    round's node rows; ``None`` when the file does not open.

    A child id can stand on several rows: its own, and one ``merge`` row for
    every node it absorbed. A merge row keeps the absorbed node's PARENT and
    overwrites its rule, so whether that node came of a split is no longer
    written on the row — but it is still in the file: a parent that emitted
    several nodes has several rows, merged or not.
    """
    try:
        table = pq.read_table(
            path, columns=["child_template_id", "parent_template_id", "rule"]
        )
        rule = _codes(table.column("rule"), _LINEAGE_RULE)
        child = _column(table, "child_template_id").astype(np.int64)
        parent = _column(table, "parent_template_id").astype(np.int64)
    except (OSError, KeyError, ValueError, TypeError, pa.ArrowException):
        return None
    walked = np.flatnonzero(rule >= 0)
    child, parent, rule = child[walked], parent[walked], rule[walked]
    if walked.shape[0]:
        _, inverse, count = np.unique(parent, return_inverse=True, return_counts=True)
        several = count[inverse.reshape(-1)] > 1
    else:
        several = np.zeros(0, dtype=bool)
    from_split = (rule == _LINEAGE_RULE["split"]) | (
        (rule == _LINEAGE_RULE["merge"]) & several
    )
    order = np.argsort(child, kind="stable")
    return child[order], parent[order], from_split[order]


def _distinct_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct ``(a, b)`` pairs, sorted."""
    if a.shape[0] < 2:
        return a, b
    order = np.lexsort((b, a))
    a, b = a[order], b[order]
    fresh = np.ones(a.shape[0], dtype=bool)
    fresh[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    return a[fresh], b[fresh]


def _kin_classes(n: int, node: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """One id per node from its ``(node, origin)`` pairs; ``-1`` = none.

    A node with one origin keeps it. Origins that meet in one node are one
    class, named by its smallest member.
    """
    out = np.full(n, -1, dtype=np.int64)
    if node.shape[0] == 0:
        return out
    node, origin = _distinct_pairs(node, origin)
    again = node[1:] == node[:-1]
    if not bool(again.any()):
        out[node] = origin
        return out
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ids, code = np.unique(origin, return_inverse=True)
    code = code.reshape(-1)
    k = ids.shape[0]
    a, b = code[:-1][again], code[1:][again]
    links = coo_matrix((np.ones(a.shape[0], dtype=np.int8), (a, b)), shape=(k, k))
    n_classes, label = connected_components(links, directed=False)
    name = np.full(n_classes, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(name, label, ids)
    out[node] = name[label[code]]
    return out


def split_origins(
    rounds_dir: Path, node_round: int, parent_template_id: np.ndarray
) -> np.ndarray:
    """The M-step split each node descends from, as a kin class; ``-1`` = none.

    ``parent_template_id[i]`` is the round-``node_round`` template that node
    ``i`` was refined from. Nodes that share a parent were split from it this
    round, and their origin is that parent. Any other node walks back through
    ``rounds_dir/rNN/lineage.parquet``, newest first, until it meets a split
    — whose parent is the origin — or runs out of lineage.

    Two nodes are **kin** iff their values are equal and ``>= 0``.

    *Why the walk.* Guarding this round's siblings alone turns the period-1
    split/merge cycle into a period-2 one: a pair split in round ``r`` is
    carried into ``r + 1`` as two templates with two different parents, and
    merged there. A line that splits again takes the newer origin.

    *Why a survivor inherits.* A template that absorbed another stands for
    both, so it has a line through each: its own, and one through every
    ``merge`` row that names it. Walking its own line alone forgets what it
    absorbed, and that re-opens the case the guard exists for one round
    later. A hub from another parent is mergeable with both children of a
    split; it takes one and is refused the other; next round the survivor's
    own origin is the hub's, the refused child is no kin of it, and the
    split is undone through the hub in two rounds instead of one.

    *Why a class and not a set.* A survivor can so descend from several
    splits, and "kin" would be "the sets of origins intersect". One id per
    node is what the grouping takes, so origins that meet in one node are
    made ONE class, named by its smallest member. That is the transitive
    closure of the set rule and so it refuses more: with ``a``, ``b`` split
    from P, ``h``, ``g`` split from Q, and ``h`` having absorbed ``a``, the
    pair ``(b, g)`` is kin by the class and not by the sets. It errs toward
    the refusal, in a guard that can only refuse, and only in a run that
    merged — without merge rows every node has one line and the class is
    the origin itself.

    A lineage file that is missing or does not open ends the walk and leaves
    the lines still walking without an origin: a seeded round has none, and
    nor does a run directory older than the file. Note which way that errs —
    "no known kin" is the PERMISSIVE answer of a guard that only ever refuses
    a merge, so a lost lineage file lets through a merge it would have
    stopped.
    """
    parent = _int64_column("parent_template_id", parent_template_id, None)
    n = parent.shape[0]
    if n == 0:
        return np.full(0, -1, dtype=np.int64)
    _, inverse, count = np.unique(parent, return_inverse=True, return_counts=True)
    sibling = count[inverse.reshape(-1)] > 1
    met_node = [np.flatnonzero(sibling)]
    met_origin = [parent[sibling]]

    # Every line still walking, as (node, the template it has reached).
    node = np.flatnonzero(~sibling)
    cur = parent[node]
    for r in range(int(node_round) - 1, 0, -1):
        if node.shape[0] == 0:
            break
        lineage = _read_lineage(Path(rounds_dir) / f"r{r:02d}" / "lineage.parquet")
        if lineage is None or lineage[0].shape[0] == 0:
            break
        child, up, from_split = lineage
        lo = np.searchsorted(child, cur, side="left")
        rows = np.searchsorted(child, cur, side="right") - lo
        line = np.repeat(np.arange(cur.shape[0]), rows)
        first = np.cumsum(rows) - rows
        at = np.repeat(lo - first, rows) + np.arange(line.shape[0])
        split = from_split[at]
        met_node.append(node[line[split]])
        met_origin.append(up[at[split]])
        # Two lines of one node can reach the same template.
        node, cur = _distinct_pairs(node[line[~split]], up[at[~split]])
    return _kin_classes(n, np.concatenate(met_node), np.concatenate(met_origin))


# ──────────────────────────────────────────────────────────────────────
# Input: one buffer, one digest
# ──────────────────────────────────────────────────────────────────────

#: Longest sequence the graph takes: lengths, overhangs and the candidate
#: diagonal are int32 columns.
_MAX_SEQUENCE_LEN = (1 << 31) - 1


def _string_chunks(
    sequences: pa.Array | pa.ChunkedArray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Per non-empty chunk, ``(row lengths int64, the rows' bytes uint8)``.

    Both are of the chunk's OWN rows. ``Array.offset`` is not zero after a
    slice while the offsets buffer still begins at the parent's row 0, so
    reading that buffer from its start returns other rows' sequences — with
    nothing to show for it but a wrong answer.
    """
    chunks = sequences.chunks if isinstance(sequences, pa.ChunkedArray) else [sequences]
    for chunk in chunks:
        kind = chunk.type
        if pa.types.is_large_string(kind) or pa.types.is_large_binary(kind):
            width = np.int64
        elif pa.types.is_string(kind) or pa.types.is_binary(kind):
            width = np.int32
        else:
            raise TypeError(f"sequences must be string or large_string, got {kind}")
        if chunk.null_count:
            raise ValueError("sequences must not contain nulls")
        if len(chunk) == 0:
            continue
        _, offsets, data = chunk.buffers()
        raw = np.frombuffer(offsets, dtype=width, count=chunk.offset + len(chunk) + 1)
        raw = raw[chunk.offset :]
        lo, hi = int(raw[0]), int(raw[-1])
        if data is None or hi == lo:
            held = np.empty(0, dtype=np.uint8)
        else:
            held = np.frombuffer(data, dtype=np.uint8, count=hi)[lo:]
        yield np.diff(raw).astype(np.int64), held


def _sequence_buffer(
    sequences: pa.Array | pa.ChunkedArray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(buffer, offsets)``: row ``i`` is ``buffer[offsets[i]:offsets[i + 1]]``.

    One chunk is viewed where it lies. Several are copied into one buffer, by
    hand: ``combine_chunks`` on a ``string`` column past 2 GB overflows its
    int32 offsets, and casting the column to avoid that copies it twice.
    """
    parts = list(_string_chunks(sequences))
    n = sum(lengths.shape[0] for lengths, _ in parts)
    offsets = np.zeros(n + 1, dtype=np.int64)
    if parts:
        np.cumsum(np.concatenate([lengths for lengths, _ in parts]), out=offsets[1:])
    if len(parts) == 1:
        return parts[0][1], offsets
    buffer = np.empty(int(offsets[-1]), dtype=np.uint8)
    at = 0
    for _, held in parts:
        buffer[at : at + held.shape[0]] = held
        at += held.shape[0]
    return buffer, offsets


def input_digest(
    sequences: pa.Array | pa.ChunkedArray,
    ids: np.ndarray,
    n_reads: np.ndarray,
    split_origin: np.ndarray | None,
) -> str:
    """xxh3-128 of everything the edges are a function of besides the
    parameters: the sequences in row order, and each row's template id, read
    count and split origin.

    A digest of the ROWS, not of how they are held: re-chunking the column,
    slicing it out of a longer one, or passing ``string`` for
    ``large_string`` leaves it unchanged. No split origins and split origins
    that are all ``-1`` are the same input, and digest the same.
    """
    import xxhash

    parts = list(_string_chunks(sequences))
    n = sum(lengths.shape[0] for lengths, _ in parts)
    ids = _int64_column("ids", ids, n)
    n_reads = _int64_column("n_reads", n_reads, n)
    if split_origin is None:
        origin = np.full(n, -1, dtype="<i8")
    else:
        origin = _int64_column("split_origin", split_origin, n)

    state = xxhash.xxh3_128()
    state.update(b"constellation.em.graph.input/1")
    state.update(np.array([n], dtype="<i8"))
    # Lengths as well as bases: the same bases cut at other places are other
    # sequences.
    for lengths, _ in parts:
        state.update(np.ascontiguousarray(lengths, dtype="<i8"))
    for _, held in parts:
        state.update(np.ascontiguousarray(held))
    for column in (ids, n_reads, origin):
        state.update(column)
    return state.hexdigest()


def graph_stamp(params: GraphParams, digest: str) -> dict:
    """What a cached edge file is a function of: the kernel, the parameters
    that decide the result, and the input.

    The merge predicate is not in it. It decides one column, ``mergeable``,
    and a caller that reads that column back has to stamp the predicate too.
    """
    return {
        "kernel_version": GRAPH_KERNEL_VERSION,
        "graph": params.semantic(),
        "input": digest,
    }


# ──────────────────────────────────────────────────────────────────────
# Byte-identical classes
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Classes:
    """The pairable rows, grouped by byte-identical sequence.

    Classes are numbered by their lowest row — the representative — so a
    lower class is a lower representative, and "ties go to the lower row"
    means over classes what it means over rows.
    """

    members: np.ndarray  # int64 (P,) node rows: class by class, ascending within
    first: np.ndarray  # int64 (C,) where each class starts in `members`
    size: np.ndarray  # int64 (C,)
    of_row: np.ndarray  # int64 (n,) the class of a node row; -1 = unpairable

    @property
    def representative(self) -> np.ndarray:
        return self.members[self.first]

    def rows_of(self, classes: np.ndarray) -> np.ndarray:
        """Every member row of ``classes``, ascending."""
        item, copy = _all_copies(self.size[classes])
        return np.sort(self.members[self.first[classes][item] + copy])


def _identical_classes(
    buffer: np.ndarray, offsets: np.ndarray, pairable: np.ndarray
) -> _Classes:
    """Group the ``pairable`` rows (ascending) by xxh3-128, confirmed by
    length. One hash call per row: a loop at node cardinality, run once."""
    import xxhash

    n = offsets.shape[0] - 1
    m = pairable.shape[0]
    digest = xxhash.xxh3_128_digest
    view = memoryview(buffer)
    lo = offsets[pairable].tolist()
    hi = offsets[pairable + 1].tolist()
    out = bytearray(16 * m)
    for i in range(m):
        out[16 * i : 16 * i + 16] = digest(view[lo[i] : hi[i]])
    key = np.frombuffer(bytes(out), dtype=np.uint64).reshape(m, 2)
    length = offsets[pairable + 1] - offsets[pairable]

    # lexsort is stable and `pairable` ascends, so rows stay ascending inside
    # a class and its first row is its lowest.
    order = np.lexsort((key[:, 1], key[:, 0], length))
    fresh = np.ones(m, dtype=bool)
    fresh[1:] = (key[order[1:]] != key[order[:-1]]).any(axis=1) | (
        length[order[1:]] != length[order[:-1]]
    )
    lowest = pairable[order[fresh]]
    rank = np.empty(lowest.shape[0], dtype=np.int64)
    rank[np.argsort(lowest, kind="stable")] = np.arange(lowest.shape[0])
    cls = rank[np.cumsum(fresh) - 1]  # class of each sorted position
    by_class = np.argsort(cls, kind="stable")
    members = pairable[order[by_class]]
    size = np.bincount(cls, minlength=lowest.shape[0]).astype(np.int64)
    of_row = np.full(n, -1, dtype=np.int64)
    of_row[members] = cls[by_class]
    return _Classes(
        members=members, first=np.cumsum(size) - size, size=size, of_row=of_row
    )


def _copies(counts: np.ndarray, limit: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expand item ``i`` into ``counts[i]`` rows, ``limit`` rows at a time.

    Yields ``(item, copy)``: each row's item and which of that item's copies
    it is. Cut in the expanded space, so one item with more copies than
    ``limit`` spans several yields instead of overrunning one.
    """
    ends = np.cumsum(counts, dtype=np.int64)
    total = int(ends[-1]) if ends.shape[0] else 0
    first = ends - counts
    for lo in range(0, total, limit):
        at = np.arange(lo, min(lo + limit, total), dtype=np.int64)
        item = np.searchsorted(ends, at, side="right")
        yield item, at - first[item]


def _all_copies(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_copies` in one piece, for expansions known to be small."""
    item = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    first = np.cumsum(counts, dtype=np.int64) - counts
    return item, np.arange(item.shape[0], dtype=np.int64) - first[item]


# ──────────────────────────────────────────────────────────────────────
# Kernel pool
# ──────────────────────────────────────────────────────────────────────

#: Summed ``len(src) * len(dst)`` of one task. Measured, an inexact pair is
#: 0.5-1.0e-10 s per unit from 700 nt up (95 us for 700 in 1,400; 0.20 ms at
#: 1.4 kb x 1.4 kb; 12 ms at 15 kb x 15 kb), so this is about half a second
#: of kernel whatever the lengths. An EXACT pair is ~4 us at any length — a
#: substring search, no alignment — and nothing says which pairs those are
#: before they are measured: a task of them is over-cut, which costs a little
#: scheduling and nothing else.
_TASK_COST = 5_000_000_000
#: And never more pairs than this. Below ~700 nt the product stops describing
#: the cost: a pair has a floor of 20-50 us (21 us for 60 in 120, 50 us for
#: 300 in 600), which at this many pairs is again about half a second.
_TASK_MAX_PAIRS = 16_384
#: Tasks submitted and not yet consumed, per worker. Results are consumed in
#: submission order, so a slow first task holds every later result in the
#: parent; this is what bounds how many.
_TASKS_IN_FLIGHT = 4
#: Rows per expansion chunk, and per row group written.
_BATCH_ROWS = 1 << 20

#: What the kernel measures, in the order a worker packs it.
_MEASURED = (
    "relation",
    "truncation",
    "src_overhang_5p",
    "src_overhang_3p",
    "dst_overhang_5p",
    "dst_overhang_3p",
    "aligned_len",
    "n_edits",
    "core_len_delta",
    "n_mismatch",
    "n_insert",
    "n_delete",
    "edit_distance_placed",
    "div_5p",
    "div_3p",
    "src_is_container",
)

# (buffer, offsets, params). Set in the parent before the pool forks and
# cleared in its `finally`; workers read it copy-on-write. numpy and bytes
# only: a list[str] here would be copied whole by every worker that read it,
# because taking a reference to a str writes its refcount and that dirties
# the page it lives on.
_KERNEL_STATE: tuple[np.ndarray, np.ndarray, GraphParams] | None = None


def _cut_tasks(
    src: np.ndarray,
    dst: np.ndarray,
    diag: np.ndarray,
    n_shared: np.ndarray,
    overflow: np.ndarray,
    lengths: np.ndarray,
) -> Iterator[tuple[np.ndarray, ...]]:
    """Consecutive runs of candidate pairs, each about ``_TASK_COST`` of
    kernel and at most ``_TASK_MAX_PAIRS`` pairs.

    By cost and not by count: a pair of 15 kb templates is ~100x a pair of
    1.4 kb ones, and equal counts would leave one worker on the long family
    while the rest sat idle. A task is five array slices — numpy pickles a
    view as its own elements, not as the array it was cut from.
    """
    m = src.shape[0]
    if m == 0:
        return
    spent = np.cumsum(lengths[src].astype(np.float64) * lengths[dst])
    lo = 0
    while lo < m:
        before = float(spent[lo - 1]) if lo else 0.0
        hi = int(np.searchsorted(spent, before + _TASK_COST, side="right"))
        hi = min(max(hi, lo + 1), lo + _TASK_MAX_PAIRS, m)
        yield src[lo:hi], dst[lo:hi], diag[lo:hi], n_shared[lo:hi], overflow[lo:hi]
        lo = hi


def _relate_task(
    src: np.ndarray,
    dst: np.ndarray,
    diag: np.ndarray,
    n_shared: np.ndarray,
    overflow: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Measure one task's pairs. Runs below the fork: numpy and edlib only.

    Returns the PRODUCED edges as columns, and how many pairs came out as
    each relation (indexed by :class:`Relation`). Nothing at candidate
    cardinality goes back to the parent.
    """
    state = _KERNEL_STATE
    assert state is not None
    buffer, offsets, params = state
    src_lo, src_hi = offsets[src].tolist(), offsets[src + 1].tolist()
    dst_lo, dst_hi = offsets[dst].tolist(), offsets[dst + 1].tolist()
    rows, hints = src.tolist(), diag.tolist()

    seen = [0] * len(Relation)
    kept: list[int] = []
    measured: list[tuple[int, ...]] = []
    last, query = -1, b""
    for i in range(len(rows)):
        # Candidates are sorted by src: its bytes are cut once per run.
        if rows[i] != last:
            last = rows[i]
            query = buffer[src_lo[i] : src_hi[i]].tobytes()
        m = relate_pair(
            query, buffer[dst_lo[i] : dst_hi[i]].tobytes(), hints[i], params
        )
        seen[m.relation] += 1
        if m.relation in PRODUCED_RELATIONS:
            kept.append(i)
            measured.append(
                (
                    int(m.relation),
                    _TRUNCATION_CODE[m.truncation],
                    m.src_overhang_5p,
                    m.src_overhang_3p,
                    m.dst_overhang_5p,
                    m.dst_overhang_3p,
                    m.aligned_len,
                    m.n_edits,
                    m.core_len_delta,
                    m.n_mismatch,
                    m.n_insert,
                    m.n_delete,
                    m.edit_distance_placed,
                    m.div_5p,
                    m.div_3p,
                    int(m.src_is_container),
                )
            )

    at = np.asarray(kept, dtype=np.int64)
    block = np.asarray(measured, dtype=np.int32).reshape(len(kept), len(_MEASURED))
    out = {name: np.ascontiguousarray(block[:, j]) for j, name in enumerate(_MEASURED)}
    out["src_row"] = src[at].astype(np.int64)
    out["dst_row"] = dst[at].astype(np.int64)
    out["n_shared_probes"] = n_shared[at].astype(np.int32)
    out["candidate_overflow"] = overflow[at].astype(bool)
    return out, np.asarray(seen, dtype=np.int64)


def _measure(
    tasks: list[tuple[np.ndarray, ...]], threads: int
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
    """Every task's result, in the order the tasks were cut — which is what
    makes the output the same for any number of workers."""
    if threads <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield _relate_task(*task)
        return
    workers = min(threads, len(tasks))
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        flying: deque = deque()
        try:
            for task in tasks:
                flying.append(pool.submit(_relate_task, *task))
                if len(flying) >= workers * _TASKS_IN_FLIGHT:
                    yield flying.popleft().result()
            while flying:
                yield flying.popleft().result()
        finally:
            # Reached with tasks still flying only when the consumer stopped.
            for future in flying:
                future.cancel()


# ──────────────────────────────────────────────────────────────────────
# From representatives to rows; annotation; tally
# ──────────────────────────────────────────────────────────────────────


def _twin_edges(
    classes: _Classes, lengths: np.ndarray
) -> Iterator[dict[str, np.ndarray]]:
    """The edges between byte-identical rows: every pair inside a class, the
    lower row as src. They are known without an alignment, and the join never
    sees them — it works on one representative per class."""
    inside = np.arange(classes.members.shape[0]) - np.repeat(
        classes.first, classes.size
    )
    later = np.repeat(classes.size, classes.size) - 1 - inside
    for item, copy in _copies(later, _BATCH_ROWS):
        src = classes.members[item]
        cols = {name: np.zeros(src.shape[0], dtype=np.int32) for name in _MEASURED}
        cols["truncation"][:] = _TRUNCATION_CODE[None]
        cols["aligned_len"] = lengths[src].astype(np.int32)
        cols["src_row"] = src
        cols["dst_row"] = classes.members[item + 1 + copy]
        cols["n_shared_probes"] = np.zeros(src.shape[0], dtype=np.int32)
        cols["candidate_overflow"] = np.zeros(src.shape[0], dtype=bool)
        yield cols


def _to_members(
    cols: dict[str, np.ndarray], classes: _Classes
) -> Iterator[dict[str, np.ndarray]]:
    """An edge between two representatives, for every pair of their classes'
    members: ``|U| x |V|`` edges carrying the one measurement."""
    u = classes.of_row[cols["src_row"]]
    v = classes.of_row[cols["dst_row"]]
    n_v = classes.size[v]
    counts = classes.size[u] * n_v
    if counts.shape[0] == 0 or int(counts.max()) == 1:
        yield cols
        return
    for item, copy in _copies(counts, _BATCH_ROWS):
        out = {name: values[item] for name, values in cols.items()}
        out["src_row"] = classes.members[classes.first[u[item]] + copy // n_v[item]]
        out["dst_row"] = classes.members[classes.first[v[item]] + copy % n_v[item]]
        yield out


@dataclass(frozen=True, slots=True)
class _Nodes:
    """What an edge is annotated from, per node row."""

    node_round: int
    ids: np.ndarray
    lengths: np.ndarray
    n_reads: np.ndarray
    origin: np.ndarray | None
    predicate: MergePredicate


def _annotate(cols: dict[str, np.ndarray], nodes: _Nodes) -> dict[str, np.ndarray]:
    """One batch of edges between node rows, as every ``TEMPLATE_EDGE_TABLE``
    column (``relation`` and ``truncation`` still as codes)."""
    c = dict(cols)
    src, dst = c["src_row"], c["dst_row"]
    # Two kinds of edge are stored the other way round from how they were
    # measured. A contained edge whose container the kernel took as src: it
    # is stored contained member first, like every other. And an equivalent
    # edge between equal lengths, stored lower row first: the kernel saw the
    # two REPRESENTATIVES in that order, and two members of their classes
    # can stand the other way round. Either way the measurement is mirrored:
    # what src overhung dst overhangs, and a base src lacked is a base dst
    # carries. edit_distance_placed stays as it was measured — it is the
    # distance of one sequence placed whole inside the other, and the other
    # way round was never aligned (13 against 12 on a clean pair).
    turn = (
        (c["relation"] == Relation.EQUIVALENT)
        & (nodes.lengths[src] == nodes.lengths[dst])
        & (src > dst)
    )
    turn |= c.pop("src_is_container") != 0
    if bool(turn.any()):
        for a, b in (
            ("src_row", "dst_row"),
            ("src_overhang_5p", "dst_overhang_5p"),
            ("src_overhang_3p", "dst_overhang_3p"),
            ("n_insert", "n_delete"),
        ):
            c[a], c[b] = np.where(turn, c[b], c[a]), np.where(turn, c[a], c[b])
        c["core_len_delta"] = np.where(turn, -c["core_len_delta"], c["core_len_delta"])
        src, dst = c["src_row"], c["dst_row"]

    n = src.shape[0]
    c["node_round"] = np.full(n, nodes.node_round, dtype=np.int32)
    c["src_template_id"] = nodes.ids[src]
    c["dst_template_id"] = nodes.ids[dst]
    c["src_len"] = nodes.lengths[src].astype(np.int32)
    c["dst_len"] = nodes.lengths[dst].astype(np.int32)
    c["delta_5p"] = c["dst_overhang_5p"] - c["src_overhang_5p"]
    c["delta_3p"] = c["dst_overhang_3p"] - c["src_overhang_3p"]
    aligned = c["aligned_len"].astype(np.float64)
    c["identity"] = np.divide(
        aligned - c["n_edits"], aligned, out=np.zeros(n), where=aligned > 0
    )
    if nodes.origin is None:
        c["same_split_origin"] = np.zeros(n, dtype=bool)
    else:
        c["same_split_origin"] = (nodes.origin[src] >= 0) & (
            nodes.origin[src] == nodes.origin[dst]
        )
    c["src_n_reads"] = nodes.n_reads[src]
    c["dst_n_reads"] = nodes.n_reads[dst]
    c["mergeable"], c["identical"] = _mergeable_mask(c, nodes.predicate, False)
    return c


def _dictionary_columns(relation: np.ndarray, truncation: np.ndarray) -> dict:
    """The two dictionary columns, with the enum value as the index."""
    return {
        "relation": pa.DictionaryArray.from_arrays(
            pa.array(relation.astype(np.int8), pa.int8()),
            pa.array([RELATION_NAMES[i] for i in range(len(Relation))], pa.string()),
        ),
        "truncation": pa.DictionaryArray.from_arrays(
            pa.array(truncation.astype(np.int8), pa.int8(), mask=truncation < 0),
            pa.array(list(TRUNCATION_NAMES), pa.string()),
        ),
    }


def _edge_table(c: dict[str, np.ndarray], schema: pa.Schema) -> pa.Table:
    coded = _dictionary_columns(c["relation"], c["truncation"])
    return pa.Table.from_arrays(
        [
            coded[f.name] if f.name in coded else pa.array(c[f.name], type=f.type)
            for f in schema
        ],
        schema=schema,
    )


_EDIT_BINS = ("0", "1", "2", "3-5", "6+")


class _Tally:
    """The stats of the edges, counted as they are written — the file is
    never read back to describe it."""

    def __init__(self) -> None:
        self.n = dict.fromkeys(
            (
                "n_edges",
                "n_equivalent",
                "n_equivalent_exact",
                "n_exact_twins",
                "n_contained",
                "n_exact_nested",
                "n_mergeable",
                "n_same_split_origin",
            ),
            0,
        )
        self.contained = np.zeros(len(TRUNCATION_NAMES), dtype=np.int64)
        self.edits = np.zeros(len(_EDIT_BINS), dtype=np.int64)

    def add(self, c: dict[str, np.ndarray]) -> None:
        equivalent = c["relation"] == Relation.EQUIVALENT
        contained = c["relation"] == Relation.CONTAINED
        exact = c["n_edits"] == 0
        n = self.n
        n["n_edges"] += int(equivalent.shape[0])
        n["n_equivalent"] += int(equivalent.sum())
        n["n_equivalent_exact"] += int((equivalent & exact).sum())
        n["n_exact_twins"] += int(c["identical"].sum())
        n["n_contained"] += int(contained.sum())
        n["n_exact_nested"] += int((contained & exact).sum())
        n["n_mergeable"] += int(c["mergeable"].sum())
        n["n_same_split_origin"] += int((equivalent & c["same_split_origin"]).sum())
        self.contained += np.bincount(
            c["truncation"][contained], minlength=len(TRUNCATION_NAMES)
        )
        self.edits += np.bincount(
            np.digitize(c["n_edits"][equivalent], (1, 2, 3, 6)),
            minlength=len(_EDIT_BINS),
        )

    def stats(self) -> dict:
        return {
            **self.n,
            "contained": dict(zip(TRUNCATION_NAMES, self.contained.tolist())),
            "n_edits_hist": dict(zip(_EDIT_BINS, self.edits.tolist())),
        }


class _EdgeSink:
    """Where finished batches go: a parquet file written beside its final
    name and renamed onto it, or — with no path — one table in memory."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._pending: list[pa.Table] = []
        self._rows = 0
        self._tables: list[pa.Table] = []
        self._tmp: Path | None = None
        self._writer: pq.ParquetWriter | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._tmp = path.with_name(path.name + ".tmp")
            self._writer = pq.ParquetWriter(self._tmp, TEMPLATE_EDGE_TABLE)

    def add(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        self._pending.append(table)
        self._rows += table.num_rows
        if self._rows >= _BATCH_ROWS:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        table = pa.concat_tables(self._pending)
        self._pending, self._rows = [], 0
        if self._writer is None:
            self._tables.append(table)
        else:
            self._writer.write_table(table)

    def close(self) -> pa.Table | None:
        self._flush()
        if self._writer is None:
            if not self._tables:
                return TEMPLATE_EDGE_TABLE.empty_table()
            return pa.concat_tables(self._tables).combine_chunks()
        # A writer closed without a batch still leaves a valid, empty file.
        self._writer.close()
        self._writer = None
        assert self._tmp is not None and self.path is not None
        self._tmp.replace(self.path)
        return None

    def abort(self) -> None:
        """Leave nothing behind. Never raises: it runs while another
        exception is on its way out."""
        with suppress(Exception):
            if self._writer is not None:
                self._writer.close()
        self._writer = None
        with suppress(OSError):
            if self._tmp is not None:
                self._tmp.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────
# The builder
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GraphResult:
    """One node set's edges, and what it took to find them.

    ``edges`` is a ``TEMPLATE_EDGE_TABLE``, or ``None`` when they were
    streamed to ``path``. ``overflow_rows`` / ``truncated_rows`` are sorted
    node rows whose candidate list came through the anchor fallback, or was
    cut at ``max_candidates``: their edge lists may be incomplete. Rows that
    are byte-identical share one candidate list, and are all named.
    """

    edges: pa.Table | None
    path: Path | None
    stats: dict
    overflow_rows: np.ndarray
    truncated_rows: np.ndarray


def _plain(value: object) -> object:
    """``value`` as something ``json.dumps`` takes."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _find_candidates(
    whole: pa.Array,
    classes: _Classes,
    lengths: np.ndarray,
    support: np.ndarray,
    params: GraphParams,
    threads: int,
    seconds: dict,
):
    """Sketch the representatives and join them.

    Parent only. This is the one place torch runs, and the modules that use
    it are imported here rather than at the top, so importing this module
    brings in nothing of the sketch.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.candidates import (
        generate_containment_candidates,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
        extract_minimizers,
    )

    rep = classes.representative
    started = time.perf_counter()
    if rep.shape[0] != len(whole):
        whole = whole.take(pa.array(rep, pa.int64()))
    # Uncapped: a bottom-m sketch preserves Jaccard similarity, and a 600-nt
    # fragment of a 5 kb template shares 12% of the union however exact.
    index = extract_minimizers(whole, k=params.kmer, w=params.window, max_per_seq=None)
    seconds["sketch"] = time.perf_counter() - started

    started = time.perf_counter()
    found = generate_containment_candidates(
        index,
        lengths[rep],
        support,
        k=params.kmer,
        probes_per_seq=params.probes_per_seq,
        bucket_cap=params.bucket_cap,
        overflow_anchors=params.overflow_anchors,
        max_candidates=params.max_candidates,
        min_shared=params.min_shared,
        diag_band=params.diag_band,
        chunk_rows=params.chunk_rows,
        max_rows=params.max_rows,
        threads=threads,
    )
    seconds["candidates"] = time.perf_counter() - started
    return found, int(index.mini_hash.shape[0])


def build_graph(
    sequences: pa.Array | pa.ChunkedArray,
    *,
    ids: np.ndarray,
    n_reads: np.ndarray,
    split_origin: np.ndarray | None = None,
    node_round: int,
    params: GraphParams = GraphParams(),
    predicate: MergePredicate = MergePredicate(),
    threads: int = 1,
    output_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> GraphResult:
    """Relate every pair of ``sequences`` worth relating.

    Row ``i`` of ``sequences`` is node row ``i``; ``ids``, ``n_reads`` and
    ``split_origin`` (``-1`` = none) are indexed by it. The edges are written
    to ``output_path`` as they are found — atomically, and as a valid empty
    file when there are none — or returned in memory without one.

    Rows shorter than ``kmer + window - 1`` cannot be sketched and take no
    part. Byte-identical rows are measured once, through their lowest row,
    and every edge of that row is then written for each of them; the rows of
    a class are themselves joined by exact ``equivalent`` edges that needed
    no alignment.

    The edges do not depend on ``threads``, nor on how the work was cut up.

    ``candidate_overflow`` is set on an edge that came through either cap of
    the join: a pair only the anchor fallback found, and every pair of a
    query whose candidate list was cut at ``max_candidates``.

    ``stats`` (plain ints, floats and dicts):

    * ``n_sequences``, ``n_unpairable_short``, ``n_unique`` (distinct
      sequences among the pairable rows), ``n_identical_classes`` (of more
      than one row), ``largest_identical_class``
    * ``candidates`` — the probe join's own stats, over the unique sequences
    * ``n_pairs_aligned`` — candidate pairs the kernel measured, which is
      ``n_edges_measured`` + the sum of ``dropped`` (by relation name)
    * ``n_edges`` and every count below are of the rows WRITTEN, after
      byte-identical rows were expanded: ``n_equivalent``,
      ``n_equivalent_exact`` (no edits), ``n_exact_twins`` (byte-identical),
      ``n_contained``, ``contained`` (by truncation), ``n_exact_nested``
      (contained, no edits), ``n_edits_hist`` (equivalent edges),
      ``n_mergeable``, ``n_same_split_origin`` (equivalent edges)
    * ``n_overflow_templates``, ``n_truncated_templates``
    * ``seconds`` — ``sketch``, ``candidates``, ``kernel``, ``total``
    """
    global _KERNEL_STATE

    began = time.perf_counter()
    say = progress if progress is not None else (lambda line: None)
    threads = max(int(threads), 1)
    seconds = {"sketch": 0.0, "candidates": 0.0, "kernel": 0.0, "total": 0.0}

    buffer, offsets = _sequence_buffer(sequences)
    n = offsets.shape[0] - 1
    lengths = np.diff(offsets)
    if n and int(lengths.max()) > _MAX_SEQUENCE_LEN:
        raise ValueError(
            f"a sequence of {int(lengths.max()):,} nt is longer than an edge "
            f"can describe ({_MAX_SEQUENCE_LEN:,})"
        )
    nodes = _Nodes(
        node_round=int(node_round),
        ids=_int64_column("ids", ids, n),
        lengths=lengths,
        n_reads=_int64_column("n_reads", n_reads, n),
        origin=(
            None
            if split_origin is None
            else _int64_column("split_origin", split_origin, n)
        ),
        predicate=predicate,
    )

    pairable = np.flatnonzero(lengths >= params.kmer + params.window - 1)
    classes = _identical_classes(buffer, offsets, pairable)
    n_unique = int(classes.size.shape[0])
    twinned = classes.size[classes.size > 1]
    say(
        f"template graph: {n:,} sequences, {n_unique:,} distinct, "
        f"{n - pairable.shape[0]:,} too short to pair"
    )

    whole = pa.LargeStringArray.from_buffers(
        n, pa.py_buffer(offsets), pa.py_buffer(buffer)
    )
    # A class is as well supported as its rows together. Used to rank the
    # anchors of an overflowing bucket and to break ties at a cap, only.
    support = np.zeros(n_unique, dtype=np.int64)
    if n_unique:
        support = np.add.reduceat(nodes.n_reads[classes.members], classes.first)
    found, n_minimizers = _find_candidates(
        whole, classes, lengths, support, params, threads, seconds
    )
    del whole
    # Rows travel to the workers, a candidate at a time: int32 is 8 bytes a
    # pair off every task, and off what the parent holds while they run.
    rep = classes.representative.astype(np.int32 if n < (1 << 31) else np.int64)
    src_class = _column(found.table, "src_row")
    src = rep[src_class]
    dst = rep[_column(found.table, "dst_row")]
    diag = _column(found.table, "diag")
    n_shared = _column(found.table, "n_shared")
    # "Came through a cap" is either cap: the anchor fallback, or a list cut
    # at max_candidates. Both leave an endpoint's edges incomplete.
    overflow = _column(found.table, "overflow") | np.isin(
        src_class, found.truncated_rows
    )
    candidate_stats = found.stats
    overflow_rows = classes.rows_of(found.overflow_rows)
    truncated_rows = classes.rows_of(found.truncated_rows)
    del found, src_class
    say(
        f"template graph: {n_minimizers:,} minimizers in "
        f"{seconds['sketch']:.1f} s, {src.shape[0]:,} candidate pairs in "
        f"{seconds['candidates']:.1f} s"
    )

    started = time.perf_counter()
    tally = _Tally()
    seen = np.zeros(len(Relation), dtype=np.int64)
    n_measured = 0
    sink = _EdgeSink(None if output_path is None else Path(output_path))
    _KERNEL_STATE = (buffer, offsets, params)
    try:
        for cols in _twin_edges(classes, lengths):
            done = _annotate(cols, nodes)
            tally.add(done)
            sink.add(_edge_table(done, TEMPLATE_EDGE_TABLE))
        tasks = list(_cut_tasks(src, dst, diag, n_shared, overflow, lengths))
        # Closed here, not when the generator is collected: a failure below
        # must stop the pool before it is reported, not some time after.
        with closing(_measure(tasks, threads)) as results:
            for cols, relations in results:
                seen += relations
                n_measured += int(cols["src_row"].shape[0])
                for part in _to_members(cols, classes):
                    done = _annotate(part, nodes)
                    tally.add(done)
                    sink.add(_edge_table(done, TEMPLATE_EDGE_TABLE))
        edges = sink.close()
    except BaseException:
        sink.abort()
        raise
    finally:
        _KERNEL_STATE = None
    seconds["kernel"] = time.perf_counter() - started
    seconds["total"] = time.perf_counter() - began

    stats = {
        "n_sequences": n,
        "n_unique": n_unique,
        "n_unpairable_short": n - int(pairable.shape[0]),
        "n_identical_classes": int(twinned.shape[0]),
        "largest_identical_class": int(twinned.max()) if twinned.shape[0] else 0,
        "candidates": candidate_stats,
        "n_pairs_aligned": int(seen.sum()),
        "n_edges_measured": n_measured,
        "dropped": {
            RELATION_NAMES[int(r)]: int(seen[r])
            for r in Relation
            if r not in PRODUCED_RELATIONS
        },
        **tally.stats(),
        "n_overflow_templates": int(overflow_rows.shape[0]),
        "n_truncated_templates": int(truncated_rows.shape[0]),
        "seconds": {name: round(value, 3) for name, value in seconds.items()},
    }
    say(
        f"template graph: {stats['n_edges']:,} edges "
        f"({stats['n_equivalent']:,} equivalent, {stats['n_contained']:,} "
        f"contained) from {stats['n_pairs_aligned']:,} pairs in "
        f"{seconds['kernel']:.1f} s"
    )
    return GraphResult(
        edges=edges,
        path=None if output_path is None else Path(output_path),
        stats=_plain(stats),
        overflow_rows=overflow_rows,
        truncated_rows=truncated_rows,
    )


# ──────────────────────────────────────────────────────────────────────
# Cluster edges
# ──────────────────────────────────────────────────────────────────────


def cluster_edges(
    edges: pa.Table,
    keep_rows: np.ndarray,
    *,
    cluster_len: np.ndarray,
    cluster_n_reads: np.ndarray,
) -> pa.Table:
    """The edges among the nodes that survive as clusters, keyed on cluster.

    ``keep_rows`` is the ascending array of surviving node rows, so a row's
    cluster id is its position there. ``cluster_len`` / ``cluster_n_reads``
    are indexed by cluster id and replace the node's own length and read
    count: a survivor holds the reads of what it absorbed.

    An edge is kept only when BOTH its rows survive. An edge to an absorbed
    node is dropped, not re-pointed at the survivor: its overhangs and edits
    were measured on the absorbed sequence, and written against the survivor
    they would be numbers nobody measured. Where the survivor relates to the
    same neighbour, that pair was a candidate in its own right and has its
    own edge.

    No self-edge can arise, and no pair twice: distinct surviving rows are
    distinct clusters, and the template edges hold each pair of rows once
    and no row against itself. For the same reason every row is decided
    alone, and a file may be passed through one record batch at a time.
    """
    keep_rows = _int64_column("keep_rows", keep_rows, None)
    k = keep_rows.shape[0]
    if k > 1 and not bool((np.diff(keep_rows) > 0).all()):
        raise ValueError("keep_rows must be strictly ascending")
    cluster_len = _int64_column("cluster_len", cluster_len, k)
    cluster_n_reads = _int64_column("cluster_n_reads", cluster_n_reads, k)
    if k == 0 or edges.num_rows == 0:
        return CLUSTER_EDGE_TABLE.empty_table()

    src_row = _column(edges, "src_row")
    dst_row = _column(edges, "dst_row")
    src = np.minimum(np.searchsorted(keep_rows, src_row), k - 1)
    dst = np.minimum(np.searchsorted(keep_rows, dst_row), k - 1)
    kept = (keep_rows[src] == src_row) & (keep_rows[dst] == dst_row)
    src, dst = src[kept], dst[kept]

    c: dict[str, np.ndarray] = {
        "src_cluster_id": src,
        "dst_cluster_id": dst,
        "relation": _codes(edges.column("relation"), _RELATION_CODE)[kept],
        "truncation": _codes(edges.column("truncation"), _TRUNCATION_CODE)[kept],
        "src_len": cluster_len[src].astype(np.int32),
        "dst_len": cluster_len[dst].astype(np.int32),
        "src_n_reads": cluster_n_reads[src],
        "dst_n_reads": cluster_n_reads[dst],
    }
    for f in EDGE_FEATURE_FIELDS:
        if f.name not in c:
            c[f.name] = _column(edges, f.name)[kept]
    return _edge_table(c, CLUSTER_EDGE_TABLE)


__all__ = [
    "CLUSTER_EDGE_TABLE",
    "EDGE_FEATURE_FIELDS",
    "GRAPH_KERNEL_VERSION",
    "GraphParams",
    "GraphResult",
    "MergePredicate",
    "PRODUCED_RELATIONS",
    "PairMeasurement",
    "RELATION_NAMES",
    "Relation",
    "TEMPLATE_EDGE_TABLE",
    "TRUNCATION_NAMES",
    "build_graph",
    "cluster_edges",
    "edit_budget",
    "effective_tolerance",
    "graph_stamp",
    "input_digest",
    "is_mergeable",
    "mergeable_pairs",
    "relate_pair",
    "split_origins",
]
