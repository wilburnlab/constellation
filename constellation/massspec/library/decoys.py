"""Decoys for a search space: an engine several methods pass through.

A decoy is a sequence that should not be in the sample, made to look
like the targets so that whatever a scorer finds for decoys says what it
finds by chance. The methods (``DECOY_METHODS``):

``reverse-interior`` (the default)
    EncyclopeDIA's rule: the first and last residue stay, the residues
    between are reversed (``PeptideUtils`` in EncyclopeDIA 6.9.23; see
    ``docs/plans/encyclopedia-6.9.23-search-algorithms.md``). Keeping both
    termini keeps a tryptic peptide tryptic and its N-terminal chemistry.
``shuffle-interior``
    The same termini kept, the interior shuffled, seeded by the target, so
    that a set is the same every time and several sets differ: the way to
    grow a decoy database past one decoy per target, for entrapment.
``reverse-c-fixed``
    The classic pseudo-reverse: everything but the C-terminal residue
    reversed. For comparison.
``reverse-protein``
    Whole protein sequences reversed before digestion, as FragPipe-style
    analyses do; the decoy proteins are then digested and modified like
    the targets. For comparison: their peptides do not share the targets'
    masses or lengths.

The three peptide-level methods permute a target peptidoform's residues
with :func:`~constellation.core.sequence.proforma.permute_peptidoform`,
so each residue keeps its modifications, terminal modifications and
isotope labels stay, and the decoy has the target's composition: the same
precursor m/z, the same isolation windows, the same MS1 channels. What
differs is the fragments.

A decoy must not be a target. As EncyclopeDIA does, a candidate whose
sequence, with I and L made equal, is a target's (or a decoy already made
in the same build) gives way to the next one: after a method's own
candidate, terminal-preserving shuffles seeded by the target's sequence;
if none qualifies, the target goes without a decoy in that set and is
counted. Seeding by the sequence, not by row, and settling the sequences
in sorted order (so that, when two propose the same decoy, which keeps it
is decided the same way every time) make a set independent of the order
targets come in, and give all modified forms of one peptide the same
decoy sequence.

A set is a method, a seed and an index: index 0 is the set that competes
with the targets, 1 and up are extra sets (entrapment). Every decoy
records its set (``PEPTIDE_TABLE.decoy_set``) and its target
(``decoy_of``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from constellation.core.sequence.proforma import Peptidoform, TaggedMod, permute_peptidoform
from constellation.core.sequence.protein import SIDE_CHAIN_CYCLIZATIONS


def _ring_key(tagged: TaggedMod) -> str | None:
    """A terminal modification's UNIMOD key, as SIDE_CHAIN_CYCLIZATIONS holds them."""
    mod = tagged.mod
    if mod is None or mod.accession is None:
        return None
    return f"{mod.cv or 'UNIMOD'}:{mod.accession}"

__all__ = [
    "DECOY_METHODS",
    "DEFAULT_DECOY_METHOD",
    "DecoyMethod",
    "DecoySet",
    "PeptideDecoys",
    "il_key",
    "peptide_decoys",
    "reverse_protein",
]

DecoyLevel = Literal["peptide", "protein"]

#: Shuffles tried, after a method's own candidate, before a target goes
#: without a decoy.
MAX_ATTEMPTS = 10


def il_key(sequence: str) -> str:
    """A sequence with I and L made equal (they have one mass)."""
    return sequence.replace("I", "L")


def _rng(sequence: str, set_index: int, seed: int, attempt: int) -> np.random.Generator:
    from constellation.massspec.library.ids import content_id

    return np.random.default_rng(content_id(f"{sequence}|{set_index}|{seed}|{attempt}"))


def _interior_shuffle(n: int, rng: np.random.Generator) -> list[int]:
    if n < 3:
        return list(range(n))
    return [0, *(1 + rng.permutation(n - 2)).tolist(), n - 1]


def _reverse_interior(n: int) -> list[int]:
    if n < 3:
        return list(range(n))
    return [0, *range(n - 2, 0, -1), n - 1]


def _reverse_c_fixed(n: int) -> list[int]:
    if n < 2:
        return list(range(n))
    return [*range(n - 2, -1, -1), n - 1]


def _with_fallback(first: Callable[[int], list[int]] | None):
    def orders(sequence: str, set_index: int, seed: int, keep_first: bool = False) -> Iterator[list[int]]:
        n = len(sequence)
        if first is not None:
            order = first(n)
            if keep_first and order and order[0] != 0:
                # The first residue's side chain is in an N-terminal ring
                # (pyro-Glu from Q or E): it cannot move, or the decoy would
                # carry the ring on a residue that cannot make it.
                order = _reverse_interior(n)
            yield order
        for attempt in range(MAX_ATTEMPTS):
            yield _interior_shuffle(n, _rng(sequence, set_index, seed, attempt))

    return orders


@dataclass(frozen=True, slots=True)
class DecoyMethod:
    """A way of making decoys. Peptide-level methods give, per target
    sequence, the residue orders to try in turn (``orders``)."""

    name: str
    level: DecoyLevel
    description: str
    orders: Callable[[str, int, int], Iterator[list[int]]] | None = None
    #: Its decoys depend on the set's index and seed: only such a method can
    #: make a second set that is not a copy of the first.
    seeded: bool = False


DECOY_METHODS: dict[str, DecoyMethod] = {
    m.name: m
    for m in (
        DecoyMethod(
            "reverse-interior",
            "peptide",
            "first and last residue kept, the interior reversed (EncyclopeDIA)",
            _with_fallback(_reverse_interior),
        ),
        DecoyMethod(
            "shuffle-interior",
            "peptide",
            "first and last residue kept, the interior shuffled (seeded; for entrapment sets)",
            _with_fallback(None),
            seeded=True,
        ),
        DecoyMethod(
            "reverse-c-fixed",
            "peptide",
            "all but the C-terminal residue reversed (pseudo-reverse)",
            _with_fallback(_reverse_c_fixed),
        ),
        DecoyMethod(
            "reverse-protein",
            "protein",
            "protein sequences reversed before digestion (FragPipe-style)",
        ),
    )
}

DEFAULT_DECOY_METHOD = "reverse-interior"


@dataclass(frozen=True, slots=True)
class DecoySet:
    """One set of decoys: how, seeded with what, and its index (0 competes
    with the targets; 1 and up are extra sets)."""

    method: str = DEFAULT_DECOY_METHOD
    index: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in DECOY_METHODS:
            raise ValueError(f"unknown decoy method {self.method!r}; known: {sorted(DECOY_METHODS)}")
        if self.index < 0:
            raise ValueError(f"a decoy set's index is >= 0, got {self.index}")

    @property
    def level(self) -> DecoyLevel:
        return DECOY_METHODS[self.method].level


@dataclass(frozen=True, slots=True)
class PeptideDecoys:
    """A set's decoys: per target that got one, its index in the input,
    the decoy, and the order its residues came from (``order[j]`` is the
    target position of decoy position ``j``)."""

    target: list[int]
    decoy: list[Peptidoform]
    order: list[list[int]]
    n_fallback: int
    n_dropped: int


def peptide_decoys(
    targets: Sequence[Peptidoform],
    decoy_set: DecoySet,
    *,
    taken: set[str],
    key: Callable[[Peptidoform], str],
) -> PeptideDecoys:
    """Decoys of ``targets`` for one peptide-level set.

    ``taken`` holds the I/L keys of every target sequence (and of decoys
    made already by other sets); this set's decoy sequences are added to
    it. ``key`` gives a peptidoform's identity (its canonical key), so
    that two targets cannot be given one decoy molecule. All forms of one
    target sequence get the same decoy sequence: the first form decides it.
    """
    method = DECOY_METHODS[decoy_set.method]
    if method.level != "peptide" or method.orders is None:
        raise ValueError(f"{decoy_set.method} is not a peptide-level method")
    # The order is decided once per I/L class of target sequences, in sorted
    # order: two sequences can propose the same decoy, and which keeps it
    # must not depend on the order the FASTA listed them in. Targets that
    # differ only by I/L have one mass and one spectrum, so their decoys
    # take one order and are I/L twins too; until 2026-11-09 the second
    # twin's proposal collided with the first's and fell back to a shuffle,
    # so one signal counted for two targets against two unrelated decoys.
    classes: dict[str, list[str]] = {}
    for seq in sorted({pf.sequence for pf in targets}):
        classes.setdefault(il_key(seq), []).append(seq)
    # Sequences some form of which closes its first residue's side chain into
    # an N-terminal ring: their first residue stays where it is. Under
    # reverse-c-fixed it moved and the ring stayed at the N-terminus, on a
    # residue that cannot make it, so the predictor refused those decoys and
    # no target (437 of them on one test; until 2026-11-10).
    ringed = {pf.sequence for pf in targets if any(_ring_key(t) in SIDE_CHAIN_CYCLIZATIONS for t in pf.n_term_mods)}
    chosen: dict[str, list[int] | None] = {}  # target sequence → order (None: dropped)
    n_fallback = n_dropped = 0
    for members in classes.values():
        seq = members[0]
        pick = None
        keep_first = any(m in ringed for m in members)
        for attempt, order in enumerate(method.orders(seq, decoy_set.index, decoy_set.seed, keep_first)):
            candidate = il_key("".join(seq[k] for k in order))
            if candidate not in taken:
                pick = order
                n_fallback += attempt > 0
                taken.add(candidate)
                break
        for member in members:
            chosen[member] = pick
    made: set[str] = set()
    target: list[int] = []
    decoy: list[Peptidoform] = []
    orders: list[list[int]] = []
    for i, pf in enumerate(targets):
        order = chosen[pf.sequence]
        if order is None:
            n_dropped += 1
            continue
        form = permute_peptidoform(pf, order)
        k = key(form)
        if k in made:
            n_dropped += 1
            continue
        made.add(k)
        target.append(i)
        decoy.append(form)
        orders.append(order)
    return PeptideDecoys(target, decoy, orders, n_fallback, n_dropped)


def reverse_protein(sequence: str) -> str:
    """``reverse-protein``: the whole sequence reversed."""
    return sequence[::-1]
