"""Content-derived ids for proteins, peptides and precursors.

The ids :func:`~constellation.massspec.library.library.assign_ids` hands
out are sequential, so they mean something only inside the table set
they were made for. Two searches run in parallel number the same peptide
differently; a search with no library has nothing to number against; and
results from the two cannot be stacked or joined until something
renumbers them against a shared library.

A content id is computed from what the entity *is*: a protein's
accession, a peptide's canonical ProForma 2.0 string, that string plus a
charge for a precursor. Any producer, anywhere, arrives at the same
number for the same molecule, so tables from independent runs
concatenate and join as they are. The price is that ids are large and
carry no order.

The recipe (``ID_SCHEME``) is small enough to redo by hand or in another
language::

    key = canonical string            (see the three ``canonical_*_key``)
    id  = first 8 bytes of BLAKE2b(key as UTF-8), big-endian,
          top bit cleared            (so it fits a signed int64, >= 0)

Opt-in. Nothing switches to these ids on its own: build keys and ids
with the functions here, or re-key a whole ``Library`` with
:func:`with_content_ids`, which also records the scheme in the library's
metadata.

What "canonical" settles
------------------------

Two strings for one molecule must give one key:

- a modification is written by accession (``[UNIMOD:35]``), whether it
  arrived as an accession or as a name the vocabulary knows
  (``[Oxidation]``); names the vocabulary does not know are kept;
- a bare mass is rounded to ``MASS_DECIMALS`` places, so ``+57.02146``
  and ``+57.0214635`` agree (it is still not *resolved*: deciding which
  modification a mass stands for needs the residue and belongs to the
  reader that met the mass);
- several modifications on one site are put in a fixed order;
- charge, adducts and the ``(>name)`` label are not part of a peptide's
  identity and are dropped (the charge comes back in a precursor key).

What it refuses, because there is no single molecule to name: ambiguous
localisation (``#g1`` groups, ``?`` unknown-position blocks, ranges),
``INFO`` tags, global fixed-modification declarations (expand them onto
the residues first), and multi-chain (cross-linked) peptidoforms.

Terminal versus side-chain placement is part of identity and is *not*
normalised away: ``[UNIMOD:1]-KVERPD`` and ``K[UNIMOD:1]VERPD`` are
different molecules and get different ids.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import replace

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import (
    ModRef,
    MultiPeptidoform,
    Peptidoform,
    TaggedMod,
    format_proforma,
    parse_proforma,
)
from constellation.massspec.library.library import Library

#: Names the recipe above. Recorded with any table set that uses these
#: ids; a change to the hash or to what "canonical" means is a new name.
ID_SCHEME = "content-v1"

#: Metadata key a re-keyed ``Library`` records the scheme under.
ID_SCHEME_METADATA_KEY = "x.constellation.id_scheme"

#: Decimal places a bare mass delta is rounded to. 0.1 mDa: finer than
#: any two distinct modifications (the closest common pairs are several
#: mDa apart), coarser than the print precision of any tool met so far
#: (four to seven places).
MASS_DECIMALS = 4

_ID_MASK = (1 << 63) - 1

# A string that is already canonical and needs no parsing: uppercase
# residues, at most one accession-form UNIMOD modification per site.
# This is what the readers emit for almost every peptide, and the full
# ProForma parser costs far more than a regular-expression test.
_RE_ALREADY_CANONICAL = re.compile(
    r"(?:\[UNIMOD:\d+\]-)?(?:[A-Z](?:\[UNIMOD:\d+\])?)+(?:-\[UNIMOD:\d+\])?"
)


class NoCanonicalKeyError(ValueError):
    """The peptidoform does not name a single molecule, so it has no key."""


class IdCollisionError(RuntimeError):
    """Two different keys hashed to the same id (about 1 in 10^7 for a
    million keys). Nothing can be done at this layer but say so."""


# ──────────────────────────────────────────────────────────────────────
# canonical keys
# ──────────────────────────────────────────────────────────────────────


def _canonical_modref(mod: ModRef, vocab: ModVocab) -> ModRef:
    if mod.cv == "INFO":
        raise NoCanonicalKeyError("an INFO tag carries no chemistry and has no canonical form")
    if mod.cv in ("Formula", "Glycan"):
        return mod
    if mod.accession is not None:
        return ModRef(cv=mod.cv, accession=mod.accession)
    if mod.name is not None:
        for candidate in (f"{mod.cv}:{mod.name}" if mod.cv else None, mod.name):
            if candidate is not None and candidate in vocab:
                cv, _, accession = vocab.get(candidate).id.partition(":")
                return ModRef(cv=cv, accession=accession)
        return ModRef(cv=mod.cv, name=mod.name)
    if mod.mass_delta is not None:
        # +0.0 rather than -0.0, so the sign printed is stable.
        return ModRef(mass_delta=round(mod.mass_delta, MASS_DECIMALS) + 0.0)
    raise NoCanonicalKeyError(f"empty modification reference: {mod!r}")


def _canonical_site(mods: Iterable[TaggedMod], vocab: ModVocab) -> tuple[TaggedMod, ...]:
    out = []
    for tagged in mods:
        if tagged.mod is None or tagged.group_id is not None:
            raise NoCanonicalKeyError(
                "ambiguous localisation (a #group label) does not name a single molecule"
            )
        out.append(TaggedMod(mod=_canonical_modref(tagged.mod, vocab)))
    return tuple(sorted(out, key=lambda t: format_proforma(Peptidoform("A", {0: (t,)}))))


def _canonical_peptidoform(peptide: Peptidoform, vocab: ModVocab) -> Peptidoform:
    for attr, what in (
        ("unknown_pos_mods", "modifications of unknown position"),
        ("ranges", "modifications on a range of residues"),
    ):
        if getattr(peptide, attr):
            raise NoCanonicalKeyError(f"{what} do not name a single molecule")
    if peptide.fixed_mods:
        raise NoCanonicalKeyError(
            "a global fixed-modification declaration (<[Mod]@X>) must be expanded onto "
            "the residues before a key is derived"
        )
    if not peptide.sequence:
        raise NoCanonicalKeyError("an empty sequence has no key")
    labile = sorted(
        (_canonical_modref(m, vocab) for m in peptide.labile_mods),
        key=lambda m: format_proforma(Peptidoform("A", labile_mods=(m,))),
    )
    return Peptidoform(
        sequence=peptide.sequence,
        residue_mods={
            position: _canonical_site(mods, vocab)
            for position, mods in sorted(peptide.residue_mods.items())
            if mods
        },
        n_term_mods=_canonical_site(peptide.n_term_mods, vocab),
        c_term_mods=_canonical_site(peptide.c_term_mods, vocab),
        labile_mods=tuple(labile),
        global_isotopes=tuple(sorted(peptide.global_isotopes)),
    )


def _as_peptidoform(peptide: Peptidoform | str) -> Peptidoform:
    parsed = parse_proforma(peptide) if isinstance(peptide, str) else peptide
    if isinstance(parsed, MultiPeptidoform):
        raise NoCanonicalKeyError(
            "multi-chain (cross-linked) peptidoforms have no canonical key yet"
        )
    return parsed


def canonical_peptide_key(peptide: Peptidoform | str, *, vocab: ModVocab = UNIMOD) -> str:
    """The canonical ProForma 2.0 string of a peptide (no charge).

    ``peptide`` is a ``Peptidoform`` or a ProForma string. The result is
    itself valid ProForma and is a fixed point: its own key is itself.
    Raises :class:`NoCanonicalKeyError` when the input does not name a
    single molecule (module docstring).
    """
    if isinstance(peptide, str) and _RE_ALREADY_CANONICAL.fullmatch(peptide):
        return peptide
    return format_proforma(_canonical_peptidoform(_as_peptidoform(peptide), vocab))


def canonical_precursor_key(
    peptide: Peptidoform | str, charge: int | None = None, *, vocab: ModVocab = UNIMOD
) -> str:
    """``<canonical peptide key>/<charge>``, which is ProForma's own
    spelling of a charged peptidoform.

    ``charge`` defaults to the one the peptidoform (or a ``/z`` suffix
    on the string) carries; given both, they must agree.
    """
    carried: int | None = None
    if not (isinstance(peptide, str) and _RE_ALREADY_CANONICAL.fullmatch(peptide)):
        peptide = _as_peptidoform(peptide)
        if peptide.adducts:
            raise NoCanonicalKeyError("adduct ions have no canonical precursor key yet")
        carried = peptide.charge
    if charge is None:
        charge = carried
    elif carried is not None and int(carried) != int(charge):
        raise ValueError(f"charge {charge} given for a peptidoform that carries /{carried}")
    if charge is None:
        raise ValueError("a precursor key needs a charge")
    if int(charge) == 0:
        raise ValueError("a precursor's charge cannot be 0")
    return f"{canonical_peptide_key(peptide, vocab=vocab)}/{int(charge)}"


def canonical_protein_key(accession: str) -> str:
    """A protein's key is its accession, with surrounding whitespace
    removed and nothing else changed.

    ``sp|P12345|ALBU_HUMAN`` and ``P12345`` are different keys: which
    part of a FASTA header is "the" accession is the FASTA reader's
    decision, and it has to be made the same way everywhere for ids to
    agree. It is not guessed here.
    """
    key = accession.strip()
    if not key:
        raise NoCanonicalKeyError("an empty accession has no key")
    return key


# ──────────────────────────────────────────────────────────────────────
# ids
# ──────────────────────────────────────────────────────────────────────


def content_id(key: str) -> int:
    """The id of a canonical key (the recipe in the module docstring)."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _ID_MASK


def content_ids(keys: Sequence[str]) -> pa.Array:
    """Ids for ``keys`` as an int64 array, in order.

    Equal keys get equal ids (that is the point); two *different* keys
    with the same id raise :class:`IdCollisionError`.
    """
    seen: dict[int, str] = {}
    out: list[int] = []
    for key in keys:
        value = content_id(key)
        other = seen.setdefault(value, key)
        if other != key:
            raise IdCollisionError(
                f"keys {other!r} and {key!r} share the {ID_SCHEME} id {value}"
            )
        out.append(value)
    return pa.array(out, type=pa.int64())


def peptide_id(peptide: Peptidoform | str, *, vocab: ModVocab = UNIMOD) -> int:
    return content_id(canonical_peptide_key(peptide, vocab=vocab))


def precursor_id(
    peptide: Peptidoform | str, charge: int | None = None, *, vocab: ModVocab = UNIMOD
) -> int:
    return content_id(canonical_precursor_key(peptide, charge, vocab=vocab))


def protein_id(accession: str) -> int:
    return content_id(canonical_protein_key(accession))


# ──────────────────────────────────────────────────────────────────────
# re-keying a Library
# ──────────────────────────────────────────────────────────────────────


def _require_distinct(keys: Sequence[str], what: str, described_by: Sequence[str]) -> None:
    first: dict[str, str] = {}
    for key, description in zip(keys, described_by, strict=True):
        if key in first:
            raise ValueError(
                f"two {what} rows are the same {what} ({key}): "
                f"{first[key]!r} and {description!r}"
            )
        first[key] = description


def _remap(column: pa.ChunkedArray, old_ids: pa.Array, new_ids: pa.Array) -> pa.Array:
    index = pc.index_in(column, value_set=old_ids)
    return pc.take(new_ids, index)


def with_content_ids(library: Library, *, vocab: ModVocab = UNIMOD) -> Library:
    """``library`` with every id replaced by its content id.

    ``protein_id``, ``peptide_id`` and ``precursor_id`` are recomputed
    from the accession, the modified sequence, and the modified sequence
    plus charge; the foreign keys in ``precursors``, ``fragments`` and
    ``protein_peptide`` follow. ``modified_sequence`` is rewritten to
    its canonical key, so a reader of the tables can check an id against
    the row it sits on. The scheme is recorded in the metadata.

    Raises ``ValueError`` when two rows turn out to be the same entity
    (two spellings of one peptide; the same peptide and charge twice),
    since the result could not have unique ids.
    """
    accessions = library.proteins.column("accession").to_pylist()
    protein_keys = [canonical_protein_key(a) for a in accessions]
    _require_distinct(protein_keys, "protein", accessions)

    modseqs = library.peptides.column("modified_sequence").to_pylist()
    peptide_keys = [canonical_peptide_key(m, vocab=vocab) for m in modseqs]
    _require_distinct(peptide_keys, "peptide", modseqs)
    key_of_peptide = dict(
        zip(library.peptides.column("peptide_id").to_pylist(), peptide_keys, strict=True)
    )

    charges = library.precursors.column("charge").to_pylist()
    parents = library.precursors.column("peptide_id").to_pylist()
    precursor_keys = [
        f"{key_of_peptide[parent]}/{int(charge)}"
        for parent, charge in zip(parents, charges, strict=True)
    ]
    _require_distinct(precursor_keys, "precursor", precursor_keys)

    old_protein, new_protein = library.proteins.column("protein_id"), content_ids(protein_keys)
    old_peptide, new_peptide = library.peptides.column("peptide_id"), content_ids(peptide_keys)
    old_precursor = library.precursors.column("precursor_id")
    new_precursor = content_ids(precursor_keys)
    old_protein, old_peptide, old_precursor = (
        c.combine_chunks() for c in (old_protein, old_peptide, old_precursor)
    )

    def _set(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
        return table.set_column(table.schema.get_field_index(name), table.schema.field(name), values)

    proteins = _set(library.proteins, "protein_id", new_protein)
    peptides = _set(library.peptides, "peptide_id", new_peptide)
    peptides = _set(peptides, "modified_sequence", pa.array(peptide_keys, type=pa.string()))
    precursors = _set(library.precursors, "precursor_id", new_precursor)
    precursors = _set(
        precursors,
        "peptide_id",
        _remap(library.precursors.column("peptide_id"), old_peptide, new_peptide),
    )
    fragments = _set(
        library.fragments,
        "precursor_id",
        _remap(library.fragments.column("precursor_id"), old_precursor, new_precursor),
    )
    edges = _set(
        library.protein_peptide,
        "protein_id",
        _remap(library.protein_peptide.column("protein_id"), old_protein, new_protein),
    )
    edges = _set(
        edges,
        "peptide_id",
        _remap(library.protein_peptide.column("peptide_id"), old_peptide, new_peptide),
    )
    return replace(
        library,
        proteins=proteins,
        peptides=peptides,
        precursors=precursors,
        fragments=fragments,
        protein_peptide=edges,
        metadata_extras={**library.metadata_extras, ID_SCHEME_METADATA_KEY: ID_SCHEME},
    )


__all__ = [
    "ID_SCHEME",
    "ID_SCHEME_METADATA_KEY",
    "MASS_DECIMALS",
    "IdCollisionError",
    "NoCanonicalKeyError",
    "canonical_peptide_key",
    "canonical_precursor_key",
    "canonical_protein_key",
    "content_id",
    "content_ids",
    "peptide_id",
    "precursor_id",
    "protein_id",
    "with_content_ids",
]
