"""Content-derived ids: the one recipe for an id independent producers agree on.

An id computed from what an entity *is* — a peptide, a precursor, a
transcript's consensus, a protein sequence — is the xxh64 of a canonical
UTF-8 key, masked to 63 bits so it is a non-negative ``int64`` (the form Arrow
and Parquet consumers handle everywhere). Two producers that agree on the key
agree on the id with no shared state and no import between them; that is what
lets a transcript cluster and a peptide be joined without one domain module
importing the other.

``massspec.library.ids`` builds its canonical peptide / precursor / protein
keys and hashes them by this recipe (its own copy predates this module and
gives identical values); the de novo EM clusterer hashes consensus and protein
sequences with it. What the KEY is belongs to each caller — equal keys must
mean the same entity — and this module only fixes how a key becomes an id.

``xxhash`` is imported on first use, so a base install still imports ``core``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pyarrow as pa

#: 63 bits: the id is a non-negative int64.
ID_MASK = (1 << 63) - 1

_digest: Callable[[bytes], int] | None = None


def _xxh64() -> Callable[[bytes], int]:
    global _digest
    if _digest is None:
        try:
            import xxhash
        except ImportError as exc:  # pragma: no cover — the extras carry it
            raise ImportError(
                "content-derived ids need the `xxhash` package "
                "(installed with the sequencing and ms extras)"
            ) from exc
        _digest = xxhash.xxh64_intdigest
    return _digest


def content_id(key: str) -> int:
    """The id of ``key``: ``xxh64(key.encode('utf-8')) & ID_MASK``."""
    return _xxh64()(key.encode("utf-8")) & ID_MASK


def content_ids(keys: Iterable[str | None]) -> pa.Array:
    """Ids of ``keys`` as an int64 array, in order; a ``None`` key is null."""
    digest = _xxh64()
    out = [
        None if key is None else digest(key.encode("utf-8")) & ID_MASK
        for key in keys
    ]
    return pa.array(out, type=pa.int64())


__all__ = ["ID_MASK", "content_id", "content_ids"]
