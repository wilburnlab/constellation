"""Templates: construction from seeds or nodes, persistence, and mmapped access.

A **template** is the sequence a read is aligned against — in round 1 a
representative read's trimmed cDNA, from round 2 that group's consensus. The
E-step's :class:`~...em.estep.TemplateSet` is the small in-memory form that
tests build by hand; :class:`TemplateStore` is the production form, an Arrow
IPC file opened with :func:`pyarrow.memory_map`.

The distinction matters for the same reason the corpus is mmapped:
``TemplateSet.sequence`` is a ``list[str]`` and ``declared_variants`` a
``list[np.ndarray]``, so at 4M templates that is ~8M PyObject headers in the
parent, inherited by every forked M-step worker and privately dirtied as soon
as they are touched. The store holds the same data as file-backed buffers.

**The FASTA naming contract.** Templates are written to the minimap2 target
FASTA named by their row index, exactly as reads are in the corpus. That is
what lets the PAF scanner parse ``t_name`` as an integer instead of hashing
4M names into a Python dict once per round. ``template_id`` remains a
separate, stable identity carried in the table — the row index is positional
and changes every round; the id does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


TEMPLATE_TABLE: pa.Schema = pa.schema(
    [
        # Stable identity, unique for the life of the run. Row index is NOT
        # this: rows are renumbered every round, ids are not.
        pa.field("template_id", pa.int64(), nullable=False),
        pa.field("sequence", pa.large_string(), nullable=False),
        pa.field("orf_start", pa.int32(), nullable=False),
        pa.field("orf_end", pa.int32(), nullable=False),
        pa.field("orf_aa_length", pa.int32(), nullable=False),
        # Assigned read mass from the previous round; in round 1 the ORF's
        # replication count from the seed analysis.
        pa.field("node_weight", pa.float64(), nullable=False),
        # Round 1's primary ranking key: how many reads carried this ORF.
        pa.field("orf_replication", pa.int64(), nullable=False),
        # Round 1's tie-break: the seed read's Dorado quality. Null from
        # round 2, where the frame is a consensus with no read of its own.
        pa.field("seed_read_quality", pa.float32(), nullable=True),
        # Corpus row of the seed read in round 1, -1 afterwards.
        pa.field("seed_read_row", pa.int32(), nullable=False),
        pa.field("declared_variants", pa.list_(pa.int64()), nullable=False),
    ],
    metadata={b"schema_name": b"EmTemplateTable"},
)

TEMPLATES_ARROW = "templates.arrow"
TEMPLATES_FASTA = "templates.fa"

_BASE_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _BASE_LUT[_b] = _i
for _i, _b in enumerate(b"acgt"):
    _BASE_LUT[_b] = _i


@dataclass(slots=True)
class TemplateStore:
    """Row-indexed, zero-copy access to a round's templates.

    ``hp_run`` — the per-base homopolymer run length the likelihood ranker
    gathers at scattered ``(template, position)`` pairs — is computed **on
    first access, not on open**, and stored as ``uint8``.

    Both details are load-bearing at scale. Only the E-step reducer needs this
    context; every M-step worker opens a store and never touches it. Building
    it eagerly as int32 cost 4 bytes per template base *per process*: measured,
    4 MB of sequence peaked at 112 MB, and round 1's ~6.2 Gb of templates would
    have wanted **~25 GB in every one of ~52 workers**. Lazy + uint8 makes it
    zero in the workers and ~6 GB once in the reducer.

    uint8 loses nothing: the epsilon table saturates at a run length of 64 and
    the error model clamps at ``hp_max`` well before that, so runs are capped
    at 255 rather than truncated into a wrong context.
    """

    table: pa.Table
    seq_offsets: np.ndarray  # int64 (T+1) — into the concatenated buffer
    seq_buffer: np.ndarray  # uint8
    _hp_run: np.ndarray | None = None
    _mm: pa.MemoryMappedFile | None = None

    @property
    def hp_run(self) -> np.ndarray:
        """Per-base homopolymer run length, built on first use."""
        if self._hp_run is None:
            self._hp_run = _homopolymer_runs_segmented(
                self.seq_buffer, self.seq_offsets
            )
        return self._hp_run

    @classmethod
    def open(cls, path: Path | str) -> TemplateStore:
        mm = pa.memory_map(str(path), "r")
        with pa.ipc.open_file(mm) as reader:
            table = reader.read_all()
        return cls._from_table(table, mm)

    @classmethod
    def from_table(cls, table: pa.Table) -> TemplateStore:
        """In-memory construction, for tests and small Jupyter use."""
        return cls._from_table(table.cast(TEMPLATE_TABLE), None)

    @classmethod
    def _from_table(
        cls, table: pa.Table, mm: pa.MemoryMappedFile | None
    ) -> TemplateStore:
        # `chunk(0)`, NOT `combine_chunks()`, when the column is already
        # contiguous: combine_chunks allocates and copies even for a single
        # chunk (measured: 32.2 MB for a 32.2 MB array, against 0 for
        # chunk(0)). On a mmapped file that turns a free view into a full copy
        # of every template base — in every M-step worker.
        col = table.column("sequence")
        if isinstance(col, pa.ChunkedArray):
            if col.num_chunks == 1:
                seq = col.chunk(0)
            elif col.num_chunks == 0:
                seq = pa.array([], pa.large_string())
            else:
                # Multi-batch file: there is no contiguous buffer to view, so
                # this one is unavoidable. `write_templates` makes it rare by
                # writing a single batch.
                seq = col.combine_chunks()
        else:
            seq = col
        offsets = np.asarray(seq.buffers()[1]).view(np.int64)[: len(seq) + 1].copy()
        data = np.asarray(seq.buffers()[2]).view(np.uint8)[: int(offsets[-1])]
        return cls(table=table, seq_offsets=offsets, seq_buffer=data, _mm=mm)

    @property
    def n_templates(self) -> int:
        return self.table.num_rows

    @property
    def template_id(self) -> np.ndarray:
        return self.table.column("template_id").to_numpy(zero_copy_only=False)

    @property
    def node_weight(self) -> np.ndarray:
        return self.table.column("node_weight").to_numpy(zero_copy_only=False)

    @property
    def orf_replication(self) -> np.ndarray:
        return self.table.column("orf_replication").to_numpy(zero_copy_only=False)

    @property
    def seed_read_quality(self) -> np.ndarray:
        col = self.table.column("seed_read_quality")
        return pc.fill_null(col, -1.0).to_numpy(zero_copy_only=False)

    @property
    def seed_read_row(self) -> np.ndarray:
        return self.table.column("seed_read_row").to_numpy(zero_copy_only=False)

    @property
    def orf_start(self) -> np.ndarray:
        return self.table.column("orf_start").to_numpy(zero_copy_only=False)

    @property
    def orf_end(self) -> np.ndarray:
        return self.table.column("orf_end").to_numpy(zero_copy_only=False)

    def lengths(self) -> np.ndarray:
        return np.diff(self.seq_offsets)

    def sequence(self, row: int) -> str:
        lo, hi = int(self.seq_offsets[row]), int(self.seq_offsets[row + 1])
        return self.seq_buffer[lo:hi].tobytes().decode("ascii")

    def declared_variants(self, row: int) -> np.ndarray:
        return np.asarray(
            self.table.column("declared_variants")[row].as_py() or [], dtype=np.int64
        )

    def hp_run_at(self, rows: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Homopolymer run length at ``(template row, template position)``.

        Positions outside a template clamp to its ends rather than raising:
        a CIGAR walk can land one past the last base at a terminal indel.
        """
        rows = np.asarray(rows, dtype=np.int64)
        lo = self.seq_offsets[rows]
        hi = self.seq_offsets[rows + 1]
        g = np.clip(
            lo + np.asarray(positions, dtype=np.int64), lo, np.maximum(hi - 1, lo)
        )
        runs = self.hp_run
        if runs.size == 0:
            return np.ones(rows.size, dtype=np.int32)
        return runs[np.clip(g, 0, runs.size - 1)].astype(np.int32)

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()


#: Bases per homopolymer-run block. The construction temporaries are int64
#: (a cumsum group index plus its gather), so peak is ~16x the block rather
#: than ~16x the whole buffer: unchunked, 4 MB of sequence peaked at 128 MB,
#: which at round 1's ~6.2 Gb of templates would have wanted ~200 GB.
_HP_BLOCK_BASES = 8_000_000


def _homopolymer_runs_segmented(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per-base homopolymer run length, reset at every template boundary.

    Chunked **on template boundaries**, which is what makes the chunking free
    of edge cases: a run never crosses one (that is the reset), so no block
    can split a run and there is nothing to stitch.

    Running :func:`homopolymer_runs` over the concatenated buffer directly
    would also merge a template ending in ``AAA`` with the next one starting
    ``AAA`` into a single run of six — a fabricated context exactly where the
    error model is most sensitive to it.
    """
    n = data.size
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    out = np.empty(n, dtype=np.uint8)
    n_templates = offsets.size - 1
    lo_row = 0
    while lo_row < n_templates:
        base = int(offsets[lo_row])
        # Take whole templates until the block budget is spent; always at
        # least one, so a single template larger than the budget still fits.
        hi_row = int(
            np.searchsorted(offsets, base + _HP_BLOCK_BASES, side="right") - 1
        )
        hi_row = min(max(hi_row, lo_row + 1), n_templates)
        stop = int(offsets[hi_row])
        if stop > base:
            out[base:stop] = _runs_one_block(
                data[base:stop], offsets[lo_row : hi_row + 1] - base
            )
        lo_row = hi_row
    return out


def _runs_one_block(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    codes = _BASE_LUT[data]
    m = codes.size
    change = np.empty(m, dtype=bool)
    change[0] = True
    change[1:] = codes[1:] != codes[:-1]
    starts = offsets[:-1]
    change[starts[(starts >= 0) & (starts < m)]] = True
    grp = np.cumsum(change) - 1
    sizes = np.bincount(grp)
    # uint8: the epsilon table saturates at 64 and the error model clamps at
    # hp_max long before that, so capping is not a truncation into a wrong
    # context — and it is 4x smaller than a per-base int32.
    return np.minimum(sizes[grp], 255).astype(np.uint8)


def write_templates(table: pa.Table, directory: Path) -> tuple[Path, Path]:
    """Write ``templates.arrow`` + ``templates.fa`` (named by row index)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table = table.cast(TEMPLATE_TABLE)

    arrow_path = directory / TEMPLATES_ARROW
    # ONE record batch, always. `write_table` emits a batch per chunk, and a
    # templates table built from the M-step's node shards has one chunk per
    # shard — which would leave every reader with a multi-chunk sequence
    # column and no contiguous buffer to memory-map. Paying one copy here, in
    # the parent, saves it in each of N workers.
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with pa.ipc.new_file(sink, TEMPLATE_TABLE) as writer:
            writer.write_table(table.combine_chunks())

    fasta_path = directory / TEMPLATES_FASTA
    seq_col = table.column("sequence")
    row = 0
    with fasta_path.open("w", encoding="utf-8") as fh:
        for lo in range(0, table.num_rows, 50_000):
            for seq in seq_col.slice(lo, 50_000).to_pylist():
                fh.write(f">{row}\n{seq}\n")
                row += 1
    return arrow_path, fasta_path


def templates_from_seeds(
    seed: pa.Table,
    *,
    group_rep_orf: np.ndarray,
    group_n_reads: np.ndarray,
    read_quality_of_row: np.ndarray | None = None,
    read_row_of_id: dict[str, int] | None = None,
    min_seed_reads: int = 1,
) -> pa.Table:
    """Round-1 templates: one per surviving fold group.

    ``node_weight`` and ``orf_replication`` both start as the fold group's
    read count — the first is what later rounds overwrite with assigned mass,
    the second is round 1's ranking key and is never overwritten, so the two
    are separate columns rather than one reused field.

    ``min_seed_reads`` defaults to 1 and should stay there: a pre-E-step
    filter erases minority proteoforms before they can be tested at all
    (Akap4's 849-aa reference is a one-read ORF group that ends round 1
    holding 26-37 reads), and there is no prune downstream to compensate.
    """
    sel = np.flatnonzero(np.asarray(group_n_reads) >= max(int(min_seed_reads), 1))
    rep = np.asarray(group_rep_orf, dtype=np.int64)[sel]
    take = pa.array(rep)
    n = rep.size

    rep_read_ids = pc.take(seed.column("representative_read_id"), take).to_pylist()
    if read_row_of_id is not None:
        seed_rows = np.array(
            [read_row_of_id.get(r, -1) for r in rep_read_ids], dtype=np.int32
        )
    else:
        seed_rows = np.full(n, -1, dtype=np.int32)

    if read_quality_of_row is not None:
        q = np.where(
            seed_rows >= 0,
            np.asarray(read_quality_of_row, dtype=np.float32)[
                np.clip(seed_rows, 0, None)
            ],
            np.nan,
        ).astype(np.float32)
        quality = pa.array(q, pa.float32())
    else:
        quality = pa.nulls(n, pa.float32())

    weights = np.asarray(group_n_reads, dtype=np.int64)[sel]
    return pa.table(
        {
            "template_id": pa.array(np.arange(n, dtype=np.int64)),
            "sequence": pc.take(seed.column("template_sequence"), take).cast(
                pa.large_string()
            ),
            "orf_start": pc.take(seed.column("orf_start_in_template"), take),
            "orf_end": pc.take(seed.column("orf_end_in_template"), take),
            "orf_aa_length": pc.take(seed.column("orf_aa_length"), take),
            "node_weight": pa.array(weights.astype(np.float64)),
            "orf_replication": pa.array(weights),
            "seed_read_quality": quality,
            "seed_read_row": pa.array(seed_rows),
            "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
        },
        schema=TEMPLATE_TABLE,
    )


__all__ = [
    "TEMPLATES_ARROW",
    "TEMPLATES_FASTA",
    "TEMPLATE_TABLE",
    "TemplateStore",
    "templates_from_seeds",
    "write_templates",
]
