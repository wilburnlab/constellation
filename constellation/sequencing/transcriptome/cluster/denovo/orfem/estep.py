"""Stage 2 — E-step: assign every read to a template.

Every read is aligned to every template with minimap2 and assigned by
alignment score, with candidates inside an **absolute** band of the best
settled by a ranked tie-break (group size → ORF length → score).

Two things about this that the design docs get subtly wrong, and which the
code and its tests treat as corrected:

* **The band is not what makes nested templates visible.** If template B is
  template A plus a 130-nt 5' exon, a read matching both scores ~260 AS points
  apart — a 40-point absolute band excludes A just as surely as minimap2's
  stock ``-p 0.8`` does. What makes A *reportable at all* is
  ``--secondary=yes -N 50 -p 0.05``. The band's real job is arbitrating among
  templates of *comparable* score, and absolute is right for that because a
  fixed ratio is far too loose on a 5 kb transcript and far too tight on a
  400 nt one.
* **Fractional tie weights stay inside the model.** ``CLUSTER_MEMBERSHIP_TABLE``
  is one row per (cluster, read) and ``cluster_counts.tsv`` writes ``int(v)``,
  so emitting fractional membership silently truncates counts and breaks the
  invariant that haplotype abundances sum to a cluster's read count. Under
  ``tie_resolution="fractional"`` the split weight is recorded on
  ``READ_ASSIGNMENT_TABLE.weight`` and drives the node-weight update and the
  M-step PWM; the reported membership and quant still come from the argmax.

Reverse-strand hits are dropped, and that is a **correctness** requirement,
not a diagnostic preference: on a ``-`` hit PAF reports ``q_start``/``q_end``
on the forward query while the CIGAR is against the reverse complement, so
feeding ``member_start = q_start`` into the PWM projection is simply wrong.
Reads and templates are both demux-oriented forward, so a ``-`` hit is an
antisense artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Literal

import numpy as np
import pyarrow as pa

from constellation.core.io.schemas import register_schema
from constellation.sequencing.transcriptome.cluster.denovo._cigar import (
    base_codes,
    parse_cigar,
)
from constellation.sequencing.transcriptome.cluster.denovo.haplotypes import (
    member_allele_row,
)


READ_ASSIGNMENT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("template_id", pa.int64(), nullable=False),
        pa.field("round", pa.int32(), nullable=False),
        # 1.0 under rank tie-resolution; a share of 1.0 under fractional.
        pa.field("weight", pa.float32(), nullable=False),
        pa.field("as_score", pa.int32(), nullable=False),
        # best_AS - this hit's AS (0 on the winner) — how contested the read is.
        pa.field("as_delta", pa.int32(), nullable=False),
        pa.field("n_banded", pa.int32(), nullable=False),
        # t_start - q_start: where the read's 5' end sits on the template.
        pa.field("offset_5p", pa.int32(), nullable=False),
        pa.field("q_start", pa.int32(), nullable=False),
        pa.field("q_end", pa.int32(), nullable=False),
        pa.field("t_start", pa.int32(), nullable=False),
        pa.field("t_end", pa.int32(), nullable=False),
        # Kept only when the M-step will reuse minimap2's own alignment.
        pa.field("cigar", pa.large_string(), nullable=True),
        # Alleles at the template's declared variant columns, '.' = uncovered.
        pa.field("allele_string", pa.string(), nullable=True),
        pa.field("sample_id", pa.int64(), nullable=True),
    ],
    metadata={b"schema_name": b"ReadAssignmentTable"},
)

# Non-exclusive: one row per (read, template) the read has a banded hit to,
# whether or not it won. The input of the graph stage.
READ_ORF_COVERAGE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("template_id", pa.int64(), nullable=False),
        pa.field("round", pa.int32(), nullable=False),
        # Fraction of the template's ORF interval the read's alignment spans.
        pa.field("orf_cov_frac", pa.float32(), nullable=False),
        pa.field("orf_identity", pa.float32(), nullable=False),
        pa.field("as_score", pa.int32(), nullable=False),
        pa.field("is_assigned", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_name": b"ReadOrfCoverageTable"},
)

register_schema("ReadAssignmentTable", READ_ASSIGNMENT_TABLE)
register_schema("ReadOrfCoverageTable", READ_ORF_COVERAGE_TABLE)


# minimap2 flags for aligning reads to de novo templates. `-p 0.05` and a
# large `-N` are load-bearing: the stock `-p 0.8` suppresses any hit scoring
# below 80% of the best, which is structurally what a nested or 5'-truncated
# proteoform of a longer template is, so every earlier E-step ran blind to
# them and undercounted ambiguity. `--eqx` is required too — without it the
# cg:Z string is all `M` and cigar_stats folds mismatches into matches, so
# every identity comes out 1.0.
TEMPLATE_MINIMAP2_ARGS: tuple[str, ...] = (
    "-x", "map-ont",
    "-c",
    "--eqx",
    "--secondary=yes",
    "-N", "50",
    "-p", "0.05",
)


@dataclass(frozen=True, slots=True)
class TemplateSet:
    """The E-step's alignment targets, as parallel arrays."""

    name: list[str]  # FASTA name per template, index == template row
    template_id: np.ndarray  # int64
    sequence: list[str]
    orf_start: np.ndarray  # int64, on the template
    orf_end: np.ndarray  # int64
    orf_aa_length: np.ndarray  # int64
    # Rule-1 group size in round 1; the previous round's assigned mass after.
    node_weight: np.ndarray  # float64
    # Declared variant columns per template, template coordinates.
    declared_variants: list[np.ndarray]

    def index_of_name(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.name)}


def rank_banded_candidates(
    as_score: np.ndarray,
    template_idx: np.ndarray,
    group_ptr: np.ndarray,
    *,
    node_weight: np.ndarray,
    orf_aa_length: np.ndarray,
    band_abs: int = 40,
    band_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick each read's template. Pure and vectorized — no minimap2 needed.

    ``group_ptr`` is a ``(G+1,)`` offset array delimiting each read's hits in
    the (contiguous) hit arrays. Returns
    ``(winner_slot, n_banded, as_delta, in_band)``: the hit-array index of each
    read's winner, how many hits tied with it, each hit's deficit from its
    read's best score, and the per-hit band membership mask.

    Ranking inside the band is ``node_weight desc → ORF length desc → score
    desc → template_idx asc``. Round 1 passes rule-1 group size as
    ``node_weight``; later rounds pass the previous round's assigned mass.
    """
    n_hits = as_score.shape[0]
    n_groups = group_ptr.shape[0] - 1
    if n_hits == 0 or n_groups <= 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, e, np.empty(0, dtype=bool)

    starts = group_ptr[:-1]
    sizes = np.diff(group_ptr)
    grp_of_hit = np.repeat(np.arange(n_groups), sizes)
    best = np.maximum.reduceat(as_score, starts)
    # Absolute band by default; band_frac > 0 opts into a hybrid where the
    # slack grows with score.
    slack = np.maximum(band_abs, band_frac * best).astype(np.int64)
    in_band = as_score >= (best - slack)[grp_of_hit]
    n_banded = np.add.reduceat(in_band.astype(np.int64), starts)
    as_delta = best[grp_of_hit] - as_score

    order = np.lexsort(
        (
            template_idx,
            -as_score,
            -orf_aa_length[template_idx],
            -node_weight[template_idx],
            ~in_band,  # in-band hits sort first
            grp_of_hit,
        )
    )
    winner_slot = order[np.searchsorted(grp_of_hit[order], np.arange(n_groups))]
    return winner_slot, n_banded, as_delta, in_band


def _allele_string(
    cigar: str,
    read_seq: str,
    *,
    t_start: int,
    q_start: int,
    columns: np.ndarray,
) -> str:
    """The read's bases at a template's declared variant columns.

    ``.`` = not covered, ``-`` = deletion. Reuses the same projection the PWM
    uses, so an allele read here and a vote cast there cannot disagree.
    """
    if columns.size == 0:
        return ""
    row = member_allele_row(
        parse_cigar(cigar),
        base_codes(read_seq),
        frame_is_query=False,  # minimap2: query = read, target = template
        frame_start=t_start,
        member_start=q_start,
        var_sorted=columns,
    )
    return "".join("." if v < 0 else "ACGT-"[v] for v in row)


@dataclass(slots=True)
class _Group:
    """One read's hits, accumulated across PAF batch boundaries."""

    read_id: str = ""
    tmpl: list[int] = None  # type: ignore[assignment]
    as_score: list[int] = None  # type: ignore[assignment]
    q_start: list[int] = None  # type: ignore[assignment]
    q_end: list[int] = None  # type: ignore[assignment]
    t_start: list[int] = None  # type: ignore[assignment]
    t_end: list[int] = None  # type: ignore[assignment]
    n_match: list[int] = None  # type: ignore[assignment]
    aln_len: list[int] = None  # type: ignore[assignment]
    cigar: list[str] = None  # type: ignore[assignment]

    def reset(self, read_id: str) -> None:
        self.read_id = read_id
        self.tmpl, self.as_score = [], []
        self.q_start, self.q_end = [], []
        self.t_start, self.t_end = [], []
        self.n_match, self.aln_len, self.cigar = [], [], []

    def __len__(self) -> int:
        return 0 if self.tmpl is None else len(self.tmpl)


def iter_read_hit_groups(
    batches: Iterable[pa.RecordBatch],
    templates: TemplateSet,
    *,
    allow_antisense: bool = False,
) -> Iterator[_Group]:
    """Regroup streaming PAF batches into one accumulator per read.

    minimap2 emits a query's hits contiguously, so a group only ever straddles
    a batch boundary — the tail carries forward rather than the whole table
    being materialised.
    """
    idx_of = templates.index_of_name()
    cur = _Group()
    cur.reset("")
    started = False
    for batch in batches:
        cols = {n: batch.column(n).to_pylist() for n in batch.schema.names}
        for i in range(batch.num_rows):
            if not allow_antisense and cols["strand"][i] != "+":
                continue
            t = idx_of.get(cols["t_name"][i])
            if t is None:
                continue
            q = cols["q_name"][i]
            if not started or q != cur.read_id:
                if started and len(cur):
                    yield cur
                nxt = _Group()
                nxt.reset(q)
                cur = nxt
                started = True
            cur.tmpl.append(t)
            cur.as_score.append(int(cols["as_score"][i] or 0))
            cur.q_start.append(int(cols["q_start"][i]))
            cur.q_end.append(int(cols["q_end"][i]))
            cur.t_start.append(int(cols["t_start"][i]))
            cur.t_end.append(int(cols["t_end"][i]))
            cur.n_match.append(int(cols["n_match"][i]))
            cur.aln_len.append(int(cols["aln_len"][i]))
            cur.cigar.append(cols["cigar"][i] or "")
    if started and len(cur):
        yield cur


def assign_reads(
    batches: Iterable[pa.RecordBatch],
    templates: TemplateSet,
    *,
    read_seq: dict[str, str] | None = None,
    sample_of_read: dict[str, int] | None = None,
    round_index: int = 1,
    band_abs: int = 40,
    band_frac: float = 0.0,
    tie_resolution: Literal["rank", "fractional"] = "rank",
    keep_cigars: bool = True,
    emit_coverage: bool = False,
    allow_antisense: bool = False,
) -> tuple[pa.Table, pa.Table]:
    """Assign every read to a template. Returns ``(assignments, coverage)``.

    ``read_seq`` is only needed to read alleles at declared variant columns;
    without it ``allele_string`` is null.
    """
    a_rows: list[tuple] = []
    c_rows: list[tuple] = []
    for grp in iter_read_hit_groups(
        batches, templates, allow_antisense=allow_antisense
    ):
        n = len(grp)
        as_score = np.asarray(grp.as_score, dtype=np.int64)
        tmpl = np.asarray(grp.tmpl, dtype=np.int64)
        ptr = np.array([0, n], dtype=np.int64)
        winner, n_banded, as_delta, in_band = rank_banded_candidates(
            as_score,
            tmpl,
            ptr,
            node_weight=templates.node_weight,
            orf_aa_length=templates.orf_aa_length,
            band_abs=band_abs,
            band_frac=band_frac,
        )
        w_slot = int(winner[0])
        nb = int(n_banded[0])
        sample = None if sample_of_read is None else sample_of_read.get(grp.read_id)

        if tie_resolution == "fractional" and nb > 1:
            slots = np.flatnonzero(in_band)
            nw = templates.node_weight[tmpl[slots]].astype(np.float64)
            share = nw / nw.sum() if nw.sum() > 0 else np.full(slots.size, 1 / slots.size)
        else:
            slots = np.array([w_slot], dtype=np.int64)
            share = np.array([1.0])

        for slot, wt in zip(slots.tolist(), share.tolist()):
            t = int(tmpl[slot])
            cols = templates.declared_variants[t]
            allele = None
            if read_seq is not None and cols.size and grp.cigar[slot]:
                allele = _allele_string(
                    grp.cigar[slot],
                    read_seq[grp.read_id],
                    t_start=grp.t_start[slot],
                    q_start=grp.q_start[slot],
                    columns=cols,
                )
            a_rows.append(
                (
                    grp.read_id,
                    int(templates.template_id[t]),
                    round_index,
                    float(wt),
                    int(as_score[slot]),
                    int(as_delta[slot]),
                    nb,
                    grp.t_start[slot] - grp.q_start[slot],
                    grp.q_start[slot],
                    grp.q_end[slot],
                    grp.t_start[slot],
                    grp.t_end[slot],
                    grp.cigar[slot] if keep_cigars else None,
                    allele,
                    sample,
                )
            )

        if emit_coverage:
            for slot in np.flatnonzero(in_band).tolist():
                t = int(tmpl[slot])
                lo = max(grp.t_start[slot], int(templates.orf_start[t]))
                hi = min(grp.t_end[slot], int(templates.orf_end[t]))
                span = max(int(templates.orf_end[t] - templates.orf_start[t]), 1)
                c_rows.append(
                    (
                        grp.read_id,
                        int(templates.template_id[t]),
                        round_index,
                        max(hi - lo, 0) / span,
                        grp.n_match[slot] / max(grp.aln_len[slot], 1),
                        int(as_score[slot]),
                        slot == w_slot,
                    )
                )

    return (
        _rows_to_table(a_rows, READ_ASSIGNMENT_TABLE),
        _rows_to_table(c_rows, READ_ORF_COVERAGE_TABLE),
    )


def _rows_to_table(rows: list[tuple], schema: pa.Schema) -> pa.Table:
    if not rows:
        return schema.empty_table()
    cols = list(zip(*rows))
    return pa.table(
        {f.name: pa.array(cols[i], type=f.type) for i, f in enumerate(schema)},
        schema=schema,
    )


def run_estep(
    templates: TemplateSet,
    reads_fasta,
    *,
    work_dir,
    threads: int = 8,
    index_batch_size: str = "16G",
    max_template_bases: int = 3_000_000_000,
    extra_minimap2_args: tuple[str, ...] = (),
    **assign_kwargs,
) -> tuple[pa.Table, pa.Table]:
    """Align every read to every template and reduce the stream to assignments.

    Composes the generic ``minimap2_stream`` primitive with the ORF-EM flag
    tuple; nothing is written to disk and the hit stream is reduced one read
    at a time.

    ``index_batch_size`` (``-I``) is deliberately large enough to force a
    **single index part**. minimap2 with a multi-part index runs the queries
    once per part and applies ``-p``, ``-N`` and the primary/secondary call
    *within* each part, so a hit that is best in part 2 can be beaten by a
    part 1 hit it was never compared against. Merging parts correctly needs a
    two-pass shuffle; refusing to run is the honest alternative here.
    """
    from pathlib import Path

    from constellation.sequencing.align.minimap2 import minimap2_stream
    from constellation.sequencing.readers.paf import iter_paf_batches

    total = sum(len(s) for s in templates.sequence)
    if total > max_template_bases:
        raise ValueError(
            f"template set is {total / 1e9:.2f} Gb over {len(templates.sequence):,} "
            f"templates, above the {max_template_bases / 1e9:.2f} Gb ceiling. "
            "This E-step is scoped to ~10^7 reads: a larger reference needs a "
            "multi-part minimap2 index, whose per-part -p/-N filtering is not "
            "comparable across parts. Raise --min-seed-reads (never above ~3 — "
            "a real minority proteoform can rest on ~5 reads) or subsample."
        )
    work_dir = Path(work_dir)
    fasta = write_template_fasta(templates, work_dir / "templates.fa")
    args = (*TEMPLATE_MINIMAP2_ARGS, "-I", index_batch_size, *extra_minimap2_args)
    stream = minimap2_stream(fasta, [Path(reads_fasta)], args=args, threads=threads)
    batches = iter_paf_batches(stream)
    return assign_reads(batches, templates, **assign_kwargs)


def write_template_fasta(templates: TemplateSet, path) -> "object":
    """Write the E-step's alignment targets. Exact-duplicate templates are
    NOT collapsed here — that is the caller's job, and it is not the same
    thing as a template-level fold (which would merge real proteoforms)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for name, seq in zip(templates.name, templates.sequence):
            fh.write(f">{name}\n{seq}\n")
    return path


__all__ = [
    "READ_ASSIGNMENT_TABLE",
    "run_estep",
    "READ_ORF_COVERAGE_TABLE",
    "TEMPLATE_MINIMAP2_ARGS",
    "TemplateSet",
    "assign_reads",
    "iter_read_hit_groups",
    "rank_banded_candidates",
    "write_template_fasta",
]
