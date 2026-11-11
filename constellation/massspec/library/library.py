"""``Library`` container — five theoretical Arrow tables + helpers.

Holds the protein → peptide → precursor → fragment DAG and the only
genuine M:N edge (protein ↔ peptide). Sample-agnostic: no per-run
observations, no calibrated transmission efficiencies — those live in
``massspec.quant``.

Construction shape:
    Library(
        proteins,                 # PROTEIN_TABLE
        peptides,                 # PEPTIDE_TABLE
        precursors,               # PRECURSOR_TABLE
        fragments,                # LIBRARY_FRAGMENT_TABLE
        protein_peptide,          # PROTEIN_PEPTIDE_EDGE
        peptide_spans=...,        # PEPTIDE_SPAN_TABLE         (optional)
        span_peptide=...,         # SPAN_PEPTIDE_EDGE          (optional)
        peptide_modifications=...,# PEPTIDE_MODIFICATION_TABLE (optional)
    )

The last three record how a peptide was made when the library was built
from a FASTA (``massspec.library.search_space``) and are empty otherwise.

Validation enforces PK uniqueness on each tier table and FK closure
across the FK columns (precursor → peptide, fragment → precursor,
protein_peptide → both endpoints; span → protein, span_peptide → span
and peptide with the span's sequence the peptide's, modification →
peptide, decoy_of → a target peptide), with Arrow compute rather than
Python sets so that it costs little at a proteome's size. When spans are
present, ``protein_peptide`` must be their projection.

``to_network`` projects to ``core.graph.Network`` using prefixed string
node ids (``"P:<id>"`` / ``"K:<id>"`` / ``"C:<id>"``) so a single graph
can hold mixed-tier nodes without id collisions.

``assign_ids`` is a convenience builder for tests and ad-hoc library
construction; it accepts plain dicts keyed on symbolic identifiers and
synthesises sequential int64 ids.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.graph.network import Network
from constellation.core.io.schemas import cast_to_schema
from constellation.core.sequence.proforma import (
    ProFormaSyntaxError,
    parse_proforma,
)
from constellation.massspec.library.schemas import (
    LIBRARY_FRAGMENT_TABLE,
    PEPTIDE_MODIFICATION_TABLE,
    PEPTIDE_SPAN_TABLE,
    PEPTIDE_TABLE,
    PRECURSOR_TABLE,
    PROTEIN_PEPTIDE_EDGE,
    PROTEIN_TABLE,
    SPAN_PEPTIDE_EDGE,
)

#: The span key: a span is where it is in which protein.
SPAN_KEY = ("protein_id", "start", "end")

#: How a search space names the decoys of a protein (``DECOY_<accession>``,
#: ``DECOY<n>_<accession>`` for decoy set n).
_DECOY_PREFIX = re.compile(r"^DECOY\d*_")

# ──────────────────────────────────────────────────────────────────────
# Container
# ──────────────────────────────────────────────────────────────────────


_TIER_PREFIXES = {"protein": "P", "peptide": "K", "precursor": "C"}


@dataclass(frozen=True, slots=True)
class Library:
    proteins: pa.Table
    peptides: pa.Table
    precursors: pa.Table
    fragments: pa.Table
    protein_peptide: pa.Table
    metadata_extras: dict[str, Any] = field(default_factory=dict)
    peptide_spans: pa.Table = field(default_factory=PEPTIDE_SPAN_TABLE.empty_table)
    span_peptide: pa.Table = field(default_factory=SPAN_PEPTIDE_EDGE.empty_table)
    peptide_modifications: pa.Table = field(default_factory=PEPTIDE_MODIFICATION_TABLE.empty_table)

    def __post_init__(self) -> None:
        object.__setattr__(self, "proteins", cast_to_schema(self.proteins, PROTEIN_TABLE))
        object.__setattr__(self, "peptides", cast_to_schema(self.peptides, PEPTIDE_TABLE))
        object.__setattr__(
            self, "precursors", cast_to_schema(self.precursors, PRECURSOR_TABLE)
        )
        object.__setattr__(
            self, "fragments", cast_to_schema(self.fragments, LIBRARY_FRAGMENT_TABLE)
        )
        object.__setattr__(
            self,
            "protein_peptide",
            cast_to_schema(self.protein_peptide, PROTEIN_PEPTIDE_EDGE),
        )
        for name, schema in (
            ("peptide_spans", PEPTIDE_SPAN_TABLE),
            ("span_peptide", SPAN_PEPTIDE_EDGE),
            ("peptide_modifications", PEPTIDE_MODIFICATION_TABLE),
        ):
            object.__setattr__(self, name, cast_to_schema(getattr(self, name), schema))
        self.validate()

    # ── validation ──────────────────────────────────────────────────
    def validate(self) -> None:
        _check_unique(self.proteins, "protein_id")
        _check_unique(self.peptides, "peptide_id")
        _check_unique(self.precursors, "precursor_id")

        protein_ids = self.proteins.column("protein_id")
        peptide_ids = self.peptides.column("peptide_id")
        precursor_ids = self.precursors.column("precursor_id")

        _check_fk(self.precursors, "peptide_id", peptide_ids, "PEPTIDE_TABLE")
        _check_fk(self.fragments, "precursor_id", precursor_ids, "PRECURSOR_TABLE")
        _check_fk(
            self.protein_peptide, "protein_id", protein_ids, "PROTEIN_TABLE"
        )
        _check_fk(
            self.protein_peptide, "peptide_id", peptide_ids, "PEPTIDE_TABLE"
        )
        self._validate_provenance(protein_ids, peptide_ids)

    def _validate_provenance(self, protein_ids: pa.ChunkedArray, peptide_ids: pa.ChunkedArray) -> None:
        """The span, edge, modification and decoy tables (all in Arrow)."""
        spans, edges = self.peptide_spans, self.span_peptide
        # ``decoy_of`` records the peptide a decoy was made from. It is not
        # a foreign key: a decoy stands on its own (PI, 2026-11-07), and a
        # library may keep it without its target. What it names, if held,
        # is not itself a decoy, and only a decoy names one.
        decoys = self.peptides.filter(pc.is_valid(self.peptides.column("decoy_of")))
        if decoys.num_rows:
            if not pc.all(pc.fill_null(decoys.column("is_decoy"), False)).as_py():
                raise ValueError("decoy_of is set on a peptide that is not a decoy")
            targets = self.peptides.filter(pc.is_in(peptide_ids, value_set=decoys.column("decoy_of").combine_chunks()))
            if pc.any(pc.fill_null(targets.column("is_decoy"), False)).as_py():
                raise ValueError("decoy_of names a peptide that is itself a decoy")
        if self.peptide_modifications.num_rows:
            _check_fk(self.peptide_modifications, "peptide_id", peptide_ids, "PEPTIDE_TABLE")
        if not (spans.num_rows or edges.num_rows):
            return
        _check_unique_key(spans, SPAN_KEY, "PEPTIDE_SPAN_TABLE")
        _check_fk(spans, "protein_id", protein_ids, "PROTEIN_TABLE")
        _check_fk(edges, "peptide_id", peptide_ids, "PEPTIDE_TABLE")
        _check_unique_key(edges, (*SPAN_KEY, "peptide_id"), "SPAN_PEPTIDE_EDGE")
        # Every edge names a span, and the span's sequence is the peptide's.
        joined = edges.join(
            spans.select([*SPAN_KEY, "sequence"]), keys=list(SPAN_KEY), join_type="left outer"
        ).join(
            self.peptides.select(["peptide_id", "sequence"]).rename_columns(["peptide_id", "peptide_sequence"]),
            keys="peptide_id",
            join_type="left outer",
        )
        orphan = pc.is_null(joined.column("sequence"))
        if pc.any(orphan).as_py():
            sample = joined.filter(orphan).select(list(SPAN_KEY)).slice(0, 3).to_pylist()
            raise ValueError(f"SPAN_PEPTIDE_EDGE names spans not in PEPTIDE_SPAN_TABLE: {sample}")
        differ = pc.not_equal(joined.column("sequence"), joined.column("peptide_sequence"))
        if pc.any(differ).as_py():
            sample = joined.filter(differ).select(["peptide_id", "sequence", "peptide_sequence"]).slice(0, 3).to_pylist()
            raise ValueError(f"a span's sequence is not its peptide's: {sample}")
        projected = _sorted_pairs(
            edges.select(["protein_id", "peptide_id"]).group_by(["protein_id", "peptide_id"]).aggregate([])
        )
        stated = _sorted_pairs(
            self.protein_peptide.group_by(["protein_id", "peptide_id"]).aggregate([])
        )
        if not projected.equals(stated):
            raise ValueError(
                "PROTEIN_PEPTIDE_EDGE is not the projection of SPAN_PEPTIDE_EDGE "
                f"({stated.num_rows} pairs stated, {projected.num_rows} from the spans)"
            )

    # ── views ───────────────────────────────────────────────────────
    @property
    def n_proteins(self) -> int:
        return self.proteins.num_rows

    @property
    def n_peptides(self) -> int:
        return self.peptides.num_rows

    @property
    def n_precursors(self) -> int:
        return self.precursors.num_rows

    @property
    def n_fragments(self) -> int:
        return self.fragments.num_rows

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_extras)

    def with_metadata(self, extras: dict[str, Any]) -> Library:
        merged = dict(self.metadata_extras)
        merged.update(extras)
        return replace(self, metadata_extras=merged)

    # ── derived structures ──────────────────────────────────────────
    def to_network(
        self,
        tier: Literal["protein-peptide", "peptide-precursor", "full"],
    ) -> Network:
        if tier == "protein-peptide":
            return _bipartite_network(
                self.proteins, "protein_id", "protein",
                self.peptides, "peptide_id", "peptide",
                self.protein_peptide, "protein_id", "peptide_id",
            )
        if tier == "peptide-precursor":
            edges = _peptide_precursor_edges(self.precursors)
            return _bipartite_network(
                self.peptides, "peptide_id", "peptide",
                self.precursors, "precursor_id", "precursor",
                edges, "peptide_id", "precursor_id",
            )
        if tier == "full":
            return _full_directed_network(
                self.proteins, self.peptides, self.precursors,
                self.protein_peptide,
            )
        raise ValueError(
            f"unknown tier {tier!r}; "
            "expected 'protein-peptide' | 'peptide-precursor' | 'full'"
        )

    def subset_proteins(self, protein_ids: Iterable[int]) -> Library:
        """The library of these proteins, their decoy proteins (the decoys a
        search space made of them, named ``DECOY_<accession>`` or
        ``DECOY<n>_<accession>``), and the peptides with an edge to one of
        them, with what those peptides need. A decoy is kept by its own
        edges, so one whose target was filtered away is kept with its
        protein (until 2026-11-10 a peptide-level decoy had no protein and
        followed its target through ``decoy_of``, so it was lost with it);
        a decoy with no protein edge at all (a library made otherwise) still
        follows the target it was made from. A search space made by older
        code, whose decoys may have no proteins, is warned about
        (``search_space.check_search_space_layout``; since 2026-11-11, review
        of 2026-11-10, B8: it was subset without a word, losing decoys)."""
        from constellation.massspec.library.search_space import check_search_space_layout

        check_search_space_layout(self)
        named = pa.array(list(protein_ids), type=pa.int64())
        proteins = self.proteins
        chosen = proteins.filter(pc.is_in(proteins.column("protein_id"), value_set=named))
        accessions = set(chosen.column("accession").to_pylist())
        decoy_rows = proteins.filter(pc.fill_null(proteins.column("is_decoy"), False))
        theirs = [
            pid for pid, acc in zip(decoy_rows.column("protein_id").to_pylist(),
                                    decoy_rows.column("accession").to_pylist(), strict=True)  # fmt: skip
            if _DECOY_PREFIX.sub("", acc, count=1) in accessions
        ]
        keep_proteins = pc.unique(pa.concat_arrays([named, pa.array(theirs, pa.int64())]))
        protein_mask = pc.is_in(
            self.proteins.column("protein_id"), value_set=keep_proteins
        )
        new_proteins = self.proteins.filter(protein_mask)

        edge_mask = pc.is_in(
            self.protein_peptide.column("protein_id"), value_set=keep_proteins
        )
        new_pp_edges = self.protein_peptide.filter(edge_mask)

        with_edges = pc.unique(new_pp_edges.column("peptide_id").combine_chunks())
        # A decoy with no protein edge at all follows the target it was made from.
        edged = pc.unique(self.protein_peptide.column("peptide_id").combine_chunks())
        unedged = pc.invert(pc.is_in(self.peptides.column("peptide_id"), value_set=edged))
        follows = pc.and_(unedged, pc.fill_null(pc.is_in(self.peptides.column("decoy_of"), value_set=with_edges), False))
        keep_peptides = pc.unique(
            pa.chunked_array(
                [with_edges, self.peptides.filter(follows).column("peptide_id").combine_chunks()],
                type=pa.int64(),
            ).combine_chunks()
        )
        peptide_mask = pc.is_in(
            self.peptides.column("peptide_id"), value_set=keep_peptides
        )
        new_peptides = self.peptides.filter(peptide_mask)

        precursor_mask = pc.is_in(
            self.precursors.column("peptide_id"), value_set=keep_peptides
        )
        new_precursors = self.precursors.filter(precursor_mask)

        keep_precursors = new_precursors.column("precursor_id").combine_chunks()
        fragment_mask = pc.is_in(
            self.fragments.column("precursor_id"), value_set=keep_precursors
        )
        new_fragments = self.fragments.filter(fragment_mask)

        return replace(
            self,
            proteins=new_proteins,
            peptides=new_peptides,
            precursors=new_precursors,
            fragments=new_fragments,
            protein_peptide=new_pp_edges,
            peptide_spans=self.peptide_spans.filter(
                pc.is_in(self.peptide_spans.column("protein_id"), value_set=keep_proteins)
            ),
            span_peptide=self.span_peptide.filter(
                pc.is_in(self.span_peptide.column("protein_id"), value_set=keep_proteins)
            ),
            peptide_modifications=self.peptide_modifications.filter(
                pc.is_in(self.peptide_modifications.column("peptide_id"), value_set=keep_peptides)
            ),
        )

    def subset_precursors(self, precursor_ids: Iterable[int] | pa.Array | pa.ChunkedArray) -> Library:
        """The library holding only these precursors and what they need:
        their fragments; the peptides left with a precursor; those
        peptides' modification rows, span edges and protein edges, and the
        spans still making one. Proteins are kept. A decoy is kept without
        its target (``decoy_of`` is a record, not a link; until 2026-11-07
        a decoy whose target had no precursor left was dropped)."""
        if isinstance(precursor_ids, pa.ChunkedArray):
            precursor_ids = precursor_ids.combine_chunks()
        elif not isinstance(precursor_ids, pa.Array):
            precursor_ids = pa.array(list(precursor_ids), type=pa.int64())
        precursors = self.precursors.filter(pc.is_in(self.precursors.column("precursor_id"), value_set=precursor_ids))
        with_precursor = pc.unique(precursors.column("peptide_id").combine_chunks())
        peptides = self.peptides.filter(pc.is_in(self.peptides.column("peptide_id"), value_set=with_precursor))
        keep = peptides.column("peptide_id").combine_chunks()
        precursors = precursors.filter(pc.is_in(precursors.column("peptide_id"), value_set=keep))
        span_peptide = self.span_peptide.filter(pc.is_in(self.span_peptide.column("peptide_id"), value_set=keep))
        spans = self.peptide_spans
        if spans.num_rows:
            used = span_peptide.select(list(SPAN_KEY)).group_by(list(SPAN_KEY)).aggregate([])
            spans = spans.join(used, keys=list(SPAN_KEY), join_type="left semi")
            spans = spans.take(pc.sort_indices(spans, sort_keys=[(k, "ascending") for k in SPAN_KEY]))
        return replace(
            self,
            peptides=peptides,
            precursors=precursors,
            fragments=self.fragments.filter(
                pc.is_in(self.fragments.column("precursor_id"), value_set=precursors.column("precursor_id").combine_chunks())
            ),
            protein_peptide=self.protein_peptide.filter(pc.is_in(self.protein_peptide.column("peptide_id"), value_set=keep)),
            peptide_spans=spans,
            span_peptide=span_peptide,
            peptide_modifications=self.peptide_modifications.filter(
                pc.is_in(self.peptide_modifications.column("peptide_id"), value_set=keep)
            ),
        )


#: The precursor values a prediction carries onto a library, each unset
#: at −1 or null (a predictor that did not predict one leaves the source's).
_PREDICTED_PRECURSOR_VALUES = ("rt_predicted", "ccs_predicted", "charge_probability")


def attach_predictions(source: Library, predicted: Library) -> tuple[Library, dict[str, int]]:
    """``source`` with ``predicted``'s fragments, retention times, CCS and
    charge probabilities.

    What a prediction of a library's own precursors should return: the
    source library, with everything it says about them (which are decoys
    and of what, the spans that make each peptide, each modification and
    why), not the flat library a predictor rebuilds from bare
    (sequence, charge) pairs. Precursors are matched on (modified
    sequence, charge); a predicted retention time, CCS or charge
    probability replaces the source's unless it is unset (null, NaN or
    exactly the −1.0 sentinel; a negative iRT is a value, and until
    2026-11-07 every negative value was taken as unset). An iRT of exactly
    −1.0 cannot be told from unset. A source precursor the predictor did
    not return (a modification the model cannot take, skipped) is left out,
    with what only it needed (:meth:`Library.subset_precursors`), and
    counted. Returns the library and the counts.
    """
    values = list(_PREDICTED_PRECURSOR_VALUES)

    def keyed(lib: Library) -> pa.Table:
        return lib.precursors.select(["precursor_id", "peptide_id", "charge", *values]).join(
            lib.peptides.select(["peptide_id", "modified_sequence"]), keys="peptide_id"
        )

    src = keyed(source).select(["precursor_id", "modified_sequence", "charge"])
    got = keyed(predicted).select(["precursor_id", "modified_sequence", "charge", *values])
    got = got.rename_columns(["predicted_id", "modified_sequence", "charge", *(f"predicted_{v}" for v in values)])
    pairs = src.join(got, keys=["modified_sequence", "charge"], join_type="inner")
    out = source.subset_precursors(pairs.column("precursor_id"))

    at = pc.index_in(out.precursors.column("precursor_id"), value_set=pairs.column("precursor_id").combine_chunks())
    precursors = out.precursors
    for name in values:
        new = pc.take(pairs.column(f"predicted_{name}"), at)
        old = precursors.column(name)
        unset = pc.or_(pc.is_null(new, nan_is_null=True), pc.fill_null(pc.equal(new, -1.0), True))
        merged_values = pc.if_else(unset, old, new)
        precursors = precursors.set_column(precursors.schema.get_field_index(name), name, merged_values)
    fragments = predicted.fragments
    to = pc.index_in(fragments.column("precursor_id"), value_set=pairs.column("predicted_id").combine_chunks())
    fragments = fragments.set_column(
        fragments.schema.get_field_index("precursor_id"), "precursor_id", pc.take(pairs.column("precursor_id"), to)
    ).filter(pc.is_valid(to))
    counts = {
        "precursors": out.precursors.num_rows,
        "precursors_not_predicted": source.precursors.num_rows - out.precursors.num_rows,
        "fragments": fragments.num_rows,
    }
    merged = {**source.metadata_extras, **predicted.metadata_extras}
    return replace(out, precursors=precursors, fragments=fragments, metadata_extras=merged), counts


# ──────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────


def _check_unique(table: pa.Table, column: str) -> None:
    col = table.column(column)
    if pc.count_distinct(col, mode="all").as_py() == len(col):
        return
    counts = table.select([column]).group_by(column).aggregate([([], "count_all")])
    dups = counts.filter(pc.greater(counts.column("count_all"), 1)).column(column).to_pylist()
    raise ValueError(
        f"{column} contains duplicate values: "
        f"{dups[:5]}{'...' if len(dups) > 5 else ''}"
    )


def _check_unique_key(table: pa.Table, columns: Sequence[str], name: str) -> None:
    counts = table.select(list(columns)).group_by(list(columns)).aggregate([([], "count_all")])
    if counts.num_rows == table.num_rows:
        return
    dups = counts.filter(pc.greater(counts.column("count_all"), 1)).select(list(columns)).slice(0, 5)
    raise ValueError(f"{name} has duplicate {tuple(columns)}: {dups.to_pylist()}")


def _check_fk(
    table: pa.Table,
    column: str,
    valid_ids: pa.Array | pa.ChunkedArray,
    target_name: str,
) -> None:
    values = table.column(column)
    if isinstance(valid_ids, pa.ChunkedArray):
        valid_ids = valid_ids.combine_chunks()
    present = pc.is_in(values, value_set=valid_ids)
    if pc.all(present).as_py() in (True, None):
        return
    missing = pc.unique(values.filter(pc.invert(present))).to_pylist()
    sample = sorted(missing)[:5]
    raise ValueError(
        f"{column} references ids not present in {target_name}: "
        f"{sample}{'...' if len(missing) > 5 else ''}"
    )


def _sorted_pairs(t: pa.Table) -> pa.Table:
    t = t.select(["protein_id", "peptide_id"])
    return t.take(pc.sort_indices(t, sort_keys=[("protein_id", "ascending"), ("peptide_id", "ascending")])).combine_chunks()


# ──────────────────────────────────────────────────────────────────────
# Network projection helpers
# ──────────────────────────────────────────────────────────────────────


def _peptide_precursor_edges(precursors: pa.Table) -> pa.Table:
    return pa.table(
        {
            "peptide_id": precursors.column("peptide_id"),
            "precursor_id": precursors.column("precursor_id"),
        }
    )


def _bipartite_network(
    a_nodes: pa.Table,
    a_id_col: str,
    a_tier: str,
    b_nodes: pa.Table,
    b_id_col: str,
    b_tier: str,
    edges: pa.Table,
    edge_a_col: str,
    edge_b_col: str,
) -> Network:
    a_prefix = _TIER_PREFIXES[a_tier]
    b_prefix = _TIER_PREFIXES[b_tier]
    a_ids = [f"{a_prefix}:{i}" for i in a_nodes.column(a_id_col).to_pylist()]
    b_ids = [f"{b_prefix}:{i}" for i in b_nodes.column(b_id_col).to_pylist()]
    nodes = pa.table(
        {
            "id": a_ids + b_ids,
            "tier": [a_tier] * len(a_ids) + [b_tier] * len(b_ids),
        }
    )
    src = [f"{a_prefix}:{i}" for i in edges.column(edge_a_col).to_pylist()]
    dst = [f"{b_prefix}:{i}" for i in edges.column(edge_b_col).to_pylist()]
    edges_tbl = pa.table({"src": src, "dst": dst})
    return Network(nodes, edges_tbl, directed=False)


def _full_directed_network(
    proteins: pa.Table,
    peptides: pa.Table,
    precursors: pa.Table,
    protein_peptide: pa.Table,
) -> Network:
    p_ids = [f"P:{i}" for i in proteins.column("protein_id").to_pylist()]
    k_ids = [f"K:{i}" for i in peptides.column("peptide_id").to_pylist()]
    c_ids = [f"C:{i}" for i in precursors.column("precursor_id").to_pylist()]
    nodes = pa.table(
        {
            "id": p_ids + k_ids + c_ids,
            "tier": (
                ["protein"] * len(p_ids)
                + ["peptide"] * len(k_ids)
                + ["precursor"] * len(c_ids)
            ),
        }
    )

    pp_src = [f"P:{i}" for i in protein_peptide.column("protein_id").to_pylist()]
    pp_dst = [f"K:{i}" for i in protein_peptide.column("peptide_id").to_pylist()]
    kc_src = [f"K:{i}" for i in precursors.column("peptide_id").to_pylist()]
    kc_dst = [f"C:{i}" for i in precursors.column("precursor_id").to_pylist()]
    edges = pa.table(
        {
            "src": pp_src + kc_src,
            "dst": pp_dst + kc_dst,
            "kind": (
                ["protein->peptide"] * len(pp_src)
                + ["peptide->precursor"] * len(kc_src)
            ),
        }
    )
    return Network(nodes, edges, directed=True)


# ──────────────────────────────────────────────────────────────────────
# assign_ids — convenience builder
# ──────────────────────────────────────────────────────────────────────


def assign_ids(
    *,
    proteins: Sequence[dict[str, Any]],
    peptides: Sequence[dict[str, Any]],
    precursors: Sequence[dict[str, Any]],
    fragments: Sequence[dict[str, Any]] = (),
    protein_peptide: Sequence[tuple[str, str]] = (),
    metadata: dict[str, Any] | None = None,
) -> Library:
    """Build a ``Library`` from records keyed by symbolic identifiers.

    ``proteins``    rows with at least ``accession``; optional ``sequence``
                    (nullable per PROTEIN_TABLE) and ``description``.
                    ``protein_id`` is assigned sequentially in input order.
    ``peptides``    rows with at least ``modified_sequence`` (a ProForma
                    2.0 string); ``sequence`` auto-derived via
                    ``parse_proforma(modseq).sequence`` when omitted.
                    Symbolic key = ``modified_sequence``.
    ``precursors``  rows with ``modified_sequence`` (or ``peptide_key``),
                    ``charge``, ``precursor_mz``; optional
                    ``rt_predicted`` / ``ccs_predicted`` (default -1.0).
                    Symbolic key = ``(modified_sequence, charge)``.
    ``fragments``   rows with ``modified_sequence``, ``charge`` (the
                    parent precursor's), and the LIBRARY_FRAGMENT_TABLE
                    fields (``ion_type``, ``position``, ion ``charge``,
                    optional ``loss_id``, ``mz_theoretical``,
                    ``intensity_predicted``, optional ``annotation``).
                    ``ion_type`` / ``position`` / ``charge`` may be
                    ``None`` (or absent) for partial-ID rows whose raw
                    annotation didn't structurally resolve; the raw
                    string rides in ``annotation`` for later re-parsing.
                    To disambiguate parent-charge from fragment-charge,
                    use ``precursor_charge`` and ``charge`` (fragment).
    ``protein_peptide`` iterable of ``(accession, modified_sequence)``.
    """
    proteins_rows = [
        {
            "protein_id": i,
            "accession": p["accession"],
            "sequence": p.get("sequence"),
            "description": p.get("description"),
        }
        for i, p in enumerate(proteins)
    ]
    accession_to_id = {p["accession"]: p["protein_id"] for p in proteins_rows}

    peptides_rows: list[dict[str, Any]] = []
    modseq_to_id: dict[str, int] = {}
    for i, k in enumerate(peptides):
        modseq = k["modified_sequence"]
        seq = k.get("sequence")
        if seq is None:
            try:
                seq = parse_proforma(modseq).sequence
            except ProFormaSyntaxError as exc:
                raise ValueError(
                    f"modified_sequence {modseq!r} is not valid ProForma 2.0: {exc}"
                ) from exc
        peptides_rows.append(
            {"peptide_id": i, "sequence": seq, "modified_sequence": modseq}
        )
        modseq_to_id[modseq] = i

    precursors_rows: list[dict[str, Any]] = []
    pkey_to_id: dict[tuple[str, int], int] = {}
    for i, c in enumerate(precursors):
        modseq = c["modified_sequence"]
        charge = int(c["charge"])
        precursors_rows.append(
            {
                "precursor_id": i,
                "peptide_id": modseq_to_id[modseq],
                "charge": charge,
                "precursor_mz": float(c["precursor_mz"]),
                "rt_predicted": float(c.get("rt_predicted", -1.0)),
                "ccs_predicted": float(c.get("ccs_predicted", -1.0)),
            }
        )
        pkey_to_id[(modseq, charge)] = i

    fragment_rows: list[dict[str, Any]] = []
    for f in fragments:
        modseq = f["modified_sequence"]
        parent_charge = int(f["precursor_charge"])
        fragment_rows.append(
            {
                "precursor_id": pkey_to_id[(modseq, parent_charge)],
                # ion_type / position / charge stay NULL when the source
                # couldn't structurally resolve the annotation — see the
                # nullability rationale on LIBRARY_FRAGMENT_TABLE. Coercing
                # None through int() here would reject exactly the
                # partial-ID rows the schema exists to preserve.
                "ion_type": _opt_int(f.get("ion_type")),
                "position": _opt_int(f.get("position")),
                "charge": _opt_int(f.get("charge")),
                "loss_id": f.get("loss_id"),
                "mz_theoretical": float(f["mz_theoretical"]),
                "intensity_predicted": float(f.get("intensity_predicted", -1.0)),
                "annotation": f.get("annotation"),
            }
        )

    pp_rows = [
        {
            "protein_id": accession_to_id[acc],
            "peptide_id": modseq_to_id[modseq],
        }
        for acc, modseq in protein_peptide
    ]

    return Library(
        proteins=_to_table(proteins_rows, PROTEIN_TABLE),
        peptides=_to_table(peptides_rows, PEPTIDE_TABLE),
        precursors=_to_table(precursors_rows, PRECURSOR_TABLE),
        fragments=_to_table(fragment_rows, LIBRARY_FRAGMENT_TABLE),
        protein_peptide=_to_table(pp_rows, PROTEIN_PEPTIDE_EDGE),
        metadata_extras=dict(metadata or {}),
    )


def _opt_int(value: Any) -> int | None:
    """``int(value)``, passing ``None`` through unchanged."""
    return None if value is None else int(value)


def _to_table(rows: Sequence[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(list(rows), schema=schema)


__all__ = ["Library", "assign_ids", "attach_predictions"]
