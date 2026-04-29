"""``Library`` container — five theoretical Arrow tables + helpers.

Holds the protein → peptide → precursor → fragment DAG and the only
genuine M:N edge (protein ↔ peptide). Sample-agnostic: no per-run
observations, no calibrated transmission efficiencies — those live in
``massspec.quant``.

Construction shape:
    Library(
        proteins,            # PROTEIN_TABLE
        peptides,            # PEPTIDE_TABLE
        precursors,          # PRECURSOR_TABLE
        fragments,           # LIBRARY_FRAGMENT_TABLE
        protein_peptide,     # PROTEIN_PEPTIDE_EDGE
    )

Validation enforces PK uniqueness on each tier table and FK closure
across the four FK columns (peptide → protein, precursor → peptide,
fragment → precursor, protein_peptide → both endpoints).

``to_network`` projects to ``core.graph.Network`` using prefixed string
node ids (``"P:<id>"`` / ``"K:<id>"`` / ``"C:<id>"``) so a single graph
can hold mixed-tier nodes without id collisions.

``assign_ids`` is a convenience builder for tests and ad-hoc library
construction; it accepts plain dicts keyed on symbolic identifiers and
synthesises sequential int64 ids.
"""

from __future__ import annotations

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
    PEPTIDE_TABLE,
    PRECURSOR_TABLE,
    PROTEIN_PEPTIDE_EDGE,
    PROTEIN_TABLE,
)

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
        self.validate()

    # ── validation ──────────────────────────────────────────────────
    def validate(self) -> None:
        _check_unique(self.proteins, "protein_id")
        _check_unique(self.peptides, "peptide_id")
        _check_unique(self.precursors, "precursor_id")

        protein_ids = set(self.proteins.column("protein_id").to_pylist())
        peptide_ids = set(self.peptides.column("peptide_id").to_pylist())
        precursor_ids = set(self.precursors.column("precursor_id").to_pylist())

        _check_fk(self.precursors, "peptide_id", peptide_ids, "PEPTIDE_TABLE")
        _check_fk(self.fragments, "precursor_id", precursor_ids, "PRECURSOR_TABLE")
        _check_fk(
            self.protein_peptide, "protein_id", protein_ids, "PROTEIN_TABLE"
        )
        _check_fk(
            self.protein_peptide, "peptide_id", peptide_ids, "PEPTIDE_TABLE"
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
        keep_proteins = pa.array(list(protein_ids), type=pa.int64())
        protein_mask = pc.is_in(
            self.proteins.column("protein_id"), value_set=keep_proteins
        )
        new_proteins = self.proteins.filter(protein_mask)

        edge_mask = pc.is_in(
            self.protein_peptide.column("protein_id"), value_set=keep_proteins
        )
        new_pp_edges = self.protein_peptide.filter(edge_mask)

        keep_peptides = pa.array(
            sorted(set(new_pp_edges.column("peptide_id").to_pylist())),
            type=pa.int64(),
        )
        peptide_mask = pc.is_in(
            self.peptides.column("peptide_id"), value_set=keep_peptides
        )
        new_peptides = self.peptides.filter(peptide_mask)

        precursor_mask = pc.is_in(
            self.precursors.column("peptide_id"), value_set=keep_peptides
        )
        new_precursors = self.precursors.filter(precursor_mask)

        keep_precursors = pa.array(
            new_precursors.column("precursor_id").to_pylist(), type=pa.int64()
        )
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
        )


# ──────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────


def _check_unique(table: pa.Table, column: str) -> None:
    values = table.column(column).to_pylist()
    if len(set(values)) != len(values):
        seen: set[Any] = set()
        dups: list[Any] = []
        for v in values:
            if v in seen:
                dups.append(v)
            else:
                seen.add(v)
        raise ValueError(
            f"{column} contains duplicate values: "
            f"{dups[:5]}{'...' if len(dups) > 5 else ''}"
        )


def _check_fk(
    table: pa.Table,
    column: str,
    valid_ids: set[Any],
    target_name: str,
) -> None:
    values = table.column(column).to_pylist()
    missing = {v for v in values if v not in valid_ids}
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{column} references ids not present in {target_name}: "
            f"{sample}{'...' if len(missing) > 5 else ''}"
        )


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
                "ion_type": int(f["ion_type"]),
                "position": int(f["position"]),
                "charge": int(f["charge"]),
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


def _to_table(rows: Sequence[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(list(rows), schema=schema)


__all__ = ["Library", "assign_ids"]
