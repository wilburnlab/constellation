"""GFF3 feature reader → ``Annotation`` (FEATURE_TABLE).

Hand-rolled streaming parser, no third-party deps. GFF3 grammar is
compact (9 tab-separated columns + ``key=value;…`` attribute column +
``##`` directives + optional embedded ``##FASTA`` section), so a
streaming Python parser building Arrow column-list arrays is the right
shape — same precedent as ``readers/sam_bam.py`` (which wraps pysam
directly rather than going through pandas) and aligned with the
no-pandas invariant.

Coordinates are converted from GFF3's 1-based inclusive to
constellation's 0-based half-open at the boundary. ``Parent=`` is
resolved via a two-pass to populate ``parent_id`` int64 FKs.

This module implements the public ``read_gff3`` and ``write_gff3``
helpers; the older ``GffReader`` ``RawReader`` subclass remains for the
read-ingest pathway and is still stubbed pending Phase 2 (it targets
``READ_TABLE``-shaped outputs, a different use case).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote, unquote

import pyarrow as pa

from constellation.core.io.readers import RawReader, ReadResult, register_reader
from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.reference.reference import GenomeReference


_PHASE = "Phase 2 (readers/gff RawReader subclass)"

# GFF3 attributes that we promote to dedicated FEATURE_TABLE columns.
# Anything else lands in attributes_json.
_RESERVED_ATTRS: frozenset[str] = frozenset(
    {"ID", "Name", "Parent"}
)


def _open_text(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _parse_attributes(attr_col: str) -> dict[str, str | list[str]]:
    """Parse the GFF3 column-9 attribute string.

    Returns a dict where values are either str or list[str] (when a key
    has multi-value comma-separated entries — e.g. ``Dbxref=NCBI:1,Ensembl:2``).
    """
    if attr_col == "" or attr_col == ".":
        return {}
    out: dict[str, str | list[str]] = {}
    for chunk in attr_col.split(";"):
        if not chunk:
            continue
        if "=" not in chunk:
            # Tolerate malformed entries — record the bare token under
            # itself so we don't lose information.
            out[chunk] = ""
            continue
        key, _, value = chunk.partition("=")
        key = unquote(key.strip())
        if "," in value:
            parts = [unquote(p) for p in value.split(",")]
            out[key] = parts
        else:
            out[key] = unquote(value)
    return out


# ──────────────────────────────────────────────────────────────────────
# read_gff3
# ──────────────────────────────────────────────────────────────────────


def read_gff3(
    path: str | Path,
    *,
    contig_name_to_id: dict[str, int] | None = None,
) -> Annotation:
    """Parse a GFF3 (optionally gzipped) into an ``Annotation``.

    ``contig_name_to_id`` is the mapping from FASTA contig names (the
    GFF3 seqid column) to the int64 ``contig_id`` namespace used in the
    paired ``GenomeReference``. When omitted, contig ids are assigned
    sequentially in encounter order; callers are responsible for
    rebinding them via the explicit map when validating against an
    existing GenomeReference.

    Coordinates are converted 1-based-inclusive → 0-based-half-open
    (``start - 1`` on entry; ``end`` is unchanged because GFF3's
    inclusive end is the same int as 0-based-half-open's exclusive end).

    Multi-parent features (``Parent=g1,g2``) are handled by emitting
    one FEATURE_TABLE row per parent — each row reuses the same source
    ``ID`` plus a ``parent_id`` int64; ``feature_id`` is unique per
    output row. The original ID is preserved as ``Name`` if no Name
    attribute was set explicitly.

    The ``##FASTA`` section, if present, terminates GFF3 parsing — the
    embedded sequence is *not* read here. Callers wanting the FASTA
    must pass the path to ``read_fasta_genome`` separately (we treat
    GFF3 + FASTA as conceptually distinct artifacts even when bundled).

    Returns the new ``Annotation``. Pair it with a ``GenomeReference``
    via ``Annotation.validate_against(genome)`` to confirm contig FK
    closure.
    """
    p = Path(path)

    # Pass-1 buffers (one entry per *output row* — multi-parent features
    # produce multiple entries).
    feature_ids: list[int] = []
    contig_ids: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    strands: list[str] = []
    types: list[str] = []
    names: list[str | None] = []
    sources: list[str | None] = []
    scores: list[float | None] = []
    phases: list[int | None] = []
    attrs_json: list[str | None] = []
    # Track each row's pending parent name (resolved in pass 2).
    parent_names: list[str | None] = []

    # Per-input-row ID → assigned feature_id (used for parent
    # resolution). Multi-parent rows reuse the same ID across multiple
    # output rows; we store the first feature_id assigned (parents
    # resolve to one of them — GFF3 semantics for multi-parent are
    # ambiguous and downstream consumers should treat parent_id as a
    # 'one of the parents' link, not a deterministic FK; the
    # alternative — emitting only one row and storing an Arrow list of
    # parent_ids — would break the column-typed FEATURE_TABLE shape).
    name_to_feature_id: dict[str, int] = {}

    contig_map: dict[str, int] = (
        dict(contig_name_to_id) if contig_name_to_id else {}
    )
    next_contig_id = max(contig_map.values(), default=-1) + 1
    next_feature_id = 0

    directives: list[str] = []
    sequence_regions: list[dict[str, Any]] = []
    saw_fasta_directive = False

    with _open_text(p) as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("##FASTA"):
                saw_fasta_directive = True
                break
            if line.startswith("##"):
                directives.append(line)
                # Capture sequence-region directives so contig length
                # can round-trip when a write_gff3 is called later.
                if line.startswith("##sequence-region"):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            sequence_regions.append(
                                {
                                    "name": parts[1],
                                    "start": int(parts[2]),
                                    "end": int(parts[3]),
                                }
                            )
                        except ValueError:
                            pass
                continue
            if line.startswith("#"):
                # Plain comment — skip
                continue

            cols = line.split("\t")
            if len(cols) < 9:
                # Malformed line — skip silently per GFF3 robustness
                # convention. Real-world Ensembl/NCBI files occasionally
                # have stray header lines; raising here would block
                # otherwise-valid imports.
                continue

            seqid, source_str, ftype, start_s, end_s, score_s, strand, phase_s, attr_col = (
                cols[0],
                cols[1],
                cols[2],
                cols[3],
                cols[4],
                cols[5],
                cols[6],
                cols[7],
                cols[8],
            )

            try:
                start_i = int(start_s) - 1  # 1-based incl → 0-based half-open
                end_i = int(end_s)
            except ValueError:
                continue

            if seqid not in contig_map:
                contig_map[seqid] = next_contig_id
                next_contig_id += 1
            contig_id = contig_map[seqid]

            score_v: float | None
            if score_s == "." or score_s == "":
                score_v = None
            else:
                try:
                    score_v = float(score_s)
                except ValueError:
                    score_v = None

            phase_v: int | None
            if phase_s == "." or phase_s == "":
                phase_v = None
            else:
                try:
                    phase_v = int(phase_s)
                except ValueError:
                    phase_v = None

            attrs = _parse_attributes(attr_col)
            id_attr = attrs.pop("ID", None)
            name_attr = attrs.pop("Name", None)
            parent_attr = attrs.pop("Parent", None)

            # Display name preference: Name attr → ID attr (since Name
            # is conceptually display-only in GFF3, but if absent we
            # fall back to ID so something appears in summary output).
            display_name: str | None
            if isinstance(name_attr, list):
                display_name = name_attr[0] if name_attr else None
            elif name_attr is not None:
                display_name = name_attr
            elif isinstance(id_attr, str):
                display_name = id_attr
            elif isinstance(id_attr, list) and id_attr:
                display_name = id_attr[0]
            else:
                display_name = None

            # Pack remaining attributes as JSON; preserve list-valued
            # attributes (Dbxref, Alias) as JSON arrays.
            attrs_payload: dict[str, Any] = {}
            for k, v in attrs.items():
                attrs_payload[k] = v
            attrs_json_str = json.dumps(attrs_payload) if attrs_payload else None

            # Multi-parent expansion: emit one row per parent name.
            parent_list: list[str | None]
            if parent_attr is None:
                parent_list = [None]
            elif isinstance(parent_attr, list):
                parent_list = list(parent_attr) if parent_attr else [None]
            else:
                parent_list = [parent_attr]

            id_str: str | None
            if isinstance(id_attr, list) and id_attr:
                id_str = id_attr[0]
            elif isinstance(id_attr, str):
                id_str = id_attr
            else:
                id_str = None

            for parent_name in parent_list:
                fid = next_feature_id
                next_feature_id += 1
                feature_ids.append(fid)
                contig_ids.append(contig_id)
                starts.append(start_i)
                ends.append(end_i)
                strands.append(strand if strand in ("+", "-", ".") else ".")
                types.append(ftype)
                names.append(display_name)
                sources.append(source_str if source_str != "." else None)
                scores.append(score_v)
                phases.append(phase_v)
                attrs_json.append(attrs_json_str)
                parent_names.append(parent_name)
                # Bind the input ID to the FIRST feature_id assigned
                # for it (subsequent multi-parent rows reuse the input
                # ID but parent-resolution returns the first row).
                if id_str is not None and id_str not in name_to_feature_id:
                    name_to_feature_id[id_str] = fid

    # Pass 2: resolve parent_names → parent_id ints.
    parent_ids: list[int | None] = []
    unresolved_parents: list[tuple[int, str]] = []
    for fid, pname in zip(feature_ids, parent_names, strict=True):
        if pname is None:
            parent_ids.append(None)
            continue
        if pname in name_to_feature_id:
            parent_ids.append(name_to_feature_id[pname])
        else:
            parent_ids.append(None)
            unresolved_parents.append((fid, pname))

    table = pa.table(
        {
            "feature_id": pa.array(feature_ids, type=pa.int64()),
            "contig_id": pa.array(contig_ids, type=pa.int64()),
            "start": pa.array(starts, type=pa.int64()),
            "end": pa.array(ends, type=pa.int64()),
            "strand": pa.array(strands, type=pa.string()),
            "type": pa.array(types, type=pa.string()),
            "name": pa.array(names, type=pa.string()),
            "parent_id": pa.array(parent_ids, type=pa.int64()),
            "source": pa.array(sources, type=pa.string()),
            "score": pa.array(scores, type=pa.float32()),
            "phase": pa.array(phases, type=pa.int32()),
            "attributes_json": pa.array(attrs_json, type=pa.string()),
        }
    )

    # Build name → contig_id mapping for round-trip and downstream use.
    metadata: dict[str, Any] = {
        "source_path": str(p),
        "contig_name_to_id": dict(contig_map),
        "saw_fasta_directive": saw_fasta_directive,
    }
    if directives:
        metadata["directives"] = directives
    if sequence_regions:
        metadata["sequence_regions"] = sequence_regions
    if unresolved_parents:
        metadata["unresolved_parents"] = [
            {"feature_id": fid, "parent_name": pn}
            for fid, pn in unresolved_parents[:50]
        ]

    return Annotation(features=table, metadata_extras=metadata)


# ──────────────────────────────────────────────────────────────────────
# write_gff3
# ──────────────────────────────────────────────────────────────────────


def write_gff3(
    annotation: Annotation,
    path: str | Path,
    *,
    genome: GenomeReference | None = None,
    contig_id_to_name: dict[int, str] | None = None,
) -> None:
    """Emit ``Annotation`` as a GFF3 file (no compression).

    Coordinates are converted back to GFF3's 1-based-inclusive
    convention. Output is sorted by ``(contig_id, start, end)``.
    Attributes are reconstructed by combining the dedicated columns
    (``Name``, ``Parent`` resolved via the parent_id → ID inverse map)
    with ``attributes_json``.

    The ``contig_id`` → name mapping is resolved in priority order:
    explicit ``contig_id_to_name`` argument, then the ``genome``
    parameter (``GenomeReference.contigs``), then
    ``annotation.metadata_extras['contig_name_to_id']`` (inverted), then
    falls back to ``"contig{id}"``.
    """
    p = Path(path)

    # Build contig_id → name map.
    cid_to_name: dict[int, str] = {}
    if contig_id_to_name:
        cid_to_name.update(contig_id_to_name)
    if genome is not None:
        for cid, name in zip(
            genome.contigs.column("contig_id").to_pylist(),
            genome.contigs.column("name").to_pylist(),
            strict=True,
        ):
            cid_to_name.setdefault(cid, name)
    name_to_cid = annotation.metadata_extras.get("contig_name_to_id")
    if isinstance(name_to_cid, dict):
        for name, cid in name_to_cid.items():
            cid_to_name.setdefault(cid, name)

    # Build feature_id → ID-attribute map (we re-mint synthetic IDs
    # that round-trip via the same name_to_feature_id table on read-back).
    fids = annotation.features.column("feature_id").to_pylist()
    fid_to_id_attr = {fid: f"f{fid}" for fid in fids}

    # Pull feature columns once.
    feats = annotation.features
    contig_ids = feats.column("contig_id").to_pylist()
    starts = feats.column("start").to_pylist()
    ends = feats.column("end").to_pylist()
    strands = feats.column("strand").to_pylist()
    types = feats.column("type").to_pylist()
    names = feats.column("name").to_pylist()
    parents = feats.column("parent_id").to_pylist()
    sources = feats.column("source").to_pylist()
    scores = feats.column("score").to_pylist()
    phases = feats.column("phase").to_pylist()
    attrs_json_col = feats.column("attributes_json").to_pylist()

    # Sort by (contig_id, start, end) to match canonical GFF3 layout.
    order = sorted(range(len(fids)), key=lambda i: (contig_ids[i], starts[i], ends[i]))

    lines: list[str] = ["##gff-version 3"]

    # ##sequence-region directives — prefer GenomeReference, else
    # round-trip recorded ones from metadata, else skip.
    if genome is not None:
        for cid, length in zip(
            genome.contigs.column("contig_id").to_pylist(),
            genome.contigs.column("length").to_pylist(),
            strict=True,
        ):
            name = cid_to_name.get(cid, f"contig{cid}")
            lines.append(f"##sequence-region {name} 1 {length}")
    else:
        for region in annotation.metadata_extras.get("sequence_regions", []) or []:
            lines.append(
                f"##sequence-region {region['name']} {region['start']} {region['end']}"
            )

    for i in order:
        seqid = cid_to_name.get(contig_ids[i], f"contig{contig_ids[i]}")
        source_str = sources[i] if sources[i] is not None else "."
        ftype = types[i]
        start_out = starts[i] + 1  # 0-based half-open → 1-based inclusive
        end_out = ends[i]
        score_str = "." if scores[i] is None else _fmt_float(scores[i])
        strand_str = strands[i] if strands[i] in ("+", "-") else "."
        phase_str = "." if phases[i] is None else str(phases[i])

        attr_pairs: list[tuple[str, str]] = []
        attr_pairs.append(("ID", fid_to_id_attr[fids[i]]))
        if names[i] is not None:
            attr_pairs.append(("Name", names[i]))
        if parents[i] is not None:
            attr_pairs.append(("Parent", fid_to_id_attr.get(parents[i], f"f{parents[i]}")))
        if attrs_json_col[i]:
            try:
                payload = json.loads(attrs_json_col[i])
            except json.JSONDecodeError:
                payload = {}
            for k, v in payload.items():
                if isinstance(v, list):
                    attr_pairs.append((k, ",".join(quote(x, safe=":") for x in v)))
                else:
                    attr_pairs.append((k, quote(str(v), safe=":")))

        attr_col = ";".join(f"{quote(k, safe=':')}={v}" for k, v in attr_pairs)

        lines.append(
            "\t".join(
                [
                    seqid,
                    source_str,
                    ftype,
                    str(start_out),
                    str(end_out),
                    score_str,
                    strand_str,
                    phase_str,
                    attr_col,
                ]
            )
        )

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_float(value: float) -> str:
    """Compact-ish float formatting that doesn't lose precision for
    typical GFF3 score values."""
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


# ──────────────────────────────────────────────────────────────────────
# RawReader subclasses (still stubbed pending Phase 2 read-ingest path)
# ──────────────────────────────────────────────────────────────────────


@register_reader
class GffReader(RawReader):
    """Decodes ``.gff`` / ``.gff3`` / ``.gtf`` → FEATURE_TABLE.

    The functional GFF3 path is via :func:`read_gff3` returning an
    ``Annotation``. This RawReader subclass is a different surface —
    it's the read-ingest pathway that returns a generic ``ReadResult``
    for the registry; pending Phase 2.
    """

    suffixes: ClassVar[tuple[str, ...]] = (".gff", ".gff3", ".gtf")
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        raise NotImplementedError(f"GffReader.read pending {_PHASE}")


@register_reader
class GffGzReader(RawReader):
    """Gzipped GFF / GTF — common in genome assembly pipelines."""

    suffixes: ClassVar[tuple[str, ...]] = (".gff.gz", ".gff3.gz", ".gtf.gz")
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        raise NotImplementedError(f"GffGzReader.read pending {_PHASE}")


# Iterator alias kept for downstream tests / callers that want streaming.
def iter_gff3_lines(path: str | Path) -> Iterator[str]:
    """Yield non-comment, non-blank GFF3 data lines (after the
    optional ``##FASTA`` boundary, lines stop)."""
    p = Path(path)
    with _open_text(p) as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line or line.startswith("#"):
                if line.startswith("##FASTA"):
                    return
                continue
            yield line


__all__ = [
    "read_gff3",
    "write_gff3",
    "iter_gff3_lines",
    "GffReader",
    "GffGzReader",
]
