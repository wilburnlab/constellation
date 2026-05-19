"""Streaming driver for the NIST .msp library reader.

The reader is **library-view** only — it expects each MSP entry to
carry a peptide identification (parseable peptidoform from ``Name`` +
``Mods``/``Modstring``/``Pep``/``Fullname``). Raw-spectra entries
(``Name`` is a free-text spectrum label, no peptide) raise
``ValueError``; a separate ``read_msp_spectra`` reader will handle the
raw-spectra view when the matching container ships.

Streaming
=========

Single-pass line iteration via ``path.open("r", encoding="utf-8",
errors="replace")``. No ``read_text()`` — peak memory is bounded by
the in-memory Arrow tables, not the file size.

State machine
=============

EXPECT_NAME       waiting for the next ``Name:`` line
READING_HEADERS   accumulating ``MW``, ``Comment``, ``Num peaks``
READING_PEAKS     consuming N peak lines; on N-th line or first
                  blank, flush the entry into the row accumulators
                  and return to EXPECT_NAME

Outputs
=======

Returns a :class:`Library` whose:

* ``peptides``         deduplicated by ProForma modseq
* ``precursors``       deduplicated by ``(modseq, charge)``;
                       ``precursor_mz`` and ``rt_predicted`` are
                       theoretical / predicted (computed from the
                       peptidoform / iRT field).
* ``fragments``        per-precursor annotated peaks with
                       ``mz_theoretical`` from ``fragment_ladder``
                       and ``intensity_predicted`` from the file
                       (raw pass-through unless ``intensity_normalize``
                       is set).
* ``proteins`` / ``protein_peptide``
                       populated iff the entries carry ``Protein=``;
                       empty otherwise.
* ``metadata_extras``  carries library-wide counters plus the
                       per-precursor Comment side-table at key
                       ``x.msp.precursor_comments``.

Per-spectrum NIST/MaxQuant/MSFragger Comment keys that don't map to a
Library column (e.g. ``Inst``, ``Collision_energy``, ``NCE``,
``Scan``, ``RTInSeconds``) are accumulated lossless into the
side-table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import pyarrow as pa

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import (
    Peptidoform,
    format_proforma,
    parse_proforma,
)
from constellation.massspec.library.library import Library
from constellation.massspec.library.schemas import (
    LIBRARY_FRAGMENT_TABLE,
    PEPTIDE_TABLE,
    PRECURSOR_TABLE,
    PROTEIN_PEPTIDE_EDGE,
    PROTEIN_TABLE,
)
from constellation.massspec.peptide.mz import precursor_mz

from constellation.massspec.peptide.ions import (
    fragment_ladder_indices_batch,
)

from constellation.massspec.io.msp._annotate import (
    AnnotateCounters,
    annotate_msp_peaks,
    scan_chunk_requirements,
)
from constellation.massspec.io.msp._comment import tokenize_comment
from constellation.massspec.io.msp._mods import (
    MspModResolutionError,
    build_peptidoform,
    parse_mods_field,
)


# Comment keys that are already represented by canonical Library
# columns or that we consume directly to drive parsing — these do NOT
# get a column in the per-precursor side-table.
_LIBRARY_REDUNDANT_KEYS = frozenset(
    {
        "iRT",
        "Charge",
        "Parent",
        "Mods",
        "Modstring",
        "Fullname",
        "Pep",
        "Name",
        "Protein",
    }
)


@dataclass(slots=True)
class _Entry:
    """One in-flight MSP entry being assembled by the state machine."""

    name: str | None = None
    mw: float | None = None
    comment: dict[str, str | bool] = field(default_factory=dict)
    n_peaks_declared: int | None = None
    peaks: list[tuple[float, float, str | None]] = field(default_factory=list)
    source_line: int = 0  # for error messages


def _iter_entries(path: Path) -> Iterator[_Entry]:
    """Yield ``_Entry`` records, one per MSP entry, by streaming the file."""

    entry = _Entry()
    state = "EXPECT_NAME"

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                # Blank line — flush if we have peaks
                if state == "READING_PEAKS" and entry.peaks:
                    yield entry
                    entry = _Entry()
                    state = "EXPECT_NAME"
                continue

            if state == "EXPECT_NAME":
                if stripped.lower().startswith("name:"):
                    entry = _Entry()
                    entry.source_line = line_no
                    entry.name = stripped[len("Name:"):].strip()
                    state = "READING_HEADERS"
                # silently skip any other lines before the first Name:
                continue

            if state == "READING_HEADERS":
                low = stripped.lower()
                if low.startswith("mw:"):
                    val = stripped.split(":", 1)[1].strip()
                    try:
                        entry.mw = float(val)
                    except ValueError:
                        entry.mw = None
                    continue
                if low.startswith("comment:"):
                    entry.comment = tokenize_comment(stripped)
                    continue
                if low.startswith("num peaks:"):
                    val = stripped.split(":", 1)[1].strip()
                    try:
                        entry.n_peaks_declared = int(val)
                    except ValueError as e:
                        raise ValueError(
                            f"line {line_no}: malformed Num peaks {val!r}"
                        ) from e
                    state = "READING_PEAKS"
                    continue
                # Other header keys (e.g. Charge:, PrecursorMZ:) — stash
                # them into comment for downstream consumers.
                if ":" in stripped:
                    k, _, v = stripped.partition(":")
                    entry.comment[k.strip()] = v.strip()
                continue

            if state == "READING_PEAKS":
                if stripped.lower().startswith("name:"):
                    # New entry without a blank-line terminator (some
                    # files do this) — flush current and restart.
                    if entry.peaks:
                        yield entry
                    entry = _Entry()
                    entry.source_line = line_no
                    entry.name = stripped[len("Name:"):].strip()
                    state = "READING_HEADERS"
                    continue
                peak = _parse_peak_line(stripped)
                if peak is not None:
                    entry.peaks.append(peak)
                if (
                    entry.n_peaks_declared is not None
                    and len(entry.peaks) >= entry.n_peaks_declared
                ):
                    yield entry
                    entry = _Entry()
                    state = "EXPECT_NAME"
                continue

        # EOF flush
        if state == "READING_PEAKS" and entry.peaks:
            yield entry


def _parse_peak_line(
    line: str,
) -> tuple[float, float, str | None] | None:
    """Parse one peak line into ``(mz, intensity, annotation?)``.

    Peak lines are whitespace- or tab-separated::

        129.1017   20123   "y1-H2O/5.34ppm"

    The third column is optional and quoted. Returns ``None`` when the
    line doesn't parse as a peak (e.g. stray text between entries).
    """
    parts = line.split(None, 2)
    if len(parts) < 2:
        return None
    try:
        mz = float(parts[0])
        intensity = float(parts[1])
    except ValueError:
        return None
    annotation: str | None = None
    if len(parts) == 3:
        ann = parts[2].strip()
        # Strip surrounding double quotes if present
        if len(ann) >= 2 and ann.startswith('"') and ann.endswith('"'):
            ann = ann[1:-1]
        # Some files append a per-peak comment after the closing quote
        # (e.g. ``"y1/5.34ppm" 1/1``). We keep only the quoted body.
        annotation = ann or None
    return mz, intensity, annotation


def _resolve_peptidoform(
    entry: _Entry, *, vocab: ModVocab
) -> tuple[Peptidoform, str, int]:
    """Derive a ``(Peptidoform, canonical_sequence, charge)`` triple
    from an MSP entry.

    Order of precedence per the project decision:

      1. ``Name`` parsed as ``<SEQUENCE>/<CHARGE>`` + ``Mods=`` field
         (the canonical NIST proteomics shape).
      2. ``Comment.Fullname`` or ``Comment.Pep`` if present (ProForma
         2.0-compatible or NIST modseq).
      3. ``Comment.Modstring`` (encyclopedia-style modseq, last
         resort — used only when 1 and 2 fail or when ``Mods=`` can't
         be resolved against the vocab).

    Raises ``ValueError`` when none of the above yields a parseable
    peptidoform — that's the signal this entry is raw spectra rather
    than a library entry.
    """
    name = entry.name or ""
    sequence: str | None = None
    charge: int | None = None
    if "/" in name:
        seq_part, charge_part = name.rsplit("/", 1)
        if seq_part.isalpha() and charge_part.isdigit():
            sequence = seq_part
            charge = int(charge_part)

    comment = entry.comment
    if charge is None and "Charge" in comment:
        try:
            charge = int(str(comment["Charge"]))
        except (TypeError, ValueError):
            pass

    if sequence is not None and charge is not None:
        mods_raw = comment.get("Mods")
        if mods_raw is not None:
            try:
                mods = parse_mods_field(str(mods_raw))
                pep = build_peptidoform(sequence, charge, mods, vocab=vocab)
                return pep, sequence, charge
            except MspModResolutionError:
                pass  # fall through to Modstring fallback
        else:
            # Mods absent — treat as unmodified.
            pep = build_peptidoform(sequence, charge, [], vocab=vocab)
            return pep, sequence, charge

    # Fullname / Pep — ProForma-compatible modseq with optional charge
    for key in ("Fullname", "Pep"):
        val = comment.get(key)
        if isinstance(val, str) and val:
            try:
                pep = parse_proforma(val)
                if isinstance(pep, Peptidoform):
                    eff_charge = pep.charge or charge
                    if eff_charge is None:
                        continue
                    # Ensure charge is reflected on the peptidoform
                    if pep.charge is None:
                        from dataclasses import replace

                        pep = replace(pep, charge=eff_charge)
                    return pep, pep.sequence, eff_charge
            except Exception:
                pass

    # Modstring fallback — encyclopedia-style modseq + trailing charge
    modstring = comment.get("Modstring")
    if isinstance(modstring, str) and "//" in modstring:
        body, _, tail = modstring.rpartition("//")
        try:
            ms_charge = int(tail) if tail.isdigit() else charge
        except ValueError:
            ms_charge = charge
        if ms_charge is not None:
            # Modstring sometimes carries inline (+mass) tokens; for
            # the v1 reader we only honour Modstring when it is a
            # plain residue sequence matching Name and Mods=0.
            if body.isalpha():
                try:
                    pep = build_peptidoform(body, ms_charge, [], vocab=vocab)
                    return pep, body, ms_charge
                except Exception:
                    pass

    raise ValueError(
        f"MSP entry at line {entry.source_line} (Name={name!r}) carries no "
        f"parseable peptide identification. Library reader requires peptide "
        f"ID + annotated fragments; raw-spectra MSPs need read_msp_spectra "
        f"(not yet implemented)."
    )


def _extract_irt(comment: dict[str, str | bool]) -> float:
    """Pull the ``iRT`` value out of the Comment dict; map ``NA`` to
    the project's ``-1.0`` not-set sentinel."""
    raw = comment.get("iRT")
    if raw is None or isinstance(raw, bool):
        return -1.0
    raw_str = str(raw).strip()
    if not raw_str or raw_str.upper() == "NA":
        return -1.0
    try:
        return float(raw_str)
    except ValueError:
        return -1.0


def _parse_proteins(comment: dict[str, str | bool]) -> list[str]:
    """Extract the protein accession list from a ``Protein=`` field.

    Supports semicolon- and pipe-separated multi-protein values.
    Returns ``[]`` when absent. Strips surrounding quotes if present.
    """
    raw = comment.get("Protein")
    if raw is None or isinstance(raw, bool):
        return []
    raw_str = str(raw).strip()
    if not raw_str:
        return []
    # Either ; or | as separator
    if ";" in raw_str:
        parts = raw_str.split(";")
    elif "|" in raw_str:
        parts = raw_str.split("|")
    else:
        parts = [raw_str]
    return [p.strip() for p in parts if p.strip()]


def _build_side_table(
    per_precursor_comments: list[tuple[int, dict[str, str | bool]]],
) -> pa.Table:
    """Build the per-precursor Comment side-table.

    ``per_precursor_comments`` is a list of ``(precursor_id,
    filtered_comment_dict)`` tuples in entry order. Keys appearing in
    any entry become string columns; rows lacking a given key are
    NULL. ``precursor_id`` is the PK.
    """
    if not per_precursor_comments:
        return pa.table({"precursor_id": pa.array([], type=pa.int64())})

    # Union of keys across all rows
    all_keys: list[str] = []
    seen: set[str] = set()
    for _, d in per_precursor_comments:
        for k in d.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    cols: dict[str, list[Any]] = {"precursor_id": []}
    for k in all_keys:
        cols[k] = []
    for pid, d in per_precursor_comments:
        cols["precursor_id"].append(pid)
        for k in all_keys:
            v = d.get(k)
            if v is None or v is True or v is False:
                cols[k].append(None if v is None else ("" if v is False else k))
            else:
                cols[k].append(str(v))

    arrays = {"precursor_id": pa.array(cols["precursor_id"], type=pa.int64())}
    for k in all_keys:
        arrays[k] = pa.array(cols[k], type=pa.string())
    return pa.table(arrays)


def read_msp_library(
    path: Path | str,
    *,
    enabled_modifications: ModVocab = UNIMOD,
    intensity_normalize: Literal["none", "max", "sum"] = "none",
    keep_unparseable_annotations: bool = True,
    fragment_tolerance_ppm: float = 20.0,  # noqa: ARG001 — reserved for fallback matcher
    chunk_size: int = 1024,
) -> Library:
    """Read a NIST .msp file as a library-view ``Library``.

    See module docstring for the data flow. Raises ``ValueError`` on
    raw-style MSPs (entries with no parseable peptide identification).
    """
    src = Path(path).expanduser().resolve()

    proteins_rows: list[dict[str, Any]] = []
    accession_to_id: dict[str, int] = {}

    peptides_rows: list[dict[str, Any]] = []
    modseq_to_id: dict[str, int] = {}

    precursors_rows: list[dict[str, Any]] = []
    pkey_to_id: dict[tuple[str, int], int] = {}

    # Fragment rows emit per-chunk as ``pa.Table`` instead of being held
    # in a single ``list[dict]`` until EOF — the Python-dict footprint
    # (~600 B/row × tens of millions of rows on a full ProteomeTools-class
    # library) is the dominant accumulator and gets freed at chunk end
    # by going through Arrow's columnar buffers.
    fragment_chunk_tables: list[pa.Table] = []
    pp_pairs: set[tuple[str, str]] = set()
    per_precursor_comments: list[tuple[int, dict[str, str | bool]]] = []

    n_entries = 0
    counters = AnnotateCounters()
    unresolved_mods: dict[str, int] = {}

    # Buffered chunk for batched fragment-ladder construction. Each
    # element is ``(peptidoform, precursor_charge, peaks, precursor_id)``.
    chunk: list[
        tuple[Peptidoform, int, list[tuple[float, float, str | None]], int]
    ] = []

    def _flush_chunk() -> None:
        if not chunk:
            return
        triples = [(p, c, peaks) for p, c, peaks, _ in chunk]
        (
            ion_types,
            loss_ids,
            max_charge,
            resolved_per_entry,
        ) = scan_chunk_requirements(triples)
        if ion_types:
            indices = fragment_ladder_indices_batch(
                [p for p, _, _, _ in chunk],
                ion_types=ion_types,
                max_fragment_charge=max_charge,
                neutral_losses=list(loss_ids) if loss_ids else None,
                vocab=enabled_modifications,
            )
        else:
            indices = [{} for _ in chunk]
        chunk_fragments_rows: list[dict[str, Any]] = []
        for (
            (pep, charge, peaks, precursor_id),
            ladder_index,
            resolved,
        ) in zip(chunk, indices, resolved_per_entry):
            frag_rows, entry_counters = annotate_msp_peaks(
                pep,
                charge,
                peaks,
                vocab=enabled_modifications,
                intensity_normalize=intensity_normalize,
                keep_unparseable_annotations=keep_unparseable_annotations,
                precomputed_ladder_index=ladder_index,
                precomputed_resolved=resolved,
            )
            counters.unparseable_annotations += entry_counters.unparseable_annotations
            counters.dropped_peaks += entry_counters.dropped_peaks
            for row in frag_rows:
                row["precursor_id"] = precursor_id
                chunk_fragments_rows.append(row)
        if chunk_fragments_rows:
            fragment_chunk_tables.append(
                _to_table(chunk_fragments_rows, LIBRARY_FRAGMENT_TABLE)
            )
        chunk.clear()

    for entry in _iter_entries(src):
        n_entries += 1
        try:
            pep, sequence, charge = _resolve_peptidoform(
                entry, vocab=enabled_modifications
            )
        except MspModResolutionError as e:
            for name in getattr(e, "args", (None,))[0] or ():
                unresolved_mods[str(name)] = unresolved_mods.get(str(name), 0) + 1
            # Skip the entry; the reader treats unresolvable mods as
            # raw-style for the purpose of this PR.
            raise ValueError(
                f"MSP entry at line {entry.source_line}: unresolved "
                f"modifications {e!s}"
            ) from e

        from dataclasses import replace

        modseq = format_proforma(replace(pep, charge=None))

        if modseq not in modseq_to_id:
            pid = len(peptides_rows)
            modseq_to_id[modseq] = pid
            peptides_rows.append(
                {
                    "peptide_id": pid,
                    "sequence": sequence,
                    "modified_sequence": modseq,
                }
            )

        pkey = (modseq, charge)
        if pkey not in pkey_to_id:
            cid = len(precursors_rows)
            pkey_to_id[pkey] = cid
            mz = precursor_mz(pep, vocab=enabled_modifications)
            precursors_rows.append(
                {
                    "precursor_id": cid,
                    "peptide_id": modseq_to_id[modseq],
                    "charge": charge,
                    "precursor_mz": mz,
                    "rt_predicted": _extract_irt(entry.comment),
                    "ccs_predicted": -1.0,
                }
            )

        precursor_id = pkey_to_id[pkey]

        # Buffer the entry; fragment-row emission happens once the
        # chunk fills up (or at EOF) so the ladder construction can
        # run in batched form.
        chunk.append((pep, charge, entry.peaks, precursor_id))
        if len(chunk) >= chunk_size:
            _flush_chunk()

        # Protein bindings
        for accession in _parse_proteins(entry.comment):
            if accession not in accession_to_id:
                accession_to_id[accession] = len(proteins_rows)
                proteins_rows.append(
                    {
                        "protein_id": accession_to_id[accession],
                        "accession": accession,
                        "sequence": None,
                        "description": None,
                    }
                )
            pp_pairs.add((accession, modseq))

        # Side-table row — keep only non-redundant keys
        side = {
            k: v
            for k, v in entry.comment.items()
            if k not in _LIBRARY_REDUNDANT_KEYS
        }
        if side:
            per_precursor_comments.append((precursor_id, side))

    _flush_chunk()

    pp_rows = [
        {
            "protein_id": accession_to_id[acc],
            "peptide_id": modseq_to_id[modseq],
        }
        for acc, modseq in sorted(pp_pairs)
    ]

    side_table = _build_side_table(per_precursor_comments)

    metadata_extras: dict[str, Any] = {
        "x.msp.source_path": str(src),
        "x.msp.format_name": "nist.msp",
        "x.msp.n_entries": n_entries,
        "x.msp.unparseable_annotations": counters.unparseable_annotations,
        "x.msp.dropped_peaks": counters.dropped_peaks,
    }
    if unresolved_mods:
        metadata_extras["x.msp.unresolved_mods"] = unresolved_mods
    if side_table.num_rows > 0:
        metadata_extras["x.msp.precursor_comments"] = side_table

    return Library(
        proteins=_to_table(proteins_rows, PROTEIN_TABLE),
        peptides=_to_table(peptides_rows, PEPTIDE_TABLE),
        precursors=_to_table(precursors_rows, PRECURSOR_TABLE),
        fragments=_concat_fragments(fragment_chunk_tables),
        protein_peptide=_to_table(pp_rows, PROTEIN_PEPTIDE_EDGE),
        metadata_extras=metadata_extras,
    )


def _to_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def _concat_fragments(tables: list[pa.Table]) -> pa.Table:
    if not tables:
        return LIBRARY_FRAGMENT_TABLE.empty_table()
    return pa.concat_tables(tables)


__all__ = ["read_msp_library"]
