"""Annotate MSP peaks against a theoretical fragment ladder.

For each entry the driver hands us:

  * a parsed ``Peptidoform``,
  * the precursor charge,
  * a list of ``(observed_mz, intensity, raw_annotation)`` peak tuples.

We compute the full ``fragment_ladder`` ``(A..Z, charges 1..N,
losses=[H2O,NH3,HPO3,H3PO4])`` once per spectrum, then resolve each
peak's annotation via ``parse_mzpaf`` to the matching ladder row by
``(ion_type, position, charge, loss_id)``. The emitted fragment row
carries ``mz_theoretical`` from the ladder (never the observed m/z)
and ``intensity_predicted`` from the file (raw pass-through unless
the caller asks for normalisation).

When the annotation can't be parsed or doesn't map to any ladder row,
behaviour is governed by ``keep_unparseable_annotations``:

  * True (default): emit a row with NULL structured columns and the
    raw annotation string. Round-trip lossless; downstream consumers
    filter on ``ion_type IS NULL``.
  * False: drop the peak; bump the dropped-peaks counter.

Multiple peaks → same theoretical fragment (chimeric / co-elution):
first-wins-by-intensity. Sum-intensity would double-count without a
deconvolution prior the library tier doesn't carry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from constellation.core.sequence.proforma import Peptidoform
from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.massspec.annotation import parse_mzpaf
from constellation.massspec.annotation.mzpaf import (
    MzPAFError,
    NeutralLoss,
)
from constellation.massspec.annotation._grammar import IonClass
from constellation.massspec.peptide.ions import IonType, fragment_ladder
from constellation.massspec.peptide.neutral_losses import LOSS_REGISTRY


_DEFAULT_LOSSES = ("H2O", "NH3", "HPO3", "H3PO4")

# letter (lowercase per mzPAF) → IonType
_LETTER_TO_ION_TYPE: dict[str, IonType] = {
    "a": IonType.A,
    "b": IonType.B,
    "c": IonType.C,
    "x": IonType.X,
    "y": IonType.Y,
    "z": IonType.Z,
}

# Canonical IonType ordering for ladder construction — matches the
# IonType enum's IntEnum values so cache keys are stable across runs.
_ALL_ION_TYPES: tuple[IonType, ...] = (
    IonType.A,
    IonType.B,
    IonType.C,
    IonType.X,
    IonType.Y,
    IonType.Z,
)

# Per project convention (encyclopedia adapter + fragment_ladder),
# the ``position`` column in LIBRARY_FRAGMENT_TABLE is the **0-indexed
# peptide-bond position from the N-terminus**, NOT the 1-indexed
# Roepstorff fragment number that mzPAF uses. For a peptide of length
# L:
#   * b_N / a_N / c_N (mzPAF, N=1..L-1) ↔ bond_pos = N - 1
#   * y_N / x_N / z_N (mzPAF, N=1..L-1) ↔ bond_pos = L - 1 - N
_C_SIDE_TYPES = frozenset({IonType.X, IonType.Y, IonType.Z})

_LOSS_IDS = frozenset(LOSS_REGISTRY.ids())


@dataclass(slots=True)
class AnnotateCounters:
    """Per-spectrum counters surfaced back to the driver."""

    unparseable_annotations: int = 0
    dropped_peaks: int = 0


def _losses_to_loss_id(losses: tuple[NeutralLoss, ...]) -> str | None:
    """Map an mzPAF NeutralLoss tuple to a LOSS_REGISTRY id.

    Returns the loss id (e.g. ``"H2O"``) for a single formula-form
    loss that matches a registered loss id. Returns ``None`` for the
    empty tuple (no-loss). Returns a sentinel ``""`` for anything
    else (multi-loss, named loss, unknown formula) — the caller
    treats it as unparseable.
    """
    if not losses:
        return None
    if len(losses) > 1:
        return ""
    (one,) = losses
    if one.sign != -1 or one.is_named:
        return ""
    if one.token in _LOSS_IDS:
        return one.token
    return ""


def _build_ladder_index(
    peptidoform: Peptidoform,
    max_fragment_charge: int,
    ion_types: tuple[IonType, ...],
    loss_ids: tuple[str, ...],
    *,
    vocab: ModVocab,
    cache: dict | None = None,
    cache_key: tuple | None = None,
) -> dict[tuple[int, int, int, str | None], float]:
    """Compute the theoretical ladder and index by lookup key.

    Only the ``ion_types`` and ``loss_ids`` actually referenced by the
    spectrum's peak annotations are passed through — the biochem
    loss-mask loop inside ``_linear_fragment_ladder`` is O(L × n_types
    × n_losses) and dominates per-entry cost on large MSPs. Restricting
    to actually-used ion types and losses gives a ~6× speedup on
    typical proteomics inputs (b/y + H2O/NH3) compared to the full
    six-letter × four-loss ladder.

    Caches by ``cache_key`` so charge-state repeats of the same modseq
    reuse the ladder.
    """
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]

    max_charge = max(max_fragment_charge, 1)
    table, _ = fragment_ladder(
        peptidoform,
        ion_types=ion_types,
        max_fragment_charge=max_charge,
        neutral_losses=list(loss_ids) if loss_ids else None,
        return_tensor=False,
        vocab=vocab,
    )
    ion_types_col = table.column("ion_type").to_pylist()
    positions = table.column("position").to_pylist()
    charges = table.column("charge").to_pylist()
    loss_id_col = table.column("loss_id").to_pylist()
    mzs = table.column("mz_theoretical").to_pylist()
    index: dict[tuple[int, int, int, str | None], float] = {}
    for it, pos, ch, lid, mz in zip(
        ion_types_col, positions, charges, loss_id_col, mzs
    ):
        index[(int(it), int(pos), int(ch), lid)] = float(mz)

    if cache is not None and cache_key is not None:
        cache[cache_key] = index
    return index


def annotate_msp_peaks(
    peptidoform: Peptidoform,
    precursor_charge: int,
    peaks: list[tuple[float, float, str | None]],
    *,
    vocab: ModVocab = UNIMOD,
    intensity_normalize: Literal["none", "max", "sum"] = "none",
    keep_unparseable_annotations: bool = True,
    ladder_cache: dict | None = None,
    ladder_cache_key: tuple | None = None,
    precomputed_ladder_index: dict[tuple[int, int, int, str | None], float]
    | None = None,
    precomputed_resolved: Sequence[
        tuple[tuple[int, int, int, str | None] | None, str | None]
    ]
    | None = None,
) -> tuple[list[dict], AnnotateCounters]:
    """Resolve each peak's annotation against the theoretical ladder.

    Returns ``(rows, counters)`` where ``rows`` is a list of dicts
    column-aligned with ``LIBRARY_FRAGMENT_TABLE`` (less
    ``precursor_id``, which the driver fills in).
    """
    counters = AnnotateCounters()
    if not peaks:
        return [], counters

    # Optional intensity transform
    intensities = [p[1] for p in peaks]
    if intensity_normalize == "max":
        m = max(intensities) if intensities else 1.0
        if m > 0:
            intensities = [x / m for x in intensities]
    elif intensity_normalize == "sum":
        s = sum(intensities) if intensities else 1.0
        if s > 0:
            intensities = [x / s for x in intensities]
    # "none": leave raw

    seq_len = len(peptidoform.sequence)

    # ── Pre-scan: parse each annotation once; collect required
    #             (ion_types, max_charge, loss_ids) so we can ask
    #             fragment_ladder for the minimal product space. The
    #             chunked MSP reader path supplies these via
    #             ``precomputed_resolved`` so we don't re-parse.
    if precomputed_resolved is not None:
        resolved = list(precomputed_resolved)
        seen_ion_types: set[IonType] = set()
        seen_loss_ids: set[str] = set()
        max_charge_seen = 1
        for key, _ in resolved:
            if key is None:
                continue
            ion_type_int, _pos, ch, lid = key
            try:
                seen_ion_types.add(IonType(ion_type_int))
            except ValueError:
                pass
            if lid is not None:
                seen_loss_ids.add(lid)
            if ch > max_charge_seen:
                max_charge_seen = ch
    else:
        resolved = []
        seen_ion_types = set()
        seen_loss_ids = set()
        max_charge_seen = 1
        for _, _, raw_ann in peaks:
            key, ann_struct = _resolve_annotation(raw_ann, seq_len)
            resolved.append((key, ann_struct))
            if key is None:
                continue
            ion_type_int, _pos, ch, lid = key
            try:
                seen_ion_types.add(IonType(ion_type_int))
            except ValueError:
                pass
            if lid is not None:
                seen_loss_ids.add(lid)
            if ch > max_charge_seen:
                max_charge_seen = ch

    # If a precomputed ladder index was supplied (typical batched
    # pathway driven by ``_read.py``), use it directly. Otherwise
    # build a per-spectrum ladder restricted to the actually-used
    # ion-types / losses.
    if precomputed_ladder_index is not None:
        ladder_index = precomputed_ladder_index
    elif seen_ion_types:
        ladder_ion_types = tuple(
            it for it in _ALL_ION_TYPES if it in seen_ion_types
        )
        ladder_losses = tuple(
            lid for lid in _DEFAULT_LOSSES if lid in seen_loss_ids
        )
        ladder_charge = max(precursor_charge, max_charge_seen, 1)
        cache_key = None
        if ladder_cache_key is not None:
            cache_key = (
                ladder_cache_key,
                ladder_ion_types,
                ladder_losses,
                ladder_charge,
            )
        ladder_index = _build_ladder_index(
            peptidoform,
            ladder_charge,
            ladder_ion_types,
            ladder_losses,
            vocab=vocab,
            cache=ladder_cache,
            cache_key=cache_key,
        )
    else:
        ladder_index = {}

    rows: list[dict] = []
    # Track first-wins-by-intensity across chimeric matches.
    used_keys: dict[tuple[int, int, int, str | None], int] = {}

    for ((mz_obs, _raw_int, raw_ann), intensity, (key, ann_struct)) in zip(
        peaks, intensities, resolved
    ):

        if key is None:
            # Unparseable / unsupported annotation
            counters.unparseable_annotations += 1
            if keep_unparseable_annotations:
                rows.append(
                    {
                        "ion_type": None,
                        "position": None,
                        "charge": None,
                        "loss_id": None,
                        "mz_theoretical": float(mz_obs),
                        "intensity_predicted": float(intensity),
                        "annotation": raw_ann,
                    }
                )
            else:
                counters.dropped_peaks += 1
            continue

        ion_type_code, position, charge, loss_id = key
        ladder_mz = ladder_index.get(key)
        if ladder_mz is None:
            # Annotation parsed but didn't match any theoretical row
            # (off-ladder charge, position out of range, etc.).
            counters.unparseable_annotations += 1
            if keep_unparseable_annotations:
                rows.append(
                    {
                        "ion_type": None,
                        "position": None,
                        "charge": None,
                        "loss_id": None,
                        "mz_theoretical": float(mz_obs),
                        "intensity_predicted": float(intensity),
                        "annotation": raw_ann,
                    }
                )
            else:
                counters.dropped_peaks += 1
            continue

        # First-wins-by-intensity for chimeric duplicates.
        if key in used_keys:
            prior = rows[used_keys[key]]
            if intensity > prior["intensity_predicted"]:
                prior["intensity_predicted"] = float(intensity)
                prior["annotation"] = ann_struct
            counters.dropped_peaks += 1
            continue

        used_keys[key] = len(rows)
        rows.append(
            {
                "ion_type": ion_type_code,
                "position": position,
                "charge": charge,
                "loss_id": loss_id,
                "mz_theoretical": ladder_mz,
                "intensity_predicted": float(intensity),
                "annotation": ann_struct,
            }
        )

    return rows, counters


def _resolve_annotation(
    raw: str | None,
    seq_len: int,
) -> tuple[tuple[int, int, int, str | None] | None, str | None]:
    """Parse one peak annotation string into a ladder lookup key.

    ``seq_len`` is the canonical residue length of the peptidoform; we
    use it to convert mzPAF's 1-indexed Roepstorff position into the
    project's 0-indexed peptide-bond position convention for C-side
    ions (y/x/z).

    Returns ``(key, canonical_string)``:

      * ``key`` is ``(ion_type, bond_position, charge, loss_id)`` when
        the annotation is a supported peptide ion. ``None`` otherwise.
      * ``canonical_string`` is the raw input — kept verbatim for
        ``LIBRARY_FRAGMENT_TABLE.annotation`` because mzPAF round-trip
        would strip useful information like ``mass_error``.
    """
    if raw is None:
        return None, None

    try:
        peak = parse_mzpaf(raw)
    except (MzPAFError, NotImplementedError):
        return None, raw

    if not peak.annotations:
        return None, raw

    ann = peak.annotations[0]  # take the first interpretation
    if ann.ion_class is not IonClass.PEPTIDE:
        return None, raw
    if ann.ion_letter is None or ann.position is None:
        return None, raw

    letter = ann.ion_letter.lower()
    ion_type = _LETTER_TO_ION_TYPE.get(letter)
    if ion_type is None:
        return None, raw

    loss_id = _losses_to_loss_id(ann.losses)
    if loss_id == "":  # sentinel for unsupported loss shape
        return None, raw

    # Roepstorff (1-indexed) → bond-position (0-indexed, from N-terminus)
    roepstorff = int(ann.position)
    if ion_type in _C_SIDE_TYPES:
        bond_pos = seq_len - 1 - roepstorff
    else:
        bond_pos = roepstorff - 1
    if bond_pos < 0 or bond_pos >= seq_len - 1:
        return None, raw

    charge = ann.charge if ann.charge is not None else 1
    key = (int(ion_type), bond_pos, int(charge), loss_id)
    return key, raw


def scan_chunk_requirements(
    chunk_entries: Sequence[tuple[Peptidoform, int, list[tuple[float, float, str | None]]]],
) -> tuple[
    tuple[IonType, ...],
    tuple[str, ...],
    int,
    list[list[tuple[tuple[int, int, int, str | None] | None, str | None]]],
]:
    """Pre-scan a chunk of (peptidoform, precursor_charge, peaks) tuples.

    Returns
    -------
    ion_types : tuple[IonType, ...]
    loss_ids : tuple[str, ...]
    max_charge : int
    resolved : list[list[(key, raw_annotation)]]
        Per-entry parsed peak resolutions. Cached here so the per-
        spectrum row-emit pass doesn't re-parse the mzPAF annotation
        strings — each ``parse_mzpaf`` call is ~10 µs and adds up at
        millions of peaks across a 200k-entry library.

    ``ion_types`` / ``loss_ids`` is the minimal product space the
    chunk's annotations reference, used to drive a single
    ``fragment_ladders_batch`` call per chunk rather than per-spectrum.
    """
    seen_ion_types: set[IonType] = set()
    seen_loss_ids: set[str] = set()
    max_charge = 1
    resolved: list[
        list[tuple[tuple[int, int, int, str | None] | None, str | None]]
    ] = []
    for peptidoform, precursor_charge, peaks in chunk_entries:
        seq_len = len(peptidoform.sequence)
        if precursor_charge > max_charge:
            max_charge = precursor_charge
        per_entry: list[
            tuple[tuple[int, int, int, str | None] | None, str | None]
        ] = []
        for _mz, _intensity, raw_ann in peaks:
            key, ann_struct = _resolve_annotation(raw_ann, seq_len)
            per_entry.append((key, ann_struct))
            if key is None:
                continue
            it_int, _pos, ch, lid = key
            try:
                seen_ion_types.add(IonType(it_int))
            except ValueError:
                pass
            if lid is not None:
                seen_loss_ids.add(lid)
            if ch > max_charge:
                max_charge = ch
        resolved.append(per_entry)

    ion_types = tuple(it for it in _ALL_ION_TYPES if it in seen_ion_types)
    loss_ids = tuple(lid for lid in _DEFAULT_LOSSES if lid in seen_loss_ids)
    return ion_types, loss_ids, max(max_charge, 1), resolved


def ladder_table_to_index(
    table: "object",
) -> dict[tuple[int, int, int, str | None], float]:
    """Convert a ``FragmentIonTable``-shaped Arrow table into the
    ``{(ion_type, position, charge, loss_id): mz}`` lookup dict used
    by ``annotate_msp_peaks``."""
    ion_types_col = table.column("ion_type").to_pylist()
    positions = table.column("position").to_pylist()
    charges = table.column("charge").to_pylist()
    loss_ids = table.column("loss_id").to_pylist()
    mzs = table.column("mz_theoretical").to_pylist()
    index: dict[tuple[int, int, int, str | None], float] = {}
    for it, pos, ch, lid, mz in zip(
        ion_types_col, positions, charges, loss_ids, mzs
    ):
        index[(int(it), int(pos), int(ch), lid)] = float(mz)
    return index


__all__ = [
    "AnnotateCounters",
    "annotate_msp_peaks",
    "scan_chunk_requirements",
    "ladder_table_to_index",
]
