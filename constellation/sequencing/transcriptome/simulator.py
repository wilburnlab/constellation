"""Synthetic nanopore-style cDNA read simulator.

Drives off :class:`LibraryDesign`: one :class:`ReadSpec` is mapped to a
sequence by walking the design's ``layout`` tuple of ``Segment``s. The
generator emits a SAM file (round-trips through ``SamReader.iter_batches``
with no new ingest code) plus a parallel ground-truth Arrow table
indexed by ``read_id``.

Use cases:

  - Stress-test :class:`HardThresholdScorer` with reads whose mutation
    count per segment is known exactly.
  - Provide a regression fixture for the future ``ProbabilisticScorer``:
    re-emit the same spec set under different mutation rates to
    calibrate joint-likelihood scoring against ground truth.

Determinism: every entry point takes (or seeds) a numpy ``Generator`` so
identical seeds produce byte-identical output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from constellation.sequencing.transcriptome.adapters import (
    AdapterSlot,
    BarcodeSlot,
    LibraryDesign,
    PolyASlot,
    Segment,
    TranscriptSlot,
)
from constellation.sequencing.transcriptome.classify import ReadStatus


# ──────────────────────────────────────────────────────────────────────
# Ground-truth schema
# ──────────────────────────────────────────────────────────────────────


GROUND_TRUTH_TABLE: pa.Schema = pa.schema([
    pa.field("read_id", pa.string(), nullable=False),
    pa.field("expected_status", pa.string(), nullable=False),
    pa.field("orientation", pa.string(), nullable=False),
    pa.field("transcript_length", pa.int32(), nullable=False),
    pa.field("transcript_id", pa.string(), nullable=False),
    pa.field("polyA_length", pa.int32(), nullable=False),
    pa.field("polyA_artifact", pa.string(), nullable=False),
    pa.field("expected_barcode_index", pa.int32(), nullable=True),
    pa.field("expected_barcode_name", pa.string(), nullable=True),
    pa.field("ssp_edits", pa.int32(), nullable=False),
    pa.field("primer3_edits", pa.int32(), nullable=False),
    pa.field("barcode_edits", pa.int32(), nullable=False),
    pa.field("include_ssp", pa.bool_(), nullable=False),
    pa.field("include_primer3", pa.bool_(), nullable=False),
    pa.field("include_polyA", pa.bool_(), nullable=False),
    pa.field("include_barcode", pa.bool_(), nullable=False),
    pa.field("is_complex", pa.bool_(), nullable=False),
    pa.field("is_fragment_expected", pa.bool_(), nullable=False),
])


# ──────────────────────────────────────────────────────────────────────
# Data model — one spec, one read
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReadSpec:
    """Recipe for one synthetic read.

    The ``include_*`` flags drive whether each segment is emitted at
    all (Missing-Barcode, 5'-Only, 3'-Only correspond to selective
    omissions). Edit counts say *how many* mutations to inject per
    segment after assembly.

    For ``is_complex`` specs the assembler builds a palindromic
    structure (forward + RC of forward) so both orientations classify
    as non-Unknown but neither is strictly Complete — the strand
    resolution rule then upgrades to Complex.
    """

    read_id: str
    expected_status: ReadStatus
    orientation: str  # "+" or "-"
    transcript_length: int
    transcript_id: str
    polyA_length: int
    polyA_artifact: str  # "clean" | "substitution" | "expanded" | "deleted"
    barcode_index: int | None
    ssp_edits: int = 0
    primer3_edits: int = 0
    barcode_edits: int = 0
    include_ssp: bool = True
    include_primer3: bool = True
    include_polyA: bool = True
    include_barcode: bool = True
    is_complex: bool = False
    is_fragment_expected: bool = False


# ──────────────────────────────────────────────────────────────────────
# Mutation primitive
# ──────────────────────────────────────────────────────────────────────


_BASES = ("A", "C", "G", "T")


def _random_base(rng: np.random.Generator) -> str:
    return _BASES[int(rng.integers(0, 4))]


def _random_other_base(b: str, rng: np.random.Generator) -> str:
    others = tuple(x for x in _BASES if x.upper() != b.upper())
    return others[int(rng.integers(0, len(others)))]


def mutate(
    seq: str,
    n_edits: int,
    *,
    kinds: tuple[str, ...] = ("sub", "ins", "del"),
    rng: np.random.Generator,
) -> str:
    """Apply ``n_edits`` operations sampled from ``kinds``.

    Each operation is independently sampled (kind, position).
    Insertions and deletions shift downstream positions; total operation
    count equals ``n_edits``. ``n_edits=0`` returns ``seq`` unchanged.
    """
    if n_edits <= 0:
        return seq
    s = list(seq)
    for _ in range(n_edits):
        if not s:
            break
        kind = kinds[int(rng.integers(0, len(kinds)))]
        pos = int(rng.integers(0, len(s)))
        if kind == "sub":
            s[pos] = _random_other_base(s[pos], rng)
        elif kind == "ins":
            s.insert(pos, _random_base(rng))
        elif kind == "del":
            s.pop(pos)
    return "".join(s)


# ──────────────────────────────────────────────────────────────────────
# Sequence assembly
# ──────────────────────────────────────────────────────────────────────


_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def _random_transcript(length: int, rng: np.random.Generator) -> str:
    """Random ATCG sequence of the requested length.

    Bias against accidental polyA / TSO substrings is not enforced here;
    on a 200+ nt transcript the chance of an accidental 15-A run is
    negligible, and the spec generator caps polyA_length explicitly.
    """
    return "".join(_BASES[int(b)] for b in rng.integers(0, 4, size=length))


def _polyA_with_artifact(
    length: int,
    artifact: str,
    rng: np.random.Generator,
) -> str:
    """Emit a polyA run with the requested artifact applied.

    Artifact catalogue:

      ``clean``         exactly ``length`` A's
      ``substitution``  one base substituted in the middle (anchors at
                        the ends remain "AAAA" so the polyA scorer's
                        edge_distance=1 merge still finds the run)
      ``expanded``      ``length + 8`` A's (homopolymer over-call sim)
      ``deleted``       ``length - 6`` A's (homopolymer under-call sim;
                        clamped to 0 minimum)
    """
    if artifact == "clean":
        return "A" * length
    if artifact == "substitution":
        s = list("A" * length)
        if length >= 5:
            mid = length // 2
            s[mid] = _random_other_base("A", rng)
        return "".join(s)
    if artifact == "expanded":
        return "A" * (length + 8)
    if artifact == "deleted":
        return "A" * max(0, length - 6)
    raise ValueError(f"unknown polyA_artifact: {artifact!r}")


def _classify_adapter_slot(slot: AdapterSlot) -> str:
    """Return ``'5p'`` or ``'3p'`` based on the slot's first adapter."""
    if not slot.adapters:
        return "unknown"
    return slot.adapters[0].kind


def _emit_segment(
    segment: Segment,
    spec: ReadSpec,
    *,
    rng: np.random.Generator,
) -> str:
    """Emit the bases for one segment of one spec, applying mutation."""
    kind = segment.kind
    if kind == "adapter":
        assert isinstance(segment, AdapterSlot)
        which = _classify_adapter_slot(segment)
        if which == "5p":
            if not spec.include_ssp:
                return ""
            target = segment.adapters[0].sequence
            return mutate(target, spec.ssp_edits, rng=rng)
        if which == "3p":
            if not spec.include_primer3:
                return ""
            target = segment.adapters[0].sequence
            return mutate(target, spec.primer3_edits, rng=rng)
        return ""
    if kind == "transcript":
        return _random_transcript(spec.transcript_length, rng)
    if kind == "polyA":
        if not spec.include_polyA:
            return ""
        return _polyA_with_artifact(spec.polyA_length, spec.polyA_artifact, rng)
    if kind == "barcode":
        if not spec.include_barcode or spec.barcode_index is None:
            return ""
        assert isinstance(segment, BarcodeSlot)
        bc = segment.barcodes[spec.barcode_index]
        # Read-as-sequenced carries the RC of the barcode (panel docstring).
        rc = _reverse_complement(bc.sequence)
        return mutate(rc, spec.barcode_edits, rng=rng)
    # UMISlot / TranscriptSlot edge cases not reached for cdna_wilburn_v1.
    return ""


def assemble_sequence(
    spec: ReadSpec,
    design: LibraryDesign,
    *,
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Build the (sequence, quality) pair for one spec.

    The sense-strand sequence is emitted by walking ``design.layout``
    5'→3' and dispatching per segment. ``simulate_panel`` applies the
    final orientation flip (reverse-complement) when ``spec.orientation
    == '-'``.

    For ``is_complex=True`` specs the structure is doubled in palindromic
    form (forward + RC(forward)) so both orientations classify as
    non-Unknown — falls into the strand-resolution Complex branch.
    """
    parts = [_emit_segment(seg, spec, rng=rng) for seg in design.layout]
    forward = "".join(parts)
    if spec.is_complex:
        # Palindromic complex: forward + RC(forward). The polyA in the
        # forward half is mirrored by a polyT (RC of polyA) in the second
        # half — but the read in the OPPOSITE orientation flips them so
        # that orientation also sees a polyA. Net: both orientations
        # find the SSP + polyA + primer3 anchors from the leading half
        # of their respective views, classify accordingly, and the
        # strand-resolution rule yields Complex.
        forward = forward + _reverse_complement(forward)
    sequence = forward
    # Q40 Phred everywhere — high enough to not mask anything.
    quality = "I" * len(sequence)
    return sequence, quality


# ──────────────────────────────────────────────────────────────────────
# Stress-test corpus generator
# ──────────────────────────────────────────────────────────────────────


# Barcodes whose RC begins with 'A' — the polyA boundary-ambiguity set
# (panel docstring in designs.py). 0-indexed within the panel.
_LEADING_A_RC_BARCODES = (3, 4, 9, 10)  # BC04, BC05, BC10, BC11

# Default in-range polyA length used when not explicitly varied.
_DEFAULT_POLYA_LEN = 22
_DEFAULT_TRANSCRIPT_LEN = 600
_FRAGMENT_TRANSCRIPT_LEN = 80
_TRANSCRIPT_MIN_LEN = 200  # mirrors panels.py constant


def _make_id(prefix: str, idx: int) -> str:
    return f"{prefix}-{idx:05d}"


def _spec_complete(
    *,
    rid: str,
    barcode_index: int,
    orientation: str = "+",
    transcript_length: int = _DEFAULT_TRANSCRIPT_LEN,
    polyA_length: int = _DEFAULT_POLYA_LEN,
    polyA_artifact: str = "clean",
    ssp_edits: int = 0,
    primer3_edits: int = 0,
    barcode_edits: int = 0,
    is_fragment: bool = False,
) -> ReadSpec:
    return ReadSpec(
        read_id=rid,
        expected_status=ReadStatus.COMPLETE,
        orientation=orientation,
        transcript_length=transcript_length,
        transcript_id=f"transcript-{rid}",
        polyA_length=polyA_length,
        polyA_artifact=polyA_artifact,
        barcode_index=barcode_index,
        ssp_edits=ssp_edits,
        primer3_edits=primer3_edits,
        barcode_edits=barcode_edits,
        include_ssp=True,
        include_primer3=True,
        include_polyA=True,
        include_barcode=True,
        is_complex=False,
        is_fragment_expected=is_fragment,
    )


def _spec_three_prime(rid: str, *, orientation: str = "+", barcode_index: int = 2) -> ReadSpec:
    return ReadSpec(
        read_id=rid,
        expected_status=ReadStatus.THREE_PRIME_ONLY,
        orientation=orientation,
        transcript_length=_DEFAULT_TRANSCRIPT_LEN,
        transcript_id=f"transcript-{rid}",
        polyA_length=_DEFAULT_POLYA_LEN,
        polyA_artifact="clean",
        barcode_index=barcode_index,
        include_ssp=False,
        include_primer3=True,
        include_polyA=True,
        include_barcode=True,
    )


def _spec_five_prime(rid: str, *, orientation: str = "+") -> ReadSpec:
    return ReadSpec(
        read_id=rid,
        expected_status=ReadStatus.FIVE_PRIME_ONLY,
        orientation=orientation,
        transcript_length=_DEFAULT_TRANSCRIPT_LEN,
        transcript_id=f"transcript-{rid}",
        polyA_length=_DEFAULT_POLYA_LEN,
        polyA_artifact="clean",
        barcode_index=None,
        include_ssp=True,
        include_primer3=False,
        include_polyA=True,
        include_barcode=False,
    )


def _spec_missing_barcode(rid: str, *, orientation: str = "+") -> ReadSpec:
    return ReadSpec(
        read_id=rid,
        expected_status=ReadStatus.MISSING_BARCODE,
        orientation=orientation,
        transcript_length=_DEFAULT_TRANSCRIPT_LEN,
        transcript_id=f"transcript-{rid}",
        polyA_length=_DEFAULT_POLYA_LEN,
        polyA_artifact="clean",
        barcode_index=None,
        include_ssp=True,
        include_primer3=True,
        include_polyA=True,
        include_barcode=False,
    )


def _spec_unknown(rid: str, *, orientation: str = "+") -> ReadSpec:
    return ReadSpec(
        read_id=rid,
        expected_status=ReadStatus.UNKNOWN,
        orientation=orientation,
        transcript_length=_DEFAULT_TRANSCRIPT_LEN,
        transcript_id=f"transcript-{rid}",
        polyA_length=0,  # no polyA emitted
        polyA_artifact="clean",
        barcode_index=None,
        include_ssp=False,
        include_primer3=False,
        include_polyA=False,
        include_barcode=False,
    )


def _spec_complex(rid: str) -> ReadSpec:
    return ReadSpec(
        read_id=rid,
        expected_status=ReadStatus.COMPLEX,
        orientation="+",
        transcript_length=_DEFAULT_TRANSCRIPT_LEN,
        transcript_id=f"transcript-{rid}",
        polyA_length=_DEFAULT_POLYA_LEN,
        polyA_artifact="clean",
        # No barcode in the palindromic structure → Missing Barcode in
        # both orientations → strand resolution upgrades to Complex.
        barcode_index=None,
        include_ssp=True,
        include_primer3=True,
        include_polyA=True,
        include_barcode=False,
        is_complex=True,
    )


def generate_stress_test_specs(
    design: LibraryDesign,
    *,
    n_per_category: int = 5,
    seed: int = 42,
) -> list[ReadSpec]:
    """Generate a deterministic stress-test corpus (≥200 specs at default).

    Coverage axes:
      - 6 base statuses (Complete, 3' Only, 5' Only, Missing Barcode,
        Unknown, Complex)
      - Forward + reverse-complement orientations
      - Fragment vs full-length transcripts
      - PolyA: in-range, below min, above max, leading-A boundary cases
      - PolyA error types: clean / substitution / expanded / deleted
      - 5' adapter mutations: 0, 1, 2, 3 edits
      - 3' adapter mutations: 0, 1, 2, 3 edits
      - Barcode mutations: 0, 1, 2, 3 edits
    """
    rng = np.random.default_rng(seed)

    # Resolve barcode panel size — needed for "vary across panel" sweeps.
    bc_slots = [s for s in design.layout if isinstance(s, BarcodeSlot)]
    if not bc_slots:
        raise ValueError("design has no BarcodeSlot")
    n_barcodes = len(bc_slots[0].barcodes)

    specs: list[ReadSpec] = []
    next_id = 0

    def _new_id(tag: str) -> str:
        nonlocal next_id
        rid = _make_id(tag, next_id)
        next_id += 1
        return rid

    # 1. Base-status sweep — 5 statuses × 2 orientations × n_per_category
    for orientation in ("+", "-"):
        for _ in range(n_per_category):
            bc = int(rng.integers(0, n_barcodes))
            specs.append(_spec_complete(
                rid=_new_id("complete"), barcode_index=bc, orientation=orientation
            ))
            bc2 = int(rng.integers(0, n_barcodes))
            specs.append(_spec_three_prime(
                _new_id("threeprime"), orientation=orientation, barcode_index=bc2
            ))
            specs.append(_spec_five_prime(
                _new_id("fiveprime"), orientation=orientation
            ))
            specs.append(_spec_missing_barcode(
                _new_id("missbc"), orientation=orientation
            ))
            specs.append(_spec_unknown(
                _new_id("unknown"), orientation=orientation
            ))

    # 2. Complex sweep (orientation=+ only; structure is palindromic)
    for _ in range(n_per_category):
        specs.append(_spec_complex(_new_id("complex")))

    # 3. PolyA length sweep — Complete spec varying polyA_length.
    for polyA_len in (8, 12, 15, 22, 30, 40, 50):
        expected = (
            ReadStatus.UNKNOWN
            if polyA_len < 15 or polyA_len > 40
            else ReadStatus.COMPLETE
        )
        for _ in range(n_per_category):
            bc = int(rng.integers(0, n_barcodes))
            spec = _spec_complete(
                rid=_new_id(f"polya-len-{polyA_len}"),
                barcode_index=bc,
                polyA_length=polyA_len,
            )
            # Override expected status if out-of-range.
            spec = ReadSpec(**{**asdict(spec), "expected_status": expected})
            specs.append(spec)

    # 4. PolyA artifact sweep — Complete spec, vary the artifact kind.
    for artifact in ("substitution", "expanded", "deleted"):
        for _ in range(n_per_category):
            bc = int(rng.integers(0, n_barcodes))
            # `expanded` at 22+8=30 stays in [15,40] → Complete; `deleted`
            # at 22-6=16 stays in [15,40] → Complete; `substitution`
            # holds 22 length → Complete (mid-substitution still leaves
            # ≥4 anchored A's).
            specs.append(_spec_complete(
                rid=_new_id(f"polya-art-{artifact}"),
                barcode_index=bc,
                polyA_artifact=artifact,
            ))

    # 5. 5' adapter (SSP) edit sweep — Complete spec, vary ssp_edits.
    for n in (0, 1, 2, 3):
        for _ in range(n_per_category):
            bc = int(rng.integers(0, n_barcodes))
            specs.append(_spec_complete(
                rid=_new_id(f"ssp-ed-{n}"),
                barcode_index=bc,
                ssp_edits=n,
            ))

    # 6. 3' adapter (primer3) edit sweep — Complete spec, vary primer3_edits.
    for n in (0, 1, 2, 3):
        for _ in range(n_per_category):
            bc = int(rng.integers(0, n_barcodes))
            specs.append(_spec_complete(
                rid=_new_id(f"p3-ed-{n}"),
                barcode_index=bc,
                primer3_edits=n,
            ))

    # 7. Barcode edit sweep — Complete spec, vary barcode_edits.
    for n in (0, 1, 2, 3):
        for _ in range(n_per_category):
            bc = int(rng.integers(0, n_barcodes))
            specs.append(_spec_complete(
                rid=_new_id(f"bc-ed-{n}"),
                barcode_index=bc,
                barcode_edits=n,
            ))

    # 8. Boundary-barcode sweep — Complete specs explicitly using
    # BC04/05/10/11 (RC starts with 'A').
    for bc in _LEADING_A_RC_BARCODES:
        for _ in range(n_per_category):
            specs.append(_spec_complete(
                rid=_new_id(f"bc-leadA-{bc:02d}"),
                barcode_index=bc,
            ))

    # 9. Fragment sweep — Complete-structure specs with short transcripts.
    for tlen in (50, 100, 150):
        for _ in range(n_per_category):
            bc = int(rng.integers(0, n_barcodes))
            specs.append(_spec_complete(
                rid=_new_id(f"fragment-t{tlen}"),
                barcode_index=bc,
                transcript_length=tlen,
                is_fragment=True,
            ))

    return specs


# ──────────────────────────────────────────────────────────────────────
# SAM emission + ground-truth materialization
# ──────────────────────────────────────────────────────────────────────


def _format_sam_record(read_id: str, sequence: str, quality: str) -> str:
    """Minimal Dorado-style SAM body line.

    FLAG=4 (unmapped), RNAME=*, POS=0, MAPQ=0, CIGAR=*, RNEXT=*, PNEXT=0,
    TLEN=0, plus optional ``ch:i:1`` and ``du:f:0.5`` so the SAM ingest
    path's tag-handling code is exercised.
    """
    return (
        f"{read_id}\t4\t*\t0\t0\t*\t*\t0\t0\t{sequence}\t{quality}"
        f"\tch:i:1\tdu:f:0.5"
    )


def simulate_panel(
    specs: Iterable[ReadSpec],
    design: LibraryDesign,
    *,
    sam_path: Path,
    ground_truth_path: Path,
    seed: int = 42,
) -> None:
    """Materialize ``specs`` into a SAM file + parallel ground-truth parquet.

    SAM body: minimal valid Dorado-style records (FLAG=4, RNAME=*,
    plain quality, ``ch:i``/``du:f`` tags) — round-trips through
    :class:`SamReader.iter_batches` with no new ingest code.

    Ground truth: one row per spec, indexed by ``read_id``, conforming
    to :data:`GROUND_TRUTH_TABLE`.
    """
    sam_path = Path(sam_path)
    ground_truth_path = Path(ground_truth_path)
    sam_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    bc_slots = [s for s in design.layout if isinstance(s, BarcodeSlot)]
    barcodes = bc_slots[0].barcodes if bc_slots else ()

    sam_lines: list[str] = ["@HD\tVN:1.6\tSO:unknown", "@PG\tID:simulator\tPN:constellation"]
    gt_rows: list[dict[str, object]] = []

    for spec in specs:
        seq, qual = assemble_sequence(spec, design, rng=rng)
        if spec.orientation == "-":
            seq = _reverse_complement(seq)
            qual = qual[::-1]
        sam_lines.append(_format_sam_record(spec.read_id, seq, qual))

        bc_name = (
            barcodes[spec.barcode_index].name
            if spec.barcode_index is not None and spec.barcode_index < len(barcodes)
            else None
        )
        gt_rows.append({
            "read_id": spec.read_id,
            "expected_status": spec.expected_status.value,
            "orientation": spec.orientation,
            "transcript_length": spec.transcript_length,
            "transcript_id": spec.transcript_id,
            "polyA_length": spec.polyA_length,
            "polyA_artifact": spec.polyA_artifact,
            "expected_barcode_index": spec.barcode_index,
            "expected_barcode_name": bc_name,
            "ssp_edits": spec.ssp_edits,
            "primer3_edits": spec.primer3_edits,
            "barcode_edits": spec.barcode_edits,
            "include_ssp": spec.include_ssp,
            "include_primer3": spec.include_primer3,
            "include_polyA": spec.include_polyA,
            "include_barcode": spec.include_barcode,
            "is_complex": spec.is_complex,
            "is_fragment_expected": spec.is_fragment_expected,
        })

    sam_path.write_text("\n".join(sam_lines) + "\n")
    table = (
        pa.Table.from_pylist(gt_rows, schema=GROUND_TRUTH_TABLE)
        if gt_rows
        else GROUND_TRUTH_TABLE.empty_table()
    )
    pq.write_table(table, ground_truth_path)


# ──────────────────────────────────────────────────────────────────────
# Helper exposed for the integration test (deterministic-clean filter).
# ──────────────────────────────────────────────────────────────────────


def is_deterministic_clean(spec: ReadSpec) -> bool:
    """A spec is deterministic-clean when zero mutations + clean polyA
    + safely-in-range polyA length + a status the hard scorer reproduces
    from structure alone (excludes Unknown / Complex).

    The polyA window is capped at 39 (one below the scorer's max_length=40)
    because a leading-A barcode RC merges with the polyA run via the
    edge_distance=1 tolerance — a polyA_length=40 spec followed by a
    BC05-style leading-A barcode produces a 41-base merged run, which
    the scorer correctly rejects as a homopolymer artifact. That
    behavior is real and documented (see designs.py boundary note);
    we exclude it from the clean roundtrip rather than mask it.

    Used by the simulator's end-to-end test to filter to specs whose
    expected_status MUST match the demux output exactly under the
    HardThresholdScorer.
    """
    if spec.ssp_edits or spec.primer3_edits or spec.barcode_edits:
        return False
    if spec.polyA_artifact != "clean":
        return False
    if spec.include_polyA and not (15 <= spec.polyA_length <= 39):
        return False
    if spec.expected_status in (ReadStatus.UNKNOWN, ReadStatus.COMPLEX):
        return False
    if spec.is_complex:
        return False
    return True


__all__ = [
    "GROUND_TRUTH_TABLE",
    "ReadSpec",
    "assemble_sequence",
    "generate_stress_test_specs",
    "is_deterministic_clean",
    "mutate",
    "simulate_panel",
]
