"""Track-kernel ABC + registry for the viz layer.

A `TrackKernel` is the data-side counterpart to a TS renderer module. Each
kernel:

1. Declares its `kind` (string id) and the wire `schema` it emits in vector
   mode — this is the Arrow IPC schema the client decodes via
   `apache-arrow` JS.
2. `discover(session)` walks the session's parquet outputs and returns one
   `TrackBinding` per renderable instance (e.g. one binding per sample for
   coverage, one binding for the full alignment table).
3. `metadata(binding)` returns the small JSON the frontend fetches eagerly to
   set up the track (palette, available samples, configured height, ...).
4. `threshold(binding, query)` returns `VECTOR` or `HYBRID` for the requested
   viewport. Vector emits per-glyph batches matching `kernel.schema`; hybrid
   emits a single one-row table matching `HYBRID_SCHEMA` containing a
   datashader-rendered PNG and its extents.
5. `fetch(binding, query)` yields `pa.RecordBatch`es. The server's IPC stream
   helper concatenates them into one frame on the wire.

All threshold defaults live as class attributes so they can be tuned per
deployment without touching kernel logic. The frontend can also force a
mode via `?force=vector|hybrid` for empirical tuning; the server echoes the
chosen mode in the `X-Track-Mode` response header.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa


# ----------------------------------------------------------------------
# Wire-shape contracts
# ----------------------------------------------------------------------

#: Schema for hybrid-mode payloads (kernel returns a one-row table). The
#: extent fields name the genomic / track-local coordinate window the PNG
#: covers; width/height carry the actual rasterized pixel size; n_items is
#: the underlying glyph count (used by the export endpoint to warn about
#: vector cost). Mode is the resolved decision, surfaced for symmetry with
#: the `X-Track-Mode` header.
HYBRID_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("png_bytes", pa.binary()),
        pa.field("extent_start", pa.int64()),
        pa.field("extent_end", pa.int64()),
        pa.field("extent_y0", pa.float64()),
        pa.field("extent_y1", pa.float64()),
        pa.field("width_px", pa.int32()),
        pa.field("height_px", pa.int32()),
        pa.field("n_items", pa.int64()),
        pa.field("mode", pa.string()),
    ],
    metadata={b"schema_name": b"VizHybridFrame"},
)


class ThresholdDecision(StrEnum):
    """Resolved render mode for a `TrackQuery`. The kernel may pick either;
    the server echoes the choice in the `X-Track-Mode` response header."""

    VECTOR = "vector"
    HYBRID = "hybrid"


# ----------------------------------------------------------------------
# Query / binding records
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TrackQuery:
    """A single read request from the frontend.

    `viewport_px` is the on-screen width the client intends to draw the
    track at; `max_glyphs` is the client's ceiling for per-glyph rendering
    (when exceeded, kernels switch to hybrid). Both are advisory — the
    kernel's threshold logic is the source of truth, but the client values
    let kernels make zoom-aware decisions.

    `force` is the optional `?force=vector|hybrid` override; when set, the
    kernel must honor it (or, in the case of `force=vector` with a too-
    expensive payload, return vector with a truncated flag in the response
    metadata — handled at the server layer, not the kernel).
    """

    contig: str
    start: int
    end: int
    samples: tuple[str, ...] = ()
    viewport_px: int = 1200
    max_glyphs: int = 50_000
    force: ThresholdDecision | None = None


@dataclass(frozen=True)
class TrackBinding:
    """A concrete renderable produced by `TrackKernel.discover`.

    `paths` is the resolved set of parquet locations this binding reads
    (e.g. `{"alignments": .../S2_align/alignments, "blocks": .../alignment_blocks}`
    for the read-pileup kernel). `config` carries per-track display state
    (palette, height, configured sample subset) — the dashboard is allowed
    to mutate this through the frontend; the server treats it as opaque.
    `label` is the human-readable name shown in the track header.
    """

    session_id: str
    kind: str
    binding_id: str
    label: str
    paths: dict[str, Path]
    config: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Kernel ABC
# ----------------------------------------------------------------------


class TrackKernel(ABC):
    """Per-modality data kernel. Concrete subclasses register via
    `@register_track` so the server can look them up by `kind`.

    Subclasses MUST set the class-level attributes `kind` and `schema`. The
    `schema` is the *vector-mode* wire schema; hybrid-mode payloads always
    use `HYBRID_SCHEMA`.

    Threshold defaults (`vector_glyph_limit`, `vector_bp_per_pixel_limit`)
    are class attributes so kernels can override them and tunings can land
    without touching the threshold method body.
    """

    #: Short string identifier — must be unique across registered kernels.
    kind: ClassVar[str]

    #: Vector-mode wire schema. Hybrid mode always uses `HYBRID_SCHEMA`.
    schema: ClassVar[pa.Schema]

    #: Default threshold knobs (kernels override as appropriate). A kernel
    #: that's always vector (e.g. coverage_histogram) leaves these unused.
    vector_glyph_limit: ClassVar[int] = 4_000
    vector_bp_per_pixel_limit: ClassVar[float] = 50.0

    @abstractmethod
    def discover(self, session: "Session") -> list[TrackBinding]:  # noqa: F821
        """Return all renderable bindings this kernel can produce for the
        given session. Empty list when no relevant outputs are present."""

    @abstractmethod
    def metadata(self, binding: TrackBinding) -> dict[str, Any]:
        """Return the small JSON the frontend uses to set up the track
        before the first data fetch (palette, samples, height, ...)."""

    @abstractmethod
    def threshold(self, binding: TrackBinding, query: TrackQuery) -> ThresholdDecision:
        """Decide vector vs hybrid for this query. Honors `query.force` if
        present (subject to the kernel's ability to satisfy it)."""

    @abstractmethod
    def fetch(
        self,
        binding: TrackBinding,
        query: TrackQuery,
        mode: ThresholdDecision,
    ) -> Iterator[pa.RecordBatch]:
        """Yield record batches matching either `self.schema` (vector) or
        `HYBRID_SCHEMA` (hybrid). The server wraps the iterator with
        `pa.ipc.new_stream(...)` and ships it as a `StreamingResponse`."""

    # ------------------------------------------------------------------
    # Helpers shared by kernel implementations
    # ------------------------------------------------------------------

    def estimate_vector_cost(
        self, binding: TrackBinding, query: TrackQuery
    ) -> int | None:
        """Optional: return an estimate of the glyph count a vector
        response would emit, for the export-cost confirmation dialog.
        Returning `None` means "unknown / can't cheaply estimate" — the
        client falls back to a generic warning."""
        return None


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


_REGISTRY: dict[str, TrackKernel] = {}


def register_track(cls: type[TrackKernel]) -> type[TrackKernel]:
    """Class decorator that instantiates and registers a kernel by `kind`.

    Usage::

        @register_track
        class CoverageHistogramKernel(TrackKernel):
            kind = "coverage_histogram"
            schema = ...

    Importing the kernel module (which constellation.viz does at package
    import) registers the kernel exactly once. Re-registration raises.
    """
    if not isinstance(cls.kind, str) or not cls.kind:
        raise TypeError(f"{cls.__name__} must set a non-empty `kind` class attribute")
    if cls.kind in _REGISTRY:
        raise ValueError(f"track kernel {cls.kind!r} already registered")
    _REGISTRY[cls.kind] = cls()
    return cls


def get_kernel(kind: str) -> TrackKernel:
    if kind not in _REGISTRY:
        raise KeyError(
            f"track kernel {kind!r} not registered "
            f"(known: {sorted(_REGISTRY)})"
        )
    return _REGISTRY[kind]


def registered_kinds() -> list[str]:
    return sorted(_REGISTRY)
