"""Bruker TopSpin reader — currently supports 1D experiments.

Bruker TopSpin lays out each experiment as a directory of files. The
minimal subset constellation cares about, by dimensionality:

    1D experiment:
        acqus       Direct-dimension acquisition parameters (JCAMP-DX text).
        fid         Raw interleaved (real, imag) binary samples.

    2D+ experiment (not yet supported by this reader):
        acqus       Direct dimension.
        acqu2s      First indirect dimension.
        acqu3s      Second indirect dimension (3D).
        ...
        ser         Single binary file containing the full nD interleaved data.

The byte-level layout (interleaved real/imag pairs in the binary) is
identical across dimensionalities — the difference is which file holds
it (``fid`` vs ``ser``), how many parameter files describe it, and how
the resulting flat array is reshaped. ``NMR_FID_TABLE`` (the output
schema) and the ``x.nmr.shape`` metadata convention are
dimension-agnostic by design, so the 2D / 3D extension keeps the same
``BrukerReader`` class, the same output schema, and reuses the
byte-decoding helper below — only the file-discovery and acqus-merging
logic grows.

The ``read`` method dispatches on directory contents and currently
raises ``NotImplementedError`` for 2D+ datasets (``ser`` file present),
pointing at the planned 2D port. The 1D path is fully implemented.

JCAMP-DX in ``acqus`` is a key-value text format with lines like
``##$KEY= value``. Fields constellation reads from the direct
dimension (``acqus``):

    TD          Total points in this dimension's binary representation
                (= 2 × N_complex for the direct dimension).
    SW_h        Spectral width in Hz (= 1 / dwell_s).
    SFO1        Actual transmitter frequency in MHz (BF1 + O1/1e6).
                Carrier for Hz ↔ ppm conversion.
    BF1         Base Larmor frequency in MHz (field-strength label).
    O1          Transmitter offset from BF1 in Hz.
    GRPDLY      Digital-filter group delay in points (modern AVANCE).
    DTYPA       FID encoding: 0 = int32, 2 = float64.
    BYTORDA     Byte order: 0 = little-endian, 1 = big-endian.
    NS          Number of scans.
    RG          Receiver gain.
    PULPROG     Pulse program name.
    NUC1        Nucleus on channel 1 (e.g. '1H', '13C', '15N').
    TE          Sample temperature in K.
    SOLVENT     Solvent name.
    INSTRUM     Instrument identifier.

Registration: this reader does **not** register with
``core.io.readers``' suffix-based dispatch — Bruker experiments are
directories with conventional filenames, not single files with
extensions. Instantiate ``BrukerReader()`` and call ``.read(path)``
directly. If a second NMR vendor reader lands later
(Varian/Agilent ``.fid``, JEOL ``.jdf``, ...), the registry may earn
its keep here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import torch

from constellation.core.io.bundle import Bundle
from constellation.core.io.readers import RawReader, ReadResult
from constellation.core.io.schemas import pack_metadata
from constellation.nmr.io.schemas import NMR_FID_TABLE


# DTYPA code → (torch dtype for direct decode, raw element size in bytes).
_DTYPE_BY_DTYPA: dict[int, tuple[torch.dtype, int]] = {
    0: (torch.int32, 4),    # int32
    2: (torch.float64, 8),  # float64
}

# BYTORDA code → numpy byte-order specifier (used when the file's order
# disagrees with the host order and the numpy boundary is required).
_BYTEORDER_BY_BYTORDA: dict[int, str] = {
    0: "<",  # little-endian
    1: ">",  # big-endian
}


class BrukerReader(RawReader):
    """Read a Bruker TopSpin NMR experiment directory.

    Currently supports **1D experiments** (``acqus`` + ``fid``). The
    schema (``NMR_FID_TABLE``), metadata convention (``x.nmr.shape``,
    ``x.nmr.sw_hz``, ...), and byte-decoding primitives are
    dimension-agnostic by design, so the 2D / 3D / nD extension reuses
    this class — only the file-discovery and per-dimension acqus
    merging logic grows.

    Output: a ``ReadResult`` whose ``primary`` table conforms to
    :data:`constellation.nmr.io.schemas.NMR_FID_TABLE` — two rows
    (component = 'real' / 'imag'), each holding the full FID as a flat
    float64 list in row-major order. The dimensional shape lives in
    schema metadata under ``x.nmr.shape``. NMR-specific parameters
    populate ``x.nmr.*`` keys; run-level identifiers
    (``run_id``, ``device``, ``instrument_id``) land in
    ``ReadResult.run_metadata``.
    """

    # The class IS a RawReader subclass for shape parity with other
    # readers, but Bruker's directory-with-known-file-names layout has
    # no extension to dispatch on. Leave the suffixes tuple empty and
    # skip the registry; callers instantiate the reader directly.
    suffixes: ClassVar[tuple[str, ...]] = ()
    modality: ClassVar[str | None] = "nmr"

    def read(self, source: Path | Bundle | str) -> ReadResult:
        """Decode a Bruker experiment into an ``NMR_FID_TABLE``.

        Dispatches on the directory contents:

        - **1D** (``fid`` present, no ``ser``): handled here. Reads
          ``acqus`` + ``fid``, deinterleaves real/imag, stamps
          ``x.nmr.shape = [N_complex]``.
        - **2D+** (``ser`` present, regardless of ``fid``): raises
          ``NotImplementedError``. The 2D port plan will land this
          path; the schema and metadata convention are already shaped
          to receive it.

        Parameters
        ----------
        source : Path, str, or Bundle
            Path to a Bruker experiment directory (the expno directory
            containing ``acqus`` and ``fid``), or a ``DirBundle`` rooted
            there.

        Returns
        -------
        ReadResult
            ``primary`` is the ``NMR_FID_TABLE``. ``run_metadata`` has
            ``run_id``, ``device``, ``instrument_id``, plus the
            unmodified parsed ``acqus`` dict under ``raw_acqus`` for
            forward debugging.

        Raises
        ------
        FileNotFoundError
            If ``acqus`` is missing, or neither ``fid`` nor ``ser`` is present.
        NotImplementedError
            If ``ser`` is present (2D+ experiment); see the project plan
            for the 2D port roadmap.
        ValueError
            If unrecognised DTYPA / BYTORDA codes appear in ``acqus``.
        """
        files = _discover_bruker_files(source)
        root = files["__root__"]
        raw_acqus = _parse_acqus(files["acqus"])

        # Dispatch on dimensionality.
        if "ser" in files:
            raise NotImplementedError(
                "Bruker 2D+ experiments (ser file present) are not yet "
                "supported. The 2D port plan lands as a follow-up; this "
                "class, schema (NMR_FID_TABLE), and metadata convention "
                "(x.nmr.shape) are already shaped to receive it. "
                f"Experiment: {root}"
            )
        if "fid" not in files:
            raise FileNotFoundError(
                f"Bruker experiment at {root} contains neither 'fid' nor 'ser'"
            )

        return _decode_1d(files["fid"], raw_acqus=raw_acqus, root=root)


# ──────────────────────────────────────────────────────────────────────
# File discovery — pulls bytes for every member constellation knows about.
# ──────────────────────────────────────────────────────────────────────


# Names we look for in a Bruker experiment directory. ``acqus`` is
# required; ``fid`` and ``ser`` are mutually exclusive (1D vs 2D+).
# ``acqu2s`` and ``acqu3s`` appear in 2D and 3D experiments respectively
# and are read so the 2D port can plug straight in.
_KNOWN_MEMBERS: tuple[str, ...] = (
    "acqus",
    "acqu2s",
    "acqu3s",
    "fid",
    "ser",
)


def _discover_bruker_files(source: Path | Bundle | str) -> dict[str, bytes | Path]:
    """Resolve ``source`` and return the bytes of every Bruker file present.

    Returns a dict with keys for each file actually found (subset of
    :data:`_KNOWN_MEMBERS`) plus a sentinel ``"__root__"`` carrying the
    directory ``Path`` for diagnostics. ``acqus`` is required; absence
    raises ``FileNotFoundError``.
    """
    if isinstance(source, Bundle):
        members = set(source.members())
        if "acqus" not in members:
            raise FileNotFoundError(
                f"acqus not found in Bruker bundle at {source.path}"
            )
        out: dict[str, bytes | Path] = {"__root__": Path(source.path)}
        for name in _KNOWN_MEMBERS:
            if name in members:
                out[name] = source.open(name)
        return out

    root = Path(source)
    if not root.is_dir():
        raise NotADirectoryError(
            f"Bruker experiment must be a directory; got {root}"
        )
    acqus = root / "acqus"
    if not acqus.exists():
        raise FileNotFoundError(f"acqus not found in {root}")

    out = {"__root__": root}
    for name in _KNOWN_MEMBERS:
        path = root / name
        if path.exists():
            out[name] = path.read_bytes()
    return out


# ──────────────────────────────────────────────────────────────────────
# 1D-specific decode path
# ──────────────────────────────────────────────────────────────────────


def _decode_1d(
    fid_bytes: bytes, *, raw_acqus: dict[str, Any], root: Path
) -> ReadResult:
    """1D-specific assembly: bytes + raw acqus → ReadResult.

    Wraps the dimension-agnostic ``_decode_fid`` byte-deinterleave with
    1D shape inference (``[TD // 2]``) and 1D metadata stamping.
    """
    td = int(raw_acqus["TD"])
    dtypa = int(raw_acqus.get("DTYPA", 0))
    bytorda = int(raw_acqus.get("BYTORDA", 0))

    real, imag = _decode_fid(fid_bytes, td=td, dtypa=dtypa, bytorda=bytorda)
    n_complex = int(real.shape[-1])

    components = pa.array(["real", "imag"], type=pa.string())
    fid_lists = pa.array(
        [real.tolist(), imag.tolist()],
        type=pa.list_(pa.float64()),
    )
    primary = pa.Table.from_arrays(
        [components, fid_lists],
        schema=NMR_FID_TABLE,
    )

    nmr_meta = _build_nmr_metadata(
        raw_acqus, shape=[n_complex], dimensionality=1
    )
    existing = NMR_FID_TABLE.metadata or {}
    primary = primary.replace_schema_metadata(
        {**existing, **pack_metadata(nmr_meta)}
    )

    run_metadata: dict[str, Any] = {
        "run_id": _derive_run_id(root),
        "device": "Bruker",
        "instrument_id": str(raw_acqus.get("INSTRUM", "")) or None,
        "raw_acqus": raw_acqus,
    }
    return ReadResult(primary=primary, companions={}, run_metadata=run_metadata)


# ──────────────────────────────────────────────────────────────────────
# acqus parsing — dimension-agnostic (used by acqus / acqu2s / acqu3s)
# ──────────────────────────────────────────────────────────────────────


# Bruker acqus keys are mostly all-uppercase (TD, SW_h, SFO1, ...), but some
# Bruker conventions use a trailing lowercase suffix to encode units —
# `SW_h` for "spectral width in Hz" (as opposed to ppm) being the canonical
# example. Allow mixed case so those round-trip.
_KV_LINE = re.compile(r"^##\$([A-Za-z0-9_]+)=\s*(.*)$")


def _parse_acqus(data: bytes) -> dict[str, Any]:
    """Parse a JCAMP-DX acqus / acqu2s / acqu3s file into ``{KEY: value}``.

    Scalar numerics (int / float) are coerced from their text form;
    angle-bracketed strings have their delimiters stripped; array-valued
    parameters whose value starts with ``(`` are returned as raw strings
    (only scalar values are needed for the 1D / 2D read paths).
    """
    text = data.decode("latin-1")  # Bruker files are latin-1, not utf-8
    params: dict[str, Any] = {}
    for line in text.splitlines():
        m = _KV_LINE.match(line.rstrip())
        if m is None:
            continue
        key, value = m.group(1), m.group(2)
        params[key] = _parse_acqus_value(value.strip())
    return params


def _parse_acqus_value(s: str) -> Any:
    """Convert an acqus value token to int / float / str."""
    if s.startswith("<") and s.endswith(">"):
        return s[1:-1]
    if s.startswith("("):
        return s  # array header, returned as-is
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ──────────────────────────────────────────────────────────────────────
# FID binary decode — dimension-agnostic
# ──────────────────────────────────────────────────────────────────────


def _decode_fid(
    data: bytes, *, td: int, dtypa: int, bytorda: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """De-interleave a Bruker FID / SER binary into two flat float64 tensors.

    Bruker stores interleaved real/imag samples in the binary: ``(R[0],
    I[0], R[1], I[1], ...)``. This works the same for 1D ``fid`` and
    nD ``ser`` files; the ``td`` argument is the total interleaved count
    (= 2 × total complex samples across all dimensions). For 1D,
    ``td`` comes straight from ``acqus['TD']``; for 2D+, the caller
    multiplies through the indirect dimensions.

    Honours the constellation invariant: little-endian on a
    little-endian host decodes straight through ``torch.frombuffer``;
    big-endian requires the numpy byte-swap boundary.
    """
    if dtypa not in _DTYPE_BY_DTYPA:
        raise ValueError(
            f"Unrecognised DTYPA={dtypa}. Supported: 0 (int32), 2 (float64)."
        )
    if bytorda not in _BYTEORDER_BY_BYTORDA:
        raise ValueError(
            f"Unrecognised BYTORDA={bytorda}. Supported: 0 (LE), 1 (BE)."
        )

    torch_dtype, elem_size = _DTYPE_BY_DTYPA[dtypa]
    expected_bytes = td * elem_size
    if len(data) < expected_bytes:
        raise ValueError(
            f"FID/SER binary is shorter than TD={td} requires: "
            f"{len(data)} bytes vs {expected_bytes} expected"
        )

    if bytorda == 0:
        # Little-endian native (Bruker AVANCE III default; matches Intel / ARM).
        # bytearray() wraps the bytes mutably so torch doesn't issue the
        # read-only-buffer warning; the array is freed when arr is replaced
        # by the float64 conversion below.
        arr = torch.frombuffer(
            bytearray(data[:expected_bytes]),
            dtype=torch_dtype,
            count=td,
        )
    else:
        # Big-endian — go through numpy to byte-swap, then bridge to torch.
        np_dtype = (
            f"{_BYTEORDER_BY_BYTORDA[bytorda]}{'i4' if dtypa == 0 else 'f8'}"
        )
        arr_np = np.frombuffer(data, dtype=np_dtype, count=td)
        # .astype(...newbyteorder('=')) byte-swaps into native order.
        arr_np = arr_np.astype(arr_np.dtype.newbyteorder("=")).copy()
        arr = torch.from_numpy(arr_np)

    arr_f64 = arr.to(torch.float64)
    real = arr_f64[0::2].contiguous()
    imag = arr_f64[1::2].contiguous()
    return real, imag


# ──────────────────────────────────────────────────────────────────────
# Metadata stamping — populates the x.nmr.* key list
# ──────────────────────────────────────────────────────────────────────


def _build_nmr_metadata(
    raw_acqus: dict[str, Any],
    *,
    shape: list[int],
    dimensionality: int,
) -> dict[str, Any]:
    """Build the ``x.nmr.*`` schema-metadata dict from parsed direct-dim acqus.

    Concrete key list, 1D today; the 2D extension adds per-dimension
    twins (``x.nmr.sw_hz_t1``, ``x.nmr.sfo1_mhz_t1``, ``x.nmr.nucleus_t1``,
    ``x.nmr.acquisition_mode_t1``) for each indirect dimension when 2D+
    readers land.

        x.nmr.shape                 [N] (1D); [N_t1, N_t2] (2D); etc.
        x.nmr.dimensionality        Integer count of dimensions.
        x.nmr.axis_domain           Per-axis FT state, ordered to match
                                    x.nmr.shape (outer→inner). Values
                                    are "time" or "freq". Time-domain
                                    everywhere at read; FFTs transition
                                    individual axes to "freq".
        x.nmr.sw_hz                 Direct-dimension sweep width in Hz.
        x.nmr.sfo1_mhz              Carrier frequency (channel 1, MHz).
        x.nmr.bf1_mhz               Base Larmor frequency (channel 1, MHz).
        x.nmr.o1_hz                 Transmitter offset from BF1 (Hz).
        x.nmr.grpdly                Digital-filter group delay in points.
        x.nmr.nucleus               Nucleus on channel 1 (e.g. '1H').
        x.nmr.dtypa                 Bruker FID encoding code (0 / 2).
        x.nmr.bytorda               Bruker FID byte-order code (0 / 1).
        x.nmr.ns                    Number of scans.
        x.nmr.rg                    Receiver gain.
        x.nmr.pulprog               Pulse program name.
        x.nmr.temp_k                Sample temperature in K (TE).
        x.nmr.solvent               Solvent name.

    Optional fields are populated only when present in ``raw_acqus`` —
    missing parameters are silently dropped rather than forcing
    sentinel values.
    """
    out: dict[str, Any] = {
        "x.nmr.shape": list(shape),
        "x.nmr.dimensionality": dimensionality,
        # Per-axis FT state, ordered to match x.nmr.shape (outer→inner).
        # Maps directly to NMRPipe's FDF*FTFLAG fields on eventual export
        # (vocabulary "time" / "freq" ↔ NMRPipe's 0 / 1). NMR-aware
        # downstream tools (CcpNmr, NMRPipe, etc.) read per-axis state
        # for the same reason: 2D processing can have asymmetric FT
        # state (e.g. t2 FT'd, t1 not yet) that a single global flag
        # can't represent.
        "x.nmr.axis_domain": ["time"] * dimensionality,
        "x.nmr.sw_hz": float(raw_acqus["SW_h"]),
        "x.nmr.sfo1_mhz": float(raw_acqus["SFO1"]),
        "x.nmr.bf1_mhz": float(raw_acqus["BF1"]),
        "x.nmr.o1_hz": float(raw_acqus["O1"]),
        "x.nmr.dtypa": int(raw_acqus.get("DTYPA", 0)),
        "x.nmr.bytorda": int(raw_acqus.get("BYTORDA", 0)),
    }
    if "GRPDLY" in raw_acqus:
        out["x.nmr.grpdly"] = float(raw_acqus["GRPDLY"])
    if "NUC1" in raw_acqus:
        out["x.nmr.nucleus"] = str(raw_acqus["NUC1"])
    if "NS" in raw_acqus:
        out["x.nmr.ns"] = int(raw_acqus["NS"])
    if "RG" in raw_acqus:
        out["x.nmr.rg"] = float(raw_acqus["RG"])
    if "PULPROG" in raw_acqus:
        out["x.nmr.pulprog"] = str(raw_acqus["PULPROG"])
    if "TE" in raw_acqus:
        out["x.nmr.temp_k"] = float(raw_acqus["TE"])
    if "SOLVENT" in raw_acqus:
        out["x.nmr.solvent"] = str(raw_acqus["SOLVENT"])
    return out


def _derive_run_id(root: Path) -> str:
    """Derive a stable run identifier from the experiment directory path.

    Bruker convention is ``<dataset_name>/<expno>/``. Combine into
    ``<dataset>/<expno>`` so the identifier survives moves of the
    absolute path.
    """
    if root.parent.name:
        return f"{root.parent.name}/{root.name}"
    return root.name


__all__ = ["BrukerReader"]
