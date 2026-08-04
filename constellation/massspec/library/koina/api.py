"""The three public entry points, layered so callers can stop early.

* :func:`predict` — raw passthrough. Whatever the model declares, you
  supply; you get its response as Arrow. No digestion, no modseq
  translation, no Library. This is the notebook path, and it stays
  deliberately thin: the point of Koina is that the round trip is
  already easy, so wrapping it in ceremony would be a regression.
* :func:`predict_fragments` — adds modseq translation, annotation
  parsing, and the local-vs-returned m/z cross-check.
* :func:`predict_library` — adds digestion and Library assembly. What
  the CLI calls.

Arrow and numpy in, Arrow out. No pandas is constructed anywhere on the
path (see ``client.KoinaClient.predict``), so the package adds no pandas
import to ``massspec.library``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from constellation.massspec.library.digest import (
    PrecursorSpec,
    precursors_from_fasta,
)
from constellation.massspec.library.koina._modseq import (
    KoinaModSeqError,
    format_koina_modseq,
)
from constellation.massspec.library.koina.assemble import (
    AssemblyStats,
    assemble_library,
)
from constellation.massspec.library.koina.client import (
    KoinaInputError,
    PredictClient,
    make_client,
)
from constellation.massspec.library.koina.models import (
    Ms2Model,
    RtModel,
    ms2_model,
    rt_model,
)
from constellation.massspec.library.koina.nce import adjust_nce
from constellation.massspec.library.library import Library
from constellation.core.sequence.proforma import parse_proforma

Inputs = Mapping[str, Any] | pa.Table | Sequence[Mapping[str, Any]]


def _to_arrays(inputs: Inputs) -> dict[str, np.ndarray]:
    """Coerce the accepted input shapes into ``{name: (N, 1) ndarray}``.

    Koina's Triton endpoints want 2-D column vectors; accepting flat
    lists and letting this reshape them is the whole ergonomic
    difference between the raw client and a usable one.
    """
    if isinstance(inputs, pa.Table):
        columns = {name: inputs.column(name).to_pylist() for name in inputs.column_names}
    elif isinstance(inputs, Mapping):
        columns = {str(k): v for k, v in inputs.items()}
    elif isinstance(inputs, Sequence):
        rows = list(inputs)
        if not rows:
            raise ValueError("cannot predict from an empty input sequence")
        keys = list(rows[0])
        columns = {k: [row[k] for row in rows] for k in keys}
    else:
        raise TypeError(
            f"unsupported input type {type(inputs).__name__}; pass a pa.Table, "
            f"a dict of arrays, or a sequence of row dicts"
        )

    out: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if name == "peptide_sequences":
            arr = arr.astype(object)
        elif name == "precursor_charges":
            arr = arr.astype(np.int32)
        elif arr.dtype.kind == "f":
            arr = arr.astype(np.float32)
        out[name] = arr
    return out


def _drop_padding(table: pa.Table, min_intensity: float) -> pa.Table:
    """Remove Prosit's fixed-grid padding rows.

    Intensity models emit a constant 174-wide grid per precursor and pad
    the positions a given peptide cannot produce with ``-1``. Those are
    absent ions, not weak ones. koinapy's DataFrame path filters them
    for you; the numpy path this client uses does not, so we do it here
    — otherwise ``predict()`` would return mostly padding and disagree
    with what the same call through koinapy returns.
    """
    if "intensities" not in table.column_names:
        return table
    intensities = np.asarray(table.column("intensities").to_pylist(), dtype=np.float64)
    return table.filter(pa.array(intensities > min_intensity))


def _response_to_table(response: Mapping[str, np.ndarray]) -> pa.Table:
    """Flatten a ``{name: (N, F)}`` response into a long Arrow table.

    Scalar-per-row heads (RT) stay one row per input; fragment heads
    expand to one row per (input, fragment) with ``row_index`` carrying
    the link back, so the caller can rejoin without guessing at shapes.
    """
    arrays = {k: np.asarray(v) for k, v in response.items()}
    widths = {k: (v.shape[1] if v.ndim > 1 else 1) for k, v in arrays.items()}
    n_rows = next(iter(arrays.values())).shape[0]
    fan = max(widths.values())

    columns: dict[str, Any] = {
        "row_index": np.repeat(np.arange(n_rows, dtype=np.int32), fan)
    }
    for name, arr in arrays.items():
        flat = np.repeat(arr.ravel(), fan) if widths[name] == 1 < fan else arr.ravel()
        if arr.dtype == object:
            flat = [
                v.decode("ascii") if isinstance(v, bytes) else v for v in flat.tolist()
            ]
        columns[name] = flat
    return pa.table(columns)


def predict(
    model: str,
    inputs: Inputs,
    *,
    server: str | None = None,
    ssl: bool = True,
    min_intensity: float = 1e-4,
    client: PredictClient | None = None,
) -> pa.Table:
    """Run one Koina model and return its response as Arrow.

    ``inputs`` may be a ``pa.Table``, a dict of arrays/lists, or a
    sequence of row dicts. Columns the model does not declare raise a
    warning and are ignored by the server — notably
    ``Prosit_2020_intensity_CID`` declares no ``collision_energies``, so
    passing one there has no effect on the prediction.

    >>> predict("Prosit_2020_intensity_HCD",
    ...         {"peptide_sequences": ["LVNELTEFAK"],
    ...          "precursor_charges": [2],
    ...          "collision_energies": [30.0]})   # doctest: +SKIP
    """
    arrays = _to_arrays(inputs)
    cl = client or make_client(model, server=server, ssl=ssl)
    table = _response_to_table(cl.predict(arrays, min_intensity=min_intensity))
    return _drop_padding(table, min_intensity)


def _validate_specs(specs: Sequence[PrecursorSpec], model: Ms2Model) -> list[str]:
    """Pre-flight the whole grid; return per-spec Koina modseq strings.

    Fails before the first network call. A 5M-precursor job that dies 40
    minutes in on peptide 3.2M with an opaque server error is the worst
    available outcome, and every check here is local.
    """
    problems: list[str] = []
    modseqs: list[str] = []
    for spec in specs:
        try:
            modseqs.append(
                format_koina_modseq(
                    parse_proforma(spec.modified_sequence),
                    supported=model.supported_mods,
                )
            )
        except KoinaModSeqError as exc:
            problems.append(f"{spec.modified_sequence}: {exc}")
            modseqs.append("")
            continue
        if len(spec.sequence) > model.max_length:
            problems.append(
                f"{spec.modified_sequence}: length {len(spec.sequence)} exceeds "
                f"{model.name}'s maximum of {model.max_length}"
            )
        if not (model.min_charge <= spec.charge <= model.max_charge):
            problems.append(
                f"{spec.modified_sequence}/{spec.charge}: charge outside "
                f"{model.name}'s supported range "
                f"{model.min_charge}-{model.max_charge}"
            )
    if problems:
        shown = "\n  ".join(problems[:10])
        more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise KoinaInputError(
            f"{len(problems)} precursor(s) cannot be predicted by "
            f"{model.name}:\n  {shown}{more}\n"
            f"Filter them out, or pass on_unsupported='skip'."
        )
    return modseqs


def predict_fragments(
    specs: Sequence[PrecursorSpec],
    *,
    model: str = "Prosit_2020_intensity_HCD",
    collision_energy: float | None = None,
    adjust_nce_for_dia: bool = True,
    server: str | None = None,
    min_intensity: float = 1e-4,
    on_unsupported: str = "error",
    fragmentation: str | None = None,
    instrument: str = "LUMOS",
    client: PredictClient | None = None,
) -> tuple[list[PrecursorSpec], dict[str, np.ndarray]]:
    """Predict fragment intensities for a precursor grid.

    Returns ``(specs_actually_sent, raw_response)`` — the first element
    matters because ``on_unsupported="skip"`` drops precursors the model
    can't handle, and the response rows align with what was *sent*, not
    with what was asked for.
    """
    overlay = ms2_model(model)
    cl = client or make_client(model, server=server)
    declared = set(cl.model_inputs)

    if on_unsupported == "skip":
        kept: list[PrecursorSpec] = []
        for spec in specs:
            try:
                _validate_specs([spec], overlay)
            except KoinaInputError:
                continue
            kept.append(spec)
        specs = kept
    if not specs:
        raise KoinaInputError("no precursors survived pre-flight validation")

    modseqs = _validate_specs(specs, overlay)
    charges = np.array([s.charge for s in specs], dtype=np.int32).reshape(-1, 1)

    arrays: dict[str, np.ndarray] = {
        "peptide_sequences": np.array(modseqs, dtype=object).reshape(-1, 1),
        "precursor_charges": charges,
    }
    if "collision_energies" in declared:
        if collision_energy is None:
            raise KoinaInputError(
                f"{model} requires a collision energy; pass collision_energy=..."
            )
        arrays["collision_energies"] = adjust_nce(
            collision_energy, charges, enabled=adjust_nce_for_dia
        )
    elif collision_energy is not None:
        raise KoinaInputError(
            f"{model} does not accept a collision energy — it declares inputs "
            f"{sorted(declared)}. Supplying one would be silently ignored and "
            f"every energy would yield an identical prediction."
        )

    # Categorical inputs the PTM-aware models require and the 2020 series
    # does not declare. Supplied only when the server asks for them, so a
    # model that ignores them never receives one.
    if "fragmentation_types" in declared:
        frag = fragmentation or overlay.fragmentation
        arrays["fragmentation_types"] = np.array(
            [frag] * len(specs), dtype=object).reshape(-1, 1)
    if "instrument_types" in declared:
        arrays["instrument_types"] = np.array(
            [instrument] * len(specs), dtype=object).reshape(-1, 1)

    return specs, cl.predict(arrays, min_intensity=min_intensity)


def predict_rt(
    modseqs: Sequence[str],
    *,
    model: str = "Chronologer_RT",
    server: str | None = None,
    client: PredictClient | None = None,
) -> dict[str, float]:
    """Predict retention time, returned in **seconds**.

    The output column name (``rt`` vs ``irt``) is read from the server's
    declared outputs rather than hardcoded; the unit conversion comes
    from the local overlay, since the server reports neither unit nor
    scale.
    """
    overlay: RtModel = rt_model(model)
    cl = client or make_client(model, server=server)
    unique = list(dict.fromkeys(modseqs))
    response = cl.predict(
        {"peptide_sequences": np.array(unique, dtype=object).reshape(-1, 1)}
    )

    outputs = list(cl.model_outputs)
    column = next((c for c in ("rt", "irt") if c in outputs), None)
    if column is None:
        if len(outputs) != 1:
            raise KoinaInputError(
                f"cannot identify the RT output of {model}; it declares "
                f"{outputs}"
            )
        column = outputs[0]

    values = np.asarray(response[column], dtype=np.float64).ravel()
    return {
        m: float(v) * overlay.to_seconds
        for m, v in zip(unique, values.tolist(), strict=True)
    }


def predict_library(
    *,
    specs: Sequence[PrecursorSpec] | None = None,
    fasta: Path | str | None = None,
    ms2_model_name: str = "Prosit_2020_intensity_HCD",
    rt_model_name: str | None = "Chronologer_RT",
    collision_energy: float | None = 30.0,
    adjust_nce_for_dia: bool = True,
    server: str | None = None,
    min_intensity: float = 1e-4,
    on_unsupported: str = "error",
    fragmentation: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    ms2_client: PredictClient | None = None,
    rt_client: PredictClient | None = None,
    **digest_kwargs: Any,
) -> tuple[Library, AssemblyStats]:
    """FASTA (or an explicit precursor grid) → a predicted ``Library``."""
    if (specs is None) == (fasta is None):
        raise ValueError("pass exactly one of specs= or fasta=")
    if specs is None:
        specs = precursors_from_fasta(fasta, **digest_kwargs)
    if not specs:
        raise ValueError("no precursors to predict")

    overlay = ms2_model(ms2_model_name)
    sent, response = predict_fragments(
        specs,
        model=ms2_model_name,
        collision_energy=collision_energy,
        adjust_nce_for_dia=adjust_nce_for_dia,
        server=server,
        min_intensity=min_intensity,
        on_unsupported=on_unsupported,
        fragmentation=fragmentation,
        client=ms2_client,
    )

    rt_seconds: dict[str, float] = {}
    if rt_model_name:
        rt_seconds = predict_rt(
            [s.modified_sequence for s in sent],
            model=rt_model_name,
            server=server,
            client=rt_client,
        )

    meta = {
        "x.koina.ms2_model": ms2_model_name,
        "x.koina.rt_model": rt_model_name or "",
        "x.koina.collision_energy": collision_energy,
        "x.koina.adjust_nce_for_dia": adjust_nce_for_dia,
        "x.koina.rt_scale": rt_model(rt_model_name).scale if rt_model_name else "",
        **(metadata or {}),
    }
    return assemble_library(
        sent,
        response,
        ms2_model=overlay,
        rt_seconds=rt_seconds,
        min_intensity=min_intensity,
        metadata=meta,
    )


__all__ = [
    "predict",
    "predict_fragments",
    "predict_library",
    "predict_rt",
]
