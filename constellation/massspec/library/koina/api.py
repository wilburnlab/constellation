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
from constellation.massspec.library.library import Library
from constellation.massspec.library.prediction.collision_energy import (
    DEFAULT_REFERENCE_CHARGE,
    CollisionEnergy,
)
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
                    allow_n_term_mods=model.supports_n_term_mods,
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


def collision_energy_for(
    collision_energy: float | CollisionEnergy | None,
    *,
    adjust_nce_for_dia: bool = True,
    nce_reference_charge: int = DEFAULT_REFERENCE_CHARGE,
) -> CollisionEnergy | None:
    """The per-charge energy a call asks for. A :class:`CollisionEnergy`
    is taken as it is; a number is scaled from ``nce_reference_charge``
    by EncyclopeDIA's rule, or used for every charge when
    ``adjust_nce_for_dia`` is false."""
    if collision_energy is None or isinstance(collision_energy, CollisionEnergy):
        return collision_energy
    if adjust_nce_for_dia:
        return CollisionEnergy.scaled_from_reference(float(collision_energy), nce_reference_charge)
    return CollisionEnergy.uniform(float(collision_energy))


def predict_fragments(
    specs: Sequence[PrecursorSpec],
    *,
    model: str = "Prosit_2020_intensity_HCD",
    collision_energy: float | CollisionEnergy | None = None,
    adjust_nce_for_dia: bool = True,
    nce_reference_charge: int = DEFAULT_REFERENCE_CHARGE,
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

    ``collision_energy`` is a :class:`CollisionEnergy` (an energy per
    charge) or a number, which ``adjust_nce_for_dia`` and
    ``nce_reference_charge`` turn into one (:func:`collision_energy_for`);
    those two do not apply to a :class:`CollisionEnergy`.
    """
    energy = collision_energy_for(
        collision_energy,
        adjust_nce_for_dia=adjust_nce_for_dia,
        nce_reference_charge=nce_reference_charge,
    )
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
        if energy is None:
            raise KoinaInputError(
                f"{model} requires a collision energy; pass collision_energy=..."
            )
        try:
            arrays["collision_energies"] = energy.resolve(charges)
        except ValueError as exc:
            raise KoinaInputError(str(exc)) from exc
    elif energy is not None:
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
    library: Library | None = None,
    ms2_model_name: str = "Prosit_2020_intensity_HCD",
    rt_model_name: str | None = "Chronologer_RT",
    collision_energy: float | CollisionEnergy | None = 30.0,
    adjust_nce_for_dia: bool = True,
    nce_reference_charge: int = DEFAULT_REFERENCE_CHARGE,
    server: str | None = None,
    min_intensity: float = 1e-4,
    on_unsupported: str = "error",
    fragmentation: str | None = None,
    instrument: str = "LUMOS",
    metadata: Mapping[str, Any] | None = None,
    ms2_client: PredictClient | None = None,
    rt_client: PredictClient | None = None,
    **digest_kwargs: Any,
) -> tuple[Library, AssemblyStats]:
    """FASTA (or an explicit precursor grid, or a library) → a predicted
    ``Library``.

    Given ``library``, its own precursors are predicted through
    :func:`~constellation.massspec.library.prediction.predict_onto` with
    :class:`~constellation.massspec.library.koina.predictor.KoinaFragments`
    (and :class:`KoinaRetentionTime`): isotope labels are taken off for the
    request and the m/z come from each precursor's own peptide; its decoys,
    spans and modification provenance survive. With ``on_unsupported=
    "error"`` a precursor the model cannot take raises before any request;
    with ``"skip"`` it is left out with its target/decoy partners and
    counted (``x.koina.precursors_not_predicted``). A fragment the model
    names that the peptide cannot have is left out and counted
    (``n_unparseable_annotations``).

    The energy each charge was predicted at is recorded as
    ``x.massspec.prediction.collision_energy``
    (:meth:`CollisionEnergy.describe` over the charges predicted), absent
    for a model that takes no energy."""
    if sum(x is not None for x in (specs, fasta, library)) != 1:
        raise ValueError("pass exactly one of specs=, fasta= or library=")
    if library is not None:
        if digest_kwargs:
            raise TypeError(f"unexpected keyword argument(s) {sorted(digest_kwargs)} with library=")
        return _predict_onto_library(
            library,
            ms2_model_name=ms2_model_name,
            rt_model_name=rt_model_name,
            energy=collision_energy_for(
                collision_energy, adjust_nce_for_dia=adjust_nce_for_dia, nce_reference_charge=nce_reference_charge
            ),
            server=server,
            min_intensity=min_intensity,
            on_unsupported=on_unsupported,
            fragmentation=fragmentation,
            instrument=instrument,
            metadata=metadata,
            ms2_client=ms2_client,
            rt_client=rt_client,
        )
    if specs is None:
        specs = precursors_from_fasta(fasta, **digest_kwargs)
    elif digest_kwargs:
        # digest options only mean something on the fasta= path. Silently
        # absorbing them let instrument= disappear into **digest_kwargs
        # and the model receive the LUMOS default instead.
        raise TypeError(
            f"unexpected keyword argument(s) {sorted(digest_kwargs)} — digest "
            f"options apply only when predicting from fasta=, not from specs="
        )
    if not specs:
        raise ValueError("no precursors to predict")

    overlay = ms2_model(ms2_model_name)
    energy = collision_energy_for(
        collision_energy,
        adjust_nce_for_dia=adjust_nce_for_dia,
        nce_reference_charge=nce_reference_charge,
    )
    sent, response = predict_fragments(
        specs,
        model=ms2_model_name,
        collision_energy=energy,
        server=server,
        min_intensity=min_intensity,
        on_unsupported=on_unsupported,
        fragmentation=fragmentation,
        instrument=instrument,
        client=ms2_client,
    )

    rt_seconds: dict[str, float] = {}
    if rt_model_name:
        # Koina's RT models read the same dialect as its fragment models;
        # the seconds are keyed back by the spec's own sequence.
        from constellation.massspec.library.koina.predictor import koina_sequence
        from constellation.massspec.tokenize.vocabulary import Sites

        on_residue = rt_model(rt_model_name).pyro_glu_on_first_residue
        koina_form = {
            s.modified_sequence: koina_sequence(
                Sites.from_peptidoform(s.modified_sequence), pyro_glu_on_first_residue=on_residue
            )
            for s in sent
        }
        by_koina = predict_rt(list(dict.fromkeys(koina_form.values())), model=rt_model_name, server=server, client=rt_client)
        rt_seconds = {m: by_koina[k] for m, k in koina_form.items()}

    meta = {
        "x.koina.ms2_model": ms2_model_name,
        "x.koina.rt_model": rt_model_name or "",
        "x.koina.collision_energy": energy.nce if energy is not None else None,
        "x.koina.instrument": instrument,
        "x.koina.adjust_nce_for_dia": energy is not None and energy.origin == "scaled_from_reference",
        "x.koina.rt_scale": rt_model(rt_model_name).scale if rt_model_name else "",
        **(metadata or {}),
    }
    if energy is not None:
        meta["x.massspec.prediction.collision_energy"] = energy.describe(
            sorted({s.charge for s in sent})
        )
    return assemble_library(
        sent,
        response,
        ms2_model=overlay,
        rt_seconds=rt_seconds,
        min_intensity=min_intensity,
        metadata=meta,
    )


def _predict_onto_library(
    library: Library,
    *,
    ms2_model_name: str,
    rt_model_name: str | None,
    energy: CollisionEnergy | None,
    server: str | None,
    min_intensity: float,
    on_unsupported: str,
    fragmentation: str | None,
    instrument: str,
    metadata: Mapping[str, Any] | None,
    ms2_client: PredictClient | None,
    rt_client: PredictClient | None,
) -> tuple[Library, AssemblyStats]:
    """``predict_library(library=...)``: the library's own precursors,
    through the prediction contract."""
    from constellation.massspec.library.koina.predictor import KoinaFragments, KoinaRetentionTime
    from constellation.massspec.library.prediction.run import PredictionRefused, predict_onto

    if on_unsupported not in ("error", "skip"):
        raise ValueError(f"on_unsupported must be 'error' or 'skip', not {on_unsupported!r}")
    fragments = KoinaFragments(
        ms2_model_name, server=server, client=ms2_client, min_intensity=min_intensity,
        fragmentation=fragmentation, instrument=instrument,
    )
    takes_energy = fragments.capabilities.needs_collision_energy
    if takes_energy and energy is None:
        raise KoinaInputError(f"{ms2_model_name} requires a collision energy; pass collision_energy=...")
    if not takes_energy and energy is not None:
        raise KoinaInputError(
            f"{ms2_model_name} does not accept a collision energy — it declares inputs "
            f"{sorted(fragments.declared)}. Supplying one would be silently ignored and "
            f"every energy would yield an identical prediction."
        )
    predictors: dict[str, Any] = {"fragments": fragments}
    if rt_model_name:
        predictors["retention_time"] = KoinaRetentionTime(rt_model_name, server=server, client=rt_client)
    try:
        predicted, report = predict_onto(library, predictors, collision_energy=energy, strict=on_unsupported == "error")
    except PredictionRefused as exc:
        raise KoinaInputError(f"{exc}\nFilter them out, or pass on_unsupported='skip'.") from exc
    except ValueError as exc:
        raise KoinaInputError(str(exc)) from exc
    meta = {
        "x.koina.ms2_model": ms2_model_name,
        "x.koina.rt_model": rt_model_name or "",
        "x.koina.collision_energy": energy.nce if energy is not None else None,
        "x.koina.instrument": instrument,
        "x.koina.adjust_nce_for_dia": energy is not None and energy.origin == "scaled_from_reference",
        "x.koina.rt_scale": rt_model(rt_model_name).scale if rt_model_name else "",
        "x.koina.precursors_not_predicted": report.dropped,
        **(metadata or {}),
    }
    stats = AssemblyStats(
        n_precursors=report.precursors_kept,
        n_fragments=report.fragments,
        n_unparseable_annotations=fragments.unparsed,
        max_abs_ppm_deviation=report.max_reported_mz_error_ppm or 0.0,
    )
    return predicted.with_metadata(meta), stats


__all__ = [
    "collision_energy_for",
    "predict",
    "predict_fragments",
    "predict_library",
    "predict_rt",
]
