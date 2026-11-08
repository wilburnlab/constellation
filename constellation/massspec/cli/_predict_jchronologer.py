"""``massspec predict-library --backend jchronologer``: JChronologer's models onto a library.

Each property's model can be swapped (``--ms2-model``, ``--rt-model``,
``--charge-model``, ``--ccs-model``: ``backend:model`` or ``none``), so
JChronologer's fragments can go beside Koina's retention times, or any
other mix the registry (``library.prediction.registry``) can build.

The library comes from ``massspec search-space`` (``--from-library``) or a
peptide list (``--peptides``); a FASTA is refused, since the search space
is the package's one FASTA enumerator and keeps how each precursor was
made. Predictions go onto that library (``library.prediction.predict_onto``)
and it is written to ``<output-dir>/library_pqdir`` with its fragments
streamed to disk, beside a manifest and a ``_SUCCESS`` marker carrying the
run's stage key.

The key (``core.stages.stage_key``) covers the upstream library (its own
recorded key, else the SHA-256 of its files), each predictor's identity
(model files and their SHA-256, batch size, minimum intensity), the
collision energy per charge and the settings that change the result.
Threads and device are recorded, not keyed: the runner's output does not
depend on the thread count. An export (``--output-dlib``) is not keyed
either: the manifest records which prediction each export holds
(``exports``), so ``--resume`` writes one that is missing or holds an
earlier prediction, and a run replaces its own exports and no other file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

#: Koina's flags that mean nothing here (a refused precursor is always
#: dropped), with their parser defaults.
KOINA_ONLY_ARGS: tuple[tuple[str, str, object], ...] = (
    ("on_unsupported_mod", "--on-unsupported-mod", "error"),
)
#: Flags only this backend implements, with their parser defaults.
JCHRONOLOGER_ONLY_ARGS: tuple[tuple[str, str, object], ...] = (
    ("no_charge_states", "--no-charge-states", False),
    ("no_ccs", "--no-ccs", False),
    ("charge_model", "--charge-model", None),
    ("ccs_model", "--ccs-model", None),
    ("min_charge_probability", "--min-charge-probability", None),
    ("threads", "--threads", None),
    ("batch_size", "--batch-size", None),
    ("device", "--device", "cpu"),
)
#: Each property's flag and the model this backend uses when it is not given.
MODEL_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("ms2_model", "fragments", "jchronologer:cartographer"),
    ("rt_model", "retention_time", "jchronologer:chronologer"),
    ("charge_model", "charge_state", "jchronologer:electrician"),
    ("ccs_model", "ccs", "jchronologer:sculptor"),
)
#: ``--min-intensity`` when not given: each backend's own floor.
DEFAULT_MIN_INTENSITY = {"jchronologer": 0.01, "koina": 1e-4}


def used_flags(args: argparse.Namespace, table) -> list[str]:
    return [flag for attr, flag, default in table if getattr(args, attr, default) != default]


def model_specs(args: argparse.Namespace) -> dict[str, str]:
    """``{property: "backend:model"}`` for a run: each property's flag
    (``backend:model`` or ``none``), JChronologer's model where none is
    given, less what ``--no-charge-states`` / ``--no-ccs`` turn off."""
    out: dict[str, str] = {}
    for attr, prop, own in MODEL_FLAGS:
        value = getattr(args, attr, None)
        flag = "--" + attr.replace("_", "-")
        if value is None:
            out[prop] = own
        elif str(value).lower() == "none":
            continue
        elif ":" not in str(value):
            raise ValueError(
                f"{flag} takes backend:model under --backend jchronologer (e.g. koina:{value}, {own}) or none"
            )
        else:
            out[prop] = str(value)
    for off, prop, flag in (("no_charge_states", "charge_state", "--charge-model"), ("no_ccs", "ccs", "--ccs-model")):
        if getattr(args, off, False):
            if getattr(args, flag[2:].replace("-", "_"), None) not in (None, "none"):
                raise ValueError(f"--{off.replace('_', '-')} and {flag} contradict each other")
            out.pop(prop, None)
    if not out:
        raise ValueError("every property is turned off; nothing to predict")
    return out


def add_jchronologer_arguments(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group(
        "jchronologer backend",
        "Only with --backend jchronologer: JChronologer's models (EncyclopeDIA's predictor) "
        "read from the installed EncyclopeDIA and run in PyTorch, onto a search space.",
    )
    group.add_argument("--no-charge-states", action="store_true",
                       help="do not predict charge-state probabilities (Electrician)")
    group.add_argument("--no-ccs", action="store_true",
                       help="do not predict CCS (Sculptor, which has no TMT tokens: TMT peptides need this)")
    group.add_argument(
        "--min-charge-probability", type=float, default=None, metavar="P",
        help=(
            "keep a precursor, target or decoy, only if Electrician gives its charge at least P, "
            "each judged on its own (default: keep every charge)"
        ),
    )
    group.add_argument(
        "--charge-model", default=None, metavar="BACKEND:MODEL",
        help="charge-state model, or none (default: jchronologer:electrician)",
    )
    group.add_argument(
        "--ccs-model", default=None, metavar="BACKEND:MODEL",
        help="CCS model, or none (default: jchronologer:sculptor)",
    )
    group.add_argument("--threads", type=int, default=None,
                       help="batches run at once, one inference thread each (default: all cores)")
    group.add_argument("--batch-size", type=int, default=None,
                       help="requests per model call (default: JChronologer's 2048); changes values at float32 noise")
    group.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="torch device (default: cpu)")


def _library_key(path: Path) -> dict[str, object]:
    """What identifies an upstream library: its own stage key if it recorded
    one, else the SHA-256 of each of its files."""
    from constellation.core.stages import file_sha256, recorded_key

    recorded = recorded_key(path) if path.is_dir() else None
    if recorded:
        return {"stage_key": recorded}
    files = sorted(p for p in (path.rglob("*") if path.is_dir() else [path]) if p.is_file() and p.name != "_SUCCESS")
    return {"files": {str(p.relative_to(path) if path.is_dir() else p.name): file_sha256(p) for p in files}}


def recorded_exports(output_dir: Path) -> dict[str, str | None]:
    """The library files a prediction directory's runs exported
    (``--output-dlib``), with the stage key each was written for (None
    where the run did not record it), read from the manifests in it:
    this backend's ``exports``, and the Koina backend's ``runs`` and
    per-energy manifests."""
    found: dict[str, str | None] = {}
    manifests = [output_dir / "manifest.json", *sorted(output_dir.glob("ce_*/manifest.json"))]
    for path in manifests:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        outputs = manifest.get("outputs")
        if isinstance(outputs, dict) and outputs.get("dlib"):
            found.setdefault(str(outputs["dlib"]), None)
        for item in [manifest, *(manifest.get("runs") or [])]:
            if isinstance(item, dict) and item.get("output_dlib"):
                found.setdefault(str(item["output_dlib"]), None)
        exports = manifest.get("exports")
        if isinstance(exports, dict):
            found.update({str(k): v for k, v in exports.items()})
    return found


def foreign_export(path: Path | None, output_dir: Path, recorded: dict[str, str | None]) -> str | None:
    """Why ``path`` may not be written, or None. A prediction replaces its
    own exports (ones its manifests record, or inside its directory) and
    nothing else; asked before anything is predicted, so a run is not
    lost to a name clash at its end."""
    if path is None or not path.exists():
        return None
    if str(path) in recorded or path.is_relative_to(output_dir):
        return None
    return f"{path} exists and was not exported from {output_dir}; delete it or choose another --output-dlib"


def _peptide_list_library(path: Path, charges: tuple[int, ...]):
    from constellation.massspec.library.digest import precursors_from_peptide_list
    from constellation.massspec.library.library import assign_ids

    specs = precursors_from_peptide_list(path, charges=charges)
    peptides = list(dict.fromkeys(s.modified_sequence for s in specs))
    return assign_ids(
        proteins=[],
        peptides=[{"modified_sequence": s} for s in peptides],
        precursors=[
            {"modified_sequence": s.modified_sequence, "charge": s.charge, "precursor_mz": s.precursor_mz}
            for s in specs
        ],
    )


def run(args: argparse.Namespace) -> int:
    from constellation import __version__ as constellation_version
    from constellation.core.stages import invalidate, is_done, mark_done, recorded_key, stage_key
    from constellation.massspec.cli.predict_library import (
        _ENCYCLOPEDIA_ONLY_ARGS,
        _PTM_NAMES,
        _camel_to_kebab,
        _energy_table,
        _parse_collision_energies,
        _ptm_default_for,
        _scalar_energy,
    )
    from constellation.massspec.library import load_library, save_library
    from constellation.massspec.library.prediction import CollisionEnergy
    from constellation.massspec.library.prediction.run import predict_onto
    from constellation.massspec.search.encyclopedia._common import build_manifest_envelope

    def fail(message: str, code: int = 1) -> int:
        print(f"error: {message}", file=sys.stderr)
        return code

    others = used_flags(args, _ENCYCLOPEDIA_ONLY_ARGS) + used_flags(args, KOINA_ONLY_ARGS)
    others += [
        f"--ptm-{_camel_to_kebab(name)}"
        for name in _PTM_NAMES
        if getattr(args, f"ptm_{_camel_to_kebab(name).replace('-', '_')}", "off") != _ptm_default_for(name)
    ]
    if others:
        return fail(f"these flags belong to another backend and have no effect with --backend jchronologer: {', '.join(others)}")
    if args.fasta is not None:
        return fail(
            "--backend jchronologer predicts a search space, not a FASTA: run "
            "`constellation massspec search-space --fasta ... -o DIR` and pass --from-library DIR"
        )
    if (args.from_library is None) == (args.peptides is None):
        return fail("pass exactly one of --from-library (a search space) or --peptides")
    if args.output_library is not None:
        return fail("--output-library is not used with --backend jchronologer; the library is <output-dir>/library_pqdir")
    try:
        specs = model_specs(args)
    except ValueError as exc:
        return fail(str(exc))
    uses_koina = any(spec.startswith("koina:") for spec in specs.values())
    if args.koina_url is not None and not uses_koina:
        return fail("--koina-url has no effect: no koina:<model> is chosen")
    if args.min_charge_probability is not None:
        if "charge_state" not in specs:
            return fail("--min-charge-probability needs charge states; drop --no-charge-states")
        if not 0.0 <= args.min_charge_probability <= 1.0:
            return fail("--min-charge-probability must be between 0 and 1")

    try:
        energies = _parse_collision_energies(args.collision_energy)
        table = _energy_table(args)
    except ValueError as exc:
        return fail(str(exc))
    if len(energies) > 1:
        return fail("--backend jchronologer predicts at one collision energy per run; give one value")
    if table is not None:
        energy = CollisionEnergy.explicit(table)
    else:
        energy = _scalar_energy(float(energies[0] if energies[0] is not None else args.default_nce), args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── the library to predict ───────────────────────────────────────
    try:
        if args.from_library is not None:
            source = Path(args.from_library).resolve()
            library = load_library(source)
        else:
            source = Path(args.peptides).resolve()
            library = _peptide_list_library(source, tuple(range(args.min_charge, args.max_charge + 1)))
    except (OSError, ValueError) as exc:
        return fail(f"could not read the precursors: {exc}")
    if library.precursors.num_rows == 0:
        return fail("no precursors to predict")
    charges = sorted(set(library.precursors.column("charge").to_pylist()))
    if table is not None:
        missing = sorted(set(charges) - set(table))
        if missing:
            return fail(
                f"--collision-energy-by-charge has no energy for charge {', '.join(map(str, missing))}, "
                "which the library's precursors include"
            )

    # ── predictors ──────────────────────────────────────────────────
    import os

    from constellation.massspec.library.jchronologer import locate_bundle
    from constellation.massspec.library.prediction.registry import predictor
    from constellation.massspec.library.prediction.torch_runner import DEFAULT_BATCH_SIZE
    from constellation.massspec.tokenize.jchronologer import UnknownModelError
    from constellation.thirdparty.registry import ToolNotFoundError

    bundle = None
    if any(spec.startswith("jchronologer:") for spec in specs.values()):
        try:
            bundle = locate_bundle()
        except (ToolNotFoundError, FileNotFoundError, UnknownModelError) as exc:
            return fail(f"JChronologer is not available: {exc}")
    threads = args.threads or os.cpu_count() or 1
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE
    predictors = {}
    for prop, spec in specs.items():
        backend = spec.partition(":")[0]
        options: dict = (
            {"bundle": bundle, "threads": threads, "batch_size": batch_size, "device": args.device}
            if backend == "jchronologer"
            else {"server": args.koina_url}
            if backend == "koina"
            else {}
        )
        if prop == "fragments":
            options["min_intensity"] = (
                args.min_intensity if args.min_intensity is not None else DEFAULT_MIN_INTENSITY.get(backend, 0.01)
            )
        try:
            predictors[prop] = predictor(spec, property=prop, **options)
        except ValueError as exc:
            return fail(str(exc))
        except Exception as exc:  # a Koina model the server does not serve, or no network
            return fail(f"could not set up {spec}: {exc}")
    fragments = predictors.get("fragments")
    if fragments is not None and not fragments.capabilities.needs_collision_energy:
        if table is not None or len(energies) and energies[0] is not None:
            return fail(f"{fragments.name} takes no collision energy; drop the energy flags")
        energy = None

    from constellation.massspec.library.prediction import PREDICTION_LAYOUT

    key = stage_key(
        stage="massspec predict-library jchronologer",
        layout=PREDICTION_LAYOUT,
        library=_library_key(source),
        peptide_charges=None if args.from_library is not None else [args.min_charge, args.max_charge],
        predictors={prop: p.identity() for prop, p in predictors.items()},
        collision_energy=energy.describe(charges) if energy is not None else None,
        min_charge_probability=args.min_charge_probability,
    )
    pqdir = output_dir / "library_pqdir"
    dlib = Path(args.output_dlib).resolve() if args.output_dlib is not None else None
    exports = recorded_exports(output_dir)
    if is_done(output_dir, key):
        if not args.resume:
            return fail(f"{output_dir} already holds this prediction; pass --resume to reuse it, or delete it")
        if dlib is not None and (exports.get(str(dlib)) != key or not dlib.is_file()):
            # The prediction is done; the export asked for now is not (a
            # new --output-dlib, or a file of an earlier prediction).
            refusal = foreign_export(dlib, output_dir, exports)
            if refusal:
                return fail(refusal)
            _export_dlib(pqdir, dlib)
            _record_export(output_dir, dlib, key)
            print(f"exported {dlib} from the completed prediction")
        print(f"already complete: {output_dir}")
        return 0
    if recorded_key(output_dir) is not None and not args.resume:
        return fail(
            f"{output_dir} holds a prediction of other inputs or settings; pick another --output-dir, "
            "or pass --resume to replace it"
        )
    refusal = foreign_export(dlib, output_dir, exports)
    if refusal:
        return fail(refusal)
    invalidate(output_dir)

    # ── predict, streaming fragments ────────────────────────────────
    import contextlib

    import pyarrow.parquet as pq

    from constellation.massspec.library.schemas import LIBRARY_FRAGMENT_TABLE

    pqdir.mkdir(parents=True, exist_ok=True)
    # Predicted fragments go to disk chunk by chunk, beside the library;
    # the library is then written (with an empty fragment table) and the
    # streamed file put in its place. Without a fragment model the
    # library keeps the fragments it came with, and they are written with it.
    streamed = pqdir / "fragments.streaming.parquet" if "fragments" in predictors else None
    started = time.perf_counter()
    try:
        with (
            pq.ParquetWriter(streamed, LIBRARY_FRAGMENT_TABLE) if streamed is not None else contextlib.nullcontext()
        ) as writer:
            predicted, report = predict_onto(
                library,
                predictors,
                collision_energy=energy,
                min_charge_probability=args.min_charge_probability,
                fragments_sink=writer.write_table if writer is not None else None,
            )
    except BaseException as exc:
        if streamed is not None:
            streamed.unlink(missing_ok=True)
        if isinstance(exc, ValueError):
            return fail(str(exc))
        raise
    save_library(predicted, pqdir, format="parquet_dir")
    if streamed is not None:
        streamed.replace(pqdir / "fragments.parquet")
    if dlib is not None:
        _export_dlib(pqdir, dlib)
    elapsed = time.perf_counter() - started

    import torch

    manifest = build_manifest_envelope(
        subcommand="massspec predict-library",
        constellation_version=constellation_version,
        constellation_argv=sys.argv,
        java_argv=None,
        tool={
            "name": "jchronologer",
            "version": bundle.version,
            "jar_path": str(bundle.jar),
            "jar_sha256": bundle.jar_sha256,
        }
        if bundle is not None
        else {"name": "koina"},
        inputs={"peptides": source} if source.is_file() else {},
        outputs={"library_pqdir": pqdir, "dlib": dlib},
        runtime={
            "elapsed_seconds": round(elapsed, 3),
            "threads": threads,
            "batch_size": batch_size,
            "device": args.device,
            "torch": torch.__version__,
        },
        extras={
            "backend": "jchronologer",
            "source": str(source),
            "stage_key": key,
            "library": _library_key(source),
            "prediction": report.to_dict(),
            # Every library file exported from this directory, with the
            # prediction (stage key) it holds; earlier ones are kept so
            # they stay this directory's to replace.
            "exports": {**exports, **({str(dlib): key} if dlib is not None else {})},
        },
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    mark_done(output_dir, key)

    if not args.no_progress:
        _print_summary(report, energy, charges, output_dir)
    return 0


def _export_dlib(pqdir: Path, dlib: Path) -> None:
    """The library as an EncyclopeDIA ``.dlib`` (its targets: the writer
    leaves decoys out), replacing an earlier export at that path."""
    from constellation.massspec.library import load_library, save_library

    dlib.parent.mkdir(parents=True, exist_ok=True)
    save_library(load_library(pqdir), dlib, format="encyclopedia.dlib", overwrite=True)


def _record_export(output_dir: Path, dlib: Path, key: str) -> None:
    """Add an export made from a completed prediction to its manifest."""
    path = output_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.setdefault("outputs", {})["dlib"] = str(dlib)
    manifest["exports"] = {**(manifest.get("exports") or {}), str(dlib): key}
    path.write_text(json.dumps(manifest, indent=2, default=str))


def _print_summary(report, energy, charges, output_dir: Path) -> None:
    table = (
        "NCE by charge: " + ", ".join(f"{z}+ {e:g}" for z, e in energy.by_charge(charges).items())
        if energy is not None
        else "no collision energy"
    )
    print(
        f"predicted {report.precursors_kept} of {report.precursors} precursors "
        f"({report.requests_predicted} requests, {report.fragments} fragments); {table}"
    )
    for prop, identity in report.predictors.items():
        print(f"  {prop}: {identity.get('backend')}:{identity.get('model')}")
    for prop, reasons in report.refused.items():
        for reason, n in reasons.items():
            shown = "; ".join(report.examples.get(prop, {}).get(reason, [])[:3])
            print(f"  refused by {prop} ({reason}): {n} precursors, e.g. {shown}")
    if report.dropped:
        print(f"  dropped: {report.dropped_targets} target and {report.dropped_decoys} decoy precursors")
    ccs = report.examples.get("ccs", {}).get("unsupported_modification", [])
    if any("UNIMOD:737" in d or "UNIMOD:739" in d for d in ccs):
        print("  Sculptor has no TMT tokens: pass --no-ccs to keep TMT precursors")
    for caveat in report.caveats:
        print(f"  note: {caveat}")
    print(f"library: {output_dir / 'library_pqdir'}")
