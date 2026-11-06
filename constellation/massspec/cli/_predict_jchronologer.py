"""``massspec predict-library --backend jchronologer``: JChronologer's models onto a library.

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
depend on the thread count.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

#: Flags only another backend implements, with their parser defaults.
KOINA_ONLY_ARGS: tuple[tuple[str, str, object], ...] = (
    ("ms2_model", "--ms2-model", "Prosit_2020_intensity_HCD"),
    ("rt_model", "--rt-model", "Chronologer_RT"),
    ("koina_url", "--koina-url", None),
    ("on_unsupported_mod", "--on-unsupported-mod", "error"),
)
#: Flags only this backend implements, with their parser defaults.
JCHRONOLOGER_ONLY_ARGS: tuple[tuple[str, str, object], ...] = (
    ("no_charge_states", "--no-charge-states", False),
    ("no_ccs", "--no-ccs", False),
    ("min_charge_probability", "--min-charge-probability", None),
    ("threads", "--threads", None),
    ("batch_size", "--batch-size", None),
    ("device", "--device", "cpu"),
)
#: ``--min-intensity`` when not given: each backend's own floor.
DEFAULT_MIN_INTENSITY = {"jchronologer": 0.01, "koina": 1e-4}


def used_flags(args: argparse.Namespace, table) -> list[str]:
    return [flag for attr, flag, default in table if getattr(args, attr, default) != default]


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
            "keep a target precursor only if Electrician gives its charge at least P; its decoys "
            "follow it (default: keep every charge)"
        ),
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
    if args.min_charge_probability is not None:
        if args.no_charge_states:
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

    try:
        bundle = locate_bundle()
    except (ToolNotFoundError, FileNotFoundError, UnknownModelError) as exc:
        return fail(f"JChronologer is not available: {exc}")
    threads = args.threads or os.cpu_count() or 1
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE
    runtime = {"bundle": bundle, "threads": threads, "batch_size": batch_size, "device": args.device}
    min_intensity = args.min_intensity if args.min_intensity is not None else DEFAULT_MIN_INTENSITY["jchronologer"]
    predictors = {
        "fragments": predictor("jchronologer:cartographer", min_intensity=min_intensity, **runtime),
        "retention_time": predictor("jchronologer:chronologer", **runtime),
    }
    if not args.no_charge_states:
        predictors["charge_state"] = predictor("jchronologer:electrician", **runtime)
    if not args.no_ccs:
        predictors["ccs"] = predictor("jchronologer:sculptor", **runtime)

    key = stage_key(
        stage="massspec predict-library jchronologer",
        library=_library_key(source),
        peptide_charges=None if args.from_library is not None else [args.min_charge, args.max_charge],
        predictors={prop: p.identity() for prop, p in predictors.items()},
        collision_energy=energy.describe(charges),
        min_charge_probability=args.min_charge_probability,
    )
    if is_done(output_dir, key):
        if not args.resume:
            return fail(f"{output_dir} already holds this prediction; pass --resume to reuse it, or delete it")
        print(f"already complete: {output_dir}")
        return 0
    if recorded_key(output_dir) is not None and not args.resume:
        return fail(
            f"{output_dir} holds a prediction of other inputs or settings; pick another --output-dir, "
            "or pass --resume to replace it"
        )
    invalidate(output_dir)

    # ── predict, streaming fragments ────────────────────────────────
    import pyarrow.parquet as pq

    from constellation.massspec.library.schemas import LIBRARY_FRAGMENT_TABLE

    pqdir = output_dir / "library_pqdir"
    pqdir.mkdir(parents=True, exist_ok=True)
    # Fragments go to disk chunk by chunk, beside the library; the library
    # is then written (with an empty fragment table) and the streamed file
    # put in its place.
    streamed = pqdir / "fragments.streaming.parquet"
    started = time.perf_counter()
    try:
        with pq.ParquetWriter(streamed, LIBRARY_FRAGMENT_TABLE) as writer:
            predicted, report = predict_onto(
                library,
                predictors,
                collision_energy=energy,
                min_charge_probability=args.min_charge_probability,
                fragments_sink=writer.write_table,
            )
    except ValueError as exc:
        streamed.unlink(missing_ok=True)
        return fail(str(exc))
    save_library(predicted, pqdir, format="parquet_dir")
    streamed.replace(pqdir / "fragments.parquet")
    dlib = None
    if args.output_dlib is not None:
        dlib = Path(args.output_dlib).resolve()
        dlib.parent.mkdir(parents=True, exist_ok=True)
        save_library(load_library(pqdir), dlib, format="encyclopedia.dlib")
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
        },
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
        },
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    mark_done(output_dir, key)

    if not args.no_progress:
        _print_summary(report, energy, charges, output_dir)
    return 0


def _print_summary(report, energy, charges, output_dir: Path) -> None:
    table = ", ".join(f"{z}+ {e:g}" for z, e in energy.by_charge(charges).items())
    print(
        f"predicted {report.precursors_kept} of {report.precursors} precursors "
        f"({report.requests_predicted} requests, {report.fragments} fragments); NCE by charge: {table}"
    )
    for prop, reasons in report.refused.items():
        for reason, n in reasons.items():
            shown = "; ".join(report.examples.get(prop, {}).get(reason, [])[:3])
            print(f"  refused by {prop} ({reason}): {n} precursors, e.g. {shown}")
    if report.dropped_with_partner:
        print(f"  dropped with a refused target or decoy: {report.dropped_with_partner} precursors")
    ccs = report.examples.get("ccs", {}).get("unsupported_modification", [])
    if any("UNIMOD:737" in d or "UNIMOD:739" in d for d in ccs):
        print("  Sculptor has no TMT tokens: pass --no-ccs to keep TMT precursors")
    for caveat in report.caveats:
        print(f"  note: {caveat}")
    print(f"library: {output_dir / 'library_pqdir'}")
