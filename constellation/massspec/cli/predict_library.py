"""``constellation massspec predict-library``: a predicted spectral library from a FASTA (EncyclopeDIA or Koina)."""

from __future__ import annotations

import argparse
from pathlib import Path

from constellation.massspec.cli._common import (
    _add_encyclopedia_passthrough_arg,
    _add_jvm_args,
    _add_output_dir_arg,
    _camel_to_kebab,
    _report_ingest_caveats,
)
from constellation.massspec.search.encyclopedia.ptm_defaults import (
    PTM_NAMES as _PTM_NAMES,
    default_for as _ptm_default_for,
)


# ── predict-library ─────────────────────────────────────────────────────


def _build_predict_library_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "predict-library",
        help=(
            "FASTA → predicted spectral library. --backend encyclopedia "
            "(default) runs EncyclopeDIA 6.5.15's bundled JChronologer "
            "(RT) + Sculptor (CCS/IMS) + Electrician (charge) in-process; "
            "--backend koina calls the Koina model server, which adds CID "
            "models and HCD at arbitrary collision energy."
        ),
    )
    p.add_argument(
        "--backend",
        choices=["encyclopedia", "koina"],
        default="encyclopedia",
        help="prediction backend (default: encyclopedia)",
    )
    p.add_argument(
        "--fasta",
        type=Path,
        help="input protein FASTA",
    )
    p.add_argument(
        "--output-dlib",
        type=Path,
        help=(
            "output predicted .dlib path (required for --backend "
            "encyclopedia; optional extra output for --backend koina)"
        ),
    )
    # ── koina backend ───────────────────────────────────────────────
    koina = p.add_argument_group(
        "koina backend",
        "Only meaningful with --backend koina.",
    )
    koina.add_argument(
        "--peptides",
        type=Path,
        default=None,
        help=(
            "explicit peptide list (TSV/CSV/parquet with a "
            "modified_sequence column, optional charge) instead of "
            "digesting a FASTA — no m/z or length filtering is applied"
        ),
    )
    koina.add_argument(
        "--from-library",
        type=Path,
        default=None,
        help=(
            "re-predict an existing library's exact precursors under a "
            "different model/energy (e.g. an HCD-built panel under a CID "
            "model)"
        ),
    )
    koina.add_argument(
        "--ms2-model",
        default="Prosit_2020_intensity_HCD",
        help="Koina fragment-intensity model (default: %(default)s)",
    )
    koina.add_argument(
        "--rt-model",
        default="Chronologer_RT",
        help=(
            "Koina retention-time model, or 'none' to skip RT prediction "
            "(default: %(default)s)"
        ),
    )
    koina.add_argument(
        "--koina-url",
        default=None,
        help=(
            "Koina server host:port (default: $CONSTELLATION_KOINA_URL, "
            "else koina.wilhelmlab.org:443)"
        ),
    )
    koina.add_argument(
        "--collision-energy",
        default=None,
        help=(
            "collision energy, or a comma-separated sweep (e.g. "
            "'20,25,30') which writes one library per energy under "
            "<output-dir>/ce_NN/. Rejected for models that declare no "
            "collision-energy input, since every energy would then "
            "produce an identical library."
        ),
    )
    koina.add_argument(
        "--output-library",
        type=Path,
        default=None,
        help=(
            "ParquetDir output path (default: <output-dir>/library_pqdir)"
        ),
    )
    koina.add_argument(
        "--on-unsupported-mod",
        choices=["error", "skip"],
        default="error",
        help=(
            "what to do with precursors the model cannot represent "
            "(default: error)"
        ),
    )
    koina.add_argument(
        "--min-intensity",
        type=float,
        default=1e-4,
        help="drop predicted fragments below this intensity (default: %(default)s)",
    )
    _add_output_dir_arg(p)
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip the jar invocation if <output-dir>/_SUCCESS exists",
    )
    p.add_argument(
        "--no-ingest",
        action="store_true",
        help=(
            "skip the read_encyclopedia(...) → ParquetDir auto-ingest "
            "of the produced .dlib"
        ),
    )
    p.add_argument(
        "--enzyme",
        default="Trypsin",
        help="enzyme name (default Trypsin)",
    )
    p.add_argument(
        "--min-charge",
        type=int,
        default=1,
        help="minimum precursor charge",
    )
    p.add_argument(
        "--max-charge",
        type=int,
        default=6,
        help="maximum precursor charge",
    )
    p.add_argument(
        "--min-mz",
        type=float,
        default=396.4,
        help="minimum precursor m/z filter",
    )
    p.add_argument(
        "--max-mz",
        type=float,
        default=1002.7,
        help="maximum precursor m/z filter",
    )
    p.add_argument(
        "--max-missed-cleavage",
        type=int,
        default=1,
    )
    p.add_argument(
        "--no-decoys",
        action="store_true",
        help="skip reversed-decoy prediction (halves prediction time)",
    )
    p.add_argument(
        "--no-adjust-nce-for-dia",
        action="store_true",
        help="skip the DIA-mode NCE adjustment EncyclopeDIA applies by default",
    )
    p.add_argument(
        "--default-nce",
        type=int,
        default=33,
        help="normalised collision energy for prediction",
    )
    p.add_argument(
        "--default-charge",
        type=int,
        default=3,
        help="default charge used for NCE adjustment",
    )
    p.add_argument(
        "--max-variable-mods",
        type=int,
        default=1,
    )
    p.add_argument(
        "--max-variable-forms",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--generate-protein-entrapments",
        action="store_true",
        help="generate one shuffled protein-level entrapment per target",
    )
    p.add_argument(
        "--entrapment-seed",
        type=int,
        default=1,
        help="RNG seed for reproducible entrapment generation",
    )
    p.add_argument(
        "--prediction-cache",
        type=Path,
        default=None,
        help=(
            "directory for the shared FASTA prediction cache (default: "
            "EncyclopeDIA's OS-specific user data dir)"
        ),
    )
    p.add_argument(
        "--ragged-n-term",
        action="store_true",
        help="enumerate N-terminal ragged variants",
    )
    # Per-PTM toggles — one --ptm-<name> for each EncyclopeDIA-recognised
    # PTM. Defaults imported from
    # ``massspec.search.encyclopedia.ptm_defaults`` so the standalone CLI
    # and ``transcriptome-to-proteome`` cannot drift.
    for name in _PTM_NAMES:
        flag = f"--ptm-{_camel_to_kebab(name)}"
        default = _ptm_default_for(name)
        p.add_argument(
            flag,
            choices=["off", "var", "fix"],
            default=default,
            help=f"{name} PTM mode (off|var|fix; default {default})",
        )
    _add_jvm_args(p)
    _add_encyclopedia_passthrough_arg(p)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress live stderr streaming of the jar's stdout/stderr",
    )
    p.set_defaults(func=_cmd_massspec_predict_library)


#: Flags that only the EncyclopeDIA backend implements, paired with the
#: parser's declared default. Silently ignoring one under --backend koina
#: would let a user believe their PTM settings or JVM heap took effect
#: when they did nothing at all — so a *non-default* value is an error.
#: Compare against the real default, not a sentinel: --jvm-heap-max
#: defaults to "12g" and --encyclopedia-arg to [], so a truthiness test
#: would reject every invocation.
_ENCYCLOPEDIA_ONLY_ARGS: tuple[tuple[str, str, object], ...] = (
    ("encyclopedia_arg", "--encyclopedia-arg", []),
    ("jvm_heap_max", "--jvm-heap-max", "12g"),
    ("jvm_heap_min", "--jvm-heap-min", None),
    ("jvm_tmpdir", "--jvm-tmpdir", None),
    ("prediction_cache", "--prediction-cache", None),
    ("generate_protein_entrapments", "--generate-protein-entrapments", False),
    ("ragged_n_term", "--ragged-n-term", False),
    ("no_decoys", "--no-decoys", False),
    ("max_variable_forms", "--max-variable-forms", 1000),
)


def _cmd_massspec_predict_library(args: argparse.Namespace) -> int:
    """Dispatch predict-library onto the selected backend."""
    if getattr(args, "backend", "encyclopedia") == "koina":
        return _cmd_predict_library_koina(args)
    return _cmd_predict_library_encyclopedia(args)


def _cmd_predict_library_encyclopedia(args: argparse.Namespace) -> int:
    """FASTA → predicted .dlib via EncyclopeDIA's JChronologer pipeline.

    Runs the jar, optionally ingests the produced .dlib into a
    ``library_pqdir/`` ParquetDir bundle, writes a ``manifest.json``
    capturing input SHA256s + jar version + JVM + runtime, and touches
    ``_SUCCESS`` last.
    """
    import sys as _sys

    if args.fasta is None:
        print("error: --fasta is required", file=_sys.stderr)
        return 1
    if args.output_dlib is None:
        print(
            "error: --output-dlib is required for --backend encyclopedia",
            file=_sys.stderr,
        )
        return 1

    from constellation import __version__ as constellation_version
    from constellation.massspec.io.encyclopedia import read_encyclopedia
    from constellation.massspec.library import save_library
    from constellation.massspec.search.encyclopedia import (
        build_manifest_envelope,
        encyclopedia_passthrough_args,
        require_min_encyclopedia,
        run_predict_library,
        write_manifest,
    )
    from constellation.thirdparty.jvm import JvmRunError
    from constellation.thirdparty.registry import ToolNotFoundError, find

    fasta = Path(args.fasta).resolve()
    output_dlib = Path(args.output_dlib).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not fasta.is_file():
        print(f"error: --fasta not found: {fasta}", file=_sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dlib.parent.mkdir(parents=True, exist_ok=True)

    success_path = output_dir / "_SUCCESS"
    if success_path.exists() and not args.resume:
        print(
            f"error: --output-dir already complete (touch _SUCCESS exists). "
            f"Pass --resume to re-use it, or delete {output_dir} to start "
            f"fresh.",
            file=_sys.stderr,
        )
        return 1
    if success_path.exists() and args.resume:
        print(f"already complete: {output_dir}")
        return 0

    # Tool version sanity-check (warn, don't error — wrappers stay
    # forward-compatible with interim builds the user may have access
    # to before a release).
    try:
        handle = find("encyclopedia")
    except ToolNotFoundError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        return 1
    if require_min_encyclopedia(handle):
        return 1

    ptms = {
        "Acetyl": args.ptm_acetyl,
        "ProteinNTermAcetyl": args.ptm_protein_n_term_acetyl,
        "Carbamidomethyl": args.ptm_carbamidomethyl,
        "Deamidation": args.ptm_deamidation,
        "Dimethyl": args.ptm_dimethyl,
        "GlyGly": args.ptm_gly_gly,
        "HexNAc": args.ptm_hex_n_ac,
        "Methyl": args.ptm_methyl,
        "Oxidation": args.ptm_oxidation,
        "Phospho": args.ptm_phospho,
        "PyroGluQ": args.ptm_pyro_glu_q,
        "Succinyl": args.ptm_succinyl,
        "Trimethyl": args.ptm_trimethyl,
        "TMT": args.ptm_tmt,
    }
    extra_args = encyclopedia_passthrough_args(args.encyclopedia_arg)

    try:
        result = run_predict_library(
            fasta=fasta,
            output_dlib=output_dlib,
            output_dir=output_dir,
            ptms=ptms,
            min_charge=args.min_charge,
            max_charge=args.max_charge,
            min_mz=args.min_mz,
            max_mz=args.max_mz,
            max_missed_cleavage=args.max_missed_cleavage,
            enzyme=args.enzyme,
            add_decoys=not args.no_decoys,
            adjust_nce_for_dia=not args.no_adjust_nce_for_dia,
            default_nce=args.default_nce,
            default_charge=args.default_charge,
            max_variable_mods=args.max_variable_mods,
            max_variable_forms=args.max_variable_forms,
            generate_protein_entrapments=args.generate_protein_entrapments,
            entrapment_seed=args.entrapment_seed,
            prediction_cache=args.prediction_cache,
            ragged_n_term=args.ragged_n_term,
            jvm_heap_max=args.jvm_heap_max,
            jvm_heap_min=args.jvm_heap_min,
            jvm_tmpdir=args.jvm_tmpdir,
            extra_args=extra_args,
            stream_to_stderr=not args.no_progress,
        )
    except JvmRunError as exc:
        print(f"error: encyclopedia jar exited {exc.returncode}", file=_sys.stderr)
        print(f"  see {exc.stderr_log} for the full log", file=_sys.stderr)
        return exc.returncode

    if not output_dlib.is_file():
        print(
            f"error: encyclopedia exited 0 but the expected .dlib was not "
            f"produced at {output_dlib}; check {result.stderr_log}",
            file=_sys.stderr,
        )
        return 2

    # ── auto-ingest ─────────────────────────────────────────────────
    ingest_info: dict[str, object] = {"skipped": bool(args.no_ingest)}
    library_pqdir: Path | None = None
    if not args.no_ingest:
        try:
            ingest_result = read_encyclopedia(output_dlib)
        except Exception as exc:  # noqa: BLE001 — surface the ingest error in the manifest
            ingest_info["error"] = f"{type(exc).__name__}: {exc}"
            _write_manifest_for_predict_library(
                args=args,
                fasta=fasta,
                output_dlib=output_dlib,
                output_dir=output_dir,
                result=result,
                handle=handle,
                ingest_info=ingest_info,
                library_pqdir=None,
                extra_args=extra_args,
                build_manifest_envelope=build_manifest_envelope,
                write_manifest=write_manifest,
                constellation_version=constellation_version,
            )
            print(
                f"error: predict-library produced {output_dlib} but "
                f"auto-ingest failed: {exc}. Manifest written; pass "
                f"--no-ingest to retry without ingest.",
                file=_sys.stderr,
            )
            return 3
        library_pqdir = output_dir / "library_pqdir"
        save_library(ingest_result.library, library_pqdir, format="parquet_dir")
        _report_ingest_caveats(ingest_result, ingest_info)
        ingest_info["library_counts"] = {
            "proteins": ingest_result.library.proteins.num_rows,
            "peptides": ingest_result.library.peptides.num_rows,
            "precursors": ingest_result.library.precursors.num_rows,
            "fragments": ingest_result.library.fragments.num_rows,
        }

    # ── manifest + _SUCCESS ─────────────────────────────────────────
    _write_manifest_for_predict_library(
        args=args,
        fasta=fasta,
        output_dlib=output_dlib,
        output_dir=output_dir,
        result=result,
        handle=handle,
        ingest_info=ingest_info,
        library_pqdir=library_pqdir,
        extra_args=extra_args,
        build_manifest_envelope=build_manifest_envelope,
        write_manifest=write_manifest,
        constellation_version=constellation_version,
    )
    success_path.write_bytes(b"")

    if not args.no_progress:
        counts = ingest_info.get("library_counts")
        if isinstance(counts, dict):
            print(
                f"predict-library done: {output_dlib} "
                f"({counts.get('peptides')} peptides, "
                f"{counts.get('precursors')} precursors)"
            )
        else:
            print(f"predict-library done: {output_dlib}")
    return 0


def _parse_collision_energies(raw: str | None) -> list[float | None]:
    """``"20,25,30"`` → ``[20.0, 25.0, 30.0]``; ``None`` → ``[None]``."""
    if raw is None:
        return [None]
    out: list[float | None] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError as exc:
            raise ValueError(f"bad collision energy {token!r}") from exc
    return out or [None]


def _cmd_predict_library_koina(args: argparse.Namespace) -> int:
    """FASTA / peptide list / existing library → predicted Library via Koina.

    Keeps the same run-dir contract as the EncyclopeDIA backend:
    ``manifest.json`` + ``_SUCCESS``, with ``--resume`` short-circuiting a
    completed directory. A multi-energy sweep writes one library per
    energy under ``ce_NN/``, each with its own ``_SUCCESS`` so a run
    interrupted at energy 12 of 17 resumes there.
    """
    import json
    import sys as _sys

    from constellation import __version__ as constellation_version
    from constellation.massspec.library import save_library
    from constellation.massspec.library.digest import (
        precursors_from_fasta,
        precursors_from_library,
        precursors_from_peptide_list,
    )
    from constellation.massspec.library.koina import (
        KoinaError,
        KoinaInputError,
        resolve_server,
    )
    from constellation.massspec.library.koina.api import predict_library
    from constellation.massspec.library.koina.client import make_client

    used = [
        flag
        for attr, flag, default in _ENCYCLOPEDIA_ONLY_ARGS
        if getattr(args, attr, default) != default
    ]
    ptm_flags = [
        f"--ptm-{_camel_to_kebab(name)}"
        for name in _PTM_NAMES
        if getattr(args, f"ptm_{_camel_to_kebab(name).replace('-', '_')}", "off")
        != _ptm_default_for(name)
    ]
    if used or ptm_flags:
        print(
            "error: these flags are EncyclopeDIA-only and have no effect "
            f"with --backend koina: {', '.join(used + ptm_flags)}",
            file=_sys.stderr,
        )
        return 1

    sources = [args.fasta, args.peptides, args.from_library]
    if sum(s is not None for s in sources) != 1:
        print(
            "error: pass exactly one of --fasta, --peptides, or "
            "--from-library",
            file=_sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    success_path = output_dir / "_SUCCESS"
    if success_path.exists():
        if not args.resume:
            print(
                f"error: --output-dir already complete (_SUCCESS exists). "
                f"Pass --resume to re-use it, or delete {output_dir} to "
                f"start fresh.",
                file=_sys.stderr,
            )
            return 1
        print(f"already complete: {output_dir}")
        return 0

    try:
        energies = _parse_collision_energies(args.collision_energy)
    except ValueError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        return 1

    server = resolve_server(args.koina_url)
    rt_model_name = None if str(args.rt_model).lower() == "none" else args.rt_model

    # ── build the precursor grid once, reused across energies ───────
    try:
        if args.fasta is not None:
            fasta = Path(args.fasta).resolve()
            if not fasta.is_file():
                print(f"error: --fasta not found: {fasta}", file=_sys.stderr)
                return 1
            specs = precursors_from_fasta(
                fasta,
                protease=args.enzyme,
                missed_cleavages=args.max_missed_cleavage,
                min_mz=args.min_mz,
                max_mz=args.max_mz,
                charges=tuple(range(args.min_charge, args.max_charge + 1)),
                fixed_mods={"C": "UNIMOD:4"},
                variable_mods={"M": "UNIMOD:35"},
                max_variable_mods=args.max_variable_mods,
            )
            source = str(fasta)
        elif args.peptides is not None:
            specs = precursors_from_peptide_list(
                args.peptides,
                charges=tuple(range(args.min_charge, args.max_charge + 1)),
            )
            source = str(Path(args.peptides).resolve())
        else:
            specs = precursors_from_library(args.from_library)
            source = str(Path(args.from_library).resolve())
    except (OSError, ValueError) as exc:
        print(f"error: could not build precursor list: {exc}", file=_sys.stderr)
        return 1

    if not specs:
        print("error: no precursors to predict", file=_sys.stderr)
        return 1

    # ── reject a sweep the model would silently collapse ────────────
    try:
        probe = make_client(args.ms2_model, server=args.koina_url)
        declared = set(probe.model_inputs)
    except KoinaError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        return 1

    takes_energy = "collision_energies" in declared
    if len(energies) > 1 and not takes_energy:
        print(
            f"error: {args.ms2_model} declares no collision-energy input, so "
            f"the {len(energies)} energies you gave would produce "
            f"{len(energies)} identical libraries. Drop --collision-energy, "
            f"or pick a model that accepts one.",
            file=_sys.stderr,
        )
        return 1
    if takes_energy and energies == [None]:
        energies = [float(args.default_nce)]
    if not takes_energy:
        energies = [None]

    multi = len(energies) > 1
    summaries: list[dict[str, object]] = []

    for energy in energies:
        run_dir = output_dir / f"ce_{energy:g}" if multi else output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = run_dir / "_SUCCESS" if multi else None
        if marker is not None and marker.exists() and args.resume:
            # Restore the completed run's summary. Skipping without it
            # drops finished energies from the aggregate manifest —
            # resuming a 20/30 sweep listed only 30, and resuming after
            # every energy had finished wrote "runs": [] and still
            # reported success.
            prior = run_dir / "manifest.json"
            if not prior.is_file():
                print(
                    f"error: {run_dir} has _SUCCESS but no manifest.json, so "
                    f"its summary cannot be restored and the aggregate would "
                    f"under-report the sweep. Delete that directory to "
                    f"recompute this energy.",
                    file=_sys.stderr,
                )
                return 1
            summaries.append(json.loads(prior.read_text()))
            print(f"  ce={energy:g}: already complete, restored from manifest")
            continue

        try:
            library, stats = predict_library(
                specs=specs,
                ms2_model_name=args.ms2_model,
                rt_model_name=rt_model_name,
                collision_energy=energy,
                adjust_nce_for_dia=not args.no_adjust_nce_for_dia,
                server=args.koina_url,
                min_intensity=args.min_intensity,
                on_unsupported=args.on_unsupported_mod,
                metadata={"x.koina.server_url": server, "x.koina.source": source},
            )
        except KoinaInputError as exc:
            print(f"error: {exc}", file=_sys.stderr)
            return 1
        except KoinaError as exc:
            print(f"error: Koina prediction failed: {exc}", file=_sys.stderr)
            return 2

        pqdir = (
            Path(args.output_library).resolve()
            if args.output_library is not None and not multi
            else run_dir / "library_pqdir"
        )
        save_library(library, pqdir, format="parquet_dir")

        dlib_path = None
        if args.output_dlib is not None:
            dlib_path = (
                Path(args.output_dlib).resolve()
                if not multi
                else run_dir / Path(args.output_dlib).name
            )
            dlib_path.parent.mkdir(parents=True, exist_ok=True)
            save_library(library, dlib_path, format="encyclopedia.dlib")

        summary = {
            "collision_energy": energy,
            "library_pqdir": str(pqdir),
            "output_dlib": str(dlib_path) if dlib_path else None,
            "counts": {
                "proteins": library.proteins.num_rows,
                "peptides": library.peptides.num_rows,
                "precursors": library.precursors.num_rows,
                "fragments": library.fragments.num_rows,
            },
            "assembly": stats.as_dict(),
        }
        summaries.append(summary)
        if marker is not None:
            (run_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
            marker.write_bytes(b"")
        if not args.no_progress:
            label = f"ce={energy:g}: " if multi else ""
            print(
                f"  {label}{library.precursors.num_rows} precursors, "
                f"{library.fragments.num_rows} fragments "
                f"(max m/z deviation {stats.max_abs_ppm_deviation:.3f} ppm)"
            )

    manifest = {
        "tool": "constellation massspec predict-library",
        "backend": "koina",
        "constellation_version": constellation_version,
        "source": source,
        "n_precursors_requested": len(specs),
        "koina": {
            "server": server,
            "ms2_model": args.ms2_model,
            "rt_model": rt_model_name,
            "adjust_nce_for_dia": not args.no_adjust_nce_for_dia,
            "min_intensity": args.min_intensity,
        },
        "runs": summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    success_path.write_bytes(b"")

    if not args.no_progress:
        print(f"predict-library done ({len(summaries)} library/libraries): {output_dir}")
    return 0


def _write_manifest_for_predict_library(
    *,
    args: argparse.Namespace,
    fasta: Path,
    output_dlib: Path,
    output_dir: Path,
    result,  # JvmResult
    handle,  # ToolHandle
    ingest_info: dict[str, object],
    library_pqdir: Path | None,
    extra_args: list[str],
    build_manifest_envelope,
    write_manifest,
    constellation_version: str,
) -> None:
    import os as _os
    import sys as _sys

    manifest = build_manifest_envelope(
        subcommand="massspec predict-library",
        constellation_version=constellation_version,
        constellation_argv=_sys.argv,
        java_argv=result.argv,
        tool={
            "name": "encyclopedia",
            "version": handle.version,
            "jar_path": str(handle.path),
            "jar_sha256": result.jar_sha256,
            "source": handle.source,
            "env_var_set": "CONSTELLATION_ENCYCLOPEDIA_HOME" in _os.environ,
            "java_version": result.java_version,
            "java_source": result.java_source,
            "java_path": str(result.java_path),
        },
        inputs={"fasta": fasta},
        outputs={
            "dlib": output_dlib,
            "library_pqdir": library_pqdir,
            "stdout_log": result.stdout_log,
            "stderr_log": result.stderr_log,
        },
        runtime={
            "elapsed_seconds": result.elapsed_seconds,
            "returncode": result.returncode,
        },
        ingest=ingest_info,
        encyclopedia_passthrough_args=extra_args,
    )
    write_manifest(output_dir / "manifest.json", manifest)
