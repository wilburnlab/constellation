"""``constellation massspec predict-library``: a predicted spectral library (EncyclopeDIA's jar, JChronologer in PyTorch, or Koina)."""

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
from constellation.massspec.cli._predict_jchronologer import (
    add_jchronologer_arguments as _add_jchronologer_arguments,
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
            "a predicted spectral library. --backend encyclopedia (default) "
            "runs the EncyclopeDIA jar on a FASTA (its JChronologer models, "
            "in the JVM) and writes a .dlib; --backend jchronologer runs "
            "those same models in PyTorch, read from the installed "
            "EncyclopeDIA, onto a search space (--from-library) or a peptide "
            "list; --backend koina calls the Koina model server, which adds "
            "CID models."
        ),
    )
    p.add_argument(
        "--backend",
        choices=["encyclopedia", "jchronologer", "koina"],
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
    # ── what to predict, for the in-Python backends ─────────────────
    given = p.add_argument_group(
        "jchronologer and koina backends",
        "What to predict instead of a FASTA (--from-library is how "
        "--backend jchronologer predicts a search space).",
    )
    given.add_argument(
        "--peptides",
        type=Path,
        default=None,
        help=(
            "explicit peptide list (TSV/CSV/parquet with a "
            "modified_sequence column, optional charge) instead of "
            "digesting a FASTA — no m/z or length filtering is applied"
        ),
    )
    given.add_argument(
        "--from-library",
        type=Path,
        default=None,
        help=(
            "re-predict an existing library's exact precursors under a "
            "different model/energy (e.g. an HCD-built panel under a CID "
            "model); the result is that library with the predictions put "
            "onto it, so its decoys and provenance (a search space's spans "
            "and modifications) are kept"
        ),
    )
    given.add_argument(
        "--min-intensity",
        type=float,
        default=None,
        help=(
            "drop predicted fragments below this intensity (default: 0.01, "
            "JChronologer's, for jchronologer; 1e-4 for koina)"
        ),
    )
    _add_jchronologer_arguments(p)
    # ── koina backend ───────────────────────────────────────────────
    koina = p.add_argument_group(
        "koina backend",
        "Only meaningful with --backend koina.",
    )
    koina.add_argument(
        "--ms2-model",
        default=None,
        help=(
            "fragment-intensity model: a Koina model name with --backend koina "
            "(default: Prosit_2020_intensity_HCD); backend:model or none with "
            "--backend jchronologer (default: jchronologer:cartographer; e.g. "
            "koina:Prosit_2020_intensity_HCD)"
        ),
    )
    koina.add_argument(
        "--rt-model",
        default=None,
        help=(
            "retention-time model, or 'none' to skip RT: a Koina model name with "
            "--backend koina (default: Chronologer_RT); backend:model with "
            "--backend jchronologer (default: jchronologer:chronologer; e.g. "
            "koina:Chronologer_RT)"
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
        "--collision-energy-by-charge",
        default=None,
        metavar="Z=NCE,...",
        help=(
            "a collision energy for each precursor charge, e.g. "
            "'1=28,2=31,3=33,4=35' (tuned for an instrument), used as given "
            "instead of one energy scaled from --default-charge. Every charge "
            "predicted must be in it. Not with --collision-energy, "
            "--default-charge or --no-adjust-nce-for-dia."
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
        help=(
            "predict every charge at the collision energy as given, instead of "
            "scaling it from --default-charge to each precursor charge "
            "(EncyclopeDIA's DIA rule, NCE·f(default)/f(z))"
        ),
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
        default=None,
        help=(
            "the method's default charge, at which the collision energy "
            "applies unchanged; other charges are scaled from it (default: 3)"
        ),
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
    backend = getattr(args, "backend", "encyclopedia")
    if backend == "jchronologer":
        from constellation.massspec.cli._predict_jchronologer import run

        return run(args)
    from constellation.massspec.cli._predict_jchronologer import JCHRONOLOGER_ONLY_ARGS, used_flags

    foreign = used_flags(args, JCHRONOLOGER_ONLY_ARGS)
    if foreign:
        import sys as _sys

        print(
            f"error: these flags are --backend jchronologer's and have no effect with "
            f"--backend {backend}: {', '.join(foreign)}",
            file=_sys.stderr,
        )
        return 1
    if backend == "koina":
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
    if getattr(args, "collision_energy_by_charge", None) is not None:
        print(
            "error: --collision-energy-by-charge is not available with --backend "
            "encyclopedia: the jar takes one energy (--default-nce) and scales it "
            "from --default-charge itself",
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
            default_charge=_reference_charge(args),
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


def _reference_charge(args: argparse.Namespace) -> int:
    """``--default-charge``, or EncyclopeDIA's 3 when it was not given."""
    from constellation.massspec.library.prediction import DEFAULT_REFERENCE_CHARGE

    given = getattr(args, "default_charge", None)
    return DEFAULT_REFERENCE_CHARGE if given is None else int(given)


def _energy_table(args: argparse.Namespace) -> dict[int, float] | None:
    """``--collision-energy-by-charge`` parsed, or ``None``. It sets every
    charge's energy, so the flags that make one energy and scale it are
    refused beside it rather than silently overridden."""
    from constellation.massspec.library.prediction import parse_energy_table

    raw = getattr(args, "collision_energy_by_charge", None)
    if raw is None:
        return None
    conflicts = [
        flag
        for flag, given in (
            ("--collision-energy", getattr(args, "collision_energy", None) is not None),
            ("--default-charge", getattr(args, "default_charge", None) is not None),
            ("--no-adjust-nce-for-dia", bool(getattr(args, "no_adjust_nce_for_dia", False))),
        )
        if given
    ]
    if conflicts:
        raise ValueError(
            "--collision-energy-by-charge gives every charge its energy, so it "
            f"cannot be combined with {', '.join(conflicts)}"
        )
    return parse_energy_table(raw)


def _scalar_energy(energy: float, args: argparse.Namespace):
    """One energy, scaled from ``--default-charge`` to each charge unless
    ``--no-adjust-nce-for-dia``."""
    from constellation.massspec.library.prediction import CollisionEnergy

    if getattr(args, "no_adjust_nce_for_dia", False):
        return CollisionEnergy.uniform(energy)
    return CollisionEnergy.scaled_from_reference(energy, _reference_charge(args))


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
    from constellation.massspec.library import load_library, save_library
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

    # The model flags default per backend; a koina: prefix is the same model.
    args.ms2_model = str(args.ms2_model or "Prosit_2020_intensity_HCD").removeprefix("koina:")
    args.rt_model = str(args.rt_model or "Chronologer_RT").removeprefix("koina:")
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

    source_library = None
    sources = [args.fasta, args.peptides, args.from_library]
    if sum(s is not None for s in sources) != 1:
        print(
            "error: pass exactly one of --fasta, --peptides, or "
            "--from-library",
            file=_sys.stderr,
        )
        return 1

    from constellation.core.stages import invalidate, mark_done, recorded_key, stage_key

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    key = stage_key(**_koina_key_parts(args))
    recorded = recorded_key(output_dir)
    # A run replaces its own exports and refuses to replace anything else,
    # before anything is predicted (a sweep writes each energy's export
    # inside its own directory).
    from constellation.massspec.cli._predict_jchronologer import foreign_export, recorded_exports

    try:
        sweep = len(_parse_collision_energies(args.collision_energy)) > 1
    except ValueError:
        sweep = False  # refused below, with the reason
    if args.output_dlib is not None and not sweep:
        refusal = foreign_export(Path(args.output_dlib).resolve(), output_dir, recorded_exports(output_dir))
        if refusal:
            print(f"error: {refusal}", file=_sys.stderr)
            return 1
    if recorded is not None:
        # A marker without a key was written before runs were keyed
        # (2026-11-06), so it cannot say what it holds: like any other
        # prediction, it is replaced only with --resume and never reused
        # (it was reused unchecked until 2026-11-07).
        if recorded == key:
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
        if not args.resume:
            print(
                f"error: {output_dir} holds a prediction of other inputs or settings; pick another "
                f"--output-dir, or pass --resume to replace it",
                file=_sys.stderr,
            )
            return 1
        invalidate(output_dir)

    try:
        energies = _parse_collision_energies(args.collision_energy)
        energy_table = _energy_table(args)
    except ValueError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        return 1
    from constellation.massspec.cli._predict_jchronologer import DEFAULT_MIN_INTENSITY

    min_intensity = args.min_intensity if args.min_intensity is not None else DEFAULT_MIN_INTENSITY["koina"]

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
            from constellation.massspec.library.search_space import check_search_space_layout

            source_library = load_library(args.from_library)
            # warned about since 2026-11-11 (review of 2026-11-10, B8)
            check_search_space_layout(source_library)
            specs = precursors_from_library(source_library)
            source = str(Path(args.from_library).resolve())
    except (OSError, ValueError) as exc:
        print(f"error: could not build precursor list: {exc}", file=_sys.stderr)
        return 1

    if not specs:
        print("error: no precursors to predict", file=_sys.stderr)
        return 1
    if energy_table is not None:
        missing = sorted({s.charge for s in specs} - set(energy_table))
        if missing:
            print(
                f"error: --collision-energy-by-charge has no energy for charge "
                f"{', '.join(map(str, missing))}, which the precursors include",
                file=_sys.stderr,
            )
            return 1

    # ── reject a sweep the model would silently collapse ────────────
    try:
        probe = make_client(args.ms2_model, server=args.koina_url)
        declared = set(probe.model_inputs)
    except KoinaError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        return 1

    takes_energy = "collision_energies" in declared
    if energy_table is not None and not takes_energy:
        print(
            f"error: {args.ms2_model} declares no collision-energy input, so "
            f"--collision-energy-by-charge would change nothing. Drop it, or "
            f"pick a model that accepts one.",
            file=_sys.stderr,
        )
        return 1
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

    # Each run's energy as a property of each charge: an explicit table,
    # or one energy scaled from the reference charge.
    if energy_table is not None:
        from constellation.massspec.library.prediction import CollisionEnergy

        plans = [CollisionEnergy.explicit(energy_table)]
    else:
        plans = [None if e is None else _scalar_energy(e, args) for e in energies]
    charges_present = sorted({s.charge for s in specs})

    multi = len(plans) > 1
    summaries: list[dict[str, object]] = []

    for energy in plans:
        run_dir = output_dir / f"ce_{energy.label}" if multi else output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        marker = run_dir / "_SUCCESS" if multi else None
        run_key = stage_key(run=key, energy=energy.describe(charges_present) if energy is not None else None)
        if marker is not None and recorded_key(run_dir) not in (None, run_key):
            # An energy run of other settings, or one whose marker carries
            # no key and so cannot say what it holds: redo it (a keyless
            # one was restored under this sweep's key until 2026-11-07).
            invalidate(run_dir)
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
            print(f"  ce={energy.label}: already complete, restored from manifest")
            continue

        try:
            library, stats = predict_library(
                specs=specs if source_library is None else None,
                library=source_library,
                ms2_model_name=args.ms2_model,
                rt_model_name=rt_model_name,
                collision_energy=energy,
                server=args.koina_url,
                min_intensity=min_intensity,
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
            # This run's own export, from an earlier prediction (checked above).
            save_library(library, dlib_path, format="encyclopedia.dlib", overwrite=True)

        summary = {
            "collision_energy": energy.nce if energy is not None else None,
            "collision_energy_by_charge": (
                energy.describe(charges_present) if energy is not None else None
            ),
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
            mark_done(run_dir, run_key)
        if not args.no_progress:
            label = f"ce={energy.label}: " if multi else ""
            print(
                f"  {label}{library.precursors.num_rows} precursors, "
                f"{library.fragments.num_rows} fragments "
                f"(max m/z deviation {stats.max_abs_ppm_deviation:.3f} ppm)"
            )
            dropped = getattr(library, "metadata_extras", {}).get("x.koina.precursors_not_predicted")
            if dropped:
                print(f"  {label}{dropped} of the library's precursors could not be predicted and are left out")

    from constellation.massspec.search.encyclopedia._common import build_manifest_envelope

    body = {
        "backend": "koina",
        "stage_key": key,
        "source": source,
        "n_precursors_requested": len(specs),
        "koina": {
            "server": server,
            "ms2_model": args.ms2_model,
            "rt_model": rt_model_name,
            "adjust_nce_for_dia": energy_table is None and not args.no_adjust_nce_for_dia,
            "nce_reference_charge": (
                _reference_charge(args)
                if energy_table is None and not args.no_adjust_nce_for_dia
                else None
            ),
            "collision_energy_by_charge": energy_table,
            "min_intensity": min_intensity,
        },
        "runs": summaries,
    }
    source_path = Path(source)
    manifest = build_manifest_envelope(
        subcommand="massspec predict-library",
        constellation_version=constellation_version,
        constellation_argv=_sys.argv,
        java_argv=None,
        tool={"name": "koina", "server": server},
        inputs={"source": source_path} if source_path.is_file() else {},
        outputs={"output_dir": output_dir},
        extras=body,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    mark_done(output_dir, key)

    if not args.no_progress:
        print(f"predict-library done ({len(summaries)} library/libraries): {output_dir}")
    return 0


def _koina_key_parts(args: argparse.Namespace) -> dict[str, object]:
    """What a Koina run's output depends on: the source (its SHA-256, or
    the search space's own key), the digestion, the models and server, the
    energies and the settings."""
    from constellation.core.stages import file_sha256
    from constellation.massspec.cli._predict_jchronologer import _library_key

    def identity(path) -> object:
        p = Path(path)
        if p.is_dir():
            return _library_key(p)
        return file_sha256(p) if p.is_file() else f"{p}:missing"

    from constellation.massspec.library.koina import resolve_server
    from constellation.massspec.library.prediction import PREDICTION_LAYOUT

    return {
        "stage": "massspec predict-library koina",
        "layout": PREDICTION_LAYOUT,
        "fasta": identity(args.fasta) if args.fasta is not None else None,
        "peptides": identity(args.peptides) if args.peptides is not None else None,
        "library": identity(args.from_library) if args.from_library is not None else None,
        "digest": [args.enzyme, args.max_missed_cleavage, args.min_mz, args.max_mz, args.min_charge,
                   args.max_charge, args.max_variable_mods],
        "models": [args.ms2_model, args.rt_model],
        "server": resolve_server(args.koina_url),
        "energy": [args.collision_energy, args.collision_energy_by_charge, args.default_nce,
                   args.default_charge, args.no_adjust_nce_for_dia],
        "min_intensity": args.min_intensity,
        "on_unsupported": args.on_unsupported_mod,
        "outputs": [str(args.output_dlib) if args.output_dlib else None,
                    str(args.output_library) if args.output_library else None],
    }


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
