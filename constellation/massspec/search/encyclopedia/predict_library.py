"""Wrapper for ``-convert -fastaToJChronologerLibrary`` — FASTA → predicted DLIB.

New in EncyclopeDIA 6.5.15. Uses the bundled JChronologer (RT) +
Sculptor (CCS/IMS) + Electrician (charge probability) predictors via
in-process PyTorch — replaces the Prosit/Koina round-trip the lab
previously used for library generation.

See [docs/plans/encyclopedia-6.5.15-utilities.md] for the full surveyed
flag table that this wrapper's kwargs mirror 1:1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from constellation.massspec.search.encyclopedia._common import (
    PtmToggle,
    ptm_toggle_args,
)
from constellation.thirdparty.jvm import JvmResult, run_jar


def build_predict_library_args(
    *,
    fasta: Path,
    output_dlib: Path,
    ptms: Mapping[str, PtmToggle] | None = None,
    min_charge: int = 1,
    max_charge: int = 6,
    min_mz: float = 396.4,
    max_mz: float = 1002.7,
    max_missed_cleavage: int = 1,
    enzyme: str = "Trypsin",
    add_decoys: bool = True,
    adjust_nce_for_dia: bool = True,
    default_nce: int = 33,
    default_charge: int = 3,
    max_variable_mods: int = 1,
    max_variable_forms: int = 1000,
    generate_protein_entrapments: bool = False,
    entrapment_seed: int = 1,
    prediction_cache: Path | None = None,
    ragged_n_term: bool = False,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Translate typed kwargs to the EncyclopeDIA CLI argv (no JVM call).

    Pure function — exists so the Tier A test can exercise the flag
    layout without spawning Java. Defaults match EncyclopeDIA's `-h`
    output exactly; see :func:`run_predict_library` for the wrapped
    invocation.
    """
    args: list[str] = [
        "-convert",
        "-fastaToJChronologerLibrary",
        "-i",
        str(fasta),
        "-o",
        str(output_dlib),
        # EncyclopeDIA accepts the booleans as `true` / `false` strings.
        "-addDecoys",
        _bool_str(add_decoys),
        "-adjustNCEForDIA",
        _bool_str(adjust_nce_for_dia),
        "-defaultCharge",
        str(int(default_charge)),
        "-defaultNCE",
        str(int(default_nce)),
        "-entrapmentSeed",
        str(int(entrapment_seed)),
        "-enzyme",
        str(enzyme),
        "-generateProteinEntrapments",
        _bool_str(generate_protein_entrapments),
        "-maxCharge",
        str(int(max_charge)),
        "-maxMissedCleavage",
        str(int(max_missed_cleavage)),
        "-maxMz",
        str(float(max_mz)),
        "-maxVariableForms",
        str(int(max_variable_forms)),
        "-maxVariableMods",
        str(int(max_variable_mods)),
        "-minCharge",
        str(int(min_charge)),
        "-minMz",
        str(float(min_mz)),
        "-raggedNTerm",
        _bool_str(ragged_n_term),
    ]
    if prediction_cache is not None:
        args.extend(["-predictionCache", str(prediction_cache)])
    args.extend(ptm_toggle_args(ptms))
    args.extend(str(a) for a in extra_args)
    return args


def run_predict_library(
    *,
    fasta: Path,
    output_dlib: Path,
    output_dir: Path,
    ptms: Mapping[str, PtmToggle] | None = None,
    min_charge: int = 1,
    max_charge: int = 6,
    min_mz: float = 396.4,
    max_mz: float = 1002.7,
    max_missed_cleavage: int = 1,
    enzyme: str = "Trypsin",
    add_decoys: bool = True,
    adjust_nce_for_dia: bool = True,
    default_nce: int = 33,
    default_charge: int = 3,
    max_variable_mods: int = 1,
    max_variable_forms: int = 1000,
    generate_protein_entrapments: bool = False,
    entrapment_seed: int = 1,
    prediction_cache: Path | None = None,
    ragged_n_term: bool = False,
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    extra_args: Sequence[str] = (),
    extra_jvm_args: Sequence[str] = (),
    stream_to_stderr: bool = True,
) -> JvmResult:
    """Predict a chromatogram-free spectral library from a FASTA.

    ``ptms`` is a dict mapping EncyclopeDIA PTM names (``Acetyl``,
    ``ProteinNTermAcetyl``, ``Carbamidomethyl``, ``Deamidation``,
    ``Dimethyl``, ``GlyGly``, ``HexNAc``, ``Methyl``, ``Oxidation``,
    ``Phospho``, ``PyroGluQ``, ``Succinyl``, ``Trimethyl``, ``TMT``) to
    one of ``"off"`` / ``"var"`` / ``"fix"``. Unspecified PTMs pick up
    EncyclopeDIA's defaults (``Carbamidomethyl`` fix,
    ``ProteinNTermAcetyl`` var, ``PyroGluQ`` var, everything else off).

    Peptide length is fixed at 7-31 residues by JChronologer.

    Streams the jar's stdout/stderr to ``<output_dir>/logs/`` and
    returns a :class:`JvmResult`. Raises :class:`JvmRunError` on
    non-zero exit.
    """
    args = build_predict_library_args(
        fasta=fasta,
        output_dlib=output_dlib,
        ptms=ptms,
        min_charge=min_charge,
        max_charge=max_charge,
        min_mz=min_mz,
        max_mz=max_mz,
        max_missed_cleavage=max_missed_cleavage,
        enzyme=enzyme,
        add_decoys=add_decoys,
        adjust_nce_for_dia=adjust_nce_for_dia,
        default_nce=default_nce,
        default_charge=default_charge,
        max_variable_mods=max_variable_mods,
        max_variable_forms=max_variable_forms,
        generate_protein_entrapments=generate_protein_entrapments,
        entrapment_seed=entrapment_seed,
        prediction_cache=prediction_cache,
        ragged_n_term=ragged_n_term,
        extra_args=extra_args,
    )
    return run_jar(
        "encyclopedia",
        args=args,
        jvm_heap_max=jvm_heap_max,
        jvm_heap_min=jvm_heap_min,
        jvm_tmpdir=jvm_tmpdir,
        extra_jvm_args=extra_jvm_args,
        log_dir=output_dir / "logs",
        stream_to_stderr=stream_to_stderr,
    )


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


__all__ = ["build_predict_library_args", "run_predict_library"]
