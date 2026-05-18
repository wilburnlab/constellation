"""Wrapper for ``-convert -fastaToJChronologerLibrary`` — FASTA → predicted DLIB.

New in EncyclopeDIA 6.5.15. Uses the bundled JChronologer (RT) +
Sculptor (CCS/IMS) + Electrician (charge probability) predictors via
in-process PyTorch — replaces the Prosit/Koina round-trip the lab
previously used for library generation.

Filled in PR 2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from constellation.massspec.search.encyclopedia._common import PtmToggle
from constellation.thirdparty.jvm import JvmResult


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

    ``ptms`` is a dict mapping EncyclopeDIA PTM names (Acetyl,
    ProteinNTermAcetyl, Carbamidomethyl, Deamidation, Dimethyl, GlyGly,
    HexNAc, Methyl, Oxidation, Phospho, PyroGluQ, Succinyl, Trimethyl,
    TMT) to one of ``"off"`` / ``"var"`` / ``"fix"``. Unspecified PTMs
    pick up EncyclopeDIA's defaults (Carbamidomethyl fix,
    ProteinNTermAcetyl var, PyroGluQ var, everything else off).

    Peptide length is fixed at 7-31 residues by JChronologer.

    Filled in PR 2.
    """
    raise NotImplementedError(
        "run_predict_library lands in PR 2 — wraps the new-in-6.5.15 "
        "`-convert -fastaToJChronologerLibrary` jar entry point. PR 0 "
        "ships the architecture only."
    )


__all__ = ["run_predict_library"]
