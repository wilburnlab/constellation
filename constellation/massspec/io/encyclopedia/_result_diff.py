"""Compare two EncyclopeDIA result libraries by what a user would act on.

Two searches of the same run never give byte-equal ``.elib`` files, and
not always the same identifications. Measured on 6.9.23: the scoring is
reproducible except for one Percolator feature, ``deltaRT``, which comes
from a retention-time fit that differs slightly from run to run (about
150 of 76,400 feature rows, by up to 0.035), and that is enough to move
up to a dozen precursors across the 1% FDR line. "Are these two results
the same" is therefore a question about overlap and agreement, with a
repeat search of the same input as the yardstick.

This reads the two files' own tables (no Constellation ids involved, so
it works on any single-run result library) and reports:

- which precursors (``PeptideModSeq`` + charge), peptides and target
  proteins each one identified, and how many they share;
- for the shared precursors, how far apart the retention times, scores
  and quantities are.

It is how a search of a Constellation-written spectra file is checked
against a search of the reference file.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from constellation.massspec.io.encyclopedia._modseq import encyclopedia_residue_form


@dataclass
class SetOverlap:
    """Sizes of two sets and their intersection."""

    n_a: int
    n_b: int
    n_shared: int

    @property
    def only_a(self) -> int:
        return self.n_a - self.n_shared

    @property
    def only_b(self) -> int:
        return self.n_b - self.n_shared

    @property
    def jaccard(self) -> float:
        union = self.n_a + self.n_b - self.n_shared
        return self.n_shared / union if union else 1.0

    def line(self, label: str) -> str:
        return (
            f"{label}: {self.n_a:,} vs {self.n_b:,}; shared {self.n_shared:,} "
            f"(only a {self.only_a:,}, only b {self.only_b:,}; Jaccard {self.jaccard:.4f})"
        )


@dataclass
class ResultComparison:
    """Result of :func:`compare_results`."""

    a: Path
    b: Path
    precursors: SetOverlap
    peptides: SetOverlap
    proteins: SetOverlap
    #: Over shared precursors. Empty when the two files share none.
    rt_abs_s: dict[str, float] = field(default_factory=dict)  # median, p99, max
    score_abs: dict[str, float] = field(default_factory=dict)
    #: ``log2(intensity_a / intensity_b)`` over shared precursors both
    #: files quantify: median absolute value, 99th percentile, the
    #: fraction within 1%, and the Pearson r of the log intensities.
    intensity: dict[str, float] = field(default_factory=dict)

    @property
    def identical_identifications(self) -> bool:
        return self.precursors.only_a == 0 and self.precursors.only_b == 0

    def report(self) -> str:
        lines = [f"a: {self.a}", f"b: {self.b}"]
        lines.append(self.precursors.line("precursors"))
        lines.append(self.peptides.line("peptides  "))
        lines.append(self.proteins.line("proteins  "))
        if self.rt_abs_s:
            r = self.rt_abs_s
            lines.append(
                f"retention time |Δ| (s) over shared precursors: median {r['median']:.4g}, "
                f"p99 {r['p99']:.4g}, max {r['max']:.4g}"
            )
        if self.score_abs:
            s = self.score_abs
            lines.append(
                f"score |Δ| over shared precursors: median {s['median']:.3g}, "
                f"p99 {s['p99']:.3g}, max {s['max']:.3g}"
            )
        if self.intensity:
            q = self.intensity
            lines.append(
                f"quantity over {int(q['n']):,} shared precursors: |log2 ratio| median "
                f"{q['median_abs_log2']:.4g}, p99 {q['p99_abs_log2']:.4g}; "
                f"{100 * q['frac_within_1pct']:.2f}% within 1%; "
                f"Pearson r (log) {q['pearson_log']:.6f}"
            )
        return "\n".join(lines)


def _open(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"no such result library: {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = {"entries", "peptidescores", "proteinscores", "peptidequants"} - tables
    if missing:
        con.close()
        raise ValueError(f"{path} is not a result library: missing table(s) {sorted(missing)}")
    return con


def _precursor_key(modseq: str, charge: int) -> tuple[str, int]:
    """One spelling per precursor across a result library's tables.

    ``entries`` and ``peptidequants`` spell an N-terminal modification
    differently (:func:`._modseq.encyclopedia_residue_form`); joined on
    the raw string, every N-terminally modified precursor (758 of 31,830
    in the first file looked at) would have no quantity.
    """
    return encyclopedia_residue_form(modseq), int(charge)


def _load(path: Path) -> tuple[dict, dict, set[str]]:
    """``{precursor: (rt, score)}``, ``{precursor: intensity}`` and target proteins."""
    con = _open(path)
    try:
        sources = [r[0] for r in con.execute("SELECT DISTINCT SourceFile FROM entries")]
        if len(sources) > 1:
            raise ValueError(
                f"{path} holds {len(sources)} runs ({', '.join(sorted(sources)[:3])}, ...); "
                f"compare_results compares single-run result libraries"
            )
        entries = {
            _precursor_key(modseq, charge): (float(rt), float(score))
            for modseq, charge, rt, score in con.execute(
                "SELECT PeptideModSeq, PrecursorCharge, RTInSeconds, Score FROM entries"
            )
        }
        quants = {
            _precursor_key(modseq, charge): float(total)
            for modseq, charge, total in con.execute(
                "SELECT PeptideModSeq, PrecursorCharge, TotalIntensity FROM peptidequants"
            )
        }
        proteins = {
            r[0]
            for r in con.execute(
                "SELECT DISTINCT ProteinAccession FROM proteinscores WHERE IsDecoy = 0"
            )
        }
    finally:
        con.close()
    return entries, quants, proteins


def _spread(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def compare_results(a: Path | str, b: Path | str) -> ResultComparison:
    """Compare two EncyclopeDIA result libraries (``.elib``, or ScribeTwo's ``.dlib``).

    A precursor is a ``(PeptideModSeq, PrecursorCharge)`` row of
    ``entries`` (N-terminal modifications in one spelling, see
    :func:`_precursor_key`); a peptide is its ``PeptideModSeq``; a protein is a
    non-decoy ``ProteinAccession`` of ``proteinscores``. Each file must
    describe one run (a multi-run quantitative report is refused); the
    run's name is not compared, so the same acquisition searched under
    two file names lines up.
    """
    a, b = Path(a), Path(b)
    entries_a, quants_a, proteins_a = _load(a)
    entries_b, quants_b, proteins_b = _load(b)

    shared = sorted(entries_a.keys() & entries_b.keys())
    peptides_a = {k[0] for k in entries_a}
    peptides_b = {k[0] for k in entries_b}
    result = ResultComparison(
        a=a,
        b=b,
        precursors=SetOverlap(len(entries_a), len(entries_b), len(shared)),
        peptides=SetOverlap(len(peptides_a), len(peptides_b), len(peptides_a & peptides_b)),
        proteins=SetOverlap(len(proteins_a), len(proteins_b), len(proteins_a & proteins_b)),
    )
    if shared:
        pairs_a = np.array([entries_a[k] for k in shared])
        pairs_b = np.array([entries_b[k] for k in shared])
        result.rt_abs_s = _spread(np.abs(pairs_a[:, 0] - pairs_b[:, 0]))
        result.score_abs = _spread(np.abs(pairs_a[:, 1] - pairs_b[:, 1]))
        quantified = [
            k for k in shared if quants_a.get(k, 0.0) > 0.0 and quants_b.get(k, 0.0) > 0.0
        ]
        if quantified:
            qa = np.log2(np.array([quants_a[k] for k in quantified]))
            qb = np.log2(np.array([quants_b[k] for k in quantified]))
            ratio = np.abs(qa - qb)
            result.intensity = {
                "n": float(len(quantified)),
                "median_abs_log2": float(np.median(ratio)),
                "p99_abs_log2": float(np.percentile(ratio, 99)),
                "frac_within_1pct": float(np.mean(ratio <= np.log2(1.01))),
                # Two identical vectors have r = 1 by definition; corrcoef
                # returns nan for a constant one.
                "pearson_log": (
                    1.0
                    if np.array_equal(qa, qb)
                    else float(np.corrcoef(qa, qb)[0, 1])
                    if len(quantified) > 1
                    else float("nan")
                ),
            }
    return result


__all__ = ["ResultComparison", "SetOverlap", "compare_results"]
