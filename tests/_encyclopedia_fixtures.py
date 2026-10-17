"""Build small EncyclopeDIA result libraries in the pinned version's layout.

The tables come from the checked-in capture of what 6.9.23 writes
(``constellation/data/_raw/encyclopedia/<version>/schema/result_library.sql``),
so a reader tested against these files is tested against the real
column set, with no jar and no data. Values follow the conventions read
off real results: ``entries`` and ``peptidescores`` spell an N-terminal
modification before the first residue (``[+42.010565]MDFQ...``) while
``peptidequants`` puts it on the residue (``M[+42.010565]DFQ...``); a
precursor the jar could not quantify has ``NumberOfQuantIons`` 0 and
``TotalIntensity`` 0; an entry's stored spectrum can hold peaks that are
not sequence fragments (the intact precursor).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

from constellation.massspec.io.encyclopedia._codec import encode_f4, encode_f8
from constellation.thirdparty.encyclopedia import MINIMUM_VERSION

SOURCE_FILE = "run.raw"


def result_library_ddl() -> str:
    return (
        resources.files("constellation.data")
        .joinpath("_raw", "encyclopedia", MINIMUM_VERSION, "schema", "result_library.sql")
        .read_text()
    )


@dataclass
class Entry:
    """One identified precursor, as the jar would store it."""

    modseq: str  # EncyclopeDIA spelling as in ``entries`` (N-terminal mods first)
    sequence: str
    charge: int
    precursor_mz: float
    mz: Sequence[float]  # the stored spectrum
    intensity: Sequence[float]
    proteins: Sequence[str] = ("P1",)
    rt: float = 600.0
    score: float = 0.001
    #: ``peptidequants.TotalIntensity``; ``None`` writes no quantity row.
    total_intensity: float | None = 1.0e6
    #: Spelling in ``peptidequants``; defaults to ``modseq``.
    quant_modseq: str | None = None
    source: str = SOURCE_FILE
    extra_quant_rows: Sequence[tuple[str, float]] = field(default_factory=tuple)
    #: ``(run, TotalIntensity)``: quantity rows for runs other than
    #: ``source``, as a report across runs (``-libexport``) has them.
    other_runs: Sequence[tuple[str, float]] = field(default_factory=tuple)


def build_result_library(
    path: Path,
    entries: Sequence[Entry],
    *,
    decoy_peptides: Sequence[tuple[str, str]] = (),
    metadata: dict[str, str] | None = None,
    source: str = SOURCE_FILE,
    protein_source: str | None = None,
) -> Path:
    """Write ``entries`` as a single-run search result (``.elib``).

    ``decoy_peptides`` are ``(sequence, accession)`` pairs written to
    ``peptidetoprotein`` and the score tables as decoys. ``source`` is
    the run the decoy and protein score rows name (each entry names its
    own); ``protein_source``, when given, the run the protein score rows
    name instead (``global`` in a report across runs).
    """
    protein_source = protein_source or source
    con = sqlite3.connect(path)
    con.executescript(result_library_ddl())
    meta = {"version": "0.1.18", "EncyclopediaVersion": MINIMUM_VERSION}
    meta.update(metadata or {})
    con.executemany("INSERT INTO metadata VALUES (?, ?)", sorted(meta.items()))

    empty_f8, empty_f8_len = encode_f8(np.zeros(0))
    empty_f4, empty_f4_len = encode_f4(np.zeros(0, dtype=np.float32))
    links: set[tuple[str, int, str]] = set()
    proteins: set[str] = set()
    for e in entries:
        mz_blob, mz_len = encode_f8(np.asarray(e.mz, dtype=np.float64))
        it_blob, it_len = encode_f4(np.asarray(e.intensity, dtype=np.float32))
        con.execute(
            "INSERT INTO entries (PrecursorMz, PrecursorCharge, PeptideModSeq, PeptideSeq, "
            "Copies, RTInSeconds, Score, MassEncodedLength, MassArray, "
            "IntensityEncodedLength, IntensityArray, SourceFile) "
            "VALUES (?,?,?,?,1,?,?,?,?,?,?,?)",
            (
                e.precursor_mz,
                e.charge,
                e.modseq,
                e.sequence,
                e.rt,
                e.score,
                mz_len,
                mz_blob,
                it_len,
                it_blob,
                e.source,
            ),
        )
        con.execute(
            "INSERT INTO peptidescores VALUES (?,?,?,?,?,?,0)",
            (e.charge, e.modseq, e.sequence, e.source, e.score, e.score),
        )
        quant_rows = [(m, t, e.source) for m, t in e.extra_quant_rows]
        if e.total_intensity is not None:
            quant_rows.insert(0, (e.quant_modseq or e.modseq, e.total_intensity, e.source))
        quant_rows += [(e.quant_modseq or e.modseq, t, run) for run, t in e.other_runs]
        for modseq, total, run in quant_rows:
            con.execute(
                "INSERT INTO peptidequants (PrecursorCharge, PeptideModSeq, PeptideSeq, "
                "SourceFile, RTInSecondsCenter, RTInSecondsStart, RTInSecondsStop, "
                "TotalIntensity, NumberOfQuantIons, QuantIonMassLength, QuantIonMassArray, "
                "BestFragmentCorrelation, BestFragmentDeltaMassPPM, "
                "MedianChromatogramEncodedLength, MedianChromatogramArray, IdentifiedTICRatio) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    e.charge,
                    modseq,
                    e.sequence,
                    run,
                    e.rt,
                    e.rt - 10.0,
                    e.rt + 10.0,
                    total,
                    3 if total > 0 else 0,
                    empty_f8_len,
                    empty_f8,
                    0.9,
                    0.0,
                    empty_f4_len,
                    empty_f4,
                    0.1,
                ),
            )
        for accession in e.proteins:
            links.add((e.sequence, 0, accession))
            proteins.add(accession)
    for sequence, accession in decoy_peptides:
        links.add((sequence, 1, accession))
        con.execute(
            "INSERT INTO peptidescores VALUES (2,?,?,?,0.5,0.5,1)",
            (sequence, sequence, source),
        )
        con.execute(
            "INSERT INTO proteinscores VALUES (?,?,?,0.5,0.5,1)",
            (len(proteins) + 1000, accession, protein_source),
        )
    con.executemany("INSERT INTO peptidetoprotein VALUES (?,?,?)", sorted(links))
    for group, accession in enumerate(sorted(proteins)):
        con.execute(
            "INSERT INTO proteinscores VALUES (?,?,?,0.001,0.001,0)",
            (group, accession, protein_source),
        )
    con.commit()
    con.close()
    return path
