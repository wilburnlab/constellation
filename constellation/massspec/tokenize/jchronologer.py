"""Tokens for JChronologer's models, as JChronologer itself makes them.

JChronologer (EncyclopeDIA's library predictor, searlelab/jchronologer,
Apache-2.0) ships each TorchScript model with a ``.preprocessing.json``:
the model's characters and their ids (``aa_to_int``), its maximum length,
and how modified residues become characters. This module turns a
Constellation peptidoform into exactly the tokens JChronologer 1.0.10
gives that model, read from the jar's source (``PeptideSequenceConverter``,
``ChronologerPreprocessor``, ``DefaultChronologerLibraryPredictor``) and
checked against tokens the jar itself produced
(``scripts/survey-jchronologer-tokens.py``).

Two conventions, by model:

* **Chronologer (RT), Cartographer (fragments), Electrician (charge
  states).** JChronologer writes the peptide with one summed mass per
  position, ``%+.6f`` (``PEPTM[+15.994915]IDEK``), and applies the JSON's
  regular expressions to it (``M\\[\\+15\\.99.{,6}\\]`` → ``m``). Masses
  come from a fixed table of fifteen UNIMOD entries
  (:data:`JAR_UNIMOD_MASSES`); a modification outside it has no spelling
  and is refused. The N-terminal token is the JSON's ``nterm_keys`` entry
  for the first six characters of the N-terminal mass, else ``)`` before
  a first residue coded ``d`` and ``(`` before one coded ``e``, else
  ``-``. Peptides of 6 to ``max_peptide_len`` residues are taken.
  Electrician reads Cartographer's tokens (the jar tokenizes once with
  Cartographer's file and feeds both models).
* **Sculptor (CCS).** Accession maps (``residue_unimod_map``,
  ``nterm_unimod_map``): one modification per position, none on the
  C-terminus, up to ``max_peptide_len`` residues.

Pyro-Glu, which Constellation writes on the N-terminus
(``[UNIMOD:28]-QPEPTIDE``), is told to each model the way it was trained:

=============  ==========================  ====================
model          ``[UNIMOD:28]-Q…``           ``[UNIMOD:27]-E…``
=============  ==========================  ====================
Chronologer    ``(`` then ``e``             ``(`` then ``e``
Cartographer   ``(`` then ``Q``             ``)`` then ``E``
Sculptor       ``(`` then ``Q``             ``)`` then ``E``
=============  ==========================  ====================

S-carbamoylmethylcysteine cyclization (``[UNIMOD:26]-C…``, which replaces
the carbamidomethyl) is not one of the fifteen; JChronologer writes the
same molecule as pyro-Glu's ammonia loss (UNIMOD:28) on the N-terminus of
a carbamidomethyl cysteine, so it is given to the models that way:
Chronologer ``)`` then ``d``, Cartographer and Sculptor ``(`` then ``c``.

One place this tokenizer departs from the jar, on purpose:
``ChronologerPreprocessor.patchModSequence`` moves a residue-0 bracket
reading exactly ``[+42.010565]`` to the N-terminus, so the jar tells
Chronologer and Cartographer that a peptide beginning with a side-chain
acetyl lysine (``K[UNIMOD:1]VERPD``) is N-terminally acetylated
(``^K…``). Constellation tells them ``-a…``, the molecule it is; Sculptor
gets ``-a…`` from the jar too.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from types import MappingProxyType

from constellation.core.sequence.proforma import Peptidoform
from constellation.massspec.tokenize.vocabulary import (
    LENGTH_OUT_OF_RANGE,
    UNKNOWN_RESIDUE,
    UNSUPPORTED_MODIFICATION,
    Sites,
    SiteTokens,
    TokenizationRefused,
    TokenizedBatch,
    Vocabulary,
)

__all__ = [
    "JAR_UNIMOD_MASSES",
    "MIN_PEPTIDE_LENGTH",
    "SNAPSHOT_NAME",
    "TOKENIZED_MODELS",
    "JChronologerTokenizer",
    "UnknownModelError",
    "jar_sites",
    "load_snapshot",
    "tokenizer_from_preprocessing",
    "tokenizers_from_files",
]

SNAPSHOT_NAME = "jchronologer_models.json"
#: The models that take tokens (Scout, the jar's all-in-one model, is not
#: run here yet).
TOKENIZED_MODELS = ("chronologer", "cartographer", "electrician", "sculptor")

#: ``PeptideSequenceConverter.createUnimodMassMap`` (jchronologer 1.0.10):
#: the only modifications JChronologer can name, with the masses it spells
#: them with.
JAR_UNIMOD_MASSES: Mapping[str, float] = MappingProxyType({
    "UNIMOD:737": 229.162932,   # TMT6plex
    "UNIMOD:35": 15.994915,     # Oxidation
    "UNIMOD:4": 57.021464,      # Carbamidomethyl
    "UNIMOD:21": 79.966331,     # Phospho
    "UNIMOD:121": 114.042927,   # GG
    "UNIMOD:1": 42.010565,      # Acetyl
    "UNIMOD:34": 14.015650,     # Methyl
    "UNIMOD:28": -17.026549,    # Gln->pyro-Glu
    "UNIMOD:43": 203.079373,    # HexNAc
    "UNIMOD:7": 0.984016,       # Deamidated
    "UNIMOD:27": -18.010565,    # Glu->pyro-Glu
    "UNIMOD:64": 100.016044,    # Succinyl
    "UNIMOD:36": 28.0313,       # Dimethyl
    "UNIMOD:37": 42.04695,      # Trimethyl
    "UNIMOD:739": 224.152478,   # TMT (zero)
})
#: ``ChronologerPreprocessor.MIN_PEPTIDE_LEN``.
MIN_PEPTIDE_LENGTH = 6

_GLN_PYRO, _GLU_PYRO, _CAM_CYCLIZED, _CAM = "UNIMOD:28", "UNIMOD:27", "UNIMOD:26", "UNIMOD:4"
_MISSING = object()


class UnknownModelError(RuntimeError):
    """A JChronologer file Constellation was not checked against."""


@cache
def load_snapshot() -> Mapping:
    """``constellation/data/jchronologer_models.json``: the SHA-256 of
    every JChronologer jar and model member Constellation was checked
    against, by version and role."""
    text = resources.files("constellation.data").joinpath(SNAPSHOT_NAME).read_text()
    return json.loads(text)


def _remove(items: list[str], value: str) -> bool:
    if value in items:
        items.remove(value)
        return True
    return False


def jar_sites(sites: Sites) -> Sites:
    """``sites`` as JChronologer's own UNIMOD form writes the molecule.

    Pyro-Glu written on the first residue (``Q[UNIMOD:28]…``) goes to the
    N-terminus (``foldFirstResiduePyrogluToNterm``), and S-carbamoylmethyl-
    cysteine cyclization (UNIMOD:26) becomes UNIMOD:28 on the N-terminus
    of a carbamidomethyl cysteine, the jar's spelling of that molecule.
    """
    if not sites.residues:
        return sites
    r0 = sites.residues[0]
    if not (
        (r0 in "QE" and not sites.n_term and sites.residue_mods[0])
        or (r0 == "C" and _CAM_CYCLIZED in sites.n_term)
    ):
        return sites
    n_term = list(sites.n_term)
    first = list(sites.residue_mods[0])
    if not n_term:
        if r0 == "Q" and _remove(first, _GLN_PYRO):
            n_term.append(_GLN_PYRO)
        elif r0 == "E" and _remove(first, _GLU_PYRO):
            n_term.append(_GLU_PYRO)
    if r0 == "C" and _remove(n_term, _CAM_CYCLIZED):
        n_term.append(_GLN_PYRO)
        first.append(_CAM)
    if n_term == list(sites.n_term) and first == list(sites.residue_mods[0]):
        return sites
    return Sites(
        residues=sites.residues,
        n_term=tuple(sorted(n_term)),
        residue_mods=(tuple(sorted(first)), *sites.residue_mods[1:]),
        c_term=sites.c_term,
    )


def _signed(mass: float) -> str:
    """``String.format(Locale.US, "%+.6f", mass)``; ``PeptideSequenceConverter.formatSignedMass``
    also writes a sum within 5e-7 of zero as ``+0.000000``."""
    return f"{0.0 if abs(mass) < 5e-7 else mass:+.6f}"


def _mass(accessions: Sequence[str], where: str) -> float:
    total = 0.0
    for accession in accessions:
        mass = JAR_UNIMOD_MASSES.get(accession)
        if mass is None:
            raise TokenizationRefused(
                UNSUPPORTED_MODIFICATION, f"{accession} on {where} (JChronologer cannot name it)"
            )
        total += mass
    return total


class _JarSpelling:
    """``ChronologerPreprocessor.modseqToCodedseq`` over the mass spelling
    of a peptide, position by position (the rules each match one residue
    and its bracket, so a position's code depends on that position only)."""

    def __init__(self, preprocessing: Mapping, *, pyro_on_first_residue: bool) -> None:
        self.rules = [(re.compile(r["pattern"]), r["token"]) for r in preprocessing["mod_regex_rules"]]
        self.nterm_keys = dict(preprocessing["nterm_keys"])
        self.max_length = int(preprocessing["max_peptide_len"])
        self.chars = frozenset(k for k in preprocessing["aa_to_int"] if len(k) == 1)
        self.pyro_on_first_residue = pyro_on_first_residue
        self._codes: dict[tuple[str, tuple[str, ...]], str | None] = {}

    def _code(self, residue: str, accessions: tuple[str, ...]) -> str | None:
        """The character the rules make of one modified residue, or None
        when no rule takes it (or the jar cannot name a modification)."""
        if any(a not in JAR_UNIMOD_MASSES for a in accessions):
            code = None
        else:
            text = f"{residue}[{_signed(sum(JAR_UNIMOD_MASSES[a] for a in accessions))}]"
            for pattern, token in self.rules:
                text = pattern.sub(token, text)
            code = None if "[" in text else text
        self._codes[(residue, accessions)] = code
        return code

    def __call__(self, sites: Sites) -> str:
        sites = jar_sites(sites)
        residues = sites.residues
        n = len(residues)
        if sites.c_term:
            # The jar writes C-terminal masses after the last residue for
            # Chronologer, where no rule takes them, and leaves them out of
            # Cartographer's tokens; neither is the molecule.
            raise TokenizationRefused(
                UNSUPPORTED_MODIFICATION, f"C-terminal modification {', '.join(sites.c_term)}"
            )
        if n < MIN_PEPTIDE_LENGTH or n > self.max_length:
            raise TokenizationRefused(
                LENGTH_OUT_OF_RANGE, f"{n} residues (takes {MIN_PEPTIDE_LENGTH}-{self.max_length})"
            )
        n_term, mods = sites.n_term, sites.residue_mods
        if self.pyro_on_first_residue and n_term:
            # PeptideSequenceConverter.foldNtermPyroToFirstResidue (the RT path).
            folded, first, r0 = list(n_term), list(mods[0]), residues[0]
            if r0 == "Q" and _remove(folded, _GLN_PYRO):
                first.append(_GLN_PYRO)
            elif r0 == "E" and _remove(folded, _GLU_PYRO):
                first.append(_GLU_PYRO)
            elif r0 == "C" and _remove(folded, _GLN_PYRO):
                if _CAM in first:
                    first.append(_GLN_PYRO)
                else:
                    folded.append(_GLN_PYRO)
            n_term, mods = tuple(sorted(folded)), (tuple(sorted(first)), *mods[1:])
        out = list(residues)
        codes = self._codes
        for i, m in enumerate(mods):
            if m:
                code = codes.get((residues[i], m), _MISSING)
                if code is _MISSING:
                    code = self._code(residues[i], m)
                if code is None:
                    unnamed = [a for a in m if a not in JAR_UNIMOD_MASSES]
                    where = residues[i] + str(i + 1) + "".join(f"[{a}]" for a in m)
                    raise TokenizationRefused(
                        UNSUPPORTED_MODIFICATION,
                        f"{where} (JChronologer cannot name {', '.join(unnamed)})" if unnamed else where,
                    )
                out[i] = code
        coded = "".join(out)
        if n_term:
            key = _signed(_mass(n_term, "the N-terminus"))[:6]
            head = self.nterm_keys.get(key)
            if head is None:
                raise TokenizationRefused(UNSUPPORTED_MODIFICATION, f"N-terminal {', '.join(n_term)}")
        else:
            head = ")" if coded[0] == "d" else "(" if coded[0] == "e" else "-"
        if not self.chars.issuperset(coded):
            i = next(i for i, char in enumerate(coded) if char not in self.chars)
            raise TokenizationRefused(UNKNOWN_RESIDUE, f"{residues[i]}{i + 1}")
        return head + coded + "_"


class _AccessionSpelling:
    """``DefaultChronologerLibraryPredictor.SculptorTokenizer``: the
    accession maps of the file, over the jar's UNIMOD form."""

    def __init__(self, preprocessing: Mapping) -> None:
        chars = {k for k in preprocessing["aa_to_int"] if len(k) == 1}
        self.tokens = SiteTokens(
            residue_tokens={c: c for c in chars if "A" <= c <= "Z"},
            modified_residue_tokens={
                (entry["residue"], entry["unimod"]): entry["token"]
                for entry in preprocessing["residue_unimod_map"]
            },
            n_term_tokens=dict(preprocessing["nterm_unimod_map"]),
            min_length=1,
            max_length=int(preprocessing["max_peptide_len"]),
        )

    def __call__(self, sites: Sites) -> str:
        return self.tokens.spell(jar_sites(sites))


@dataclass(frozen=True)
class JChronologerTokenizer:
    """One JChronologer model's tokenizer."""

    model: str
    vocabulary: Vocabulary
    spelling: Callable[[Sites], str]

    def spell(self, peptide: Peptidoform | Sites | str) -> str:
        """The model's characters for one peptide (``-PEPTIDEK_``).
        Raises :class:`TokenizationRefused`."""
        sites = peptide if isinstance(peptide, Sites) else Sites.from_peptidoform(peptide)
        return self.spelling(sites)

    def encode(self, peptides: Sequence[Peptidoform | Sites | str]) -> TokenizedBatch:
        """Tokens for many peptides; refused rows are padding, with the
        reason and detail beside them."""
        spellings: list[str | None] = []
        reasons: list[str | None] = []
        details: list[str | None] = []
        for peptide in peptides:
            try:
                spellings.append(self.spell(peptide))
                reasons.append(None)
                details.append(None)
            except TokenizationRefused as refusal:
                spellings.append(None)
                reasons.append(refusal.reason)
                details.append(refusal.detail)
        return TokenizedBatch(self.vocabulary.encode(spellings), tuple(reasons), tuple(details))


def _vocabulary(preprocessing: Mapping) -> Vocabulary:
    return Vocabulary(
        ids={k: int(v) for k, v in preprocessing["aa_to_int"].items() if len(k) == 1},
        max_length=int(preprocessing["max_peptide_len"]),
        padding_index=int(preprocessing.get("padding_index", 0)),
    )


def tokenizer_from_preprocessing(model: str, preprocessing: Mapping) -> JChronologerTokenizer:
    """The tokenizer for ``model`` from its parsed ``.preprocessing.json``.

    Not checked against the snapshot (:func:`tokenizers_from_files` is):
    for files already verified, or for tests.
    """
    if model == "sculptor":
        spelling: Callable[[Sites], str] = _AccessionSpelling(preprocessing)
    elif model in ("cartographer", "electrician"):
        spelling = _JarSpelling(preprocessing, pyro_on_first_residue=False)
    elif model == "chronologer":
        spelling = _JarSpelling(preprocessing, pyro_on_first_residue=True)
    else:
        raise ValueError(f"no tokenizer for JChronologer model {model!r} (takes {', '.join(TOKENIZED_MODELS)})")
    return JChronologerTokenizer(model=model, vocabulary=_vocabulary(preprocessing), spelling=spelling)


def tokenizers_from_files(version: str, files: Mapping[str, bytes]) -> dict[str, JChronologerTokenizer]:
    """Tokenizers for JChronologer ``version`` from its preprocessing
    files (``{"chronologer": b"...", "cartographer": ..., "electrician":
    ..., "sculptor": ...}``, as read from the jar).

    Each file's SHA-256 must be the snapshot's for that version and role;
    anything else raises :class:`UnknownModelError`. Electrician gets
    Cartographer's tokens, as in the jar, and its own file must agree with
    Cartographer's on the characters and length.
    """
    known = load_snapshot()["versions"].get(version)
    if known is None:
        raise UnknownModelError(
            f"JChronologer {version} is not one Constellation was checked against "
            f"(known: {', '.join(load_snapshot()['versions'])}); run "
            "scripts/build-jchronologer-models-json.py and scripts/survey-jchronologer-tokens.py on it"
        )
    parsed: dict[str, Mapping] = {}
    for model in TOKENIZED_MODELS:
        if model not in files:
            raise ValueError(f"no preprocessing file for {model}")
        expected = known["models"][model]["preprocessing_sha256"]
        actual = hashlib.sha256(files[model]).hexdigest()
        if actual != expected:
            raise UnknownModelError(
                f"the {model} preprocessing file of JChronologer {version} has SHA-256 {actual}, "
                f"not the {expected} Constellation was checked against; rerun "
                "scripts/build-jchronologer-models-json.py and scripts/survey-jchronologer-tokens.py"
            )
        parsed[model] = json.loads(files[model])
    cart, elec = parsed["cartographer"], parsed["electrician"]
    for key in ("aa_to_int", "max_peptide_len", "mod_regex_rules", "nterm_keys"):
        if cart.get(key) != elec.get(key):
            raise UnknownModelError(
                f"Electrician's {key} differs from Cartographer's in JChronologer {version}, "
                "but the jar feeds Electrician Cartographer's tokens"
            )
    tokenizers = {m: tokenizer_from_preprocessing(m, parsed[m]) for m in ("chronologer", "cartographer", "sculptor")}
    tokenizers["electrician"] = JChronologerTokenizer(
        model="electrician",
        vocabulary=tokenizers["cartographer"].vocabulary,
        spelling=tokenizers["cartographer"].spelling,
    )
    return tokenizers
