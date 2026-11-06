"""Capture the tokens JChronologer itself gives a corpus of peptides.

Writes ``tests/data/jchronologer/<version>/tokens.tsv``: for each peptide
of a fixed corpus (written as Constellation writes peptidoforms, in
ProForma), the input JChronologer was given and, for Chronologer,
Cartographer and Sculptor, whether the jar took it and the tokens it
made. ``tests/test_tokenize_jchronologer.py`` checks
``constellation.massspec.tokenize.jchronologer`` against it, row for row.

The tokens come from the jar's own code: ``scripts/jchronologer/TokenOracle.java``
runs each model's preprocessing path of ``DefaultChronologerLibraryPredictor``
(see its header). It is compiled with the JDK on ``PATH`` (the conda env's
``openjdk``) and run with the JRE EncyclopeDIA ships, on the installed
the jars the installed EncyclopeDIA's manifest ``Class-Path`` names
(never ``lib/*``: it also holds a leftover ``jchronologer-1.0.9.jar``).

The corpus covers every modification JChronologer or Sculptor has a token
for at its residue, the N-terminal forms (acetyl, TMT, both pyro-Glu, the
cysteine cyclization), lengths at each model's limits, refusals (unknown
residues, modifications the jar cannot name, two on one residue, a
C-terminal one), and 200 random peptidoforms carrying the fifteen
modifications the jar can name.

Run from project root (needs EncyclopeDIA installed):
    python3 scripts/survey-jchronologer-tokens.py
"""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
import zipfile
from pathlib import Path

from constellation.massspec.tokenize.jchronologer import load_snapshot
from constellation.massspec.tokenize.vocabulary import Sites
from constellation.thirdparty.encyclopedia import jchronologer_jar, manifest_class_path
from constellation.thirdparty.registry import find

ORACLE = Path(__file__).parent / "jchronologer" / "TokenOracle.java"
OUT_DIR = Path("tests/data/jchronologer")
AA = "ACDEFGHIKLMNPQRSTVWY"


def _maps(jar: Path, snapshot_models: dict, role: str) -> tuple[list[tuple[str, str]], list[str]]:
    """(residue, accession) pairs and N-terminal accessions a model's file
    names (Scout's are the fifteen modifications the jar can name;
    Sculptor's add others the jar never sends it)."""
    with zipfile.ZipFile(jar) as archive:
        data = json.loads(archive.read(snapshot_models[role]["preprocessing_member"]))
    pairs = sorted((e["residue"], e["unimod"]) for e in data["residue_unimod_map"])
    return pairs, sorted(data["nterm_unimod_map"])


def corpus(pairs: list[tuple[str, str]], n_terms: list[str],
           jar_pairs: list[tuple[str, str]], jar_n_terms: list[str]) -> list[str]:
    peptides = [
        "PEPTIDEK", "LVNELTEFAK", "PEPTID", "PEPTI", "PEPTIDEKR",
        "A" * 30 + "K", "A" * 31 + "K", "A" * 49 + "K", "A" * 50 + "K",
        "PEPTUDEK", "PEPTXDEK", "PEPTBDEK",
    ]
    peptides += [f"GLA{residue}[{acc}]TEFAK" for residue, acc in pairs]
    peptides += [f"[{acc}]-AGLTEFAK" for acc in n_terms if acc not in ("UNIMOD:27", "UNIMOD:28")]
    peptides += [
        "[UNIMOD:28]-QPEPTIDEK", "Q[UNIMOD:28]PEPTIDEK", "[UNIMOD:27]-EPEPTIDEK", "E[UNIMOD:27]PEPTIDEK",
        "[UNIMOD:28]-QPEPTM[UNIMOD:35]IDEK", "[UNIMOD:28]-QC[UNIMOD:4]PEPTIDEK",
        "[UNIMOD:26]-CPEPTIDEK", "[UNIMOD:26]-CPEPTM[UNIMOD:35]IDEK",
        "[UNIMOD:1]-K[UNIMOD:1]VERPDK", "K[UNIMOD:1]VERPDK", "[UNIMOD:1]-KVERPDK",
        "[UNIMOD:1]-C[UNIMOD:4]PEPTIDEK", "[UNIMOD:1]-S[UNIMOD:21]PEPTIDEK", "[UNIMOD:1]-M[UNIMOD:35]PEPTIDEK",
        "[UNIMOD:737]-PEPTK[UNIMOD:737]IDEK", "[UNIMOD:739]-PEPTK[UNIMOD:739]IDEK",
        "[UNIMOD:1]-[UNIMOD:737]-PEPTIDEK".replace("]-[", "]["),
        "PEPTK[UNIMOD:1][UNIMOD:121]IDEK", "PEPTK[UNIMOD:280]IDEK", "PEPTIDEK-[UNIMOD:2]", "PEPTIDEK-[UNIMOD:35]",
        "[UNIMOD:58]-PEPTIDEK", "C[UNIMOD:4]C[UNIMOD:4]C[UNIMOD:4]PEPTIDEK",
        "PEPTM[UNIMOD:35]M[UNIMOD:35]M[UNIMOD:35]IDEK", "S[UNIMOD:21]T[UNIMOD:21]Y[UNIMOD:21]PEPTIDEK",
    ]
    rng = random.Random(20261106)
    by_residue: dict[str, list[str]] = {}
    for residue, acc in jar_pairs:
        by_residue.setdefault(residue, []).append(acc)
    for _ in range(200):
        length = rng.choice([rng.randint(6, 15), rng.randint(6, 35), rng.randint(30, 52)])
        residues = [rng.choice(AA) for _ in range(length)]
        text = ""
        if rng.random() < 0.25:
            text = f"[{rng.choice(jar_n_terms)}]-"
            if text == "[UNIMOD:28]-":
                residues[0] = "Q"
            elif text == "[UNIMOD:27]-":
                residues[0] = "E"
        for residue in residues:
            text += residue
            if residue in by_residue and rng.random() < 0.3:
                text += f"[{rng.choice(by_residue[residue])}]"
        peptides.append(text)
    return list(dict.fromkeys(peptides))


def jar_input(proforma: str) -> str:
    """JChronologer's UNIMOD form of a peptide; S-carbamoylmethylcysteine
    cyclization (UNIMOD:26), which the jar cannot name, as the jar writes
    that molecule (UNIMOD:28 on the N-terminus of a carbamidomethyl C)."""
    sites = Sites.from_peptidoform(proforma)
    n_term = list(sites.n_term)
    mods = [list(m) for m in sites.residue_mods]
    if "UNIMOD:26" in n_term and sites.residues.startswith("C"):
        n_term.remove("UNIMOD:26")
        n_term.append("UNIMOD:28")
        mods[0].append("UNIMOD:4")

    def block(accs: list[str]) -> str:
        return "".join(f"[{a}]" for a in accs) or "[]"

    body = "".join(r + "".join(f"[{a}]" for a in m) for r, m in zip(sites.residues, mods))
    return f"{block(n_term)}-{body}-{block(list(sites.c_term))}"


def main() -> int:
    encyclopedia = find("encyclopedia")
    jar = jchronologer_jar(encyclopedia.path)
    java = encyclopedia.path.parent / "jre" / "bin" / "java"
    snapshot = load_snapshot()
    with zipfile.ZipFile(jar) as archive:
        version = next(
            line.split("=", 1)[1].strip()
            for line in archive.read("META-INF/maven/org.searlelab/jchronologer/pom.properties").decode().splitlines()
            if line.startswith("version=")
        )
    models = snapshot["versions"][version]["models"]
    jar_pairs, jar_n_terms = _maps(jar, models, "scout")
    sculptor_pairs, sculptor_n_terms = _maps(jar, models, "sculptor")
    peptides = corpus(
        sorted(set(jar_pairs) | set(sculptor_pairs)), sorted(set(jar_n_terms) | set(sculptor_n_terms)),
        jar_pairs, jar_n_terms,
    )
    inputs = [jar_input(p) for p in peptides]

    # Exactly the jars EncyclopeDIA loads: lib/ also holds a leftover
    # jchronologer-1.0.9.jar, and a wildcard classpath can load its classes.
    class_path = ":".join(str(p) for p in manifest_class_path(encyclopedia.path))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["javac", "--release", "21", "-cp", class_path, "-d", tmp, str(ORACLE)], check=True
        )
        result = subprocess.run(
            [str(java if java.exists() else "java"), "-cp", f"{tmp}:{class_path}", "TokenOracle",
             models["chronologer"]["preprocessing_member"], models["cartographer"]["preprocessing_member"],
             models["sculptor"]["preprocessing_member"]],
            input="\n".join(inputs) + "\n", capture_output=True, text=True, check=True,
        )
    by_input: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        by_input.setdefault(line.split("\t", 1)[0], []).append(line)

    out = OUT_DIR / version / "tokens.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"# Tokens JChronologer {version} gives each peptide (scripts/survey-jchronologer-tokens.py; do not edit).",
        f"# jar {jar.name} sha256 {snapshot['versions'][version]['jar_sha256']}",
        "# tokens: the ids up to the end token; the rest of the row (width columns) is padding (0)",
        "proforma\tjar_input\tmodel\tstatus\tdetail\twidth\ttokens",
    ]
    for peptide, given in zip(peptides, inputs):
        lines = by_input.get(given, [])
        if len(lines) != 3:
            raise SystemExit(f"the oracle printed {len(lines)} rows for {given!r}")
        for line in lines:
            *head, tokens = line.split("\t")
            ids = tokens.split()
            cut = ids.index("0") if "0" in ids else len(ids)
            if any(i != "0" for i in ids[cut:]):
                raise SystemExit(f"padding is not trailing in {line!r}")
            rows.append("\t".join([peptide, *head, str(len(ids)) if ids else "", " ".join(ids[:cut])]))
    out.write_text("\n".join(rows) + "\n")
    body = [r for r in rows if not r.startswith(("#", "proforma\t"))]
    n_accepted = sum(1 for r in body if r.split("\t")[3] == "accepted")
    print(f"wrote {out}: {len(peptides)} peptides, {len(body)} rows, {n_accepted} accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
