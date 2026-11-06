"""Capture JChronologer's own library predictions for a set of requests.

Writes ``tests/data/jchronologer/1.0.10/predictions.tsv``: for each
(peptide, charges, NCE) request below, what ``ChronologerLibraryPredictor``
itself returns (``scripts/jchronologer/PredictionOracle.java``): the
retention time, every reported fragment with its m/z and intensity, the
CCS and, for automatic charge selection, each charge's probability. Or
that it refused the request. ``tests/test_jchronologer_golden.py`` runs
Constellation's predictors on the same requests and compares (it needs
the installed jar for the models, so it is skipped without one).

The peptides are written as Constellation writes them and given to the jar
in its UNIMOD form (``scripts/survey-jchronologer-tokens.py``'s
``jar_input``). Energies are float32-exact, so the jar's double path and
Constellation's float32 requests carry the same value.

Run from project root (needs EncyclopeDIA installed and a JDK on PATH):
    python3 scripts/survey-jchronologer-predictions.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from pathlib import Path

from constellation.massspec.tokenize.jchronologer import load_snapshot
from constellation.thirdparty.encyclopedia import jchronologer_jar, manifest_class_path
from constellation.thirdparty.registry import find

HERE = Path(__file__).parent
ORACLE = HERE / "jchronologer" / "PredictionOracle.java"
OUT = Path("tests/data/jchronologer")

_spec = importlib.util.spec_from_file_location("survey_tokens", HERE / "survey-jchronologer-tokens.py")
_tokens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tokens)
jar_input = _tokens.jar_input

PEPTIDES = [
    # Bovine serum albumin tryptic peptides, carbamidomethyl C.
    "LVNELTEFAK", "YLYEIAR", "HLVDEPQNLIK", "AEFVEVTK", "LGEYGFQNALIVR", "KVPQVSTPTLVEVSR",
    "DAFLGSFLYEYSR", "RHPEYAVSVLLR", "LKPDPNTLC[UNIMOD:4]DEFK", "SHC[UNIMOD:4]IAEVEK",
    "EC[UNIMOD:4]C[UNIMOD:4]HGDLLEC[UNIMOD:4]ADDR", "QTALVELLK", "FKDLGEEHFK", "TVMENFVAFVDK",
    "TVM[UNIMOD:35]ENFVAFVDK", "LFTFHADIC[UNIMOD:4]TLPDTEK", "GLVLIAFSQYLQQC[UNIMOD:4]PFDEHVK",
    "VPQVSTPTLVEVSRSLGK", "EYEATLEEC[UNIMOD:4]C[UNIMOD:4]AKDDPHAC[UNIMOD:4]YSTVFDK", "NYQEAK", "ATEEQLK",
    # Terminal forms.
    "[UNIMOD:1]-ADEEKLPPGWEK", "[UNIMOD:1]-M[UNIMOD:35]PEPTIDEK", "[UNIMOD:28]-QTALVELLK",
    "[UNIMOD:27]-EAC[UNIMOD:4]FAVEGPK", "[UNIMOD:26]-CC[UNIMOD:4]TESLVNR", "[UNIMOD:1]-K[UNIMOD:1]VERPDK",
    "[UNIMOD:737]-LVNELTEFAK[UNIMOD:737]", "[UNIMOD:739]-YLYEIAR",
    # Other modifications the models take.
    "LVNELTEFAK[UNIMOD:121]", "S[UNIMOD:21]HC[UNIMOD:4]IAEVEK", "AEFVEVT[UNIMOD:21]K", "DAFLGSFLY[UNIMOD:21]EYSR",
    "K[UNIMOD:36]VPQVSTPTLVEVSR", "LVNELTEFAK[UNIMOD:34]", "RHPEYAVSVLLR[UNIMOD:36]", "LVNELTEFAK[UNIMOD:64]",
    # Refused by the jar (Chronologer has no deamidation, HexNAc or oxidised W).
    "LGEYGFQN[UNIMOD:7]ALIVR", "HLVDEPQN[UNIMOD:43]LIK", "W[UNIMOD:35]LVNELTEFAK",
]
REQUESTS = (
    [(p, "1,2,3,4", "27.5") for p in PEPTIDES]
    + [(p, "2,3", "33.0") for p in PEPTIDES[:12]]
    + [(p, "2,3", "20.0") for p in PEPTIDES[:6]]
    + [(p, "2,3", "45.0") for p in PEPTIDES[:6]]
    + [(p, "auto:0.0", "30.0") for p in PEPTIDES[:8] + PEPTIDES[21:29]]
)


def main() -> int:
    encyclopedia = find("encyclopedia")
    jar = jchronologer_jar(encyclopedia.path)
    java = encyclopedia.path.parent / "jre" / "bin" / "java"
    class_path = ":".join(str(p) for p in manifest_class_path(encyclopedia.path))
    version = jar.stem.split("-", 1)[1]
    inputs = [(jar_input(p), z, nce) for p, z, nce in REQUESTS]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["javac", "--release", "21", "-cp", class_path, "-d", tmp, str(ORACLE)], check=True)
        result = subprocess.run(
            [str(java if java.exists() else "java"), "-cp", f"{tmp}:{class_path}", "PredictionOracle"],
            input="".join(f"{a}\t{z}\t{n}\n" for a, z, n in inputs), capture_output=True, text=True, check=True,
        )
    by_request: dict[tuple[str, str, str], list[str]] = {}
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        by_request.setdefault(tuple(fields[:3]), []).append("\t".join(fields[3:]))
    out = OUT / version / "predictions.tsv"
    rows = [
        f"# JChronologer {version}'s own library predictions (scripts/survey-jchronologer-predictions.py; do not edit).",
        f"# jar {jar.name} sha256 {load_snapshot()['versions'][version]['jar_sha256']}",
        "proforma\tcharges\tnce\tunimod\tcharge\tprecursor_mz\trt_seconds\tccs\tcharge_probability\tions",
    ]
    for (peptide, charges, nce), given in zip(REQUESTS, inputs):
        got = by_request.get(given)
        if not got:
            raise SystemExit(f"the oracle printed nothing for {given!r}")
        rows += [f"{peptide}\t{charges}\t{nce}\t{line}" for line in got]
    out.write_text("\n".join(rows) + "\n")
    errors = sum(1 for r in rows[3:] if r.split("\t")[3] == "ERROR")
    print(f"wrote {out}: {len(REQUESTS)} requests, {len(rows) - 3 - errors} entries, {errors} refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
