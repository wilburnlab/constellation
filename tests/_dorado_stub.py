"""A mock ``dorado`` for tests, and an argv parser that reads it as Dorado does.

The stub is a bash script found through ``$CONSTELLATION_DORADO_HOME``. It
records every invocation and can be told to fail, so tests can check what
a run asked Dorado for and what happens to its files when Dorado dies:

- ``$MOCK_ARGV_FILE``: the last invocation's argv, one token per line;
- ``$MOCK_CALLS_FILE``: one line per invocation, appended;
- ``$MOCK_BAM_SRC``: a file to write as the basecalled output (a real BAM
  when a later stage reads its header), instead of ``MOCKBAM``;
- ``$MOCK_FAIL_ON``: when any argv token contains it, write
  ``$MOCK_PARTIAL_BYTES`` bytes (or the file ``$MOCK_PARTIAL_SRC``, when a
  later run must read it as a BAM) and exit 1, as a run killed part-way
  would;
- ``$MOCK_DORADO_VERSION``: what ``dorado --version`` reports.

``--resume-from FILE`` is honoured the way Dorado honours it: the resume
file's reads are written to the new output first, then the new calls.

Substring checks on a joined argv cannot tell whether Dorado reads a token
as a positional or as an option's value, which is how a model swallowed by
``--modified-bases`` went unnoticed. :func:`parse_dorado_argv` parses the
recorded tokens with Dorado's arities instead, so a test can assert what
the positionals are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_VERSION = "2.0.0"

STUB = r"""#!/usr/bin/env bash
verb="${1:-}"
if [[ "$verb" == "--version" ]]; then
  echo "dorado ${MOCK_DORADO_VERSION:-__DEFAULT_VERSION__}+mock"; exit 0
fi
if [[ -n "${MOCK_ARGV_FILE:-}" ]]; then printf '%s\n' "$@" > "${MOCK_ARGV_FILE}"; fi
if [[ -n "${MOCK_CALLS_FILE:-}" ]]; then printf '%s\n' "$*" >> "${MOCK_CALLS_FILE}"; fi
if [[ -n "${MOCK_FAIL_ON:-}" ]]; then
  for a in "$@"; do
    if [[ "$a" == *"${MOCK_FAIL_ON}"* ]]; then
      if [[ -n "${MOCK_PARTIAL_SRC:-}" ]]; then cat "${MOCK_PARTIAL_SRC}";
      else head -c "${MOCK_PARTIAL_BYTES:-0}" /dev/zero | tr '\0' 'P'; fi
      echo "[error] mock failure" >&2
      exit 1
    fi
  done
fi
resume=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  [[ "${args[$i]}" == "--resume-from" ]] && resume="${args[$((i+1))]}"
done
case "$verb" in
  basecaller|duplex)
    [[ -n "$resume" ]] && cat "$resume"
    if [[ -n "${MOCK_BAM_SRC:-}" ]]; then cat "${MOCK_BAM_SRC}"; else printf 'MOCKBAM'; fi
    echo "[info] basecalling 50% done" >&2; exit 0 ;;
  aligner) printf 'MOCKALN'; exit 0 ;;
  polish) draft="${@: -1}"; cat "$draft"; exit 0 ;;
  summary) printf 'read_id\tn\nr1\t1\n'; exit 0 ;;
  download|demux) exit 0 ;;
  *) printf 'MOCKOUT'; exit 0 ;;
esac
""".replace("__DEFAULT_VERSION__", DEFAULT_VERSION)


@dataclass(frozen=True)
class DoradoStub:
    home: Path
    argv_file: Path
    calls_file: Path

    def argv(self) -> list[str]:
        """The last invocation's argv (without the binary)."""
        return self.argv_file.read_text().splitlines()

    def calls(self) -> list[str]:
        """Every invocation so far, each as one space-joined line."""
        if not self.calls_file.exists():
            return []
        return self.calls_file.read_text().splitlines()


def install(tmp_path: Path, monkeypatch, *, name: str = "dorado_home") -> DoradoStub:
    """Write the stub under ``tmp_path`` and point the registry at it."""
    home = tmp_path / name
    (home / "bin").mkdir(parents=True, exist_ok=True)
    binary = home / "bin" / "dorado"
    binary.write_text(STUB)
    binary.chmod(0o755)
    monkeypatch.setenv("CONSTELLATION_DORADO_HOME", str(home))
    stub = DoradoStub(
        home=home,
        argv_file=tmp_path / f"{name}.argv",
        calls_file=tmp_path / f"{name}.calls",
    )
    monkeypatch.setenv("MOCK_ARGV_FILE", str(stub.argv_file))
    monkeypatch.setenv("MOCK_CALLS_FILE", str(stub.calls_file))
    return stub


# Dorado's options as its argument parser reads them: how many values each
# takes. ``"+"`` is a greedy list — every following token up to the next
# option, which is why such an option has to come after the positionals.
_ARITY: dict[str, int | str] = {
    "-x": 1,
    "--device": 1,
    "--resume-from": 1,
    "--emit-moves": 0,
    "--modified-bases": "+",
    "--modified-bases-models": 1,
    "--modified-bases-threshold": 1,
    "--models-directory": 1,
    "--model-override": 1,
    "--RG": 1,
    "-t": 1,
    "--threads": 1,
    "--model": 1,
    "--output-dir": 1,
    "--kit-name": 1,
    "--no-classify": 0,
    "--emit-fastq": 0,
}


def parse_dorado_argv(tokens: list[str]) -> tuple[str, list[str], dict[str, list[str]]]:
    """``(verb, positionals, options)`` as Dorado would read ``tokens``.

    An option this table does not know raises, so a test cannot pass by
    accident on a flag whose arity it guessed.
    """
    if not tokens:
        raise ValueError("empty argv")
    verb, rest = tokens[0], tokens[1:]
    positionals: list[str] = []
    options: dict[str, list[str]] = {}
    i = 0
    while i < len(rest):
        token = rest[i]
        if token.startswith("-"):
            if token not in _ARITY:
                raise ValueError(f"unknown dorado option in test argv: {token!r}")
            arity = _ARITY[token]
            i += 1
            values: list[str] = []
            if arity == "+":
                while i < len(rest) and not rest[i].startswith("-"):
                    values.append(rest[i])
                    i += 1
                if not values:
                    raise ValueError(f"{token} needs at least one value")
            else:
                values = rest[i : i + int(arity)]
                if len(values) != int(arity):
                    raise ValueError(f"{token} needs {arity} value(s)")
                i += int(arity)
            options.setdefault(token, []).extend(values)
        else:
            positionals.append(token)
            i += 1
    return verb, positionals, options
