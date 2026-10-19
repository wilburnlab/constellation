"""EncyclopeDIA's command-line surface, as data.

The jar's programs and flags are not written down by hand anywhere in
Constellation. They are *captured* from the pinned jar's own ``-h``
output and checked in, so a wrapper can be tested against what the jar
actually accepts without a JVM, and a version bump shows up as a diff:

    constellation/data/_raw/encyclopedia/<version>/help/<program>.txt
        Normalised ``-h`` output, one file per program (written by
        :func:`capture_help`; needs the jar).
    constellation/data/encyclopedia_cli.json
        Programs and flags parsed from those files (written by
        :func:`build_cli_spec`; pure text → JSON, so CI can rebuild it).

``scripts/survey-encyclopedia.py`` and
``scripts/build-encyclopedia-cli-json.py`` are the two entry points.

Why this exists: the hand-maintained flag notes it replaces listed
``-scribetwodda`` / ``-scribetwodia`` for a jar whose ScribeTwo is a
single ``-scribetwo -searchType ...`` program. Nothing caught that,
because nothing compared the notes with a jar.

Programs are named by their argv without dashes, dot-joined:
``scribetwo``, ``libexport``, ``convert.processDIA``. Two names are
special — ``main`` is the dispatcher's own ``-h`` (it lists the other
programs), and ``encyclopedia`` is the default, peptide-centric search,
which runs when no program flag is given.

What help does not tell you: the jar has parameters it does not print
(``-quantifyAcrossSamples``, ``-proteinGroupScoreMode``, ...; a result
library's ``metadata`` table shows the full effective set). So "not in
the snapshot" means "not documented by the jar", which is a reason to
look, not proof the jar will reject it. The ones seen so far are listed
in :data:`UNDOCUMENTED_PARAMETERS`, as a record only.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = [
    "CLI_JSON_NAME",
    "HELP_NAME_WIDTH",
    "UNDOCUMENTED_PARAMETERS",
    "HelpProbe",
    "EXTRA_PROBES",
    "normalize_help",
    "parse_help",
    "capture_help",
    "write_capture",
    "raw_help_dir",
    "build_cli_spec",
    "load_cli_spec",
    "program_names",
    "program_argv",
    "known_flags",
    "flag_default",
    "is_known_flag",
    "unknown_flags",
]

CLI_JSON_NAME = "encyclopedia_cli.json"

#: Placeholder written in place of a default that is the host's core
#: count, so a snapshot captured on one machine equals one from another.
HOST_CORES = "<host cores>"

_TIMESTAMP_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] ?")
_THREADS_DEFAULT_RE = re.compile(r"^(\t-numberOfThreadsUsed\s+\(default: )\d+(\))", re.M)
_SECTION_RE = re.compile(r"^(?P<name>[A-Z][^\t]*?):\s*$")
_FLAG_TOKEN_RE = re.compile(r"^-[A-Za-z][A-Za-z0-9]*$")
# A default is always inside a parenthesised group, alone or with other
# comma-separated parts:
#   (default: X)   (default=X)   (default: X, values: a|b|c)
#   (true/false, default: [X])   (DDA mode only, default: false)
_DEFAULT_GROUP_RE = re.compile(r"\(([^()]*\bdefault\s*[:=][^()]*)\)")
_DEFAULT_PART_RE = re.compile(r"default\s*[:=]\s*(.*)$")
_ENUM_PART_RE = re.compile(r"[\w.]+(?:/[\w.]+)+")
_SUBPROGRAM_SECTIONS = frozenset({"Other Programs", "Available converters"})
#: The shared-parameter table in the jar's help cuts flag names off at this
#: many characters: it prints ``-numberOfExtraDecoyLibrariesSearche`` for
#: ``-numberOfExtraDecoyLibrariesSearched``. A documented name of exactly
#: this length is therefore treated as a prefix.
HELP_NAME_WIDTH = 35
_REQUIRED_SECTIONS = frozenset({"Required Parameters"})


#: Parameters the 6.9.23 jar knows about but does not print in ``-h``.
#:
#: A record, not an interface: nothing in Constellation passes these, and
#: none is needed for the searches we run today. They are kept so that
#: whoever next needs one (or wonders what a result's metadata key means)
#: does not have to rediscover them.
#:
#: Provenance, per program. ``encyclopedia`` and ``scribetwo``: names a
#: result library's ``metadata`` table records as effective parameters,
#: from a peptide-centric DIA search and a ScribeTwo DDA search run in the
#: GUI on 6.9.23. ``libexport``: option strings read from the jar's
#: classes. In neither case has the name been tried on the command line,
#: so "the jar has this parameter" is established and "the CLI accepts
#: this flag" is not. Some are clearly derived rather than settable
#: (``-percolatorVersionNumber``). The top-level
#: ``-enableAdvancedOptions`` flag may be what exposes the rest.
#:
#: ``tests/test_encyclopedia_cli_spec.py`` fails if one of these turns up
#: in the help snapshot, which is the cue to delete it from here.
UNDOCUMENTED_PARAMETERS: Mapping[str, tuple[str, ...]] = {
    "encyclopedia": (
        "-instrument",
        "-percolatorModelFile",
        "-percolatorTrainingIterations",
        "-percolatorVersionNumber",
        "-precursorIsolationRangeFile",
        "-proteinGroupScoreMode",
        "-quantifyAcrossSamples",
        "-targetWindowCenter",
    ),
    "scribetwo": (
        "-acceptWeakPrecursors",
        "-adjustTolerances",
        "-chimericSearching",
        "-instrument",
        "-isomerSearch",
        "-percolatorModelFile",
        "-percolatorTrainingIterations",
        "-percolatorVersionNumber",
        "-precursorIsolationRangeFile",
        "-proteinGroupScoreMode",
        "-quantifyAcrossSamples",
        "-scribeTwoBucketSize",
        "-scribeTwoMinFragMz",
        "-scribeTwoTopCandidates",
        "-scribeTwoUseProteinContextFeatures",
        "-scribeTwoUseSharedChargeStateSupportFeatures",
        "-targetWindowCenter",
    ),
    "libexport": (
        "-alignOnly",
        "-alignmentFrom",
        "-quantifyAcrossSamples",
    ),
}


@dataclass(frozen=True)
class HelpProbe:
    """One ``<argv> -h`` invocation to record.

    ``listed`` is whether a parent program's help advertises it;
    ``alias_of`` names the program whose help it repeats.
    """

    name: str
    argv: tuple[str, ...]
    listed: bool = True
    alias_of: str | None = None
    note: str | None = None


#: Probes the crawl cannot discover on its own: the two roots, plus
#: programs the jar accepts without listing them. ``-browser`` is left out
#: on purpose — it opens the GUI instead of printing help.
EXTRA_PROBES: tuple[HelpProbe, ...] = (
    HelpProbe("main", (), note="dispatcher help: `-h` with no program"),
    HelpProbe(
        "encyclopedia",
        ("-encyclopedia",),
        note=(
            "the default, peptide-centric search; runs when no program flag "
            "is given, so its argv prefix is empty"
        ),
    ),
    HelpProbe("pecan", ("-pecan",), listed=False, alias_of="walnut"),
    HelpProbe(
        "convert.mergeDIA",
        ("-convert", "-mergeDIA"),
        listed=False,
        alias_of="convert.processDIA",
    ),
    HelpProbe(
        "convert.fastaToKoinaLibrary",
        ("-convert", "-fastaToKoinaLibrary"),
        listed=False,
    ),
)


# ──────────────────────────────────────────────────────────────────────
# Text → structure
# ──────────────────────────────────────────────────────────────────────


# The helper thread's line, with its own timestamp when it starts
# mid-line: "... [HH:MM:SS] Thermo server: <message>".
_THERMO_LINE_RE = re.compile(r"(?:\[\d{2}:\d{2}:\d{2}\] )?Thermo server:.*$")


def normalize_help(raw: str) -> str:
    """Strip what varies between runs from a jar ``-h`` capture.

    Removes the ``[HH:MM:SS]`` log prefixes and the MSRawJava
    "Thermo server" line (written from another thread, so it lands at a
    different place each run), and replaces the thread-count default,
    which is the host's core count, with :data:`HOST_CORES`.

    The helper thread does not wait for a line break. About one capture
    in thirty it writes into the middle of another line::

        Exception in thread "main" [23:21:49] Thermo server: extracting...
        java.lang.NullPointerException

    so what stood before it is kept and rejoined with what follows,
    rather than dropped along with it.
    """
    lines: list[str] = []
    carry = ""
    for line in raw.replace("\r\n", "\n").split("\n"):
        line = carry + _TIMESTAMP_RE.sub("", line)
        carry = ""
        interrupted = _THERMO_LINE_RE.search(line)
        if interrupted:
            carry = line[: interrupted.start()]
            continue
        lines.append(line.rstrip())
    if carry.strip():
        lines.append(carry.rstrip())
    while lines and not lines[-1]:
        lines.pop()
    text = "\n".join(lines) + "\n"
    return _THREADS_DEFAULT_RE.sub(rf"\g<1>{HOST_CORES}\g<2>", text)


def _split_default(text: str) -> tuple[str | None, list[str] | None, str | None]:
    """Pull ``(default, values, remaining description)`` out of a flag's text.

    Parts of the group that are neither the default nor an enumeration
    of values (``DDA mode only``) stay in the description.
    """
    m = _DEFAULT_GROUP_RE.search(text)
    if m is None:
        return None, None, text.strip() or None
    default: str | None = None
    values: list[str] | None = None
    kept: list[str] = []
    for part in (p.strip() for p in m.group(1).split(",")):
        dm = _DEFAULT_PART_RE.match(part)
        if dm is not None:
            default = dm.group(1).strip()
            if default.startswith("[") and default.endswith("]"):
                default = default[1:-1].strip()
        elif part.startswith("values:"):
            values = [v.strip() for v in part[len("values:"):].split("|")]
        elif _ENUM_PART_RE.fullmatch(part):
            values = part.split("/")
        elif part:
            kept.append(part)
    middle = f"({', '.join(kept)})" if kept else ""
    rest = (text[: m.start()] + middle + text[m.end():]).strip(" ,\t")
    return default, values, rest or None


def parse_help(text: str) -> dict[str, Any]:
    """Parse one program's normalised ``-h`` text.

    Returns ``{"title", "summary", "flags", "flag_patterns",
    "subprograms", "notes"}``. A flag entry is ``{"required", "section"}``
    plus ``description`` / ``default`` / ``values`` when the line gives
    them. Lines the grammar does not recognise are kept verbatim under
    ``notes`` rather than dropped, so nothing in the capture is lost.

    The jar's help is hand-formatted, so the grammar is the handful of
    shapes it actually uses::

        \\t-flag\\tdescription (default: X)
        \\t-flag      (default: X, values: a|b|c)
        \\t-a/-b\\tdescription              two flags, one line
        \\t-ptm...\\tdescription            a family, not one flag
        \\t\\tcontinuation of the line above
    """
    lines = text.split("\n")
    title = lines[0].strip() if lines else ""
    summary: list[str] = []
    flags: dict[str, dict[str, Any]] = {}
    flag_patterns: dict[str, str] = {}
    subprograms: dict[str, str] = {}
    notes: list[str] = []
    section: str | None = None
    last: dict[str, Any] | None = None

    for line in lines[1:]:
        if not line.strip():
            last = None
            continue
        m = _SECTION_RE.match(line)
        if m is not None:
            section = m.group("name").strip()
            last = None
            continue
        if not line.startswith("\t"):
            (summary if section is None else notes).append(line.strip())
            last = None
            continue
        if line.startswith("\t\t"):
            extra = line.strip()
            if last is not None:
                last["description"] = f"{last.get('description') or ''} {extra}".strip()
            else:
                notes.append(extra)
            continue

        body = line[1:]
        head, _, tail = body.partition("\t")
        if not tail:
            # `-flag   (default: X)` — spaces, not a tab, before the text.
            head, _, tail = body.partition(" ")
        head, tail = head.strip(), tail.strip()

        if section in _SUBPROGRAM_SECTIONS and _FLAG_TOKEN_RE.match(head):
            subprograms[head] = tail
            last = None
            continue
        if head.endswith("..."):
            flag_patterns[head[:-3] + "*"] = tail
            last = None
            continue
        names = head.split("/")
        if not all(_FLAG_TOKEN_RE.match(n) for n in names):
            notes.append(body.strip())
            last = None
            continue

        default, values, description = _split_default(tail)
        entry: dict[str, Any] = {
            # The jar lists a few defaulted flags under "Required"
            # (`-libexport -a`); a flag with a default is not required.
            "required": section in _REQUIRED_SECTIONS and default is None,
            "section": section,
        }
        if description:
            entry["description"] = description
        if default is not None:
            entry["default"] = default
        if values:
            entry["values"] = values
        for name in names:
            # A flag printed twice (ScribeTwo repeats `-fixed`) keeps the
            # first, more specific line and gains the later default.
            if name in flags:
                for key, value in entry.items():
                    flags[name].setdefault(key, value)
            else:
                flags[name] = dict(entry)
        last = flags[names[0]] if len(names) == 1 else None

    return {
        "title": title,
        "summary": " ".join(summary),
        "flags": flags,
        "flag_patterns": flag_patterns,
        "subprograms": subprograms,
        "notes": notes,
    }


# ──────────────────────────────────────────────────────────────────────
# Jar → text (needs an install)
# ──────────────────────────────────────────────────────────────────────


def _run_help(java: Path, jar: Path, argv: Sequence[str], *, timeout: float) -> str:
    """Run ``<argv> -h`` and return normalised stdout+stderr.

    Runs in a throwaway directory with a small heap: some programs write
    incidental files to the working directory before they look at their
    arguments. The same directory is the JVM's home: ``-scribetwo -h``
    creates an empty prediction cache in ``~/.encyclopedia/`` (measured
    2026-10-17), which a help capture has no business leaving behind.
    """
    from constellation.thirdparty.jvm import capture_jar

    with tempfile.TemporaryDirectory(prefix="encyclopedia-help-") as cwd:
        out = capture_jar(
            java, jar, [*argv, "-h"], jvm_args=["-Xmx1g", f"-Duser.home={cwd}"],
            timeout=timeout, cwd=Path(cwd),
        )  # fmt: skip
    return normalize_help(out)


def _sub_name(parent: str, flag: str) -> str:
    return flag.lstrip("-") if parent == "main" else f"{parent}.{flag.lstrip('-')}"


def capture_help(
    *,
    jar: Path | None = None,
    java: Path | None = None,
    timeout: float = 120.0,
) -> tuple[dict[str, str], list[HelpProbe]]:
    """Capture ``-h`` for every program of an installed jar.

    Starts from the dispatcher (``-h``), follows the programs it lists
    and the converters ``-convert -h`` lists, then adds
    :data:`EXTRA_PROBES`. Returns ``(texts, probes)`` keyed by program
    name. ``jar`` / ``java`` default to the registry-resolved install
    and the Java the runner would use for it.
    """
    if jar is None or java is None:
        from constellation.thirdparty import encyclopedia as _adapter  # noqa: F401
        from constellation.thirdparty.jvm import _resolve_java
        from constellation.thirdparty.registry import find

        handle = find("encyclopedia")
        jar = jar or handle.path
        java = java or _resolve_java(handle)[0]

    probes: dict[str, HelpProbe] = {p.name: p for p in EXTRA_PROBES}
    texts: dict[str, str] = {}

    def _visit(probe: HelpProbe) -> dict[str, Any]:
        texts[probe.name] = _run_help(java, jar, probe.argv, timeout=timeout)
        return parse_help(texts[probe.name])

    main = _visit(probes["main"])
    for flag in main["subprograms"]:
        name = _sub_name("main", flag)
        if name == "browser":  # opens the GUI rather than printing help
            continue
        probes.setdefault(name, HelpProbe(name, (flag,)))
    convert = _visit(probes.setdefault("convert", HelpProbe("convert", ("-convert",))))
    for flag in convert["subprograms"]:
        name = _sub_name("convert", flag)
        probes.setdefault(name, HelpProbe(name, ("-convert", flag)))
    for name in sorted(probes):
        if name not in texts:
            _visit(probes[name])
    return texts, [probes[name] for name in sorted(probes)]


# ──────────────────────────────────────────────────────────────────────
# Text → JSON
# ──────────────────────────────────────────────────────────────────────


def build_cli_spec(help_dir: Path, *, version: str) -> dict[str, Any]:
    """Build the packaged CLI document from a captured ``help/`` directory.

    Reads ``<help_dir>/<program>.txt`` plus ``<help_dir>/../programs.json``
    (the probe list :func:`capture_help` returned). Pure: no jar needed.
    """
    index = json.loads((help_dir.parent / "programs.json").read_text())
    programs: dict[str, Any] = {}
    for entry in index["programs"]:
        name = entry["name"]
        parsed = parse_help((help_dir / f"{name}.txt").read_text())
        # The default search runs with no program flag at all.
        argv = [] if name in ("main", "encyclopedia") else list(entry["argv"])
        programs[name] = {
            "argv": argv,
            "listed": entry["listed"],
            **({"alias_of": entry["alias_of"]} if entry.get("alias_of") else {}),
            **({"note": entry["note"]} if entry.get("note") else {}),
            **parsed,
        }
    return {"encyclopedia_version": version, "programs": programs}


# ──────────────────────────────────────────────────────────────────────
# JSON → queries
# ──────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def load_cli_spec() -> Mapping[str, Any]:
    """The packaged CLI document for the pinned EncyclopeDIA version."""
    with resources.files("constellation.data").joinpath(CLI_JSON_NAME).open() as fh:
        return json.load(fh)


def _program(name: str) -> Mapping[str, Any]:
    programs = load_cli_spec()["programs"]
    if name not in programs:
        known = ", ".join(sorted(programs))
        raise KeyError(f"unknown EncyclopeDIA program {name!r}; known: {known}")
    return programs[name]


def program_names() -> tuple[str, ...]:
    return tuple(sorted(load_cli_spec()["programs"]))


def program_argv(name: str) -> tuple[str, ...]:
    """The argv prefix that selects ``name`` (empty for the default search)."""
    return tuple(_program(name)["argv"])


def known_flags(name: str) -> frozenset[str]:
    """Flags the jar's help documents for ``name`` (families excluded)."""
    return frozenset(_program(name)["flags"])


def flag_default(name: str, flag: str) -> str | None:
    """The default the jar's help prints for ``flag`` of program ``name``
    (``None`` when it prints none, or does not document the flag)."""
    return _program(name)["flags"].get(flag, {}).get("default")


def is_known_flag(name: str, flag: str) -> bool:
    """Whether ``flag`` is documented for ``name``, by name or by family
    (ScribeTwo documents its PTM flags as ``-ptm...``)."""
    program = _program(name)
    if flag in program["flags"]:
        return True
    if any(flag.startswith(p[:-1]) for p in program["flag_patterns"]):
        return True
    return any(len(f) == HELP_NAME_WIDTH and flag.startswith(f) for f in program["flags"])


def unknown_flags(name: str, argv: Iterable[str]) -> list[str]:
    """Flag-shaped tokens in ``argv`` that the jar's help does not document.

    ``argv`` is the arguments *after* the program prefix. A token counts
    as a flag when it starts with ``-`` and a letter, so negative numbers
    and the bare ``-`` ScribeTwo uses for "none" are values, not flags.
    """
    prefix = set(program_argv(name))
    return [
        tok
        for tok in argv
        if _FLAG_TOKEN_RE.match(str(tok)) and tok not in prefix and not is_known_flag(name, tok)
    ]


def _default_raw_root() -> Path:
    return Path(str(resources.files("constellation.data"))) / "_raw" / "encyclopedia"


def raw_help_dir(version: str) -> Path:
    """``constellation/data/_raw/encyclopedia/<version>/help`` in this checkout."""
    return _default_raw_root() / version / "help"


def write_capture(
    texts: Mapping[str, str],
    probes: Sequence[HelpProbe],
    *,
    version: str,
    jar_sha256: str,
    java_version: str | None,
    root: Path | None = None,
) -> Path:
    """Write a :func:`capture_help` result under ``_raw/encyclopedia/<version>/``.

    Replaces the ``help/`` directory wholesale, so a program dropped by a
    new jar disappears from the snapshot instead of lingering.
    """
    version_dir = (root or _default_raw_root()) / version
    help_dir = version_dir / "help"
    if help_dir.is_dir():
        for stale in help_dir.glob("*.txt"):
            stale.unlink()
    help_dir.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(texts.items()):
        (help_dir / f"{name}.txt").write_text(text)
    index = {
        "encyclopedia_version": version,
        "jar_sha256": jar_sha256,
        "java": java_version,
        "host_cores_placeholder": HOST_CORES,
        "programs": [
            {
                "name": p.name,
                "argv": list(p.argv),
                "listed": p.listed,
                "alias_of": p.alias_of,
                "note": p.note,
            }
            for p in probes
        ],
    }
    (version_dir / "programs.json").write_text(json.dumps(index, indent=1) + "\n")
    return version_dir
