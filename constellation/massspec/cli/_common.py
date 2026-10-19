"""Argument groups and small helpers shared by the ``constellation massspec`` verbs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# ── shared arg helpers ──────────────────────────────────────────────────


def _add_output_dir_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "run directory: holds the jar's output file, the auto-ingest "
            "parquet bundles (when applicable), logs/{stdout,stderr}.log, "
            "manifest.json, and the _SUCCESS sentinel"
        ),
    )


#: The three EncyclopeDIA search tolerances, as
#: ``(flag-stem, dest-stem, jar-flag, help-tail)``. Shared by
#: :func:`_add_tolerance_args` and :func:`_resolve_tolerance_args` so the
#: flag surface and the validation can't drift apart.
_TOLERANCES = (
    (
        "precursor-tolerance",
        "precursor_tolerance",
        "-ptol",
        "precursor m/z window",
    ),
    (
        "fragment-tolerance",
        "fragment_tolerance",
        "-ftol",
        "window applied to the acquired spectrum's fragment peaks",
    ),
    (
        "library-fragment-tolerance",
        "library_fragment_tolerance",
        "-lftol",
        (
            "window applied to the LIBRARY's fragment peaks. Independent of "
            "--fragment-tolerance on purpose: a predicted library's m/z are "
            "exact theoretical values and want a tight ppm window even when "
            "the acquired data is ion-trap. Widen this only when searching a "
            "measured chromatogram .elib whose peaks carry the instrument's "
            "own mass error"
        ),
    ),
)


def _add_tolerance_args(p: argparse.ArgumentParser, *, default: float | None = None) -> None:
    """Add the value + unit flag pair for each EncyclopeDIA tolerance.

    ``default`` is the shared default for the *value* flags — ``None``
    (``massspec search``) omits the flag entirely so the jar's built-in
    10 ppm applies; ``transcriptome-to-proteome`` passes ``10.0`` to
    preserve its pipeline's explicit defaults.
    """
    default_note = (
        f"default {default}, in the unit given by the paired --*-unit flag"
        if default is not None
        else "default: jar's built-in 10 ppm"
    )
    for flag, dest, jar_flag, tail in _TOLERANCES:
        p.add_argument(
            f"--{flag}",
            type=float,
            default=default,
            help=f"EncyclopeDIA {jar_flag} value — {tail} ({default_note}).",
        )
        p.add_argument(
            f"--{flag}-unit",
            choices=["ppm", "Da"],
            default="ppm",
            help=(
                f"unit for --{flag} (default: %(default)s). 'Da' is sent to "
                f"the jar as its 'AMU' token — use it for ion-trap data, "
                f"where the window is absolute rather than mass-proportional."
            ),
        )


def _resolve_tolerance_args(args: argparse.Namespace) -> dict[str, object]:
    """Validate the tolerance flags and project them onto runner kwargs.

    Guards the one real footgun: because the unit defaults to ``ppm``, a
    bare ``--fragment-tolerance-unit Da`` with no matching value flag is
    a silent no-op. That combination is always a mistake, so error on it.
    """
    resolved: dict[str, object] = {}
    for flag, dest, _jar_flag, _tail in _TOLERANCES:
        value = getattr(args, dest, None)
        unit = getattr(args, f"{dest}_unit", "ppm")
        if value is None and unit != "ppm":
            raise ValueError(
                f"--{flag}-unit {unit} was given without --{flag}; the unit "
                f"alone has no effect. Pass a value, or drop the unit flag."
            )
        resolved[dest] = value
        resolved[f"{dest}_unit"] = unit
    return resolved


def _format_tolerance_summary(resolved: dict[str, object]) -> str:
    """One-line human summary of the resolved tolerances, for stderr."""
    parts = []
    for flag, dest, _jar_flag, _tail in _TOLERANCES:
        value = resolved[dest]
        label = flag.removesuffix("-tolerance")
        if value is None:
            parts.append(f"{label} jar-default")
        else:
            parts.append(f"{label} {value} {resolved[f'{dest}_unit']}")
    return "search tolerances: " + ", ".join(parts)


_JVM_HEAP_RE = re.compile(r"^\d+[kKmMgGtT]$")


def _jvm_heap_value(raw: str) -> str:
    """argparse type validator for ``--jvm-heap`` / ``--jvm-heap-min``.

    The JVM wants ``<integer><unit>`` (e.g. ``24g``, ``2048m``). A bare
    integer like ``24`` parses as 24 BYTES and crashes with the
    cryptic ``Too small maximum heap``. Reject up front with a clear
    message.
    """
    if not _JVM_HEAP_RE.match(raw):
        raise argparse.ArgumentTypeError(
            f"invalid JVM heap value {raw!r}: must be <integer><unit>, "
            f"e.g. '24g' (24 gigabytes), '2048m' (2048 megabytes). "
            f"A bare number is interpreted as bytes by the JVM and will "
            f"crash with 'Too small maximum heap'."
        )
    return raw


def _add_jvm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--jvm-heap",
        dest="jvm_heap_max",
        type=_jvm_heap_value,
        default="12g",
        help="max JVM heap (-Xmx); default 12g. Suffix required (g/m/k).",
    )
    p.add_argument(
        "--jvm-heap-min",
        type=_jvm_heap_value,
        default=None,
        help="initial JVM heap (-Xms); default unset. Suffix required (g/m/k).",
    )
    p.add_argument(
        "--jvm-tmpdir",
        type=Path,
        default=None,
        help="-Djava.io.tmpdir override (default: system tmp)",
    )


#: Set in a detached run's environment, so that it runs in the foreground
#: whatever its command line says (an abbreviation of --detach would
#: otherwise detach it again, without end).
_DETACHED_ENV = "CONSTELLATION_DETACHED"


def _add_detach_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--detach",
        action="store_true",
        help=(
            "start in the background, detached from this terminal, and return at "
            "once. Output goes to <output-dir>/runs/<time>-<command>.log; "
            "`constellation massspec summarize <output-dir> --follow` follows it, "
            "and `kill -TERM -<pid>` stops it in order."
        ),
    )


def _detach_if_asked(args: argparse.Namespace, output_dir: Path, label: str) -> int | None:
    """With ``--detach``: start this same command again, detached, and
    return the exit status for this one. ``None`` to carry on here."""
    import os

    if not getattr(args, "detach", False) or os.environ.get(_DETACHED_ENV):
        return None
    from constellation.core.detach import launch

    raw = getattr(args, "_raw_argv", None)
    if raw is None:
        print("error: --detach works from the `constellation` command line", file=sys.stderr)
        return 2
    experiment = Path(output_dir).resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    argv = [a for a in raw if a != "--detach"]
    run = launch(
        [sys.executable, "-m", "constellation.cli", *argv],
        run_dir=experiment / "runs",
        label=label,
        env={**os.environ, _DETACHED_ENV: "1"},
    )
    print(f"detached: pid {run.pid}, log {run.log}")
    print(f"follow with: constellation massspec summarize {experiment} --follow")
    print(f"stop with:   kill -TERM -{run.pid}")
    return 0


def _note_stopped(what: str) -> None:
    """The last line a stopped run leaves in its log (Ctrl-C, or a
    SIGTERM / SIGHUP that the JVM runner turned into an orderly stop)."""
    print(
        f"stopped: {what} recorded as cancelled; the same command with --resume runs "
        f"only what did not finish",
        file=sys.stderr,
    )


def _add_encyclopedia_passthrough_arg(p: argparse.ArgumentParser, *, bare_flags: bool = True) -> None:
    """``--encyclopedia-arg``. ``bare_flags=False`` for the commands that
    read it as a flag mapping (``search``, ``report``, ``quantify``),
    which refuse an item without ``=``."""
    tail = (
        "Items without '=' pass through as bare flags."
        if bare_flags
        else "Every item is -flag=value; an empty value is -flag= (EncyclopeDIA has no bare flags)."
    )
    p.add_argument(
        "--encyclopedia-arg",
        action="append",
        default=[],
        metavar="FLAG=VALUE",
        help=(
            "repeatable escape hatch for unwrapped EncyclopeDIA flags "
            f"(e.g. --encyclopedia-arg '-percolatorVersion=v3-05'). {tail}"
        ),
    )


def _camel_to_kebab(name: str) -> str:
    """``ProteinNTermAcetyl`` → ``protein-n-term-acetyl``; ``TMT`` → ``tmt``.

    Insert a hyphen (a) before an upper-then-lowercase pair that starts
    a CamelCase word, and (b) after a lowercase letter or digit when
    followed by uppercase. All-caps acronyms (``TMT``) stay glued
    because rule (a) never matches and rule (b) requires a preceding
    lowercase.
    """
    s = re.sub(r"(?<!^)(?=[A-Z][a-z])", "-", name)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", s)
    return s.lower()


# ── handlers (PR 0 stubs) ───────────────────────────────────────────────


def _not_yet_wired(subcommand: str) -> int:
    print(
        f"error: `constellation massspec {subcommand}` is registered for "
        f"dashboard introspection but the runner wiring is not yet "
        f"implemented. See docs/plans/encyclopedia-6.9.23-survey.md for "
        f"the rollout order.",
        file=sys.stderr,
    )
    return 2


def _report_ingest_caveats(ingest_result, ingest_info: dict[str, object]) -> None:
    """Show, and record in the manifest, where the Parquet copy of a
    ``.dlib`` / ``.elib`` is not the whole file (dropped fragment peaks,
    unmatched quantity rows, what ``precursor_quant.intensity`` is).
    The file itself is complete; this is about the copy."""
    caveats = list(getattr(ingest_result, "caveats", ()))
    if not caveats:
        return
    ingest_info["caveats"] = caveats
    for caveat in caveats:
        print(f"note: ingest: {caveat}", file=sys.stderr)
