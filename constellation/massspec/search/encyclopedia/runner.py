"""Run any program of the EncyclopeDIA jar, checked against what the jar documents.

The four legacy wrappers (``library_search``, ``predict_library``,
``process_dia``, ``library_export``) each spell out one program's flags
by hand. That is how a wrapper came to emit ``-scribetwodda`` for a jar
that had no such program. This runner takes a program *name* and a
mapping of flags, and before Java starts it checks both against the
snapshot of the pinned jar's own help
(:mod:`.cli_spec`, ``data/encyclopedia_cli.json``):

- the program must be one the jar has;
- every flag must be one the jar documents for that program (by name,
  by family such as ``-ptm...``, or as a help-truncated prefix), unless
  the caller names it in ``allow_undocumented``, which is how a
  parameter the jar accepts but does not print is passed on purpose;
- every flag must have a value. EncyclopeDIA takes ``-flag value``
  throughout; a bare flag is a program selector, and the program is
  chosen by name here.

So a typo, a flag from another program, or a flag that a version bump
removed fails in a millisecond with the near-misses listed, instead of
minutes later inside the JVM, or never (the jar ignores some unknown
flags silently).

The legacy wrappers stay as they are: the transcriptome-to-proteome
pipeline imports them and its tests patch them by name.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from pathlib import Path

from constellation.massspec.search.encyclopedia._common import (
    MINIMUM_ENCYCLOPEDIA_VERSION,
    is_supported_version,
)
from constellation.massspec.search.encyclopedia.cli_spec import (
    is_known_flag,
    known_flags,
    program_argv,
    program_names,
)
from constellation.thirdparty.jvm import JvmResult, run_jar
from constellation.thirdparty.registry import ToolHandle, find

TOOL = "encyclopedia"

FlagValue = str | int | float | bool | Path


class UnknownProgramError(ValueError):
    """Not a program of the pinned jar."""


class UnknownFlagError(ValueError):
    """A flag the pinned jar does not document for the program."""


class UnsupportedVersionError(RuntimeError):
    """The installed jar is older than the version Constellation is pinned to."""


def format_value(value: FlagValue) -> str:
    """A flag value as the jar reads it: booleans as ``true`` / ``false``,
    numbers in their shortest exact form, paths and text as they are.
    An empty string is a legitimate value (``-fixed ""`` means no fixed
    modification)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def compose_argv(
    program: str,
    flags: Mapping[str, FlagValue] | Sequence[tuple[str, FlagValue]],
    *,
    allow_undocumented: Sequence[str] = (),
) -> list[str]:
    """The jar arguments for ``program`` with ``flags``, validated.

    ``flags`` keeps its order. Pure: no jar and no JVM are needed, so
    this is what tests and ``--dry-run`` call.
    """
    if program not in program_names():
        near = difflib.get_close_matches(program, program_names(), n=3)
        hint = f" (did you mean {', '.join(near)}?)" if near else ""
        raise UnknownProgramError(
            f"EncyclopeDIA {MINIMUM_ENCYCLOPEDIA_VERSION} has no program {program!r}{hint}; "
            f"it has: {', '.join(program_names())}"
        )
    items = list(flags.items()) if isinstance(flags, Mapping) else list(flags)
    allowed = set(allow_undocumented)
    argv = list(program_argv(program))
    unknown: list[str] = []
    for name, value in items:
        if not name.startswith("-"):
            raise UnknownFlagError(f"{name!r} is not a flag: EncyclopeDIA flags start with '-'")
        if value is None:
            raise UnknownFlagError(
                f"{name} has no value. EncyclopeDIA takes `-flag value`; a bare flag selects "
                f"a program, and the program here is {program!r}"
            )
        if name not in allowed and not is_known_flag(program, name):
            unknown.append(name)
            continue
        argv += [name, format_value(value)]
    if unknown:
        documented = sorted(known_flags(program))
        lines = []
        for name in unknown:
            near = difflib.get_close_matches(name, documented, n=3)
            lines.append(f"  {name}" + (f"  (did you mean {', '.join(near)}?)" if near else ""))
        raise UnknownFlagError(
            f"EncyclopeDIA {MINIMUM_ENCYCLOPEDIA_VERSION} does not document these flags for "
            f"{program!r}:\n" + "\n".join(lines) + "\n"
            "`constellation massspec encyclopedia flags " + program + "` lists the ones it does. "
            "A parameter the jar accepts without documenting has to be allowed by name."
        )
    return argv


def resolve_jar() -> ToolHandle:
    """The installed jar, refused if it is older than the pin."""
    handle = find(TOOL)
    if not is_supported_version(handle.version):
        raise UnsupportedVersionError(
            f"EncyclopeDIA {handle.version} resolved at {handle.path}, but Constellation "
            f"requires >= {MINIMUM_ENCYCLOPEDIA_VERSION}. Install a current build: "
            f"bash scripts/install-encyclopedia.sh --installer <install4j .sh> "
            f"(or --from-dir <extracted install dir>)"
        )
    return handle


def run_program(
    program: str,
    flags: Mapping[str, FlagValue] | Sequence[tuple[str, FlagValue]],
    *,
    log_dir: Path,
    cwd: Path | None = None,
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    extra_jvm_args: Sequence[str] = (),
    allow_undocumented: Sequence[str] = (),
    stream_to_stderr: bool = True,
    check: bool = True,
) -> JvmResult:
    """Run ``program`` of the installed jar.

    The flags are validated first (:func:`compose_argv`), then the jar's
    version, then the JVM starts through :func:`thirdparty.jvm.run_jar`
    (own session, private temp directory, cleanup, cancellation). Logs
    go to ``log_dir``. Raises
    :class:`~constellation.thirdparty.jvm.JvmRunError` on a non-zero
    exit unless ``check`` is false.
    """
    argv = compose_argv(program, flags, allow_undocumented=allow_undocumented)
    resolve_jar()
    return run_jar(
        TOOL,
        args=argv,
        jvm_heap_max=jvm_heap_max,
        jvm_heap_min=jvm_heap_min,
        jvm_tmpdir=jvm_tmpdir,
        extra_jvm_args=extra_jvm_args,
        log_dir=log_dir,
        cwd=cwd,
        stream_to_stderr=stream_to_stderr,
        check=check,
    )


__all__ = [
    "TOOL",
    "FlagValue",
    "UnknownFlagError",
    "UnknownProgramError",
    "UnsupportedVersionError",
    "compose_argv",
    "format_value",
    "resolve_jar",
    "run_program",
]
