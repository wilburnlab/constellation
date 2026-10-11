"""Whether the Percolator a search is about to use can start on this machine.

EncyclopeDIA does not link Percolator. It carries several builds inside
the jar (``bin/percolator-<version>.<platform>``), unpacks the selected
one into ``java.io.tmpdir`` and runs it as a separate process, twice per
search, once all the scoring is done. ``-percolatorVersion`` selects the
build; it also accepts the path of any Percolator executable, which the
help does not say (read from the jar's ``PercolatorVersion`` class and
confirmed by running a search that way).

On Linux the default build, 3.01, is a statically linked executable from
2017 whose C library reads the clock by calling into the legacy
*vsyscall page*. A kernel booted with ``vsyscall=none`` does not map
that page, so the process is killed with SIGSEGV the first time it asks
what time it is, which is before it prints anything. All the jar reports
is ``Percolator exited with non-zero status: 139``, and it reports it
after the whole run has been scored. The WSL2 kernel is booted that way,
as are some hardened distributions; RHEL-family kernels are not.

:func:`percolator_blocker` answers the question before the JVM starts,
so the failure costs a second instead of the search. Diagnosed
2026-10-11 on WSL2 kernel 6.6.87 from the kernel log line
``vsyscall attempted with vsyscall=none ip:ffffffffff600400``; details
in ``docs/plans/encyclopedia-6.9.23-survey.md``.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from constellation.massspec.search.encyclopedia.cli_spec import flag_default

#: Bundled Linux builds that call the legacy vsyscall page. Established
#: by searching each ``bin/percolator-*.lin`` of the 6.9.23 jar for the
#: call sequence (``mov $0xffffffffff600X00,%rax; call *%rax``): 3.01
#: has one, for ``time()``; 2.10 and 3.05 are dynamically linked and
#: have none.
LINUX_BUILDS_NEEDING_VSYSCALL: frozenset[str] = frozenset({"v3-01"})

_PROC_MAPS = Path("/proc/self/maps")


def default_percolator_version() -> str:
    """The build the default search uses when none is named (from the
    captured help, so it follows the pinned jar)."""
    return flag_default("encyclopedia", "-percolatorVersion") or "v3-01"


def _canonical(version: str) -> str:
    """``v3p01`` is the jar's other spelling of ``v3-01``."""
    v = version.strip().lower()
    if len(v) == 5 and v[0] == "v" and v[2] == "p":
        v = f"{v[:2]}-{v[3:]}"
    return v


def _is_linux_x86_64() -> bool:
    return sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64")


def legacy_vsyscall_mapped() -> bool | None:
    """Whether this kernel maps the legacy vsyscall page into processes.

    ``True`` under ``vsyscall=emulate`` or ``xonly`` (the page is listed
    in every process's memory map as ``[vsyscall]``), ``False`` under
    ``vsyscall=none``, ``None`` when it cannot be told (no ``/proc``).
    """
    try:
        maps = _PROC_MAPS.read_text()
    except OSError:
        return None
    return any(line.rstrip().endswith("[vsyscall]") for line in maps.splitlines())


def is_wsl() -> bool:
    return "microsoft" in platform.release().lower()


def effective_percolator(
    version: str | None, extra_args: Sequence[str] = ()
) -> str | None:
    """The ``-percolatorVersion`` value a search will run with.

    ``version`` is the typed option; ``extra_args`` are pass-through jar
    arguments, which can also carry ``-percolatorVersion`` (the last one
    wins, as in the jar) or switch Percolator off. Returns ``None`` when
    the run will not use Percolator at all.
    """
    tokens = [str(a) for a in extra_args]
    for flag, value in zip(tokens, tokens[1:]):
        if flag == "-usePercolator" and value.strip().lower() == "false":
            return None
        if flag == "-percolatorVersion":
            version = value
    return version if version is not None else default_percolator_version()


def percolator_blocker(
    version: str | None = None, extra_args: Sequence[str] = ()
) -> str | None:
    """Why the selected Percolator cannot run here, or ``None`` if nothing is known against it.

    Only the one known case is checked (module docstring). A path to an
    executable is taken on trust.
    """
    selected = effective_percolator(version, extra_args)
    if selected is None or _canonical(selected) not in LINUX_BUILDS_NEEDING_VSYSCALL:
        return None
    if not _is_linux_x86_64() or legacy_vsyscall_mapped() is not False:
        return None
    number = _canonical(selected)[1:].replace("-", ".")
    kernel_fix = (
        "add `kernelCommandLine = vsyscall=emulate` under `[wsl2]` in "
        "%UserProfile%\\.wslconfig, then run `wsl --shutdown`"
        if is_wsl()
        else "boot the kernel with `vsyscall=xonly`"
    )
    return (
        f"EncyclopeDIA's bundled Percolator {number} cannot run on this kernel.\n"
        f"  It is a static 2017 build that reads the clock through the legacy "
        f"vsyscall page, and this kernel\n"
        f"  was booted with vsyscall=none, so it would be killed (the jar reports "
        f"\"Percolator exited with\n"
        f"  non-zero status: 139\") once the whole search had been scored. Any one of:\n"
        f"    - {kernel_fix}\n"
        f"      (keeps Percolator {number}, so results stay comparable with other machines);\n"
        f"    - --percolator-version v3-05: the jar's newer bundled build, which needs "
        f"libgomp.so.1 on the\n"
        f"      library path (the conda env has one) and scores slightly differently;\n"
        f"    - --percolator-version /path/to/percolator: any Percolator executable "
        f"that runs here."
    )


__all__ = [
    "LINUX_BUILDS_NEEDING_VSYSCALL",
    "default_percolator_version",
    "effective_percolator",
    "is_wsl",
    "legacy_vsyscall_mapped",
    "percolator_blocker",
]
