"""What a finished EncyclopeDIA run left behind, and whether it is all there.

The list of files each program writes was captured from a real run of
the pinned jar (``scripts/survey-encyclopedia-outputs.py`` ->
``data/_raw/encyclopedia/<version>/outputs/<program>.json``), with the
input's file name replaced by ``<input>``. This module reads that
inventory back and holds a search's directory to it, so "the search
finished" means "the result library and the files that belong with it
exist", not "Java exited 0".

**Where the jar writes, and what it takes for a finished search.** By
default everything goes beside the input, named for it:
``run.dia`` -> ``run.dia.elib``, ``run.dia.encyclopedia.txt``,
``run.dia.features.txt`` and so on. A result already there is the jar's
own completion marker, by design: it logs "Previously found N
peptides" and returns that result without searching, whether or not
the settings match. It is the jar's ``_SUCCESS`` flag.
``-o X.encyclopedia.txt`` moves both the outputs and that check to
``X.*``.

So a search that is meant to run needs a directory with no result in
it. Constellation gives each search its own, holding a link to the
spectra file and nothing else, and passes no ``-o``: the jar writes
into that directory under its own names, which is also the layout
``-libexport`` expects (a spectra file with ``<file>.elib`` beside it).
Measured on 6.9.23 with a symlinked ``.dia``: the jar does not follow
the link back, so nothing is written beside the link's target and a
result sitting there is not seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from constellation.thirdparty.encyclopedia import MINIMUM_VERSION

INPUT = "<input>"


class MissingOutputError(RuntimeError):
    """A run finished without writing its result library."""


@lru_cache(maxsize=None)
def output_inventory(program: str) -> dict[str, Any]:
    """The captured inventory for ``program`` of the pinned jar."""
    resource = resources.files("constellation.data").joinpath(
        "_raw", "encyclopedia", MINIMUM_VERSION, "outputs", f"{program}.json"
    )
    if not resource.is_file():
        raise KeyError(
            f"no output inventory was captured for EncyclopeDIA program {program!r} "
            f"(scripts/survey-encyclopedia-outputs.py records one from a finished run)"
        )
    return json.loads(resource.read_text())


def expected_names(program: str, input_name: str) -> dict[str, str]:
    """``inventory name -> file name`` for a search of the file called
    ``input_name`` (``"run.dia"``)."""
    return {
        f["name"]: f["name"].replace(INPUT, input_name, 1)
        for f in output_inventory(program)["files"]
    }


@dataclass(frozen=True)
class RunOutputs:
    """The files of one finished search."""

    directory: Path
    input_name: str
    program: str
    #: The SQLite result (``<input>.elib``).
    result_library: Path
    #: Inventory name (``<input>.features.txt``) -> path, for files present.
    files: dict[str, Path] = field(default_factory=dict)
    #: Inventory names of expected files that are not there.
    missing: tuple[str, ...] = ()

    def path(self, name: str) -> Path:
        """The file the inventory calls ``name`` (``"<input>.features.txt"``)."""
        return self.files[name]

    def relative(self, base: Path) -> dict[str, str]:
        """``inventory name -> path relative to base``, for a manifest."""
        return {name: str(path.relative_to(base)) for name, path in sorted(self.files.items())}


def locate_outputs(
    directory: Path, input_name: str, program: str = "encyclopedia"
) -> RunOutputs:
    """Find the outputs of a finished search of ``directory/input_name``.

    Raises :class:`MissingOutputError` when the result library is not
    there. Other expected files that are absent are listed in
    ``missing`` rather than raised: the plots and fit tables are not
    needed to read a result, and whether their absence matters is the
    caller's call.
    """
    directory = Path(directory)
    inventory = output_inventory(program)
    files: dict[str, Path] = {}
    missing: list[str] = []
    result: Path | None = None
    for entry in inventory["files"]:
        path = directory / entry["name"].replace(INPUT, input_name, 1)
        if path.is_file():
            files[entry["name"]] = path
            if entry.get("kind") == "sqlite" and result is None:
                result = path
        else:
            missing.append(entry["name"])
    if result is None:
        expected = [e["name"] for e in inventory["files"] if e.get("kind") == "sqlite"]
        raise MissingOutputError(
            f"{program} finished but wrote no result library: expected "
            f"{', '.join(n.replace(INPUT, input_name, 1) for n in expected)} in {directory}"
        )
    return RunOutputs(
        directory=directory,
        input_name=input_name,
        program=program,
        result_library=result,
        files=files,
        missing=tuple(missing),
    )


__all__ = [
    "INPUT",
    "MissingOutputError",
    "RunOutputs",
    "expected_names",
    "locate_outputs",
    "output_inventory",
]
