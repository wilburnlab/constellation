"""TypedDict definitions for the dashboard's CLI-schema wire format.

Shared between the introspect walker, the FastAPI endpoint, and the TS
form generator. The values below are the *only* valid `type` strings
emitted by the walker — the TS side has a 1:1 union it switches on.

Wire shape (one example, abbreviated):

```json
{
  "prog": "constellation",
  "arguments": [],
  "subcommands": [
    {
      "name": "transcriptome",
      "path": ["transcriptome"],
      "help": "Sequencing transcriptomics pipeline",
      "arguments": [],
      "subcommands": [
        {
          "name": "align",
          "path": ["transcriptome", "align"],
          "help": "Genome-guided alignment",
          "arguments": [
            {"dest": "demux_dir", "option_strings": ["--demux-dir"],
             "type": "path", "required": true, ...}
          ],
          "subcommands": []
        }
      ]
    }
  ]
}
```
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

ArgumentType = Literal["str", "int", "float", "flag", "enum", "path", "multi"]
"""All valid wire types for ArgumentSchema.type.

- ``str`` — free-text input
- ``int`` / ``float`` — numeric input
- ``flag`` — boolean toggle (no value); store_true / store_false /
  BooleanOptionalAction collapse here
- ``enum`` — argparse ``choices=`` set → dropdown
- ``path`` — heuristically detected by ``dest`` suffix
  (``_dir`` / ``_file`` / ``_path`` …); rendered by the frontend as a
  text box plus a "Browse…" button that opens the FilePicker over
  ``GET /api/fs/list``. The ``path_kind`` sub-field says whether the
  picker opens in directory or file mode.
- ``multi`` — ``nargs in ('+', '*', N>1)`` → repeating list input. When
  the dest is also path-ish, ``path_kind`` is set so the frontend can
  offer a Browse-to-append affordance.
"""

PathKind = Literal["dir", "file", "either"]
"""Sub-classification for path-ish args (``type == 'path'`` or a
path-ish ``multi``). Derived from the ``dest`` heuristic; overridable
per-argument via ``curated.json`` ``arg_hints``. ``either`` — the arg
accepts a directory or a file (e.g. ``--reads`` takes a run directory
or individual BAM/SAM files); the picker lets the user select both."""

WidgetHint = Literal["reference"]
"""Optional richer-widget hint decoupled from ``type``.

- ``reference`` — the arg is a reference *cache handle* (dest
  ``reference``), rendered as an editable dropdown fed by
  ``GET /api/references`` rather than a plain text box.
"""


class ArgumentSchema(TypedDict, total=False):
    """Wire shape for a single argparse argument."""

    dest: str
    option_strings: list[str]
    metavar: str | None
    help: str | None
    type: ArgumentType
    path_kind: PathKind | None
    widget: WidgetHint | None
    glob: str | None
    default: Any
    choices: list[Any] | None
    required: bool
    nargs: str | int | None
    is_positional: bool


class CommandSchema(TypedDict, total=False):
    """Wire shape for a (sub)command. Recursive via ``subcommands``."""

    name: str
    path: list[str]
    help: str | None
    arguments: list[ArgumentSchema]
    subcommands: list["CommandSchema"]


class CuratedEntry(TypedDict, total=False):
    """One row from ``curated.json`` ``curated`` array (the Common overlay)."""

    path: list[str]
    label: str
    group: str
    hint: str


class ArgPathHint(TypedDict, total=False):
    """One row from ``curated.json`` ``arg_hints`` array.

    Refines the auto-walked classification of a single argument the pure
    ``dest`` heuristic can't get right (e.g. ``--samples`` is a TSV file
    even though its dest lacks a path suffix). Applied as a post-pass in
    :func:`build_cli_schema`. ``path`` + ``dest`` locate the argument.
    """

    path: list[str]
    dest: str
    is_path: bool
    path_kind: PathKind
    widget: WidgetHint
    glob: str


class CliSchema(TypedDict):
    """Top-level response shape for ``GET /api/cli/schema``."""

    prog: str
    help: str | None
    arguments: list[ArgumentSchema]
    subcommands: list[CommandSchema]
    curated: list[CuratedEntry]
