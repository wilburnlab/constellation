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
  (``_dir`` / ``_file`` / ``_path``); rendered as plain text in v1,
  FilePicker in a follow-up PR
- ``multi`` — ``nargs in ('+', '*', N>1)`` → repeating list input
"""


class ArgumentSchema(TypedDict, total=False):
    """Wire shape for a single argparse argument."""

    dest: str
    option_strings: list[str]
    metavar: str | None
    help: str | None
    type: ArgumentType
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
    """One row from ``curated.json``."""

    path: list[str]
    label: str
    group: str
    hint: str


class CliSchema(TypedDict):
    """Top-level response shape for ``GET /api/cli/schema``."""

    prog: str
    help: str | None
    arguments: list[ArgumentSchema]
    subcommands: list[CommandSchema]
    curated: list[CuratedEntry]
