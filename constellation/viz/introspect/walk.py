"""Walk an ``argparse`` parser tree and emit a JSON schema.

Used by the dashboard's ``/api/cli/schema`` endpoint to power the
sidebar's command tree + the CommandForm's auto-generated inputs.

The walker is intentionally tolerant of unknown action shapes: anything
it doesn't recognize maps to ``type='str'`` with the original ``help``
preserved, so the form still renders. The TS side only branches on the
seven types declared in :class:`ArgumentType` — no special-case fallback
needed on the client.

``walk_parser()`` accesses argparse's private ``_actions`` /
``_SubParsersAction.choices`` attributes. This is the only sanctioned
way to introspect an already-built parser; stdlib argparse hasn't
exposed a public reflection API since 3.0. Stable enough for our use
(the attributes have been load-bearing for two decades).
"""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from typing import Any

from constellation.viz.introspect.schema import (
    ArgumentSchema,
    ArgumentType,
    CliSchema,
    CommandSchema,
    CuratedEntry,
)

_PATH_DEST_SUFFIXES = ("_dir", "_file", "_path", "dir", "path", "file")
_PATH_DEST_EXACT = {"output_dir", "demux_dir", "align_dir", "session"}


def walk_parser(
    parser: argparse.ArgumentParser, *, _path: tuple[str, ...] = ()
) -> CommandSchema:
    """Recursively walk ``parser`` and emit a :class:`CommandSchema`.

    Top-level callers pass the parser produced by
    ``constellation.cli.__main__._build_parser()`` with no ``_path``; the
    function descends into ``_SubParsersAction.choices`` and accumulates
    the command path.
    """
    arguments: list[ArgumentSchema] = []
    subcommands: list[CommandSchema] = []

    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if isinstance(action, argparse._SubParsersAction):
            # Sort for deterministic output; the sidebar respects this order.
            for sub_name in sorted(action.choices):
                sub_parser = action.choices[sub_name]
                child_path = (*_path, sub_name)
                child = walk_parser(sub_parser, _path=child_path)
                child["name"] = sub_name
                child["path"] = list(child_path)
                # Help comes from the parent SubParsersAction's choices_actions,
                # not the child parser's description — surface it here.
                child["help"] = _subparser_help(action, sub_name) or sub_parser.description
                subcommands.append(child)
            continue
        arguments.append(_action_to_schema(action))

    out: CommandSchema = {
        "name": _path[-1] if _path else (parser.prog or ""),
        "path": list(_path),
        "help": parser.description,
        "arguments": arguments,
        "subcommands": subcommands,
    }
    return out


def _subparser_help(action: argparse._SubParsersAction, name: str) -> str | None:
    """Pull the per-choice help string from a SubParsersAction.

    ``add_parser(name, help=...)`` stashes the help string on
    ``_ChoicesPseudoAction`` instances in ``_choices_actions``. Recover
    it so the sidebar shows the same one-liners that ``--help`` prints.
    """
    for pseudo in getattr(action, "_choices_actions", ()):
        if getattr(pseudo, "dest", None) == name:
            return getattr(pseudo, "help", None)
    return None


def _action_to_schema(action: argparse.Action) -> ArgumentSchema:
    """Map one argparse ``Action`` to wire shape."""
    arg_type: ArgumentType
    if isinstance(action, argparse.BooleanOptionalAction):
        arg_type = "flag"
    elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        arg_type = "flag"
    elif action.choices is not None:
        arg_type = "enum"
    elif action.nargs in ("+", "*") or (
        isinstance(action.nargs, int) and action.nargs > 1
    ):
        arg_type = "multi"
    elif _looks_like_path(action):
        arg_type = "path"
    elif action.type is int:
        arg_type = "int"
    elif action.type is float:
        arg_type = "float"
    else:
        arg_type = "str"

    schema: ArgumentSchema = {
        "dest": action.dest,
        "option_strings": list(action.option_strings),
        "metavar": _metavar_str(action.metavar),
        "help": action.help,
        "type": arg_type,
        "default": _json_safe(action.default),
        "choices": _json_safe_list(action.choices),
        "required": bool(getattr(action, "required", False)),
        "nargs": _nargs_str(action.nargs),
        "is_positional": not action.option_strings,
    }
    return schema


def _looks_like_path(action: argparse.Action) -> bool:
    """Heuristic — argparse doesn't carry semantic types beyond callables.

    Matches when ``dest`` ends in a path-ish suffix or is a known
    file-input dest (``output_dir``, ``demux_dir``, ``align_dir``,
    ``session``). Misses get rendered as plain text inputs; the v2
    FilePicker will refine this via the curated overlay.
    """
    dest = action.dest or ""
    if dest in _PATH_DEST_EXACT:
        return True
    return any(dest.endswith(suffix) for suffix in _PATH_DEST_SUFFIXES)


def _metavar_str(metavar: Any) -> str | None:
    if metavar is None:
        return None
    if isinstance(metavar, tuple):
        return " ".join(str(m) for m in metavar)
    return str(metavar)


def _nargs_str(nargs: Any) -> str | int | None:
    """Coerce nargs to a JSON-safe primitive.

    argparse sometimes stores sentinel objects (``argparse.PARSER``,
    ``argparse.REMAINDER``); fall back to str() for those.
    """
    if nargs is None:
        return None
    if isinstance(nargs, (int, str)):
        return nargs
    return str(nargs)


def _json_safe(value: Any) -> Any:
    """Coerce a default value to something ``json.dumps`` can handle."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # argparse SUPPRESS, callable defaults, Paths — stringify.
    return str(value)


def _json_safe_list(choices: Any) -> list[Any] | None:
    if choices is None:
        return None
    return [_json_safe(c) for c in choices]


def load_curated() -> list[CuratedEntry]:
    """Read the packaged ``curated.json`` overlay.

    Returns an empty list if the file is missing or malformed — the
    sidebar falls back to the full auto-walked tree in that case.
    """
    try:
        raw = files("constellation.viz.introspect").joinpath("curated.json").read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    entries = data.get("curated")
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict) and "path" in e]


def build_cli_schema(parser: argparse.ArgumentParser) -> CliSchema:
    """Walk ``parser`` and pair the result with the curated overlay."""
    root = walk_parser(parser)
    return {
        "prog": parser.prog or "constellation",
        "help": parser.description,
        "arguments": root.get("arguments", []),
        "subcommands": root.get("subcommands", []),
        "curated": load_curated(),
    }
