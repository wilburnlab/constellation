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
    ArgPathHint,
    ArgumentSchema,
    ArgumentType,
    CliSchema,
    CommandSchema,
    CuratedEntry,
)

# Directory-vs-file classification for path-ish argparse dests. argparse
# doesn't carry semantic types beyond callables, so we lean on the dest
# naming convention used throughout the CLI. Directory dests win over
# file dests when both would match (checked first below).
_DIR_DEST_SUFFIXES = ("_dir",)
_DIR_DEST_EXACT = {
    "output_dir",
    "demux_dir",
    "align_dir",
    "cluster_dir",
    "reference_dir",
    "session",
    "dir",
}
_FILE_DEST_SUFFIXES = (
    "_file",
    "_path",
    "_bed",
    "_tsv",
    "_bam",
    "_sam",
    "_fasta",
    "_fa",
    "_gff",
    "_gff3",
    "_vcf",
)
_FILE_DEST_EXACT = {"samples", "reads", "file", "path"}
# Dests that look path-ish (or would collide with a suffix) but are NOT
# filesystem paths — cache handles, free-text labels. Guarded out of the
# picker entirely.
_NOT_PATH_DEST = {"reference", "library_design"}


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

    # Stamp the directory-vs-file sub-kind on path-ish args (both plain
    # `path` and repeatable `multi` — a `nargs="+"` path arg stays multi
    # but the frontend offers a Browse-to-append affordance).
    kind = _path_kind_or_none(action)
    if kind is not None and arg_type in ("path", "multi"):
        schema["path_kind"] = kind

    # Reference cache handles (dest `reference`) get an editable dropdown
    # fed by GET /api/references — universal for any command that takes
    # `--reference`, no curated entry required.
    if action.dest == "reference":
        schema["widget"] = "reference"

    return schema


def _path_kind_or_none(action: argparse.Action) -> str | None:
    """Return ``"dir"``, ``"file"``, or ``None`` for an argparse action.

    Pure ``dest``-name heuristic (argparse carries no semantic type).
    Directory dests are checked before file dests so a ``*_dir`` wins.
    Dests in :data:`_NOT_PATH_DEST` (cache handles, labels) return
    ``None`` even if they'd otherwise match. Misses that are genuine file
    inputs get refined via the ``arg_hints`` curated overlay.
    """
    dest = action.dest or ""
    if dest in _NOT_PATH_DEST:
        return None
    if dest in _DIR_DEST_EXACT or any(
        dest.endswith(suffix) for suffix in _DIR_DEST_SUFFIXES
    ):
        return "dir"
    if dest in _FILE_DEST_EXACT or any(
        dest.endswith(suffix) for suffix in _FILE_DEST_SUFFIXES
    ):
        return "file"
    return None


def _looks_like_path(action: argparse.Action) -> bool:
    """True when ``dest`` classifies as a directory or file path."""
    return _path_kind_or_none(action) is not None


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


def load_arg_hints() -> list[ArgPathHint]:
    """Read the packaged ``curated.json`` ``arg_hints`` overlay.

    Each hint refines one argument's classification (see
    :class:`ArgPathHint`). Returns an empty list when the file is missing
    or malformed — the auto-walked classification stands unchanged.
    """
    try:
        raw = files("constellation.viz.introspect").joinpath("curated.json").read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    hints = data.get("arg_hints")
    if not isinstance(hints, list):
        return []
    return [
        h
        for h in hints
        if isinstance(h, dict) and "path" in h and "dest" in h
    ]


def _find_command(root: CommandSchema, path: list[str]) -> CommandSchema | None:
    """Descend the walked tree to the command node at ``path``."""
    node = root
    for name in path:
        nxt = None
        for child in node.get("subcommands", ()):
            if child.get("name") == name:
                nxt = child
                break
        if nxt is None:
            return None
        node = nxt
    return node


def _apply_arg_hints(root: CommandSchema, hints: list[ArgPathHint]) -> None:
    """Patch the walked tree in place from the ``arg_hints`` overlay.

    Locates each hint's ``(path, dest)`` argument and applies:
    ``is_path`` (force ``type`` to/from ``path``), ``path_kind``,
    ``widget``, and ``glob``. Unknown paths / dests are skipped silently
    so a stale hint never breaks the schema.
    """
    for hint in hints:
        path = hint.get("path")
        dest = hint.get("dest")
        if not isinstance(path, list) or not isinstance(dest, str):
            continue
        node = _find_command(root, path)
        if node is None:
            continue
        for arg in node.get("arguments", ()):
            if arg.get("dest") != dest:
                continue
            if "is_path" in hint:
                if hint["is_path"] and arg.get("type") not in ("path", "multi"):
                    arg["type"] = "path"
                elif not hint["is_path"] and arg.get("type") == "path":
                    arg["type"] = "str"
                    arg.pop("path_kind", None)
            if "path_kind" in hint:
                arg["path_kind"] = hint["path_kind"]
                # A path_kind hint implies the arg IS a path unless it's a
                # repeatable multi (which keeps its type).
                if arg.get("type") not in ("path", "multi"):
                    arg["type"] = "path"
            if "widget" in hint:
                arg["widget"] = hint["widget"]
            if "glob" in hint:
                arg["glob"] = hint["glob"]
            break


def build_cli_schema(parser: argparse.ArgumentParser) -> CliSchema:
    """Walk ``parser`` and pair the result with the curated overlay."""
    root = walk_parser(parser)
    _apply_arg_hints(root, load_arg_hints())
    return {
        "prog": parser.prog or "constellation",
        "help": parser.description,
        "arguments": root.get("arguments", []),
        "subcommands": root.get("subcommands", []),
        "curated": load_curated(),
    }
