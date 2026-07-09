"""argparse → JSON schema walker.

Exercises the real production CLI parser (no test fixture) so any
addition / removal of subcommands or argument types surfaces here
immediately. The walker has no fastapi / pyarrow dependencies, so this
file doesn't gate on ``[viz]`` extras.
"""

from __future__ import annotations

import json
from pathlib import Path

from constellation.cli.__main__ import _build_parser
from constellation.viz.introspect import (
    build_cli_schema,
    load_curated,
    walk_parser,
)
from constellation.viz.introspect.walk import load_arg_hints


def _commands_by_path(schema, path):
    """Drill into ``schema`` along ``path`` and return the CommandSchema."""
    node = schema
    for name in path:
        children = node.get("subcommands", [])
        match = next((c for c in children if c.get("name") == name), None)
        assert match is not None, f"missing subcommand path {path}; failed at {name}"
        node = match
    return node


def _arg_by_dest(command, dest):
    return next((a for a in command["arguments"] if a["dest"] == dest), None)


def test_walk_emits_top_level_subcommands():
    schema = build_cli_schema(_build_parser())
    assert schema["prog"] == "constellation"
    names = {c["name"] for c in schema["subcommands"]}
    for required in (
        "doctor",
        "transcriptome",
        "reference",
        "viz",
        "dashboard",
    ):
        assert required in names, f"missing top-level subcommand: {required}"


def test_walk_descends_into_nested_subparsers():
    schema = build_cli_schema(_build_parser())
    trans = _commands_by_path(schema, ["transcriptome"])
    trans_subs = {c["name"] for c in trans["subcommands"]}
    assert trans_subs >= {"demultiplex", "align", "cluster"}
    ref = _commands_by_path(schema, ["reference"])
    ref_subs = {c["name"] for c in ref["subcommands"]}
    assert ref_subs >= {"import", "fetch", "summary", "validate"}
    viz = _commands_by_path(schema, ["viz"])
    viz_subs = {c["name"] for c in viz["subcommands"]}
    assert viz_subs >= {"genome", "install-frontend"}


def test_argument_types_mapped_correctly():
    """Spot-check the type-mapping rules on transcriptome align — it
    has the widest type surface in the CLI: paths, ints, floats, flags,
    enums, and store_true / BooleanOptionalAction toggles."""
    schema = build_cli_schema(_build_parser())
    align = _commands_by_path(schema, ["transcriptome", "align"])

    demux_dir = _arg_by_dest(align, "demux_dir")
    assert demux_dir["type"] == "path"
    assert demux_dir["required"] is True

    threads = _arg_by_dest(align, "threads")
    assert threads["type"] == "int"
    assert threads["default"] == 1

    min_aligned = _arg_by_dest(align, "min_aligned_fraction")
    assert min_aligned["type"] == "float"

    matrix_format = _arg_by_dest(align, "matrix_format")
    assert matrix_format["type"] == "enum"
    assert set(matrix_format["choices"]) == {"count", "tpm", "both", "none"}

    resume = _arg_by_dest(align, "resume")
    assert resume["type"] == "flag"

    allow_antisense = _arg_by_dest(align, "allow_antisense")
    assert allow_antisense["type"] == "flag"


def test_positional_arguments_flagged():
    """`constellation reference fetch <spec>` has a positional ``spec``;
    the walker must mark it ``is_positional=True``."""
    schema = build_cli_schema(_build_parser())
    fetch = _commands_by_path(schema, ["reference", "fetch"])
    spec = _arg_by_dest(fetch, "spec")
    assert spec is not None
    assert spec["is_positional"] is True
    assert spec["option_strings"] == []
    # Required-by-virtue-of-being-positional surfaces as required=True
    # in argparse for positionals without nargs.
    assert spec["required"] is True


def test_boolean_optional_action_maps_to_flag():
    """``constellation transcriptome cluster --write-fasta`` uses
    BooleanOptionalAction (line 611 in __main__.py). The walker must
    map this to ``type='flag'`` so the form renders a checkbox, not a
    text input."""
    schema = build_cli_schema(_build_parser())
    cluster = _commands_by_path(schema, ["transcriptome", "cluster"])
    write_fasta = _arg_by_dest(cluster, "write_fasta")
    assert write_fasta is not None
    assert write_fasta["type"] == "flag"


def test_subparser_help_surfaces():
    """The walker pulls per-choice help strings from the parent
    SubParsersAction (not the child parser's description) so the
    sidebar matches what ``--help`` prints."""
    schema = build_cli_schema(_build_parser())
    doctor = _commands_by_path(schema, ["doctor"])
    # 'doctor' subparser was added with help= in __main__.py:134
    assert doctor["help"] is not None and "third-party" in doctor["help"].lower()


def test_curated_overlay_loads_and_paths_resolve():
    """Every curated path must resolve against the real parser tree —
    otherwise the sidebar would show a Common-mode entry that vanishes
    on click."""
    schema = build_cli_schema(_build_parser())
    curated = schema["curated"]
    assert len(curated) > 0
    # Sanity-check shape
    for entry in curated:
        assert "path" in entry
        assert "label" in entry
        # Every path resolves
        _commands_by_path(schema, entry["path"])


def test_curated_json_is_valid_json():
    """Guard against a malformed curated.json silently disabling the
    overlay (load_curated swallows JSONDecodeError to keep the
    dashboard alive)."""
    from importlib.resources import files

    raw = files("constellation.viz.introspect").joinpath("curated.json").read_text()
    data = json.loads(raw)
    assert "curated" in data
    assert isinstance(data["curated"], list)


def test_walk_output_is_json_serializable():
    """The endpoint serializes the schema via FastAPI's JSON encoder.
    Any non-serializable default (callable, custom class) would crash
    the request — catch it here at the unit-test level."""
    schema = build_cli_schema(_build_parser())
    # Will raise TypeError if any value can't be encoded.
    json.dumps(schema)


def test_path_heuristic_recognizes_dest_suffixes():
    """The path-detection heuristic uses dest suffixes (``_dir``,
    ``_file``, ``_path``) plus an exact-match set. Spot-check it picks
    up common patterns across the CLI."""
    schema = build_cli_schema(_build_parser())
    # `--reference REF_DIR` on transcriptome align — `reference` dest
    # is in _PATH_DEST_EXACT? No, but the metavar suggests it's a path.
    # We rely on `_dir` suffix detection.
    demux = _commands_by_path(schema, ["transcriptome", "demultiplex"])
    output_dir = _arg_by_dest(demux, "output_dir")
    assert output_dir["type"] == "path"
    # `--align-dir` on viz genome maps via the `_dir` suffix detection.
    genome = _commands_by_path(schema, ["viz", "genome"])
    align_dir = _arg_by_dest(genome, "align_dir")
    assert align_dir["type"] == "path"


def test_path_kind_directory_vs_file():
    """Path args carry a `path_kind` sub-field so the FilePicker opens
    in the right mode. Directories (``*_dir``) vs files must split."""
    schema = build_cli_schema(_build_parser())
    align = _commands_by_path(schema, ["transcriptome", "align"])
    # Directory dests
    for dest in ("demux_dir", "output_dir"):
        arg = _arg_by_dest(align, dest)
        assert arg["type"] == "path"
        assert arg["path_kind"] == "dir"
    # File dest — junc_bed matches the `_bed` suffix (dir-first, file-next)
    junc = _arg_by_dest(align, "junc_bed")
    assert junc["type"] == "path"
    assert junc["path_kind"] == "file"

    demux = _commands_by_path(schema, ["transcriptome", "demultiplex"])
    samples = _arg_by_dest(demux, "samples")
    assert samples["type"] == "path"
    assert samples["path_kind"] == "file"


def test_repeatable_path_arg_stays_multi_with_path_kind():
    """``--reads`` is ``nargs='+'`` so it stays ``type='multi'`` (the
    textarea), but carries a ``path_kind`` so the frontend can offer a
    Browse-to-append affordance. The curated overlay marks it 'either'
    (a run dir or individual BAM/SAM files)."""
    schema = build_cli_schema(_build_parser())
    demux = _commands_by_path(schema, ["transcriptome", "demultiplex"])
    reads = _arg_by_dest(demux, "reads")
    assert reads["type"] == "multi"
    assert reads["path_kind"] == "either"


def test_reference_handle_is_widget_not_path():
    """``--reference`` is a cache handle, NOT a filesystem path — it must
    stay ``type='str'`` (no path_kind, no picker) and carry
    ``widget='reference'`` so the form renders the references dropdown."""
    schema = build_cli_schema(_build_parser())
    align = _commands_by_path(schema, ["transcriptome", "align"])
    ref = _arg_by_dest(align, "reference")
    assert ref["type"] == "str"
    assert ref.get("path_kind") is None
    assert ref["widget"] == "reference"

    # `--library-design` (a plain label) must stay a free-text str.
    demux = _commands_by_path(schema, ["transcriptome", "demultiplex"])
    lib = _arg_by_dest(demux, "library_design")
    assert lib["type"] == "str"
    assert lib.get("path_kind") is None
    assert lib.get("widget") is None


def test_arg_hints_apply_globs():
    """The curated ``arg_hints`` overlay refines path args with file
    globs the pure dest heuristic can't infer."""
    schema = build_cli_schema(_build_parser())
    demux = _commands_by_path(schema, ["transcriptome", "demultiplex"])
    samples = _arg_by_dest(demux, "samples")
    assert samples["glob"] == "*.tsv,*.txt"
    reads = _arg_by_dest(demux, "reads")
    assert reads["glob"] == "*.bam,*.sam"
    align = _commands_by_path(schema, ["transcriptome", "align"])
    junc = _arg_by_dest(align, "junc_bed")
    assert junc["glob"] == "*.bed"


def test_arg_hints_loader_shape():
    """load_arg_hints returns only well-formed rows (path + dest)."""
    hints = load_arg_hints()
    assert isinstance(hints, list)
    assert len(hints) >= 3
    for h in hints:
        assert isinstance(h.get("path"), list)
        assert isinstance(h.get("dest"), str)


def test_arg_hints_do_not_break_json():
    """The full schema (with path_kind / widget / glob) still serializes
    for the FastAPI JSON encoder."""
    schema = build_cli_schema(_build_parser())
    json.dumps(schema)
    # arg_hints block in curated.json must be a list when present.
    from importlib.resources import files

    raw = files("constellation.viz.introspect").joinpath("curated.json").read_text()
    data = json.loads(raw)
    assert isinstance(data.get("arg_hints", []), list)
