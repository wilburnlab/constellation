"""Record what an EncyclopeDIA run writes: file inventory and table definitions.

Companion to ``survey-encyclopedia.py`` (which captures the jar's
``-h``). That one needs only the jar; this one needs a finished run,
because what a program writes, where, and with which tables can only be
read off real output. It writes::

    constellation/data/_raw/encyclopedia/<version>/schema/result_library.sql
    constellation/data/_raw/encyclopedia/<version>/schema/spectra_file.sql
    constellation/data/_raw/encyclopedia/<version>/outputs/<program>.json

``result_library.sql`` and ``spectra_file.sql`` are the ``CREATE``
statements of a result library (``.elib`` / ``.dlib``) and of a spectra
file (``.dia``), so readers can be tested against the pinned version's
layout without the jar or the data. ``outputs/<program>.json`` lists the
files a run of that program leaves beside its input (the input's file
name replaced by ``<input>``), the column headers of the tab-separated ones,
the tables each SQLite file fills, and the keys of the result library's
``metadata`` table, which is where the jar records every effective
parameter, including ones its help does not list.

Nothing run-specific is kept: no row counts, no values.

The directory to capture from is one search's own: a fresh directory
holding the input (or a link to it) and nothing else before the run,
searched without ``-o``, so that everything in it is named the jar's
own way (``<input>.elib``, ``<input>.encyclopedia.txt``, ...).

Example (the 6.9.23 capture, from a peptide-centric search)::

    python3 scripts/survey-encyclopedia-outputs.py \\
        --program encyclopedia \\
        --output-dir <the search's directory> \\
        --spectra-file <any .dia written by the pinned toolchain>

Run from project root.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from constellation.thirdparty import encyclopedia as _adapter  # noqa: F401  (registers the tool)
from constellation.thirdparty.encyclopedia import MINIMUM_VERSION

_SQLITE_SUFFIXES = (".elib", ".dlib", ".dia")
_TEXT_TABLE_SUFFIXES = (".txt",)
# Text files that are logs or fitted-model dumps, not tables with a header row.
_NOT_TABLES = (".log", ".model")

_NOTES = {
    "encyclopedia": (
        "Files a search leaves beside its input when no -o is given: the "
        "jar's own naming. <input> is the input's file name, extension "
        "included (run.dia -> run.dia.elib). With -o X.encyclopedia.txt "
        "the same files are written as X.* in X's directory instead."
    ),
    "scribetwo": (
        "Files a ScribeTwo search leaves when no -o is given: the result "
        "library beside its input (<input>.dlib) and its reports in a "
        "directory named for the input (<input>.scribe2/). <input> is the "
        "input's file name, extension included (run.dia -> run.dia.dlib)."
    ),
}


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def _ddl(path: Path) -> list[str]:
    """``CREATE`` statements, tables then indexes, each whitespace-normalised."""
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL "
            "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name"
        ).fetchall()
    finally:
        con.close()
    return [" ".join(sql.split()) + ";" for _type, _name, sql in rows]


def _metadata(path: Path) -> dict[str, str]:
    con = _connect(path)
    try:
        return {str(k): str(v) for k, v in con.execute("SELECT Key, Value FROM metadata")}
    finally:
        con.close()


def _filled_tables(path: Path) -> dict[str, bool]:
    con = _connect(path)
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {
            n: con.execute(f'SELECT EXISTS(SELECT 1 FROM "{n}")').fetchone()[0] == 1
            for n in sorted(names)
        }
    finally:
        con.close()


def _write_sql(path: Path, title: str, statements: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"-- {title}\n" + "\n".join(statements) + "\n")


def _input_name(output_dir: Path) -> str:
    """The input a directory's outputs are named for: the result library's
    name without ``.elib`` / ``.dlib`` (``run.dia`` for ``run.dia.elib``)."""
    libraries = sorted(
        p for p in output_dir.iterdir() if p.suffix in (".elib", ".dlib") and p.is_file()
    )
    if len(libraries) != 1:
        raise SystemExit(
            f"error: expected one result library in {output_dir}, found "
            f"{[p.name for p in libraries]}"
        )
    return libraries[0].name.removesuffix(libraries[0].suffix)


def _outputs(output_dir: Path, input_name: str) -> list[tuple[str, Path]]:
    """``(inventory name, path)`` for every file the run named for its input:
    beside the input, and inside a directory named for it (ScribeTwo keeps
    its reports in ``<input>.scribe2/``)."""
    prefix = input_name + "."
    out: list[tuple[str, Path]] = []
    for path in sorted(output_dir.iterdir()):
        if not path.name.startswith(prefix):
            continue  # the input itself, logs directories, anything unrelated
        name = "<input>" + path.name[len(input_name) :]
        if path.is_file():
            out.append((name, path))
        elif path.is_dir():
            for inner in sorted(path.iterdir()):
                if inner.is_file() and inner.name.startswith(prefix):
                    out.append((f"{name}/<input>{inner.name[len(input_name):]}", inner))
    return out


def _inventory(output_dir: Path, input_name: str) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for name, path in _outputs(output_dir, input_name):
        entry: dict[str, object] = {"name": name}
        if path.suffix in _SQLITE_SUFFIXES:
            entry["kind"] = "sqlite"
            entry["tables_with_rows"] = [t for t, filled in _filled_tables(path).items() if filled]
        elif path.suffix in _TEXT_TABLE_SUFFIXES and not path.name.endswith(_NOT_TABLES):
            entry["kind"] = "tsv"
            with path.open() as fh:
                entry["columns"] = fh.readline().rstrip("\n").split("\t")
        else:
            entry["kind"] = path.suffix.lstrip(".") or "file"
        files.append(entry)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--program", required=True, help="program name, as in programs.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="a finished run's outputs")
    parser.add_argument(
        "--spectra-file",
        type=Path,
        default=None,
        help="a .dia written by the pinned toolchain, for its table definitions",
    )
    parser.add_argument("--version", default=MINIMUM_VERSION, help="EncyclopeDIA version")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("constellation/data/_raw/encyclopedia"),
        help="where the per-version capture lives",
    )
    args = parser.parse_args()

    root = args.raw_root / args.version
    input_name = _input_name(args.output_dir)
    library = next(
        p
        for p in args.output_dir.iterdir()
        if p.name in (f"{input_name}.elib", f"{input_name}.dlib")
    )
    metadata = _metadata(library)
    container = metadata.get("version", "unknown")
    recorded = metadata.get("EncyclopediaVersion")
    if recorded is not None and recorded != args.version:
        print(
            f"error: {library.name} was written by EncyclopeDIA {recorded}, not {args.version}",
            file=sys.stderr,
        )
        return 1

    _write_sql(
        root / "schema" / "result_library.sql",
        f"EncyclopeDIA {args.version} result library (.elib / .dlib), container {container}",
        _ddl(library),
    )
    if args.spectra_file is not None:
        dia_version = _metadata(args.spectra_file).get("version", "unknown")
        _write_sql(
            root / "schema" / "spectra_file.sql",
            f"EncyclopeDIA {args.version} spectra file (.dia), format {dia_version}",
            _ddl(args.spectra_file),
        )

    # Three keys are named for the run: "<source file> search type",
    # "<source file> library" and "TIC_<source file>".
    source = metadata.get("sourcefile")

    def _generic(key: str) -> str:
        if source and key in (f"{source} search type", f"{source} library"):
            return "<source file>" + key[len(source) :]
        if source and key == f"TIC_{source}":
            return "TIC_<source file>"
        return key

    keys = sorted(_generic(k) for k in metadata)
    doc = {
        "encyclopedia_version": args.version,
        "program": args.program,
        "container_version": container,
        "note": _NOTES.get(args.program, _NOTES["encyclopedia"]),
        "files": _inventory(args.output_dir, input_name),
        "result_metadata_keys": keys,
    }
    out = root / "outputs" / f"{args.program}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {root / 'schema'} and {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
