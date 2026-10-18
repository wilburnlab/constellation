"""What an experiment directory holds, said plainly.

A search's directory under ``units/`` is named for the run it searched
plus a digest of how it was searched (``<run>-<8 hex digits>``). That
keeps every search apart, so nothing is ever replaced, but the name
says nothing else. :func:`summarize` says the rest: for an experiment,
or for one search in it, what was searched, with which library, FASTA
and settings, whether it finished, what it found, and what sets it
apart from the other searches of the same acquisition.
:func:`format_summary` renders that as text, and ``constellation
massspec summarize`` prints it (or, with ``--json``, the summary itself).

Everything comes from what the directory records about itself: the
experiment and search manifests and the acquisitions table. Nothing is
recomputed and no result file is opened. A search's manifest is written
when the search starts, with the flags it was given, and again when it
ends, with every setting the engine recorded in its result. So a search
that failed or was stopped can be described too; its settings are then
the flags it was given over the engine's documented defaults, and the
summary says which of the two a search's settings are.
"""

from __future__ import annotations

import math
import textwrap
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from constellation.core import stages
from constellation.core.manifests import ManifestError, manifest_kind, read_manifest
from constellation.massspec.acquisitions import load_acquisitions
from constellation.massspec.manifest import (
    ExperimentManifest,
    QuantReportManifest,
    SearchManifest,
)

#: Flags that name a search's input files; shown as inputs, not settings.
_INPUT_FLAGS = frozenset({"-i", "-f", "-l", "-o"})

#: Where a search's settings come from (``settings_source``).
RECORDED = "recorded"  # by the engine, in its result: what the search ran with
PASSED = "passed"  # the flags it was given, over the engine's documented defaults

_WRAP = 100


# ──────────────────────────────────────────────────────────────────────
# the engine-specific parts
# ──────────────────────────────────────────────────────────────────────


def _execution_only(engine: str) -> frozenset[str]:
    if engine == "encyclopedia":
        from constellation.massspec.search.encyclopedia.params import EXECUTION_ONLY_FLAGS

        return EXECUTION_ONLY_FLAGS
    return frozenset()


def _documented_defaults(engine: str, program: str) -> dict[str, str]:
    if engine != "encyclopedia":
        return {}
    from constellation.massspec.search.encyclopedia import cli_spec

    flags = cli_spec.load_cli_spec()["programs"].get(program, {}).get("flags", {})
    return {
        flag: str(entry["default"])
        for flag, entry in flags.items()
        if entry.get("default") is not None and flag not in _INPUT_FLAGS
    }


def _readable(
    engine: str, settings: Mapping[str, str], flags: Iterable[str] | None = None
) -> list[tuple[str, str]]:
    if engine == "encyclopedia":
        from constellation.massspec.search.encyclopedia.params import readable_settings

        return readable_settings(settings, flags=flags)
    keys = sorted(settings if flags is None else flags)
    return [(k, str(settings.get(k, "(not recorded)"))) for k in keys]


def _headline_flags(engine: str) -> frozenset[str]:
    if engine != "encyclopedia":
        return frozenset()
    from constellation.massspec.search.encyclopedia.params import HEADLINE_SETTINGS

    return frozenset(
        flag for _, value, unit in HEADLINE_SETTINGS for flag in (value, unit) if flag
    )


# ──────────────────────────────────────────────────────────────────────
# reading
# ──────────────────────────────────────────────────────────────────────


def summarize(directory: Path | str) -> dict[str, Any]:
    """Summary of an experiment directory, or of one search directory in it.

    Returns plain data (it serialises as JSON as it is)::

        path, name, created_at, constellation_version, id_scheme
        acquisitions   [{acquisition_id, name, source_file, source_kind,
                         acquired_at, source_sha256, aliases, merged_from}]
        searches       [{unit, directory, status, error, ...}], one per
                       directory under units/, finished or not
        reports        [{key, directory, status, units, acquisition_ids,
                         found, ...}], one per directory under reports/
        runs           [{label, state, pid, started_at, log}], the detached
                       runs recorded under runs/
        focus          the search's or report's key when ``directory`` was
                       one of them, else None

    ``status`` is ``finished`` (its ``_SUCCESS`` marker is there),
    ``failed`` or ``cancelled`` (it said so as it stopped),
    ``not finished`` (it started and has not said anything since: still
    running, or stopped without the chance) or ``no record`` (a
    directory with no manifest in it). Raises ``ValueError`` for a
    directory that is neither an experiment nor a search in one.
    """
    directory = Path(directory).resolve()
    focus: str | None = None
    kind = manifest_kind(directory)
    if kind == ExperimentManifest.KIND:
        experiment_dir = directory
    elif directory.parent.name in ("units", "reports") and (
        manifest_kind(directory.parent.parent) == ExperimentManifest.KIND
    ):
        experiment_dir, focus = directory.parent.parent, directory.name
    else:
        what = f"a {kind} directory" if kind else "not an experiment directory"
        raise ValueError(
            f"{directory} is {what}; give an experiment directory (what `massspec search "
            f"--output-dir` names) or one of the search directories under its units/"
        )

    root = read_manifest(experiment_dir, expect=ExperimentManifest)
    units = experiment_dir / "units"
    searches = [
        _search(experiment_dir, unit_dir, registered=unit_dir.name in root.units)
        for unit_dir in sorted(units.iterdir() if units.is_dir() else ())
        if unit_dir.is_dir()
    ]
    _compare_within_acquisitions(searches)
    for search in searches:
        search.pop("_compare", None)
    reports_dir = experiment_dir / "reports"
    reports = [
        _report(report_dir, registered=report_dir.name in root.reports)
        for report_dir in sorted(reports_dir.iterdir() if reports_dir.is_dir() else ())
        if report_dir.is_dir()
    ]
    return {
        "path": str(experiment_dir),
        "name": root.name,
        "created_at": root.created_at,
        "constellation_version": root.constellation_version,
        "id_scheme": root.id_scheme,
        "acquisitions": _acquisitions(experiment_dir, root),
        "searches": searches,
        "reports": reports,
        "runs": _runs(experiment_dir),
        "focus": focus,
    }


def _file_name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _acquisitions(experiment_dir: Path, root: ExperimentManifest) -> list[dict[str, Any]]:
    try:
        table = load_acquisitions(experiment_dir / root.acquisitions).table
    except FileNotFoundError:
        return []
    out = []
    for row in table.to_pylist():
        out.append(
            {
                "acquisition_id": row["acquisition_id"],
                "name": _file_name(str(row["source_file"])),
                "source_file": row["source_file"],
                "source_kind": row.get("source_kind"),
                # ISO 8601 as the instrument wrote it (its clock, its zone).
                "acquired_at": row.get("acquisition_datetime"),
                "source_sha256": row.get("source_sha256"),
                "aliases": list(row.get("source_aliases") or []),
                "merged_from": list(row["merged_from"]) if row.get("merged_from") else None,
            }
        )
    return sorted(out, key=lambda a: a["acquisition_id"])


def _last_activity(unit_dir: Path) -> str | None:
    """When anything in the search's directory was last written."""
    times = [
        p.lstat().st_mtime
        for sub in ("logs", "jar")
        for p in (unit_dir / sub).glob("*")
        if (unit_dir / sub).is_dir()
    ]
    if not times:
        return None
    return datetime.fromtimestamp(max(times), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _settings_view(record: SearchManifest) -> tuple[dict[str, str], str]:
    if record.engine_settings:
        return dict(record.engine_settings), RECORDED
    passed = {k: v for k, v in record.engine_flags.items() if k not in _INPUT_FLAGS}
    return {**_documented_defaults(record.engine, record.program), **passed}, PASSED


def _search(experiment_dir: Path, unit_dir: Path, *, registered: bool) -> dict[str, Any]:
    key = unit_dir.name
    entry: dict[str, Any] = {
        "unit": key,
        "directory": f"units/{key}",
        "in_experiment_manifest": registered,
    }
    try:
        record = read_manifest(unit_dir, expect=SearchManifest)
    except ManifestError:
        entry.update(status="no record", last_activity=_last_activity(unit_dir))
        return entry

    finished = stages.is_done(unit_dir)
    if finished:
        status = "finished"
    elif record.status in ("failed", "cancelled"):
        status = record.status
    else:
        status = "not finished"
    view, source = _settings_view(record)
    rows = record.runtime.get("table_rows") or {}
    threshold = view.get("-percolatorThreshold")
    result_library = record.outputs.get("result_library")
    entry.update(
        status=status,
        error=record.error or None,
        started_at=record.created_at or None,
        finished_at=record.runtime.get("finished_at"),
        elapsed_seconds=record.runtime.get("elapsed_seconds"),
        last_activity=None if finished else _last_activity(unit_dir),
        engine=record.engine,
        engine_version=record.engine_version,
        program=record.program,
        search_mode=record.search_mode,
        acquisition_ids=list(record.acquisition_ids),
        library=(record.inputs.get("library") or {}).get("path"),
        library_sha256=(record.inputs.get("library") or {}).get("sha256"),
        fasta=(record.inputs.get("fasta") or {}).get("path"),
        fasta_sha256=(record.inputs.get("fasta") or {}).get("sha256"),
        fasta_copy=(record.inputs.get("fasta") or {}).get("copy"),
        search_type=record.engine_flags.get("-searchType"),
        spectra_file=dict(record.inputs.get("spectra_file") or {}) or None,
        settings=[list(pair) for pair in _spectrum_centric_view(record, _readable(record.engine, view))],
        settings_source=source,
        all_settings=view,
        set_explicitly={k: v for k, v in record.engine_flags.items() if k not in _INPUT_FLAGS},
        found=(
            {
                "precursors": rows.get("precursor"),
                "peptides": rows.get("peptide"),
                "proteins": rows.get("protein_score"),
                "fdr": _fdr(threshold),
            }
            if finished and rows
            else None
        ),
        caveats=list(record.caveats),
        result_library=f"units/{key}/{result_library}" if result_library else None,
        logs=f"units/{key}/logs",
        command=list(record.argv),
        same_acquisitions_as=[],
        differs_by=[],
    )
    entry["_compare"] = {
        "library": entry["library"] or "none",
        "FASTA": entry["fasta"] or "none",
        "engine": f"{record.engine} {record.engine_version}".strip(),
        "search mode": record.search_mode,
        **{k: v for k, v in view.items() if k not in _execution_only(record.engine)},
    }
    return entry


#: Headline settings that say nothing about a ScribeTwo search: it scores
#: with its own JavaPot, whatever Percolator it records.
_NOT_SCRIBETWO = frozenset({"Percolator"})


def _spectrum_centric_view(record: SearchManifest, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The headline settings of a ScribeTwo search: its search type first,
    its scorer instead of a Percolator it does not run, and a tolerance it
    adjusted shown with the value it was given."""
    if record.program != "scribetwo":
        return pairs
    given = {
        "precursor tolerance": record.engine_flags.get("-ptol"),
        "fragment tolerance": record.engine_flags.get("-ftol"),
    }
    out: list[tuple[str, str]] = []
    if record.engine_flags.get("-searchType"):
        out.append(("search type", record.engine_flags["-searchType"]))
    for label, value in pairs:
        if label in _NOT_SCRIBETWO:
            continue
        asked = given.get(label)
        if asked is not None and record.engine_settings and not value.startswith(f"{asked} "):
            number, _, unit = value.partition(" ")
            try:
                number = f"{float(number):.4g}"
            except ValueError:
                pass
            value = f"{number} {unit}".strip() + f" (adjusted by ScribeTwo; given {asked})"
        out.append((label, value))
    out.append(("scoring", "JavaPot (ScribeTwo's own)"))
    return out


def _library_line(s: dict[str, Any]) -> str:
    if s.get("library"):
        return _file_name(s["library"])
    return "none: predicted from the FASTA" if s.get("program") == "scribetwo" else "none"


def _fdr(threshold: str | None) -> str | None:
    try:
        return f"{float(threshold) * 100:g}%" if threshold is not None else None
    except ValueError:
        return None


def _report(report_dir: Path, *, registered: bool) -> dict[str, Any]:
    key = report_dir.name
    entry: dict[str, Any] = {
        "key": key,
        "directory": f"reports/{key}",
        "in_experiment_manifest": registered,
    }
    try:
        record = read_manifest(report_dir, expect=QuantReportManifest)
    except ManifestError:
        entry.update(status="no record", last_activity=_last_activity(report_dir))
        return entry
    finished = stages.is_done(report_dir)
    if finished:
        status = "finished"
    elif record.status in ("failed", "cancelled"):
        status = record.status
    else:
        status = "not finished"
    rows = record.runtime.get("table_rows") or {}
    entry.update(
        status=status,
        error=record.error or None,
        started_at=record.created_at or None,
        finished_at=record.runtime.get("finished_at"),
        elapsed_seconds=record.runtime.get("elapsed_seconds"),
        last_activity=None if finished else _last_activity(report_dir),
        engine=record.engine,
        engine_version=record.engine_version,
        units=list(record.units),
        acquisition_ids=list(record.acquisition_ids),
        library=(record.inputs.get("library") or {}).get("path"),
        fasta=(record.inputs.get("fasta") or {}).get("path"),
        align=record.parameters.get("align"),
        found=(
            {
                "precursors": rows.get("precursor"),
                "proteins": rows.get("protein_score"),
            }
            if finished and rows
            else None
        ),
        outputs={k: f"reports/{key}/{v}" for k, v in record.outputs.items()},
        caveats=list(record.caveats),
        command=list(record.argv),
        logs=f"reports/{key}/logs",
    )
    return entry


def _runs(experiment_dir: Path) -> list[dict[str, Any]]:
    from constellation.core.detach import runs

    return [
        {
            "label": run.label,
            "state": run.state(),
            "pid": run.pid,
            "started_at": run.started_at,
            "exit_code": run.exit_code(),
            "log": str(run.log.relative_to(experiment_dir)),
        }
        for run in runs(experiment_dir / "runs")
    ]


# ──────────────────────────────────────────────────────────────────────
# what sets searches of one acquisition apart
# ──────────────────────────────────────────────────────────────────────


def _same(a: object, b: object) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(a).casefold() == str(b).casefold()


_INPUT_KEYS = ("library", "FASTA", "engine", "search mode")


def _compare_within_acquisitions(searches: list[dict[str, Any]]) -> None:
    """Fill ``same_acquisitions_as`` and ``differs_by`` for every search
    that shares its acquisitions with another.

    Only what both sides know is compared: a setting one search lacks
    (an engine-version difference in what is recorded) is skipped rather
    than called a difference.
    """
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for search in searches:
        if "_compare" in search:
            groups[tuple(sorted(search["acquisition_ids"]))].append(search)
    for group in groups.values():
        if len(group) < 2:
            continue
        views = [s["_compare"] for s in group]
        shared = set.intersection(*(set(v) for v in views))
        differing = sorted(
            k for k in shared if not all(_same(v[k], views[0][k]) for v in views[1:])
        )
        for search in group:
            search["same_acquisitions_as"] = [o["unit"] for o in group if o is not search]
            search["differs_by"] = [list(p) for p in _render_differences(search, group, differing)]


def _render_differences(
    search: dict[str, Any], group: list[dict[str, Any]], differing: list[str]
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key in _INPUT_KEYS:
        if key not in differing:
            continue
        value = search["_compare"][key]
        if key in ("library", "FASTA") and value != "none":
            full = {o["_compare"][key] for o in group}
            names = {_file_name(str(v)) for v in full}
            value = _file_name(value) if len(names) == len(full) else value
        out.append((key, str(value)))
    flags = [k for k in differing if k not in _INPUT_KEYS]
    if flags:
        out.extend(_readable(search["engine"], search["_compare"], flags=flags))
    return out


# ──────────────────────────────────────────────────────────────────────
# text
# ──────────────────────────────────────────────────────────────────────


def _when(iso: str | None) -> str:
    if not iso:
        return "(unknown time)"
    return iso.replace("T", " ").replace("Z", "")[:16] + " UTC"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def _field(label: str, text: str, indent: int = 6) -> list[str]:
    head = " " * indent + f"{label:<10}"
    return textwrap.wrap(
        text,
        width=_WRAP,
        initial_indent=head,
        subsequent_indent=" " * len(head),
        break_on_hyphens=False,
        break_long_words=False,  # a path or a command line stays whole, to copy
    ) or [head.rstrip()]


def _status_line(s: dict[str, Any]) -> str:
    status = s["status"]
    if status == "finished":
        took = _duration(s.get("elapsed_seconds"))
        what = "the report" if "key" in s else "the search"
        return f"finished {_when(s.get('finished_at'))}" + (f", {what} took {took}" if took else "")
    if status in ("failed", "cancelled"):
        first = (s.get("error") or "").splitlines()[0] if s.get("error") else ""
        return f"{status} {_when(s.get('finished_at'))}" + (f": {first}" if first else "")
    if status == "not finished":
        return (
            f"not finished: started {_when(s.get('started_at'))}, last written "
            f"{_when(s.get('last_activity'))} (still running, or stopped without a chance "
            f"to say so)"
        )
    return (
        f"no record: nothing in this directory says what it is (last written "
        f"{_when(s.get('last_activity'))})"
    )


def _found(s: dict[str, Any]) -> str | None:
    found = s.get("found")
    if not found:
        return None
    parts = [
        f"{found[k]:,} {k}" for k in ("precursors", "peptides", "proteins") if found.get(k) is not None
    ]
    return ", ".join(parts) + (f" at {found['fdr']} FDR" if found.get("fdr") else "")


def _pairs(pairs: Iterable[Iterable[str]]) -> str:
    return "; ".join(f"{label} {value}" for label, value in pairs)


def _acquisition_label(ids: list[int]) -> str:
    word = "acquisition" if len(ids) == 1 else "acquisitions"
    return f"{word} " + ", ".join(str(i) for i in ids)


def _search_block(s: dict[str, Any], names: Mapping[int, str]) -> list[str]:
    lines = [f"  {s['unit']}"]
    lines += _field("status", _status_line(s))
    if s["status"] == "no record":
        return lines
    lines += _field(
        "searched",
        f"{_acquisition_label(s['acquisition_ids'])}, {s['search_mode']}, "
        f"{s['engine']} {s['engine_version']}",
    )
    lines += _field("library", _library_line(s))
    if s.get("fasta"):
        lines += _field("FASTA", _file_name(s["fasta"]))
    source = "" if s["settings_source"] == RECORDED else " (as given, the rest at the jar's defaults)"
    lines += _field("settings", _pairs(s["settings"]) + source)
    found = _found(s)
    if found:
        lines += _field("found", found)
    if s.get("same_acquisitions_as"):
        if s["differs_by"]:
            lines += _field("differs", _pairs(s["differs_by"]) + "  (from the other searches of this acquisition)")
        else:
            lines += _field(
                "differs",
                "in no setting from " + ", ".join(s["same_acquisitions_as"])
                + "; the input files themselves differ",
            )
    if s["status"] in ("failed", "cancelled", "not finished"):
        lines += _field("logs", s["logs"])
    if s.get("caveats"):
        n = len(s["caveats"])
        lines += _field("notes", f"{n} caveat{'s' if n != 1 else ''}; summarize {s['directory']} to read them")
    return lines


def _detail_block(s: dict[str, Any], names: Mapping[int, str]) -> list[str]:
    lines = [f"search {s['unit']}"]
    lines += _field("status", _status_line(s), indent=2)
    if s["status"] == "no record":
        return lines
    if s.get("error") and "\n" in s["error"]:
        lines += ["      " + line for line in s["error"].splitlines()[1:]]
    if s.get("started_at"):
        lines += _field("started", _when(s["started_at"]), indent=2)
    lines += _field(
        "searched",
        "; ".join(
            f"acquisition {i} ({names.get(i, '?')})" for i in s["acquisition_ids"]
        ),
        indent=2,
    )
    lines += _field(
        "engine", f"{s['engine']} {s['engine_version']}, program {s['program']}, {s['search_mode']}", indent=2
    )
    for label, path, sha in (
        ("library", s.get("library"), s.get("library_sha256")),
        ("FASTA", s.get("fasta"), s.get("fasta_sha256")),
    ):
        if path:
            lines += _field(label, f"{path}" + (f"  (sha256 {sha[:16]}…)" if sha else ""), indent=2)
        elif label == "library" and s.get("program") == "scribetwo":
            lines += _field(label, _library_line(s), indent=2)
    if s.get("fasta_copy"):
        lines += _field(
            "", f"ScribeTwo read the experiment's copy, {s['fasta_copy']}, and indexed it there",
            indent=2,
        )  # fmt: skip
    spectra = s.get("spectra_file")
    if spectra:
        how = "written by Constellation" if spectra.get("written_by") == "constellation" else "given"
        lines += _field(
            "spectra",
            f"{spectra.get('path')}  ({how}; in the search's directory as "
            f"{spectra.get('link')}, a {spectra.get('linked_as')})",
            indent=2,
        )
    found = _found(s)
    if found:
        lines += _field("found", found, indent=2)
    if s.get("result_library"):
        lines += _field("result", s["result_library"], indent=2)
    lines += _field("logs", s["logs"], indent=2)
    source = (
        "as the engine recorded them in its result"
        if s["settings_source"] == RECORDED
        else "as given, the rest at the jar's documented defaults: the search did not finish"
    )
    lines.append(f"  settings ({source})")
    headline = [tuple(p) for p in s["settings"]]
    width = max([len(label) for label, _ in headline] + [len(k) for k in s["all_settings"]] + [10])
    for label, value in headline:
        lines.append(f"      {label:<{width}}  {value}")
    shown = _headline_flags(s["engine"])
    for flag in sorted(set(s["all_settings"]) - shown, key=str.lower):
        lines.append(f"      {flag:<{width}}  {s['all_settings'][flag]}")
    if s.get("set_explicitly"):
        lines += _field(
            "given",
            "  ".join(f"{k} {v!r}" if v == "" else f"{k} {v}" for k, v in s["set_explicitly"].items()),
            indent=2,
        )
    if s.get("command"):
        lines += _field("command", " ".join(s["command"]), indent=2)
    for caveat in s.get("caveats") or []:
        lines += _field("caveat", caveat, indent=2)
    if s.get("same_acquisitions_as"):
        lines.append("  other searches of the same acquisition in this experiment")
        for other in s["same_acquisitions_as"]:
            lines.append(f"      {other}")
        if s["differs_by"]:
            lines += _field("this one", _pairs(s["differs_by"]), indent=6)
    return lines


def format_summary(summary: Mapping[str, Any]) -> str:
    """The summary as text for a terminal."""
    names = {a["acquisition_id"]: a["name"] for a in summary["acquisitions"]}
    lines = [f"experiment {summary['name']}  ({summary['path']})"]
    made = f"created {_when(summary.get('created_at'))}"
    if summary.get("constellation_version"):
        made += f" by Constellation {summary['constellation_version']}"
    if summary.get("id_scheme"):
        made += f"; content-derived ids ({summary['id_scheme']})"
    lines.append("  " + made)
    focus = summary.get("focus")
    if focus is not None:
        found = [s for s in summary["searches"] if s["unit"] == focus]
        lines.append("")
        if found:
            lines += _detail_block(found[0], names)
        else:
            (report,) = [r for r in summary.get("reports", []) if r["key"] == focus]
            lines += _report_detail(report, names)
        return "\n".join(lines) + "\n"

    acquisitions = summary["acquisitions"]
    lines += ["", f"acquisitions ({len(acquisitions)})"]
    for a in acquisitions:
        extra = []
        if a.get("acquired_at"):
            extra.append(f"acquired {str(a['acquired_at']).replace('T', ' ')[:16]}")
        if a.get("source_sha256"):
            extra.append(f"sha256 {a['source_sha256'][:12]}…")
        if a.get("merged_from"):
            extra.append("merged from " + ", ".join(str(i) for i in a["merged_from"]))
        lines.append(f"  {a['acquisition_id']:>3}  {a['name']}" + (f"  ({', '.join(extra)})" if extra else ""))
    searches = summary["searches"]
    counts = defaultdict(int)
    for s in searches:
        counts[s["status"]] += 1
    tally = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    lines += ["", f"searches ({len(searches)}{': ' + tally if tally else ''})"]
    for s in searches:
        lines += _search_block(s, names)
    reports = summary.get("reports") or []
    if reports:
        lines += ["", f"reports ({len(reports)})"]
        for r in reports:
            lines += _report_block(r)
    runs = summary.get("runs") or []
    if runs:
        lines += ["", "detached runs"]
        for run in runs:
            lines.append(
                f"  {run['label']:<9} {run['state']:<9} started {_when(run['started_at'])}, "
                f"pid {run['pid']}, log {run['log']}"
            )
    if searches or reports:
        lines += [
            "",
            "`constellation massspec summarize <experiment>/units/<search>` (or "
            "/reports/<report>) shows one in full.",
        ]
    return "\n".join(lines) + "\n"


def _report_block(r: dict[str, Any]) -> list[str]:
    lines = [f"  {r['key']}"]
    lines += _field("status", _status_line(r))
    if r["status"] == "no record":
        return lines
    ids = r.get("acquisition_ids") or []
    lines += _field(
        "covers",
        f"{len(r.get('units') or [])} searches, acquisitions {', '.join(str(i) for i in ids)}"
        + ("" if r.get("align") is None else f"; retention times {'aligned' if r['align'] else 'not aligned'}"),
    )
    if r.get("library"):
        lines += _field("library", _file_name(r["library"]))
    if r.get("found"):
        found = r["found"]
        lines += _field(
            "found",
            f"{found.get('precursors') or 0:,} precursors, {found.get('proteins') or 0:,} proteins",
        )
    if r["status"] in ("failed", "cancelled", "not finished"):
        lines += _field("logs", r["logs"])
    return lines


def _report_detail(r: dict[str, Any], names: Mapping[int, str]) -> list[str]:
    lines = [f"report {r['key']}"]
    lines += _field("status", _status_line(r), indent=2)
    if r["status"] == "no record":
        return lines
    if r.get("error") and "\n" in r["error"]:
        lines += ["      " + line for line in r["error"].splitlines()[1:]]
    lines += _field("engine", f"{r['engine']} {r['engine_version']}", indent=2)
    for label, path in (("library", r.get("library")), ("FASTA", r.get("fasta"))):
        if path:
            lines += _field(label, path, indent=2)
    lines.append("  searches")
    for unit in r.get("units") or []:
        lines.append(f"      {unit}")
    lines += _field(
        "runs",
        "; ".join(f"acquisition {i} ({names.get(i, '?')})" for i in r.get("acquisition_ids") or []),
        indent=2,
    )
    if r.get("found"):
        found = r["found"]
        lines += _field(
            "found", f"{found.get('precursors') or 0:,} precursors, {found.get('proteins') or 0:,} proteins",
            indent=2,
        )
    for label, path in sorted((r.get("outputs") or {}).items()):
        lines += _field("output", path, indent=2)
    if r.get("command"):
        lines += _field("command", " ".join(r["command"]), indent=2)
    for caveat in r.get("caveats") or []:
        lines += _field("caveat", caveat, indent=2)
    return lines


__all__ = ["PASSED", "RECORDED", "format_summary", "summarize"]
