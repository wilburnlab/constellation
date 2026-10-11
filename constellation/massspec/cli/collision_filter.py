"""``constellation massspec collision-filter``: the DIA co-elution filter on a chromatogram library."""

from __future__ import annotations

import argparse
from pathlib import Path

from constellation.massspec.cli._common import (
    _add_output_dir_arg,
    _report_ingest_caveats,
)


# ── collision-filter ────────────────────────────────────────────────────


def _build_collision_filter_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "collision-filter",
        help=(
            "Apply the DIA collision filter to a search .elib and write a "
            "filtered .elib with the collided identifications removed. "
            "Drops co-eluting peptides in the same GPF isolation window "
            "that share fragment ions (keeping the lowest-Score per "
            "cluster), at the SQLite level so the output is consumable by "
            "library-export."
        ),
    )
    p.add_argument(
        "--elib",
        required=True,
        type=Path,
        help="search .elib to filter (the EncyclopeDIA search output)",
    )
    p.add_argument(
        "--dia",
        required=True,
        type=Path,
        help=(
            "combined .dia file — supplies the GPF isolation-window ranges "
            "the filter groups co-eluting peptides by"
        ),
    )
    p.add_argument(
        "--output-elib",
        required=True,
        type=Path,
        help="destination for the collision-filtered .elib",
    )
    _add_output_dir_arg(p)
    p.add_argument(
        "--rt-threshold-s",
        type=float,
        default=5.0,
        help="max |ΔRT| (s) for a candidate co-elution pair (default 5.0)",
    )
    p.add_argument(
        "--frag-ppm-tol",
        type=float,
        default=20.0,
        help="fragment m/z match tolerance in ppm (default 20.0)",
    )
    p.add_argument(
        "--min-shared-ions",
        type=int,
        default=4,
        help="min shared observed fragment ions to flag a pair (default 4)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip if <output-dir>/_SUCCESS exists",
    )
    p.add_argument(
        "--no-ingest",
        action="store_true",
        help=(
            "skip auto-ingesting the filtered .elib into ParquetDir bundles "
            "(library_pqdir / quant_pqdir / search_pqdir)"
        ),
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress progress messages on stderr",
    )
    p.set_defaults(func=_cmd_massspec_collision_filter)


def _cmd_massspec_collision_filter(args: argparse.Namespace) -> int:
    """Collision-filter a search .elib and write a filtered .elib.

    Runs ``apply_collision_filter`` to find the collided identifications,
    writes a SQLite-filtered .elib with the loser rows removed (via
    ``filter_elib_by_losers``), drops a ``collision_metadata.json``
    sidecar, optionally auto-ingests the filtered .elib into ParquetDir
    bundles, writes ``manifest.json``, and touches ``_SUCCESS``.
    """
    import json as _json
    import os as _os
    import platform
    import socket
    import sys as _sys

    from constellation import __version__ as constellation_version
    from constellation.massspec.search import (
        apply_collision_filter,
        filter_elib_by_losers,
    )

    elib = Path(args.elib).resolve()
    dia = Path(args.dia).resolve()
    output_elib = Path(args.output_elib).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not elib.is_file():
        print(f"error: --elib not found: {elib}", file=_sys.stderr)
        return 1
    if not dia.is_file():
        print(f"error: --dia not found: {dia}", file=_sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_elib.parent.mkdir(parents=True, exist_ok=True)

    success_path = output_dir / "_SUCCESS"
    if success_path.exists() and not args.resume:
        print(
            f"error: --output-dir already complete (touch _SUCCESS exists). "
            f"Pass --resume to re-use it, or delete {output_dir} to start "
            f"fresh.",
            file=_sys.stderr,
        )
        return 1
    if success_path.exists() and args.resume:
        print(f"already complete: {output_dir}")
        return 0

    if not args.no_progress:
        print(
            f"[collision-filter] running on {elib} + {dia} "
            f"(rt<{args.rt_threshold_s}s, {args.frag_ppm_tol} ppm, "
            f">={args.min_shared_ions} shared ions) ...",
            file=_sys.stderr,
        )
    losers, collision_metadata = apply_collision_filter(
        elib_path=elib,
        dia_path=dia,
        rt_threshold_s=args.rt_threshold_s,
        frag_ppm_tol=args.frag_ppm_tol,
        min_shared_ions=args.min_shared_ions,
        return_metadata=True,
    )
    rows_deleted = filter_elib_by_losers(elib, losers, output_elib)
    collision_metadata["n_losers"] = len(losers)
    collision_metadata["rows_deleted"] = rows_deleted
    (output_dir / "collision_metadata.json").write_text(
        _json.dumps(collision_metadata, indent=2) + "\n"
    )

    if not args.no_progress:
        print(
            f"[collision-filter] dropped {len(losers):,} modseqs across "
            f"{len(collision_metadata['clusters']):,} clusters "
            f"(of {collision_metadata['n_entries']:,} entries) → "
            f"{output_elib}",
            file=_sys.stderr,
        )

    # ── auto-ingest the filtered .elib ──────────────────────────────
    ingest_info: dict[str, object] = {"skipped": bool(args.no_ingest)}
    library_pqdir: Path | None = None
    quant_pqdir: Path | None = None
    search_pqdir: Path | None = None
    if not args.no_ingest:
        from constellation.massspec.io.encyclopedia import read_encyclopedia
        from constellation.massspec.library import save_library
        from constellation.massspec.quant import save_quant
        from constellation.massspec.search import save_search

        try:
            ingest = read_encyclopedia(output_elib)
        except Exception as exc:  # noqa: BLE001 — surface in manifest
            ingest_info["error"] = f"{type(exc).__name__}: {exc}"
        else:
            library_pqdir = output_dir / "library_pqdir"
            save_library(ingest.library, library_pqdir, format="parquet_dir")
            _report_ingest_caveats(ingest, ingest_info)
            ingest_info["library_counts"] = {
                "proteins": ingest.library.proteins.num_rows,
                "peptides": ingest.library.peptides.num_rows,
                "precursors": ingest.library.precursors.num_rows,
                "fragments": ingest.library.fragments.num_rows,
            }
            if ingest.quant is not None:
                quant_pqdir = output_dir / "quant_pqdir"
                save_quant(ingest.quant, quant_pqdir, format="parquet_dir")
            if ingest.search is not None:
                search_pqdir = output_dir / "search_pqdir"
                save_search(ingest.search, search_pqdir, format="parquet_dir")

    manifest = {
        "constellation_version": constellation_version,
        "subcommand": "massspec collision-filter",
        "argv": _sys.argv,
        "inputs": {"elib": str(elib), "dia": str(dia)},
        "outputs": {
            "filtered_elib": str(output_elib),
            "library_pqdir": str(library_pqdir) if library_pqdir else None,
            "quant_pqdir": str(quant_pqdir) if quant_pqdir else None,
            "search_pqdir": str(search_pqdir) if search_pqdir else None,
            "collision_metadata": str(output_dir / "collision_metadata.json"),
        },
        "params": {
            "rt_threshold_s": args.rt_threshold_s,
            "frag_ppm_tol": args.frag_ppm_tol,
            "min_shared_ions": args.min_shared_ions,
        },
        "counts": {
            "n_losers": len(losers),
            "n_clusters": len(collision_metadata["clusters"]),
            "n_entries": collision_metadata["n_entries"],
            "rows_deleted": rows_deleted,
        },
        "ingest": ingest_info,
        "runtime": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": _sys.version.split()[0],
            "user": _os.environ.get("USER", "unknown"),
        },
    }
    (output_dir / "manifest.json").write_text(
        _json.dumps(manifest, indent=2) + "\n"
    )
    success_path.write_bytes(b"")
    return 0
