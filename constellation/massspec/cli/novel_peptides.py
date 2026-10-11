"""``constellation massspec classify-novel-peptides``: detected peptides against a novel-vs-reference protein alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

from constellation.massspec.cli._common import _add_output_dir_arg


# ── classify-novel-peptides ─────────────────────────────────────────────


def _build_classify_novel_peptides_parser(
    subs: argparse._SubParsersAction,
) -> None:
    p = subs.add_parser(
        "classify-novel-peptides",
        help=(
            "Classify detected novel peptides against a reference-vs-novel "
            "mmseqs2 alignment. Walks each per-hit CIGAR string and assigns "
            "one of 11 deviation classes (snp / insertion / deletion / "
            "complex / truncations / cutsite mutation / deviations / "
            "unknown / non_reference). Ports the cartographer "
            "nanopore.classify_novel_peptides workflow."
        ),
    )
    p.add_argument(
        "--search-results",
        required=True,
        type=Path,
        help=(
            "Path to a ParquetDir Search bundle, OR a .elib SQLite file. "
            "Only the detected-peptide sequence set is consulted (column "
            "PeptideSeq for .elib; peptide_id+modseq-joined-to-library for "
            "ParquetDir)."
        ),
    )
    p.add_argument(
        "--library",
        required=True,
        type=Path,
        help=(
            "Path to a ParquetDir Library bundle. Used to resolve "
            "peptide_id ↔ peptide_sequence when --search-results is a "
            "ParquetDir Search."
        ),
    )
    p.add_argument(
        "--alignment-hits",
        required=True,
        type=Path,
        help=(
            "Path to an AlignmentHits ParquetDir bundle, OR a headerless "
            "mmseqs2 .tab TSV with the canonical 8-column layout "
            "(query, target, evalue, qstart, qend, tstart, tend, cigar). "
            "May contain mixed reference + swissprot hits per query — "
            "tier inference happens internally based on target membership "
            "in the reference proteome."
        ),
    )
    p.add_argument(
        "--reference-fasta",
        required=True,
        type=Path,
        help=(
            "Reference proteome FASTA. Also serves as the tier-inference "
            "source: any hit whose target accession is not in this FASTA "
            "is treated as non_reference and short-circuits to that "
            "classification."
        ),
    )
    p.add_argument(
        "--novel-fasta",
        required=True,
        type=Path,
        help="Novel proteome FASTA (transcript-derived novel proteins).",
    )
    p.add_argument(
        "--annotation-fasta",
        type=Path,
        default=None,
        action="append",
        help=(
            "Optional additional FASTA(s) scanned for gene-symbol "
            "annotations to populate the `gene` column. Recognises "
            "bracketed `[gene=SYMBOL]` tokens (constellation "
            "combined.fasta convention), bare `gene=SYMBOL` tokens "
            "(cartographer convention), and `GN=SYMBOL` (UniProt / "
            "SwissProt). Pass once per file; can be repeated."
        ),
    )
    p.add_argument(
        "--protein-counts",
        type=Path,
        default=None,
        help=(
            "Optional path to counts_tpm.parquet (PROTEIN_COUNT_TABLE "
            "schema: protein_id, sample_name, tpm, ...). When supplied, "
            "the classifier picks the most-abundant `protein_id` "
            "(highest mean-total TPM across samples) as the "
            "representative when multiple novel proteins produce the "
            "same peptide with the same classification."
        ),
    )
    _add_output_dir_arg(p)
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip the classification if <output-dir>/_SUCCESS exists",
    )
    p.add_argument(
        "--enzyme",
        default="Trypsin",
        help="enzyme for tryptic digestion (default Trypsin)",
    )
    p.add_argument(
        "--max-missed-cleavages",
        type=int,
        default=1,
        help="max missed cleavages in tryptic digest (default 1)",
    )
    p.add_argument(
        "--min-peptide-length",
        type=int,
        default=7,
        help="min peptide length kept in the digest (default 7)",
    )
    p.add_argument(
        "--max-peptide-length",
        type=int,
        default=50,
        help="max peptide length kept in the digest (default 50)",
    )
    p.add_argument(
        "--collision-filter",
        action="store_true",
        help=(
            "Apply cartographer's DIA collision filter: drop detected "
            "modseqs that lose co-elution clusters in the .elib's "
            "entries table. Requires --dia and a .elib --search-results. "
            "Opt-in pending v6.5.15-sweep validation of whether the "
            "newer search internalises this filter."
        ),
    )
    p.add_argument(
        "--dia",
        type=Path,
        default=None,
        help=(
            "Path to combined.dia (the merged-fractions DIA file). "
            "Required when --collision-filter is set; supplies the GPF "
            "isolation windows."
        ),
    )
    p.add_argument(
        "--collision-rt-threshold-s",
        type=float,
        default=5.0,
        help="--collision-filter: max |ΔRT| (s) for a pair (default 5.0)",
    )
    p.add_argument(
        "--collision-frag-ppm-tol",
        type=float,
        default=20.0,
        help="--collision-filter: fragment m/z tolerance ppm (default 20.0)",
    )
    p.add_argument(
        "--collision-min-shared-ions",
        type=int,
        default=4,
        help="--collision-filter: min shared ions to flag (default 4)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress progress messages on stderr",
    )
    p.set_defaults(func=_cmd_massspec_classify_novel_peptides)


def _cmd_massspec_classify_novel_peptides(args: argparse.Namespace) -> int:
    """Classify detected novel peptides against a reference-vs-novel
    mmseqs2 alignment and write a NOVEL_PEPTIDE_TABLE-shaped bundle.
    """
    import json as _json
    import sys as _sys

    from constellation import __version__ as constellation_version
    from constellation.core.io.schemas import (
        ALIGNMENT_HIT_TABLE,
        read_mmseqs_tab,
    )
    from constellation.massspec.search import (
        apply_collision_filter,
        build_gene_map_from_fasta_headers,
        classify_novel_peptides,
        read_fasta_proteins,
        save_novel_peptides,
    )

    search_results = Path(args.search_results).resolve()
    library_path = Path(args.library).resolve()
    alignment_hits = Path(args.alignment_hits).resolve()
    reference_fasta = Path(args.reference_fasta).resolve()
    novel_fasta = Path(args.novel_fasta).resolve()
    output_dir = Path(args.output_dir).resolve()
    dia_path: Path | None = (
        Path(args.dia).resolve() if args.dia is not None else None
    )

    if args.collision_filter:
        if dia_path is None:
            print(
                "error: --collision-filter requires --dia (path to combined.dia)",
                file=_sys.stderr,
            )
            return 1
        if search_results.suffix.lower() != ".elib":
            print(
                "error: --collision-filter requires a .elib --search-results "
                "(observed fragment m/z lives in the elib entries table)",
                file=_sys.stderr,
            )
            return 1

    for label, p in [
        ("--search-results", search_results),
        ("--library", library_path),
        ("--alignment-hits", alignment_hits),
        ("--reference-fasta", reference_fasta),
        ("--novel-fasta", novel_fasta),
    ]:
        if not p.exists():
            print(f"error: {label} not found: {p}", file=_sys.stderr)
            return 1
    if dia_path is not None and not dia_path.exists():
        print(f"error: --dia not found: {dia_path}", file=_sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
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

    # 1. Load detected peptides
    if not args.no_progress:
        print(
            f"[classify-novel-peptides] loading detected peptides from "
            f"{search_results} ...",
            file=_sys.stderr,
        )
    detected_peptides = _load_detected_peptides(search_results, library_path)

    # 1b. Optional: drop collision-loser modseqs before classification.
    collision_metadata: dict | None = None
    if args.collision_filter:
        if not args.no_progress:
            print(
                f"[classify-novel-peptides] running collision filter "
                f"(rt<{args.collision_rt_threshold_s}s, "
                f"{args.collision_frag_ppm_tol} ppm, "
                f">={args.collision_min_shared_ions} shared ions) "
                f"on {search_results} + {dia_path} ...",
                file=_sys.stderr,
            )
        losers, collision_metadata = apply_collision_filter(
            elib_path=search_results,
            dia_path=dia_path,
            rt_threshold_s=args.collision_rt_threshold_s,
            frag_ppm_tol=args.collision_frag_ppm_tol,
            min_shared_ions=args.collision_min_shared_ions,
            return_metadata=True,
        )
        if losers and detected_peptides.num_rows > 0:
            detected_peptides = _drop_collision_losers(detected_peptides, losers)
        collision_metadata["n_losers"] = len(losers)
        if not args.no_progress:
            print(
                f"[classify-novel-peptides] collision filter dropped "
                f"{len(losers):,} modseqs across "
                f"{len(collision_metadata['clusters']):,} clusters; "
                f"{detected_peptides.num_rows:,} detected peptides remain",
                file=_sys.stderr,
            )

    # 2. Load alignments
    if not args.no_progress:
        print(
            f"[classify-novel-peptides] loading alignments from "
            f"{alignment_hits} ...",
            file=_sys.stderr,
        )
    if alignment_hits.is_dir():
        # ParquetDir bundle
        import pyarrow.parquet as pq
        candidate = alignment_hits / "alignment_hits.parquet"
        if not candidate.is_file():
            candidate = alignment_hits / "alignments.parquet"
        if not candidate.is_file():
            print(
                "error: --alignment-hits dir has no recognised parquet file "
                "(looked for alignment_hits.parquet and alignments.parquet)",
                file=_sys.stderr,
            )
            return 1
        alignments = pq.read_table(candidate)
    else:
        alignments = read_mmseqs_tab(alignment_hits)
    # Ensure schema match before passing to the classifier
    from constellation.core.io.schemas import cast_to_schema
    alignments = cast_to_schema(alignments, ALIGNMENT_HIT_TABLE)

    # 3. Load proteomes
    if not args.no_progress:
        print(
            "[classify-novel-peptides] loading reference + novel proteomes ...",
            file=_sys.stderr,
        )
    reference_proteins = read_fasta_proteins(reference_fasta)
    novel_proteins = read_fasta_proteins(novel_fasta)
    # The bare reference/novel FASTAs typically don't carry gene tags.
    # An optional --annotation-fasta (combined.fasta with [gene=...] tags
    # or SwissProt with GN=...) is the actual source for the gene map.
    annotation_fastas = args.annotation_fasta or []
    gene_map_inputs = [reference_fasta, novel_fasta, *annotation_fastas]
    gene_map = build_gene_map_from_fasta_headers(gene_map_inputs)
    if not args.no_progress:
        print(
            f"[classify-novel-peptides] gene_map: "
            f"{len(gene_map):,} accessions tagged "
            f"from {len(gene_map_inputs)} FASTA(s)",
            file=_sys.stderr,
        )

    # Optional per-protein abundance map (highest-abundance ORF wins
    # the row when multiple novel proteins produce the same peptide).
    protein_abundance: dict[str, float] | None = None
    if args.protein_counts is not None:
        if not args.no_progress:
            print(
                f"[classify-novel-peptides] loading protein abundance from "
                f"{args.protein_counts}",
                file=_sys.stderr,
            )
        import pyarrow.parquet as _pq
        counts_tbl = _pq.read_table(args.protein_counts)
        # PROTEIN_COUNT_TABLE has columns: protein_id, sample_name, count, tpm
        # Aggregate to protein_id -> mean tpm across samples.
        ids = counts_tbl.column("protein_id").to_pylist()
        tpms = counts_tbl.column("tpm").to_pylist()
        from collections import defaultdict as _dd
        sums: dict[str, float] = _dd(float)
        n_obs: dict[str, int] = _dd(int)
        for pid, t in zip(ids, tpms):
            if pid is None or t is None:
                continue
            sums[pid] += float(t)
            n_obs[pid] += 1
        protein_abundance = {
            pid: sums[pid] / n_obs[pid] for pid in sums
        }
        if not args.no_progress:
            print(
                f"[classify-novel-peptides] protein_abundance: "
                f"{len(protein_abundance):,} proteins with mean TPM",
                file=_sys.stderr,
            )

    # 4. Classify
    if not args.no_progress:
        print(
            f"[classify-novel-peptides] classifying "
            f"{detected_peptides.num_rows:,} detected peptides against "
            f"{alignments.num_rows:,} alignment hits "
            f"(novel proteins: {novel_proteins.num_rows:,}; "
            f"reference proteins: {reference_proteins.num_rows:,}) ...",
            file=_sys.stderr,
        )
    result = classify_novel_peptides(
        detected_peptides=detected_peptides,
        alignments=alignments,
        reference_proteins=reference_proteins,
        novel_proteins=novel_proteins,
        gene_map=gene_map,
        protein_abundance=protein_abundance,
        enzyme=args.enzyme,
        max_missed_cleavages=args.max_missed_cleavages,
        min_peptide_length=args.min_peptide_length,
        max_peptide_length=args.max_peptide_length,
    )

    # 5. Write output
    save_novel_peptides(
        result,
        output_dir,
        metadata={
            "constellation_version": constellation_version,
            "search_results": str(search_results),
            "library": str(library_path),
            "alignment_hits": str(alignment_hits),
            "reference_fasta": str(reference_fasta),
            "novel_fasta": str(novel_fasta),
            "enzyme": args.enzyme,
            "max_missed_cleavages": args.max_missed_cleavages,
            "min_peptide_length": args.min_peptide_length,
            "max_peptide_length": args.max_peptide_length,
        },
    )
    if collision_metadata is not None:
        (output_dir / "collision_metadata.json").write_text(
            _json.dumps(collision_metadata, indent=2) + "\n"
        )

    # 6. Run-level manifest
    import os as _os
    import socket
    import platform

    classification_counts: dict[str, int] = {}
    for c in result.column("classification").to_pylist():
        classification_counts[c] = classification_counts.get(c, 0) + 1
    manifest = {
        "constellation_version": constellation_version,
        "subcommand": "massspec classify-novel-peptides",
        "argv": _sys.argv,
        "inputs": {
            "search_results": str(search_results),
            "library": str(library_path),
            "alignment_hits": str(alignment_hits),
            "reference_fasta": str(reference_fasta),
            "novel_fasta": str(novel_fasta),
        },
        "outputs": {
            "novel_peptides_pqdir": str(output_dir),
        },
        "params": {
            "enzyme": args.enzyme,
            "max_missed_cleavages": args.max_missed_cleavages,
            "min_peptide_length": args.min_peptide_length,
            "max_peptide_length": args.max_peptide_length,
            "collision_filter": args.collision_filter,
            "collision_rt_threshold_s": args.collision_rt_threshold_s,
            "collision_frag_ppm_tol": args.collision_frag_ppm_tol,
            "collision_min_shared_ions": args.collision_min_shared_ions,
            "dia": (str(dia_path) if dia_path is not None else None),
        },
        "runtime": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": _sys.version.split()[0],
            "user": _os.environ.get("USER", "unknown"),
        },
        "counts": {
            "novel_peptides": result.num_rows,
            "classifications": classification_counts,
            "collision_losers": (
                collision_metadata["n_losers"]
                if collision_metadata is not None
                else None
            ),
            "collision_clusters": (
                len(collision_metadata["clusters"])
                if collision_metadata is not None
                else None
            ),
        },
    }
    (output_dir / "manifest.json").write_text(_json.dumps(manifest, indent=2) + "\n")
    success_path.write_bytes(b"")

    if not args.no_progress:
        print(
            f"classify-novel-peptides done: {result.num_rows:,} unique novel "
            f"peptides classified → {output_dir}/novel_peptides.parquet",
            file=_sys.stderr,
        )
        for cls in sorted(
            classification_counts, key=lambda c: classification_counts[c],
            reverse=True,
        ):
            print(f"  {cls:30s}  {classification_counts[cls]:>6,}", file=_sys.stderr)
    return 0


def _load_detected_peptides(
    search_results: Path, library_path: Path
) -> "pa.Table":  # noqa: F821 — pa import-on-demand below
    """Load detected peptides as a (peptide_sequence, peptide_id?,
    modified_sequence?) Arrow table.

    Two supported inputs:
      * ``.elib`` SQLite — pull the ``PeptideSeq`` column from the
        ``entries`` table (the cartographer-compatible path).
      * ParquetDir Search bundle — join ``peptide_scores.parquet`` to
        ``<library>/peptides.parquet`` on ``peptide_id`` to recover the
        canonical sequence.
    """
    import pyarrow as pa
    import sqlite3

    if search_results.suffix in (".elib", ".dlib"):
        con = sqlite3.connect(str(search_results))
        try:
            rows = con.execute(
                "SELECT DISTINCT PeptideSeq, PeptideModSeq "
                "FROM entries WHERE PeptideSeq IS NOT NULL"
            ).fetchall()
        finally:
            con.close()
        return pa.table(
            {
                "peptide_sequence": pa.array(
                    [r[0] for r in rows], type=pa.string()
                ),
                "modified_sequence": pa.array(
                    [r[1] for r in rows], type=pa.string()
                ),
            }
        )

    # ParquetDir Search bundle: search-results dir + a library to join on
    import pyarrow.parquet as pq

    from constellation.massspec.library.io import load_library

    library = load_library(library_path, format="parquet_dir")
    peptide_id_to_seq = {
        pid: (seq, modseq)
        for pid, seq, modseq in zip(
            library.peptides.column("peptide_id").to_pylist(),
            library.peptides.column("sequence").to_pylist(),
            library.peptides.column("modified_sequence").to_pylist(),
            strict=True,
        )
    }
    peptide_scores = pq.read_table(search_results / "peptide_scores.parquet")
    seen: set[int] = set()
    peptide_ids: list[int] = []
    seqs: list[str] = []
    modseqs: list[str | None] = []
    for pid in peptide_scores.column("peptide_id").to_pylist():
        if pid in seen:
            continue
        seen.add(pid)
        entry = peptide_id_to_seq.get(pid)
        if entry is None:
            continue
        seqs.append(entry[0])
        modseqs.append(entry[1])
        peptide_ids.append(pid)
    return pa.table(
        {
            "peptide_id": pa.array(peptide_ids, type=pa.int64()),
            "peptide_sequence": pa.array(seqs, type=pa.string()),
            "modified_sequence": pa.array(modseqs, type=pa.string()),
        }
    )


def _drop_collision_losers(
    detected_peptides: "pa.Table",  # noqa: F821 — pa import-on-demand
    losers: set[str],
) -> "pa.Table":  # noqa: F821
    """Filter ``detected_peptides`` to drop rows whose
    ``modified_sequence`` is in the collision-loser set.

    ``losers`` are PeptideModSeq strings in EncyclopeDIA notation
    (``[+N.NNN]`` mass-delta form); ``_load_detected_peptides`` writes
    the same notation into ``modified_sequence`` when the input is an
    ``.elib`` — the only path on which ``--collision-filter`` is
    permitted.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if "modified_sequence" not in detected_peptides.column_names:
        return detected_peptides
    keep_mask = pc.invert(
        pc.is_in(
            detected_peptides.column("modified_sequence"),
            value_set=pa.array(sorted(losers), type=pa.string()),
        )
    )
    return detected_peptides.filter(keep_mask)
