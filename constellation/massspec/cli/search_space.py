"""``constellation massspec search-space``: FASTA files → every precursor
to look for, with how each one was made (``massspec.library.search_space``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_search_space_parser(subs: argparse._SubParsersAction) -> None:
    from constellation.core.sequence.protein import PROTEASES
    from constellation.massspec.library.decoys import DECOY_METHODS
    from constellation.massspec.library.search_space import DEFAULT_MAX_MZ, DEFAULT_MIN_MZ

    p = subs.add_parser(
        "search-space",
        help=(
            "Build a search space from FASTA files: every protein span, peptidoform and "
            "precursor that the digestion, modification, label and charge rules give, "
            "with decoys, each recorded with how it was made (where in which protein, "
            "how its ends were cut, which modification and why). Writes a library "
            "directory that `massspec chromatogram extract --targets` and "
            "`massspec predict-library --from-library` read. No fragments are stored: "
            "the sweep makes them, so one search space serves any fragmentation settings."
        ),
    )
    p.set_defaults(func=_cmd_search_space)
    p.add_argument("--fasta", type=Path, action="append", required=True, metavar="FILE",
                   help="A protein FASTA (gzip or plain). Repeat for several.")
    p.add_argument("-o", "--output-dir", type=Path, required=True,
                   help="Where the library directory is written.")
    p.add_argument("--resume", action="store_true",
                   help="Keep a finished search space with the same inputs and settings.")
    p.add_argument("--count", action="store_true",
                   help="Print what each stage would hold and write nothing.")

    g = p.add_argument_group("digestion")
    g.add_argument("--enzyme", default="Trypsin", choices=sorted(PROTEASES.ids()),
                   help="Protease rule (default Trypsin). No_enzyme never cuts.")
    g.add_argument("--missed-cleavages", type=int, default=2,
                   help="Cut sites a span may contain (default 2; unspecific ignores it).")
    g.add_argument("--specificity", choices=("specific", "semi", "unspecific"), default="specific",
                   help="Both ends by the rule, at least one, or neither (every span in the "
                        "length bounds). Default specific.")
    g.add_argument("--min-length", type=int, default=6, help="Shortest peptide (default 6).")
    g.add_argument("--max-length", type=int, default=30, help="Longest peptide (default 30).")
    g.add_argument("--no-initiator-met-excision", action="store_true",
                   help="Do not also make the Met-clipped form of each protein-N-terminal span.")
    g.add_argument("--protein-terminus", action="append", default=[], metavar="ACC=START[-END]",
                   help="A mature chain's first (and last) residue, 1-based as UniProt numbers "
                        "them (P02769=25-607): spans there count as protein termini. Repeatable.")

    g = p.add_argument_group("modifications")
    g.add_argument("--fixed", action="append", default=[], metavar="MOD[@SITE,...]",
                   help="A modification on every site that UNIMOD lists it at. MOD is a UNIMOD "
                        "accession or name; SITE a residue, N-term, C-term, Protein N-term, "
                        "Protein C-term, or * (every UNIMOD placement); no @ for UNIMOD's "
                        "shown sites. Repeatable.")
    g.add_argument("--variable", action="append", default=[], metavar="MOD[@SITE,...][:max=N]",
                   help="A modification enumerated over its sites, at most N in one peptide. "
                        "Same spelling as --fixed. Repeatable.")
    g.add_argument("--max-variable-mods", type=int, default=2,
                   help="Variable modifications per peptidoform, all rules together (default 2).")

    g = p.add_argument_group("isotope labels")
    g.add_argument("--label", action="append", default=[], metavar="TARGET=ISOTOPES",
                   help="Label a FASTA's (or one accession's) peptides: 13C,15N, or light for "
                        "the unlabelled forms. Several labels for one target give several forms. "
                        "Repeatable.")

    g = p.add_argument_group("precursors")
    g.add_argument("--min-charge", type=int, default=2, help="Lowest charge (default 2).")
    g.add_argument("--max-charge", type=int, default=4, help="Highest charge (default 4).")
    g.add_argument("--min-mz", type=float, default=DEFAULT_MIN_MZ,
                   help=f"Lowest precursor m/z kept (default {DEFAULT_MIN_MZ:g}).")
    g.add_argument("--max-mz", type=float, default=DEFAULT_MAX_MZ,
                   help=f"Highest precursor m/z kept (default {DEFAULT_MAX_MZ:g}).")

    g = p.add_argument_group("decoys")
    g.add_argument("--decoys", action="append", default=None, metavar="METHOD[:sets=N][:seed=S]",
                   help="A decoy method (" + ", ".join(sorted(DECOY_METHODS)) + "), or none. "
                        "Default reverse-interior (EncyclopeDIA's: first and last residue kept, "
                        "the rest reversed). Each one given makes the next decoy set: the "
                        "first set competes with the targets, the rest are extra (entrapment) "
                        "sets; sets=N makes N of them. Repeatable.")

    g = p.add_argument_group("budget")
    g.add_argument("--max-peptidoforms", type=int, default=5_000_000,
                   help="Refuse (before building any) past this many peptidoforms; 0 for no limit.")
    g.add_argument("--max-precursors", type=int, default=5_000_000,
                   help="Refuse (before building tables) past this many precursors, decoys "
                        "included; 0 for no limit.")


def _decoy_sets(specs: list[str] | None):
    from constellation.massspec.library.decoys import DEFAULT_DECOY_METHOD, DecoySet

    if specs is None:
        return (DecoySet(DEFAULT_DECOY_METHOD, 0, 0),)
    out = []
    for spec in specs:
        method, *opts = spec.strip().split(":")
        if method == "none":
            if len(specs) > 1:
                raise ValueError("--decoys none cannot be combined with other decoy sets")
            return ()
        n, seed = 1, 0
        for opt in opts:
            key, eq, value = opt.partition("=")
            if not eq or key not in ("sets", "seed") or not value.isdigit():
                raise ValueError(f"bad decoy option {opt!r} in {spec!r} (sets=N, seed=S)")
            if key == "sets":
                n = int(value)
            else:
                seed = int(value)
        if n < 1:
            raise ValueError(f"sets must be at least 1 in {spec!r}")
        for _ in range(n):
            out.append(DecoySet(method, len(out), seed))
    return tuple(out)


def _settings(args: argparse.Namespace):
    from constellation.massspec.library.search_space import (
        Digestion,
        ModRule,
        SearchSpaceSettings,
        parse_label,
        parse_protein_terminus,
    )

    return SearchSpaceSettings(
        digestion=Digestion(
            protease=args.enzyme,
            missed_cleavages=args.missed_cleavages,
            specificity=args.specificity,
            min_length=args.min_length,
            max_length=args.max_length,
            excise_initiator_met=not args.no_initiator_met_excision,
        ),
        rules=tuple(ModRule.parse(t, "fixed") for t in args.fixed)
        + tuple(ModRule.parse(t, "variable") for t in args.variable),
        max_variable=args.max_variable_mods,
        labels=tuple(parse_label(t) for t in args.label),
        protein_termini=tuple(parse_protein_terminus(t) for t in args.protein_terminus),
        charges=tuple(range(args.min_charge, args.max_charge + 1)),
        min_mz=args.min_mz,
        max_mz=args.max_mz,
        decoys=_decoy_sets(args.decoys),
        max_peptidoforms=args.max_peptidoforms or None,
        max_precursors=args.max_precursors or None,
    )


def _print_counts(counts: dict) -> None:
    rows = [
        ("proteins", counts["proteins"]),
        ("decoy proteins", counts["decoy_proteins"]),
        ("spans", counts["spans"]),
        ("spans skipped (residue outside the 20)", counts["spans_noncanonical_skipped"]),
        ("peptidoforms (enumerated)", counts.get("peptidoforms_enumerated", "-")),
        ("peptidoforms outside the m/z range", counts.get("peptidoforms_out_of_mz_range", "-")),
        ("peptidoforms", counts["peptidoforms"]),
        ("precursors", counts["precursors"]),
        ("decoy peptidoforms", counts["decoy_peptidoforms"]),
        ("decoy precursors", counts["decoy_precursors"]),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"  {name:<{width}}  {value:>12,}" if isinstance(value, int) else f"  {name:<{width}}  {value:>12}")
    for d in counts.get("decoy_sets", []):
        extra = "".join(f", {k} {d[k]:,}" for k in ("fell_back", "dropped") if k in d)
        print(f"  decoy set {d['set']} ({d['method']}, seed {d['seed']}): {d['made']:,} made{extra}")


def _cmd_search_space(args: argparse.Namespace) -> int:
    from constellation.core.stages import invalidate, is_done, mark_done
    from constellation.massspec.library.io import save_library
    from constellation.massspec.library.search_space import (
        SEARCH_SPACE_METADATA_KEY,
        SearchSpaceBudgetError,
        build_search_space,
        search_space_key,
    )

    for fasta in args.fasta:
        if not fasta.is_file():
            print(f"error: FASTA not found: {fasta}", file=sys.stderr)
            return 2
    try:
        settings = _settings(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.count:
        try:
            counts = build_search_space(args.fasta, settings, count_only=True)
        except (SearchSpaceBudgetError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_counts(counts)
        return 0

    # Not ``__version__``: an editable install never rewrites it, so it let a
    # search space made by older code pass as current (until 2026-11-10).
    key = search_space_key(args.fasta, settings)
    out = args.output_dir
    if is_done(out):
        if is_done(out, key) and args.resume:
            print(f"already built: {out}")
            return 0
        if not args.resume:
            print(f"error: {out} already holds a finished search space; pass --resume to keep it "
                  "(if its inputs and settings are these), or choose another directory", file=sys.stderr)  # fmt: skip
            return 1
    invalidate(out)
    try:
        library = build_search_space(args.fasta, settings)
    except (SearchSpaceBudgetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    save_library(library, out, format="parquet_dir")
    mark_done(out, key)
    counts = library.metadata[SEARCH_SPACE_METADATA_KEY]["counts"]
    print(f"search space → {out}")
    _print_counts(counts)
    return 0


__all__ = ["_build_search_space_parser", "_cmd_search_space"]
