"""``constellation massspec counter {calibrate, estimate}``: ion-count estimation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from constellation.massspec.cli.chromatogram import _DEFAULT_N_ISOTOPES


# ──────────────────────────────────────────────────────────────────────
# counter — panel-shaped ion-count estimation + per-acquisition calibration
# ──────────────────────────────────────────────────────────────────────


def _build_counter_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "counter",
        help=(
            "Counter ion-count estimation. `calibrate` fits a per-acquisition "
            "GlobalCalibration from spiked calibrants; `estimate` runs the "
            "panel-shaped per-seed estimate (one panel per target, parallel "
            "across seeds) producing N_total + credible intervals."
        ),
    )
    csubs = p.add_subparsers(dest="counter_subcommand", required=True)
    _build_counter_calibrate_parser(csubs)
    _build_counter_estimate_parser(csubs)


def _add_counter_common_inputs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trace", type=Path, required=True,
                   help="XIC_TRACE_TABLE .parquet (MS1, ideally an all_in_window extraction).")
    p.add_argument("--scan-metadata", type=Path, required=True,
                   help="SCAN_METADATA_TABLE .parquet (the scan axis; level-1 rows are used).")
    p.add_argument("--rt-window", type=float, default=60.0,
                   help="Half-width (s) of the scan-axis window around each target's rt_center.")
    p.add_argument("--n-isotopes", type=int, default=_DEFAULT_N_ISOTOPES)
    p.add_argument("--acquisition-id", type=int, default=0)
    p.add_argument("--no-progress", action="store_true")


def _build_counter_calibrate_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "calibrate",
        help="Fit a per-acquisition GlobalCalibration (+ peptide params) from calibrants.",
    )
    _add_counter_common_inputs(p)
    p.add_argument("--calibrants", type=Path, required=True,
                   help="XIC_TARGET_TABLE .parquet of calibrant peptides (target_id, "
                        "modified_sequence, precursor_charge, rt_center).")
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("--gain", action="store_true",
                   help="Co-fit the gain alpha(z) (weakly identified; off by default).")
    p.set_defaults(func=_cmd_counter_calibrate)


def _build_counter_estimate_parser(subs: argparse._SubParsersAction) -> None:
    p = subs.add_parser(
        "estimate",
        help="Panel-shaped per-seed ion-count estimate (parallel across seeds).",
    )
    _add_counter_common_inputs(p)
    p.add_argument("--targets", type=Path, required=True,
                   help="XIC_TARGET_TABLE .parquet of seed peptides to quantify.")
    p.add_argument("--calibration", type=Path, required=True,
                   help="COUNTER_GLOBAL_CALIBRATION_TABLE .parquet (from `counter calibrate`).")
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1,
                   help="Process-pool size over seeds (each seed is an independent panel).")
    p.add_argument("--neighborhood-ppm", type=float, default=100.0,
                   help="Per-channel m/z neighborhood retained as discovery candidate nodes.")
    p.add_argument("--detect-threshold", type=float, default=8.0)
    p.add_argument("--max-candidates", type=int, default=1,
                   help="Max same-grid interferers per panel (default 1; >1 is "
                        "weakly identifiable and overfits until the parsimony pass lands).")
    p.add_argument("--no-background", dest="background", action="store_false",
                   help="Disable the additive background channel.")
    p.add_argument("--collide-ppm", type=float, default=20.0,
                   help="m/z tolerance for grouping targets into channel-overlap "
                        "components that are co-fit jointly (default 20). Auto-clamped to "
                        "the trace's recorded XIC extraction tolerance — a member beyond "
                        "that tolerance has no extracted signal on the reference grid.")
    p.add_argument("--extraction-tolerance-ppm", type=float, default=None,
                   help="Override the trace's recorded XIC extraction tolerance (ppm) used "
                        "to clamp --collide-ppm. Default: read from the trace metadata.")
    p.add_argument("--rt-overlap-s", type=float, default=None,
                   help="RT-center span (s) within which component members are co-fit as "
                        "one unit (default + max: --rt-window; larger values are clamped, "
                        "since a unit can't span beyond the reference observation window).")
    p.add_argument("--max-component-size", type=int, default=8,
                   help="Co-fit units larger than this fall back to independent "
                        "single-target fits (guards against a dense isobaric region "
                        "chaining into one pool-stalling mega-panel).")
    p.add_argument("--emit-attribution", action="store_true",
                   help="Also write the sparse ion→progenitor soft-attribution map "
                        "(COUNTER_PEAK_ATTRIBUTION_TABLE) — the 'what's left' / interference "
                        "bookkeeping — alongside the count table.")
    p.set_defaults(func=_cmd_counter_estimate)


def _counter_targets(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    need = ("target_id", "modified_sequence", "precursor_charge", "rt_center")
    missing = [c for c in need if c not in t.column_names]
    if missing:
        raise SystemExit(f"error: targets missing columns {missing}")
    rows = t.select(list(need)).to_pylist()
    tids = [r["target_id"] for r in rows]
    if len(set(tids)) != len(tids):
        from collections import Counter

        dups = sorted(tid for tid, n in Counter(tids).items() if n > 1)
        raise SystemExit(
            f"error: targets has duplicate target_id(s) (must be unique): {dups[:10]}"
        )
    return rows


# -- estimate: spawn worker pool over seeds (each an independent panel) --

_COUNTER_CTX: dict = {}


def _counter_worker_init(trace_path: str, scan_meta_path: str, calibration_path: str, opts: dict) -> None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from constellation.massspec.counter import calibration_from_table
    from constellation.massspec.quant.chromatogram import load_xic

    # load_xic handles both a `chromatogram extract` bundle directory and a bare parquet.
    trace = load_xic(trace_path)
    sm = pq.read_table(scan_meta_path)
    if "level" in sm.column_names:
        sm = sm.filter(pc.equal(sm.column("level"), 1))
    _COUNTER_CTX.update(
        trace=trace,
        ms1=sm.select(["scan", "rt", "iit"]),
        cal=calibration_from_table(pq.read_table(calibration_path)),
        opts=opts,
    )


def _counter_worker_estimate(target: dict) -> tuple[list[dict], Any]:
    """Singleton per-target fit. Returns (records, attribution) — the uniform worker
    contract; `attribution` is a COUNTER_PEAK_ATTRIBUTION_TABLE (or None unless
    `--emit-attribution`)."""
    import pyarrow.compute as pc

    from constellation.core.sequence.proforma import Peptidoform
    from constellation.massspec.counter import (
        DiscoverConfig,
        Panel,
        Progenitor,
        estimate_panel,
        observation_for_region,
        panel_attribution_table,
    )

    opts, cal, trace, ms1 = (_COUNTER_CTX[k] for k in ("opts", "cal", "trace", "ms1"))
    tid = int(target["target_id"])
    modseq = target["modified_sequence"]
    base = {"acquisition_id": opts["acquisition_id"], "target_id": tid,
            "modified_sequence": modseq, "precursor_charge": target["precursor_charge"]}
    try:
        # Coerce inside the try: a null charge / rt_center / modseq is one bad seed,
        # not a run-aborting crash.
        z = int(target["precursor_charge"])
        rtc = float(target["rt_center"])
        prog = Progenitor.for_peptide(Peptidoform(sequence=modseq), [z], cal,
                                      n_isotopes=opts["n_isotopes"])
        rt = ms1.column("rt")
        win = ms1.filter(pc.less_equal(pc.abs(pc.subtract(rt, rtc)), opts["rt_window"]))
        obs, region = observation_for_region(trace, win, prog, target_id=tid,
                                             neighborhood_ppm=opts["neighborhood_ppm"])
        if int(obs.mask.sum()) == 0:
            return [{**base, "status": "no_signal"}], None
        panel = Panel([prog], cal, background=opts["background"])
        cfg = DiscoverConfig(detect_threshold=opts["detect_threshold"],
                             max_candidates=opts["max_candidates"])
        res = estimate_panel(panel, obs, region, config=cfg, rt_prior_ms=rtc * 1000.0,
                             inference="map")
        attr = (
            panel_attribution_table(panel, obs, acquisition_id=opts["acquisition_id"], target_id=tid)
            if opts.get("emit_attribution") else None
        )
        return [{**base, **res, "status": "ok"}], attr
    except Exception as exc:  # noqa: BLE001 — one bad seed shouldn't kill the run
        return [{**base, "status": f"error:{type(exc).__name__}"}], None


def _counter_cofit_units(targets: list[dict], opts: dict) -> tuple[list[list[dict]], int]:
    """Partition targets into co-fit units: channel-overlap components (shared m/z)
    refined to co-eluting sub-clusters (shared RT), with oversized units split back
    to singletons (the mega-component pool-stall guard). Each unit is a list of
    target dicts sorted by `target_id` ([0] is the reference grid). Targets without a
    `modified_sequence` (no envelope) stay singletons. Returns (units, n_capped)."""
    from constellation.core.sequence.proforma import Peptidoform
    from constellation.massspec.counter import (
        TheoreticalCandidateIndex,
        channel_overlap_components,
        refine_components_by_rt,
        restrict_to_reference_star,
    )

    by_tid = {int(t["target_id"]): t for t in targets}
    entries: list[tuple] = []
    rt_centers: dict[int, float] = {}
    no_envelope: list[int] = []
    for t in targets:
        tid = int(t["target_id"])
        if t.get("rt_center") is not None:
            rt_centers[tid] = float(t["rt_center"])
        # A target needs a modseq AND a charge to form a theoretical envelope; any
        # missing/malformed field → keep it a singleton (the worker reports the error,
        # rather than the parent partitioner crashing the whole run).
        try:
            if t["modified_sequence"] and t.get("precursor_charge") is not None:
                entries.append(
                    (tid, Peptidoform(sequence=t["modified_sequence"]), [int(t["precursor_charge"])])
                )
            else:
                no_envelope.append(tid)
        except Exception:  # noqa: BLE001 — malformed modseq/charge → singleton fallback
            no_envelope.append(tid)

    unit_tids: list = []
    if entries:
        index = TheoreticalCandidateIndex.from_peptides(entries, n_isotopes=opts["n_isotopes"])
        comps = channel_overlap_components(index, collide_ppm=opts["collide_ppm"])
        unit_tids = refine_components_by_rt(comps, rt_centers, rt_overlap_s=opts["rt_overlap_s"])
        # Reference-star restriction: a transitive m/z component co-fit on the
        # reference grid would under-score members that only indirectly overlap (their
        # peaks were never extracted onto the reference grid). Keep only members
        # DIRECTLY overlapping each star's reference; the rest fall into their own
        # stars (singletons if they overlap nobody).
        unit_tids = restrict_to_reference_star(unit_tids, index, collide_ppm=opts["collide_ppm"])
    unit_tids += [frozenset({tid}) for tid in no_envelope]

    cap = int(opts["max_component_size"])
    units: list[list[dict]] = []
    n_capped = 0
    for u in unit_tids:
        members = sorted(u)
        if len(members) > cap:
            n_capped += 1
            units.extend([[by_tid[tid]] for tid in members])  # fall back to singletons
        else:
            units.append([by_tid[tid] for tid in members])
    units.sort(key=lambda u: int(u[0]["target_id"]))  # deterministic work-item order
    return units, n_capped


def _counter_worker_component(unit: list[dict]) -> tuple[list[dict], Any]:
    """Joint co-fit of a multi-member unit — one panel over the reference member's
    grid (`unit[0]`), each member scored at its own mass defect. Returns (records,
    attribution) per the uniform worker contract (one record per member)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    from constellation.core.sequence.proforma import Peptidoform
    from constellation.massspec.counter import (
        DiscoverConfig,
        Progenitor,
        estimate_component,
        observation_for_region,
        panel_attribution_table,
    )

    opts, cal, trace, ms1 = (_COUNTER_CTX[k] for k in ("opts", "cal", "trace", "ms1"))
    emit = bool(opts.get("emit_attribution"))
    members = sorted(unit, key=lambda t: int(t["target_id"]))  # reference = min target_id
    bases = [
        {
            "acquisition_id": opts["acquisition_id"],
            "target_id": int(t["target_id"]),
            "modified_sequence": t["modified_sequence"],
            "precursor_charge": int(t["precursor_charge"]),
        }
        for t in members
    ]
    try:
        progs = [
            Progenitor.for_peptide(
                Peptidoform(sequence=t["modified_sequence"]),
                [int(t["precursor_charge"])],
                cal,
                n_isotopes=opts["n_isotopes"],
            )
            for t in members
        ]
        rtc_ref = float(members[0]["rt_center"])
        rt = ms1.column("rt")
        win = ms1.filter(pc.less_equal(pc.abs(pc.subtract(rt, rtc_ref)), opts["rt_window"]))
        obs, _region = observation_for_region(
            trace, win, progs[0], target_id=int(members[0]["target_id"]),
            neighborhood_ppm=opts["neighborhood_ppm"],
        )
        if int(obs.mask.sum()) == 0:
            # faint/absent reference grid → don't blanket-no_signal the co-members
            # (the reference is min target_id, arbitrary w.r.t. abundance); fit each
            # member on its OWN grid as an independent singleton.
            subs = [_counter_worker_estimate(t) for t in members]
            records = [r for s in subs for r in s[0]]
            attrs = [s[1] for s in subs if s[1] is not None]
            return records, (pa.concat_tables(attrs) if attrs else None)
        cfg = DiscoverConfig(
            detect_threshold=opts["detect_threshold"], max_candidates=opts["max_candidates"]
        )
        out = estimate_component(
            progs, obs, config=cfg, background=opts["background"],
            rt_priors_ms=[float(t["rt_center"]) * 1000.0 for t in members],
            return_panel=emit,
        )
        if emit:
            results, panel = out
            attr = panel_attribution_table(
                panel, obs, acquisition_id=opts["acquisition_id"],
                target_id=int(members[0]["target_id"]),
                # the panel's progenitors ARE the members in order (a component
                # does no discovery), so stamp each row with its member's real
                # target_id rather than flattening co-members into anonymous
                # interferers under the reference's id.
                progenitor_target_ids=[int(m["target_id"]) for m in members],
            )
        else:
            results, attr = out, None
        return [{**b, **r, "status": "ok"} for b, r in zip(bases, results)], attr
    except Exception as exc:  # noqa: BLE001 — one bad component shouldn't kill the run
        return [{**b, "status": f"error:{type(exc).__name__}"} for b in bases], None


def _counter_worker_unit(unit: list[dict]) -> tuple[list[dict], Any]:
    """Dispatch a co-fit unit: a singleton → the per-target discovery path; a
    multi-member unit → the joint component co-fit. Both return (records, attribution)."""
    if len(unit) == 1:
        return _counter_worker_estimate(unit[0])
    return _counter_worker_component(unit)


def _resolve_collide_ppm(args: argparse.Namespace) -> float:
    """Clamp --collide-ppm to the XIC extraction tolerance: a member beyond the
    tolerance the trace was extracted at has NO signal on the reference grid, so
    grouping it would silently under-score it. The tolerance is read from the trace's
    schema metadata (stamped by `chromatogram extract`); --extraction-tolerance-ppm
    overrides; absent + no override → warn and leave --collide-ppm as-is."""
    import pyarrow.parquet as pq

    extraction_tol = args.extraction_tolerance_ppm
    if extraction_tol is None:
        # --trace may be a `chromatogram extract` BUNDLE DIRECTORY (xic_trace.parquet +
        # manifest.json) or a bare parquet; read the schema from the right place.
        schema_path = (
            args.trace / "xic_trace.parquet" if args.trace.is_dir() else args.trace
        )
        try:
            meta = pq.read_schema(schema_path).metadata or {}
        except Exception:  # noqa: BLE001 — unreadable footer → treat as unrecorded
            meta = {}
        tol_b = meta.get(b"x.massspec.extraction_tolerance")
        if tol_b is not None and meta.get(b"x.massspec.extraction_tolerance_unit") == b"ppm":
            extraction_tol = float(tol_b.decode("utf-8"))
    if extraction_tol is None:
        if not args.no_progress:
            print(
                f"warning: trace records no ppm extraction tolerance; --collide-ppm "
                f"{args.collide_ppm} not validated (pass --extraction-tolerance-ppm)",
                file=sys.stderr,
            )
        return float(args.collide_ppm)
    if args.collide_ppm > extraction_tol:
        if not args.no_progress:
            print(
                f"warning: --collide-ppm {args.collide_ppm} > extraction tolerance "
                f"{extraction_tol} ppm; clamping (a member beyond it has no extracted "
                "signal on the reference grid)",
                file=sys.stderr,
            )
        return float(extraction_tol)
    return float(args.collide_ppm)


def _cmd_counter_estimate(args: argparse.Namespace) -> int:
    import multiprocessing as mp


    from constellation.massspec.counter import CounterResult, counter_n_table, save_counter

    for path in (args.trace, args.scan_metadata, args.calibration, args.targets):
        if not path.exists():
            print(f"error: not found: {path}", file=sys.stderr)
            return 2
    success = args.output_dir / "_SUCCESS"
    if success.exists():
        print(f"error: --output-dir already complete ({success} exists)", file=sys.stderr)
        return 1

    targets = _counter_targets(args.targets)
    rt_overlap_s = args.rt_overlap_s if args.rt_overlap_s is not None else args.rt_window
    if rt_overlap_s > args.rt_window:
        # a unit's RT span can't exceed the reference's ± rt_window obs window, or a
        # co-member's elution falls outside the observation entirely.
        if not args.no_progress:
            print(
                f"warning: --rt-overlap-s {rt_overlap_s} > --rt-window {args.rt_window}; "
                f"clamping to {args.rt_window}",
                file=sys.stderr,
            )
        rt_overlap_s = args.rt_window
    collide_ppm = _resolve_collide_ppm(args)
    opts = dict(
        acquisition_id=args.acquisition_id, n_isotopes=args.n_isotopes,
        rt_window=args.rt_window, neighborhood_ppm=args.neighborhood_ppm,
        detect_threshold=args.detect_threshold, max_candidates=args.max_candidates,
        background=args.background, collide_ppm=collide_ppm,
        rt_overlap_s=rt_overlap_s, max_component_size=args.max_component_size,
        emit_attribution=args.emit_attribution,
    )
    # Partition into co-fit units in the parent (cheap, data-independent): m/z-overlap
    # components × RT co-elution. Singletons stay on the per-target discovery path; a
    # multi-member unit is one joint panel. The worker maps over UNITS, not targets.
    units, n_capped = _counter_cofit_units(targets, opts)
    initargs = (str(args.trace), str(args.scan_metadata), str(args.calibration), opts)
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_counter_worker_init, initargs=initargs) as pool:
            grouped = pool.map(_counter_worker_unit, units)
    else:
        _counter_worker_init(*initargs)
        try:
            grouped = [_counter_worker_unit(u) for u in units]
        finally:
            _COUNTER_CTX.clear()  # don't leak this run's trace/cal into a later in-process call
    records = [r for group, _attr in grouped for r in group]
    attr_tables = [attr for _group, attr in grouped if attr is not None]

    ok = [r for r in records if r.get("status") == "ok"]
    skipped = len(records) - len(ok)
    n_multi = sum(1 for u in units if len(u) > 1)
    peak_attribution = None
    if attr_tables:
        import pyarrow as pa

        peak_attribution = pa.concat_tables(attr_tables)
    elif args.emit_attribution:
        # --emit-attribution was requested but no unit produced rows (every target
        # no_signal / errored). Write the schema-correct EMPTY table so the requested
        # output file always exists rather than being silently omitted.
        from constellation.massspec.counter import COUNTER_PEAK_ATTRIBUTION_TABLE

        peak_attribution = COUNTER_PEAK_ATTRIBUTION_TABLE.empty_table()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_counter(
        CounterResult(counter_n=counter_n_table(ok), peak_attribution=peak_attribution),
        args.output_dir,
    )
    success.touch()
    if not args.no_progress:
        extra = f", {n_multi} co-fit components" if n_multi else ""
        extra += f", {n_capped} oversized split" if n_capped else ""
        if peak_attribution is not None:
            extra += f", {peak_attribution.num_rows} attribution rows"
        print(f"counter estimate: {len(ok)} estimated, {skipped} skipped{extra} → {args.output_dir}",
              file=sys.stderr)
    return 0


def _cmd_counter_calibrate(args: argparse.Namespace) -> int:

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from constellation.core.sequence.proforma import Peptidoform
    from constellation.massspec.counter import (
        GlobalCalibration,
        Progenitor,
        StagedCalibration,
        calibration_to_table,
        observation_for_progenitor,
        peptide_params_to_table,
    )

    for path in (args.trace, args.scan_metadata, args.calibrants):
        if not path.exists():
            print(f"error: not found: {path}", file=sys.stderr)
            return 2
    success = args.output_dir / "_SUCCESS"
    if success.exists():
        print(f"error: --output-dir already complete ({success} exists)", file=sys.stderr)
        return 1

    trace = pq.read_table(args.trace)
    sm = pq.read_table(args.scan_metadata)
    if "level" in sm.column_names:
        sm = sm.filter(pc.equal(sm.column("level"), 1))
    ms1 = sm.select(["scan", "rt", "iit"])
    calibrants = _counter_targets(args.calibrants)

    cal = GlobalCalibration(n_isotopes=args.n_isotopes, charges=(1, 2, 3, 4))
    progs, obss, tids, mods = [], [], [], []
    rt = ms1.column("rt")
    for c in calibrants:
        tid, modseq, z, rtc = (int(c["target_id"]), c["modified_sequence"],
                               int(c["precursor_charge"]), float(c["rt_center"]))
        prog = Progenitor.for_peptide(Peptidoform(sequence=modseq), [z], cal, n_isotopes=args.n_isotopes)
        win = ms1.filter(pc.less_equal(pc.abs(pc.subtract(rt, rtc)), args.rt_window))
        obs = observation_for_progenitor(prog, trace, win, target_id=tid)
        if int(obs.mask.sum()) == 0:
            continue
        progs.append(prog)
        obss.append(obs)
        tids.append(tid)
        mods.append(modseq)
    if not progs:
        print("error: no calibrant yielded signal", file=sys.stderr)
        return 1

    from constellation.massspec.counter.orchestrate import _DEFAULT_GLOBAL_PARAMS, _GAIN_PARAM_NAMES

    gp = tuple(_DEFAULT_GLOBAL_PARAMS) + (_GAIN_PARAM_NAMES if args.gain else ())
    StagedCalibration(progs, obss, cal).run(global_params=gp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(calibration_to_table(cal, acquisition_id=args.acquisition_id),
                   args.output_dir / "global_calibration.parquet")
    pq.write_table(peptide_params_to_table(progs, acquisition_id=args.acquisition_id,
                                           target_ids=tids, modified_sequences=mods),
                   args.output_dir / "peptide_params.parquet")
    success.touch()
    if not args.no_progress:
        print(f"counter calibrate: {len(progs)} calibrants → {args.output_dir}", file=sys.stderr)
    return 0
