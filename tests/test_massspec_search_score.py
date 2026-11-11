"""Scorer v1 (massspec.search.score): a KL trace over MS2, its elution peaks
found by a peak picker, on planted runs whose fragments' true shares are
known."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

from _massspec_fixtures import planted_run, write_bundle, write_bundle_manifest
from constellation.core.stats.losses import kld
from constellation.massspec.peptide.channels import ChannelSettings
from constellation.massspec.precursor_major import build_precursor_major
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.search.schemas import CANDIDATE_TABLE
from constellation.massspec.search.score import KlScorer, ScorerSettings, score_acquisition
from constellation.massspec.sweep import compile_queries
from constellation.massspec.sweep.engine import SweepSettings, sweep

_TOL = (0.35, "Da")


@dataclass(frozen=True)
class PlantedShares:
    """A test-only fragment abundance source: each precursor's planted
    fragments by m/z (``frag_mz`` → ``frag_ab``, as shares), every other
    candidate ion predicted absent; a precursor not planted (a decoy) gets
    seeded random shares, as a library predicts something for a decoy."""

    planted: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = field(default_factory=dict)
    seed: int = 0
    ranks: bool = True

    def model_losses(self) -> tuple[str, ...]:
        return ()

    def describe(self) -> dict:
        return {"source": "planted"}

    def abundance(self, ions) -> np.ndarray:
        out = np.zeros(ions.mz.size, np.float64)
        q = ions.query_of()
        for i in range(ions.offsets.size - 1):
            at = np.flatnonzero(q == i)
            key = int(ions.key[i])
            if key in self.planted:
                mz, ab = (np.asarray(v) for v in self.planted[key])
                share = ab / ab.sum()
                near = np.abs(ions.mz[at, None] - mz[None, :])
                hit = near.min(-1) < 1e-6
                out[at[hit]] = share[near.argmin(-1)[hit]]
            else:
                g = np.random.default_rng(self.seed + key % 9973)
                r = g.uniform(0.1, 1.0, at.size)
                out[at] = r / r.sum()
        return out


def _setup(tmp: Path, name: str, peptides=None, cycles: int = 60):
    kw = dict(acquisition="dia", cycles=cycles, window_width=100.0, noise_peaks=60)
    if peptides is not None:
        kw["peptides"] = peptides
    run = planted_run(**kw)
    # Shot noise: a peak's intensity is a count of ions times a gain, so its
    # fragments' ratios scatter more where it is weaker (the planted run's
    # are exact at every intensity, so a KL is 0 across a whole elution).
    g = np.random.default_rng(17)
    gain = 2e3
    scans = [dict(s_, intensity=gain * g.poisson(np.asarray(s_["intensity"]) / gain).astype(np.float64))
             for s_ in run.scans]  # fmt: skip
    for s_ in scans:
        keep = s_["intensity"] > 0
        s_["mz"], s_["intensity"] = s_["mz"][keep], s_["intensity"][keep]
    bundle = write_bundle_manifest(write_bundle(tmp, name, scans))
    build_precursor_major(bundle)
    planted = {5000 + i: (tuple(p["frag_mz"]), tuple(p["frag_ab"])) for i, p in enumerate(run.planted)}
    rows = run.targets.to_pylist()
    for r in rows:
        r.update(is_decoy=False, rt_center=None)
    decoys = [dict(r, target_id=r["target_id"] + 500, precursor_id=r["precursor_id"] + 500, is_decoy=True, decoy_set=0,
                   modified_sequence=r["modified_sequence"][-2::-1] + r["modified_sequence"][-1])
              for r in rows if "[" not in r["modified_sequence"]]  # fmt: skip
    targets = pa.Table.from_pylist(rows + decoys, schema=XIC_TARGET_TABLE)
    return run, bundle, targets, PlantedShares(planted)


@functools.lru_cache(maxsize=None)
def _cached(root: str):
    return _setup(Path(root), "planted")


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    return _cached(str(tmp_path_factory.mktemp("score")))


def _channels() -> ChannelSettings:
    return ChannelSettings(top_fragments=10, max_fragment_charge=2)


def _score(bundle, targets, shares, **kw):
    settings = kw.pop("settings", ScorerSettings())
    channels = kw.pop("channels", _channels())
    return score_acquisition(bundle, targets, fragment_abundance=shares, tolerance=_TOL, channels=channels,
                             settings=settings, **kw)  # fmt: skip


def test_every_query_gets_rows_of_the_candidate_table_with_how_they_were_made(planted):
    run, bundle, targets, shares = planted
    result = _score(bundle, targets, shares)
    t = result.table("candidates")
    assert t.schema.remove_metadata().equals(CANDIDATE_TABLE.remove_metadata())
    meta = t.schema.metadata
    assert json.loads(meta[b"x.massspec.feature_set"]) == {"name": "kl-trace", "version": 1}
    assert json.loads(meta[b"x.constellation.peak_picker"])["name"] == "peak_picker"
    assert len(meta[b"x.massspec.scorer.key"]) == 32
    chosen = t.filter(t.column("chosen"))
    # One chosen row per query, targets and decoys alike.
    assert sorted(chosen.column("target_id").to_pylist()) == sorted(targets.column("target_id").to_pylist())
    frags = result.table("candidate_fragments")
    assert frags.num_rows > 0 and set(frags.column("target_id").to_pylist()) <= set(chosen.column("target_id").to_pylist())


def test_the_planted_peptides_are_found_at_their_apex_and_outscore_their_decoys(planted):
    run, bundle, targets, shares = planted
    t = _score(bundle, targets, shares).table("candidates")
    chosen = t.filter(t.column("chosen")).to_pylist()
    apex = {1000 + i: p["apex"] for i, p in enumerate(run.planted)}
    found = {r["target_id"]: r for r in chosen if not r["is_decoy"]}
    shared = {"PEPTMIDEK", "PEPTM[Oxidation]IDEK"}  # 6 s apart, most fragments in common
    for i, (target_id, rt) in enumerate(apex.items()):
        r = found[target_id]
        if run.planted[i]["modseq"] in shared:
            # The KL is blind to intensity: where a neighbour shares its ions
            # the cleanest match can be on the peak's own tail.
            assert abs(r["rt_apex"] - rt) < 2.5 * 5.0  # inside its elution
        else:
            assert r["rt_apex"] == pytest.approx(rt, abs=3.5)  # within about a cycle
        assert r["kl_apex"] < 0.3 and r["n_coeluting"] >= 3
    decoys = [r["score"] for r in chosen if r["is_decoy"]]
    targets_ = [found[k]["score"] for k in apex]
    assert min(targets_) > np.nanmedian([d if d is not None else -np.inf for d in decoys])


def test_a_slot_with_too_few_observed_channels_is_no_evidence(planted):
    run, bundle, targets, shares = planted
    strict = ScorerSettings(min_observed=50)  # more than any query has
    t = _score(bundle, targets, shares, settings=strict).table("candidates")
    assert pc.all(pc.equal(t.column("rank"), -1)).as_py()
    assert t.column("score").null_count == t.num_rows
    assert (np.asarray(t.column("n_candidates")) == 0).all()


class _KlProbe:
    """Records each slot's KL from the dense view as the scorer computes it."""

    name = "kl_probe"
    schema = pa.schema([("query", pa.int64()), ("slot", pa.int64()), ("kl", pa.float64()), ("scan", pa.int64())])
    gated = False
    zero_fills = True

    def reduce(self, batch):
        d = batch.dense(2, dtype=torch.float64)
        p = d.abundance.to(torch.float64)
        model = d.channel_valid & torch.isfinite(p) & (p > 0)
        kept = d.acquired & model.unsqueeze(1) & d.slot_valid.unsqueeze(-1)
        o = torch.where(kept, torch.nan_to_num(d.observed()), torch.zeros_like(d.intensity))
        lib = torch.where(model, p, torch.zeros_like(p)).unsqueeze(1).expand_as(o)
        kl = kld(pred=lib, target=o, mask=kept, pseudocount=1e-3, pseudocount_units="probability", reduce="none")
        qi, si = torch.nonzero(d.slot_valid, as_tuple=True)
        return pa.table({"query": d.global_queries[qi.numpy()], "slot": si.numpy(), "kl": kl[qi, si].numpy(),
                         "scan": d.scan[qi, si].numpy()}, schema=self.schema)  # fmt: skip


def test_the_kl_at_an_apex_is_the_slots_kl_of_observed_and_library_shares(planted):
    run, bundle, targets, shares = planted
    plan = compile_queries(targets, levels=[2], channels=_channels(), fragment_abundance=shares)
    result = sweep(bundle, plan, SweepSettings(tolerances={2: _TOL}), reducers=[_KlProbe(), KlScorer()])
    probe = result.table("kl_probe")
    by = {(q, s): k for q, s, k in zip(probe.column("query").to_pylist(), probe.column("scan").to_pylist(),
                                        probe.column("kl").to_pylist())}  # fmt: skip
    t = result.table("candidates")
    rows = t.filter(pc.is_valid(t.column("kl_apex_scan"))).to_pylist()
    assert rows
    for r in rows:
        assert r["kl_apex"] == pytest.approx(by[(r["query"], r["kl_apex_scan"])], abs=1e-9)
        assert r["kl_apex_h0"] == pytest.approx(r["kl_apex"], abs=1e-12)  # no sliding sum by default


def test_an_interferer_with_the_same_ions_in_other_ratios_loses_to_the_planted_apex(tmp_path):
    """The same peptide eluting twice with its fragments in different
    ratios: only the first is the library's (the planted shares of the
    first elution), so the scorer must choose it."""
    peptides = (("LVNELTEFAK", 2, 60.0), ("LVNELTEFAK", 2, 120.0))
    run, bundle, targets, shares = _setup(tmp_path, "interfered", peptides=peptides)
    first = targets.filter(pc.equal(targets.column("target_id"), 1000))
    t = _score(bundle, first, PlantedShares({5000: shares.planted[5000]})).table("candidates")
    chosen = t.filter(t.column("chosen")).to_pylist()[0]
    assert chosen["rt_apex"] == pytest.approx(60.0, abs=3.5)
    assert chosen["n_candidates"] >= 2  # the other elution is a candidate too, and loses


def test_the_chosen_rows_do_not_depend_on_how_the_sweep_is_cut(planted, tmp_path):
    run, bundle, targets, shares = planted
    keys = ["target_id", "rank"]
    base = _score(bundle, targets, shares).table("candidates").sort_by([(k, "ascending") for k in keys])
    small = ScorerSettings(max_cells=2_000)
    for kw in (dict(settings=small), dict(layout="scan_major"), dict(workers=2, work_dir=tmp_path / "w")):
        got = _score(bundle, targets, shares, **kw).table("candidates").sort_by([(k, "ascending") for k in keys])
        assert got.column("target_id").equals(base.column("target_id")), kw
        assert got.column("chosen").equals(base.column("chosen")), kw
        assert got.column("apex_scan").equals(base.column("apex_scan")), kw
        a, b = np.asarray(got.column("score").to_pylist(), float), np.asarray(base.column("score").to_pylist(), float)
        np.testing.assert_allclose(a, b, rtol=1e-6, equal_nan=True)


# ── what the scorer cannot score still gets its row (Codex review) ────

#: The candidate table's columns measured per candidate: null on a query's
#: no-candidate row.
_MEASURES = ("score", "detector_score", "kl_apex", "kl_apex_h0", "kl_apex_smoothed", "kl_integrated", "rt_apex",
             "rt_left", "rt_right", "apex_scan", "rt_kl_apex", "kl_apex_scan", "height", "prominence", "snr", "width",
             "area", "n_observed", "n_coeluting", "ms2_ppm")  # fmt: skip


def _row(targets: pa.Table, of: int, **changes) -> dict:
    """Target ``of``'s row, with ``changes``."""
    row = targets.filter(pc.equal(targets.column("target_id"), of)).to_pylist()[0]
    return {**row, **changes}


def _unscorable(case: str, targets: pa.Table) -> tuple[dict, ChannelSettings, int]:
    """A target the planted run cannot score, the channels it is compiled
    with and how many of its channels the library models."""
    if case == "rt_outside_the_run":  # the run is 0-180 s
        return _row(targets, 1000, target_id=2001, rt_start=1000.0, rt_end=1100.0), _channels(), 10
    if case == "precursor_in_no_window":  # 6+ is 194.8 Th, below the windows' 350-1250
        return _row(targets, 1000, target_id=2002, precursor_charge=6, precursor_mz=None), _channels(), 10
    # WW's ions (b1 187, y1 205 Th) are all below the m/z the channels start at.
    target = _row(targets, 1000, target_id=2003, modified_sequence="WW", precursor_charge=1, precursor_mz=None,
                  precursor_id=7777)  # fmt: skip
    return target, ChannelSettings(top_fragments=10, max_fragment_charge=2, min_fragment_mz=300.0), 0


@pytest.mark.parametrize("layout", ["precursor_major", "scan_major"])
@pytest.mark.parametrize("case", ["rt_outside_the_run", "precursor_in_no_window", "no_channels"])
def test_a_target_with_no_scan_or_no_channel_gets_its_no_candidate_row(planted, case, layout):
    """Codex review of the scorer: a chunk whose dense view had no scan slot
    or no channel raised (a reshape in the picker, an empty ``amax``)
    instead of giving each query its row. Alone, such a target is that
    chunk; beside a target that scores, it is a query without slots or
    channels in a chunk of others."""
    run, bundle, targets, shares = planted
    target, channels, modelled = _unscorable(case, targets)
    for rows in ([target], [_row(targets, 1000), target]):
        t = _score(bundle, pa.Table.from_pylist(rows, schema=XIC_TARGET_TABLE), shares, channels=channels,
                   layout=layout).table("candidates")  # fmt: skip
        mine = t.filter(pc.equal(t.column("target_id"), target["target_id"])).to_pylist()
        assert len(mine) == 1, (case, len(rows))
        r = mine[0]
        assert (r["rank"], r["chosen"], r["n_candidates"], r["n_channels"]) == (-1, True, 0, modelled)
        assert all(r[k] is None for k in _MEASURES), [k for k in _MEASURES if r[k] is not None]
        if len(rows) == 2:  # the other target is scored as it is alone
            other = t.filter(pc.and_(pc.equal(t.column("target_id"), 1000), t.column("chosen"))).to_pylist()[0]
            assert other["n_candidates"] >= 1 and other["score"] is not None


def test_a_querys_channel_count_does_not_depend_on_its_batch(planted):
    """Codex review: ``n_channels`` was taken through the candidate axis,
    whose width is the chunk's, so a target with no candidate counted 0
    modelled channels alone and its true 10 beside a target with one."""
    run, bundle, targets, shares = planted
    late = _row(targets, 1000, target_id=2004, rt_start=150.0, rt_end=180.0)  # scans, but nothing elutes there
    alone = _score(bundle, pa.Table.from_pylist([late], schema=XIC_TARGET_TABLE), shares).table("candidates")
    assert alone.num_rows == 1 and alone.column("n_candidates")[0].as_py() == 0
    both = _score(bundle, pa.Table.from_pylist([_row(targets, 1000), late], schema=XIC_TARGET_TABLE), shares)
    t = both.table("candidates")
    assert pc.max(t.column("n_candidates")).as_py() >= 1  # the other target has a candidate
    beside = t.filter(pc.equal(t.column("target_id"), 2004))
    assert alone.column("n_channels").to_pylist() == beside.column("n_channels").to_pylist() == [10]


@dataclass(frozen=True)
class _PartlyModelled(PlantedShares):
    """Planted shares with the ions at ``unmodelled`` m/z not modelled (NaN):
    a library that predicts nothing for them."""

    unmodelled: tuple[float, ...] = ()

    def abundance(self, ions) -> np.ndarray:
        out = super().abundance(ions)
        for mz in self.unmodelled:
            out[np.abs(ions.mz - mz) < 1e-6] = np.nan
        return out


def test_a_measured_fragment_the_library_does_not_model_keeps_its_intensity(planted):
    """Codex review: ``candidate_fragments`` took its intensities from the
    observation masked for the KL, which keeps only modelled channels, so a
    fragment the run measured and the library has no value for was written
    with zero intensity. Its quantities are the run's, modelled or not."""
    run, bundle, targets, shares = planted
    p = run.planted[0]
    strongest = float(p["frag_mz"][np.argmax(p["frag_ab"])])
    target = pa.Table.from_pylist([_row(targets, 1000)], schema=XIC_TARGET_TABLE)
    channels = ChannelSettings(max_fragment_charge=2)  # every ion: top-N ranks an unmodelled one last
    out = {}
    for name, source in (("modelled", shares), ("unmodelled", _PartlyModelled(shares.planted, unmodelled=(strongest,)))):
        result = _score(bundle, target, source, channels=channels)
        c = result.table("candidates")
        f = result.table("candidate_fragments")
        ion = f.filter(pc.less(pc.abs(pc.subtract(f.column("mz_theoretical"), strongest)), 1e-6)).to_pylist()
        assert len(ion) == 1
        out[name] = (c.filter(c.column("chosen")).select(["apex_scan", "rt_left", "rt_right"]).to_pylist(), ion[0])
    (where, unmodelled), (where_m, modelled) = out["unmodelled"], out["modelled"]
    assert where == where_m  # the same apex and extent, so the same cells
    assert not unmodelled["modelled"] and unmodelled["abundance"] is None and modelled["modelled"]
    assert unmodelled["intensity_apex"] == modelled["intensity_apex"] > 0
    assert unmodelled["intensity_summed"] == modelled["intensity_summed"] > unmodelled["intensity_apex"]
