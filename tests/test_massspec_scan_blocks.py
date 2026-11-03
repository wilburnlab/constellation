"""Reading a bundle as blocks of whole scans (`massspec.bundle.iter_scan_blocks`)
and cutting it into scan ranges (`scan_range_shards`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from _massspec_fixtures import dda_scans, dia_scans, write_bundle, write_bundle_manifest
from constellation.massspec.bundle import iter_scan_blocks, scan_range_shards
from constellation.massspec.io.encyclopedia._dia_write import _iter_scan_peaks


def _bundle(tmp_path: Path, scans, name: str = "run", **kw) -> Path:
    bundle = write_bundle(tmp_path, name, scans, **kw)
    write_bundle_manifest(bundle)
    return bundle


def _flatten(blocks) -> list[tuple[int, np.ndarray, np.ndarray]]:
    out = []
    for b in blocks:
        assert b.offsets[0] == 0 and b.offsets[-1] == b.columns["mz"].size
        assert np.all(np.diff(b.offsets) > 0), "a scan in a block with no points"
        for i, scan in enumerate(b.scan.tolist()):
            a, e = b.offsets[i], b.offsets[i + 1]
            out.append((scan, b.columns["mz"][a:e], b.columns["intensity"][a:e]))
    return out


def _expected(scans, lo=None, hi=None):
    return [
        (s["scan"], s["mz"], s["intensity"])
        for s in scans
        if s["mz"].size
        and (lo is None or s["scan"] >= lo)
        and (hi is None or s["scan"] < hi)
    ]


def _assert_same(got, expected):
    assert [g[0] for g in got] == [e[0] for e in expected]
    for (_, gm, gi), (_, em, ei) in zip(got, expected):
        np.testing.assert_array_equal(gm, em)
        np.testing.assert_array_equal(gi, ei)


@pytest.mark.parametrize("block_rows", [1, 5, 17, 10_000])
def test_blocks_hold_whole_scans_in_stored_order(tmp_path, block_rows):
    # Row groups of 17 rows cut through scans; two parts; one scan has no peaks.
    scans = dia_scans(cycles=12, empty_scan=7)
    bundle = _bundle(tmp_path, scans, row_group_size=17)
    blocks = list(iter_scan_blocks(bundle, block_rows=block_rows))
    seen = np.concatenate([b.scan for b in blocks])
    assert np.all(np.diff(seen) > 0), "a scan split across blocks"
    _assert_same(_flatten(blocks), _expected(scans))
    assert 7 not in seen.tolist()


def test_blocks_agree_with_the_spectra_writer_iterator(tmp_path):
    scans = dda_scans(n_ms1=6, per_ms1=7)
    bundle = _bundle(tmp_path, scans, row_group_size=11)
    reference = list(_iter_scan_peaks(bundle, batch_rows=9))
    _assert_same(_flatten(iter_scan_blocks(bundle, block_rows=9)), reference)


def test_scan_number_gaps(tmp_path):
    scans = [s for s in dia_scans(cycles=8) if not 10 <= s["scan"] <= 14]
    bundle = _bundle(tmp_path, scans, row_group_size=13)
    _assert_same(_flatten(iter_scan_blocks(bundle, block_rows=7)), _expected(scans))


@pytest.mark.parametrize("lo,hi", [(1, 5), (5, 6), (9, 30), (30, 10_000), (0, 1), (17, 17)])
def test_scan_range_keeps_lo_inclusive_hi_exclusive(tmp_path, lo, hi):
    scans = dia_scans(cycles=12)
    bundle = _bundle(tmp_path, scans, row_group_size=17)
    got = _flatten(iter_scan_blocks(bundle, block_rows=6, scan_range=(lo, hi)))
    _assert_same(got, _expected(scans, lo, hi))


def test_scan_range_reads_only_overlapping_row_groups(tmp_path, monkeypatch):
    scans = dia_scans(cycles=12)
    bundle = _bundle(tmp_path, scans, row_group_size=17)
    read: list[list[int]] = []
    original = pq.ParquetFile.iter_batches

    def spy(self, *args, row_groups=None, **kw):
        read.append(list(row_groups))
        return original(self, *args, row_groups=row_groups, **kw)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    list(iter_scan_blocks(bundle, scan_range=(3, 6)))
    total = sum(
        pq.ParquetFile(p).metadata.num_row_groups for p in sorted((bundle / "peaks").glob("*"))
    )
    assert sum(len(r) for r in read) < total


@pytest.mark.parametrize("target_rows", [1, 20, 50, 10**9])
def test_shards_partition_the_run(tmp_path, target_rows):
    scans = dia_scans(cycles=12, empty_scan=20)
    bundle = _bundle(tmp_path, scans, row_group_size=17)
    shards = scan_range_shards(bundle, target_rows=target_rows)
    assert shards[0][0] <= scans[0]["scan"]
    assert shards[-1][1] > scans[-1]["scan"]
    for (_, hi), (lo, _) in zip(shards, shards[1:]):
        assert hi == lo
    got = []
    for lo, hi in shards:
        got.extend(_flatten(iter_scan_blocks(bundle, block_rows=8, scan_range=(lo, hi))))
    _assert_same(got, _expected(scans))
    if target_rows == 10**9:
        assert len(shards) == 1
    if target_rows == 1:
        assert len(shards) > 2


def test_shards_do_not_depend_on_block_size(tmp_path):
    bundle = _bundle(tmp_path, dia_scans(cycles=12), row_group_size=17)
    assert scan_range_shards(bundle, target_rows=40) == scan_range_shards(bundle, target_rows=40)


def test_profile_points_need_asking_for(tmp_path):
    scans = dia_scans(cycles=3)
    bundle = _bundle(tmp_path, scans, name="run.profile", profile=True)
    with pytest.raises(ValueError, match="profile bundle"):
        list(iter_scan_blocks(bundle))
    _assert_same(_flatten(iter_scan_blocks(bundle, points=True)), _expected(scans))


def test_extra_columns_ride_along(tmp_path):
    scans = dia_scans(cycles=4)
    bundle = _bundle(tmp_path, scans)
    for b in iter_scan_blocks(bundle, columns=("mz", "intensity", "rt", "level")):
        for i in range(b.n_scans):
            a, e = b.offsets[i], b.offsets[i + 1]
            assert np.unique(b.columns["rt"][a:e]).size == 1
            assert np.unique(b.columns["level"][a:e]).size == 1
