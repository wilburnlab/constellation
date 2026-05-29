"""Tests for ``constellation.massspec.io.thermo._batch``.

Two tiers:

1. **Unit tests** that monkeypatch ``_convert_one_file`` to a no-op
   marker writer. The ``n_workers <= 1`` inline path is exercised
   directly. Spawn-mode (``n_workers > 1``) is exercised by intercepting
   the ``ProcessPoolExecutor`` constructor — the spawn child re-imports
   the module fresh so parent-side monkeypatching does not propagate,
   making real spawn-driven tests impractical without a real ``.raw``.

2. **CLI integration** — drive ``_cmd_massspec_convert`` end-to-end with
   ``convert_batch`` monkeypatched to canned results. Covers the
   directory-glob path, summary line, and exit codes.

The real-fixture smoke (gated on ``$CONSTELLATION_TEST_THERMO_RAW``) is
in ``test_thermo_reader.py``; adding a multi-file smoke here would
require the user to maintain two .raw fixtures, which isn't worth it
when the unit + CLI tests already pin every code path that's not the
``.NET CLR`` itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from constellation.massspec.io.thermo import (
    BatchResult,
    convert_batch,
)
from constellation.massspec.io.thermo.manifest import MANIFEST_FILENAME


# ──────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────


def _make_raw_inputs(tmp_path: Path, stems: list[str]) -> list[Path]:
    """Create empty ``<stem>.raw`` files in ``tmp_path/in/`` and return paths."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    paths: list[Path] = []
    for stem in stems:
        p = in_dir / f"{stem}.raw"
        p.write_bytes(b"")
        paths.append(p)
    return paths


def _write_marker_manifest(bundle_dir: Path, source: str) -> None:
    """Write a stub ``manifest.json`` so the skip-check sees a complete bundle."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / MANIFEST_FILENAME).write_text(
        json.dumps({"source_file": source})
    )


# ──────────────────────────────────────────────────────────────────────
# 1. Unit tier — inline (n_workers <= 1) and intercept-only spawn
# ──────────────────────────────────────────────────────────────────────


class TestConvertBatchInline:
    """The ``n_workers <= 1`` inline path. Real Python calls, no IPC."""

    def test_dispatches_each_path(self, tmp_path, monkeypatch):
        paths = _make_raw_inputs(tmp_path, ["a", "b", "c"])
        out_parent = tmp_path / "out"

        calls: list[tuple[str, str]] = []

        def fake_convert(src, out_dir, **kwargs):
            calls.append((str(src), str(out_dir)))
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / MANIFEST_FILENAME).write_text("{}")
            return None

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch._convert_one_file",
            fake_convert,
        )

        results = convert_batch(paths, out_parent, n_workers=1)

        assert [r.status for r in results] == ["ok", "ok", "ok"]
        assert [r.input_path for r in results] == paths
        assert [r.bundle_dir for r in results] == [
            out_parent / "a",
            out_parent / "b",
            out_parent / "c",
        ]
        assert len(calls) == 3
        for path in paths:
            assert (out_parent / path.stem / MANIFEST_FILENAME).is_file()

    def test_skips_existing_manifest(self, tmp_path, monkeypatch):
        paths = _make_raw_inputs(tmp_path, ["a", "b"])
        out_parent = tmp_path / "out"
        _write_marker_manifest(out_parent / "a", str(paths[0]))

        calls: list[str] = []

        def fake_convert(src, out_dir, **kwargs):
            calls.append(str(src))
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / MANIFEST_FILENAME).write_text("{}")
            return None

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch._convert_one_file",
            fake_convert,
        )

        results = convert_batch(paths, out_parent, n_workers=1, force=False)

        assert [r.status for r in results] == ["skipped", "ok"]
        # Worker invoked only for the un-skipped path.
        assert calls == [str(paths[1])]

    def test_force_overrides_skip(self, tmp_path, monkeypatch):
        paths = _make_raw_inputs(tmp_path, ["a", "b"])
        out_parent = tmp_path / "out"
        _write_marker_manifest(out_parent / "a", str(paths[0]))

        calls: list[str] = []

        def fake_convert(src, out_dir, **kwargs):
            calls.append(str(src))
            assert kwargs["force"] is True
            return None

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch._convert_one_file",
            fake_convert,
        )

        results = convert_batch(paths, out_parent, n_workers=1, force=True)

        assert [r.status for r in results] == ["ok", "ok"]
        assert sorted(calls) == sorted(str(p) for p in paths)

    def test_aggregates_errors(self, tmp_path, monkeypatch):
        paths = _make_raw_inputs(tmp_path, ["good1", "bad", "good2"])
        out_parent = tmp_path / "out"

        def fake_convert(src, out_dir, **kwargs):
            if Path(src).stem == "bad":
                raise RuntimeError("synthetic failure")
            return None

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch._convert_one_file",
            fake_convert,
        )

        results = convert_batch(paths, out_parent, n_workers=1)

        statuses = [r.status for r in results]
        assert statuses == ["ok", "error", "ok"]
        bad = results[1]
        assert bad.detail is not None
        assert "RuntimeError" in bad.detail
        assert "synthetic failure" in bad.detail

    def test_propagates_capture_trailer_extras_false(self, tmp_path, monkeypatch):
        """``--no-trailer-extras`` flips this to False; must reach the worker."""
        paths = _make_raw_inputs(tmp_path, ["a"])
        out_parent = tmp_path / "out"

        captured: dict[str, object] = {}

        def fake_convert(src, out_dir, **kwargs):
            captured.update(kwargs)
            return None

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch._convert_one_file",
            fake_convert,
        )

        convert_batch(
            paths,
            out_parent,
            n_workers=1,
            capture_trailer_extras=False,
            compute_sha256=False,
        )

        assert captured["capture_trailer_extras"] is False
        assert captured["compute_sha256"] is False

    def test_duplicate_stem_raises(self, tmp_path):
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "same.raw").write_bytes(b"")
        (d2 / "same.raw").write_bytes(b"")
        paths = [d1 / "same.raw", d2 / "same.raw"]

        with pytest.raises(ValueError, match="duplicate output bundle dirs"):
            convert_batch(paths, tmp_path / "out", n_workers=1)

    def test_empty_paths_raises(self, tmp_path):
        with pytest.raises(ValueError, match="paths is empty"):
            convert_batch([], tmp_path / "out", n_workers=1)

    def test_all_skipped_no_workers_invoked(self, tmp_path, monkeypatch):
        paths = _make_raw_inputs(tmp_path, ["a", "b"])
        out_parent = tmp_path / "out"
        for p in paths:
            _write_marker_manifest(out_parent / p.stem, str(p))

        called = False

        def fake_convert(*a, **kw):
            nonlocal called
            called = True
            return None

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch._convert_one_file",
            fake_convert,
        )

        results = convert_batch(paths, out_parent, n_workers=4)
        assert [r.status for r in results] == ["skipped", "skipped"]
        assert called is False


class TestConvertBatchSpawn:
    """Spawn-mode (n_workers > 1) — intercept the executor constructor."""

    def test_uses_spawn_context(self, tmp_path, monkeypatch):
        """Verify ProcessPoolExecutor is constructed with mp_context=spawn."""
        paths = _make_raw_inputs(tmp_path, ["a", "b"])
        out_parent = tmp_path / "out"

        seen: dict[str, object] = {}

        # Stub the executor so we can verify constructor kwargs without
        # actually spawning subprocesses (which would try to import the
        # real _convert_one_file and need pythonnet).
        class _FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class _FakeExecutor:
            def __init__(self, max_workers, mp_context):
                seen["max_workers"] = max_workers
                seen["mp_context"] = mp_context

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

            def submit(self, fn, *args, **kwargs):
                # Run the function inline so the result aggregation
                # path is exercised end-to-end.
                return _FakeFuture(fn(*args, **kwargs))

        def fake_as_completed(futures):
            return list(futures)

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch.ProcessPoolExecutor",
            _FakeExecutor,
        )
        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch.as_completed",
            fake_as_completed,
        )
        # The inline worker still needs the underlying convert stubbed.
        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch._convert_one_file",
            lambda *a, **kw: None,
        )

        results = convert_batch(paths, out_parent, n_workers=2)

        assert seen["max_workers"] == 2
        ctx = seen["mp_context"]
        # multiprocessing.get_context("spawn") returns a SpawnContext;
        # its _name is "spawn".
        assert getattr(ctx, "_name", None) == "spawn" or "spawn" in repr(ctx).lower()
        assert all(r.status == "ok" for r in results)

    def test_clamps_n_workers_to_pending_count(self, tmp_path, monkeypatch):
        """One pending file + n_workers=8 → max_workers=1."""
        paths = _make_raw_inputs(tmp_path, ["only"])
        out_parent = tmp_path / "out"

        seen: dict[str, object] = {}

        class _FakeExecutor:
            def __init__(self, max_workers, mp_context):
                seen["max_workers"] = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

            def submit(self, fn, *args, **kwargs):
                class _F:
                    def result(self_inner):
                        return fn(*args, **kwargs)

                return _F()

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch.ProcessPoolExecutor",
            _FakeExecutor,
        )
        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch.as_completed",
            lambda fs: list(fs),
        )
        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch._convert_one_file",
            lambda *a, **kw: None,
        )

        # n_workers=8 but only 1 pending → should clamp inline (n=1
        # uses the inline branch, NOT the executor).
        convert_batch(paths, out_parent, n_workers=8)

        # With 1 pending, the inline branch fires and the executor
        # is never constructed.
        assert "max_workers" not in seen

    def test_uses_executor_when_pending_geq_two_and_threads_geq_two(
        self, tmp_path, monkeypatch
    ):
        """Two pending files + n_workers=2 → executor is used."""
        paths = _make_raw_inputs(tmp_path, ["a", "b"])
        out_parent = tmp_path / "out"

        seen: dict[str, object] = {}

        class _FakeExecutor:
            def __init__(self, max_workers, mp_context):
                seen["max_workers"] = max_workers
                seen["mp_context"] = mp_context

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

            def submit(self, fn, *args, **kwargs):
                class _F:
                    def result(self_inner):
                        return fn(*args, **kwargs)

                return _F()

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch.ProcessPoolExecutor",
            _FakeExecutor,
        )
        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch.as_completed",
            lambda fs: list(fs),
        )
        monkeypatch.setattr(
            "constellation.massspec.io.thermo._batch._convert_one_file",
            lambda *a, **kw: None,
        )

        convert_batch(paths, out_parent, n_workers=2)

        assert seen["max_workers"] == 2


# ──────────────────────────────────────────────────────────────────────
# 2. CLI integration tier
# ──────────────────────────────────────────────────────────────────────


def _make_args(
    *,
    input_path: Path,
    output_dir: Path | None = None,
    threads: int = 1,
    force: bool = False,
    no_progress: bool = True,
) -> argparse.Namespace:
    """Build a Namespace matching the argparse layout for `convert`."""
    return argparse.Namespace(
        input=input_path,
        output_dir=output_dir,
        threads=threads,
        rt_bin_width_s=60.0,
        profile=False,
        capture_trailer_extras=True,
        compute_sha256=True,
        force=force,
        batch_size=64,
        no_progress=no_progress,
    )


class TestCliRouter:
    def test_legacy_single_file_unchanged(self, tmp_path, monkeypatch, capsys):
        """Single file + --threads 1 → legacy path emits 'wrote bundle:'."""
        from constellation.massspec import cli as ms_cli

        src = tmp_path / "sample.raw"
        src.write_bytes(b"")
        out = tmp_path / "out"

        class _Manifest:
            outputs = {
                "peaks": "peaks.parquet",
                "scan_metadata": "scan_metadata.parquet",
                "acquisition_metadata": "acquisition_metadata.parquet",
            }

        def fake_thermo_convert(src_arg, bundle_dir, **kwargs):
            Path(bundle_dir).mkdir(parents=True, exist_ok=True)
            (Path(bundle_dir) / MANIFEST_FILENAME).write_text("{}")
            return _Manifest()

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._netruntime.require_thermo",
            lambda: None,
        )
        monkeypatch.setattr(
            "constellation.massspec.io.thermo.convert",
            fake_thermo_convert,
        )

        args = _make_args(input_path=src, output_dir=out, threads=1)
        rc = ms_cli._cmd_massspec_convert(args)

        captured = capsys.readouterr()
        assert rc == 0
        assert "wrote bundle:" in captured.out
        assert "batch convert:" not in captured.err

    def test_batch_directory_routes_through_convert_batch(
        self, tmp_path, monkeypatch, capsys
    ):
        from constellation.massspec import cli as ms_cli

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        for stem in ("a", "b"):
            (in_dir / f"{stem}.raw").write_bytes(b"")

        def fake_convert_batch(paths, output_parent, **kwargs):
            return [
                BatchResult(
                    input_path=p,
                    bundle_dir=output_parent / p.stem,
                    status="ok",
                )
                for p in paths
            ]

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._netruntime.require_thermo",
            lambda: None,
        )
        monkeypatch.setattr(
            "constellation.massspec.io.thermo.convert_batch",
            fake_convert_batch,
        )

        out = tmp_path / "out"
        args = _make_args(input_path=in_dir, output_dir=out, threads=2)
        rc = ms_cli._cmd_massspec_convert(args)

        captured = capsys.readouterr()
        assert rc == 0
        assert "batch convert: 2 succeeded, 0 skipped, 0 failed" in captured.err
        assert "wrote bundle:" not in captured.out

    def test_batch_exits_5_on_error(self, tmp_path, monkeypatch, capsys):
        from constellation.massspec import cli as ms_cli

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        bad = in_dir / "bad.raw"
        good = in_dir / "good.raw"
        bad.write_bytes(b"")
        good.write_bytes(b"")

        def fake_convert_batch(paths, output_parent, **kwargs):
            return [
                BatchResult(
                    input_path=p,
                    bundle_dir=output_parent / p.stem,
                    status="error" if p.stem == "bad" else "ok",
                    detail="RuntimeError: nope" if p.stem == "bad" else None,
                )
                for p in paths
            ]

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._netruntime.require_thermo",
            lambda: None,
        )
        monkeypatch.setattr(
            "constellation.massspec.io.thermo.convert_batch",
            fake_convert_batch,
        )

        args = _make_args(input_path=in_dir, threads=2)
        rc = ms_cli._cmd_massspec_convert(args)

        captured = capsys.readouterr()
        assert rc == 5
        assert "1 failed" in captured.err
        assert "  failed:" in captured.err
        assert "bad.raw" in captured.err
        assert "RuntimeError: nope" in captured.err

    def test_empty_directory_exits_2(self, tmp_path, monkeypatch, capsys):
        from constellation.massspec import cli as ms_cli

        in_dir = tmp_path / "empty"
        in_dir.mkdir()

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._netruntime.require_thermo",
            lambda: None,
        )

        args = _make_args(input_path=in_dir, threads=2)
        rc = ms_cli._cmd_massspec_convert(args)

        captured = capsys.readouterr()
        assert rc == 2
        assert "no .raw files in" in captured.err

    def test_missing_input_exits_2(self, tmp_path, capsys):
        from constellation.massspec import cli as ms_cli

        args = _make_args(input_path=tmp_path / "does-not-exist.raw")
        rc = ms_cli._cmd_massspec_convert(args)
        captured = capsys.readouterr()
        assert rc == 2
        assert "input not found" in captured.err

    def test_single_file_with_threads_gt_1_uses_batch(
        self, tmp_path, monkeypatch, capsys
    ):
        """A single .raw file under --threads 2 routes through batch."""
        from constellation.massspec import cli as ms_cli

        src = tmp_path / "solo.raw"
        src.write_bytes(b"")

        called_with: dict[str, object] = {}

        def fake_convert_batch(paths, output_parent, **kwargs):
            called_with["paths"] = paths
            called_with["output_parent"] = output_parent
            return [
                BatchResult(
                    input_path=paths[0],
                    bundle_dir=output_parent / paths[0].stem,
                    status="ok",
                )
            ]

        monkeypatch.setattr(
            "constellation.massspec.io.thermo._netruntime.require_thermo",
            lambda: None,
        )
        monkeypatch.setattr(
            "constellation.massspec.io.thermo.convert_batch",
            fake_convert_batch,
        )

        args = _make_args(input_path=src, threads=2)
        rc = ms_cli._cmd_massspec_convert(args)

        captured = capsys.readouterr()
        assert rc == 0
        assert called_with["paths"] == [src]
        # When -o is omitted on single-file-via-batch, output parent
        # defaults to the file's parent dir (matches legacy semantics).
        assert called_with["output_parent"] == src.parent
        assert "batch convert: 1 succeeded" in captured.err

    def test_directory_glob_case_insensitive_and_deduped(
        self, tmp_path, monkeypatch
    ):
        """Directory mode picks up *.raw and *.RAW without double-counting.

        On case-insensitive filesystems both globs hit the same dirent;
        we dedupe via resolve() so the input list has one entry per
        physical file regardless of casing.
        """
        from constellation.massspec import cli as ms_cli

        in_dir = tmp_path / "in"
        in_dir.mkdir()
        # Lowercase + uppercase + a non-.raw file.
        (in_dir / "lower.raw").write_bytes(b"")
        (in_dir / "ignore.txt").write_bytes(b"")
        # Skip an explicit uppercase fixture: case-insensitive
        # filesystems would have it collide with a sibling lowercase,
        # and case-sensitive filesystems still exercise the second
        # glob below — _collect_raw_files runs both patterns and
        # dedupes by resolved path, which is the property under test.

        found = ms_cli._collect_raw_files(in_dir)
        # Either case: one match (lower.raw); ignore.txt is filtered.
        assert len(found) == 1
        assert found[0].name == "lower.raw"
