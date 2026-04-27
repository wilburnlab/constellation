"""Unit tests for constellation.core.io.readers."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.core.io.readers import (
    RawReader,
    ReaderNotFoundError,
    ReadResult,
    find_reader,
    register_reader,
    registered_readers,
)


def _empty_table() -> pa.Table:
    return pa.table({"time_s": pa.array([], type=pa.float64())})


def test_register_reader_requires_a_suffix():
    class NoSuffix(RawReader):
        suffixes = ()
        modality = "test"

        def read(self, source):
            return ReadResult(primary=_empty_table())

    with pytest.raises(ValueError, match="must declare at least one suffix"):
        register_reader(NoSuffix)


def test_register_reader_requires_dot_prefix():
    class BadSuffix(RawReader):
        suffixes = ("foo",)
        modality = "test"

        def read(self, source):
            return ReadResult(primary=_empty_table())

    with pytest.raises(ValueError, match="must start with a dot"):
        register_reader(BadSuffix)


def test_find_reader_dispatches_on_suffix():
    @register_reader
    class FooReader(RawReader):
        suffixes = (".foo_test_a",)
        modality = "domain-a"

        def read(self, source):
            return ReadResult(primary=_empty_table())

    reader = find_reader("/some/path/example.foo_test_a")
    assert isinstance(reader, FooReader)


def test_find_reader_unknown_suffix_raises():
    with pytest.raises(ReaderNotFoundError, match="no reader registered"):
        find_reader("/some/path/example.never_registered_suffix")


def test_find_reader_modality_disambiguation():
    @register_reader
    class ReaderA(RawReader):
        suffixes = (".shared_test",)
        modality = "alpha"

        def read(self, source):
            return ReadResult(primary=_empty_table())

    @register_reader
    class ReaderB(RawReader):
        suffixes = (".shared_test",)
        modality = "beta"

        def read(self, source):
            return ReadResult(primary=_empty_table())

    # Ambiguous without modality —
    with pytest.raises(ReaderNotFoundError, match="claimed by multiple readers"):
        find_reader("/path/example.shared_test")

    # Disambiguates by modality.
    assert isinstance(find_reader("/p/x.shared_test", modality="alpha"), ReaderA)
    assert isinstance(find_reader("/p/x.shared_test", modality="beta"), ReaderB)

    # Wrong modality raises with a useful message.
    with pytest.raises(ReaderNotFoundError, match="no reader for suffix"):
        find_reader("/p/x.shared_test", modality="gamma")


def test_suffix_match_is_case_insensitive():
    @register_reader
    class CaseReader(RawReader):
        suffixes = (".case_test",)
        modality = "case"

        def read(self, source):
            return ReadResult(primary=_empty_table())

    assert isinstance(find_reader(Path("/p/X.CASE_TEST")), CaseReader)


def test_registered_readers_snapshot_is_a_copy():
    snapshot = registered_readers()
    snapshot.clear()
    # Mutating the returned dict must not affect the live registry.
    assert registered_readers(), "live registry should be unaffected"


def test_read_result_companions_default_empty():
    rr = ReadResult(primary=_empty_table())
    assert rr.companions == {}
    assert rr.run_metadata == {}
