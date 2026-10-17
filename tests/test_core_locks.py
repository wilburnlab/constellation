"""``core.locks.file_lock``: one holder at a time, across threads and processes."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from constellation.core.locks import LockHeld, file_lock


def test_threads_take_turns(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    inside = []
    overlap = []

    def work(name: str) -> None:
        with file_lock(lock):
            if inside:
                overlap.append(name)
            inside.append(name)
            time.sleep(0.05)
            inside.remove(name)

    threads = [threading.Thread(target=work, args=(str(i),)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlap == []


def test_without_waiting_a_held_lock_is_refused_before_the_block_runs(tmp_path: Path) -> None:
    lock = tmp_path / "sub" / "x.lock"
    ran = []
    with file_lock(lock):
        with pytest.raises(LockHeld):
            with file_lock(lock, wait=False):
                ran.append(1)
    assert ran == []
    with file_lock(lock, wait=False):  # free again
        ran.append(1)
    assert ran == [1]


def test_another_process_holding_it_is_seen(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time\n"
         "from constellation.core.locks import file_lock\n"
         f"with file_lock({str(lock)!r}):\n"
         "    print('held', flush=True)\n"
         "    sys.stdin.readline()\n"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(LockHeld):
            with file_lock(lock, wait=False):
                pass
    finally:
        holder.communicate("\n", timeout=10)
    with file_lock(lock, wait=False):
        pass
