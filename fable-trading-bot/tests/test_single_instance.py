"""The lock that stops two cycles trading at once.

Sequential re-runs are already handled by the journal's `already_ran` check.
This covers the case that check cannot see: two cycles overlapping in time, both
reading an empty position book and both sending the same entry. A watcher that
gets started twice, or a manual run launched during a scheduled one, is enough
to cause it.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from fable_bot.auto_run import single_instance


@pytest.fixture
def lock_dir(tmp_path):
    """Keep test locks out of the project's logs/ directory, which a running
    watcher is using for real."""
    return tmp_path


def test_the_lock_is_acquired_when_free(lock_dir):
    with single_instance("test-free.lock", lock_dir) as acquired:
        assert acquired


def test_the_lock_is_released_afterwards(lock_dir):
    with single_instance("test-reuse.lock", lock_dir) as first:
        assert first
    with single_instance("test-reuse.lock", lock_dir) as second:
        assert second


def test_a_second_holder_in_another_process_is_refused(lock_dir):
    """Threads in one process share file handles, so this has to cross a real
    process boundary to mean anything."""
    holder = textwrap.dedent("""
        import pathlib, sys, time
        from fable_bot.auto_run import single_instance
        with single_instance("test-contended.lock", pathlib.Path(sys.argv[1])) as acquired:
            print("ACQUIRED" if acquired else "REFUSED", flush=True)
            time.sleep(30)
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", holder, str(lock_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ACQUIRED"

        with single_instance("test-contended.lock", lock_dir) as acquired:
            assert not acquired, "a second cycle acquired the lock while one was held"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_the_lock_survives_the_holder_being_killed(lock_dir):
    """A crashed run must not leave a lock that silences every later session.

    This is why the implementation uses an OS advisory lock rather than a PID
    file: the kernel drops it when the process dies, with nothing to clean up.
    """
    holder = textwrap.dedent("""
        import pathlib, sys, time
        from fable_bot.auto_run import single_instance
        with single_instance("test-crash.lock", pathlib.Path(sys.argv[1])) as acquired:
            print("ACQUIRED" if acquired else "REFUSED", flush=True)
            time.sleep(30)
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", holder, str(lock_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdout.readline().strip() == "ACQUIRED"
    proc.kill()
    proc.wait(timeout=10)

    with single_instance("test-crash.lock", lock_dir) as acquired:
        assert acquired, "the lock outlived the process that held it"
