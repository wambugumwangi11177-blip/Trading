"""The kill-switch must survive process death, because the runner is one-shot.

The bug these cover: peak_equity lived in a module global, the unattended
runner is a fresh interpreter per session, so the peak reset every morning and
a multi-day drawdown could never be detected. Every test here therefore builds
a NEW monitor from the same path rather than reusing an object -- reusing one
would pass against the old broken code too, which is how this went unnoticed
through 266 passing tests.
"""
from __future__ import annotations

import json

import pytest

from fable_bot.risk.manager import DrawdownMonitor


@pytest.fixture()
def state_path(tmp_path):
    return tmp_path / "drawdown_state.json"


def _fresh(state_path, equity, limit=10.0):
    """A monitor as a brand-new process would build it."""
    return DrawdownMonitor.load(state_path, max_drawdown_pct=limit, current_equity=equity)


def test_peak_survives_a_new_process(state_path):
    day_one = _fresh(state_path, 100_000.0)
    day_one.update(117_000.0)

    day_two = _fresh(state_path, 113_000.0)
    assert day_two.peak_equity == pytest.approx(117_000.0)


def test_multi_day_drawdown_trips_the_switch(state_path):
    """The exact shape of the Aug-Sep slide, scaled to a clean 10%."""
    _fresh(state_path, 100_000.0).update(117_000.0)

    # A later session, fresh process, equity down 10.3% from the stored peak.
    later = _fresh(state_path, 105_000.0)
    assert later.update(104_900.0) is True
    assert later.halted is True


def test_same_day_only_move_does_not_trip(state_path):
    monitor = _fresh(state_path, 100_000.0)
    assert monitor.update(95_000.0) is False
    assert monitor.halted is False


def test_halt_is_sticky_across_processes(state_path):
    first = _fresh(state_path, 100_000.0)
    first.update(89_000.0)
    assert first.halted is True

    # Equity recovers completely. The halt must NOT clear itself: only a human
    # reset may resume trading after a kill-switch fires.
    recovered = _fresh(state_path, 100_000.0)
    assert recovered.halted is True


def test_manual_reset_clears_and_rebaselines(state_path):
    monitor = _fresh(state_path, 100_000.0)
    monitor.update(85_000.0)
    assert monitor.halted is True

    monitor.manual_reset(new_peak=85_000.0)
    after = _fresh(state_path, 85_000.0)
    assert after.halted is False
    assert after.peak_equity == pytest.approx(85_000.0)


def test_peak_never_regresses_to_current_equity(state_path):
    """Loading at a lower equity must not quietly lower the peak."""
    _fresh(state_path, 100_000.0).update(120_000.0)
    assert _fresh(state_path, 50_000.0).peak_equity == pytest.approx(120_000.0)


def test_corrupt_state_degrades_instead_of_raising(state_path):
    state_path.write_text("{ this is not json", encoding="utf-8")
    monitor = _fresh(state_path, 99_000.0)
    assert monitor.peak_equity == pytest.approx(99_000.0)
    assert monitor.halted is False


def test_missing_state_file_starts_clean(state_path):
    monitor = _fresh(state_path, 42_000.0)
    assert monitor.peak_equity == pytest.approx(42_000.0)
    assert state_path.exists()
    assert json.loads(state_path.read_text())["peak_equity"] == pytest.approx(42_000.0)


def test_drawdown_pct_reports_against_stored_peak(state_path):
    _fresh(state_path, 100_000.0).update(117_148.02)
    monitor = _fresh(state_path, 113_662.34)
    assert monitor.drawdown_pct(113_662.34) == pytest.approx(2.98, abs=0.02)
