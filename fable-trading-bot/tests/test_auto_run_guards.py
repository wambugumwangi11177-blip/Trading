"""Tests for the guards that only matter when nobody is watching: the journal's
idempotency record, and trimming the in-progress bar before signals are computed.

Both exist to stop the same class of failure -- a scheduled run doing damage
quietly. Double-entering a position because the task retried, or sizing off an
ATR collapsed by a one-print bar, are invisible in the moment and expensive
later.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fable_bot.journal import Journal, already_ran, completed_session_days
from fable_bot.live_runner import drop_incomplete_bar


@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"


# ── idempotency ──────────────────────────────────────────────────────────


def test_a_completed_run_marks_its_session_day(log_dir):
    journal = Journal(log_dir=log_dir)
    journal.event("run_end", status="ok", session_day="2026-08-06")
    assert completed_session_days(log_dir) == {"2026-08-06"}
    assert already_ran(date(2026, 8, 6), log_dir)


def test_a_failed_run_does_not_block_a_retry(log_dir):
    """Dying in preflight means no orders were placed, so re-running is safe."""
    journal = Journal(log_dir=log_dir)
    journal.event("run_end", status="preflight_failed", session_day="2026-08-06")
    assert not already_ran(date(2026, 8, 6), log_dir)


def test_a_dry_run_does_not_block_the_real_run(log_dir):
    journal = Journal(log_dir=log_dir)
    journal.event("run_end", status="dry_run", session_day="2026-08-06")
    assert not already_ran(date(2026, 8, 6), log_dir)


def test_unrelated_sessions_are_not_confused(log_dir):
    journal = Journal(log_dir=log_dir)
    journal.event("run_end", status="ok", session_day="2026-08-05")
    assert already_ran(date(2026, 8, 5), log_dir)
    assert not already_ran(date(2026, 8, 6), log_dir)


def test_missing_log_dir_is_not_an_error(tmp_path):
    assert completed_session_days(tmp_path / "nope") == set()


def test_a_torn_final_line_does_not_break_the_read(log_dir):
    """A hard kill mid-write must not make the whole journal unreadable."""
    journal = Journal(log_dir=log_dir)
    journal.event("run_end", status="ok", session_day="2026-08-06")
    with journal.path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "run_end", "status": "ok", "sess')  # truncated
    assert completed_session_days(log_dir) == {"2026-08-06"}


def test_events_are_one_json_object_per_line(log_dir):
    journal = Journal(log_dir=log_dir)
    journal.event("signal", symbol="SPY", side="long")
    journal.event("order", symbol="AMEX:SPY", qty=12)
    lines = journal.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("{") and line.endswith("}") for line in lines)


# ── the in-progress bar ──────────────────────────────────────────────────


def _frame(days: int, end: str) -> pd.DataFrame:
    index = pd.date_range(end=end, periods=days, freq="D")
    close = pd.Series(np.linspace(100, 110, days), index=index)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1_000},
        index=index,
    )


def test_todays_partial_bar_is_dropped():
    df = _frame(10, "2026-08-06")
    trimmed = drop_incomplete_bar(df, today=date(2026, 8, 6))
    assert len(trimmed) == len(df) - 1
    assert trimmed.index[-1].date() == date(2026, 8, 5)


def test_a_history_ending_yesterday_is_left_alone():
    df = _frame(10, "2026-08-05")
    trimmed = drop_incomplete_bar(df, today=date(2026, 8, 6))
    assert trimmed.equals(df)


def test_a_bar_dated_ahead_of_today_is_also_dropped():
    """A UTC-indexed crypto feed can print a bar dated past the ET session date."""
    df = _frame(10, "2026-08-07")
    trimmed = drop_incomplete_bar(df, today=date(2026, 8, 6))
    assert trimmed.index[-1].date() == date(2026, 8, 6)


def test_empty_history_survives():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert drop_incomplete_bar(empty, today=date(2026, 8, 6)).empty


def test_trimming_changes_atr_enough_to_matter():
    """The actual risk: a one-print bar collapses ATR, which sets position size.

    A tighter stop means a larger quantity for the same 1% risk budget, so an
    understated ATR silently oversizes the trade.
    """
    from fable_bot.strategies.indicators import atr

    df = _frame(30, "2026-08-05")
    partial = df.copy()
    # Today opens and barely moves: a bar with essentially zero range.
    partial.loc[pd.Timestamp("2026-08-06")] = {
        "open": 110.0, "high": 110.02, "low": 109.98, "close": 110.0, "volume": 900,
    }

    with_partial = atr(partial, period=14).iloc[-1]
    without_partial = atr(drop_incomplete_bar(partial, today=date(2026, 8, 6)), period=14).iloc[-1]

    assert with_partial < without_partial


# ── --no-wait must actually run now ──────────────────────────────────────


@pytest.fixture
def stubbed_cycle(monkeypatch, tmp_path):
    """A cycle with the broker and preflight stubbed out, clocked to 05:00 ET."""
    from datetime import datetime

    import fable_bot.auto_run as ar
    import fable_bot.broker as broker_mod
    import fable_bot.journal as jr
    import fable_bot.lessons as lessons_mod
    import fable_bot.live_runner as lr
    import fable_bot.reconcile as reconcile_mod
    from fable_bot.market_calendar import ET
    from fable_bot.preflight import PreflightResult

    monkeypatch.setattr(jr, "LOG_DIR", tmp_path)
    # The post-run learning hook must not touch the project's real memory/ or
    # spawn the real trading CLI: no history support -> reconciliation skips.
    memory_dir = tmp_path / "memory"
    monkeypatch.setattr(reconcile_mod, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(lessons_mod, "MEMORY_DIR", memory_dir)

    class StubBroker:
        def get_order_history(self, limit=100):
            raise NotImplementedError("stub broker has no history")

    monkeypatch.setattr(broker_mod, "build_broker", lambda config: StubBroker())

    # 2026-08-06 is a Thursday session; 05:00 ET is 265 minutes before the bell,
    # well past the 150-minute wait budget.
    monkeypatch.setattr(ar, "now_et", lambda: datetime(2026, 8, 6, 5, 0, tzinfo=ET))
    monkeypatch.setattr(ar, "already_ran", lambda *a, **k: False)

    class FakePreflight:
        def __init__(self, config):
            pass

        def ensure_ready(self, **kwargs):
            return PreflightResult(ok=True, detail="stub", cdp_connected=True,
                                   broker_connected=True, is_paper=True, equity=100_000.0)

        def cdp_up(self):
            return True

    monkeypatch.setattr(ar, "Preflight", FakePreflight)
    monkeypatch.setattr(lr, "run_signal_check", lambda **kwargs: ["ORDER stub"])
    return ar


def test_no_wait_runs_immediately_even_when_the_open_is_far_off(stubbed_cycle):
    """The regression: the too-early guard used to fire before --no-wait was
    considered, so the flag did nothing whenever it was most useful."""
    outcome = stubbed_cycle.auto_run(wait_for_open=False, dry_run=True)
    assert outcome.status != "too_early"
    assert outcome.actions == ["ORDER stub"]


def test_the_too_early_guard_still_protects_the_waiting_path(stubbed_cycle):
    """A scheduled run that would have to wait 265 minutes has a misconfigured
    trigger, and should say so rather than sleep half the day."""
    outcome = stubbed_cycle.auto_run(wait_for_open=True, dry_run=True)
    assert outcome.status == "too_early"
