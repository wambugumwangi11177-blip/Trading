"""Does the watcher actually re-arm, every day, forever?

Being armed once is easy and proves nothing. The property that matters is that
after running a cycle the loop advances to the *next* session rather than
re-firing on the one it just handled or stalling on it -- and that it keeps
going after a bad day, since an exception on Tuesday must not mean silence for
the rest of the year.

The clock here is driven by the fake `_sleep_until`, so the loop's own sense of
time advances exactly as it would in production, without waiting.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import fable_bot.auto_run as ar
from fable_bot.auto_run import RunOutcome, watch
from fable_bot.market_calendar import ET


@pytest.fixture
def fake_clock(monkeypatch, tmp_path):
    """Runs the watch loop against a controllable clock and a stubbed cycle."""
    import fable_bot.journal as jr

    monkeypatch.setattr(jr, "LOG_DIR", tmp_path)
    state = {"now": datetime(2026, 8, 6, 5, 0, tzinfo=ET), "ran": [], "wakes": []}

    def fake_now():
        return state["now"]

    def fake_sleep_until(target, *, label):
        state["wakes"].append(target)
        state["now"] = target

    def fake_auto_run(**kwargs):
        state["ran"].append(state["now"].date())
        state["now"] += timedelta(minutes=5)
        return RunOutcome(status="ok", exit_code=0, detail="stub", actions=[])

    monkeypatch.setattr(ar, "now_et", fake_now)
    monkeypatch.setattr(ar, "_sleep_until", fake_sleep_until)
    monkeypatch.setattr(ar, "auto_run", fake_auto_run)
    monkeypatch.setattr(ar.time, "sleep", lambda seconds: None)
    return state


def test_it_runs_on_five_consecutive_sessions(fake_clock):
    watch(max_cycles=5, max_retries=0)
    assert fake_clock["ran"] == [
        __import__("datetime").date(2026, 8, 6),
        __import__("datetime").date(2026, 8, 7),
        __import__("datetime").date(2026, 8, 10),
        __import__("datetime").date(2026, 8, 11),
        __import__("datetime").date(2026, 8, 12),
    ]


def test_it_skips_the_weekend_rather_than_firing_into_a_closed_market(fake_clock):
    watch(max_cycles=3, max_retries=0)
    days = fake_clock["ran"]
    assert days[1].isoweekday() == 5   # Friday 2026-08-07
    assert days[2].isoweekday() == 1   # straight to Monday 2026-08-10
    assert all(d.isoweekday() <= 5 for d in days)


def test_it_wakes_the_configured_lead_before_each_bell(fake_clock):
    watch(max_cycles=3, lead_minutes=10, max_retries=0)
    for wake in fake_clock["wakes"]:
        assert (wake.hour, wake.minute) == (9, 20), f"woke at {wake}, not 09:20 ET"


def test_a_different_lead_is_honoured(fake_clock):
    watch(max_cycles=2, lead_minutes=25, max_retries=0)
    for wake in fake_clock["wakes"]:
        assert (wake.hour, wake.minute) == (9, 5)


def test_it_never_re_fires_the_session_it_just_handled(fake_clock):
    """The loop's own guard: after a cycle, `next_session` must move past today."""
    watch(max_cycles=4, max_retries=0)
    assert len(set(fake_clock["ran"])) == len(fake_clock["ran"]), "a session ran twice"


def test_a_thrown_cycle_does_not_end_the_loop(fake_clock, monkeypatch):
    """One bad day must not silence every day after it."""
    calls = {"n": 0}

    def sometimes_explodes(**kwargs):
        calls["n"] += 1
        fake_clock["ran"].append(fake_clock["now"].date())
        fake_clock["now"] += timedelta(minutes=5)
        if calls["n"] == 2:
            raise RuntimeError("broker melted down")
        return RunOutcome(status="ok", exit_code=0, detail="stub", actions=[])

    monkeypatch.setattr(ar, "auto_run", sometimes_explodes)
    watch(max_cycles=4, max_retries=0)

    assert calls["n"] == 4, "the loop stopped after the failure"
    assert len(fake_clock["ran"]) == 4


def test_it_keeps_going_across_a_holiday(monkeypatch, tmp_path):
    """Thanksgiving 2026 (Nov 26) must be skipped, not fired into."""
    import fable_bot.journal as jr

    monkeypatch.setattr(jr, "LOG_DIR", tmp_path)
    state = {"now": datetime(2026, 11, 24, 5, 0, tzinfo=ET), "ran": []}

    monkeypatch.setattr(ar, "now_et", lambda: state["now"])
    monkeypatch.setattr(ar, "_sleep_until", lambda target, *, label: state.update(now=target))
    monkeypatch.setattr(ar.time, "sleep", lambda seconds: None)

    def fake_auto_run(**kwargs):
        state["ran"].append(state["now"].date())
        state["now"] += timedelta(minutes=5)
        return RunOutcome(status="ok", exit_code=0, detail="stub", actions=[])

    monkeypatch.setattr(ar, "auto_run", fake_auto_run)
    watch(max_cycles=4, max_retries=0)

    from datetime import date

    assert date(2026, 11, 26) not in state["ran"], "fired on Thanksgiving"
    assert state["ran"] == [date(2026, 11, 24), date(2026, 11, 25),
                            date(2026, 11, 27), date(2026, 11, 30)]


# ── retries on failure, but bounded ──────────────────────────────────────


def test_a_failed_cycle_is_retried_on_the_same_session(fake_clock, monkeypatch):
    """TradingView may simply have been slow; one more attempt is worth it."""
    seen = []

    def always_fails(**kwargs):
        seen.append(fake_clock["now"].date())
        return RunOutcome(status="preflight_failed", exit_code=2, detail="no broker", actions=[])

    monkeypatch.setattr(ar, "auto_run", always_fails)
    watch(max_cycles=3, max_retries=2, retry_delay_seconds=0)

    from datetime import date
    assert seen == [date(2026, 8, 6)] * 3, "retries should stay on the same session"


def test_retries_are_bounded_and_then_it_moves_on(fake_clock, monkeypatch):
    """A permanently broken day must not consume the rest of the week."""
    seen = []

    def always_fails(**kwargs):
        seen.append(fake_clock["now"].date())
        fake_clock["now"] += timedelta(minutes=1)
        return RunOutcome(status="error", exit_code=1, detail="boom", actions=[])

    monkeypatch.setattr(ar, "auto_run", always_fails)
    watch(max_cycles=4, max_retries=1, retry_delay_seconds=0)

    from datetime import date
    # One attempt + one retry on the 6th, then it gives up and moves to the 7th.
    assert seen == [date(2026, 8, 6), date(2026, 8, 6),
                    date(2026, 8, 7), date(2026, 8, 7)]


def test_a_successful_cycle_is_never_retried(fake_clock, monkeypatch):
    watch(max_cycles=2, max_retries=3, retry_delay_seconds=0)
    assert len(set(fake_clock["ran"])) == 2, "a successful session was repeated"
