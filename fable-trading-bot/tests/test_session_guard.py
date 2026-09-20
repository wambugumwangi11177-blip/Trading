"""Tests for the intraday futures session risk guard.

These cover the rules that decide whether a bad session stays survivable: the
daily loss halt, contract sizing off the real stop distance, and the time
windows. Sizing is the one where an off-by-one is expensive rather than
embarrassing.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from fable_bot.risk.session_guard import SessionRiskConfig, SessionRiskGuard

TODAY = date(2026, 8, 4)


def _guard(**overrides) -> SessionRiskGuard:
    config = SessionRiskConfig(**{
        "max_risk_per_trade_pct": 1.0, "max_daily_loss_pct": 3.0,
        "max_trades_per_session": 4, "max_concurrent_positions": 1,
        "no_entry_first_minutes": 5, "no_entry_last_minutes": 20,
        **overrides,
    })
    g = SessionRiskGuard(config=config, starting_equity=111_822.0)
    g.start_session(TODAY, 111_822.0)
    return g


def at(hour: int, minute: int) -> datetime:
    return datetime(2026, 8, 4, hour, minute)


# ── Sizing ───────────────────────────────────────────────────────────────

def test_sizes_contracts_off_stop_distance_and_multiplier():
    g = _guard()
    # NQ at $20/pt with a 50-point stop risks $1,000 per contract; the budget
    # is $1,118, so exactly one contract fits.
    qty, reason = g.contracts_for(29_100.0, 29_050.0, point_value=20.0)
    assert qty == 1
    assert reason == ""


def test_refuses_when_one_contract_exceeds_the_budget():
    g = _guard()
    # A 1.5x daily-ATR stop on full-size NQ: 1043 points x $20 = $20,860.
    qty, reason = g.contracts_for(29_100.0, 28_057.0, point_value=20.0)
    assert qty == 0
    assert "over the" in reason


def test_a_daily_atr_stop_fits_neither_nq_nor_mnq():
    """1.5x daily ATR on NQ is ~1043 points: $20,860 on NQ, $2,086 even on MNQ.

    Both exceed the $1,118 budget, which is exactly why this account cannot
    swing-trade index futures on daily bars regardless of contract size.
    """
    g = _guard()
    entry, stop = 29_100.0, 28_057.0
    assert g.contracts_for(entry, stop, point_value=20.0)[0] == 0  # NQ
    assert g.contracts_for(entry, stop, point_value=2.0)[0] == 0   # MNQ


def test_an_intraday_stop_fits_both_with_micros_sized_larger():
    """At a 55-point intraday stop the budget affords 1 NQ or 10 MNQ.

    The margin is genuinely this thin: at 56 points one NQ contract costs
    $1,120 against a $1,118 budget and the trade is correctly refused, which
    is why the micros carry the sizing granularity.
    """
    g = _guard()
    entry, stop = 29_100.0, 29_045.0
    assert g.contracts_for(entry, stop, point_value=20.0)[0] == 1
    assert g.contracts_for(entry, stop, point_value=2.0)[0] == 10
    # One point wider and full-size NQ no longer fits at all.
    assert g.contracts_for(29_100.0, 29_044.0, point_value=20.0)[0] == 0


def test_sizing_rounds_down_never_up():
    g = _guard()
    # $1,118 budget / (10pts x $50) = 2.236 contracts -> 2, not 3.
    qty, _ = g.contracts_for(7_640.0, 7_630.0, point_value=50.0)
    assert qty == 2


def test_zero_stop_distance_is_rejected():
    g = _guard()
    assert g.contracts_for(7_640.0, 7_640.0, point_value=50.0)[0] == 0


# ── Daily loss limit ─────────────────────────────────────────────────────

def test_daily_loss_limit_halts_the_session():
    g = _guard()
    g.record_fill(opened=True)
    g.record_fill(pnl=-3_400.0, closed=True)  # over 3% of 111,822
    allowed, reason = g.can_enter(at(11, 0))
    assert not allowed
    assert "daily loss limit" in reason


def test_halt_persists_even_after_a_later_win():
    """Once the day is done it stays done -- no trading back out of the hole."""
    g = _guard()
    g.record_fill(opened=True)
    g.record_fill(pnl=-3_400.0, closed=True)
    g.can_enter(at(11, 0))            # trips the halt
    g.record_fill(pnl=+5_000.0, closed=True)
    allowed, reason = g.can_enter(at(12, 0))
    assert not allowed
    assert "daily loss limit" in reason


def test_losses_within_the_limit_still_allow_trading():
    g = _guard()
    g.record_fill(opened=True)
    g.record_fill(pnl=-1_000.0, closed=True)
    assert g.can_enter(at(11, 0))[0]


# ── Trade and position caps ──────────────────────────────────────────────

def test_max_trades_per_session_is_enforced():
    g = _guard(max_trades_per_session=2)
    for _ in range(2):
        g.record_fill(opened=True)
        g.record_fill(pnl=10.0, closed=True)
    allowed, reason = g.can_enter(at(11, 0))
    assert not allowed
    assert "max trades" in reason


def test_only_one_position_at_a_time_by_default():
    """ES and NQ are ~0.9 correlated; two index longs is one bet at 2x size."""
    g = _guard()
    g.record_fill(opened=True)
    allowed, reason = g.can_enter(at(11, 0))
    assert not allowed
    assert "concurrent positions" in reason


def test_closing_frees_the_slot():
    g = _guard()
    g.record_fill(opened=True)
    g.record_fill(pnl=50.0, closed=True)
    assert g.can_enter(at(11, 0))[0]


# ── Time windows ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("hour,minute,ok", [
    (9, 20, False),   # pre-open
    (9, 31, False),   # inside the opening auction window
    (9, 36, True),    # settled
    (12, 0, True),
    (15, 39, True),
    (15, 41, False),  # inside the last 20 minutes
    (16, 30, False),  # after the bell
])
def test_entry_time_window(hour, minute, ok):
    assert _guard().can_enter(at(hour, minute))[0] is ok


def test_must_flatten_near_the_close():
    g = _guard()
    assert not g.must_flatten(at(15, 50))
    assert g.must_flatten(at(15, 56))
    assert g.must_flatten(at(16, 30))


# ── Event blackout passthrough ───────────────────────────────────────────

def test_calendar_blackout_blocks_entry():
    g = _guard()
    allowed, reason = g.can_enter(at(11, 0), event_blocked_reason="blocked: CPI in 12min")
    assert not allowed
    assert "CPI" in reason


# ── Session lifecycle ────────────────────────────────────────────────────

def test_new_session_resets_counters_and_halt():
    g = _guard()
    g.record_fill(opened=True)
    g.record_fill(pnl=-3_400.0, closed=True)
    g.can_enter(at(11, 0))
    assert g.halted_reason
    g.start_session(date(2026, 8, 5), 108_400.0)
    assert not g.halted_reason
    assert g.trades_taken == 0
    assert g.realized_pnl == 0.0
    assert g.risk_budget == pytest.approx(1084.0)


def test_start_session_is_idempotent_within_a_day():
    g = _guard()
    g.record_fill(opened=True)
    g.start_session(TODAY, 999_999.0)  # same day -- must not wipe state
    assert g.trades_taken == 1
    assert g.starting_equity == 111_822.0


def test_summary_reports_budget_and_usage():
    g = _guard()
    g.record_fill(opened=True)
    g.record_fill(pnl=-1_677.0, closed=True)
    s = g.summary()
    assert s["risk_budget_per_trade"] == pytest.approx(1118.22)
    assert s["daily_loss_limit"] == pytest.approx(3354.66)
    assert s["daily_loss_used_pct"] == pytest.approx(50.0, abs=0.5)
    assert s["trades_remaining"] == 3
