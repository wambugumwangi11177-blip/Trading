"""Book-wide risk ceilings.

Per-trade sizing bounded one order and nothing bounded the book: 2026-09-08
sent eight entries in two minutes (~8% of equity at risk) and 2026-09-11 held
seven positions at once. These tests pin both ceilings, and the small-account
sizing window that made a $25 balance untradeable.
"""
from __future__ import annotations

import dataclasses

import pytest

from fable_bot.config import RISK, RiskConfig
from fable_bot.risk.manager import portfolio_gate, sizing_sanity

BOUNDS = {
    "risk_pct_min": 0.25, "risk_pct_max": 1.5,
    "max_notional_pct_equity": 2.0, "min_notional_usd": 100.0,
    "min_notional_pct_equity": 0.10,
}


def cfg(**overrides) -> RiskConfig:
    return dataclasses.replace(RISK, **overrides)


# ── concurrent position ceiling ───────────────────────────────────────────

def test_allows_entry_below_the_position_cap():
    ok, _ = portfolio_gate(open_positions=2, risk_committed=0.0, equity=100_000.0,
                           config=cfg(max_concurrent_positions=3))
    assert ok is True


def test_blocks_entry_at_the_position_cap():
    ok, reason = portfolio_gate(open_positions=3, risk_committed=0.0, equity=100_000.0,
                                config=cfg(max_concurrent_positions=3))
    assert ok is False
    assert "3 position(s) already open" in reason


def test_the_seven_position_book_is_refused():
    """2026-09-11 carried seven at once under the old code."""
    ok, _ = portfolio_gate(open_positions=7, risk_committed=0.0, equity=113_662.0,
                           config=cfg(max_concurrent_positions=3))
    assert ok is False


# ── new-risk-per-run ceiling ──────────────────────────────────────────────

def test_blocks_once_the_run_has_committed_its_risk_budget():
    equity = 113_000.0
    config = cfg(max_concurrent_positions=99, max_new_risk_per_run_pct=2.0)
    # Two entries at ~1% each exhausts the budget.
    ok, reason = portfolio_gate(open_positions=2, risk_committed=equity * 0.02,
                                equity=equity, config=config)
    assert ok is False
    assert "already risked this run" in reason


def test_the_eight_order_burst_stops_after_two():
    """Replays 2026-09-08: eight entries, each sized to ~1% of equity."""
    equity = 115_625.0
    config = cfg(max_concurrent_positions=99, max_new_risk_per_run_pct=2.0)
    committed, sent = 0.0, 0
    for _ in range(8):
        ok, _ = portfolio_gate(open_positions=sent, risk_committed=committed,
                               equity=equity, config=config)
        if not ok:
            break
        committed += equity * 0.01
        sent += 1
    assert sent == 2, f"expected the burst to stop at 2 entries, got {sent}"


def test_zero_disables_a_cap():
    ok, _ = portfolio_gate(open_positions=50, risk_committed=0.0, equity=100_000.0,
                           config=cfg(max_concurrent_positions=0, max_new_risk_per_run_pct=0.0))
    assert ok is True


def test_non_positive_equity_does_not_divide_by_zero():
    ok, _ = portfolio_gate(open_positions=0, risk_committed=5.0, equity=0.0,
                           config=cfg(max_concurrent_positions=3))
    assert ok is True


# ── small-account sizing window ───────────────────────────────────────────

def test_small_account_is_not_structurally_banned():
    """At $25 the absolute $100 floor sat ABOVE the 200% ceiling ($50).

    Every order was refused: not risky, impossible. The floor now scales with
    equity, so a proportionate order is admissible again.
    """
    ok, reason, metrics = sizing_sanity(
        qty=0.02, price=401.0, stop_price=384.0, equity=25.0,
        point_value=1.0, bounds=BOUNDS)
    assert ok is True, reason
    assert metrics["notional"] == pytest.approx(8.02, abs=0.01)


def test_small_account_still_refuses_true_dust():
    ok, reason, _ = sizing_sanity(
        qty=0.0001, price=401.0, stop_price=384.0, equity=25.0,
        point_value=1.0, bounds=BOUNDS)
    assert ok is False
    assert "below minimum" in reason


def test_large_account_floor_is_unchanged_at_100_dollars():
    """The scaling must be inert at six figures: min(100, 10% of 113k) = 100."""
    _, _, metrics = sizing_sanity(
        qty=1.0, price=150.0, stop_price=140.0, equity=113_000.0,
        point_value=1.0, bounds=BOUNDS)
    assert metrics["min_notional_applied"] == pytest.approx(100.0)


def test_oversized_notional_is_still_refused():
    """EURGBP at 587% of equity, 2026-09-02."""
    ok, reason, _ = sizing_sanity(
        qty=786_000, price=0.8574, stop_price=0.8589, equity=114_659.0,
        point_value=1.0, bounds=BOUNDS)
    assert ok is False
    assert "of equity" in reason


def test_unsatisfiable_bounds_are_reported_as_such():
    tight = {**BOUNDS, "min_notional_pct_equity": 1.0, "max_notional_pct_equity": 0.5}
    ok, reason, _ = sizing_sanity(qty=1.0, price=10.0, stop_price=9.0, equity=25.0,
                                  point_value=1.0, bounds=tight)
    assert ok is False
    assert "unsatisfiable" in reason
