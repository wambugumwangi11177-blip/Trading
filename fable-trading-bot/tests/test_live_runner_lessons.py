"""Integration tests for the learning hooks in live_runner.run_signal_check.

Broker, market data, and strategies are stubbed; what gets exercised is the
safety wiring: untradable series never reach submit_order, lessons gate
entries (never exits), out-of-bounds sizes are refused with evidence, and the
intent->submit->verify ordering survives a broker failure.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fable_bot.broker.base import AccountSnapshot, HistoryOrder, OrderResult
from fable_bot.broker.tradingview_adapter import TradingViewError
from fable_bot.config import INSTRUMENTS
from fable_bot.journal import Journal
from fable_bot.lessons import LessonBook
from fable_bot.strategies.base import Signal

UNTRADABLE_TV = {i.tv_symbol for i in INSTRUMENTS if not i.tradable}
UNTRADABLE_SYMBOLS = [i.symbol for i in INSTRUMENTS if not i.tradable]


def _events(journal: Journal) -> list[dict]:
    return [json.loads(line) for line in
            journal.path.read_text(encoding="utf-8").strip().splitlines() if line.strip()]


def test_untradable_series_are_skipped_and_never_reach_the_broker(cycle):
    actions = cycle.lr.run_signal_check(journal=cycle.journal)
    events = _events(cycle.journal)

    assert len(UNTRADABLE_SYMBOLS) == 6
    skips = {(e["symbol"], e["gate"]) for e in events
             if e["kind"] == "skip" and e.get("gate") == "tradability"}
    assert {s for s, _ in skips} == set(UNTRADABLE_SYMBOLS)

    # Not fetched, not ordered: the NDX/IXIC rejection class is closed off.
    assert not set(UNTRADABLE_SYMBOLS) & set(cycle.requested[0])
    assert cycle.broker.submitted == []
    assert actions == []


def test_lessons_loaded_event_reports_the_gate(cycle):
    cycle.lr.run_signal_check(journal=cycle.journal)
    loaded = next(e for e in _events(cycle.journal) if e["kind"] == "lessons_loaded")
    assert loaded["active"] == 0
    assert "sizing_bounds" in loaded and "verification" in loaded


def test_block_symbol_lesson_gates_the_entry(cycle):
    LessonBook(memory_dir=cycle.memory_dir).add(
        "block_symbol", {"symbols": ["AMEX:SPY"]}, source="seeded", incident="test")
    cycle.signals["SPY"] = ("long", 110.0, 108.0)

    cycle.lr.run_signal_check(journal=cycle.journal)

    skips = [e for e in _events(cycle.journal)
             if e["kind"] == "skip" and e.get("gate") == "lesson"]
    assert len(skips) == 1 and skips[0]["symbol"] == "AMEX:SPY"
    assert skips[0]["lesson_id"]
    assert cycle.broker.submitted == []


def test_out_of_bounds_size_is_refused_with_evidence(cycle):
    # Notional capped at 10% of equity: the stub's $55k SPY order is 55%.
    LessonBook(memory_dir=cycle.memory_dir).add(
        "sizing_notional_bounds", {"max_notional_pct_equity": 0.1},
        source="seeded", incident="test")
    cycle.signals["SPY"] = ("long", 110.0, 108.0)

    actions = cycle.lr.run_signal_check(journal=cycle.journal)

    events = _events(cycle.journal)
    skip = next(e for e in events if e["kind"] == "skip" and e.get("gate") == "sizing_sanity")
    evidence = next(e for e in events if e["kind"] == "sizing_out_of_bounds")
    assert skip["symbol"] == evidence["symbol"] == "AMEX:SPY"
    assert evidence["notional_pct_equity"] > 10.0
    assert cycle.broker.submitted == []
    assert any(a.startswith("SKIP AMEX:SPY") for a in actions)


def test_in_bounds_order_flows_intent_then_verified_order(cycle):
    # Submit returns no usable id (the lost-response case); verification
    # supplies the broker row instead.
    cycle.broker.order_id = None
    cycle.broker.verify_row = HistoryOrder(
        id="777", symbol="AMEX:SPY", side="buy", type="market",
        status="filled", qty=500, placed_at="2026-08-31T13:31:00Z")
    cycle.signals["SPY"] = ("long", 110.0, 108.0)

    actions = cycle.lr.run_signal_check(journal=cycle.journal)

    events = _events(cycle.journal)
    intent = next(e for e in events if e["kind"] == "order_intent")
    order = next(e for e in events if e["kind"] == "order")
    assert events.index(intent) < events.index(order)
    assert order["intent_id"] == intent["intent_id"]
    assert order["verified"] is True
    assert order["order_id"] == "777"  # from the history row, not the submit response
    assert any(a.startswith("ORDER AMEX:SPY") and "verified" in a for a in actions)


def test_venue_rejection_keeps_the_intent_and_leaves_evidence(cycle):
    cycle.broker.submit_exc = TradingViewError(
        "trade buy AMEX:SPY failed: Invalid symbol", venue_rejection=True)
    cycle.signals["SPY"] = ("long", 110.0, 108.0)

    cycle.lr.run_signal_check(journal=cycle.journal)

    events = _events(cycle.journal)
    kinds = [e["kind"] for e in events]
    # The intent is journalled before submission even when the broker refuses:
    # that ordering is what lets reconciliation prove what happened.
    assert "order_intent" in kinds and "order_rejected" in kinds
    assert kinds.index("order_intent") < kinds.index("order_rejected")
    rejected = next(e for e in events if e["kind"] == "order_rejected")
    assert rejected["symbol"] == "AMEX:SPY"
    assert rejected["intent_id"]
    assert "order" not in kinds


def test_connectivity_failure_is_not_learned_as_a_rejection(cycle):
    """The app being down must produce order_failed, not order_rejected:
    only the latter becomes a 30-day halt lesson for the symbol."""
    cycle.broker.submit_exc = TradingViewError("Cannot reach TradingView Desktop")
    cycle.signals["SPY"] = ("long", 110.0, 108.0)

    cycle.lr.run_signal_check(journal=cycle.journal)

    kinds = [e["kind"] for e in _events(cycle.journal)]
    assert "order_failed" in kinds
    assert "order_rejected" not in kinds


def test_dry_run_leaves_no_intent_or_verification(cycle):
    cycle.signals["SPY"] = ("long", 110.0, 108.0)
    actions = cycle.lr.run_signal_check(dry_run=True, journal=cycle.journal)
    kinds = [e["kind"] for e in _events(cycle.journal)]
    assert "order" in kinds and "order_intent" not in kinds
    assert cycle.broker.submitted == []
    assert any(a.startswith("WOULD ORDER") for a in actions)
