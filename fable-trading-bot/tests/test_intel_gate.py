"""The positioning gate: four refusals, fail-open, entries only."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fable_bot.intel.cot import COMMODITY, CotReport
from fable_bot.intel.gamma import GammaProfile
from fable_bot.intel.liquidity import LiquidityMap, LiquidityPool
from fable_bot.intel.report import SymbolIntel
from fable_bot.intel.sentiment import Sentiment
from fable_bot.risk.intel_gate import evaluate_entry

NOW = datetime(2026, 9, 21, tzinfo=timezone.utc)


def gamma(net):
    return GammaProfile(symbol="X", spot=100.0, as_of=NOW, by_strike=[], net_gex=net,
                        gamma_flip=None, call_wall=None, put_wall=None)


def cot(percentile):
    return CotReport(market="X", report_date="2026-09-15", open_interest=1000.0,
                     producer_long=0, producer_short=0, managed_money_long=600, managed_money_short=400,
                     swap_long=0, swap_short=0, days_stale=5, report_type=COMMODITY,
                     managed_money_net_percentile=percentile, history_weeks=156)


def pool(price, kind, side, swept=False, spot=100.0):
    return LiquidityPool(price=price, kind=kind, side=side, touches=1, bar="2026-09-10",
                         swept=swept, distance_pct=(price - spot) / spot * 100)


def entry(side="long", strategy="trend_following", intel=None, price=100.0, stop=97.0):
    return evaluate_entry(side=side, strategy=strategy, entry_price=price, stop_price=stop, intel=intel)


# ── fail-open ──────────────────────────────────────────────────────────────

def test_no_coverage_allows_and_says_so():
    d = entry(intel=None)
    assert d.allowed is True
    assert d.evidence == {"coverage": False}


def test_empty_intel_allows():
    assert entry(intel=SymbolIntel(symbol="X")).allowed is True


# ── 1. trapped with the crowd ─────────────────────────────────────────────

def test_extreme_crowd_on_same_side_is_refused():
    intel = SymbolIntel(symbol="X", sentiment=Sentiment("X", 78.0, 22.0, "test", "2026-09-15"))
    d = entry(side="long", intel=intel)
    assert d.allowed is False
    assert "crowd" in d.reason


def test_extreme_crowd_on_the_other_side_is_fine():
    intel = SymbolIntel(symbol="X", sentiment=Sentiment("X", 78.0, 22.0, "test", "2026-09-15"))
    assert entry(side="short", intel=intel).allowed is True


def test_crowd_below_threshold_is_fine():
    intel = SymbolIntel(symbol="X", sentiment=Sentiment("X", 70.0, 30.0, "test", "2026-09-15"))
    assert entry(side="long", intel=intel).allowed is True


# ── 2. against the synthesis ──────────────────────────────────────────────

def _short_leaning_intel():
    """COT crowded long + retail crowded long (not extreme): two forced shorts."""
    return SymbolIntel(symbol="X",
                       cot=cot(88.0),
                       sentiment=Sentiment("X", 68.0, 32.0, "test", "2026-09-15"))


def test_entry_against_an_agreed_lean_is_refused():
    d = entry(side="long", intel=_short_leaning_intel())
    assert d.allowed is False
    assert "lean SHORT" in d.reason


def test_entry_with_the_lean_is_allowed():
    assert entry(side="short", intel=_short_leaning_intel()).allowed is True


def test_one_forced_section_alone_does_not_refuse():
    intel = SymbolIntel(symbol="X", cot=cot(88.0))
    assert entry(side="long", intel=intel).allowed is True


def test_conflicted_sections_let_the_strategy_decide():
    intel = SymbolIntel(symbol="X", cot=cot(88.0),                       # short
                        sentiment=Sentiment("X", 30.0, 70.0, "test", "2026-09-15"))  # long
    assert entry(side="long", intel=intel).allowed is True
    assert entry(side="short", intel=intel).allowed is True


# ── 3. strategy vs gamma regime ───────────────────────────────────────────

def test_breakout_in_long_gamma_is_refused():
    intel = SymbolIntel(symbol="X", gamma=gamma(+5e6))
    d = entry(strategy="momentum", intel=intel)
    assert d.allowed is False and "LONG gamma" in d.reason


def test_breakout_in_short_gamma_is_allowed():
    intel = SymbolIntel(symbol="X", gamma=gamma(-5e6))
    assert entry(strategy="momentum", intel=intel).allowed is True


def test_fade_in_short_gamma_is_refused():
    intel = SymbolIntel(symbol="X", gamma=gamma(-5e6))
    d = entry(strategy="mean_reversion", intel=intel)
    assert d.allowed is False and "SHORT gamma" in d.reason


def test_fade_in_long_gamma_is_allowed():
    intel = SymbolIntel(symbol="X", gamma=gamma(+5e6))
    assert entry(strategy="mean_reversion", intel=intel).allowed is True


def test_trend_strategies_are_not_gamma_gated():
    for regime in (+5e6, -5e6):
        intel = SymbolIntel(symbol="X", gamma=gamma(regime))
        assert entry(strategy="trend_following", intel=intel).allowed is True
        assert entry(strategy="long_or_flat_trend", intel=intel).allowed is True


# ── 4. stop inside a pool ─────────────────────────────────────────────────

def test_stop_on_an_untouched_pool_is_refused():
    lm = LiquidityMap(symbol="X", spot=100.0, as_of_bar="2026-09-18",
                      below=[pool(97.1, "swing_low", "below")])
    d = entry(side="long", price=100.0, stop=97.0, intel=SymbolIntel(symbol="X", liquidity=lm))
    assert d.allowed is False and "pool" in d.reason
    assert d.evidence["pool_at_stop"]["kind"] == "swing_low"


def test_stop_clear_of_pools_is_allowed():
    lm = LiquidityMap(symbol="X", spot=100.0, as_of_bar="2026-09-18",
                      below=[pool(95.0, "swing_low", "below")])
    assert entry(side="long", price=100.0, stop=97.0,
                 intel=SymbolIntel(symbol="X", liquidity=lm)).allowed is True


def test_swept_pool_does_not_count():
    lm = LiquidityMap(symbol="X", spot=100.0, as_of_bar="2026-09-18",
                      below=[pool(97.1, "swing_low", "below", swept=True)])
    assert entry(side="long", price=100.0, stop=97.0,
                 intel=SymbolIntel(symbol="X", liquidity=lm)).allowed is True


def test_short_entry_checks_pools_above():
    lm = LiquidityMap(symbol="X", spot=100.0, as_of_bar="2026-09-18",
                      above=[pool(103.1, "equal_highs", "above")])
    d = entry(side="short", price=100.0, stop=103.0, intel=SymbolIntel(symbol="X", liquidity=lm))
    assert d.allowed is False


# ── wiring: entries only, exits never ─────────────────────────────────────

def test_gate_blocks_an_entry_in_the_runner(cycle):
    cycle.enable_intel_gate()
    cycle.intel["SPY"] = SymbolIntel(symbol="SPY",
                                     sentiment=Sentiment("SPY", 80.0, 20.0, "test", "2026-09-15"))
    cycle.signals["SPY"] = ("long", 110.0, 108.0)
    cycle.lr.run_signal_check(dry_run=True, journal=cycle.journal)
    events = cycle.events()
    assert any(e.get("gate") == "intel" for e in events)
    assert not [e for e in events if e["kind"] == "order"]


def test_gate_never_blocks_an_exit(cycle):
    cycle.enable_intel_gate()
    cycle.intel["SPY"] = SymbolIntel(symbol="SPY",
                                     sentiment=Sentiment("SPY", 80.0, 20.0, "test", "2026-09-15"))
    cycle.open_position("AMEX:SPY", "long")
    cycle.signals["SPY"] = ("flat", 110.0, 110.0)
    cycle.lr.run_signal_check(dry_run=False, journal=cycle.journal)
    assert "AMEX:SPY" in cycle.closed
    assert not any(e.get("gate") == "intel" for e in cycle.events())


def test_gate_journals_its_evidence_when_allowing(cycle):
    cycle.enable_intel_gate()
    cycle.intel["SPY"] = SymbolIntel(symbol="SPY", gamma=gamma(+1e6))
    cycle.signals["SPY"] = ("long", 110.0, 108.0)
    cycle.lr.run_signal_check(dry_run=True, journal=cycle.journal)
    gate_events = [e for e in cycle.events() if e["kind"] == "intel_gate"]
    assert gate_events and gate_events[0]["allowed"] is True
    assert gate_events[0]["gamma_regime"] == "long_gamma"
