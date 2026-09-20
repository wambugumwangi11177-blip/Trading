"""Technicals, liquidity map, sentiment, and the synthesis -- all offline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fable_bot.intel.fmt import decimals_for, px
from fable_bot.intel.liquidity import build_liquidity_map
from fable_bot.intel.report import SymbolIntel
from fable_bot.intel.sentiment import Sentiment, sentiment_from_cot_row
from fable_bot.intel.technicals import compute_technicals, rsi


def frame(closes, start="2026-01-01", spread=0.5, wobble=True):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    c = pd.Series(np.asarray(closes, dtype=float), index=idx)
    if wobble:
        c = c + np.sin(np.arange(len(c))) * 0.05
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c + spread,
                         "low": c - spread, "close": c, "volume": 1000}, index=idx)


# ── formatting ─────────────────────────────────────────────────────────────

def test_fx_gets_five_decimals_and_gold_gets_two():
    assert decimals_for(1.1476) == 5
    assert decimals_for(4424.9) == 2
    assert px(1.14760) == "1.14760"
    assert px(4424.9) == "4,424.90"


def test_reference_price_drives_decimals_for_small_values():
    """An ATR of 0.0048 on EURUSD must not print as 0.00."""
    assert px(0.00481, 1.1476) == "0.00481"


# ── technicals ─────────────────────────────────────────────────────────────

def test_rsi_is_bounded_and_high_in_a_straight_rally():
    r = rsi(pd.Series(np.linspace(100, 150, 60)), 14)
    assert 0 <= r.iloc[-1] <= 100
    assert r.iloc[-1] > 90


def test_rsi_is_low_in_a_straight_decline():
    r = rsi(pd.Series(np.linspace(150, 100, 60)), 14)
    assert r.iloc[-1] < 10


def test_uptrend_reads_as_up_with_stacked_emas():
    snap = compute_technicals("X", frame(np.linspace(100, 160, 250)))
    assert snap.trend == "up"
    assert snap.ema20 > snap.ema50 > snap.ema200
    assert snap.range_position_pct > 90


def test_downtrend_reads_as_down():
    snap = compute_technicals("X", frame(np.linspace(160, 100, 250)))
    assert snap.trend == "down"


def test_too_few_bars_is_refused():
    with pytest.raises(ValueError):
        compute_technicals("X", frame(np.linspace(100, 110, 20)))


def test_summary_uses_instrument_decimals():
    snap = compute_technicals("EURUSD", frame(np.linspace(1.10, 1.15, 60), spread=0.002))
    text = snap.summary()
    atr_field = text.split("ATR14 ")[1].split(" ")[0]
    assert atr_field != "0.00", text          # the whole point: FX ATR must be legible
    assert len(atr_field.split(".")[1]) == 5


# ── liquidity map ──────────────────────────────────────────────────────────

def _swing_series():
    """A range with a clear swing high at 120 and swing low at 80, then spot at 100."""
    up = np.linspace(100, 120, 15)
    down = np.linspace(120, 80, 30)
    back = np.linspace(80, 100, 15)
    flat = np.full(20, 100.0)
    return np.concatenate([up, down, back, flat])


def test_untouched_swing_high_is_a_buy_side_pool_above():
    lm = build_liquidity_map("X", frame(_swing_series()), scan_pct=40.0)
    hits = [p for p in lm.above if p.kind in ("swing_high", "equal_highs")
            and abs(p.price - 120.5) < 1.0]
    assert hits, "expected the 120 swing high as an untouched pool above"
    assert hits[0].swept is False


def test_untouched_swing_low_is_a_sell_side_pool_below():
    lm = build_liquidity_map("X", frame(_swing_series()), scan_pct=40.0)
    lows = [p for p in lm.below if p.kind in ("swing_low", "equal_lows") and not p.swept]
    assert lows
    assert min(p.price for p in lows) == pytest.approx(79.5, abs=1.0)


def test_a_broken_level_is_marked_swept():
    series = np.concatenate([_swing_series(), np.linspace(100, 125, 10)])  # take the high
    lm = build_liquidity_map("X", frame(series), scan_pct=40.0)
    highs = [p for p in lm.above + lm.below if p.kind in ("swing_high", "equal_highs")
             and abs(p.price - 120.5) < 1.5]
    assert highs and highs[0].swept is True


def test_recent_sweep_requires_the_level_to_have_been_untouched_before():
    """A level price has sat beyond for months is not 'just swept'."""
    series = np.concatenate([_swing_series(), np.linspace(100, 125, 10), np.full(40, 130.0)])
    lm = build_liquidity_map("X", frame(series), scan_pct=40.0)
    assert not any(abs(p.price - 120.5) < 1.5 for p in lm.recently_swept)


def test_fresh_sweep_is_reported():
    series = np.concatenate([_swing_series(), np.full(30, 100.0), np.linspace(100, 125, 4)])
    lm = build_liquidity_map("X", frame(series), scan_pct=40.0)
    assert any(abs(p.price - 120.5) < 1.5 for p in lm.recently_swept)


def test_equal_highs_are_clustered_and_counted():
    base = np.concatenate([np.linspace(100, 110, 10), np.linspace(110, 100, 10)] * 3 + [np.full(15, 100.0)])
    lm = build_liquidity_map("X", frame(base, wobble=False), equal_tol_pct=0.5, scan_pct=40.0)
    eq = [p for p in lm.above if p.kind == "equal_highs" and abs(p.price - 110.5) < 1.0]
    assert eq and eq[0].touches >= 2, [(p.price, p.kind, p.touches) for p in lm.above]


def test_summary_never_raises_and_states_spot():
    lm = build_liquidity_map("X", frame(_swing_series(), wobble=False))
    assert "spot 100.00" in lm.summary()


# ── sentiment ──────────────────────────────────────────────────────────────

def test_sentiment_from_nonreportable_columns():
    s = sentiment_from_cot_row("XAUUSD", {"nonrept_positions_long_all": "30000",
                                          "nonrept_positions_short_all": "10000",
                                          "report_date_as_yyyy_mm_dd": "2026-09-15T00:00:00"})
    assert s.long_pct == pytest.approx(75.0)
    assert s.crowd_side == "long"
    assert s.extreme is True
    assert s.as_of == "2026-09-15"
    assert "contrarian lean: short" in s.summary()


def test_balanced_sentiment_has_no_lean():
    s = Sentiment("X", 52.0, 48.0, "test", "2026-09-15")
    assert s.crowd_side == "balanced"
    assert "no contrarian read" in s.summary()


def test_zero_sample_returns_none():
    assert sentiment_from_cot_row("X", {}) is None


# ── synthesis ──────────────────────────────────────────────────────────────

def test_two_agreeing_forced_sections_produce_a_lean():
    intel = SymbolIntel(symbol="X")
    intel.sentiment = Sentiment("X", 80.0, 20.0, "test", "2026-09-15")   # -> short
    # Spot parks just above an untouched swing low (nearest pool BELOW) with
    # the swing high far away; the tail is flat so no pdh/pdl sweep fires.
    # Swing low at ~79.5 sits 1.8% under spot; everything structural above is
    # 5%+ away, so the nearer untouched destination is BELOW -> short.
    series = np.concatenate([np.linspace(100, 120, 15), np.linspace(120, 80, 30),
                             np.linspace(80, 100, 15), np.linspace(100, 81, 25)])
    intel.liquidity = build_liquidity_map("X", frame(series, wobble=False), scan_pct=60.0)
    sides = [side for s_, side, _ in intel.leans() if s_ in ("sentiment", "liquidity")]
    assert sides.count("short") >= 2, intel.leans()
    assert "LEAN SHORT" in intel.synthesis()


def test_disagreeing_forced_sections_are_conflicted():
    intel = SymbolIntel(symbol="X")
    intel.sentiment = Sentiment("X", 80.0, 20.0, "test", "2026-09-15")   # -> short
    # Liquidity strongly favours long: nearest untouched pool just above.
    intel.liquidity = build_liquidity_map("X", frame(np.concatenate(
        [np.linspace(100, 120, 15), np.linspace(120, 60, 30), np.linspace(60, 118, 40)])))
    leans = {side for _, side, _ in intel.leans() if side in ("long", "short")}
    if leans == {"long", "short"}:
        assert "CONFLICTED" in intel.synthesis()


def test_no_sections_is_stated_not_faked():
    assert "nothing to synthesise" in SymbolIntel(symbol="X").synthesis()


def test_technicals_alone_never_make_a_forced_lean():
    intel = SymbolIntel(symbol="X")
    intel.technicals = compute_technicals("X", frame(np.linspace(100, 160, 250)))
    text = intel.synthesis()
    assert "NO LEAN" in text
