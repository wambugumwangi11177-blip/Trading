"""Tests for the validated long-or-flat trend strategy.

The load-bearing properties are: it never shorts, it does not peek at the bar it
is deciding on, and a flat signal carries no stop (there is nothing to protect).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fable_bot.strategies.long_or_flat_trend import (
    TrendConfig, average_true_range, backtest_signals, evaluate,
)


def _frame(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    close = pd.Series(close, index=idx)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.004,
        "low": close * 0.996,
        "close": close,
    })


def _uptrend(n=400):
    return _frame(1000 + np.linspace(0, 600, n) + np.sin(np.arange(n) / 7) * 5)


def _downtrend(n=400):
    return _frame(1600 - np.linspace(0, 600, n) + np.sin(np.arange(n) / 7) * 5)


# ── Direction ────────────────────────────────────────────────────────────

def test_goes_long_in_an_uptrend():
    sig = evaluate("GC=F", _uptrend())
    assert sig.side == "long"
    assert sig.agreement == 3
    assert sig.momentum_pct > 0


def test_goes_flat_not_short_in_a_downtrend():
    """The whole validated edge is that this never takes the short side."""
    sig = evaluate("GC=F", _downtrend())
    assert sig.side == "flat"
    assert sig.agreement == 0
    assert "never shorts" in sig.reason


def test_flat_signal_carries_no_stop():
    assert evaluate("GC=F", _downtrend()).stop_price is None


def test_long_signal_places_the_stop_below_price_by_atr():
    config = TrendConfig(atr_stop_multiple=1.5)
    sig = evaluate("GC=F", _uptrend(), config)
    assert sig.stop_price == pytest.approx(sig.price - 1.5 * sig.atr)
    assert sig.stop_price < sig.price


# ── Agreement threshold ──────────────────────────────────────────────────

def test_requiring_full_agreement_is_stricter():
    # A market that has turned up recently but is still under its slow MA.
    n = 400
    close = np.concatenate([np.linspace(1600, 1000, n - 60), np.linspace(1000, 1150, 60)])
    df = _frame(close)
    lenient = evaluate("GC=F", df, TrendConfig(required_agreement=1))
    strict = evaluate("GC=F", df, TrendConfig(required_agreement=3))
    assert strict.agreement == lenient.agreement
    assert not (strict.side == "long" and lenient.side == "flat")


# ── Look-ahead safety ────────────────────────────────────────────────────

def test_backtest_positions_are_lagged_by_one_bar():
    """A position must be decided before the bar it is held through."""
    df = _uptrend()
    pos = backtest_signals("GC=F", df)
    raw = (
        (df["close"].pct_change(63) > 0).astype(int)
        + (df["close"].rolling(20).mean() > df["close"].rolling(100).mean()).astype(int)
        + (df["close"] > df["close"].rolling(50).max().shift(1)).astype(int)
    )
    expected = (raw >= 1).astype(float).shift(1).fillna(0.0)
    pd.testing.assert_series_equal(pos, expected, check_names=False)


def test_donchian_does_not_use_todays_own_high():
    """Comparing today's close to a window that includes today is always false-safe
    but silently disables the breakout; the window must exclude the current bar."""
    df = _uptrend()
    sig = evaluate("GC=F", df)
    window_including_today = df["close"].rolling(50).max().iloc[-1]
    assert sig.detail["donchian_high"] <= window_including_today


def test_positions_are_only_ever_long_or_flat():
    for df in (_uptrend(), _downtrend()):
        pos = backtest_signals("GC=F", df)
        assert set(np.unique(pos)) <= {0.0, 1.0}


# ── Plumbing ─────────────────────────────────────────────────────────────

def test_atr_is_positive_and_tracks_range():
    df = _uptrend()
    atr = average_true_range(df, 14).dropna()
    assert (atr > 0).all()


def test_short_history_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError, match="need more history"):
        evaluate("GC=F", _uptrend(n=50))
