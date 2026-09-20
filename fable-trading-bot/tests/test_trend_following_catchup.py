"""Catch-up reconciliation for the trend-following strategy.

Instruments added to the universe mid-trend (the FX majors, 2026-08-25) carry a
simulated position whose entry bar was never traded. catch_up_signal must
reprice exactly that simulated state on the last bar -- same side, last close,
trailing stop as generate_signals maintains it -- so the live runner can
reconcile the book instead of skipping a stale signal forever.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fable_bot.strategies import build_strategy
from fable_bot.strategies.indicators import atr
from fable_bot.strategies.trend_following import TrendFollowingStrategy


@pytest.fixture
def fx_like_df():
    """Uptrend with swings, then a regime change -- enough for the 50-bar EMA
    and for SMC confluence to fire on noisy swings."""
    rng = np.random.default_rng(11)
    up = np.linspace(1.05, 1.18, 220)
    down = np.linspace(1.18, 1.10, 130)
    close = np.concatenate([up, down]) * (1 + rng.normal(0, 0.0022, 350))
    index = pd.date_range("2024-01-01", periods=len(close), freq="D")
    close = pd.Series(close, index=index)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.003,
        "low": close * 0.997,
        "close": close,
        "volume": 100_000,
    }, index=index)


def test_empty_frame_returns_none():
    assert TrendFollowingStrategy().catch_up_signal("EURUSD=X", pd.DataFrame()) is None


def test_catch_up_reprices_the_simulated_state(fx_like_df):
    strategy = TrendFollowingStrategy()
    signals = strategy.generate_signals("EURUSD=X", fx_like_df)
    assert signals, "fixture produced no signals; it cannot pin anything"

    catchup = strategy.catch_up_signal("EURUSD=X", fx_like_df)
    assert catchup is not None
    last = signals[-1]

    assert catchup.date == fx_like_df.index[-1]
    assert catchup.side == last.side
    assert catchup.price == pytest.approx(float(fx_like_df["close"].iloc[-1]))

    if last.side == "flat":
        assert catchup.stop_price == catchup.price
    else:
        held = fx_like_df.loc[last.date:]
        a = float(atr(fx_like_df, strategy.atr_period).iloc[-1])
        extreme = float(held["close"].max()) if last.side == "long" else float(held["close"].min())
        expected = extreme - strategy.atr_trail_multiple * a if last.side == "long" \
            else extreme + strategy.atr_trail_multiple * a
        assert catchup.stop_price == pytest.approx(expected)
        assert "catch-up" in catchup.reason


def test_registered_strategy_exposes_catch_up():
    assert isinstance(build_strategy("trend_following"), TrendFollowingStrategy)
    assert build_strategy("trend_following").catch_up_signal("X", pd.DataFrame()) is None
