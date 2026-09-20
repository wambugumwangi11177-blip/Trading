"""The registry adapter for the validated gold trend strategy.

`config.GOLD_FUTURES` names this strategy, but the registry had no entry for it,
so `build_strategy("long_or_flat_trend")` raised KeyError and the one strategy
that cleared validation could not be backtested or traded.

The risk in fixing that is subtler than the bug: a second implementation of a
validated strategy that quietly disagrees with the first is worse than no
implementation at all. So the tests below pin the adapter to `evaluate()` --
the function the published numbers came from -- bar by bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fable_bot.strategies import STRATEGY_REGISTRY, build_strategy
from fable_bot.strategies.long_or_flat_trend import (
    LongOrFlatTrendStrategy,
    TrendConfig,
    evaluate,
    vote_series,
)


@pytest.fixture
def trending_df():
    """Long enough for the 100-bar MA plus a regime change to trade against."""
    rng = np.random.default_rng(7)
    up = np.linspace(100, 160, 200)
    down = np.linspace(160, 110, 150)
    close = np.concatenate([up, down]) * (1 + rng.normal(0, 0.004, 350))
    index = pd.date_range("2024-01-01", periods=len(close), freq="D")
    close = pd.Series(close, index=index)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.006,
        "low": close * 0.994,
        "close": close,
        "volume": 1_000_000,
    }, index=index)


def test_the_strategy_is_registered():
    assert "long_or_flat_trend" in STRATEGY_REGISTRY
    assert isinstance(build_strategy("long_or_flat_trend"), LongOrFlatTrendStrategy)


def test_config_gold_futures_can_now_be_built():
    """The exact lookup that used to raise KeyError."""
    from fable_bot.config import GOLD_FUTURES

    for inst in GOLD_FUTURES:
        assert build_strategy(inst.strategy) is not None


def test_every_signal_agrees_with_evaluate_on_its_own_bar(trending_df):
    """The pin: for each emitted signal, `evaluate` on history up to that bar
    must reach the same side."""
    strategy = LongOrFlatTrendStrategy()
    signals = strategy.generate_signals("MGC=F", trending_df)
    assert signals, "fixture produced no signals; it cannot pin anything"

    for signal in signals:
        position = trending_df.index.get_loc(signal.date)
        truth = evaluate("MGC=F", trending_df.iloc[: position + 1])
        assert signal.side == truth.side, (
            f"bar {signal.date}: adapter said {signal.side}, evaluate said {truth.side}"
        )
        assert signal.price == pytest.approx(truth.price)
        if signal.side == "long":
            assert signal.stop_price == pytest.approx(truth.stop_price)


def test_reconstructed_position_matches_evaluate_on_every_bar(trending_df):
    """Stronger than the above: walk the whole series, not just the turns.

    A signal list can agree on the bars it emits and still be wrong about the
    bars it stays silent on -- a missed exit looks like nothing at all.
    """
    strategy = LongOrFlatTrendStrategy()
    signals = {s.date: s.side for s in strategy.generate_signals("MGC=F", trending_df)}
    config = TrendConfig()
    min_bars = max(config.ma_slow, config.momentum_days, config.donchian_days) + 1

    position = "flat"
    for i in range(min_bars, len(trending_df)):
        date = trending_df.index[i]
        position = signals.get(date, position)
        expected = evaluate("MGC=F", trending_df.iloc[: i + 1]).side
        assert position == expected, f"bar {date}: held {position}, evaluate said {expected}"


def test_it_never_shorts(trending_df):
    """Long-or-flat is the whole premise; a short would be a different strategy."""
    sides = {s.side for s in LongOrFlatTrendStrategy().generate_signals("MGC=F", trending_df)}
    assert "short" not in sides
    assert sides <= {"long", "flat"}


def test_signals_alternate_and_never_repeat_a_side(trending_df):
    """Entries and exits must interleave, or the backtester would double-enter."""
    sides = [s.side for s in LongOrFlatTrendStrategy().generate_signals("MGC=F", trending_df)]
    for earlier, later in zip(sides, sides[1:]):
        assert earlier != later, f"repeated {earlier} signal without an intervening flip"


def test_stop_is_one_and_a_half_atr_below_entry(trending_df):
    from fable_bot.strategies.long_or_flat_trend import average_true_range

    config = TrendConfig()
    atr = average_true_range(trending_df, config.atr_days)
    for signal in LongOrFlatTrendStrategy().generate_signals("MGC=F", trending_df):
        if signal.side != "long":
            continue
        i = trending_df.index.get_loc(signal.date)
        expected = signal.price - config.atr_stop_multiple * atr.iloc[i]
        assert signal.stop_price == pytest.approx(expected)
        assert signal.stop_price < signal.price


def test_backtest_signals_still_thresholds_the_shared_votes(trending_df):
    """`backtest_signals` was refactored onto the shared vote series; it must
    still be the one-bar-shifted exposure the validation harness expects."""
    from fable_bot.strategies.long_or_flat_trend import backtest_signals

    config = TrendConfig()
    expected = (vote_series(trending_df, config) >= config.required_agreement) \
        .astype(float).shift(1).fillna(0.0)
    pd.testing.assert_series_equal(backtest_signals("MGC=F", trending_df), expected)


def test_the_universe_now_backtests_end_to_end(trending_df):
    """What actually broke: simulate() over GOLD_FUTURES raised KeyError."""
    from fable_bot.backtester import simulate
    from fable_bot.config import GOLD_FUTURES

    result = simulate({inst.symbol: trending_df for inst in GOLD_FUTURES}, GOLD_FUTURES)
    assert result.num_trades >= 0  # the point is that it completes at all


def test_catch_up_signal_agrees_with_evaluate_on_the_last_bar(trending_df):
    """A missed flip reconciles to exactly what evaluate() says now: same side,
    repriced on the evaluated bar, stop recomputed from its ATR."""
    strategy = LongOrFlatTrendStrategy()
    truth = evaluate("MGC=F", trending_df)
    catchup = strategy.catch_up_signal("MGC=F", trending_df)
    assert catchup is not None
    assert catchup.date == trending_df.index[-1]
    assert catchup.side == truth.side
    assert catchup.price == pytest.approx(truth.price)
    if truth.side == "long":
        assert catchup.stop_price == pytest.approx(truth.stop_price)
        assert catchup.stop_price < catchup.price
    else:
        assert catchup.stop_price == catchup.price


def test_catch_up_is_opt_in_for_other_strategies():
    """Only state-based strategies may reconcile; the rest keep the historical
    stale-signal skip."""
    for name in ("mean_reversion", "momentum"):
        assert build_strategy(name).catch_up_signal("SYM", pd.DataFrame()) is None
