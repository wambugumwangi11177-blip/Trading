"""Trend following strategy for gold and oil (GLD / USO proxies).

Bias: 20/50 EMA crossover defines the prevailing trend.
Entry: only taken *with* the trend, on a pullback into an SMC confluence zone
(liquidity sweep or FVG retest) rather than blindly on the crossover itself --
i.e. buy the dip in an uptrend, sell the rally in a downtrend.
Exit: trend flips (fast EMA crosses the slow EMA) or an ATR trailing stop, whichever hits first.
"""
from __future__ import annotations

import pandas as pd

from . import smc
from .base import Side, Signal, Strategy
from .indicators import atr, ema


class TrendFollowingStrategy(Strategy):
    name = "trend_following"

    def __init__(self, fast_period: int = 20, slow_period: int = 50, atr_period: int = 14,
                 atr_trail_multiple: float = 2.5, swing_window: int = 5,
                 confluence_lookback: int = 3):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_trail_multiple = atr_trail_multiple
        self.swing_window = swing_window
        self.confluence_lookback = confluence_lookback

    def generate_signals(self, symbol: str, df: pd.DataFrame) -> list[Signal]:
        fast_ema = ema(df["close"], self.fast_period)
        slow_ema = ema(df["close"], self.slow_period)
        atr_series = atr(df, self.atr_period)
        bull_confluence = smc.recently_true(smc.bullish_confluence(df, self.swing_window), self.confluence_lookback)
        bear_confluence = smc.recently_true(smc.bearish_confluence(df, self.swing_window), self.confluence_lookback)

        signals: list[Signal] = []
        position: Side = "flat"
        trail_extreme = 0.0
        min_bars = max(self.slow_period, self.atr_period, self.swing_window * 2 + 1)

        for i in range(min_bars, len(df)):
            date = df.index[i]
            price = df["close"].iloc[i]
            a = atr_series.iloc[i]
            uptrend = fast_ema.iloc[i] > slow_ema.iloc[i]
            downtrend = fast_ema.iloc[i] < slow_ema.iloc[i]
            if pd.isna(a):
                continue

            if position == "flat":
                if uptrend and bull_confluence.iloc[i]:
                    stop = price - self.atr_trail_multiple * a
                    signals.append(Signal(date, symbol, "long", price, stop,
                                           "uptrend pullback into SMC bullish zone"))
                    position, trail_extreme = "long", price
                elif downtrend and bear_confluence.iloc[i]:
                    stop = price + self.atr_trail_multiple * a
                    signals.append(Signal(date, symbol, "short", price, stop,
                                           "downtrend rally into SMC bearish zone"))
                    position, trail_extreme = "short", price
            elif position == "long":
                trail_extreme = max(trail_extreme, price)
                trail_stop = trail_extreme - self.atr_trail_multiple * a
                if downtrend or price < trail_stop:
                    signals.append(Signal(date, symbol, "flat", price, price, "trend flip / trail stop"))
                    position = "flat"
            elif position == "short":
                trail_extreme = min(trail_extreme, price)
                trail_stop = trail_extreme + self.atr_trail_multiple * a
                if uptrend or price > trail_stop:
                    signals.append(Signal(date, symbol, "flat", price, price, "trend flip / trail stop"))
                    position = "flat"

        return signals

    def catch_up_signal(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        """Current simulated state, repriced on the last bar.

        The state machine may be holding a position whose entry bar predates
        this instrument going live (added to the universe mid-trend), so the
        book is flat while the simulation is not. Reprice that simulated
        position on the last bar with its trailing stop as maintained in
        generate_signals; a flat simulation reconciles by closing any orphan.
        """
        if df.empty:
            return None
        signals = self.generate_signals(symbol, df)
        if not signals:
            return None
        price = float(df["close"].iloc[-1])
        last = signals[-1]
        if last.side == "flat":
            return Signal(df.index[-1], symbol, "flat", price, price,
                          "catch-up: strategy is flat (flip was missed)")
        held = df.loc[last.date:]
        a = float(atr(df, self.atr_period).iloc[-1])
        if last.side == "long":
            stop = float(held["close"].max()) - self.atr_trail_multiple * a
        else:
            stop = float(held["close"].min()) + self.atr_trail_multiple * a
        return Signal(df.index[-1], symbol, last.side, price, stop,
                      f"catch-up: {last.side} since {last.date.date()} "
                      f"({last.reason}); flip was missed")
