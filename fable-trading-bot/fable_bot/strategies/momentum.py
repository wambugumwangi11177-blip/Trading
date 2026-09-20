"""Momentum breakout strategy for BTC.

Entry: close breaks above/below the prior N-day Donchian channel extreme,
confirmed by SMC confluence (a liquidity sweep of the opposing side just
happened) so we favor breakouts that already shook out the other side's stops
over ones running straight into untested liquidity.
Exit: ATR-based trailing stop (chandelier exit) from the best close since entry,
or a breakout of the opposite channel.
"""
from __future__ import annotations

import pandas as pd

from . import smc
from .base import Side, Signal, Strategy
from .indicators import atr, donchian_channel


class MomentumBreakoutStrategy(Strategy):
    name = "momentum"

    def __init__(self, channel_period: int = 20, atr_period: int = 14,
                 atr_trail_multiple: float = 3.0, swing_window: int = 5,
                 confluence_lookback: int = 3):
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.atr_trail_multiple = atr_trail_multiple
        self.swing_window = swing_window
        self.confluence_lookback = confluence_lookback

    def generate_signals(self, symbol: str, df: pd.DataFrame) -> list[Signal]:
        upper, lower = donchian_channel(df, self.channel_period)
        atr_series = atr(df, self.atr_period)
        bull_confluence = smc.recently_true(smc.bullish_confluence(df, self.swing_window), self.confluence_lookback)
        bear_confluence = smc.recently_true(smc.bearish_confluence(df, self.swing_window), self.confluence_lookback)

        signals: list[Signal] = []
        position: Side = "flat"
        trail_extreme = 0.0
        min_bars = max(self.channel_period, self.atr_period, self.swing_window * 2 + 1) + 1

        for i in range(min_bars, len(df)):
            date = df.index[i]
            price = df["close"].iloc[i]
            a = atr_series.iloc[i]
            prior_upper = upper.iloc[i - 1]
            prior_lower = lower.iloc[i - 1]
            if pd.isna(a) or pd.isna(prior_upper) or pd.isna(prior_lower):
                continue

            if position == "flat":
                if price > prior_upper and bull_confluence.iloc[i]:
                    stop = price - self.atr_trail_multiple * a
                    signals.append(Signal(date, symbol, "long", price, stop,
                                           f"breakout above {self.channel_period}d high + SMC sweep confluence"))
                    position, trail_extreme = "long", price
                elif price < prior_lower and bear_confluence.iloc[i]:
                    stop = price + self.atr_trail_multiple * a
                    signals.append(Signal(date, symbol, "short", price, stop,
                                           f"breakdown below {self.channel_period}d low + SMC sweep confluence"))
                    position, trail_extreme = "short", price
            elif position == "long":
                trail_extreme = max(trail_extreme, price)
                trail_stop = trail_extreme - self.atr_trail_multiple * a
                if price < trail_stop or price < prior_lower:
                    signals.append(Signal(date, symbol, "flat", price, price, "chandelier trail stop / reversal"))
                    position = "flat"
            elif position == "short":
                trail_extreme = min(trail_extreme, price)
                trail_stop = trail_extreme + self.atr_trail_multiple * a
                if price > trail_stop or price > prior_upper:
                    signals.append(Signal(date, symbol, "flat", price, price, "chandelier trail stop / reversal"))
                    position = "flat"

        return signals
