"""Mean reversion strategy for range-bound, index-like instruments (SPY, QQQ).

Entry: price stretched >2 std devs from its 20-day mean (z-score), confirmed by
an SMC confluence signal in the reversion direction (liquidity sweep of the
recent extreme, or a retest of an unfilled FVG) so we're not just fading
momentum blindly.
Exit: z-score reverts back inside +/-0.5 of the mean.
"""
from __future__ import annotations

import pandas as pd

from . import smc
from .base import Side, Signal, Strategy
from .indicators import atr, zscore


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(self, lookback: int = 20, entry_z: float = 2.0, exit_z: float = 0.5,
                 atr_period: int = 14, atr_stop_multiple: float = 1.5, swing_window: int = 5,
                 confluence_lookback: int | None = None):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.swing_window = swing_window
        # A z-score extreme builds up over `lookback` days; an SMC sweep/FVG can
        # confirm anywhere in that same window rather than on the exact bar the
        # z-score happens to peak, so the confluence check spans the same
        # window the statistical extreme itself is measured over.
        self.confluence_lookback = confluence_lookback if confluence_lookback is not None else lookback

    def generate_signals(self, symbol: str, df: pd.DataFrame) -> list[Signal]:
        z = zscore(df["close"], self.lookback)
        atr_series = atr(df, self.atr_period)
        bull_confluence = smc.recently_true(smc.bullish_confluence(df, self.swing_window), self.confluence_lookback)
        bear_confluence = smc.recently_true(smc.bearish_confluence(df, self.swing_window), self.confluence_lookback)

        signals: list[Signal] = []
        position: Side = "flat"
        min_bars = max(self.lookback, self.atr_period, self.swing_window * 2 + 1)

        for i in range(min_bars, len(df)):
            date = df.index[i]
            price = df["close"].iloc[i]
            zi = z.iloc[i]
            a = atr_series.iloc[i]
            if pd.isna(zi) or pd.isna(a):
                continue

            if position == "flat":
                if zi <= -self.entry_z and bull_confluence.iloc[i]:
                    stop = price - self.atr_stop_multiple * a
                    signals.append(Signal(date, symbol, "long", price, stop,
                                           f"z={zi:.2f} oversold + SMC bullish confluence"))
                    position = "long"
                elif zi >= self.entry_z and bear_confluence.iloc[i]:
                    stop = price + self.atr_stop_multiple * a
                    signals.append(Signal(date, symbol, "short", price, stop,
                                           f"z={zi:.2f} overbought + SMC bearish confluence"))
                    position = "short"
            elif position == "long" and zi >= -self.exit_z:
                signals.append(Signal(date, symbol, "flat", price, price, f"z={zi:.2f} reverted to mean"))
                position = "flat"
            elif position == "short" and zi <= self.exit_z:
                signals.append(Signal(date, symbol, "flat", price, price, f"z={zi:.2f} reverted to mean"))
                position = "flat"

        return signals
