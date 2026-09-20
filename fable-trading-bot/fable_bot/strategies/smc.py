"""Smart-money-concepts confluence filter.

TJR Trades' free Bootcamp material (publicly available; see
https://www.jointjrtrades.com/course) teaches a simplified version of ICT-style
"Smart Money Concepts": read market structure, wait for a liquidity sweep of a
prior swing point, then look for entries at a fair value gap or order block in
the direction of the sweep. None of this code reproduces paid/gated course
material -- it's a standard, independently-implemented version of concepts that
are widely documented (market structure, liquidity sweeps, FVGs, order blocks).

This module doesn't generate trade signals on its own. It's a confluence filter
that the three core strategies (mean reversion, momentum, trend following) call
into before firing a signal, so entries land where SMC theory says institutional
order flow is more likely to support the move rather than blindly on an
indicator crossing a threshold.

All of this operates on daily bars for backtest simplicity. Session-window
filtering (NY/London open) is a hook for a future intraday version and is a
no-op at the daily timeframe.

A swing high/low can only be *identified* once `window` bars have printed after
it (that's what makes it a swing point rather than just the latest bar). Using
a centered rolling window without accounting for that would let the backtest
"see" a swing the same day it happens -- a look-ahead bias a live bot could
never replicate. Every flag below is therefore delayed by `window` bars from
the underlying extreme so a live run and a backtest agree on what was knowable
on a given day.
"""
from __future__ import annotations

import pandas as pd


def swing_points(df: pd.DataFrame, window: int = 5) -> tuple[pd.Series, pd.Series]:
    """Boolean series marking bars where a swing high / swing low is *confirmed*.

    The flag for a swing that occurred at bar i is set at bar i + window (the
    earliest point it's actually knowable), not at bar i itself.
    """
    high, low = df["high"], df["low"]
    raw_high = high == high.rolling(window * 2 + 1, center=True).max()
    raw_low = low == low.rolling(window * 2 + 1, center=True).min()
    is_swing_high = raw_high.fillna(False).shift(window).fillna(False).astype(bool)
    is_swing_low = raw_low.fillna(False).shift(window).fillna(False).astype(bool)
    return is_swing_high, is_swing_low


def _swing_values(df: pd.DataFrame, is_sh: pd.Series, is_sl: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Price level of each confirmed swing, aligned to its confirmation bar (not the extreme's own bar)."""
    swing_high_vals = df["high"].shift(window).where(is_sh)
    swing_low_vals = df["low"].shift(window).where(is_sl)
    return swing_high_vals, swing_low_vals


def market_structure(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Rolling trend label from swing structure: 'bullish', 'bearish', or 'neutral'.

    Bullish once the most recently confirmed swing high is higher than the one
    before it, AND the most recently confirmed swing low is higher than the one
    before it (higher highs + higher lows). Bearish on the mirror condition.
    Each side's state persists until the next swing of that type updates it --
    swing highs and swing lows essentially never confirm on the same calendar
    bar, so this must be tracked independently per side rather than compared
    row-by-row.
    """
    is_sh, is_sl = swing_points(df, window)
    swing_high_vals, swing_low_vals = _swing_values(df, is_sh, is_sl, window)

    sh_seq = swing_high_vals.dropna()
    sl_seq = swing_low_vals.dropna()

    higher_high = (sh_seq > sh_seq.shift(1)).reindex(df.index).ffill().fillna(False)
    lower_high = (sh_seq < sh_seq.shift(1)).reindex(df.index).ffill().fillna(False)
    higher_low = (sl_seq > sl_seq.shift(1)).reindex(df.index).ffill().fillna(False)
    lower_low = (sl_seq < sl_seq.shift(1)).reindex(df.index).ffill().fillna(False)

    structure = pd.Series("neutral", index=df.index)
    structure[(higher_high & higher_low)] = "bullish"
    structure[(lower_high & lower_low)] = "bearish"
    return structure


def liquidity_sweeps(df: pd.DataFrame, window: int = 5) -> tuple[pd.Series, pd.Series]:
    """Detect stop-hunt style sweeps of a prior swing point that reverse intrabar.

    Returns (swept_sell_side, swept_buy_side):
      - swept_sell_side: today's high pierced the last confirmed swing high but
        closed back below it -- buy-side liquidity grabbed, bearish reversal signal.
      - swept_buy_side: today's low pierced the last confirmed swing low but
        closed back above it -- sell-side liquidity grabbed, bullish reversal signal.
    """
    is_sh, is_sl = swing_points(df, window)
    swing_high_vals, swing_low_vals = _swing_values(df, is_sh, is_sl, window)
    last_swing_high = swing_high_vals.ffill()
    last_swing_low = swing_low_vals.ffill()

    swept_sell_side = (df["high"] > last_swing_high) & (df["close"] < last_swing_high)
    swept_buy_side = (df["low"] < last_swing_low) & (df["close"] > last_swing_low)
    return swept_sell_side.fillna(False), swept_buy_side.fillna(False)


def fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """3-bar imbalance zones. One row per bar with bullish/bearish zone bounds (NaN if none)."""
    high, low = df["high"], df["low"]
    prev_high, prev_low = high.shift(2), low.shift(2)

    bullish = prev_high < low  # gap between bar[i-2].high and bar[i].low
    bearish = prev_low > high  # gap between bar[i].high and bar[i-2].low

    out = pd.DataFrame(index=df.index)
    out["bullish_fvg_low"] = prev_high.where(bullish)
    out["bullish_fvg_high"] = low.where(bullish)
    out["bearish_fvg_low"] = high.where(bearish)
    out["bearish_fvg_high"] = prev_low.where(bearish)
    return out


def price_near_bullish_zone(df: pd.DataFrame, fvg: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """True when today's low dips into an unfilled bullish FVG from the last `lookback` bars."""
    near = pd.Series(False, index=df.index)
    lows = fvg["bullish_fvg_low"].ffill(limit=lookback)
    highs = fvg["bullish_fvg_high"].ffill(limit=lookback)
    valid = lows.notna() & highs.notna()
    near[valid] = (df["low"][valid] <= highs[valid]) & (df["low"][valid] >= lows[valid])
    return near


def price_near_bearish_zone(df: pd.DataFrame, fvg: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """True when today's high pokes into an unfilled bearish FVG from the last `lookback` bars."""
    near = pd.Series(False, index=df.index)
    lows = fvg["bearish_fvg_low"].ffill(limit=lookback)
    highs = fvg["bearish_fvg_high"].ffill(limit=lookback)
    valid = lows.notna() & highs.notna()
    near[valid] = (df["high"][valid] >= lows[valid]) & (df["high"][valid] <= highs[valid])
    return near


def bullish_confluence(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Combined long-side confluence: not-bearish structure + (sweep or FVG retest)."""
    structure = market_structure(df, window)
    _, swept_buy_side = liquidity_sweeps(df, window)
    fvg = fair_value_gaps(df)
    near_bull_zone = price_near_bullish_zone(df, fvg)
    return (structure != "bearish") & (swept_buy_side | near_bull_zone)


def bearish_confluence(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Combined short-side confluence: not-bullish structure + (sweep or FVG retest)."""
    structure = market_structure(df, window)
    swept_sell_side, _ = liquidity_sweeps(df, window)
    fvg = fair_value_gaps(df)
    near_bear_zone = price_near_bearish_zone(df, fvg)
    return (structure != "bullish") & (swept_sell_side | near_bear_zone)


def recently_true(flags: pd.Series, lookback: int = 3) -> pd.Series:
    """True if `flags` was true at any point in the last `lookback` bars (inclusive of today).

    A sweep/FVG confluence signal and a statistical trigger (z-score, breakout,
    trend pullback) rarely land on the exact same bar in practice -- one often
    confirms a bar or two before or after the other. Requiring literal same-bar
    coincidence would be unrealistically strict for both backtest and live use,
    so strategies check confluence over a short trailing window instead of a
    single bar.
    """
    return flags.rolling(lookback, min_periods=1).max().astype(bool)
