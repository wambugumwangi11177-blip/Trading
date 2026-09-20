"""Shared indicator math used across strategy modules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(period).mean()


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std.replace(0, np.nan)


def donchian_channel(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series]:
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    return upper, lower


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()
