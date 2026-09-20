"""Historical data access, backed by yfinance (free, no API key required).

Live/paper trading gets its own price snapshots from the broker adapter, so this
module is only responsible for backtesting and for the daily bar history that
strategies need to compute indicators (ATR, moving averages, swing points, etc).
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV history for a single symbol as a clean, flat-column DataFrame."""
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No historical data returned for {symbol!r} (period={period!r})")
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index.name = "date"
    return df


def fetch_universe_history(symbols: list[str], period: str = "6mo", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Fetch history for every symbol in the trading universe."""
    return {symbol: fetch_history(symbol, period=period, interval=interval) for symbol in symbols}
