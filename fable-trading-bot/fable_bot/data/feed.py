"""Historical data access, backed by yfinance (free, no API key required).

Live/paper trading gets its own price snapshots from the broker adapter, so this
module is only responsible for backtesting and for the daily bar history that
strategies need to compute indicators (ATR, moving averages, swing points, etc).
"""
from __future__ import annotations

from datetime import date

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


def _bar_date(stamp) -> date:
    """Calendar date of a bar index entry, tz-aware or not."""
    return stamp.date() if isinstance(stamp, pd.Timestamp) else pd.Timestamp(stamp).date()


def drop_incomplete_bar(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """Trim today's still-forming daily bar off the end of a history frame.

    Only relevant when the cycle runs *during* a session. yfinance happily
    returns a partial row for the current day, and at 09:30 that row is a
    single print: near-zero range, near-zero volume. Feeding it to the
    strategies would contaminate every indicator computed off it -- most
    dangerously ATR, which sets the stop distance and therefore position size.
    A collapsed ATR yields a tight stop, a tight stop yields a large quantity,
    and the 1%-per-trade rule quietly becomes something else.

    Dropping it makes the live signal identical to the one the backtest
    computed at yesterday's close, which is the signal actually being traded.
    """
    if df.empty:
        return df
    cutoff = today or _now_et().date()
    if _bar_date(df.index[-1]) >= cutoff:
        return df.iloc[:-1]
    return df



def _now_et():
    # Imported lazily so the data layer stays free of scheduler dependencies.
    from ..market_calendar import now_et
    return now_et()
