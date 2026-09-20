"""Technical snapshot: where price sits against its own recent history.

This is the least privileged information in the briefing and it is placed
last on purpose. Every participant sees the same moving averages; nobody is
forced to act on them. What technicals earn their place for is CONTEXT for the
constrained-flow sections: a gamma flip is worth more when it coincides with
the 20-day low, and a crowded COT long is worth more when RSI is already
stretched. On their own they are a description, not a reason.

Everything here is computed from daily bars and states which bar it was
computed on, so a reader can tell a Friday close from a Monday morning.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..strategies.indicators import atr, ema
from .fmt import px


@dataclass(frozen=True)
class TechnicalSnapshot:
    symbol: str
    as_of_bar: str
    close: float
    ema20: float
    ema50: float
    ema200: float | None
    rsi14: float
    atr14: float
    atr_pct: float
    high20: float
    low20: float
    high52w: float | None
    low52w: float | None
    macd: float
    macd_signal: float
    ret_5d_pct: float
    ret_20d_pct: float

    @property
    def trend(self) -> str:
        """EMA stack reading: 'up' when 20>50(>200), 'down' when inverted, else 'mixed'."""
        if self.ema200 is not None:
            if self.ema20 > self.ema50 > self.ema200:
                return "up"
            if self.ema20 < self.ema50 < self.ema200:
                return "down"
            return "mixed"
        if self.ema20 > self.ema50:
            return "up"
        if self.ema20 < self.ema50:
            return "down"
        return "mixed"

    @property
    def rsi_state(self) -> str:
        if self.rsi14 >= 70:
            return "overbought"
        if self.rsi14 <= 30:
            return "oversold"
        return "neutral"

    @property
    def range_position_pct(self) -> float:
        """Where close sits inside the 20-day range: 0 = at the low, 100 = at the high."""
        span = self.high20 - self.low20
        if span <= 0:
            return 50.0
        return (self.close - self.low20) / span * 100

    def summary(self) -> str:
        c = self.close
        lines = [
            f"{self.symbol} technicals (bar {self.as_of_bar}) -- close {px(c)}",
            f"  trend {self.trend.upper()}: EMA20 {px(self.ema20, c)} / EMA50 {px(self.ema50, c)}"
            + (f" / EMA200 {px(self.ema200, c)}" if self.ema200 is not None else ""),
            f"  RSI14 {self.rsi14:.1f} ({self.rsi_state}), MACD {px(self.macd, c)} vs signal {px(self.macd_signal, c)}",
            f"  ATR14 {px(self.atr14, c)} ({self.atr_pct:.2f}% of price) -- a 1.5x ATR stop is {px(1.5 * self.atr14, c)}",
            f"  20d range {px(self.low20, c)} - {px(self.high20, c)}, close at {self.range_position_pct:.0f}% of it",
            f"  5d {self.ret_5d_pct:+.2f}%   20d {self.ret_20d_pct:+.2f}%",
        ]
        if self.high52w is not None and self.low52w is not None:
            lines.append(f"  52w range {px(self.low52w, c)} - {px(self.high52w, c)}")
        return "\n".join(lines)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Kept local: indicators.py has no RSI and nothing else needs one."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    out = 100 - 100 / (1 + rs)
    # All-gain windows divide by zero above; that is RSI 100 by definition.
    return out.fillna(100.0).where(avg_loss.notna(), float("nan"))


def compute_technicals(symbol: str, df: pd.DataFrame) -> TechnicalSnapshot:
    """Build the snapshot from a daily OHLC frame with lowercase columns."""
    if df is None or len(df) < 30:
        raise ValueError(f"{symbol}: need at least 30 daily bars, got {0 if df is None else len(df)}")
    close = df["close"].astype(float)
    n = len(close)

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200) if n >= 200 else None
    a14 = atr(df, 14)
    r14 = rsi(close, 14)
    macd_line = ema(close, 12) - ema(close, 26)
    macd_sig = ema(macd_line, 9)

    last = close.iloc[-1]
    return TechnicalSnapshot(
        symbol=symbol,
        as_of_bar=str(pd.Timestamp(df.index[-1]).date()),
        close=float(last),
        ema20=float(e20.iloc[-1]),
        ema50=float(e50.iloc[-1]),
        ema200=float(e200.iloc[-1]) if e200 is not None else None,
        rsi14=float(r14.iloc[-1]),
        atr14=float(a14.iloc[-1]),
        atr_pct=float(a14.iloc[-1] / last * 100) if last else 0.0,
        high20=float(df["high"].tail(20).max()),
        low20=float(df["low"].tail(20).min()),
        high52w=float(df["high"].tail(252).max()) if n >= 200 else None,
        low52w=float(df["low"].tail(252).min()) if n >= 200 else None,
        macd=float(macd_line.iloc[-1]),
        macd_signal=float(macd_sig.iloc[-1]),
        ret_5d_pct=float((last / close.iloc[-6] - 1) * 100) if n > 6 else 0.0,
        ret_20d_pct=float((last / close.iloc[-21] - 1) * 100) if n > 21 else 0.0,
    )
