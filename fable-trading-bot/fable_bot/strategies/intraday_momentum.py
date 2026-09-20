"""Intraday momentum for index futures — a noise-band breakout from the session anchor.

Why this and not the existing mean-reversion strategy
-----------------------------------------------------
Mean reversion was measured on intraday ES/NQ and produced a Sharpe of -1.57
(15m) and -2.01 (1h), tripping the drawdown kill-switch both times. Index
futures intraday are not a mean-reverting regime, so this strategy takes the
opposite side of that: it trades *continuation* once price escapes the range
that a normal session would produce by that time of day.

The design follows two published results rather than being invented here:

  * Gao, Han, Li & Zhou, "Market Intraday Momentum", Journal of Financial
    Economics (2018). The first half-hour return predicts the last half-hour
    return; predictive R^2 ~1.6%. Crucially for us, they find the effect is
    STRONGER on high-volatility days, high-volume days, and major macroeconomic
    news release days.

  * Zarattini, Aziz & Barbon, "Beat the Market: An Effective Intraday Momentum
    Strategy for S&P500 ETF (SPY)" (2024). Builds a volatility "noise area"
    around a session anchor, goes long/short when price breaks out of it, uses
    session VWAP as a trailing stop, and is always flat by the close.

The mechanics below are that second paper's structure, adapted to futures.

  sigma(k)  = mean over the last `lookback` sessions of
              |close(k) - anchor| / anchor        for bar-of-session k
  upper(k)  = anchor * (1 + band_mult * sigma(k))
  lower(k)  = anchor * (1 - band_mult * sigma(k))

Computing sigma per bar-of-session (rather than one number per day) is what
captures intraday volatility seasonality: the band is naturally tight at 09:35
and wide by 15:00, so a breakout means the same thing at any hour.

Entries are long above `upper`, short below `lower`. Exits are, in priority
order: the hard stop (which is what makes the 1% sizing real), the VWAP
trailing stop, and the session close — the strategy never holds overnight.

IMPORTANT: sigma is computed from *prior* sessions only. Including the current
session would leak the day's realised volatility into its own entry decision
and produce a backtest that cannot be traded.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.intraday import session_opens, session_vwap


@dataclass(frozen=True)
class IntradayConfig:
    lookback: int = 14          # sessions used to build the noise band
    band_mult: float = 1.0      # breakout threshold, in units of sigma
    stop_mult: float = 1.0      # hard stop distance, in units of the band width
    max_trades_per_session: int = 2
    use_vwap_stop: bool = True
    # Skip the first bar: the anchor is that bar's open, so its own close is
    # mechanically inside the band and carries no information.
    min_bar_of_session: int = 1
    # Flatten before the bell rather than at it -- the closing bar is where
    # slippage is worst and a market-on-close fill is not what the backtest
    # assumes.
    flat_before_close_bars: int = 1


@dataclass(frozen=True)
class IntradayTrade:
    symbol: str
    session: object
    side: str              # "long" | "short"
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    points: float


def noise_bands(df: pd.DataFrame, config: IntradayConfig) -> pd.DataFrame:
    """Per-bar upper/lower breakout boundaries, built from prior sessions only."""
    out = df.copy()
    out["anchor"] = session_opens(df)
    out["rel_move"] = (out["close"] - out["anchor"]).abs() / out["anchor"]

    # Mean |move| for this bar-of-session across the previous `lookback`
    # sessions. shift(1) inside each bucket is what enforces "prior sessions
    # only" -- without it the current bar informs its own threshold.
    sigma = (out.groupby("bar_of_session")["rel_move"]
                .transform(lambda s: s.shift(1).rolling(config.lookback, min_periods=config.lookback).mean()))
    out["sigma"] = sigma
    out["upper"] = out["anchor"] * (1 + config.band_mult * sigma)
    out["lower"] = out["anchor"] * (1 - config.band_mult * sigma)
    out["vwap"] = session_vwap(df)
    return out


def generate_trades(symbol: str, df: pd.DataFrame, config: IntradayConfig | None = None) -> list[IntradayTrade]:
    """Walk each session bar by bar and produce completed round trips.

    Deliberately event-driven rather than vectorised: the exit priority (hard
    stop before VWAP before close) and the one-position-at-a-time rule are the
    parts that decide whether the risk model holds, and they are much easier to
    get right -- and to read -- as an explicit loop.
    """
    config = config or IntradayConfig()
    data = noise_bands(df, config)
    trades: list[IntradayTrade] = []

    for session, day in data.groupby("session", sort=True):
        if day["sigma"].isna().all():
            continue  # not enough history yet to define a band
        last_index = len(day) - 1 - config.flat_before_close_bars
        position = None
        taken = 0

        for i, (ts, bar) in enumerate(day.iterrows()):
            if position is not None:
                exit_price, reason = None, None
                # 1. Hard stop first: it is the only exit that bounds the loss
                #    to what the position was sized for, so it must win any tie
                #    with a softer exit on the same bar.
                if position["side"] == "long" and bar["low"] <= position["stop"]:
                    exit_price, reason = position["stop"], "hard stop"
                elif position["side"] == "short" and bar["high"] >= position["stop"]:
                    exit_price, reason = position["stop"], "hard stop"
                # 2. VWAP trailing stop: momentum has failed when the average
                #    participant of the session is on the other side.
                elif config.use_vwap_stop and position["side"] == "long" and bar["close"] < bar["vwap"]:
                    exit_price, reason = bar["close"], "vwap stop"
                elif config.use_vwap_stop and position["side"] == "short" and bar["close"] > bar["vwap"]:
                    exit_price, reason = bar["close"], "vwap stop"
                # 3. Time: never carry an intraday position overnight.
                elif i >= last_index:
                    exit_price, reason = bar["close"], "session close"

                if exit_price is not None:
                    direction = 1 if position["side"] == "long" else -1
                    trades.append(IntradayTrade(
                        symbol=symbol, session=session, side=position["side"],
                        entry_time=position["time"], entry_price=position["price"],
                        stop_price=position["stop"], exit_time=ts, exit_price=exit_price,
                        exit_reason=reason,
                        points=(exit_price - position["price"]) * direction,
                    ))
                    position = None

            if (position is None and i < last_index and taken < config.max_trades_per_session
                    and i >= config.min_bar_of_session and pd.notna(bar["sigma"])):
                band_width = bar["upper"] - bar["anchor"]
                if band_width <= 0:
                    continue
                if bar["close"] > bar["upper"]:
                    position = {"side": "long", "price": bar["close"], "time": ts,
                                "stop": bar["close"] - config.stop_mult * band_width}
                    taken += 1
                elif bar["close"] < bar["lower"]:
                    position = {"side": "short", "price": bar["close"], "time": ts,
                                "stop": bar["close"] + config.stop_mult * band_width}
                    taken += 1

    return trades
