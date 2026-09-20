"""Validation harness for the intraday futures strategy.

Separate from backtester.py because the shape of the problem is different: the
daily backtester holds one position per symbol across days, while this one runs
many round trips inside a session and is flat every night.

The point of this module is to make it hard to fool ourselves. A 60-session
intraday sample will happily fit almost any rule, so results are reported as a
train/test split with the test period never touched during parameter choice,
plus a per-parameter sweep so the sensitivity is visible rather than hidden
behind one flattering number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .strategies.intraday_momentum import IntradayConfig, IntradayTrade, generate_trades

# Round-turn cost per contract in index points, covering commission plus one
# tick of slippage. ES ticks 0.25pt/$12.50 and NQ 0.25pt/$5, and a market order
# routinely gives up a tick per side, so ~1 tick round turn is optimistic-to-
# fair rather than conservative. Results are quoted net of this.
COST_POINTS = {"ES": 0.5, "NQ": 2.0, "MES": 0.5, "MNQ": 2.0}


@dataclass
class IntradayResult:
    symbol: str
    label: str
    trades: list[IntradayTrade]
    sessions: int
    net_points: float
    net_cash: float
    win_rate_pct: float
    avg_win_points: float
    avg_loss_points: float
    profit_factor: float
    expectancy_points: float
    sharpe: float
    max_drawdown_cash: float
    return_pct: float

    def as_row(self) -> str:
        return (f"{self.label:<22}{len(self.trades):>7}{self.win_rate_pct:>8.1f}%"
                f"{self.expectancy_points:>11.2f}{self.profit_factor:>9.2f}"
                f"{self.sharpe:>8.2f}{self.net_cash:>12,.0f}{self.return_pct:>9.2f}%")


def evaluate(symbol: str, trades: list[IntradayTrade], sessions: int, point_value: float,
              starting_equity: float, cost_points: float, label: str = "") -> IntradayResult:
    """Turn a list of round trips into the statistics that decide go/no-go."""
    net = [t.points - cost_points for t in trades]
    wins = [p for p in net if p > 0]
    losses = [p for p in net if p <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    equity, curve, peak, max_dd = starting_equity, [], starting_equity, 0.0
    for p in net:
        equity += p * point_value
        curve.append(equity)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    # Sharpe on per-trade returns, annualised by the observed trade frequency.
    # Using trades rather than calendar days keeps it meaningful when the
    # strategy sits out whole sessions.
    rets = pd.Series([p * point_value / starting_equity for p in net])
    trades_per_year = (len(net) / sessions * 252) if sessions else 0
    sharpe = (rets.mean() / rets.std() * math.sqrt(trades_per_year)) \
        if len(rets) > 1 and rets.std() > 0 else 0.0

    return IntradayResult(
        symbol=symbol, label=label or symbol, trades=trades, sessions=sessions,
        net_points=sum(net), net_cash=sum(net) * point_value,
        win_rate_pct=(len(wins) / len(net) * 100) if net else 0.0,
        avg_win_points=(gross_win / len(wins)) if wins else 0.0,
        avg_loss_points=(-gross_loss / len(losses)) if losses else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        expectancy_points=(sum(net) / len(net)) if net else 0.0,
        sharpe=sharpe, max_drawdown_cash=max_dd,
        return_pct=(sum(net) * point_value / starting_equity * 100),
    )


def run(symbol: str, df: pd.DataFrame, point_value: float, config: IntradayConfig,
        starting_equity: float = 100_000.0, cost_key: str = "NQ",
        label: str = "") -> IntradayResult:
    trades = generate_trades(symbol, df, config)
    return evaluate(symbol, trades, df["session"].nunique(), point_value,
                    starting_equity, COST_POINTS.get(cost_key, 2.0), label)


def split_sessions(df: pd.DataFrame, train_frac: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split on whole sessions.

    Chronological, never random: shuffling bars across a time series lets the
    model see the future, and splitting mid-session would hand the test set an
    entry whose exit lives in the training set.
    """
    sessions = sorted(df["session"].unique())
    cut = int(len(sessions) * train_frac)
    train_days, test_days = set(sessions[:cut]), set(sessions[cut:])
    return df[df["session"].isin(train_days)].copy(), df[df["session"].isin(test_days)].copy()


HEADER = (f"{'sample':<22}{'trades':>7}{'win%':>9}{'exp(pts)':>11}"
          f"{'PF':>9}{'Sharpe':>8}{'net $':>12}{'return':>10}")
