"""Runs the full 5-instrument strategy suite over historical data.

Wires together: data feed -> per-instrument strategy signals -> risk manager
(sizing, correlation filter) -> portfolio tracker -> drawdown kill-switch,
then reports Sharpe ratio, max drawdown, win rate, and per-trade detail so a
negative-Sharpe strategy can be caught before it's ever pointed at a paper or
live account.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import RISK, Instrument
from .data.feed import fetch_universe_history
from .portfolio import PortfolioTracker
from .risk.manager import DrawdownMonitor, RiskManager, correlation_matrix
from .strategies import Signal, build_strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    halted: bool


def _all_signals(price_history: dict[str, pd.DataFrame], instruments: list[Instrument]) -> list[Signal]:
    signals: list[Signal] = []
    for inst in instruments:
        strategy = build_strategy(inst.strategy)
        signals.extend(strategy.generate_signals(inst.symbol, price_history[inst.symbol]))
    return sorted(signals, key=lambda s: s.date)


def enforce_hard_stops(portfolio: PortfolioTracker, price_history: dict[str, pd.DataFrame],
                        date: pd.Timestamp) -> None:
    """Force-close any open position whose fixed entry stop was breached intrabar today.

    The 1%-of-equity sizing at entry only holds if the stop is actually
    enforced. A strategy's own exit condition (z-score, trend flip) is a
    softer, usually-earlier exit; this is the non-negotiable backstop that
    caps the loss at what was sized, checked before today's strategy signals
    are evaluated.
    """
    for symbol in list(portfolio.positions):
        df = price_history.get(symbol)
        if df is None or date not in df.index:
            continue
        pos = portfolio.positions[symbol]
        bar = df.loc[date]
        if pos.side == "long" and bar["low"] <= pos.stop_price:
            portfolio.close_position(symbol, pos.stop_price, date, "hard stop loss (max 1% risk)")
        elif pos.side == "short" and bar["high"] >= pos.stop_price:
            portfolio.close_position(symbol, pos.stop_price, date, "hard stop loss (max 1% risk)")


# Bars per year, used to annualise Sharpe. Daily bars are the original case;
# intraday backtests must pass their own or the ratio is understated by
# sqrt(bars per day) -- roughly 5x on 15-minute bars.
PERIODS_PER_YEAR = {"1d": 252, "1h": 252 * 7, "30m": 252 * 13, "15m": 252 * 26, "5m": 252 * 78}


def _performance(equity_curve: list[tuple[pd.Timestamp, float]], trades: list,
                  periods_per_year: int = 252) -> tuple[float, float, float, float]:
    if len(equity_curve) < 2:
        return 0.0, 0.0, 0.0, 0.0
    eq = pd.Series([e for _, e in equity_curve], index=[d for d, _ in equity_curve])
    period_returns = eq.pct_change().dropna()
    total_return_pct = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    sharpe = (period_returns.mean() / period_returns.std() * math.sqrt(periods_per_year)) \
        if period_returns.std() > 0 else 0.0
    running_peak = eq.cummax()
    drawdown = (eq - running_peak) / running_peak * 100
    max_drawdown_pct = drawdown.min()
    wins = [t for t in trades if t.pnl > 0]
    win_rate_pct = (len(wins) / len(trades) * 100) if trades else 0.0
    return total_return_pct, sharpe, max_drawdown_pct, win_rate_pct


def run_backtest(instruments: list[Instrument], period: str = "6mo",
                  interval: str = "1d") -> BacktestResult:
    price_history = fetch_universe_history([i.symbol for i in instruments],
                                            period=period, interval=interval)
    return simulate(price_history, instruments,
                    periods_per_year=PERIODS_PER_YEAR.get(interval, 252))


def simulate(price_history: dict[str, pd.DataFrame], instruments: list[Instrument],
              periods_per_year: int = 252) -> BacktestResult:
    """Core event loop, separated from data-fetching so it's testable without network access."""
    corr = correlation_matrix(price_history)

    risk_manager = RiskManager(config=RISK, correlations=corr)
    portfolio = PortfolioTracker(cash=RISK.starting_equity)
    drawdown_monitor = DrawdownMonitor(max_drawdown_pct=RISK.max_drawdown_pct, peak_equity=RISK.starting_equity)

    signals = _all_signals(price_history, instruments)
    # Signals only carry a symbol, so keep the contract multiplier reachable by
    # symbol for sizing and P&L. Defaults to 1.0 for anything not in the
    # universe, which is correct for shares and ETFs.
    point_values = {inst.symbol: inst.point_value for inst in instruments}
    master_dates = sorted(set().union(*[df.index for df in price_history.values()]))
    last_price: dict[str, float] = {}
    signals_by_date: dict[pd.Timestamp, list[Signal]] = {}
    for s in signals:
        signals_by_date.setdefault(s.date, []).append(s)

    for date in master_dates:
        for symbol, df in price_history.items():
            if date in df.index:
                last_price[symbol] = df.loc[date, "close"]

        if drawdown_monitor.halted:
            for symbol in list(portfolio.positions):
                portfolio.close_position(symbol, last_price[symbol], date, "drawdown kill-switch")
            portfolio.record_equity(date, last_price)
            continue

        enforce_hard_stops(portfolio, price_history, date)

        for sig in signals_by_date.get(date, []):
            if sig.side == "flat":
                if sig.symbol in portfolio.positions:
                    portfolio.close_position(sig.symbol, sig.price, sig.date, sig.reason)
                continue

            if sig.symbol in portfolio.positions:
                continue  # already in a position for this symbol

            blocked, _reason = risk_manager.is_blocked_by_correlation(
                sig.symbol, sig.side, portfolio.open_sides())
            if blocked:
                continue

            equity_now = portfolio.equity(last_price)
            point_value = point_values.get(sig.symbol, 1.0)
            qty = risk_manager.position_size(equity_now, sig.price, sig.stop_price,
                                              point_value=point_value)
            if qty <= 0:
                continue
            portfolio.open_position(sig.symbol, sig.side, qty, sig.price, sig.stop_price,
                                     sig.date, point_value=point_value)

        equity_now = portfolio.record_equity(date, last_price)
        if drawdown_monitor.update(equity_now):
            for symbol in list(portfolio.positions):
                portfolio.close_position(symbol, last_price[symbol], date, "drawdown kill-switch")
            portfolio.record_equity(date, last_price)

    total_return_pct, sharpe, max_drawdown_pct, win_rate_pct = _performance(
        portfolio.equity_curve, portfolio.trades, periods_per_year=periods_per_year)
    eq_series = pd.Series([e for _, e in portfolio.equity_curve], index=[d for d, _ in portfolio.equity_curve])

    return BacktestResult(
        equity_curve=eq_series,
        trades=portfolio.trades,
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_drawdown_pct,
        win_rate_pct=win_rate_pct,
        num_trades=len(portfolio.trades),
        halted=drawdown_monitor.halted,
    )
