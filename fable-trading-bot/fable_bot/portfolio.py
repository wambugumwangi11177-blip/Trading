"""Tracks cash, open positions, realized/unrealized P&L, and the equity curve."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    symbol: str
    side: str  # "long" | "short"
    qty: float
    entry_price: float
    stop_price: float
    opened_at: pd.Timestamp
    # Cash per 1.00 of price move, per unit. 1.0 for shares/ETFs; the contract
    # multiplier for futures (E-mini S&P = $50/point). P&L is multiplied by it,
    # so leaving it at 1.0 for a futures position understates the result 50x.
    point_value: float = 1.0


@dataclass
class Trade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp
    pnl: float
    reason: str


@dataclass
class PortfolioTracker:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    def open_position(self, symbol: str, side: str, qty: float, price: float,
                       stop_price: float, date: pd.Timestamp, point_value: float = 1.0) -> None:
        self.positions[symbol] = Position(symbol, side, qty, price, stop_price, date, point_value)

    def close_position(self, symbol: str, price: float, date: pd.Timestamp, reason: str) -> float:
        pos = self.positions.pop(symbol)
        direction = 1 if pos.side == "long" else -1
        pnl = direction * (price - pos.entry_price) * pos.qty * pos.point_value
        self.cash += pnl
        self.trades.append(Trade(symbol, pos.side, pos.qty, pos.entry_price, price,
                                  pos.opened_at, date, pnl, reason))
        return pnl

    def unrealized_pnl(self, symbol: str, price: float) -> float:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        direction = 1 if pos.side == "long" else -1
        return direction * (price - pos.entry_price) * pos.qty * pos.point_value

    def equity(self, prices: dict[str, float]) -> float:
        unrealized = sum(self.unrealized_pnl(sym, prices[sym]) for sym in self.positions if sym in prices)
        return self.cash + unrealized

    def record_equity(self, date: pd.Timestamp, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        self.equity_curve.append((date, eq))
        return eq

    def open_sides(self) -> dict[str, str]:
        return {symbol: pos.side for symbol, pos in self.positions.items()}
