"""Broker-agnostic interface so a paper/live venue can be swapped without
touching strategy, risk, or portfolio code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    is_paper: bool


@dataclass
class OrderResult:
    symbol: str
    side: str
    qty: float
    status: str
    broker_order_id: str | None = None


@dataclass
class HistoryOrder:
    """One row of broker order history, normalized across adapters."""
    id: str | None
    symbol: str
    side: str
    type: str  # "market" | "limit" | "stop" | ...
    status: str  # "filled" | "working" | "rejected" | "cancelled" | ...
    qty: float
    avg_fill_price: float | None = None
    placed_at: str | None = None  # UTC ISO, may be None for old rows


class BrokerAdapter(ABC):
    @abstractmethod
    def venue_symbol(self, instrument) -> str:
        """Ticker for this instrument as the venue names it.

        Alpaca wants "BTC/USD"; TradingView wants "BITSTAMP:BTCUSD". Keeping the
        mapping behind the adapter means live_runner never has to know which
        broker is connected.
        """
        raise NotImplementedError

    @abstractmethod
    def quantize(self, instrument, qty: float) -> float:
        """Round a sized quantity to something the venue will accept.

        Always rounds DOWN: ATR sizing produces fractions, and rounding up would
        risk more than the 1% of equity the position was sized for. May return
        0, which callers must treat as "too small to trade".
        """
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_open_positions(self) -> dict[str, dict]:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, symbol: str, qty: float, side: str, stop_price: float | None = None) -> OrderResult:
        """side: 'buy' or 'sell' (long-entry / short-entry or a closing trade).

        stop_price, when given, must be attached as a broker-side stop so the
        1%-risk sizing is actually enforced between polling cycles -- a daily
        signal-check loop can't react to an intrabar breach in time on its own.
        """
        raise NotImplementedError

    @abstractmethod
    def close_position(self, symbol: str) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def close_all(self) -> list[OrderResult]:
        raise NotImplementedError

    # Optional capabilities. Deliberately concrete (not abstract): an adapter
    # without order-history support -- Alpaca today -- needs no changes, and
    # callers (reconcile.py) degrade gracefully instead of failing the run.

    def get_order_history(self, limit: int = 100) -> list[HistoryOrder]:
        """Most-recent broker order history, normalized. Raises NotImplementedError
        when the venue doesn't expose it."""
        raise NotImplementedError(f"{type(self).__name__} does not support order history")

    def verify_order(self, symbol: str, side: str, qty: float, *,
                     attempts: int = 3, delay_seconds: float = 2.0) -> HistoryOrder | None:
        """Confirm a just-submitted order actually exists at the broker.

        Learned 2026-08: a submission response can be lost (timeout, unparseable
        JSON) while the order still lands at the venue -- or never lands at all.
        A bounded poll of the history turns both cases into evidence instead of
        a guess. Returns the matching row, or None if unverified.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support order history")
