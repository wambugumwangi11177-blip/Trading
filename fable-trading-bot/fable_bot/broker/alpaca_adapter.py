"""Alpaca implementation of BrokerAdapter.

Defaults to the paper endpoint. Only routes to the live endpoint when both
TRADING_MODE=live and ALLOW_LIVE_TRADING=true are set in the environment --
see config.BrokerConfig.is_paper. There is no code path here that can promote
paper to live on its own; that's a deliberate two-flag human decision.
"""
from __future__ import annotations

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest

from ..config import BrokerConfig
from .base import AccountSnapshot, BrokerAdapter, OrderResult


class AlpacaBrokerAdapter(BrokerAdapter):
    def __init__(self, config: BrokerConfig):
        if not config.is_configured:
            raise RuntimeError(
                "Alpaca API key/secret not set. Add ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "to .env (see config/.env.example) before starting the live/paper runner."
            )
        self.config = config
        self.client = TradingClient(config.api_key, config.secret_key, paper=config.is_paper)

    def venue_symbol(self, instrument) -> str:
        return instrument.broker_symbol

    def quantize(self, instrument, qty: float) -> float:
        # Alpaca accepts fractional shares on equities and crypto, so sizing is
        # passed through untouched; TradingView is the one that needs rounding.
        return abs(qty)

    def get_account(self) -> AccountSnapshot:
        account = self.client.get_account()
        return AccountSnapshot(equity=float(account.equity), cash=float(account.cash),
                                is_paper=self.config.is_paper)

    def get_open_positions(self) -> dict[str, dict]:
        positions = self.client.get_all_positions()
        return {
            p.symbol: {"qty": float(p.qty), "side": "long" if float(p.qty) > 0 else "short",
                       "avg_entry_price": float(p.avg_entry_price)}
            for p in positions
        }

    def submit_order(self, symbol: str, qty: float, side: str, stop_price: float | None = None) -> OrderResult:
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        if stop_price is not None:
            # Bracket order: the stop leg lives at the broker, so the hard 1%-risk
            # stop still fires even though signal-check only polls once a day.
            request = MarketOrderRequest(
                symbol=symbol, qty=abs(qty), side=order_side, time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
            )
        else:
            request = MarketOrderRequest(symbol=symbol, qty=abs(qty), side=order_side,
                                          time_in_force=TimeInForce.DAY)
        order = self.client.submit_order(request)
        return OrderResult(symbol=symbol, side=side, qty=qty, status=order.status.value,
                            broker_order_id=str(order.id))

    def close_position(self, symbol: str) -> OrderResult:
        self.client.close_position(symbol)
        return OrderResult(symbol=symbol, side="close", qty=0, status="submitted")

    def close_all(self) -> list[OrderResult]:
        self.client.close_all_positions(cancel_orders=True)
        return [OrderResult(symbol="ALL", side="close", qty=0, status="submitted")]
