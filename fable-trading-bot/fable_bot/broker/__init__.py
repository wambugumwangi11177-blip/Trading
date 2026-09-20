from .base import AccountSnapshot, BrokerAdapter, OrderResult
from .tradingview_adapter import TradingViewBrokerAdapter

__all__ = [
    "AccountSnapshot", "BrokerAdapter", "OrderResult",
    "TradingViewBrokerAdapter", "build_broker",
]


def build_broker(config) -> BrokerAdapter:
    """Construct the adapter named by BROKER_PROVIDER.

    Alpaca is imported lazily so its SDK stays an optional dependency -- running
    on TradingView shouldn't require alpaca-py to be installed.
    """
    provider = (config.provider or "tradingview").lower()
    if provider == "tradingview":
        return TradingViewBrokerAdapter(config)
    if provider == "alpaca":
        from .alpaca_adapter import AlpacaBrokerAdapter
        return AlpacaBrokerAdapter(config)
    raise ValueError(
        f"Unknown BROKER_PROVIDER {provider!r}. Expected 'tradingview' or 'alpaca'."
    )
