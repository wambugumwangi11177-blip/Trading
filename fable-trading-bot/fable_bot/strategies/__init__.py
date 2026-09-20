from .base import Signal, Strategy
from .long_or_flat_trend import LongOrFlatTrendStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumBreakoutStrategy
from .trend_following import TrendFollowingStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumBreakoutStrategy,
    "trend_following": TrendFollowingStrategy,
    # config.GOLD_FUTURES names this; without the entry every path through
    # build_strategy raised KeyError on the one validated strategy.
    "long_or_flat_trend": LongOrFlatTrendStrategy,
}


def build_strategy(name: str) -> Strategy:
    return STRATEGY_REGISTRY[name]()


__all__ = [
    "Signal",
    "Strategy",
    "LongOrFlatTrendStrategy",
    "MeanReversionStrategy",
    "MomentumBreakoutStrategy",
    "TrendFollowingStrategy",
    "STRATEGY_REGISTRY",
    "build_strategy",
]
