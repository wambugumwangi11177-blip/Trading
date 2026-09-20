"""Market-structure intelligence: what the large, hedging-constrained players
are positioned in, as opposed to what price has done.

Everything under fable_bot/strategies reads PRICE -- moving averages, ATR,
swing sweeps, fair value gaps. Price is the output of the machinery. This
package reads inputs to it:

  * gamma.py -- dealer gamma exposure from listed option open interest. Option
    market makers run delta-neutral books, so their hedging is not discretion,
    it is obligation: the sign of their aggregate gamma decides whether they
    must BUY dips and SELL rallies (long gamma, volatility suppressed, price
    pins to strikes) or SELL dips and BUY rallies (short gamma, volatility
    amplified, moves extend). That is a mechanical, publishable constraint on
    the largest continuous flow in the market.

  * cot.py -- CFTC Commitments of Traders. Weekly, by regulation: how much of
    the futures open interest is held by commercial hedgers (producers,
    refiners, bullion banks) versus managed money (trend-following funds).

Neither predicts direction. Both say where flow is forced, which is a
different and more durable claim than "this indicator crossed that one".
"""
from __future__ import annotations

from .cot import CotError, CotReport, NoCotMarket, fetch_cot, parse_cot_rows
from .gamma import GammaProfile, StrikeGamma, black_scholes_gamma, compute_gamma_profile
from .liquidity import LiquidityMap, LiquidityPool, build_liquidity_map
from .options_feed import (OptionsFeedError, fetch_gamma_profile, fetch_option_chains,
                           fetch_spot)
from .sentiment import Sentiment, fetch_sentiment, sentiment_from_cot_row
from .technicals import TechnicalSnapshot, compute_technicals, rsi

__all__ = [
    "GammaProfile",
    "StrikeGamma",
    "black_scholes_gamma",
    "compute_gamma_profile",
    "CotReport",
    "CotError",
    "NoCotMarket",
    "fetch_cot",
    "parse_cot_rows",
    "OptionsFeedError",
    "fetch_gamma_profile",
    "fetch_option_chains",
    "fetch_spot",
    "LiquidityMap",
    "LiquidityPool",
    "build_liquidity_map",
    "Sentiment",
    "fetch_sentiment",
    "sentiment_from_cot_row",
    "TechnicalSnapshot",
    "compute_technicals",
    "rsi",
]
