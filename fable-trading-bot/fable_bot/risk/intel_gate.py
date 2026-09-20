"""The positioning gate: a strategy entry must not fight the forced flow.

WHERE THIS SITS
---------------
The strategies decide WHAT looks like a trade (z-score extreme, Donchian
break, EMA pullback, trend votes). Every gate before this one decides whether
the account is ALLOWED to take it (lessons, portfolio caps, event blackout,
correlation, sizing). This one asks a different question: is the trade on the
same side as the players who are obligated to move price, or against them?

It reads the intel briefing (gamma, COT, retail sentiment, liquidity) and
refuses an entry in four situations, each of which is a documented way to be
the liquidity rather than take it:

  1. TRAPPED WITH THE CROWD. Retail is at an extreme (>= 75%) on the SAME
     side as the entry. Their stops sit behind them; running those stops is
     the cheapest fill available to size, and an entry on that side is a bet
     that this time they will not be run.

  2. AGAINST THE SYNTHESIS. The forced-flow sections agree on a lean and the
     entry is the other way. One section disagreeing is noise; two or more
     agreeing against the trade is a reason.

  3. WRONG STRATEGY FOR THE GAMMA REGIME. Momentum/breakout entries when
     dealers are LONG gamma: their hedging sells rallies and buys dips, which
     is exactly the flow that fails breakouts. Mean-reversion entries when
     dealers are SHORT gamma: their hedging chases, which is exactly the flow
     that runs a fade over. Only enforced where a gamma profile exists.

  4. STOP INSIDE A POOL. The ATR stop would sit within 0.25% of an untouched
     liquidity pool on the stop side. That pool IS the target; a stop placed in
     it is a stop placed where the machinery is heading next.

Everything else passes, including CONFLICTED syntheses (the strategy decides)
and instruments with no coverage (fail-OPEN, journalled). Fail-open is the
correct default because this gate refines entries it can see; refusing all
entries whenever a feed hiccups would silently turn a data outage into a
trading halt, and the drawdown switch already owns the halt decision.

Exits are never gated here. Nothing in this file is consulted on the close path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..intel.report import SymbolIntel

logger = logging.getLogger("fable_bot.risk.intel_gate")

CROWD_EXTREME_PCT = 75.0
POOL_PROXIMITY_PCT = 0.25

# Which strategies are breakout-natured and which are fade-natured, for the
# gamma-regime rule. long_or_flat_trend holds through regimes by design and
# is not gated on gamma.
BREAKOUT_STRATEGIES = frozenset({"momentum", "intraday_momentum"})
FADE_STRATEGIES = frozenset({"mean_reversion"})


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str
    evidence: dict = field(default_factory=dict)


def _synthesis_lean(intel: SymbolIntel) -> str | None:
    """'long' / 'short' when forced-flow sections agree, else None."""
    forced = [(s, side) for s, side, _ in intel.leans()
              if s != "technicals" and side in ("long", "short")]
    longs = [1 for _, side in forced if side == "long"]
    shorts = [1 for _, side in forced if side == "short"]
    if len(longs) >= 2 and not shorts:
        return "long"
    if len(shorts) >= 2 and not longs:
        return "short"
    return None


def evaluate_entry(*, side: str, strategy: str, entry_price: float, stop_price: float,
                   intel: SymbolIntel | None) -> GateDecision:
    """side is 'long' or 'short'. Returns the decision with what it looked at."""
    if intel is None:
        return GateDecision(True, "no intel coverage for this instrument (fail-open)",
                            {"coverage": False})

    evidence: dict = {"coverage": True}

    # 1. Trapped with the crowd.
    if intel.sentiment is not None:
        s = intel.sentiment
        evidence["retail_long_pct"] = round(s.long_pct, 1)
        same_side_pct = s.long_pct if side == "long" else s.short_pct
        if same_side_pct >= CROWD_EXTREME_PCT:
            return GateDecision(
                False,
                f"retail is {same_side_pct:.0f}% {side} ({s.source}); entering with the crowd "
                f"at an extreme is being the liquidity",
                evidence)

    # 2. Against the synthesis.
    lean = _synthesis_lean(intel)
    evidence["synthesis_lean"] = lean
    if lean is not None and lean != side:
        reasons = [r for s_, sd, r in intel.leans() if sd == lean and s_ != "technicals"]
        return GateDecision(
            False,
            f"forced-flow sections lean {lean.upper()} against a {side} entry: "
            + "; ".join(reasons[:3]),
            evidence)

    # 3. Strategy vs gamma regime.
    if intel.gamma is not None:
        regime = intel.gamma.regime
        evidence["gamma_regime"] = regime
        if strategy in BREAKOUT_STRATEGIES and regime == "long_gamma":
            return GateDecision(
                False,
                f"{strategy} is a breakout strategy and dealers are LONG gamma "
                f"(net {intel.gamma.net_gex / 1e6:+.1f}M): their hedging sells rallies "
                "and buys dips, which is the flow that fails breakouts",
                evidence)
        if strategy in FADE_STRATEGIES and regime == "short_gamma":
            return GateDecision(
                False,
                f"{strategy} fades extremes and dealers are SHORT gamma "
                f"(net {intel.gamma.net_gex / 1e6:+.1f}M): their hedging chases moves, "
                "which is the flow that runs a fade over",
                evidence)

    # 4. Stop inside a pool.
    if intel.liquidity is not None and entry_price > 0:
        pools = intel.liquidity.below if side == "long" else intel.liquidity.above
        for pool in pools:
            if pool.swept:
                continue
            gap_pct = abs(stop_price - pool.price) / entry_price * 100
            if gap_pct <= POOL_PROXIMITY_PCT:
                evidence["pool_at_stop"] = {"price": pool.price, "kind": pool.kind,
                                            "gap_pct": round(gap_pct, 3)}
                return GateDecision(
                    False,
                    f"the stop ({stop_price:,.5g}) sits {gap_pct:.2f}% from an untouched "
                    f"{pool.kind.replace('_', ' ')} at {pool.price:,.5g}; that pool is where "
                    "price is being driven, not where a stop belongs",
                    evidence)

    return GateDecision(True, "positioning does not oppose the entry", evidence)
