"""The daily positioning briefing: what the constrained players are holding.

This is the "what are the banks and funds actually doing" report. It answers
that with disclosure and hedging mechanics rather than inference from price:

  * GAMMA (gamma.py)      -- where dealer hedging flow is FORCED this week, and
                             which side of the gamma flip price is on.
  * COT (cot.py)          -- how commercial hedgers and managed money are
                             POSITIONED, by CFTC filing, and how stretched that
                             positioning is against its own history.

Every section degrades independently. A feed outage prints as a stated gap in
that section, never as a missing line or a silent zero -- an intelligence
report whose failures are invisible is worse than no report, because it gets
trusted at full weight while quietly saying nothing.

No section emits a trade signal, and nothing here is wired into order routing.
This informs a human; it does not size a position.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .cot import CotError, CotReport, NoCotMarket, fetch_cot
from .gamma import GammaProfile
from .options_feed import OptionsFeedError, fetch_gamma_profile

logger = logging.getLogger("fable_bot.intel.report")

# Underlyings with listed options liquid enough for a gamma profile to mean
# anything. Spot FX and world cash indices have no US listed option chain, so
# they are deliberately absent rather than silently producing an empty profile.
GAMMA_UNIVERSE: tuple[str, ...] = ("SPY", "QQQ", "GLD", "USO", "IWM")


@dataclass
class SymbolIntel:
    """Everything known about one symbol's positioning, plus what failed."""

    symbol: str
    gamma: GammaProfile | None = None
    cot: CotReport | None = None
    gaps: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out: list[str] = []
        if self.gamma is not None:
            out.append(self.gamma.summary())
        if self.cot is not None:
            out.append(self.cot.summary())
        for gap in self.gaps:
            out.append(f"  [gap] {gap}")
        return out or [f"{self.symbol}: nothing available."]


def gather(symbol: str, *, max_expiries: int = 4, cot_weeks: int = 156) -> SymbolIntel:
    """Collect gamma and COT for one symbol, recording rather than raising failures."""
    intel = SymbolIntel(symbol=symbol)

    try:
        intel.gamma = fetch_gamma_profile(symbol, max_expiries=max_expiries)
    except OptionsFeedError as exc:
        intel.gaps.append(f"gamma unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001 - a briefing must not die on a feed
        logger.exception("gamma failed for %s", symbol)
        intel.gaps.append(f"gamma failed: {type(exc).__name__}: {exc}")

    try:
        intel.cot = fetch_cot(symbol, weeks=cot_weeks)
    except NoCotMarket:
        # Not a failure: spot FX pairs and cash indices genuinely have no CFTC
        # futures market. Saying nothing is correct; saying "unavailable" would
        # imply an outage that is not happening.
        pass
    except CotError as exc:
        intel.gaps.append(f"COT unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("COT failed for %s", symbol)
        intel.gaps.append(f"COT failed: {type(exc).__name__}: {exc}")

    return intel


def build_market_report(symbols: tuple[str, ...] | list[str] = GAMMA_UNIVERSE,
                        *, now: datetime | None = None,
                        max_expiries: int = 4) -> str:
    """Render the full positioning briefing as plain text."""
    now = now or datetime.now(timezone.utc)
    header = [
        "POSITIONING BRIEFING",
        f"generated {now.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Dealer gamma is computed under the standard naive convention (dealers long",
        "calls, short puts) from end-of-day open interest, so it describes the book",
        "as of the last settlement, not the live tape. COT is filed Tuesday and",
        "published Friday; its staleness is stated per market.",
        "",
    ]
    body: list[str] = []
    for symbol in symbols:
        intel = gather(symbol, max_expiries=max_expiries)
        body.append(f"--- {symbol} " + "-" * max(0, 60 - len(symbol)))
        body.extend(intel.lines())
        body.append("")
    return "\n".join(header + body).rstrip() + "\n"
