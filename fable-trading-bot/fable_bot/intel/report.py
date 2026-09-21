"""The daily positioning briefing: what the constrained players are holding,
where the crowd is, where the stops are, and what price is doing about it.

Sections, in order of how much they are FORCED rather than chosen:

  1. GAMMA       -- where dealer hedging flow is obligated this week (gamma.py)
  2. COT         -- how banks, hedgers and funds are positioned by CFTC filing (cot.py)
  3. SENTIMENT   -- where the retail crowd is, so as not to be in it (sentiment.py)
  4. LIQUIDITY   -- where resting stops are pooled and which pools were taken (liquidity.py)
  5. TECHNICALS  -- trend, RSI, ATR, ranges: context, not reason (technicals.py)
  6. SYNTHESIS   -- do the forced-flow sections agree, and on which side

Every section degrades independently. A feed outage prints as a stated gap in
that section, never as a missing line or a silent zero -- a briefing whose
failures are invisible is worse than no briefing, because it gets trusted at
full weight while quietly saying nothing.

No section emits a trade signal, and nothing here is wired into order routing.
The synthesis is a lean with reasons attached. This informs a human; it does
not size a position.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..data.feed import drop_incomplete_bar, fetch_history
from .fmt import px
from .cot import CotError, CotReport, NoCotMarket, fetch_cot
from .gamma import GammaProfile
from .liquidity import LiquidityMap, build_liquidity_map
from .options_feed import OptionsFeedError, fetch_gamma_profile
from .sentiment import Sentiment, fetch_sentiment
from .technicals import TechnicalSnapshot, compute_technicals

logger = logging.getLogger("fable_bot.intel.report")


@dataclass(frozen=True)
class Coverage:
    """Which data source serves each section for one display symbol.

    Spot gold has no listed options of its own; GLD's chain is the liquid
    proxy and tracks it ~1:1, which is stated on the profile. Spot FX has no
    liquid US-listed option chain worth reading, so gamma is None there and
    the section says so instead of producing an empty profile.
    """
    display: str
    history: str                 # yfinance ticker for daily bars
    gamma: str | None            # options underlying, or None
    cot: str | None              # key into cot.MARKET_NAMES, or None


COVERAGE: dict[str, Coverage] = {
    "XAUUSD": Coverage("XAUUSD", "GC=F", "GLD", "GLD"),
    "EURUSD": Coverage("EURUSD", "EURUSD=X", None, "EURUSD=X"),
    "GBPUSD": Coverage("GBPUSD", "GBPUSD=X", None, "GBPUSD=X"),
    "AUDUSD": Coverage("AUDUSD", "AUDUSD=X", None, "AUDUSD=X"),
    "GLD": Coverage("GLD", "GLD", "GLD", "GLD"),
    "SPY": Coverage("SPY", "SPY", "SPY", "SPY"),
    "QQQ": Coverage("QQQ", "QQQ", "QQQ", "QQQ"),
    "USO": Coverage("USO", "USO", "USO", "USO"),
    "IWM": Coverage("IWM", "IWM", "IWM", None),
}

def coverage_for_ticker(ticker: str) -> Coverage | None:
    """Coverage whose history feed is `ticker` (a yfinance symbol), if any.

    live_runner instruments are keyed by their yfinance ticker; the briefing
    is keyed by display name. MGC=F has no entry of its own because gold
    futures and spot read the same positioning -- it resolves to XAUUSD.
    """
    aliases = {"MGC=F": "GC=F", "GC=F": "GC=F"}
    wanted = aliases.get(ticker, ticker)
    for cov in COVERAGE.values():
        if cov.history == wanted:
            return cov
    return None


# What the small account will trade, in the user's order of preference.
SMALL_ACCOUNT_UNIVERSE: tuple[str, ...] = ("XAUUSD", "EURUSD", "AUDUSD")
GAMMA_UNIVERSE: tuple[str, ...] = ("SPY", "QQQ", "GLD", "USO", "IWM")
DEFAULT_UNIVERSE: tuple[str, ...] = SMALL_ACCOUNT_UNIVERSE


@dataclass
class SymbolIntel:
    """Everything known about one symbol's positioning, plus what failed."""

    symbol: str
    gamma: GammaProfile | None = None
    cot: CotReport | None = None
    sentiment: Sentiment | None = None
    liquidity: LiquidityMap | None = None
    technicals: TechnicalSnapshot | None = None
    gaps: list[str] = field(default_factory=list)

    # ── synthesis ────────────────────────────────────────────────────────

    def leans(self) -> list[tuple[str, str, str]]:
        """(section, side, reason) for each section with an opinion.

        side is 'long', 'short' or 'flat'. Sections with nothing to say are
        omitted rather than counted as flat -- an outage is not a vote.
        """
        out: list[tuple[str, str, str]] = []

        if self.gamma is not None:
            g = self.gamma
            if g.regime == "long_gamma":
                out.append(("gamma", "flat",
                            f"dealers long gamma; hedging leans AGAINST moves, expect pinning"
                            + (f" near {g.call_wall:,.0f}/{g.put_wall:,.0f}" if g.call_wall and g.put_wall else "")))
            else:
                out.append(("gamma", "trend",
                            "dealers short gamma; hedging leans WITH moves, expect extension"))

        if self.cot is not None:
            c = self.cot
            if c.is_stale:
                out.append(("cot", "flat", "report is stale -- ignored"))
            elif c.crowding in ("extreme_long", "crowded_long"):
                out.append(("cot", "short",
                            f"{c.spec_label} {c.managed_money_net_percentile:.0f}th pct long: "
                            "marginal buyer already in"))
            elif c.crowding in ("extreme_short", "crowded_short"):
                out.append(("cot", "long",
                            f"{c.spec_label} {c.managed_money_net_percentile:.0f}th pct short: "
                            "marginal seller already in"))
            else:
                out.append(("cot", "flat", f"{c.spec_label} positioning neutral"))

        if self.sentiment is not None:
            s = self.sentiment
            if s.crowd_side == "long":
                out.append(("sentiment", "short",
                            f"retail {s.long_pct:.0f}% long; their stops are below"))
            elif s.crowd_side == "short":
                out.append(("sentiment", "long",
                            f"retail {s.short_pct:.0f}% short; their stops are above"))
            else:
                out.append(("sentiment", "flat", "retail balanced"))

        if self.liquidity is not None:
            lq = self.liquidity
            na, nb = lq.nearest_structural_above, lq.nearest_structural_below
            if na and nb:
                # The nearer untouched pool is the likelier first destination.
                if abs(na.distance_pct) < abs(nb.distance_pct) * 0.6:
                    out.append(("liquidity", "long",
                                f"nearest untouched pool is ABOVE at {px(na.price)} ({na.distance_pct:+.2f}%)"))
                elif abs(nb.distance_pct) < abs(na.distance_pct) * 0.6:
                    out.append(("liquidity", "short",
                                f"nearest untouched pool is BELOW at {px(nb.price)} ({nb.distance_pct:+.2f}%)"))
                else:
                    out.append(("liquidity", "flat",
                                f"pools roughly equidistant: {px(na.price)} above, {px(nb.price)} below"))
            # Only STRUCTURAL sweeps vote. Every down day takes the prior-day
            # low and every up day takes the prior-day high; calling each one
            # a "reversal setup" would have the section flip-flop daily.
            structural_sweeps = [p for p in lq.recently_swept if p.kind in lq.STRUCTURAL]
            if structural_sweeps:
                last = structural_sweeps[0]
                out.append(("liquidity", "long" if last.side == "below" else "short",
                            f"{last.kind.replace('_', ' ')} at {px(last.price)} was just SWEPT; "
                            "reversal from a taken pool is the classic setup"))

        if self.technicals is not None:
            t = self.technicals
            if t.trend == "up":
                out.append(("technicals", "long", f"EMA stack up, RSI {t.rsi14:.0f}"))
            elif t.trend == "down":
                out.append(("technicals", "short", f"EMA stack down, RSI {t.rsi14:.0f}"))
            else:
                out.append(("technicals", "flat", "EMAs mixed"))
            if t.rsi_state != "neutral":
                out.append(("technicals", "short" if t.rsi_state == "overbought" else "long",
                            f"RSI {t.rsi_state} at {t.rsi14:.0f}"))

        return out

    def synthesis(self) -> str:
        leans = self.leans()
        if not leans:
            return "  no sections available -- nothing to synthesise."
        longs = [r for s, side, r in leans if side == "long"]
        shorts = [r for s, side, r in leans if side == "short"]
        regime = next((r for s, side, r in leans if s == "gamma"), None)

        lines = []
        # Forced-flow sections (cot, sentiment, liquidity) outweigh technicals.
        forced_long = [r for s, side, r in leans if side == "long" and s != "technicals"]
        forced_short = [r for s, side, r in leans if side == "short" and s != "technicals"]
        if len(forced_long) >= 2 and not forced_short:
            verdict = "LEAN LONG -- the forced-flow sections agree"
        elif len(forced_short) >= 2 and not forced_long:
            verdict = "LEAN SHORT -- the forced-flow sections agree"
        elif forced_long and forced_short:
            verdict = "CONFLICTED -- forced-flow sections disagree; the edge is in waiting"
        elif forced_long or forced_short:
            side = "LONG" if forced_long else "SHORT"
            verdict = f"WEAK LEAN {side} -- one forced-flow section, unconfirmed"
        else:
            verdict = "NO LEAN -- nothing forced is pointing anywhere"
        lines.append(f"  {verdict}")
        if regime:
            lines.append(f"  regime: {regime}")
        for reason in longs:
            lines.append(f"    + long : {reason}")
        for reason in shorts:
            lines.append(f"    - short: {reason}")
        return "\n".join(lines)

    # ── rendering ────────────────────────────────────────────────────────

    def lines(self) -> list[str]:
        out: list[str] = []
        if self.gamma is not None:
            out.append(self.gamma.summary())
        if self.cot is not None:
            out.append(self.cot.summary())
        if self.sentiment is not None:
            out.append(self.sentiment.summary())
        if self.liquidity is not None:
            out.append(self.liquidity.summary())
        if self.technicals is not None:
            out.append(self.technicals.summary())
        for gap in self.gaps:
            out.append(f"  [gap] {gap}")
        out.append("SYNTHESIS")
        out.append(self.synthesis())
        return out


def gather(symbol: str, *, max_expiries: int = 4, cot_weeks: int = 156,
           history_period: str = "1y") -> SymbolIntel:
    """Collect every section for one symbol, recording rather than raising failures."""
    cov = COVERAGE.get(symbol) or Coverage(symbol, symbol, symbol, symbol)
    intel = SymbolIntel(symbol=symbol)

    if cov.gamma is not None:
        try:
            intel.gamma = fetch_gamma_profile(cov.gamma, max_expiries=max_expiries)
        except OptionsFeedError as exc:
            intel.gaps.append(f"gamma unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001 - a briefing must not die on a feed
            logger.exception("gamma failed for %s", symbol)
            intel.gaps.append(f"gamma failed: {type(exc).__name__}: {exc}")
    else:
        intel.gaps.append("gamma: no liquid listed option chain for this instrument")

    if cov.cot is not None:
        try:
            intel.cot = fetch_cot(cov.cot, weeks=cot_weeks)
        except NoCotMarket:
            pass
        except CotError as exc:
            intel.gaps.append(f"COT unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("COT failed for %s", symbol)
            intel.gaps.append(f"COT failed: {type(exc).__name__}: {exc}")

        try:
            intel.sentiment = fetch_sentiment(cov.cot)
            if intel.sentiment is not None:
                intel.sentiment = Sentiment(**{**intel.sentiment.__dict__, "symbol": symbol})
        except NoCotMarket:
            pass
        except CotError as exc:
            intel.gaps.append(f"sentiment unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("sentiment failed for %s", symbol)
            intel.gaps.append(f"sentiment failed: {type(exc).__name__}: {exc}")

    try:
        history = fetch_history(cov.history, period=history_period)
        if history is not None and not history.empty:
            # yfinance returns today's forming bar; on a Sunday that is a
            # few hours of thin FX trade masquerading as a daily candle.
            history = drop_incomplete_bar(history)
        if history is None or history.empty:
            intel.gaps.append(f"no price history for {cov.history}")
        else:
            try:
                intel.technicals = compute_technicals(symbol, history)
            except Exception as exc:  # noqa: BLE001
                intel.gaps.append(f"technicals failed: {type(exc).__name__}: {exc}")
            try:
                intel.liquidity = build_liquidity_map(symbol, history)
            except Exception as exc:  # noqa: BLE001
                intel.gaps.append(f"liquidity map failed: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        intel.gaps.append(f"price history failed: {type(exc).__name__}: {exc}")

    return intel


def build_market_report(symbols: tuple[str, ...] | list[str] = DEFAULT_UNIVERSE,
                        *, now: datetime | None = None,
                        max_expiries: int = 4) -> str:
    """Render the full positioning briefing as plain text."""
    now = now or datetime.now(timezone.utc)
    header = [
        "POSITIONING BRIEFING",
        f"generated {now.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Sections run from most-forced to least: dealer gamma (obligated hedging),",
        "COT (regulated disclosure), retail sentiment (the crowd), liquidity (where",
        "stops pool), technicals (context). Gamma uses end-of-day open interest and",
        "the naive dealer convention; COT and sentiment are weekly and state their",
        "staleness. Spot gold reads GLD's option chain as its gamma proxy.",
        "",
    ]
    body: list[str] = []
    for symbol in symbols:
        intel = gather(symbol, max_expiries=max_expiries)
        body.append(f"=== {symbol} " + "=" * max(0, 60 - len(symbol)))
        body.extend(intel.lines())
        body.append("")
    return "\n".join(header + body).rstrip() + "\n"
