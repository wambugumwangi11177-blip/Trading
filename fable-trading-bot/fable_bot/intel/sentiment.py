"""Retail long/short sentiment: where the crowd is, so as not to be in it.

WHY
---
Retail positioning is the one input that is reliably WRONG at extremes, and
that is exactly what makes it useful. A crowd that is 80% long has already
bought; its stops sit below, in a pool the liquidity map can see, and it has
no marginal buyer left to push price up. The size on the other side knows
this. The user's phrase for it -- "so you're not trapped with the other
traders" -- is the whole thesis in eight words.

SOURCE, AND ITS LIMIT
---------------------
Every free broker-sentiment page tried on 2026-09-20 (Myfxbook outlook, DailyFX
/ IG client sentiment, FXBlue) either returns 403 to a script or renders its
numbers client-side, so there is no static feed to read without an account key.

What IS free and regulated is the CFTC's NON-REPORTABLE category: positions
too small to hit the reporting threshold, which is the small-speculator crowd
in the futures market. It is weekly and it is futures, not spot CFD retail --
but it is real, disclosed, and it covers gold, the euro and the aussie, which
are the three instruments the small account will trade.

A live broker-sentiment feed can be plugged in through `fetch_live_sentiment`
without changing anything downstream; until then this module says which
source it used on every record so the reader can weight it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .cot import (COMMODITY, MARKET_NAMES, NoCotMarket, CotError, DISAGGREGATED_ENDPOINT,
                  TFF_ENDPOINT, FINANCIAL)

# Symbol aliases so the small-account instruments map onto the COT market.
_ALIASES = {
    "XAUUSD": "GLD", "OANDA:XAUUSD": "GLD", "GC=F": "GLD",
    "OANDA:EURUSD": "EURUSD=X", "EURUSD": "EURUSD=X",
    "OANDA:GBPUSD": "GBPUSD=X", "GBPUSD": "GBPUSD=X",
    "OANDA:AUDUSD": "AUDUSD=X", "AUDUSD": "AUDUSD=X",
}


@dataclass(frozen=True)
class Sentiment:
    symbol: str
    long_pct: float          # 0-100
    short_pct: float         # 0-100
    source: str              # e.g. "cftc_nonreportable" or a broker name
    as_of: str
    sample: float = 0.0      # contracts or accounts behind the number

    @property
    def crowd_side(self) -> str:
        if self.long_pct >= 65:
            return "long"
        if self.short_pct >= 65:
            return "short"
        return "balanced"

    @property
    def extreme(self) -> bool:
        return max(self.long_pct, self.short_pct) >= 75

    def summary(self) -> str:
        lines = [f"{self.symbol} retail sentiment ({self.source}, as of {self.as_of}): "
                 f"{self.long_pct:.0f}% long / {self.short_pct:.0f}% short"]
        if self.crowd_side == "balanced":
            lines.append("  crowd is balanced; no contrarian read.")
        else:
            other = "short" if self.crowd_side == "long" else "long"
            stress = "EXTREME -- " if self.extreme else ""
            lines.append(f"  {stress}crowd is {self.crowd_side.upper()}. Their stops are on the "
                         f"{'downside' if self.crowd_side == 'long' else 'upside'}; the path of "
                         f"least resistance for size is to run them before any move {self.crowd_side}.")
            lines.append(f"  contrarian lean: {other}. Not a signal -- a reason not to be "
                         f"{self.crowd_side} without a better one.")
        return "\n".join(lines)


def _f(row: dict, *keys: str) -> float:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def sentiment_from_cot_row(symbol: str, row: dict) -> Sentiment | None:
    """Small-trader long/short share from one CFTC row (either schema)."""
    long_ = _f(row, "nonrept_positions_long_all", "nonrept_positions_long")
    short = _f(row, "nonrept_positions_short_all", "nonrept_positions_short")
    total = long_ + short
    if total <= 0:
        return None
    return Sentiment(
        symbol=symbol,
        long_pct=long_ / total * 100,
        short_pct=short / total * 100,
        source="cftc_nonreportable",
        as_of=str(row.get("report_date_as_yyyy_mm_dd", ""))[:10],
        sample=total,
    )


def fetch_cot_sentiment(symbol: str, *, timeout: float = 30.0) -> Sentiment | None:
    """Latest small-trader positioning for `symbol` from the CFTC.

    Returns None when no futures market maps; raises CotError on a feed
    failure -- the same contract as cot.fetch_cot, for the same reason.
    """
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    key = _ALIASES.get(symbol, symbol)
    mapped = MARKET_NAMES.get(key)
    if mapped is None:
        raise NoCotMarket(f"no CFTC futures market mapped for {symbol}")
    market, report_type = mapped
    endpoint = TFF_ENDPOINT if report_type == FINANCIAL else DISAGGREGATED_ENDPOINT
    query = urllib.parse.urlencode({
        "market_and_exchange_names": market,
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": "1",
    })
    request = urllib.request.Request(f"{endpoint}?{query}",
                                     headers={"User-Agent": "fable-trading-bot/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise CotError(f"could not fetch sentiment for {symbol}: {exc}") from exc
    if not payload:
        raise CotError(f"CFTC returned no rows for {symbol} ({market!r})")
    return sentiment_from_cot_row(symbol, payload[0])


# Hook for a live broker feed (IG, OANDA, Myfxbook with an API key). Assign a
# callable taking a symbol and returning a Sentiment or None; the report will
# prefer it and fall back to COT when it returns None or raises.
fetch_live_sentiment: Callable[[str], Sentiment | None] | None = None


def fetch_sentiment(symbol: str) -> Sentiment | None:
    if fetch_live_sentiment is not None:
        try:
            live = fetch_live_sentiment(symbol)
            if live is not None:
                return live
        except Exception:  # noqa: BLE001 - a broken live hook must not lose the COT read
            pass
    return fetch_cot_sentiment(symbol)
