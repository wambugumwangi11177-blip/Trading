"""CFTC Commitments of Traders: who is actually holding the futures.

WHY THIS EXISTS
---------------
Gamma exposure (gamma.py) says where hedging flow is FORCED this week. COT says
who is POSITIONED, by legally-mandated disclosure, over weeks and months. They
answer different questions and neither substitutes for the other.

Every large futures position in a US-regulated market is reported to the CFTC
and published weekly. The disaggregated report splits open interest into:

  * PRODUCER / MERCHANT / PROCESSOR / USER -- commercial hedgers. Miners,
    refiners, jewellers, bullion banks. They are short gold because they own
    gold, not because they are bearish. They are the "smart money" only in the
    sense that they are closest to physical supply; they are price-insensitive
    and usually early.

  * MANAGED MONEY -- CTAs and hedge funds. Trend followers. Their position IS
    the crowded trade, and extremes in it mark where a trend has run out of
    marginal buyers. This is the series worth watching for exhaustion.

  * SWAP DEALERS -- banks intermediating OTC exposure, hedging client swaps.

The tradeable observation is not the level but the EXTREME: managed money at a
multi-year net-long percentile with commercials at the mirror-image short is
the configuration that precedes most sharp gold reversals. That is a
positioning statement, not a timing one, and this module does not pretend
otherwise: it reports percentiles and never emits a signal.

DATA AND ITS LAG
----------------
Source is the CFTC's own Socrata endpoint (public, no key). The report is taken
Tuesday and published Friday afternoon, so the freshest number available is
always at least three days stale and can be eight. `days_stale` is on every
record because acting on COT as if it were live is the standard way to be wrong
with it.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger("fable_bot.intel.cot")

# The CFTC splits its weekly data across two reports with DIFFERENT trader
# categories, and using the wrong one for a market returns zero rows -- which
# is easy to misread as "no positioning" rather than "wrong dataset".
#
#   DISAGGREGATED (commodities: metals, energy, ags)
#       producer/merchant | swap dealers | managed money | other reportables
#
#   TFF, Traders in Financial Futures (equity indices, rates, FX, bitcoin)
#       dealer/intermediary | asset manager | leveraged funds | other
#
# "Dealer/Intermediary" in the TFF report is the sell-side bank desk -- the
# closest thing to a published bank positioning series, and the counterpart to
# the dealer gamma estimated in gamma.py. One is disclosed, the other inferred.
DISAGGREGATED_ENDPOINT = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
TFF_ENDPOINT = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"

COMMODITY = "disaggregated"
FINANCIAL = "tff"

# Market names are the agency's exact strings; a near-miss returns an empty
# result rather than the wrong market, which is why they are pinned here.
MARKET_NAMES: dict[str, tuple[str, str]] = {
    "GLD": ("GOLD - COMMODITY EXCHANGE INC.", COMMODITY),
    "GC=F": ("GOLD - COMMODITY EXCHANGE INC.", COMMODITY),
    "MGC=F": ("GOLD - COMMODITY EXCHANGE INC.", COMMODITY),
    "XAUUSD": ("GOLD - COMMODITY EXCHANGE INC.", COMMODITY),
    "SI=F": ("SILVER - COMMODITY EXCHANGE INC.", COMMODITY),
    "XAGUSD": ("SILVER - COMMODITY EXCHANGE INC.", COMMODITY),
    "USO": ("WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE", COMMODITY),
    "CL=F": ("WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE", COMMODITY),
    # Verified against the live TFF dataset on 2026-09-20. These strings drift:
    # an earlier guess ("E-MINI S&P 500 STOCK INDEX ...") matched only rows
    # that stopped updating in 2022 and returned a 1,692-day-old report that
    # looked perfectly well-formed. That is why STALE_REPORT_DAYS exists below.
    "SPY": ("E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", FINANCIAL),
    "MES=F": ("MICRO E-MINI S&P 500 INDEX - CHICAGO MERCANTILE EXCHANGE", FINANCIAL),
    "QQQ": ("NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE", FINANCIAL),
    "MNQ=F": ("MICRO E-MINI NASDAQ-100 INDEX - CHICAGO MERCANTILE EXCHANGE", FINANCIAL),
    "BTC-USD": ("BITCOIN - CHICAGO MERCANTILE EXCHANGE", FINANCIAL),
    "EURUSD=X": ("EURO FX - CHICAGO MERCANTILE EXCHANGE", FINANCIAL),
    "GBPUSD=X": ("BRITISH POUND - CHICAGO MERCANTILE EXCHANGE", FINANCIAL),
    "USDJPY=X": ("JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", FINANCIAL),
    "AUDUSD=X": ("AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", FINANCIAL),
}


# COT is taken Tuesday and published Friday, so 3-8 days stale is normal and
# ~14 days covers a holiday week. Anything beyond that is not a late report --
# it means the pinned market name is matching an archived series that stopped
# updating, which is exactly how a 2022 row was nearly served as current.
STALE_REPORT_DAYS = 21


class NoCotMarket(LookupError):
    """This symbol has no CFTC futures market mapped. Not an error, a fact."""


class CotError(RuntimeError):
    """The COT feed could not be read. Callers degrade; they do not crash."""


@dataclass(frozen=True)
class CotReport:
    """One week's positioning for one market."""

    market: str
    report_date: str
    open_interest: float
    # Hedger side. In the disaggregated report this is producer/merchant; in
    # TFF it is dealer/intermediary -- the sell-side bank desk. `report_type`
    # says which, because they are not the same players.
    producer_long: float
    producer_short: float
    # Speculative side: managed money (disaggregated) or leveraged funds (TFF).
    managed_money_long: float
    managed_money_short: float
    swap_long: float
    swap_short: float
    days_stale: int
    report_type: str = COMMODITY
    # Percentile of managed-money net positioning within the supplied history
    # (0-100). None when there was not enough history to rank it.
    managed_money_net_percentile: float | None = None
    history_weeks: int = 0

    @property
    def managed_money_net(self) -> float:
        return self.managed_money_long - self.managed_money_short

    @property
    def producer_net(self) -> float:
        return self.producer_long - self.producer_short

    @property
    def hedger_label(self) -> str:
        return "dealer/intermediary (banks)" if self.report_type == FINANCIAL else "commercial hedgers"

    @property
    def spec_label(self) -> str:
        return "leveraged funds" if self.report_type == FINANCIAL else "managed money"

    @property
    def managed_money_net_pct_oi(self) -> float:
        if self.open_interest <= 0:
            return 0.0
        return self.managed_money_net / self.open_interest * 100

    @property
    def is_stale(self) -> bool:
        """True when this report is too old to be the current week's."""
        return self.days_stale > STALE_REPORT_DAYS

    @property
    def crowding(self) -> str:
        """Plain reading of how stretched the speculative side is."""
        pct = self.managed_money_net_percentile
        if pct is None:
            return "unknown"
        if pct >= 90:
            return "extreme_long"
        if pct >= 75:
            return "crowded_long"
        if pct <= 10:
            return "extreme_short"
        if pct <= 25:
            return "crowded_short"
        return "neutral"

    def summary(self) -> str:
        lines = [
            f"{self.market} -- COT week ending {self.report_date} ({self.days_stale}d stale)",
            f"  {self.spec_label} net {self.managed_money_net:+,.0f} "
            f"({self.managed_money_net_pct_oi:+.1f}% of OI)",
            f"  {self.hedger_label} net {self.producer_net:+,.0f}",
        ]
        if self.is_stale:
            lines.insert(1, f"  *** STALE: {self.days_stale} days old (expected <= "
                            f"{STALE_REPORT_DAYS}). The pinned market name is probably "
                            "matching an archived series -- do not trade this. ***")
        if self.managed_money_net_percentile is not None:
            lines.append(f"  speculative crowding: {self.crowding.replace('_', ' ').upper()} "
                         f"({self.managed_money_net_percentile:.0f}th percentile "
                         f"of {self.history_weeks} weeks)")
        if self.crowding in ("extreme_long", "extreme_short"):
            lines.append("  reading: positioning is stretched. Trend continuation needs NEW "
                         "money, and the marginal buyer is largely already in.")
        return "\n".join(lines)


def _f(row: dict, *keys: str) -> float:
    """First present numeric field among `keys`.

    The Socrata schema has renamed these columns more than once. Trying several
    spellings costs nothing and prevents a silent zero -- which would read as
    "flat positioning" rather than "field not found", the more dangerous of the
    two failures.
    """
    for key in keys:
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def parse_cot_rows(rows: list[dict], *, now: datetime | None = None,
                   report_type: str = COMMODITY) -> CotReport | None:
    """Fold raw Socrata rows (newest first) into the latest report, with percentile.

    Pure: no network. The percentile ranks the newest managed-money net against
    every week supplied, so passing three years of history gives a three-year
    percentile.
    """
    if not rows:
        return None
    now = now or datetime.now(timezone.utc)

    # Column names differ per report; each tuple is tried in order so one
    # parser serves both schemas.
    if report_type == FINANCIAL:
        spec_long = ("lev_money_positions_long_all", "lev_money_positions_long")
        spec_short = ("lev_money_positions_short_all", "lev_money_positions_short")
        hedge_long = ("dealer_positions_long_all", "dealer_positions_long")
        hedge_short = ("dealer_positions_short_all", "dealer_positions_short")
        swap_long = ("asset_mgr_positions_long_all", "asset_mgr_positions_long")
        swap_short = ("asset_mgr_positions_short_all", "asset_mgr_positions_short")
    else:
        spec_long = ("m_money_positions_long_all", "m_money_positions_long")
        spec_short = ("m_money_positions_short_all", "m_money_positions_short")
        hedge_long = ("prod_merc_positions_long_all", "prod_merc_positions_long")
        hedge_short = ("prod_merc_positions_short_all", "prod_merc_positions_short")
        swap_long = ("swap_positions_long_all", "swap_positions_long")
        swap_short = ("swap__positions_short_all", "swap_positions_short_all",
                      "swap_positions_short")

    def net_of(row: dict) -> float:
        return _f(row, *spec_long) - _f(row, *spec_short)

    # Newest first is the endpoint's order; do not rely on it.
    ordered = sorted(rows, key=lambda r: str(r.get("report_date_as_yyyy_mm_dd", "")), reverse=True)
    latest = ordered[0]
    report_date = str(latest.get("report_date_as_yyyy_mm_dd", ""))[:10]

    days_stale = 0
    try:
        stamp = datetime.fromisoformat(report_date).replace(tzinfo=timezone.utc)
        days_stale = max(0, (now - stamp).days)
    except ValueError:
        pass

    history = [net_of(r) for r in ordered]
    percentile: float | None = None
    if len(history) >= 8:
        current = history[0]
        below = sum(1 for value in history if value < current)
        percentile = below / len(history) * 100

    return CotReport(
        market=str(latest.get("market_and_exchange_names", "")),
        report_date=report_date,
        open_interest=_f(latest, "open_interest_all", "open_interest"),
        producer_long=_f(latest, *hedge_long),
        producer_short=_f(latest, *hedge_short),
        managed_money_long=_f(latest, *spec_long),
        managed_money_short=_f(latest, *spec_short),
        swap_long=_f(latest, *swap_long),
        swap_short=_f(latest, *swap_short),
        days_stale=days_stale,
        report_type=report_type,
        managed_money_net_percentile=percentile,
        history_weeks=len(history),
    )


def fetch_cot(symbol: str, *, weeks: int = 156, timeout: float = 30.0) -> CotReport | None:
    """Latest COT for the market behind `symbol`, ranked against `weeks` of history.

    Raises NoCotMarket when the symbol has no mapped futures market, and
    CotError on an actual feed failure. Those are different facts and callers
    report them differently: "this instrument has no COT" is not an outage.
    """
    mapped = MARKET_NAMES.get(symbol)
    if mapped is None:
        raise NoCotMarket(f"no CFTC futures market mapped for {symbol}")
    market, report_type = mapped
    endpoint = TFF_ENDPOINT if report_type == FINANCIAL else DISAGGREGATED_ENDPOINT

    query = urllib.parse.urlencode({
        "market_and_exchange_names": market,
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": str(int(weeks)),
    })
    url = f"{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "fable-trading-bot/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise CotError(f"could not fetch COT for {symbol} ({market}): {exc}") from exc

    if not isinstance(payload, list):
        raise CotError(f"unexpected COT payload shape for {symbol}")
    if not payload:
        # An empty result almost always means the pinned market string no
        # longer matches the agency's, not that positioning vanished. Say so.
        raise CotError(
            f"CFTC returned no rows for {symbol} (market {market!r}, {report_type} report) "
            "-- the pinned market name may no longer match")
    return parse_cot_rows(payload, report_type=report_type)
