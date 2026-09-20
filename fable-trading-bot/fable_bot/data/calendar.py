"""Economic calendar feed -- the bot's market intelligence source.

Price history tells the strategies what *has* happened; this module tells them
what is *scheduled* to happen. High-impact macro releases (FOMC, CPI, NFP, EIA
inventories) routinely gap every instrument in the universe at once, which is
exactly the kind of move ATR-based stops are worst at containing: the stop is
sized off recent volatility, then the release prints a bar many multiples wider
than that. Knowing a release is 20 minutes away is what lets the bot stand aside
instead of sizing a position as if it were a normal session.

Source and its caveats
----------------------
Data comes from Investing.com's economic-calendar AJAX endpoint. There is no
maintained Python client for it: `investpy` has been Cloudflare-blocked (403)
since 2022, and its successor `investiny` covers historical prices only, not
calendars. So this module talks to the endpoint directly and parses the HTML
fragment it returns.

That means this is scraping, with the usual consequences -- it is unofficial,
subject to Investing.com's terms, and a markup change on their side will break
parsing. Two design choices follow from that:

  * `fetch_economic_calendar` raises `CalendarUnavailable` on *any* failure
    (network, HTTP, empty parse) rather than returning a silently empty list.
    An empty calendar and an unreachable calendar mean opposite things for
    risk, and callers must be able to tell them apart.
  * Everything below the `EconomicEvent` dataclass is provider-specific. To
    swap sources, write another `fetch_*` returning `list[EconomicEvent]`; the
    consumer in risk/event_filter.py needs no changes.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Sequence

import requests

from ..config import PROJECT_ROOT

logger = logging.getLogger("fable_bot.data.calendar")

_ENDPOINT = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"

# Investing.com's internal timezone id for UTC. Verified empirically: a release
# known to print at 09:45 US Eastern comes back as 13:45 under this id during
# EDT (UTC-4). Requesting UTC explicitly keeps DST out of the parsing path.
_TZ_UTC_ID = 55

# Investing.com's internal country ids. US only by default -- the whole
# instrument universe (SPY/QQQ/BTC/GLD/USO) is USD-denominated and moves on US
# macro, so other countries would add noise without adding signal.
COUNTRY_US = 5

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

_CACHE_PATH = PROJECT_ROOT / ".cache" / "economic_calendar.json"


class CalendarUnavailable(RuntimeError):
    """Raised when the calendar could not be fetched or parsed.

    Deliberately distinct from "there are no events": callers decide whether an
    unavailable calendar should block trading or be traded through.
    """


@dataclass(frozen=True)
class EconomicEvent:
    """One scheduled macro release, normalised and provider-agnostic."""

    event_id: str
    when: datetime  # always timezone-aware UTC
    currency: str  # e.g. "USD"
    country: str  # e.g. "United States"
    importance: int  # 1 low, 2 medium, 3 high
    name: str
    actual: str = ""
    forecast: str = ""
    previous: str = ""

    def minutes_until(self, now: datetime) -> float:
        """Signed minutes from `now` to the release (negative once it has passed)."""
        return (self.when - now).total_seconds() / 60.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["when"] = self.when.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EconomicEvent":
        d = dict(d)
        d["when"] = datetime.fromisoformat(d["when"])
        return cls(**d)


class _EventRowParser(HTMLParser):
    """Extracts event rows from the HTML fragment the endpoint returns.

    Anchors on stable ids (`eventRowId_*`, `eventActual_*`) and on individual
    class tokens rather than on cell order, so an inserted column or a reordered
    class attribute does not silently shift every field by one.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[EconomicEvent] = []
        self._row: dict | None = None
        self._field: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag == "tr":
            self._commit_row()
            classes = a.get("class", "").split()
            if "js-event-item" in classes:
                self._row = {
                    "event_id": a.get("id", "").replace("eventRowId_", ""),
                    "raw_datetime": a.get("data-event-datetime", ""),
                    "importance": 0,
                    "currency": "",
                    "country": "",
                    "name": "",
                    "actual": "",
                    "forecast": "",
                    "previous": "",
                }
            return

        if self._row is None:
            return

        if tag == "td":
            self._flush_field()
            cell_id = a.get("id", "")
            classes = a.get("class", "").split()
            if cell_id.startswith("eventActual_"):
                self._field = "actual"
            elif cell_id.startswith("eventForecast_"):
                self._field = "forecast"
            elif cell_id.startswith("eventPrevious_"):
                self._field = "previous"
            elif "flagCur" in classes:
                self._field = "currency"
            elif "sentiment" in classes:
                self._field = None
                m = re.fullmatch(r"bull(\d)", a.get("data-img_key", ""))
                if m:
                    self._row["importance"] = int(m.group(1))
            elif "event" in classes:
                # Note: the actual/forecast/previous cells carry classes like
                # "event-553426-actual", which is why this tests for the exact
                # "event" token and is checked after the id-based branches.
                self._field = "name"
            else:
                self._field = None
            self._text = []
        elif tag == "span" and self._field == "currency" and a.get("title"):
            # The flag span carries the full country name; the cell text is the
            # currency code.
            self._row["country"] = a["title"]

    def handle_endtag(self, tag: str) -> None:
        if tag == "td":
            self._flush_field()
        elif tag == "tr":
            self._commit_row()

    def handle_data(self, data: str) -> None:
        if self._row is not None and self._field:
            self._text.append(data)

    def close(self) -> None:  # noqa: D102 - inherited behaviour
        super().close()
        self._commit_row()

    def _flush_field(self) -> None:
        if self._row is not None and self._field:
            value = " ".join("".join(self._text).split())
            # The endpoint uses a non-breaking space as its "no value" filler.
            self._row[self._field] = "" if value in {"\xa0", "-"} else value
        self._field = None
        self._text = []

    def _commit_row(self) -> None:
        row = self._row
        self._row = None
        self._field = None
        self._text = []
        if not row or not row["raw_datetime"]:
            return
        try:
            when = datetime.strptime(row["raw_datetime"], "%Y/%m/%d %H:%M:%S")
        except ValueError:
            # All-day entries and bank holidays carry no usable timestamp;
            # they have no release moment to stand aside for.
            return
        self.events.append(EconomicEvent(
            event_id=row["event_id"],
            when=when.replace(tzinfo=timezone.utc),
            currency=row["currency"],
            country=row["country"],
            importance=row["importance"],
            name=row["name"],
            actual=row["actual"],
            forecast=row["forecast"],
            previous=row["previous"],
        ))


def parse_calendar_html(html: str) -> list[EconomicEvent]:
    """Parse the endpoint's HTML fragment into events, sorted by release time."""
    parser = _EventRowParser()
    parser.feed(html)
    parser.close()
    return sorted(parser.events, key=lambda e: e.when)


def fetch_economic_calendar(
    *,
    countries: Sequence[int] = (COUNTRY_US,),
    min_importance: int = 3,
    window: str = "thisWeek",
    session: requests.Session | None = None,
    timeout: float = 20.0,
) -> list[EconomicEvent]:
    """Fetch upcoming macro releases.

    Args:
        countries: Investing.com country ids (see COUNTRY_US).
        min_importance: 1 low / 2 medium / 3 high -- requests this level and above.
        window: endpoint tab, e.g. "today", "tomorrow", "thisWeek", "nextWeek".

    Raises:
        CalendarUnavailable: on any network, HTTP, decoding, or parse failure.
    """
    payload = [("country[]", str(c)) for c in countries]
    payload += [("importance[]", str(i)) for i in range(min_importance, 4)]
    payload += [
        ("timeZone", str(_TZ_UTC_ID)),
        ("timeFilter", "timeRemain"),
        ("currentTab", window),
        ("limit_from", "0"),
    ]
    headers = {
        "User-Agent": _DEFAULT_USER_AGENT,
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://www.investing.com/economic-calendar/",
        "Accept": "*/*",
    }
    http = session or requests
    try:
        response = http.post(_ENDPOINT, data=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        html = response.json()["data"]
    except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
        raise CalendarUnavailable(f"could not fetch economic calendar: {exc}") from exc

    events = parse_calendar_html(html)
    if not events:
        # A successful response that yields nothing almost always means their
        # markup changed, not that the world has no scheduled data this week.
        raise CalendarUnavailable(
            f"fetched {len(html)} chars but parsed 0 events -- the source markup likely changed"
        )
    return events


def load_economic_calendar(
    *,
    cache_ttl_minutes: int = 180,
    cache_path: Path | None = None,
    now: datetime | None = None,
    **fetch_kwargs,
) -> list[EconomicEvent]:
    """Fetch with a disk cache, so repeated runs in a day hit the source once.

    On a fetch failure a stale cache is reused if one exists -- a calendar from
    this morning is far better than no calendar at all. `CalendarUnavailable`
    propagates only when there is also nothing cached to fall back on.
    """
    path = cache_path or _CACHE_PATH
    now = now or datetime.now(timezone.utc)

    cached: list[EconomicEvent] | None = None
    cached_at: datetime | None = None
    if path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(blob["fetched_at"])
            cached = [EconomicEvent.from_dict(e) for e in blob["events"]]
        except Exception as exc:  # noqa: BLE001 - a bad cache is not fatal
            logger.warning("ignoring unreadable calendar cache %s: %s", path, exc)

    if cached is not None and cached_at is not None:
        if now - cached_at < timedelta(minutes=cache_ttl_minutes):
            return cached

    try:
        events = fetch_economic_calendar(**fetch_kwargs)
    except CalendarUnavailable:
        if cached is not None:
            logger.warning("economic calendar fetch failed; using cache from %s", cached_at)
            return cached
        raise

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "fetched_at": now.isoformat(),
            "events": [e.to_dict() for e in events],
        }, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("could not write calendar cache %s: %s", path, exc)

    return events
