"""NYSE trading calendar: which days have a session, and when it opens.

The unattended runner is triggered by the OS clock, which knows nothing about
market holidays or US daylight saving. This module is the authority instead:
the scheduled task fires early every weekday and asks here whether there is
actually a session today and when it starts.

Everything is computed in America/New_York and converted at the edges. That
matters on this machine specifically -- it runs on UTC+3 (Nairobi), which has
no DST, while New York shifts twice a year. A fixed local-time trigger would
silently drift an hour off the open every March and November; deriving the open
from ET means the drift is absorbed by the wait instead.

No dependency on pandas_market_calendars / exchange_calendars: the NYSE rule set
is small and stable enough to state directly, and an unattended trader that
can't start because a calendar package failed to install is worse than one that
carries forty lines of holiday arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth (1-based) `weekday` of a month. weekday: Monday=0 .. Sunday=6."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last `weekday` of a month."""
    next_month = date(year + (month == 12), (month % 12) + 1, 1)
    last = next_month - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm). Good Friday hangs off this."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(day: date) -> date | None:
    """Weekend-shift a fixed-date holiday to the day the NYSE actually closes.

    Saturday -> the Friday before, Sunday -> the Monday after. Returns None for
    the one exception: a Saturday New Year's Day is not observed at all (the
    exchange does not close the last trading day of the prior year for it).
    """
    if day.weekday() == 5:  # Saturday
        if (day.month, day.day) == (1, 1):
            return None
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday
        return day + timedelta(days=1)
    return day


def holidays(year: int) -> set[date]:
    """Full-day NYSE closures for a calendar year."""
    fixed = [date(year, 1, 1), date(year, 7, 4), date(year, 12, 25)]
    # Juneteenth became an NYSE holiday in 2022; before that the exchange traded.
    if year >= 2022:
        fixed.append(date(year, 6, 19))

    days: set[date] = set()
    for day in fixed:
        shifted = _observed(day)
        if shifted is not None:
            days.add(shifted)

    days.update({
        _nth_weekday(year, 1, 0, 3),           # MLK Day, 3rd Monday of January
        _nth_weekday(year, 2, 0, 3),           # Washington's Birthday, 3rd Monday of February
        _easter(year) - timedelta(days=2),     # Good Friday -- NYSE closes, federal offices do not
        _last_weekday(year, 5, 0),             # Memorial Day, last Monday of May
        _nth_weekday(year, 9, 0, 1),           # Labor Day, 1st Monday of September
        _nth_weekday(year, 11, 3, 4),          # Thanksgiving, 4th Thursday of November
    })
    return days


def early_closes(year: int) -> set[date]:
    """Days the NYSE closes at 13:00 ET instead of 16:00."""
    days: set[date] = set()

    # Day after Thanksgiving.
    days.add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1))

    # July 3rd and December 24th, but only when they are themselves weekdays and
    # not already a full closure (July 3 is the observed holiday when the 4th
    # lands on a Saturday, in which case there is no half day at all).
    full = holidays(year)
    for candidate in (date(year, 7, 3), date(year, 12, 24)):
        if candidate.weekday() < 5 and candidate not in full:
            days.add(candidate)

    return {d for d in days if d.weekday() < 5 and d not in full}


def is_trading_day(day: date) -> bool:
    """True if the NYSE holds a session on this date."""
    return day.weekday() < 5 and day not in holidays(day.year)


def why_closed(day: date) -> str:
    """Human-readable reason a date has no session. Empty string if it does."""
    if day.weekday() == 5:
        return "Saturday"
    if day.weekday() == 6:
        return "Sunday"
    if day in holidays(day.year):
        return "NYSE holiday"
    return ""


@dataclass(frozen=True)
class Session:
    """One NYSE trading session, as timezone-aware ET datetimes."""

    day: date
    open_at: datetime
    close_at: datetime
    is_early_close: bool

    def opens_in(self, now: datetime) -> timedelta:
        """How long until the bell. Negative once the session has opened."""
        return self.open_at - now.astimezone(ET)

    def is_open(self, now: datetime) -> bool:
        return self.open_at <= now.astimezone(ET) < self.close_at


def session_for(day: date) -> Session | None:
    """The session on `day`, or None if the market is closed that day."""
    if not is_trading_day(day):
        return None
    early = day in early_closes(day.year)
    return Session(
        day=day,
        open_at=datetime.combine(day, REGULAR_OPEN, tzinfo=ET),
        close_at=datetime.combine(day, EARLY_CLOSE if early else REGULAR_CLOSE, tzinfo=ET),
        is_early_close=early,
    )


def next_session(after: datetime, *, include_today: bool = True) -> Session:
    """The next session at or after `after` (searching forward day by day)."""
    now_et = after.astimezone(ET)
    day = now_et.date() if include_today else now_et.date() + timedelta(days=1)
    for _ in range(370):  # a year of lookahead is more than any holiday run needs
        session = session_for(day)
        if session is not None and (day != now_et.date() or session.open_at > now_et):
            return session
        day += timedelta(days=1)
    raise RuntimeError("No trading session found within a year -- calendar logic is broken.")


def now_et() -> datetime:
    return datetime.now(ET)
