"""Tests for the NYSE session calendar.

The holiday dates below are the exchange's published closures, not derived from
the same helpers under test -- a calendar that agrees with itself proves
nothing. The daylight-saving cases are the point of the module: this bot runs on
a UTC+3 machine with no DST, so the whole scheduling design rests on the open
being computed in ET rather than assumed in local time.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from fable_bot.market_calendar import (
    ET,
    early_closes,
    holidays,
    is_trading_day,
    next_session,
    session_for,
    why_closed,
)

NAIROBI = ZoneInfo("Africa/Nairobi")

# Published NYSE closures for 2026.
HOLIDAYS_2026 = {
    date(2026, 1, 1),    # New Year's Day (Thu)
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Washington's Birthday
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth (Fri)
    date(2026, 7, 3),    # Independence Day observed (the 4th is a Saturday)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas (Fri)
}


def test_2026_holidays_match_the_published_calendar():
    assert holidays(2026) == HOLIDAYS_2026


def test_good_friday_is_a_closure_though_it_is_not_a_federal_holiday():
    # The reason this calendar can't just reuse pandas' USFederalHolidayCalendar.
    assert date(2026, 4, 3) in holidays(2026)
    assert date(2025, 4, 18) in holidays(2025)


def test_columbus_day_and_veterans_day_are_federal_but_the_exchange_trades():
    assert is_trading_day(date(2026, 10, 12))  # Columbus Day
    assert is_trading_day(date(2026, 11, 11))  # Veterans Day


@pytest.mark.parametrize("year,expected", [
    (2027, date(2027, 7, 5)),   # Jul 4 is a Sunday -> observed Monday
    (2026, date(2026, 7, 3)),   # Jul 4 is a Saturday -> observed Friday
    (2025, date(2025, 7, 4)),   # already a weekday
])
def test_independence_day_weekend_observance(year, expected):
    assert expected in holidays(year)


def test_saturday_new_year_is_not_observed_on_the_preceding_friday():
    # Jan 1 2022 fell on a Saturday; the NYSE traded a full session on Dec 31 2021.
    assert is_trading_day(date(2021, 12, 31))
    assert date(2021, 12, 31) not in holidays(2021)


def test_sunday_holiday_rolls_forward_to_monday():
    # Dec 25 2022 was a Sunday -> closed Monday the 26th.
    assert date(2022, 12, 26) in holidays(2022)


def test_juneteenth_only_from_2022():
    assert date(2021, 6, 18) not in holidays(2021)  # observed date had it been a holiday
    assert date(2022, 6, 20) in holidays(2022)      # the 19th was a Sunday -> Monday


def test_weekends_have_no_session():
    assert session_for(date(2026, 8, 8)) is None   # Saturday
    assert session_for(date(2026, 8, 9)) is None   # Sunday
    assert why_closed(date(2026, 8, 8)) == "Saturday"
    assert why_closed(date(2026, 11, 26)) == "NYSE holiday"
    assert why_closed(date(2026, 8, 6)) == ""


def test_session_open_is_0930_et_regardless_of_the_season():
    for day in (date(2026, 8, 6), date(2026, 12, 1)):
        session = session_for(day)
        assert (session.open_at.hour, session.open_at.minute) == (9, 30)


def test_the_open_moves_against_local_time_across_the_us_dst_change():
    """The reason the scheduled task fires early and then waits.

    Nairobi never changes clocks, so the same 09:30 ET bell lands an hour later
    in local time once the US leaves daylight saving.
    """
    summer = session_for(date(2026, 8, 6)).open_at.astimezone(NAIROBI)
    winter = session_for(date(2026, 12, 1)).open_at.astimezone(NAIROBI)
    assert (summer.hour, summer.minute) == (16, 30)
    assert (winter.hour, winter.minute) == (17, 30)


def test_early_closes():
    closes = early_closes(2026)
    assert date(2026, 11, 27) in closes           # day after Thanksgiving
    assert date(2026, 12, 24) in closes           # Christmas Eve (Thursday)
    assert session_for(date(2026, 11, 27)).close_at.hour == 13
    assert session_for(date(2026, 11, 27)).is_early_close
    assert not session_for(date(2026, 11, 25)).is_early_close


def test_early_close_list_never_contains_a_full_closure():
    for year in range(2022, 2031):
        assert not (early_closes(year) & holidays(year))


def test_next_session_skips_the_weekend():
    friday_evening = datetime(2026, 8, 7, 18, 0, tzinfo=ET)
    assert next_session(friday_evening).day == date(2026, 8, 10)


def test_next_session_skips_a_holiday():
    # Wednesday evening before Thanksgiving -> Friday, not Thursday.
    before = datetime(2026, 11, 25, 18, 0, tzinfo=ET)
    assert next_session(before).day == date(2026, 11, 27)


def test_next_session_returns_today_before_the_bell_and_tomorrow_after():
    session_day = date(2026, 8, 6)
    pre_open = datetime(2026, 8, 6, 8, 0, tzinfo=ET)
    post_open = datetime(2026, 8, 6, 10, 0, tzinfo=ET)
    assert next_session(pre_open).day == session_day
    assert next_session(post_open).day == date(2026, 8, 7)


def test_session_open_and_close_predicates():
    session = session_for(date(2026, 8, 6))
    assert not session.is_open(datetime(2026, 8, 6, 9, 29, tzinfo=ET))
    assert session.is_open(datetime(2026, 8, 6, 9, 30, tzinfo=ET))
    assert session.is_open(datetime(2026, 8, 6, 15, 59, tzinfo=ET))
    assert not session.is_open(datetime(2026, 8, 6, 16, 0, tzinfo=ET))
    assert session.opens_in(datetime(2026, 8, 6, 9, 0, tzinfo=ET)) == timedelta(minutes=30)


def test_opens_in_accepts_a_foreign_timezone():
    """The runner compares against local time; the conversion must happen inside."""
    session = session_for(date(2026, 8, 6))
    local_nine_am_et = datetime(2026, 8, 6, 16, 0, tzinfo=NAIROBI)  # 09:00 ET
    assert session.opens_in(local_nine_am_et) == timedelta(minutes=30)
