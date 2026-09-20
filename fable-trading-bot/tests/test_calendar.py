"""Tests for the economic calendar feed.

The parser tests run against `fixtures/economic_calendar_sample.html`, which is
a verbatim capture of a real Investing.com response (day-header rows included),
not hand-written markup. That matters: the whole risk of this module is their
markup drifting, and a fixture we invented would only prove the parser agrees
with itself.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fable_bot.data.calendar import (
    CalendarUnavailable,
    EconomicEvent,
    fetch_economic_calendar,
    load_economic_calendar,
    parse_calendar_html,
)

FIXTURE = Path(__file__).parent / "fixtures" / "economic_calendar_sample.html"


@pytest.fixture
def sample_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


class _FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("403 Forbidden")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers})
        if self._exc:
            raise self._exc
        return self._response


# ── Parsing ──────────────────────────────────────────────────────────────

def test_parses_only_event_rows(sample_html):
    events = parse_calendar_html(sample_html)
    # fixture has 5 day-header rows and 4 event rows
    assert len(events) == 4


def test_parses_all_fields_of_first_event(sample_html):
    event = parse_calendar_html(sample_html)[0]
    assert event.event_id == "553426"
    assert event.name == "S&P Global Manufacturing PMI (Jul)"
    assert event.currency == "USD"
    assert event.country == "United States"
    assert event.importance == 3
    assert event.forecast == "53.8"
    assert event.previous == "53.8"
    assert event.actual == ""  # not yet released -- source sends &nbsp;


def test_times_are_timezone_aware_utc(sample_html):
    events = parse_calendar_html(sample_html)
    assert all(e.when.tzinfo is timezone.utc for e in events)
    # 09:45 US Eastern during EDT == 13:45 UTC; this is the assertion that
    # catches a silent timezone regression in the request parameters.
    assert events[0].when == datetime(2026, 8, 3, 13, 45, tzinfo=timezone.utc)


def test_events_are_sorted_by_release_time(sample_html):
    events = parse_calendar_html(sample_html)
    assert [e.when for e in events] == sorted(e.when for e in events)


def test_adjacent_cells_do_not_bleed_into_each_other(sample_html):
    """The actual/forecast/previous cells carry classes containing 'event'."""
    events = parse_calendar_html(sample_html)
    jolts = [e for e in events if "JOLTS" in e.name][0]
    assert jolts.name == "JOLTS Job Openings (Jun)"
    assert jolts.forecast == "7.420M"
    assert jolts.previous == "7.594M"


def test_parsing_empty_html_yields_no_events():
    assert parse_calendar_html("") == []


def test_minutes_until_is_signed():
    when = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    event = EconomicEvent("1", when, "USD", "United States", 3, "CPI")
    assert event.minutes_until(when - timedelta(minutes=30)) == pytest.approx(30)
    assert event.minutes_until(when + timedelta(minutes=15)) == pytest.approx(-15)


def test_event_round_trips_through_dict():
    original = EconomicEvent(
        "1", datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        "USD", "United States", 3, "CPI", forecast="3.1%",
    )
    assert EconomicEvent.from_dict(original.to_dict()) == original


# ── Fetching ─────────────────────────────────────────────────────────────

def test_fetch_requests_utc_and_importance_range(sample_html):
    session = _FakeSession(_FakeResponse({"data": sample_html}))
    fetch_economic_calendar(session=session, min_importance=2)
    sent = dict(session.calls[0]["data"])
    assert sent["timeZone"] == "55"  # Investing.com's UTC id
    importances = [v for k, v in session.calls[0]["data"] if k == "importance[]"]
    assert importances == ["2", "3"]


def test_fetch_raises_when_network_fails():
    session = _FakeSession(exc=ConnectionError("dns failure"))
    with pytest.raises(CalendarUnavailable, match="could not fetch"):
        fetch_economic_calendar(session=session)


def test_fetch_raises_on_http_error():
    session = _FakeSession(_FakeResponse({}, status_ok=False))
    with pytest.raises(CalendarUnavailable):
        fetch_economic_calendar(session=session)


def test_fetch_raises_rather_than_returning_empty_on_markup_change():
    """A 200 that parses to nothing means their HTML changed, not a quiet week."""
    session = _FakeSession(_FakeResponse({"data": "<table><tr><td>redesigned</td></tr></table>"}))
    with pytest.raises(CalendarUnavailable, match="markup likely changed"):
        fetch_economic_calendar(session=session)


# ── Caching ──────────────────────────────────────────────────────────────

def _write_cache(path: Path, events: list[EconomicEvent], fetched_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fetched_at": fetched_at.isoformat(),
        "events": [e.to_dict() for e in events],
    }), encoding="utf-8")


def test_fresh_cache_is_used_without_fetching(tmp_path, monkeypatch):
    now = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    cached = [EconomicEvent("1", now, "USD", "United States", 3, "Cached CPI")]
    cache = tmp_path / "cal.json"
    _write_cache(cache, cached, now - timedelta(minutes=5))

    def _boom(**kwargs):
        raise AssertionError("should not fetch when cache is fresh")

    monkeypatch.setattr("fable_bot.data.calendar.fetch_economic_calendar", _boom)
    events = load_economic_calendar(cache_path=cache, now=now, cache_ttl_minutes=180)
    assert [e.name for e in events] == ["Cached CPI"]


def test_stale_cache_triggers_refetch_and_rewrite(tmp_path, monkeypatch):
    now = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    _write_cache(cache := tmp_path / "cal.json", [
        EconomicEvent("1", now, "USD", "United States", 3, "Old"),
    ], now - timedelta(hours=9))
    fresh = [EconomicEvent("2", now, "USD", "United States", 3, "Fresh")]
    monkeypatch.setattr("fable_bot.data.calendar.fetch_economic_calendar", lambda **kw: fresh)

    events = load_economic_calendar(cache_path=cache, now=now, cache_ttl_minutes=180)
    assert [e.name for e in events] == ["Fresh"]
    assert json.loads(cache.read_text())["events"][0]["name"] == "Fresh"


def test_stale_cache_is_reused_when_refetch_fails(tmp_path, monkeypatch):
    """A calendar from this morning beats no calendar at all."""
    now = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    _write_cache(cache := tmp_path / "cal.json", [
        EconomicEvent("1", now, "USD", "United States", 3, "Stale but usable"),
    ], now - timedelta(hours=9))

    def _fail(**kwargs):
        raise CalendarUnavailable("blocked")

    monkeypatch.setattr("fable_bot.data.calendar.fetch_economic_calendar", _fail)
    events = load_economic_calendar(cache_path=cache, now=now, cache_ttl_minutes=180)
    assert [e.name for e in events] == ["Stale but usable"]


def test_failure_with_no_cache_propagates(tmp_path, monkeypatch):
    def _fail(**kwargs):
        raise CalendarUnavailable("blocked")

    monkeypatch.setattr("fable_bot.data.calendar.fetch_economic_calendar", _fail)
    with pytest.raises(CalendarUnavailable):
        load_economic_calendar(cache_path=tmp_path / "missing.json")


def test_corrupt_cache_is_ignored_not_fatal(tmp_path, monkeypatch):
    cache = tmp_path / "cal.json"
    cache.write_text("{not json", encoding="utf-8")
    fresh = [EconomicEvent("2", datetime.now(timezone.utc), "USD", "US", 3, "Fresh")]
    monkeypatch.setattr("fable_bot.data.calendar.fetch_economic_calendar", lambda **kw: fresh)
    assert [e.name for e in load_economic_calendar(cache_path=cache)] == ["Fresh"]
