"""Tests for the macro-release blackout filter."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fable_bot.config import EventFilterConfig
from fable_bot.data.calendar import EconomicEvent
from fable_bot.risk.event_filter import EventRiskFilter

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def _config(**overrides) -> EventFilterConfig:
    return EventFilterConfig(**{
        "enabled": True,
        "min_importance": 3,
        "blackout_minutes_before": 60,
        "blackout_minutes_after": 30,
        "fail_closed": False,
        **overrides,
    })


def _event(minutes_from_now: float, *, currency="USD", importance=3, name="FOMC Rate Decision"):
    return EconomicEvent(
        event_id="1",
        when=NOW + timedelta(minutes=minutes_from_now),
        currency=currency,
        country="United States",
        importance=importance,
        name=name,
    )


def _filter(events, **config_overrides) -> EventRiskFilter:
    return EventRiskFilter(config=_config(**config_overrides), events=events)


# ── Blackout window ──────────────────────────────────────────────────────

def test_blocks_shortly_before_release():
    blocked, reason = _filter([_event(20)]).is_blocked(["USD"], now=NOW)
    assert blocked
    assert "FOMC Rate Decision" in reason
    assert "in 20min" in reason


def test_blocks_shortly_after_release():
    blocked, reason = _filter([_event(-15)]).is_blocked(["USD"], now=NOW)
    assert blocked
    assert "15min ago" in reason


def test_allows_well_before_the_window_opens():
    blocked, _ = _filter([_event(90)]).is_blocked(["USD"], now=NOW)
    assert not blocked


def test_allows_once_the_window_has_passed():
    blocked, _ = _filter([_event(-45)]).is_blocked(["USD"], now=NOW)
    assert not blocked


def test_window_boundaries_are_inclusive():
    assert _filter([_event(60)]).is_blocked(["USD"], now=NOW)[0]
    assert _filter([_event(-30)]).is_blocked(["USD"], now=NOW)[0]


def test_window_sizes_are_configurable():
    blocked, _ = _filter([_event(90)], blackout_minutes_before=120).is_blocked(["USD"], now=NOW)
    assert blocked


# ── Relevance filtering ──────────────────────────────────────────────────

def test_ignores_events_in_unrelated_currencies():
    blocked, _ = _filter([_event(20, currency="JPY")]).is_blocked(["USD"], now=NOW)
    assert not blocked


def test_currency_matching_is_case_insensitive():
    blocked, _ = _filter([_event(20, currency="usd")]).is_blocked(["USD"], now=NOW)
    assert blocked


def test_ignores_events_below_the_importance_threshold():
    blocked, _ = _filter([_event(20, importance=2)]).is_blocked(["USD"], now=NOW)
    assert not blocked


def test_lowering_the_threshold_catches_medium_impact():
    blocked, _ = _filter([_event(20, importance=2)], min_importance=2).is_blocked(["USD"], now=NOW)
    assert blocked


def test_reports_the_first_blocking_event_when_several_overlap():
    events = [_event(20, name="CPI"), _event(25, name="Retail Sales")]
    _, reason = _filter(events).is_blocked(["USD"], now=NOW)
    assert "CPI" in reason


# ── Disabled / unavailable policy ────────────────────────────────────────

def test_disabled_filter_never_blocks():
    blocked, _ = _filter([_event(0)], enabled=False).is_blocked(["USD"], now=NOW)
    assert not blocked


def test_unavailable_calendar_fails_open_by_default():
    f = EventRiskFilter(config=_config(), events=[], available=False, unavailable_reason="403")
    blocked, _ = f.is_blocked(["USD"], now=NOW)
    assert not blocked


def test_unavailable_calendar_blocks_when_fail_closed():
    f = EventRiskFilter(
        config=_config(fail_closed=True), events=[], available=False, unavailable_reason="403",
    )
    blocked, reason = f.is_blocked(["USD"], now=NOW)
    assert blocked
    assert "unavailable" in reason


def test_fail_open_warns_so_it_is_distinguishable_from_no_events(caplog):
    f = EventRiskFilter(config=_config(), events=[], available=False, unavailable_reason="403")
    with caplog.at_level("WARNING"):
        f.is_blocked(["USD"], now=NOW)
    assert any("unavailable" in r.getMessage() for r in caplog.records)


# ── Briefing helper ──────────────────────────────────────────────────────

def test_upcoming_respects_horizon_and_importance():
    events = [
        _event(60, name="Soon"),
        _event(60 * 30, name="Beyond horizon"),
        _event(90, importance=1, name="Low impact"),
        _event(-60, name="Already released"),
    ]
    names = [e.name for e in _filter(events).upcoming(within_hours=24, now=NOW)]
    assert names == ["Soon"]


def test_upcoming_is_empty_when_calendar_unavailable():
    f = EventRiskFilter(config=_config(), events=[_event(30)], available=False)
    assert f.upcoming(now=NOW) == []
