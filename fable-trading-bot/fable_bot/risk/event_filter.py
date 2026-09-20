"""Blackout windows around scheduled high-impact macro releases.

This is the fourth risk rule, sitting alongside the three from the source
design (1% risk per trade, correlation blocking, 10% drawdown kill-switch). The
others are all reactive -- they measure what price has already done. This one is
the only forward-looking check: it reads the economic calendar and declines to
open new positions in the minutes around a release.

The rationale is specifically about position *sizing*, not about predicting
direction. Size is derived from ATR, i.e. from recent realised volatility. A
CPI or FOMC print is a scheduled discontinuity in that volatility, so a position
opened just before one is sized off an ATR that no longer describes the next
bar, and the "1% risk" it was built to respect is not the risk actually taken.
Standing aside for a defined window keeps sizing honest.

Only *new entries* are gated. Exits, stops, and the drawdown kill-switch are
never blocked -- being unable to close a position during a release would invert
the whole point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from ..config import EventFilterConfig
from ..data.calendar import CalendarUnavailable, EconomicEvent, load_economic_calendar

logger = logging.getLogger("fable_bot.risk.event_filter")


@dataclass
class EventRiskFilter:
    """Decides whether a new entry is too close to a scheduled release.

    `available` is False when the calendar could not be loaded. What that means
    is a policy question, not a data question, so it is deferred to
    `config.fail_closed`.
    """

    config: EventFilterConfig
    events: list[EconomicEvent] = field(default_factory=list)
    available: bool = True
    unavailable_reason: str = ""

    def is_blocked(
        self,
        currencies: Sequence[str],
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        """Return (blocked, reason) for opening a new position right now."""
        if not self.config.enabled:
            return False, ""

        now = now or datetime.now(timezone.utc)

        if not self.available:
            if self.config.fail_closed:
                return True, f"blocked: economic calendar unavailable ({self.unavailable_reason})"
            # Fail open, but never quietly -- a silent skip here would look
            # identical to "no events scheduled".
            logger.warning(
                "economic calendar unavailable (%s); trading through it because "
                "EVENT_FILTER_FAIL_CLOSED is false",
                self.unavailable_reason,
            )
            return False, ""

        wanted = {c.upper() for c in currencies}
        for event in self.blocking_events(now):
            if event.currency.upper() not in wanted:
                continue
            minutes = event.minutes_until(now)
            when = f"in {minutes:.0f}min" if minutes >= 0 else f"{-minutes:.0f}min ago"
            return True, (
                f"blocked: {event.name} ({event.currency}, impact {event.importance}) {when}"
            )
        return False, ""

    def blocking_events(self, now: datetime | None = None) -> Iterable[EconomicEvent]:
        """Events whose blackout window currently contains `now`."""
        now = now or datetime.now(timezone.utc)
        before = timedelta(minutes=self.config.blackout_minutes_before)
        after = timedelta(minutes=self.config.blackout_minutes_after)
        for event in self.events:
            if event.importance < self.config.min_importance:
                continue
            if event.when - before <= now <= event.when + after:
                yield event

    def upcoming(
        self,
        within_hours: float = 24.0,
        now: datetime | None = None,
        currencies: Sequence[str] | None = None,
    ) -> list[EconomicEvent]:
        """Releases due in the next `within_hours` -- used for briefings."""
        if not self.available:
            return []
        now = now or datetime.now(timezone.utc)
        horizon = now + timedelta(hours=within_hours)
        wanted = {c.upper() for c in currencies} if currencies else None
        return [
            e for e in self.events
            if e.importance >= self.config.min_importance
            and now <= e.when <= horizon
            and (wanted is None or e.currency.upper() in wanted)
        ]


def build_event_filter(config: EventFilterConfig) -> EventRiskFilter:
    """Load the calendar and wrap it in a filter, degrading rather than raising.

    A disabled filter does no network I/O at all, so the calendar stays an
    opt-out dependency: the backtester and anyone without internet access are
    unaffected.
    """
    if not config.enabled:
        return EventRiskFilter(config=config, events=[], available=True)
    try:
        events = load_economic_calendar(min_importance=config.min_importance)
    except CalendarUnavailable as exc:
        logger.warning("economic calendar unavailable: %s", exc)
        return EventRiskFilter(
            config=config, events=[], available=False, unavailable_reason=str(exc),
        )
    logger.info("economic calendar loaded: %d event(s)", len(events))
    return EventRiskFilter(config=config, events=events, available=True)
