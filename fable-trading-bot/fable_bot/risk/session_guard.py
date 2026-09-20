"""Intraday session risk guard for futures.

The daily-bar risk rules (1% per trade, correlation block, 10% drawdown halt)
were built for a strategy that takes a handful of positions a month. Intraday
futures break two of their assumptions: trades arrive many times a session, and
a single contract is large enough that a bad hour can do what previously took a
bad quarter. A 10%-equity kill-switch is far too slow to be the only backstop
when you can take six trades before lunch.

So this adds the guards that matter on an intraday clock, checked before every
entry:

  * per-trade risk, sized in cash off the actual stop distance and the contract
    multiplier -- not off notional, which for futures is meaningless
  * a daily loss limit, after which the session is done regardless of how good
    the next setup looks
  * a cap on trades per session, which bounds both commission drag and the
    revenge-trading failure mode
  * a cap on concurrent positions, since ES and NQ are ~0.9 correlated and two
    index longs is one bet at double size
  * time guards: no entries in the first minutes while the opening auction
    settles, none near the close, and nothing held overnight
  * the economic-calendar blackout, so entries stand aside around scheduled
    high-impact releases

Everything is advisory-by-return-value rather than raising: the caller gets
(allowed, reason) and decides. That keeps the guard testable without a broker
and makes the reason string available for the trade log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone

logger = logging.getLogger("fable_bot.risk.session_guard")

# Regular session in exchange time.
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


@dataclass(frozen=True)
class SessionRiskConfig:
    max_risk_per_trade_pct: float = 1.0
    # Daily stop. Three losing trades at full per-trade risk ends the session:
    # past that point the evidence is that the day is not cooperating, and the
    # marginal trade is far more likely to be an emotional one.
    max_daily_loss_pct: float = 3.0
    max_trades_per_session: int = 4
    max_concurrent_positions: int = 1
    # Skip the opening minutes: the first prints are an auction, spreads are
    # widest, and a stop placed there is measured off noise.
    no_entry_first_minutes: int = 5
    # Stop opening new risk near the bell; positions still get flattened.
    no_entry_last_minutes: int = 20
    flatten_before_close_minutes: int = 5


@dataclass
class SessionRiskGuard:
    """Tracks one session's realised risk and vetoes entries that breach it."""

    config: SessionRiskConfig
    starting_equity: float
    session: date | None = None
    realized_pnl: float = 0.0
    trades_taken: int = 0
    open_positions: int = 0
    halted_reason: str = ""
    _events: list = field(default_factory=list)

    # ── session lifecycle ────────────────────────────────────────────────

    def start_session(self, session: date, equity: float) -> None:
        """Reset counters for a new session. Idempotent within the same day."""
        if self.session == session:
            return
        self.session = session
        self.starting_equity = equity
        self.realized_pnl = 0.0
        self.trades_taken = 0
        self.open_positions = 0
        self.halted_reason = ""
        logger.info("session %s started, equity %.2f", session, equity)

    def record_fill(self, pnl: float | None = None, opened: bool = False, closed: bool = False) -> None:
        """Update state after an execution. `pnl` applies to closes."""
        if opened:
            self.trades_taken += 1
            self.open_positions += 1
        if closed:
            self.open_positions = max(0, self.open_positions - 1)
            if pnl is not None:
                self.realized_pnl += pnl

    # ── budgets ─────────────────────────────────────────────────────────

    @property
    def risk_budget(self) -> float:
        return self.starting_equity * self.config.max_risk_per_trade_pct / 100.0

    @property
    def daily_loss_limit(self) -> float:
        return self.starting_equity * self.config.max_daily_loss_pct / 100.0

    @property
    def daily_loss_used_pct(self) -> float:
        if self.realized_pnl >= 0:
            return 0.0
        return abs(self.realized_pnl) / self.daily_loss_limit * 100.0

    def contracts_for(self, entry_price: float, stop_price: float, point_value: float,
                       qty_step: float = 1.0) -> tuple[float, str]:
        """Contracts that risk at most the per-trade budget, rounded DOWN.

        Down, never nearest: rounding a 1.6-contract budget up to 2 would risk
        125% of the intended amount. A 0 result means the stop is too wide for
        this account and the trade should be skipped, not shrunk by widening
        the risk.
        """
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0 or point_value <= 0:
            return 0.0, "invalid stop distance or point value"
        raw = self.risk_budget / (stop_distance * point_value)
        qty = (raw // qty_step) * qty_step
        if qty <= 0:
            risk_one = stop_distance * point_value
            return 0.0, (f"one contract risks ${risk_one:,.0f}, over the "
                          f"${self.risk_budget:,.0f} per-trade budget")
        return qty, ""

    # ── the gate ────────────────────────────────────────────────────────

    def can_enter(self, now: datetime, event_blocked_reason: str = "") -> tuple[bool, str]:
        """Whether a new position may be opened right now."""
        if self.halted_reason:
            return False, self.halted_reason

        if self.realized_pnl <= -self.daily_loss_limit:
            self.halted_reason = (
                f"daily loss limit hit: {self.realized_pnl:+,.0f} vs limit "
                f"{-self.daily_loss_limit:,.0f} ({self.config.max_daily_loss_pct}% of equity)"
            )
            logger.warning("session halted: %s", self.halted_reason)
            return False, self.halted_reason

        if self.trades_taken >= self.config.max_trades_per_session:
            return False, (f"max trades per session reached "
                            f"({self.trades_taken}/{self.config.max_trades_per_session})")

        if self.open_positions >= self.config.max_concurrent_positions:
            return False, (f"max concurrent positions reached "
                            f"({self.open_positions}/{self.config.max_concurrent_positions})")

        in_session, why = self.within_entry_window(now)
        if not in_session:
            return False, why

        if event_blocked_reason:
            return False, event_blocked_reason

        return True, ""

    def within_entry_window(self, now: datetime) -> tuple[bool, str]:
        """Time-of-day gate, evaluated in exchange-local time."""
        t = now.time()
        if t < RTH_OPEN or t >= RTH_CLOSE:
            return False, "outside regular trading hours"

        minutes_in = (t.hour - RTH_OPEN.hour) * 60 + (t.minute - RTH_OPEN.minute)
        if minutes_in < self.config.no_entry_first_minutes:
            return False, (f"within the opening {self.config.no_entry_first_minutes} min "
                            "(auction still settling)")

        minutes_left = (RTH_CLOSE.hour - t.hour) * 60 - t.minute
        if minutes_left <= self.config.no_entry_last_minutes:
            return False, f"within {self.config.no_entry_last_minutes} min of the close"

        return True, ""

    def must_flatten(self, now: datetime) -> bool:
        """True once open positions should be closed for the day."""
        t = now.time()
        if t >= RTH_CLOSE:
            return True
        minutes_left = (RTH_CLOSE.hour - t.hour) * 60 - t.minute
        return minutes_left <= self.config.flatten_before_close_minutes

    def summary(self) -> dict:
        return {
            "session": str(self.session),
            "starting_equity": round(self.starting_equity, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "trades_taken": self.trades_taken,
            "trades_remaining": max(0, self.config.max_trades_per_session - self.trades_taken),
            "open_positions": self.open_positions,
            "risk_budget_per_trade": round(self.risk_budget, 2),
            "daily_loss_limit": round(self.daily_loss_limit, 2),
            "daily_loss_used_pct": round(self.daily_loss_used_pct, 1),
            "halted": bool(self.halted_reason),
            "halted_reason": self.halted_reason,
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
