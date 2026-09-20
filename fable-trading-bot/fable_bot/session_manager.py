"""Unattended intraday session manager.

Runs from entry to the closing bell without supervision: watches the open
position, enforces the daily loss limit, and flattens before the close so
nothing is carried overnight.

What it deliberately does NOT do is add discretionary management -- no moving
stops to breakeven, no scaling out, no re-entries. None of that was validated
in testing, and unvalidated management rules applied to an unvalidated edge is
how a small planned loss becomes an improvised large one. The protective stop
sits at the broker where it survives this process dying, the network dropping,
or the machine sleeping; this manager is a supervisor, not the safety net.

Output is deliberately sparse: one line per state change, plus a heartbeat, so
it can be attached to a notifier without becoming noise.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .broker import build_broker
from .config import BROKER
from .risk.session_guard import SessionRiskConfig, SessionRiskGuard

NY = ZoneInfo("America/New_York")
logger = logging.getLogger("fable_bot.session_manager")


@dataclass
class ManagerConfig:
    poll_seconds: int = 60
    heartbeat_minutes: int = 30
    # Consecutive broker errors tolerated before shouting. The stop is already
    # resting at the broker, so a transient outage is survivable -- what is not
    # survivable is failing to say anything about it.
    max_consecutive_errors: int = 5


def _emit(kind: str, message: str) -> None:
    """One event per line, prefixed so a notifier can filter on severity."""
    stamp = datetime.now(NY).strftime("%H:%M:%S")
    print(f"[{stamp} NY] {kind}: {message}", flush=True)


def run_session(symbol: str, config: ManagerConfig | None = None,
                 guard_config: SessionRiskConfig | None = None) -> int:
    """Manage one session. Returns an exit code: 0 clean, 1 error-exit."""
    config = config or ManagerConfig()
    broker = build_broker(BROKER)

    account = broker.get_account()
    now = datetime.now(NY)
    guard = SessionRiskGuard(config=guard_config or SessionRiskConfig(),
                              starting_equity=account.equity)
    guard.start_session(now.date(), account.equity)
    start_equity = account.equity

    _emit("START", f"managing {symbol} | equity {account.equity:,.2f} | "
                    f"daily loss limit ${guard.daily_loss_limit:,.0f}")

    seen_position = False
    errors = 0
    last_heartbeat = time.time()
    peak_unrealized = 0.0

    while True:
        now = datetime.now(NY)
        try:
            positions = broker.get_open_positions()
            account = broker.get_account()
            errors = 0
        except Exception as exc:  # noqa: BLE001 - keep supervising through blips
            errors += 1
            _emit("WARN", f"broker read failed ({errors}/{config.max_consecutive_errors}): {exc}")
            if errors >= config.max_consecutive_errors:
                _emit("ERROR", "broker unreachable repeatedly — the resting stop still "
                                "protects the position, but this manager is blind. Check TradingView.")
                return 1
            time.sleep(config.poll_seconds)
            continue

        position = positions.get(symbol)
        pnl_today = account.equity - start_equity

        if position and not seen_position:
            seen_position = True
            _emit("ENTRY", f"{symbol} {position['side']} qty={position['qty']:g} "
                            f"@ {position['avg_entry_price']:.2f}")

        # Position closed on its own: the broker stop fired, or price ran to a
        # bracket. Either way the session's job is done.
        if seen_position and not position:
            outcome = "STOPPED OUT" if pnl_today < 0 else "CLOSED"
            _emit(outcome, f"{symbol} flat | session P&L {pnl_today:+,.2f} "
                            f"| equity {account.equity:,.2f}")
            return 0

        if position:
            unrealized = position.get("unrealized_pnl") or 0.0
            peak_unrealized = max(peak_unrealized, unrealized)

            # Daily loss limit is checked against live equity, not just closed
            # trades: a position deep enough underwater has already spent the
            # day's budget whether or not it has been realised.
            if pnl_today <= -guard.daily_loss_limit:
                _emit("RISK", f"daily loss limit breached ({pnl_today:+,.2f}) — flattening now")
                broker.close_position(symbol)
                _emit("FLAT", f"closed by daily loss limit | session P&L {pnl_today:+,.2f}")
                return 0

            if guard.must_flatten(now):
                _emit("CLOSE", "approaching the bell — flattening")
                broker.close_position(symbol)
                final = broker.get_account()
                _emit("FLAT", f"closed at session end | session P&L "
                               f"{final.equity - start_equity:+,.2f} | equity {final.equity:,.2f}")
                return 0

        # Never entered (or already closed) and the bell is near: nothing left
        # to supervise.
        if not position and guard.must_flatten(now):
            _emit("END", f"session over, flat | P&L {pnl_today:+,.2f}")
            return 0

        if time.time() - last_heartbeat >= config.heartbeat_minutes * 60:
            last_heartbeat = time.time()
            if position:
                _emit("HOLD", f"{symbol} {position['side']} qty={position['qty']:g} "
                               f"@ {position['avg_entry_price']:.2f} | "
                               f"session P&L {pnl_today:+,.2f}")
            else:
                _emit("WAIT", f"flat | session P&L {pnl_today:+,.2f}")

        time.sleep(config.poll_seconds)
