"""Orchestrates one live/paper trading cycle: fetch data -> signals -> risk
checks -> broker orders -> briefing. Designed to be invoked once per trading
day (this is a daily-bar strategy suite, matching the backtest timeframe).

Orders route to whichever venue BROKER_PROVIDER names -- TradingView's Paper
Trading simulator by default (no API keys, just the desktop app running with a
broker connected), or Alpaca with ALPACA_API_KEY/ALPACA_SECRET_KEY. Symbols and
quantity rounding differ between the two, so both go through the adapter
(venue_symbol / quantize) rather than being hardcoded here.

Briefings additionally require TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID. Nothing
places an order or sends a message if those aren't configured -- see config.py.
"""
from __future__ import annotations

import logging
from datetime import date
from uuid import uuid4

import pandas as pd

from .broker import build_broker
from .broker.tradingview_adapter import TradingViewError
from .config import BROKER, EVENT_FILTER, INSTRUMENTS, MEMORY_DIR, RISK, TELEGRAM
from .data.feed import drop_incomplete_bar, fetch_universe_history  # noqa: F401 - re-exported for callers/tests
from .journal import Journal
from .lessons import LessonBook, LessonGate
from .market_calendar import now_et
from .risk.event_filter import build_event_filter
from .risk.intel_gate import evaluate_entry
from .risk.manager import (DrawdownMonitor, RiskManager, correlation_matrix,
                          portfolio_gate, sizing_sanity)
from .strategies import build_strategy
from .telegram_bot import TelegramBriefing

logger = logging.getLogger("fable_bot.live_runner")

DRAWDOWN_STATE_FILE = "drawdown_state.json"

_drawdown_monitor: DrawdownMonitor | None = None


def drawdown_state_path():
    """Resolved per call, not pinned at import.

    A module-level constant baked in the real MEMORY_DIR at import time, so a
    test that redirected memory to a tmp dir still loaded the live peak -- and
    tripped the kill-switch against a stubbed equity. Reading the module
    attribute here keeps one source of truth and lets it be redirected.
    """
    return MEMORY_DIR / DRAWDOWN_STATE_FILE


def _get_drawdown_monitor(starting_equity: float) -> DrawdownMonitor:
    """The kill-switch, restored from disk.

    This used to hold peak_equity in the process only. The unattended runner
    is a ONE-SHOT process -- a fresh interpreter per session -- so the peak
    reset to that morning's equity on every run and a multi-day drawdown
    could never be observed. Between 2026-08-25 and 2026-09-11 the account
    slid 117,148 -> 113,662 without the switch seeing a single percent of it.

    Loading from MEMORY_DIR makes the peak outlive the process, which is the
    only way a daily-bar kill-switch can mean anything.
    """
    global _drawdown_monitor
    if _drawdown_monitor is None:
        _drawdown_monitor = DrawdownMonitor.load(
            drawdown_state_path(),
            max_drawdown_pct=RISK.max_drawdown_pct,
            current_equity=starting_equity,
        )
    return _drawdown_monitor


def run_signal_check(
    *,
    require_completed_bar: bool = False,
    dry_run: bool = False,
    journal: Journal | None = None,
) -> list[str]:
    """Pull the latest bar per instrument, generate signals, size and submit orders.

    require_completed_bar: drop today's in-progress bar before evaluating. Set
        this for any run that happens while the market is open (see
        drop_incomplete_bar); leave it off for the post-close run, where the
        final bar is the one being signalled on.
    dry_run: do everything except place or close orders. The actions list still
        reports what would have been sent.
    """
    def record(kind: str, **fields) -> None:
        if journal is not None:
            journal.event(kind, **fields)

    broker = build_broker(BROKER)
    account = broker.get_account()
    open_positions = broker.get_open_positions()
    record("account", equity=account.equity, cash=account.cash, is_paper=account.is_paper,
           open_positions={s: p["side"] for s, p in open_positions.items()}, dry_run=dry_run)
    if account.equity < RISK.small_account_equity:
        # Said every run, on purpose. At this size the 1% rule cannot be met
        # on a standard minimum lot (sizing_check.py has the arithmetic), so
        # whatever MAX_RISK_PER_TRADE_PCT is set to is the real risk.
        record("small_account", equity=account.equity, threshold=RISK.small_account_equity,
               risk_pct_per_trade=RISK.max_risk_per_trade_pct)
        logger.warning("SMALL ACCOUNT: equity %.2f < %.0f; per-trade risk is %.1f%% by config, "
                       "and minimum lots may exceed it -- see `fable_bot size`",
                       account.equity, RISK.small_account_equity, RISK.max_risk_per_trade_pct)

    # Positioning intel, fetched at most once per coverage key per run. It is
    # network-bound (option chains, CFTC), so it is only gathered for an
    # instrument that actually reaches the entry path.
    intel_cache: dict[str, object] = {}

    def intel_for(inst):
        if not RISK.intel_gate_enabled:
            return None
        from .intel.report import coverage_for_ticker, gather
        cov = coverage_for_ticker(inst.symbol)
        if cov is None:
            return None
        if cov.display not in intel_cache:
            try:
                intel_cache[cov.display] = gather(cov.display)
            except Exception as exc:  # noqa: BLE001 - the gate fails open
                logger.warning("intel gather failed for %s: %s", cov.display, exc)
                intel_cache[cov.display] = None
        return intel_cache[cov.display]

    # Lesson memory, fail-soft: a corrupt lessons.jsonl degrades to an empty
    # gate -- never worse than before lessons existed -- with the load errors
    # journalled as evidence.
    book = LessonBook()
    gate = LessonGate(book)
    record("lessons_loaded", active=gate.active_count, sizing_bounds=gate.sizing_bounds(),
           verification=gate.verification_config())
    if book.load_errors:
        record("lessons_load_error", errors=book.load_errors)

    # Intelligence-only series (tradable=False) are fetched nowhere and can
    # never reach submit_order: ordering the unorderable is what the venue
    # rejections of 2026-08 were.
    price_history = fetch_universe_history([i.symbol for i in INSTRUMENTS if i.tradable],
                                           period="1y")
    if require_completed_bar:
        trimmed = {}
        for symbol, df in price_history.items():
            kept = drop_incomplete_bar(df)
            if len(kept) < len(df):
                record("bar_trimmed", symbol=symbol, dropped=str(df.index[-1]), now_evaluating=str(kept.index[-1]))
            trimmed[symbol] = kept
        price_history = trimmed

    corr = correlation_matrix(price_history)
    risk_manager = RiskManager(config=RISK, correlations=corr)
    event_filter = build_event_filter(EVENT_FILTER)
    actions: list[str] = []
    monitor = _get_drawdown_monitor(account.equity)
    record("drawdown_state", peak_equity=monitor.peak_equity, equity=account.equity,
           drawdown_pct=round(monitor.drawdown_pct(account.equity), 3),
           limit_pct=RISK.max_drawdown_pct, halted=monitor.halted)
    if monitor.halted:
        # Sticky across restarts by design: only manual_reset() clears it.
        actions.append("RISK HALT ACTIVE: drawdown kill-switch previously tripped. "
                       "No new entries until it is manually reset.")
        record("skip_all", gate="drawdown_halt",
               reason="kill-switch previously tripped")
        return actions
    if not event_filter.available:
        record("calendar_unavailable", reason=event_filter.unavailable_reason,
               fail_closed=EVENT_FILTER.fail_closed)

    if monitor.update(account.equity):
        if dry_run:
            actions.append("DRAWDOWN KILL-SWITCH would fire (dry run: nothing closed).")
            record("kill_switch", equity=account.equity, dry_run=True)
            return actions
        results = broker.close_all()
        record("kill_switch", equity=account.equity, closed=[r.symbol for r in results],
               statuses=[r.status for r in results])
        actions.append(f"DRAWDOWN KILL-SWITCH: closed all positions ({len(results)} order(s)).")
        return actions

    open_sides = {sym: pos["side"] for sym, pos in open_positions.items()}

    # Book-wide risk accounting for this run. open_count starts at whatever is
    # already on and grows as entries go out, so the caps bound the book rather
    # than each order in isolation.
    open_count = len(open_positions)
    risk_committed = 0.0

    for inst in INSTRUMENTS:
        if not inst.tradable:
            record("skip", symbol=inst.symbol, gate="tradability",
                   reason="intelligence-only series")
            continue
        try:
            df = price_history[inst.symbol]
            if df.empty:
                record("skip", symbol=inst.symbol, reason="no price history after trimming")
                continue
            strategy = build_strategy(inst.strategy)
            signals = strategy.generate_signals(inst.symbol, df)
            if not signals:
                continue
            latest = signals[-1]
            if latest.date != df.index[-1]:
                # The flip bar was missed (machine down, instrument added
                # mid-trend). Journalling the skip makes the hole visible;
                # state-based strategies may still reconcile via catch-up.
                record("skip", symbol=inst.symbol, gate="stale",
                       reason=f"signal {latest.side} is from {latest.date.date()}, "
                              f"not the bar being evaluated")
                catchup = strategy.catch_up_signal(inst.symbol, df)
                if catchup is None:
                    continue
                latest = catchup
                record("catchup", symbol=inst.symbol, side=latest.side,
                       price=latest.price, stop_price=latest.stop_price,
                       reason=latest.reason)

            venue_symbol = broker.venue_symbol(inst)
            record("signal", symbol=inst.symbol, venue_symbol=venue_symbol, side=latest.side,
                   price=latest.price, stop_price=latest.stop_price, bar=str(latest.date),
                   reason=latest.reason)

            already_open = venue_symbol in open_positions
            if latest.side == "flat":
                if already_open:
                    if not dry_run:
                        broker.close_position(venue_symbol)
                    record("close", symbol=venue_symbol, reason=latest.reason, dry_run=dry_run)
                    actions.append(f"{'WOULD CLOSE' if dry_run else 'CLOSE'} {venue_symbol}: {latest.reason}")
                continue

            if already_open:
                adds_lesson = gate.max_adds_authority()
                if adds_lesson is not None:
                    record("skip", symbol=venue_symbol, reason="position already open",
                           gate="max_adds", lesson_id=adds_lesson)
                else:
                    record("skip", symbol=venue_symbol, reason="position already open")
                continue

            # Entries only, for the same reason the lesson gate below is:
            # the flat/close branch above has already run. Gating this any
            # earlier would strand every open position in a quarantined
            # instrument -- USDCHF, UKX and EURGBP were all open when this
            # quarantine was introduced, and an exit-blocking gate would have
            # meant the bot could never close them again.
            if not inst.validated and not RISK.trade_unvalidated:
                record("skip", symbol=venue_symbol, gate="unvalidated",
                       reason="no documented backtest support; TRADE_UNVALIDATED is off")
                continue

            # Lesson gate: entries only -- the flat/close branch above runs
            # first, so a lesson can never block getting out of a position.
            blocked, reason, lesson_id = gate.blocked_entry(venue_symbol)
            if blocked:
                record("skip", symbol=venue_symbol, reason=reason, gate="lesson",
                       lesson_id=lesson_id)
                actions.append(f"SKIP {venue_symbol}: {reason}")
                continue

            # Positioning gate: the entry must not fight the forced flow. Sits
            # with the other entry-only gates; the close branch above has run.
            if RISK.intel_gate_enabled:
                decision = evaluate_entry(
                    side=latest.side, strategy=inst.strategy,
                    entry_price=latest.price, stop_price=latest.stop_price,
                    intel=intel_for(inst))
                record("intel_gate", symbol=venue_symbol, side=latest.side,
                       strategy=inst.strategy, allowed=decision.allowed,
                       reason=decision.reason, **decision.evidence)
                if not decision.allowed:
                    record("skip", symbol=venue_symbol, gate="intel", reason=decision.reason)
                    actions.append(f"SKIP {venue_symbol}: positioning -- {decision.reason}")
                    continue

            # Forward-looking check: stand aside near a scheduled macro release, so
            # entries are never sized off an ATR that the next print invalidates.
            # Deliberately placed after the "flat" branch above -- exits and stops
            # must never be gated by this.
            blocked, reason = event_filter.is_blocked(inst.event_currencies)
            if blocked:
                record("skip", symbol=venue_symbol, reason=reason, gate="event_filter")
                actions.append(f"SKIP {venue_symbol}: {reason}")
                continue

            blocked, reason = risk_manager.is_blocked_by_correlation(inst.symbol, latest.side, open_sides)
            if blocked:
                record("skip", symbol=venue_symbol, reason=reason, gate="correlation")
                actions.append(f"SKIP {venue_symbol}: {reason}")
                continue

            allowed, why_not = portfolio_gate(
                open_positions=open_count, risk_committed=risk_committed,
                equity=account.equity, config=RISK)
            if not allowed:
                record("skip", symbol=venue_symbol, gate="portfolio", reason=why_not,
                       open_positions=open_count,
                       risk_committed=round(risk_committed, 2))
                actions.append(f"SKIP {venue_symbol}: {why_not}")
                continue

            qty = broker.quantize(inst, risk_manager.position_size(
                account.equity, latest.price, latest.stop_price, point_value=inst.point_value,
            ))
            if qty <= 0:
                # The 1% risk budget doesn't cover one tradable unit at this stop
                # distance. Skipping is correct: the alternative is rounding up and
                # risking more than the position was sized for.
                record("skip", symbol=venue_symbol, reason="sized below the venue's minimum quantity",
                       gate="sizing")
                actions.append(f"SKIP {venue_symbol}: sized below the venue's minimum quantity")
                continue

            side = "buy" if latest.side == "long" else "sell"
            risk_amount = qty * abs(latest.price - latest.stop_price) * inst.point_value

            ok, why_not, metrics = sizing_sanity(qty, latest.price, latest.stop_price,
                                                 account.equity, inst.point_value,
                                                 gate.sizing_bounds())
            if not ok:
                record("skip", symbol=venue_symbol, gate="sizing_sanity", reason=why_not,
                       **metrics)
                record("sizing_out_of_bounds", symbol=venue_symbol, side=side, qty=qty,
                       price=latest.price, stop_price=latest.stop_price, **metrics)
                actions.append(f"SKIP {venue_symbol}: {why_not}")
                continue

            if dry_run:
                open_count += 1
                risk_committed += risk_amount
                record("order", symbol=venue_symbol, side=side, qty=qty, stop_price=latest.stop_price,
                       risk_amount=risk_amount, status="dry_run", dry_run=True)
                actions.append(f"WOULD ORDER {venue_symbol} {side} qty={qty:g} "
                               f"stop={latest.stop_price:.2f} risk={risk_amount:,.0f} ({latest.reason})")
                continue

            # Counted BEFORE submission, deliberately. A submitted order whose
            # response is lost still consumed risk at the venue (2026-08-20),
            # so optimistic accounting is the safe direction to be wrong in.
            open_count += 1
            risk_committed += risk_amount

            intent_id = uuid4().hex
            # Journalled BEFORE submission: if the submit call itself is lost
            # (timeout, crash), the intent remains as evidence that an order
            # was meant to go out, and reconciliation decides what happened.
            record("order_intent", intent_id=intent_id, symbol=venue_symbol, side=side,
                   qty=qty, stop_price=latest.stop_price, risk_amount=risk_amount)
            try:
                result = broker.submit_order(venue_symbol, qty, side, stop_price=latest.stop_price)
            except TradingViewError as exc:
                if exc.venue_rejection:
                    record("order_rejected", symbol=venue_symbol, side=side, qty=qty,
                           intent_id=intent_id, error=str(exc))
                    actions.append(f"REJECTED {venue_symbol} {side}: {exc}")
                else:
                    record("order_failed", symbol=venue_symbol, side=side, qty=qty,
                           intent_id=intent_id, error=str(exc))
                    actions.append(f"ERROR {venue_symbol}: {exc}")
                continue
            verified_row = None
            try:
                verification = gate.verification_config()
                verified_row = broker.verify_order(
                    venue_symbol, side, qty,
                    attempts=verification["verify_attempts"],
                    delay_seconds=verification["verify_delay_seconds"])
            except Exception as exc:  # noqa: BLE001 - verification is evidence, never a gate
                logger.warning("verify_order failed for %s: %s", venue_symbol, exc)
            record("order", symbol=venue_symbol, side=side, qty=qty, stop_price=latest.stop_price,
                   risk_amount=risk_amount, status=result.status,
                   order_id=result.broker_order_id or (verified_row.id if verified_row else None),
                   verified=verified_row is not None, intent_id=intent_id)
            actions.append(f"ORDER {venue_symbol} {side} qty={qty:g} status={result.status} "
                           f"{'verified' if verified_row else 'UNVERIFIED'} ({latest.reason})")

        except Exception as exc:  # noqa: BLE001
            # One instrument failing must not cancel the rest of the book. On an
            # unattended run there is nobody to notice and re-run the remainder.
            logger.exception("instrument %s failed", inst.symbol)
            record("error", symbol=inst.symbol, error=f"{type(exc).__name__}: {exc}")
            actions.append(f"ERROR {inst.symbol}: {type(exc).__name__}: {exc}")

    return actions


def send_morning_briefing() -> None:
    if not TELEGRAM.is_configured:
        logger.info("Telegram not configured; skipping morning briefing.")
        return
    broker = build_broker(BROKER)
    account = broker.get_account()
    positions = broker.get_open_positions()
    monitor = _get_drawdown_monitor(account.equity)
    briefing = TelegramBriefing(TELEGRAM)

    lines = ["Good morning. Fable trading bot -- morning briefing.", ""]
    if monitor.halted:
        lines.append("RISK HALT ACTIVE: drawdown kill-switch triggered. No new trades until manually reset.")
    if not positions:
        lines.append("Open positions: none.")
    else:
        lines.append("Open positions:")
        for symbol, pos in positions.items():
            lines.append(f"  {symbol} {pos['side']} qty={pos['qty']:.4f} entry={pos['avg_entry_price']:.2f}")
    lines.append("")
    lines.append(f"Equity: {account.equity:,.2f}")

    event_filter = build_event_filter(EVENT_FILTER)
    if not event_filter.available:
        lines.append("")
        lines.append("Economic calendar: UNAVAILABLE (entries are not being event-gated today).")
    else:
        todays_events = event_filter.upcoming(within_hours=24)
        lines.append("")
        if todays_events:
            lines.append("High-impact releases in the next 24h (entries pause around these):")
            for event in todays_events:
                stamp = event.when.strftime("%H:%M UTC")
                forecast = f", forecast {event.forecast}" if event.forecast else ""
                lines.append(f"  {stamp}  {event.currency}  {event.name}{forecast}")
        else:
            lines.append("High-impact releases in the next 24h: none scheduled.")

    briefing._send("\n".join(lines))


def send_evening_briefing(actions_today: list[str]) -> None:
    if not TELEGRAM.is_configured:
        logger.info("Telegram not configured; skipping evening briefing.")
        return
    broker = build_broker(BROKER)
    account = broker.get_account()
    briefing = TelegramBriefing(TELEGRAM)
    lines = ["Evening briefing.", ""]
    lines.extend(actions_today or ["No signal actions today."])
    lines.append("")
    lines.append(f"Equity: {account.equity:,.2f}")
    briefing._send("\n".join(lines))
