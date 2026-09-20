"""Command-line entry point.

    python -m fable_bot.cli backtest [--period 6mo]
    python -m fable_bot.cli signal-check      # one paper/live trading cycle
    python -m fable_bot.cli morning-briefing
    python -m fable_bot.cli evening-briefing
    python -m fable_bot.cli telegram-test
    python -m fable_bot.cli broker-check      # verify the broker connection
    python -m fable_bot.cli calendar          # upcoming high-impact macro releases
    python -m fable_bot.cli lessons list|show|disable|enable|seed
    python -m fable_bot.cli reconcile         # broker history vs journal, on demand
    python -m fable_bot.cli serve             # run signal-check + briefings on a daily schedule
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import INSTRUMENTS, TELEGRAM


def cmd_backtest(args: argparse.Namespace) -> None:
    from .backtester import run_backtest
    from .config import EMINI_FUTURES, FUTURES

    universe = {"equities": INSTRUMENTS, "futures": FUTURES, "emini": EMINI_FUTURES}[args.universe]
    result = run_backtest(universe, period=args.period, interval=args.interval)
    print(f"\n=== Backtest ({args.universe}, {args.period}, {args.interval}) ===")
    print(f"Total return:   {result.total_return_pct:+.2f}%")
    print(f"Sharpe ratio:   {result.sharpe_ratio:.2f}")
    print(f"Max drawdown:   {result.max_drawdown_pct:.2f}%")
    print(f"Win rate:       {result.win_rate_pct:.1f}% ({result.num_trades} trades)")
    print(f"Kill-switch:    {'TRIGGERED' if result.halted else 'not triggered'}")
    if result.trades:
        print("\nTrade log:")
        for t in result.trades:
            print(f"  {t.opened_at.date()} -> {t.closed_at.date()}  {t.symbol:8s} {t.side:5s} "
                  f"pnl={t.pnl:+9.2f}  {t.reason}")


def cmd_signal_check(args: argparse.Namespace) -> None:
    from .journal import Journal
    from .live_runner import run_signal_check

    journal = Journal()
    for line in run_signal_check(dry_run=getattr(args, "dry_run", False), journal=journal):
        print(line)


def cmd_auto_run(args: argparse.Namespace) -> None:
    """One unattended cycle -- what the scheduled task invokes."""
    from .auto_run import auto_run

    outcome = auto_run(
        wait_for_open=not args.no_wait,
        max_wait_minutes=args.max_wait,
        dry_run=args.dry_run,
        force=args.force,
        allow_launch=not args.no_launch,
        ignore_session=args.ignore_session,
    )
    print(f"[{outcome.status}] {outcome.detail}")
    for line in outcome.actions:
        print(f"  {line}")
    raise SystemExit(outcome.exit_code)


def cmd_watch(args: argparse.Namespace) -> None:
    """Stay resident and run a cycle at every market open."""
    from .auto_run import watch

    watch(lead_minutes=args.lead, dry_run=args.dry_run)


def cmd_next_session(_args: argparse.Namespace) -> None:
    """Show how the calendar resolves the next few sessions, in both timezones."""
    from datetime import datetime, timedelta

    from .market_calendar import ET, next_session, now_et, session_for, why_closed

    local_tz = datetime.now().astimezone().tzinfo
    now = now_et()
    print(f"Now: {now:%Y-%m-%d %H:%M:%S} ET  |  {now.astimezone(local_tz):%Y-%m-%d %H:%M:%S} local ({local_tz})\n")

    today = session_for(now.date())
    if today is None:
        print(f"Today ({now.date()}): CLOSED -- {why_closed(now.date())}\n")
    else:
        state = "open" if today.is_open(now) else ("pre-open" if now < today.open_at else "closed for the day")
        print(f"Today ({today.day}): session {today.open_at:%H:%M}-{today.close_at:%H:%M} ET "
              f"[{state}]{'  EARLY CLOSE' if today.is_early_close else ''}\n")

    print("Next sessions:")
    cursor = now
    for _ in range(5):
        session = next_session(cursor, include_today=True)
        local_open = session.open_at.astimezone(local_tz)
        print(f"  {session.day}  open {session.open_at:%H:%M} ET = {local_open:%H:%M} local"
              f"{'   (early close)' if session.is_early_close else ''}")
        cursor = session.open_at + timedelta(minutes=1)


def cmd_morning_briefing(_args: argparse.Namespace) -> None:
    from .live_runner import send_morning_briefing

    send_morning_briefing()
    print("Morning briefing sent." if TELEGRAM.is_configured else "Telegram not configured; nothing sent.")


def cmd_evening_briefing(_args: argparse.Namespace) -> None:
    from .journal import Journal
    from .live_runner import run_signal_check, send_evening_briefing

    actions = run_signal_check(journal=Journal())
    send_evening_briefing(actions)
    print("Evening briefing sent." if TELEGRAM.is_configured else "Telegram not configured; nothing sent.")


def cmd_telegram_test(_args: argparse.Namespace) -> None:
    from .telegram_bot import TelegramBriefing

    TelegramBriefing(TELEGRAM).send_test_message()
    print("Test message sent.")


def cmd_telegram_setup(args: argparse.Namespace) -> None:
    """Resolve the chat ID from a token, and write both into .env.

    The chat ID is the fiddly half of Telegram setup -- it is not shown
    anywhere in the app. Rather than sending people to a third-party
    "get my id" bot, this reads it from the bot's own getUpdates feed: message
    the bot once, run this, and it finds you.
    """
    import re

    import requests

    from .config import PROJECT_ROOT

    token = args.token or TELEGRAM.bot_token
    if not token:
        print("No bot token. Get one from @BotFather in Telegram (/newbot), then re-run:")
        print("    python -m fable_bot.cli telegram-setup --token <TOKEN>")
        raise SystemExit(1)

    try:
        me = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=20).json()
    except requests.RequestException as exc:
        print(f"Could not reach Telegram: {exc}")
        raise SystemExit(1) from exc
    if not me.get("ok"):
        print(f"Telegram rejected that token: {me.get('description', me)}")
        raise SystemExit(1)
    username = me["result"].get("username")
    print(f"Bot authenticated: @{username}")

    updates = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20).json()
    chats: dict[str, str] = {}
    for update in updates.get("result", []):
        message = update.get("message") or update.get("channel_post") or {}
        chat = message.get("chat") or {}
        if chat.get("id") is not None:
            label = chat.get("username") or chat.get("title") or chat.get("first_name") or chat.get("type")
            chats[str(chat["id"])] = str(label)

    if not chats:
        print(f"\nNo messages yet. Open Telegram, search @{username}, send it any message,")
        print("then run this command again -- it will pick up your chat ID automatically.")
        raise SystemExit(2)

    chat_id = args.chat_id or next(iter(chats))
    print("\nChats found:")
    for cid, label in chats.items():
        print(f"  {cid}  ({label}){'   <- using this' if cid == chat_id else ''}")

    env_path = PROJECT_ROOT / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    for key, value in (("TELEGRAM_BOT_TOKEN", token), ("TELEGRAM_CHAT_ID", chat_id)):
        line = f"{key}={value}"
        if re.search(rf"^{key}=.*$", existing, flags=re.M):
            existing = re.sub(rf"^{key}=.*$", line, existing, flags=re.M)
        else:
            existing = existing.rstrip("\n") + ("\n" if existing.strip() else "") + line + "\n"
    env_path.write_text(existing, encoding="utf-8")

    print(f"\nWrote TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to {env_path}")
    print("Verify with:  python -m fable_bot.cli telegram-test")


def cmd_trend_signal(args: argparse.Namespace) -> None:
    """Evaluate the validated long-or-flat trend strategy on today's bar."""
    from .broker import build_broker
    from .config import BROKER, GOLD_FULL, GOLD_FUTURES, RISK
    from .data.feed import fetch_history
    from .risk.session_guard import SessionRiskConfig, SessionRiskGuard
    from .strategies.long_or_flat_trend import TrendConfig, evaluate

    universe = GOLD_FUTURES + GOLD_FULL if args.include_full else GOLD_FUTURES
    config = TrendConfig(required_agreement=args.agreement)

    equity = RISK.starting_equity
    try:
        equity = build_broker(BROKER).get_account().equity
    except Exception as exc:  # noqa: BLE001 - signal is still useful without a broker
        print(f"(broker unavailable, sizing off configured equity: {exc})\n")

    guard = SessionRiskGuard(config=SessionRiskConfig(), starting_equity=equity)
    print(f"Equity {equity:,.2f} | risk budget ${guard.risk_budget:,.2f} "
          f"| agreement required {args.agreement}/3\n")

    for inst in universe:
        df = fetch_history(inst.symbol, period="2y")
        signal = evaluate(inst.symbol, df, config)
        print(f"=== {inst.display_name} ({inst.tv_symbol}) ===")
        print(f"  close {signal.price:,.2f}   ATR(14) {signal.atr:,.2f}   "
              f"{config.momentum_days}d momentum {signal.momentum_pct:+.2f}%")
        for name, vote in signal.detail.items():
            if isinstance(vote, bool):
                print(f"    {name:<20} {'UP' if vote else 'down'}")
        print(f"  SIGNAL: {signal.side.upper()} — {signal.reason}")
        if signal.side == "long":
            qty, why = guard.contracts_for(signal.price, signal.stop_price,
                                            point_value=inst.point_value, qty_step=inst.qty_step)
            risk = qty * abs(signal.price - signal.stop_price) * inst.point_value
            print(f"  stop {signal.stop_price:,.2f} ({abs(signal.price-signal.stop_price):.1f} pts)"
                  f"  -> qty {qty:g}  risk ${risk:,.0f} ({risk/equity*100:.2f}%)  {why}")
        print()


def cmd_manage_session(args: argparse.Namespace) -> None:
    """Supervise an open intraday position until it closes or the bell rings."""
    from .session_manager import ManagerConfig, run_session

    raise SystemExit(run_session(args.symbol, ManagerConfig(poll_seconds=args.poll)))


def cmd_broker_check(_args: argparse.Namespace) -> None:
    """Confirm the configured broker is reachable before trusting a live run."""
    from .broker import build_broker
    from .config import BROKER, INSTRUMENTS

    print(f"Provider:     {BROKER.provider}")
    print(f"Mode:         {'paper' if BROKER.is_paper else 'LIVE'} "
          f"(TRADING_MODE={BROKER.trading_mode}, ALLOW_LIVE_TRADING={BROKER.allow_live_trading})")
    if BROKER.provider == "tradingview":
        print(f"MCP path:     {BROKER.tv_mcp_path}")

    try:
        broker = build_broker(BROKER)
        account = broker.get_account()
    except Exception as exc:  # noqa: BLE001 - this command exists to report failures
        print(f"\nBROKER UNREACHABLE: {exc}")
        raise SystemExit(1)

    print(f"\nConnected. equity={account.equity:,.2f} cash={account.cash:,.2f} "
          f"is_paper={account.is_paper}")
    positions = broker.get_open_positions()
    if positions:
        print("Open positions:")
        for symbol, pos in positions.items():
            print(f"  {symbol} {pos['side']} qty={pos['qty']:g} entry={pos['avg_entry_price']:.2f}")
    else:
        print("Open positions: none.")

    print("\nInstrument -> venue symbol:")
    for inst in INSTRUMENTS:
        print(f"  {inst.symbol:8s} -> {broker.venue_symbol(inst)}")


def cmd_calendar(args: argparse.Namespace) -> None:
    from datetime import datetime, timezone

    from .config import EVENT_FILTER
    from .risk.event_filter import build_event_filter

    event_filter = build_event_filter(EVENT_FILTER)
    if not event_filter.available:
        print(f"Economic calendar UNAVAILABLE: {event_filter.unavailable_reason}")
        print("Entries will not be event-gated "
              f"({'blocked' if EVENT_FILTER.fail_closed else 'traded through'} per EVENT_FILTER_FAIL_CLOSED).")
        return

    now = datetime.now(timezone.utc)
    events = event_filter.upcoming(within_hours=args.hours, now=now)
    print(f"=== High-impact releases, next {args.hours:g}h (importance >= {EVENT_FILTER.min_importance}) ===")
    if not events:
        print("None scheduled.")
    for event in events:
        blackout = "  <-- BLACKOUT NOW" if event in set(event_filter.blocking_events(now)) else ""
        print(f"  {event.when:%Y-%m-%d %H:%M} UTC  {event.currency}  {event.name}"
              f"  (fc {event.forecast or '-'} / prev {event.previous or '-'}){blackout}")
    print(f"\nBlackout window: -{EVENT_FILTER.blackout_minutes_before}min "
          f"/ +{EVENT_FILTER.blackout_minutes_after}min around each release.")


def cmd_serve(_args: argparse.Namespace) -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler

    from .journal import Journal
    from .live_runner import run_signal_check, send_evening_briefing, send_morning_briefing

    scheduler = BlockingScheduler()
    scheduler.add_job(send_morning_briefing, "cron", day_of_week="mon-fri", hour=9, minute=0)
    scheduler.add_job(lambda: send_evening_briefing(run_signal_check(journal=Journal())),
                       "cron", day_of_week="mon-fri", hour=16, minute=15)
    print("Serving on a daily schedule (09:00 morning briefing, 16:15 signal check + evening briefing). Ctrl+C to stop.")
    scheduler.start()


def cmd_lessons(args: argparse.Namespace) -> None:
    """Inspect and manage the lessons the bot has learned from its mistakes."""
    from .lessons import LessonBook

    book = LessonBook()
    if book.load_errors:
        print("WARNING: lessons file had load errors (degraded to what parsed):")
        for error in book.load_errors:
            print(f"  {error}")

    action = args.lessons_action
    if action == "seed":
        added = book.seed()
        if added:
            print(f"Seeded {len(added)} lesson(s):")
            for lesson in added:
                print(f"  {lesson.id}  {lesson.rule_type}  {lesson.params}")
        else:
            print("All seed lessons already present; nothing added.")
        _baseline_reconciliation(force=args.force)
        return

    if action == "show":
        lesson = book.get(args.id)
        if lesson is None:
            print(f"No lesson with id {args.id}")
            raise SystemExit(1)
        import json

        print(json.dumps(lesson.to_record(), indent=2))
        return

    if action in ("disable", "enable"):
        status = "disabled" if action == "disable" else "active"
        if book.set_status(args.id, status):
            print(f"{args.id} -> {status}")
        else:
            print(f"No lesson with id {args.id}")
            raise SystemExit(1)
        return

    lessons = book.all_lessons() if args.all else book.active_lessons()
    if not lessons:
        print("No active lessons." if not args.all else "No lessons at all.")
        return
    label = f"{len(lessons)} lesson(s), including disabled" if args.all else f"{len(lessons)} active lesson(s)"
    print(f"{label}:")
    for lesson in lessons:
        status = "" if lesson.status == "active" else f"  [{lesson.status}]"
        print(f"  {lesson.id}  {lesson.rule_type:<24} {lesson.params}  "
              f"({lesson.source}){status}")


def _baseline_reconciliation(*, force: bool) -> None:
    """Point reconciliation at 'now' so history already encoded in the seed
    lessons is not re-flagged. Only touches state when none exists, unless forced."""
    from .config import MEMORY_DIR
    from .reconcile import ReconState, load_state, save_state
    from .timeutil import utc_now_iso

    state = load_state(MEMORY_DIR)
    if state.last_reconciled_ts is None or force:
        save_state(MEMORY_DIR, ReconState(last_reconciled_ts=utc_now_iso()))
        print("Reconciliation baselined to now.")
    else:
        print(f"Reconciliation state left as-is (last reconciled {state.last_reconciled_ts}).")


def cmd_reconcile(_args: argparse.Namespace) -> None:
    """One manual reconciliation pass: broker reality vs the journal."""
    from .auto_run import EXIT_LOCKED, single_instance
    from .broker import build_broker
    from .config import BROKER
    from .journal import Journal
    from .reconcile import run_reconciliation

    with single_instance() as acquired:
        if not acquired:
            print("Another cycle holds the auto-run lock; try again when it finishes.")
            raise SystemExit(EXIT_LOCKED)
        journal = Journal()
        try:
            findings = run_reconciliation(build_broker(BROKER), journal)
        except Exception as exc:  # noqa: BLE001 - report, don't traceback
            journal.event("reconciliation_error", error=f"{type(exc).__name__}: {exc}")
            print(f"RECONCILIATION FAILED: {exc}")
            raise SystemExit(1)

    if not findings:
        print("Reconciliation clean: no unjournaled fills, no lost intents, no missed sessions.")
        return
    print(f"{len(findings)} finding(s):")
    for finding in findings:
        print(f"  {finding.kind:<18} {finding.detail}")


def cmd_intel(args: argparse.Namespace) -> None:
    """Print the positioning briefing: dealer gamma + COT.

    Read-only and network-bound. Nothing here touches the broker or the order
    path -- it informs a decision, it does not take one.
    """
    from .intel.report import GAMMA_UNIVERSE, build_market_report

    symbols = tuple(args.symbols) if args.symbols else GAMMA_UNIVERSE
    print(build_market_report(symbols, max_expiries=args.expiries))


def cmd_drawdown(args: argparse.Namespace) -> None:
    """Show, or manually reset, the durable drawdown kill-switch."""
    from .broker import build_broker
    from .config import BROKER, RISK
    from .live_runner import drawdown_state_path
    from .risk.manager import DrawdownMonitor

    equity = float(args.equity) if args.equity else build_broker(BROKER).get_account().equity
    state_path = drawdown_state_path()
    monitor = DrawdownMonitor.load(state_path,
                                   max_drawdown_pct=RISK.max_drawdown_pct,
                                   current_equity=equity)
    if args.reset:
        # Re-baselining to current equity is the right default for a REAL halt:
        # resuming means accepting today's equity as the new high-water mark.
        # --keep-peak is for a halt that should never have fired (a bad state
        # write, a bogus equity read); re-baselining then would silently lower
        # the bar and under-protect from that point on.
        new_peak = None if args.keep_peak else equity
        monitor.manual_reset(new_peak=new_peak)
        if args.keep_peak:
            print(f"Kill-switch reset. Peak preserved at {monitor.peak_equity:,.2f} "
                  f"(drawdown now {monitor.drawdown_pct(equity):.2f}%).")
        else:
            print(f"Kill-switch reset. Peak re-baselined to {equity:,.2f}.")
        return
    print(f"state file    : {state_path}")
    print(f"peak equity   : {monitor.peak_equity:,.2f}")
    print(f"current equity: {equity:,.2f}")
    print(f"drawdown      : {monitor.drawdown_pct(equity):.2f}% (limit {RISK.max_drawdown_pct:.2f}%)")
    print(f"halted        : {monitor.halted}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="fable_bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_backtest = sub.add_parser("backtest", help="Run the historical backtest")
    p_backtest.add_argument("--period", default="6mo", help="yfinance period, e.g. 6mo, 1y, 2y, 60d")
    p_backtest.add_argument("--interval", default="1d",
                            help="bar size: 1d (default), 1h, 30m, 15m, 5m. Intraday history is capped by yfinance (~60d for 15m).")
    p_backtest.add_argument("--universe", default="equities", choices=["equities", "futures", "emini"],
                            help="equities (SPY/QQQ/BTC/GLD/USO), futures (micros), emini (full-size ES/NQ)")
    p_backtest.set_defaults(func=cmd_backtest)

    p_sig = sub.add_parser("signal-check", help="Run one live/paper signal check + order cycle")
    p_sig.add_argument("--dry-run", action="store_true", help="do everything except place or close orders")
    p_sig.set_defaults(func=cmd_signal_check)

    p_auto = sub.add_parser("auto-run", help="Unattended cycle: calendar gate, preflight, wait for the open, trade")
    p_auto.add_argument("--dry-run", action="store_true", help="do everything except place or close orders")
    p_auto.add_argument("--no-wait", action="store_true", help="do not sleep until the open; run immediately")
    p_auto.add_argument("--max-wait", type=float, default=150.0,
                        help="minutes it may wait for the open before giving up (default 150)")
    p_auto.add_argument("--no-launch", action="store_true",
                        help="fail instead of cold-launching TradingView Desktop")
    p_auto.add_argument("--force", action="store_true",
                        help="run even if this session already completed a run")
    p_auto.add_argument("--ignore-session", action="store_true",
                        help="run even on a weekend or NYSE holiday (for testing)")
    p_auto.set_defaults(func=cmd_auto_run)

    p_watch = sub.add_parser("watch", help="Stay resident and run a cycle at every market open (no admin needed)")
    p_watch.add_argument("--lead", type=float, default=10.0,
                         help="minutes before the open to wake and run preflight (default 10)")
    p_watch.add_argument("--dry-run", action="store_true", help="never place orders")
    p_watch.set_defaults(func=cmd_watch)

    sub.add_parser("next-session", help="Show the next NYSE sessions in ET and local time").set_defaults(func=cmd_next_session)
    sub.add_parser("morning-briefing", help="Send the morning Telegram briefing").set_defaults(func=cmd_morning_briefing)
    sub.add_parser("evening-briefing", help="Run signal check then send the evening briefing").set_defaults(func=cmd_evening_briefing)
    sub.add_parser("telegram-test", help="Send a test Telegram message").set_defaults(func=cmd_telegram_test)

    p_tg = sub.add_parser("telegram-setup", help="Find your chat ID and write Telegram creds into .env")
    p_tg.add_argument("--token", help="bot token from @BotFather (defaults to TELEGRAM_BOT_TOKEN)")
    p_tg.add_argument("--chat-id", help="use this chat ID instead of the first one discovered")
    p_tg.set_defaults(func=cmd_telegram_setup)
    sub.add_parser("broker-check", help="Verify the configured broker is reachable").set_defaults(func=cmd_broker_check)

    p_trend = sub.add_parser("trend-signal", help="Evaluate the validated long-or-flat gold trend strategy")
    p_trend.add_argument("--agreement", type=int, default=1, choices=[1, 2, 3],
                         help="how many of the 3 trend signals must agree (1 = as validated)")
    p_trend.add_argument("--include-full", action="store_true",
                         help="also show the full-size GC contract (usually too large for the 1%% rule)")
    p_trend.set_defaults(func=cmd_trend_signal)

    p_manage = sub.add_parser("manage-session", help="Supervise an open intraday position to the close")
    p_manage.add_argument("--symbol", default="CME_MINI:MES1!", help="venue symbol to supervise")
    p_manage.add_argument("--poll", type=int, default=60, help="poll interval in seconds")
    p_manage.set_defaults(func=cmd_manage_session)

    p_calendar = sub.add_parser("calendar", help="Show upcoming high-impact macro releases")
    p_calendar.add_argument("--hours", type=float, default=48.0, help="look-ahead horizon in hours")
    p_calendar.set_defaults(func=cmd_calendar)

    p_lessons = sub.add_parser("lessons", help="Inspect and manage the bot's learned lessons")
    lessons_sub = p_lessons.add_subparsers(dest="lessons_action", required=True)
    p_lessons_list = lessons_sub.add_parser("list", help="show lessons (active by default)")
    p_lessons_list.add_argument("--all", action="store_true", help="include disabled lessons")
    p_lessons_show = lessons_sub.add_parser("show", help="show one lesson in full")
    p_lessons_show.add_argument("id")
    p_lessons_disable = lessons_sub.add_parser("disable", help="disable a lesson (human veto)")
    p_lessons_disable.add_argument("id")
    p_lessons_enable = lessons_sub.add_parser("enable", help="re-enable a disabled lesson")
    p_lessons_enable.add_argument("id")
    p_lessons_seed = lessons_sub.add_parser(
        "seed", help="append the 2026-08 incident seed lessons (idempotent)")
    p_lessons_seed.add_argument("--force", action="store_true",
                                help="re-baseline reconciliation state to now even if it exists")
    p_lessons.set_defaults(func=cmd_lessons)

    sub.add_parser("reconcile",
                   help="Manually reconcile broker history against the journal").set_defaults(
                       func=cmd_reconcile)

    p_intel = sub.add_parser(
        "intel", help="Positioning briefing: dealer gamma exposure + CFTC COT")
    p_intel.add_argument("symbols", nargs="*",
                         help="underlyings to profile (default: SPY QQQ GLD USO IWM)")
    p_intel.add_argument("--expiries", type=int, default=4,
                         help="option expiries to include per symbol (default 4)")
    p_intel.set_defaults(func=cmd_intel)

    p_dd = sub.add_parser("drawdown", help="Show or reset the durable drawdown kill-switch")
    p_dd.add_argument("--reset", action="store_true",
                      help="clear a halt and re-baseline the peak to current equity")
    p_dd.add_argument("--keep-peak", action="store_true",
                      help="clear the halt but keep the existing peak (for a halt "
                           "that should never have fired)")
    p_dd.add_argument("--equity", type=float,
                      help="use this equity instead of querying the broker")
    p_dd.set_defaults(func=cmd_drawdown)

    sub.add_parser("serve", help="Run signal-check + briefings on a daily schedule").set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
