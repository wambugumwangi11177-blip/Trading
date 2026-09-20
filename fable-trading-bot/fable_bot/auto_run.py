"""The unattended entry point: one full trading cycle, start to finish, with
nobody watching.

Invoked by the OS scheduler. Everything the interactive commands leave to the
operator's judgement has to be decided here instead:

  * Is there a session today? (weekends and NYSE holidays -- the scheduler will
    happily fire on Thanksgiving)
  * Has today's run already happened? (a retry, or a human running it by hand,
    must not double a position)
  * Is TradingView actually up, and pointed at the paper account?
  * Has the opening bell rung yet? The task fires early on purpose, because the
    ET open moves against this machine's clock twice a year.

Only after all four does it place an order. Every step is journalled, so a run
that decides to do nothing says why.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as dt_time

from .config import BROKER
from .journal import Journal, already_ran, setup_file_logging
from .market_calendar import ET, next_session, now_et, session_for, why_closed
from .preflight import Preflight

logger = logging.getLogger("fable_bot.auto_run")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PREFLIGHT = 2
EXIT_LOCKED = 3


# Outcomes worth a message. The valuable ones are the failures: a run that
# could not trade looks exactly like a run with no signals if the only evidence
# is silence. Benign no-ops (weekend, already handled today) stay quiet so the
# alerts keep meaning something.
ANNOUNCED = {
    "ok": "",
    "dry_run": "DRY RUN — no orders were placed.",
    "preflight_failed": "COULD NOT TRADE — TradingView was not reachable.",
    "error": "RUN FAILED.",
    "too_early": "DID NOT RUN — the schedule fired too far before the open.",
    "locked": "SKIPPED — another cycle was already running.",
    # A session that ended before anything ran means the machine was off,
    # asleep, or logged out at the open. That is a missed trading day, and the
    # one failure most likely to go unnoticed -- it leaves no error anywhere.
    "market_closed": "MISSED THE SESSION — nothing ran before the close.",
}


def _announce(status: str, detail: str, actions: list[str], session_day) -> None:
    """Send the run's outcome to Telegram, if it is configured."""
    if status not in ANNOUNCED:
        return
    from .telegram_bot import notify

    headline = ANNOUNCED[status]
    day = session_day.isoformat() if session_day else now_et().date().isoformat()
    lines = [f"Fable bot — {day}"]
    if headline:
        lines.append(headline)
    lines.append("")

    if actions:
        lines.extend(actions)
    elif status == "ok":
        lines.append("No signals on today's bar. Nothing traded.")
    else:
        lines.append(detail)

    if status == "ok" and actions:
        lines.append("")
        lines.append(detail)

    notify("\n".join(lines))


@contextmanager
def single_instance(name: str = "auto-run.lock", directory=None):
    """Refuse to run a second cycle concurrently.

    The journal's `already_ran` check makes a *sequential* re-run harmless, but
    it cannot help with two cycles running at once: both would read an empty
    position book, both would size the same entry, and both would send it. Two
    watchers waking together, or a manual run during a scheduled one, is enough.

    An OS-level advisory lock is used rather than a PID file because it is
    released automatically when the holder dies -- a crashed run must not leave
    a lock that silences every session afterwards.
    """
    from .journal import LOG_DIR

    lock_dir = directory or LOG_DIR
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / name
    handle = open(path, "a+")  # noqa: SIM115 - closed on every path below
    handle.seek(0)

    if not _try_lock(handle):
        # Closing a handle whose locked byte range is held by another process
        # raises PermissionError on Windows, so this close has to be forgiving:
        # failing to tidy up must not turn "someone else is running" into a
        # crash.
        _quiet_close(handle)
        yield False
        return

    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()} {now_et().isoformat()}\n")
        handle.flush()
        yield True
    finally:
        _unlock(handle)
        _quiet_close(handle)


def _try_lock(handle) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _quiet_close(handle) -> None:
    try:
        handle.close()
    except OSError:
        pass


@dataclass
class RunOutcome:
    status: str
    exit_code: int
    detail: str
    actions: list[str]


def _sleep_until(target, *, label: str) -> None:
    """Sleep to a wall-clock target, reporting progress on the way."""
    while True:
        remaining = (target - now_et()).total_seconds()
        if remaining <= 0:
            return
        if remaining > 300 and int(remaining) % 300 < 60:
            logger.info("waiting for %s -- %.0f min to go", label, remaining / 60)
        time.sleep(min(60.0, remaining))


def auto_run(**kwargs) -> RunOutcome:
    """One cycle, guaranteed to be the only one running. See `_auto_run`."""
    with single_instance() as acquired:
        if not acquired:
            setup_file_logging()
            logger.warning("another cycle already holds the lock -- exiting without trading.")
            return RunOutcome(
                status="locked", exit_code=EXIT_LOCKED, actions=[],
                detail="Another auto-run cycle is in progress; refusing to trade concurrently.",
            )
        return _auto_run(**kwargs)


def _auto_run(
    *,
    wait_for_open: bool = True,
    max_wait_minutes: float = 150.0,
    dry_run: bool = False,
    force: bool = False,
    allow_launch: bool = True,
    ignore_session: bool = False,
) -> RunOutcome:
    setup_file_logging()
    journal = Journal()
    started = now_et()

    journal.event("run_start", local_time=str(started), tz=str(ET), dry_run=dry_run,
                  provider=BROKER.provider, mode="paper" if BROKER.is_paper else "LIVE")
    logger.info("auto-run %s starting (%s ET)", journal.run_id, started.strftime("%Y-%m-%d %H:%M:%S"))

    def finish(status: str, code: int, detail: str, actions: list[str] | None = None,
               session_day=None) -> RunOutcome:
        journal.event("run_end", status=status, detail=detail, actions=actions or [],
                      session_day=session_day.isoformat() if session_day else None)
        logger.info("auto-run %s finished: %s -- %s", journal.run_id, status, detail)
        _announce(status, detail, actions or [], session_day)
        return RunOutcome(status=status, exit_code=code, detail=detail, actions=actions or [])

    # 1. Is there a session today?
    session = session_for(started.date())
    if session is None and not ignore_session:
        reason = why_closed(started.date())
        journal.event("no_session", day=started.date().isoformat(), reason=reason)
        return finish("no_session", EXIT_OK, f"No NYSE session today ({reason}).")

    if session is not None:
        journal.event("session", day=session.day.isoformat(), open_at=str(session.open_at),
                      close_at=str(session.close_at), early_close=session.is_early_close)

        # 2. Already traded this session?
        if already_ran(session.day) and not force:
            return finish("already_ran", EXIT_OK,
                          f"A run for {session.day} already completed; not trading again. "
                          "Pass --force to override.", session_day=session.day)

        if started >= session.close_at and not force:
            return finish("market_closed", EXIT_OK,
                          f"Session for {session.day} already closed at "
                          f"{session.close_at:%H:%M} ET.", session_day=session.day)

    session_day = session.day if session else started.date()

    # 3. Get TradingView up and a paper broker attached. Done before the wait so
    #    a cold launch happens on the scheduler's early trigger, not at the bell.
    preflight = Preflight(BROKER)
    result = preflight.ensure_ready(allow_launch=allow_launch)
    journal.event("preflight", **result.as_event())
    if not result.ok:
        logger.error("preflight failed: %s", result.detail)
        return finish("preflight_failed", EXIT_PREFLIGHT, result.detail, session_day=session_day)
    logger.info("preflight ok: %s", result.detail)

    # 4. Wait for the opening bell.
    if session is not None and started < session.open_at:
        wait_minutes = session.opens_in(now_et()).total_seconds() / 60
        # The too-early guard belongs to the waiting path only. --no-wait means
        # "run now regardless", and refusing it whenever the open happens to be
        # far off would make the flag useless precisely when it is needed.
        if wait_for_open and wait_minutes > max_wait_minutes:
            return finish("too_early", EXIT_ERROR,
                          f"Open is {wait_minutes:.0f} min away, beyond the "
                          f"{max_wait_minutes:.0f} min wait budget. Check the task's trigger time.",
                          session_day=session_day)
        if wait_for_open:
            journal.event("waiting_for_open", minutes=round(wait_minutes, 1),
                          open_at=str(session.open_at))
            logger.info("waiting %.0f min for the open (%s ET)", wait_minutes,
                        session.open_at.strftime("%H:%M"))
            _sleep_until(session.open_at, label="the opening bell")
            journal.event("open_reached", at=str(now_et()))

            # A long wait can outlive the connection -- re-verify rather than
            # discovering it inside the first order.
            if not preflight.cdp_up():
                logger.warning("CDP dropped during the wait; re-running preflight.")
                result = preflight.ensure_ready(allow_launch=allow_launch)
                journal.event("preflight", stage="post_wait", **result.as_event())
                if not result.ok:
                    return finish("preflight_failed", EXIT_PREFLIGHT, result.detail,
                                  session_day=session_day)

    # 5. Trade.
    from .live_runner import run_signal_check

    try:
        actions = run_signal_check(require_completed_bar=True, dry_run=dry_run, journal=journal)
    except Exception as exc:  # noqa: BLE001 - the scheduler only sees the exit code
        logger.exception("signal check failed")
        journal.event("error", stage="signal_check", error=f"{type(exc).__name__}: {exc}")
        return finish("error", EXIT_ERROR, f"{type(exc).__name__}: {exc}", session_day=session_day)

    for line in actions:
        logger.info("%s", line)

    # 6. Learn from this run. Reconciliation compares broker reality with the
    #    journal; the reviewer distills any mistake into a lesson that takes
    #    effect from the next run. Fail-soft by contract: a broken learning
    #    path must never stop trading -- the exit status is decided below,
    #    from the trading results alone.
    from .broker import build_broker
    from .lessons import LessonBook
    from .reconcile import run_reconciliation
    from .review import run_review

    findings = []
    try:
        findings = run_reconciliation(build_broker(BROKER), journal)
    except Exception as exc:  # noqa: BLE001
        logger.exception("reconciliation failed")
        journal.event("reconciliation_error", error=f"{type(exc).__name__}: {exc}")
    try:
        learned = run_review(journal, LessonBook(), findings)
        for lesson in learned:
            actions.append(f"LEARNED: {lesson.rule_type} {lesson.params}")
        for finding in findings:
            if finding.kind == "missing_run":
                actions.append(f"MISSED SESSION DETECTED {finding.day}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("review failed")
        journal.event("review_error", error=f"{type(exc).__name__}: {exc}")

    # No separate evening briefing here: `finish` announces every outcome,
    # including the failures a success-only briefing would stay silent about.
    failed = [line for line in actions if line.startswith("ERROR")]
    equity = f"Equity {result.equity:,.2f}" if result.equity is not None else ""
    summary = ", ".join(filter(None, [
        f"{len(actions)} action(s)" if actions else "no signals on today's bar", equity,
    ]))
    if dry_run:
        status = "dry_run"
        detail = summary
    elif failed:
        # An order that errored is not a completed cycle, whatever happened to
        # the others: marking the run "ok" here is what lets a day of failed
        # orders look exactly like a day of no signals. Reporting "error"
        # leaves the session unhandled, so the watcher retries -- safe, because
        # each cycle re-reads the position book before sizing anything, so an
        # order that actually landed late is seen as a position, not re-entered.
        status = "error"
        detail = f"{len(failed)} of {len(actions)} order(s) FAILED; {summary}"
    else:
        status = "ok"
        detail = summary
    return finish(status, EXIT_ERROR if failed and not dry_run else EXIT_OK, detail,
                  actions=actions, session_day=session_day)


RETRYABLE_STATUSES = {"preflight_failed", "error"}


def _next_unhandled_session(last_handled):
    """The next session strictly after the one already dealt with.

    Asking `next_session` alone is not enough. The watcher wakes *before* the
    bell, so a cycle can finish while today's open is still in the future --
    at which point "the next session" is the one just handled, and the loop
    runs it again. Tracking the handled day explicitly is what makes advancing
    a property of the loop rather than a side effect of how long `auto_run`
    happened to take.
    """
    session = next_session(now_et(), include_today=True)
    if last_handled is not None and session.day <= last_handled:
        probe = datetime.combine(last_handled, dt_time(12, 0), tzinfo=ET)
        session = next_session(probe, include_today=False)
    return session


def watch(*, lead_minutes: float = 10.0, dry_run: bool = False, max_cycles: int | None = None,
          max_retries: int = 2, retry_delay_seconds: float = 300.0) -> None:
    """Resident scheduler: sleep until each session, run a cycle, repeat.

    The alternative to a Windows Scheduled Task, and the one that needs no
    administrator rights -- registering a task does. It costs nothing in
    robustness here, because driving TradingView already requires a logged-in
    interactive desktop; a process living in that same session is subject to
    exactly the same condition.

    Nothing in this loop is allowed to terminate it. An exception on Tuesday
    must not mean silence for the rest of the year, so failures are logged and
    the loop advances to the next session.
    """
    setup_file_logging()
    logger.info("watcher started (lead %.0f min%s)", lead_minutes, ", DRY RUN" if dry_run else "")

    cycles = 0
    last_handled = None   # the session day already dealt with
    session = None        # held across retries of the same day
    attempts = 0

    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        try:
            if session is None:
                session = _next_unhandled_session(last_handled)
                attempts = 0
                wake_at = session.open_at - timedelta(minutes=lead_minutes)
                logger.info("next session %s, open %s ET -- waking at %s ET",
                            session.day, session.open_at.strftime("%H:%M"), wake_at.strftime("%H:%M"))
            else:
                wake_at = session.open_at - timedelta(minutes=lead_minutes)

            if wake_at > now_et():
                _sleep_until(wake_at, label=f"the {session.day} session")

            outcome = auto_run(wait_for_open=True, dry_run=dry_run)
            logger.info("cycle result: %s -- %s", outcome.status, outcome.detail)
            should_retry = outcome.status in RETRYABLE_STATUSES
        except KeyboardInterrupt:
            logger.info("watcher stopped by user.")
            return
        except Exception:  # noqa: BLE001 - a resident scheduler must never die
            logger.exception("watch cycle failed")
            should_retry = True

        # A failed cycle is worth another go -- TradingView may have been slow
        # to come up -- but a bounded number of them. Retrying forever would
        # hammer the app for the rest of the day.
        if should_retry and session is not None and attempts < max_retries:
            attempts += 1
            logger.warning("retrying the %s session (attempt %d of %d) in %.0f min",
                           session.day, attempts, max_retries, retry_delay_seconds / 60)
            time.sleep(retry_delay_seconds)
            continue

        if session is not None:
            last_handled = session.day
        session = None
        attempts = 0
        time.sleep(90)
