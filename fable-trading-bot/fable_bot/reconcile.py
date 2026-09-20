"""Reconciliation: broker reality vs the bot's own journal.

Learned from the 2026-08 incident, when fills existed in the paper account
that the journal had never recorded, and a dead watcher silently skipped a
whole session. This module closes both holes, once per run:

  * every broker order since the last reconciliation must match a journal
    ``order_intent``/``order`` event, or it is flagged ``unjournaled_fill``
    (someone -- or some path -- traded without the bot knowing);
  * every journal intent older than the matching window with no broker row is
    flagged ``unverified_intent`` (a submission that was lost, or a timeout
    whose order never landed);
  * every NYSE session day since the last reconciliation with no completed
    run is flagged ``missing_run`` (the dead-watcher case).

Fail-soft by contract: reconciliation can add evidence to a run, it can never
fail one. Callers still wrap it defensively.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from .config import MEMORY_DIR
from .journal import Journal, completed_session_days
from .market_calendar import ET, is_trading_day, now_et
from .timeutil import parse_utc_iso, utc_now, utc_now_iso

logger = logging.getLogger("fable_bot.reconcile")

MATCH_WINDOW = timedelta(minutes=15)
RECONCILED_STATUSES = ("filled", "working")
GAP_LOOKBACK_DAYS = 7


@dataclass
class Finding:
    kind: str  # "unjournaled_fill" | "unverified_intent" | "missing_run"
    symbol: str = ""
    side: str = ""
    qty: float = 0.0
    day: str = ""
    detail: str = ""


@dataclass
class ReconState:
    last_reconciled_ts: str | None = None
    last_session_day: str | None = None


def _state_path(memory_dir: Path) -> Path:
    return memory_dir / "reconciliation_state.json"


def load_state(memory_dir: Path) -> ReconState:
    path = _state_path(memory_dir)
    if not path.exists():
        return ReconState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ReconState()
    return ReconState(
        last_reconciled_ts=data.get("last_reconciled_ts"),
        last_session_day=data.get("last_session_day"),
    )


def save_state(memory_dir: Path, state: ReconState) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(memory_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "last_reconciled_ts": state.last_reconciled_ts,
        "last_session_day": state.last_session_day,
    }, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _journal_events(log_dir: Path, days: list[date]) -> list[dict]:
    """Events from the journal files for the given dates, torn-line tolerant."""
    events: list[dict] = []
    for day in days:
        path = log_dir / f"journal-{day.isoformat()}.jsonl"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return events


def _intent_events(events: list[dict]) -> list[dict]:
    """Journal events that represent an order the bot meant to place."""
    intents = []
    for event in events:
        kind = event.get("kind")
        if kind not in ("order_intent", "order"):
            continue
        if event.get("status") == "dry_run" or event.get("dry_run"):
            continue
        intents.append(event)
    return intents


def _matches(event: dict, row, window: timedelta = MATCH_WINDOW) -> bool:
    if event.get("symbol") != row.symbol or event.get("side") != row.side:
        return False
    try:
        if abs(float(event.get("qty") or 0) - row.qty) > 1e-9:
            return False
        event_ts = parse_utc_iso(event["ts"])
        row_ts = parse_utc_iso(row.placed_at)
    except (ValueError, KeyError, TypeError):
        return False
    return abs((row_ts - event_ts)) <= window


def run_reconciliation(broker, journal: Journal, *, log_dir: Path | None = None,
                       memory_dir: Path | None = None,
                       now: datetime | None = None) -> list[Finding]:
    """Compare broker history and session history against the journal.

    Returns findings for the reviewer; never raises for expected broker
    limitations (a venue without history support skips reconciliation).
    """
    log_dir = log_dir or journal.dir
    memory_dir = Path(memory_dir) if memory_dir else MEMORY_DIR
    now = now or now_et()
    today = now.astimezone(ET).date()
    findings: list[Finding] = []

    try:
        history = broker.get_order_history(limit=100)
    except NotImplementedError:
        journal.event("reconciliation_skipped", reason="broker does not expose order history")
        return findings

    state = load_state(memory_dir)
    if state.last_reconciled_ts is None:
        # Bootstrap: baseline to now. The known August incident is encoded as
        # seed lessons, not re-flagged forever.
        save_state(memory_dir, ReconState(last_reconciled_ts=utc_now_iso(),
                                          last_session_day=today.isoformat()))
        journal.event("reconciliation_baseline", ts=utc_now_iso())
        return findings

    cutoff = parse_utc_iso(state.last_reconciled_ts)

    # Broker rows that matter: real orders (not bracket stops) placed since the
    # last reconciliation.
    candidates = []
    max_placed = cutoff
    for row in history:
        if row.type == "stop" or row.status not in RECONCILED_STATUSES:
            continue
        if row.placed_at is None:
            journal.event("reconciliation_no_timestamp", symbol=row.symbol,
                          side=row.side, qty=row.qty, status=row.status)
            continue
        row_ts = parse_utc_iso(row.placed_at)
        if row_ts <= cutoff:
            continue
        candidates.append(row)
        max_placed = max(max_placed, row_ts)

    # Journal intents: today's and yesterday's files cover a run and its
    # aftermath across the ET/UTC boundary.
    events = _journal_events(log_dir, [today, today - timedelta(days=1)])
    intents = _intent_events(events)

    matched_intent_ids: set[int] = set()
    for row in candidates:
        match = next((i for i, event in enumerate(intents)
                      if i not in matched_intent_ids and _matches(event, row)), None)
        if match is None:
            finding = Finding(kind="unjournaled_fill", symbol=row.symbol, side=row.side,
                              qty=row.qty, detail=f"broker order {row.id or '?'} "
                              f"{row.status} at {row.placed_at} has no journal record")
            findings.append(finding)
            journal.event("reconciliation_mismatch", finding=finding.kind, symbol=row.symbol,
                          side=row.side, qty=row.qty, order_id=row.id, status=row.status,
                          placed_at=row.placed_at)
        else:
            matched_intent_ids.add(match)

    # Intents nobody filled. The floor mirrors the broker-side filter exactly:
    # an intent newer than the cutoff fills with a placed_at newer than the
    # cutoff, so the row would have been a candidate above; anything at or
    # before the cutoff was already adjudicated by the previous pass. Widening
    # the floor flags pre-baseline events whose fills were filtered out --
    # that happened live on the first seeded baseline.
    intent_floor = cutoff
    age_cutoff = utc_now() - MATCH_WINDOW
    for i, event in enumerate(intents):
        if i in matched_intent_ids:
            continue
        try:
            event_ts = parse_utc_iso(event["ts"])
        except (ValueError, KeyError):
            continue
        if event_ts <= intent_floor or event_ts > age_cutoff:
            continue
        finding = Finding(kind="unverified_intent", symbol=str(event.get("symbol", "")),
                          side=str(event.get("side", "")), qty=float(event.get("qty") or 0),
                          detail=f"intent {event.get('intent_id', '?')} from {event['ts']} "
                          "never appeared in broker history")
        findings.append(finding)
        journal.event("reconciliation_mismatch", finding=finding.kind, symbol=finding.symbol,
                      side=finding.side, qty=finding.qty, intent_id=event.get("intent_id"))

    # Missing runs: session days since the last reconciliation with no
    # completed run. Today is excluded -- it is the run we're inside.
    try:
        last_seen = date.fromisoformat(state.last_session_day) if state.last_session_day else today
    except ValueError:
        last_seen = today
    completed = completed_session_days(log_dir)
    for offset in range(1, GAP_LOOKBACK_DAYS + 1):
        day = today - timedelta(days=offset)
        if day <= last_seen:
            break
        if not is_trading_day(day):
            continue
        if day.isoformat() in completed:
            continue
        finding = Finding(kind="missing_run", day=day.isoformat(),
                          detail=f"no completed run recorded for the {day.isoformat()} session")
        findings.append(finding)
        journal.event("reconciliation_mismatch", finding=finding.kind, day=day.isoformat())

    save_state(memory_dir, ReconState(
        last_reconciled_ts=max(max_placed, utc_now()).isoformat(),
        last_session_day=today.isoformat(),
    ))
    if not findings:
        journal.event("reconciliation_clean", checked_rows=len(candidates))
    return findings
