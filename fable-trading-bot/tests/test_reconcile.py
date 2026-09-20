"""Tests for broker-vs-journal reconciliation.

Reconciliation is the part that turns the 2026-08 incident -- fills the journal
never recorded, and a dead watcher that silently skipped a session -- into
findings the reviewer can learn from. The broker is faked; what gets exercised
is the matching, the time-window math, and the gap detection.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fable_bot.broker.base import HistoryOrder
from fable_bot.market_calendar import ET
from fable_bot.reconcile import ReconState, load_state, run_reconciliation, save_state
from fable_bot.timeutil import parse_utc_iso, utc_now


class FakeBroker:
    def __init__(self, history=None, *, unsupported=False):
        self.history = history or []
        self.unsupported = unsupported

    def get_order_history(self, limit=100):
        if self.unsupported:
            raise NotImplementedError("no history")
        return self.history


class FakeJournal:
    """Records events in memory; hands reconcile a log_dir to read from."""

    def __init__(self, log_dir: Path):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []

    def event(self, kind, **fields):
        record = {"kind": kind, **fields}
        self.events.append(record)
        return record

    @property
    def kinds(self) -> list[str]:
        return [e["kind"] for e in self.events]


def _write_journal(log_dir: Path, day: str, events: list[dict]) -> None:
    with (log_dir / f"journal-{day}.jsonl").open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def _seed_state(memory_dir: Path, last_reconciled_ts: str, last_session_day: str | None) -> None:
    save_state(memory_dir, ReconState(last_reconciled_ts=last_reconciled_ts,
                                      last_session_day=last_session_day))


# ── Bootstrap ────────────────────────────────────────────────────────────

def test_first_run_baselines_and_reports_nothing(tmp_path):
    """The known August incident is encoded as seed lessons, not re-flagged forever."""
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    broker = FakeBroker([HistoryOrder("1", "AMEX:SPY", "buy", "market", "filled", 12,
                                      placed_at="2026-08-20T10:00:00Z")])
    findings = run_reconciliation(broker, journal, log_dir=log_dir, memory_dir=memory)
    assert findings == []
    assert "reconciliation_baseline" in journal.kinds
    state = load_state(memory)
    assert state.last_reconciled_ts is not None
    assert state.last_session_day is not None


# ── Matched intent + fill ────────────────────────────────────────────────

def test_matched_intent_and_fill_is_clean_and_state_advances(tmp_path):
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    _write_journal(log_dir, "2026-09-15", [
        {"ts": "2026-09-15T11:55:00+00:00", "kind": "order_intent", "symbol": "AMEX:SPY",
         "side": "buy", "qty": 12, "intent_id": "abc"},
    ])
    _seed_state(memory, "2026-09-15T10:00:00+00:00", "2026-09-14")
    broker = FakeBroker([HistoryOrder("77", "AMEX:SPY", "buy", "market", "filled", 12,
                                      placed_at="2026-09-15T11:56:00+00:00")])
    findings = run_reconciliation(broker, journal, log_dir=log_dir, memory_dir=memory,
                                  now=datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc))
    assert findings == []
    assert "reconciliation_clean" in journal.kinds
    state = load_state(memory)
    assert parse_utc_iso(state.last_reconciled_ts) >= datetime(2026, 9, 15, 11, 56, tzinfo=timezone.utc)
    assert state.last_session_day == "2026-09-15"


# ── Mismatches ───────────────────────────────────────────────────────────

def test_unjournaled_fill_is_flagged(tmp_path):
    """A broker order with no journal record is exactly the August incident."""
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    _seed_state(memory, "2026-09-15T10:00:00+00:00", "2026-09-14")
    broker = FakeBroker([HistoryOrder("77", "AMEX:SPY", "buy", "market", "filled", 12,
                                      placed_at="2026-09-15T11:56:00Z")])
    findings = run_reconciliation(broker, journal, log_dir=log_dir, memory_dir=memory,
                                  now=datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc))
    assert [f.kind for f in findings] == ["unjournaled_fill"]
    assert findings[0].symbol == "AMEX:SPY"
    assert findings[0].qty == 12
    assert "reconciliation_mismatch" in journal.kinds


def test_unverified_intent_when_no_broker_row_appears(tmp_path):
    """A submission whose response was lost and never shows up in history."""
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    now = utc_now()
    today_et = now.astimezone(ET).date().isoformat()
    _write_journal(log_dir, today_et, [
        {"ts": (now - timedelta(hours=2)).isoformat(timespec="seconds"),
         "kind": "order_intent", "symbol": "AMEX:SPY", "side": "buy", "qty": 5,
         "intent_id": "xyz"},
    ])
    _seed_state(memory, (now - timedelta(days=1)).isoformat(timespec="seconds"), today_et)
    findings = run_reconciliation(FakeBroker([]), journal, log_dir=log_dir, memory_dir=memory)
    assert [f.kind for f in findings] == ["unverified_intent"]
    assert findings[0].symbol == "AMEX:SPY"


def test_events_before_the_baseline_are_not_reflagged(tmp_path):
    """Live regression: the first seeded reconcile flagged two pre-baseline
    order events as unverified because their fills were filtered out on the
    broker side. Journal scope must mirror the broker scope exactly."""
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    _write_journal(log_dir, "2026-09-15", [
        {"ts": "2026-09-15T09:50:00+00:00", "kind": "order", "symbol": "OANDA:EURUSD",
         "side": "buy", "qty": 118000, "status": "filled", "order_id": "1"},
    ])
    _seed_state(memory, "2026-09-15T10:00:00+00:00", "2026-09-14")
    findings = run_reconciliation(FakeBroker([]), journal, log_dir=log_dir, memory_dir=memory,
                                  now=datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc))
    assert findings == []
    assert "reconciliation_clean" in journal.kinds


def test_dry_run_intents_are_not_reconciled(tmp_path):
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    _write_journal(log_dir, "2026-09-15", [
        {"ts": "2026-09-15T11:55:00+00:00", "kind": "order_intent", "symbol": "AMEX:SPY",
         "side": "buy", "qty": 12, "status": "dry_run"},
    ])
    _seed_state(memory, "2026-09-15T10:00:00+00:00", "2026-09-15")
    findings = run_reconciliation(FakeBroker([]), journal, log_dir=log_dir, memory_dir=memory,
                                  now=datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc))
    assert findings == []


# ── Timestamp forms ──────────────────────────────────────────────────────

def test_z_and_offset_forms_of_the_same_moment_both_match(tmp_path):
    """CLI returns ...Z, the journal writes ...+00:00; neither may fail to match."""
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    _write_journal(log_dir, "2026-09-15", [
        {"ts": "2026-09-15T11:55:00+00:00", "kind": "order", "symbol": "AMEX:SPY",
         "side": "buy", "qty": 3},
        {"ts": "2026-09-15T12:10:00+00:00", "kind": "order", "symbol": "AMEX:GLD",
         "side": "buy", "qty": 7},
    ])
    _seed_state(memory, "2026-09-15T10:00:00+00:00", "2026-09-14")
    broker = FakeBroker([
        HistoryOrder("1", "AMEX:SPY", "buy", "market", "filled", 3,
                     placed_at="2026-09-15T11:55:30Z"),
        HistoryOrder("2", "AMEX:GLD", "buy", "market", "filled", 7,
                     placed_at="2026-09-15T12:10:30+00:00"),
    ])
    findings = run_reconciliation(broker, journal, log_dir=log_dir, memory_dir=memory,
                                  now=datetime(2026, 9, 15, 12, 30, tzinfo=timezone.utc))
    assert findings == []


# ── Row filtering ────────────────────────────────────────────────────────

def test_bracket_protective_stops_are_not_reconciled(tmp_path):
    """Our own bracket stops live in history; treating them as orders would be
    a permanent false positive."""
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    _seed_state(memory, "2026-09-15T10:00:00+00:00", "2026-09-14")
    broker = FakeBroker([HistoryOrder("88", "AMEX:SPY", "sell", "stop", "working", 12,
                                      placed_at="2026-09-15T11:56:00Z")])
    findings = run_reconciliation(broker, journal, log_dir=log_dir, memory_dir=memory,
                                  now=datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc))
    assert findings == []
    clean = next(e for e in journal.events if e["kind"] == "reconciliation_clean")
    assert clean["checked_rows"] == 0


def test_rejected_and_stale_rows_are_out_of_scope(tmp_path):
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    _seed_state(memory, "2026-09-15T10:00:00+00:00", "2026-09-14")
    broker = FakeBroker([
        HistoryOrder("1", "NASDAQ:NDX", "buy", "market", "rejected", 1,
                     placed_at="2026-09-15T11:00:00Z"),
        HistoryOrder("2", "AMEX:SPY", "buy", "market", "filled", 5,
                     placed_at="2026-09-15T09:00:00Z"),  # before the cutoff
    ])
    findings = run_reconciliation(broker, journal, log_dir=log_dir, memory_dir=memory,
                                  now=datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc))
    assert findings == []


# ── Missing runs ─────────────────────────────────────────────────────────

def test_missing_session_day_is_flagged(tmp_path):
    """Replay the incident: watcher died Friday 2026-08-28; Monday's run must notice."""
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    _seed_state(memory, "2026-08-27T21:00:00+00:00", "2026-08-27")
    findings = run_reconciliation(FakeBroker([]), journal, log_dir=log_dir, memory_dir=memory,
                                  now=datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc))  # Mon 16:00 ET
    assert [f.kind for f in findings] == ["missing_run"]
    assert findings[0].day == "2026-08-28"


def test_completed_session_day_is_not_flagged(tmp_path):
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    _write_journal(log_dir, "2026-08-28", [
        {"ts": "2026-08-28T14:00:00+00:00", "kind": "run_end", "status": "ok",
         "session_day": "2026-08-28"},
    ])
    _seed_state(memory, "2026-08-27T21:00:00+00:00", "2026-08-27")
    findings = run_reconciliation(FakeBroker([]), journal, log_dir=log_dir, memory_dir=memory,
                                  now=datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc))
    assert findings == []


# ── Broker limitations ───────────────────────────────────────────────────

def test_broker_without_history_support_is_skipped_not_fatal(tmp_path):
    log_dir, memory = tmp_path / "logs", tmp_path / "memory"
    journal = FakeJournal(log_dir)
    findings = run_reconciliation(FakeBroker(unsupported=True), journal,
                                  log_dir=log_dir, memory_dir=memory)
    assert findings == []
    assert "reconciliation_skipped" in journal.kinds
    assert not (memory / "reconciliation_state.json").exists()
