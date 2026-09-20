"""Tests for the mistake-to-lesson reviewer.

The reviewer is what makes the loop autonomous, so its tests guard the two
properties that make autonomy safe: every lesson it can produce restricts
entries or observes, and the same mistake is learned exactly once.
"""
from __future__ import annotations

from datetime import timedelta

from fable_bot.journal import Journal
from fable_bot.lessons import DEFAULT_GAP, DEFAULT_SIZING_BOUNDS, DEFAULT_VERIFICATION, LessonBook
from fable_bot.reconcile import Finding
from fable_bot.review import run_review
from fable_bot.timeutil import parse_utc_iso, utc_now


def _setup(tmp_path, run_id="20260831T120000-abc123"):
    journal = Journal(run_id=run_id, log_dir=tmp_path / "logs")
    book = LessonBook(memory_dir=tmp_path / "memory")
    return journal, book


def test_order_rejection_becomes_time_boxed_halt(tmp_path):
    journal, book = _setup(tmp_path)
    journal.event("order_rejected", symbol="NASDAQ:NDX", side="buy",
                  error="Invalid symbol for paper trading")
    learned = run_review(journal, book, [])
    assert len(learned) == 1
    lesson = learned[0]
    assert lesson.rule_type == "halt_entries_for_symbol"
    assert lesson.params["symbol"] == "NASDAQ:NDX"
    assert lesson.source == "reviewer"
    # Time-boxed, never permanent: ~30 days out.
    remaining = parse_utc_iso(lesson.params["until"]) - utc_now()
    assert timedelta(days=29, hours=23) < remaining <= timedelta(days=30, minutes=1)
    assert any(e["kind"] == "lesson_learned" and e["lesson_id"] == lesson.id
               for e in _events(journal))


def test_rejecting_the_same_symbol_twice_is_learned_once(tmp_path):
    journal, book = _setup(tmp_path)
    journal.event("order_rejected", symbol="NASDAQ:NDX", error="nope")
    journal.event("order_rejected", symbol="NASDAQ:NDX", error="nope again")
    learned = run_review(journal, book, [])
    assert len(learned) == 1


def test_second_run_with_another_rejection_of_the_same_symbol_is_a_noop(tmp_path):
    """Deterministic behaviour across runs: the halt exists, so nothing is added,
    even though a fresh 30-day expiry would hash to a new signature."""
    journal, book = _setup(tmp_path)
    journal.event("order_rejected", symbol="NASDAQ:IXIC", error="nope")
    assert len(run_review(journal, book, [])) == 1

    later_journal = Journal(run_id="20260901T120000-def456", log_dir=tmp_path / "logs")
    later_journal.event("order_rejected", symbol="NASDAQ:IXIC", error="nope")
    assert run_review(later_journal, book, []) == []
    halts = [l for l in book.all_lessons() if l.rule_type == "halt_entries_for_symbol"]
    assert len(halts) == 1


def test_disabled_halt_still_vetoes_a_new_one(tmp_path):
    """A human who disables a halt has decided the symbol is fine; a repeat
    rejection must not silently re-arm it."""
    journal, book = _setup(tmp_path)
    journal.event("order_rejected", symbol="CBOE:VIX", error="nope")
    added = run_review(journal, book, [])
    assert len(added) == 1
    assert book.set_status(added[0].id, "disabled")
    later = Journal(run_id="20260902T120000-aaa111", log_dir=tmp_path / "logs")
    later.event("order_rejected", symbol="CBOE:VIX", error="nope")
    assert run_review(later, book, []) == []


def test_sizing_out_of_bounds_dedupes_against_the_seed(tmp_path):
    """The reviewer emits default bounds; seed lesson #5 already encodes exactly
    those, so the finding is a no-op by signature."""
    journal, book = _setup(tmp_path)
    seeded = book.seed()
    assert any(l.rule_type == "sizing_notional_bounds" for l in seeded)
    journal.event("sizing_out_of_bounds", symbol="FX:EURJPY", risk_pct=46.0)
    assert run_review(journal, book, []) == []


def test_sizing_out_of_bounds_adds_bounds_when_unseeded(tmp_path):
    journal, book = _setup(tmp_path)
    journal.event("sizing_out_of_bounds", symbol="FX:EURJPY", risk_pct=46.0)
    learned = run_review(journal, book, [])
    assert len(learned) == 1
    assert learned[0].rule_type == "sizing_notional_bounds"
    assert learned[0].params == DEFAULT_SIZING_BOUNDS


def test_unjournaled_fill_becomes_require_verification(tmp_path):
    journal, book = _setup(tmp_path)
    findings = [Finding(kind="unjournaled_fill", symbol="AMEX:SPY", side="buy", qty=12)]
    learned = run_review(journal, book, findings)
    assert len(learned) == 1
    assert learned[0].rule_type == "require_verification"
    assert learned[0].params == DEFAULT_VERIFICATION


def test_order_disagreement_findings_learn_the_lesson_once(tmp_path):
    journal, book = _setup(tmp_path)
    findings = [
        Finding(kind="unjournaled_fill", symbol="AMEX:SPY", side="buy", qty=12),
        Finding(kind="unverified_intent", symbol="AMEX:GLD", side="buy", qty=7),
    ]
    assert len(run_review(journal, book, findings)) == 1


def test_missing_run_becomes_alert_on_gap(tmp_path):
    journal, book = _setup(tmp_path)
    findings = [Finding(kind="missing_run", day="2026-08-28")]
    learned = run_review(journal, book, findings)
    assert len(learned) == 1
    assert learned[0].rule_type == "alert_on_gap"
    assert learned[0].params == DEFAULT_GAP


def test_events_from_other_runs_are_ignored(tmp_path):
    journal, book = _setup(tmp_path)
    other = Journal(run_id="some-other-run", log_dir=tmp_path / "logs")
    other.event("order_rejected", symbol="NASDAQ:NDX", error="nope")
    assert run_review(journal, book, []) == []


def test_reviewer_can_never_emit_note_lessons(tmp_path):
    """note is seed-only: throw everything at the reviewer and check the
    produced rule types stay inside the restrictive/observing set."""
    journal, book = _setup(tmp_path)
    journal.event("order_rejected", symbol="NASDAQ:NDX", error="nope")
    journal.event("sizing_out_of_bounds", symbol="FX:EURJPY", risk_pct=46.0)
    findings = [
        Finding(kind="unjournaled_fill", symbol="AMEX:SPY", side="buy", qty=12),
        Finding(kind="missing_run", day="2026-08-28"),
    ]
    learned = run_review(journal, book, findings)
    assert {l.rule_type for l in learned} == {
        "halt_entries_for_symbol", "sizing_notional_bounds",
        "require_verification", "alert_on_gap",
    }
    assert all(l.rule_type != "note" for l in learned)


def _events(journal: Journal) -> list[dict]:
    import json
    events = []
    with journal.path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
