"""The reviewer: observed mistakes -> lessons, automatically.

Runs after every cycle (auto_run.py), looking at two evidence streams:

  * this run's journal events -- ``order_rejected`` (the venue refused a
    symbol) and ``sizing_out_of_bounds`` (a sized order left sane bounds);
  * reconciliation findings -- broker reality disagreeing with the journal,
    or a trading session that passed with no completed run.

Each maps to exactly one lesson type through a FIXED mapping. Combined with
the closed rule vocabulary in lessons.py, that is what makes auto-activation
safe: the reviewer cannot invent rule types, cannot emit params outside the
clamps, and can only ever restrict entries or observe. Repeats are no-ops:
signature dedupe in the book, plus symbol-level dedupe for halts, so the same
mistake is learned exactly once.
"""
from __future__ import annotations

import json
import logging

from .journal import Journal
from .lessons import (
    DEFAULT_GAP,
    DEFAULT_SIZING_BOUNDS,
    DEFAULT_VERIFICATION,
    Lesson,
    LessonBook,
    halt_until_param,
)
from .reconcile import Finding

logger = logging.getLogger("fable_bot.review")

HALT_DAYS = 30

# Findings that mean "an order's fate was unknown or unjournaled": the
# answer is always a stricter post-submission verification poll.
_VERIFICATION_FINDINGS = {"unjournaled_fill", "unverified_intent"}


def _run_events(journal: Journal) -> list[dict]:
    """Events belonging to this run only, torn-line tolerant."""
    events: list[dict] = []
    try:
        with journal.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("run_id") == journal.run_id:
                    events.append(record)
    except OSError:
        pass
    return events


def _halt_exists(book: LessonBook, symbol: str) -> bool:
    """Any halt for this symbol, in ANY status -- a human veto is respected,
    and the expiry clock is not restarted by a repeat rejection."""
    return any(
        lesson.rule_type == "halt_entries_for_symbol"
        and lesson.params.get("symbol") == symbol
        for lesson in book.all_lessons()
    )


def run_review(journal: Journal, book: LessonBook, findings: list[Finding]) -> list[Lesson]:
    """Distill this run's mistakes into lessons. Returns the lessons actually
    added (dedupe turns repeats into nothing). Never raises for a missing or
    malformed journal -- callers still wrap it defensively."""
    planned: list[tuple[str, dict, str]] = []  # (rule_type, params, note)
    until = halt_until_param(HALT_DAYS)  # one expiry per run keeps signatures stable
    halted_symbols: set[str] = set()

    for event in _run_events(journal):
        kind = event.get("kind")
        if kind == "order_rejected":
            symbol = str(event.get("symbol") or "").strip()
            if not symbol or symbol in halted_symbols:
                continue
            halted_symbols.add(symbol)
            planned.append((
                "halt_entries_for_symbol",
                {"symbol": symbol, "until": until},
                f"venue rejected an order for {symbol}: {event.get('error', 'unknown error')}",
            ))
        elif kind == "sizing_out_of_bounds":
            planned.append((
                "sizing_notional_bounds", dict(DEFAULT_SIZING_BOUNDS),
                "a sized order fell outside sane risk/notional bounds",
            ))

    kinds = {finding.kind for finding in findings}
    if kinds & _VERIFICATION_FINDINGS:
        planned.append((
            "require_verification", dict(DEFAULT_VERIFICATION),
            "broker reality and the journal disagreed about an order",
        ))
    if "missing_run" in kinds:
        planned.append((
            "alert_on_gap", dict(DEFAULT_GAP),
            "a trading session passed with no completed run",
        ))

    learned: list[Lesson] = []
    incident = f"auto-review {journal.run_id}"
    for rule_type, params, note in planned:
        # The reviewer's mapping never emits note/seed-only rules, and book.add
        # re-validates against RULE_REGISTRY anyway: unknown types and
        # out-of-clamp params are rejected here, not executed.
        if rule_type == "halt_entries_for_symbol" and _halt_exists(book, params["symbol"]):
            continue
        lesson = book.add(rule_type, params, source="reviewer", incident=incident, note=note)
        if lesson is None:
            continue
        journal.event("lesson_learned", lesson_id=lesson.id, rule_type=lesson.rule_type,
                      params=lesson.params, mistake_signature=lesson.mistake_signature,
                      note=note)
        logger.info("lesson learned: %s %s", lesson.id, rule_type)
        learned.append(lesson)
    return learned
