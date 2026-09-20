"""The bot's lesson memory: mistakes it has made and the rules it now follows.

The point is *learning*, not just patching: when something goes wrong (a
venue rejection, an unjournaled fill, a sizing anomaly, a missed session) the
reviewer (review.py) distills it into a Lesson -- a rule drawn from a closed
vocabulary below -- and appends it here. Every run loads the active lessons
and lets LessonGate enforce them before any order goes out.

Safety invariant that makes auto-activation defensible: every rule type can
only RESTRICT entries or OBSERVE. Nothing in the vocabulary can enable
trading, enlarge sizing, add symbols, or weaken an existing gate. Params are
clamped by RULE_REGISTRY at both load and add time, so a malformed lesson can
never take effect.

Storage is append-only and event-sourced, same discipline as journal.py:
lesson records are appended as single lines, status changes append a
``status_change`` line rather than rewriting, and readers fold the file,
tolerating a torn final line. That removes rewrite/locking hazards on an
unattended Windows host.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from .config import MEMORY_DIR
from .timeutil import parse_utc_iso, utc_now, utc_now_iso

logger = logging.getLogger("fable_bot.lessons")

ACTIVE = "active"
DISABLED = "disabled"
STATUSES = (ACTIVE, DISABLED)


class InvalidLesson(ValueError):
    """A lesson failed rule-registry validation and was rejected."""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


# ── rule vocabulary ──────────────────────────────────────────────────────
# Closed on purpose: the reviewer may only emit (rule_type, params) pairs the
# registry knows, and the registry clamps every numeric param. Adding a rule
# type is a code change with tests, never something a run can do on its own.


def _validate_block_symbol(params: dict) -> dict:
    symbols = params.get("symbols")
    if not isinstance(symbols, (list, tuple)) or not symbols:
        raise InvalidLesson("block_symbol requires a non-empty 'symbols' list")
    cleaned = sorted({str(s) for s in symbols if str(s).strip()})
    if not cleaned:
        raise InvalidLesson("block_symbol 'symbols' contains no usable entries")
    return {"symbols": cleaned}


def _validate_halt_entries(params: dict) -> dict:
    symbol = str(params.get("symbol") or "").strip()
    if not symbol:
        raise InvalidLesson("halt_entries_for_symbol requires 'symbol'")
    until = params.get("until")
    if until is not None:
        try:
            until = parse_utc_iso(str(until)).isoformat(timespec="seconds")
        except ValueError as exc:
            raise InvalidLesson(f"halt_entries_for_symbol 'until' unparseable: {until}") from exc
    return {"symbol": symbol, "until": until}


def _validate_sizing_bounds(params: dict) -> dict:
    risk_min = _clamp(params.get("risk_pct_min", 0.25), 0.0, 2.0)
    risk_max = _clamp(params.get("risk_pct_max", 1.5), 0.1, 1.5)
    if risk_min > risk_max:
        risk_min = risk_max
    return {
        "risk_pct_min": risk_min,
        "risk_pct_max": risk_max,
        "max_notional_pct_equity": _clamp(params.get("max_notional_pct_equity", 2.0), 0.1, 10.0),
        "min_notional_usd": _clamp(params.get("min_notional_usd", 100.0), 0.0, 10000.0),
        # The dust floor as a FRACTION of equity. min_notional_usd alone is an
        # absolute number written for a six-figure account; on a small account
        # it rises above the notional ceiling and bans every order outright
        # (at $25 equity: floor $100 > ceiling $50, admissible window empty).
        # sizing_sanity takes the lesser of the two, so this bounds the floor
        # on small accounts and is inert on large ones.
        "min_notional_pct_equity": _clamp(params.get("min_notional_pct_equity", 0.10), 0.0, 1.0),
    }


def _validate_max_adds(params: dict) -> dict:
    try:
        max_adds = int(params.get("max_adds", 0))
    except (TypeError, ValueError) as exc:
        raise InvalidLesson("max_adds_per_position 'max_adds' must be an integer") from exc
    return {"max_adds": int(_clamp(max_adds, 0, 3))}


def _validate_require_verification(params: dict) -> dict:
    return {
        "verify_attempts": int(_clamp(params.get("verify_attempts", 3), 1, 10)),
        "verify_delay_seconds": _clamp(params.get("verify_delay_seconds", 2.0), 0.0, 30.0),
    }


def _validate_alert_on_gap(params: dict) -> dict:
    return {"max_missing_sessions": int(_clamp(params.get("max_missing_sessions", 1), 1, 5))}


def _validate_note(params: dict) -> dict:
    topic = str(params.get("topic") or "").strip()
    if not topic:
        raise InvalidLesson("note requires a 'topic'")
    return {"topic": topic}


RULE_REGISTRY: dict[str, Callable[[dict], dict]] = {
    "block_symbol": _validate_block_symbol,
    "halt_entries_for_symbol": _validate_halt_entries,
    "sizing_notional_bounds": _validate_sizing_bounds,
    "max_adds_per_position": _validate_max_adds,
    "require_verification": _validate_require_verification,
    "alert_on_gap": _validate_alert_on_gap,
    "note": _validate_note,
}


def validate_lesson(rule_type: str, params: dict) -> dict:
    """Return clamped/canonical params, or raise InvalidLesson."""
    validator = RULE_REGISTRY.get(rule_type)
    if validator is None:
        raise InvalidLesson(f"unknown rule_type: {rule_type!r}")
    return validator(params or {})


def mistake_signature(rule_type: str, params: dict) -> str:
    """Stable identity of a lesson: same rule + same canonical params = same mistake."""
    canonical = json.dumps({"rule_type": rule_type, "params": params}, sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ── data model ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Lesson:
    id: str
    ts: str
    source: str  # "seeded" | "reviewer"
    incident: str
    mistake_signature: str
    rule_type: str
    params: dict
    status: str = ACTIVE
    note: str = ""

    def to_record(self) -> dict:
        return {
            "kind": "lesson", "id": self.id, "ts": self.ts, "source": self.source,
            "incident": self.incident, "mistake_signature": self.mistake_signature,
            "rule_type": self.rule_type, "params": self.params,
            "status": self.status, "note": self.note,
        }


# ── seed lessons: the 2026-08 weekend incident ───────────────────────────

SEED_INCIDENT = "2026-08 weekend incident"

SEED_LESSONS: list[tuple[str, dict, str]] = [
    ("block_symbol",
     {"symbols": ["NASDAQ:NDX", "NASDAQ:IXIC", "TVC:DXY", "CBOE:VIX", "CBOE:VXN", "TVC:TNX"]},
     "Orders to non-tradable intelligence-only index series were rejected by the "
     "paper venue (2026-08-27)."),
    ("require_verification",
     {"verify_attempts": 3, "verify_delay_seconds": 2.0},
     "A filled order went unjournaled when the submission response was lost "
     "(2026-08-20 BTC buy: 'submitted but no order appeared')."),
    ("note",
     {"topic": "weekend_unmanaged_positions"},
     "24/7 BTC and 24/5 FX positions are unmanaged over weekends because the "
     "NYSE calendar gate stops all runs; weekend care runs deferred by the user."),
    ("alert_on_gap",
     {"max_missing_sessions": 1},
     "The watcher died 2026-08-28 (Fri) mid-wait and the session was silently "
     "missed; missing sessions must surface within one day."),
    ("sizing_notional_bounds",
     {"risk_pct_min": 0.25, "risk_pct_max": 1.5,
      "max_notional_pct_equity": 2.0, "min_notional_usd": 100.0},
     "EURJPY sized to 628 units next to EURUSD 85000 in the same batch "
     "(2026-08-27); order magnitude must stay inside sane bounds."),
    ("max_adds_per_position",
     {"max_adds": 0},
     "GLD was added to at an extended price (426.43) then stopped out at 407.3 "
     "(-$362 realized); never pyramid into an open position."),
]


# ── storage ──────────────────────────────────────────────────────────────


class LessonBook:
    """Append-only lesson store with signature dedupe.

    ``add`` is a no-op if the signature already exists in ANY status --
    including disabled, so a human veto is respected and the same mistake is
    learned exactly once.
    """

    def __init__(self, memory_dir: Path | None = None):
        self.dir = Path(memory_dir) if memory_dir else MEMORY_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "lessons.jsonl"
        self._lessons: dict[str, Lesson] = {}
        self._signatures: set[str] = set()
        self.load_errors: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        self.load_errors.append("torn or malformed line skipped")
                        continue  # torn final line from a hard kill; skip it
                    self._apply(record)
        except OSError as exc:
            self.load_errors.append(f"read failed: {exc}")

    def _apply(self, record: dict) -> None:
        if record.get("kind") == "status_change":
            lesson = self._lessons.get(record.get("id"))
            if lesson is not None and record.get("status") in STATUSES:
                self._lessons[lesson.id] = Lesson(
                    id=lesson.id, ts=lesson.ts, source=lesson.source,
                    incident=lesson.incident, mistake_signature=lesson.mistake_signature,
                    rule_type=lesson.rule_type, params=lesson.params,
                    status=record["status"], note=lesson.note,
                )
            return
        if record.get("kind") != "lesson":
            return
        try:
            params = validate_lesson(record["rule_type"], record.get("params") or {})
        except (InvalidLesson, KeyError) as exc:
            self.load_errors.append(f"lesson {record.get('id')} rejected: {exc}")
            return
        lesson = Lesson(
            id=str(record["id"]), ts=str(record.get("ts", "")),
            source=str(record.get("source", "reviewer")),
            incident=str(record.get("incident", "")),
            mistake_signature=str(record.get("mistake_signature", "")),
            rule_type=str(record["rule_type"]), params=params,
            status=str(record.get("status", ACTIVE)),
            note=str(record.get("note", "")),
        )
        if lesson.status not in STATUSES:
            lesson = Lesson(**{**lesson.__dict__, "status": ACTIVE})
        self._lessons[lesson.id] = lesson
        self._signatures.add(lesson.mistake_signature)

    def _append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()

    def has_signature(self, signature: str) -> bool:
        return signature in self._signatures

    def add(self, rule_type: str, params: dict, *, source: str, incident: str,
            note: str = "") -> Lesson | None:
        """Validate, dedupe, persist and return a new lesson -- or None if rejected/known."""
        try:
            validated = validate_lesson(rule_type, params)
        except InvalidLesson as exc:
            logger.warning("lesson rejected: %s (%s)", exc, rule_type)
            return None
        signature = mistake_signature(rule_type, validated)
        if signature in self._signatures:
            return None
        lesson = Lesson(
            id="L-" + signature.removeprefix("sha256:")[:12],
            ts=utc_now_iso(), source=source, incident=incident,
            mistake_signature=signature, rule_type=rule_type,
            params=validated, status=ACTIVE, note=note,
        )
        self._append(lesson.to_record())
        self._lessons[lesson.id] = lesson
        self._signatures.add(signature)
        logger.info("lesson learned: %s %s %s", lesson.id, rule_type, validated)
        return lesson

    def set_status(self, lesson_id: str, status: str) -> bool:
        if status not in STATUSES or lesson_id not in self._lessons:
            return False
        self._append({"kind": "status_change", "id": lesson_id,
                      "ts": utc_now_iso(), "status": status})
        lesson = self._lessons[lesson_id]
        self._lessons[lesson_id] = Lesson(**{**lesson.__dict__, "status": status})
        return True

    def get(self, lesson_id: str) -> Lesson | None:
        return self._lessons.get(lesson_id)

    def all_lessons(self) -> list[Lesson]:
        return list(self._lessons.values())

    def active_lessons(self) -> list[Lesson]:
        return [lesson for lesson in self._lessons.values() if lesson.status == ACTIVE]

    def seed(self) -> list[Lesson]:
        """Append the incident seed lessons that are not already known. Idempotent."""
        added: list[Lesson] = []
        for rule_type, params, note in SEED_LESSONS:
            lesson = self.add(rule_type, params, source="seeded",
                              incident=SEED_INCIDENT, note=note)
            if lesson is not None:
                added.append(lesson)
        return added


# ── enforcement ──────────────────────────────────────────────────────────


DEFAULT_SIZING_BOUNDS = {
    "risk_pct_min": 0.25, "risk_pct_max": 1.5,
    "max_notional_pct_equity": 2.0, "min_notional_usd": 100.0,
    "min_notional_pct_equity": 0.10,
}
DEFAULT_VERIFICATION = {"verify_attempts": 3, "verify_delay_seconds": 2.0}
DEFAULT_GAP = {"max_missing_sessions": 1}


class LessonGate:
    """Read-side view of the lesson book, consulted before entries.

    Exits are never gated -- same precedent as the event filter in
    live_runner. Multiple active lessons of one type combine to the most
    restrictive interpretation.
    """

    def __init__(self, book: LessonBook, now: Any | None = None):
        self._blocked_symbols: dict[str, str] = {}
        self._halts: dict[str, tuple[str, str]] = {}  # symbol -> (until_iso, lesson_id)
        self._max_adds_lesson: Lesson | None = None
        self._max_adds: int | None = None
        bounds = dict(DEFAULT_SIZING_BOUNDS)
        verification = dict(DEFAULT_VERIFICATION)
        gap = dict(DEFAULT_GAP)
        self.active_count = 0

        now = now or utc_now()

        for lesson in book.active_lessons():
            self.active_count += 1
            if lesson.rule_type == "block_symbol":
                for symbol in lesson.params["symbols"]:
                    self._blocked_symbols[symbol] = lesson.id
            elif lesson.rule_type == "halt_entries_for_symbol":
                until = lesson.params.get("until")
                if until is None:
                    self._halts.setdefault(lesson.params["symbol"], ("", lesson.id))
                    continue
                try:
                    if parse_utc_iso(until) <= now:
                        continue  # time-boxed halt has decayed
                except ValueError:
                    continue
                existing = self._halts.get(lesson.params["symbol"])
                if existing is None or until > existing[0]:
                    self._halts[lesson.params["symbol"]] = (until, lesson.id)
            elif lesson.rule_type == "max_adds_per_position":
                if self._max_adds is None or lesson.params["max_adds"] < self._max_adds:
                    self._max_adds = lesson.params["max_adds"]
                    self._max_adds_lesson = lesson
            elif lesson.rule_type == "sizing_notional_bounds":
                p = lesson.params
                bounds["risk_pct_min"] = max(bounds["risk_pct_min"], p["risk_pct_min"])
                bounds["risk_pct_max"] = min(bounds["risk_pct_max"], p["risk_pct_max"])
                bounds["max_notional_pct_equity"] = min(
                    bounds["max_notional_pct_equity"], p["max_notional_pct_equity"])
                bounds["min_notional_usd"] = max(bounds["min_notional_usd"], p["min_notional_usd"])
                # A HIGHER floor is the more restrictive one, same as the
                # absolute floor above -- the safety invariant is that folding
                # two lessons can only ever narrow what is permitted.
                bounds["min_notional_pct_equity"] = max(
                    bounds["min_notional_pct_equity"], p["min_notional_pct_equity"])
            elif lesson.rule_type == "require_verification":
                verification["verify_attempts"] = max(
                    verification["verify_attempts"], lesson.params["verify_attempts"])
                verification["verify_delay_seconds"] = max(
                    verification["verify_delay_seconds"], lesson.params["verify_delay_seconds"])
            elif lesson.rule_type == "alert_on_gap":
                gap["max_missing_sessions"] = min(
                    gap["max_missing_sessions"], lesson.params["max_missing_sessions"])

        if bounds["risk_pct_min"] > bounds["risk_pct_max"]:
            bounds["risk_pct_min"] = bounds["risk_pct_max"]
        self._sizing_bounds = bounds
        self._verification = verification
        self._gap = gap

    def blocked_entry(self, venue_symbol: str) -> tuple[bool, str, str | None]:
        lesson_id = self._blocked_symbols.get(venue_symbol)
        if lesson_id is not None:
            return True, f"blocked by lesson {lesson_id} (block_symbol)", lesson_id
        halt = self._halts.get(venue_symbol)
        if halt is not None:
            until, lesson_id = halt
            detail = f" until {until}" if until else ""
            return True, f"entries halted by lesson {lesson_id}{detail}", lesson_id
        return False, "", None

    def max_adds_authority(self) -> str | None:
        """Lesson id behind the no-pyramiding policy, if one is active."""
        return self._max_adds_lesson.id if self._max_adds_lesson is not None else None

    def sizing_bounds(self) -> dict:
        return dict(self._sizing_bounds)

    def verification_config(self) -> dict:
        return dict(self._verification)

    def gap_config(self) -> dict:
        return dict(self._gap)


def halt_until_param(days: int = 30) -> str:
    """UTC ISO expiry for a time-boxed halt lesson."""
    return (utc_now() + timedelta(days=days)).isoformat(timespec="seconds")
